"""Batch OCR and caption generator for extracted exam images.

This script walks `data/images/` (or a supplied directory), runs OCR on each
image, extracts lightweight metadata (figure references, axis labels, units,
numeric tokens), and writes a JSON caption alongside the image under
`data/captions/`. It also records a run log in `logs/cleanup/` for auditing.

Example:

    python scripts/ocr_images.py --image-dir data/images --workers 4

Dependencies:
    pip install pillow pytesseract numpy

Optionally install Tesseract OCR engine:
    - Windows: https://github.com/UB-Mannheim/tesseract/wiki
    - Ensure the binary is on PATH or configure TESSDATA_PREFIX.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import pytesseract  # type: ignore
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "pytesseract is required. Install via `pip install pytesseract` and "
        "ensure the Tesseract OCR engine is available on your system."
    ) from exc


DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_CAPTION_ROOT = Path("data/captions")
DEFAULT_LOG_DIR = Path("logs/cleanup")
IMAGE_PATTERN = re.compile(r"\.(png|jpg|jpeg|bmp|tif|tiff)$", re.IGNORECASE)
EMBEDDED_PATTERN = re.compile(r"page_(\d{3})_img_(\d{2})", re.IGNORECASE)
VECTOR_PATTERN = re.compile(r"page_(\d{3})_vector_(\d{2})", re.IGNORECASE)
RENDER_PATTERN = re.compile(r"page_(\d{3})_render", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OCR captions for physics diagrams")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Root directory containing extracted images")
    parser.add_argument("--caption-dir", type=Path, default=DEFAULT_CAPTION_ROOT,
                        help="Destination directory for caption JSON files")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="Directory to store run logs")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads (use >1 for speed)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing caption files")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional limit on number of images to process")
    parser.add_argument("--language", default="eng",
                        help="Tesseract language code (default: eng)")
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"ocr_run_{timestamp}.log"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logging.info("Logging to %s", log_path)
    return log_path


def iter_image_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    for path in sorted(root.rglob("*")):
        if path.is_file() and IMAGE_PATTERN.search(path.name):
            yield path


FIGURE_REGEX = re.compile(r"figure\s*([0-9]+)", re.IGNORECASE)
AXIS_REGEX = re.compile(r"(?:axis|axes|time|distance|velocity|current|voltage|force)", re.IGNORECASE)
UNIT_REGEX = re.compile(r"\b(?:m|cm|mm|km|N|J|W|s|ms|kg|g|A|V|Hz|Pa|rad|°C)\b")
NUMERIC_REGEX = re.compile(r"[-+]?\d+(?:\.\d+)?")


def extract_metadata(text: str) -> dict:
    lower_text = text.lower()
    figure_refs = sorted({match.group(0) for match in FIGURE_REGEX.finditer(lower_text)})
    axis_terms = sorted({match.group(0) for match in AXIS_REGEX.finditer(lower_text)})
    units = sorted({match.group(0) for match in UNIT_REGEX.finditer(lower_text)})
    numeric_tokens = sorted({match.group(0) for match in NUMERIC_REGEX.finditer(text)})
    return {
        "figure_references": figure_refs,
        "axis_terms": axis_terms,
        "units": units,
        "numeric_tokens": numeric_tokens,
    }


def derive_image_context(image_path: Path, image_root: Path, size: Tuple[int, int]) -> dict:
    relative = image_path.relative_to(image_root)
    name = relative.name
    page_number: Optional[int] = None
    asset_index: Optional[int] = None
    asset_kind = "unknown"

    match = EMBEDDED_PATTERN.search(name)
    if match:
        page_number = int(match.group(1))
        asset_index = int(match.group(2))
        asset_kind = "embedded"
    else:
        match = VECTOR_PATTERN.search(name)
        if match:
            page_number = int(match.group(1))
            asset_index = int(match.group(2))
            asset_kind = "vector_clip"
        else:
            match = RENDER_PATTERN.search(name)
            if match:
                page_number = int(match.group(1))
                asset_kind = "page_render"

    width, height = size
    context = {
        "asset_kind": asset_kind,
        "page_number": page_number,
        "asset_index": asset_index,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 4) if height else None,
        "filesize_bytes": image_path.stat().st_size,
        "relative_path": str(relative),
    }
    return context


def image_to_text(image_path: Path, language: str) -> str:
    with Image.open(image_path) as img:
        # Convert to grayscale for better OCR results.
        gray = img.convert("L")
        arr = np.array(gray)
        # Adaptive threshold to reduce noise.
        arr = np.clip((arr > 200) * 255, 0, 255).astype(np.uint8)
        processed = Image.fromarray(arr)
        text = pytesseract.image_to_string(processed, lang=language)
    return text.strip()


@dataclass
class CaptionResult:
    image: str
    caption_path: str
    status: str
    message: str = ""


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_caption(image_rel: Path, caption_dir: Path, text: str, metadata: dict,
                  overwrite: bool) -> Path:
    caption_path = caption_dir / image_rel.with_suffix(".json")
    ensure_parent(caption_path)
    if caption_path.exists() and not overwrite:
        return caption_path
    payload = {
        "image": str(image_rel),
        "text": text,
        "metadata": metadata,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    caption_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return caption_path


def process_image(image_path: Path, image_root: Path, caption_root: Path,
                  language: str, overwrite: bool) -> CaptionResult:
    try:
        logging.debug("Processing image %s", image_path)
        relative = image_path.relative_to(image_root)
        with Image.open(image_path) as img:
            source_size = img.size
        text = image_to_text(image_path, language=language)
        metadata = extract_metadata(text)
        metadata.update(derive_image_context(image_path, image_root, source_size))
        caption_path = write_caption(relative, caption_root, text, metadata, overwrite)
        status = "ok"
        message = ""
        if not text:
            status = "warn"
            message = "OCR produced empty text"
        return CaptionResult(image=str(relative), caption_path=str(caption_path), status=status, message=message)
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed OCR for %s", image_path)
        return CaptionResult(image=str(image_path), caption_path="", status="error", message=str(exc))


def write_run_json(results: Iterable[CaptionResult], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"ocr_run_{timestamp}.jsonl"
    with json_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return json_path


def main() -> None:
    args = parse_args()

    log_path = configure_logging(args.log_dir)

    image_root: Path = args.image_dir
    caption_root: Path = args.caption_dir

    caption_root.mkdir(parents=True, exist_ok=True)

    images = list(iter_image_files(image_root))
    if args.max_files is not None:
        images = images[: args.max_files]

    if not images:
        logging.warning("No images found under %s", image_root)
        return

    logging.info("Found %s images to process", len(images))

    results: List[CaptionResult] = []
    workers = max(1, args.workers)
    if workers == 1 or len(images) == 1:
        for img_path in images:
            results.append(process_image(img_path, image_root, caption_root, args.language, args.overwrite))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(process_image, img_path, image_root, caption_root, args.language, args.overwrite): img_path
                for img_path in images
            }
            for future in as_completed(future_map):
                results.append(future.result())

    jsonl_path = write_run_json(results, args.log_dir)
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    err = sum(1 for r in results if r.status == "error")
    logging.info(
        "OCR complete | ok=%s warn=%s error=%s | log=%s | jsonl=%s",
        ok,
        warn,
        err,
        log_path,
        jsonl_path,
    )


if __name__ == "__main__":
    main()

