"""Batch extract text and images from exam PDFs.

This utility walks the `data/pdfs/` directory (or a supplied root), writes
per-page text files under `data/raw/`, exports embedded images to
`data/images/`, and records a JSONL log in `logs/extract/` summarising each
processed file.

Example:

    python scripts/extract_pdf.py --pdf-root data/pdfs --workers 1

Requirements:
    pip install pdfplumber pymupdf
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
from typing import Iterable, List

import pdfplumber  # type: ignore
import fitz  # PyMuPDF


DEFAULT_PDF_ROOT = Path("data/pdfs")
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_LOG_DIR = Path("logs/extract")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract text/images from PDFs")
    parser.add_argument("--pdf-root", type=Path, default=DEFAULT_PDF_ROOT,
                        help="Root directory containing board/paper PDF folders")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_ROOT,
                        help="Destination for per-page text outputs")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Destination for extracted images")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="Directory to store extraction logs")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional limit on number of PDFs to process")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of concurrent workers (set to 1 for serial)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing outputs instead of skipping")
    parser.add_argument("--include-data-sheets", action="store_true",
                        help="Process files whose names include 'data sheet'")
    parser.add_argument("--single-file", type=Path, default=None,
                        help="Process just one PDF (overrides directory walk)")
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"run_{timestamp}.log"
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )
    logging.info("Logging to %s", log_path)
    return log_path


def sanitise_segment(segment: str) -> str:
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", segment.strip())
    return slug.strip("._") or "unnamed"


def iter_pdf_files(root: Path, include_data_sheets: bool = False) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"PDF root does not exist: {root}")
    for path in sorted(root.rglob("*.pdf")):
        name_lower = path.name.lower()
        if not include_data_sheets and "data sheet" in name_lower:
            continue
        yield path


@dataclass
class ExtractionResult:
    pdf: str
    pages: int
    text_files: int
    images: int
    status: str
    message: str = ""


def ensure_parents(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def export_text(pdf_path: Path, pdf_root: Path, raw_root: Path,
                overwrite: bool) -> List[Path]:
    outputs: List[Path] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            relative = pdf_path.relative_to(pdf_root)
            segments = [sanitise_segment(seg) for seg in relative.parts[:-1]]
            segments.append(sanitise_segment(pdf_path.stem))
            out_path = raw_root.joinpath(*segments, f"page_{page_index:03d}.txt")
            ensure_parents(out_path)
            if out_path.exists() and not overwrite:
                continue
            out_path.write_text(text, encoding="utf-8")
            outputs.append(out_path)
    return outputs


def export_images(pdf_path: Path, pdf_root: Path, image_root: Path,
                  overwrite: bool) -> int:
    image_count = 0
    doc = fitz.open(pdf_path)
    try:
        for page_index, page in enumerate(doc, start=1):
            images = page.get_images(full=True)
            if not images:
                continue
            for img_index, img in enumerate(images, start=1):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                try:
                    if pix.alpha:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    relative = pdf_path.relative_to(pdf_root)
                    segments = [sanitise_segment(seg) for seg in relative.parts[:-1]]
                    segments.append(sanitise_segment(pdf_path.stem))
                    out_path = image_root.joinpath(
                        *segments,
                        f"page_{page_index:03d}_img_{img_index:02d}.png",
                    )
                    ensure_parents(out_path)
                    if out_path.exists() and not overwrite:
                        continue
                    pix.save(out_path)
                    image_count += 1
                finally:
                    pix = None  # free resources
    finally:
        doc.close()
    return image_count


def process_pdf(pdf_path: Path, pdf_root: Path, raw_root: Path,
                image_root: Path, overwrite: bool) -> ExtractionResult:
    try:
        logging.info("Processing %s", pdf_path)
        text_outputs = export_text(pdf_path, pdf_root, raw_root, overwrite)
        image_count = export_images(pdf_path, pdf_root, image_root, overwrite)
        pages = len(text_outputs)
        status = "ok"
        msg = ""
        if pages == 0:
            status = "warn"
            msg = "No text extracted"
        return ExtractionResult(
            pdf=str(pdf_path.relative_to(pdf_root)),
            pages=pages,
            text_files=len(text_outputs),
            images=image_count,
            status=status,
            message=msg,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Failed to process %s", pdf_path)
        return ExtractionResult(
            pdf=str(pdf_path.relative_to(pdf_root)),
            pages=0,
            text_files=0,
            images=0,
            status="error",
            message=str(exc),
        )


def write_jsonl(results: Iterable[ExtractionResult], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = log_dir / f"run_{timestamp}.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return jsonl_path


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_dir)

    pdf_root: Path = args.pdf_root
    raw_root: Path = args.raw_dir
    image_root: Path = args.image_dir

    raw_root.mkdir(parents=True, exist_ok=True)
    image_root.mkdir(parents=True, exist_ok=True)

    if args.single_file:
        pdfs = [args.single_file]
    else:
        pdf_iter = iter_pdf_files(pdf_root, include_data_sheets=args.include_data_sheets)
        pdfs = list(pdfs if (limit := args.max_files) is None else
                    [p for _, p in zip(range(limit), pdf_iter)])
        if not pdfs:
            logging.warning("No PDFs found under %s", pdf_root)
            return

    results: List[ExtractionResult] = []
    workers = max(1, args.workers)
    if workers == 1 or len(pdfs) == 1:
        for pdf_path in pdfs:
            results.append(process_pdf(pdf_path, pdf_root, raw_root, image_root, args.overwrite))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(process_pdf, pdf_path, pdf_root, raw_root, image_root, args.overwrite): pdf_path
                for pdf_path in pdfs
            }
            for future in as_completed(future_map):
                results.append(future.result())

    jsonl_path = write_jsonl(results, args.log_dir)
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    err = sum(1 for r in results if r.status == "error")
    logging.info("Extraction complete | ok=%s warn=%s error=%s | log=%s | jsonl=%s",
                 ok, warn, err, log_path, jsonl_path)


if __name__ == "__main__":
    main()

