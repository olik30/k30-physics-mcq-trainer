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
import io
import json
import logging
import math
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple

import pdfplumber  # type: ignore
import fitz  # PyMuPDF
from PIL import Image, ImageStat  # type: ignore


DEFAULT_PDF_ROOT = Path("data/pdfs")
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_LOG_DIR = Path("logs/extract")

DEFAULT_RENDER_DPI = 450
DEFAULT_MIN_RASTER_AREA = 2048
DEFAULT_MIN_RASTER_CONTRAST = 8.0
DEFAULT_DRAWING_MIN_AREA = 5000
DEFAULT_DRAWING_MARGIN = 5.0
DEFAULT_DRAWING_CLUSTER_GAP = 12.0


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
    parser.add_argument("--include-ma", action="store_true",
                        help="Process PDFs whose filename contains standalone 'MA' (model answers)")
    parser.add_argument("--single-file", type=Path, default=None,
                        help="Process just one PDF (overrides directory walk)")
    parser.add_argument("--render-dpi", type=int, default=DEFAULT_RENDER_DPI,
                        help="DPI for fallback/drawing renders")
    parser.add_argument("--min-raster-area", type=int, default=DEFAULT_MIN_RASTER_AREA,
                        help="Minimum pixel area for embedded raster images to keep")
    parser.add_argument("--min-raster-contrast", type=float, default=DEFAULT_MIN_RASTER_CONTRAST,
                        help="Minimum grayscale std-dev for embedded rasters (filters monotone strips)")
    parser.add_argument("--drawing-min-area", type=float, default=DEFAULT_DRAWING_MIN_AREA,
                        help="Minimum area (px^2) for drawing clusters to render")
    parser.add_argument("--drawing-margin", type=float, default=DEFAULT_DRAWING_MARGIN,
                        help="Margin (px) to pad around drawing bounding boxes before clipping")
    parser.add_argument("--drawing-cluster-gap", type=float, default=DEFAULT_DRAWING_CLUSTER_GAP,
                        help="Maximum gap (px) between drawing items to treat as same cluster")
    parser.add_argument("--max-recursion", type=int, default=20000,
                        help="Recursion limit passed to pdfminer to avoid depth errors")
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


def evaluate_contrast(image_bytes: bytes) -> float:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            # If image is empty, stddev may be zero-length list
            if not stat.stddev:
                return 0.0
            return float(stat.stddev[0])
    except Exception:  # pylint: disable=broad-except
        return 0.0


def box_area(bbox: Tuple[float, float, float, float]) -> float:
    x0, y0, x1, y1 = bbox
    return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))


def expand_box(bbox: Tuple[float, float, float, float], margin: float,
               width: float, height: float) -> Tuple[float, float, float, float]:
    x0, y0, x1, y1 = bbox
    return (
        max(0.0, x0 - margin),
        max(0.0, y0 - margin),
        min(width, x1 + margin),
        min(height, y1 + margin),
    )


def merge_boxes(boxes: List[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    return x0, y0, x1, y1


def box_center(bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = bbox
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def iter_pdf_files(root: Path, include_data_sheets: bool = False, include_ma: bool = False) -> Iterable[Path]:
    if not root.exists():
        raise FileNotFoundError(f"PDF root does not exist: {root}")
    for path in sorted(root.rglob("*.pdf")):
        name_lower = path.name.lower()
        if not include_data_sheets and "data sheet" in name_lower:
            continue
        if not include_ma:
            stem = path.stem.lower()
            if re.search(r"\bma\b", stem):
                logging.debug("Skipping MA PDF: %s", path)
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
                overwrite: bool, max_recursion: int) -> tuple[list[Path], int, int]:
    outputs: List[Path] = []
    skipped = 0
    original_limit = sys.getrecursionlimit()
    if max_recursion > original_limit:
        sys.setrecursionlimit(max_recursion)
    try:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                for page_index, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text() or ""
                    relative = pdf_path.relative_to(pdf_root)
                    segments = [sanitise_segment(seg) for seg in relative.parts[:-1]]
                    segments.append(sanitise_segment(pdf_path.stem))
                    out_path = raw_root.joinpath(*segments, f"page_{page_index:03d}.txt")
                    ensure_parents(out_path)
                    if out_path.exists() and not overwrite:
                        skipped += 1
                        continue
                    out_path.write_text(text, encoding="utf-8")
                    outputs.append(out_path)
                return outputs, total_pages, skipped
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning("pdfplumber failed for %s (%s), falling back to PyMuPDF text extraction", pdf_path, exc)
            return export_text_pymupdf(pdf_path, pdf_root, raw_root, overwrite)
    finally:
        if sys.getrecursionlimit() != original_limit:
            try:
                sys.setrecursionlimit(original_limit)
            except RecursionError:
                pass


def export_text_pymupdf(pdf_path: Path, pdf_root: Path, raw_root: Path,
                        overwrite: bool) -> tuple[list[Path], int, int]:
    outputs: List[Path] = []
    skipped = 0
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            relative = pdf_path.relative_to(pdf_root)
            segments = [sanitise_segment(seg) for seg in relative.parts[:-1]]
            segments.append(sanitise_segment(pdf_path.stem))
            out_path = raw_root.joinpath(*segments, f"page_{page_index + 1:03d}.txt")
            ensure_parents(out_path)
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            out_path.write_text(text, encoding="utf-8")
            outputs.append(out_path)
    return outputs, total_pages, skipped


def export_images(pdf_path: Path, pdf_root: Path, image_root: Path,
                  overwrite: bool, render_dpi: int,
                  min_raster_area: int,
                  min_raster_contrast: float,
                  drawing_min_area: float,
                  drawing_margin: float,
                  drawing_cluster_gap: float) -> tuple[int, int]:
    image_count = 0
    fallback_count = 0
    doc = fitz.open(pdf_path)
    try:
        relative = pdf_path.relative_to(pdf_root)
        segments = [sanitise_segment(seg) for seg in relative.parts[:-1]]
        segments.append(sanitise_segment(pdf_path.stem))
        out_dir = image_root.joinpath(*segments)
        ensure_parents(out_dir / "placeholder")
        if (out_dir / "placeholder").exists():
            (out_dir / "placeholder").unlink(missing_ok=True)

        for page_index, page in enumerate(doc, start=1):
            page_had_raster = False
            images = page.get_images(full=True)
            for img_index, img in enumerate(images, start=1):
                xref = img[0]
                image_info = doc.extract_image(xref)
                image_bytes = image_info.get("image") if image_info else None
                ext = (image_info.get("ext") if image_info else "png").lower()
                out_path = out_dir / f"page_{page_index:03d}_img_{img_index:02d}.{ext}"

                if out_path.exists() and not overwrite:
                    if out_path.stat().st_size == 0:
                        out_path.unlink()
                    else:
                        page_had_raster = True
                        continue

                if not image_bytes or len(image_bytes) == 0:
                    if out_path.exists():
                        out_path.unlink()
                    logging.warning("Missing or empty image payload (xref=%s) in %s", xref, pdf_path)
                    continue

                width = image_info.get("width", 0)
                height = image_info.get("height", 0)
                area = width * height
                if area < min_raster_area:
                    logging.debug("Skipping small raster (area=%s) from %s page %s", area, pdf_path, page_index)
                    continue
                contrast = evaluate_contrast(image_bytes)
                if contrast < min_raster_contrast:
                    logging.debug("Skipping low-contrast raster (contrast=%.2f) from %s page %s", contrast, pdf_path, page_index)
                    continue

                with out_path.open("wb") as fh:
                    fh.write(image_bytes)
                page_had_raster = True
                image_count += 1

            page_drawings = page.get_drawings()
            draw_boxes: List[Tuple[float, float, float, float]] = []
            for item in page_drawings:
                rect = item.get("rect")
                if not rect:
                    continue
                bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                if box_area(bbox) < drawing_min_area:
                    continue
                draw_boxes.append(bbox)

            clusters: List[List[Tuple[float, float, float, float]]] = []
            for bbox in sorted(draw_boxes):
                matched = False
                for cluster in clusters:
                    last_box = cluster[-1]
                    last_center = box_center(last_box)
                    current_center = box_center(bbox)
                    gap = math.dist(last_center, current_center)
                    if gap <= drawing_cluster_gap:
                        cluster.append(bbox)
                        matched = True
                        break
                if not matched:
                    clusters.append([bbox])

            page_width = page.rect.width
            page_height = page.rect.height
            for cluster_index, cluster in enumerate(clusters, start=1):
                merged = merge_boxes(cluster)
                expanded = expand_box(merged, drawing_margin, page_width, page_height)
                if box_area(expanded) < drawing_min_area:
                    continue
                render_path = out_dir / f"page_{page_index:03d}_vector_{cluster_index:02d}.png"
                if render_path.exists() and not overwrite:
                    if render_path.stat().st_size == 0:
                        render_path.unlink()
                    else:
                        page_had_raster = True
                        continue
                try:
                    matrix = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                    pix = page.get_pixmap(matrix=matrix, alpha=False, clip=expanded)
                    pix.save(render_path)
                    image_count += 1
                    logging.info("Rendered drawing clip for %s page %s cluster %s", pdf_path, page_index, cluster_index)
                    page_had_raster = True
                except Exception as err:  # pylint: disable=broad-except
                    if render_path.exists():
                        render_path.unlink()
                    logging.error("Failed drawing clip render for %s page %s cluster %s: %s",
                                  pdf_path, page_index, cluster_index, err)

            if not page_had_raster:
                render_path = out_dir / f"page_{page_index:03d}_render.png"
                if render_path.exists() and not overwrite:
                    if render_path.stat().st_size == 0:
                        render_path.unlink()
                    else:
                        fallback_count += 1
                        continue
                try:
                    zoom = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                    pix = page.get_pixmap(matrix=zoom, alpha=False)
                    pix.save(render_path)
                    fallback_count += 1
                    image_count += 1
                    logging.info("Rendered fallback image for %s page %s", pdf_path, page_index)
                except Exception as err:  # pylint: disable=broad-except
                    if render_path.exists():
                        render_path.unlink()
                    logging.error("Failed to render fallback for %s page %s: %s", pdf_path, page_index, err)
    finally:
        doc.close()
    return image_count, fallback_count


def process_pdf(pdf_path: Path, pdf_root: Path, raw_root: Path,
                image_root: Path, overwrite: bool,
                render_dpi: int,
                min_raster_area: int,
                min_raster_contrast: float,
                drawing_min_area: float,
                drawing_margin: float,
                drawing_cluster_gap: float,
                max_recursion: int) -> ExtractionResult:
    try:
        logging.info("Processing %s", pdf_path)
        text_outputs, total_pages, skipped_pages = export_text(pdf_path, pdf_root, raw_root, overwrite, max_recursion)
        image_count, fallback_count = export_images(
            pdf_path,
            pdf_root,
            image_root,
            overwrite,
            render_dpi,
            min_raster_area,
            min_raster_contrast,
            drawing_min_area,
            drawing_margin,
            drawing_cluster_gap,
        )
        status = "ok"
        message_parts = []
        if total_pages == 0:
            status = "warn"
            message_parts.append("No pages detected")
        elif skipped_pages and len(text_outputs) == 0 and not overwrite:
            message_parts.append(f"Skipped (already extracted {skipped_pages} pages)")
        if fallback_count:
            message_parts.append(f"Rendered {fallback_count} page fallback(s)")
        msg = " ".join(message_parts)
        return ExtractionResult(
            pdf=str(pdf_path.relative_to(pdf_root)),
            pages=total_pages,
            text_files=max(len(text_outputs), total_pages),
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
        pdf_iter = iter_pdf_files(
            pdf_root,
            include_data_sheets=args.include_data_sheets,
            include_ma=args.include_ma,
        )
        if args.max_files is None:
            pdfs = list(pdf_iter)
        else:
            pdfs = []
            for _, path in zip(range(args.max_files), pdf_iter):
                pdfs.append(path)
        if not pdfs:
            logging.warning("No PDFs found under %s", pdf_root)
            return

    results: List[ExtractionResult] = []
    workers = max(1, args.workers)
    if workers == 1 or len(pdfs) == 1:
        for pdf_path in pdfs:
            results.append(process_pdf(
                pdf_path,
                pdf_root,
                raw_root,
                image_root,
                args.overwrite,
                args.render_dpi,
                args.min_raster_area,
                args.min_raster_contrast,
                args.drawing_min_area,
                args.drawing_margin,
                args.drawing_cluster_gap,
                args.max_recursion,
            ))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(
                    process_pdf,
                    pdf_path,
                    pdf_root,
                    raw_root,
                    image_root,
                    args.overwrite,
                    args.render_dpi,
                    args.min_raster_area,
                    args.min_raster_contrast,
                    args.drawing_min_area,
                    args.drawing_margin,
                    args.drawing_cluster_gap,
                    args.max_recursion,
                ): pdf_path
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

