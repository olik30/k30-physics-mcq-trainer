"""Batch extract text and images from exam PDFs.

This utility walks the `data/pdfs/` directory (or a supplied root), writes
per-page text files under `data/raw/`, exports embedded images to
`data/images/`, and records a JSONL log in `artifacts/logs/extract/` summarising each
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

# Allow very large exam renders without Pillow's decompression bomb guard.
Image.MAX_IMAGE_PIXELS = None
import numpy as np  # type: ignore
import cv2  # type: ignore


DEFAULT_PDF_ROOT = Path("data/pdfs")
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_LOG_DIR = Path("artifacts/logs/extract")

DEFAULT_RENDER_DPI = 600
DEFAULT_MIN_RASTER_AREA = 2048
DEFAULT_MIN_RASTER_CONTRAST = 8.0
DEFAULT_DRAWING_MIN_AREA = 5000
DEFAULT_DRAWING_MARGIN = 5.0
DEFAULT_DRAWING_CLUSTER_GAP = 12.0
DEFAULT_DRAWING_GAP_RATIO = 0.04
DEFAULT_LOGO_MAX_AREA_RATIO = 0.02
DEFAULT_LOGO_MIN_EDGE_RATIO = 0.4
DEFAULT_LOGO_SOLIDITY = 0.92
DEFAULT_RASTER_OUTPUT_DPI = 450
DEFAULT_MAX_RECURSION = 100000
DEFAULT_TEXT_ENGINE = "pdfplumber"


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
    parser.add_argument("--drawing-gap-ratio", type=float, default=DEFAULT_DRAWING_GAP_RATIO,
                        help="Optional page-size multiplier for drawing cluster gap (set 0 to disable)")
    parser.add_argument("--logo-max-area-ratio", type=float, default=DEFAULT_LOGO_MAX_AREA_RATIO,
                        help="Maximum page-area ratio for raster logos to auto-drop (0 disables)")
    parser.add_argument("--logo-min-edge-ratio", type=float, default=DEFAULT_LOGO_MIN_EDGE_RATIO,
                        help="Minimum aspect ratio edge length (relative to page) to flag logos")
    parser.add_argument("--logo-solidity", type=float, default=DEFAULT_LOGO_SOLIDITY,
                        help="Solidity threshold (0-1) for detecting solid logo blobs when rendering")
    parser.add_argument("--enable-vector-clips", action="store_true",
                        help="Opt-in: render vector drawing clusters in addition to full-page fallbacks")
    parser.add_argument("--board-depth", type=int, default=1,
                        help="Directory depth under pdf_root that identifies the board (default 1)")
    parser.add_argument("--raster-output-dpi", type=int, default=DEFAULT_RASTER_OUTPUT_DPI,
                        help="When >0, rescale embedded rasters using this DPI before saving")
    parser.add_argument("--no-render-rescale", action="store_true",
                        help="Keep fallback renders at native size (ignore render_dpi scaling)")
    parser.add_argument("--max-recursion", type=int, default=DEFAULT_MAX_RECURSION,
                        help="Recursion limit passed to pdfplumber/PyMuPDF for deep-nested PDFs")
    parser.add_argument("--text-engine", choices=("auto", "pdfplumber", "pymupdf"), default=DEFAULT_TEXT_ENGINE,
                        help="Which text extractor to use: auto=pdfplumber with PyMuPDF fallback")
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


def board_relative(pdf_path: Path, pdf_root: Path, board_depth: int) -> Tuple[str, Path]:
    relative = pdf_path.relative_to(pdf_root)
    parts = relative.parts
    board = parts[0] if parts else "unknown_board"
    if board_depth > 1 and len(parts) >= board_depth:
        board = parts[board_depth - 1]
    board_slug = sanitise_segment(board)
    relative_subpath = Path(*[sanitise_segment(seg) for seg in parts[:board_depth]]) if len(parts) >= board_depth else Path(board_slug)
    return board_slug, relative_subpath


def evaluate_contrast(image_bytes: bytes) -> float:
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            if not stat.stddev:
                return 0.0
            return float(stat.stddev[0])
    except Exception:  # pylint: disable=broad-except
        return 0.0


def raster_is_logo(image_bytes: bytes, width: int, height: int,
                   page_width: float, page_height: float,
                   max_area_ratio: float, min_edge_ratio: float,
                   solidity_threshold: float) -> bool:
    if max_area_ratio <= 0:
        return False
    area = width * height
    page_area = page_width * page_height
    if area == 0 or page_area == 0:
        return False
    if (area / page_area) > max_area_ratio:
        return False
    longest_edge = max(width, height) / max(page_width, page_height)
    if longest_edge < min_edge_ratio:
        return False
    try:
        data = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        total_area = sum(cv2.contourArea(c) for c in contours)
        hull = cv2.convexHull(np.vstack(contours))
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            return False
        solidity = total_area / hull_area
        return solidity >= solidity_threshold
    except Exception:
        return False


def compute_drawing_cluster_gap(page_width: float, page_height: float,
                                base_gap: float, gap_ratio: float) -> float:
    if gap_ratio <= 0:
        return base_gap
    return max(base_gap, min(page_width, page_height) * gap_ratio)


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
                overwrite: bool, max_recursion: int, board_depth: int,
                engine: str) -> tuple[list[Path], int, int]:
    if engine == "pymupdf":
        return export_text_pymupdf(pdf_path, pdf_root, raw_root, overwrite, board_depth)

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
                    segments = [sanitise_segment(seg) for seg in relative.parts[:board_depth]]
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
            if engine == "pdfplumber":
                raise
            logging.warning("pdfplumber failed for %s (%s), falling back to PyMuPDF text extraction", pdf_path, exc)
            return export_text_pymupdf(pdf_path, pdf_root, raw_root, overwrite, board_depth)
    finally:
        if sys.getrecursionlimit() != original_limit:
            try:
                sys.setrecursionlimit(original_limit)
            except RecursionError:
                pass


def export_text_pymupdf(pdf_path: Path, pdf_root: Path, raw_root: Path,
                        overwrite: bool, board_depth: int) -> tuple[list[Path], int, int]:
    outputs: List[Path] = []
    skipped = 0
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
        for page_index in range(total_pages):
            page = doc.load_page(page_index)
            text = page.get_text("text") or ""
            relative = pdf_path.relative_to(pdf_root)
            segments = [sanitise_segment(seg) for seg in relative.parts[:board_depth]]
            segments.append(sanitise_segment(pdf_path.stem))
            out_path = raw_root.joinpath(*segments, f"page_{page_index + 1:03d}.txt")
            ensure_parents(out_path)
            if out_path.exists() and not overwrite:
                skipped += 1
                continue
            out_path.write_text(text, encoding="utf-8")
            outputs.append(out_path)
    return outputs, total_pages, skipped


def rescale_raster(image_bytes: bytes, target_dpi: int) -> bytes:
    if target_dpi <= 0:
        return image_bytes
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if hasattr(img, "info"):
                original_dpi = img.info.get("dpi")
                if original_dpi and original_dpi[0] >= target_dpi:
                    return image_bytes
            scale = target_dpi / 72.0
            new_size = (int(round(img.width * scale)), int(round(img.height * scale)))
            resized = img.resize(new_size, Image.LANCZOS)
            output = io.BytesIO()
            resized.save(output, format="PNG", dpi=(target_dpi, target_dpi))
            return output.getvalue()
    except Exception:
        return image_bytes


def export_images(pdf_path: Path, pdf_root: Path, image_root: Path,
                  overwrite: bool, render_dpi: int,
                  min_raster_area: int,
                  min_raster_contrast: float,
                  drawing_min_area: float,
                  drawing_margin: float,
                  drawing_cluster_gap: float,
                  drawing_gap_ratio: float,
                  logo_max_area_ratio: float,
                  logo_min_edge_ratio: float,
                  logo_solidity: float,
                  enable_vector_clips: bool,
                  board_depth: int,
                  raster_output_dpi: int,
                  no_render_rescale: bool) -> tuple[int, int]:
    image_count = 0
    fallback_count = 0
    doc = fitz.open(pdf_path)
    try:
        relative = pdf_path.relative_to(pdf_root)
        segments = [sanitise_segment(seg) for seg in relative.parts[:board_depth]]
        segments.append(sanitise_segment(pdf_path.stem))
        out_dir = image_root.joinpath(*segments)
        ensure_parents(out_dir / "placeholder")
        if (out_dir / "placeholder").exists():
            (out_dir / "placeholder").unlink(missing_ok=True)

        for page_index, page in enumerate(doc, start=1):
            page_saved_raster = False
            page_has_meaningful_raster = False
            images = page.get_images(full=True)
            page_width = page.rect.width
            page_height = page.rect.height
            draw_boxes: List[Tuple[float, float, float, float]] = []
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
                        page_saved_raster = True
                        page_has_meaningful_raster = True
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

                shorter_edge = min(width, height)
                longer_edge = max(width, height)
                aspect_ratio = (longer_edge / shorter_edge) if shorter_edge else float("inf")
                if shorter_edge < 160 and aspect_ratio >= 6.0:
                    logging.debug("Skipping narrow raster (size=%sx%s, aspect=%.2f) from %s page %s", width, height, aspect_ratio, pdf_path, page_index)
                    continue
                if shorter_edge <= 800 and aspect_ratio >= 12.0:
                    logging.debug("Skipping extreme aspect raster (size=%sx%s, aspect=%.2f) from %s page %s", width, height, aspect_ratio, pdf_path, page_index)
                    continue
                if raster_is_logo(image_bytes, width, height, page_width, page_height,
                                  logo_max_area_ratio, logo_min_edge_ratio, logo_solidity):
                    logging.debug("Dropping probable logo raster from %s page %s", pdf_path, page_index)
                    continue
                contrast = evaluate_contrast(image_bytes)
                if contrast < min_raster_contrast:
                    logging.debug("Skipping low-contrast raster (contrast=%.2f) from %s page %s", contrast, pdf_path, page_index)
                    continue

                image_bytes = rescale_raster(image_bytes, raster_output_dpi)

                with out_path.open("wb") as fh:
                    fh.write(image_bytes)
                page_saved_raster = True
                page_has_meaningful_raster = True
                image_count += 1

            page_drawings = page.get_drawings()
            gap_limit = compute_drawing_cluster_gap(page_width, page_height, drawing_cluster_gap, drawing_gap_ratio)
            for item in page_drawings:
                rect = item.get("rect")
                if not rect:
                    continue
                bbox = (rect.x0, rect.y0, rect.x1, rect.y1)
                if box_area(bbox) < drawing_min_area:
                    continue
                draw_boxes.append(bbox)

            if enable_vector_clips and draw_boxes:
                clusters: List[List[Tuple[float, float, float, float]]] = []
                for bbox in sorted(draw_boxes):
                    matched = False
                    for cluster in clusters:
                        last_box = cluster[-1]
                        last_center = box_center(last_box)
                        current_center = box_center(bbox)
                        gap = math.dist(last_center, current_center)
                        if gap <= gap_limit:
                            cluster.append(bbox)
                            matched = True
                            break
                    if not matched:
                        clusters.append([bbox])

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
                            page_saved_raster = True
                            page_has_meaningful_raster = True
                            continue
                    try:
                        matrix = fitz.Matrix(render_dpi / 72, render_dpi / 72)
                        pix = page.get_pixmap(matrix=matrix, alpha=False, clip=expanded)
                        pix.save(render_path)
                        image_count += 1
                        logging.info("Rendered drawing clip for %s page %s cluster %s", pdf_path, page_index, cluster_index)
                        page_saved_raster = True
                        page_has_meaningful_raster = True
                    except Exception as err:  # pylint: disable=broad-except
                        if render_path.exists():
                            render_path.unlink()
                        logging.error("Failed drawing clip render for %s page %s cluster %s: %s",
                                      pdf_path, page_index, cluster_index, err)

            should_render_fallback = (not page_has_meaningful_raster) or bool(draw_boxes)
            if should_render_fallback:
                render_path = out_dir / f"page_{page_index:03d}_render.png"
                if render_path.exists() and not overwrite:
                    if render_path.stat().st_size == 0:
                        render_path.unlink()
                    else:
                        fallback_count += 1
                        continue
                try:
                    if no_render_rescale:
                        pix = page.get_pixmap()
                    else:
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
                drawing_gap_ratio: float,
                logo_max_area_ratio: float,
                logo_min_edge_ratio: float,
                logo_solidity: float,
                max_recursion: int,
                enable_vector_clips: bool,
                board_depth: int,
                raster_output_dpi: int,
                no_render_rescale: bool,
                text_engine: str) -> ExtractionResult:
    try:
        logging.info("Processing %s", pdf_path)
        text_outputs, total_pages, skipped_pages = export_text(pdf_path, pdf_root, raw_root, overwrite, max_recursion, board_depth, text_engine)
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
            drawing_gap_ratio,
            logo_max_area_ratio,
            logo_min_edge_ratio,
            logo_solidity,
            enable_vector_clips,
            board_depth,
            raster_output_dpi,
            no_render_rescale,
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
                args.drawing_gap_ratio,
                args.logo_max_area_ratio,
                args.logo_min_edge_ratio,
                args.logo_solidity,
                args.max_recursion,
                args.enable_vector_clips,
                args.board_depth,
                args.raster_output_dpi,
                args.no_render_rescale,
                args.text_engine,
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
                    args.drawing_gap_ratio,
                    args.logo_max_area_ratio,
                    args.logo_min_edge_ratio,
                    args.logo_solidity,
                    args.max_recursion,
                    args.enable_vector_clips,
                    args.board_depth,
                    args.raster_output_dpi,
                    args.no_render_rescale,
                    args.text_engine,
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

