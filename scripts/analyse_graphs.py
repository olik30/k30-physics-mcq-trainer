"""Graph analysis pipeline for extracted exam diagrams.

This script detects chart-like images in `data/images/` (or a supplied root),
applies lightweight heuristics to confirm chart presence, merges related vector
clips emitted by `scripts/extract_pdf.py` into per-page composites, converts the
image into structured features (axis score, line density, chart-type guess,
sampled keypoints), and writes the metadata to `data/graph_analysis/`. Each
run produces a timestamped log (`artifacts/logs/graph_analysis/`) summarising successes,
warnings, and errors. Full-page fallback renders are only analysed when no
vector/embedded clips are available, keeping the focus on actual diagrams.

Example:

    python scripts/analyse_graphs.py --image-dir data/images --workers 2

Dependencies (install once):
    pip install pillow opencv-python numpy

Notes:
    - Chart features are generated with OpenCV heuristics. If you have a more
      specialised digitiser, adapt `digitise_chart` accordingly.
    - Chart detection is heuristic-based (Hough lines for axes + ratio checks);
      adjust thresholds if false positives/negatives are unacceptable.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Tuple

import cv2  # type: ignore
import numpy as np
from PIL import Image

MAX_ANALYSIS_DIM = 1800

DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_OUTPUT_ROOT = Path("data/graph_analysis")
DEFAULT_LOG_DIR = Path("artifacts/logs/graph_analysis")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
EMBEDDED_PATTERN = re.compile(r"page_(\d{3})_img_(\d{2})", re.IGNORECASE)
VECTOR_PATTERN = re.compile(r"page_(\d{3})_vector_(\d{2})", re.IGNORECASE)
RENDER_PATTERN = re.compile(r"page_(\d{3})_render", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse graphs and digitise charts")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Root directory containing extracted images")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT,
                        help="Destination directory for graph metadata JSON")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR,
                        help="Directory to store analysis logs")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of worker threads for digitisation")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing graph metadata files")
    parser.add_argument("--max-files", type=int, default=None,
                        help="Optional limit on number of images to process")
    parser.add_argument("--det-threshold", type=float, default=0.25,
                        help="Minimum axis-line score (0–1) to treat an image as a graph")
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"graph_run_{timestamp}.log"
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
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def detect_axis_score(image_path: Path) -> float:
    """Return heuristic score (0–1) indicating likelihood of chart axes."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 150)
    if lines is None:
        return 0.0
    vertical = 0
    horizontal = 0
    for rho, theta in lines[:, 0]:
        angle = theta * 180 / np.pi
        if abs(angle) < 10 or abs(angle - 180) < 10:
            vertical += 1
        elif abs(angle - 90) < 10:
            horizontal += 1
    if vertical == 0 or horizontal == 0:
        return 0.0
    total = len(lines)
    score = (vertical + horizontal) / max(total, 1)
    return float(min(score, 1.0))


def digitise_chart(image_path: Path) -> dict:
    """Derive rich geometric features from a candidate graph image."""

    color = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError("Unable to load image for digitisation")

    height, width = color.shape[:2]
    scale_factor = 1.0
    max_dim = max(height, width)
    if max_dim > MAX_ANALYSIS_DIM:
        scale_factor = MAX_ANALYSIS_DIM / float(max_dim)
        new_size = (int(round(width * scale_factor)), int(round(height * scale_factor)))
        color = cv2.resize(color, new_size, interpolation=cv2.INTER_AREA)
        height, width = color.shape[:2]

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

    # Basic contrast metrics
    mean_intensity = float(np.mean(gray))
    std_intensity = float(np.std(gray))

    # Edge map and density
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 80, 200)
    edge_pixels = int(np.count_nonzero(edges))
    edge_density = float(edge_pixels / max(height * width, 1))

    # Keypoints via Shi-Tomasi (cap to 250 points for JSON size)
    min_distance = max(3, min(width, height) // 120)
    corners = cv2.goodFeaturesToTrack(blur, maxCorners=300, qualityLevel=0.01, minDistance=min_distance)
    keypoints_set = set()
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < width and 0 <= yi < height:
                keypoints_set.add((xi, yi))
            if len(keypoints_set) >= 250:
                break
    keypoints = sorted(keypoints_set)

    # Axis orientation heuristics
    structure_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(2, height // 40)))
    structure_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(2, width // 40), 1))
    vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, structure_v)
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, structure_h)
    v_score = float(np.count_nonzero(vertical_lines) / max(height * width, 1))
    h_score = float(np.count_nonzero(horizontal_lines) / max(height * width, 1))

    # Line segments via Hough transform (probabilistic)
    min_length = int(max(width, height) * 0.08)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=max(20, min_length), maxLineGap=12)
    angle_bins: DefaultDict[int, int] = defaultdict(int)
    vertical_segments = 0
    horizontal_segments = 0
    diagonal_segments = 0
    line_lengths: List[float] = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0, :]:
            dx = x2 - x1
            dy = y2 - y1
            length = float(math.hypot(dx, dy))
            if length < max(width, height) * 0.03:
                continue
            angle = (math.degrees(math.atan2(-dy, dx)) + 360.0) % 180.0
            bin_key = int(round(angle / 5.0) * 5) % 180
            angle_bins[bin_key] += 1
            line_lengths.append(length)
            if 75 <= angle <= 105:
                vertical_segments += 1
            elif angle <= 15 or angle >= 165:
                horizontal_segments += 1
            else:
                diagonal_segments += 1

    dominant_angles = [
        {"angle": int(k), "count": int(v)}
        for k, v in sorted(angle_bins.items(), key=lambda item: item[1], reverse=True)[:6]
    ]

    # Component analysis using contours and morph shapes
    _, thresh_inv = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    triangle_like = 0
    rectangle_like = 0
    complex_polygons = 0
    component_boxes: List[Dict[str, int]] = []
    area_floor = (width * height) * 0.0005
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < area_floor:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        vertices = len(approx)
        x, y, w, h = cv2.boundingRect(contour)
        component_boxes.append({"x": int(x), "y": int(y), "width": int(w), "height": int(h)})
        if vertices == 3:
            triangle_like += 1
        elif vertices == 4:
            rectangle_like += 1
        elif vertices >= 8:
            complex_polygons += 1

    circles_detected = 0
    try:
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min(width, height) // 8, param1=120, param2=30)
        if circles is not None:
            circles_detected = int(circles.shape[1])
    except cv2.error:
        circles_detected = 0

    component_counts = {
        "triangle_like": int(triangle_like),
        "rectangle_like": int(rectangle_like),
        "complex_polygons": int(complex_polygons),
        "circle_like": int(circles_detected),
    }

    axis_presence = {
        "vertical_score": v_score,
        "horizontal_score": h_score,
        "has_vertical": v_score >= 0.0015,
        "has_horizontal": h_score >= 0.0015,
    }

    # Heuristic chart classification
    arrow_like = component_counts["triangle_like"]
    circle_like = component_counts["circle_like"]
    rectangle_like = component_counts["rectangle_like"]
    if axis_presence["has_horizontal"] and axis_presence["has_vertical"] and edge_density < 0.15:
        chart_type = "cartesian_graph"
    elif axis_presence["has_horizontal"] or axis_presence["has_vertical"]:
        chart_type = "single_axis_plot"
    elif circle_like >= 1 and arrow_like >= 1:
        chart_type = "circular_diagram"
    elif arrow_like >= 2:
        chart_type = "vector_diagram"
    elif rectangle_like >= 2:
        chart_type = "schematic_diagram"
    else:
        chart_type = "unknown"

    normalized_keypoints = [
        [round(x / width, 4), round(1.0 - (y / height), 4)]
        for x, y in keypoints
    ]

    line_stats = {
        "total_segments": int(len(line_lengths)),
        "mean_length": float(np.mean(line_lengths)) if line_lengths else 0.0,
        "max_length": float(max(line_lengths)) if line_lengths else 0.0,
        "vertical_segments": int(vertical_segments),
        "horizontal_segments": int(horizontal_segments),
        "diagonal_segments": int(diagonal_segments),
        "dominant_angles": dominant_angles,
    }

    return {
        "chart_type": chart_type,
        "image_size": {"width": int(width), "height": int(height)},
        "contrast": {"mean": mean_intensity, "std": std_intensity},
        "edge_density": edge_density,
        "keypoints": keypoints,
        "keypoints_normalized": normalized_keypoints,
        "axis_presence": axis_presence,
        "line_stats": line_stats,
        "component_counts": component_counts,
        "bounding_regions": component_boxes,
        "vertical_line_score": v_score,
        "horizontal_line_score": h_score,
        "processing": {"scale_factor": scale_factor},
    }


@dataclass
class GraphCandidate:
    display_path: Path
    metadata_rel_path: Path
    logical_image: str
    asset_kind: str
    source_images: List[str]
    fallback_path: Optional[Path] = None
    fallback_logical: Optional[str] = None


@dataclass
class GraphResult:
    image: str
    metadata_path: str
    asset_kind: str
    status: str
    message: str = ""


def classify_image(image_path: Path) -> Tuple[str, Optional[int], Optional[int]]:
    name = image_path.name
    match = VECTOR_PATTERN.search(name)
    if match:
        return "vector", int(match.group(1)), int(match.group(2))
    match = EMBEDDED_PATTERN.search(name)
    if match:
        return "embedded", int(match.group(1)), int(match.group(2))
    match = RENDER_PATTERN.search(name)
    if match:
        return "render", int(match.group(1)), None
    return "other", None, None


def default_assets() -> Dict[str, List[Tuple[int, Path]]]:
    return {"vector": [], "embedded": [], "render": []}


def create_single_candidate(image_path: Path, image_root: Path, asset_kind: str,
                           source_override: Optional[List[str]] = None,
                           metadata_override: Optional[Path] = None) -> GraphCandidate:
    relative = image_path.relative_to(image_root)
    metadata_rel = metadata_override or relative.with_suffix(".json")
    logical_image = str(relative)
    sources = source_override if source_override is not None else [logical_image]
    return GraphCandidate(
        display_path=image_path,
        metadata_rel_path=metadata_rel,
        logical_image=logical_image,
        asset_kind=asset_kind,
        source_images=sources,
    )


def merge_vector_clips(relative_parent: Path, page: int, vector_assets: List[Tuple[int, Path]],
                      image_root: Path, output_root: Path,
                      fallback_render: Optional[Path]) -> GraphCandidate:
    if not vector_assets:
        raise ValueError("No vector assets provided for merge")
    vectors_sorted = [path for _, path in sorted(vector_assets, key=lambda item: item[0])]
    pil_images: List[Image.Image] = []
    for path in vectors_sorted:
        with Image.open(path) as img:
            pil_images.append(img.convert("RGB"))
    margin = 12
    max_width = max(img.width for img in pil_images) + margin * 2
    total_height = margin * (len(pil_images) + 1) + sum(img.height for img in pil_images)
    composite = Image.new("RGB", (max_width, total_height), color="white")
    y_offset = margin
    for img in pil_images:
        x_offset = (max_width - img.width) // 2
        composite.paste(img, (x_offset, y_offset))
        y_offset += img.height + margin

    composite_dir = output_root / "__composites__" / relative_parent
    composite_dir.mkdir(parents=True, exist_ok=True)
    composite_name = f"page_{page:03d}_composite.png"
    composite_path = composite_dir / composite_name
    composite.save(composite_path)

    logical_image = str(Path("__composites__") / relative_parent / composite_name)
    metadata_rel = relative_parent / f"{Path(composite_name).stem}.json"
    source_images = [str(path.relative_to(image_root)) for path in vectors_sorted]
    fallback_logical = None
    if fallback_render is not None:
        fallback_logical = str(fallback_render.relative_to(image_root))

    return GraphCandidate(
        display_path=composite_path,
        metadata_rel_path=metadata_rel,
        logical_image=logical_image,
        asset_kind="vector_composite",
        source_images=source_images,
        fallback_path=fallback_render,
        fallback_logical=fallback_logical,
    )


def collect_candidates(image_root: Path, output_root: Path) -> List[GraphCandidate]:
    page_assets: DefaultDict[Tuple[Path, int], Dict[str, List[Tuple[int, Path]]]] = defaultdict(default_assets)
    others: List[Path] = []

    for image_path in iter_image_files(image_root):
        kind, page, index = classify_image(image_path)
        if kind in {"vector", "embedded", "render"} and page is not None:
            relative_parent = image_path.relative_to(image_root).parent
            assets = page_assets[(relative_parent, page)]
            assets[kind].append((index or 0, image_path))
        else:
            others.append(image_path)

    candidates: List[GraphCandidate] = []

    for (relative_parent, page), assets in sorted(page_assets.items(), key=lambda item: (item[0][0].as_posix(), item[0][1])):
        vector_assets = assets["vector"]
        render_assets = sorted(assets["render"], key=lambda item: item[0])
        fallback_render = render_assets[0][1] if render_assets else None
        if vector_assets:
            candidates.append(merge_vector_clips(relative_parent, page, vector_assets, image_root, output_root, fallback_render))

        embedded_assets = sorted(assets["embedded"], key=lambda item: item[0])
        if embedded_assets:
            for _, path in embedded_assets:
                candidates.append(create_single_candidate(path, image_root, "embedded"))
        elif not vector_assets and fallback_render is not None:
            candidates.append(create_single_candidate(fallback_render, image_root, "page_render"))

    for path in sorted(others, key=lambda p: p.relative_to(image_root).as_posix()):
        candidates.append(create_single_candidate(path, image_root, "other"))

    return candidates


def write_metadata(candidate: GraphCandidate, output_root: Path, metadata: dict,
                   overwrite: bool) -> Path:
    output_path = output_root / candidate.metadata_rel_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return output_path
    payload = {
        "image": candidate.logical_image,
        "asset_kind": candidate.asset_kind,
        "source_images": candidate.source_images,
        "graph_metadata": metadata,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def process_candidate(candidate: GraphCandidate, image_root: Path, output_root: Path,
                      threshold: float, overwrite: bool) -> GraphResult:
    try:
        logging.debug("Graph analysis: %s", candidate.display_path)
        score = detect_axis_score(candidate.display_path)
        if score < threshold:
            if candidate.asset_kind == "vector_composite" and candidate.fallback_path:
                fallback_sources = list(dict.fromkeys(candidate.source_images + ([candidate.fallback_logical] if candidate.fallback_logical else [])))
                fallback_candidate = GraphCandidate(
                    display_path=candidate.fallback_path,
                    metadata_rel_path=candidate.metadata_rel_path,
                    logical_image=candidate.fallback_logical or candidate.logical_image,
                    asset_kind="page_render_fallback",
                    source_images=fallback_sources,
                )
                fallback_result = process_candidate(fallback_candidate, image_root, output_root, threshold, overwrite)
                if fallback_result.status != "error":
                    message = f"Composite axis score {score:.2f}; used fallback render"
                    fallback_result.message = f"{message} | {fallback_result.message}".strip(" |")
                return fallback_result
            return GraphResult(
                image=candidate.logical_image,
                metadata_path="",
                asset_kind=candidate.asset_kind,
                status="skip",
                message=f"Axis score {score:.2f} below threshold",
            )

        if candidate.asset_kind not in {"embedded", "vector_composite", "page_render", "page_render_fallback"}:
            return GraphResult(
                image=candidate.logical_image,
                metadata_path="",
                asset_kind=candidate.asset_kind,
                status="skip",
                message="Asset not graph-like",
            )

        metadata = digitise_chart(candidate.display_path)
        keypoints = metadata.get("keypoints_normalized") or []
        axis_presence = metadata.get("axis_presence", {})

        if len(keypoints) < 6 and not (axis_presence.get("has_horizontal") or axis_presence.get("has_vertical")):
            return GraphResult(
                image=candidate.logical_image,
                metadata_path="",
                asset_kind=candidate.asset_kind,
                status="skip",
                message=f"Insufficient graph structure (keypoints={len(keypoints)})",
            )

        if len(keypoints) < 10:
            status = "warn"
            message = f"Sparse keypoints ({len(keypoints)})"
        else:
            status = "ok"
            message = ""
        metadata.setdefault("analysis", {})
        metadata["analysis"]["axis_detection_score"] = round(float(score), 4)
        metadata["analysis"]["used_fallback"] = candidate.asset_kind == "page_render_fallback"
        metadata["analysis"]["asset_kind"] = candidate.asset_kind

        metadata_path = write_metadata(candidate, output_root, metadata, overwrite)
        return GraphResult(
            image=candidate.logical_image,
            metadata_path=str(metadata_path),
            asset_kind=candidate.asset_kind,
            status=status,
            message=message,
        )
    except Exception as exc:  # pylint: disable=broad-except
        logging.exception("Graph analysis failed for %s", candidate.display_path)
        return GraphResult(
            image=candidate.logical_image,
            metadata_path="",
            asset_kind=candidate.asset_kind,
            status="error",
            message=str(exc),
        )


def write_run_json(results: Iterable[GraphResult], log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = log_dir / f"graph_run_{timestamp}.jsonl"
    with json_path.open("w", encoding="utf-8") as fh:
        for result in results:
            fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
    return json_path


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_dir)

    image_root: Path = args.image_dir
    output_root: Path = args.output_dir

    candidates = collect_candidates(image_root, output_root)
    if args.max_files is not None:
        candidates = candidates[: args.max_files]

    if not candidates:
        logging.warning("No graph candidates found under %s", image_root)
        return

    logging.info("Graph analysis candidates: %s", len(candidates))

    results: List[GraphResult] = []
    workers = max(1, args.workers)
    if workers == 1 or len(candidates) == 1:
        for candidate in candidates:
            results.append(process_candidate(candidate, image_root, output_root, args.det_threshold, args.overwrite))
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_map = {
                executor.submit(process_candidate, candidate, image_root, output_root, args.det_threshold, args.overwrite): candidate
                for candidate in candidates
            }
            for future in as_completed(future_map):
                results.append(future.result())

    jsonl_path = write_run_json(results, args.log_dir)
    ok = sum(1 for r in results if r.status == "ok")
    warn = sum(1 for r in results if r.status == "warn")
    skip = sum(1 for r in results if r.status == "skip")
    err = sum(1 for r in results if r.status == "error")
    logging.info(
        "Graph analysis complete | ok=%s warn=%s skip=%s error=%s | log=%s | jsonl=%s",
        ok,
        warn,
        skip,
        err,
        log_path,
        jsonl_path,
    )


if __name__ == "__main__":
    main()



