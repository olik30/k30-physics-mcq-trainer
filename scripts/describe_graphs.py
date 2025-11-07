"""Generate natural-language descriptions for analysed graphs.

This script reads the metadata emitted by `scripts/analyse_graphs.py`,
optionally combines it with OCR captions, and writes human-readable
descriptions plus structured observations back into the graph JSON files.
It favours a data-first summary (keypoints, detected axes, component counts)
but can call out to optional third-party digitisers when available.

Example:

    python scripts/describe_graphs.py \
        --graph-json-dir data/captions/graphs/AQA \
        --caption-dir data/captions/AQA \
        --image-dir data/images/AQA

Dependencies (core):
    pip install pillow opencv-python numpy

Optional (richer descriptions when installed):
    pip install chartocr  # https://github.com/TencentARC/ChartOCR
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


DEFAULT_GRAPH_JSON_DIR = Path("data/graph_analysis")
DEFAULT_OUTPUT_ROOT = Path("data/graph_descriptions")
DEFAULT_CAPTION_DIR = Path("data/captions")
DEFAULT_IMAGE_ROOT = Path("data/images")
DEFAULT_LOG_DIR = Path("artifacts/logs/graph_descriptions")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add textual descriptions to analysed graph metadata"
    )
    parser.add_argument(
        "--graph-json-dir",
        type=Path,
        default=DEFAULT_GRAPH_JSON_DIR,
        help="Directory containing graph metadata JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Destination directory for enriched graph JSON files",
    )
    parser.add_argument(
        "--caption-dir",
        type=Path,
        default=DEFAULT_CAPTION_DIR,
        help="Directory containing OCR caption JSON files (for context)",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help="Root directory containing extracted images",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory to write description run logs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of worker processes (currently informational)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing graph_summary blocks",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional limit on number of graph JSON files to process",
    )
    parser.add_argument(
        "--chartocr",
        action="store_true",
        help="Attempt to use ChartOCR (if installed) for richer extraction",
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Write compact summary-only JSON instead of cloning full metadata",
    )
    return parser.parse_args()


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"describe_run_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.info("Logging to %s", log_path)
    return log_path


def iter_graph_json(graph_dir: Path) -> Iterable[Path]:
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph JSON directory does not exist: {graph_dir}")
    for path in sorted(graph_dir.rglob("*.json")):
        if path.is_file():
            yield path


def load_caption_text(caption_dir: Path, image_rel: str) -> Optional[str]:
    caption_path = caption_dir / Path(image_rel).with_suffix(".json")
    if not caption_path.exists():
        return None
    try:
        payload = json.loads(caption_path.read_text(encoding="utf-8"))
        return payload.get("text")
    except Exception:  # pylint: disable=broad-except
        return None


def attempt_chartocr(image_path: Path) -> Optional[Dict[str, str]]:
    """Call ChartOCR if available, returning a concise summary dict."""

    try:
        from chartocr import ChartOCR  # type: ignore
    except Exception:  # pylint: disable=broad-except
        return None

    try:
        predictor = ChartOCR()
        result = predictor(str(image_path))
    except Exception as exc:  # pylint: disable=broad-except
        logging.warning("ChartOCR failed for %s: %s", image_path, exc)
        return {"error": str(exc)}

    summary = {
        "chart_type": result.get("chart_type"),
        "x_axis": result.get("x_axis"),
        "y_axis": result.get("y_axis"),
        "legends": result.get("legend_text", []),
        "data_series": result.get("data_series", []),
    }
    return summary


def regression_summary(points: List[Tuple[float, float]]) -> Dict[str, float]:
    if len(points) < 6:
        return {
            "slope": 0.0,
            "variance": 0.0,
            "trend_strength": 0.0,
        }

    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]
    slope = float(np.polyfit(xs, ys, 1)[0])
    variance = float(np.var(ys))
    trend_strength = float(abs(slope) * (variance ** 0.5))
    return {
        "slope": slope,
        "variance": variance,
        "trend_strength": trend_strength,
    }


def detect_peaks(points: List[Tuple[float, float]], min_prominence: float = 0.05) -> Dict[str, List[float]]:
    if len(points) < 10:
        return {"maxima": [], "minima": []}
    xs = np.array([p[0] for p in points])
    ys = np.array([p[1] for p in points])
    order = np.argsort(xs)
    ys_sorted = ys[order]
    maxima: List[float] = []
    minima: List[float] = []
    for idx in range(1, len(ys_sorted) - 1):
        prev_val, curr_val, next_val = ys_sorted[idx - 1 : idx + 2]
        if curr_val > prev_val and curr_val > next_val:
            if curr_val - min(prev_val, next_val) >= min_prominence:
                maxima.append(float(curr_val))
        if curr_val < prev_val and curr_val < next_val:
            if max(prev_val, next_val) - curr_val >= min_prominence:
                minima.append(float(curr_val))
    return {"maxima": maxima, "minima": minima}


def component_summary(component_counts: Dict[str, int]) -> List[str]:
    notes: List[str] = []
    if component_counts.get("triangle_like", 0) >= 2:
        notes.append("multiple arrow-head shapes detected (vector/force indicators)")
    if component_counts.get("rectangle_like", 0) >= 3:
        notes.append("rectangular blocks present (possible electrical components or tables)")
    if component_counts.get("circle_like", 0) >= 1:
        notes.append("circular elements present (wheels, pulleys, or nodes)")
    if component_counts.get("complex_polygons", 0) >= 1:
        notes.append("irregular closed shapes detected")
    return notes


def classify_overall_kind(features: Dict[str, object]) -> str:
    chart_type = features.get("chart_type", "unknown")
    axis_presence = features.get("axis_presence", {})
    line_stats = features.get("line_stats", {})
    component_counts = features.get("component_counts", {})

    has_axes = bool(axis_presence.get("has_horizontal") and axis_presence.get("has_vertical"))
    segments = int(line_stats.get("total_segments", 0))
    rectangle_like = int(component_counts.get("rectangle_like", 0))
    triangle_like = int(component_counts.get("triangle_like", 0))

    if has_axes and chart_type == "cartesian_graph":
        return "cartesian_graph"
    if has_axes and segments >= 2:
        return "axes_plot"
    if triangle_like >= 2:
        return "vector_diagram"
    if rectangle_like >= 3:
        return "schematic_diagram"
    return chart_type or "unknown"


def combine_summary(
    base_features: Dict[str, object],
    caption_text: Optional[str],
    chartocr_payload: Optional[Dict[str, object]],
) -> Dict[str, object]:
    axis_presence = base_features.get("axis_presence", {})
    keypoints_normalized = base_features.get("keypoints_normalized", [])
    line_stats = base_features.get("line_stats", {})
    component_counts = base_features.get("component_counts", {})
    edge_density = float(base_features.get("edge_density", 0.0))
    size = base_features.get("image_size", {})

    trend_info = regression_summary(keypoints_normalized)
    peaks = detect_peaks(keypoints_normalized)
    component_notes = component_summary(component_counts)
    dominant_angles = line_stats.get("dominant_angles", [])

    observations: List[str] = []

    has_axes = axis_presence.get("has_horizontal") or axis_presence.get("has_vertical")
    if axis_presence.get("has_horizontal") and axis_presence.get("has_vertical"):
        observations.append("clear horizontal and vertical axes detected")
    elif axis_presence.get("has_horizontal"):
        observations.append("horizontal axis detected; vertical axis weak/absent")
    elif axis_presence.get("has_vertical"):
        observations.append("vertical axis detected; horizontal axis weak/absent")
    else:
        observations.append("no strong axes; treated as schematic/diagram")

    total_segments = int(line_stats.get("total_segments", 0))
    if total_segments:
        observations.append(
            f"{total_segments} prominent line segments with strongest angles: "
            + ", ".join(
                f"~{entry['angle']}° ({entry['count']})" for entry in dominant_angles[:3]
            )
            if dominant_angles
            else ""
        )

    if component_notes:
        observations.extend(component_notes)

    if edge_density > 0.2:
        observations.append("dense ink coverage (possible shaded region or complex schematic)")

    direction: str
    slope = trend_info.get("slope", 0.0)
    variance = trend_info.get("variance", 0.0)
    direction_threshold = 0.08
    if len(keypoints_normalized) >= 10 and variance >= 0.0025:
        if slope >= direction_threshold:
            direction = "increasing"
        elif slope <= -direction_threshold:
            direction = "decreasing"
        else:
            direction = "flat"
    elif len(keypoints_normalized) >= 10:
        direction = "flat"
    else:
        direction = "undetermined"

    peak_note = ""
    if peaks["maxima"]:
        peak_note = f"{len(peaks['maxima'])} peak(s) detected"
    if peaks["minima"]:
        if peak_note:
            peak_note += ", "
        peak_note += f"{len(peaks['minima'])} dip(s) detected"
    if peak_note:
        observations.append(peak_note)

    dimensions = (
        f"approx. {int(size.get('width', 0))}×{int(size.get('height', 0))} px"
        if size
        else "unknown size"
    )

    if caption_text:
        observations.append("OCR text context available")

    chartocr_summary = None
    if chartocr_payload:
        # If ChartOCR returned meaningful info, prefer its structured output.
        if not chartocr_payload.get("error"):
            chartocr_summary = chartocr_payload
            axis_labels = []
            if chartocr_payload.get("x_axis"):
                axis_labels.append(f"x-axis: {chartocr_payload['x_axis']}")
            if chartocr_payload.get("y_axis"):
                axis_labels.append(f"y-axis: {chartocr_payload['y_axis']}")
            if axis_labels:
                observations.extend(axis_labels)
            if chartocr_payload.get("legends"):
                legend_text = ", ".join(chartocr_payload["legends"])
                observations.append(f"legend entries: {legend_text}")
        else:
            observations.append(f"ChartOCR error: {chartocr_payload['error']}")

    overall_kind = classify_overall_kind(base_features)

    # Build final paragraph.
    lines: List[str] = []
    if overall_kind == "cartesian_graph":
        lines.append("Line/curve graph with resolved Cartesian axes.")
    elif overall_kind == "axes_plot":
        lines.append("Graph-like figure with partial axis structure.")
    elif overall_kind == "vector_diagram":
        lines.append("Diagram dominated by arrow shapes (vector/force layout).")
    elif overall_kind == "schematic_diagram":
        lines.append("Schematic diagram with multiple rectangular components.")
    else:
        lines.append("Diagram with ambiguous chart type (treated generically).")

    if has_axes and direction in {"increasing", "decreasing", "flat"}:
        dir_phrase = {
            "increasing": "overall trend moves upward",
            "decreasing": "overall trend slopes downward",
            "flat": "values stay roughly level",
        }.get(direction, "")
        if dir_phrase:
            lines.append(dir_phrase + ".")
    elif direction == "undetermined":
        lines.append("Not enough geometry to determine a clear trend.")

    if peak_note:
        lines.append(peak_note.capitalize() + ".")

    if component_notes and overall_kind in {"vector_diagram", "schematic_diagram"}:
        lines.append("Detected components: " + "; ".join(component_notes) + ".")

    lines.append(f"Image size {dimensions}.")

    summary_text = " ".join(lines)

    return {
        "text": summary_text.strip(),
        "observations": observations,
        "trend": {
            "direction": direction,
            "slope": round(trend_info.get("slope", 0.0), 4),
            "variance": round(trend_info.get("variance", 0.0), 5),
            "peak_counts": {
                "maxima": len(peaks["maxima"]),
                "minima": len(peaks["minima"]),
            },
        },
        "sources": {
            "caption_excerpt": caption_text[:280] if caption_text else None,
            "chartocr": chartocr_summary,
        },
    }


def process_graph_json(
    json_path: Path,
    graph_root: Path,
    output_root: Path,
    caption_dir: Path,
    image_root: Path,
    overwrite: bool,
    use_chartocr: bool,
    minimal_output: bool,
) -> Tuple[str, str]:
    relative_path = json_path.relative_to(graph_root)
    target_path = output_root / relative_path

    if not overwrite and target_path.exists():
        try:
            existing_payload = json.loads(target_path.read_text(encoding="utf-8"))
            if existing_payload.get("graph_summary"):
                return (existing_payload.get("image", str(json_path)), "skip_existing")
        except Exception:  # pylint: disable=broad-except
            pass

    payload = json.loads(json_path.read_text(encoding="utf-8"))

    base_features = payload.get("graph_metadata")
    if not base_features:
        return (payload.get("image", str(json_path)), "missing_metadata")

    image_rel = payload.get("image")
    caption_text = None
    chartocr_payload = None

    if image_rel:
        caption_text = load_caption_text(caption_dir, image_rel)
        image_path = image_root / image_rel
        if use_chartocr and image_path.exists():
            chartocr_payload = attempt_chartocr(image_path)
    else:
        image_path = None

    summary = combine_summary(base_features, caption_text, chartocr_payload)

    payload["graph_summary"] = summary
    payload["updated_at"] = datetime.utcnow().isoformat() + "Z"

    if minimal_output:
        output_payload = {
            "image": payload.get("image"),
            "graph_summary": payload["graph_summary"],
            "graph_metadata_source": str(json_path.relative_to(graph_root)),
            "updated_at": payload["updated_at"],
        }
    else:
        output_payload = payload

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return (payload.get("image", str(json_path)), "updated")


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_dir)

    graph_paths = list(iter_graph_json(args.graph_json_dir))
    if args.max_files is not None:
        graph_paths = graph_paths[: args.max_files]

    if not graph_paths:
        logging.warning("No graph JSON files found under %s", args.graph_json_dir)
        return

    logging.info("Generating descriptions for %s graph files", len(graph_paths))

    status_counter: Counter[str] = Counter()

    for json_path in graph_paths:
        image_id, status = process_graph_json(
            json_path,
        graph_root=args.graph_json_dir,
        output_root=args.output_dir,
        caption_dir=args.caption_dir,
        image_root=args.image_dir,
        overwrite=args.overwrite,
        use_chartocr=args.chartocr,
            minimal_output=args.minimal_output,
        )
        status_counter[status] += 1
        logging.debug("%s -> %s", image_id, status)

    logging.info(
        "Description run complete | updated=%s skipped=%s missing=%s | log=%s",
        status_counter.get("updated", 0),
        status_counter.get("skip_existing", 0),
        status_counter.get("missing_metadata", 0),
        log_path,
    )


if __name__ == "__main__":  # pragma: no cover
    main()


