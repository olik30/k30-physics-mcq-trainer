"""Convert a human supervision CSV into per-image JSON sidecar files.

The CSV is expected to contain (case-insensitive) columns for:
    - Graph (image filename)
    - Title
    - Visual Content
    - Description (human-written)
    - Comments (optional)
    - GPT description (optional)

Example usage:
    python scripts/csv_to_sidecar_json.py \
        "Human Supervised AQA  - Sheet1.csv" \
        "Graphs For Human Supervision/AQA"

Each row will produce a JSON file alongside the target image, sharing the same
stem (e.g. `AQA_graph_001.json`). Existing JSON files with the same name will be
overwritten.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable


EXPECTED_COLUMNS = {
    "graph": "image",
    "title": "title",
    "visual content": "visual_content",
    "description": "human_description",
    "comments": "comments",
    "gpt description": "gpt_description",
}


def normalise_headers(headers: Iterable[str | None]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for header in headers:
        key = (header or "").strip().lower()
        if not key or key in mapping:
            continue
        if key not in EXPECTED_COLUMNS:
            continue
        mapping[key] = EXPECTED_COLUMNS[key]
    missing = [raw for raw in EXPECTED_COLUMNS if raw not in mapping]
    if missing:
        logging.warning("Missing expected columns: %s", ", ".join(missing))
    return mapping


def read_csv_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header_map = normalise_headers(reader.fieldnames or [])
        if "graph" not in header_map:
            raise ValueError("CSV must include a 'Graph' column")

        for row_index, row in enumerate(reader, start=2):
            payload: dict[str, str] = {}
            for raw_key, value in row.items():
                key = (raw_key or "").strip().lower()
                if key not in header_map:
                    continue
                cleaned = (value or "").strip()
                if cleaned:
                    payload[header_map[key]] = cleaned
            if not payload:
                logging.debug("Row %d skipped: no recognised columns", row_index)
                continue
            payload["row_index"] = str(row_index)
            yield payload


def build_lookup(image_dir: Path) -> Dict[str, Path]:
    lookup: Dict[str, Path] = {}
    for path in image_dir.glob("*.png"):
        lookup[path.name.lower()] = path
    return lookup


def write_sidecars(rows: Iterable[dict[str, str]], image_dir: Path) -> None:
    lookup = build_lookup(image_dir)
    missing_images = 0
    written = 0
    for row in rows:
        image_name = row.get("image")
        if not image_name:
            logging.warning(
                "Row %s skipped: no image filename provided", row.get("row_index", "?")
            )
            continue
        image_path = lookup.get(image_name.lower())
        if not image_path:
            logging.error(
                "Row %s: image '%s' not found in %s",
                row.get("row_index", "?"),
                image_name,
                image_dir,
            )
            missing_images += 1
            continue

        json_path = image_path.with_suffix(".json")
        payload = {
            "image": image_path.name,
            "title": row.get("title"),
            "visual_content": row.get("visual_content"),
            "human_description": row.get("human_description"),
            "comments": row.get("comments"),
            "gpt_description": row.get("gpt_description"),
        }
        # Remove keys with None values to keep JSON tidy.
        payload = {key: value for key, value in payload.items() if value}

        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logging.info("Wrote %s", json_path.relative_to(image_dir.parent.parent))
        written += 1

    logging.info("Conversion complete: %d JSON files written", written)
    if missing_images:
        logging.warning("%d rows referenced missing images", missing_images)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create JSON sidecars for human annotations")
    parser.add_argument("csv_path", type=Path, help="Path to the CSV export")
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing the annotated PNG files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    if not args.csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {args.csv_path}")
    if not args.image_dir.is_dir():
        raise NotADirectoryError(f"Image directory not found: {args.image_dir}")

    rows = list(read_csv_rows(args.csv_path))
    if not rows:
        logging.warning("No rows parsed from CSV — nothing to do")
        return

    write_sidecars(rows, args.image_dir)


if __name__ == "__main__":
    main()

