"""Sample a balanced set of graph images for manual benchmarking.

This script walks the extracted image tree (as produced by scripts/extract_pdf.py),
selects a configurable number of diagrams per exam board, merges vector clips into
composite previews, and copies everything into a benchmark folder for human review.
A manifest JSONL is written alongside the images so you know the asset source and
why it was selected.

Example:

    python scripts/sample_graph_benchmark.py \
        --image-dir data/images \
        --output-dir data/graph_benchmark/samples \
        --total 120 --seed 42

Key behaviours:
    • picks both embedded bitmap graphs and stitched vector composites
    • aims for an even per-board mix (adjusts automatically if a board has fewer assets)
    • records metadata (board, exam, page, asset_kind, source paths) in a manifest
    • only processes the assets that end up in the sample list

Delete the generated folder when you no longer need the temporary copies.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

EMBEDDED_PATTERN = Path("page_000_img_00.png")  # placeholder so mypy sees the attribute

# Reimport the regex patterns from analyse_graphs to stay consistent.
try:
    from scripts.analyse_graphs import EMBEDDED_PATTERN, VECTOR_PATTERN, RENDER_PATTERN
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Run this script from the project root so 'scripts' is importable."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample graph images for benchmarking")
    parser.add_argument("--image-dir", type=Path, default=Path("data/images"),
                        help="Root directory containing extracted images")
    parser.add_argument("--output-dir", type=Path, default=Path("data/graph_benchmark/samples"),
                        help="Destination folder for sampled images")
    parser.add_argument("--total", type=int, default=120,
                        help="Total number of samples across all boards (default: 120)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--min-vectors", type=int, default=5,
                        help="Minimum vector composites to pick per board when available")
    parser.add_argument("--vector-ratio", type=float, default=0.35,
                        help="Target ratio of vector composites per board (0-1)")
    return parser.parse_args()


def list_board_dirs(image_root: Path) -> List[Path]:
    return sorted([p for p in image_root.iterdir() if p.is_dir()])


def group_vector_assets(board_dir: Path) -> Dict[Tuple[Path, int], Dict[str, List[Path]]]:
    grouped: Dict[Tuple[Path, int], Dict[str, List[Path]]] = defaultdict(lambda: {"vector": [], "embedded": [], "render": []})
    for path in sorted(board_dir.rglob("*")):
        if not path.is_file():
            continue
        name = path.name
        if VECTOR_PATTERN.search(name):
            page = int(VECTOR_PATTERN.search(name).group(1))  # type: ignore[union-attr]
            relative_parent = path.relative_to(board_dir).parent
            grouped[(relative_parent, page)]["vector"].append(path)
        elif EMBEDDED_PATTERN.search(name):
            page = int(EMBEDDED_PATTERN.search(name).group(1))  # type: ignore[union-attr]
            relative_parent = path.relative_to(board_dir).parent
            grouped[(relative_parent, page)]["embedded"].append(path)
        elif RENDER_PATTERN.search(name):
            page = int(RENDER_PATTERN.search(name).group(1))  # type: ignore[union-attr]
            relative_parent = path.relative_to(board_dir).parent
            grouped[(relative_parent, page)]["render"].append(path)
    return grouped


def safe_name(parts: List[str]) -> str:
    cleaned: List[str] = []
    for part in parts:
        cleaned.append(part.replace(" ", "_").replace("-", "_"))
    return "__".join(cleaned)


def build_vector_composite(paths: List[Path], dest_dir: Path, board: str,
                            relative_parent: Path, page: int) -> Tuple[Path, Dict[str, object]]:
    if not paths:
        raise ValueError("No vector paths provided")
    dest_dir.mkdir(parents=True, exist_ok=True)
    pil_images: List[Image.Image] = []
    for path in paths:
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

    rel_parts = [board] + list(relative_parent.parts)
    filename = f"{safe_name(rel_parts)}__page_{page:03d}_vector_combo.png"
    output_path = dest_dir / filename
    composite.save(output_path)

    manifest = {
        "asset_kind": "vector_composite",
        "page": page,
        "source_images": [str(path) for path in paths],
    }
    return output_path, manifest


def copy_asset(src: Path, dest_dir: Path, board: str, tag: str) -> Tuple[Path, Dict[str, object]]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    rel_parts = [board] + list(src.parent.parts[-3:])  # limit to keep names short
    filename = f"{safe_name(rel_parts)}__{src.name}"
    output_path = dest_dir / filename
    shutil.copy2(src, output_path)
    manifest = {
        "asset_kind": tag,
        "source_images": [str(src)],
    }
    return output_path, manifest


def choose_samples(board: str, board_dir: Path,
                   grouped: Dict[Tuple[Path, int], Dict[str, List[Path]]],
                   goal: int, min_vectors: int, vector_ratio: float,
                   rng: random.Random, output_board_dir: Path) -> List[Dict[str, object]]:
    vector_groups: List[Tuple[Path, int, List[Path]]] = []
    embedded_assets: List[Path] = []
    render_assets: List[Path] = []

    for (rel_parent, page), assets in grouped.items():
        vectors = assets["vector"]
        if vectors:
            vector_groups.append((rel_parent, page, sorted(vectors)))
        embedded_assets.extend(sorted(assets["embedded"]))
        render_assets.extend(sorted(assets["render"]))

    rng.shuffle(vector_groups)
    rng.shuffle(embedded_assets)
    rng.shuffle(render_assets)

    vector_goal = min(len(vector_groups), max(min_vectors, int(round(goal * vector_ratio))))
    embedded_goal = goal - vector_goal

    selections: List[Dict[str, object]] = []

    for rel_parent, page, vectors in vector_groups[:vector_goal]:
        output_path, meta = build_vector_composite(vectors, output_board_dir, board, rel_parent, page)
        meta.update({
            "board": board,
            "page": page,
            "relative_parent": str(rel_parent),
            "output_image": str(output_path),
        })
        selections.append(meta)

    def pull_assets(pool: List[Path], count: int, tag: str) -> None:
        for src in pool[:count]:
            output_path, meta = copy_asset(src, output_board_dir, board, tag)
            meta.update({
                "board": board,
                "page": extract_page_number(src),
                "relative_parent": str(src.relative_to(board_dir).parent),
                "output_image": str(output_path),
            })
            selections.append(meta)

    pull_assets(embedded_assets, max(0, embedded_goal), "embedded")

    remaining = goal - len(selections)
    if remaining > 0:
        pull_assets(render_assets, min(len(render_assets), remaining), "page_render")

    return selections[:goal]


def extract_page_number(path: Path) -> Optional[int]:
    name = path.name
    if EMBEDDED_PATTERN.search(name):
        return int(EMBEDDED_PATTERN.search(name).group(1))  # type: ignore[union-attr]
    if VECTOR_PATTERN.search(name):
        return int(VECTOR_PATTERN.search(name).group(1))  # type: ignore[union-attr]
    if RENDER_PATTERN.search(name):
        return int(RENDER_PATTERN.search(name).group(1))  # type: ignore[union-attr]
    return None


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    board_dirs = list_board_dirs(args.image_dir)
    if not board_dirs:
        raise SystemExit(f"No board directories found under {args.image_dir}")

    total = max(args.total, len(board_dirs))
    per_board = total // len(board_dirs)
    remainder = total % len(board_dirs)

    manifest: List[Dict[str, object]] = []

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for idx, board_dir in enumerate(board_dirs):
        goal = per_board + (1 if idx < remainder else 0)
        board_name = board_dir.name
        grouped = group_vector_assets(board_dir)
        board_output_dir = args.output_dir / board_name
        board_output_dir.mkdir(parents=True, exist_ok=True)

        board_manifest = choose_samples(
            board_name,
            board_dir,
            grouped,
            goal,
            args.min_vectors,
            args.vector_ratio,
            rng,
            board_output_dir,
        )
        manifest.extend(board_manifest)

    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fh:
        for entry in manifest:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    write_readme(args.output_dir, len(manifest))

    print(f"Sampled {len(manifest)} images -> {args.output_dir}")
    print(f"Manifest written to {manifest_path}")


def write_readme(output_dir: Path, count: int) -> None:
    readme_path = output_dir / "README.txt"
    blurb = (
        "These are temporary copies for the Day 4 graph benchmark.\n"
        "Annotate each image with component notes and a plain-language summary,\n"
        "then record the status in manifest.jsonl or your annotation sheet.\n"
        "Delete this folder when you are finished.\n"
        f"Total assets: {count}\n"
    )
    readme_path.write_text(blurb, encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    main()
