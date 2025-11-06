"""Assemble a Day 14 handoff bundle containing key artefacts and manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import List, Optional, Sequence


DEFAULT_OUTPUT_DIR = Path("handoff/day14_artifacts")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create handoff bundle")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--paths",
        nargs="+",
        required=True,
        help="Files or directories to copy into the bundle",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "manifest.json",
        help="Path to write the JSON manifest",
    )
    parser.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "README.md",
        help="Path to write the README overview",
    )
    return parser.parse_args(argv)


def sha256_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


REPO_ROOT = Path.cwd().resolve()


def copy_path(src: Path, dest_root: Path) -> List[Path]:
    try:
        rel = src.relative_to(REPO_ROOT)
    except ValueError:
        rel = src.name
    dest = dest_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
        copied_files: List[Path] = [p for p in dest.rglob("*") if p.is_file()]
    else:
        shutil.copy2(src, dest)
        copied_files = [dest]
    return copied_files


def write_manifest(manifest_path: Path, records: List[dict]) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def write_readme(readme_path: Path, records: List[dict]) -> None:
    lines = [
        "# Day 14 Handoff Bundle",
        "",
        "This directory contains the artefacts needed for the post-Day 14 human review loop.",
        "",
        "| Path | Size (bytes) | SHA256 |",
        "| --- | ---: | --- |",
    ]
    for record in records:
        lines.append(
            f"| `{record['path']}` | {record['size']} | `{record['sha256']}` |"
        )
    readme_path.parent.mkdir(parents=True, exist_ok=True)
    readme_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    output_root = args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_records: List[dict] = []
    for raw_path in args.paths:
        src = Path(raw_path).resolve()
        if not src.exists():
            raise FileNotFoundError(f"{src} does not exist")
        copied = copy_path(src, output_root)
        for dst in copied:
            rel_path = dst.relative_to(output_root)
            manifest_records.append(
                {
                    "path": str(rel_path).replace("\\", "/"),
                    "size": dst.stat().st_size,
                    "sha256": sha256_digest(dst),
                }
            )

    manifest_records.sort(key=lambda item: item["path"])
    write_manifest(args.manifest, manifest_records)
    write_readme(args.readme, manifest_records)
    print(f"Bundle created at {output_root}")
    print(f"Manifest written to {args.manifest}")


if __name__ == "__main__":
    main()


