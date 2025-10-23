"""End-to-end comparison between human graph notes and automated summaries.

Steps performed by this helper script:

1. Re-run the graph analysis + description pipeline on the curated
   benchmark PNGs so we have fresh automated summaries that line up with the
   current code.
2. Pair each human-written description with the matching machine summary.
3. Compute simple similarity scores and keyword coverage metrics.
4. Emit tabular + JSONL + Markdown reports that flag low-alignment cases.

Example usage:

    python scripts/graph_benchmark_compare.py \
        --manual-dir data/graph_human_descriptions/AQA \
        --work-dir data/graph_human_descriptions_for_comparison \
        --workers 4

The reports will be written under the chosen work directory. Existing
analysis/summary folders are refreshed unless `--reuse-analysis` or
`--reuse-description` is provided.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import shutil
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


DEFAULT_MANUAL_DIR = Path("data/graph_human_descriptions/AQA")
DEFAULT_WORK_DIR = Path("data/graph_human_descriptions_for_comparison")
STAGING_IMAGE_DIRNAME = "staging_images"


# Lightweight stopword list for keyword coverage checks.
STOPWORDS: Set[str] = {
    "the",
    "and",
    "that",
    "with",
    "from",
    "this",
    "into",
    "have",
    "there",
    "which",
    "also",
    "over",
    "under",
    "through",
    "because",
    "between",
    "where",
    "about",
    "their",
    "while",
    "within",
    "without",
    "along",
    "across",
    "such",
    "been",
    "were",
    "when",
    "each",
    "onto",
    "only",
    "many",
    "some",
    "made",
    "make",
    "show",
    "shows",
    "figure",
    "labelled",
    "labelled",
    "image",
    "diagram",
}


@dataclass
class ComparisonResult:
    image: str
    manual_title: Optional[str]
    manual_path: Path
    auto_path: Optional[Path]
    human_description: str
    auto_summary: Optional[str]
    similarity: Optional[float]
    missing_keywords: List[str]
    manual_keywords: List[str]
    auto_keywords: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare human and machine graph summaries")
    parser.add_argument(
        "--manual-dir",
        type=Path,
        default=DEFAULT_MANUAL_DIR,
        help="Directory containing curated PNG+JSON benchmark pairs",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=DEFAULT_WORK_DIR,
        help="Workspace to host analysis outputs and comparison reports",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of workers to hand to the analysis pipeline",
    )
    parser.add_argument(
        "--det-threshold",
        type=float,
        default=0.0,
        help="Axis-detection threshold to pass into analyse_graphs.py",
    )
    parser.add_argument(
        "--reuse-analysis",
        action="store_true",
        help="Skip rerunning analyse_graphs.py if analysis output already exists",
    )
    parser.add_argument(
        "--reuse-description",
        action="store_true",
        help="Skip rerunning describe_graphs.py if auto summaries already exist",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.60,
        help="Flag entries with cosine similarity below this value",
    )
    return parser.parse_args()


def configure_logging(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "comparison.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )
    logging.info("Logging to %s", log_path)


def run_cli(command: Sequence[str]) -> None:
    logging.info("Running: %s", " ".join(command))
    subprocess.run(command, check=True)


def ensure_subdir(path: Path, reuse: bool) -> None:
    if path.exists() and not reuse:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def sanitise_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.split()).strip()


def extract_keywords(text: str) -> Set[str]:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    keywords = {tok for tok in tokens if len(tok) >= 4 and tok not in STOPWORDS}
    return keywords


def compute_similarity(manual_text: str, auto_text: str) -> Optional[float]:
    manual_clean = manual_text.strip()
    auto_clean = auto_text.strip()
    if not manual_clean and not auto_clean:
        return 1.0
    if not manual_clean or not auto_clean:
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([manual_clean, auto_clean])
    score = float(cosine_similarity(matrix[0], matrix[1])[0, 0])
    return score


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def gather_manual_entries(manual_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for json_path in sorted(manual_dir.glob("*.json")):
        mapping[json_path.stem] = json_path
    return mapping


def gather_auto_entries(auto_dir: Path) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for json_path in sorted(auto_dir.glob("*.json")):
        mapping[json_path.stem] = json_path
    return mapping


def prepare_staging_images(manual_dir: Path, staging_dir: Path, reuse: bool) -> Dict[str, str]:
    if staging_dir.exists() and not reuse:
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    mapping: Dict[str, str] = {}
    manual_pngs = sorted(manual_dir.glob("*.png"))
    if not manual_pngs:
        raise FileNotFoundError(f"No PNG files found in {manual_dir}")

    for idx, image_path in enumerate(manual_pngs, start=1):
        staged_name = f"benchmark_page_{idx:03d}_render{image_path.suffix.lower()}"
        staged_path = staging_dir / staged_name
        if not reuse or not staged_path.exists():
            shutil.copy2(image_path, staged_path)
        mapping[Path(staged_name).stem] = image_path.stem

    return mapping


def mirror_analysis_to_manual_names(dir_path: Path, mapping: Dict[str, str]) -> None:
    for staged_stem, manual_stem in mapping.items():
        staged_path = dir_path / f"{staged_stem}.json"
        if not staged_path.exists():
            continue
        manual_path = dir_path / f"{manual_stem}.json"
        shutil.copy2(staged_path, manual_path)
        if manual_path != staged_path:
            staged_path.unlink()


def mirror_auto_to_manual_names(auto_dir: Path, mapping: Dict[str, str]) -> None:
    for staged_stem, manual_stem in mapping.items():
        staged_path = auto_dir / f"{staged_stem}.json"
        if not staged_path.exists():
            continue
        manual_path = auto_dir / f"{manual_stem}.json"
        shutil.copy2(staged_path, manual_path)
        if manual_path != staged_path:
            staged_path.unlink()


def compare_summaries(
    manual_dir: Path,
    auto_dir: Path,
) -> List[ComparisonResult]:
    manual_map = gather_manual_entries(manual_dir)
    auto_map = gather_auto_entries(auto_dir)

    results: List[ComparisonResult] = []

    for stem, manual_path in manual_map.items():
        manual_payload = load_json(manual_path)
        manual_title = manual_payload.get("title")
        human_text = sanitise_text(
            manual_payload.get("human_description")
            or manual_payload.get("visual_content")
            or manual_payload.get("title")
            or ""
        )

        auto_path = auto_map.get(stem)
        if auto_path is None:
            results.append(
                ComparisonResult(
                    image=f"{stem}.png",
                    manual_title=manual_title,
                    manual_path=manual_path,
                    auto_path=None,
                    human_description=human_text,
                    auto_summary=None,
                    similarity=None,
                    missing_keywords=sorted(extract_keywords(human_text)),
                    manual_keywords=sorted(extract_keywords(human_text)),
                    auto_keywords=[],
                )
            )
            continue

        auto_payload = load_json(auto_path)
        auto_summary = ""
        if isinstance(auto_payload, dict):
            summary_block = auto_payload.get("graph_summary")
            if isinstance(summary_block, dict):
                auto_summary = sanitise_text(summary_block.get("text"))
        auto_summary = auto_summary or ""

        similarity = compute_similarity(human_text, auto_summary)
        manual_keywords = extract_keywords(human_text)
        auto_keywords = extract_keywords(auto_summary)
        missing_keywords = sorted(manual_keywords - auto_keywords)

        results.append(
            ComparisonResult(
                image=f"{stem}.png",
                manual_title=manual_title,
                manual_path=manual_path,
                auto_path=auto_path,
                human_description=human_text,
                auto_summary=auto_summary,
                similarity=similarity,
                missing_keywords=missing_keywords,
                manual_keywords=sorted(manual_keywords),
                auto_keywords=sorted(auto_keywords),
            )
        )

    return results


def write_jsonl(results: Iterable[ComparisonResult], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        for item in results:
            payload = {
                "image": item.image,
                "manual_title": item.manual_title,
                "manual_path": str(item.manual_path),
                "auto_path": str(item.auto_path) if item.auto_path else None,
                "human_description": item.human_description,
                "auto_summary": item.auto_summary,
                "similarity": item.similarity,
                "missing_keywords": item.missing_keywords,
                "manual_keywords": item.manual_keywords,
                "auto_keywords": item.auto_keywords,
            }
            handle.write(json.dumps(payload, ensure_ascii=False))
            handle.write("\n")


def write_csv(results: Iterable[ComparisonResult], output_path: Path) -> None:
    fieldnames = [
        "image",
        "manual_title",
        "similarity",
        "missing_keywords",
        "human_description",
        "auto_summary",
        "manual_path",
        "auto_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            writer.writerow(
                {
                    "image": item.image,
                    "manual_title": item.manual_title or "",
                    "similarity": f"{item.similarity:.3f}" if item.similarity is not None else "",
                    "missing_keywords": ", ".join(item.missing_keywords),
                    "human_description": item.human_description,
                    "auto_summary": item.auto_summary or "",
                    "manual_path": str(item.manual_path),
                    "auto_path": str(item.auto_path) if item.auto_path else "",
                }
            )


def write_markdown(
    results: List[ComparisonResult],
    output_path: Path,
    similarity_threshold: float,
) -> None:
    total = len(results)
    sims = [item.similarity for item in results if item.similarity is not None]
    below = [item for item in results if (item.similarity or 0.0) < similarity_threshold]
    missing_auto = [item for item in results if item.auto_summary is None]

    with output_path.open("w", encoding="utf-8") as handle:
        handle.write("# Graph Benchmark Comparison\n\n")
        handle.write(f"Total graphs compared: **{total}**\n\n")
        if sims:
            handle.write(
                f"Average similarity: **{statistics.mean(sims):.3f}**, "
                f"median: **{statistics.median(sims):.3f}**\n\n"
            )
            handle.write(
                f"Lowest similarity: **{min(sims):.3f}**, highest: **{max(sims):.3f}**\n\n"
            )
        handle.write(
            f"Similarity threshold: **{similarity_threshold:.2f}** — "
            f"entries below threshold: **{len(below)}**\n\n"
        )
        if missing_auto:
            handle.write(
                f"⚠️ Automated summary missing for {len(missing_auto)} image(s).\n\n"
            )

        if below:
            handle.write("## Lowest-scoring graphs\n\n")
            handle.write("| Image | Similarity | Missing keywords |\n")
            handle.write("| --- | --- | --- |\n")
            for item in sorted(below, key=lambda obj: (obj.similarity or 0.0))[:10]:
                sim_value = "—" if item.similarity is None else f"{item.similarity:.3f}"
                missing = ", ".join(item.missing_keywords) if item.missing_keywords else "—"
                handle.write(f"| {item.image} | {sim_value} | {missing} |\n")
            handle.write("\n")

        handle.write("## Detailed pairs\n\n")
        for item in results:
            handle.write(f"### {item.image}\n\n")
            if item.manual_title:
                handle.write(f"**Title:** {item.manual_title}\n\n")
            if item.similarity is not None:
                handle.write(f"**Similarity:** {item.similarity:.3f}\n\n")
            else:
                handle.write("**Similarity:** —\n\n")
            handle.write("**Human description**\n\n")
            handle.write(f"> {item.human_description}\n\n")
            handle.write("**Automated summary**\n\n")
            if item.auto_summary:
                handle.write(f"> {item.auto_summary}\n\n")
            else:
                handle.write("> _Missing automated summary_\n\n")
            if item.missing_keywords:
                handle.write(
                    f"**Keywords missing in automated text:** {', '.join(item.missing_keywords)}\n\n"
                )


def main() -> None:
    args = parse_args()
    configure_logging(args.work_dir)

    if not args.manual_dir.exists():
        raise FileNotFoundError(f"Manual directory not found: {args.manual_dir}")

    analysis_dir = args.work_dir / "analysis"
    auto_dir = args.work_dir / "auto_summaries"
    logs_dir = args.work_dir / "logs"
    captions_dir = args.work_dir / "captions_dummy"
    staging_dir = args.work_dir / STAGING_IMAGE_DIRNAME

    ensure_subdir(analysis_dir, args.reuse_analysis)
    ensure_subdir(auto_dir, args.reuse_description)
    logs_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    reuse_staging = args.reuse_analysis and args.reuse_description
    stem_mapping = prepare_staging_images(args.manual_dir, staging_dir, reuse_staging)

    if not args.reuse_analysis:
        run_cli(
            [
                sys.executable,
                "scripts/analyse_graphs.py",
                "--image-dir",
                str(staging_dir),
                "--output-dir",
                str(analysis_dir),
                "--log-dir",
                str(logs_dir / "analysis"),
                "--workers",
                str(args.workers),
                "--det-threshold",
                str(args.det_threshold),
                "--overwrite",
            ]
        )
        mirror_analysis_to_manual_names(analysis_dir, stem_mapping)

    if not args.reuse_description:
        run_cli(
            [
                sys.executable,
                "scripts/describe_graphs.py",
                "--graph-json-dir",
                str(analysis_dir),
                "--output-dir",
                str(auto_dir),
                "--image-dir",
                str(staging_dir),
                "--caption-dir",
                str(captions_dir),
                "--log-dir",
                str(logs_dir / "describe"),
                "--workers",
                str(args.workers),
                "--minimal-output",
                "--overwrite",
            ]
        )
        mirror_auto_to_manual_names(auto_dir, stem_mapping)

    logging.info("Comparing human notes with automated summaries…")
    results = compare_summaries(args.manual_dir, auto_dir)
    if not results:
        logging.warning("No manual entries found in %s", args.manual_dir)
        return

    jsonl_path = args.work_dir / "comparison.jsonl"
    csv_path = args.work_dir / "comparison.csv"
    md_path = args.work_dir / "comparison.md"

    write_jsonl(results, jsonl_path)
    write_csv(results, csv_path)
    write_markdown(results, md_path, args.similarity_threshold)

    logging.info("Comparison complete: %s", args.work_dir)
    logging.info("  JSONL: %s", jsonl_path)
    logging.info("  CSV:    %s", csv_path)
    logging.info("  Markdown: %s", md_path)

    low_scores = [
        item for item in results if (item.similarity or 0.0) < args.similarity_threshold
    ]
    if low_scores:
        logging.warning(
            "Found %d entries below threshold %.2f",
            len(low_scores),
            args.similarity_threshold,
        )
        for item in sorted(low_scores, key=lambda obj: (obj.similarity or 0.0))[:5]:
            logging.warning(
                "  %s -> similarity %.3f (missing keywords: %s)",
                item.image,
                item.similarity if item.similarity is not None else -1.0,
                ", ".join(item.missing_keywords) if item.missing_keywords else "-",
            )


if __name__ == "__main__":
    main()

