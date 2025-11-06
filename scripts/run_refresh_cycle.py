"""Run an end-to-end refresh cycle: regenerate MCQs, retrain, evaluate, and optionally bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from scripts import (
    auto_generate,
    compare_adapters,
    create_handoff_bundle,
    eval,
    filter_balance,
    format_for_sft,
    prepare_refresh,
    train_lora,
)


DEFAULT_EVALUATION = Path("results/adapter_v2/samples.jsonl")
DEFAULT_FORMATTED_MANIFEST = Path("data/formatted/manifest.json")
DEFAULT_REFRESH_SOURCES = Path("data/parsed/refresh_sources.jsonl")
DEFAULT_REFRESH_DRAFTS = Path("data/parsed/refresh_drafts.jsonl")
DEFAULT_REFRESH_CANDIDATES = Path("data/filtered/refresh_candidates.jsonl")
DEFAULT_SEED_DATA = Path("data/filtered/seed_train.filtered.jsonl")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate the feedback → refresh → train loop")
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=DEFAULT_EVALUATION,
        help="Evaluation samples JSONL to analyse (defaults to latest adapter)",
    )
    parser.add_argument(
        "--formatted-manifest",
        type=Path,
        default=DEFAULT_FORMATTED_MANIFEST,
        help="Manifest JSON produced by format_for_sft.py",
    )
    parser.add_argument(
        "--refresh-sources",
        type=Path,
        default=DEFAULT_REFRESH_SOURCES,
        help="Output path for prepare_refresh (question parts JSONL)",
    )
    parser.add_argument(
        "--refresh-drafts",
        type=Path,
        default=DEFAULT_REFRESH_DRAFTS,
        help="Output path for auto_generate drafts",
    )
    parser.add_argument(
        "--refresh-candidates",
        type=Path,
        default=DEFAULT_REFRESH_CANDIDATES,
        help="Filtered refresh dataset ready for review/training",
    )
    parser.add_argument(
        "--seed-data",
        type=Path,
        default=DEFAULT_SEED_DATA,
        help="Seed dataset to merge when formatting for SFT",
    )
    parser.add_argument(
        "--adapter-name",
        required=True,
        help="Name for the new adapter (e.g. adapter_v3)",
    )
    parser.add_argument(
        "--device-map",
        default="cpu",
        help="Device map for training/evaluation (default: cpu)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1,
        help="Training max steps (default: 1 for smoke test—raise for real runs)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate for LoRA training",
    )
    parser.add_argument(
        "--skip-format",
        action="store_true",
        help="Skip re-running format_for_sft.py",
    )
    parser.add_argument(
        "--compare-with",
        type=Path,
        default=None,
        help="Optional metrics.json from the previous adapter for comparison",
    )
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="Create a fresh handoff bundle after training",
    )
    parser.add_argument(
        "--bundle-paths",
        nargs="*",
        default=[],
        help="Additional paths to include in the bundle (defaults include key artefacts)",
    )
    parser.add_argument(
        "--auto-generate-limit",
        type=int,
        default=60,
        help="Max drafts to generate per cycle",
    )
    return parser.parse_args(argv)


def run_refresh_cycle(args: argparse.Namespace) -> None:
    print("Step 1/7: prepare_refresh.py")
    summary_json = Path("results") / f"{args.adapter_name}_refresh_summary.json"
    summary_md = Path("results") / f"{args.adapter_name}_refresh_summary.md"
    prepare_refresh_args = [
        "--evaluation",
        str(args.evaluation),
        "--formatted",
        str(args.formatted_manifest.parent / "test.jsonl"),
        "--parsed",
        "data/parsed/questions.jsonl",
        "--output-sources",
        str(args.refresh_sources),
        "--summary-json",
        str(summary_json),
        "--summary-md",
        str(summary_md),
        "--include-answer-mismatch",
    ]
    prepare_refresh.main(prepare_refresh_args)

    if not args.refresh_sources.exists():
        print("No refresh_sources.jsonl created; aborting cycle.")
        return

    if args.refresh_drafts.exists():
        args.refresh_drafts.unlink()

    print("Step 2/7: auto_generate.py")
    auto_generate_args = [
        "--input",
        str(args.refresh_sources),
        "--output",
        str(args.refresh_drafts),
        "--limit",
        str(args.auto_generate_limit),
        "--allow-existing",
    ]
    auto_generate.main(auto_generate_args)

    print("Step 3/7: filter_balance.py")
    reports_tag = args.adapter_name
    filter_args = [
        "--input",
        str(args.refresh_drafts),
        "--output",
        str(args.refresh_candidates),
        "--report",
        f"reports/{reports_tag}_refresh.json",
        "--coverage-report",
        f"reports/{reports_tag}_refresh.md",
        "--emit-flagged",
        f"data/review/{reports_tag}_refresh_flagged.jsonl",
    ]
    filter_balance.main(filter_args)

    if not args.skip_format:
        print("Step 4/7: format_for_sft.py")
        format_args = [
            "--inputs",
            str(args.seed_data),
            str(args.refresh_candidates),
            "--manifest",
            str(args.formatted_manifest),
            "--stats-path",
            "results/dataset_stats.json",
            "--output-dir",
            "data/formatted",
        ]
        format_for_sft.main(format_args)
    else:
        print("Step 4/7: format_for_sft.py skipped")

    print("Step 5/7: train_lora.py")
    adapter_dir = Path("models/adapters") / args.adapter_name
    log_path = Path("logs/train") / f"{args.adapter_name}.jsonl"
    train_args = [
        "--base-model",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "--adapter-name",
        args.adapter_name,
        "--output-dir",
        str(adapter_dir),
        "--run-name",
        args.adapter_name,
        "--manifest",
        str(args.formatted_manifest),
        "--log-path",
        str(log_path),
        "--config-path",
        "config/adapters.yaml",
        "--device-map",
        args.device_map,
        "--max-steps",
        str(args.max_steps),
        "--learning-rate",
        str(args.learning_rate),
        "--per-device-train-batch-size",
        "1",
        "--per-device-eval-batch-size",
        "1",
        "--gradient-accumulation-steps",
        "1",
        "--logging-steps",
        "1",
        "--save-steps",
        "1",
        "--eval-steps",
        "1",
        "--warmup-steps",
        "0",
        "--no-eval",
        "--max-seq-length",
        "512",
        "--trust-remote-code",
    ]
    train_lora.main(train_args)

    print("Step 6/7: eval.py")
    eval_output = Path("results") / args.adapter_name
    eval_args = [
        "--trust-remote-code",
        "--device",
        args.device_map,
        "--adapter-path",
        str(adapter_dir),
        "--output-dir",
        str(eval_output),
    ]
    eval.main(eval_args)

    if args.compare_with and args.compare_with.exists():
        print("Step 7/7: compare_adapters.py")
        comparison_args = [
            "--adapter-a",
            str(args.compare_with),
            "--label-a",
            args.compare_with.parent.name,
            "--adapter-b",
            str(eval_output / "metrics.json"),
            "--label-b",
            args.adapter_name,
        ]
        compare_adapters.main(comparison_args)
    else:
        print("Step 7/7: comparison skipped (provide --compare-with to enable)")

    if args.bundle:
        print("Creating handoff bundle...")
        default_paths = [
            str(args.refresh_candidates),
            "data/review/day13_feedback_log.jsonl",
            str(eval_output / "metrics.json"),
            str(eval_output / "samples.jsonl"),
            "results/adapter_comparison.json",
            "results/adapter_comparison.md",
            "docs/feedback-playbook.md",
            "scripts/feedback_queue.py",
            "config/adapters.yaml",
            str(adapter_dir),
        ]
        bundle_paths = default_paths + args.bundle_paths
        bundle_args = ["--paths"] + bundle_paths
        create_handoff_bundle.main(bundle_args)

    print("Refresh cycle complete.")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    try:
        run_refresh_cycle(args)
    except Exception as exc:  # pragma: no cover - cascade failure guard
        print(f"Refresh cycle failed: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()


