"""Run an end-to-end refresh cycle: regenerate MCQs, retrain, evaluate, and optionally bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:  # pragma: no cover
    from . import (
        auto_generate,
        compare_adapters,
        create_handoff_bundle,
        eval,
        filter_balance,
        format_for_sft,
        prepare_refresh,
        train_lora,
    )
except ImportError:  # pragma: no cover
    import auto_generate  # type: ignore
    import compare_adapters  # type: ignore
    import create_handoff_bundle  # type: ignore
    import eval  # type: ignore
    import filter_balance  # type: ignore
    import format_for_sft  # type: ignore
    import prepare_refresh  # type: ignore
    import train_lora  # type: ignore

LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

DEFAULT_EVALUATION = Path("results/adapter_v2/samples.jsonl")
DEFAULT_FORMATTED_MANIFEST = Path("data/formatted/manifest.json")
DEFAULT_REFRESH_SOURCES = Path("data/parsed/refresh_sources.jsonl")
DEFAULT_REFRESH_DRAFTS = Path("data/parsed/refresh_drafts.jsonl")
DEFAULT_REFRESH_CANDIDATES = Path("data/filtered/refresh_candidates.jsonl")
DEFAULT_APPROVED_REFRESH = Path("data/filtered/refresh_approved.jsonl")
DEFAULT_HOLDBACK_REFRESH = Path("data/filtered/refresh_holdback.jsonl")
DEFAULT_FEEDBACK_LOG = Path("data/review/day13_feedback_log.jsonl")
DEFAULT_SEED_DATA = Path("data/filtered/seed_train.filtered.jsonl")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automate the feedback -> refresh -> train loop")
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
        help="Filtered refresh dataset for human review",
    )
    parser.add_argument(
        "--approved-refresh",
        type=Path,
        default=DEFAULT_APPROVED_REFRESH,
        help="Output file for reviewer-approved refresh items",
    )
    parser.add_argument(
        "--holdback-refresh",
        type=Path,
        default=DEFAULT_HOLDBACK_REFRESH,
        help="Output file for items that need regeneration (needs-work/reject/unreviewed)",
    )
    parser.add_argument(
        "--feedback-log",
        type=Path,
        default=DEFAULT_FEEDBACK_LOG,
        help="JSONL log of reviewer decisions produced by feedback_queue.py",
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
        help="Training max steps (default: 1 for smoke test - raise for real runs)",
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


def compute_record_id(record: Dict[str, object]) -> str:
    question_id = ""
    part_code = ""
    for source in record.get("sources") or []:
        if isinstance(source, dict) and source.get("type") == "question_part":
            question_id = str(source.get("question_id") or "")
            part_code = str(source.get("part_code") or "")
            break
    if not question_id:
        question_id = str(record.get("id") or "")
    identifier = question_id
    if part_code:
        identifier = f"{identifier}#{part_code}"
    return identifier or "unknown"


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON (line {line_number}) in {path}: {exc}") from exc
    return records


def write_jsonl(path: Path, records: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")


def index_to_letter(index: Optional[int], options_len: int) -> Optional[str]:
    if isinstance(index, int) and 0 <= index < options_len and index < len(LETTER_LABELS):
        return LETTER_LABELS[index]
    return None


def letter_to_index(letter: str, options_len: int) -> Optional[int]:
    if not letter:
        return None
    if options_len <= 0:
        return None
    max_letter_index = min(options_len, len(LETTER_LABELS)) - 1
    if max_letter_index < 0:
        return None
    text = letter.strip().upper()
    if len(text) == 1 and "A" <= text <= LETTER_LABELS[max_letter_index]:
        idx = ord(text) - ord("A")
        if idx < options_len:
            return idx
    if text.isdigit():
        value = int(text)
        if 0 <= value < options_len:
            return value
        if 1 <= value <= options_len:
            return value - 1
    return None


def resolve_answer_index(entry: Dict[str, object], options_len: int, model_index: Optional[int]) -> Optional[int]:
    answer_index = entry.get("answer_index")
    if isinstance(answer_index, int) and 0 <= answer_index < options_len:
        return answer_index
    answer_choice = entry.get("answer_choice")
    if isinstance(answer_choice, str):
        idx = letter_to_index(answer_choice, options_len)
        if idx is not None:
            return idx
    if isinstance(model_index, int) and 0 <= model_index < options_len:
        return model_index
    return None


def load_feedback(log_path: Path) -> Dict[str, Dict[str, object]]:
    entries: Dict[str, Dict[str, object]] = {}
    for entry in load_jsonl(log_path):
        record_id = entry.get("record_id")
        if isinstance(record_id, str):
            entries[record_id] = entry
    return entries


def apply_review_feedback(
    refresh_candidates: Path,
    feedback_log: Path,
    approved_output: Path,
    holdback_output: Path,
) -> Dict[str, int]:
    records = load_jsonl(refresh_candidates)
    feedback_map = load_feedback(feedback_log)

    approved: List[Dict[str, object]] = []
    holdback: List[Dict[str, object]] = []

    stats = {
        "total": len(records),
        "approved": 0,
        "needs_work": 0,
        "rejected": 0,
        "pending": 0,
    }

    for record in records:
        record_id = compute_record_id(record)
        options = record.get("options") or []
        options_len = len(options) if isinstance(options, list) else 0
        model_index = record.get("correct_index") if isinstance(record.get("correct_index"), int) else None
        model_choice = index_to_letter(model_index, options_len)

        entry = feedback_map.get(record_id)
        if not entry:
            stats["pending"] += 1
            holdback.append(record)
            continue

        decision = str(entry.get("decision") or "").lower()
        if decision == "approve":
            answer_index = resolve_answer_index(entry, options_len, model_index)
            if answer_index is not None and 0 <= answer_index < options_len:
                record["correct_index"] = answer_index
            metadata = dict(record.get("metadata") or {})
            review_meta = {
                "decision": "approve",
                "note": entry.get("note") or "",
                "answer_index": record.get("correct_index"),
                "answer_choice": index_to_letter(record.get("correct_index"), options_len),
                "model_choice": entry.get("model_choice") or model_choice,
            }
            if entry.get("answer_choice"):
                review_meta["provided_answer"] = str(entry["answer_choice"]).upper()
            metadata["review"] = review_meta
            record["metadata"] = metadata
            approved.append(record)
            stats["approved"] += 1
        elif decision in {"needs-work", "needs_work", "needs"}:
            holdback.append(record)
            stats["needs_work"] += 1
        elif decision == "reject":
            holdback.append(record)
            stats["rejected"] += 1
        else:
            holdback.append(record)
            stats["pending"] += 1

    if approved:
        write_jsonl(approved_output, approved)
    elif approved_output.exists():
        approved_output.unlink()

    if holdback:
        write_jsonl(holdback_output, holdback)
    elif holdback_output.exists():
        holdback_output.unlink()

    return stats


def run_refresh_cycle(args: argparse.Namespace) -> None:
    print("Step 1/8: prepare_refresh.py")
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

    print("Step 2/8: auto_generate.py")
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

    print("Step 3/8: filter_balance.py")
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

    print("Step 4/8: apply reviewer decisions")
    stats = apply_review_feedback(
        args.refresh_candidates,
        args.feedback_log,
        args.approved_refresh,
        args.holdback_refresh,
    )
    print(
        f"Applied review log: {stats['approved']} approved, "
        f"{stats['needs_work']} needs-work, {stats['rejected']} rejected, "
        f"{stats['pending']} pending."
    )

    if not args.skip_format:
        print("Step 5/8: format_for_sft.py")
        inputs: List[str] = [str(args.seed_data)]
        if stats["approved"] > 0:
            inputs.append(str(args.approved_refresh))
        format_args: List[str] = ["--inputs", *inputs, "--manifest", str(args.formatted_manifest)]
        format_args.extend(
            [
                "--stats-path",
                "results/dataset_stats.json",
                "--output-dir",
                "data/formatted",
            ]
        )
        format_for_sft.main(format_args)
    else:
        print("Step 5/8: format_for_sft.py skipped")

    print("Step 6/8: train_lora.py")
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

    print("Step 7/8: eval.py")
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
        print("Step 8/8: compare_adapters.py")
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
        print("Step 8/8: comparison skipped (provide --compare-with to enable)")

    if args.bundle:
        print("Creating handoff bundle...")
        default_paths = [
            str(args.refresh_candidates),
            str(args.approved_refresh),
            str(args.holdback_refresh),
            str(args.feedback_log),
            str(eval_output / "metrics.json"),
            str(eval_output / "samples.jsonl"),
            "results/adapter_comparison.json",
            "results/adapter_comparison.md",
            "docs/feedback-playbook.md",
            "scripts/feedback_queue.py",
            "config/adapters.yaml",
            str(adapter_dir),
        ]
        bundle_paths = [path for path in default_paths if path] + args.bundle_paths
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

