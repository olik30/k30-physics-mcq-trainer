"""Lightweight CLI for surfacing MCQs and recording reviewer feedback.

Day 13 calls for a simple tool that:
* Lists MCQs from any JSONL dataset (seed, auto, refresh)
* Skips items that already have decisions in a review log
* Lets reviewers append approve/reject decisions with free-text notes

Example usage:

    # Show the first 3 unreviewed items from the refresh candidates file
    python scripts/feedback_queue.py list \
        --input data/filtered/refresh_candidates.jsonl \
        --log data/review/day13_feedback_log.jsonl \
        --limit 3

    # Record a decision
    python scripts/feedback_queue.py annotate \
        --input data/filtered/refresh_candidates.jsonl \
        --log data/review/day13_feedback_log.jsonl \
        --record-id AQA/AQA_Paper_1/June_2017_QP/Q02#02.1 \
        --decision reject \
        --note "Explanation copies prompt; needs rewrite."
"""

from __future__ import annotations

import argparse
import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT = Path("data/filtered/refresh_candidates.jsonl")
DEFAULT_LOG = Path("data/review/day13_feedback_log.jsonl")


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return records


def ensure_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("", encoding="utf-8")


def load_log(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    decisions: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            record_id = entry.get("record_id")
            if isinstance(record_id, str):
                decisions[record_id] = entry
    return decisions


def compute_record_id(record: Dict[str, object]) -> str:
    question_id = ""
    part_code = ""
    for source in record.get("sources") or []:
        if source.get("type") == "question_part":
            question_id = str(source.get("question_id") or "")
            part_code = str(source.get("part_code") or "")
            break
    if not question_id:
        question_id = str(record.get("id") or "")
    identifier = question_id
    if part_code:
        identifier = f"{identifier}#{part_code}"
    return identifier or "unknown"


def wrap_text(text: str, width: int = 88, indent: int = 2) -> str:
    wrapper = textwrap.TextWrapper(width=width, subsequent_indent=" " * indent)
    return wrapper.fill(text)


def render_record(record: Dict[str, object], record_id: str, current_decision: Optional[Dict[str, object]] = None) -> None:
    print("=" * 88)
    print(f"ID: {record_id}")
    if current_decision:
        print(f"Current decision: {current_decision.get('decision')} ({current_decision.get('note', '')})")
    question = str(record.get("question") or "")
    if question:
        print("Question:")
        print(wrap_text(question))
    options = record.get("options") or []
    if isinstance(options, list):
        for idx, opt in enumerate(options):
            label = chr(ord("A") + idx)
            print(f"  {label}. {opt}")
    hint = record.get("hint")
    if hint:
        print("Hint:")
        print(wrap_text(str(hint)))
    explanation = record.get("explanation")
    if explanation:
        print("Explanation:")
        print(wrap_text(str(explanation)))
    ao = record.get("ao")
    topic = record.get("topic")
    difficulty = record.get("difficulty")
    print(f"Meta: AO={ao} | Topic={topic} | Difficulty={difficulty}")


def list_records(
    dataset_path: Path,
    log_path: Path,
    limit: int,
    show_reviewed: bool,
) -> None:
    records = load_jsonl(dataset_path)
    log_entries = load_log(log_path)
    shown = 0
    for record in records:
        record_id = compute_record_id(record)
        decision = log_entries.get(record_id)
        if not show_reviewed and decision:
            continue
        render_record(record, record_id, decision)
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        print("No records to display (all reviewed or dataset empty).")


def annotate_record(
    dataset_path: Path,
    log_path: Path,
    record_id: str,
    decision: str,
    note: str,
) -> None:
    records = load_jsonl(dataset_path)
    target = None
    for record in records:
        rid = compute_record_id(record)
        if rid == record_id:
            target = record
            break
    if target is None:
        raise ValueError(f"Record {record_id} not found in {dataset_path}")

    ensure_log(log_path)
    entry = {
        "record_id": record_id,
        "decision": decision,
        "note": note,
    }
    with log_path.open("a", encoding="utf-8") as stream:
        json.dump(entry, stream, ensure_ascii=False)
        stream.write("\n")
    print(f"Recorded decision for {record_id}: {decision}")


def interactive_review(
    dataset_path: Path,
    log_path: Path,
    show_reviewed: bool,
) -> None:
    records = load_jsonl(dataset_path)
    log_entries = load_log(log_path)
    ensure_log(log_path)

    decision_map = {
        "a": "approve",
        "approve": "approve",
        "n": "needs-work",
        "needs": "needs-work",
        "needs-work": "needs-work",
        "r": "reject",
        "reject": "reject",
    }

    reviewed_count = 0
    for record in records:
        record_id = compute_record_id(record)
        if not show_reviewed and record_id in log_entries:
            continue

        render_record(record, record_id, log_entries.get(record_id))

        while True:
            choice = input(
                "Decision [a=approve, n=needs-work, r=reject, s=skip, q=quit]: "
            ).strip().lower()
            if not choice:
                continue
            if choice in {"q", "quit"}:
                print("Stopping review loop.")
                return
            if choice in {"s", "skip"}:
                break
            mapped = decision_map.get(choice)
            if not mapped:
                print("Unrecognised choice. Please enter a, n, r, s, or q.")
                continue
            note = input("Note (optional): ").strip()
            entry = {
                "record_id": record_id,
                "decision": mapped,
                "note": note,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                json.dump(entry, stream, ensure_ascii=False)
                stream.write("\n")
            log_entries[record_id] = entry
            reviewed_count += 1
            print(f"Recorded decision for {record_id}: {mapped}")
            break

    if reviewed_count == 0:
        print("No records required review (all already decided or dataset empty).")
    else:
        print(f"Completed {reviewed_count} reviews.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feedback queue helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="Display unreviewed MCQs")
    list_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    list_parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    list_parser.add_argument("--limit", type=int, default=5)
    list_parser.add_argument(
        "--show-reviewed",
        action="store_true",
        help="Include items that already have decisions logged",
    )

    review_parser = subparsers.add_parser("review", help="Interactively review MCQs and record decisions")
    review_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    review_parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    review_parser.add_argument(
        "--show-reviewed",
        action="store_true",
        help="Include items that already have decisions logged",
    )

    annotate_parser = subparsers.add_parser("annotate", help="Record a decision for a specific MCQ")
    annotate_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    annotate_parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    annotate_parser.add_argument("--record-id", required=True)
    annotate_parser.add_argument(
        "--decision",
        choices=["approve", "reject", "needs-work"],
        required=True,
    )
    annotate_parser.add_argument("--note", default="", help="Optional reviewer note")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.command == "list":
        list_records(
            dataset_path=args.input,
            log_path=args.log,
            limit=args.limit,
            show_reviewed=args.show_reviewed,
        )
    elif args.command == "review":
        interactive_review(
            dataset_path=args.input,
            log_path=args.log,
            show_reviewed=args.show_reviewed,
        )
    elif args.command == "annotate":
        annotate_record(
            dataset_path=args.input,
            log_path=args.log,
            record_id=args.record_id,
            decision=args.decision,
            note=args.note,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()


