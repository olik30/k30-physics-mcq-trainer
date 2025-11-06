"""Select evaluation failures and build a targeted generation source file.

Day 13 requires us to:
* Parse the structured evaluation log (`results/adapter_v1/samples.jsonl`)
* Tag problematic items (JSON parse errors, schema failures, duplicate distractors, etc.)
* Produce a small JSONL containing just those question parts so `scripts/auto_generate.py`
  can regenerate them without crawling the full source dataset.

The script emits three artefacts:
1. `results/<adapter>/refresh_summary.json` – overall counts + reasons
2. `results/<adapter>/refresh_summary.md` – human-readable checklist
3. `data/parsed/refresh_sources.jsonl` – the trimmed question-part records

Example usage:

    python scripts/prepare_refresh.py \
        --evaluation results/adapter_v1/samples.jsonl \
        --formatted data/formatted/test.jsonl \
        --parsed data/parsed/questions.jsonl \
        --output-sources data/parsed/refresh_sources.jsonl \
        --summary-json results/adapter_v1/refresh_summary.json \
        --summary-md results/adapter_v1/refresh_summary.md \
        --unique-threshold 0.75
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_EVALUATION = Path("results/adapter_v1/samples.jsonl")
DEFAULT_FORMATTED = Path("data/formatted/test.jsonl")
DEFAULT_PARSED = Path("data/parsed/questions.jsonl")
DEFAULT_OUTPUT_SOURCES = Path("data/parsed/refresh_sources.jsonl")
DEFAULT_SUMMARY_JSON = Path("results/adapter_v1/refresh_summary.json")
DEFAULT_SUMMARY_MD = Path("results/adapter_v1/refresh_summary.md")


STRUCTURAL_ERROR_KEYS = {
    "options must be distinct",
    "options must be a list of 4 strings",
    "correct_index must be an int between 0 and 3",
    "question missing",
    "explanation missing",
    "ao missing",
    "topic missing",
    "difficulty missing",
}


@dataclass
class EvaluationFailure:
    record_id: str
    question_id: str
    part_code: str
    reasons: List[str]
    metrics: Dict[str, object]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build refresh sources from evaluation logs")
    parser.add_argument("--evaluation", type=Path, default=DEFAULT_EVALUATION)
    parser.add_argument("--formatted", type=Path, default=DEFAULT_FORMATTED)
    parser.add_argument("--parsed", type=Path, default=DEFAULT_PARSED)
    parser.add_argument("--output-sources", type=Path, default=DEFAULT_OUTPUT_SOURCES)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument(
        "--unique-threshold",
        type=float,
        default=0.75,
        help="Flag items whose distractor uniqueness falls below this ratio",
    )
    parser.add_argument(
        "--include-answer-mismatch",
        action="store_true",
        help="Also refresh items where the answer disagrees (even if schema is otherwise valid)",
    )
    return parser.parse_args(argv)


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
                raise ValueError(f"Invalid JSON (line {line_number}) in {path}: {exc}") from exc
    return records


def parse_sources(formatted_records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, str]]:
    """Map dataset ID to its question/part source metadata."""
    mapping: Dict[str, Dict[str, str]] = {}
    for record in formatted_records:
        record_id = record.get("id")
        if not isinstance(record_id, str):
            continue
        metadata = record.get("metadata") or {}
        sources = metadata.get("sources") or []
        question_id = None
        part_code = None
        for source in sources:
            if source.get("type") == "question_part":
                question_id = str(source.get("question_id"))
                part_code = str(source.get("part_code"))
                break
        mapping[record_id] = {
            "question_id": question_id or "",
            "part_code": part_code or "",
        }
    return mapping


def load_parsed_index(path: Path) -> Dict[str, Dict[str, object]]:
    """Return question_id -> full question record (with parts)."""
    index: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON (line {line_number}) in {path}: {exc}") from exc
            question_id = record.get("question_id")
            if isinstance(question_id, str):
                index[question_id] = record
    return index


def extract_failures(
    samples: Iterable[Dict[str, object]],
    id_map: Dict[str, Dict[str, str]],
    unique_threshold: float,
    include_answer_mismatch: bool,
) -> List[EvaluationFailure]:
    failures: List[EvaluationFailure] = []

    for sample in samples:
        record_id = sample.get("id")
        if not isinstance(record_id, str):
            continue
        source_info = id_map.get(record_id) or {"question_id": "", "part_code": ""}
        question_id = source_info["question_id"]
        part_code = source_info["part_code"]
        if not question_id or not part_code:
            continue

        reasons = []
        if sample.get("parse_error"):
            reasons.append(f"parse_error:{sample['parse_error']}")
        if not sample.get("json_valid", False):
            reasons.append("json_invalid")
        schema_errors = sample.get("schema_errors") or []
        if schema_errors:
            reasons.extend(f"schema:{err}" for err in schema_errors)
        unique_ratio = float(sample.get("distractor_unique_ratio") or 0.0)
        if unique_ratio < unique_threshold:
            reasons.append("low_distractor_diversity")
        if not sample.get("numeric_ok", True):
            reasons.append("numeric_placeholder")
        if include_answer_mismatch and not sample.get("answer_match", False):
            reasons.append("answer_mismatch")

        structural_fail = any(
            key.startswith("schema:") and key.split("schema:", 1)[1] in STRUCTURAL_ERROR_KEYS
            for key in reasons
        ) or "json_invalid" in reasons or any(r.startswith("parse_error") for r in reasons)

        diversity_fail = "low_distractor_diversity" in reasons
        should_include = structural_fail or diversity_fail or ("answer_mismatch" in reasons)

        if not should_include:
            continue

        failures.append(
            EvaluationFailure(
                record_id=record_id,
                question_id=question_id,
                part_code=part_code,
                reasons=reasons,
                metrics={
                    "json_valid": sample.get("json_valid"),
                    "schema_valid": sample.get("schema_valid"),
                    "answer_match": sample.get("answer_match"),
                    "unique_ratio": unique_ratio,
                    "hint_chars": sample.get("hint_chars"),
                    "explanation_chars": sample.get("explanation_chars"),
                },
            )
        )

    return failures


def build_refresh_dataset(
    failures: Iterable[EvaluationFailure],
    parsed_index: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    grouped_parts: Dict[str, List[str]] = defaultdict(list)
    for failure in failures:
        grouped_parts[failure.question_id].append(failure.part_code)

    refresh_records: List[Dict[str, object]] = []
    for question_id, parts in grouped_parts.items():
        original = parsed_index.get(question_id)
        if not original:
            continue
        new_record = {key: value for key, value in original.items() if key != "parts"}
        new_record["parts"] = [
            part
            for part in original.get("parts", [])
            if part.get("code") in parts
        ]
        if new_record["parts"]:
            refresh_records.append(new_record)
    return refresh_records


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")


def write_summary_json(
    path: Path,
    failures: Sequence[EvaluationFailure],
    reason_counts: Counter,
    output_sources: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "total_flagged": len(failures),
        "reason_counts": dict(reason_counts),
        "output_sources": str(output_sources),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_summary_md(
    path: Path,
    failures: Sequence[EvaluationFailure],
    reason_counts: Counter,
    output_sources: Path,
) -> None:
    lines = [
        "# Day 13 Refresh Summary",
        "",
        f"- **Flagged items**: {len(failures)}",
        "- **Reason breakdown**:",
    ]
    for reason, count in reason_counts.most_common():
        lines.append(f"  - {reason}: {count}")
    lines.extend(
        [
            "",
            f"- **Refresh source file**: `{output_sources}`",
            "",
            "## Itemised List",
            "",
        ]
    )
    for failure in failures:
        reason_str = ", ".join(failure.reasons) or "n/a"
        lines.append(
            f"- `{failure.record_id}` → `{failure.question_id}` / `{failure.part_code}` ({reason_str})"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    evaluation_records = load_jsonl(args.evaluation)
    formatted_records = load_jsonl(args.formatted)
    id_map = parse_sources(formatted_records)
    parsed_index = load_parsed_index(args.parsed)

    failures = extract_failures(
        samples=evaluation_records,
        id_map=id_map,
        unique_threshold=args.unique_threshold,
        include_answer_mismatch=args.include_answer_mismatch,
    )

    if not failures:
        print("No records met the refresh criteria. Nothing to do.")
        return

    refresh_records = build_refresh_dataset(failures, parsed_index)
    if not refresh_records:
        print("Failed to build refresh dataset (missing parsed entries).")
        return

    write_jsonl(args.output_sources, refresh_records)

    reason_counter = Counter()
    for failure in failures:
        for reason in failure.reasons:
            reason_counter[reason] += 1

    write_summary_json(args.summary_json, failures, reason_counter, args.output_sources)
    write_summary_md(args.summary_md, failures, reason_counter, args.output_sources)

    print(f"Prepared {len(refresh_records)} question records for regeneration.")
    print(f"Summary written to {args.summary_json} and {args.summary_md}.")


if __name__ == "__main__":
    main()


