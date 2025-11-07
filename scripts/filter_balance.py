"""Filter and balance MCQ datasets with reusable sanitisation and coverage stats.

Day 9 requires a deterministic way to sanity check every record that feeds
fine-tuning.  This script reuses the Day 8 sanitisation/validation helpers,
adds duplicate and metadata checks, and emits both a cleaned dataset and a
report that highlights coverage gaps (AO/topic/difficulty).

Example usage:

    python scripts/filter_balance.py \
        --input data/parsed/seed_train.jsonl \
        --output data/filtered/seed_train.filtered.jsonl \
        --report reports/day9_seed_train.json \
        --coverage-report reports/day9_seed_train.md \
        --emit-flagged data/review/day9_seed_train_flagged.jsonl

Use ``--strict`` if you want warnings (e.g. generic topic tags) to become
drop-worthy errors.  ``--notes-mode`` summarises review decision logs instead of
running the full MCQ pipeline.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover - import fallbacks for script vs package execution
    from . import auto_generate
except ImportError:  # pragma: no cover
    import auto_generate  # type: ignore


# Canonical difficulty labels that we accept after sanitisation.  Values outside
# this mapping are preserved but surfaced as warnings so we can clean them up
# manually.
DIFFICULTY_ALIASES = {
    "AS": "AS",
    "A2": "A2",
    "A-LEVEL": "A2",
    "A LEVEL": "A2",
    "GCSE": "GCSE",
    "FOUNDATION": "Foundation",
    "HIGHER": "Higher",
    "EASY": "Easy",
    "MEDIUM": "Medium",
    "HARD": "Hard",
    "E": "Easy",
    "M": "Medium",
    "H": "Hard",
    "1": "Easy",
    "2": "AS",
    "3": "A2",
}

GENERIC_TOPIC_TOKENS = {
    "misc",
    "general",
    "other",
}


@dataclass
class ProcessedRecord:
    raw: Dict[str, Any]
    cleaned: Dict[str, Any]
    errors: List[str]
    warnings: List[str]


@dataclass
class FilterSummary:
    kept: List[Dict[str, Any]]
    flagged: List[Dict[str, Any]]
    error_counts: collections.Counter
    warning_counts: collections.Counter
    coverage: Dict[str, Dict[str, int]]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def serialise_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")


def normalise_difficulty(value: Optional[str], warnings: List[str]) -> str:
    if not value:
        return ""
    normalised = value.strip()
    if not normalised:
        return ""
    alias = DIFFICULTY_ALIASES.get(normalised.upper())
    if alias:
        return alias
    warnings.append(f"difficulty_unmapped:{normalised}")
    return normalised


def analyse_record(raw: Dict[str, Any]) -> ProcessedRecord:
    cleaned = copy.deepcopy(raw)
    cleaned = auto_generate.sanitise_record(cleaned)
    errors = list(auto_generate.validate_record(cleaned))
    warnings: List[str] = []

    difficulty = cleaned.get("difficulty")
    if not difficulty or difficulty == "UNSPECIFIED":
        errors.append("difficulty_missing")
    else:
        cleaned["difficulty"] = normalise_difficulty(str(difficulty), warnings)

    ao_value = str(cleaned.get("ao") or "").strip()
    if not ao_value:
        errors.append("ao_missing")
    elif "AO" not in ao_value.upper():
        warnings.append("ao_suspect")

    topic_value = str(cleaned.get("topic") or "").strip()
    if not topic_value:
        errors.append("topic_missing")
    else:
        topic_lower = topic_value.lower()
        if any(token in topic_lower for token in GENERIC_TOPIC_TOKENS):
            warnings.append("topic_generic")

    hint_value = str(cleaned.get("hint") or "").strip()
    if not hint_value:
        warnings.append("hint_missing")

    explanation = str(cleaned.get("explanation") or "").strip()
    if not explanation:
        errors.append("explanation_missing")
    elif len(explanation) < 40:
        warnings.append("explanation_short")

    question = str(cleaned.get("question") or "").strip()
    if len(question) < 30:
        warnings.append("question_short")

    correct_index = cleaned.get("correct_index")
    options: Sequence[str] = cleaned.get("options") or []
    if isinstance(correct_index, int) and 0 <= correct_index < len(options):
        option_text = str(options[correct_index]).strip()
        if not option_text:
            errors.append("correct_option_empty")
    metadata = cleaned.get("metadata") or {}
    if metadata:
        status = str(metadata.get("status") or "").lower()
        if status not in {"draft", "approved", "seed"}:
            warnings.append("metadata_status_unexpected")

    return ProcessedRecord(raw=raw, cleaned=cleaned, errors=errors, warnings=warnings)


def detect_duplicates(processed: List[ProcessedRecord]) -> None:
    question_index: Dict[str, List[int]] = collections.defaultdict(list)
    provenance_index: Dict[Tuple[str, str, str], List[int]] = collections.defaultdict(list)

    for idx, record in enumerate(processed):
        metadata = record.cleaned.get("metadata") or {}
        if metadata.get("variant_id"):
            # Allow multiple variants of the same question without duplicate penalties
            continue
        question = str(record.cleaned.get("question") or "").lower()
        if question:
            question_index[question].append(idx)

        seen_sources: set[Tuple[str, str, str]] = set()
        for source in record.cleaned.get("sources") or []:
            if not isinstance(source, dict):
                continue
            source_type = str(source.get("type") or "")
            qid = str(source.get("question_id") or "")
            part = str(source.get("part_code") or "")
            key = (source_type, qid, part)
            if (qid or part) and key not in seen_sources:
                provenance_index[key].append(idx)
                seen_sources.add(key)

    for indices in question_index.values():
        if len(indices) > 1:
            for dup_idx in indices[1:]:
                processed[dup_idx].errors.append("duplicate_question_text")

    for indices in provenance_index.values():
        if len(indices) > 1:
            for dup_idx in indices[1:]:
                processed[dup_idx].errors.append("duplicate_provenance")


def collect_coverage(records: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    ao_counter: collections.Counter = collections.Counter()
    topic_counter: collections.Counter = collections.Counter()
    topic_root_counter: collections.Counter = collections.Counter()
    difficulty_counter: collections.Counter = collections.Counter()

    for record in records:
        ao_counter[str(record.get("ao") or "").strip() or "(missing)"] += 1
        topic_value = str(record.get("topic") or "").strip() or "(missing)"
        topic_counter[topic_value] += 1
        root = topic_value.split(">", 1)[0].strip()
        topic_root_counter[root] += 1
        difficulty_counter[str(record.get("difficulty") or "").strip() or "(missing)"] += 1

    return {
        "ao": dict(ao_counter),
        "topic": dict(topic_counter),
        "topic_root": dict(topic_root_counter),
        "difficulty": dict(difficulty_counter),
    }


def summarise_values(counter: Dict[str, int], top: int = 10) -> List[Tuple[str, int]]:
    ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:top]


def create_markdown_report(summary: FilterSummary) -> str:
    total = len(summary.kept) + len(summary.flagged)
    kept = len(summary.kept)
    lines = [
        f"# Filter Report",
        "",
        f"- Total records: {total}",
        f"- Kept: {kept}",
        f"- Flagged: {len(summary.flagged)}",
        f"- Error types: {len(summary.error_counts)}",
        f"- Warning types: {len(summary.warning_counts)}",
        "",
        "## AO coverage (top 10)",
    ]
    for key, count in summarise_values(summary.coverage["ao"]):
        lines.append(f"- {key or '(missing)'}: {count}")

    lines.extend(["", "## Topic roots (top 10)"])
    for key, count in summarise_values(summary.coverage["topic_root"]):
        lines.append(f"- {key or '(missing)'}: {count}")

    lines.extend(["", "## Difficulty counts"])
    for key, count in summarise_values(summary.coverage["difficulty"]):
        lines.append(f"- {key or '(missing)'}: {count}")

    if summary.warning_counts:
        lines.extend(["", "## Warnings"])
        for key, count in summary.warning_counts.most_common():
            lines.append(f"- {key}: {count}")

    if summary.error_counts:
        lines.extend(["", "## Errors"])
        for key, count in summary.error_counts.most_common():
            lines.append(f"- {key}: {count}")

    return "\n".join(lines) + "\n"


def run_filter(records: List[Dict[str, Any]], strict: bool = False) -> FilterSummary:
    processed = [analyse_record(record) for record in records]
    detect_duplicates(processed)

    kept: List[Dict[str, Any]] = []
    flagged: List[Dict[str, Any]] = []
    error_counts: collections.Counter = collections.Counter()
    warning_counts: collections.Counter = collections.Counter()

    for record in processed:
        for err in record.errors:
            error_counts[err] += 1
        for warn in record.warnings:
            warning_counts[warn] += 1

        has_errors = bool(record.errors)
        has_warnings = bool(record.warnings)

        if has_errors or (strict and has_warnings):
            flagged.append({
                "record": record.cleaned,
                "errors": record.errors,
                "warnings": record.warnings,
            })
        else:
            kept.append(record.cleaned)

    coverage = collect_coverage(kept)
    return FilterSummary(
        kept=kept,
        flagged=flagged,
        error_counts=error_counts,
        warning_counts=warning_counts,
        coverage=coverage,
    )


def summarise_notes(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    decision_counter: collections.Counter = collections.Counter()
    tag_counter: collections.Counter = collections.Counter()
    question_counter: collections.Counter = collections.Counter()

    for entry in records:
        decision = str(entry.get("decision") or "(missing)")
        decision_counter[decision] += 1
        for tag in entry.get("tags") or []:
            tag_counter[str(tag)] += 1
        question_id = str(entry.get("question_id") or "")
        if question_id:
            question_counter[question_id] += 1

    most_flagged = summarise_values(question_counter, top=10)
    return {
        "total_entries": len(records),
        "decisions": dict(decision_counter),
        "tags": dict(tag_counter),
        "most_flagged_questions": most_flagged,
    }


def build_report_dict(summary: FilterSummary) -> Dict[str, Any]:
    total = len(summary.kept) + len(summary.flagged)
    return {
        "total": total,
        "kept": len(summary.kept),
        "flagged": len(summary.flagged),
        "error_counts": dict(summary.error_counts),
        "warning_counts": dict(summary.warning_counts),
        "coverage": summary.coverage,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter, normalise, and summarise MCQ datasets")
    parser.add_argument("--input", type=Path, required=True, help="Path to the source JSONL file")
    parser.add_argument("--output", type=Path, help="Write cleaned records to this JSONL path")
    parser.add_argument("--report", type=Path, help="Write a JSON report with summary statistics")
    parser.add_argument("--coverage-report", type=Path, help="Write a Markdown coverage summary")
    parser.add_argument("--emit-flagged", type=Path, help="Write flagged records (errors/warnings) to JSONL")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as fatal and drop the record")
    parser.add_argument("--notes-mode", action="store_true", help="Summarise review notes instead of MCQ data")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    records = load_jsonl(args.input)

    if args.notes_mode:
        summary = summarise_notes(records)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        else:
            print(json.dumps(summary, indent=2))
        if args.coverage_report:
            args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
            lines = ["# Review notes summary", ""]
            lines.append(f"- Total entries: {summary['total_entries']}")
            lines.append("- Decisions:")
            for key, value in summary["decisions"].items():
                lines.append(f"  - {key}: {value}")
            lines.append("- Tags:")
            for key, value in summary["tags"].items():
                lines.append(f"  - {key}: {value}")
            lines.append("- Most flagged question_ids:")
            for question_id, count in summary["most_flagged_questions"]:
                lines.append(f"  - {question_id}: {count}")
            args.coverage_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    summary = run_filter(records, strict=args.strict)

    if args.output:
        serialise_jsonl(args.output, summary.kept)

    if args.emit_flagged:
        serialise_jsonl(args.emit_flagged, summary.flagged)

    report_dict = build_report_dict(summary)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report_dict, indent=2))

    if args.coverage_report:
        args.coverage_report.parent.mkdir(parents=True, exist_ok=True)
        args.coverage_report.write_text(create_markdown_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()


