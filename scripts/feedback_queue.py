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
import sys
import textwrap
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUT = Path("data/filtered/refresh_candidates.jsonl")
DEFAULT_LOG = Path("data/review/day13_feedback_log.jsonl")
DEFAULT_VARIANT_LOG = Path("data/review/variant_choices.jsonl")
DEFAULT_QUESTIONS_INDEX = Path("data/parsed/questions.jsonl")
DEFAULT_REFRESH_INDEX = Path("data/parsed/refresh_sources.jsonl")


LETTER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
ASCII_REPLACEMENTS = {
    ord("∝"): "proportional to",
    ord("√"): "sqrt",
    ord("–"): "-",
    ord("—"): "-",
    ord("×"): "x",
    ord("·"): "*",
    ord("°"): " degrees",
    ord("±"): "+/-",
    ord("µ"): "mu",
    ord("μ"): "mu",
    ord("π"): "pi",
    ord("“"): '"',
    ord("”"): '"',
    ord("‘"): "'",
    ord("’"): "'",
}


@dataclass
class SourceContext:
    question_id: str
    part_code: str
    mark_scheme: Optional[str]
    prompt: Optional[str]
    board: Optional[str]
    paper: Optional[str]
    session: Optional[str]
    question_pdf: Optional[str]
    mark_scheme_pdf: Optional[str]
    assets: List[str]


def safe_print(*values: object, sep: str = " ", end: str = "\n") -> None:
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    combined = sep.join(str(value) for value in values)
    combined = combined.translate(ASCII_REPLACEMENTS)
    text = combined + (end or "")
    sys.stdout.write(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def iter_jsonl(path: Path) -> Iterable[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return []

def extract_source_keys(record: Dict[str, object]) -> Tuple[Optional[str], Optional[str]]:
    question_id = None
    part_code = None
    for source in record.get("sources") or []:
        if not isinstance(source, dict):
            continue
        if source.get("type") == "question_part":
            question_id = str(source.get("question_id") or "") or None
            part_code = str(source.get("part_code") or "") or None
            break
    if not question_id:
        question_id = str(record.get("id") or "") or None
    return question_id, part_code


def resolve_asset_paths(raw_assets: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    base_paths = [
        Path("data/images"),
        Path("data/graph_analysis"),
        Path("data/graph_descriptions"),
        Path("data/captions"),
    ]
    for asset in raw_assets:
        if not asset:
            continue
        candidate_rel = Path(str(asset).replace("\\", "/"))
        found = None
        for base in base_paths:
            candidate = base / candidate_rel
            if candidate.exists():
                found = candidate
                break
        if found is not None:
            resolved.append(str(found))
        else:
            resolved.append(str(candidate_rel))
    return resolved


def find_pdf_path(
    board: Optional[str],
    paper: Optional[str],
    session: Optional[str],
    suffixes: Sequence[str],
    include_original: bool = True,
) -> Optional[str]:
    if not board or not paper or not session:
        return None
    base_dir = Path("data/pdfs") / board / paper
    if not base_dir.exists():
        return None
    tokens = [token for token in str(session).replace(".pdf", "").split("_") if token]
    candidates: List[str] = []
    if include_original and tokens:
        candidates.append(" ".join(tokens))
    for suffix in suffixes:
        if not tokens:
            continue
        candidate_tokens = list(tokens[:-1]) + [suffix]
        candidates.append(" ".join(candidate_tokens))

    for name in candidates:
        candidate = base_dir / f"{name}.pdf"
        if candidate.exists():
            return str(candidate)
    return None


def extract_assets(record: Dict[str, object]) -> List[str]:
    assets: List[str] = []
    asset_block = record.get("assets") or {}
    for key in ("captions", "graphs", "images"):
        items = asset_block.get(key)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            path = item.get("image") or item.get("path")
            if path:
                assets.append(str(path))
    unique_assets = list(dict.fromkeys(assets))
    return resolve_asset_paths(unique_assets)


def build_source_index() -> Dict[str, Dict[str, SourceContext]]:
    index: Dict[str, Dict[str, SourceContext]] = {}
    for path in (DEFAULT_QUESTIONS_INDEX, DEFAULT_REFRESH_INDEX):
        for record in iter_jsonl(path):
            question_id = record.get("question_id")
            if not isinstance(question_id, str) or not question_id:
                continue
            board = record.get("board") or record.get("metadata", {}).get("board")
            paper = record.get("paper") or record.get("metadata", {}).get("paper")
            session = record.get("session") or record.get("metadata", {}).get("session")
            parts = record.get("parts") or []
            assets = extract_assets(record)
            question_pdf = find_pdf_path(
                board, paper, session, suffixes=[session.split("_")[-1]] if session else [], include_original=True
            )
            mark_scheme_pdf = find_pdf_path(board, paper, session, suffixes=["MS", "MA"], include_original=False)
            for part in parts:
                if not isinstance(part, dict):
                    continue
                code = str(part.get("code") or "")
                if not code:
                    continue
                mark_scheme = part.get("mark_scheme")
                prompt = part.get("prompt")
                context = SourceContext(
                    question_id=question_id,
                    part_code=code,
                    mark_scheme=mark_scheme if isinstance(mark_scheme, str) else None,
                    prompt=prompt if isinstance(prompt, str) else None,
                    board=board if isinstance(board, str) else None,
                    paper=paper if isinstance(paper, str) else None,
                    session=session if isinstance(session, str) else None,
                    question_pdf=question_pdf,
                    mark_scheme_pdf=mark_scheme_pdf,
                    assets=assets,
                )
                index.setdefault(question_id, {})[code] = context
    return index


SOURCE_INDEX: Optional[Dict[str, Dict[str, SourceContext]]] = None


def get_source_context(question_id: Optional[str], part_code: Optional[str]) -> Optional[SourceContext]:
    global SOURCE_INDEX
    if not question_id:
        return None
    if SOURCE_INDEX is None:
        SOURCE_INDEX = build_source_index()
    part_map = SOURCE_INDEX.get(question_id)
    if not part_map:
        return None
    if part_code and part_code in part_map:
        return part_map[part_code]
    # Fallback: return any context for question if part missing.
    return next(iter(part_map.values()))


def determine_variant_group(record: Dict[str, object]) -> Optional[str]:
    metadata = record.get("metadata") or {}
    group_id = metadata.get("variant_group_id")
    if group_id:
        return str(group_id)
    question_id, part_code = extract_source_keys(record)
    if not question_id:
        return None
    return f"{question_id}#{part_code or ''}"


def resolve_variant_id(record: Dict[str, object], fallback: int) -> int:
    metadata = record.get("metadata") or {}
    variant_id = record.get("variant_id") or metadata.get("variant_id")
    if isinstance(variant_id, int):
        return variant_id
    try:
        return int(str(variant_id))
    except (TypeError, ValueError):
        return fallback


def group_variant_records(records: Sequence[Dict[str, object]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []
    for record in records:
        group_id = determine_variant_group(record)
        if not group_id:
            continue
        question_id, part_code = extract_source_keys(record)
        group = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "question_id": question_id,
                "part_code": part_code,
                "records": [],
            },
        )
        variant_index = resolve_variant_id(record, len(group["records"]) + 1)
        group["records"].append({"variant_id": variant_index, "record": record})
        if group_id not in order:
            order.append(group_id)

    grouped_list: List[Dict[str, Any]] = []
    for group_id in order:
        group = grouped[group_id]
        group_records = sorted(group["records"], key=lambda item: item["variant_id"])
        group["records"] = group_records
        grouped_list.append(group)
    return grouped_list

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


def parse_answer_choice(raw: str, options_len: int) -> Optional[int]:
    if not raw:
        return None
    idx = letter_to_index(raw, options_len)
    return idx


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


def load_variant_log(path: Path) -> Dict[str, Dict[str, object]]:
    if not path.exists():
        return {}
    entries: Dict[str, Dict[str, object]] = {}
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            group_id = str(entry.get("variant_group_id") or "")
            if not group_id:
                question_id = entry.get("question_id")
                part_code = entry.get("part_code")
                if not question_id:
                    continue
                group_id = f"{question_id}#{part_code or ''}"
            entries[group_id] = entry
    return entries


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


def render_record(
    record: Dict[str, object],
    record_id: str,
    current_decision: Optional[Dict[str, object]] = None,
    source_context: Optional[SourceContext] = None,
) -> None:
    safe_print("=" * 88)
    safe_print(f"ID: {record_id}")
    if current_decision:
        decision = current_decision.get("decision")
        note = current_decision.get("note", "")
        answer_choice = current_decision.get("answer_choice") or current_decision.get("answer")
        if answer_choice:
            safe_print(f"Current decision: {decision} (answer={answer_choice}; {note})")
        else:
            safe_print(f"Current decision: {decision} ({note})")
    question = str(record.get("question") or "")
    if question:
        safe_print("Question:")
        safe_print(wrap_text(question))
    options = record.get("options") or []
    if isinstance(options, list):
        for idx, opt in enumerate(options):
            label = chr(ord("A") + idx)
            safe_print(f"  {label}. {opt}")
    if isinstance(options, list):
        model_letter = index_to_letter(record.get("correct_index"), len(options))
        if model_letter:
            safe_print(f"Model answer: {model_letter}")
    hint = record.get("hint")
    if hint:
        safe_print("Hint:")
        safe_print(wrap_text(str(hint)))
    explanation = record.get("explanation")
    if explanation:
        safe_print("Explanation:")
        safe_print(wrap_text(str(explanation)))
    ao = record.get("ao")
    topic = record.get("topic")
    difficulty = record.get("difficulty")
    safe_print(f"Meta: AO={ao} | Topic={topic} | Difficulty={difficulty}")
    if source_context:
        origin_bits = []
        if source_context.board:
            origin_bits.append(str(source_context.board))
        if source_context.paper:
            origin_bits.append(str(source_context.paper))
        if source_context.session:
            origin_bits.append(str(source_context.session))
        source_label = " · ".join(origin_bits) if origin_bits else ""
        part_label = source_context.part_code or "(part not specified)"
        if source_context.question_id or source_label:
            descriptor = source_context.question_id or "Unknown question"
            if source_label:
                descriptor = f"{descriptor} ({source_label})"
            safe_print(f"Source: {descriptor} — part {part_label}")
        if source_context.prompt:
            safe_print("Original prompt:")
            safe_print(wrap_text(source_context.prompt, indent=4))
        if source_context.mark_scheme:
            safe_print("Mark scheme note:")
            safe_print(wrap_text(source_context.mark_scheme, indent=4))
        file_lines: List[str] = []
        if source_context.question_pdf:
            file_lines.append(f"Question PDF: {source_context.question_pdf}")
        if source_context.mark_scheme_pdf:
            file_lines.append(f"Mark scheme PDF: {source_context.mark_scheme_pdf}")
        if file_lines:
            safe_print("Source files:")
            for line in file_lines:
                safe_print(f"  - {line}")
        if source_context.assets:
            safe_print("Asset snapshots:")
            for asset in source_context.assets[:5]:
                safe_print(f"  - {asset}")
            if len(source_context.assets) > 5:
                safe_print(f"  ... ({len(source_context.assets) - 5} more)")


def prompt_preferred_variant(total_variants: int) -> Tuple[Optional[int], bool]:
    while True:
        choice = input(
            f"Preferred variant [1-{total_variants}, none=n, q=quit]: "
        ).strip().lower()
        if choice in {"", "n", "none"}:
            return None, False
        if choice in {"q", "quit"}:
            return None, True
        if choice.isdigit():
            value = int(choice)
            if 1 <= value <= total_variants:
                return value, False
        safe_print("Please enter a number within range, 'n' for none, or 'q' to quit.")


def prompt_variant_decision(variant_id: int) -> Tuple[Optional[str], bool]:
    while True:
        choice = input(
            f"Variant {variant_id} decision [a=accept, r=reject, q=quit]: "
        ).strip().lower()
        if choice in {"a", "accept"}:
            return "accept", False
        if choice in {"r", "reject"}:
            return "reject", False
        if choice in {"q", "quit"}:
            return None, True
        safe_print("Please enter 'a', 'r', or 'q'.")


def variant_review(
    dataset_path: Path,
    log_path: Path,
    show_reviewed: bool,
) -> None:
    records = load_jsonl(dataset_path)
    groups = group_variant_records(records)
    if not groups:
        safe_print("No variant groups found in dataset.")
        return

    ensure_log(log_path)
    log_entries = load_variant_log(log_path)
    reviewed = 0

    for group in groups:
        group_id = group["group_id"]
        if not show_reviewed and group_id in log_entries:
            continue

        question_id = group.get("question_id")
        part_code = group.get("part_code")
        source_context = get_source_context(question_id, part_code)

        safe_print("=" * 88)
        safe_print(f"Variant group: {group_id}")
        for item in group["records"]:
            variant_id = item["variant_id"]
            record = item["record"]
            safe_print("-" * 88)
            safe_print(f"Variant {variant_id}")
            render_record(record, compute_record_id(record), source_context=source_context)

        preferred, stop_now = prompt_preferred_variant(len(group["records"]))
        if stop_now:
            safe_print("Stopping review loop.")
            break

        variant_entries: List[Dict[str, object]] = []
        stop_requested = False
        for item in group["records"]:
            variant_id = item["variant_id"]
            decision, quit_now = prompt_variant_decision(variant_id)
            if quit_now:
                safe_print("Stopping review loop.")
                stop_requested = True
                break
            note = input("Note (why accepted/rejected / why not preferred) [optional]: ").strip()
            variant_entries.append({
                "variant_id": variant_id,
                "decision": decision,
                "note": note,
                "is_preferred": bool(preferred and variant_id == preferred),
            })

        if stop_requested:
            break

        entry = {
            "variant_group_id": group_id,
            "question_id": question_id,
            "part_code": part_code,
            "preferred_variant": preferred if isinstance(preferred, int) else None,
            "variants": variant_entries,
            "timestamp": datetime.utcnow().isoformat(),
        }
        with log_path.open("a", encoding="utf-8") as stream:
            json.dump(entry, stream, ensure_ascii=False)
            stream.write("\n")
        log_entries[group_id] = entry
        reviewed += 1

    if reviewed:
        safe_print(f"Completed {reviewed} variant reviews.")
    else:
        safe_print("No variant groups were reviewed (already completed or skipped).")


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
        question_id, part_code = extract_source_keys(record)
        source_context = get_source_context(question_id, part_code)
        if not show_reviewed and decision:
            continue
        render_record(record, record_id, decision, source_context)
        shown += 1
        if shown >= limit:
            break
    if shown == 0:
        safe_print("No records to display (all reviewed or dataset empty).")


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

    question_id, part_code = extract_source_keys(target)
    options = target.get("options") or []
    options_len = len(options) if isinstance(options, list) else 0
    model_index = target.get("correct_index") if isinstance(target.get("correct_index"), int) else None

    ensure_log(log_path)
    entry = {
        "record_id": record_id,
        "decision": decision,
        "note": note,
    }
    if question_id:
        entry["question_id"] = question_id
    if part_code:
        entry["part_code"] = part_code
    if model_index is not None:
        entry["model_choice"] = index_to_letter(model_index, options_len)
    with log_path.open("a", encoding="utf-8") as stream:
        json.dump(entry, stream, ensure_ascii=False)
        stream.write("\n")
    safe_print(f"Recorded decision for {record_id}: {decision}")


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

        existing_decision = log_entries.get(record_id)
        question_id, part_code = extract_source_keys(record)
        source_context = get_source_context(question_id, part_code)
        render_record(record, record_id, existing_decision, source_context)

        while True:
            choice = input(
                "Decision [a=approve, n=needs-work, r=reject, s=skip, q=quit]: "
            ).strip().lower()
            if not choice:
                continue
            if choice in {"q", "quit"}:
                safe_print("Stopping review loop.")
                return
            if choice in {"s", "skip"}:
                break
            mapped = decision_map.get(choice)
            if not mapped:
                safe_print("Unrecognised choice. Please enter a, n, r, s, or q.")
                continue
            options = record.get("options") or []
            options_len = len(options) if isinstance(options, list) else 0
            model_index = record.get("correct_index")
            default_index: Optional[int] = None
            if existing_decision:
                if isinstance(existing_decision.get("answer_index"), int):
                    default_index = existing_decision["answer_index"]
                elif existing_decision.get("answer_choice"):
                    default_index = letter_to_index(existing_decision["answer_choice"], options_len)
            if default_index is None and isinstance(model_index, int):
                default_index = model_index if 0 <= model_index < options_len else None
            default_letter = index_to_letter(default_index, options_len) if default_index is not None else None
            while True:
                prompt_default = default_letter or ("model" if model_index is not None else "none")
                answer_raw = input(
                    f"Correct answer? [A-{chr(ord('A') + max(options_len - 1, 0))}] (default: {prompt_default}): "
                ).strip()
                if not answer_raw:
                    answer_index = default_index
                    break
                parsed_index = parse_answer_choice(answer_raw, options_len)
                if parsed_index is not None:
                    answer_index = parsed_index
                    break
                safe_print("Please enter a valid option letter (e.g., A) or digit.")
            answer_choice = (
                index_to_letter(answer_index, options_len) if answer_index is not None else None
            )
            note = input("Reason / note (optional): ").strip()
            entry = {
                "record_id": record_id,
                "decision": mapped,
                "answer_choice": answer_choice,
                "answer_index": answer_index,
                "note": note,
                "model_choice": index_to_letter(model_index, options_len),
            }
            if question_id:
                entry["question_id"] = question_id
            if part_code:
                entry["part_code"] = part_code
            with log_path.open("a", encoding="utf-8") as stream:
                json.dump(entry, stream, ensure_ascii=False)
                stream.write("\n")
            log_entries[record_id] = entry
            reviewed_count += 1
            safe_print(f"Recorded decision for {record_id}: {mapped}")
            break

    if reviewed_count == 0:
        safe_print("No records required review (all already decided or dataset empty).")
    else:
        safe_print(f"Completed {reviewed_count} reviews.")


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

    variant_parser = subparsers.add_parser(
        "variants",
        help="Review multiple variants of each MCQ and pick the preferred ones",
    )
    variant_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    variant_parser.add_argument("--log", type=Path, default=DEFAULT_VARIANT_LOG)
    variant_parser.add_argument(
        "--show-reviewed",
        action="store_true",
        help="Include variant groups that already have decisions logged",
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
    elif args.command == "variants":
        variant_review(
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


