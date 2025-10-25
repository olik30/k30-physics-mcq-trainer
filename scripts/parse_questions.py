"""Parse exam questions into structured JSONL records.

This script walks the OCR text exported under ``data/raw`` together with
captions/graph metadata, segments each question, attaches the matching mark
scheme text, and writes one tidy JSON object per question to
``data/parsed/questions.jsonl``.

The output schema (per record):

```
{
  "question_id": "AQA/AQA_Paper_1/June_2019_QP/Q02",
  "board": "AQA",
  "paper": "AQA_Paper_1",
  "session": "June_2019_QP",
  "question_number": "02",
  "page_numbers": [4, 5, 6],
  "stem": "...",
  "parts": [
      {
         "code": "02.1",
         "prompt": "...",
         "marks": 2,
         "mark_scheme": "..."
      },
      ...
  ],
  "mark_scheme_stem": "...",  # optional text covering whole question
  "assets": {
      "captions": [ {"image": "...", "text": "...", "page": 10} ],
      "graphs": [ {"image": "...", "chart_type": "...", "page": 10} ],
  },
  "metadata": {
      "status": "pending",
      "reviewer": null
  }
}
```

Usage example (process everything):

    python scripts/parse_questions.py

Process a single board/paper/session:

    python scripts/parse_questions.py --board AQA --paper AQA_Paper_1 --session June_2019_QP

The script avoids copying images. It simply records the relative paths so later
steps can load them on demand.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


RAW_ROOT = Path("data/raw")
CAPTION_ROOT = Path("data/captions")
GRAPH_ROOT = Path("data/graph_analysis")
OUTPUT_PATH = Path("data/parsed/questions.jsonl")
RUN_LOG_DIR = Path("logs/parse")


HEADER_EXACT = {
    "pppmmmttt",
    "do not write",
    "do not write outside the box",
    "outside the",
    "the box",
    "turn over ►",
    "turn over",
    "turn over for the next question",
    "question continues on the next page",
    "* *",
    "*  *",
    "\ufeff",
}

HEADER_FRAGMENT = (
    "ib/m/",
    "time allowed",
    "please write clearly",
    "candidate number",
    "surname",
    "forename",
    "candidate signature",
    "materials",
    "for examiner’s use",
    "question mark",
    "instructions",
    "information",
    "copyright",
    "this paper",
    "question continues",
    "blank page",
    "mark scheme",
    "instructions to examiners",
    "emboldening",
    "marking points",
    "level of response",
    "answer all questions",
    "comments/guidance",
    "general marking guidance",
    "marking guidance",
    "ignore / insufficient / do not allow",
    "significant figure",
    "unit penalties",
    "marking points",
    "marking of lists",
    "marking procedure",
    "phonetic spelling",
    "interpretation of",
    "errors carried forward",
    "level of response mark schemes",
    "determining a level",
    "indicative content",
)


QUESTION_PATTERN = re.compile(r"^(?P<prefix>[Qq]?)(?P<code>\d{1,2})(?P<suffix>(?:\.\d{1,2})?)(?::|\.|\)|-|\s)*(?P<rest>.*)$")
MARK_PATTERN = re.compile(r"\[(?P<marks>\d+)\s+marks?\]", re.IGNORECASE)


@dataclass
class CaptionRecord:
    image: str
    text: str
    page_number: int
    asset_index: Optional[int] = None
    figure_references: Sequence[str] = field(default_factory=list)


@dataclass
class GraphRecord:
    image: str
    page_number: int
    chart_type: Optional[str] = None
    graph_metadata: Optional[Dict[str, object]] = None


@dataclass
class PageBundle:
    page_number: int
    text_lines: List[str]
    captions: List[CaptionRecord] = field(default_factory=list)
    graphs: List[GraphRecord] = field(default_factory=list)


@dataclass
class QuestionPart:
    code: str
    prompt: str
    marks: Optional[int]
    mark_scheme: Optional[str]


@dataclass
class QuestionRecord:
    question_id: str
    board: str
    paper: str
    session: str
    question_number: str
    page_numbers: List[int]
    stem: str
    parts: List[QuestionPart]
    mark_scheme_stem: Optional[str]
    assets: Dict[str, List[Dict[str, object]]]
    metadata: Dict[str, object]


HEADER_TRAILERS = (
    "turn over for the next question",
    "turn over",
    "turn over >",
    "turn over ►",
    "question continues on the next page",
    "question continues",
    "do not write outside the box",
    "outside the box",
    "box",
)

TRAILER_RE = re.compile(r"\b(turn over for the next question|turn over( >|►)?|question continues.*|do not write outside the box|outside the box|box)\b", re.IGNORECASE)

RUBRIC_TOKENS = (
    "do not allow",
    "no credit",
    "allow",
    "accept",
    "ignore",
    "condone",
    "reward",
    "marking points",
    "marking guidance",
    "marking point",
    "marking procedure",
    "marking of",
    "indicative content",
)

RUBRIC_PREFIXES = (
    "allow",
    "condone",
    "accept",
    "ignore",
    "insufficient",
    "credit",
    "ecf",
    "max",
    "total",
    "penalise",
    "mark",
    "marks",
)

BULLET_RE = re.compile(r"[•●▪■▲◦▪■❖✱✳✦✧✪✫✬✭✮✯✰✳✴✵✶✷✸✹✺✻✼✽✾❁❂❃❇❈❉❊❋◾◼◻◽◾✕✗✘✚✛✜✢✣✤✥✦✧★☆]+")
RUBRIC_INLINE_RE = re.compile(r"\b(?:allow|accept|ignore|condone|insufficient|ecf|max(?:imum)?|total|credit|penalise|reward)\b.*", re.IGNORECASE)

MS_HEADER_RE = re.compile(r"\b(question\s+answer\s+comments?|guidance|mark\s*scheme|comments?/guidance)\b", re.IGNORECASE)

BULLET_LEAD_RE = re.compile(r"^(?:[-–•●▪■\*]+\s*)", re.IGNORECASE)
BULLET_INFIX_RE = re.compile(r"\b(?:any\s+two\s+from|any\s+three\s+from|max\s+\d+|any\s+\d+\s+from)\b", re.IGNORECASE)
DISCARD_STARTERS = re.compile(r"^(?:allow|accept|ignore|condone|insufficient|do not|give|award|max|min|ecf)\b", re.IGNORECASE)
RUBRIC_WORD_RE = re.compile(r"\b(?:" + "|".join(re.escape(word) for word in RUBRIC_PREFIXES) + r")\b", re.IGNORECASE)


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"parse_{timestamp}.log"
    handler_console = logging.StreamHandler(sys.stdout)
    handler_file = logging.FileHandler(log_path, encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[handler_console, handler_file],
    )
    logging.info("Logging to %s", log_path)
    return log_path


def sanitise_line(line: str) -> str:
    line = line.strip().replace("\u2013", "-")
    # Merge hyphenated line breaks (e.g., "anti-" followed by "neutrino")
    if line.endswith("-"):
        line = line[:-1]
    if not line:
        return ""
    lowered = line.lower()
    lowered_no_number = re.sub(r"^[0-9\s.]+", "", lowered).strip()
    if lowered_no_number in HEADER_EXACT:
        return ""
    if lowered_no_number.startswith("do not write"):
        return ""
    if lowered in HEADER_EXACT:
        return ""
    if lowered.startswith("answer all questions"):
        return ""
    if any(fragment in lowered for fragment in HEADER_FRAGMENT):
        return ""
    if any(fragment in lowered_no_number for fragment in HEADER_FRAGMENT):
        return ""
    if not any(ch.isalpha() for ch in line):
        return ""
    return line


def collapse_digit_spacing(text: str) -> str:
    # Join digit sequences split by spaces (common in OCR for question numbers).
    def replacer(match: re.Match[str]) -> str:
        token = match.group(0)
        return token.replace(" ", "")

    collapsed = re.sub(r"(?<!\d)(?:\d\s+)+\d(?!\d)", replacer, text)
    collapsed = re.sub(r"(\d)\s*\.\s*(\d)", r"\1.\2", collapsed)
    return collapsed


def extract_leading_number(text: str) -> Optional[str]:
    match = re.search(r"\d+(?:\.\d+)?", text)
    if match:
        return match.group(0)
    return None


def strip_trailers(text: str) -> str:
    lowered = text.lower()
    for trailer in HEADER_TRAILERS:
        lowered_trailer = trailer.lower()
        if lowered.endswith(lowered_trailer):
            return text[: -len(trailer)].rstrip(" .-:")
    text = TRAILER_RE.sub("", text)
    text = re.sub(r"\bturn over\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def load_page_bundles(
    qp_dir: Path,
    board: str,
    paper: str,
    session: str,
    caption_root: Path,
    graph_root: Path,
) -> List[PageBundle]:
    pages: List[PageBundle] = []
    caption_dir = caption_root / board / paper / session
    graph_dir = graph_root / board / paper / session

    for text_path in sorted(qp_dir.glob("page_*.txt")):
        page_number = int(text_path.stem.split("_")[-1])
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        raw_text = collapse_digit_spacing(raw_text)
        lines = [line for line in (sanitise_line(l) for l in raw_text.splitlines()) if line]

        captions: List[CaptionRecord] = []
        if caption_dir.exists():
            for cap_path in sorted(caption_dir.glob(f"page_{page_number:03d}*.json")):
                try:
                    data = json.loads(cap_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    logging.warning("Failed to load caption %s: %s", cap_path, exc)
                    continue
                metadata = data.get("metadata", {})
                captions.append(
                    CaptionRecord(
                        image=data.get("image", ""),
                        text=data.get("text", ""),
                        page_number=metadata.get("page_number", page_number),
                        asset_index=metadata.get("asset_index"),
                        figure_references=tuple(metadata.get("figure_references", []) or []),
                    )
                )

        graphs: List[GraphRecord] = []
        if graph_dir.exists():
            for graph_path in sorted(graph_dir.glob(f"page_{page_number:03d}*.json")):
                try:
                    data = json.loads(graph_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    logging.warning("Failed to load graph %s: %s", graph_path, exc)
                    continue

                graph_meta = data.get("graph_metadata", {})
                graphs.append(
                    GraphRecord(
                        image=data.get("image", ""),
                        page_number=page_number,
                        chart_type=graph_meta.get("chart_type"),
                        graph_metadata=graph_meta,
                    )
                )

        pages.append(PageBundle(page_number=page_number, text_lines=lines, captions=captions, graphs=graphs))

    return pages


def parse_mark_scheme(ms_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, List[str]] = {}
    if not ms_dir.exists():
        logging.warning("Mark scheme directory missing: %s", ms_dir)
        return {}

    for text_path in sorted(ms_dir.glob("page_*.txt")):
        raw_text = text_path.read_text(encoding="utf-8", errors="ignore")
        raw_text = collapse_digit_spacing(raw_text)

        current_code: Optional[str] = None
        buffer: List[str] = []

        def flush_buffer() -> None:
            nonlocal buffer, current_code
            if current_code and buffer:
                mapping.setdefault(current_code, []).extend(buffer)
            buffer = []

        for line in raw_text.splitlines():
            raw_line = line.strip()
            line = sanitise_line(raw_line)
            if not line:
                continue
            match = QUESTION_PATTERN.match(line)
            if not match:
                if current_code:
                    buffer.append(raw_line)
                continue
            prefix = match.group("prefix") or ""
            code = match.group("code")
            rest = match.group("rest").strip()
            full_code = code
            if prefix and prefix.lower() == "q":
                rest = rest.lstrip(":.- ")
            suffix = match.group("suffix") or ""
            if suffix:
                full_code = f"{code}{suffix}"
            cleaned = sanitise_line(rest)
            if cleaned:
                flush_buffer()
                current_code = full_code
                mapping.setdefault(full_code, []).append(cleaned)
            else:
                flush_buffer()
                current_code = full_code
                mapping.setdefault(full_code, [])

        flush_buffer()

    merged: Dict[str, str] = {}
    for code, lines in mapping.items():
        if not code:
            continue
        if len(code) == 1:
            continue
        clean_lines: List[str] = []
        for raw_line in lines:
            candidate = raw_line.strip()
            candidate = MARK_PATTERN.sub("", candidate).strip()
            candidate = BULLET_RE.sub(" ", candidate)
            candidate = strip_trailers(candidate)
            candidate = sanitise_line(candidate)
            if not candidate:
                continue
            candidate = BULLET_LEAD_RE.sub("", candidate).strip()
            if not candidate or BULLET_INFIX_RE.search(candidate):
                continue
            candidate = re.sub(r"^(\d{1,2}(?:\.\d{1,2})?)\s*", "", candidate).strip()
            inline_match = RUBRIC_INLINE_RE.search(candidate)
            if inline_match:
                candidate = candidate[: inline_match.start()].strip()
                if not candidate:
                    continue
            words = candidate.split()
            if not words:
                continue
            first_word = re.sub(r"[^a-z]", "", words[0].lower())
            if first_word in RUBRIC_PREFIXES:
                remainder = " ".join(words[1:]).strip()
                if not remainder:
                    continue
                candidate = remainder
            if DISCARD_STARTERS.match(candidate):
                continue
            keyword_stop = re.search(r"\b(allow|condone|ignore|range|ecf|credit|score|min|max|all\s+\d|both|either)\b", candidate, re.IGNORECASE)
            if keyword_stop:
                candidate = candidate[: keyword_stop.start()].strip()
                if not candidate:
                    continue
            candidate = re.sub(r"\b\d+\s*$", "", candidate).strip()
            candidate = candidate.rstrip(" ,;:")
            candidate = re.split(r"[.;]", candidate)[0].strip()
            if not candidate:
                continue
            lowered_candidate = candidate.lower()
            if any(phrase in lowered_candidate for phrase in ("emboldened", "used to indicate", "brackets", "mark scheme")):
                continue
            numeric = extract_leading_number(candidate)
            if numeric and len(candidate.split()) > 3:
                candidate = numeric
            clean_lines.append(candidate)
            # Prefer the first clean statement per code to avoid rubric duplicates
            break

        if not clean_lines:
            continue

        text = " ".join(clean_lines).strip()
        # Keep only the first short clause before rubric keywords or additional guidance
        first_sentence = re.split(r"[.;]", text, maxsplit=1)[0].strip()
        if first_sentence:
            text = first_sentence
        lowered_text = text.lower()
        if len(text.split()) > 40 and not any(char.isdigit() for char in text):
            continue
        if any(fragment in lowered_text for fragment in HEADER_FRAGMENT):
            continue
        if MS_HEADER_RE.search(lowered_text):
            continue
        if any(token in lowered_text for token in RUBRIC_TOKENS):
            continue

        if text:
            merged[code] = text
    return merged


def extract_marks(text: str) -> Optional[int]:
    match = MARK_PATTERN.search(text)
    if not match:
        return None
    try:
        return int(match.group("marks"))
    except ValueError:
        return None


def remove_mark_tokens(text: str) -> str:
    return MARK_PATTERN.sub("", text).strip()


def parse_questions_from_pages(
    pages: Sequence[PageBundle],
    mark_scheme: Dict[str, str],
    board: str,
    paper: str,
    session: str,
) -> List[QuestionRecord]:
    records: List[QuestionRecord] = []

    current_question_code: Optional[str] = None
    current_stem_lines: List[str] = []
    current_pages: List[int] = []
    current_assets: Dict[str, List[Dict[str, object]]] = {
        "captions": [],
        "graphs": [],
    }
    parts: Dict[str, List[str]] = {}
    current_part_code: Optional[str] = None

    def ensure_page_recorded(page_number: int) -> None:
        if page_number not in current_pages:
            current_pages.append(page_number)

    def flush_current() -> None:
        nonlocal current_question_code, current_stem_lines, current_pages, parts, current_part_code, current_assets
        if not current_question_code:
            return

        stem_text = " ".join(current_stem_lines).strip()
        stem_text = strip_trailers(stem_text)
        stem_text = remove_mark_tokens(stem_text)

        question_parts: List[QuestionPart] = []
        for part_code in sorted(parts.keys(), key=lambda c: [int(x) for x in c.split(".")]):
            part_lines = parts[part_code]
            part_text = " ".join(part_lines).strip()
            part_text = strip_trailers(part_text)
            marks = extract_marks(part_text)
            part_text = remove_mark_tokens(part_text)
            question_parts.append(
                QuestionPart(
                    code=part_code,
                    prompt=part_text,
                    marks=marks,
                    mark_scheme=mark_scheme.get(part_code),
                )
            )

        question_index = len([r for r in records if r.board == board and r.paper == paper and r.session == session]) + 1
        question_id = f"{board}/{paper}/{session}/Q{current_question_code.zfill(2)}"
        page_set = sorted(set(current_pages))
        assets = {
            "captions": current_assets["captions"],
            "graphs": current_assets["graphs"],
        }

        records.append(
            QuestionRecord(
                question_id=question_id,
                board=board,
                paper=paper,
                session=session,
                question_number=current_question_code or str(question_index),
                page_numbers=page_set,
                stem=stem_text,
                parts=question_parts,
                mark_scheme_stem=mark_scheme.get(current_question_code),
                assets=assets,
                metadata={"status": "pending", "reviewer": None},
            )
        )

        current_question_code = None
        current_stem_lines = []
        current_pages = []
        parts = {}
        current_part_code = None
        current_assets = {
            "captions": [],
            "graphs": [],
        }

    for bundle in pages:
        for raw_line in bundle.text_lines:
            line = sanitise_line(raw_line)
            if not line:
                continue

            match = QUESTION_PATTERN.match(line)
            if not match:
                if current_part_code:
                    ensure_page_recorded(bundle.page_number)
                    parts.setdefault(current_part_code, []).append(line)
                elif current_question_code:
                    ensure_page_recorded(bundle.page_number)
                    current_stem_lines.append(line)
                continue

            prefix = match.group("prefix") or ""
            code = match.group("code")
            suffix = match.group("suffix") or ""
            rest = match.group("rest").strip()
            if code == "0" and not suffix:
                continue

            rest_clean = sanitise_line(rest)
            if suffix and not rest_clean:
                rest_clean = rest

            if not rest and not suffix:
                continue

            if prefix and prefix.lower() == "q":
                if current_question_code and current_question_code != code:
                    flush_current()
                current_question_code = code
                current_part_code = None
                ensure_page_recorded(bundle.page_number)
                if rest_clean:
                    current_stem_lines.append(rest_clean)
                continue

            if suffix:
                main_code = code
                full_code = f"{code}{suffix}"
                if current_question_code and main_code != current_question_code:
                    flush_current()
                current_question_code = main_code
                current_part_code = full_code
                ensure_page_recorded(bundle.page_number)
                if rest_clean and not rest_clean.lower().startswith("figure"):
                    parts.setdefault(current_part_code, []).append(rest_clean)
                continue

            if current_question_code and code != current_question_code:
                flush_current()
            current_question_code = code
            current_part_code = None
            ensure_page_recorded(bundle.page_number)
            if rest_clean:
                current_stem_lines.append(rest_clean)

        if current_question_code:
            for cap in bundle.captions:
                entry = {
                    "page": cap.page_number,
                    "image": cap.image,
                    "text": cap.text,
                    "figure_refs": list(cap.figure_references),
                    "asset_index": cap.asset_index,
                }
                if entry not in current_assets["captions"]:
                    current_assets["captions"].append(entry)
            for graph in bundle.graphs:
                entry = {
                    "page": graph.page_number,
                    "image": graph.image,
                    "chart_type": graph.chart_type,
                    "metadata": graph.graph_metadata,
                }
                if entry not in current_assets["graphs"]:
                    current_assets["graphs"].append(entry)

    flush_current()

    deduped: Dict[str, QuestionRecord] = {}
    for record in records:
        key = record.question_id
        if key not in deduped:
            deduped[key] = record
        else:
            existing = deduped[key]
            existing.stem = strip_trailers(" ".join(filter(None, [existing.stem, record.stem])).strip())
            existing.page_numbers = sorted(set(existing.page_numbers + record.page_numbers))
            existing.parts.extend(record.parts)
            for kind in ("captions", "graphs"):
                existing.assets[kind].extend(record.assets.get(kind, []))

    cleaned_records: List[QuestionRecord] = []
    for record in deduped.values():
        unique_parts: Dict[str, QuestionPart] = {}
        for part in record.parts:
            unique_parts[part.code] = part
        record.parts = [unique_parts[code] for code in sorted(unique_parts.keys(), key=lambda c: [int(x) for x in c.split(".")])]
        for part in record.parts:
            if part.mark_scheme:
                part.mark_scheme = strip_trailers(part.mark_scheme)
        record.page_numbers = sorted(set(record.page_numbers))
        cleaned_records.append(record)

    return sorted(cleaned_records, key=lambda r: r.question_id)


def iter_question_papers(
    raw_root: Path, board_filter: Optional[str], paper_filter: Optional[str], session_filter: Optional[str]
) -> Iterator[tuple[str, str, str, Path]]:
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_root}")
    for board_dir in sorted(raw_root.iterdir()):
        if not board_dir.is_dir():
            continue
        board = board_dir.name
        if board_filter and board_filter != board:
            continue
        for paper_dir in sorted(board_dir.iterdir()):
            if not paper_dir.is_dir():
                continue
            paper = paper_dir.name
            if paper_filter and paper_filter != paper:
                continue
            for session_dir in sorted(paper_dir.iterdir()):
                if not session_dir.is_dir():
                    continue
                session = session_dir.name
                if not session.endswith("_QP"):
                    continue
                if session_filter and session_filter != session:
                    continue
                yield board, paper, session, session_dir


def find_mark_scheme_dir(qp_dir: Path) -> Optional[Path]:
    name = qp_dir.name
    if name.endswith("_QP"):
        candidate = qp_dir.parent / name.replace("_QP", "_MS")
        if candidate.exists():
            return candidate
        candidate = qp_dir.parent / name.replace("_QP", "_MA")
        if candidate.exists():
            return candidate
    return None


def write_jsonl(records: Iterable[QuestionRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = asdict(record)
            # Convert dataclass parts to plain dicts.
            payload["parts"] = [asdict(part) for part in record.parts]
            json.dump(payload, stream, ensure_ascii=False)
            stream.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse OCR’d exam questions into JSONL records")
    parser.add_argument("--board", help="Limit to a specific board (e.g. AQA)")
    parser.add_argument("--paper", help="Limit to a specific paper (e.g. AQA_Paper_1)")
    parser.add_argument("--session", help="Limit to a session (e.g. June_2019_QP)")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Destination JSONL file (default: data/parsed/questions.jsonl)",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing JSONL instead of overwriting",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=RUN_LOG_DIR,
        help="Directory for parser logs (default: logs/parse)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_dir)
    logging.info("Starting question parsing")

    all_records: List[QuestionRecord] = []
    for board, paper, session, qp_dir in iter_question_papers(RAW_ROOT, args.board, args.paper, args.session):
        logging.info("Processing %s / %s / %s", board, paper, session)
        pages = load_page_bundles(qp_dir, board, paper, session, CAPTION_ROOT, GRAPH_ROOT)
        if not pages:
            logging.warning("No pages found for %s", qp_dir)
            continue
        ms_dir = find_mark_scheme_dir(qp_dir)
        mark_scheme = parse_mark_scheme(ms_dir) if ms_dir else {}
        records = parse_questions_from_pages(pages, mark_scheme, board, paper, session)
        logging.info("Extracted %d questions", len(records))
        all_records.extend(records)

    if not all_records:
        logging.warning("No question records parsed; exiting without writing output")
        return

    if args.append and args.output.exists():
        existing = list(args.output.read_text(encoding="utf-8").splitlines())
        with args.output.open("a", encoding="utf-8") as stream:
            for record in all_records:
                payload = asdict(record)
                payload["parts"] = [asdict(part) for part in record.parts]
                json.dump(payload, stream, ensure_ascii=False)
                stream.write("\n")
        logging.info("Appended %d records to %s", len(all_records), args.output)
    else:
        write_jsonl(all_records, args.output)
        logging.info("Wrote %d records to %s", len(all_records), args.output)

    logging.info("Run finished; see %s for details", log_path)


if __name__ == "__main__":
    main()

