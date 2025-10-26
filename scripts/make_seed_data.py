"""Assemble draft MCQs from parsed question parts.

This utility ingests ``data/parsed/questions.jsonl`` (written by
``scripts/parse_questions.py``), filters for human-friendly parts, and emits a
draft JSONL file with multiple-choice scaffolds. Each draft contains a stem,
four options (auto-suggested distractors), metadata, and provenance links.
Records that need manual polishing are flagged via metadata so reviewers can
triage quickly before approval.

Usage example::

    python scripts/make_seed_data.py \
        --input data/parsed/questions.jsonl \
        --output data/parsed/seed_drafts.jsonl \
        --max-records 200

The script errs on the side of caution: if we cannot generate credible
distractors or explanations, a record is tagged with ``status: needs_review``
and notes explaining why.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


DEFAULT_INPUT = Path("data/parsed/questions.jsonl")
DEFAULT_OUTPUT = Path("data/parsed/seed_drafts.jsonl")
DEFAULT_MAX_RECORDS = 250


NUMERIC_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
DEPENDENCY_RE = re.compile(r"(?:part|answer to|result from)\s+(?:part|question)\s*[A-Za-z0-9]+", re.I)
NOISE_FRAGMENT_RE = re.compile(r"(PMT|Turn over|Do not write outside the box|ANSWER IN THE SPACES PROVIDED)", re.I)


TEXT_DISTRACTOR_BANK: Dict[str, Sequence[str]] = {
    "particle": ["Proton", "Neutron", "Electron", "Muon", "Pion", "Kaon"],
    "interaction": [
        "Strong interaction",
        "Weak interaction",
        "Electromagnetic interaction",
        "Gravitational interaction",
    ],
    "electric": [
        "Ohm's law is not satisfied",
        "Current decreases to zero",
        "Potential difference doubles",
        "Charge remains constant",
    ],
    "wave": [
        "Amplitude decreases to zero",
        "Frequency doubles",
        "Phase difference is 90 degrees",
        "Wavelength increases",
    ],
    "nuclear": [
        "Alpha emission",
        "Beta-minus emission",
        "Gamma emission",
        "Spontaneous fission",
    ],
    "motion": [
        "Uniform acceleration",
        "Terminal velocity reached",
        "Resultant force is zero",
        "Momentum is conserved",
    ],
}

GENERIC_DISTRACTORS = [
    "Not enough information provided",
    "The value depends on external conditions",
    "It cannot occur under these assumptions",
    "This contradicts the conservation laws",
    "Measurement uncertainty prevents a conclusion",
]


TOPIC_KEYWORDS: Sequence[Tuple[str, str]] = (
    ("momentum", "Mechanics > Momentum"),
    ("velocity", "Mechanics > Kinematics"),
    ("oscillation", "Waves > Simple harmonic motion"),
    ("diffraction", "Waves > Diffraction"),
    ("capacitor", "Electricity > Capacitors"),
    ("potential difference", "Electricity > Circuits"),
    ("magnetic", "Fields > Magnetic fields"),
    ("gravitational", "Fields > Gravitational fields"),
    ("nucleus", "Nuclear physics > Structure"),
    ("quark", "Particle physics > Quark model"),
    ("half-life", "Nuclear physics > Radioactivity"),
    ("photoelectric", "Quantum physics > Photoelectric effect"),
)


AO_MAP: Sequence[Tuple[Sequence[str], str]] = (
    (("state", "define", "name", "identify"), "AO1"),
    (("calculate", "determine", "estimate", "show that"), "AO1+AO2"),
    (("explain", "justify", "discuss", "derive"), "AO2+AO3"),
)


@dataclass
class QuestionPart:
    question_id: str
    board: str
    paper: str
    session: str
    question_number: str
    stem: str
    code: str
    prompt: str
    mark_scheme: str
    marks: Optional[int]
    assets: Dict[str, Sequence[Dict[str, object]]]


def load_records(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSON on line %d: %s", line_number, exc)


def iter_candidate_parts(records: Iterable[Dict[str, object]]) -> Iterator[QuestionPart]:
    for record in records:
        board = str(record.get("board"))
        question_id = str(record.get("question_id"))
        paper = str(record.get("paper"))
        session = str(record.get("session"))
        question_number = str(record.get("question_number"))
        stem = (record.get("stem") or "").strip()
        assets = record.get("assets") or {}
        for part in record.get("parts") or []:
            prompt = (part.get("prompt") or "").strip()
            mark_scheme = (part.get("mark_scheme") or "").strip()
            if not prompt or not mark_scheme:
                continue
            marks = part.get("marks")
            try:
                marks = int(marks) if marks is not None else None
            except (TypeError, ValueError):
                marks = None
            code = str(part.get("code") or "")
            yield QuestionPart(
                question_id=question_id,
                board=board,
                paper=paper,
                session=session,
                question_number=question_number,
                stem=stem,
                code=code,
                prompt=prompt,
                mark_scheme=mark_scheme,
                marks=marks,
                assets=assets,
            )


def is_dependency(prompt: str) -> bool:
    return bool(DEPENDENCY_RE.search(prompt))


def clean_prompt(text: str) -> str:
    text = NOISE_FRAGMENT_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compose_question(stem: str, prompt: str) -> str:
    prompt = clean_prompt(prompt)
    if len(prompt.split()) < 6 and stem:
        candidate = f"{clean_prompt(stem)} {prompt}".strip()
    else:
        candidate = prompt
    candidate = candidate.replace(" ?", "?")
    if not candidate.endswith(("?", ".")):
        candidate += "?"
    return candidate


def infer_topic(prompt: str, stem: str) -> Optional[str]:
    text = f"{stem} {prompt}".lower()
    for keyword, topic in TOPIC_KEYWORDS:
        if keyword in text:
            return topic
    return "Physics > Misc"


def infer_ao(prompt: str) -> Optional[str]:
    text = prompt.lower()
    for triggers, ao in AO_MAP:
        if any(trigger in text for trigger in triggers):
            return ao
    return "AO1+AO2"


def is_numeric_answer(answer: str) -> Optional[float]:
    match = NUMERIC_RE.search(answer)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def format_numeric(value: float, template: str) -> str:
    match = NUMERIC_RE.search(template)
    if not match:
        return f"{value:.3g}"
    digits = match.group()
    decimals = 0
    if "." in digits:
        decimals = len(digits.split(".")[1])
    formatted = f"{value:.{decimals}f}" if decimals else str(int(round(value)))
    return NUMERIC_RE.sub(formatted, template, count=1)


def generate_numeric_options(answer: str) -> Tuple[List[str], List[str]]:
    base = is_numeric_answer(answer)
    if base is None:
        return [], ["numeric_parse_failed"]
    deltas = [0.0, -0.1 * base, 0.1 * base, 0.2 * base]
    variants = []
    flags: List[str] = []
    for delta in deltas:
        candidate = base + delta
        variants.append(format_numeric(candidate, answer))
    if len(set(variants)) < 4:
        # Fall back to simple rounding variants
        variants = [
            format_numeric(base, answer),
            format_numeric(base * 0.9, answer),
            format_numeric(base * 1.1, answer),
            format_numeric(base * 1.2, answer),
        ]
    if len(set(variants)) < 4:
        flags.append("numeric_duplicates")
    return variants, flags


def classify_prompt(prompt: str, stem: str) -> Optional[str]:
    text = f"{prompt} {stem}".lower()
    for keyword in TEXT_DISTRACTOR_BANK:
        if keyword in text:
            return keyword
    return None


def generate_text_options(answer: str, prompt: str, stem: str, extra_pool: Sequence[str]) -> Tuple[List[str], List[str]]:
    answer_clean = answer.strip()
    pool_key = classify_prompt(prompt, stem)
    distractors = list(extra_pool)
    if pool_key:
        distractors.extend(TEXT_DISTRACTOR_BANK[pool_key])
    distractors.extend(GENERIC_DISTRACTORS)
    distractors = [opt for opt in distractors if opt and opt.lower() != answer_clean.lower()]
    unique = []
    for opt in distractors:
        if opt not in unique:
            unique.append(opt)
    options = [answer_clean]
    options.extend(unique[:3])
    flags: List[str] = []
    if len(options) < 4:
        flags.append("insufficient_distractors")
        while len(options) < 4:
            options.append(f"Placeholder option {len(options)}")
    return options, flags


def deduplicate_options(options: List[str]) -> Tuple[List[str], List[str]]:
    seen = set()
    unique = []
    for opt in options:
        if opt.lower() in seen:
            continue
        seen.add(opt.lower())
        unique.append(opt)
    flags: List[str] = []
    if len(unique) < 4:
        flags.append("option_dedup_shrink")
        while len(unique) < 4:
            unique.append(f"Placeholder option {len(unique)}")
    return unique[:4], flags


def generate_options(part: QuestionPart, sibling_answers: Sequence[str]) -> Tuple[List[str], List[str]]:
    answer = part.mark_scheme.strip()
    sibling_pool = [s for s in sibling_answers if s and s != answer]
    flags: List[str] = []
    numeric = is_numeric_answer(answer)
    if numeric is not None:
        options, extra_flags = generate_numeric_options(answer)
        flags.extend(extra_flags)
    else:
        options, extra_flags = generate_text_options(answer, part.prompt, part.stem, sibling_pool)
        flags.extend(extra_flags)
    options, dedup_flags = deduplicate_options(options)
    flags.extend(dedup_flags)
    random.shuffle(options)
    try:
        correct_index = options.index(answer)
    except ValueError:
        options[0] = answer
        correct_index = 0
        flags.append("answer_not_found_post_shuffle")
    return options, flags, correct_index


def build_hint(prompt: str) -> str:
    text = prompt.lower()
    if "calculate" in text or "determine" in text:
        return "Use the given values with the appropriate formula before rounding carefully."
    if "state" in text or "name" in text:
        return "Recall the key definition or fact highlighted in this topic."
    if "explain" in text or "justify" in text:
        return "Link the underlying principle to the observation described in the question."
    return "Focus on the core principle this question targets."


def build_explanation(answer: str, prompt: str) -> str:
    prompt_clean = clean_prompt(prompt)
    if is_numeric_answer(answer) is not None:
        return f"Evaluating the quantities described in the prompt yields {answer} as required."
    return f"The correct response is {answer} because it directly addresses: {prompt_clean}."


def build_sources(part: QuestionPart) -> List[Dict[str, object]]:
    sources: List[Dict[str, object]] = [
        {
            "type": "question_part",
            "question_id": part.question_id,
            "part_code": part.code,
        },
        {
            "type": "mark_scheme",
            "question_id": part.question_id,
            "part_code": part.code,
        },
    ]
    return sources


def build_image_context(part: QuestionPart) -> List[Dict[str, object]]:
    context: List[Dict[str, object]] = []
    graphs = part.assets.get("graphs") if isinstance(part.assets, dict) else None
    if graphs:
        for graph in graphs[:1]:
            path = graph.get("image")
            if isinstance(path, str):
                context.append({"type": "graph", "path": path})
    captions = part.assets.get("captions") if isinstance(part.assets, dict) else None
    if captions and not context:
        snippet = captions[0].get("image") if isinstance(captions[0], dict) else None
        if isinstance(snippet, str):
            context.append({"type": "caption", "path": snippet})
    return context


def compose_id(part: QuestionPart) -> str:
    slug = part.question_id.replace("/", "_").replace(" ", "")
    code = part.code.replace(".", "_").replace(" ", "")
    return f"{slug}_{code}" if code else slug


def assemble_record(part: QuestionPart, sibling_answers: Sequence[str]) -> Dict[str, object]:
    question_text = compose_question(part.stem, part.prompt)
    options, flags, correct_index = generate_options(part, sibling_answers)
    topic = infer_topic(part.prompt, part.stem)
    ao = infer_ao(part.prompt)
    record = {
        "id": compose_id(part),
        "question": question_text,
        "options": options,
        "correct_index": correct_index,
        "explanation": build_explanation(part.mark_scheme, part.prompt),
        "hint": build_hint(part.prompt),
        "ao": ao,
        "topic": topic,
        "difficulty": infer_difficulty(part.paper),
        "sources": build_sources(part),
        "image_context": build_image_context(part),
        "metadata": {
            "board": part.board,
            "paper": part.paper,
            "session": part.session,
            "question_number": part.question_number,
            "marks": part.marks,
            "original_prompt": clean_prompt(part.prompt),
            "flags": flags,
        },
    }
    record["metadata"]["status"] = "draft" if not flags else "needs_review"
    return record


def infer_difficulty(paper: str) -> Optional[str]:
    paper_lower = paper.lower()
    if "as" in paper_lower:
        return "AS"
    if "a2" in paper_lower or "paper_3" in paper_lower or "paper_2" in paper_lower:
        return "A2"
    if "gcse" in paper_lower:
        return "GCSE"
    return None


def eligible(part: QuestionPart) -> bool:
    if not part.mark_scheme or len(part.mark_scheme.strip()) < 2:
        return False
    if part.marks is not None and part.marks > 4:
        return False
    if is_dependency(part.prompt):
        return False
    prompt = clean_prompt(part.prompt)
    if len(prompt.split()) < 3 and not part.stem:
        return False
    return True


def collect_sibling_answers(parts: Sequence[QuestionPart], target: QuestionPart) -> List[str]:
    answers = []
    for part in parts:
        if part is target:
            continue
        candidate = (part.mark_scheme or "").strip()
        if candidate:
            answers.append(candidate)
    return answers


def generate_drafts(candidates: Sequence[QuestionPart], max_records: int) -> List[Dict[str, object]]:
    grouped: Dict[str, List[QuestionPart]] = {}
    for part in candidates:
        grouped.setdefault(part.question_id, []).append(part)

    drafts: List[Dict[str, object]] = []
    for question_parts in grouped.values():
        for part in question_parts:
            if not eligible(part):
                continue
            sibling_answers = collect_sibling_answers(question_parts, part)
            drafts.append(assemble_record(part, sibling_answers))
    drafts.sort(key=lambda rec: rec["id"])
    if max_records:
        drafts = drafts[:max_records]
    return drafts


def write_jsonl(records: Sequence[Dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            json.dump(record, stream, ensure_ascii=True)
            stream.write("\n")


def configure_logging(log_level: str) -> None:
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(levelname)s | %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate draft MCQs from parsed question parts")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSONL file from Day 6 parser")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination JSONL for draft MCQs")
    parser.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS, help="Cap on number of draft items")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle eligible parts before slicing to max-records",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.input.exists():
        logging.error("Input file not found: %s", args.input)
        raise SystemExit(1)

    logging.info("Loading parsed questions from %s", args.input)
    records = list(load_records(args.input))
    candidates = list(iter_candidate_parts(records))
    logging.info("Loaded %d question parts", len(candidates))

    eligible_parts = [part for part in candidates if eligible(part)]
    if args.shuffle:
        random.shuffle(eligible_parts)
    logging.info("Eligible parts after filtering: %d", len(eligible_parts))

    drafts = generate_drafts(eligible_parts, args.max_records)
    logging.info("Generated %d draft MCQs", len(drafts))

    write_jsonl(drafts, args.output)
    logging.info("Drafts written to %s", args.output)


if __name__ == "__main__":
    main()


