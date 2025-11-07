"""Generate MCQ drafts using a local model (Ollama qwen2.5-7b-instruct).

This script takes parsed question parts, feeds them to the model with
mark-scheme context, and writes provisional MCQ drafts for human review.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple
ASCII_REPLACEMENTS = {
    "π": "pi",
    "Π": "Pi",
    "μ": "mu",
    "Μ": "Mu",
    "Ω": "ohm",
    "ω": "omega",
    "β": "beta",
    "α": "alpha",
    "γ": "gamma",
    "λ": "lambda",
    "ν": "nu",
    "×": " x ",
    "·": " ",
    "−": "-",
    "–": "-",
    "—": "-",
    "⁺": "+",
    "⁻": "-",
    "°": " deg ",
    "Δ": "delta",
    "θ": "theta",
    "σ": "sigma",
    "φ": "phi",
    "η": "eta",
    "τ": "tau",
    "κ": "kappa",
    "ρ": "rho",
}

WHITESPACE_RE = re.compile(r"\s+")
DOUBLE_SPACE_RE = re.compile(r"  +")
BROKEN_SCI_NOTATION_RE = re.compile(r"\b\d+\s+10\^[+-]?\d+")
PLACEHOLDER_RE = re.compile(r"x10\^[0-9-]+\b")
RAW_SCI_RE = re.compile(r"x10\^\s*$")
NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*x\s*10\^[+-]?\d+)?")




DEFAULT_INPUT = Path("data/parsed/questions.jsonl")
DEFAULT_OUTPUT = Path("data/parsed/auto_drafts.jsonl")
DEFAULT_LOG_DIR = Path("logs/auto_generate")
SEED_TRAIN_PATH = Path("data/parsed/seed_train.jsonl")
SEED_DRAFTS_PATH = Path("data/parsed/seed_drafts.jsonl")

SYSTEM_PROMPT = (
    "You are an assistant that produces strict JSON MCQs for AQA physics exams. "
    "Return valid JSON with keys: question (string), options (array of 4 distinct strings), correct_index (int), "
    "hint (string), explanation (string), ao (string), topic (string), difficulty (string). "
    "Every answer must respect the original mark scheme magnitude—if the mark scheme is numeric, reuse that value "
    "with sensible significant figures, e.g. '1.4 x10^3'. Never output bare tokens like 'x10^3'. "
    "Always infer an AO tag (AO1/AO2/AO3 combos) and a short topic label (e.g. 'Mechanics > Forces'). "
    "Output ASCII text only. No commentary or markdown beyond the JSON."
)


def load_question_parts(path: Path) -> Iterator[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stem = (record.get("stem") or "").strip()
            for part in record.get("parts") or []:
                prompt = (part.get("prompt") or "").strip()
                mark_scheme = (part.get("mark_scheme") or "").strip()
                if stem and prompt and mark_scheme:
                    figures: List[Dict[str, object]] = []
                    for fig in (record.get("assets") or {}).get("figures") or []:
                        description = fig.get("caption") or fig.get("summary") or fig.get("ocr")
                        if description:
                            figures.append({
                                "path": fig.get("path"),
                                "description": description,
                            })

                    yield {
                        "question_id": record.get("question_id"),
                        "part_code": part.get("code"),
                        "stem": stem,
                        "prompt": prompt,
                        "mark_scheme": mark_scheme,
                        "paper": record.get("paper"),
                        "session": record.get("session"),
                        "figures": figures,
                    }


def build_user_prompt(entry: Dict[str, object]) -> str:
    prompt_lines = [
        "Question (as seen in the paper):",
        entry["stem"],
        entry["prompt"],
        "",
        "Mark scheme highlights (the correct idea/value):",
        entry["mark_scheme"],
    ]
    figures: Sequence[Dict[str, object]] = entry.get("figures") or []
    if figures:
        prompt_lines.extend([
            "",
            "Image context (text descriptions only):",
        ])
        for fig in figures:
            description = fig.get("description")
            if description:
                prompt_lines.append(str(description))

    prompt_lines.extend([
        "",
        "Produce one MCQ following the schema. Keep language student-friendly, avoid rubric wording, and ensure "
        "all numbers and units match the mark scheme."
    ])
    return "\n".join(prompt_lines)


def run_ollama(prompt: str, model: str) -> str:
    command = [
        "ollama",
        "run",
        model,
        "--format",
        "json",
    ]
    result = subprocess.run(command, input=prompt.encode("utf-8"), capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Ollama command failed: {result.stderr.decode('utf-8', errors='ignore')}")
    return result.stdout.decode("utf-8", errors="ignore")


def sanitise_text(value: Optional[str]) -> str:
    text = (value or "").replace("\r\n", "\n").strip()
    if not text:
        return ""
    for source, target in ASCII_REPLACEMENTS.items():
        text = text.replace(source, target)
    normalised = unicodedata.normalize("NFKD", text)
    ascii_bytes = normalised.encode("ascii", "ignore")
    cleaned = ascii_bytes.decode("ascii")
    cleaned = WHITESPACE_RE.sub(" ", cleaned)
    cleaned = DOUBLE_SPACE_RE.sub(" ", cleaned)
    return cleaned.strip()


def sanitise_record(data: Dict[str, object]) -> Dict[str, object]:
    data["question"] = sanitise_text(str(data.get("question") or ""))
    data["hint"] = sanitise_text(str(data.get("hint") or ""))
    data["explanation"] = sanitise_text(str(data.get("explanation") or ""))
    options = data.get("options") or []
    data["options"] = [sanitise_text(str(opt)) for opt in options]
    ao = data.get("ao")
    data["ao"] = sanitise_text(str(ao)) if ao not in (None, "") else None
    topic = data.get("topic")
    data["topic"] = sanitise_text(str(topic)) if topic not in (None, "") else None
    difficulty = data.get("difficulty")
    data["difficulty"] = sanitise_text(str(difficulty)) or "UNSPECIFIED"
    for key in ("question", "explanation", "hint"):
        if BROKEN_SCI_NOTATION_RE.search(data[key]):
            data[key] = PLACEHOLDER_RE.sub(" x10^n", data[key])
        if RAW_SCI_RE.search(data[key]):
            data[key] = data[key].strip().rstrip("x10^").strip()
    return data


def validate_record(data: Dict[str, object]) -> List[str]:
    errors: List[str] = []
    options = data.get("options")
    if not isinstance(options, list) or len(options) != 4:
        errors.append("options must be a list of 4 strings")
    else:
        dedup = {opt.lower() for opt in options}
        if len(dedup) != 4:
            errors.append("options must be distinct")
        for opt in options:
            if RAW_SCI_RE.search(opt) or PLACEHOLDER_RE.search(opt):
                errors.append("option contains invalid scientific notation placeholder")
                break
    correct_index = data.get("correct_index")
    if not isinstance(correct_index, int) or not 0 <= correct_index < 4:
        errors.append("correct_index must be an int between 0 and 3")
    else:
        if isinstance(options, list) and (correct_index >= len(options) or not options[correct_index]):
            errors.append("correct option text missing")
    for field in ("question", "explanation"):
        if not data.get(field):
            errors.append(f"{field} missing")
        elif RAW_SCI_RE.search(data[field]) or PLACEHOLDER_RE.search(data[field]):
            errors.append(f"{field} contains invalid scientific notation placeholder")
    for field in ("ao", "topic", "difficulty"):
        value = data.get(field)
        if not value or not str(value).strip():
            errors.append(f"{field} missing")
    return errors


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MCQ drafts via local model")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default="qwen2.5:7b-instruct")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of question parts to process",
    )
    parser.add_argument(
        "--variants-per-source",
        type=int,
        default=1,
        help="Number of MCQ variants to generate for each question part",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Generate even if question part already exists in seed/train datasets",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    output_path = args.output
    log_dir = DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / "run.jsonl"

    if output_path.exists():
        logging.warning("Output file %s already exists; aborting to avoid overwrite", output_path)
        return

    existing_keys: Set[Tuple[str, str]] = set()
    if not args.allow_existing:
        for existing_path in (SEED_TRAIN_PATH, SEED_DRAFTS_PATH):
            if not existing_path.exists():
                continue
            with existing_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for source in record.get("sources") or []:
                        if source.get("type") == "question_part":
                            existing_keys.add((str(source.get("question_id")), str(source.get("part_code"))))

    question_parts = list(load_question_parts(args.input))
    if args.allow_existing:
        entries = question_parts
    else:
        entries = [
            entry
            for entry in question_parts
            if (entry.get("question_id"), entry.get("part_code")) not in existing_keys
        ]
    logging.info("Loaded %d question parts", len(entries))

    output_records: List[Dict[str, object]] = []
    log_entries: List[Dict[str, object]] = []

    max_questions = max(args.limit, 0)
    variants = max(args.variants_per_source, 1)

    processed_questions = 0
    for entry in entries:
        if max_questions and processed_questions >= max_questions:
            break

        processed_questions += 1
        prompt_text = build_user_prompt(entry)

        for variant_index in range(variants):
            full_prompt = (
                f"<|system|>\n{SYSTEM_PROMPT}\n<|assistant|>\n"
                f"<|user|>\n{prompt_text}\n"
            )
            raw_response = ""
            try:
                raw_response = run_ollama(full_prompt, args.model)
                try:
                    data = json.loads(raw_response)
                except json.JSONDecodeError:
                    start = raw_response.find("{")
                    end = raw_response.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        data = json.loads(raw_response[start:end + 1])
                    else:
                        raise
                data = sanitise_record(data)
                validation_errors = validate_record(data)
                if validation_errors:
                    raise ValueError("; ".join(validation_errors))
            except Exception as exc:
                logging.error(
                    "Generation failed for %s/%s (variant %s): %s",
                    entry.get("question_id"),
                    entry.get("part_code"),
                    variant_index + 1,
                    exc,
                )
                log_entries.append({
                    "question_id": entry.get("question_id"),
                    "part_code": entry.get("part_code"),
                    "variant_id": variant_index + 1,
                    "status": "error",
                    "error": str(exc),
                    "prompt": prompt_text,
                    "raw": raw_response or None,
                })
                continue

            data["variant_id"] = variant_index + 1
            data.setdefault("metadata", {})
            metadata = data["metadata"] or {}
            metadata.update({
                "board": metadata.get("board") or "AQA",
                "paper": entry.get("paper"),
                "session": entry.get("session"),
                "source": "model_generated",
                "status": metadata.get("status") or "draft",
                "variant_id": variant_index + 1,
                "variants_per_source": variants,
                "variant_group_id": f"{entry.get('question_id')}#{entry.get('part_code')}",
            })
            data["metadata"] = metadata
            data["variant_group_id"] = metadata["variant_group_id"]
            data["sources"] = [
                {
                    "type": "question_part",
                    "question_id": entry.get("question_id"),
                    "part_code": entry.get("part_code"),
                },
                {
                    "type": "mark_scheme",
                    "question_id": entry.get("question_id"),
                    "part_code": entry.get("part_code"),
                },
            ]

            output_records.append(data)
            log_entries.append({
                "question_id": entry.get("question_id"),
                "part_code": entry.get("part_code"),
                "variant_id": variant_index + 1,
                "status": "ok",
            })

    if not output_records:
        logging.warning("No drafts were generated.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for record in output_records:
            json.dump(record, stream, ensure_ascii=False)
            stream.write("\n")
    logging.info("Wrote %d drafts to %s", len(output_records), output_path)

    with run_log.open("w", encoding="utf-8") as stream:
        for entry in log_entries:
            json.dump(entry, stream, ensure_ascii=False)
            stream.write("\n")


if __name__ == "__main__":
    main()


