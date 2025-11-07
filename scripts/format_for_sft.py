"""Convert filtered MCQs into chat-completion format for supervised fine-tuning.

This script ingests one or more JSONL files containing validated MCQs
(as produced by `scripts/filter_balance.py`), runs an additional schema check,
then emits train/val/test splits in the OpenAI-style chat format.  It also
produces dataset statistics so downstream steps can track coverage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:  # pragma: no cover
    from . import filter_balance
except ImportError:  # pragma: no cover
    import filter_balance  # type: ignore


SYSTEM_PROMPT = (
    "You are an assistant that produces STRICT JSON for physics MCQs. "
    "Follow the exam board tone and conventions. Output exactly this JSON shape "
    "with no extra text: {question, options, correct_index, hint, explanation, ao, topic, difficulty}. "
    "No meta options (like 'All of the above'). Use correct units and significant figures."
)


def load_records(paths: Sequence[Path]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:  # pragma: no cover - guard
                    raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
    return records


def sanitise_and_validate(records: Iterable[Dict[str, object]]) -> List[Dict[str, object]]:
    clean: List[Dict[str, object]] = []
    for record in records:
        processed = filter_balance.analyse_record(record)
        if processed.errors:
            raise ValueError(f"Record failed validation: {processed.errors} (id={record.get('id')})")
        clean.append(processed.cleaned)
    return clean


def build_user_prompt(record: Dict[str, object]) -> str:
    metadata = record.get("metadata") or {}
    sources = record.get("sources") or []
    image_context = record.get("image_context") or []

    lines = [
        "Create a single physics MCQ in strict JSON using the details below.",
        "",
        f"Question stem: {record.get('question')}",
        f"Target AO: {record.get('ao')}",
        f"Topic label: {record.get('topic')}",
        f"Difficulty tag: {record.get('difficulty')}",
    ]

    if metadata:
        meta_lines = []
        for key in ("board", "paper", "session", "question_number", "marks"):
            value = metadata.get(key)
            if value is not None and value != "":
                meta_lines.append(f"  - {key}: {value}")
        if meta_lines:
            lines.append("Provenance:")
            lines.extend(meta_lines)

        orig_prompt = metadata.get("original_prompt")
        if orig_prompt and orig_prompt != record.get("question"):
            lines.append(f"Original mark-scheme note: {orig_prompt}")

    if sources:
        lines.append("Associated past-paper references:")
        for source in sources:
            if not isinstance(source, dict):
                continue
            src_type = source.get("type")
            qid = source.get("question_id")
            part = source.get("part_code")
            lines.append(f"  - {src_type}: {qid} / {part}")

    if image_context:
        lines.append("Image/snippet context (text description only):")
        for snippet in image_context:
            description = snippet.get("description") or snippet.get("path")
            if description:
                lines.append(f"  - {description}")

    lines.extend([
        "",
        "Requirements:",
        "- Invent four distinct options (A–D) and mark the correct one via the JSON field `correct_index`.",
        "- Provide a short student-friendly hint and a fuller explanation.",
        "- Keep wording plain English with exam-style tone.",
        "- Return JSON only—no markdown, prose, or extra keys.",
    ])

    return "\n".join(lines)


def build_assistant_response(record: Dict[str, object]) -> str:
    payload = {
        "question": record.get("question"),
        "options": record.get("options"),
        "correct_index": record.get("correct_index"),
        "hint": record.get("hint"),
        "explanation": record.get("explanation"),
        "ao": record.get("ao"),
        "topic": record.get("topic"),
        "difficulty": record.get("difficulty"),
    }
    return json.dumps(payload, ensure_ascii=False)


def convert_to_conversations(records: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    conversations: List[Dict[str, object]] = []
    for record in records:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(record)},
            {"role": "assistant", "content": build_assistant_response(record)},
        ]
        conversations.append({
            "id": record.get("id"),
            "messages": messages,
            "metadata": {
                "ao": record.get("ao"),
                "topic": record.get("topic"),
                "difficulty": record.get("difficulty"),
                "sources": record.get("sources"),
                "original_metadata": record.get("metadata"),
            },
        })
    return conversations


def split_dataset(records: Sequence[Dict[str, object]], train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1:
        raise ValueError("Ratios must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    total = len(records)
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train = [records[i] for i in indices[:train_end]]
    val = [records[i] for i in indices[train_end:val_end]]
    test = [records[i] for i in indices[val_end:]]
    return train, val, test


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            data = json.dumps(record, ensure_ascii=False)
            stream.write(data)
            stream.write("\n")
            digest.update(data.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def compute_stats(split_name: str, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    ao_counts: Dict[str, int] = {}
    topic_counts: Dict[str, int] = {}
    difficulty_counts: Dict[str, int] = {}

    for record in records:
        metadata = record.get("metadata") or {}
        ao = metadata.get("ao") or "(missing)"
        topic = metadata.get("topic") or "(missing)"
        difficulty = metadata.get("difficulty") or "(missing)"
        ao_counts[ao] = ao_counts.get(ao, 0) + 1
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1

    return {
        "split": split_name,
        "count": len(records),
        "ao_distribution": ao_counts,
        "topic_distribution": topic_counts,
        "difficulty_distribution": difficulty_counts,
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Format MCQs for supervised fine-tuning")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input JSONL files (filtered MCQs)")
    parser.add_argument("--output-dir", type=Path, default=Path("data/formatted"), help="Directory for formatted splits")
    parser.add_argument("--stats-path", type=Path, default=Path("artifacts/results/dataset_stats.json"), help="Path to write dataset stats JSON")
    parser.add_argument("--manifest", type=Path, default=Path("data/formatted/manifest.json"), help="Write split hashes and counts here")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation split ratio (default 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    records = load_records(args.inputs)
    clean_records = sanitise_and_validate(records)
    conversations = convert_to_conversations(clean_records)

    train, val, test = split_dataset(
        conversations,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    output_dir = args.output_dir
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    test_path = output_dir / "test.jsonl"

    train_hash = write_jsonl(train_path, train)
    val_hash = write_jsonl(val_path, val)
    test_hash = write_jsonl(test_path, test)

    stats = {
        "total_records": len(conversations),
        "splits": [
            compute_stats("train", train),
            compute_stats("val", val),
            compute_stats("test", test),
        ],
    }
    args.stats_path.parent.mkdir(parents=True, exist_ok=True)
    args.stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    manifest = {
        "train": {"path": str(train_path), "count": len(train), "sha256": train_hash},
        "val": {"path": str(val_path), "count": len(val), "sha256": val_hash},
        "test": {"path": str(test_path), "count": len(test), "sha256": test_hash},
        "source_inputs": [str(p) for p in args.inputs],
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1 - args.train_ratio - args.val_ratio,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


