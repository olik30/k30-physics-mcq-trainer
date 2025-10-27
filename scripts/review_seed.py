"""Lightweight CLI review tool for seed MCQ drafts.

This utility loads draft MCQs (produced by ``scripts/make_seed_data.py``),
iterates through each record, and prompts the reviewer to accept/reject the
item with optional notes. Decisions and comments are appended to
``data/review/seed_notes.jsonl`` so that the curation step can ingest human
feedback.

Usage example::

    python scripts/review_seed.py \
        --drafts data/parsed/seed_drafts.jsonl \
        --notes data/review/seed_notes.jsonl \
        --start 0

Within the interactive loop, commands are kept intentionally simple:

* ``a`` – accept (status ``approved``)
* ``r`` – reject (status ``rejected``)
* ``s`` – skip (no change)
* ``sample <tag>`` – mark as stratified QA sample (status ``sampled_ok``)
* ``img`` – open linked image/graph assets in the default viewer
* ``b`` – go back to previous item
* ``q`` – quit review

All actions can include an optional comment, entered after the command.
``review_seed.py`` keeps a local cache of prior decisions to avoid duplicate
entries when re-running the tool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


DEFAULT_DRAFTS = Path("data/parsed/seed_drafts.jsonl")
DEFAULT_NOTES = Path("data/review/seed_notes.jsonl")
IMAGES_ROOT = Path("data/images")


@dataclass
class DraftRecord:
    raw: Dict[str, object]
    index: int

    @property
    def id(self) -> str:
        return str(self.raw.get("id"))


def load_jsonl(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False)
        stream.write("\n")


def summarize(record: DraftRecord) -> str:
    raw = record.raw
    options = raw.get("options") or []
    options_fmt = "\n".join(f"    [{idx}] {opt}" for idx, opt in enumerate(options))
    metadata = raw.get("metadata") or {}
    flags = metadata.get("flags") or []
    sources = raw.get("sources") or []
    lines = [
        f"ID: {raw.get('id')}",
        f"Question: {raw.get('question')}",
        "Options:",
        options_fmt,
        f"Correct index: {raw.get('correct_index')}",
        f"Explanation: {raw.get('explanation')}",
        f"Hint: {raw.get('hint')}",
        f"AO: {raw.get('ao')} | Topic: {raw.get('topic')} | Difficulty: {raw.get('difficulty')}",
        f"Flags: {', '.join(flags) if flags else 'None'}",
        f"Sources: {sources}",
        f"Image context: {raw.get('image_context') or 'None'}",
    ]
    return "\n".join(lines)


def load_previous_notes(path: Path) -> Dict[str, Dict[str, object]]:
    notes: Dict[str, Dict[str, object]] = {}
    for entry in load_jsonl(path):
        draft_id = entry.get("id")
        if isinstance(draft_id, str):
            notes[draft_id] = entry
    return notes


def collect_decision(notes_cache: Dict[str, Dict[str, object]], draft_id: str) -> Optional[Dict[str, object]]:
    try:
        command = input("Command [a/r/s/sample/img/b/q] > ").strip()
    except EOFError:
        return {"action": "quit"}
    if not command:
        return None
    if command.lower() in {"q", "quit"}:
        return {"action": "quit"}
    if command.lower() in {"b", "back"}:
        return {"action": "back"}
    if command.lower() in {"img", "image"}:
        return {"action": "open_image"}

    parts = command.split(maxsplit=1)
    action = parts[0].lower()
    comment = parts[1].strip() if len(parts) > 1 else ""
    if action == "sample":
        status = "sampled_ok"
        tag = comment or ""
        return {"action": status, "comment": tag}
    if action not in {"a", "r", "s"}:
        print("Unrecognised command; enter a/r/s/sample/b/q/img optionally followed by a comment.")
        return None
    return {
        "action": {"a": "approved", "r": "rejected", "s": "skipped"}[action],
        "comment": comment,
    }


def open_image_context(record: DraftRecord) -> None:
    context = record.raw.get("image_context") or []
    if not context:
        print("No image context available for this draft.")
        return

    for entry in context:
        path = entry.get("path")
        if not isinstance(path, str):
            continue
        primary = IMAGES_ROOT / path
        candidates = [primary]
        if not primary.exists():
            candidates.append(Path(path))
        for full_path in candidates:
            if not full_path.exists():
                continue
            try:
                if os.name == "nt":
                    os.startfile(str(full_path))  # type: ignore[attr-defined]
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(full_path)], check=False)
                else:
                    subprocess.run(["xdg-open", str(full_path)], check=False)
                print(f"Opened image: {full_path}")
                break
            except Exception as exc:  # pragma: no cover - best effort
                print(f"Failed to open image {full_path}: {exc}")
        else:
            print(f"Image not found: {primary}")
        if not full_path.exists():
            print(f"Image not found: {full_path}")
            continue
        try:
            if os.name == "nt":
                os.startfile(str(full_path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", str(full_path)], check=False)
            else:
                subprocess.run(["xdg-open", str(full_path)], check=False)
            print(f"Opened image: {full_path}")
        except Exception as exc:  # pragma: no cover - best effort
            print(f"Failed to open image {full_path}: {exc}")


def log_sample_activity(notes_path: Path, tag: str) -> None:
    if not tag:
        return
    sample_log = notes_path.with_name("seed_samples.log")
    with sample_log.open("a", encoding="utf-8") as stream:
        stream.write(f"{tag}\n")


def review_loop(drafts: List[DraftRecord], notes_path: Path) -> None:
    notes_cache = load_previous_notes(notes_path)
    logging.info("Loaded %d prior notes", len(notes_cache))

    position = 0
    while 0 <= position < len(drafts):
        record = drafts[position]
        draft_id = record.id
        os.system("cls" if os.name == "nt" else "clear")
        print(f"Draft {position + 1}/{len(drafts)}\n")
        print(summarize(record))
        if draft_id in notes_cache:
            existing = notes_cache[draft_id]
            print("\nExisting decision:", existing.get("status"), existing.get("comment"))

        decision = collect_decision(notes_cache, draft_id)
        if decision is None:
            continue
        action = decision["action"]

        if action == "open_image":
            open_image_context(record)
            continue
        if action == "sampled_ok":
            entry = {
                "id": draft_id,
                "status": "sampled_ok",
                "comment": decision.get("comment"),
            }
            append_jsonl(notes_path, entry)
            notes_cache[draft_id] = entry
            log_sample_activity(notes_path, decision.get("comment", ""))
            logging.info("%s -> sampled_ok", draft_id)
            position += 1
            continue
        if action == "quit":
            logging.info("Review session terminated by user at position %d", position)
            break
        if action == "back":
            position = max(position - 1, 0)
            continue
        if action == "skipped":
            position += 1
            continue

        entry = {
            "id": draft_id,
            "status": decision["action"],
            "comment": decision.get("comment"),
        }
        append_jsonl(notes_path, entry)
        notes_cache[draft_id] = entry
        logging.info("%s -> %s", draft_id, entry["status"])
        position += 1

    logging.info("Review complete. Processed %d drafts", position)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive review CLI for seed MCQ drafts")
    parser.add_argument("--drafts", type=Path, default=DEFAULT_DRAFTS, help="Path to seed draft JSONL")
    parser.add_argument("--notes", type=Path, default=DEFAULT_NOTES, help="Path to append reviewer decisions")
    parser.add_argument("--start", type=int, default=0, help="Index to start reviewing from (0-based)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    return parser.parse_args()


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(levelname)s | %(message)s")


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    if not args.drafts.exists():
        logging.error("Draft file not found: %s", args.drafts)
        raise SystemExit(1)

    raw_records = load_jsonl(args.drafts)
    if not raw_records:
        logging.warning("No draft records found in %s", args.drafts)
        return

    drafts = [DraftRecord(raw=rec, index=idx) for idx, rec in enumerate(raw_records)]
    start = max(0, min(args.start, len(drafts) - 1))
    logging.info("Starting review from index %d", start)
    review_loop(drafts[start:], args.notes)


if __name__ == "__main__":
    main()


