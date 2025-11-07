"""Build a lightweight SQLite index for parsed exam questions.

This utility ingests ``data/parsed/questions.jsonl`` (written by
``scripts/parse_questions.py``), normalises the records into relational tables,
and writes the results into ``data/index/questions.db``. The index lets later
pipelines issue fast lookups by board/paper/session, question code, or asset
paths without re-reading the JSONL each time. A run log is emitted to
``artifacts/logs/index/`` for traceability.

Example:

    python scripts/build_question_index.py \
        --input data/parsed/questions.jsonl \
        --database data/index/questions.db

To rebuild from scratch (dropping existing tables):

    python scripts/build_question_index.py --force
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence


DEFAULT_INPUT = Path("data/parsed/questions.jsonl")
DEFAULT_DATABASE = Path("data/index/questions.db")
DEFAULT_LOG_DIR = Path("artifacts/logs/index")


def configure_logging(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"index_build_{timestamp}.log"
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, encoding="utf-8")]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers)
    logging.info("Logging to %s", log_path)
    return log_path


def ensure_schema(conn: sqlite3.Connection, force: bool) -> None:
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    if force:
        conn.executescript(
            """
            DROP TABLE IF EXISTS assets;
            DROP TABLE IF EXISTS parts;
            DROP TABLE IF EXISTS questions;
            """
        )

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS questions (
            question_id TEXT PRIMARY KEY,
            board TEXT NOT NULL,
            paper TEXT NOT NULL,
            session TEXT NOT NULL,
            question_number TEXT NOT NULL,
            stem TEXT,
            mark_scheme_stem TEXT,
            page_numbers TEXT,
            status TEXT,
            reviewer TEXT
        );

        CREATE TABLE IF NOT EXISTS parts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT NOT NULL,
            code TEXT NOT NULL,
            prompt TEXT,
            marks INTEGER,
            mark_scheme TEXT,
            FOREIGN KEY(question_id) REFERENCES questions(question_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            page INTEGER,
            image TEXT,
            payload TEXT,
            FOREIGN KEY(question_id) REFERENCES questions(question_id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_questions_board_paper_session
            ON questions(board, paper, session);
        CREATE INDEX IF NOT EXISTS idx_parts_question ON parts(question_id);
        CREATE INDEX IF NOT EXISTS idx_assets_question ON assets(question_id);
        """
    )


def clear_existing(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DELETE FROM assets;
        DELETE FROM parts;
        DELETE FROM questions;
        VACUUM;
        """
    )


def load_records(jsonl_path: Path) -> Iterator[Dict[str, object]]:
    with jsonl_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logging.warning("Skipping malformed JSON on line %d: %s", line_number, exc)


def insert_records(conn: sqlite3.Connection, records: Iterable[Dict[str, object]]) -> int:
    question_rows: List[tuple] = []
    part_rows: List[tuple] = []
    asset_rows: List[tuple] = []

    for record in records:
        question_id = record.get("question_id")
        if not question_id:
            logging.debug("Skipping record with missing question_id: %s", record)
            continue

        board = record.get("board")
        paper = record.get("paper")
        session = record.get("session")
        question_number = record.get("question_number")
        stem = record.get("stem")
        mark_scheme_stem = record.get("mark_scheme_stem")
        page_numbers = json.dumps(record.get("page_numbers", []), ensure_ascii=False)
        metadata = record.get("metadata") or {}
        status = metadata.get("status") if isinstance(metadata, dict) else None
        reviewer = metadata.get("reviewer") if isinstance(metadata, dict) else None

        question_rows.append(
            (
                question_id,
                board,
                paper,
                session,
                question_number,
                stem,
                mark_scheme_stem,
                page_numbers,
                status,
                reviewer,
            )
        )

        parts = record.get("parts") or []
        if isinstance(parts, Sequence):
            for part in parts:
                if not isinstance(part, dict):
                    continue
                code = part.get("code")
                prompt = part.get("prompt")
                marks = part.get("marks")
                if marks is not None:
                    try:
                        marks = int(marks)
                    except (TypeError, ValueError):
                        marks = None
                mark_scheme = part.get("mark_scheme")
                part_rows.append((question_id, code, prompt, marks, mark_scheme))

        assets = record.get("assets") or {}
        if isinstance(assets, dict):
            for kind in ("captions", "graphs"):
                for payload in assets.get(kind) or []:
                    if not isinstance(payload, dict):
                        continue
                    page = payload.get("page")
                    if page is not None:
                        try:
                            page = int(page)
                        except (TypeError, ValueError):
                            page = None
                    image = payload.get("image") if isinstance(payload.get("image"), str) else None
                    payload_json = json.dumps(payload, ensure_ascii=False)
                    asset_rows.append((question_id, kind.rstrip("s"), page, image, payload_json))

    with conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO questions (
                question_id, board, paper, session, question_number,
                stem, mark_scheme_stem, page_numbers, status, reviewer
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            question_rows,
        )

        conn.executemany(
            """
            INSERT INTO parts (question_id, code, prompt, marks, mark_scheme)
            VALUES (?, ?, ?, ?, ?)
            """,
            part_rows,
        )

        conn.executemany(
            """
            INSERT INTO assets (question_id, kind, page, image, payload)
            VALUES (?, ?, ?, ?, ?)
            """,
            asset_rows,
        )

    logging.info(
        "Inserted %d questions, %d parts, %d assets",
        len(question_rows),
        len(part_rows),
        len(asset_rows),
    )
    return len(question_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a SQLite index for parsed questions")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Source JSONL file")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="SQLite database output path")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR, help="Directory for run logs")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Drop and recreate tables before inserting records",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Keep existing rows and append new ones (conflicts replace questions)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_dir)

    if not args.input.exists():
        logging.error("Input JSONL not found: %s", args.input)
        raise SystemExit(1)

    args.database.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(args.database) as conn:
        ensure_schema(conn, force=args.force)
        if not args.append:
            clear_existing(conn)

        count = insert_records(conn, load_records(args.input))
        if count == 0:
            logging.warning("No questions were inserted; check input file")

    logging.info("Index build complete; database at %s", args.database)
    logging.info("Run log saved to %s", log_path)


if __name__ == "__main__":
    main()

