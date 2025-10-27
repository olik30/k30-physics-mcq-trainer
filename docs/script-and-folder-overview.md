# Scripts & Folders (Quick Reference)

## Scripts (with day introduced)
- `scripts/extract_pdf.py` (Day 2) – pulls text and images from each PDF, saves to `data/raw/` and `data/images/`.
- `scripts/ocr_images.py` (Day 3) – OCR for figures; writes captions to `data/captions/`.
- `scripts/analyse_graphs.py` (Day 3) – detects chart metadata (axes, scales) into `data/graph_analysis/`.
- `scripts/describe_graphs.py` (Day 3) – turns graph metadata into plain-language summaries in `data/graph_descriptions/`.
- `scripts/parse_questions.py` (Day 6) – merges paper text, captions, and mark schemes into structured question records (`data/parsed/questions.jsonl`).
- `scripts/build_question_index.py` (Day 6) – loads the parsed questions into a SQLite DB (`data/index/questions.db`).
- `scripts/train_graph_model.py` (Day 5) – trains the diagram understanding adapter.
- `scripts/eval_graph_model.py` (Day 5) – evaluates the adapter; logs metrics in `results/graph_reader_v1/`.
- `scripts/make_seed_data.py` (Day 7) – generates MCQ drafts with flags for risky items (`data/parsed/seed_drafts.jsonl`).
- `scripts/review_seed.py` (Day 7) – CLI to approve/reject/sample drafts; updates `data/review/seed_notes.jsonl`.

## Root folders
- `docs/` – all project notes, daily recaps, coverage tables.
- `scripts/` – Python tools for extraction, OCR, parsing, MCQ curation, and model training.
- `data/` – extracted text/images, parsed questions, MCQ drafts/train sets, indices.
- `logs/` – run logs for extraction, parsing, analysis, training.
- `models/` – trained adapters (Day 5 graph model currently).
- `results/` – evaluation metrics and sample outputs.
- `.venv*/` – local Python environments (ignored in Git).

