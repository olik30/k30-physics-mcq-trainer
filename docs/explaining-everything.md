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
- `results/` – evaluation metrics and sample outputs.
- `.venv*/` – local Python environments (ignored in Git).

## MOST IMPORTANT FOLDER
- `models/` – this is where the trained model actually lives.
  - **Weight checkpoints** (e.g., `model.safetensors`, `adapter.bin`): huge tensor files that hold the learned parameters—the “brain” we fine-tune. 
  - Think of this as the master recipe that tells the AI how to answer questions.

  - **Adapter layers** (if we use LoRA/PEFT): smaller delta-weight files that sit beside the base checkpoint so we can swap or merge variations without retraining from scratch. 
  - Picture them as lightweight add-ons that give the base model new tricks without rebuilding everything.

  - **Config files** (e.g., `config.json`, `generation_config.json`): describe the architecture, layer sizes, and default decoding settings so any runtime knows how to wire the weights. - They act like blueprints so the software knows how to assemble the brain correctly.

  - **Tokenizer assets** (e.g., `tokenizer.json`, `tokenizer.model`, `special_tokens_map.json`): map between raw text and token IDs so the model can read prompts and write answers.
  - They’re the translation dictionary between human words and what the model understands.

  - **Metadata docs** (optional README or YAML): notes about which dataset/run produced the weights, useful for provenance when we ship updates. 
  - This is the travel diary that says where the brain came from and what it’s been trained on.

## When the project finishes, what the AI can do
- Generate fresh AQA-style MCQs from the parsed past papers, complete with hints and explanations.
- Keep answers numerically accurate and student-friendly, so they look like real exam items.
- Auto-tag each question with AO, topic, and difficulty for easy filtering or lesson planning.
- Run locally/offline via Ollama—no outside API needed once the weights sit in `models/`.
- Feed approved questions straight into `auto_train_candidates.jsonl`, so new training runs can start immediately.
