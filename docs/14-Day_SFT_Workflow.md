**Two-Week Training Plan (Physics MCQ Generator)**

**Why train our own model?**
The current Azure-backed decoder still drops accuracy despite multiple guardrails: it hallucinates intermediate steps, produces weak hints, and sometimes misaligns MCQ steps with mark-scheme logic. Maintaining that stack is costly (per-call Azure fees) and requires layered post-processing to patch symptoms. Training a local SFT model lets us ground generation directly on our own papers, enforce board-specific structure natively, and eliminate recurring cloud costs while keeping the guardrails we already built.

**Assumptions**
- Local Windows machine with RTX 5080-class GPU (≥16 GB VRAM), Ryzen 9800X3D-class CPU, 48 GB RAM
- Only free/on-device tools (no paid APIs or cloud training)
- Git (or similar) available for tracking scripts and dataset snapshots

Flow: (Collect PDFs) -> (Extract Text+Images) -> (OCR/Captions) -> (Parse Q + Mark Scheme) -> (Index + Review) -> (Make MCQs) -> (Filter/Balance + Validate) -> (Format + Snapshot) -> (Fine-Tune) -> (Evaluate + Log) -> (Serve + Test)

In short: we turn your papers and mark schemes into a clean, labeled dataset (including simple diagram info), teach a local model to produce strict JSON MCQs in your exam style, and use light checks and lookups to keep answers correct—all running on your PC with little to no cost.

---

**What we’re building**

\- A local model that outputs strict JSON MCQs (4 options, correct index, hint, explanation, AO)  
\- It understands diagrams/graphs by using simple image captions or OCR text  
\- It stays accurate by looking up the mark scheme (“open‑book”) before answering

---

**Current dataset scope (Oct 2025 update)**
- Active board: AQA (all raw text, images, OCR captions, graph analysis, and descriptions live under `data/`)
- Archived boards: EDEXCEL, Eduqas, OCR_A, OCR_B, WJEC, CAIE (moved to `other_syllabuses/` at repo root until we resume multi-board runs)
- When you resume a board, move its tree back under `data/` before rerunning extraction/OCR scripts and refresh the coverage tables.

---

NOTE: MARKDOWN PROGRESS + WHAT'S IMPLEMENENTED + DELIVERABLES WHEN YOU FINISH A DAY

**Day 1 — Collect, catalogue & secure data**
- Put all PDFs into `data/pdfs/` (papers and their matching mark schemes)
- Keep filenames consistent (e.g., `AQA_2019_P1.pdf`, `AQA_2019_P1_MS.pdf`)
- Track coverage (board, year, topic, marks) inside `docs/internal-notes.md` so the gap table lives with the rest of the project notes
- If material is sensitive, store `data/` on encrypted storage (BitLocker) and document handling rules
- Initialise Git LFS and track `data/` + `logs/` so large binaries stay out of normal git history; record commands in `docs/internal-notes.md`
- Goal: clean, organized, auditable source files

**Day 2 — Extract text + images (batch + log)**
- Create `scripts/extract_pdf.py` to iterate over `data/pdfs/`, pull text per page, export images, and emit structured logs/errors.
- Remove stale `data/raw/` + `data/images/` before a rerun so new outputs reflect the latest heuristics (high-DPI vector clips, raster filters).
- Run `scripts/extract_pdf.py` (tune with flags like `--render-dpi`, `--min-raster-contrast`, `--drawing-cluster-gap`) to extract per-page text and targeted image crops; store logs in `logs/extract/`.
- New dependency: install OpenCV (`pip install opencv-python`) so the extractor can rescale rasters and filter logo artifacts.
- Output:
  - `data/raw/<paper>_page<N>.txt`
  - `data/images/<paper>_page<N>_img<M>.png` and `..._render.png` for graphs or page snapshots (vector clips optional via `--enable-vector-clips`)
- Review `logs/extract/run_*.jsonl` for any `status: "error"` PDFs, adjust parameters if needed, and note issues/fixes in `docs/internal-notes.md`
- For the current AQA-only phase, confirm `other_syllabuses/` still holds the archived boards so fresh runs don’t mix scopes; update coverage notes in `docs/internal-notes.md` to reflect "AQA complete, others pending".
- Create first git commit (`extract-baseline`) for reproducibility

**Day 3 — OCR, graph analysis & parallel cleanup**
- Create `scripts/ocr_images.py` to batch OCR stored page images, capture axis/label text, and write tidy JSON captions per asset.
- Create `scripts/analyse_graphs.py` to detect chart-like images, merge related vector clips emitted by `scripts/extract_pdf.py` into per-graph composites, and harvest rich geometry features (axis detection, line statistics, keypoints, component counts) into `data/graph_analysis/` while logging to `logs/graph_analysis/`.
- Graph composites run through an axis sanity-check—if the stitched vector view fails, the script automatically reuses the sharp page render as a fallback so we never digitise a broken collage.
- Create `scripts/describe_graphs.py` to read the graph metadata + OCR captions, optionally call open-source chart readers (ChartOCR/UniChart), and write a plain-language `graph_summary` back to each JSON (trend, peaks, component notes) for every real graph while skipping non-graph assets.
- Run `scripts/ocr_images.py` to capture labels (axes, units, numbers, component names) and save captions to `data/captions/<image>.json`.
- Run `scripts/analyse_graphs.py` to populate geometry metadata; run `scripts/describe_graphs.py` afterwards so every graph has both structured features and readable text (logs go to `logs/graph_descriptions/`).
- While OCR/graph passes run, start cleaning noisy text (headers, watermarks) and log fixes in `logs/cleanup/`.
- **Human**: skim any `status="warn"`/`skip` outputs and record diagrams that still read poorly (feed these into the benchmark on Day 4).
- Update coverage table with OCR + graph-analysis + description status.
- Note: with only AQA active, mark the remaining boards (EDEXCEL, Eduqas, OCR_A, OCR_B, WJEC, CAIE) as "queued" so the Day 4 benchmark knows its current scope.

**Day 4 — Graph benchmark & manual summaries**
- Sample 30-40 representative diagrams (mix of standard charts and complex schematics) into `data/graph_benchmark/`.
- **Human**: label each with component notes, expected textual summary, and correctness checks; this becomes the target set for automated graph reading.
- Record which boards/topics are covered and log gaps in `INTERNAL-NOTES.md`.

**Day 5 — Graph understanding pre-training**
- Set up a fresh Python 3.12 virtual environment (`.venv312`) and install the core Hugging Face stack in one go: `transformers`, `peft`, `accelerate`, `bitsandbytes`, `datasets`, `sentence-transformers`, `scikit-learn`, `einops`, `safetensors`, `torch`, `torchvision`, `torchaudio` (CUDA 12.4 builds).
- Confirm the 40 human-labelled graphs in `data/graph_human_descriptions/AQA` all share the same fields and filenames so the trainer can link each PNG with its plain-English notes.
- Build `scripts/train_graph_model.py`: it streams each PNG directly from disk, pairs it with the human summary, and fine-tunes `Qwen/Qwen2.5-VL-7B-Instruct` in 4-bit (QLoRA) with LoRA rank 16 / α 32 / dropout 0.05, gradient accumulation 4 (effective batch size 4). Training logs land in `logs/train/graph_reader_v1/run_*.json` and the adapter saves to `models/adapters/graph_reader_v1/`.
- Add `scripts/eval_graph_model.py`: reload the base model + adapter, generate summaries for the 8 held-out diagrams, and measure Rouge-L, BLEU-1, and sentence-transformer cosine similarity (flag anything <0.70). Metrics write to `results/graph_reader_v1/metrics.json` and `samples.jsonl` for easy review.
- **Human**: keep an eye on GPU VRAM and temperatures (stick with the default batch size if the RTX 5080 is under 12 GB load; drop to grad_accum 2 if needed), then skim the evaluation JSON to be sure the summaries stay factual before moving to Day 6.

**Day 6 — Parse questions, align mark schemes & seed retrieval**
- Create `scripts/parse_questions.py` to combine raw text, captions, graph metadata/summary, and mark schemes into structured question objects with metadata links.
- Run `scripts/parse_questions.py` to detect questions, sections, marks and write the cleaned output to `data/parsed/questions.jsonl` (1004 questions).
- Matched each item to its mark-scheme text, trimming rubric-only lines so we keep concise answers (numeric answers survive as plain numbers).
- Linked images/graphs by page and figure reference; graph metadata is attached so later steps can pull axes and trend info.
- Built the SQLite lookup (`scripts/build_question_index.py --force`) so `data/index/questions.db` now holds questions (1004), parts (1126) and assets (3194) with run logs in `logs/index/`.
- **Human next**: spot-check a few entries in `data/parsed/questions.jsonl` and the SQLite DB (`data/index/questions.db`) to confirm question text, parts, mark schemes, and assets line up.

- **Day 7 — Create first training examples + review lane**
- Harden `scripts/make_seed_data.py` so each draft includes clear flags (missing figure, duplicate option, rubric text leak) before hitting review.
- Run the generator over all parsed parts (~505 drafts this run) and store them in `data/parsed/seed_drafts.jsonl`.
- Use `scripts/review_seed.py` to log reviewer decisions; first reject the flagged drafts, then sample a handful of clean items before approving the remainder.
- Promote all approved items into `data/parsed/seed_train.jsonl` (current total: 460 approved MCQs) and record notes in `data/review/seed_notes.jsonl`.
- **Human**: tidy the rejected list for future rewrites, double-check any diagram-heavy items, and confirm reviewer counts before moving on.
- **Result**: Clean MCQ starter pack (460 items in `seed_train.jsonl`), 45 rejected items tagged for rework, plus a review log we can reuse on new drafts.

**Day 8 — Expand with local model help + human triage**
- `scripts/auto_generate.py` now calls Ollama `qwen2.5:7b-instruct` with stricter prompts (ASCII-only, schema enforced) and sanitises outputs.
- Generated 100 auto drafts (`data/parsed/auto_drafts.jsonl`) plus run log (`logs/auto_generate/run.jsonl`). Common issues captured: poor numeric accuracy (e.g. mass mis-scaling), weak particle-physics reasoning, AO/topic left null.
- Started review via `scripts/review_seed.py`; rejected flawed examples and queued the remainder for stratified sampling before promotion.
- Saved reviewer decisions to `data/review/seed_notes.jsonl`; flagged items listed there for rewrite on Day 9.
- **Human next**: finish the review pass, build a clean `auto_train_candidates.jsonl` subset, and document any prompt tweaks needed before the next batch.
- Follow-up pilot with refined prompt/validators (`--limit 20`) still produced weak circuit reasoning, so all items remain rejected pending another generation pass.

**Day 9 — Filter, unit-test & balance**
- Ship `scripts/filter_balance.py` with flags for `--output`, `--report`, `--coverage-report`, `--emit-flagged`, and `--strict` so we can reuse it on any JSONL pile (seeds, auto drafts, review notes).
- Reuse the Day 8 sanitiser/validator inside the script, then add extra guards for empty options, missing AO/topic/difficulty tags, short explanations, and duplicate question/provenance combos.
- Emit coverage snapshots for AO/topic/difficulty plus a JSONL of anything we drop (`data/review/day9_seed_train_flagged.jsonl`) so manual fixes stay traceable.
- Add targeted unit tests in `tests/test_filter_balance.py` (Pytest) to lock in difficulty normalisation, duplicate detection, and strict-mode warnings.
- Run the filter over `data/parsed/seed_train.jsonl` (459 kept, 1 dropped) and the fresh `data/parsed/auto_drafts.jsonl`, saving markdown + JSON reports under `reports/day9_*.{md,json}` and cleaned copies in `data/filtered/`.
- Summarise `data/review/seed_notes.jsonl` in notes mode to highlight the 623 undecided rows so future reviewers can fill the gaps; defer triage of flagged items until the post-Day 14 human handoff.

- **Day 10 — Format, validate & stage (fully automated)**
- Build `scripts/format_for_sft.py` to convert validated MCQs into chat/completion format and enforce the JSON schema during export.
- Generate train/val/test splits (80/10/10) automatically, writing stats to `results/dataset_stats.json` and recording snapshot hashes in `docs/internal-notes.md`.
- Produce `data/formatted/train.jsonl`, `val.jsonl`, and `test.jsonl`, plus a machine-readable manifest that downstream tooling can consume when humans begin review.

**Day 11 — Fine-tune locally (LoRA) + structured logging**
- Use `scripts/train_lora.py` with configurable hyperparameters (LoRA rank 16–32, bf16/fp16, lr ≈2e-4, gradient checkpointing as needed).
- Stream training metrics to `logs/train/adapter_v1.jsonl` and store checkpoints in `models/adapters/adapter_v1/`, updating `config/adapters.yaml` automatically.
- Capture key performance traces (loss curves, throughput) so reviewers have immediate context once they log on after Day 14.
- ✅ First smoke test completed: ran `python scripts/train_lora.py --base-model Qwen/Qwen2.5-0.5B-Instruct --device-map cpu --max-steps 1 --no-eval` to prove end-to-end wiring. Saved adapter weights under `models/adapters/adapter_v1/`, metrics to `logs/train/adapter_v1.jsonl`, and catalogued the run in `config/adapters.yaml`. Full 7B training will resume once a CUDA build that recognises the RTX 5080 is installed.

**Day 12 — Automated evaluation & surfacing weak spots**
- Run `scripts/eval.py` on `test.jsonl` to compute JSON validity %, accuracy, distractor diversity, numeric tolerance, AO distribution alignment, and hint quality heuristics.
- Save metrics to `results/adapter_v1/metrics.json` and representative samples to `results/adapter_v1/samples.jsonl`; generate CSV/HTML dashboards that can be opened by reviewers later.
- Auto-tag items that fall below numeric or reasoning thresholds and queue them for further synthesis.
- ✅ `python scripts/eval.py --trust-remote-code --device cpu` completed (≈15 min on the 0.5B adapter). Metrics: JSON validity 83.7%, schema-valid 69.4%, numeric-clean 100%, but answer_match 0% and AO1/AO2 skew (AO2+AO3 missing). Samples logged to `results/adapter_v1/samples.jsonl` for Day 13 triage.

**Day 13 — Synthetic refresh & feedback tooling**
- Use the evaluation tags to auto-select candidate questions for regeneration with `scripts/auto_generate.py`, passing them through the sanitise/filter chain into `data/filtered/refresh_candidates.jsonl`.
- Stand up a lightweight feedback API/CLI (`scripts/feedback_queue.py`) that can surface any JSONL file (seed, auto, refresh) and record reviewer inputs when the human loop begins.
- Document the feedback workflow in `docs/feedback-playbook.md` so the human reviewers have a ready-made guide.
- ✅ `python scripts/prepare_refresh.py --include-answer-mismatch` flagged 46 weak outputs and wrote `data/parsed/refresh_sources.jsonl` plus a summary (`results/adapter_v1/refresh_summary.{json,md}`).
- ✅ `python scripts/auto_generate.py --input data/parsed/refresh_sources.jsonl --output data/parsed/refresh_drafts.jsonl --limit 60 --allow-existing` produced 40 clean drafts (one stubborn part still fails due to sci-notation placeholders) which were filtered into `data/filtered/refresh_candidates.jsonl` with reports under `reports/day13_refresh.{json,md}`.
- ✅ Added `scripts/feedback_queue.py` for listing/annotating MCQs and logged a sample decision in `data/review/day13_feedback_log.jsonl`; reviewer playbook lives in `docs/feedback-playbook.md`.

**Day 14 — Compare adapters & prep the review station**
- Run a second LoRA pass (`models/adapters/adapter_v2/`) to compare against adapter_v1, storing metrics in `results/adapter_comparison.json` and rendering charts in `results/adapter_comparison.md`.
- Pre-populate the feedback tooling with the latest metrics, sample outputs, and flagged items; generate a consolidated `handoff/day14_artifacts/` bundle containing datasets, adapters, evaluation dashboards, and a ready-to-use review queue.
- Confirm all automation scripts can run unattended (cron/batch) so the human team only needs to open the feedback tool, review MCQs, and submit decisions post-Day 14.
- Trained `adapter_v2` (`python scripts/train_lora.py ... --adapter-name adapter_v2 --device-map cpu`) and evaluated it with `scripts/eval.py` → outputs in `results/adapter_v2/`.
- Ran `python scripts/compare_adapters.py` to produce `results/adapter_comparison.{json,md}` (adapter_v2 nudges JSON-valid to 85.7% and schema-valid to 75.5%, but answer-match remains 0%).
- Assembled `handoff/day14_artifacts/` via `scripts/create_handoff_bundle.py`, bundling adapters, metrics, refresh candidates, feedback log, and the reviewer playbook with a manifest/README for the QA team.

**Post-Day 14 — Human QA & iterative improvement (starts after automation ends)**
- Use the bundled feedback tool to review flagged MCQs, confirm the correct option (override the model’s guess if needed), and capture a short note; every decision lands in `data/review/day13_feedback_log.jsonl`.
- `run_refresh_cycle.py` now reads that log, writes approved items + corrected answers to `data/filtered/refresh_approved.jsonl`, and parks all `needs-work` / `reject` / pending rows in `data/filtered/refresh_holdback.jsonl` for the next regeneration pass.
- Inspect evaluation dashboards, choose the preferred adapter checkpoint, and note follow-up training requests; no deployment or rollout tasks are required at this stage.
- Repeat the generate → filter → review → retrain loop as often as needed—the tooling handles retraining once you’re done reviewing.
- Shortcut commands:
  - `python scripts/feedback_queue.py review --input data/filtered/refresh_candidates.jsonl --log data/review/day13_feedback_log.jsonl`
  - `python scripts/run_refresh_cycle.py --adapter-name adapter_v3 --compare-with results/adapter_v2/metrics.json --bundle`

---

**Scripts you’ll create (simple, small)**
- `scripts/extract_pdf.py` — save per-page text and images
- `scripts/ocr_images.py` — run OCR on images and build simple JSON (labels/axes/units)
- `scripts/analyse_graphs.py` — detect chart-like diagrams and extract axis scales, sampled data points, and chart metadata
- `scripts/parse_questions.py` — split into questions and match mark schemes
- `scripts/make_seed_data.py` — hand-curate the first 100 great MCQs
- `scripts/auto_generate.py` — use a local model to draft more MCQs (with mark scheme context)
- `scripts/review_seed.py` — lightweight UI/CLI for human approval + notes
- `scripts/filter_balance.py` — validate JSON, units/sig-figs, dedup, and balance the set
- `scripts/format_for_sft.py` — turn into chat format for training
- `scripts/feedback_queue.py` — surface queued MCQs for human grading after Day 14
- `tests/test_filter_balance.py` — automated checks for schema, duplicates, AO tags, units
- `scripts/train_lora.py` — fine-tune with LoRA (logs metrics)
- `scripts/eval.py` — compute metrics and sample outputs
- `scripts/serve_model.py` — start a local HTTP API for your model
- `scripts/regression_suite.py` — run archived questions against adapters and compare outputs

Each script can be small and focused. We can scaffold these if you want.

---

Prompts (keep them consistent)

System (training and inference)  
\`\`\`  
You are an assistant that produces STRICT JSON for physics MCQs.  
Follow the exam board tone and conventions.  
Output exactly this JSON shape with no extra text: { mcqs: \[ ... \], solution: { ... } }  
No meta options (like “All of the above”). Use correct units and significant figures.  
\`\`\`

User (with image context when present)  
\`\`\`  
Question:  
\<question\_text\>

Ground truth (from mark scheme):  
\<answer\_text or key steps\>

If diagram:  
\<image\_context JSON: axes, labels, units, key points\>

Please produce one high‑quality MCQ with 4 options, the correct index, a short hint, and a clear explanation. Include an AO tag.  
\`\`\`

Assistant  
\`\`\`  
{ "mcqs": \[ ... \], "solution": { ... } }  
\`\`\`

---

**Tips to avoid hallucinations**
- Always include the mark-scheme snippet in prompts during data creation and at inference (ground truth!)
- Validate after generation: JSON schema, units/sig-figs, duplicates, meta options
- Keep temperature low (0.2–0.4) and cap max tokens
- For numeric answers, check against the mark-scheme value with a small tolerance

**Future syllabus backlog**
- Re-run extraction → OCR → graph analysis for: EDEXCEL, Eduqas, OCR_A, OCR_B, WJEC, CAIE (currently archived under `other_syllabuses/`).
- Once the AQA pipeline is validated end-to-end, restore each board to `data/`, refresh coverage tables, and append their diagrams to the benchmark before broader fine-tuning.

---

**What success looks like by Day 14**  
\- A working local model that generates board‑style MCQs in strict JSON  
\- Solid accuracy on a test set (and easy to measure)  
\- A clear path to keep improving: add weak topics weekly, re‑train short runs

