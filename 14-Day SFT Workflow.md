**Two-Week Training Plan (Physics MCQ Generator)**

This guide is written for non‑technical readers. Follow it day by day to build a local, low‑cost AI that turns physics exam questions into high‑quality MCQs with hints and explanations.

We’ll use your own papers \+ mark schemes, run everything on your PC, and avoid paid APIs.

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

**Install these once (Day 0\)**

**Programs**  
\- Python 3.10 or newer  
\- VS Code (code editor)  
\- Git (version control)  
\- Ollama (simple local model runner) or vLLM (faster, optional)

Windows setup (PowerShell)

**Python packages (install once)**  
pip install \--upgrade pip  
pip install transformers datasets peft bitsandbytes accelerate sentence-transformers scikit-learn pdfplumber pymupdf pytesseract pillow camelot-py\[cv\] pandas tqdm fastapi uvicorn jsonschema

**Optional math OCR (skip if not needed)**  
pip install nougat-ocr  
\`\`\`  
Also install:  
\- Poppler for Windows (needed by some PDF tools): https://github.com/oschwartz10612/poppler-windows/releases/  
\- Tesseract OCR (Windows installer): https://github.com/UB-Mannheim/tesseract/wiki

**Folder layout (we’ll create as we go)**
```
project/
  data/
    pdfs/                # your papers + mark schemes
    raw/                 # extracted text
    images/              # extracted images
    parsed/              # per-question JSONL + QA metadata
    captions/            # image captions (text)
    formatted/           # training-ready JSONL
    index/               # retrieval index (FAISS/sqlite)
    review/              # curator notes, approvals
  models/
    adapters/            # LoRA adapters from training
  scripts/               # small helper scripts
  results/               # evaluation reports
  config/
    adapters.yaml        # adapter status tracking
  logs/                  # training, evaluation, extraction logs
```

**Current dataset scope (Oct 2025 update)**
- Active board: AQA (all raw text, images, OCR captions, graph analysis, and descriptions live under `data/`)
- Archived boards: EDEXCEL, Eduqas, OCR_A, OCR_B, WJEC, CAIE (moved to `other_syllabuses/` at repo root until we resume multi-board runs)
- When you resume a board, move its tree back under `data/` before rerunning extraction/OCR scripts and refresh the coverage tables.

---

NOTE: MARKDOWN PROGRESS + WHAT'S IMPLEMENENTED + DELIVERABLES WHEN YOU FINISH A DAY

**Day 1 — Collect, catalogue & secure data**
- Put all PDFs into `data/pdfs/` (papers and their matching mark schemes)
- Keep filenames consistent (e.g., `AQA_2019_P1.pdf`, `AQA_2019_P1_MS.pdf`)
- Create `data/README.md` with coverage table (board, year, topic, marks) to track gaps
- If material is sensitive, store `data/` on encrypted storage (BitLocker) and document handling rules
- Initialise Git LFS and track `data/` + `logs/` so large binaries stay out of normal git history; record commands in `INTERNAL-NOTES.md`
- Goal: clean, organized, auditable source files

**Day 2 — Extract text + images (batch + log)**
- Create `scripts/extract_pdf.py` to iterate over `data/pdfs/`, pull text per page, export images, and emit structured logs/errors.
- Remove stale `data/raw/` + `data/images/` before a rerun so new outputs reflect the latest heuristics (high-DPI vector clips, raster filters).
- Run `scripts/extract_pdf.py` (tune with flags like `--render-dpi`, `--min-raster-contrast`, `--drawing-cluster-gap`) to extract per-page text and targeted image crops; store logs in `logs/extract/`.
- New dependency: install OpenCV (`pip install opencv-python`) so the extractor can rescale rasters and filter logo artifacts.
- Output:
  - `data/raw/<paper>_page<N>.txt`
  - `data/images/<paper>_page<N>_img<M>.png` and `..._render.png` for graphs or page snapshots (vector clips optional via `--enable-vector-clips`)
- Review `logs/extract/run_*.jsonl` for any `status: "error"` PDFs, adjust parameters if needed, and note issues/fixes in `data/README.md`
- For the current AQA-only phase, confirm `other_syllabuses/` still holds the archived boards so fresh runs don’t mix scopes; update coverage notes to reflect "AQA complete, others pending".
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
- Build a lightweight coverage script/notebook to compare automated summaries vs. benchmark ground truth and flag low-confidence cases.

**Day 5 — Graph understanding pre-training**
- Create `scripts/train_graph_model.py` to fine-tune a lightweight vision-language model (or the base LLM) using benchmark annotations + auto metadata so it can answer graph-centric questions.
- Optional: integrate ChartOCR/UniChart outputs as additional supervised signals.
- Build `scripts/eval_graph_model.py` to score the model on the benchmark set.
- **Human**: review sample predictions—especially challenging schematics—to confirm the model now reads them correctly; only move forward once results are acceptable.
- Snapshot weights/config (`models/adapters/graph_reader_v1/`) and document in `config/adapters.yaml`.

**Day 6 — Parse questions, align mark schemes & seed retrieval**
- Create `scripts/parse_questions.py` to combine raw text, captions, graph metadata/summary, and mark schemes into structured question objects with metadata links.
- Run `scripts/parse_questions.py` to detect questions, sections, marks.
- Match each to mark scheme answers; log ambiguities for manual review.
- Link images by page number and keywords (“Figure 1”); attach graph JSON (axis info, sampled data, chart type) when available so downstream MCQs can reason about trends and numeric relationships.
- Build lightweight retrieval index under `data/index/` (FAISS/sqlite) and document parameters in `logs/index/`.
- Output: `data/parsed/questions.jsonl` with fields including metadata `{status:'pending', reviewer?:string}`.
- **Human**: spot-check parsed questions and linked graph summaries; correct any mismatches before continuing.

**Day 7 — Create first training examples + review lane**
- Create `scripts/make_seed_data.py` to help curate MCQ JSON by merging parsed questions with mark-scheme context and enforcing the schema.
- Curate ~120 balanced questions into clean MCQ JSON (options, correct_index, hint, explanation, AO).
- Use mark scheme, image_context, and retrieved snippets.
- Create `scripts/review_seed.py` to provide a lightweight approval UI/CLI that writes reviewer decisions and notes to disk.
- Build lightweight review UI (`scripts/review_seed.py`, Streamlit/Gradio/CLI) for approvals and notes saved in `data/review/seed_notes.jsonl`.
- Save approved items to `data/parsed/seed_train.jsonl` and update coverage tracker.
- **Human**: review every seed MCQ; confirm diagram references align with graph summaries before approving.

**Day 8 — Expand with local model help + human triage**
- Create `scripts/auto_generate.py` to call the local model with mark-scheme snippets and produce draft MCQs tagged with provenance.
- Use local model (Ollama Qwen2.5-7B-Instruct or Llama-3.1-8B) via `scripts/auto_generate.py`.
- Prompt with mark scheme snippets, image_context, retrieval results.
- Route outputs through the same review UI for accept/reject tagging (store rejected reasons).
- Save accepted items to `data/parsed/auto_train_candidates.jsonl` with status metadata.
- **Human**: triage auto-generated MCQs, reject hallucinations, and add reviewer notes for problematic diagrams/topics.

**Day 9 — Filter, unit-test & balance**
- Create `scripts/filter_balance.py` with CLI flags to validate JSON records, enforce AO/topic balance, and deduplicate by embeddings.
- Implement `scripts/filter_balance.py` checks:
  - Valid JSON, 4 unique options, correct_index 0–3.
  - No meta options, length limits, unit/significant-figure consistency.
- Create `tests/test_filters.py` to cover schema violations, duplicate detection, AO gaps, and sig-fig/unit edge cases.
- Add Pytest suite `tests/test_filters.py` covering duplicates, AO gaps, sig-fig errors; run daily.
- Deduplicate via sentence-transformer cosine similarity; enforce AO/topic balance using coverage tracker.
- Output: `data/parsed/train_clean.jsonl` (~2–4k items).
- **Human**: review filter warnings (e.g., near-duplicate questions, unit mismatches) and decide whether to keep/fix/drop each case.

**Day 10 — Format, validate & snapshot**
- Create `scripts/format_for_sft.py` to convert validated MCQs into chat-completion format and run JSON schema checks during export.
- Use `scripts/format_for_sft.py` to create chat format (system/user/assistant).
- Run jsonschema validation on formatted files; fail fast on malformed records.
- Split into train/val/test (80/10/10) and record dataset stats in `results/dataset_stats.json`.
- Snapshot dataset (`git tag dataset-v1` or record hash in `data/README.md`).
- Output: `data/formatted/train.jsonl`, `val.jsonl`, `test.jsonl`.
- **Human**: sanity-check a handful of formatted samples to ensure instructions/images/graph summaries still align after formatting.

**Day 11 — Fine-tune locally (LoRA) + log metrics**
- Create `scripts/train_lora.py` to launch LoRA fine-tuning with configurable hyperparameters and structured logging.
- Train with `scripts/train_lora.py` (LoRA rank 16–32, bf16/fp16, lr ~2e-4, gradient checkpointing if needed).
- Log training metrics to `logs/train/adapter_v1.jsonl`.
- Save adapter to `models/adapters/adapter_v1/` and update `config/adapters.yaml` (`adapter_v1: status=staging`).

**Day 12 — Evaluate, benchmark & log**
- Create `scripts/eval.py` to load adapters, run validation splits, compute core metrics, and emit sample outputs for review.
- Run `scripts/eval.py` on `test.jsonl`.
- Capture metrics: JSON validity %, answer accuracy, distractor diversity, numeric tolerance, AO distribution, hint quality flags.
- Save to `results/adapter_v1/metrics.json` and `results/adapter_v1/samples.jsonl`.
- **Human**: review 30–50 outputs (especially diagram-heavy questions) and log notes in `data/review/eval_notes.md`; update coverage tracker with weak topics.

**Day 13 — Target weak spots & refresh dataset**
- Add ~100 curated examples covering weak boards/topics/AO gaps.
- Run full review + filter pipeline (Days 7–10).
- Snapshot dataset (`dataset-v1.1`) and document changes in `data/README.md`.
- Keep original seed set as high-quality reference.
- **Human**: choose which weak areas to target, curate/approve new examples, and document rationale.

**Day 14 — Second fine-tune + adapter management**
- Update `scripts/train_lora.py` (if needed) or add config hooks to support multiple adapters cleanly.
- Train new adapter `models/adapters/adapter_v2/`.
- Update `config/adapters.yaml` with metrics (statuses: staging/prod/archive).
- Evaluate both adapters; compare results in `results/adapter_comparison.md`.
- Promote higher-performing adapter to `status=prod`.

**Day 15 — Serve, integrate & regression test**
- Create `scripts/serve_model.py` to expose the trained adapter over HTTP with health checks and latency logging.
- Serve adapter locally with `scripts/serve_model.py`; add health check and latency logging.
- Create `scripts/regression_suite.py` to replay archived questions against both Azure and local paths and compare structured outputs.
- Update `netlify/functions/decode.ts` to call the local endpoint while retaining validators.
- Run regression suite (`scripts/regression_suite.py`) over 20–30 archived questions; compare against Azure outputs and log diffs in `results/integration_tests.md`.
- Maintain temporary Azure fallback until local path proves stable.
- **Human**: inspect regression diffs, verify deployment checklist, and green-light rollout.

**Day 16 — Trial run, sign-off & roadmap**
- Run 50–100 real questions end-to-end; log telemetry (latency, validity, user issues) in `results/post_launch_log.jsonl`.
- Update `config/adapters.yaml` to mark shipped adapter as `prod` and archive older versions.
- Review failure log, create action items for next sprint (data gaps, prompt tweaks).
- Decide rollout: keep local model primary, Azure fallback until confidence >95%; schedule fallback retirement once metrics stay stable.
- **Human**: coordinate dry-run testing, review telemetry, and sign off on the production plan.

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
- `tests/test_filters.py` — automated checks for schema, options, AO tags, units
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

