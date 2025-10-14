# K30 Creator Codebase Recap

This document summarizes the current K30 Creator repository so a new AI (or developer) can understand what exists, how the pieces interact, and which constraints any future training/integration work must respect.

## Product Snapshot
- **Purpose**: Help A-level students turn exam questions (typed text, images, PDFs, DOCX) into guided multiple-choice sequences with hints and a final worked solution.
- **Current decoder**: Azure OpenAI (o4-mini/gpt-4o family) accessed via Netlify functions. Despite aggressive guardrails, accuracy still drifts and per-request cost remains high.
- **Planned direction**: Replace Azure dependency with a locally fine-tuned model (see `14-Day SFT Workflow.md`) that produces board-specific MCQs in strict JSON while reusing existing validation scaffolding.

## High-Level Flow
1. **User input** (text/photo/file) enters the React SPA (`src/App.tsx` → routed views).
2. **If file/photo**: client sends base64 to `/api/extract` (Netlify function) → Azure Document Intelligence returns raw text.
3. **Client cleanup**: verification step normalizes fractions, trims boilerplate.
4. **Optional augment**: `/api/augment` (Azure OpenAI vision) merges OCR text + diagrams into a clean statement.
5. **Decode**: `/api/ai-decode` (`netlify/functions/decode.ts`) sends text (+ up to 2 images) to Azure OpenAI, enforces schema, performs retries, synthesizes solution metadata, and returns `{ mcqs: [...], solution: {...}, usage, meta }`.
6. **Client rendering**: components such as `QuestionDecoder.tsx`, `MCQInterface.tsx`, `SolutionSummary.tsx`, and dashboard/history pages display the content, track streaks, and manage user review cycles.

## Frontend Structure (Vite + React + TypeScript)
- **Entry**: `src/main.tsx` mounts `App.tsx` and global styles (`src/styles/globals.css`, `src/index.css`).
- **UI primitives**: `src/components/ui/` holds Shadcn-inspired components (buttons, cards, dialog, tabs, etc.) reused across pages. These are pure client-side; no custom logic relevant to training.
- **Feature components**:
  - `QuestionInput.tsx`: collects user prompts, manages file uploads, chooses marks (1–8), subject, syllabus, etc.
  - `QuestionDecoder.tsx`: orchestrates the decoding spinner, handles image compression/PDF rendering, and calls the Netlify decode function with fallback logic. Performs client-side hint strengthening and deduplication.
  - `MCQInterface.tsx`: displays the generated MCQs sequentially, capturing user answers and progress.
  - `SolutionSummary.tsx`: presents final answer, working steps, key formulas, key points, pitfalls, and applications returned by the decoder.
  - `QuestionHistory.tsx`, `Dashboard.tsx`, `RecentPerformanceCard.tsx`, `RadarChart.tsx`: provide analytics/history views over Supabase data.
  - `ReviewPage.tsx`, `MilestonesPage.tsx`, `StreaksPage.tsx`, etc. round out the learner experience.
- **State/data**: Authentication and persistence rely on Supabase (`src/lib/supabase.ts`). Most question/MCQ data lives in local component state backed by API responses.

## Constants & Domain Knowledge
- `netlify/constants/boardProfiles.ts`: defines exam-board profiles (`BoardProfile`) with command words, conventions (e.g., g = 9.81 m/s² vs 9.8), distractor styles, and AO distributions. The decoder consults this file to shape MCQ tone per syllabus/level.
- `src/constants/catalog.ts`: enumerates themes, achievements, and UI catalog metadata.
- `src/guidelines/Guidelines.md`: developer-facing guidelines (UI, accessibility, etc.).
- `docs/AI-Pipeline.md`: narrative documentation of current extract → augment → decode pipeline, aligned with Azure dependencies.

## Netlify Functions (Azure Path)
All live under `netlify/functions/` and export default handlers with `config = { path: ... }`.

### `decode.ts` (aliased as `/api/ai-decode`)
- **Inputs**: `{ text?, images?, marks?, subject?, syllabus?, level?, specCode?, maxTokens? }`.
- **Steps**:
  1. Normalizes notation, blocks multi-question payloads, and resolves board profile.
  2. Stage 1 (parse): prompts model to extract quantities/relations/plan JSON (length = marks).
  3. Stage 2 (generate): instructs model to produce exactly `marks` MCQs + solution skeleton, optionally with retrieval snippets (Azure Cognitive Search). Handles token budgeting, attachments for up to 2 images, retries without images, fallback deployments.
  4. Stage 3 (synthesize): calls model again to produce working steps, key points, applications.
  5. Stage 4 (pitfalls): separate prompt for pitfalls list; fallback to static map if empty.
  6. Stage 5 (final answer): ensures finalAnswer/unit present.
  7. Post-processing: validates MCQs (no meta options, 4 choices), reindexes correct answers, improves hints, deduplicates, enforces step numbering, optionally runs judge pass, compiles usage stats.
- **Outputs**: `mcqs`, `solution`, `usage` (token totals), `meta` (retrieval metadata, prompt estimates, judge results).

### Other functions
- `ai-decode.ts`: re-exports `decode.ts` for backward compatibility (`/.netlify/functions/ai-decode`).
- `decode-hints.ts` + `ai-refine-hints.ts`: rewrites hints for existing MCQs; mode-detection heuristics choose hint templates.
- (Documentation mentions `/api/extract` & `/api/augment`; their implementations may exist in another branch/deployment but conceptually follow the pipeline described in `docs/AI-Pipeline.md`.)

## Data Contracts
- **MCQ object** (frontend & backend): `{ id, question, options[4], correctAnswer (0-based), hint, explanation, step, ao?, calculationStep? }`.
- **SolutionSummary**: `{ finalAnswer, unit, workingSteps[], keyFormulas[], keyPoints?, applications?, pitfalls? }`.
- Client-side fallback generator (in `QuestionDecoder.tsx`) mirrors these shapes for deterministic defaults if the API fails.

## Known Pain Points (Root Cause for New Training)
- Azure decoder still fabricates or mis-orders steps, even with staged prompts and salvage logic.
- Hints can be generic; client adds heuristics to patch them.
- Strict JSON enforcement incurs multiple round trips (expensive latency/cost).
- Dependency on Azure cognitive stack conflicts with goal of running locally and reducing per-request expense.

## Planned Solution (Hand-off Context)
- `14-Day SFT Workflow.md` (and `docs/Two-Week-Training-Plan.md`) detail a two-week schedule to build a local supervised fine-tuning pipeline: collect + OCR + parse dataset, curate MCQs grounded on mark schemes, apply automated validation, fine-tune via LoRA, evaluate, and integrate a locally served model.
- The new model must:
  - Produce the existing response shape exactly (`{ mcqs: [...], solution: {...} }`).
  - Respect board profiles (command words, AO distribution, sig figs, g-value, distractor style).
  - Honor constraints already enforced in `decode.ts` (no meta options, one question per request, numeric tolerance, retrieval integration optional).
  - Support images via textual captions/contexts when available.

## Key Files for Reference
- `README.md`: public product description, feature list, licensing.
- `14-Day SFT Workflow.md`: detailed training plan (with rationale, assumptions, day-by-day tasks, scripts to build, QA steps, integration guidance).
- `docs/Two-Week-Training-Plan.md`: mirrored plan for documentation.
- `docs/AI-Pipeline.md`: existing Azure pipeline explanation.
- `netlify/functions/decode.ts`: current decoder implementation to mirror with local model.
- `src/components/QuestionDecoder.tsx`: client decode orchestration, hint improvements, fallback logic.
- `netlify/constants/boardProfiles.ts`: exam board conventions.

## Expectations for the New AI
When you (the incoming AI) build or fine-tune the local model:
1. Align outputs with the contracts above; the frontend expects 4 options, AO tags, and step-aligned MCQs equal to the mark count.
2. Rely on mark-scheme text for grounding—hallucinations are unacceptable. LoRA training data will include ground truth fields.
3. Maintain hint/explanation quality so the client’s hint improver becomes a safety net rather than a crutch.
4. Support plug-in retrieval (local FAISS/sqlite index) so contextual snippets can feed prompts.
5. Provide token usage/metrics if possible; the UI logs them today.
6. Keep latency low enough for an interactive experience (<5 seconds per decode preferred).

This recap, combined with the workflow plan, should give full context for reproducing and improving the decoder in a fresh repository without direct access to the original source. Good luck, and keep the guardrails tight!
