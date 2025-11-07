# Review → Retrain Loop Overview

## Current Adapter Quality
- Metrics from `results/adapter_v2/metrics.json`:
  - `json_valid_pct`: **85.7 %** — adapter usually emits parseable JSON.
  - `schema_valid_pct`: **75.5 %** — three quarters satisfy all required fields/length checks.
  - `answer_match_pct`: **0.0 %** — it still picks the wrong option every time, so wording/structure is decent but physics correctness is missing.

## Human Review Steps (Variant Phase)
1. Queue five variants per question and pick the best ones:
   ```powershell
   python scripts/feedback_queue.py variants \
     --input data/filtered/refresh_candidates.jsonl \
     --log data/review/variant_choices.jsonl
   ```
   _Prefer a GUI?_ Launch the Streamlit app:
   ```powershell
   streamlit run ui/variant_review.py
   ```
   (Install once with `pip install streamlit`.)
   - First choose the preferred variant (`1-5`, or `n` if none stand out).
   - For every variant, answer `a` (accept) or `r` (reject) and leave a short note:
     - Accepted-but-not-preferred notes should mention why it’s still good and why another variant wins.
     - Rejected notes call out the flaw (e.g. “hint restates the question”, “explanation missing units”).

2. Retrain with the accepted variants:
   ```powershell
   python scripts/run_refresh_cycle.py \
     --adapter-name adapter_v3 \
     --compare-with results/adapter_v2/metrics.json \
     --variants-per-source 5 \
     --bundle
   ```
   - Accepted variants flow into `data/filtered/refresh_accept.jsonl` with preferred ones flagged and weighted heavier.
   - Rejected/pending variants are archived in `data/filtered/refresh_reject.jsonl` for regeneration.
   - Once wording quality is solid, switch back to `feedback_queue.py review …` to focus purely on answer correctness.

## What Happens During the Cycle
- `prepare_refresh.py` selects weak samples (JSON/schema failures, answer mismatches) from the last evaluation and writes question parts to `data/parsed/refresh_sources.jsonl`.
- `auto_generate.py` (with `--variants-per-source`) and `filter_balance.py` regenerate drafts and output `data/filtered/refresh_candidates.jsonl` containing all variants.
- `feedback_queue.py variants` records decisions in `data/review/variant_choices.jsonl` and splits them into:
  - `data/filtered/refresh_accept.jsonl` (accepted variants; preferred ones flagged via `metadata.review`).
  - `data/filtered/refresh_reject.jsonl` (rejected or pending variants held for another regeneration pass).
- `format_for_sft.py` merges `refresh_accept.jsonl` with the seed dataset and emits chat-tuned splits under `data/formatted/` + hashes in `data/formatted/manifest.json`.
- `train_lora.py` writes the new adapter into `models/adapters/<adapter_name>/` (checkpoints, tokenizer, logs).
- `eval.py` stores metrics/samples in `results/<adapter_name>/`.
- Optional `compare_adapters.py` creates deltas in `results/adapter_comparison.{json,md}`.
- If `--bundle` is set, `handoff/<day14_artifacts|...>/` is refreshed with everything reviewers need.

## Output Locations
- **MCQs awaiting review:** `data/filtered/refresh_candidates.jsonl`
- **Accepted variants feeding the next train:** `data/filtered/refresh_accept.jsonl`
- **Rejected / pending variants:** `data/filtered/refresh_reject.jsonl`
- **Variant review log:** `data/review/variant_choices.jsonl`
- **(Answer QA phase) Per-MCQ review log:** `data/review/day13_feedback_log.jsonl`
- **Adapter checkpoints & training logs:** `models/adapters/<adapter_name>/`
- **Evaluation reports:** `results/<adapter_name>/`

## Comment Guidance
- **Variant phase:** focus on wording quality. Explain why a variant is accepted/rejected and, if it isn’t the preferred one, why another variant reads better.
- **Answer QA phase:** keep notes concise (1–2 sentences) with the correct option letter and a short justification tied to the mark scheme.

