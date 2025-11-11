# Feedback Playbook

This guide explains how to review generated MCQs once the fully automated Days 1–14 pipeline has finished running. The human review loop does **not** start until after Day 14, but the tooling is ready now so you can hit the ground running.

---

## 1. What you will review

- Focus on the JSON files produced by the automated filter:
  - `data/filtered/refresh_candidates.jsonl` – the latest machine-generated drafts that passed schema checks.
  - `data/filtered/seed_train.filtered.jsonl` – the curated “good” seed set (useful for comparison or spot checks).
- Each line is a single MCQ with the fields:
  - `question`, `options[4]`, `correct_index`, `hint`, `explanation`
  - `ao`, `topic`, `difficulty`
  - `sources` → links back to the original past-paper part for context.

## 2. Variant review (format & wording)

```bash
python scripts/feedback_queue.py variants \
  --input data/filtered/refresh_candidates.jsonl \
  --log data/review/variant_choices.jsonl
```
or launch the Streamlit UI (after `pip install streamlit`):
```bash
streamlit run ui/variant_review.py
```

- Each screen shows five variants for the same question part.
- First pick the preferred variant (`1-5`, or `n` if none are standout).
- For every variant, decide `a` (accept) or `r` (reject) and add a short note:
  - Accepted-but-not-preferred notes explain why it is still good and why another variant won.
  - Rejected notes call out the flaw (“hint restates question”, “explanation missing units”, etc.).
- Decisions append to `data/review/variant_choices.jsonl`; accepted variants flow into `data/filtered/refresh_accept.jsonl`, rejected/pending ones into `data/filtered/refresh_reject.jsonl`.

## 3. Answer correctness pass (optional once wording is solid)

```bash
python scripts/feedback_queue.py review \
  --input data/filtered/refresh_candidates.jsonl \
  --log data/review/day13_feedback_log.jsonl
```

- Items are shown one-by-one. For each MCQ, type:
  - `a` → approve
  - `n` → needs-work
  - `r` → reject
  - `s` → skip this item
  - `q` → quit the loop
- After choosing `a/n/r`, confirm the correct answer letter. Press **Enter** to accept the model’s choice or type the real letter (A–D) to override it.
- Jot a concise note (why the answer is right/wrong). The CLI prints the original prompt, mark-scheme snippet, and source PDF paths for quick cross-checks.
- Notes land in `data/review/day13_feedback_log.jsonl`; use `--show-reviewed` to revisit previous entries.

## 4. What “good” looks like

- **Accuracy** – the correct option must align with the mark scheme value or reasoning.
- **Options** – four distinct, plausible distractors; avoid “All of the above”.
- **Hint/Explanation** – short hint points a student in the right direction; explanation walks through the reasoning.
- **Tags** – AO/topic/difficulty should match the intent of the original part.
- **Style** – exam tone, no markdown, ASCII only, units with sensible sig figs.

## 5. Triage tips

- If an item is completely unusable (e.g. wrong physics, repeated options), mark `reject`.
- If the structure is fine but needs massaging (e.g. better hint wording), pick `needs-work`.
- Only choose `approve` when you’d be happy to push the MCQ to students as-is.

## 6. After feedback

- Variant decisions create:
  1. `data/filtered/refresh_accept.jsonl` – all accepted variants (preferred ones flagged for higher training weight).
  2. `data/filtered/refresh_reject.jsonl` – rejected or pending variants kept for regeneration.
- Answer-phase decisions (if run) update `data/filtered/refresh_approved.jsonl` / `data/filtered/refresh_holdback.jsonl` and correct the answer indices inside each record.
- Every log entry includes the original `question_id` and `part_code`, keeping the audit trail intact for future regeneration.
- No manual editing of JSON files is required; stick to the CLI logs unless you’re doing deep fixes.

## 7. One-command refresh cycle

Once you’ve logged enough feedback, run the end-to-end automation:

```bash
python scripts/run_refresh_cycle.py \
  --adapter-name adapter_v2 \
  --compare-with artifacts/results/adapter_v1/metrics.json \
  --variants-per-source 5 \
  --bundle
```

This executes: `prepare_refresh` → `auto_generate` (with variants) → `filter_balance` → **apply reviewer decisions** → (optional) `format_for_sft` → `train_lora` → `eval` → `compare_adapters` → `create_handoff_bundle`.  
Only the variants you accepted feed training; everything else stays in `refresh_reject.jsonl` for another regeneration pass. Adjust flags if you want a longer training run (increase `--max-steps`, change `--device-map`, etc.).

---

**Need help?**  
Open an issue referencing the record ID, paste the CLI output, and describe the concern (physics correctness, formatting, etc.). The engineering team will adjust prompts, filters, or regeneration strategies accordingly. Thanks! 

