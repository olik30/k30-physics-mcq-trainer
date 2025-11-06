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

## 2. Review & log decisions (interactive)

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
- After choosing `a/n/r`, the CLI asks you to confirm the correct option. Press **Enter** to accept the model’s choice or type the real answer letter (A–D) to correct it.
- Then add a short reason/note (optional). Everything is logged to `data/review/day13_feedback_log.jsonl`, including the answer you supplied.
- The CLI now prints the original prompt, the mark-scheme snippet, and direct paths to the source PDFs/images so you can double-check the official answer without leaving the terminal.
- Use `--show-reviewed` if you want to revisit items that already have decisions.

> Still want the old behaviour? `list` + `annotate` remain available for ad-hoc use.

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

- Decisions feed the next automated loop:
  1. `needs-work` / `reject` entries are held in `data/filtered/refresh_holdback.jsonl` for regeneration.
  2. Approved items (with your confirmed answer index) are written to `data/filtered/refresh_approved.jsonl` and merged into the next training set.
  3. The evaluation suite (`scripts/eval.py`) is re-run to track progress.
- Each log entry also records the original `question_id` and `part_code`, keeping the audit trail intact for future regeneration.
- No manual editing of JSON files is required; stick to the CLI log unless you’re doing deep fixes.

## 7. One-command refresh cycle

Once you’ve logged enough feedback, run the end-to-end automation:

```bash
python scripts/run_refresh_cycle.py \
  --adapter-name adapter_v3 \
  --compare-with results/adapter_v2/metrics.json \
  --bundle
```

This executes: `prepare_refresh` → `auto_generate` → `filter_balance` → **apply reviewer decisions** → (optional) `format_for_sft` → `train_lora` → `eval` → `compare_adapters` → `create_handoff_bundle`.  
Only the records you approved (with corrected answers) feed training; everything else stays in the holdback file for another regeneration pass. Adjust flags if you want a longer training run (increase `--max-steps`, change `--device-map`, etc.).

---

**Need help?**  
Open an issue referencing the record ID, paste the CLI output, and describe the concern (physics correctness, formatting, etc.). The engineering team will adjust prompts, filters, or regeneration strategies accordingly. Thanks! 

