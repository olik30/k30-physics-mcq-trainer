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

## 2. Bring an item into view

```bash
python scripts/feedback_queue.py list \
  --input data/filtered/refresh_candidates.jsonl \
  --log data/review/day13_feedback_log.jsonl \
  --limit 5
```

- The CLI prints the first five unreviewed items (skipping anything already logged).
- Use `--show-reviewed` if you ever need to revisit decisions.
- Copy the `ID:` value; it’s the key you’ll use when recording feedback (e.g. `AQA/AQA_Paper_1/June_2017_QP/Q02#02.1`).

## 3. Record a decision

```bash
python scripts/feedback_queue.py annotate \
  --input data/filtered/refresh_candidates.jsonl \
  --log data/review/day13_feedback_log.jsonl \
  --record-id AQA/AQA_Paper_1/June_2017_QP/Q02#02.1 \
  --decision needs-work \
  --note "Correct option repeats prompt; regenerate explanation."
```

- Allowed decisions: `approve`, `reject`, `needs-work`.
- Notes are optional but strongly encouraged so the automation can respond (e.g. regenerate vs. keep).
- All entries append to `data/review/day13_feedback_log.jsonl`; no existing rows are overwritten.

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
  1. `needs-work` / `reject` items get regenerated or hand-edited.
  2. Newly-approved items roll into the training dataset.
  3. The evaluation suite (`scripts/eval.py`) is re-run to track progress.
- No manual editing of JSON files is required; stick to the CLI log unless you’re doing deep fixes.

---

**Need help?**  
Open an issue referencing the record ID, paste the CLI output, and describe the concern (physics correctness, formatting, etc.). The engineering team will adjust prompts, filters, or regeneration strategies accordingly. Thanks! 

