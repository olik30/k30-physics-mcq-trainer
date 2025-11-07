# Review → Retrain Cycle (Quick Guide)

This page explains, in plain language, how to review MCQ drafts, pick the best ones, and retrain the model. Follow the steps in order.

---

## 1. One-Time Setup

Install the tools we use:

```powershell
pip install streamlit transformers datasets accelerate
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

(The second command installs the CPU build of PyTorch so the training script can run anywhere.)

---

## 2. Generate Fresh Drafts (5 variants per question)

Click **“Generate new variants & retrain”** in the Streamlit sidebar, or run the command below if you prefer the terminal:

```powershell
python scripts/run_refresh_cycle.py --adapter-name adapter_v3 --compare-with results/adapter_v2/metrics.json --variants-per-source 5 --bundle
```

This creates:

- `data/filtered/refresh_candidates.jsonl` – the 5 variants for each question.
- `data/parsed/refresh_sources.jsonl` – the source question parts that were regenerated.
- `reports/adapter_v3_refresh.*` – quick quality summaries.

Run this command whenever you want a new batch to review. (It also performs a short training pass; increase `--max-steps` later for longer runs.)

---

## 3. Review the Variants

### Option A – Command line

```powershell
python scripts/feedback_queue.py variants \
  --input data/filtered/refresh_candidates.jsonl \
  --log data/review/variant_choices.jsonl
```

You’ll see each question with five columns. For every group:

1. Pick the **preferred** variant (`1-5`, or `n` if none stand out).
2. For each variant, type `a` (accept) or `r` (reject) and add a short note.

### Option B – Streamlit web app

```powershell
streamlit run ui/variant_review.py
```

This opens a simple browser page with buttons instead of typing. The app writes the same log file (`data/review/variant_choices.jsonl`).

Your notes should call out what works (or doesn’t): e.g. “Accept – clear hint, numerical check matches mark scheme” or “Reject – explanation repeats question.”

The log file records:

- `refresh_accept.jsonl` – all variants marked “accept” (preferred ones are flagged).
- `refresh_reject.jsonl` – variants marked “reject” or left undecided.

---

## 4. Retrain With Your Choices

After you’ve reviewed enough questions, rerun the cycle (same command as step 2). The script will:

1. Pull in your decisions from `data/review/variant_choices.jsonl`.
2. Rebuild the training set using the accepted variants.
3. Fine-tune the adapter and produce new evaluation metrics in `results/<adapter_name>/`.

Repeat steps 2–4 as many times as needed until all five variants for each question are acceptable.

---

## 5. (Optional) Answer-Only Review Stage

Once the wording/hints look good, switch to the original answer checker to fix incorrect answers:

```powershell
python scripts/feedback_queue.py review \
  --input data/filtered/refresh_candidates.jsonl \
  --log data/review/day13_feedback_log.jsonl
```

This prompt lets you mark each MCQ `approve` / `needs-work` / `reject`, confirm the correct option letter, and leave a short rationale.

---

### Where Files Live

- `data/filtered/refresh_candidates.jsonl` – drafts waiting for review.
- `data/review/variant_choices.jsonl` – your variant decisions.
- `data/filtered/refresh_accept.jsonl` – variants that will train the next adapter.
- `data/filtered/refresh_reject.jsonl` – variants to regenerate later.
- `models/adapters/<adapter_name>/` – saved LoRA adapters.
- `results/<adapter_name>/` – evaluation reports (metrics + sample outputs).

Keep this cheat sheet handy and repeat the loop: **Generate → Review variants → Retrain → (Optional) Answer review**.

