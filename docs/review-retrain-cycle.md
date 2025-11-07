# Review → Retrain Cycle (UI version)

Everything here can be done inside the **MCQ Variant Reviewer** Streamlit page—no terminals or commands required.

---

## Step 0 — Open the reviewer
- Launch the MCQ Variant Reviewer page (your team already has a desktop shortcut or browser bookmark).
- If the page says “No variant groups left”, click the sidebar button **Generate new variants & retrain** once to create a fresh batch.

---

## Step 1 — Review the five variants
For each question row you see:
1. Read the five variants shown side-by-side.
2. Choose a **Preferred variant** (`1–5`) or leave “None” if nothing stands out yet.
3. For every variant, pick **Accept** or **Reject** and leave a short note (e.g. “Accept – hint mentions g correctly”, “Reject – explanation copies the question”).
4. Click **Save decisions**.
5. Move between questions with **Previous**, **Skip group**, or **Next (no save)**.

Your notes are saved automatically to the project log so the training script knows which variants to keep.

---

## Step 2 — Retrain from the sidebar
Once you have a batch of decisions:
1. Open the sidebar (tap the “>” chevron if it’s hidden).
2. Check the settings:
   - **Adapter name** — pick the label for the new training run (e.g. `adapter_v3`).
   - **Compare-with metrics** — usually the previous adapter’s metrics file.
   - **Variants per question** — default is 5; lower it if you want fewer drafts.
   - **Bundle artefacts** — keep on if you want the hand-off bundle refreshed.
3. Press **Generate new variants & retrain**.
4. The centre panel streams live progress; when it finishes successfully the page refreshes with the newly generated variants ready for another pass.

Each retrain reuses your accepted variants and regenerates new drafts for anything you rejected or skipped.

---

## Step 3 — Repeat until happy
Keep looping on the same page:
**Review → Save decisions → Generate new variants & retrain**
until every question has at least one acceptable variant (and ideally several good ones for variety).

---

### Optional: answer-only check
When the wording and hints look right, switch to the “Answer Review” tab inside the app to double-check final answers and log a quick rationale for each.

---

### Behind the scenes (for reference only)
- `data/review/variant_choices.jsonl` stores every decision you make.
- `data/filtered/refresh_accept.jsonl` keeps the variants you approved (preferred ones are weighted more).
- `data/filtered/refresh_reject.jsonl` holds the variants you rejected or skipped so they can be regenerated next round.
- `models/adapters/<name>/` and `results/<name>/` contain the newly trained adapter and its evaluation reports.

That’s all—work entirely inside the reviewer: review the variants, press the sidebar button, and the system takes care of the rest.

