# Review → Retrain Cycle (UI version)

Everything here can be done inside the **MCQ Variant Reviewer** Streamlit page—no terminals or commands required.

---

## Step 0 — One-time install & launch
Run these commands in PowerShell (inside your virtualenv if you use one):

```powershell
pip uninstall -y torch torchvision torchaudio
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install streamlit transformers datasets accelerate peft opencv-contrib-python
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print([torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())])"
streamlit run ui/variant_review.py
```
- The uninstall step keeps you from mixing the old CPU-only wheels with the new CUDA 12.8 nightly that supports RTX 50‑series cards.
- The verification line should print something like `2.7.0.dev…`, `12.8`, `True`, and `[(18, 0)]`—that tells you PyTorch can see your GPU.
- If the page says “No variant groups left”, click **Generate new variants & retrain** once to produce the first batch.

---

## Step 1 — Review the five variants
Each cycle picks **five random question parts** from the whole past-paper dataset. For the group currently on screen:
1. Read the five variants shown side-by-side.
2. Choose a **Preferred variant** (`1–5`) or leave “None” if nothing stands out yet.
3. For every variant, pick **Accept** or **Reject** and leave a short note (e.g. “Accept – hint mentions g correctly”, “Reject – explanation copies the question”).
4. Click **Save decisions**.
5. Move between questions with **Previous**, **Skip group**, or **Next**.

Your notes are saved automatically to the project log so the training script knows which variants to keep.

---

## Step 2 — Retrain from the sidebar
Once you have a batch of decisions:
1. Open the sidebar (tap the “>” chevron if it’s hidden).
2. Check the settings:
   - **Adapter name** — pick the label for the new training run (e.g. `adapter_v3`).
   - **Compare-with metrics** — usually the previous adapter’s metrics file.
   - **Questions per batch** — default is 5 (change this if you want to review more or fewer questions next time).
   - **Variants per question** — default is 5; lower it if you want fewer drafts per question.
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
- `data/eval/eval_set.jsonl` is the fixed 300-question evaluation deck every run uses for comparable metrics.
- `models/adapters/<name>/` and `artifacts/results/<name>/` contain the newly trained adapter and its evaluation reports.

That’s all—work entirely inside the reviewer: review the variants, press the sidebar button, and the system takes care of the rest.

