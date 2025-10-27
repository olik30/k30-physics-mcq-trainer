Quick daily snapshot (non-technical):
- Day 1: Organised the exam folders so everything is tidy and logged coverage in `data/README.md` (no new scripts, just cleaner structure).
- Day 2: Built the PDF extractor (`scripts/extract_pdf.py`) so we can pull clear text and images straight from the papers.
- Day 3: Added the auto-caption and graph tools (`scripts/ocr_images.py`, `scripts/analyse_graphs.py`, `scripts/describe_graphs.py`) so every diagram has a description ready to use.
- Day 4: Wrote human-friendly diagram notes saved in `data/graph_human_descriptions/AQA/` to guide the model later.
- Day 5: Trained and evaluated the visual helper with `scripts/train_graph_model.py` / `scripts/eval_graph_model.py`, saving adapters in `models/adapters/graph_reader_v1/` and metrics in `results/graph_reader_v1/`.
- Day 6: Parsed the past-paper content into `data/parsed/questions.jsonl` using `scripts/parse_questions.py` and indexed it with `scripts/build_question_index.py` (`data/index/questions.db`).
- Day 7: Turned the best MCQ drafts into a 460-question starter pack, logged reviewer decisions, and parked the messy ones for follow-up.
- Day 8: Hooked up an auto-generation draft loop, but the model’s answers still need work—everything landed in the reject pile for another pass.

==============================================================================================================

Day 1 recap:
- organised every exam board into tidy folders (kept question paper + mark scheme pairs together)
- logged what we have in `data/README.md`, showing years covered and any gaps
- deleted old pre-2017 papers and the unused WJEC Unit 5 folder so we only keep the years we agreed on

Day 2 recap:
- first version of the PDF extractor could pull page text + screenshots, but it sometimes saved blank/low-quality diagrams and crashed on the big mark-scheme PDFs
- strengthened it so the diagrams stay sharp and the mark-scheme files no longer crash
- created `scripts/extract_pdf.py`, which reads each exam PDF and saves the text plus clear pictures (uses pdfplumber + PyMuPDF to pull both photos and vector drawings)
- erased the old text/image outputs, ran a fresh full pass (2025-10-16 00:09) and confirmed every file processed cleanly
- we now have high-quality diagrams ready for the Day 3 caption/graph pass

Day 3 recap:
- created and ran the image caption script (`scripts/ocr_images.py`), so every image has their own "denoised" text json, which complements the raw text
- each run finished with thousands of “ok” files and “warns” only when the picture is just a logo or a pure graph (no text for Tesseract to read); there were no errors or crashes
- the warnings are expected: they simply flag blank captions for images that are pure visuals, which is fine because the graph analysis script will handle those diagrams next
- finished the graph analysis pass (`scripts/analyse_graphs.py`) so every diagram (including composites) now has a JSON with its key graph features; anything skipped was just a page with no axes/plot to analyse
- ran the graph description script (`scripts/describe_graphs.py`) so each analysed diagram now has a short plain-English blurb; the files are trimmed to just the summary so non-coders can skim them quickly

Day 4 recap:
- picked 40 AQA diagrams that cover mechanics, electricity, waves, fields, and data-handling charts so we have a good mix for manual checks
- wrote human-friendly descriptions for each picture (added as matching JSON files next to the PNGs under `data/graph_human_descriptions/AQA`)
- the CSV we used for editing is gone so only the cleaned images + notes remain in the repo

Day 5 recap:
- set up a fresh Python 3.12 workspace and installed all the Hugging Face tools we need so future commands run without surprises.
- Opened a handful of the 40 human-labelled graph files and confirmed every JSON lines up with the matching image and uses the same field names.
- Planned how the training script will read each diagram: the script pulls the original PNG straight from disk, pairs it with the human summary, and keeps everything in place so we don’t duplicate files or waste space.
- wrote a training helper (`scripts/train_graph_model.py`) so the computer can learn from those human notes without touching the original images.
- added a simple checker (`scripts/eval_graph_model.py`) that compares the model’s summaries with our human ones and highlights any that fall below the similarity threshold.
- saved the trained weights as `models/adapters/graph_reader_v1` and kept a note of the run in `logs/train/graph_reader_v1` so we remember what happened.
- wrote the score sheet to `results/graph_reader_v1/metrics.json` and the eight example write-ups to `results/graph_reader_v1/samples.jsonl`; two of those examples are still a bit too different from the human wording.
- anything in `models/` is a saved model we can load later, and anything in `results/` is a score report from the most recent test run.

Day 6 recap:
- `scripts/parse_questions.py` now reads each paper, skips the noisy headers, and writes clean question records to `data/parsed/questions.jsonl` (about 1000 items in plain English). This file is the master list of questions we will sample from on Day 7, so we avoid editing the raw OCR again.
- `scripts/build_question_index.py --force` turns that JSONL into a handy `data/index/questions.db` file so we can look up any question, part, or linked image quickly. It is just a lightweight SQLite file, no special software needed.
- Row counts (questions 1004 / parts 1126 / assets 3194) checked out, so the database matches the JSON and is ready for the next day. If those numbers ever drift, rebuild with `--force` and re-run the quick row-count command.
- what we basically did: we now have the raw questions, mark schemes, and linked images living in one tidy source (`questions.jsonl`) plus a searchable version in SQLite, so Day 7 can focus on curating MCQs instead of parsing paperwork.

Day 7 recap:
- Re-ran the MCQ builder so every question part now shows up as a tidy draft with flags when something looks off (missing figure, repeated option, mark-scheme spill, etc.).
- Went through the flagged list and rejected 45 bad drafts (messy wording, duplicate answers, or mark-scheme language showing).
- Spot-checked 18 clean drafts, then approved the rest that passed the checklist—now 460 questions sit in the “good” pile with reviewer notes.
- Promoted the approved set into `data/parsed/seed_train.jsonl`; any draft not approved stays in `seed_drafts.jsonl` for a later fix pass.
- what we basically did: Day 7 gave us a 460-question starter pack we trust, plus a smaller reject list so we know exactly which items still need hand-cleaning.

Day 8 recap:
- Set up the local draft generator (`scripts/auto_generate.py`) and confirmed Ollama (`qwen2.5:7b-instruct`) runs locally.
- First 100-model batch (`data/parsed/auto_drafts.jsonl`, log in `logs/auto_generate/run.jsonl`) still gave wobbly physics and bad numerics, so we rejected them all via `scripts/review_seed.py`.
- Tweaked the prompt/sanitiser and tried a 20-question pilot (`python scripts/auto_generate.py --limit 20`), but those explanations were still off, so the second batch also stays in `data/review/seed_notes.jsonl` as “rejected”.
- Bottom line: Day 8 pipeline works end-to-end, but the drafts aren’t good enough yet—we’ll revisit the prompt and heuristics on Day 9 before generating another set.