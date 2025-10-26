Quick daily snapshot (non-technical):
- Day 1: Organised the exam folders so everything is tidy and logged coverage in `data/README.md` (no new scripts, just cleaner structure).
- Day 2: Built the PDF extractor (`scripts/extract_pdf.py`) so we can pull clear text and images straight from the papers.
- Day 3: Added the auto-caption and graph tools (`scripts/ocr_images.py`, `scripts/analyse_graphs.py`, `scripts/describe_graphs.py`) so every diagram has a description ready to use.
- Day 4: Wrote human-friendly diagram notes saved in `data/graph_human_descriptions/AQA/` to guide the model later.
- Day 5: Trained and evaluated the visual helper with `scripts/train_graph_model.py` / `scripts/eval_graph_model.py`, saving adapters in `models/adapters/graph_reader_v1/` and metrics in `results/graph_reader_v1/`.
- Day 6: Parsed the past-paper content into `data/parsed/questions.jsonl` using `scripts/parse_questions.py` and indexed it with `scripts/build_question_index.py` (`data/index/questions.db`).
- Day 7: Generated seed MCQs via `scripts/make_seed_data.py` and logged reviews with `scripts/review_seed.py`, producing `data/parsed/seed_drafts.jsonl`, `data/parsed/seed_train.jsonl`, and `data/review/seed_notes.jsonl`.

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
- Checked `questions.jsonl` so I know which parts already have answers and which are diagram-heavy or missing mark schemes.
- Locked down the MCQ template (stem, four options, correct index, hint, explanation, AO, topic, difficulty, provenance) so every seed item looks the same.
- Wrote `scripts/make_seed_data.py` to pull 500 draft MCQs with auto-filled distractors, AO/topic guesses, and flag tags when something looks off.
- Added `scripts/review_seed.py`, a quick CLI that lets us approve/reject drafts and logs the decision in `data/review/seed_notes.jsonl`.
- Auto-approved a clean set of 120 items and saved them in `data/parsed/seed_train.jsonl` with `status=approved`, ready for Day 8 to build on.
- what we basically did: Day 7 stitched the parsed content into living MCQ seeds—500 drafts plus 120 signed-off items—so the upcoming automation can start refining instead of rebuilding from scratch.
- seed_train vs seed_notes: seed_drafts.jsonl holds every auto-generated draft (good and bad) along with flags so reviewers know what still needs work. seed_train.jsonl is the filtered subset that passed review—only those 120 approved MCQs we’ll actually train or ship with. 
- basically, we run "python scripts/review_seed.py --drafts data/parsed/seed_drafts.jsonl --notes data/review/seed_notes.jsonl" to launch manual review of the drafts, if we accept them, it'll go to the seed_train.jsonl
- then we can check how many we approved using "python -c "import json,collections; entries=[json.loads(l) for l in open('data/review/seed_notes.jsonl', encoding='utf-8') if l.strip()]; print(collections.Counter(e['status'] for e in entries))", there's already 120 pre-approved. These are MCQS SEED CANDIDATES, not actual ready MCQs, it's like a starter park the model will try to learn from