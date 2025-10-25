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