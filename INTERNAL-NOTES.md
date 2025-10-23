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

Day 4 recap (so far):
- picked 40 AQA diagrams that cover mechanics, electricity, waves, fields, and data-handling charts so we have a good mix for manual checks
- wrote human-friendly descriptions for each picture (added as matching JSON files next to the PNGs under `data/graph_human_descriptions/AQA`)
- the CSV we used for editing is gone so only the cleaned images + notes remain in the repo
- still to do: make a small comparison script/notebook that lines up our manual notes with the auto summaries and flags any gaps or oddities before we move on