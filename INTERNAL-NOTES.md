Day 1 recap:
- organised every exam board into tidy folders (kept question paper + mark scheme pairs together)
- logged what we have in `data/README.md`, showing years covered and any gaps
- deleted old pre-2017 papers and the unused WJEC Unit 5 folder so we only keep the years we agreed on

Day 2 recap:
- first version of the PDF extractor could pull page text + screenshots, but it sometimes saved blank/low-quality diagrams and crashed on the big mark-scheme PDFs
- strengthened it so the diagrams stay sharp and the mark-scheme files no longer crash
- created `scripts/extract_pdf.py`, which reads each exam PDF and saves the text plus clear pictures (uses pdfplumber + PyMuPDF to pull both photos and vector drawings)
- erased the old text/image outputs, ran a fresh full pass (2025-10-16 00:09) and confirmed every file processed cleanly (`ok=1163 warn=0 error=0`)
- we now have high-quality diagrams ready for the Day 3 caption/graph pass

Day 3 recap (so far):
- created and ran the image caption script (`scripts/ocr_images.py`) across the all boards, so every diagram now has a small text note saved beside it
- each run finished with thousands of “ok” files and “warns” only when the picture is just a logo or a pure graph (no text for Tesseract to read); there were no errors or crashes
- the warnings are expected: they simply flag blank captions for images that are pure visuals, which is fine because the graph analysis script will handle those diagrams next