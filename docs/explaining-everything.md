Day 1: We gathered every exam paper and mark scheme and organised them neatly so nothing goes missing.
Day 2: We pulled the text and pictures out of those papers so they’re easy to browse without opening each PDF.
Day 3: We read the diagrams and charts, wrote plain descriptions for them, and cleaned up any messy text.
Day 4: We built a small reference set of diagrams and wrote human notes so we know what “good” looks like.
Day 5: We taught a helper model how to understand those diagrams using the human notes as examples.
Day 6: We turned every exam question into tidy records that link the question, answer, and any images together.
Day 7: We created a first batch of multiple-choice questions, reviewed them, and kept the 460 best ones.
Day 8: We hooked up a local AI to draft new questions and logged where its answers still fell short.
Day 9: We built an automatic checker that cleans the question files, spots broken entries, and shows topic gaps.
Day 10: We bundled the approved questions into the training format and sliced them into train/validation/test files automatically.
Day 11: We let the AI train on that cleaned set (one quick sanity run so far) and recorded the training stats.
Day 12: We ran an automatic report card on the trained AI; ~84% of the outputs were valid JSON, ~69% passed the schema checks, but none matched the reference answers—so we have work to do before trusting it.
Day 13: We regenerated 40 of the weakest MCQs, logged why they failed, and built a tiny CLI + playbook so humans can approve or reject them after Day 14.
Day 14: We trained a second adapter, compared it to the first, and bundled every artifact (datasets, metrics, adapters, review tools) into `artifacts/handoff/day14_artifacts/` for human QA.

Inside data/
pdfs/ – the original exam papers and mark schemes, untouched.
raw/ – plain text dumps made from those PDFs (one file per page).
images/ – every picture or diagram clipped out of the papers.
captions/ – short OCR descriptions of each image so we remember what it shows.
graph_analysis/ – measurements and geometry pulled from chart-style diagrams (axes, slopes, etc.).
graph_descriptions/ – tidy one-paragraph explanations of those graphs.
graph_human_descriptions/ – the human-written “gold” summaries we used for training.
parsed/ – structured question files (question text, mark schemes, linked assets).
index/ – a quick-lookup database built from the parsed questions.
filtered/ – the cleaned MCQ drafts that passed Day 9 checks.
eval/ – rotating evaluation decks (three 60-question cores plus the larger audit set).
review/ – notes on what we approved/rejected during manual reviews.
other_syllabuses/ – archived boards we’re not using right now, kept out of the AQA pipeline.

Other top-level folders
docs/ – all written guides, daily recaps, and workflow plans.
artifacts/logs/ – run-by-run transcripts from scripts (extraction, OCR, training, etc.).
artifacts/state/ – small JSON files that track things like which evaluation deck to use next.
models/ – saved model files and adapter checkpoints.
config/ – machine-readable notes about each adapter run (`config/adapters.yaml`).
artifacts/results/ – evaluation metrics, dataset stats, and sample outputs.
scripts/ – every command-line helper we’ve written (extract, filter, format, train…).
tests/ – automated checks to make sure scripts behave before we rely on them.