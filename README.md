# Quire

Quire is a reflowable-EPUB and multi-format conversion pipeline for image-heavy
or mojibake-laden PDFs, with first-class support for Arabic, Persian, Hebrew,
Urdu, and other RTL/multi-script content. It is designed to scale from a single
book to large batches of heterogeneous books in many languages.

The pipeline takes a PDF (or a manifest of PDFs) and produces clean, reviewable
outputs with:

- Reflowable XHTML chapters (no fixed-layout pages).
- EPUB3, standalone HTML, Markdown, and plain-text exports from one structured
  chapter model.
- Preserved printed page numbers (`<span epub:type="pagebreak">`).
- Pluggable OCR back-ends. macOS Vision (`ocrmac`) is supported out of the box
  for scanned or mojibake-heavy PDFs; the `text` engine reads existing PDF
  text layers for files that already contain reliable text.
- Pluggable post-processors:
  - **Vocabulary** — substitute transliterations with canonical script.
  - **Glossary auto-extract** — bootstrap the vocabulary from a printed glossary.
  - **Canonical sources** — replace OCR'd Quran/Bible/etc. by citation.
  - **Script detection** — tag Persian, Hebrew, Greek runs with `lang=`.
  - **Mojibake cleanup** — repair PDF text-layer bleed-through.
- Quality audit with per-page metrics, JSON output, structured logs, and
  optional EPUBCheck integration.
- Batch processing with bounded parallelism, per-book status manifests, and
  partial-failure isolation.

## Layout

```
quire/
├── quire/                        # generic pipeline package
│   ├── extract/                  # PDF + OCR layer extraction
│   ├── structure/                # paragraph / heading / footnote reconstruction
│   ├── render/                   # XHTML + EPUB packaging + audit
│   ├── postprocess/              # pluggable language / canonical helpers
│   ├── batch.py                  # multi-book batch runner
│   ├── logging_utils.py          # structured logging
│   └── pipeline.py               # orchestrator
├── books/<slug>/                 # one folder per book
│   ├── book.toml                 # book-specific config
│   ├── source.pdf                # input PDF
│   ├── vocabulary.json           # optional canonical vocabulary
│   └── artifacts/                # outputs (epub, caches, audit)
├── data/                         # shared canonical corpora (Quran, fonts, …)
├── tests/                        # pytest suite
└── pyproject.toml
```

## Quickstart

```bash
# Default install includes the Tesseract OCR engine
brew install tesseract tesseract-lang   # macOS
# or: sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fas  (Debian/Ubuntu)
pip install -e ".[tesseract]"

# For the optional macOS Vision OCR engine:
pip install -e ".[vision]"

quire build books/my-book
quire build books/my-book --all-formats
quire build books/my-book --retry-failed   # resume after partial OCR failure
quire check books/my-book                  # validate book.toml without building
quire audit books/my-book

# Batch many books in one run with bounded parallelism:
quire batch --manifest batch.toml --workers 4
```

### OCR engines

Quire ships with three OCR engines. New books default to **Tesseract 5**
because it works everywhere and covers 163 trained languages including
Arabic, Persian, Urdu, Hebrew, Chinese (simplified + traditional), Russian,
Greek, Hindi, Japanese, and Korean — all with native RTL reading order.

| Engine | When to use | Setup |
|---|---|---|
| `tesseract` (default) | Most books, all platforms, all scripts. | `brew install tesseract tesseract-lang` + `pip install ".[tesseract]"` |
| `vision` | macOS-only; faster than Tesseract on Apple Silicon. ~30 languages. | `pip install ".[vision]"` (macOS only) |
| `text` / `pdf` / `pymupdf` | The PDF already has a reliable embedded text layer. No OCR is run. | Built-in |

Configure per book in `book.toml`:

```toml
[ocr]
engine = "tesseract"          # or "vision" or "text"
languages = ["ar-SA", "en-US"] # BCP-47 codes; auto-mapped to tesseract codes
workers = 4
dpi_scale = 3                  # higher = better quality, slower
retries = 1                    # retry transient OCR failures
```

The `languages` field accepts either BCP-47 tags (`ar-SA`, `fa-IR`,
`zh-CN`, …) or Tesseract's 3-letter codes (`ara`, `fas`, `chi_sim`, …)
directly. Vision uses BCP-47 natively; for Tesseract, Quire translates.

For RTL/LTR mixed pages Tesseract runs **two independent OCR passes** —
one per script direction — and merges the line outputs by position.
Combined-language mode (e.g. `lang=eng+ara`) silently drops the minority
script in the LSTM model; the two-pass strategy mirrors what macOS Vision
does internally.

The optional `[postprocess.arabic_refine]` Vision-based re-OCR step is
**skipped automatically for non-Vision engines** — Vision's Arabic
detection is more conservative than Tesseract's, and applying it to
Tesseract output would overwrite ~65 % of the recognized Arabic with
Vision's narrower view of the same regions.

Tesseract builds also pass through three hallucination guards that
together drop ~95 % of the `ara` LSTM's worst failure mode (random
Arabic glyphs emitted over Latin text). All three run **only when
`engine = "tesseract"`** so Vision builds stay bit-identical.

**1. Per-block strict-digits filter** (`_filter_vision_blocks` in
`quire/pipeline.py`). Hallucinated Arabic blocks always carry an
unusual density of ASCII digits (citation page numbers, years, volume
refs) that real Arabic prose never has. The filter rejects:

- blocks where ASCII digits exceed 20 % of the Arabic character count;
- short blocks (< 30 Arabic chars) containing any ASCII digit at
  confidence < 60.

**2. Block-level embedded-text filter**
(`_is_embedded_text_hallucination` in `quire/pipeline.py`). The
strongest of the three guards — it uses the PDF's own embedded text
layer (typically added by a prior pass through Adobe Acrobat or
ABBYY FineReader on scanned books) as **ground-truth English
positions** that are independent of anything Tesseract does. An
Arabic block is rejected when **all** of:

- the PDF has an embedded text layer at all (no-op on pure scans);
- the block's horizontal extent is ≥ 50 % covered by embedded
  English-word bboxes at the same y-position;
- the block's confidence is `< 30`;
- the block's text contains **no** Arabic diacritics (tashkeel).

The diacritic check is the safety: religious / classical Arabic —
the dominant Arabic content in books that need this filter — is
heavily diacritised, so a "block of Arabic" without a single
tashkeel mark AND overlaying confirmed English AND at low conf is
not real Arabic with overwhelming probability. Measured precision on
a 161-page bilingual Arabic/English scanned reference book: ≥ 96 %
(≈5.3 K chars of hallucination dropped, ≈200 chars of real Arabic
lost). The filter is a silent no-op for scans with no embedded
layer.

**3. Page-level bibliography filter**
(`_page_is_english_citation_dense` in `quire/pipeline.py`). Catches
the few pages where every individual block looks "plausibly noisy"
but the page as a whole is clearly a Latin bibliography or hadith
index — and the embedded-text filter happens not to fire (e.g. the
prior OCR layer is incomplete). A page is suppressed when **all** of:

- `en_chars >= 1000` — substantial English captured;
- `digit_chars / en_chars >= 3 %` — citation-density signature
  (years, volume nums, page refs) absent from flowing prose;
- no anchoring Arabic block on the page (no single block with
  `ar_chars >= 100` AND `conf >= 65`), which would prove real Arabic
  prose exists.

On the bilingual reference corpus above, this page filter flags ~8
additional bibliography pages the embedded-text filter alone misses;
together the three guards remove well over 90 % of Tesseract's
bibliography hallucination while losing fewer than 200 chars of
confirmed real Arabic in 161 pages. The audit reports the post-filter
numbers as `ocr_arabic_chars_real` so metrics never silently inflate
from hallucination.

### Reproducible installs

Lock files for the runtime and dev environments are generated with
[`pip-compile`](https://github.com/jazzband/pip-tools):

```bash
pip install pip-tools
pip-compile --strip-extras --output-file requirements.lock pyproject.toml
pip-compile --strip-extras --extra dev --output-file requirements-dev.lock pyproject.toml
```

Install from the lock files when you need byte-identical environments
(e.g. CI runners that build releases):

```bash
pip install -r requirements.lock      # runtime only
pip install -r requirements-dev.lock  # tests + lint + mypy
```

The `build` command reads `books/<slug>/book.toml`, runs the pipeline, and
writes outputs plus caches to `books/<slug>/artifacts/`. The `audit` command writes:

- `audit.txt` — coverage/link sanity summary (human-readable).
- `audit.json` — machine-readable metrics for batch dashboards.
- `ocr_corrections.tsv` / `ocr_corrections.diff` — every automatic OCR correction and its rule/evidence.
- `ocr_review.tsv` — build-time OCR candidates that were not safe enough to apply automatically.
- `manual_review.tsv` — spreadsheet-friendly review queue.
- `corrections.md` — readable copy of the review queue.

## book.toml

```toml
[book]
slug = "my-book"
title = "My Book"
author = "Author Name"
language = "en"

[input]
pdf = "source.pdf"            # relative to this folder
cover_pdf_page = 1            # 1-indexed; defaults to first page

[ocr]
engine = "vision"             # vision | text
languages = ["en-US", "ar-SA"]
workers = 4                   # optional; default 4
dpi_scale = 4                 # optional; rendering scale for OCR
retries = 1                   # optional; per-page retry count

[structure]
# Optional, but recommended for reliable chapter splitting.
headings_module = "headings"   # books/<slug>/headings.py exports HEADINGS
# or: headings = [{ title = "Chapter One", level = 1 }]

[postprocess]
strict = true                  # fail if a configured plugin cannot load
plugins = [
  "vocabulary",
  "glossary_extract",
  "script_detect",
  "mojibake_cleanup",
  "common_ocr",
  "canonical_quran",
]

[postprocess.vocabulary]
path = "vocabulary.json"      # optional; relative to book folder

[postprocess.canonical_quran]
corpus = "quran/tanzil-uthmani.txt"   # relative to data/canonical/

[postprocess.common_ocr]
high_confidence = 88
low_confidence = 82
min_occurrences = 2
dominant_min_occurrences = 5
max_variant_occurrences = 1

[render]
embed_fonts = ["NotoSerif-Regular.ttf", "NotoNaskhArabic-Regular.ttf"]
include_corrections_md = true
formats = ["epub"]            # any of: epub, html, markdown, text
```

OCR caches are fingerprinted by source PDF content hash, OCR languages, and
plugin set. `--force-ocr` refreshes both the primary OCR cache and
language-specific refinement caches.

Shared exact OCR typo fixes live in `data/ocr/common_fixes.toml`. To add a
book-only correction, create `books/<slug>/ocr_fixes.toml` with the same
`[phrase]` and `[word]` sections; build output records each applied fix in
`books/<slug>/artifacts/ocr_corrections.tsv`.

## Batch processing

For thousands-of-books workloads, use a batch manifest:

```toml
# batch.toml
[[book]]
path = "books/book-001"

[[book]]
path = "books/book-002"
formats = ["epub", "markdown"]

[[book]]
path = "books/book-003"
force_ocr = true
```

```bash
quire batch --manifest batch.toml --workers 4 --status batch-status.json
```

The batch runner isolates per-book failures, writes a JSON status file, and
exits non-zero when any book fails (configurable with `--fail-fast` or
`--continue-on-error`).

## Adding a new language

1. Create a vocabulary file (`books/<slug>/vocabulary.json`) mapping
   normalized transliterations to canonical-script.
2. Optionally add a canonical source under `data/canonical/<lang>/` and
   register it in `quire/postprocess/canonical/`.
3. Configure the post-processor pipeline in `book.toml`.

## Security note

`book.toml` can reference a Python `headings_module` and a Python
`vocabulary` module that Quire imports from the book folder. These run
arbitrary Python; only ingest book directories you trust. Plugins listed
outside the built-in registry are loaded by name from the Python environment
under the same trust assumption.

## License

Quire is released under the **MIT License**. See [`LICENSE`](LICENSE)
for the full text.

### Third-party assets and dependencies

Quire bundles a small amount of third-party data and links to several
third-party Python packages at runtime. The full attribution list is
in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md); a short summary:

- **Tanzil Quran text** (`data/canonical/quran/tanzil-uthmani.txt`) —
  CC-BY 3.0, © Tanzil Project, redistributed unmodified.
- **Liberation fonts** and **Noto fonts** in `data/fonts/` — SIL Open
  Font License 1.1; full license text at `data/fonts/OFL.txt`.
- **`PyMuPDF`** runtime dependency — **AGPL-3.0** (or commercial from
  Artifex). Quire itself is MIT and only imports PyMuPDF as a library;
  downstream redistributors who bundle PyMuPDF (Docker images, frozen
  binaries, hosted services that expose the program over a network)
  inherit AGPL's source-disclosure obligation for that component. If
  that's a problem for your deployment, Artifex offers a commercial
  PyMuPDF license.
- Other Python dependencies (`pillow`, `pyspellchecker`, `tomli`,
  `ocrmac`, `pytesseract`) ship under MIT / MIT-CMU / Apache-2.0.
