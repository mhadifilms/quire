# The Quire book workspace

Import → review → translate → check → publish. A project contains the original
PDF, proof images, a structured manuscript, translations, edit history, and
versioned editions. Everything stays in ordinary local files.

## Install and open

Python 3.10+ on macOS or Linux is supported. Install Tesseract and the language
packs needed by your books, then install Quire's `studio` extra:

```bash
pip install -e '.[studio]'
quire studio                         # library: books/workspace; local port 8765
quire studio /path/to/library --port 8770
quire review /path/to/project
```

The service binds to `127.0.0.1`. It is a local editing application, not a
multi-user hosted service. Browser changes use a session token and revision
checks. Two windows cannot silently overwrite each other's saved edits.

## 1. Import a book

Drop a PDF into the library or use **Import a book**. Confirm a language if
its detection is uncertain. The command line exposes explicit routing too:

```bash
quire import source.pdf books/workspace/my-book --language auto
quire import source.pdf books/workspace/my-book --language fa --engine tesseract
```

A readable embedded text layer is used page by page. Scans use OCR; weak pages
and uncovered regions receive bounded fallback attempts. RTL and Latin OCR
passes are kept separate to avoid losing the minority script. Available
language packs determine OCR coverage. Unsupported or uncertain language
identification stays visible for review.

Rerunning the same import resumes unfinished pages without replacing completed
pages or approved edits. A changed source PDF requires a new project. Recover
an omitted passage by drawing a box over the scan; an illustration becomes a
cropped image with an editable description.

## 2. Review against the source

Select a passage in either pane to highlight its source region. Edits are
attached to stable passage IDs. **Structure, notes and source** exposes:

- Passage type, heading level, and language.
- Reading order and joining split paragraphs on the same page.
- Footnote links, table cells, and illustration descriptions.
- Explicit exclusions and editorial notes for unreadable source text.
- The original extraction, including every region of a joined passage.

Save changes, then approve the passage after comparing it with the scan.
**Check completeness** is separate: it confirms that the page's content has
been accounted for, including anything the extractor missed. **History** can
undo passage edits, translations, reading-order changes, joins, recovered
regions, and page approval. Excluded material retains its source and reason.

Unsaved drafts remain available while navigating passages or the library in
the current browser session. Leaving the tab with drafts triggers a warning;
use **Save changes** for durable storage. On smaller screens, **More** exposes
book details, history, page navigation, and the review queue.

## 3. Translate with source alignment

Choose **Translate**, select a target language, and save terminology as
`source term = preferred translation`. **Write translation** opens manual
editing. Automatic translation uses Gemini, retaining chapter context, passage
IDs, table dimensions, notes, and uncertainty signals. Model drafts need review.

```bash
# Set GEMINI_API_KEY in the environment before opening the workspace, or:
quire translate books/workspace/my-book --language en --token-budget 100000
```

Automatic translation sends the book's text, context, and glossary to Google's
Gemini API. Import, local OCR, manual review, and publication do not require
that service. The token budget is per invocation, not a dollar spending limit.
Completed batches are checkpointed. A truncated response is never partly applied;
rerun to resume. Human and approved translations are protected from replacement.

Changing source text or terminology makes affected translations need review.
Passage alignment checks do not prove that prose is faithful: compare the
translation against the source before approving it.

To work without an API key or with another translator:

```bash
quire translate books/workspace/my-book --language en --export-request request.json
# Produce response.json containing all and only the passage IDs in request.json:
quire translate books/workspace/my-book --language en --request request.json --import-response response.json
```

A response has this shape:

```json
{"translations":[{"id":"p0001-b0001","text":"Full translated passage","uncertainties":[]}]}
```

Table results additionally include `cells`, preserving every row and column.
The original request binds the response to the project, source text, and glossary.

## 4. Check and publish

```bash
quire project-check books/workspace/my-book --language en
quire publish books/workspace/my-book --language en --bilingual --template study \
  --format pdf --format epub --format html --format markdown --format text
```

The same saved manuscript supplies all five outputs. Templates are **reading**
(A5), **study** (A4), and **large-print** (A4). Bilingual editions put the original
before its translation, with independent script direction. PDF includes bookmarks;
EPUB includes source-page navigation, fonts, tables, images, and notes.

Unreviewed content, missing pages, stale translations, broken notes, or unexplained
exclusions block a reviewed edition. `--draft` permits a clearly labeled proof.
No character-count ratio or OCR confidence is called recognition accuracy.

Install `epubcheck` and `@daisy/ace` to run both EPUB validators. Ace also needs
its supported browser; follow the [Ace installation guide](https://daisy.github.io/ace/getting-started/installation/).
Results and output checksums appear in each edition's `release.json`:

- `draft`: content still needs review, regardless of validator results.
- `validation_pending`: manuscript review is complete; a validator is unavailable.
- `ready`: manuscript and the requested format's available validation gates passed.

A failed installed validator blocks a reviewed release. Passing automated
accessibility checks is not accessibility certification. PDF exports are visually
reviewable, but are not claimed to be tagged PDF/UA documents. HTML, Markdown,
and text may reference accompanying `fonts` or `assets`; keep the edition folder
together when moving those formats.

## 5. Move or back up a project

```bash
quire bundle books/workspace/my-book /backups/my-book.quire.zip
quire restore /backups/my-book.quire.zip /new-library/my-book
```

Bundles preserve source files, corrections, translations, history, and editions.
Restoration requires a new destination and verifies all checksums, source identity,
and archive paths before making the project available. No database is required.

## Optional layout analysis

```bash
pip install -e '.[studio,layout]'
quire layout books/workspace/my-book
# Or import an existing Docling JSON export:
quire layout-import books/workspace/my-book docling.json
```

Run layout analysis before human review or in a separate comparison project.
It may download model weights on first use. Quire uses Docling's table, picture,
heading, caption, and footnote structure while retaining source coordinates and
raw JSON. Complex merged table cells and reading order still need review.

Docling remains opt-in because it lost more text than Quire's default routing
on the measured mixed-script sample. Its language-aware Tesseract configuration
was tested, as was the adapter with tables and pictures.

## Reference benchmarks

```bash
python -m scripts.create_benchmark_corpus /tmp/quire-corpus
quire benchmark /tmp/quire-corpus/manifest.json --output /tmp/quire-results
quire benchmark /tmp/quire-corpus/manifest.json --output /tmp/quire-docling \
  --engine docling --baseline /tmp/quire-results/benchmark.json
```

Use separate output directories for fresh timing comparisons. Reports label
resumed projects, reference provenance, character/word error, omitted words,
runtime, model usage, and cost. Human review time is unavailable for an automated
extraction benchmark; real editing time is recorded in the project. Optional
per-case `limits` make regressions return a failing exit status in CI.

The included corpus contains five original authored passages rendered as PDFs,
including Persian, Arabic, mixed-script, and degraded/rotated scans. These are
small regression samples, not a claim about accuracy across complete books.
A local run on September 7, 2026 measured:

| Case | Quire character error | Omitted words | Docling + Tesseract character error | Omitted words |
|---|---:|---:|---:|---:|
| English prose | 0.00% | 0 | 0.00% | 0 |
| Persian scan | 0.91% | 0 | 5.74% | 3 |
| Arabic scan | 0.31% | 0 | 1.57% | 0 |
| Mixed Persian / English | 0.00% | 0 | 20.90% | 13 |
| Degraded rotated English | 0.00% | 0 | 0.00% | 0 |

The benchmark workflow builds the wheel and keeps JSON/Markdown reports as CI
artifacts. Extend the manifest with independently transcribed real book samples
and their PDF SHA-256 before drawing conclusions about a larger collection.

## Compatibility with configured books

Existing `build`, `convert`, `batch`, and `audit` commands remain available.
Automatic conversion now uses language evidence instead of assuming English.
New AI QC corrections in `qc_fixes.toml` have a source fingerprint and page scope.
A correction applies only to one exact occurrence on that page; ambiguous,
cross-page, or stale matches appear in `qc_scope_report.json` for review. Previous
files are saved under `qc_fixes.history/`. Existing human-authored `[phrase]`
rules retain their deliberate book-wide behavior.

## Validation performed

Regression coverage includes checkpoint recovery, duplicate phrases on different
pages, stale revisions, reversible edits and joins, translation alignment and
budgets, table cells, missing content, archive traversal/checksums, the live local
HTTP service, and text retained in every export format. Gemini's request/response
protocol is tested with a simulated service; no live API key was available for
this implementation run. A Persian/English edition passed real EPUBCheck and
DAISY Ace, and its PDF was rendered and visually inspected. Browser checks covered
editing, draft retention, recovery/undo, terminology, manual translation, and
portrait/landscape layouts.
