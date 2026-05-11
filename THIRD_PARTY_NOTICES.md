# Third-party notices

Quire is distributed under the MIT License (see `LICENSE`). This file
documents the third-party assets and runtime dependencies that ship with
or are required by Quire, along with their license terms and attribution
requirements.

---

## Bundled data assets

### Tanzil Quran text (Uthmani, Version 1.1)

- **File:** `data/canonical/quran/tanzil-uthmani.txt`
- **License:** [Creative Commons Attribution 3.0](https://creativecommons.org/licenses/by/3.0/)
- **Copyright:** © 2007–2026 [Tanzil Project](https://tanzil.net/)
- **Source:** <https://tanzil.net/download/>

The text is redistributed verbatim and unmodified. The Tanzil Project
forbids modification of the text. The original copyright notice is
preserved inside the file (see the trailing comment block).

When you redistribute Quire or any artifact derived from this corpus you
must:

1. Keep the Tanzil copyright notice in the file.
2. Clearly indicate Tanzil Project as the source of the Quran text.
3. Provide a link to <https://tanzil.net/> so users can track updates.

### Bundled fonts

- **Files:**
  - `data/fonts/LiberationSerif-Regular.ttf`
  - `data/fonts/LiberationSerif-Bold.ttf`
  - `data/fonts/LiberationSerif-Italic.ttf`
  - `data/fonts/LiberationSerif-BoldItalic.ttf`
  - `data/fonts/NotoNaskhArabic.ttf`
  - `data/fonts/NotoNastaliqUrdu.ttf`
- **License:** [SIL Open Font License 1.1](https://scripts.sil.org/OFL)
- **Liberation copyright:** © Red Hat, Inc. and contributors.
- **Noto copyright:** © Google LLC and contributors.

The full SIL OFL 1.1 license text is reproduced at
`data/fonts/OFL.txt`. The OFL permits redistribution, embedding, and
modification, but the fonts themselves may not be sold by themselves and
any derivative font must keep the OFL.

---

## Runtime dependencies

### PyMuPDF

- **Package:** `PyMuPDF` (imported as `fitz`)
- **License:** [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html)
  or a commercial license from Artifex Software, Inc.
- **Project:** <https://github.com/pymupdf/PyMuPDF>

PyMuPDF is **AGPL-3.0**. Quire itself is MIT, and Quire only imports
PyMuPDF as a library — Quire does not vendor or redistribute the PyMuPDF
sources. Downstream redistributors who bundle PyMuPDF (e.g. in a
container image, a frozen binary, or a hosted service) inherit the
AGPL-3.0 obligation to offer the corresponding source code of any
modifications to users who interact with the running program. If that
constraint is unacceptable for your use case, Artifex offers PyMuPDF
under a commercial license — see <https://artifex.com/licensing/>.

### Other Python dependencies

The full transitive dependency set with versions is recorded in
`requirements.lock` and `requirements-dev.lock`. Each package retains
its own license; the primary direct dependencies are:

| Package          | License             | Project |
|------------------|---------------------|---------|
| `pillow`         | MIT-CMU             | <https://github.com/python-pillow/Pillow> |
| `pyspellchecker` | MIT                 | <https://github.com/barrust/pyspellchecker> |
| `tomli`          | MIT                 | <https://github.com/hukkin/tomli> |
| `ocrmac` (opt.)  | MIT                 | <https://github.com/straussmaximilian/ocrmac> |
| `pytesseract`    | Apache-2.0          | <https://github.com/madmaze/pytesseract> |

Tesseract OCR itself is an external system dependency (Apache-2.0); it
is not bundled with Quire. The macOS Vision OCR engine is provided by
Apple's operating system and is accessed only at runtime via `ocrmac`.

---

## Reporting omissions

If you believe an asset has been redistributed here without the correct
attribution, please open an issue.
