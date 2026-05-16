"""AI-assisted QC pass for Quire.

This package implements a page-by-page proofreading stage powered by a
Vision-Language Model (Gemini 2.5 Flash by default). For every rendered
PDF page, the page image plus the EPUB's extracted plain text are sent
to the model, which proposes ``find`` / ``replace`` corrections as
structured JSON. Validated corrections are merged into the book's
``qc_fixes.toml``, which the existing post-render typography stage
applies on the next build.

Public surface:

- :func:`run_qc` — orchestrator used by both the CLI and the build pipeline.
- :class:`QCResult` — summary of a single QC run.
- :class:`Correction` — one validated find/replace pair.
"""

from __future__ import annotations

from .models import Correction, QCResult
from .runner import run_qc

__all__ = ["Correction", "QCResult", "run_qc"]
