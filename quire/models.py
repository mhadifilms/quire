"""Typed data contracts for the pipeline.

The pipeline historically threaded loosely-typed ``dict`` objects between
stages. These TypedDicts document the shape contracts and make refactors and
plugins safer without forcing a full migration to a dataclass-based pipeline.

We deliberately keep the dict-based runtime representation so existing
post-processors and renderers keep working; the TypedDicts are a static
contract that tooling and tests can rely on.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class PageMeta(TypedDict, total=False):
    """Per-page metadata captured by the PDF extractor.

    The ``pno`` field is the 0-indexed PDF page number; ``printed_page`` is
    the human-readable page number printed in the source (or ``None`` if the
    extractor couldn't infer one).
    """

    pno: int
    printed_page: int | None
    width: float
    height: float
    body_size: float
    lines: list[dict]
    text_layer: str
    error: str


class Element(TypedDict, total=False):
    """A structured block that lives inside an OCRPage.elements list.

    The discriminator field ``kind`` controls which extra keys are set:

    - ``"heading"``  : ``level`` (int), ``text``
    - ``"paragraph"``: ``text``, optional ``indent``, ``centered``
    - ``"arabic"``   : ``text``, optional ``conf``, ``script_lang``, ``is_quran``
    - ``"image"``    : ``href``, ``caption``
    """

    kind: Literal["heading", "paragraph", "arabic", "image", "blockquote"]
    text: str
    level: int
    y: float
    indent: bool
    centered: bool
    script_lang: str
    conf: float
    is_quran: bool
    href: str
    caption: str
    _pdf_pno: int
    _printed: int


class Footnote(TypedDict, total=False):
    """A footnote attached to a chapter."""

    number: str
    text: str
    y: float
    _pdf_pno: int
    _printed: int


class OCRPage(TypedDict, total=False):
    """A single page after OCR + structuring.

    ``elements`` is the rendered content list; ``arabic_blocks`` is kept on
    the page for downstream refinement plugins.
    """

    pno: int
    printed_page: int | None
    elements: list[Element]
    arabic_blocks: list[dict]
    error: str


class AuditFinding(TypedDict, total=False):
    """One row of suspicious content surfaced by the audit module."""

    kind: str
    file: str
    page: int
    excerpt: str
    suggested_fix: str


class AuditResult(TypedDict, total=False):
    """Machine-readable audit summary written to ``audit.json``."""

    slug: str
    title: str
    language: str
    ocr_engine: str
    chapter_files: int
    english_words: int
    arabic_chars: int
    pdf_english_words: int
    pdf_english_words_real: int
    ocr_arabic_chars_real: int
    english_coverage_pct: float
    arabic_coverage_pct: float
    unresolved_links: int
    suspicious_count: int
    per_chapter: list[dict[str, Any]]
    findings: list[AuditFinding]
    epubcheck_status: str
    epubcheck_messages: list[dict[str, Any]]


__all__ = [
    "PageMeta",
    "Element",
    "Footnote",
    "OCRPage",
    "AuditFinding",
    "AuditResult",
]
