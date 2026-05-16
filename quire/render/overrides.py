"""Chapter-override layer.

Some scanned-book pages defy the OCR + structure + typography pipeline:
a four-column romanization table, a heavily italicized glossary, a
column-merged index. For those cases the pipeline output is corrupt
enough that no amount of small find/replace can recover it — but
because it is per-book content (not a pipeline bug), the right escape
hatch is an author-provided XHTML override that replaces the rendered
chapter entirely.

Layout
------

Drop an override at::

    books/<slug>/chapter_overrides/<chapter-slug>.xhtml

The file is a full XHTML 1.1 document (``<?xml version="1.0" ...?>``
prolog, ``<html xmlns=...>`` root, ``<body epub:type="bodymatter">``
content). The chapter slug must match the slug emitted by the
``assemble_chapters`` stage; see ``ls`` on a prior build's
``OEBPS/text/`` directory for the exact filenames.

The override is applied AFTER the typography stage so that any
pipeline transformations (hyphen stitching, footnote-digit
normalization, ``qc_fixes.toml`` substitutions) are bypassed for that
chapter. The page-break ``<span epub:type="pagebreak" id="page-N">``
anchors and footnote markup must be preserved inside the override so
the ``nav.xhtml`` / ``content.opf`` cross-references continue to
resolve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OverrideReport:
    applied: list[str] = field(default_factory=list)
    stale: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        parts: list[str] = []
        if self.applied:
            parts.append(f"{len(self.applied)} chapter override(s) applied")
        if self.stale:
            parts.append(f"{len(self.stale)} stale override(s) skipped")
        return "; ".join(parts) if parts else "no chapter overrides"


def apply_chapter_overrides(
    rendered: list[tuple[str, str]],
    overrides_dir: Path | None,
) -> tuple[list[tuple[str, str]], OverrideReport]:
    """Replace rendered chapter XHTML with author-provided overrides.

    Returns ``(new_rendered, report)`` where ``new_rendered`` mirrors
    the input list with any overridden chapter slug's HTML swapped for
    the contents of ``overrides_dir/<slug>.xhtml``. The report records
    which slugs were applied and which override files were ignored
    because no rendered chapter shared their slug (typo / stale file).

    If ``overrides_dir`` is ``None`` or does not exist, the input
    ``rendered`` list is returned unchanged with an empty report.
    """
    rep = OverrideReport()
    if overrides_dir is None or not overrides_dir.is_dir():
        return list(rendered), rep

    overrides: dict[str, str] = {}
    for path in sorted(overrides_dir.glob("*.xhtml")):
        slug = path.stem
        overrides[slug] = path.read_text("utf-8")

    rendered_slugs = {slug for slug, _ in rendered}

    out: list[tuple[str, str]] = []
    for slug, html in rendered:
        if slug in overrides:
            out.append((slug, overrides[slug]))
            rep.applied.append(slug)
        else:
            out.append((slug, html))

    rep.stale = sorted(s for s in overrides if s not in rendered_slugs)
    return out, rep
