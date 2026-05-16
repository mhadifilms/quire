"""Per-page plain-text extraction from rendered chapter XHTML.

The post-render typography stage operates on the rendered XHTML where
footnote refs have been inlined as ``<sup>N</sup>`` and pagebreak
anchors have been embedded as ``<span epub:type="pagebreak" ... />``.
This module splits that XHTML on pagebreak anchors to produce one
plain-text payload per PDF page, suitable for feeding to a VLM.

``<sup>N</sup>`` runs are rewritten to the literal token ``[^N]`` so the
model understands the digit is a footnote marker, not real inline text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..render.typography import html_to_plain
from .models import PageText

# Matches the pagebreak anchors emitted by render/chapters.py and
# the simpler form used by render/export.py. We capture id and aria-label
# to map back to the PDF page index via the supplied page map.
_PAGEBREAK_RE = re.compile(
    r'<span[^>]*\bid="(?P<id>page-(?:pdf-)?[^"]+)"[^>]*></span>',
    re.IGNORECASE,
)

_SUP_RE = re.compile(r"<sup\b[^>]*>(?P<digits>\d{1,3})</sup>", re.IGNORECASE)
_NOTEREF_A_RE = re.compile(
    r'<a\b[^>]*epub:type="noteref"[^>]*>\s*<sup\b[^>]*>(?P<digits>\d{1,3})</sup>\s*</a>',
    re.IGNORECASE | re.S,
)


@dataclass(frozen=True)
class _Pagebreak:
    anchor_id: str
    start: int
    end: int


def _scan_pagebreaks(xhtml: str) -> list[_Pagebreak]:
    return [
        _Pagebreak(anchor_id=m.group("id"), start=m.start(), end=m.end())
        for m in _PAGEBREAK_RE.finditer(xhtml)
    ]


def _normalize_footnote_refs(html: str) -> str:
    """Rewrite footnote-ref ``<sup>N</sup>`` markers to literal ``[^N]``.

    This stops the model from interpreting bare superscript digits as
    OCR errors. The literal placeholder ``[^N]`` is unambiguous and
    matches how the rendered EPUB displays the marker.
    """
    html = _NOTEREF_A_RE.sub(lambda m: f"[^{m.group('digits')}]", html)
    html = _SUP_RE.sub(lambda m: f"[^{m.group('digits')}]", html)
    return html


def extract_page_texts(
    rendered: list[tuple[str, str]],
    page_map: list[tuple[int, int | None]],
) -> list[PageText]:
    """Split rendered XHTML into per-PDF-page plain text payloads.

    Parameters
    ----------
    rendered :
        ``[(chapter_slug, xhtml), ...]`` -- the output of ``render_chapter``
        before typography fixes are applied.
    page_map :
        ``[(pdf_pno, printed_or_None), ...]`` in the order the pagebreak
        anchors appear in the rendered XHTML across all chapters. Built
        from ``chapter.page_breaks`` for each chapter in chapter order.

    Returns one :class:`PageText` per pagebreak anchor in ``page_map``
    that contains at least one non-whitespace character.
    """
    flat_breaks: list[tuple[int, int, _Pagebreak, str]] = []
    for chap_idx, (_slug, xhtml) in enumerate(rendered):
        for pb in _scan_pagebreaks(xhtml):
            flat_breaks.append((chap_idx, pb.start, pb, xhtml))

    if len(flat_breaks) != len(page_map):
        # The caller passed a page map that doesn't match the rendered
        # XHTML. We still emit what we can pairwise; any extra entries
        # in either list are dropped.
        pass

    out: list[PageText] = []
    by_chapter: dict[int, list[tuple[_Pagebreak, str]]] = {}
    for chap_idx, _start, pb, xhtml in flat_breaks:
        by_chapter.setdefault(chap_idx, []).append((pb, xhtml))

    paired_idx = 0
    chap_indices = sorted(by_chapter.keys())
    for chap_idx in chap_indices:
        breaks = by_chapter[chap_idx]
        if not breaks:
            continue
        xhtml = breaks[0][1]
        for j, (pb, _x) in enumerate(breaks):
            slice_start = pb.end
            slice_end = breaks[j + 1][0].start if j + 1 < len(breaks) else len(xhtml)
            chunk = xhtml[slice_start:slice_end]
            chunk = _normalize_footnote_refs(chunk)
            plain, _posmap = html_to_plain(chunk)
            plain = _trim_text(plain)
            if paired_idx < len(page_map):
                pdf_pno, printed = page_map[paired_idx]
            else:
                pdf_pno = paired_idx + 1
                printed = None
            paired_idx += 1
            if not plain:
                continue
            out.append(PageText(pdf_pno=pdf_pno, printed=printed, plain_text=plain))
    return out


def _trim_text(s: str) -> str:
    s = re.sub(r"[\u00a0\u2007\u202f]+", " ", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def build_page_map(chapters) -> list[tuple[int, int | None]]:
    """Flatten ``chapter.page_breaks`` into a single ordered list.

    Skips empty chapters (those with no pagebreak anchors).
    """
    out: list[tuple[int, int | None]] = []
    for chap in chapters:
        for pdf_pno, printed in chap.page_breaks:
            out.append((pdf_pno, printed))
    return out
