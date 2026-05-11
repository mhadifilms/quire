"""Assemble per-page elements into chapters and emit XHTML.

Chapter splitting strategy:

  - Walk pages in order, building a flat element list with per-page boundaries.
  - When we encounter an h2 heading whose normalized title matches a known
    chapter title, start a new chapter. Sub-headings (h3) stay inside the
    current chapter.
  - Front matter (cover, title page, copyright, dedication) before the first
    chapter heading goes into a dedicated "Front Matter" chapter.
  - Back matter (Bibliography, Glossary, Indices, Romanization Table) are
    each chapters.

XHTML emission produces:
  - One XHTML per chapter (`OEBPS/text/ch-NN.xhtml`).
  - Pagebreak anchors with `epub:type="pagebreak"` and `role="doc-pagebreak"`
    inserted at the start of each printed page's content.
  - Footnotes rendered as EPUB3 `<aside epub:type="footnote">` after the body
    of the chapter, with `<a epub:type="noteref">` markers in body text.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from ..structure.pdf_based import (
    PLACEHOLDER_FN,
    _norm_heading,
    known_headings,
    normalize_known_title,
)


def _level1_titles_norm() -> dict[str, str]:
    return {
        _norm_heading(title).lower(): title
        for title, level in known_headings()
        if level == 1
    }


@dataclass
class Chapter:
    title: str
    slug: str
    page_start: int
    elements: list[dict] = field(default_factory=list)
    footnotes: list[dict] = field(default_factory=list)
    page_breaks: list[tuple[int, int | None]] = field(default_factory=list)
    # tuples of (pdf_pno, printed_pno_or_None) inserted as pagebreak anchors

    def add_pagebreak(self, pdf_pno: int, printed: int | None):
        self.page_breaks.append((pdf_pno, printed))


# --- chapter assembly -------------------------------------------------------


def slugify(text: str) -> str:
    text = re.sub(r"[\x02-\x07]", "", text)
    text = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return text or "section"


def _disambiguate_slug(seen: set[str], slug: str) -> str:
    """Return ``slug`` if not in ``seen``, else append ``-2``, ``-3``, ….

    The chapter slug doubles as the XHTML filename and OPF manifest id, so
    duplicates would produce an EPUB that fails epubcheck. Callers should
    pass a single mutable set; this helper updates it in place.
    """
    if slug not in seen:
        seen.add(slug)
        return slug
    n = 2
    while f"{slug}-{n}" in seen:
        n += 1
    final = f"{slug}-{n}"
    seen.add(final)
    return final


def assemble_chapters(
    pages_meta: list[dict],
    ocr_pages: list[dict],
    pdf_doc=None,
    *,
    cfg=None,
) -> list[Chapter]:
    """Build chapters from per-page element streams.

    Each ``ocr_pages[i]`` must have an ``elements`` list (as produced by the
    structure layer); ``pages_meta[i]`` carries ``pno`` and ``printed_page``.
    """
    chapters: list[Chapter] = []
    seen_slugs: set[str] = set()
    front = Chapter(title="Front Matter",
                    slug=_disambiguate_slug(seen_slugs, "front-matter"),
                    page_start=1)
    chapters.append(front)
    current = front

    for i, op in enumerate(ocr_pages):
        elements = op.get("elements", [])
        meta = pages_meta[i] if i < len(pages_meta) else {}
        pdf_pno = op.get("pno", i + 1)
        printed = meta.get("printed_page")
        if cfg is not None and pdf_pno == getattr(cfg, "cover_pdf_page", None):
            continue
        current.add_pagebreak(pdf_pno, printed)

        for elem in elements:
            if elem["kind"] == "heading":
                canonical = normalize_known_title(elem["text"])
                level1 = _level1_titles_norm()
                if canonical and canonical in level1.values():
                    title = canonical
                    slug = _disambiguate_slug(
                        seen_slugs, f"ch-{len(chapters):02d}-{slugify(title)}"
                    )
                    current = Chapter(title=title, slug=slug, page_start=pdf_pno)
                    chapters.append(current)
                    current.add_pagebreak(pdf_pno, printed)
                    continue
                if not level1 and elem.get("level", 3) <= 2 and current.elements:
                    title = elem["text"]
                    slug = _disambiguate_slug(
                        seen_slugs, f"ch-{len(chapters):02d}-{slugify(title)}"
                    )
                    current = Chapter(title=title, slug=slug, page_start=pdf_pno)
                    chapters.append(current)
                    current.add_pagebreak(pdf_pno, printed)
                    continue
            current.elements.append(dict(elem, _pdf_pno=pdf_pno, _printed=printed))

        for fn in [e for e in elements if e["kind"] == "footnote"]:
            current.footnotes.append({**fn, "_pdf_pno": pdf_pno, "_printed": printed})

    return chapters


# --- XHTML emission ---------------------------------------------------------

XHTML_HEAD = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{lang}" xml:lang="{lang}" dir="{dir}">
  <head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <link rel="stylesheet" type="text/css" href="../styles/book.css" />
  </head>
  <body epub:type="bodymatter" dir="{dir}">
"""

XHTML_TAIL = """  </body>\n</html>\n"""


def _convert_format_markers(s: str) -> str:
    def _convert(s, open_marker, close_marker, open_tag, close_tag):
        result = []
        i = 0
        while i < len(s):
            j = s.find(open_marker, i)
            if j < 0:
                result.append(s[i:])
                break
            result.append(s[i:j])
            k = s.find(close_marker, j + 1)
            if k < 0:
                result.append(s[j + 1:])
                break
            inner = s[j + 1:k]
            result.append(open_tag + inner + close_tag)
            i = k + 1
        return "".join(result)

    s = _convert(s, "\x02", "\x03", "<em>", "</em>")
    s = _convert(s, "\x04", "\x05", "<strong>", "</strong>")
    s = _convert(s, "\x06", "\x07", "<strong><em>", "</em></strong>")
    # Inline Arabic: \x10..\x11 wraps an Arabic span. The inner text is
    # already raw Arabic (no escaping needed) and gets a lang/dir wrapper.
    # Inline Arabic/Persian spans: detect script and tag accordingly.
    def _wrap_rtl(inner: str) -> str:
        lang = detect_script_lang(inner)
        cls = "persian-inline" if lang == "fa" else "arabic-inline"
        return (
            f'<span lang="{lang}" xml:lang="{lang}" dir="rtl" class="{cls}">'
            f"{inner}</span>"
        )

    out = []
    i = 0
    while i < len(s):
        nx = s.find("\x10", i)
        if nx < 0:
            out.append(s[i:])
            break
        out.append(s[i:nx])
        end = s.find("\x11", nx + 1)
        if end < 0:
            out.append(s[nx + 1:])
            break
        out.append(_wrap_rtl(s[nx + 1:end]))
        i = end + 1
    s = "".join(out)
    return s


def render_inline(
    text: str,
    page_pno: int,
    available_fns: set[str],
    fn_occurrences: dict[tuple[int, str], int],
    emitted_refs: set[tuple[int, str]] | None = None,
    valid_noterefs: set[tuple[int, str]] | None = None,
    note_ids: dict[tuple[int, str], str] | None = None,
    first_ref_ids: dict[tuple[int, str], str] | None = None,
) -> str:
    """Convert formatting markers and footnote placeholders into XHTML.

    Footnote hrefs use semantic chapter-local IDs when `note_ids` is provided.
    Source PDF page numbers remain internal lookup keys only; they are not part
    of user-facing text or the visible reading flow.

    If `emitted_refs` is provided, every (pdf, num) pair that produced a real
    `<a>` element is added so the footnote aside emitter can decide whether to
    include a back-link.
    """
    if available_fns:
        text = _normalize_noisy_noteref_markers(text, page_pno, available_fns, fn_occurrences)
    s = html.escape(text, quote=False)
    s = _convert_format_markers(s)
    s = _convert_explicit_noterefs(
        s,
        fn_occurrences,
        emitted_refs,
        valid_noterefs,
        note_ids,
        first_ref_ids,
    )

    def _fn_repl(m):
        num = m.group(1)
        if num in available_fns:
            key = (page_pno, num)
            fn_occurrences[key] = fn_occurrences.get(key, 0) + 1
            occ = fn_occurrences[key]
            note_id = (note_ids or {}).get(key, f"fn-src-{page_pno}-{num}")
            ref_id = f"{note_id}-ref-{occ}"
            if emitted_refs is not None:
                emitted_refs.add((page_pno, num))
            if first_ref_ids is not None:
                first_ref_ids.setdefault(key, ref_id)
            return (
                f'<a epub:type="noteref" role="doc-noteref" id="{ref_id}" '
                f'href="#{note_id}"><sup>{num}</sup></a>'
            )
        return f"<sup>{html.escape(num, quote=False)}</sup>"

    s = re.sub(rf"{PLACEHOLDER_FN}([\d*]+){PLACEHOLDER_FN}", _fn_repl, s)
    return s


# Format markers (\x02..\x07) are inserted by structure_page_vision to mark
# italic / bold / bold-italic runs *before* render_inline runs its noteref
# repair. Permit those markers — and the inline-Arabic wrappers \x10..\x11 —
# to appear between the body word and the misread marker punctuation, so
# patterns like ``<em>kaafir</em>?`` (where ? is a misread superscript) are
# still detected and converted to a footnote ref.
_FMT = r"[\x02-\x07\x10\x11]*"
# IMPORTANT: superscript footnote markers misread as ASCII punctuation
# typically sit directly after a word or a comma. Misreads after a
# sentence-terminating ``.``/``?``/``!`` are vanishingly rare — and those
# positions are precisely where legitimate closing single quotes live in
# books with quoted dialogue (e.g. ``reward.' With this sacrifice``). So
# the body intentionally allows a trailing ``,``/``;``/``:`` but NOT a
# trailing ``.``/``?``/``!`` — otherwise we'd corrupt every closing
# quote in the manuscript into a fake footnote ref.
_NOISY_NOTEREF_RE = re.compile(
    rf"(?P<body>[A-Za-z][A-Za-z\-]*(?:-[A-Za-z]+)?[,;:]?{_FMT})['’](?=(?:\s|$))"
)
# ``?`` / ``!`` after a period (``mustahab).!``) is almost always a misread
# superscript, NOT legitimate punctuation — books don't end a sentence with
# both ``.`` and ``!``. So we keep the broader body that allows ``.{1,3}``.
_QUESTION_NOTEREF_RE = re.compile(
    rf"(?P<body>[A-Za-z][A-Za-z\-]*(?:\))?(?:\.{{1,3}}|[,;:!]){_FMT})[?!](?=\s|$)"
)
_LOWERCASE_FLOW_NOTEREF_RE = re.compile(
    rf"(?P<body>[A-Za-z][A-Za-z\-]*{_FMT})\?(?=\s+[a-z])"
)
_EXPLICIT_NOTEREF_RE = re.compile(r"\x12(\d+):([\d*]+)\x13")


def _normalize_noisy_noteref_markers(
    text: str,
    page_pno: int,
    available_fns: set[str],
    fn_occurrences: dict[tuple[int, str], int],
) -> str:
    """Convert Vision's quote-shaped superscript markers into placeholders.

    Vision often reads tiny superscript note numbers as a trailing apostrophe.
    We only run this when the renderer knows the current page actually has
    footnotes, then map occurrences in order.
    """
    nums = sorted([n for n in available_fns if n.isdigit()], key=lambda n: int(n))
    if not nums:
        return text
    current_placeholders = set(
        re.findall(rf"{PLACEHOLDER_FN}([\d*]+){PLACEHOLDER_FN}", text)
    )
    used = {
        n for (p, n), count in fn_occurrences.items()
        if p == page_pno and count > 0
    } | current_placeholders
    remaining = [n for n in nums if n not in used]
    if not remaining:
        remaining = nums
    idx = 0

    def take_next() -> str | None:
        nonlocal idx
        if idx >= len(remaining):
            return None
        num = remaining[idx]
        idx += 1
        return num

    # Vision sometimes reads a tiny superscript as punctuation after a word.
    # Handle these before quote-shaped markers so this becomes the next unused
    # note, not a later closing quote.
    def q_repl(m: re.Match) -> str:
        num = take_next()
        if num is None:
            return m.group(0)
        return f"{m.group('body')}{PLACEHOLDER_FN}{num}{PLACEHOLDER_FN}"

    text = _QUESTION_NOTEREF_RE.sub(q_repl, text)
    text = _LOWERCASE_FLOW_NOTEREF_RE.sub(q_repl, text)

    def repl(m: re.Match) -> str:
        num = take_next()
        if num is None:
            return m.group(0)
        return f"{m.group('body')}{PLACEHOLDER_FN}{num}{PLACEHOLDER_FN}"

    return _NOISY_NOTEREF_RE.sub(repl, text)


def _convert_explicit_noterefs(
    s: str,
    fn_occurrences: dict[tuple[int, str], int],
    emitted_refs: set[tuple[int, str]] | None,
    valid_noterefs: set[tuple[int, str]] | None,
    note_ids: dict[tuple[int, str], str] | None = None,
    first_ref_ids: dict[tuple[int, str], str] | None = None,
) -> str:
    """Convert page-qualified noteref placeholders created by cross-page merge."""

    def repl(m: re.Match) -> str:
        pdf_pno, num = int(m.group(1)), m.group(2)
        key = (pdf_pno, num)
        if valid_noterefs is not None and key not in valid_noterefs:
            return ""
        fn_occurrences[key] = fn_occurrences.get(key, 0) + 1
        occ = fn_occurrences[key]
        note_id = (note_ids or {}).get(key, f"fn-src-{pdf_pno}-{num}")
        ref_id = f"{note_id}-ref-{occ}"
        if emitted_refs is not None:
            emitted_refs.add((pdf_pno, num))
        if first_ref_ids is not None:
            first_ref_ids.setdefault(key, ref_id)
        return (
            f'<a epub:type="noteref" role="doc-noteref" id="{ref_id}" '
            f'href="#{note_id}"><sup>{num}</sup></a>'
        )

    return _EXPLICIT_NOTEREF_RE.sub(repl, s)


def detect_script_lang(text: str) -> str:
    """Backward-compatible thin wrapper: prefer the shared script-utils.

    Returns ``"fa"`` for Persian, ``"ur"`` for Urdu, else ``"ar"``.
    """
    from ..script_utils import detect_script
    lang = detect_script(text)
    if lang in {"fa", "ur"}:
        return lang
    return "ar"


def render_arabic_block(
    text: str,
    is_quran: bool,
    conf: float,
    *,
    script_lang: str | None = None,
) -> str:
    """Render an Arabic block as a paragraph (or quote) with proper lang/dir.

    If ``script_lang`` is provided (e.g. set by the ``script_detect`` plugin),
    we trust it rather than re-detecting. Otherwise we fall back to detection.
    """
    from ..script_utils import css_class_for
    text = _normalize_arabic_punctuation(text, is_quran)
    lang = script_lang or detect_script_lang(text)
    lines = [html.escape(line, quote=False) for line in text.split("\n") if line.strip()]
    inner = '<br />'.join(lines)
    base_cls = css_class_for(lang) or "arabic"
    cls = f"{base_cls} quran" if is_quran else base_cls
    title_attr = f' title="OCR confidence {conf:.0f}"' if conf >= 0 else ""
    return (
        f'    <p class="{cls}" lang="{lang}" xml:lang="{lang}" dir="rtl"{title_attr}>{inner}</p>\n'
    )


def _normalize_arabic_punctuation(text: str, is_quran: bool) -> str:
    """Replace Vision OCR's ASCII brackets with proper Arabic decorative quotes.

    Quranic verses in the source PDF use decorative ornate brackets ﴾ ﴿ which
    Vision frequently renders as ``(`` ``)`` or ``{`` ``}``. We restore them
    for blocks classified as Quranic. Hadith quotes use ``«»`` which Vision
    captures correctly.
    """
    if "﴿" in text or "﴾" in text:
        return text
    if is_quran:
        # Common patterns Vision produces for the verse delimiters
        # Leading "(... " or "(. " becomes "﴿... "
        text = re.sub(r"^\s*\(\.\s*\.\.\s*", "﴿... ", text)
        text = re.sub(r"^\s*\(\.\s*", "﴿", text)
        text = re.sub(r"^\s*\(\s*\.\.\.", "﴿...", text)
        text = re.sub(r"^\s*\(", "﴿", text)
        # Trailing `)` or `}` becomes `﴾`
        text = re.sub(r"[\)\}]\s*$", "﴾", text)
        # Any internal `}` is most likely the wrong half-bracket
        text = text.replace("}", "﴾").replace("{", "﴿")
    return text


def _base_dir(language: str) -> str:
    return "rtl" if language.split("-", 1)[0].lower() in {"ar", "fa", "he", "ur"} else "ltr"


def render_chapter(
    chapter: Chapter,
    *,
    all_chapters: list[Chapter] | None = None,
    cfg=None,
) -> tuple[str, set[int]]:
    """Render a chapter to XHTML.

    Returns (xhtml_string, emitted_printed_pages).
    """
    _ = all_chapters  # reserved for future cross-chapter linking
    lang = html.escape(getattr(cfg, "language", "en") or "en", quote=False)
    out = [XHTML_HEAD.format(
        title=html.escape(chapter.title, quote=False),
        lang=lang,
        dir=_base_dir(lang),
    )]
    out.append(f'    <section epub:type="chapter" aria-labelledby="title-{chapter.slug}">\n')
    out.append(f'      <h1 id="title-{chapter.slug}">{html.escape(chapter.title, quote=False)}</h1>\n')

    # Index of footnotes available per (pdf_pno, num).
    fn_by_page: dict[int, set[str]] = {}
    fn_dedup: dict[tuple[int, str], dict] = {}
    for fn in chapter.footnotes:
        key = (fn["_pdf_pno"], fn["number"])
        if key in fn_dedup:
            continue
        fn_dedup[key] = fn
        fn_by_page.setdefault(fn["_pdf_pno"], set()).add(fn["number"])
    valid_noterefs = set(fn_dedup)
    note_ids = {
        key: f"fn-{chapter.slug}-note-{idx:03d}"
        for idx, key in enumerate(sorted(fn_dedup), start=1)
    }

    # Track inline footnote occurrences for unique IDs.
    fn_occurrences: dict[tuple[int, str], int] = {}
    emitted_refs: set[tuple[int, str]] = set()
    first_ref_ids: dict[tuple[int, str], str] = {}

    emitted_printed_pages: set[int] = set()
    last_pno: int | None = None
    for elem in chapter.elements:
        pdf_pno = elem["_pdf_pno"]
        printed = elem["_printed"]
        if pdf_pno != last_pno:
            if printed is not None:
                out.append(
                    f'      <span epub:type="pagebreak" role="doc-pagebreak" '
                    f'id="page-{printed}" aria-label="{printed}" title="page {printed}"></span>\n'
                )
                emitted_printed_pages.add(printed)
            else:
                out.append(
                    f'      <span epub:type="pagebreak" role="doc-pagebreak" '
                    f'id="page-pdf-{pdf_pno}" aria-label="pdf-{pdf_pno}" title="pdf page {pdf_pno}"></span>\n'
                )
            last_pno = pdf_pno

        if elem["kind"] == "heading":
            level = elem["level"]
            text = render_inline(
                elem["text"], pdf_pno, set(), fn_occurrences, emitted_refs,
                valid_noterefs, note_ids, first_ref_ids
            )
            out.append(f'      <h{level}>{text}</h{level}>\n')
        elif elem["kind"] == "paragraph":
            classes = ["body"]
            if elem.get("indent"):
                classes.append("indent")
            if elem.get("centered"):
                classes.append("center")
            # A paragraph that crosses a page boundary keeps the *first* page's
            # pno on its element. When that page has no detected footnote-list
            # (the footnotes live at the bottom of the *next* page), the
            # noteref repair would otherwise short-circuit and leave misread
            # superscript glyphs (``act?``) in place. So fall back to the
            # next page's footnotes if the current page doesn't have any.
            available = fn_by_page.get(pdf_pno, set())
            if not available:
                # Pull from the nearest forward page that has any footnotes.
                # We don't reach across chapter boundaries — fn_by_page is
                # already scoped to this chapter's pages.
                for ahead_pno in sorted(p for p in fn_by_page if p > pdf_pno):
                    available = fn_by_page[ahead_pno]
                    pdf_pno_for_ref = ahead_pno
                    break
                else:
                    pdf_pno_for_ref = pdf_pno
            else:
                pdf_pno_for_ref = pdf_pno
            text = render_inline(
                elem["text"], pdf_pno_for_ref, available, fn_occurrences, emitted_refs,
                valid_noterefs, note_ids, first_ref_ids
            )
            out.append(f'      <p class="{" ".join(classes)}">{text}</p>\n')
        elif elem["kind"] == "arabic":
            # If the block was already replaced by canonical Quran text, the
            # confidence is effectively 100 and the brackets are already correct.
            conf = 100.0 if elem.get("canonical") else elem.get("conf", -1)
            out.append(render_arabic_block(
                elem["text"],
                elem.get("is_quran", False),
                conf,
                script_lang=elem.get("script_lang"),
            ))

    out.append("    </section>\n")

    if fn_dedup:
        out.append('    <aside epub:type="footnotes" class="footnotes">\n')
        out.append('      <h2 class="footnotes-title">Notes</h2>\n')
        for (pdf_pno, num), fn in sorted(fn_dedup.items()):
            text = render_inline(fn["text"], pdf_pno, set(), {})
            if (pdf_pno, num) in emitted_refs:
                ref_id = first_ref_ids.get((pdf_pno, num), "")
                back = (
                    f' <a href="#{ref_id}" class="fn-back" '
                    f'role="doc-backlink" aria-label="Back to text">\u21a9</a>'
                ) if ref_id else ""
            else:
                back = ""
            out.append(
                f'      <aside epub:type="footnote" role="doc-footnote" '
                f'id="{note_ids[(pdf_pno, num)]}" data-source-page="{pdf_pno}" '
                f'data-note-number="{html.escape(num, quote=False)}">'
                f'<p><span class="fn-num">{num}.</span> {text}{back}</p></aside>\n'
            )
        out.append("    </aside>\n")

    out.append(XHTML_TAIL)
    return "".join(out), emitted_printed_pages
