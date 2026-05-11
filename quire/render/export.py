"""Standalone HTML, Markdown, and plain-text exporters.

These exporters consume the same chapter model used by the EPUB packager. That
keeps OCR cleanup, paragraph reconstruction, footnotes, and canonical-script
post-processing in one shared pipeline instead of format-specific branches.
"""

from __future__ import annotations

import html
import re
from pathlib import Path

from ..config import BookConfig
from ..io_utils import atomic_write_text
from ..structure.pdf_based import PLACEHOLDER_FN
from .chapters import (
    _EXPLICIT_NOTEREF_RE,
    Chapter,
    _convert_format_markers,
    _normalize_noisy_noteref_markers,
    slugify,
)
from .package import CSS_BODY


def _strip_control_markers(text: str) -> str:
    text = re.sub(r"[\x02-\x07\x10-\x13]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _markdown_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _plain_inline(text: str, page_pno: int, available_fns: set[str]) -> str:
    occurrences: dict[tuple[int, str], int] = {}
    if available_fns:
        text = _normalize_noisy_noteref_markers(text, page_pno, available_fns, occurrences)
    text = _EXPLICIT_NOTEREF_RE.sub(lambda m: f"[{m.group(2)}]", text)
    text = re.sub(rf"{PLACEHOLDER_FN}([\d*]+){PLACEHOLDER_FN}", r"[\1]", text)
    return _strip_control_markers(text)


def _markdown_inline(
    text: str,
    page_pno: int,
    available_fns: set[str],
    note_ids: dict[tuple[int, str], str] | None = None,
) -> str:
    occurrences: dict[tuple[int, str], int] = {}
    if available_fns:
        text = _normalize_noisy_noteref_markers(text, page_pno, available_fns, occurrences)
    text = _EXPLICIT_NOTEREF_RE.sub(
        lambda m: f"[^{(note_ids or {}).get((int(m.group(1)), m.group(2)), f'note-{m.group(1)}-{m.group(2)}')}]",
        text,
    )
    text = re.sub(
        rf"{PLACEHOLDER_FN}([\d*]+){PLACEHOLDER_FN}",
        lambda m: f"[^{(note_ids or {}).get((page_pno, m.group(1)), f'note-{page_pno}-{m.group(1)}')}]",
        text,
    )
    replacements = [
        ("\x06", "***"), ("\x07", "***"),
        ("\x04", "**"), ("\x05", "**"),
        ("\x02", "*"), ("\x03", "*"),
        ("\x10", ""), ("\x11", ""),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return _markdown_escape(re.sub(r"\s+", " ", text).strip())


def _normal_html_inline(
    text: str,
    page_pno: int,
    available_fns: set[str],
    note_ids: dict[tuple[int, str], str] | None = None,
) -> str:
    occurrences: dict[tuple[int, str], int] = {}
    if available_fns:
        text = _normalize_noisy_noteref_markers(text, page_pno, available_fns, occurrences)
    text = html.escape(text, quote=False)
    text = _convert_format_markers(text)
    text = _EXPLICIT_NOTEREF_RE.sub(
        lambda m: (
            f'<sup><a href="#{(note_ids or {}).get((int(m.group(1)), m.group(2)), f"fn-src-{m.group(1)}-{m.group(2)}")}">'
            f'{html.escape(m.group(2))}</a></sup>'
        ),
        text,
    )
    return re.sub(
        rf"{PLACEHOLDER_FN}([\d*]+){PLACEHOLDER_FN}",
        lambda m: (
            f'<sup><a href="#{(note_ids or {}).get((page_pno, m.group(1)), f"fn-src-{page_pno}-{m.group(1)}")}">'
            f'{html.escape(m.group(1))}</a></sup>'
        ),
        text,
    )


def _footnotes_by_page(chapter: Chapter) -> dict[int, set[str]]:
    out: dict[int, set[str]] = {}
    for fn in chapter.footnotes:
        out.setdefault(fn["_pdf_pno"], set()).add(str(fn["number"]))
    return out


def _note_ids(chapter: Chapter) -> dict[tuple[int, str], str]:
    keys: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for fn in chapter.footnotes:
        key = (int(fn["_pdf_pno"]), str(fn["number"]))
        if key not in seen:
            keys.append(key)
            seen.add(key)
    return {
        key: f"fn-{chapter.slug}-note-{idx:03d}"
        for idx, key in enumerate(sorted(keys), start=1)
    }


def render_markdown(cfg: BookConfig, chapters: list[Chapter]) -> str:
    lines = [f"# {cfg.title}", "", f"*{cfg.author}*", ""]
    for chapter in chapters:
        if not chapter.elements and not chapter.footnotes:
            continue
        fn_by_page = _footnotes_by_page(chapter)
        note_ids = _note_ids(chapter)
        lines.extend([f"## {chapter.title}", ""])
        for elem in chapter.elements:
            pdf_pno = int(elem["_pdf_pno"])
            available = fn_by_page.get(pdf_pno, set())
            if elem["kind"] == "heading":
                level = max(3, min(6, int(elem.get("level", 3)) + 1))
                lines.extend([f"{'#' * level} {_markdown_inline(elem['text'], pdf_pno, available, note_ids)}", ""])
            elif elem["kind"] == "paragraph":
                lines.extend([_markdown_inline(elem["text"], pdf_pno, available, note_ids), ""])
            elif elem["kind"] == "arabic":
                lines.extend([elem["text"].strip(), ""])
        if chapter.footnotes:
            lines.extend(["### Notes", ""])
            seen: set[tuple[int, str]] = set()
            for fn in chapter.footnotes:
                key = (int(fn["_pdf_pno"]), str(fn["number"]))
                if key in seen:
                    continue
                seen.add(key)
                text = _markdown_inline(fn["text"], key[0], set(), note_ids)
                lines.append(f"[^{note_ids[key]}]: {text}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_text(cfg: BookConfig, chapters: list[Chapter]) -> str:
    lines = [cfg.title, cfg.author, ""]
    for chapter in chapters:
        if not chapter.elements and not chapter.footnotes:
            continue
        fn_by_page = _footnotes_by_page(chapter)
        lines.extend([chapter.title, "=" * len(chapter.title), ""])
        for elem in chapter.elements:
            pdf_pno = int(elem["_pdf_pno"])
            available = fn_by_page.get(pdf_pno, set())
            if elem["kind"] in {"heading", "paragraph"}:
                lines.extend([_plain_inline(elem["text"], pdf_pno, available), ""])
            elif elem["kind"] == "arabic":
                lines.extend([elem["text"].strip(), ""])
        if chapter.footnotes:
            lines.extend(["Notes", "-----"])
            seen: set[tuple[int, str]] = set()
            for fn in chapter.footnotes:
                key = (int(fn["_pdf_pno"]), str(fn["number"]))
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"[{key[1]}] {_plain_inline(fn['text'], key[0], set())}")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_chapter_html(chapter: Chapter) -> str:
    fn_by_page = _footnotes_by_page(chapter)
    note_ids = _note_ids(chapter)
    out = [f'<section id="{html.escape(chapter.slug)}">', f"<h1>{html.escape(chapter.title)}</h1>"]
    last_pno: int | None = None
    for elem in chapter.elements:
        pdf_pno = int(elem["_pdf_pno"])
        if pdf_pno != last_pno:
            printed = elem.get("_printed")
            label = printed if printed is not None else f"pdf-{pdf_pno}"
            out.append(f'<span class="pagebreak" id="page-{html.escape(str(label))}"></span>')
            last_pno = pdf_pno
        available = fn_by_page.get(pdf_pno, set())
        if elem["kind"] == "heading":
            level = max(2, min(6, int(elem.get("level", 3))))
            out.append(f"<h{level}>{_normal_html_inline(elem['text'], pdf_pno, available, note_ids)}</h{level}>")
        elif elem["kind"] == "paragraph":
            cls = "body"
            if elem.get("indent"):
                cls += " indent"
            if elem.get("centered"):
                cls += " center"
            out.append(f'<p class="{cls}">{_normal_html_inline(elem["text"], pdf_pno, available, note_ids)}</p>')
        elif elem["kind"] == "arabic":
            from ..script_utils import css_class_for, detect_script
            lang = elem.get("script_lang") or detect_script(elem["text"]) or "ar"
            cls = css_class_for(lang) or "arabic"
            out.append(f'<p class="{cls}" lang="{lang}" dir="rtl">{html.escape(elem["text"])}</p>')
    if chapter.footnotes:
        out.extend(['<aside class="footnotes">', "<h2>Notes</h2>"])
        seen: set[tuple[int, str]] = set()
        for fn in chapter.footnotes:
            key = (int(fn["_pdf_pno"]), str(fn["number"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(
                f'<p id="{note_ids[key]}" data-source-page="{key[0]}" data-note-number="{html.escape(key[1])}">'
                f'<strong>{html.escape(key[1])}.</strong> '
                f'{_normal_html_inline(fn["text"], key[0], set(), note_ids)}</p>'
            )
        out.append("</aside>")
    out.append("</section>")
    return "\n".join(out)


def render_html(cfg: BookConfig, chapters: list[Chapter]) -> str:
    body = "\n".join(_render_chapter_html(c) for c in chapters if c.elements or c.footnotes)
    title = html.escape(cfg.title)
    return f"""<!doctype html>
<html lang="{html.escape(cfg.language)}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
{CSS_BODY}
.pagebreak {{ display: none; }}
  </style>
</head>
<body>
  <main>
{body}
  </main>
</body>
</html>
"""


def write_exports(cfg: BookConfig, chapters: list[Chapter], formats: set[str]) -> dict[str, Path]:
    written: dict[str, Path] = {}
    if "markdown" in formats:
        atomic_write_text(cfg.markdown_path, render_markdown(cfg, chapters))
        written["markdown"] = cfg.markdown_path
    if "text" in formats:
        atomic_write_text(cfg.text_path, render_text(cfg, chapters))
        written["text"] = cfg.text_path
    if "html" in formats:
        cfg.html_dir.mkdir(parents=True, exist_ok=True)
        path = cfg.html_dir / f"{slugify(cfg.title)}.html"
        atomic_write_text(path, render_html(cfg, chapters))
        written["html"] = path
    return written
