"""Tests for the Markdown / text / HTML exporters."""

from __future__ import annotations

from quire.render.chapters import Chapter
from quire.render.export import (
    _apply_export_qc,
    render_html,
    render_markdown,
    render_text,
)


class _Cfg:
    title = "Test Book"
    author = "Author"
    language = "en"


def _make_chapter() -> Chapter:
    chapter = Chapter(title="Hello", slug="ch-01", page_start=1)
    chapter.elements = [
        {"kind": "heading", "level": 2, "text": "Section", "y": 50,
         "_pdf_pno": 1, "_printed": 1, "indent": False},
        {"kind": "paragraph", "text": "Body para.", "y": 80,
         "_pdf_pno": 1, "_printed": 1, "indent": False},
        {"kind": "arabic", "text": "السلام", "y": 120,
         "_pdf_pno": 1, "_printed": 1, "is_quran": False, "conf": 90},
    ]
    chapter.footnotes = [
        {"number": "1", "text": "Note.", "y": 700, "_pdf_pno": 1, "_printed": 1},
    ]
    return chapter


def test_render_markdown_contains_title_and_chapter() -> None:
    out = render_markdown(_Cfg(), [_make_chapter()])
    assert out.startswith("# Test Book")
    assert "## Hello" in out
    assert "Body para." in out
    assert "السلام" in out
    assert "[^fn-" in out  # footnote ref


def test_render_text_plain() -> None:
    out = render_text(_Cfg(), [_make_chapter()])
    assert "Hello" in out
    assert "=====" in out  # title underline
    assert "Body para." in out
    assert "[1] Note." in out


def test_render_html_includes_aside() -> None:
    out = render_html(_Cfg(), [_make_chapter()])
    assert "<!doctype html>" in out
    assert "<section" in out
    assert 'class="footnotes"' in out
    assert "Body para." in out


def test_export_qc_updates_text_and_removes_empty_chapter() -> None:
    chapter = _make_chapter()
    front = Chapter(title="Front Matter", slug="front", page_start=1)
    front.elements = [
        {"kind": "paragraph", "text": "OCR noise", "_pdf_pno": 1},
    ]

    corrected = _apply_export_qc(
        [front, chapter],
        {"OCR noise": "", "Body para.": "Corrected body."},
    )

    assert [item.title for item in corrected] == ["Hello"]
    assert corrected[0].elements[1]["text"] == "Corrected body."


def test_export_qc_applies_generic_typography_repairs() -> None:
    chapter = _make_chapter()
    chapter.elements[1]["text"] = "'I like him' Dina said."

    corrected = _apply_export_qc([chapter], {})

    assert corrected[0].elements[1]["text"] == "'I like him.' Dina said."
