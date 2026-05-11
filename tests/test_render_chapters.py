"""Tests for chapter assembly and XHTML rendering."""

from __future__ import annotations

from quire.render.chapters import (
    Chapter,
    assemble_chapters,
    detect_script_lang,
    render_chapter,
    slugify,
)
from quire.structure.pdf_based import configure_known_headings


def test_slugify_basic() -> None:
    assert slugify("Hello World!") == "hello-world"
    assert slugify("") == "section"


def test_slugify_unicode_fallback() -> None:
    # Currently strips non-ASCII; the slug should not be empty.
    out = slugify("مقدمة")
    assert out  # must produce a usable slug, not crash


def test_detect_script_lang_persian() -> None:
    assert detect_script_lang("کتاب") == "fa"


def test_detect_script_lang_arabic() -> None:
    assert detect_script_lang("الكتاب") == "ar"


def test_assemble_chapters_splits_on_known_heading() -> None:
    configure_known_headings([("Chapter One", 1), ("Chapter Two", 1)])
    try:
        pages_meta = [{"printed_page": 1}, {"printed_page": 2}]
        ocr_pages = [
            {
                "pno": 1,
                "elements": [
                    {"kind": "heading", "level": 2, "text": "Chapter One", "y": 50},
                    {"kind": "paragraph", "text": "Body one.", "y": 80},
                ],
            },
            {
                "pno": 2,
                "elements": [
                    {"kind": "heading", "level": 2, "text": "Chapter Two", "y": 50},
                    {"kind": "paragraph", "text": "Body two.", "y": 80},
                ],
            },
        ]
        chapters = assemble_chapters(pages_meta, ocr_pages)
        titles = [c.title for c in chapters]
        assert "Chapter One" in titles
        assert "Chapter Two" in titles
    finally:
        configure_known_headings([])


def test_assemble_chapters_skips_configured_cover_page() -> None:
    class Cfg:
        cover_pdf_page = 1

    pages_meta = [{"printed_page": 1}, {"printed_page": 2}]
    ocr_pages = [
        {
            "pno": 1,
            "elements": [{"kind": "paragraph", "text": "Cover duplicate", "y": 50}],
        },
        {
            "pno": 2,
            "elements": [{"kind": "paragraph", "text": "Real front matter", "y": 50}],
        },
    ]
    chapters = assemble_chapters(pages_meta, ocr_pages, cfg=Cfg())
    front_text = [e["text"] for e in chapters[0].elements]
    assert front_text == ["Real front matter"]


def test_render_chapter_emits_xhtml_skeleton() -> None:
    chapter = Chapter(title="Hello", slug="ch-01", page_start=1)
    chapter.elements = [
        {"kind": "paragraph", "text": "Hello world.", "y": 100,
         "_pdf_pno": 1, "_printed": 1, "indent": False},
    ]
    class Cfg:
        language = "en"
    xhtml, emitted = render_chapter(chapter, cfg=Cfg())
    assert "<?xml" in xhtml
    assert "Hello world." in xhtml
    assert "<h1" in xhtml
    assert 'lang="en"' in xhtml
    assert 1 in emitted


def test_render_chapter_with_footnote_creates_aside() -> None:
    chapter = Chapter(title="Notes", slug="ch-notes", page_start=1)
    chapter.elements = [
        {"kind": "paragraph", "text": "Body \u20201\u2020 text.", "y": 80,
         "_pdf_pno": 1, "_printed": 1, "indent": False},
    ]
    chapter.footnotes = [
        {"number": "1", "text": "First note.", "y": 700, "_pdf_pno": 1, "_printed": 1},
    ]
    class Cfg:
        language = "en"
    xhtml, _ = render_chapter(chapter, cfg=Cfg())
    assert 'epub:type="noteref"' in xhtml
    assert 'epub:type="footnote"' in xhtml
    assert "First note." in xhtml


def test_render_arabic_block_trusts_plugin_script_lang() -> None:
    from quire.render.chapters import render_arabic_block
    # Text contains only Arabic glyphs; plugin tags it Urdu.
    out = render_arabic_block("السلام", is_quran=False, conf=80,
                              script_lang="ur")
    assert 'lang="ur"' in out
    assert 'class="urdu"' in out


def test_render_arabic_block_falls_back_to_detection() -> None:
    from quire.render.chapters import render_arabic_block
    # No script_lang provided => persian glyph forces fa.
    out = render_arabic_block("کتاب", is_quran=False, conf=80)
    assert 'lang="fa"' in out


def test_render_chapter_rtl_language() -> None:
    chapter = Chapter(title="عنوان", slug="ch-ar", page_start=1)
    chapter.elements = [
        {"kind": "paragraph", "text": "محتوى", "y": 100,
         "_pdf_pno": 1, "_printed": 1, "indent": False},
    ]
    class Cfg:
        language = "ar"
    xhtml, _ = render_chapter(chapter, cfg=Cfg())
    assert 'dir="rtl"' in xhtml
