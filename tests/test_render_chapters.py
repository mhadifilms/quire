"""Tests for chapter assembly and XHTML rendering."""

from __future__ import annotations

from quire.render.chapters import (
    Chapter,
    assemble_chapters,
    detect_script_lang,
    render_chapter,
    slugify,
)
from quire.render.package import render_nav
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
        assert "Front Matter" not in titles
        assert "Chapter One" in titles
        assert "Chapter Two" in titles
    finally:
        configure_known_headings([])


def test_assemble_chapters_skips_configured_cover_page() -> None:
    class Cfg:
        cover_pdf_page = 1
        raw = {}

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


def test_assemble_chapters_keeps_cover_page_when_it_is_story_content() -> None:
    class Cfg:
        cover_pdf_page = 1
        raw = {"input": {"cover_is_content": True}}

    chapters = assemble_chapters(
        [{"printed_page": None}],
        [{
            "pno": 1,
            "elements": [{"kind": "paragraph", "text": "Story starts here.", "y": 50}],
        }],
        cfg=Cfg(),
    )

    assert [element["text"] for element in chapters[0].elements] == ["Story starts here."]


def test_auto_detected_book_drops_short_annotation_before_story() -> None:
    class Cfg:
        cover_pdf_page = 1
        raw = {"input": {"cover_is_content": True, "auto_detected": True}}

    configure_known_headings([("Sample Story", 1)])
    try:
        chapters = assemble_chapters(
            [{"printed_page": None}],
            [{
                "pno": 1,
                "elements": [
                    {"kind": "paragraph", "text": "Handwritten source citation", "y": 20},
                    {"kind": "heading", "level": 2, "text": "Sample Story", "y": 150},
                    {"kind": "paragraph", "text": "Story starts here.", "y": 220},
                ],
            }],
            cfg=Cfg(),
        )
    finally:
        configure_known_headings([])

    assert [chapter.title for chapter in chapters] == ["Sample Story"]
    assert chapters[0].elements[0]["text"] == "Story starts here."


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


def test_render_chapter_preserves_pagebreak_inside_merged_paragraph() -> None:
    chapter = Chapter(title="Hello", slug="ch-01", page_start=1)
    chapter.elements = [
        {
            "kind": "paragraph",
            "text": "First half second half.",
            "y": 100,
            "_pdf_pno": 1,
            "_printed": 3,
            "indent": False,
            "_continuation_pagebreaks": [
                {"offset": 11, "pdf_pno": 2, "printed": 4},
            ],
        },
    ]

    class Cfg:
        language = "en"

    xhtml, emitted = render_chapter(chapter, cfg=Cfg())

    assert 'id="page-4"' in xhtml
    assert "First half " in xhtml
    assert "second half." in xhtml
    assert emitted == {3, 4}


def test_navigation_omits_empty_page_list() -> None:
    class Cfg:
        language = "en"

    chapter = Chapter(title="Story", slug="story", page_start=1)
    nav = render_nav(Cfg(), [chapter], [])

    assert 'epub:type="toc"' in nav
    assert 'epub:type="page-list"' not in nav


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
