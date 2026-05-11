"""Tests for the noteref-misread repair in :mod:`quire.render.chapters`.

The repair converts OCR misreads of superscript footnote markers (``?``,
``!``, ``'``) into real footnote refs. Coverage focuses on:

  * Italic / format markers between the body word and the misread glyph
    don't block the repair (regression for the bug where ``<em>kaafir</em>?``
    was leaving ``?`` in the rendered output).
  * Legitimate closing single quotes after sentence-end punctuation
    (``reward.' With``) are NOT converted into noterefs.
  * ``?`` / ``!`` after sentence-end punctuation (``mustahab).!``) IS still
    converted — that combination is almost always a misread.
  * Cross-page paragraphs fall back to the next page's footnote list when
    the starting page has none of its own.
"""

from __future__ import annotations

from quire.render.chapters import _normalize_noisy_noteref_markers
from quire.structure.pdf_based import PLACEHOLDER_FN


def _repair(text: str, fns: set[str]) -> str:
    return _normalize_noisy_noteref_markers(
        text, page_pno=1, available_fns=fns, fn_occurrences={},
    )


def test_repair_question_marker_after_word_followed_by_lowercase() -> None:
    out = _repair("The kaafir? is a person", {"1", "2"})
    assert "kaafir?" not in out
    assert f"kaafir{PLACEHOLDER_FN}" in out


def test_repair_works_through_italic_markers() -> None:
    out = _repair("The \x02kaafir\x03? is a person", {"1"})
    assert "kaafir\x03?" not in out
    assert f"{PLACEHOLDER_FN}" in out


def test_repair_works_through_inline_arabic_wrapper() -> None:
    """A Latin word followed by an inline Arabic span and a misread ``?``
    still gets repaired — the wrapper chars are absorbed into the body
    capture."""
    out = _repair("the word \x10كاف\x11 means kafir\x11? is here", {"1"})
    # Body 'kafir\x11' captured; misread ? converted to placeholder.
    assert "kafir\x11?" not in out
    assert f"{PLACEHOLDER_FN}" in out


def test_repair_question_after_sentence_end_punctuation() -> None:
    out = _repair("recommended (mustahab).! For instance", {"1"})
    assert "mustahab).!" not in out
    assert f"{PLACEHOLDER_FN}" in out


def test_repair_question_after_dot_followed_by_capital() -> None:
    out = _repair("negation of ma-siwa-llah.? Ma-siwa is", {"1"})
    assert "ma-siwa-llah.?" not in out
    assert f"{PLACEHOLDER_FN}" in out


def test_repair_quote_after_letter() -> None:
    out = _repair("the pres' Ibn Babawayh", {"1"})
    assert "pres'" not in out
    assert f"{PLACEHOLDER_FN}" in out


def test_repair_quote_after_comma() -> None:
    out = _repair("ritual prayer,' fasting, paying khums,' or zakat is", {"1", "2"})
    assert "prayer,'" not in out
    assert "khums,'" not in out


def test_skip_legit_closing_quote_after_period() -> None:
    """``reward.' With`` is a legit closing quote, NOT a footnote marker."""
    text = "all be a noble reward.' With this sacrifice"
    out = _repair(text, {"1"})
    assert "reward.'" in out
    assert f"reward{PLACEHOLDER_FN}" not in out


def test_skip_legit_closing_quote_after_question_mark() -> None:
    """``Lord?' They`` is a legit quoted question, NOT a misread."""
    out = _repair('Was it ", "Am I not your Lord?\' They said: yes', {"1"})
    assert "Lord?'" in out
    assert f"Lord?{PLACEHOLDER_FN}" not in out


def test_skip_legit_closing_quote_after_bang() -> None:
    out = _repair("they shouted truth!' She walked away", {"1"})
    assert "truth!'" in out


def test_no_fns_no_repair() -> None:
    """When the page has no detected footnotes, leave text unchanged."""
    out = _repair("The kaafir? is a person", set())
    assert out == "The kaafir? is a person"


def test_cross_page_paragraph_uses_next_page_footnotes(tmp_path) -> None:
    """Integration: a paragraph whose ``?`` is on a footnote-less page should
    pick up the next page's footnote list via :func:`render_chapter`."""
    from quire.render.chapters import Chapter, render_chapter

    chapter = Chapter(title="Test", slug="ch-test", page_start=1)
    chapter.elements = [
        {
            "kind": "paragraph",
            "text": "This devotional act? is not a collective obligation.",
            "indent": False,
            "centered": False,
            "y": 100,
            "conf": 90,
            "_pdf_pno": 13,
            "_printed": 13,
        }
    ]
    chapter.footnotes = [
        {
            "number": "1",
            "text": "First footnote on the next page.",
            "_pdf_pno": 14,
            "_printed": 14,
            "y": 600,
        }
    ]

    class _Cfg:
        language = "en"

    xhtml, _ = render_chapter(chapter, all_chapters=[chapter], cfg=_Cfg())
    assert "act?" not in xhtml, "page-13 misread should have been repaired from page 14's footnote list"
    assert "epub:type=\"noteref\"" in xhtml
