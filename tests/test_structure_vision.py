"""Tests for Vision structure helpers."""

from __future__ import annotations

from quire.extract.pdf import parse_running_header
from quire.structure.pdf_based import _merge_drop_caps, merge_paragraphs, rejoin_text
from quire.structure.vision_based import (
    _is_bottom_page_number,
    _is_running_header,
    _looks_like_centered_title_credits_page,
    _prefer_fallback_ocr,
    _prefer_secondary_ocr,
    _rejoin_hyphenation,
    detect_spread_columns,
    merge_same_y_lines,
)


def _line(text: str, y: float, x0: float = 150.0, x1: float = 300.0) -> dict:
    return {"text": text, "x0": x0, "x1": x1, "y0": y, "y1": y + 10}


def test_pdf_text_layer_detects_title_case_running_headers() -> None:
    line = {
        "bbox": (140, 25, 260, 40),
        "spans": [{"text": "Something Wonderful"}],
    }
    assert parse_running_header(line, 5, 400) == ("Something Wonderful", None)


def test_pdf_text_layer_merges_drop_cap_and_preserves_compounds() -> None:
    cap = {
        "text": "L",
        "median_size": 38.0,
        "x0": 56,
        "y": 228,
        "y_bottom": 280,
        "spans": [{"text": "L", "size": 38.0, "bold": False, "italic": False}],
    }
    body = {
        "text": "ess than a year",
        "median_size": 11.8,
        "x0": 80,
        "y": 238,
        "y_bottom": 254,
        "spans": [{"text": "ess than a year", "size": 11.8}],
    }
    assert _merge_drop_caps([cap, body], 11.8)[0]["text"] == "Less than a year"
    assert rejoin_text(["sixty-", "eight years"]) == "sixty-eight years"
    assert rejoin_text(["maid-of-", "all-work"]) == "maid-of-all-work"


def test_pdf_text_layer_keeps_large_italic_section_label_separate() -> None:
    label = {
        "text": "Stewart",
        "median_size": 14.0,
        "x0": 58,
        "y": 100,
        "y_bottom": 115,
        "is_centered": False,
        "spans": [{"text": "Stewart", "italic": True, "bold": False}],
    }
    body = {
        "text": "Stewart Doig, master of the ship, came home.",
        "median_size": 11.75,
        "x0": 72,
        "y": 130,
        "y_bottom": 145,
        "is_centered": False,
        "spans": [{"text": "Stewart Doig, master of the ship, came home."}],
    }

    paragraphs = merge_paragraphs([label, body], 11.75, 396)

    assert len(paragraphs) == 2
    assert paragraphs[0]["heading"] is True


def test_detects_centered_title_credits_page() -> None:
    """A typical title/credits page: a few centered all-caps title lines
    followed by author / editor credits, none of which are flush-left."""
    lines = [
        _line("A SAMPLE", 100),
        _line("BOOK TITLE", 115),
        _line("HERE", 130),
        _line("Author One", 160),
        _line("Translated by", 190),
        _line("Translator A and Translator B", 205),
        _line("Edited by", 230),
        _line("Editor X, Editor Y and Editor Z", 245, 90, 360),
    ]
    assert _looks_like_centered_title_credits_page(lines, 450)


def test_rejects_table_of_contents_page() -> None:
    """A TOC is flush-left with trailing page numbers; it must not be
    misclassified as a title page."""
    lines = [
        _line("Table of Contents", 100),
        _line("Foreword 11", 130, 70, 140),
        _line("Chapter One: An Introduction 15", 145, 70, 290),
        _line("Chapter Two: Continuing On 25", 160, 70, 210),
    ]
    assert not _looks_like_centered_title_credits_page(lines, 450)


def test_detects_two_page_scan_before_same_y_merge() -> None:
    lines = []
    for index in range(12):
        y = 70 + index * 14
        lines.append(_line(f"Left page line {index}", y, 55, 330))
        lines.append(_line(f"Right page line {index}", y, 400, 685))
    lines.append(_line("Right line extends into gutter", 250, 370, 685))

    spread = detect_spread_columns(lines, page_width=745, page_height=590)

    assert spread is not None
    left, right, gutter = spread
    assert 330 < gutter < 400
    assert all("Left page" in line["text"] for line in left)
    assert all("Right" in line["text"] for line in right)
    assert len(merge_same_y_lines(left)) == 12
    assert len(merge_same_y_lines(right)) == 13


def test_does_not_split_single_column_landscape_page() -> None:
    lines = [
        _line(f"One wide body line {index}", 70 + index * 14, 80, 665)
        for index in range(20)
    ]
    assert detect_spread_columns(lines, page_width=745, page_height=590) is None


def test_bottom_page_number_detection_is_column_relative() -> None:
    assert _is_bottom_page_number(
        _line("158", 510, 235, 255),
        page_height=590,
        column_left=50,
        column_right=340,
    )
    assert not _is_bottom_page_number(
        _line("1944", 200, 235, 270),
        page_height=590,
        column_left=50,
        column_right=340,
    )
    assert _is_bottom_page_number(
        _line("77.", 510, 235, 255),
        page_height=590,
        column_left=50,
        column_right=340,
    )


def test_running_headers_support_portrait_and_leading_page_numbers() -> None:
    assert _is_running_header(
        _line("Something Wonderful", 30, 140, 260),
        page_width=400,
    )
    assert _is_running_header(
        _line("74 BIRDS OF PARADISE LOST", 30, 28, 165),
        page_width=792,
    )
    assert _is_running_header(_line("II9", 35, 720, 738), page_width=792)
    assert not _is_running_header(
        _line("Body prose starts here", 80, 400, 685),
        page_width=792,
    )
    assert not _is_running_header(
        _line('"What?" asked Kayden.', 56, 422, 535),
        page_width=792,
        page_height=612,
    )


def test_secondary_ocr_never_crosses_spread_gutter() -> None:
    primary = [
        {**_line("Left page prose", 100, 50, 330), "conf": 0.9},
        {**_line("Right page prose", 100, 400, 685), "conf": 0.9},
    ]
    secondary = [
        {**_line("Unrelated cleaner right text", 100, 400, 685), "conf": 0.9},
    ]

    result = _prefer_secondary_ocr(primary, secondary)

    assert result[0]["text"] == "Left page prose"


def test_fallback_ocr_repairs_clear_lexical_and_quote_damage() -> None:
    primary = [
        {**_line('"Goddamn it, he mutters.', 100, 50, 330), "conf": 1.0},
        {**_line("under the yellow cauTIoN ropes", 115, 50, 330), "conf": 1.0},
        {**_line("there's no salad left.?", 130, 50, 330), "conf": 1.0},
        {**_line("It was a black day when I first saw your father' They watched", 145, 50, 330), "conf": 1.0},
    ]
    fallback = [
        {**_line("“Goddamn it,” he mutters.", 100, 50, 330), "conf": 96.0},
        {**_line("under the yellow caution ropes", 115, 50, 330), "conf": 96.0},
        {**_line("there’s no salad left.’", 130, 50, 330), "conf": 96.0},
        {**_line("It was a black day when I first saw your father.’ They watched", 145, 50, 330), "conf": 96.0},
    ]

    result = _prefer_fallback_ocr(primary, fallback)

    assert result[0]["text"] == "“Goddamn it,” he mutters."
    assert result[1]["text"] == "under the yellow caution ropes"
    assert result[2]["text"] == "there’s no salad left.’"
    assert result[3]["text"] == "It was a black day when I first saw your father.’ They watched"


def test_fallback_ocr_repairs_systematic_closing_quote_confusion() -> None:
    primary = [
        {**_line("'We've run out of chapattis?", 100, 50, 330), "conf": 1.0},
        {**_line("why ...? He sat up", 115, 50, 330), "conf": 1.0},
    ]
    fallback = [
        {**_line("‘We’ve run out of chapattis.’", 100, 50, 330), "conf": 96.0},
        {**_line("why ...? He sat up", 115, 50, 330), "conf": 96.0},
    ]

    result = _prefer_fallback_ocr(primary, fallback)

    assert result[0]["text"] == "‘We’ve run out of chapattis.’"
    assert result[1]["text"] == "why ...? He sat up"


def test_fallback_ocr_recovers_omitted_drop_cap() -> None:
    primary = [
        {**_line("The died AFTER Mama came home", 100, 50, 330), "conf": 1.0},
    ]
    fallback = [
        {**_line("HE DAY AFTER Mama came home", 100, 50, 330), "conf": 96.0},
    ]

    result = _prefer_fallback_ocr(primary, fallback)

    assert result[0]["text"] == "THE DAY AFTER Mama came home"


def test_fallback_ocr_repairs_hyphenated_word_with_dictionary_evidence() -> None:
    primary = [
        {**_line("domes appear to be mor-", 100, 400, 680), "conf": 1.0},
        {**_line("ing, somehow, spinning", 115, 400, 680), "conf": 1.0},
    ]
    fallback = [
        {**_line("domes appear to be mov-", 100, 400, 680), "conf": 95.0},
        {**_line("ing, somehow, spinning", 115, 400, 680), "conf": 95.0},
    ]

    result = _prefer_fallback_ocr(primary, fallback)

    assert result[0]["text"].endswith("mov-")


def test_fallback_ocr_does_not_replace_with_adjacent_different_line() -> None:
    primary = [
        {**_line("Big Red couldn't remember who had", 100, 40, 330), "conf": 1.0},
    ]
    fallback = [
        {**_line("inside the shells?", 98, 30, 330), "conf": 96.0},
    ]

    assert _prefer_fallback_ocr(primary, fallback) == primary


def test_fallback_ocr_adds_high_confidence_line_omitted_by_primary() -> None:
    primary = [
        {**_line("at first. She had on a sack-shaped dress", 100, 400, 680), "conf": 1.0},
        {**_line("could have belonged to anyone", 130, 400, 680), "conf": 1.0},
    ]
    fallback = [
        {**_line("at first. She had on a sack-shaped dress", 100, 400, 680), "conf": 96.0},
        {
            **_line(
                "dress ordained for housework. Her veiny, brittle-looking shins",
                115,
                400,
                680,
            ),
            "conf": 95.0,
        },
        {**_line("could have belonged to anyone", 130, 400, 680), "conf": 96.0},
    ]

    result = _prefer_fallback_ocr(primary, fallback)

    assert [line["text"] for line in result] == [
        "at first. She had on a sack-shaped dress",
        "dress ordained for housework. Her veiny, brittle-looking shins",
        "could have belonged to anyone",
    ]


def test_rejoin_hyphenation_preserves_semantic_compound() -> None:
    assert _rejoin_hyphenation("a fourth- grade level") == "a fourth-grade level"
    assert _rejoin_hyphenation("preternatural dark- ness") == "preternatural darkness"
    assert _rejoin_hyphenation("it was sug gested in Eng lish") == "it was suggested in English"
    assert _rejoin_hyphenation("As he left, freeze Grand ma?") == "As he left, freeze Grand ma?"
