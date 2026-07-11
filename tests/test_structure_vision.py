"""Tests for Vision structure helpers."""

from __future__ import annotations

from quire.structure.vision_based import (
    _is_bottom_page_number,
    _looks_like_centered_title_credits_page,
    _prefer_fallback_ocr,
    _prefer_secondary_ocr,
    _rejoin_hyphenation,
    detect_spread_columns,
    merge_same_y_lines,
)


def _line(text: str, y: float, x0: float = 150.0, x1: float = 300.0) -> dict:
    return {"text": text, "x0": x0, "x1": x1, "y0": y, "y1": y + 10}


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

    spread = detect_spread_columns(lines, page_width=745, page_height=590)

    assert spread is not None
    left, right, gutter = spread
    assert 330 < gutter < 400
    assert all("Left page" in line["text"] for line in left)
    assert all("Right page" in line["text"] for line in right)
    assert len(merge_same_y_lines(left)) == 12
    assert len(merge_same_y_lines(right)) == 12


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
    ]
    fallback = [
        {**_line("“Goddamn it,” he mutters.", 100, 50, 330), "conf": 96.0},
        {**_line("under the yellow caution ropes", 115, 50, 330), "conf": 96.0},
    ]

    result = _prefer_fallback_ocr(primary, fallback)

    assert result[0]["text"] == "“Goddamn it,” he mutters."
    assert result[1]["text"] == "under the yellow caution ropes"


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


def test_rejoin_hyphenation_preserves_semantic_compound() -> None:
    assert _rejoin_hyphenation("a fourth- grade level") == "a fourth-grade level"
    assert _rejoin_hyphenation("preternatural dark- ness") == "preternatural darkness"
