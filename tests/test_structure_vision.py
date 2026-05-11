"""Tests for Vision structure helpers."""

from __future__ import annotations

from quire.structure.vision_based import _looks_like_centered_title_credits_page


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
