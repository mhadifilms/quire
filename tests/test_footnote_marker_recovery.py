"""Regression tests for ``_strip_footnote_markers``.

The recovery converts OCR-misread footnote markers (``?``, ``!``, ``'``
in place of tiny superscript digits) into ``PLACEHOLDER_FN<n>PLACEHOLDER_FN``
tokens that the renderer can resolve to real ``<a epub:type="noteref">``
elements. Misclassifying a genuine end-of-sentence ``?`` as a footnote
marker would silently corrupt body text, so the cases below pin both the
positive and negative behaviours.
"""

from __future__ import annotations

import pytest

from quire.structure.pdf_based import PLACEHOLDER_FN
from quire.structure.vision_based import _strip_footnote_markers


@pytest.mark.parametrize(
    "text,expected",
    [
        # --- POSITIVE: should be recovered as footnote markers --------------
        ("Khabir.? while at the next", 1),
        ("miqat,! at Makkah", 1),
        ("walayah,? which is the divinely", 1),
        ("hanif:? then", 1),
        ("tawhid? and is the secure", 1),
        ("purification? in order", 1),
        ("jamah? 'the middle", 1),
        ("\x02jamrah\x03?", 1),
        ("\x02jamrah\x03? and", 1),
        ("Adam,' Regarding", 1),
        ("labbayk.3 followed", 1),
    ],
)
def test_recovers_misread_footnote_markers(text: str, expected: int) -> None:
    _, count = _strip_footnote_markers(text)
    assert count == expected, f"expected {expected} markers, got {count} in: {text!r}"


@pytest.mark.parametrize(
    "text",
    [
        # --- NEGATIVE: genuine punctuation that must pass through unchanged -
        "What? The next sentence.",
        "Is it real?",
        "He said: yes?",
        "Hello, world! How are you?",
        "He said: 'hello'",
        '"jamrah?"',
    ],
)
def test_preserves_genuine_punctuation(text: str) -> None:
    out, count = _strip_footnote_markers(text)
    assert count == 0, f"unexpectedly converted {count} markers in: {text!r}"
    assert out == text


def test_emits_sequential_placeholders() -> None:
    """Multiple markers in one paragraph get numbered 1, 2, 3…"""
    text = "First.? Then second,? and third tawhid? finally."
    out, count = _strip_footnote_markers(text)
    assert count == 3
    assert f"{PLACEHOLDER_FN}1{PLACEHOLDER_FN}" in out
    assert f"{PLACEHOLDER_FN}2{PLACEHOLDER_FN}" in out
    assert f"{PLACEHOLDER_FN}3{PLACEHOLDER_FN}" in out
