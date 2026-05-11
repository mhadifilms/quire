"""Tests for ``quire.postprocess.canonical.quran`` lookup helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from quire.postprocess.canonical import quran


@pytest.fixture(autouse=True)
def _reset_corpus():
    """Restore default corpus path between tests."""
    default = quran.QURAN_TXT
    yield
    quran.set_corpus_path(default)


def test_normalize_surah_name() -> None:
    assert quran.normalize_surah_name("Al-Fatihah") == 1
    assert quran.normalize_surah_name("baqarah") == 2
    assert quran.normalize_surah_name("Aal-'Imran") == 3
    assert quran.normalize_surah_name("Ya-sin") == 36
    assert quran.normalize_surah_name("Not a Surah") is None


def test_find_citations_basic() -> None:
    text = "...as the verse Aal-'Imran 3:97 makes clear..."
    cits = quran.find_citations(text)
    assert any(c["surah"] == 3 and c["ayah"] == 97 for c in cits)


def test_find_citations_rejects_mismatched_name() -> None:
    # If the name doesn't map to the surah number, drop the match.
    text = "Fatihah 12:1 is wrong"
    cits = quran.find_citations(text)
    assert all(c["surah"] != 12 or c["ayah"] != 1 for c in cits)


def test_strip_diacritics_normalizes_alif() -> None:
    s = "ٱلْحَجُّ آدَمُ"
    out = quran._strip_diacritics(s)
    assert "ٱ" not in out
    assert "آ" not in out
    assert "ا" in out


def test_set_corpus_clears_cache(tmp_path: Path) -> None:
    corpus = tmp_path / "fake.txt"
    corpus.write_text("1|1|بسم الله الرحمن الرحيم\n", encoding="utf-8")
    quran.set_corpus_path(corpus)
    assert quran.get_verse(1, 1) == "بسم الله الرحمن الرحيم"
    # Replace, then set again — old cache must be discarded.
    corpus.write_text("1|1|تَجْرِبَة\n", encoding="utf-8")
    quran.set_corpus_path(corpus)
    assert quran.get_verse(1, 1) == "تَجْرِبَة"


def test_load_missing_corpus_returns_empty(tmp_path: Path) -> None:
    quran.set_corpus_path(tmp_path / "does-not-exist.txt")
    assert quran.get_verse(1, 1) is None
