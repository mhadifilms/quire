"""Tests for the transliteration vocabulary helpers."""

from __future__ import annotations

import pytest

from quire.postprocess import vocabulary


@pytest.fixture(autouse=True)
def _reset_vocab():
    saved = dict(vocabulary.ARABIC_VOCAB)
    vocabulary.ARABIC_VOCAB.clear()
    yield
    vocabulary.ARABIC_VOCAB.clear()
    vocabulary.ARABIC_VOCAB.update(saved)


def test_normalize_translit_strips_diacritics_and_punct() -> None:
    norm = vocabulary._normalize_translit("Ḥajj-1")
    assert norm == "hajj"


def test_lookup_exact_and_al_prefix() -> None:
    vocabulary.ARABIC_VOCAB["hajj"] = "حَجّ"
    assert vocabulary.lookup("Hajj") == "حَجّ"
    assert vocabulary.lookup("Al-hajj") == "حَجّ"


def test_lookup_missing_returns_none() -> None:
    assert vocabulary.lookup("nonexistentterm") is None


def test_correct_ocr_transliteration_known_variant() -> None:
    vocabulary.ARABIC_VOCAB["miqat"] = "مِيقَات"
    assert vocabulary.correct_ocr_transliteration("Migat") == "Miqat"


def test_correct_ocr_transliteration_returns_none_for_real_word() -> None:
    vocabulary.ARABIC_VOCAB["miqat"] = "مِيقَات"
    # No phonetic similarity hop into vocab => None
    assert vocabulary.correct_ocr_transliteration("hello") is None


def test_update_with_pairs_does_not_overwrite() -> None:
    vocabulary.ARABIC_VOCAB["hajj"] = "حَجّ"
    vocabulary.update_with_pairs({"hajj": "WRONG"})
    assert vocabulary.ARABIC_VOCAB["hajj"] == "حَجّ"


def test_update_with_pairs_fills_new_keys() -> None:
    vocabulary.update_with_pairs({"umrah": "عُمْرَة"})
    assert vocabulary.ARABIC_VOCAB.get("umrah") == "عُمْرَة"
