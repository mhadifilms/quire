"""Tests for the shared script utilities."""

from __future__ import annotations

from quire.script_utils import css_class_for, detect_script, is_rtl


def test_detect_script_arabic() -> None:
    assert detect_script("الكتاب") == "ar"


def test_detect_script_persian() -> None:
    assert detect_script("کتاب گفت") == "fa"


def test_detect_script_urdu() -> None:
    assert detect_script("یہ اردو ہے") == "ur"


def test_detect_script_hebrew() -> None:
    assert detect_script("שלום") == "he"


def test_detect_script_greek() -> None:
    assert detect_script("Καλημέρα") == "el"


def test_detect_script_cjk() -> None:
    assert detect_script("你好世界") == "zh"


def test_detect_script_none_for_english() -> None:
    assert detect_script("Hello world") is None


def test_is_rtl_arabic() -> None:
    assert is_rtl("ar")
    assert is_rtl("fa")
    assert is_rtl("he")
    assert is_rtl("ur")
    assert is_rtl("ar-EG")


def test_is_rtl_english() -> None:
    assert not is_rtl("en")
    assert not is_rtl("en-US")
    assert not is_rtl(None)


def test_css_class_for() -> None:
    assert css_class_for("ar") == "arabic"
    assert css_class_for("fa") == "persian"
    assert css_class_for("ur") == "urdu"
    assert css_class_for("he") == "hebrew"
    assert css_class_for("en") is None
    assert css_class_for(None) is None
