"""Tests for the Tesseract OCR engine.

Most tests exercise the pure-Python helpers (language mapping, dict→lines
conversion, script splitting) so they pass on CI without the binary.

The ``test_tesseract_full_pipeline`` test only runs when the local
``tesseract`` binary is on ``$PATH`` (skipped otherwise).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quire.extract.tesseract_engine import (
    TESSERACT_LANG_MAP,
    _group_words_to_lines,
    _is_rtl_lang_code,
    _normalize_langs,
    _split_langs_by_directionality,
)


def test_language_normalisation_bcp47_to_tesseract() -> None:
    assert _normalize_langs(["en-US"]) == ["eng"]
    assert _normalize_langs(["ar-SA", "en-US"]) == ["ara", "eng"]
    assert _normalize_langs(["fa-IR"]) == ["fas"]
    assert _normalize_langs(["ur-PK"]) == ["urd"]
    # Already a tesseract code: pass through.
    assert _normalize_langs(["ara"]) == ["ara"]
    # Unknown code: pass through (Tesseract will surface an error if needed).
    assert _normalize_langs(["xyz"]) == ["xyz"]
    # Empty defaults to English.
    assert _normalize_langs([]) == ["eng"]


def test_split_langs_by_directionality() -> None:
    ltr, rtl = _split_langs_by_directionality(["eng", "ara"])
    assert ltr == ["eng"] and rtl == ["ara"]
    # Order preserved within each group.
    ltr, rtl = _split_langs_by_directionality(["fra", "ara", "eng", "fas"])
    assert ltr == ["fra", "eng"] and rtl == ["ara", "fas"]
    # All-LTR list yields empty RTL.
    ltr, rtl = _split_langs_by_directionality(["eng", "deu"])
    assert ltr == ["eng", "deu"] and rtl == []
    # All-RTL list yields empty LTR.
    ltr, rtl = _split_langs_by_directionality(["ara", "heb"])
    assert ltr == [] and rtl == ["ara", "heb"]


def test_rtl_detection() -> None:
    assert _is_rtl_lang_code("ara")
    assert _is_rtl_lang_code("fas")
    assert _is_rtl_lang_code("urd")
    assert _is_rtl_lang_code("heb")
    assert not _is_rtl_lang_code("eng")
    assert not _is_rtl_lang_code("chi_sim")


def test_lang_map_covers_top_scripts() -> None:
    for tag, code in [("en", "eng"), ("ar", "ara"), ("fa", "fas"),
                       ("ur", "urd"), ("he", "heb"), ("zh", "chi_sim"),
                       ("ja", "jpn"), ("ko", "kor"), ("hi", "hin")]:
        assert TESSERACT_LANG_MAP.get(tag) == code, f"{tag} → {code}"


def test_group_words_to_lines_converts_pixels_to_points() -> None:
    # Two words on one line, one on a second line.
    data = {
        "text": ["Hello", "world", "Bye"],
        "conf": ["90.0", "88.5", "75.0"],
        "left": [100, 200, 100],
        "top":  [50, 50, 100],
        "width":[80, 70, 60],
        "height":[20, 20, 20],
        "page_num": [1, 1, 1],
        "block_num":[1, 1, 1],
        "par_num":  [1, 1, 1],
        "line_num": [1, 1, 2],
    }
    lines = _group_words_to_lines(data, scale=4)
    assert len(lines) == 2
    line1, line2 = lines
    assert line1["text"] == "Hello world"
    # x0 = 100 / 4 = 25, y0 = 50 / 4 = 12.5, x1 = (200+70)/4 = 67.5, y1 = (50+20)/4 = 17.5
    assert line1["x0"] == 25
    assert line1["y0"] == 12.5
    assert line1["x1"] == 67.5
    assert line1["y1"] == 17.5
    assert 88 <= line1["conf"] <= 91
    assert line2["text"] == "Bye"


def test_group_words_skips_negative_confidence() -> None:
    data = {
        "text": ["good", "junk"],
        "conf": ["80.0", "-1"],
        "left": [0, 0], "top": [0, 0],
        "width": [10, 10], "height": [10, 10],
        "page_num": [1, 1], "block_num": [1, 1], "par_num": [1, 1], "line_num": [1, 2],
    }
    lines = _group_words_to_lines(data, scale=1)
    assert [L["text"] for L in lines] == ["good"]


pytest_skip_no_binary = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract binary not found on PATH",
)


@pytest_skip_no_binary
def test_tesseract_ocr_on_real_pdf(tiny_pdf_path: Path) -> None:
    """Sanity check that the engine returns Vision-shaped dicts for a real PDF."""
    from quire.extract.tesseract_engine import ocr_pdf_tesseract

    pages = ocr_pdf_tesseract(str(tiny_pdf_path), languages=["en-US"], workers=1)
    assert len(pages) == 2
    for p in pages:
        assert p is not None
        assert "pno" in p and "page_size_pt" in p
        assert "en_lines" in p and "ar_lines" in p and "arabic_blocks" in p
        # The tiny PDF contains "Tiny Book" on each page.
        assert any("tiny" in L["text"].lower() or "book" in L["text"].lower()
                   for L in p["en_lines"])


@pytest_skip_no_binary
def test_tesseract_build_book_end_to_end(tmp_path: Path, tiny_pdf_path: Path) -> None:
    """Full pipeline: build an EPUB using the Tesseract engine."""
    from quire.config import load_book_config
    from quire.pipeline import build_book

    book_dir = tmp_path / "books" / "tess-book"
    book_dir.mkdir(parents=True)
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "tess-book"
title = "Tess Book"
[input]
pdf = "source.pdf"
[ocr]
engine = "tesseract"
languages = ["en-US"]
workers = 1
[postprocess]
plugins = []
[render]
formats = ["epub"]
""",
        encoding="utf-8",
    )
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    assert cfg.ocr_engine == "tesseract"
    outputs = build_book(cfg)
    assert "epub" in outputs
    assert outputs["epub"].exists()
    # Confirm the Tesseract cache landed where we expect (so resume works).
    assert (cfg.caches_dir / "tesseract_ocr.pkl").exists()


def test_arabic_refine_skipped_for_tesseract_engine(tmp_path: Path) -> None:
    """Regression: Vision-based Arabic refine must NOT run for Tesseract builds.

    When Vision overwrites Tesseract Arabic blocks with its own (narrower)
    re-OCR, it drops ~65 % of the recognized Arabic characters on this
    corpus — see ``_run_arabic_refine`` for the rationale. This test pins
    the behaviour: ``_run_arabic_refine`` returns ``None`` for any engine
    other than Vision, regardless of whether ``[postprocess.arabic_refine]``
    is configured.
    """
    from types import SimpleNamespace

    from quire.pipeline import _run_arabic_refine

    cfg = SimpleNamespace(
        ocr_engine="tesseract",
        caches_dir=tmp_path,
        plugin_config=lambda name: (
            {"languages": ["ar-SA"]} if name == "arabic_refine" else None
        ),
    )
    assert _run_arabic_refine(cfg, pages=[], force=False) is None

    cfg_vision_no_plugin = SimpleNamespace(
        ocr_engine="vision",
        caches_dir=tmp_path,
        plugin_config=lambda name: None,
    )
    assert _run_arabic_refine(cfg_vision_no_plugin, pages=[], force=False) is None


def test_tesseract_default_engine_for_new_books(tmp_path: Path, tiny_pdf_path: Path) -> None:
    """When [ocr].engine is omitted, new books should default to tesseract."""
    from quire.config import load_book_config

    book_dir = tmp_path / "books" / "default-engine"
    book_dir.mkdir(parents=True)
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "default-engine"
[input]
pdf = "source.pdf"
""",
        encoding="utf-8",
    )
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    assert cfg.ocr_engine == "tesseract"
