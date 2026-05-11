"""Tests for the OCR engine abstraction."""

from __future__ import annotations

from pathlib import Path

import pytest

from quire.extract.ocr_engine import (
    OCR_ENGINES,
    OCREngineOptions,
    TextLayerEngine,
    get_engine,
)


def test_get_engine_known_names() -> None:
    assert get_engine("text").name == "text"
    assert get_engine("pdf").name == "text"
    assert get_engine("vision").name == "vision"


def test_get_engine_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_engine("doesnt-exist")


def test_text_engine_returns_one_dict_per_page(tiny_pdf_path: Path) -> None:
    engine = TextLayerEngine()
    options = OCREngineOptions(
        languages=["en-US"], workers=1, dpi_scale=2, retries=0,
    )
    pages = engine.ocr_pdf(str(tiny_pdf_path), options)
    assert len(pages) == 2
    for p in pages:
        assert "pno" in p
        assert "page_size_pt" in p


def test_engine_registry_complete() -> None:
    # All supported engine names map to actual classes
    for name in ("text", "pdf", "pymupdf", "vision"):
        assert name in OCR_ENGINES
