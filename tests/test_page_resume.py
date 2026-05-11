"""Per-page OCR cache resume."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from quire import pipeline
from quire.config import load_book_config


def _make_vision_book(tmp_path: Path, pdf: Path) -> Path:
    import shutil
    book_dir = tmp_path / "books" / "viz"
    book_dir.mkdir(parents=True)
    shutil.copy(pdf, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "viz"
title = "Viz"
[input]
pdf = "source.pdf"
[ocr]
engine = "vision"
workers = 1
[postprocess]
plugins = []
[render]
formats = ["epub"]
""",
        encoding="utf-8",
    )
    return book_dir


def test_partial_cache_rejected_without_retry(tiny_pdf_path: Path, tmp_path: Path) -> None:
    book_dir = _make_vision_book(tmp_path, tiny_pdf_path)
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    cfg.ensure_artifact_dirs()
    cache_path = cfg.caches_dir / "vision_ocr.pkl"
    meta = pipeline._cache_meta(cfg, "vision_ocr", extra={
        "workers": cfg.ocr_workers,
        "scale": cfg.ocr_dpi_scale,
    })
    # Cache with one good page + one errored page.
    good_page = {
        "pno": 1, "en_lines": [], "ar_lines": [],
        "arabic_blocks": [], "page_size_pt": (612, 792),
    }
    bad_page = {**good_page, "pno": 2, "error": "synthetic failure"}
    pipeline._write_cache(cache_path, meta, [good_page, bad_page])
    # Confirm the cache shape was written correctly.
    payload = pickle.loads(cache_path.read_bytes())
    assert payload["pages"][1]["error"] == "synthetic failure"
    with pytest.raises(RuntimeError, match="retry-failed"):
        pipeline._run_vision_ocr(cfg, force=False, retry_failed=False)


def test_retry_failed_only_reocrs_bad_pages(
    tiny_pdf_path: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    book_dir = _make_vision_book(tmp_path, tiny_pdf_path)
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    cfg.ensure_artifact_dirs()
    cache_path = cfg.caches_dir / "vision_ocr.pkl"
    meta = pipeline._cache_meta(cfg, "vision_ocr", extra={
        "workers": cfg.ocr_workers,
        "scale": cfg.ocr_dpi_scale,
    })
    good_page = {
        "pno": 1, "en_lines": [], "ar_lines": [],
        "arabic_blocks": [], "page_size_pt": (612, 792), "marker": "kept-from-cache",
    }
    bad_page = {**good_page, "pno": 2, "marker": "would-be-discarded",
                "error": "synthetic"}
    pipeline._write_cache(cache_path, meta, [good_page, bad_page])

    # Stub out ocr_all so we can assert it's invoked with only the failed pno.
    called_with: dict = {}

    def fake_ocr_all(pdf_path, **kwargs):
        called_with.update(kwargs)
        # Real ``ocr_all`` returns None placeholders for pages outside
        # ``page_numbers``; mirror that here.
        return [
            None,
            {"pno": 2, "marker": "freshly-ocred",
             "en_lines": [], "ar_lines": [], "arabic_blocks": [],
             "page_size_pt": (612, 792)},
        ]

    monkeypatch.setattr("quire.extract.ocr.ocr_all", fake_ocr_all)

    pages = pipeline._run_vision_ocr(cfg, force=False, retry_failed=True)
    assert called_with["page_numbers"] == [2]
    # Good page kept (marker preserved), bad page replaced.
    assert pages[0]["marker"] == "kept-from-cache"
    assert pages[1]["marker"] == "freshly-ocred"
    # Cache rewritten.
    payload = pickle.loads(cache_path.read_bytes())
    assert payload["pages"][1]["marker"] == "freshly-ocred"
    assert "error" not in payload["pages"][1]
