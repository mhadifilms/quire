"""Tests for the batch runner."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from quire.batch import BatchEntry, load_manifest, run_batch


def _write_book(tmp_path: Path, slug: str, pdf: Path) -> Path:
    book_dir = tmp_path / "books" / slug
    book_dir.mkdir(parents=True)
    shutil.copy(pdf, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        f"""
[book]
slug = "{slug}"
title = "Book {slug}"
[input]
pdf = "source.pdf"
[ocr]
engine = "text"
[postprocess]
plugins = []
[render]
formats = ["epub"]
""",
        encoding="utf-8",
    )
    return book_dir


def test_load_manifest_parses_entries(tmp_path: Path) -> None:
    manifest = tmp_path / "batch.toml"
    manifest.write_text(
        """
[[book]]
path = "books/a"

[[book]]
path = "books/b"
formats = ["epub", "markdown"]
force_ocr = true
""",
        encoding="utf-8",
    )
    entries = load_manifest(manifest)
    assert len(entries) == 2
    assert entries[0].path == "books/a"
    assert entries[1].formats == ["epub", "markdown"]
    assert entries[1].force_ocr is True


def test_load_manifest_rejects_empty(tmp_path: Path) -> None:
    manifest = tmp_path / "empty.toml"
    manifest.write_text("# no books here\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(manifest)


def test_load_manifest_missing_path(tmp_path: Path) -> None:
    manifest = tmp_path / "bad.toml"
    manifest.write_text("[[book]]\ntitle = \"X\"\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(manifest)


def test_run_batch_sequential_ok(tmp_path: Path, tiny_pdf_path: Path,
                                 monkeypatch: pytest.MonkeyPatch) -> None:
    # Use tmp_path as the repo root so books/<slug> resolves correctly.
    monkeypatch.setattr("quire.batch.REPO_ROOT", tmp_path)
    _write_book(tmp_path, "alpha", tiny_pdf_path)
    _write_book(tmp_path, "beta", tiny_pdf_path)
    entries = [BatchEntry(path="alpha"), BatchEntry(path="beta")]
    status_path = tmp_path / "status.json"
    summary = run_batch(
        entries, workers=1, status_path=status_path, repo_root=tmp_path,
    )
    assert summary["total"] == 2
    assert summary["ok"] == 2
    assert summary["failed"] == 0
    data = json.loads(status_path.read_text())
    assert data["ok"] == 2
    assert {b["slug"] for b in data["books"]} == {"alpha", "beta"}


def test_run_batch_isolates_failures(tmp_path: Path, tiny_pdf_path: Path,
                                     monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quire.batch.REPO_ROOT", tmp_path)
    _write_book(tmp_path, "good", tiny_pdf_path)
    bad = tmp_path / "books" / "bad"
    bad.mkdir(parents=True)
    (bad / "book.toml").write_text(
        '[book]\nslug = "bad"\n[input]\npdf = "missing.pdf"\n',
        encoding="utf-8",
    )
    entries = [BatchEntry(path="good"), BatchEntry(path="bad")]
    summary = run_batch(entries, workers=1, repo_root=tmp_path)
    assert summary["ok"] == 1
    assert summary["failed"] == 1
    failed = [b for b in summary["books"] if b["status"] != "ok"]
    assert "missing.pdf" in failed[0]["error"]


def test_run_batch_fail_fast(tmp_path: Path, tiny_pdf_path: Path,
                             monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("quire.batch.REPO_ROOT", tmp_path)
    bad = tmp_path / "books" / "bad"
    bad.mkdir(parents=True)
    (bad / "book.toml").write_text(
        '[book]\nslug = "bad"\n[input]\npdf = "missing.pdf"\n',
        encoding="utf-8",
    )
    _write_book(tmp_path, "good", tiny_pdf_path)
    entries = [BatchEntry(path="bad"), BatchEntry(path="good")]
    summary = run_batch(entries, workers=1, fail_fast=True, repo_root=tmp_path)
    # With fail_fast, the second book should not have been built.
    assert summary["total"] == 1
    assert summary["failed"] == 1
