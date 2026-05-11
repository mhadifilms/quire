"""Tests for ``quire.config``: loading, validation, and resolution rules."""

from __future__ import annotations

from pathlib import Path

import pytest

from quire.config import BookConfig, find_book_dir, load_book_config


def test_load_minimal_book(text_engine_book: Path, tmp_path: Path) -> None:
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    assert isinstance(cfg, BookConfig)
    assert cfg.slug == "tiny"
    assert cfg.title == "Tiny Book"
    assert cfg.author == "Test Author"
    assert cfg.language == "en"
    assert cfg.pdf_path.exists()
    assert cfg.artifact_dir == text_engine_book / "artifacts"
    assert cfg.caches_dir == text_engine_book / "artifacts" / "caches"
    assert cfg.ocr_engine == "text"
    assert cfg.postprocess_plugins == []
    assert {"epub", "html", "markdown", "text"} <= set(cfg.output_formats)


def test_missing_book_dir_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_book_config(tmp_path / "nope", repo_root=tmp_path)


def test_missing_pdf_raises(tmp_path: Path) -> None:
    book_dir = tmp_path / "books" / "broken"
    book_dir.mkdir(parents=True)
    (book_dir / "book.toml").write_text(
        '[book]\nslug = "broken"\ntitle = "X"\n[input]\npdf = "missing.pdf"\n',
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        load_book_config(book_dir, repo_root=tmp_path)


def test_invalid_format_raises(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = tmp_path / "books" / "bad-fmt"
    book_dir.mkdir(parents=True)
    import shutil
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        '[book]\nslug = "bad-fmt"\n[input]\npdf = "source.pdf"\n[render]\nformats = ["docx"]\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_book_config(book_dir, repo_root=tmp_path)


def test_headings_module_loaded(headings_book: Path, tmp_path: Path) -> None:
    cfg = load_book_config(headings_book, repo_root=tmp_path)
    assert ("Chapter One", 1) in cfg.structure_headings
    assert ("Section A", 2) in cfg.structure_headings


def test_find_book_dir_by_slug(text_engine_book: Path, tmp_path: Path) -> None:
    # Layout: tmp_path / books / tiny
    found = find_book_dir(tmp_path, "tiny")
    assert found == text_engine_book


def test_find_book_dir_by_path(text_engine_book: Path, tmp_path: Path) -> None:
    found = find_book_dir(tmp_path, str(text_engine_book))
    assert found == text_engine_book


def test_find_book_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        find_book_dir(tmp_path, "nonexistent")


def test_missing_font_lenient(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = tmp_path / "books" / "fontless"
    book_dir.mkdir(parents=True)
    import shutil
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "fontless"
[input]
pdf = "source.pdf"
[render]
embed_fonts = ["Nonexistent.ttf"]
""",
        encoding="utf-8",
    )
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    assert cfg.missing_fonts == ["Nonexistent.ttf"]
    assert cfg.embed_fonts == []
    assert any("font not found" in w for w in cfg.warnings)


def test_missing_font_strict_raises(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = tmp_path / "books" / "fontstrict"
    book_dir.mkdir(parents=True)
    import shutil
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "fontstrict"
[input]
pdf = "source.pdf"
[render]
strict_fonts = true
embed_fonts = ["Nonexistent.ttf"]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict_fonts"):
        load_book_config(book_dir, repo_root=tmp_path)
