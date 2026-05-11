"""Tests for the opt-in ``[structure] book_heuristics`` flag."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quire.config import load_book_config


def _write_book(tmp_path: Path, tiny_pdf: Path, *, heuristics_block: str = "") -> Path:
    book_dir = tmp_path / "books" / "h"
    book_dir.mkdir(parents=True)
    shutil.copy(tiny_pdf, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        f"""
[book]
slug = "h"
[input]
pdf = "source.pdf"
[ocr]
engine = "text"
[structure]
{heuristics_block}
""",
        encoding="utf-8",
    )
    return book_dir


def test_default_book_heuristics_empty(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = _write_book(tmp_path, tiny_pdf_path)
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    assert cfg.book_heuristics == []


def test_book_heuristics_known_values_accepted(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = _write_book(
        tmp_path, tiny_pdf_path,
        heuristics_block='book_heuristics = ["imprint-fix"]',
    )
    cfg = load_book_config(book_dir, repo_root=tmp_path)
    assert cfg.book_heuristics == ["imprint-fix"]


def test_book_heuristics_unknown_value_rejected(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = _write_book(
        tmp_path, tiny_pdf_path,
        heuristics_block='book_heuristics = ["greek-frontmatter"]',
    )
    with pytest.raises(ValueError, match="unknown book_heuristics"):
        load_book_config(book_dir, repo_root=tmp_path)


def test_book_heuristics_wrong_type_rejected(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_dir = _write_book(
        tmp_path, tiny_pdf_path,
        heuristics_block='book_heuristics = "imprint-fix"',
    )
    with pytest.raises(ValueError, match="must be a list of strings"):
        load_book_config(book_dir, repo_root=tmp_path)


def test_clean_frontmatter_helpers_callable() -> None:
    """Sanity: the standalone imprint-cleanup helpers are still importable
    so the opt-in path in :func:`structure_page_vision` can dispatch to them.
    """
    from quire.structure.vision_based import _clean_frontmatter_line

    # The most generic of the imprint fixes: ASCII '@' -> '©'.
    assert _clean_frontmatter_line("Copyright @ 2024") == "Copyright © 2024"
