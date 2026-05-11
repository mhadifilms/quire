"""Shared pytest fixtures.

We deliberately avoid relying on the local ``books/`` directory (it contains
gitignored inputs and generated outputs). Each fixture builds the minimum file
tree it needs in ``tmp_path``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture
def tiny_pdf_path(tmp_path: Path) -> Path:
    """Create a small but real PDF with a known text layer.

    Pages contain "Chapter One\nHello world.\n2" and "Hello again.\n3" so
    we have a header-like first line, a body paragraph, and a printed page
    number — enough to exercise the text-engine code path end-to-end.
    """
    import fitz  # PyMuPDF

    pdf_path = tmp_path / "tiny.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page(width=420, height=600)
        page.insert_text(
            (60, 60),
            "Tiny Book",
            fontname="helv",
            fontsize=18,
        )
        page.insert_text(
            (60, 120),
            f"Page {i + 1} body text. The quick brown fox jumps over the lazy dog.",
            fontname="helv",
            fontsize=12,
        )
        page.insert_text(
            (60, 560),
            str(i + 1),
            fontname="helv",
            fontsize=9,
        )
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def text_engine_book(tmp_path: Path, tiny_pdf_path: Path) -> Path:
    """A minimal book folder using engine = 'text' so tests run anywhere
    (no macOS Vision dependency).
    """
    book_dir = tmp_path / "books" / "tiny"
    book_dir.mkdir(parents=True)
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "tiny"
title = "Tiny Book"
author = "Test Author"
language = "en"

[input]
pdf = "source.pdf"

[ocr]
engine = "text"

[postprocess]
strict = false
plugins = []

[render]
formats = ["epub", "html", "markdown", "text"]
""",
        encoding="utf-8",
    )
    return book_dir


@pytest.fixture
def book_with_vocab(tmp_path: Path, tiny_pdf_path: Path) -> Path:
    """Book folder with a vocabulary.json file the vocabulary plugin can load."""
    book_dir = tmp_path / "books" / "with-vocab"
    book_dir.mkdir(parents=True)
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "vocabulary.json").write_text(
        json.dumps({"hajj": "حَجّ", "miqat": "مِيقَات"}),
        encoding="utf-8",
    )
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "with-vocab"
title = "With Vocab"
author = "Test"
language = "en"

[input]
pdf = "source.pdf"

[ocr]
engine = "text"

[postprocess]
plugins = ["vocabulary"]

[postprocess.vocabulary]
path = "vocabulary.json"

[render]
formats = ["epub"]
""",
        encoding="utf-8",
    )
    return book_dir


@pytest.fixture
def headings_book(tmp_path: Path, tiny_pdf_path: Path) -> Path:
    """Book folder with a headings.py module."""
    book_dir = tmp_path / "books" / "with-headings"
    book_dir.mkdir(parents=True)
    shutil.copy(tiny_pdf_path, book_dir / "source.pdf")
    (book_dir / "headings.py").write_text(
        'HEADINGS = [("Chapter One", 1), ("Section A", 2)]\n',
        encoding="utf-8",
    )
    (book_dir / "book.toml").write_text(
        """
[book]
slug = "with-headings"
title = "With Headings"
author = "Test"
language = "en"

[input]
pdf = "source.pdf"

[ocr]
engine = "text"

[structure]
headings_module = "headings"

[postprocess]
plugins = []

[render]
formats = ["epub"]
""",
        encoding="utf-8",
    )
    return book_dir
