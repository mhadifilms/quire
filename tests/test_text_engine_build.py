"""End-to-end test using the ``ocr.engine = 'text'`` path so we don't need
macOS Vision. Builds an EPUB + auxiliary formats from a synthetic PDF.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from quire.config import load_book_config
from quire.pipeline import build_book


def test_build_book_text_engine(text_engine_book: Path, tmp_path: Path) -> None:
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    outputs = build_book(cfg, formats=["epub", "html", "markdown", "text"])

    epub = outputs["epub"]
    assert epub.exists()
    assert epub.suffix == ".epub"
    assert epub.parent == text_engine_book / "artifacts"

    # EPUB is a ZIP with mimetype as first entry.
    with zipfile.ZipFile(epub) as zf:
        names = zf.namelist()
        assert names[0] == "mimetype"
        assert zf.read("mimetype") == b"application/epub+zip"
        assert any(n.endswith("content.opf") for n in names)
        assert any(n.endswith("nav.xhtml") for n in names)

    assert outputs["markdown"].exists()
    assert outputs["text"].exists()
    assert outputs["html"].exists()
