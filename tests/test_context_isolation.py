"""Two sequential builds should not bleed state between books."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from quire.config import load_book_config
from quire.context import reset_for_build
from quire.pipeline import build_book
from quire.postprocess import vocabulary as vocab_mod
from quire.structure import pdf_based


def _make_book(tmp_path: Path, slug: str, pdf: Path, *, headings_module: bool = False,
               vocab: dict | None = None) -> Path:
    book_dir = tmp_path / "books" / slug
    book_dir.mkdir(parents=True)
    shutil.copy(pdf, book_dir / "source.pdf")
    headings_block = ""
    if headings_module:
        (book_dir / "headings.py").write_text(
            'HEADINGS = [("Bespoke Heading", 1)]\n', encoding="utf-8"
        )
        headings_block = '[structure]\nheadings_module = "headings"\n'
    pp_block = "[postprocess]\nplugins = []\n"
    vocab_block = ""
    if vocab is not None:
        (book_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        pp_block = '[postprocess]\nplugins = ["vocabulary"]\n'
        vocab_block = '[postprocess.vocabulary]\npath = "vocab.json"\n'
    (book_dir / "book.toml").write_text(
        f"""
[book]
slug = "{slug}"
title = "{slug}"
[input]
pdf = "source.pdf"
[ocr]
engine = "text"
{headings_block}
{pp_block}
{vocab_block}
[render]
formats = ["epub"]
""",
        encoding="utf-8",
    )
    return book_dir


def test_sequential_builds_reset_globals(tmp_path: Path, tiny_pdf_path: Path) -> None:
    book_a = _make_book(tmp_path, "alpha", tiny_pdf_path, headings_module=True,
                        vocab={"alphaword": "α"})
    book_b = _make_book(tmp_path, "beta", tiny_pdf_path)

    cfg_a = load_book_config(book_a, repo_root=tmp_path)
    build_book(cfg_a, formats=["epub"])

    # After build A: vocabulary populated, known headings populated.
    assert "alphaword" in vocab_mod.ARABIC_VOCAB
    assert pdf_based.KNOWN_HEADINGS  # type: ignore[truthy-iterable]

    cfg_b = load_book_config(book_b, repo_root=tmp_path)
    build_book(cfg_b, formats=["epub"])

    # After build B: alpha state has been cleared, B has no headings/vocab.
    assert "alphaword" not in vocab_mod.ARABIC_VOCAB
    assert pdf_based.KNOWN_HEADINGS == []


def test_reset_for_build_returns_context(tmp_path: Path, text_engine_book: Path) -> None:
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    ctx = reset_for_build(cfg)
    assert ctx.cfg is cfg
    assert ctx.warnings == []
