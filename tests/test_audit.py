"""Tests for the audit module's helpers (link/text scanning, JSON shape)."""

from __future__ import annotations

import zipfile
from pathlib import Path

from quire.config import load_book_config
from quire.pipeline import build_book
from quire.render import audit as audit_mod


def test_collect_ids_and_links_parses_internal_refs() -> None:
    xhtml = b"""<?xml version='1.0' encoding='utf-8'?>
<html xmlns='http://www.w3.org/1999/xhtml'>
<body><p id='para-1'><a href='#para-1'>link</a></p></body>
</html>"""
    ids, links = audit_mod._collect_ids_and_links(xhtml)
    assert "para-1" in ids
    assert ("para-1", "_self") in links


def test_count_english_words_simple() -> None:
    xhtml = b"""<?xml version='1.0' encoding='utf-8'?>
<html xmlns='http://www.w3.org/1999/xhtml'>
<body><p>hello world</p><h2>chapter</h2></body>
</html>"""
    n = audit_mod._count_english_words(xhtml)
    assert n == 3


def test_run_audit_on_text_engine_build(text_engine_book: Path, tmp_path: Path) -> None:
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    build_book(cfg, formats=["epub"])
    result = audit_mod.run_audit(cfg, ocr_pages=[])
    assert "english_pct" in result
    assert "unresolved_links" in result
    assert cfg.audit_path.exists()
