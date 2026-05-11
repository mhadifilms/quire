"""Tests for stable book IDs, slug disambiguation, and OPF manifest sanity."""

from __future__ import annotations

import zipfile
from pathlib import Path

from quire.config import load_book_config
from quire.pipeline import build_book
from quire.render.chapters import _disambiguate_slug
from quire.render.package import _stable_book_id


def test_stable_book_id_is_deterministic(text_engine_book: Path, tmp_path: Path) -> None:
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    a = _stable_book_id(cfg)
    b = _stable_book_id(cfg)
    assert a == b
    # Roughly UUID-shaped
    assert len(a) == 36 and a.count("-") == 4


def test_stable_book_id_changes_with_pdf_content(text_engine_book: Path, tmp_path: Path) -> None:
    cfg1 = load_book_config(text_engine_book, repo_root=tmp_path)
    base = _stable_book_id(cfg1)
    # Mutate the source PDF: re-render with different content.
    import fitz
    doc = fitz.open()
    page = doc.new_page(width=420, height=600)
    page.insert_text((60, 60), "DIFFERENT BOOK", fontname="helv", fontsize=18)
    doc.save(str(cfg1.pdf_path))
    doc.close()
    cfg2 = load_book_config(text_engine_book, repo_root=tmp_path)
    assert _stable_book_id(cfg2) != base


def test_disambiguate_slug_suffixes_collisions() -> None:
    seen: set[str] = set()
    assert _disambiguate_slug(seen, "intro") == "intro"
    assert _disambiguate_slug(seen, "intro") == "intro-2"
    assert _disambiguate_slug(seen, "intro") == "intro-3"
    assert "intro-2" in seen and "intro-3" in seen


def test_epub_opf_consistency(text_engine_book: Path, tmp_path: Path) -> None:
    """Every manifest item exists in the zip; every spine itemref maps to a
    manifest item."""
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    build_book(cfg, formats=["epub"])
    import re
    with zipfile.ZipFile(cfg.epub_path) as zf:
        names = set(zf.namelist())
        opf = zf.read("OEBPS/content.opf").decode()
    manifest_ids = {m.group(1): m.group(2)
                    for m in re.finditer(r'<item id="([^"]+)" href="([^"]+)"', opf)}
    for _id, href in manifest_ids.items():
        assert f"OEBPS/{href}" in names, f"manifest href missing in zip: {href}"
    spine_refs = re.findall(r'<itemref idref="([^"]+)"', opf)
    for ref in spine_refs:
        assert ref in manifest_ids, f"spine ref has no manifest item: {ref}"
