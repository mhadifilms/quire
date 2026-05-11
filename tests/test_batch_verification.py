"""End-to-end batch verification: text + RTL + failing config + skip.

The original audit spec asks for a small batch manifest covering at least one
text-layer PDF, one RTL/mixed-script book, one intentionally failing config,
and the runner should keep going past a single book's failure. We can't
exercise Vision OCR portably (macOS-only), so the RTL case uses the text
engine with an Arabic-only body — still routes through the script-detection
and RTL packaging branches without needing the OS framework.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import fitz
import pytest

from quire.batch import load_manifest, run_batch


def _write_book(
    root: Path,
    slug: str,
    body: str,
    *,
    language: str = "en",
    extra_render: str = "",
    bad_pdf: bool = False,
) -> Path:
    book_dir = root / "books" / slug
    book_dir.mkdir(parents=True)
    if bad_pdf:
        # intentionally missing PDF; book.toml will reference it
        pass
    else:
        pdf_path = book_dir / "source.pdf"
        doc = fitz.open()
        page = doc.new_page(width=420, height=600)
        page.insert_text((60, 60), body, fontname="helv", fontsize=14)
        page.insert_text((60, 200), "Body text for testing batch processing.",
                         fontname="helv", fontsize=11)
        doc.save(str(pdf_path))
        doc.close()
    (book_dir / "book.toml").write_text(
        f"""
[book]
slug = "{slug}"
title = "{slug}"
language = "{language}"

[input]
pdf = "source.pdf"

[ocr]
engine = "text"

[postprocess]
plugins = []

[render]
formats = ["epub"]
{extra_render}
""",
        encoding="utf-8",
    )
    return book_dir


def test_batch_verification(tmp_path: Path) -> None:
    repo_root = tmp_path
    # 1. Text-layer EN book.
    _write_book(repo_root, "verify-text", "Tiny Text Book")
    # 2. RTL/mixed-script book (language=ar so the renderer takes the RTL
    #    branch; text engine handles the PDF text layer either way).
    _write_book(repo_root, "verify-rtl", "كتاب صغير", language="ar")
    # 3. Intentionally failing config: PDF declared but missing.
    _write_book(repo_root, "verify-broken", "", bad_pdf=True)

    manifest = repo_root / "batch.toml"
    manifest.write_text(
        """
[[book]]
path = "books/verify-text"

[[book]]
path = "books/verify-rtl"

[[book]]
path = "books/verify-broken"
""",
        encoding="utf-8",
    )

    status_path = repo_root / "batch_status.json"
    entries = load_manifest(manifest)
    summary = run_batch(
        entries,
        workers=1,
        fail_fast=False,
        repo_root=repo_root,
        status_path=status_path,
    )
    assert summary["total"] == 3
    assert summary["ok"] == 2
    assert summary["failed"] == 1
    by_slug = {b["slug"]: b for b in summary["books"]}
    assert by_slug["verify-text"]["status"] == "ok"
    assert by_slug["verify-rtl"]["status"] == "ok"
    assert by_slug["verify-broken"]["status"] == "failed"
    # The status file should match the in-memory summary.
    on_disk = json.loads(status_path.read_text())
    assert on_disk["total"] == 3
    # Built EPUBs landed in each book's artifacts dir.
    assert (repo_root / "books" / "verify-text" / "artifacts" / "verify-text.epub").exists()
    assert (repo_root / "books" / "verify-rtl" / "artifacts" / "verify-rtl.epub").exists()


def test_batch_fail_fast_stops_after_first_failure(tmp_path: Path) -> None:
    _write_book(tmp_path, "broken1", "", bad_pdf=True)
    _write_book(tmp_path, "later-ok", "Tiny Text Book")
    manifest = tmp_path / "batch.toml"
    manifest.write_text(
        """
[[book]]
path = "books/broken1"

[[book]]
path = "books/later-ok"
""",
        encoding="utf-8",
    )
    entries = load_manifest(manifest)
    summary = run_batch(
        entries, workers=1, fail_fast=True, repo_root=tmp_path,
    )
    by_slug = {b["slug"]: b for b in summary["books"]}
    assert by_slug["broken1"]["status"] == "failed"
    later = (tmp_path / "books" / "later-ok" / "artifacts" / "later-ok.epub")
    assert not later.exists()
