"""Tests for zero-configuration PDF detection and mapping."""

from __future__ import annotations

from pathlib import Path

import fitz

from quire.auto import _filename_metadata, _prepare_workspace, detect_pdf


def _write_text_pdf(path: Path) -> None:
    doc = fitz.open()
    pages = [
        "Collected Stories\nAlex Writer",
        "Sample Story",
        " ".join(["This is reliable story prose."] * 55),
        " ".join(["The story continues with readable words."] * 45),
    ]
    for text in pages:
        page = doc.new_page(width=396, height=612)
        page.insert_textbox(
            fitz.Rect(40, 40, 356, 572),
            text,
            fontsize=10,
        )
    doc.save(path)
    doc.close()


def test_filename_metadata_maps_author_and_underscored_title() -> None:
    title, author = _filename_metadata(
        Path("Alex Writer, _Sample Story_ from Collected Stories.pdf")
    )
    assert title == "Sample Story"
    assert author == "Alex Writer"


def test_filename_metadata_recovers_author_from_numbered_scan() -> None:
    title, author = _filename_metadata(Path("Avery Stone 0001-1.pdf"))

    assert title == "Avery Stone 0001-1"
    assert author == "Avery Stone"


def test_detect_pdf_selects_text_engine_and_story_start(tmp_path: Path) -> None:
    source = tmp_path / "Alex Writer, _Sample Story_ from Collected Stories.pdf"
    _write_text_pdf(source)

    detected = detect_pdf(source)

    assert detected.title == "Sample Story"
    assert detected.author == "Alex Writer"
    assert detected.ocr_engine == "text"
    assert detected.content_start_page == 2


def test_prepare_workspace_trims_detected_frontmatter(tmp_path: Path) -> None:
    source = tmp_path / "Alex Writer, _Sample Story_ from Collected Stories.pdf"
    _write_text_pdf(source)
    detected = detect_pdf(source)

    cfg = _prepare_workspace(detected, tmp_path / "work")

    with fitz.open(cfg.pdf_path) as trimmed:
        assert trimmed.page_count == 3
        assert trimmed[0].get_text().strip() == "Sample Story"
    assert cfg.title == "Sample Story"
    assert cfg.author == "Alex Writer"
    assert cfg.structure_headings == [("Sample Story", 1)]
    assert cfg.postprocess_plugins == ["common_ocr"]
