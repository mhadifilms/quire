"""The audit module should write a JSON sidecar with batch-ready metrics."""

from __future__ import annotations

import json
from pathlib import Path

from quire.config import load_book_config
from quire.pipeline import build_book
from quire.render import audit as audit_mod


def test_audit_writes_json_sidecar(text_engine_book: Path, tmp_path: Path) -> None:
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    build_book(cfg, formats=["epub"])
    # Skip EPUBCheck in this test; it may not be on the runner.
    audit_mod.run_audit(cfg, ocr_pages=[], run_epubcheck=False)
    assert cfg.audit_json_path.exists()
    data = json.loads(cfg.audit_json_path.read_text())
    assert data["slug"] == "tiny"
    assert "english_coverage_pct" in data
    assert "unresolved_links" in data
    assert "per_chapter" in data
    assert isinstance(data["per_chapter"], list)
    assert data["opf_issues"] == []


def test_audit_epubcheck_unavailable_status(text_engine_book: Path, tmp_path: Path,
                                            monkeypatch) -> None:
    from quire import epubcheck
    monkeypatch.setattr(epubcheck, "epubcheck_executable", lambda: None)
    cfg = load_book_config(text_engine_book, repo_root=tmp_path)
    build_book(cfg, formats=["epub"])
    result = audit_mod.run_audit(cfg, ocr_pages=[], run_epubcheck=True)
    data = json.loads(cfg.audit_json_path.read_text())
    assert data.get("epubcheck_status") == "unavailable"
    assert result.get("epubcheck_status") == "unavailable"
