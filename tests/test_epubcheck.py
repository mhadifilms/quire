"""Tests for the optional EPUBCheck integration."""

from __future__ import annotations

from pathlib import Path

import pytest

from quire import epubcheck


def test_epubcheck_unavailable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(epubcheck, "epubcheck_executable", lambda: None)
    out = epubcheck.run_epubcheck(tmp_path / "fake.epub")
    assert out["status"] == "unavailable"
    assert out["messages"] == []


def test_epubcheck_missing_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Pretend the binary exists but the EPUB doesn't.
    fake = tmp_path / "fake-epubcheck"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(0o755)
    monkeypatch.setattr(epubcheck, "epubcheck_executable", lambda: str(fake))
    out = epubcheck.run_epubcheck(tmp_path / "nope.epub")
    assert out["status"] == "fail"
    assert any("EPUB not found" in m["message"] for m in out["messages"])
