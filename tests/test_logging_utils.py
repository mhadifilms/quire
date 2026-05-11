"""Tests for the structured logging shim."""

from __future__ import annotations

import json

import pytest

from quire.logging_utils import log_event


def test_log_event_text(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.delenv("QUIRE_LOG", raising=False)
    log_event("ocr_done", pages=42, engine="text")
    out = capsys.readouterr().err
    assert "[quire] ocr_done" in out
    assert "pages=42" in out
    assert "engine=text" in out


def test_log_event_json(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setenv("QUIRE_LOG", "json")
    log_event("ocr_done", pages=42, engine="text")
    line = capsys.readouterr().err.strip()
    data = json.loads(line)
    assert data["event"] == "ocr_done"
    assert data["pages"] == 42
    assert data["engine"] == "text"
    assert "ts" in data
