"""Tests for the CLI subcommands beyond `--help`."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from quire import cli


def test_check_command_succeeds(
    text_engine_book: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    # Point REPO_ROOT into our fixture tree.
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    rc = cli.main(["check", "tiny"])
    err = capsys.readouterr().err
    assert rc == 0
    assert "book: tiny" in err
    assert "OK" in err


def test_check_command_invalid_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    book_dir = tmp_path / "books" / "broken"
    book_dir.mkdir(parents=True)
    (book_dir / "book.toml").write_text(
        '[book]\nslug = "broken"\n[input]\npdf = "missing.pdf"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        cli.main(["check", "broken"])


def test_build_command_text_engine(
    text_engine_book: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    rc = cli.main(["build", "tiny", "--format", "epub"])
    assert rc == 0
    epub = text_engine_book / "artifacts" / "tiny.epub"
    assert epub.exists()


def test_build_workers_override(
    text_engine_book: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)
    # text engine ignores worker count for actual OCR but accepting the flag
    # confirms wiring.
    rc = cli.main(["build", "tiny", "--workers", "2", "--format", "epub"])
    assert rc == 0
