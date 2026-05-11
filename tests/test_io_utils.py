"""Tests for ``quire.io_utils``."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from quire.io_utils import (
    atomic_write_bytes,
    atomic_write_text,
    content_fingerprint,
    file_lock,
)


def test_atomic_write_bytes_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "sub" / "x.bin"
    atomic_write_bytes(out, b"hello")
    assert out.read_bytes() == b"hello"


def test_atomic_write_text_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "x.txt"
    atomic_write_text(out, "hello\n")
    assert out.read_text() == "hello\n"


def test_atomic_write_no_leftover_tmp(tmp_path: Path) -> None:
    out = tmp_path / "x.bin"
    atomic_write_bytes(out, b"a")
    leftovers = [p for p in tmp_path.iterdir() if p.name != "x.bin"]
    assert leftovers == []


def test_content_fingerprint_stable(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world" * 1024)
    a = content_fingerprint(p)
    b = content_fingerprint(p)
    assert a == b


def test_content_fingerprint_changes_with_content(tmp_path: Path) -> None:
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello")
    a = content_fingerprint(p)
    p.write_bytes(b"world")
    b = content_fingerprint(p)
    assert a != b


def test_file_lock_is_exclusive(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"

    inside = threading.Event()
    release = threading.Event()
    second_acquired = threading.Event()

    def t1() -> None:
        with file_lock(lock):
            inside.set()
            release.wait(timeout=2)

    def t2() -> None:
        inside.wait(timeout=2)
        with file_lock(lock, timeout=2):
            second_acquired.set()

    a = threading.Thread(target=t1)
    b = threading.Thread(target=t2)
    a.start()
    b.start()
    # First thread holds the lock; ensure second doesn't acquire yet.
    inside.wait(timeout=2)
    time.sleep(0.2)
    assert not second_acquired.is_set()
    release.set()
    a.join(timeout=3)
    b.join(timeout=3)
    assert second_acquired.is_set()


def test_file_lock_timeout(tmp_path: Path) -> None:
    lock = tmp_path / "x.lock"
    with file_lock(lock):
        with pytest.raises(TimeoutError):
            with file_lock(lock, timeout=0.1):
                pass
