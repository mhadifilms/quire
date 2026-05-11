"""Small I/O utilities used throughout the pipeline.

- ``atomic_write_bytes`` / ``atomic_write_text``: write-and-rename to avoid
  half-written files on crashes or concurrent batch runs.
- ``file_lock``: cooperative per-book lock so batch workers don't race.
- ``content_fingerprint``: cheap content-based fingerprint for cache keys.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import os
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically (write to a sibling tmp file
    in the same directory, then ``os.replace``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


@contextlib.contextmanager
def file_lock(
    path: Path,
    *,
    timeout: float | None = 30.0,
    poll_interval: float = 0.25,
) -> Iterator[None]:
    """Block until we acquire an exclusive lock on ``path``.

    Uses ``O_CREAT | O_EXCL`` for cross-platform portability (works on macOS
    Linux without ``fcntl``). The lock file holds the owning PID for debugging.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = None if timeout is None else time.monotonic() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(f"timed out waiting for lock: {path}") from None
            time.sleep(poll_interval)
    try:
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        fd = None
        yield
    finally:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(str(path))


def content_fingerprint(path: Path, *, sample_bytes: int = 4 * 1024 * 1024) -> str:
    """Return a stable content fingerprint for ``path``.

    Uses SHA-256 of a fixed-size sample (header + footer) plus the total
    file size. This is dramatically faster than hashing entire multi-GB
    PDFs while still being content-aware enough to defeat the
    mtime-changed-but-content-same problem.
    """
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(size.to_bytes(8, "big"))
    with open(p, "rb") as f:
        head = f.read(sample_bytes)
        h.update(head)
        if size > sample_bytes * 2:
            f.seek(max(0, size - sample_bytes))
            tail = f.read(sample_bytes)
            h.update(tail)
    return h.hexdigest()
