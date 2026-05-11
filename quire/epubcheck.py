"""Optional EPUBCheck integration.

If the ``epubcheck`` executable (https://www.w3.org/publishing/epubcheck/) is
on ``PATH``, we run it on the built EPUB and surface its findings inside the
audit. EPUBCheck supports a ``--json <file>`` flag whose output we parse for
structured findings; we also fall back to reading stdout when JSON is not
available.

This module is intentionally side-effect-free and degrades to ``status =
"unavailable"`` when EPUBCheck is not installed.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

EPUBCHECK_EXECUTABLE_ENV = "QUIRE_EPUBCHECK"


def epubcheck_executable() -> str | None:
    """Return the resolved path to the EPUBCheck CLI, or ``None``."""
    env = os.environ.get(EPUBCHECK_EXECUTABLE_ENV)
    if env:
        return env if Path(env).exists() else None
    return shutil.which("epubcheck")


def run_epubcheck(epub_path: Path, *, timeout: float = 120.0) -> dict[str, Any]:
    """Run EPUBCheck and return a dict with ``status``, ``messages`` etc.

    ``status`` is one of ``"unavailable"``, ``"ok"``, ``"warn"``, ``"fail"``.
    ``messages`` is a list of ``{severity, message, location}`` dicts.
    """
    exe = epubcheck_executable()
    if not exe:
        return {
            "status": "unavailable",
            "messages": [],
            "reason": "epubcheck not on PATH or QUIRE_EPUBCHECK unset",
        }
    epub_path = Path(epub_path)
    if not epub_path.exists():
        return {"status": "fail", "messages": [{
            "severity": "FATAL",
            "message": f"EPUB not found: {epub_path}",
        }]}
    with tempfile.NamedTemporaryFile(
        prefix="quire_epubcheck_", suffix=".json", delete=False
    ) as tmp:
        json_path = Path(tmp.name)
    try:
        proc = subprocess.run(
            [exe, "--json", str(json_path), "--quiet", str(epub_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return _parse_result(json_path, proc.returncode, proc.stdout, proc.stderr)
    except subprocess.TimeoutExpired:
        return {"status": "fail", "messages": [{
            "severity": "FATAL",
            "message": f"epubcheck timed out after {timeout}s",
        }]}
    finally:
        try:
            json_path.unlink()
        except FileNotFoundError:
            pass


def _parse_result(json_path: Path, rc: int, stdout: str, stderr: str) -> dict[str, Any]:
    messages: list[dict[str, Any]] = []
    if json_path.exists() and json_path.stat().st_size > 0:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            for m in data.get("messages") or []:
                messages.append({
                    "severity": str(m.get("severity") or "INFO"),
                    "message": str(m.get("message") or ""),
                    "id": str(m.get("ID") or m.get("id") or ""),
                    "location": _stringify_locations(m.get("locations") or []),
                })
        except (json.JSONDecodeError, OSError):
            pass

    if rc == 0 and not any(m["severity"].upper() in {"ERROR", "FATAL"} for m in messages):
        status = "warn" if messages else "ok"
    else:
        status = "fail"

    return {
        "status": status,
        "messages": messages,
        "returncode": rc,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
    }


def _stringify_locations(locations: list[dict[str, Any]]) -> str:
    parts = []
    for loc in locations:
        fname = loc.get("fileName") or loc.get("path") or ""
        line = loc.get("line")
        col = loc.get("column")
        bits = [fname]
        if line is not None:
            bits.append(f"line {line}")
        if col is not None:
            bits.append(f"col {col}")
        parts.append(" ".join(b for b in bits if b))
    return "; ".join(parts)
