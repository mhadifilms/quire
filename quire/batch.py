"""Batch processing for many books in one command.

A batch manifest is a TOML file with one ``[[book]]`` entry per book:

    [[book]]
    path = "books/book-001"

    [[book]]
    path = "books/book-002"
    formats = ["epub", "markdown"]
    force_ocr = true

The runner builds each book in turn (or in parallel via process workers),
isolates per-book failures, and writes a JSON status manifest with the
per-book outcome. The CLI command is ``quire batch``.

Process-based workers are used so each book starts with a clean Python
interpreter — this sidesteps the historical module-level globals (vocabulary,
known headings, Quran corpus) that would otherwise bleed across books.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from .config import REPO_ROOT, find_book_dir, load_book_config
from .io_utils import atomic_write_text


@dataclass
class BatchEntry:
    """One book in a batch manifest."""

    path: str
    formats: list[str] | None = None
    force_ocr: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Outcome of one book's build."""

    slug: str
    path: str
    status: str
    elapsed_s: float
    outputs: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    traceback: str | None = None
    audit: dict[str, Any] | None = None


def load_manifest(path: Path) -> list[BatchEntry]:
    """Parse a batch manifest TOML and return :class:`BatchEntry` objects."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    raw = data.get("book") or []
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"manifest {path} has no [[book]] entries")
    out: list[BatchEntry] = []
    for entry in raw:
        if not isinstance(entry, dict):
            raise ValueError(f"manifest entry must be a table, got {type(entry).__name__}")
        if "path" not in entry:
            raise ValueError(f"manifest entry missing required 'path': {entry}")
        out.append(BatchEntry(
            path=str(entry["path"]),
            formats=list(entry["formats"]) if "formats" in entry else None,
            force_ocr=bool(entry.get("force_ocr", False)),
            extra={k: v for k, v in entry.items()
                   if k not in {"path", "formats", "force_ocr"}},
        ))
    return out


def _build_one(entry_dict: dict[str, Any], repo_root_str: str) -> dict[str, Any]:
    """Worker entry point. Imports the pipeline lazily so each process starts
    with a clean Python interpreter (and clean module-level globals)."""
    from .pipeline import build_book
    from .render.audit import run_audit

    started = time.monotonic()
    repo_root = Path(repo_root_str)
    entry = BatchEntry(**entry_dict)
    try:
        book_dir = find_book_dir(repo_root, entry.path)
        cfg = load_book_config(book_dir, repo_root=repo_root)
        cfg.ensure_artifact_dirs()
        outputs = build_book(cfg, force_ocr=entry.force_ocr, formats=entry.formats)
        audit_result = run_audit(cfg, ocr_pages=None)
        return BatchResult(
            slug=cfg.slug,
            path=str(book_dir),
            status="ok",
            elapsed_s=round(time.monotonic() - started, 2),
            outputs={k: str(v) for k, v in outputs.items()},
            audit=audit_result,
        ).__dict__
    except Exception as e:  # noqa: BLE001
        return BatchResult(
            slug=Path(entry.path).name,
            path=entry.path,
            status="failed",
            elapsed_s=round(time.monotonic() - started, 2),
            error=f"{type(e).__name__}: {e}",
            traceback=traceback.format_exc(),
        ).__dict__


def run_batch(
    entries: list[BatchEntry],
    *,
    workers: int = 1,
    status_path: Path | None = None,
    fail_fast: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Run every book in the manifest with bounded parallelism."""
    repo_root = (repo_root or REPO_ROOT).resolve()
    results: list[dict[str, Any]] = []
    started_at = time.time()

    workers = max(1, int(workers))
    entry_dicts = [{
        "path": e.path,
        "formats": e.formats,
        "force_ocr": e.force_ocr,
        "extra": e.extra,
    } for e in entries]

    if workers == 1:
        for ed in entry_dicts:
            result = _build_one(ed, str(repo_root))
            _log_result(result)
            results.append(result)
            if fail_fast and result["status"] != "ok":
                break
    else:
        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {
                pool.submit(_build_one, ed, str(repo_root)): ed
                for ed in entry_dicts
            }
            for fut in cf.as_completed(futs):
                result = fut.result()
                _log_result(result)
                results.append(result)
                if fail_fast and result["status"] != "ok":
                    for other in futs:
                        other.cancel()
                    break

    summary = {
        "started_at": started_at,
        "elapsed_s": round(time.time() - started_at, 2),
        "total": len(results),
        "ok": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] != "ok"),
        "books": results,
    }
    if status_path is not None:
        atomic_write_text(Path(status_path), json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def _log_result(result: dict[str, Any]) -> None:
    status = result["status"]
    slug = result.get("slug") or result.get("path")
    if status == "ok":
        print(f"[quire batch] OK    {slug} ({result['elapsed_s']}s)", file=sys.stderr)
    else:
        print(
            f"[quire batch] FAIL  {slug} ({result['elapsed_s']}s): {result.get('error')}",
            file=sys.stderr,
        )
