"""Shared OCR correction rules and audit logging.

The exact-fix rules are data-driven so they can be reused by every book:

* ``data/ocr/common_fixes.toml`` for repository-wide safe fixes.
* ``books/<slug>/ocr_fixes.toml`` for optional book-local additions.

Every automatic correction is recorded in-memory during a build and written to
``books/<slug>/artifacts/ocr_corrections.tsv`` by the pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from ..config import REPO_ROOT, BookConfig
from ..io_utils import atomic_write_text


@dataclass
class ExactFixes:
    phrase: dict[str, str]
    word: dict[str, str]


@dataclass
class Correction:
    source: str
    rule: str
    page: str
    kind: str
    y: str
    before: str
    after: str
    evidence: str = ""


_CORRECTIONS: list[Correction] = []
_REVIEW_ROWS: list[dict[str, str]] = []
_FIX_CACHE: dict[tuple[str, ...], ExactFixes] = {}


def reset_report() -> None:
    _CORRECTIONS.clear()
    _REVIEW_ROWS.clear()


def _read_fix_file(path: Path) -> ExactFixes:
    if not path.exists():
        return ExactFixes({}, {})
    with open(path, "rb") as f:
        data = tomllib.load(f)
    phrase = data.get("phrase", {})
    word = data.get("word", {})
    return ExactFixes(
        phrase={str(k): str(v) for k, v in phrase.items()},
        word={str(k): str(v) for k, v in word.items()},
    )


def _configured_fix_paths(cfg: Any) -> list[Path]:
    paths = [REPO_ROOT / "data" / "ocr" / "common_fixes.toml"]
    book_dir = getattr(cfg, "book_dir", None)
    if book_dir is not None:
        paths.append(Path(book_dir) / "ocr_fixes.toml")
    plugin_config = getattr(cfg, "plugin_config", None)
    settings = plugin_config("ocr_corrections") if callable(plugin_config) else {}
    extra = settings.get("path") if isinstance(settings, dict) else None
    if extra and book_dir is not None:
        paths.append((Path(book_dir) / str(extra)).resolve())
    return paths


def load_exact_fixes(cfg: Any) -> ExactFixes:
    paths = _configured_fix_paths(cfg)
    cache_key = tuple(str(p.resolve()) for p in paths)
    cached = _FIX_CACHE.get(cache_key)
    if cached is not None:
        return cached
    phrase: dict[str, str] = {}
    word: dict[str, str] = {}
    for path in paths:
        fixes = _read_fix_file(path)
        phrase.update(fixes.phrase)
        word.update(fixes.word)
    merged = ExactFixes(phrase=phrase, word=word)
    _FIX_CACHE[cache_key] = merged
    return merged


def _word_re(words: dict[str, str]) -> re.Pattern[str] | None:
    if not words:
        return None
    pattern = "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"\b(?:{pattern})\b")


def _clean_snippet(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:220]


def _context(text: str, start: int, end: int, repl: str) -> tuple[str, str]:
    left = max(0, start - 70)
    right = min(len(text), end + 70)
    before = text[left:right]
    after = text[left:start] + repl + text[end:right]
    return _clean_snippet(before), _clean_snippet(after)


def record_correction(
    cfg: Any,
    *,
    source: str,
    rule: str,
    page: Any = "",
    element: dict | None = None,
    before: str,
    after: str,
    evidence: str = "",
) -> None:
    if getattr(cfg, "artifact_dir", None) is None:
        return
    element = element or {}
    _CORRECTIONS.append(
        Correction(
            source=source,
            rule=rule,
            page=str(page or ""),
            kind=str(element.get("kind", "")),
            y=str(element.get("y", "")),
            before=_clean_snippet(before),
            after=_clean_snippet(after),
            evidence=evidence,
        )
    )


def record_review(
    cfg: Any,
    *,
    kind: str,
    page: Any = "",
    excerpt: str,
    suggested_fix: str = "",
    file: str = "",
) -> None:
    if getattr(cfg, "artifact_dir", None) is None:
        return
    _REVIEW_ROWS.append({
        "kind": kind,
        "file": file,
        "page": str(page or ""),
        "excerpt": _clean_snippet(excerpt),
        "suggested_fix": _clean_snippet(suggested_fix),
    })


def apply_exact_fixes(
    text: str,
    cfg: Any,
    *,
    page: Any = "",
    element: dict | None = None,
    source: str = "shared_exact",
) -> str:
    fixes = load_exact_fixes(cfg)
    out = text
    for old, new in fixes.phrase.items():
        start = 0
        while True:
            idx = out.find(old, start)
            if idx < 0:
                break
            before, after = _context(out, idx, idx + len(old), new)
            record_correction(
                cfg,
                source=source,
                rule=f"phrase:{old}",
                page=page,
                element=element,
                before=before,
                after=after,
            )
            out = out[:idx] + new + out[idx + len(old):]
            start = idx + len(new)

    word_re = _word_re(fixes.word)
    if word_re is None:
        return out

    def repl(m: re.Match[str]) -> str:
        replacement = fixes.word[m.group(0)]
        before, after = _context(out, m.start(), m.end(), replacement)
        record_correction(
            cfg,
            source=source,
            rule=f"word:{m.group(0)}",
            page=page,
            element=element,
            before=before,
            after=after,
        )
        return replacement

    return word_re.sub(repl, out)


def write_report(cfg: BookConfig) -> None:
    path = getattr(cfg, "ocr_corrections_path", cfg.artifact_dir / "ocr_corrections.tsv")
    header = ["source", "rule", "page", "kind", "y", "before", "after", "evidence"]

    def cell(value: str) -> str:
        return str(value).replace("\t", " ").replace("\n", " ").strip()

    rows = [
        "\t".join(cell(getattr(c, h)) for h in header)
        for c in _CORRECTIONS
    ]
    atomic_write_text(path, "\t".join(header) + "\n" + "\n".join(rows) + ("\n" if rows else ""))

    diff_path = getattr(cfg, "ocr_corrections_diff_path", cfg.artifact_dir / "ocr_corrections.diff")
    diff_lines: list[str] = []
    for i, c in enumerate(_CORRECTIONS, start=1):
        label = f"{c.source}:{c.rule}:p{c.page or '?'}:{i}"
        diff_lines.extend([
            f"--- {label}",
            f"+++ {label}",
            f"- {c.before}",
            f"+ {c.after}",
            "",
        ])
    atomic_write_text(diff_path, "\n".join(diff_lines))

    review_path = getattr(cfg, "ocr_review_tsv_path", cfg.artifact_dir / "ocr_review.tsv")
    review_header = ["kind", "file", "page", "excerpt", "suggested_fix"]
    review_rows = [
        "\t".join(cell(row.get(h, "")) for h in review_header)
        for row in _REVIEW_ROWS
    ]
    atomic_write_text(
        review_path,
        "\t".join(review_header) + "\n" + "\n".join(review_rows) + ("\n" if review_rows else ""),
    )


def correction_count() -> int:
    return len(_CORRECTIONS)
