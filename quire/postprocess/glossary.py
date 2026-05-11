"""Glossary auto-extract plugin.

Many books include a printed glossary of the form::

    <english> (<arabic> [<transliteration>]) : <definition>

This plugin scans the OCR'd pages, finds the glossary chapter heuristically,
extracts every (transliteration, arabic) pair, and registers them in the
shared vocabulary so that downstream substitution gets coverage for free.

Configurable via ``book.toml``::

    [postprocess.glossary_extract]
    pattern = "glossary"             # case-insensitive; selects pages
    pair_regex = "default"           # or a custom regex with named groups <ar> <tr>
"""

from __future__ import annotations

import re
import sys

from .vocabulary import update_with_pairs

DEFAULT_PAIR_RE = re.compile(
    r"\(\s*"
    r"(?P<ar>[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff][\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff\u064b-\u065f\u0670\s]{1,30})"
    r"\s*"
    r"\["
    r"(?P<tr>[A-Za-z'\u00e0-\u017f\u02bb\u02bf\.\- ]{2,30}?)"
    r"\s*\]\s*\)",
)


def _find_pages(ocr_pages: list[dict], pattern: str) -> list[int]:
    needle = pattern.upper()
    hits: list[int] = []
    for i, p in enumerate(ocr_pages):
        en = " ".join(L["text"] for L in p.get("en_lines", []))
        ar = " ".join(L["text"] for L in p.get("ar_lines", []))
        head = (en + " " + ar)[:200].upper()
        if needle in head:
            hits.append(i + 1)  # 1-indexed PDF page numbers
    if not hits:
        return []
    return list(range(hits[0], hits[-1] + 1))


def _extract_pairs(lines: list[dict], pair_re: re.Pattern[str]) -> dict[str, str]:
    text = "\n".join(L["text"] for L in lines)
    pairs: dict[str, str] = {}
    for m in pair_re.finditer(text):
        ar = m.group("ar").strip()
        tr = m.group("tr").strip()
        if not ar or not tr:
            continue
        ar_chars = sum(1 for c in ar if "\u0600" <= c <= "\u06ff")
        if ar_chars < 2:
            continue
        pairs.setdefault(tr, ar)
    return pairs


def pre_structure(cfg, ocr_pages: list[dict]) -> None:
    settings = cfg.plugin_config("glossary_extract")
    pattern = (settings.get("pattern") if settings else None) or "GLOSSARY"
    pair_regex = settings.get("pair_regex") if settings else None
    pair_re = (
        DEFAULT_PAIR_RE if not pair_regex or pair_regex == "default"
        else re.compile(pair_regex)
    )

    pno_range = _find_pages(ocr_pages, pattern)
    if not pno_range:
        return
    pairs: dict[str, str] = {}
    for pno in pno_range:
        lines = sorted(
            ocr_pages[pno - 1].get("ar_lines", []), key=lambda L: L["y0"]
        )
        for tr, ar in _extract_pairs(lines, pair_re).items():
            pairs.setdefault(tr, ar)
    if pairs:
        update_with_pairs(pairs)
        print(
            f"[quire] glossary auto-extract: {len(pairs)} pairs from "
            f"PDF pages {pno_range[0]}-{pno_range[-1]}",
            file=sys.stderr,
        )
