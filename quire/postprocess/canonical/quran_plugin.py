"""Canonical Quran plugin (post-processor).

For every page:
  1. Scan footnotes for ``Quran, <Surah> <s>:<v>`` citations.
  2. For each Arabic-kind element, try to match against the canonical Quran
     text using a 2-gram anchor under a citation hint, OR a global 4-gram
     anchor without a hint.
  3. If the match score is strong enough, replace the OCR'd text with the
     canonical Tanzil substring wrapped in decorative ﴿…﴾ brackets.

Configurable in ``book.toml``::

    [postprocess.canonical_quran]
    corpus = "quran/tanzil-uthmani.txt"   # relative to data/canonical/
    score_threshold_with_citation = 0.40
    score_threshold_without = 0.55
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from . import quran as _quran


def _resolve_corpus(cfg) -> Path | None:
    settings = cfg.plugin_config("canonical_quran")
    rel = (settings or {}).get("corpus", "quran/tanzil-uthmani.txt")
    candidate = Path(rel)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    # Try <repo>/data/canonical/<rel>
    from ...config import REPO_ROOT
    full = REPO_ROOT / "data" / "canonical" / rel
    if full.exists():
        return full
    return None


def _replace_with_canonical(text: str, citations: list[tuple[int, int]],
                            score_with: float, score_without: float) -> str | None:
    result = None
    if citations:
        result = _quran.find_best_verse(text, citations)
        if result and result[3] < score_with:
            result = None
    if result is None:
        result = _quran.find_verse_by_anchor(text)
        if result and result[3] < score_without:
            result = None
    if result is None:
        return None
    surah, ayah, slice_text, score = result
    has_lead = bool(re.match(r"^\s*[\(\[\{﴿]?\s*\.{2,}", text))
    has_trail = bool(re.search(r"\.{2,}\s*[\)\]\}﴾]?\s*$", text))
    canonical = slice_text.strip()
    if has_lead:
        canonical = "... " + canonical
    if has_trail:
        canonical = canonical + " ..."
    return "﴿" + canonical + "﴾"


def post_structure(cfg, ocr_pages: list[dict]) -> None:
    settings = cfg.plugin_config("canonical_quran") or {}
    corpus = _resolve_corpus(cfg)
    if not corpus:
        print("[quire] canonical_quran: no corpus available, skipping",
              file=sys.stderr)
        return
    _quran.set_corpus_path(corpus)

    score_with = float(settings.get("score_threshold_with_citation", 0.40))
    score_without = float(settings.get("score_threshold_without", 0.55))

    n_replaced = 0
    for page in ocr_pages:
        # Collect Quran citations from this page's footnotes
        cits: list[tuple[int, int]] = []
        for el in page.get("elements", []):
            if el.get("kind") == "footnote":
                for c in _quran.find_citations(el.get("text", "")):
                    cits.append((c["surah"], c["ayah"]))
        page["_quran_citations"] = cits

        for el in page.get("elements", []):
            if el.get("kind") != "arabic":
                continue
            replaced = _replace_with_canonical(
                el["text"], cits, score_with, score_without
            )
            if replaced is not None:
                el["text"] = replaced
                el["canonical"] = True
                el["is_quran"] = True
                n_replaced += 1
    if n_replaced:
        print(f"[quire] canonical_quran: replaced {n_replaced} blocks",
              file=sys.stderr)
