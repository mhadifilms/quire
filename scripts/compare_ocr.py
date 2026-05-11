"""A/B comparison of two OCR engines on the same source.

Builds two book folders side-by-side (typically a Vision-configured and a
Tesseract-configured copy of the same PDF), runs the audit on both, and
prints a single-row scorecard so the relative strength of each engine is
obvious. Used to iterate on engine tuning.

Both book folders must already exist under ``books/`` and reference the
same source PDF; only ``[ocr] engine = ...`` should differ.

Usage::

    python3 scripts/compare_ocr.py BOOK_A BOOK_B
    python3 scripts/compare_ocr.py BOOK_A BOOK_B --force        # re-OCR both
    python3 scripts/compare_ocr.py BOOK_A BOOK_B --force-b      # re-OCR only B

``--force-b`` is handy when iterating on the second engine (B) without
re-running the (already-known-good) baseline (A).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from quire.config import load_book_config  # noqa: E402
from quire.pipeline import build_book  # noqa: E402
from quire.render.audit import run_audit  # noqa: E402


def build_and_audit(book_slug: str, *, force_ocr: bool) -> dict:
    cfg = load_book_config(f"books/{book_slug}", repo_root=REPO)
    t0 = time.monotonic()
    build_book(cfg, force_ocr=force_ocr)
    build_s = time.monotonic() - t0
    run_audit(cfg, run_epubcheck=False)
    data = json.loads(cfg.audit_json_path.read_text())
    return {
        "slug": cfg.slug,
        "engine": cfg.ocr_engine,
        "english_pct": data["english_coverage_pct"],
        "english_words": data["english_words"],
        "arabic_chars_epub": data["arabic_chars"],
        "ocr_arabic_chars_real": data["ocr_arabic_chars_real"],
        "suspicious": data["suspicious_count"],
        "build_s": round(build_s, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="A/B compare OCR engines on two pre-configured book folders."
    )
    parser.add_argument("book_a", help="Slug of the baseline book (under books/)")
    parser.add_argument("book_b", help="Slug of the candidate book (under books/)")
    parser.add_argument(
        "--force", action="store_true",
        help="Force-re-OCR both books (discard caches)",
    )
    parser.add_argument(
        "--force-b", action="store_true",
        help="Force-re-OCR only the second (candidate) book",
    )
    args = parser.parse_args()

    rows = [
        build_and_audit(args.book_a, force_ocr=args.force),
        build_and_audit(args.book_b, force_ocr=args.force or args.force_b),
    ]
    cols = ["engine", "english_pct", "english_words", "arabic_chars_epub",
            "ocr_arabic_chars_real", "suspicious", "build_s"]
    header = " | ".join(f"{c:>22}" for c in cols)
    print()
    print(header)
    print("-" * len(header))
    for r in rows:
        cells = [
            f"{r['engine']:>22}",
            f"{r['english_pct']:>21.2f}%",
            f"{r['english_words']:>22}",
            f"{r['arabic_chars_epub']:>22}",
            f"{r['ocr_arabic_chars_real']:>22}",
            f"{r['suspicious']:>22}",
            f"{r['build_s']:>22}",
        ]
        print(" | ".join(cells))

    a, b = rows[0], rows[1]
    print()
    print(f"  ENGLISH WORDS Δ:  {b['english_words'] - a['english_words']:+d}   "
          f"({b['english_pct'] - a['english_pct']:+.2f} pp coverage)")
    print(f"  EPUB ARABIC Δ:    {b['arabic_chars_epub'] - a['arabic_chars_epub']:+d}   "
          f"(B = {b['arabic_chars_epub'] / max(a['arabic_chars_epub'], 1) * 100:.1f}% of A)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
