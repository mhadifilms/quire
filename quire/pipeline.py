"""End-to-end pipeline orchestrator.

This module is intentionally book-agnostic: every concrete decision (which OCR
engine, which post-processors, which canonical sources, what title etc.) is
driven by ``BookConfig``. To support a new book, write a new ``book.toml``;
to support a new language, register a post-processor plugin.
"""

from __future__ import annotations

import hashlib
import pickle
import re
import statistics
import sys
from pathlib import Path
from typing import Any

import fitz

from .config import BookConfig
from .context import reset_for_build
from .extract.pdf import extract_all
from .extract.refine import refine_all as refine_arabic_all
from .io_utils import atomic_write_bytes, content_fingerprint, file_lock
from .logging_utils import log_event
from .postprocess import ocr_corrections
from .postprocess import registry as pp_registry
from .render.chapters import assemble_chapters, render_chapter
from .render.export import write_exports
from .render.package import build_epub, render_cover_jpeg
from .render.typography import (
    apply_typography_fixes,
    load_qc_fixes,
)
from .render.typography import (
    build_vocab as _build_typography_vocab,
)
from .structure.pdf_based import (
    estimate_global_body_size,
)
from .structure.pdf_based import (
    structure_page as structure_page_text,
)
from .structure.vision_based import structure_page_vision


def log(msg: str) -> None:
    print(f"[quire] {msg}", file=sys.stderr, flush=True)


# ---------- OCR ----------

CACHE_SCHEMA_VERSION = 3


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_meta_matches(cached: dict, expected: dict) -> bool:
    keys = ("kind", "pdf_size", "ocr_engine", "ocr_languages", "extra")
    if not all(cached.get(k) == expected.get(k) for k in keys):
        return False
    if cached.get("pdf_fingerprint") == expected.get("pdf_fingerprint"):
        return True
    # Schema v2 used a full-file SHA and an absolute source path. Accept it
    # after verifying the current source content, then the reader rewrites it.
    if cached.get("pdf_sha256") and expected.get("pdf_path"):
        return cached["pdf_sha256"] == _file_sha256(Path(expected["pdf_path"]))
    return False


def _cache_meta(cfg: BookConfig, kind: str, extra: dict | None = None) -> dict:
    stat = cfg.pdf_path.stat()
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "kind": kind,
        "pdf_path": str(cfg.pdf_path.resolve()),
        "pdf_size": stat.st_size,
        "pdf_fingerprint": content_fingerprint(cfg.pdf_path),
        "ocr_engine": cfg.ocr_engine,
        "ocr_languages": list(cfg.ocr_languages),
        "extra": extra or {},
    }


def _read_pickle_payload(path: Path) -> object | None:
    """Read a pickle file written by Quire.

    SECURITY: pickle deserialization can execute arbitrary code. Only call
    this on caches inside ``cfg.caches_dir`` which Quire itself wrote.
    """
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_cache(path: Path, expected_meta: dict) -> list[dict] | None:
    payload = _read_pickle_payload(path)
    if payload is None:
        return None
    if isinstance(payload, dict) and isinstance(payload.get("meta"), dict):
        cached_meta = payload["meta"]
        if _cache_meta_matches(cached_meta, expected_meta):
            pages = payload.get("pages")
            if not isinstance(pages, list):
                return None
            if cached_meta != expected_meta:
                _write_cache(path, expected_meta, pages)
            return pages
        log(f"cache fingerprint changed; ignoring stale cache: {path}")
        return None
    log(f"unrecognized cache payload at {path}; ignoring")
    return None


def read_pages_cache(path: Path) -> list[dict] | None:
    """Public helper for the audit/render modules: read the page list from
    a Quire-written pickle cache, ignoring meta wrapper."""
    payload = _read_pickle_payload(path)
    if isinstance(payload, dict):
        pages = payload.get("pages")
        return pages if isinstance(pages, list) else None
    if isinstance(payload, list):
        return payload
    return None


def _write_cache(path: Path, meta: dict, pages: list[dict]) -> None:
    data = pickle.dumps({"meta": meta, "pages": pages}, protocol=pickle.HIGHEST_PROTOCOL)
    atomic_write_bytes(path, data)


def _ocr_errors(pages: list[dict]) -> list[str]:
    return [
        f"p. {p.get('pno')}: {p.get('error')}"
        for p in pages
        if p.get("error")
    ]


def _run_vision_ocr(
    cfg: BookConfig,
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """Run Vision OCR with per-page resume.

    See :func:`_run_engine_ocr` for the generic implementation; this is the
    Vision-specific wrapper that keeps the cache filename stable for back
    compat with existing builds.
    """
    return _run_engine_ocr(cfg, engine_name="vision", force=force, retry_failed=retry_failed)


def _run_tesseract_ocr(
    cfg: BookConfig,
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    return _run_engine_ocr(cfg, engine_name="tesseract", force=force, retry_failed=retry_failed)


def _run_engine_ocr(
    cfg: BookConfig,
    *,
    engine_name: str,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """Run OCR through any engine, with per-page caching and resume.

    Behaviour:
      - ``force=True``      → discard the cache and re-OCR every page.
      - ``retry_failed=True`` → keep good pages from the cache, only re-OCR
        the failed ones (pages with an ``error`` key).
      - default             → if any cached page has an error, raise — the
        caller must opt in to resume via ``retry_failed``.
    """
    cache_name = "vision_ocr" if engine_name == "vision" else f"{engine_name}_ocr"
    cache_path = cfg.caches_dir / f"{cache_name}.pkl"
    meta = _cache_meta(cfg, cache_name, extra={
        "workers": cfg.ocr_workers,
        "scale": cfg.ocr_dpi_scale,
    })
    def _ocr_call(page_numbers: list[int] | None = None) -> list[dict | None]:
        if engine_name == "vision":
            from .extract.ocr import ocr_all as vision_ocr_all
            return vision_ocr_all(
                str(cfg.pdf_path),
                languages=cfg.ocr_languages,
                workers=cfg.ocr_workers,
                scale=cfg.ocr_dpi_scale,
                retries=cfg.ocr_retries,
                page_numbers=page_numbers,
            )
        if engine_name == "tesseract":
            from .extract.tesseract_engine import ocr_pdf_tesseract
            return ocr_pdf_tesseract(
                str(cfg.pdf_path),
                languages=cfg.ocr_languages,
                workers=cfg.ocr_workers,
                scale=cfg.ocr_dpi_scale,
                retries=cfg.ocr_retries,
                page_numbers=page_numbers,
            )
        raise ValueError(f"_run_engine_ocr: unsupported engine_name={engine_name!r}")

    label = "Vision" if engine_name == "vision" else "Tesseract"
    cached = None if force else _read_cache(cache_path, meta)
    if cached is not None:
        errors = _ocr_errors(cached)
        if not errors:
            log(f"loading cached {label} OCR from {cache_path}")
            return cached
        if not retry_failed:
            raise RuntimeError(
                "cached OCR contains failed pages — pass --retry-failed to "
                "resume:\n" + "\n".join(errors[:10])
            )
        failed_pnos = [p["pno"] for p in cached if p.get("error")]
        log(
            f"resuming OCR: {len(failed_pnos)} failed page(s) out of "
            f"{len(cached)} (pages {failed_pnos[:10]}{'…' if len(failed_pnos) > 10 else ''})"
        )
        partial = _ocr_call(page_numbers=failed_pnos)
        merged: list[dict] = []
        for i, old in enumerate(cached):
            new = partial[i] if i < len(partial) else None
            merged.append(new if new is not None else old)
        errors_after = _ocr_errors(merged)
        if errors_after:
            _write_cache(cache_path, meta, merged)
            raise RuntimeError(
                f"{label} OCR still failed for page(s) after resume:\n"
                + "\n".join(errors_after[:10])
            )
        _write_cache(cache_path, meta, merged)
        log(f"resume succeeded; wrote {cache_path}")
        return merged

    log(
        f"running {label} OCR on {cfg.pdf_path} "
        f"(workers={cfg.ocr_workers}, scale={cfg.ocr_dpi_scale}, retries={cfg.ocr_retries})"
    )
    raw_pages = _ocr_call(page_numbers=None)
    # Without ``page_numbers`` every slot is populated; narrow the type.
    pages: list[dict] = [p for p in raw_pages if p is not None]
    errors = _ocr_errors(pages)
    if errors:
        # Write a partial cache so users can ``--retry-failed`` later instead
        # of starting from scratch.
        _write_cache(cache_path, meta, pages)
        raise RuntimeError(
            f"{label} OCR failed for page(s) (partial cache written; rerun with "
            "--retry-failed to resume):\n" + "\n".join(errors[:10])
        )
    _write_cache(cache_path, meta, pages)
    log(f"wrote {cache_path}")
    return pages


def _run_arabic_refine(
    cfg: BookConfig,
    pages: list[dict],
    *,
    force: bool = False,
) -> list[dict] | None:
    """Optional: re-OCR only Arabic-region crops with ar-only language pref.

    The refine pass uses macOS Vision (``ocrmac``) and was tuned against
    Vision's primary OCR output. Running it on **Tesseract** blocks is
    destructive: Vision systematically returns less Arabic than Tesseract
    on this corpus, and ``_pick_best_refined`` would overwrite Tesseract's
    text with Vision's narrower version, dropping ~65 % of the Arabic.
    So we only run refine when the primary engine is Vision.
    """
    engine = cfg.ocr_engine.lower()
    if engine != "vision":
        return None
    refined_path = cfg.caches_dir / "vision_arabic_refined.pkl"
    refined_pages: list[dict] | None = None
    refine_cfg = cfg.plugin_config("arabic_refine") or cfg.plugin_config("vocabulary")
    if not refine_cfg:
        return None
    languages = refine_cfg.get("languages") if isinstance(refine_cfg, dict) else None
    if not languages:
        return None
    meta = _cache_meta(cfg, "vision_arabic_refined", {"languages": list(languages)})
    if not force:
        cached = _read_cache(refined_path, meta)
        if cached is not None:
            log(f"loading refined Arabic OCR from {refined_path}")
            return cached
    log(
        f"running targeted Arabic-only OCR (workers={cfg.ocr_workers}, "
        f"scale={cfg.ocr_dpi_scale}, retries={cfg.ocr_retries})"
    )
    refined_pages = refine_arabic_all(
        str(cfg.pdf_path),
        pages,
        languages=languages,
        workers=cfg.ocr_workers,
        scale=cfg.ocr_dpi_scale,
        retries=cfg.ocr_retries,
    )
    _write_cache(refined_path, meta, refined_pages)
    return refined_pages


# ---------- block filter helpers ----------

_AR_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")
_AR_WORD = re.compile(r"[\u0600-\u06ff]{4,}")
_AR_WORD3 = re.compile(r"[\u0600-\u06ff]{3,}")


def _arabic_dominant(text: str) -> bool:
    ar = sum(1 for c in text if _AR_RE.match(c))
    if ar < 2:
        return False
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return ar >= latin * 0.8


# --- block-level filter: drop hallucination overlaid on embedded English ---

# A Tesseract Arabic block that (a) sits geometrically over text the PDF's
# *embedded* layer marks as English, (b) is low-confidence, and (c)
# contains zero Arabic diacritics is almost certainly hallucination over
# Latin glyphs. The three signals together give ~98 % precision on a
# bilingual Arabic/English reference corpus while losing well under 0.5
# chars of real Arabic per page.
#
# Note: this filter is a no-op for PDFs that have no embedded text layer
# (pure scans) — ``page.get_text("words")`` returns an empty list and
# coverage is always 0. It is also a no-op for the Vision engine; we only
# wire it in for Tesseract.
_AR_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u06d6-\u06ed]")

# --- thresholds for the line-level embedded-text hallucination filter ---
#
# We filter at the line level (BEFORE clustering into blocks) because the
# block-level filter cannot split clusters that bundle a hallucinated page
# header / footnote with adjacent real Arabic prose: the real text dominates
# the cluster's average confidence and an "any diacritic" check skips the
# whole block, even though some of its constituent lines are pure noise.
#
# A line is dropped when **all** of:
#   - it sits geometrically over text the PDF's embedded layer marks as
#     English (>= 50 % horizontal coverage at the line's y-range);
#   - one of these fingerprints fires:
#       (a) confidence < 35 AND diacritic density < 5 % (very low conf is
#           almost always reversed-Latin noise UNLESS it's a heavily
#           diacritised classical Arabic passage that the engine just had
#           trouble reading — 5 % diacritic density is comfortably below
#           normal classical-Arabic density of 15-25 %, so this protects
#           real prose without weakening the hallucination guard).
#       (b) confidence < 50 AND no Arabic diacritics
#       (c) confidence < 50 AND high ASCII clutter (>= 20 % of arabic char
#           count, capturing the digit-and-bracket signature of bibliography
#           hallucination)
#       (d) confidence < 60 AND diacritic density < 3 % AND ascii clutter
#           >= 10 % (catches mid-conf "fake diacritic" hallucinations like
#           Tesseract reading "[alastu]" and emitting a sprinkled-diacritic
#           Arabic mock-up of it).
#
# The geometric overlap requirement is the safety net: real Arabic prose
# never overlays English-marked PDF regions (unless the embedded layer is
# itself wrong, which Adobe / ABBYY rarely are for English).
_EMBED_MIN_HORIZ_COV = 0.50
_LINE_VERY_LOW_CONF = 35.0
_LINE_LOW_CONF = 50.0
_LINE_MID_CONF = 60.0
_LINE_CLUTTER_HIGH = 0.20
_LINE_CLUTTER_MID = 0.10
_LINE_DIACRITIC_MIN_DENSITY = 0.03
# Lines with high diacritic density (>= 5%) are protected from the
# "very low confidence" reject: heavily-diacritised classical Arabic
# can read at conf 20-35 when the scan is faded, but it's real text.
_LINE_PROTECT_DIACRITIC_DENSITY = 0.05

# --- thresholds for the (legacy) block-level embedded-text filter ---
#
# The line-level filter above does most of the work; the block filter is a
# safety net for edge cases where lines were already clustered (no per-line
# bbox preserved) but the cluster as a whole sits on embedded English at
# low confidence. We raise the cap from 30 -> 45 because hallucinations in
# the 30-45 conf band were slipping through (e.g. all-caps running headers
# in the page margin being read RTL as Arabic-script and scoring ~38 conf),
# and we replace the binary "any diacritic" check with a density threshold
# (Tesseract sprinkles 1-2 fake diacritics into its hallucinations, which
# trivially defeats a binary check).
_EMBED_MAX_CONF = 45.0
_EMBED_DIACRITIC_MIN_DENSITY = 0.03


def _embedded_english_words(
    doc: Any,
    page_idx: int,
) -> list[tuple[float, float, float, float]]:
    """Return PDF-point bboxes of English words from the PDF's embedded
    text layer for the given page.

    "English" = a word containing at least 2 ASCII alphabetical letters.
    Skips punctuation-only tokens and embedded Arabic (which on our
    target corpora is left empty by Adobe/ABBYY anyway). Returns ``[]``
    on a pure scan with no embedded layer.

    All coordinates are in PDF user units (1 unit = 1 pt), the same
    space the Tesseract engine produces line bboxes in (see
    ``_group_words_to_lines`` in ``tesseract_engine.py`` which divides
    raw pixel coordinates by the render scale).
    """
    try:
        page = doc[page_idx]
        words = page.get_text("words")
    except Exception:
        return []
    out: list[tuple[float, float, float, float]] = []
    for w in words:
        if len(w) < 5:
            continue
        text = w[4]
        ascii_alpha = sum(1 for c in text if c.isascii() and c.isalpha())
        if ascii_alpha < 2:
            continue
        out.append((float(w[0]), float(w[1]), float(w[2]), float(w[3])))
    return out


def _block_horiz_coverage_on_english(
    block: dict,
    embedded_words: list[tuple[float, float, float, float]],
) -> float:
    """Fraction of the block's horizontal extent covered (at any y inside
    the block's y-range) by embedded English word bboxes.

    Uses merged x-intervals: a block whose entire visible width is shared
    with embedded English words at the same y-position scores ~1.0; a
    block sitting next to (rather than on top of) English scores ~0.0.

    We measure horizontal coverage (rather than area coverage) because
    Arabic clusters span multiple lines vertically; per-area coverage
    underestimates due to inter-line whitespace.
    """
    bx0, by0, bx1, by1 = block["x0"], block["y0"], block["x1"], block["y1"]
    width = max(bx1 - bx0, 1.0)
    intervals: list[tuple[float, float]] = []
    for ex0, ey0, ex1, ey1 in embedded_words:
        if ey1 < by0 or ey0 > by1:
            continue
        a = max(ex0, bx0)
        b = min(ex1, bx1)
        if b > a:
            intervals.append((a, b))
    if not intervals:
        return 0.0
    intervals.sort()
    merged = 0.0
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged += cur_e - cur_s
            cur_s, cur_e = s, e
    merged += cur_e - cur_s
    return min(merged / width, 1.0)


def _is_line_geometric_artifact(line: dict) -> bool:
    """Return True for OCR "lines" that can only be a scan artifact.

    Tesseract's RTL pass occasionally outputs a tall, thin column of
    garbage at the scan's binding / spine / page edge: a 10pt-wide,
    200pt-tall "line" of low-confidence Arabic glyphs that is in fact
    the dark gutter of a book scan being misread as text. Real text
    lines on this corpus run roughly 8-22pt tall and 100-400pt wide
    (width-to-height ratio ≥ 5:1). Any "line" tall enough to span
    multiple real lines but too narrow to hold a single word is
    overwhelmingly likely to be a binding artifact, regardless of OCR
    confidence or embedded-text overlap. The rule is purely geometric
    and applies uniformly to all PDFs.
    """
    try:
        width = float(line.get("x1", 0)) - float(line.get("x0", 0))
        height = float(line.get("y1", 0)) - float(line.get("y0", 0))
    except (TypeError, ValueError):
        return False
    if width <= 0 or height <= 0:
        return False
    # Tall + narrow: height substantially exceeds width AND width is
    # less than ~40pt (less than the width of three real Arabic chars
    # at 12pt body size). Real lines never look like this.
    if height > width and width < 40.0:
        return True
    # Extreme aspect ratio in either direction is also suspicious.
    if height > 0 and width / height < 0.5 and width < 60.0:
        return True
    return False


def _ascii_clutter_count(text: str) -> int:
    """ASCII letters / digits / punctuation embedded in an Arabic-glyph block.

    Real Arabic prose rarely contains ASCII clutter: a typical body
    paragraph or classical citation has zero ASCII chars. Hallucinations
    over Latin glyphs almost always carry brackets, digits, periods,
    commas — the typographical scaffolding of footnote / bibliography
    text — because Tesseract's RTL pass is literally reading the
    underlying English text right-to-left.
    """
    return sum(
        1 for c in text
        if c.isascii() and (c.isalnum() or c in "[](){}<>,.:;'\"!?-/\\|=+*")
    )


def _is_line_hallucination(
    line: dict,
    embedded_words: list[tuple[float, float, float, float]],
) -> bool:
    """Return True if a single OCR line looks like Tesseract hallucinating
    Arabic glyphs over text the PDF's embedded layer marks as English.

    Runs BEFORE clustering so that a hallucinated page header / footnote
    bundled with adjacent real Arabic prose can be dropped without
    discarding the real text. See the threshold constants at the top of
    this module for the rationale behind each fingerprint.
    """
    if not embedded_words:
        return False
    text = line.get("text", "") or ""
    if not text.strip():
        return False
    ar_chars = sum(1 for c in text if _AR_RE.match(c))
    if ar_chars == 0:
        return False

    horiz_cov = _block_horiz_coverage_on_english(line, embedded_words)
    if horiz_cov < _EMBED_MIN_HORIZ_COV:
        return False

    conf = line.get("conf", 0)
    if 0 < conf <= 1:
        conf = conf * 100

    diacritics = len(_AR_DIACRITICS_RE.findall(text))
    diacritic_density = diacritics / max(ar_chars, 1)
    clutter = _ascii_clutter_count(text)
    clutter_ratio = clutter / max(ar_chars, 1)

    # (a) very-low-conf line over embedded English → reject, UNLESS the
    #     line has substantial diacritic density (heavily diacritised
    #     classical Arabic is real text, even at conf 25-30 on a faded
    #     scan).
    if (
        conf < _LINE_VERY_LOW_CONF
        and diacritic_density < _LINE_PROTECT_DIACRITIC_DENSITY
    ):
        return True
    # (b) low-conf line over embedded English with no diacritics at all.
    if conf < _LINE_LOW_CONF and diacritics == 0:
        return True
    # (c) low-conf line over embedded English with heavy ASCII clutter.
    if conf < _LINE_LOW_CONF and clutter_ratio >= _LINE_CLUTTER_HIGH:
        return True
    # (d) mid-conf line over embedded English with sparse fake diacritics
    #     AND some ASCII clutter. Catches Tesseract's "fake-diacritic
    #     hallucination" mode where it sprinkles 1-2 tashkeel marks into
    #     a stream of reversed Latin to evade a binary diacritic gate.
    if (
        conf < _LINE_MID_CONF
        and diacritic_density < _LINE_DIACRITIC_MIN_DENSITY
        and clutter_ratio >= _LINE_CLUTTER_MID
    ):
        return True
    return False


def _is_embedded_text_hallucination(
    block: dict,
    embedded_words: list[tuple[float, float, float, float]],
) -> bool:
    """Block-level safety net behind :func:`_is_line_hallucination`.

    Conditions (all must hold):

    1. ``conf < 45`` — Tesseract is at most moderately sure. Raised from
       the original 30 because we observed hallucinated page headers
       (page numbers + all-caps title lines in scan margins) scoring
       35-42 conf when read RTL as Arabic.
    2. Arabic-diacritic density below 3 %. We do **not** use a binary
       "has-any-diacritic" gate: Tesseract sprinkles 1-2 fake tashkeel
       marks into hallucinated reversed-Latin chains, trivially
       defeating the binary check. Real classical Arabic prose runs at
       5-10 % diacritic density on this corpus.
    3. Horizontal coverage on embedded English ≥ 50 %.

    Safety: for PDFs without an embedded text layer (pure scans),
    ``embedded_words`` is empty and the filter is a silent no-op.
    """
    if not embedded_words:
        return False
    conf = block.get("conf", 0)
    if 0 < conf <= 1:
        conf = conf * 100
    if conf >= _EMBED_MAX_CONF:
        return False
    text = block.get("text", "") or ""
    ar_chars = sum(1 for c in text if _AR_RE.match(c))
    if ar_chars == 0:
        return False
    diacritics = len(_AR_DIACRITICS_RE.findall(text))
    if diacritics / max(ar_chars, 1) >= _EMBED_DIACRITIC_MIN_DENSITY:
        return False
    return (
        _block_horiz_coverage_on_english(block, embedded_words)
        >= _EMBED_MIN_HORIZ_COV
    )


# --- page-level filter: drop bibliography / index pages ---

# Heuristic thresholds for the page-level bibliography filter. These were
# tuned on a corpus of bilingual Arabic/English religious texts where
# Tesseract's ``ara`` LSTM hallucinates Arabic glyphs over Latin citation
# pages. They are deliberately conservative — a page must show *all* of
# the following before we silence its Arabic blocks:
#   - lots of recognised English (≥ 1000 chars)
#   - high ASCII-digit density in that English (≥ 3 %, the signature of
#     citation pages: years, volume nums, page refs)
#   - no anchoring Arabic block on the page (≥ 100 chars AND conf ≥ 65),
#     which would prove this page actually does carry Arabic prose.
_BIB_MIN_EN_CHARS = 1000
_BIB_MIN_DIGIT_FRAC = 0.03
_BIB_ANCHOR_AR_CHARS = 100
_BIB_ANCHOR_AR_CONF = 65.0


def _page_is_english_citation_dense(page: dict) -> bool:
    """Return ``True`` for pages that look like English bibliography / index.

    Tesseract's ``ara`` LSTM happily hallucinates Arabic glyphs over a page
    of Latin citations (author names + years + page numbers + italic
    titles). Vision's primary OCR returns zero Arabic on those same pages,
    so we have a clean ground-truth signal: a citation-heavy English page
    in this corpus contains *no* real Arabic. The heuristic identifies
    such pages by their structural shape, not by content, so it
    generalises beyond a single book.

    The page must show all of:
      - ``en_chars >= 1000`` — substantial English captured.
      - ``digit_chars / en_chars >= 0.03`` — citation pages are unusually
        digit-rich (years, volumes, page numbers); flowing prose runs
        well under 1 %.
      - no anchoring Arabic block: any block with ``ar_chars >= 100`` AND
        ``conf >= 65`` proves real Arabic prose exists on the page and we
        leave it alone.

    Empirically: on a 161-page bilingual reference corpus this flags
    ~15 pages — *all* of them pure bibliography or citation-index pages
    where the Vision baseline also reports zero Arabic — and zero pages
    where Vision agrees Arabic is present.
    """
    en_lines = page.get("en_lines", [])
    en_chars = sum(len(L.get("text", "")) for L in en_lines)
    if en_chars < _BIB_MIN_EN_CHARS:
        return False
    digit_chars = sum(
        1 for L in en_lines for c in L.get("text", "") if c.isascii() and c.isdigit()
    )
    if digit_chars / max(en_chars, 1) < _BIB_MIN_DIGIT_FRAC:
        return False
    for b in page.get("arabic_blocks", []):
        ar = sum(1 for c in b.get("text", "") if _AR_RE.match(c))
        conf = b.get("conf", 0)
        if 0 < conf <= 1:
            conf = conf * 100
        if ar >= _BIB_ANCHOR_AR_CHARS and conf >= _BIB_ANCHOR_AR_CONF:
            return False
    return True


def _filter_vision_blocks(
    blocks: list[dict],
    *,
    strict_digits: bool = False,
) -> list[dict]:
    """Drop Arabic blocks that look like OCR noise or hallucination.

    ``strict_digits=True`` adds two extra reject rules tuned for Tesseract's
    well-known failure mode of hallucinating Arabic glyphs in Latin
    bibliography / index regions. The Tesseract ``ara`` LSTM happily
    "translates" a page of English citations into a stream of arabic-block
    code points; those hallucinations almost always carry an unusual
    density of ASCII digits (citation page numbers, years, volume refs)
    that real Arabic prose never has. The two rules are:

    - **digit-dominated reject**: any block where ASCII digits exceed 20 %
      of the Arabic char count.
    - **short-fragment reject**: any block with fewer than 30 Arabic chars
      that contains at least one ASCII digit and has conf < 60.

    Vision rarely emits either shape, so this is left off for Vision to
    keep the Vision build bit-identical. Note that this filter operates
    on individual blocks; the page-level
    :func:`_page_is_english_citation_dense` filter complements it by
    catching pages where every block is hallucinated (and so no
    individual block looks particularly bad).
    """
    out = []
    for b in blocks:
        text = b["text"]
        ar_chars = sum(1 for c in text if _AR_RE.match(c))
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        if ar_chars < 5:
            continue
        if latin_chars > ar_chars * 0.6:
            continue
        conf = b.get("conf", 100)
        if 0 < conf <= 1:
            conf *= 100
        if strict_digits and ar_chars > 0:
            digit_chars = sum(1 for c in text if c.isascii() and c.isdigit())
            if digit_chars >= ar_chars * 0.20:
                continue
            if ar_chars < 30 and digit_chars >= 1 and conf < 60:
                continue
        if conf < 50 and not _AR_WORD.search(text):
            continue
        nonblank = [s for s in text.split("\n") if s.strip()]
        if nonblank and all(len(s.strip()) <= 3 for s in nonblank):
            continue
        if conf < 70 and not any(_AR_WORD.search(s) for s in nonblank):
            continue
        longest = max(
            (len(m.group(0)) for s in nonblank for m in _AR_WORD3.finditer(s)),
            default=0,
        )
        if longest < 4 and conf < 80:
            continue
        out.append(b)
    return out


def _pick_best_refined(orig: dict, refined: list[dict]) -> dict | None:
    """For each original Arabic block, find a refined block with overlapping bbox."""
    if not refined:
        return None
    best = None
    best_overlap = 0.0
    for r in refined:
        ix0 = max(orig["x0"], r["x0"])
        iy0 = max(orig["y0"], r["y0"])
        ix1 = min(orig["x1"], r["x1"])
        iy1 = min(orig["y1"], r["y1"])
        if ix1 <= ix0 or iy1 <= iy0:
            continue
        inter = (ix1 - ix0) * (iy1 - iy0)
        area_o = (orig["x1"] - orig["x0"]) * (orig["y1"] - orig["y0"])
        if area_o <= 0:
            continue
        ov = inter / area_o
        if ov > best_overlap:
            best_overlap = ov
            best = r
    if best_overlap < 0.3:
        return None
    return best


# ---------- main entry points ----------


def _run_engine(
    cfg: BookConfig,
    text_pages: list[dict],
    *,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """Single OCR dispatch path: route through ``ocr_engine.get_engine``.

    For Vision OCR we still need the cache + Arabic-refine pipeline, so we
    keep the higher-level :func:`_ocr_pages_for` orchestrator. For lighter
    engines (text-layer-only) we use the engine directly and synthesise the
    minimal page envelopes the pipeline expects.
    """
    from .extract.ocr_engine import get_engine

    name = cfg.ocr_engine.lower()
    if name in {"vision", "tesseract"}:
        return _ocr_pages_for(cfg, force=force, retry_failed=retry_failed)

    log(f"OCR engine: {name} (no Arabic refine; text-layer source)")
    # text/pdf/pymupdf engines do not call out to OCR; we just return the
    # per-page envelopes the structure stage expects. The engine call itself
    # is kept so adding a 3rd-party engine (Tesseract, cloud) is a drop-in.
    engine = get_engine(name)
    from .extract.ocr_engine import OCREngineOptions
    options = OCREngineOptions(
        languages=cfg.ocr_languages,
        workers=cfg.ocr_workers,
        dpi_scale=cfg.ocr_dpi_scale,
        retries=cfg.ocr_retries,
        force=force,
    )
    raw = engine.ocr_pdf(str(cfg.pdf_path), options)
    return [
        {
            "pno": i + 1,
            "printed_page": text_pages[i].get("printed_page") if i < len(text_pages) else None,
            **page,
        }
        for i, page in enumerate(raw)
    ]


def _ocr_pages_for(
    cfg: BookConfig,
    force: bool = False,
    retry_failed: bool = False,
) -> list[dict]:
    """OCR + (optional) Arabic refine, for engines that need the full path.

    Vision and Tesseract both share the cache/resume + Arabic-refine
    pipeline; the difference is which raw OCR backend is invoked.
    """
    engine_name = cfg.ocr_engine.lower()
    if engine_name == "tesseract":
        pages = _run_tesseract_ocr(cfg, force=force, retry_failed=retry_failed)
    else:
        pages = _run_vision_ocr(cfg, force=force, retry_failed=retry_failed)
    refined = _run_arabic_refine(cfg, pages, force=force)

    # Normalize confidences and re-cluster Arabic blocks. Tesseract's
    # ``ara`` LSTM hallucinates Arabic on Latin bibliography and index
    # pages; ``strict_digits`` adds digit-density / short-fragment rejects
    # tuned for that failure mode (no-op for Vision, which doesn't produce
    # that pattern).
    from .extract.ocr import cluster_arabic_blocks
    strict_digits = engine_name == "tesseract"
    bib_suppress = engine_name == "tesseract"
    embed_suppress = engine_name == "tesseract"
    suppressed_pages = 0
    embed_drop_chars = 0
    embed_drop_blocks = 0
    line_drop_count = 0
    line_drop_chars = 0
    # Open the source PDF once so we can read its embedded text layer
    # for the line- and block-level hallucination filters. We close it
    # after the loop. For Vision builds we don't need it and skip the
    # open.
    embed_doc = None
    if embed_suppress:
        try:
            embed_doc = fitz.open(str(cfg.pdf_path))
        except Exception:
            embed_doc = None
    for p in pages:
        for b in p.get("arabic_blocks", []):
            if b.get("conf", 0) <= 1.0:
                b["conf"] = b["conf"] * 100
        ar_lines = [L for L in p.get("ar_lines", []) if _arabic_dominant(L["text"])]

        # Geometric artifact filter (Tesseract only): drop OCR "lines" that
        # are tall-thin scan binding / page-edge regions. Runs unconditionally
        # (doesn't need an embedded text layer).
        if embed_suppress and ar_lines:
            before = len(ar_lines)
            ar_lines = [L for L in ar_lines if not _is_line_geometric_artifact(L)]
            geom_drop = before - len(ar_lines)
            if geom_drop:
                line_drop_count += geom_drop
                # We approximate: assume each artifact line contributed its
                # full text length (cheap, since we already filtered).
                # (Not used downstream beyond logging.)

        # Line-level embedded-text hallucination filter (Tesseract only).
        # Runs BEFORE clustering so that a hallucinated page-header line
        # bundled with an adjacent real-Arabic citation line can be
        # dropped without losing the real text. See
        # :func:`_is_line_hallucination`.
        if embed_suppress and embed_doc is not None and ar_lines:
            pno = p.get("pno")
            page_idx = (pno - 1) if isinstance(pno, int) and pno >= 1 else None
            embedded_for_lines: list[tuple[float, float, float, float]] = []
            if page_idx is not None and 0 <= page_idx < len(embed_doc):
                embedded_for_lines = _embedded_english_words(embed_doc, page_idx)
            if embedded_for_lines:
                kept_lines = []
                for L in ar_lines:
                    if _is_line_hallucination(L, embedded_for_lines):
                        line_drop_count += 1
                        line_drop_chars += sum(
                            1 for c in L.get("text", "") if _AR_RE.match(c)
                        )
                    else:
                        kept_lines.append(L)
                ar_lines = kept_lines
                # Also persist the filtered ar_lines on the page so
                # downstream consumers (e.g. structure_page_vision)
                # don't see the dropped lines.
                p["ar_lines"] = ar_lines

        if ar_lines:
            heights = [L["y1"] - L["y0"] for L in ar_lines if L["y1"] > L["y0"]]
            avg_h = statistics.median(heights) if heights else 14.0
            new_blocks = cluster_arabic_blocks(ar_lines, avg_h)
            for b in new_blocks:
                if 0 < b.get("conf", -1) <= 1.0:
                    b["conf"] *= 100
            p["arabic_blocks"] = _filter_vision_blocks(
                new_blocks, strict_digits=strict_digits,
            )
        else:
            p["arabic_blocks"] = []
        # Block-level embedded-text hallucination filter: safety net for
        # clusters the line-level filter missed (e.g. a block where
        # surviving lines happen to align with embedded English).
        if embed_suppress and embed_doc is not None and p["arabic_blocks"]:
            pno = p.get("pno")
            page_idx = (pno - 1) if isinstance(pno, int) and pno >= 1 else None
            if page_idx is not None and 0 <= page_idx < len(embed_doc):
                embedded = _embedded_english_words(embed_doc, page_idx)
                if embedded:
                    kept = []
                    for b in p["arabic_blocks"]:
                        if _is_embedded_text_hallucination(b, embedded):
                            embed_drop_chars += sum(
                                1 for c in b.get("text", "") if _AR_RE.match(c)
                            )
                            embed_drop_blocks += 1
                        else:
                            kept.append(b)
                    p["arabic_blocks"] = kept
        # Page-level bibliography suppression: only enabled for Tesseract,
        # where the ara LSTM hallucinates Arabic on citation-dense Latin
        # pages. Run AFTER the per-block filter so the heuristic sees the
        # final blocks the pipeline would otherwise emit.
        if bib_suppress and p["arabic_blocks"] and _page_is_english_citation_dense(p):
            log(
                f"  suppressing Arabic on bibliography page {p.get('pno', '?')}: "
                f"{sum(1 for b in p['arabic_blocks'] for c in b.get('text', '') if _AR_RE.match(c))} chars"
            )
            p["arabic_blocks"] = []
            p["ar_lines"] = []
            suppressed_pages += 1
    if embed_doc is not None:
        embed_doc.close()
    if embed_suppress and line_drop_count:
        log(
            f"suppressed {line_drop_count} Arabic line(s) ({line_drop_chars} chars) "
            f"pre-clustering that hallucinated over embedded-English text "
            f"(Tesseract line-level guard)"
        )
    if embed_suppress and embed_drop_blocks:
        log(
            f"suppressed {embed_drop_blocks} Arabic block(s) ({embed_drop_chars} chars) "
            f"post-clustering that overlay embedded-English text at low confidence "
            f"(Tesseract block-level guard)"
        )
    if bib_suppress and suppressed_pages:
        log(f"suppressed Arabic on {suppressed_pages} bibliography-like pages (Tesseract hallucination guard)")

    if refined:
        for p, rp in zip(pages, refined, strict=False):
            blocks = p.get("arabic_blocks", [])
            for orig in blocks:
                best = _pick_best_refined(orig, rp.get("refined_blocks", []))
                if best is not None:
                    orig["text"] = best["text"]
                    if best.get("conf", 0) > 0:
                        orig["conf"] = best["conf"]
    return pages


def _normalize_formats(formats: list[str] | tuple[str, ...] | set[str] | None, cfg: BookConfig) -> set[str]:
    selected = list(formats) if formats is not None else list(cfg.output_formats)
    if not selected:
        selected = ["epub"]
    normalized = {"markdown" if f == "md" else "text" if f == "txt" else str(f).lower() for f in selected}
    invalid = normalized - {"epub", "html", "markdown", "text"}
    if invalid:
        raise ValueError(f"unsupported output format(s): {', '.join(sorted(invalid))}")
    return normalized


_PARAGRAPH_OPEN_RE = re.compile(r"^[a-z\u00e0-\u017f]")
_SENTENCE_END_RE = re.compile(r"[\.!?\u2026][\"'\)\]\u201c\u201d]?\s*$")
_NOISY_NOTEREF_RE = re.compile(
    r"(?P<body>[A-Za-z][A-Za-z.\-]*(?:-[A-Za-z]+)?[.!?]?)['’](?=(?:\s|$))"
)
EXPLICIT_NOTEREF_OPEN = "\x12"
EXPLICIT_NOTEREF_CLOSE = "\x13"


def _last_body_paragraph(elements: list[dict]) -> dict | None:
    for el in reversed(elements):
        if el.get("kind") == "paragraph":
            return el
    return None


def _first_body_element(elements: list[dict]) -> dict | None:
    for el in elements:
        if el.get("kind") in ("paragraph", "heading"):
            return el
    return None


def _footnote_numbers(elements: list[dict]) -> list[str]:
    nums = [str(e.get("number")) for e in elements if e.get("kind") == "footnote"]
    return sorted([n for n in nums if n.isdigit()], key=lambda n: int(n))


def _encode_cross_page_noterefs(text: str, page_no: int, numbers: list[str]) -> str:
    """Encode quote-shaped note markers with their original page number.

    This is needed when the first paragraph of page B is merged onto page A:
    its noterefs must still point to page B's footnotes.
    """
    if not numbers:
        return text
    idx = 0

    def repl(m: re.Match) -> str:
        nonlocal idx
        if idx >= len(numbers):
            return m.group(0)
        num = numbers[idx]
        idx += 1
        return (
            f"{m.group('body')}"
            f"{EXPLICIT_NOTEREF_OPEN}{page_no}:{num}{EXPLICIT_NOTEREF_CLOSE}"
        )

    return _NOISY_NOTEREF_RE.sub(repl, text)


def _merge_cross_page_paragraphs(ocr_pages: list[dict]) -> None:
    """Stitch the last paragraph of each page to the first paragraph of the next
    when the sentence clearly continues across the page break.
    """
    merged = 0
    for i in range(len(ocr_pages) - 1):
        elems_a = ocr_pages[i].get("elements", [])
        elems_b = ocr_pages[i + 1].get("elements", [])
        last = _last_body_paragraph(elems_a)
        first = _first_body_element(elems_b)
        if not last or not first or first.get("kind") != "paragraph":
            continue
        a_text = last.get("text", "").rstrip()
        b_text = first.get("text", "").lstrip()
        if not a_text or not b_text:
            continue
        # Conditions:
        #   - page A's last paragraph does NOT end with sentence-final punctuation
        #   - page B's first paragraph starts with a lowercase letter (clear
        #     mid-sentence continuation)
        if _SENTENCE_END_RE.search(a_text):
            continue
        if not _PARAGRAPH_OPEN_RE.match(b_text):
            continue
        b_text = _encode_cross_page_noterefs(
            b_text,
            int(ocr_pages[i + 1].get("pno", i + 2)),
            _footnote_numbers(elems_b),
        )
        last["text"] = a_text + " " + b_text
        a_conf = last.get("conf", -1)
        b_conf = first.get("conf", -1)
        if a_conf >= 0 and b_conf >= 0:
            last["conf"] = min(float(a_conf), float(b_conf))
        # Drop the now-merged paragraph from page B.
        elems_b.remove(first)
        merged += 1
    if merged:
        log(f"  merged {merged} cross-page paragraphs")


def _structure_and_assemble(cfg: BookConfig, *, pdf_doc, pages, ocr_pages):
    """Run structure + post-processors + chapter assembly.

    Shared between ``build_book`` and the standalone ``quire qc`` path
    so both produce identical chapter objects from the same OCR cache.
    """
    body_size = estimate_global_body_size(pages)
    engine = cfg.ocr_engine.lower()
    pp_registry.run_pre_structure(cfg, ocr_pages)
    if engine in {"text", "pdf", "pymupdf"}:
        for i, op in enumerate(ocr_pages):
            op["elements"] = structure_page_text(pages[i], [], body_size)
    elif engine in {"vision", "tesseract"}:
        for i, op in enumerate(ocr_pages):
            op["pno"] = i + 1
            op.setdefault("printed_page", pages[i].get("printed_page"))
            op["elements"] = structure_page_vision(
                pdf_doc[i], op, body_size, heuristics=cfg.book_heuristics,
            )
    _merge_cross_page_paragraphs(ocr_pages)
    pp_registry.run_post_structure(cfg, ocr_pages)
    return assemble_chapters(pages, ocr_pages, pdf_doc, cfg=cfg)


def _build_rendered_chapters_for_qc(cfg: BookConfig):
    """Re-run only the steps needed to produce ``(rendered, chapters)``.

    Used by the standalone ``quire qc`` CLI when no in-memory build
    context exists. Re-uses OCR / structure caches so a typical
    invocation only does chapter assembly + chapter rendering.
    """
    cfg.ensure_artifact_dirs()
    reset_for_build(cfg)
    ocr_corrections.reset_report()

    log("extracting PDF text layer with PyMuPDF")
    pdf_doc, pages = extract_all(str(cfg.pdf_path))
    try:
        ocr_pages = _run_engine(cfg, pages, force=False, retry_failed=False)
        if len(ocr_pages) != len(pages):
            raise RuntimeError(
                f"page count mismatch: PDF has {len(pages)} pages but OCR returned {len(ocr_pages)} pages"
            )
        chapters = _structure_and_assemble(
            cfg, pdf_doc=pdf_doc, pages=pages, ocr_pages=ocr_pages,
        )
        rendered: list[tuple[str, str]] = []
        for c in chapters:
            xhtml, _emitted = render_chapter(c, all_chapters=chapters, cfg=cfg)
            rendered.append((c.slug, xhtml))
    finally:
        try:
            pdf_doc.close()
        except Exception:  # noqa: BLE001
            pass
    return rendered, chapters


def _maybe_run_qc(
    cfg: BookConfig,
    *,
    rendered: list[tuple[str, str]],
    chapters,
) -> None:
    """Auto-run AI QC when ``[qc] enabled = true`` in book.toml.

    The QC stage is intentionally a soft dependency: if it fails (no
    API key, network error, etc.) we log and continue with the build
    rather than abort. Corrections it writes are picked up by the
    next ``load_qc_fixes`` call in the build flow.
    """
    settings = cfg.qc_settings
    if settings is None or not settings.enabled:
        return
    try:
        from .qc import run_qc
        from .qc.page_text import build_page_map, extract_page_texts
    except ImportError as e:
        log(f"qc: skipped (import error: {e})")
        return
    try:
        page_map = build_page_map(chapters)
        page_texts = extract_page_texts(rendered, page_map)
        if not page_texts:
            log("qc: no page texts found; skipped")
            return
        log(f"qc: running on {len(page_texts)} page(s) via {settings.engine}")
        result = run_qc(cfg, page_texts=page_texts)
        log(f"  {result.summary_line()}")
    except Exception as e:
        log(f"qc: failed ({e}); continuing build without AI corrections")
        log_event("qc_pipeline_error", slug=cfg.slug, error=str(e))


def render_page_images(cfg: BookConfig, dpi: int = 160) -> int:
    """Render every PDF page as PNG into the book's artifact image cache."""
    out = cfg.page_images_dir
    out.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(cfg.pdf_path))
    written = 0
    for pno in range(1, len(doc) + 1):
        target = out / f"p{pno:03d}.png"
        if target.exists():
            continue
        pix = doc[pno - 1].get_pixmap(dpi=dpi)
        pix.save(str(target))
        written += 1
    log(f"rendered {written} page images to {out}")
    return written


def build_book(
    cfg: BookConfig,
    *,
    force_ocr: bool = False,
    retry_failed: bool = False,
    render_pages: bool = False,
    formats: list[str] | tuple[str, ...] | set[str] | None = None,
) -> dict[str, Path]:
    """Build the requested outputs end-to-end."""
    selected_formats = _normalize_formats(formats, cfg)
    log(f"=== building '{cfg.slug}' ({cfg.title}) ===")
    log(f"source PDF: {cfg.pdf_path}")
    log(f"language: {cfg.language}; OCR engine: {cfg.ocr_engine}; OCR langs: {cfg.ocr_languages}")
    log(f"output formats: {', '.join(sorted(selected_formats))}")
    if "epub" in selected_formats:
        log(f"target EPUB: {cfg.epub_path}")
    cfg.ensure_artifact_dirs()
    for w in cfg.warnings:
        log(f"warning: {w}")
    reset_for_build(cfg)
    ocr_corrections.reset_report()
    if cfg.structure_headings:
        log(f"loaded {len(cfg.structure_headings)} configured headings")
    else:
        log("no configured headings; using geometry-only chapter detection")

    with file_lock(cfg.lock_path, timeout=None):
        return _build_book_locked(
            cfg,
            selected_formats=selected_formats,
            force_ocr=force_ocr,
            retry_failed=retry_failed,
            render_pages_flag=render_pages,
        )


def _build_book_locked(
    cfg: BookConfig,
    *,
    selected_formats: set[str],
    force_ocr: bool,
    retry_failed: bool,
    render_pages_flag: bool,
) -> dict[str, Path]:
    if render_pages_flag:
        render_page_images(cfg)

    # 1. Extract PDF text layer
    log("extracting PDF text layer with PyMuPDF")
    pdf_doc, pages = extract_all(str(cfg.pdf_path))
    log(f"  {len(pages)} pages parsed")
    try:
        return _build_book_with_doc(
            cfg,
            pdf_doc=pdf_doc,
            pages=pages,
            selected_formats=selected_formats,
            force_ocr=force_ocr,
            retry_failed=retry_failed,
        )
    finally:
        try:
            pdf_doc.close()
        except Exception:  # noqa: BLE001
            pass


def _build_book_with_doc(
    cfg: BookConfig,
    *,
    pdf_doc,
    pages,
    selected_formats: set[str],
    force_ocr: bool,
    retry_failed: bool,
) -> dict[str, Path]:
    # 3. Resolve global body font size
    body_size = estimate_global_body_size(pages)
    log(f"  global body size: {body_size:.2f}pt")

    engine = cfg.ocr_engine.lower()
    ocr_pages = _run_engine(cfg, pages, force=force_ocr, retry_failed=retry_failed)
    if len(ocr_pages) != len(pages):
        raise RuntimeError(
            f"page count mismatch: PDF has {len(pages)} pages but OCR returned {len(ocr_pages)} pages"
        )

    # 4. Run pre-structure post-processors that bootstrap shared state
    #    (e.g. glossary auto-extract → expand vocabulary).
    pp_registry.run_pre_structure(cfg, ocr_pages)

    # 5. Per-page structuring (uses Vision OCR + PDF text-layer cues)
    if engine in {"text", "pdf", "pymupdf"}:
        log("structuring elements per page (PDF text layer)")
        for i, op in enumerate(ocr_pages):
            op["elements"] = structure_page_text(pages[i], [], body_size)
    elif engine in {"vision", "tesseract"}:
        log(f"structuring elements per page ({engine}-based)")
        for i, op in enumerate(ocr_pages):
            op["pno"] = i + 1
            op.setdefault("printed_page", pages[i].get("printed_page"))
            op["elements"] = structure_page_vision(
                pdf_doc[i], op, body_size, heuristics=cfg.book_heuristics,
            )

    # 5b. Cross-page paragraph merge: if page N's last paragraph ends without
    # sentence-final punctuation AND page N+1's first paragraph isn't a
    # heading and starts lowercase / mid-sentence, merge them.
    _merge_cross_page_paragraphs(ocr_pages)

    # 6. Per-element post-processors (mojibake cleanup, canonical Quran etc).
    pp_registry.run_post_structure(cfg, ocr_pages)
    ocr_corrections.write_report(cfg)
    correction_count = ocr_corrections.correction_count()
    if correction_count:
        log(f"recorded {correction_count} OCR correction(s) at {cfg.ocr_corrections_path}")

    # 7. Assemble chapters
    log("assembling chapters")
    chapters = assemble_chapters(pages, ocr_pages, pdf_doc, cfg=cfg)
    log(f"  {len(chapters)} chapters")

    emitted_pp_total: set[int] = set()
    outputs: dict[str, Path] = {}
    if "epub" in selected_formats:
        # 8. Render XHTML
        log("rendering chapter XHTML")
        rendered: list[tuple[str, str]] = []
        for c in chapters:
            xhtml, emitted = render_chapter(c, all_chapters=chapters, cfg=cfg)
            rendered.append((c.slug, xhtml))
            emitted_pp_total |= emitted
        log(f"  {len(emitted_pp_total)} unique printed-page anchors")

        # 8b. Optional AI-assisted QC.
        # When [qc] enabled = true in book.toml, send each rendered
        # page image + extracted text to a VLM (Gemini by default) and
        # merge validated find/replace corrections into qc_fixes.toml.
        # Those corrections are then consumed by the typography stage
        # right below. See quire/qc/ for the full implementation.
        _maybe_run_qc(cfg, rendered=rendered, chapters=chapters)

        # 9. Post-render typography fixes.
        # These operate on the rendered XHTML, where OCR / typesetting
        # artifacts surface only after footnote refs are inlined:
        #   - hyphen-across-line-break stitching
        #   - loose footnote-digit -> Unicode superscript
        #   - stray footnote-quote stripping (Name" verb, word." Next)
        #   - per-book qc_fixes.toml HTML-layer substitutions
        # See quire.render.typography for the full safety contract.
        qc_fixes = load_qc_fixes(cfg.qc_fixes_path)
        vocab_text = "\n".join(html for _slug, html in rendered)
        vocab = _build_typography_vocab(vocab_text)
        rendered, typo_report = apply_typography_fixes(
            rendered, vocab=vocab, qc_fixes=qc_fixes,
        )
        log(f"  {typo_report.summary_line()}")
        if typo_report.hyphen_examples:
            for ex in typo_report.hyphen_examples[:5]:
                log(f"    hyphen: {ex}")
        if typo_report.qc_tag_skipped_entries:
            log(
                "  qc_fixes.toml: "
                f"{len(typo_report.qc_tag_skipped_entries)} entry/entries "
                "skipped because their find span crosses an inline tag "
                "(<em>, <span>, etc.). Rewrite the find string to start/end "
                "outside the markup so the substitution can apply."
            )
            for entry in typo_report.qc_tag_skipped_entries[:5]:
                preview = entry if len(entry) <= 80 else entry[:77] + "..."
                log(f"    tag-skipped: {preview!r}")
        if typo_report.qc_no_op_entries:
            log_event(
                "qc_fixes_no_op",
                slug=cfg.slug,
                count=len(typo_report.qc_no_op_entries),
                examples=typo_report.qc_no_op_entries[:10],
            )

        # 10. Cover
        cover_jpeg = cfg.caches_dir / "cover.jpg"
        render_cover_jpeg(str(cfg.pdf_path), str(cover_jpeg), page_index=cfg.cover_pdf_page - 1)

        # 11. Package EPUB
        log("packaging EPUB")
        build_epub(
            cfg=cfg,
            chapters=chapters,
            rendered=rendered,
            cover_jpeg=cover_jpeg,
            emitted_printed_pages=emitted_pp_total,
        )
        outputs["epub"] = cfg.epub_path
        log(f"wrote {cfg.epub_path} ({cfg.epub_path.stat().st_size / 1024:.1f} KB)")

    extra_formats = selected_formats - {"epub"}
    if extra_formats:
        log(f"writing export format(s): {', '.join(sorted(extra_formats))}")
        outputs.update(write_exports(cfg, chapters, extra_formats))
        for fmt, path in sorted(outputs.items()):
            if fmt != "epub":
                log(f"wrote {fmt}: {path}")

    log_event(
        "build_done",
        slug=cfg.slug,
        outputs=sorted(outputs),
        chapters=len(chapters),
        pages=len(pages),
        ocr_engine=cfg.ocr_engine,
    )
    return outputs
