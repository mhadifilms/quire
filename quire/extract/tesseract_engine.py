"""Tesseract 5 OCR engine.

Tesseract is the cross-platform default for Quire: 163 trained languages
(including ``ara``, ``fas``, ``urd``, ``heb``, ``chi_sim``, ``chi_tra``,
``ell``, ``rus`` etc.), Apache-2.0, CPU-friendly, no GPU required. We use
the Python binding :mod:`pytesseract` which shells out to the ``tesseract``
binary (must be on ``$PATH``).

The engine output is structurally identical to the macOS Vision engine
output so the rest of the pipeline (cache, structure stage, Arabic refine,
audit, packager) doesn't care which engine produced it.
"""

from __future__ import annotations

import concurrent.futures as cf
import statistics
from typing import Any

import fitz
from PIL import Image

__all__ = ["ocr_pdf_tesseract", "ocr_page_tesseract", "TESSERACT_LANG_MAP"]


# Map BCP-47 language tags to the 3-letter codes Tesseract expects. Falls
# through unchanged if the caller already passed a tesseract code (e.g.
# ``ara``, ``fas``). Unknown codes are dropped with a warning at call time.
TESSERACT_LANG_MAP: dict[str, str] = {
    "en": "eng", "en-us": "eng", "en-gb": "eng",
    "ar": "ara", "ar-sa": "ara", "ar-eg": "ara",
    "fa": "fas", "fa-ir": "fas",
    "ur": "urd", "ur-pk": "urd",
    "he": "heb", "he-il": "heb",
    "el": "ell", "el-gr": "ell",
    "ru": "rus", "ru-ru": "rus",
    "uk": "ukr",
    "tr": "tur",
    "fr": "fra", "fr-fr": "fra",
    "de": "deu", "de-de": "deu",
    "it": "ita",
    "es": "spa", "es-es": "spa", "es-mx": "spa",
    "pt": "por", "pt-br": "por", "pt-pt": "por",
    "pl": "pol",
    "nl": "nld",
    "ja": "jpn",
    "ko": "kor",
    "zh-cn": "chi_sim", "zh-hans": "chi_sim", "zh": "chi_sim",
    "zh-tw": "chi_tra", "zh-hant": "chi_tra",
    "hi": "hin",
    "bn": "ben",
    "ta": "tam",
    "th": "tha",
    "vi": "vie",
    "id": "ind",
    "ms": "msa",
}


def _normalize_langs(langs: list[str]) -> list[str]:
    """Translate BCP-47 codes to Tesseract codes, dedupe, preserve order."""
    out: list[str] = []
    for raw in langs or []:
        key = str(raw).lower()
        code = TESSERACT_LANG_MAP.get(key) or TESSERACT_LANG_MAP.get(key.split("-")[0]) or key
        if code not in out:
            out.append(code)
    return out or ["eng"]


def _is_rtl_lang_code(code: str) -> bool:
    return code in {"ara", "fas", "urd", "heb", "yid", "pus", "snd", "div"}


def _render_page_to_pil(pdf_path: str, pno: int, scale: int) -> tuple[Image.Image, float, float]:
    """Render PDF page ``pno`` (1-indexed) to a PIL image at the requested scale.

    Returns ``(image, page_width_pt, page_height_pt)``. Always closes the
    underlying ``fitz.Document``.
    """
    doc = fitz.open(pdf_path)
    try:
        page = doc[pno - 1]
        w_pt, h_pt = page.rect.width, page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()
    return img, w_pt, h_pt


def _group_words_to_lines(data: dict[str, Any], scale: int) -> list[dict[str, Any]]:
    """Convert ``pytesseract.image_to_data`` dict output into per-line records.

    Coordinates are converted from pixels to PDF points by dividing by
    ``scale``. Lines with negative confidence (Tesseract's sentinel for
    "no recognition attempted") are dropped.
    """
    n = len(data.get("text", []))
    groups: dict[tuple[int, int, int, int], list[int]] = {}
    for i in range(n):
        # ``pytesseract`` returns confidences as strings in some versions.
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        text = data["text"][i] or ""
        if not text.strip() or conf < 0:
            continue
        key = (
            int(data["page_num"][i]),
            int(data["block_num"][i]),
            int(data["par_num"][i]),
            int(data["line_num"][i]),
        )
        groups.setdefault(key, []).append(i)

    lines: list[dict[str, Any]] = []
    for _key, idxs in groups.items():
        words = [data["text"][i] for i in idxs]
        text = " ".join(w for w in words if w.strip())
        if not text:
            continue
        xs0 = [int(data["left"][i]) for i in idxs]
        ys0 = [int(data["top"][i]) for i in idxs]
        xs1 = [int(data["left"][i]) + int(data["width"][i]) for i in idxs]
        ys1 = [int(data["top"][i]) + int(data["height"][i]) for i in idxs]
        confs = [float(data["conf"][i]) for i in idxs if float(data["conf"][i]) >= 0]
        lines.append({
            "text": text,
            "x0": min(xs0) / scale,
            "y0": min(ys0) / scale,
            "x1": max(xs1) / scale,
            "y1": max(ys1) / scale,
            "conf": float(statistics.mean(confs)) if confs else 0.0,
        })
    lines.sort(key=lambda L: (round(L["y0"], 1), L["x0"]))
    return lines


def _cluster_arabic_blocks(ar_lines: list[dict]) -> list[dict]:
    """Lazy wrapper around the existing Vision-side block clusterer."""
    if not ar_lines:
        return []
    from .ocr import cluster_arabic_blocks
    heights = [L["y1"] - L["y0"] for L in ar_lines if L["y1"] > L["y0"]]
    avg_h = statistics.median(heights) if heights else 12.0
    blocks = cluster_arabic_blocks(ar_lines, avg_h)
    for b in blocks:
        if 0 < b.get("conf", -1) <= 1.0:
            b["conf"] *= 100
    return blocks


def _split_langs_by_directionality(langs: list[str]) -> tuple[list[str], list[str]]:
    """Partition a tesseract-code list into (LTR, RTL) sublists, preserving order."""
    ltr = [c for c in langs if not _is_rtl_lang_code(c)]
    rtl = [c for c in langs if _is_rtl_lang_code(c)]
    return ltr, rtl


def _run_tesseract_pass(
    img: Image.Image,
    lang_arg: str,
    config: str,
    scale: int,
) -> list[dict[str, Any]]:
    """One ``image_to_data`` call → list of line dicts in PDF points."""
    import pytesseract
    from pytesseract import Output

    data = pytesseract.image_to_data(
        img, lang=lang_arg, config=config, output_type=Output.DICT,
    )
    return _group_words_to_lines(data, scale)


def _arabic_char_ratio(text: str) -> float:
    """Fraction of letters in ``text`` that are Arabic-block characters.

    Used to discard lines where Tesseract's Arabic LSTM hallucinated Arabic
    glyphs in regions of Latin text (a common failure mode when running
    ``lang=ara`` on a mixed-script page).
    """
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    total = ar + latin
    return ar / total if total > 0 else 0.0


def _looks_like_arabic_text(text: str, *, min_chars: int = 3, min_ratio: float = 0.6) -> bool:
    """True when the line plausibly contains real Arabic, not OCR noise.

    Reasoning:
      - ``ara`` traineddata happily outputs Arabic glyphs even when shown
        Latin text — those outputs are mostly disjoint single chars or have
        a high latin-to-arabic ratio.
      - Real Arabic prose tends to be ≥3 chars and dominated by the Arabic
        Unicode block.
    """
    ar = sum(1 for c in text if "\u0600" <= c <= "\u06ff")
    return ar >= min_chars and _arabic_char_ratio(text) >= min_ratio


def _is_marginal_zone(L: dict, page_h_pt: float, *,
                       top_pt: float = 55.0, bottom_pt: float = 35.0) -> bool:
    """Lines whose centre falls inside the running-header or footer band."""
    yc = (L["y0"] + L["y1"]) / 2
    return yc < top_pt or yc > (page_h_pt - bottom_pt)


def ocr_page_tesseract(
    pdf_path: str,
    pno: int,
    *,
    languages: list[str] | None = None,
    scale: int = 3,
    psm_ltr: int = 3,
    psm_rtl: int = 3,
    oem: int = 1,
    retries: int = 1,
    min_arabic_conf: float = 0.0,
    min_arabic_char_ratio: float = 0.0,
    min_arabic_chars: int = 0,
    strip_margin_band: bool = False,
) -> dict[str, Any]:
    """OCR a single page with Tesseract. Returns Vision-shaped page dict.

    Knobs (callers normally use the defaults):

    - ``psm_ltr`` PSM for the LTR pass (default 3 = automatic with OSD).
    - ``psm_rtl`` PSM for the RTL pass (default 6 = single uniform block;
      gives better recall on inline Arabic phrases than auto-segment).
    - ``oem`` OCR engine mode (default 1 = LSTM only, recommended for 5.x).
    - ``min_arabic_conf`` drop Arabic lines whose Tesseract confidence is
      below this threshold (Tesseract 0–100 scale). The ``ara`` model
      hallucinates Arabic in Latin text but does so at conf=0.
    - ``min_arabic_char_ratio`` drop Arabic lines where less than this
      fraction of the letters are Arabic-block characters.
    - ``min_arabic_chars`` drop Arabic lines with fewer than this many
      Arabic-block characters (filters page-number/header glyph noise).
    - ``strip_margin_band`` drop Arabic lines that fall entirely in the
      running-header or footer band.
    - ``retries`` extra attempts on transient failures.

    Mixed-script pages: when the language list contains both LTR and RTL
    codes, Tesseract's combined ``lang=eng+ara`` mode reliably drops the
    minority script (the LSTM model biases toward the dominant script).
    We work around this by running one pass per script direction and
    merging the line-level outputs by position — same strategy macOS
    Vision uses internally.
    """
    langs = _normalize_langs(languages or ["eng"])
    ltr_langs, rtl_langs = _split_langs_by_directionality(langs)
    ltr_config = f"--oem {oem} --psm {psm_ltr}"
    rtl_config = f"--oem {oem} --psm {psm_rtl}"

    last_exc: Exception | None = None
    img = None
    w_pt = h_pt = 0.0
    for attempt in range(max(1, retries + 1)):
        try:
            img, w_pt, h_pt = _render_page_to_pil(pdf_path, pno, scale)
            ltr_lines: list[dict] = []
            rtl_lines: list[dict] = []
            try:
                if ltr_langs:
                    ltr_lines = _run_tesseract_pass(
                        img, "+".join(ltr_langs), ltr_config, scale,
                    )
                if rtl_langs:
                    rtl_lines = _run_tesseract_pass(
                        img, "+".join(rtl_langs), rtl_config, scale,
                    )
            finally:
                img.close()
                img = None
            break
        except Exception as e:  # noqa: BLE001
            if img is not None:
                try:
                    img.close()
                except Exception:  # noqa: BLE001
                    pass
                img = None
            last_exc = e
            if attempt >= retries:
                return {
                    "pno": pno,
                    "en_lines": [],
                    "ar_lines": [],
                    "arabic_blocks": [],
                    "page_size_pt": (612.0, 792.0),
                    "error": f"tesseract failure (pno={pno}): {type(e).__name__}: {e}",
                }
    else:  # pragma: no cover — defensive fallthrough
        if last_exc is not None:
            raise last_exc

    en_lines: list[dict] = []
    ar_lines: list[dict] = []
    # The LTR pass output rarely contains real Arabic (the eng model can't
    # recognize Arabic glyphs); any "arabic-script" detection here is
    # essentially always Latin-with-diacritics. Route it to en_lines.
    for L in ltr_lines:
        en_lines.append(L)

    # Build a quick lookup of high-confidence English line bboxes. The
    # Arabic LSTM hallucinates Arabic in Latin text, so any RTL-pass line
    # whose bbox nearly *duplicates* a high-conf English line is noise.
    # Inline Arabic phrases on an English baseline are KEPT — they share a
    # baseline but cover only a fraction of the horizontal extent.
    high_conf_en = [L for L in en_lines if L.get("conf", 0) >= 80.0]

    def _duplicates_high_conf_english(L: dict) -> bool:
        L_area = max(1e-6, (L["x1"] - L["x0"]) * (L["y1"] - L["y0"]))
        for E in high_conf_en:
            # Vertical overlap > 60% of the candidate's height (same baseline).
            v_overlap = max(0.0, min(L["y1"], E["y1"]) - max(L["y0"], E["y0"]))
            v_height = max(1e-6, L["y1"] - L["y0"])
            if v_overlap / v_height < 0.6:
                continue
            # Intersection / candidate-area > 0.85 → the English line
            # covers nearly the entire Arabic candidate horizontally too,
            # which means the Arabic is a hallucinated overlay on the
            # whole English line (not a discrete inline phrase).
            x_overlap = max(0.0, min(L["x1"], E["x1"]) - max(L["x0"], E["x0"]))
            inter = x_overlap * v_overlap
            if inter / L_area >= 0.85:
                return True
        return False

    # The RTL pass is the noise source: ara/fas LSTMs hallucinate Arabic
    # in Latin text at low confidence. Apply a confidence + script-ratio +
    # margin-zone filter + overlap-with-English filter so the cache stores
    # only plausible Arabic lines.
    for L in rtl_lines:
        text = L["text"]
        has_ar_glyph = any("\u0600" <= ch <= "\u06ff" for ch in text)
        if not has_ar_glyph:
            # Numbers / pure-Latin from RTL pass — already in en_lines.
            continue
        if L.get("conf", 0) < min_arabic_conf:
            continue
        if not _looks_like_arabic_text(
            text, min_chars=min_arabic_chars, min_ratio=min_arabic_char_ratio,
        ):
            continue
        if strip_margin_band and _is_marginal_zone(L, h_pt):
            continue
        # Only fight English-overlap noise on low/mid confidence Arabic;
        # high-conf Arabic is reliable enough to trust even when it shares
        # a baseline with an English line (i.e. mixed-script context).
        if L.get("conf", 0) < 60.0 and _duplicates_high_conf_english(L):
            continue
        ar_lines.append(L)

    arabic_blocks = _cluster_arabic_blocks(ar_lines)
    return {
        "pno": pno,
        "page_size_pt": (w_pt, h_pt),
        "en_lines": en_lines,
        "ar_lines": ar_lines,
        "arabic_blocks": arabic_blocks,
    }


def ocr_pdf_tesseract(
    pdf_path: str,
    *,
    languages: list[str] | None = None,
    workers: int = 4,
    scale: int = 3,
    psm_ltr: int = 3,
    psm_rtl: int = 3,
    oem: int = 1,
    retries: int = 1,
    min_arabic_conf: float = 0.0,
    min_arabic_char_ratio: float = 0.0,
    min_arabic_chars: int = 0,
    strip_margin_band: bool = False,
    page_numbers: list[int] | None = None,
    progress=None,
) -> list[dict | None]:
    """OCR every page (or just ``page_numbers``) of a PDF with Tesseract."""
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
    finally:
        doc.close()
    pages: list[dict | None] = [None] * n
    pno_list = list(page_numbers) if page_numbers is not None else list(range(1, n + 1))
    pno_list = [p for p in pno_list if 1 <= p <= n]
    if not pno_list:
        return pages
    workers = max(1, int(workers))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                ocr_page_tesseract, pdf_path, pno,
                languages=languages, scale=scale,
                psm_ltr=psm_ltr, psm_rtl=psm_rtl, oem=oem, retries=retries,
                min_arabic_conf=min_arabic_conf,
                min_arabic_char_ratio=min_arabic_char_ratio,
                min_arabic_chars=min_arabic_chars,
                strip_margin_band=strip_margin_band,
            ): pno
            for pno in pno_list
        }
        done = 0
        for fut in cf.as_completed(futs):
            pno = futs[fut]
            try:
                pages[pno - 1] = fut.result()
            except Exception as e:  # noqa: BLE001
                pages[pno - 1] = {
                    "pno": pno,
                    "en_lines": [],
                    "ar_lines": [],
                    "arabic_blocks": [],
                    "page_size_pt": (612.0, 792.0),
                    "error": f"{type(e).__name__}: {e}",
                }
            done += 1
            if progress:
                progress(done, len(pno_list))
    return pages
