"""Per-page PDF text-layer extraction.

Produces a structured per-page representation from ordinary PDF text spans.
Some heuristics are tuned for ABBYY/Arial mojibake-heavy religious texts, but
the output shape is generic and is also used by the fast `ocr.engine = "text"`
pipeline for PDFs with a usable text layer:

  Page = {
    'pno': int,
    'page_size': (w, h),
    'header': str | None,             # the running header text we strip
    'printed_page': int | None,       # printed page number parsed from header
    'body_lines': [Line],             # English lines in body zone (mojibake removed)
    'footnote_lines': [Line],         # English lines in footnote zone
    'mojibake_y_ranges': [(y0,y1)],   # zones where Arabic mojibake was dropped
  }

  Line = {
    'y': float, 'x0': float, 'x1': float,
    'spans': [Span],
    'text': str,                      # joined text of kept spans
    'median_size': float,
    'is_centered': bool,
  }

  Span = {
    'text': str, 'font': str, 'size': float, 'flags': int, 'bbox': (x0,y0,x1,y1),
    'bold': bool, 'italic': bool, 'is_mojibake': bool,
  }
"""

from __future__ import annotations

import re
import statistics

import fitz

ARABIC_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")
MOJIBAKE_HINTS = set("^\\{}<>|~`")
SOFT_HYPHEN = "\u00ad"

# Italic flag (bit 1, value 2) and bold flag (bit 4, value 16) per PyMuPDF docs.
FLAG_ITALIC = 1 << 1
FLAG_BOLD = 1 << 4
FLAG_SUPERSCRIPT = 1 << 0


def is_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def looks_like_mojibake(text: str, font: str) -> bool:
    """Detect ABBYY-OCR Arabic mojibake at the span level.

    In this PDF, real Arabic was encoded by ABBYY as glyph-index runs in
    Arial-family fonts, which PyMuPDF decodes into nonsense Latin/punctuation.
    The only legitimate Arial usage in the book is:
      - Arial-ItalicMT: italic Latin transliterations (e.g. ``[arafat]``).
      - Single-character Arial punctuation/digits used as quotes or footnote
        markers.

    Any other Arial span longer than two characters is mojibake.
    """
    t = text.strip()
    if not t:
        return False
    if is_arabic(t):
        return False
    if any(ch in MOJIBAKE_HINTS for ch in t):
        return True

    is_arial = font.startswith("Arial") and not font.startswith("Arial Unicode")
    if is_arial:
        if font == "Arial-ItalicMT":
            # Italic transliterations are legitimate, but reject ones with
            # embedded digits or hint chars (already handled above).
            if re.search(r"\d.*[a-zA-Z]|[a-zA-Z].*\d", t):
                return True
            return False
        # All other Arial faces: keep only short single tokens (digits,
        # punctuation, single letters that act as footnote refs).
        if len(t) <= 2:
            if re.fullmatch(r"[\d\*•\(\)\[\]\.,:;!\?\-\u2013\u2014]+", t):
                return False
            if re.fullmatch(r"[A-Za-z]", t):
                return False
            # Two-character tokens like "1.", "2)" are fine; otherwise treat
            # as mojibake (Arabic-letter glyph indexes).
            if not re.fullmatch(r"[\dA-Za-z][\.,)\]]?", t):
                return True
            return False
        return True
    return False


def line_text(line: dict) -> str:
    return "".join(s.get("text", "") for s in line.get("spans", []))


def line_bbox(line: dict) -> tuple[float, float, float, float]:
    return tuple(line["bbox"])


def collect_lines(page: fitz.Page) -> list[dict]:
    """Return raw PyMuPDF lines (as dicts) sorted top-to-bottom, left-to-right."""
    out = []
    d = page.get_text("dict")
    for block in d["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            out.append({"raw": line, "spans": spans, "bbox": tuple(line["bbox"])})
    out.sort(key=lambda L: (round(L["bbox"][1] / 2), L["bbox"][0]))
    return out


def make_span(s: dict) -> dict:
    flags = s.get("flags", 0)
    font = s.get("font", "")
    return {
        "text": s.get("text", ""),
        "font": font,
        "size": float(s.get("size", 0.0)),
        "flags": flags,
        "bbox": tuple(s.get("bbox", (0, 0, 0, 0))),
        "bold": bool(flags & FLAG_BOLD) or "Bold" in font,
        "italic": bool(flags & FLAG_ITALIC) or "Italic" in font or "Oblique" in font,
        "superscript": bool(flags & FLAG_SUPERSCRIPT),
        "is_mojibake": looks_like_mojibake(s.get("text", ""), font),
    }


HEADER_RE = re.compile(
    r"^\s*(?:(?P<num>\d{1,3})\s+)?(?P<title>(?:THE\s+)?[A-Z][A-Z'’ \-]{2,})(?:\s+(?P<num2>\d{1,3}))?\s*$"
)


def parse_running_header(line: dict, page_no: int) -> tuple[str | None, int | None]:
    """If the line looks like a running header, return (title, printed_page)."""
    text = "".join(s["text"] for s in line["spans"]).strip()
    if not text:
        return None, None
    if line["bbox"][1] > 60:
        return None, None
    m = HEADER_RE.match(text)
    if not m:
        return None, None
    num = m.group("num") or m.group("num2")
    try:
        printed = int(num) if num else None
    except ValueError:
        printed = None
    title = m.group("title").strip()
    return title or None, printed


def _first_text_span(line: dict) -> dict | None:
    for s in line.get("all_spans", []):
        if s["text"].strip():
            return s
    return None


def _is_footnote_marker_line(line: dict, body_size: float) -> bool:
    """A line whose first non-space span is a tiny digit (< body * 0.70) and
    text matches r'^\\d{1,3}[\\s.,)]' or r'^[*•]' is the start of a numbered
    footnote.
    """
    s0 = _first_text_span(line)
    if s0 is None:
        return False
    if s0["size"] >= body_size * 0.70:
        return False
    text = s0["text"].strip()
    if not text:
        return False
    if re.fullmatch(r"\d{1,3}[\.,)]?", text) or text in {"*", "•"}:
        return True
    return False


def find_footnote_boundary(lines: list[dict], body_size: float, page_height: float) -> int | None:
    """Return the index where the footnote zone begins.

    The boundary heuristic is: in the bottom 30% of the page, find the first
    line whose leading span is a small footnote marker (tiny digit / asterisk).
    If that fails, fall back to a sustained median-size drop.
    """
    bottom_threshold_y = page_height * 0.70
    for i, ln in enumerate(lines):
        if ln["y"] < bottom_threshold_y:
            continue
        if _is_footnote_marker_line(ln, body_size):
            return i

    # Fallback: median-size sustained drop (handles pages whose footnote
    # numbers were merged into adjacent spans).
    threshold = body_size * 0.82
    for i, ln in enumerate(lines):
        if ln["y"] < page_height * 0.50:
            continue
        size = ln["median_size"]
        if size <= 0:
            raw_sizes = [s["size"] for s in ln.get("all_spans", []) if s["text"].strip()]
            if not raw_sizes:
                continue
            size = statistics.median(raw_sizes)
        if size < threshold:
            tail_sizes: list[float] = []
            for L in lines[i:]:
                ms = L["median_size"]
                if ms <= 0:
                    raws = [s["size"] for s in L.get("all_spans", []) if s["text"].strip()]
                    if not raws:
                        continue
                    ms = statistics.median(raws)
                tail_sizes.append(ms)
            if tail_sizes and statistics.median(tail_sizes) < threshold + 1.0:
                return i
    return None


def is_centered(
    line: dict, page_width: float, body_left: float, body_right: float, tol: float = 18.0
) -> bool:
    """A line is centered if it is materially narrower than the body width
    AND its center sits near the body center.
    """
    x0, _, x1, _ = line["bbox"]
    line_w = x1 - x0
    body_w = max(body_right - body_left, 1.0)
    if line_w > body_w * 0.78:
        return False
    line_center = (x0 + x1) / 2
    body_center = (body_left + body_right) / 2
    return abs(line_center - body_center) < tol


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def line_dict(line: dict, page_width: float, body_left: float, body_right: float) -> dict:
    spans = [make_span(s) for s in line["spans"]]
    # Keep all non-mojibake spans, including space-only spans, so word
    # boundaries are preserved when we join.
    kept = [s for s in spans if not s["is_mojibake"]]
    text = normalize_text("".join(s["text"] for s in kept))
    text_sizes = [s["size"] for s in kept if s["text"].strip()]
    if not text_sizes:
        text_sizes = [s["size"] for s in spans if s["text"].strip()]
    median_size = statistics.median(text_sizes) if text_sizes else 0.0
    bbox = line["bbox"]
    return {
        "y": bbox[1],
        "y_bottom": bbox[3],
        "x0": bbox[0],
        "x1": bbox[2],
        "spans": kept,
        "all_spans": spans,
        "text": text,
        "median_size": median_size,
        "is_centered": is_centered(line, page_width, body_left, body_right),
        "raw_bbox": bbox,
    }


def find_mojibake_zones(lines: list[dict]) -> list[tuple[float, float]]:
    zones: list[tuple[float, float]] = []
    cur_start: float | None = None
    cur_end: float | None = None
    for L in lines:
        spans = L["all_spans"]
        if not spans:
            continue
        moji_chars = sum(len(s["text"]) for s in spans if s["is_mojibake"])
        good_chars = sum(len(s["text"].strip()) for s in spans if not s["is_mojibake"])
        if moji_chars > good_chars and moji_chars > 1:
            y0, y1 = L["raw_bbox"][1], L["raw_bbox"][3]
            if cur_start is None:
                cur_start, cur_end = y0, y1
            else:
                cur_end = max(cur_end if cur_end is not None else y1, y1)
        else:
            if cur_start is not None and cur_end is not None:
                zones.append((cur_start, cur_end))
                cur_start = cur_end = None
    if cur_start is not None and cur_end is not None:
        zones.append((cur_start, cur_end))
    # Merge close zones (within 6pt vertical gap).
    merged: list[tuple[float, float]] = []
    for z in zones:
        if merged and z[0] - merged[-1][1] < 6:
            merged[-1] = (merged[-1][0], z[1])
        else:
            merged.append(z)
    return merged


def extract_page(doc: fitz.Document, pno: int) -> dict:
    page = doc[pno - 1]
    pw, ph = page.rect.width, page.rect.height
    raw_lines = collect_lines(page)

    # Compute body size median from non-header, non-footer-ish region for stability.
    body_sizes = []
    for L in raw_lines:
        y0 = L["bbox"][1]
        if y0 < 70 or y0 > ph * 0.85:
            continue
        for s in L["spans"]:
            if s.get("text", "").strip():
                body_sizes.append(s.get("size", 0.0))
    body_size = statistics.median(body_sizes) if body_sizes else 11.2

    header_title = None
    printed_page = None
    body_start_idx = 0
    # Try the first few top-of-page lines: ABBYY sometimes leaves stray glyph
    # fragments above or beside the running header.
    for idx in range(min(4, len(raw_lines))):
        if raw_lines[idx]["bbox"][1] >= 65:
            break
        h_title, h_num = parse_running_header(raw_lines[idx], pno)
        if h_title:
            header_title = h_title
            printed_page = h_num
            body_start_idx = idx + 1
            break

    # Estimate body left/right margins from text-rich lines only (so single
    # short centered lines don't pull the margin in).
    candidate_lines = [
        L for L in raw_lines[body_start_idx:] if (L["bbox"][2] - L["bbox"][0]) > pw * 0.4
    ]
    if candidate_lines:
        body_left = min(L["bbox"][0] for L in candidate_lines)
        body_right = max(L["bbox"][2] for L in candidate_lines)
    else:
        body_left, body_right = 0.0, pw

    # Cooked lines for everything below header. We detect the footnote zone
    # using the *full* line stream (including lines that will later be dropped
    # because they fall inside an OCR-Arabic zone) so the boundary still works
    # when OCR happens to catch the first small-font footnote line.
    cooked = [line_dict(L, pw, body_left, body_right) for L in raw_lines[body_start_idx:]]
    cooked_nonempty = [L for L in cooked if L["text"] or L["all_spans"]]
    fn_idx = find_footnote_boundary(cooked_nonempty, body_size, ph)
    if fn_idx is None:
        body_lines = [L for L in cooked_nonempty if L["text"]]
        footnote_lines: list[dict] = []
    else:
        body_raw = cooked_nonempty[:fn_idx]
        fn_raw = cooked_nonempty[fn_idx:]
        body_lines = [L for L in body_raw if L["text"]]
        footnote_lines = [L for L in fn_raw if L["text"]]

    mojibake_zones = find_mojibake_zones(cooked)

    return {
        "pno": pno,
        "page_size": (pw, ph),
        "body_size": body_size,
        "header": header_title,
        "printed_page": printed_page,
        "body_lines": body_lines,
        "footnote_lines": footnote_lines,
        "mojibake_y_ranges": mojibake_zones,
    }


def extract_all(pdf_path: str) -> tuple[fitz.Document, list[dict]]:
    doc = fitz.open(pdf_path)
    pages = [extract_page(doc, pno) for pno in range(1, doc.page_count + 1)]
    return doc, pages
