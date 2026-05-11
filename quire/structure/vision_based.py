"""Per-page structuring using macOS Vision OCR as the canonical text source.

Why this exists
---------------
The original pipeline (``structure.py``) used the PDF text layer as the
source of truth for English. The text layer in this PDF is full of:
  * stray "i" / "I" / smart-quote glyphs that are really mis-mapped
    Arabic-mojibake bytes;
  * unreliable footnote markers (the printed superscript "1" comes through
    as the curly quote ``'`` in Times New Roman PSMT);
  * missing apostrophes ("Beloveds" instead of "Beloved's");
  * ABBYY-Arabic spans that PyMuPDF decodes as Latin nonsense.

macOS Vision (used via ``ocrmac``) reads the rendered page image cleanly:
no mojibake, correct apostrophes, properly identified Arabic with full
diacritics. This module rebuilds the per-page element list from Vision
data, with the PDF text layer used only for italic/bold flag overlay and
footnote-marker geometry.

Output is the same shape that ``chapters.py`` / ``render_chapter`` expects.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass

from ..postprocess.vocabulary import (
    correct_ocr_transliteration,
)
from ..postprocess.vocabulary import (
    lookup as vocab_lookup,
)
from .pdf_based import (
    PLACEHOLDER_FN,
    has_arabic,
    known_heading_level,
    looks_like_real_english,
    normalize_known_title,
)

ARABIC_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")
SOFT_HYPHEN = "\u00ad"

# Vision often misreads tiny footnote marker glyphs as one of these symbols
# directly after an italic transliteration or a closing single-quote /
# punctuation. We treat any of these as a footnote marker placeholder.
FN_MARKER_NOISY = set("'\"`?¢‘’\u00b0\u00b4\u00b8\u02bc\u02bb")


def _fold_apostrophes(s: str) -> str:
    return s.replace("\u2019", "'").replace("\u2018", "'")


# ---------- italic / bold range overlay from PDF spans ----------


def build_format_ranges(pdf_page) -> tuple[list[dict], list[dict]]:
    """Return (text_spans, all_spans) from the PDF text layer.

    ``text_spans`` are filtered to spans that look like real Latin text
    (font flags / italics / bold are usable here). ``all_spans`` includes
    everything (even digit-only or single-char spans), used for footnote
    marker detection.
    """
    text_spans = []
    all_spans = []
    d = pdf_page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for s in line.get("spans", []):
                font = s.get("font", "")
                text = s.get("text", "")
                if not text.strip():
                    continue
                italic = ("Italic" in font) or ("Oblique" in font) or bool(s.get("flags", 0) & 2)
                bold = ("Bold" in font) or bool(s.get("flags", 0) & 16)
                size = float(s.get("size", 0.0))
                bbox = s.get("bbox") or (0, 0, 0, 0)
                entry = {
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "italic": italic,
                    "bold": bold,
                    "size": size,
                    "text": text,
                    "font": font,
                }
                all_spans.append(entry)
                # Latin-text filter for the format-overlay span list.
                if not re.search(r"[A-Za-z]", text):
                    continue
                if font.startswith("Arial") and "Italic" not in font and len(text.strip()) <= 2:
                    continue
                text_spans.append(entry)
    return text_spans, all_spans


def _y_overlap(a0, a1, b0, b1) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def line_format_for(line: dict, pdf_spans: list[dict]) -> dict:
    """Aggregate format flags from PDF spans whose Y range overlaps this
    Vision line. Returns dict with 'italic_words', 'bold_words', 'size'.
    """
    y0, y1 = line["y0"], line["y1"]
    overlap_spans = [
        s for s in pdf_spans
        if _y_overlap(y0, y1, s["y0"], s["y1"]) >= (y1 - y0) * 0.4
    ]
    italic_words: set[str] = set()
    bold_words: set[str] = set()
    sizes: list[float] = []
    for s in overlap_spans:
        sizes.append(s["size"])
        if s["italic"]:
            for w in re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", s["text"]):
                if len(w) >= 2:
                    italic_words.add(w.lower())
        if s["bold"]:
            for w in re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", s["text"]):
                if len(w) >= 2:
                    bold_words.add(w.lower())
    return {
        "italic_words": italic_words,
        "bold_words": bold_words,
        "size": statistics.median(sizes) if sizes else 0.0,
    }


# ---------- Vision line classification ----------


def find_footnote_y_threshold(
    pdf_page,
    pdf_spans: list[dict],
    page_height: float,
    body_size: float,
) -> float:
    """Return the Y above which content is body, below which is footnotes.

    Tries three signals in order:
      1. A horizontal-rule drawing between body and footnotes (the page's
         ``[bookbody] ____________ [footnotes]`` separator).
      2. A Y position at which the local median span size sustains a drop
         to <= ``body_size * 0.92`` for several consecutive lines.
      3. Default: page_height * 1.1 (no footnote zone on this page).
    """
    # Signal 1: horizontal rule via PDF drawings.
    try:
        drawings = pdf_page.get_drawings()
    except Exception:
        drawings = []
    rule_ys = []
    for d in drawings:
        for item in d.get("items", []):
            # 'l' is a line segment: ('l', Point, Point) per PyMuPDF.
            if not item or len(item) < 3:
                continue
            tag = item[0]
            if tag != "l":
                continue
            p0, p1 = item[1], item[2]
            x0, y0 = float(p0.x), float(p0.y)
            x1, y1 = float(p1.x), float(p1.y)
            if abs(y0 - y1) < 1.5 and abs(x1 - x0) > 60 and y0 > page_height * 0.4:
                rule_ys.append((y0 + y1) / 2)
    if rule_ys:
        return min(rule_ys)

    # Signal 1b: tiny footnote marker digits in the lower margin. Some pages
    # have footnotes whose text is close enough to body size that the sustained
    # small-text detector starts at note 2 instead of note 1. The marker glyphs
    # themselves are much smaller, so use their Y as a robust lower-page cue.
    tiny_marker_ys = []
    page_width = float(pdf_page.rect.width)
    for s in pdf_spans:
        text = s["text"].strip()
        if s["y0"] < page_height * 0.72:
            continue
        # Only footnote-list markers live near the left text margin. Inline
        # superscript markers in the body can also be tiny digits, but their
        # x-position follows the body text and must not define the note zone.
        if s["x0"] > page_width * 0.22:
            continue
        if s["size"] >= body_size * 0.80:
            continue
        if re.fullmatch(r"\d{1,3}\.?", text) or text in {"*", "•"}:
            tiny_marker_ys.append(s["y0"])
    if tiny_marker_ys:
        return max(page_height * 0.40, min(tiny_marker_ys) - 8)

    # Signal 2: sustained small-text run.
    threshold = body_size * 0.92
    pool = sorted(
        [s for s in pdf_spans if s["y0"] >= page_height * 0.40 and re.search(r"[A-Za-z]", s["text"])],
        key=lambda s: s["y0"],
    )
    # Group spans into lines by Y proximity.
    lines: list[list[dict]] = []
    for s in pool:
        if not lines:
            lines.append([s])
            continue
        last = lines[-1]
        if abs(s["y0"] - last[0]["y0"]) <= 4:
            last.append(s)
        else:
            lines.append([s])
    line_summaries = [
        {"y": min(L[0]["y0"], min(s["y0"] for s in L)), "med": statistics.median([s["size"] for s in L])}
        for L in lines
    ]
    # Find first line where median <= threshold AND next line also small.
    for i in range(len(line_summaries) - 1):
        if line_summaries[i]["med"] <= threshold and line_summaries[i + 1]["med"] <= threshold:
            return line_summaries[i]["y"]

    # Signal 3: a single small-text line in the lower half of the page that
    # follows a body-size line is almost always a footnote (e.g. a one-line
    # citation). We require:
    #   - the candidate line median <= body_size * 0.92
    #   - the previous line median >= body_size * 0.95 (clearly body)
    #   - candidate sits in the bottom 35% of the page
    bottom_threshold = page_height * 0.65
    for i in range(1, len(line_summaries)):
        cur = line_summaries[i]
        prev = line_summaries[i - 1]
        if cur["y"] < bottom_threshold:
            continue
        if cur["med"] <= threshold and prev["med"] >= body_size * 0.95:
            return cur["y"]
    return page_height * 1.1


def is_footnote_line(line: dict, footnote_y_threshold: float) -> bool:
    # Vision bbox tops sit a few points above the PDF span y0 (ascenders), so
    # we relax the comparison by ~6pt to keep footnote first-lines from being
    # classified as body text.
    return line["y0"] >= footnote_y_threshold - 6


def _clean_frontmatter_line(text: str) -> str:
    """Fix high-confidence recurring OCR slips on centered imprint pages.

    Generic, book-agnostic only. Anything book-specific (publisher
    names, translator names, place names, websites) belongs in a
    per-book ``books/<slug>/ocr_fixes.toml`` file and is applied by
    the ``common_ocr`` post-processor.
    """
    return text.replace("Copyright @", "Copyright ©")


def _clean_frontmatter_arabic_block(text: str) -> str:
    """Canonicalize a publisher's Arabic/Persian organization name.

    Book-agnostic. Per-book Arabic frontmatter substitutions belong
    in the book's vocabulary file or per-book OCR fixes.
    """
    return text


def _clean_footnote_text(text: str) -> str:
    """Fix recurring OCR slips inside short English footnotes.

    Generic, book-agnostic cleanup only. Book-specific footnote
    substitutions (bracket placeholders, transliteration tokens, etc.)
    belong in ``books/<slug>/ocr_fixes.toml`` and are applied by the
    ``common_ocr`` post-processor.
    """
    text = re.sub(r"^\s*[-–—]\s*", "", text)
    text = re.sub(r"^\s*\d+\S?\s+(?=\[|[A-Za-z])", "", text)
    text = re.sub(r"^\s*['’]\s*(?=Quran\b)", "", text)
    return text


def _prefer_secondary_ocr(primary: list[dict], secondary: list[dict]) -> list[dict]:
    """Use the secondary Vision pass when it clearly read Latin text better.

    The Arabic-preferred Vision pass sometimes reads mixed-script
    imprint lines more accurately than the English-preferred pass,
    even for Latin text (Vision occasionally drops or misreads ASCII
    around adjacent Arabic glyphs). We swap a same-baseline line only
    when:

    - both candidates are Latin-only (no Arabic in either);
    - the secondary line contains the same number of *or more*
      English-letter runs than the primary; and
    - the secondary line has strictly fewer non-alphanumeric noise
      characters (a proxy for a cleaner OCR pass).

    This is a generic heuristic; per-book corrections belong in
    ``books/<slug>/ocr_fixes.toml``.
    """
    out: list[dict] = []
    secondary_sorted = sorted(secondary, key=lambda L: L["y0"])
    _noise_re = re.compile(r"[^A-Za-z0-9 ]")
    _word_re = re.compile(r"[A-Za-z]+")
    for line in primary:
        repl = None
        text = line.get("text", "")
        for cand in secondary_sorted:
            if abs(cand["y0"] - line["y0"]) > 2.5:
                continue
            ctext = cand.get("text", "")
            if has_arabic(text) or has_arabic(ctext):
                continue
            if not re.search(r"[A-Za-z]", ctext):
                continue
            primary_words = len(_word_re.findall(text))
            secondary_words = len(_word_re.findall(ctext))
            if secondary_words < primary_words:
                continue
            primary_noise = len(_noise_re.findall(text))
            secondary_noise = len(_noise_re.findall(ctext))
            if secondary_noise < primary_noise:
                repl = dict(line)
                repl["text"] = ctext
                break
        out.append(repl or line)
    return out


def _looks_like_centered_imprint_page(lines: list[dict], page_width: float) -> bool:
    """Detect pages made of centered copyright / publisher imprint lines."""
    meaningful = [
        L for L in lines
        if len(L.get("text", "").strip()) >= 5 and re.search(r"[A-Za-z0-9]", L.get("text", ""))
    ]
    if len(meaningful) < 6:
        return False
    center = page_width / 2
    close = 0
    for L in meaningful:
        mid = (L["x0"] + L["x1"]) / 2
        if abs(mid - center) < page_width * 0.18:
            close += 1
    text = " ".join(L["text"] for L in meaningful)
    imprint_terms = sum(
        term in text
        for term in ("ISBN", "Copyright", "Printed", "Printing", "Foundation", "www.")
    )
    return close / len(meaningful) >= 0.75 and imprint_terms >= 2


def _drop_margin_noise_on_centered_page(lines: list[dict], page_width: float) -> list[dict]:
    """Drop tiny margin OCR garbage (e.g. logo specks read as ``0000``)."""
    kept = []
    for L in lines:
        t = L.get("text", "").strip()
        if len(t) <= 4 and L["x0"] < page_width * 0.20:
            continue
        kept.append(L)
    return kept


def _looks_like_centered_publisher_footer(lines: list[dict], page_width: float) -> bool:
    """Detect title-page publisher footer lines that are not footnotes."""
    meaningful = [
        L for L in lines
        if len(L.get("text", "").strip()) >= 3 and re.search(r"[A-Za-z]", L.get("text", ""))
    ]
    if len(meaningful) < 2 or len(meaningful) > 5:
        return False
    center = page_width / 2
    centered = sum(
        1 for L in meaningful
        if abs(((L["x0"] + L["x1"]) / 2) - center) < page_width * 0.18
    )
    text = " ".join(L["text"] for L in meaningful)
    return centered == len(meaningful) and any(
        term in text for term in ("Foundation", "Cooperation", "IslamIFC")
    )


def _looks_like_centered_title_credits_page(lines: list[dict], page_width: float) -> bool:
    """Detect title / credits pages whose centered lines must not be
    paragraph-joined.

    Generic heuristic: at least one stacked ALL-CAPS title line at the
    top of the page, at least one ``"... by"`` credit line below, and
    the bulk of the meaningful lines centered within ±22 % of the page
    midline. Page must not be too sparse (< 4 lines) or too dense
    (> 24 lines, which is then a TOC / chapter, not a title page).
    """
    meaningful = [
        L for L in lines
        if len(L.get("text", "").strip()) >= 3 and re.search(r"[A-Za-z]", L.get("text", ""))
    ]
    if len(meaningful) < 4 or len(meaningful) > 24:
        return False
    center = page_width / 2
    centered = sum(
        1 for L in meaningful
        if abs(((L["x0"] + L["x1"]) / 2) - center) < page_width * 0.22
        and L["x0"] > page_width * 0.05
        and L["x1"] < page_width * 0.95
    )
    centered_ratio = centered / len(meaningful)
    if centered_ratio < 0.75:
        return False

    # Title stack: at least one ALL-CAPS line of >= 3 letters near the
    # top of the meaningful run. Title pages stack title fragments in
    # all caps; TOCs and chapter bodies don't.
    top_third = meaningful[: max(1, len(meaningful) // 3 + 1)]
    has_title_stack = any(
        re.search(r"[A-Z]{3,}", L.get("text", "")) and L.get("text", "").upper() == L.get("text", "")
        for L in top_third
    )
    if not has_title_stack:
        return False

    # Credit line: any of the conventional credit phrases. These are
    # standard publishing-industry English; no specific names baked in.
    text = " ".join(L["text"] for L in meaningful)
    credit_terms = (
        "Edited by",
        "Editor",
        "Editors",
        "Translated by",
        "Translator",
        "Revised by",
        "Researched by",
        "Compiled by",
        "Foreword by",
        "Introduction by",
        "Edited and translated by",
    )
    has_credit_line = any(term in text for term in credit_terms)
    return has_credit_line


def merge_same_y_lines(lines: list[dict], y_tol: float = 4.0) -> list[dict]:
    """Vision often splits a single visual line into adjacent text fragments
    when they're separated by extra whitespace (TOC entries, e.g.
    ``"Foreword"  +  "11"``). Merge fragments that share the same baseline.
    """
    if not lines:
        return []
    items = [dict(L) for L in lines]
    items.sort(key=lambda L: (L["y0"] + L["y1"]) / 2)
    groups: list[list[dict]] = []
    for L in items:
        cy = (L["y0"] + L["y1"]) / 2
        placed = False
        for g in groups:
            g_cy = sum((x["y0"] + x["y1"]) / 2 for x in g) / len(g)
            if abs(g_cy - cy) <= y_tol:
                g.append(L)
                placed = True
                break
        if not placed:
            groups.append([L])
    out: list[dict] = []
    for g in groups:
        g.sort(key=lambda L: L["x0"])
        merged = dict(g[0])
        for L in g[1:]:
            merged["text"] = merged["text"] + "  " + L["text"]
            merged["x0"] = min(merged["x0"], L["x0"])
            merged["x1"] = max(merged["x1"], L["x1"])
            merged["y0"] = min(merged["y0"], L["y0"])
            merged["y1"] = max(merged["y1"], L["y1"])
        out.append(merged)
    out.sort(key=lambda L: L["y0"])
    return out


# ---------- footnote marker detection ----------


_INLINE_TR_RE = re.compile(
    # Opening bracket: real `[` or a Vision misread `l`/`I` followed by `'`
    # or apostrophe. We need a word-boundary or non-letter before to avoid
    # matching inside English words.
    r"(?:(?<=^)|(?<=[\s\(\)\.,;:!\?\"\u2018\u2019]))"
    r"(?P<open>\[|[lI](?=[\u02bb\u02bf']))"
    r"\s*(?P<tr>[A-Za-z'\u00e0-\u017f\u02bb\u02bf\-]+(?:[ ][A-Za-z'\u00e0-\u017f\u02bb\u02bf\-]+){0,3})"
    r"\s*(?P<close>[\]\)]|l(?![A-Za-z]))",
)


_COMMON_SHORT_WORDS = {
    "a", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in",
    "is", "it", "me", "my", "no", "of", "on", "or", "so", "to", "up",
    "us", "we", "and", "but", "for", "her", "his", "not", "the",
    "are", "all", "had", "has", "him", "its", "may", "now", "one",
    "out", "see", "she", "two", "use", "way", "you",
    "from", "have", "into", "make", "more", "such", "they",
    "this", "true", "were", "with", "your",
}


_KNOWN_ENGLISH = None


def _is_known_english(word: str) -> bool:
    """Check whether `word` is a real English word.

    Uses pyspellchecker's frequency-based dictionary (~80k words including
    inflections like ``means`` and ``prayer``).
    """
    global _KNOWN_ENGLISH
    if _KNOWN_ENGLISH is None:
        try:
            from spellchecker import SpellChecker
            sc = SpellChecker()
            _KNOWN_ENGLISH = sc.word_frequency.dictionary
        except ImportError:
            _KNOWN_ENGLISH = set()
    return word.lower() in _KNOWN_ENGLISH


def _scan_noise_back(text: str, prefix_start: int, has_canonical: bool) -> int:
    """Walk backward from `prefix_start` collecting mojibake noise tokens.

    Returns the slice start index. The slice text[start:prefix_start] is
    the noise to drop when substituting canonical Arabic.

    Strategy: collect up to 4 tokens going backward from `[`, then find the
    boundary between mojibake (closer to `[`) and real English (further
    away). A token is "definitely English" if it's 5+ chars and in dict.
    Short or non-ASCII tokens are noise.
    """
    pos = prefix_start
    # Skip whitespace just before `[`
    while pos > 0 and text[pos - 1] == ' ':
        pos -= 1
    end_after_ws = pos
    if pos == 0:
        return prefix_start
    # Stop at sentence period or newline immediately before
    if text[pos - 1] in ".\n":
        return end_after_ws
    tokens: list[tuple[int, int, str]] = []  # (start, end, token)
    max_back = max(0, prefix_start - 40)
    while pos > max_back and len(tokens) < 5:
        prev = text[pos - 1]
        if prev in ".\n":
            break
        if prev == ' ':
            pos -= 1
            continue
        tok_end = pos
        while pos > 0 and text[pos - 1] not in (' ', '\n', '.'):
            pos -= 1
        token = text[pos:tok_end]
        if not token:
            break
        tokens.append((pos, tok_end, token))
    # tokens are in order: [closest_to_bracket, ..., furthest]
    # Determine where mojibake ends and English begins.
    # Walk from closest-to-bracket outward. Consume noise; stop at English.
    noise_start = end_after_ws
    for i, (tstart, _tend, tok) in enumerate(tokens):
        word_only = re.sub(r"[^A-Za-z]", "", tok)
        has_non_ascii = any(ord(c) > 127 for c in tok) or any(
            c in "!?" for c in tok
        )
        is_short = len(word_only) <= 4
        # English check
        is_real_english = (
            len(word_only) >= 5
            and _is_known_english(word_only)
            and not has_non_ascii
        )
        if is_real_english:
            # Stop, this is real text
            break
        # If short and known, it could be a real short word (the, of, in, ...)
        if (
            is_short
            and _is_known_english(word_only)
            and not has_non_ascii
            and i > 1  # Only after we've consumed at least 2 tokens
        ):
            # Likely a real short word at this distance; stop.
            break
        # Treat as noise
        noise_start = tstart
    return noise_start


def _looks_like_english_word(word: str) -> bool:
    """Heuristic: True if `word` plausibly is a normal English word."""
    if not word:
        return False
    w = word.lower()
    if w in _COMMON_SHORT_WORDS:
        return True
    if len(w) < 4:
        return False
    # Must contain at least one vowel
    if not re.search(r"[aeiouy]", w):
        return False
    # Reject words with implausible 4+ consonant runs
    if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", w):
        return False
    # Reject all-caps short words (likely abbreviation or noise)
    if word.isupper() and len(word) <= 4:
        return False
    return True


def _vocab_lookup_relaxed(tr: str) -> str | None:
    """Try vocab with common Vision substitution errors.

    Vision sometimes misreads ``t`` as ``r`` and ``c`` as ``e`` in italic
    transliterations. Try a few targeted substitutions.
    """
    candidate = vocab_lookup(tr)
    if candidate:
        return candidate
    # Try common single-character substitutions
    subs = [
        ("r", "t"),  # ralbiyah -> talbiyah
        ("t", "r"),
        ("c", "e"),
        ("e", "c"),
        ("g", "j"),
        ("j", "g"),
    ]
    for old, new in subs:
        if old in tr.lower():
            candidate = vocab_lookup(tr.replace(old, new).replace(old.upper(), new.upper()))
            if candidate:
                return candidate
    return None


def _substitute_inline_arabic(text: str) -> str:
    """Substitute mojibake noise before a ``[transliteration]`` with canonical Arabic.

    The Vision English pass garbles inline Arabic glyphs as 2-5 random Latin
    characters (e.g. ``jes [man kafaral`` for ``﴾مَن كَفَرَ﴿ [man kafara]``).
    For every ``[token]`` whose token resolves to a known Arabic word in our
    vocabulary, we:
      1. Replace any garbled prefix (1-6 chars / 2-3 short tokens, no full
         English words) with the canonical Arabic in inline-arabic markers.
      2. Fix a trailing 'l' that's really a misread ']'.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        m = _INLINE_TR_RE.search(text, i)
        if not m:
            out.append(text[i:])
            break
        tr = m.group("tr").strip().rstrip(".,;:")
        canonical = _vocab_lookup_relaxed(tr)
        # Decide if the closing was actually `]` or a misread `l`/`)`
        close_char = m.group("close")
        had_misread_close = (close_char in ('l', ')'))
        open_char = m.group("open")
        had_misread_open = (open_char in ('l', 'I'))
        # If close was 'l'/I and tr ends with another 'l' that didn't fold to ']',
        # the actual transliteration may have shed a final letter (e.g.
        # ``[wordl`` from ``[word]`` - tr captured ``wordl``). Try both.
        if not canonical and tr and tr[-1] in ('l', 'I'):
            canonical = vocab_lookup(tr[:-1])
            if canonical:
                tr = tr[:-1]
        # Also try collapsing whitespace: ``man kafara`` not in vocab but
        # the whole phrase might be a single Arabic word. Try without spaces.
        if not canonical:
            canonical = vocab_lookup(tr.replace(" ", ""))
        # Look back for noise: prefer to consume 1-3 short tokens that look
        # like mojibake (no real English word).
        prefix_start = m.start()
        noise_start = _scan_noise_back(text, prefix_start, has_canonical=bool(canonical))
        noise = text[noise_start:prefix_start].strip()
        replace_start = noise_start
        # If we have canonical and either noise or had_l_close: substitute.
        prefix_chunk = text[i:replace_start]
        out.append(prefix_chunk)
        if canonical and (noise or had_misread_close or had_misread_open):
            # Ensure space before the Arabic if the prefix ends with a letter
            # or punctuation that abuts the Arabic awkwardly.
            if prefix_chunk and not prefix_chunk.endswith(" "):
                out.append(" ")
            out.append("\x10" + canonical + "\x11 ")
            out.append("[" + tr + "]")
        else:
            out.append(text[replace_start:m.end()])
        i = m.end()
    return "".join(out)


def _strip_footnote_markers(text: str) -> tuple[str, list[int]]:
    """Find footnote-marker positions in body text.

    A footnote marker in this book follows an italic transliteration (or a
    closing punctuation) and is rendered by the OCR engine as one of:

      - the literal digit ``1``/``2``/``3``… (Vision usually preserves
        these correctly);
      - ``?`` or ``!`` (Tesseract systematically misreads tiny superscript
        digits as these); or
      - a closing apostrophe ``'`` (likewise a Tesseract misread of a
        superscript digit).

    We replace each occurrence with a placeholder
    ``PLACEHOLDER_FN<sequence>PLACEHOLDER_FN`` so ``chapters.render_inline``
    can substitute the real footnote anchor later.

    The patterns below intentionally mirror the
    ``SUSPICIOUS_TEXT_PATTERNS`` checks in :mod:`quire.render.audit` — what
    the auditor flags as a probable footnote-marker misread is exactly what
    we recover here, so the rendered EPUB stops shipping bare ``?`` /
    ``!`` glyphs and the audit stops flagging them.

    Returns ``(text_with_placeholders, count)``.
    """
    # Italic/bold wrap markers separate the word body from any following
    # punctuation, so treat them as "word-end" for marker recovery too.
    WORD_END_MARK = {"\x03", "\x05", "\x07"}

    def _looks_like_word_end(prev: str) -> bool:
        return prev.isalpha() or prev in WORD_END_MARK

    out: list[str] = []
    seq = 0
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        prev = text[i - 1] if i > 0 else ""
        # Case A: direct digit superscript marker (Vision usually saw it).
        if c.isdigit() and prev in '."\')]':
            seq += 1
            out.append(f"{PLACEHOLDER_FN}{seq}{PLACEHOLDER_FN}")
            i += 1
            continue
        # Case B: ?/!/' immediately after closing punctuation. Real
        # question/exclamation marks never follow another punctuation
        # character, so this can only be a misread tiny digit.
        if c in "?!'" and prev in ".,;:":
            nxt = text[i + 1] if i + 1 < n else ""
            if nxt == "" or nxt.isspace():
                seq += 1
                out.append(f"{PLACEHOLDER_FN}{seq}{PLACEHOLDER_FN}")
                i += 1
                continue
        # Case C: bare ``word?`` mid-sentence (followed by whitespace and a
        # lowercase letter or an opening quote) — always a misread digit
        # since a real question mark would precede the next sentence's
        # capitalised start.
        if c == "?" and _looks_like_word_end(prev):
            tail = text[i + 1:]
            if re.match(r"\s+(['\"]?[a-z]|['\"][A-Za-z])", tail):
                seq += 1
                out.append(f"{PLACEHOLDER_FN}{seq}{PLACEHOLDER_FN}")
                i += 1
                continue
        # Case D: ``word?`` at the very end of a line/paragraph fragment,
        # restricted to words wrapped in italic/bold markers (which is how
        # transliterated technical terms — the only end-of-paragraph
        # positions where misread footnote markers actually occur — appear
        # by the time this runs). A genuine end-of-paragraph ``?`` on
        # plain prose ("Is it real?") still passes through unchanged.
        if c == "?" and prev in WORD_END_MARK:
            tail = text[i + 1:]
            if tail == "" or re.match(r"\s*$", tail):
                seq += 1
                out.append(f"{PLACEHOLDER_FN}{seq}{PLACEHOLDER_FN}")
                i += 1
                continue
        out.append(c)
        i += 1
    return "".join(out), seq


# ---------- paragraph reconstruction ----------


@dataclass
class Para:
    text: str
    y0: float
    y1: float
    indent: bool
    heading: bool
    size: float
    conf: float = -1.0
    centered: bool = False


def _line_conf_percent(line: dict) -> float:
    conf = line.get("conf", -1)
    if conf is None:
        return -1.0
    try:
        value = float(conf)
    except (TypeError, ValueError):
        return -1.0
    if 0 <= value <= 1:
        value *= 100
    return value


def vision_lines_to_paragraphs(
    lines: list[dict],
    pdf_spans: list[dict],
    body_size: float,
    body_left: float,
    body_right: float,
) -> list[Para]:
    if not lines:
        return []
    line_heights = [L["y1"] - L["y0"] for L in lines if L["y1"] > L["y0"]]
    avg_h = statistics.median(line_heights) if line_heights else 14.0
    paragraphs: list[Para] = []
    current: dict | None = None
    prev_y_bottom: float | None = None
    prev_x1: float | None = None
    body_width = max(1.0, body_right - body_left)
    # In justified text, a paragraph's last line is almost always significantly
    # shorter than the column. Anything that fills less than ~80% of the body
    # width AND ends on sentence punctuation is a strong end-of-paragraph cue.
    short_line_x1 = body_right - body_width * 0.20

    def commit():
        nonlocal current
        if current is None:
            return
        paragraphs.append(
            Para(
                text=" ".join(current["texts"]).strip(),
                y0=current["y0"],
                y1=current["y_bottom"],
                indent=current["indent"],
                heading=current["heading"],
                size=current["size"],
                conf=statistics.mean(current["confs"]) if current["confs"] else -1.0,
                centered=current.get("centered", False),
            )
        )
        current = None

    for L in lines:
        text = _fold_apostrophes(L["text"])
        if not text.strip():
            continue
        fmt = line_format_for(L, pdf_spans)
        size = fmt["size"] or body_size
        # Replace mojibake-noise + [transliteration] with canonical Arabic
        # before adding italic markers; otherwise <em> markers inside the
        # transliteration can prevent the vocabulary lookup from matching.
        text_for_format = _substitute_inline_arabic(text)
        text_for_format, italic_words = _correct_cursive_transliterations(
            text_for_format,
            fmt["italic_words"],
        )
        # Italic markup: surround any italic word with markers so chapters.py
        # turns them into <em>. We only italicize whole words that PDF spans
        # marked italic.
        text_with_em = _apply_italics(text_for_format, italic_words)
        # Footnote markers: convert digit suffixes after punctuation into
        # placeholders.
        text_with_em, _ = _strip_footnote_markers(text_with_em)

        x0 = L["x0"]
        indented = (x0 - body_left) > avg_h * 0.6
        # Heading detection: large font OR centered + matches known title
        is_centered_line = (L["x0"] > body_left + 5) and (body_right - L["x1"] > 5) and (
            abs(((L["x0"] + L["x1"]) / 2) - ((body_left + body_right) / 2)) < 18
        )
        is_heading = False
        title_match = normalize_known_title(text)
        if title_match is not None and looks_like_real_english(title_match):
            is_heading = True
        elif size >= body_size * 1.25 and looks_like_real_english(text):
            is_heading = True
        elif (
            is_centered_line
            and size >= body_size * 1.05
            and len(text) <= 60
            and looks_like_real_english(text)
        ):
            words = text.split()
            if 1 <= len(words) <= 10:
                is_heading = True

        centered_line = (
            L["x0"] > body_left + avg_h * 2.0
            and body_right - L["x1"] > avg_h * 2.0
        )
        signature_line = (
            centered_line
            and not is_heading
            and len(text) <= 90
            and (
                re.search(r"\bPresident of\b", text)
                or re.fullmatch(r"[A-Z][A-Za-z]+ \d{4}", text.strip())
                or re.fullmatch(r"[A-Z][A-Za-z]+,\s+[A-Z][A-Za-z]+", text.strip())
            )
        )

        gap = (L["y0"] - prev_y_bottom) if prev_y_bottom is not None else 0
        prev_text = current["texts"][-1] if current and current["texts"] else ""
        prev_ended_sentence = bool(re.search(r"[\.!?][\"'\)\]]?\s*$", prev_text)) if prev_text else True
        prev_ends_with_number = bool(re.search(r"\d{1,3}\s*$", prev_text)) if prev_text else False
        cur_starts_capital = bool(re.match(r"\s*[A-Z(\u2018\u2019']", text))
        toc_break = prev_ends_with_number and cur_starts_capital
        # In justified text, the last line of a paragraph is short of the
        # right margin AND ends on sentence punctuation. Vision OCR sometimes
        # misses the first-line indent of the *next* paragraph; this signal
        # reliably catches paragraph breaks even without an indent.
        short_last_line_break = (
            current is not None
            and prev_x1 is not None
            and prev_ended_sentence
            and prev_x1 <= short_line_x1
            and cur_starts_capital
        )

        new_para = (
            current is None
            or is_heading
            or (current and current["heading"])
            or signature_line
            or (current and current.get("signature"))
            or gap > avg_h * 1.6
            or (indented and prev_ended_sentence)
            or toc_break
            or short_last_line_break
        )
        if new_para:
            commit()
            current = {
                "texts": [text_with_em],
                "y0": L["y0"],
                "y_bottom": L["y1"],
                "indent": indented and not is_heading,
                "heading": is_heading,
                "size": size,
                "confs": [c for c in [_line_conf_percent(L)] if c >= 0],
                "centered": signature_line,
                "signature": signature_line,
            }
        else:
            current["texts"].append(text_with_em)
            current["y_bottom"] = max(current["y_bottom"], L["y1"])
            conf = _line_conf_percent(L)
            if conf >= 0:
                current["confs"].append(conf)
        prev_y_bottom = L["y1"]
        prev_x1 = L["x1"]
    commit()

    # Join hyphenated words across line breaks.
    for p in paragraphs:
        p.text = _rejoin_hyphenation(p.text)
        p.text = _fix_split_cursive_markers(p.text)
        # Replace mojibake-noise + [transliteration] with canonical Arabic.
        p.text = _substitute_inline_arabic(p.text)
    return paragraphs


def _apply_italics(text: str, italic_words: set[str]) -> str:
    if not italic_words:
        return text
    def _repl(m):
        w = m.group(0)
        if w.lower() in italic_words:
            return "\x02" + w + "\x03"
        return w
    return re.sub(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", _repl, text)


def _correct_cursive_transliterations(
    text: str,
    italic_words: set[str],
) -> tuple[str, set[str]]:
    """Fix OCR spellings for transliterated terms before italic overlay.

    The visual OCR pass often reads italic/cursive transliteration glyphs with
    nearby Latin letters, e.g. ``miqat`` as ``migat``. The PDF font layer still
    tells us which words were italic, so correct against the configured
    vocabulary first and then let `_apply_italics` preserve the styling.
    """
    corrected_words = set(italic_words)

    def repl(m: re.Match) -> str:
        word = m.group(0)
        corrected = correct_ocr_transliteration(word, preferred=italic_words)
        if not corrected:
            return word
        corrected_words.add(corrected.lower())
        return corrected

    corrected_text = re.sub(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", repl, text)
    return corrected_text, corrected_words


_SPLIT_CURSIVE_MARKER_RE = re.compile(
    r"(?<![A-Za-z])(?P<pre1>[A-Za-z]{1,8})\x02(?P<em1>[A-Za-z]+(?:[-'][A-Za-z]+)*)\x03(?=[^A-Za-z]|$)"
    r"|(?<![A-Za-z])\x02(?P<em2>[A-Za-z]+(?:[-'][A-Za-z]+)*)\x03(?P<post2>[A-Za-z]{1,8})(?![A-Za-z])"
)


def _fix_split_cursive_markers(text: str) -> str:
    """Repair words that OCR split before the italic overlay was applied."""

    def repl(m: re.Match) -> str:
        if m.group("pre1") is not None:
            word = m.group("pre1") + m.group("em1")
        else:
            word = m.group("em2") + m.group("post2")
        corrected = correct_ocr_transliteration(word)
        if not corrected:
            return m.group(0)
        return "\x02" + corrected + "\x03"

    return _SPLIT_CURSIVE_MARKER_RE.sub(repl, text)


def _rejoin_hyphenation(text: str) -> str:
    text = re.sub(r"-\s+", lambda m: "" if m.start() > 0 and text[m.start() - 1].islower() else m.group(0), text)
    text = re.sub(r"\u00ad", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------- footnote parsing ----------


_FN_START_RE = re.compile(r"^\s*([\d]{1,3}|[*])\s+(.*)$", re.S)
_BRACKET_RE = re.compile(r"[\[\(]\s*([A-Za-z\u00e0-\u017f\u02bf\u02bb' .\-]{2,30}?)\s*[\]\)]")


_FOOTNOTE_START_BRACKET = re.compile(r"^.{0,25}?[\[\(][A-Za-z\u00e0-\u017f' .\-]{2,20}[\]\)]")
_FOOTNOTE_START_DIGIT = re.compile(
    r"^\s*\d{1,3}(?:\s+|[^\sA-Za-z]{1,3}\s+)(?:[A-Za-z]|\[)"
)


def _find_footnote_start_lines(fn_lines: list[dict]) -> list[int]:
    """Return indices into ``fn_lines`` that start a new footnote.

    Heuristic: a footnote begins on a line whose text contains either
      * a ``[transliteration]`` or ``(transliteration)`` pattern within its
        first 25 characters (Vision misreads the leading Arabic word so the
        clean signal is the bracketed Latin term that follows), or
      * a leading digit + space + word (Quran citation / cross-reference
        footnote).
    The first line of the zone is always a start line.
    """
    out: list[int] = []
    for i, L in enumerate(fn_lines):
        text = _fold_apostrophes(L["text"]).strip()
        if not text:
            continue
        if i == 0:
            out.append(i)
            continue
        if _FOOTNOTE_START_BRACKET.match(text) or _FOOTNOTE_START_DIGIT.match(text):
            out.append(i)
    return out


def find_footnote_marker_ys(
    pdf_spans: list[dict], fn_y_threshold: float, body_size: float
) -> list[float]:
    """Return Y positions of footnote-marker glyphs in the footnote zone.

    A footnote marker in this book is a tiny digit (or '*'/'•') rendered in
    Times New Roman family at ~60-78% of body size. We require the span to
    be:
      * in the footnote zone,
      * Times-family (not Arial mojibake),
      * size < body_size * 0.78,
      * text is exactly a 1-3 digit string, an asterisk, or a bullet.
    """
    markers: list[float] = []
    for s in pdf_spans:
        if s["y0"] < fn_y_threshold - 6:
            continue
        if s["size"] >= body_size * 0.78:
            continue
        if not s["font"].startswith("TimesNewRoman"):
            continue
        text = s["text"].strip()
        if not text:
            continue
        if re.fullmatch(r"\d{1,3}\.?", text) or text in {"*", "•"}:
            markers.append((s["y0"] + s["y1"]) / 2)
    markers.sort()
    merged: list[float] = []
    for m in markers:
        if merged and abs(m - merged[-1]) <= 6:
            continue
        merged.append(m)
    return merged


def parse_footnotes_from_vision(
    fn_lines: list[dict],
    pdf_spans: list[dict],
    body_size: float,
    arabic_blocks_in_zone: list[dict],
    page_pno: int,
    fn_y_threshold: float = 0.0,
) -> list[dict]:
    """Parse footnote bodies from Vision lines below the footnote separator.

    Strategy: use PDF tiny-span Y positions as footnote start anchors. Each
    Vision line is assigned to the most recent marker whose Y is at or above
    the line's Y center. Inline Arabic words at the start are replaced by
    canonical Arabic looked up via vocabulary.
    """
    if not fn_lines:
        return []
    fn_lines = sorted(fn_lines, key=lambda L: L["y0"])
    # Use Vision-line text patterns to find footnote start lines, then
    # supplement with PDF tiny-marker Y positions for missed cases.
    marker_idx = _find_footnote_start_lines(fn_lines)
    if not marker_idx:
        marker_ys = find_footnote_marker_ys(pdf_spans, fn_y_threshold, body_size)
    else:
        marker_ys = []
    if marker_idx:
        # Bucket lines by start indices.
        notes: list[dict] = []
        for k, idx in enumerate(marker_idx):
            end = marker_idx[k + 1] if k + 1 < len(marker_idx) else len(fn_lines)
            bucket = fn_lines[idx:end]
            if not bucket:
                continue
            text = " ".join(_fold_apostrophes(L["text"]).strip() for L in bucket if L["text"].strip())
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"^[\d.,\s'\"\u2018\u2019]+(?=[A-Za-z\(\[])", "", text)
            notes.append(
                {
                    "number": str(k + 1),
                    "text_lines": [text],
                    "y0": min(L["y0"] for L in bucket),
                    "y1": max(L["y1"] for L in bucket),
                    "x0": min(L["x0"] for L in bucket),
                    "x1": max(L["x1"] for L in bucket),
                }
            )
    elif marker_ys:
        # Assign each Vision line to a marker bucket
        buckets: list[list[dict]] = [[] for _ in marker_ys]
        for L in fn_lines:
            cy = (L["y0"] + L["y1"]) / 2
            idx = None
            for i, my in enumerate(marker_ys):
                if my <= cy + 4:
                    idx = i
                else:
                    break
            if idx is None:
                buckets[0].append(L)
            else:
                buckets[idx].append(L)
        notes: list[dict] = []
        for i, bucket in enumerate(buckets):
            if not bucket:
                continue
            text = " ".join(_fold_apostrophes(L["text"]).strip() for L in bucket if L["text"].strip())
            text = re.sub(r"\s+", " ", text).strip()
            text = re.sub(r"^[\d.,\s'\"\u2018\u2019]+", "", text)
            notes.append(
                {
                    "number": str(i + 1),
                    "text_lines": [text],
                    "y0": min(L["y0"] for L in bucket),
                    "y1": max(L["y1"] for L in bucket),
                    "x0": min(L["x0"] for L in bucket),
                    "x1": max(L["x1"] for L in bucket),
                }
            )
    else:
        notes = _parse_footnotes_by_bracket(fn_lines, arabic_blocks_in_zone)

    # Join lines and apply italic overlay using PDF spans
    out = []
    for n in notes:
        joined = " ".join(n["text_lines"])
        joined = _rejoin_hyphenation(joined)
        joined = _clean_footnote_text(joined)
        # Collect italic ranges within this footnote
        italic_words: set[str] = set()
        for s in pdf_spans:
            if not s["italic"]:
                continue
            if not (n["y0"] - 2 <= s["y0"] and s["y1"] <= n["y1"] + 2):
                continue
            for w in re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", s["text"]):
                if len(w) >= 2:
                    italic_words.add(w.lower())
        joined, italic_words = _correct_cursive_transliterations(joined, italic_words)
        joined = _apply_italics(joined, italic_words)
        joined = _fix_split_cursive_markers(joined)
        # Substitute inline Arabic at start: pattern is
        #   "<garbled-arabic-or-symbols> [transliteration] meaning..."
        # Replace the leading garbled section with canonical Arabic via vocab.
        joined, leading_arabic = _substitute_leading_arabic(joined)
        # Also substitute any subsequent inline mojibake-glossed terms.
        joined = _substitute_inline_arabic(joined)
        # Embed any Arabic blocks whose Y range falls inside this footnote.
        embedded = []
        for blk in arabic_blocks_in_zone:
            blk_yc = (blk["y0"] + blk["y1"]) / 2
            if n["y0"] - 4 <= blk_yc <= n["y1"] + 8:
                embedded.append(blk)
        # Remove the embedded blocks from the zone list so they're not
        # double-emitted (mutating list passed in).
        for blk in embedded:
            try:
                arabic_blocks_in_zone.remove(blk)
            except ValueError:
                pass
        out.append(
            {
                "number": n["number"],
                "text": joined,
                "leading_arabic": leading_arabic,
                "embedded_arabic": embedded,
                "y": n["y0"],
            }
        )
    return out


def _parse_footnotes_by_bracket(fn_lines: list[dict], arabic_blocks_in_zone: list[dict]) -> list[dict]:
    """Fallback footnote splitter when no PDF marker spans are available.

    Splits the footnote-zone text on each ``[transliteration]`` occurrence;
    each split chunk becomes a numbered footnote. Quran-citation footnotes
    (without a bracket) are preserved as continuation of the previous note.
    """
    full_text = " ".join(_fold_apostrophes(L["text"]).strip() for L in fn_lines if L["text"].strip())
    full_text = re.sub(r"\s+", " ", full_text).strip()
    if not full_text:
        return []
    # Find bracket starts.
    matches = list(_BRACKET_RE.finditer(full_text))
    if not matches:
        return [{
            "number": "1",
            "text_lines": [full_text],
            "y0": fn_lines[0]["y0"],
            "y1": fn_lines[-1]["y1"],
            "x0": fn_lines[0]["x0"],
            "x1": fn_lines[-1]["x1"],
        }]
    notes = []
    for i, m in enumerate(matches):
        # Find a small left-context window (mis-OCR'd Arabic word + maybe a digit).
        # Walk back to start of previous note (or text start).
        prev_end = matches[i - 1].end() if i > 0 else 0
        start = max(prev_end, m.start() - 30)
        end = matches[i + 1].start() - 30 if i + 1 < len(matches) else len(full_text)
        # Move start forward past obvious noise (single chars, digits)
        chunk = full_text[start:end].strip()
        chunk = re.sub(r"^[^A-Za-z\[]*", "", chunk)
        notes.append({
            "number": str(i + 1),
            "text_lines": [chunk],
            "y0": fn_lines[0]["y0"],
            "y1": fn_lines[-1]["y1"],
            "x0": fn_lines[0]["x0"],
            "x1": fn_lines[-1]["x1"],
        })
    return notes


_LEAD_RE = re.compile(r"^[^A-Za-z\[]*\[([A-Za-z\u2019'\u00e0-\u017f\u02bf\u02bb\.\-\s]+?)\]")


def _substitute_leading_arabic(text: str) -> tuple[str, str | None]:
    """If the footnote starts with a transliteration bracket like ``[miqat]``,
    look up the canonical Arabic and return (text_unchanged, arabic).

    Vision text often has noise BEFORE the bracket (the misread Arabic word).
    We strip that noise and prepend canonical Arabic where available.
    """
    m = _LEAD_RE.match(text)
    if not m:
        return text, None
    tr = m.group(1).strip().rstrip(".,;:")
    # Clean up: take only the first short bracket content.
    if len(tr) > 30:
        return text, None
    canonical = vocab_lookup(tr)
    if canonical is None:
        return text, None
    # Replace the entire pre-bracket prefix (mojibake noise) with the canonical
    # Arabic. We wrap canonical Arabic in \x10..\x11 markers so chapters.py
    # renders it as a span lang="ar".
    rest = text[m.start(0):]
    new_text = "\x10" + canonical + "\x11" + " " + rest
    return new_text, canonical


# ---------- main entry point ----------


def structure_page_vision(
    pdf_page,
    vision_page: dict,
    body_size_global: float,
    heuristics: list[str] | None = None,
) -> list[dict]:
    """Convert a Vision-OCR'd page into typed elements.

    Some books need bespoke cleanup of their imprint / colophon pages
    (e.g. canonicalising the publisher's Arabic organisation name when
    OCR mangles a known phrase). These transforms are opt-in via
    ``cfg.book_heuristics``; new books default to no book-specific
    transforms. See :func:`_clean_frontmatter_line` and
    :func:`_clean_frontmatter_arabic_block` for the actual fixes
    triggered by the ``"imprint-fix"`` heuristic.
    """
    _heur: set[str] = set(heuristics or [])
    _frontmatter_clean = (
        _clean_frontmatter_line if "imprint-fix" in _heur else (lambda t: t)
    )
    _frontmatter_clean_ar = (
        _clean_frontmatter_arabic_block
        if "imprint-fix" in _heur
        else (lambda t: t)
    )
    pw = vision_page["page_size_pt"][0]
    ph = vision_page["page_size_pt"][1]
    pdf_spans, all_pdf_spans = build_format_ranges(pdf_page)
    # Body size: estimate from the most-common body-zone span size, preferring
    # the larger mode (footnotes use smaller text and can dominate the count
    # when body text is short).
    from collections import Counter
    body_zone_sizes = [
        round(s["size"] * 4) / 4
        for s in pdf_spans
        if 9.5 <= s["size"] <= 14 and s["y0"] < ph * 0.7
    ]
    if body_zone_sizes:
        counts = Counter(body_zone_sizes).most_common()
        # If a "tall" mode (>= 11pt) has at least 30% of the most-common count,
        # prefer it (footnotes are typically <= 10.5pt).
        top_count = counts[0][1]
        tall_modes = [(sz, c) for sz, c in counts if sz >= 11.0 and c >= top_count * 0.3]
        if tall_modes:
            body_size = max(tall_modes, key=lambda t: t[1])[0]
        else:
            body_size = counts[0][0]
    else:
        body_size = body_size_global

    fn_y0 = find_footnote_y_threshold(pdf_page, all_pdf_spans, ph, body_size)

    en_lines = _prefer_secondary_ocr(
        list(vision_page.get("en_lines", [])),
        list(vision_page.get("ar_lines", [])),
    )
    # Drop running header. Two passes:
    #   1. Anything in the top 60pt is always header / margin noise.
    #   2. Within the top 100pt, drop a line that looks like a running
    #      header. We only flag it as such when the line carries BOTH an
    #      all-caps title AND a trailing page number (e.g. "TAWAF 89") —
    #      that combination doesn't appear in real chapter titles, but is
    #      the standard running-header layout.
    def _is_running_header(L: dict) -> bool:
        if L["y0"] >= 100:
            return False
        t = L["text"].strip()
        if not t:
            return True
        m = re.search(r"^(.+?)\s+(\d{1,3})\s*$", t)
        if not m:
            return False
        title = m.group(1).strip()
        letters = [c for c in title if c.isalpha()]
        if len(letters) < 4:
            return False
        return all(c.isupper() for c in letters)

    en_lines = [L for L in en_lines if L["y0"] >= 60 and not _is_running_header(L)]

    # Merge fragments that share a baseline (Vision splits TOC entries into
    # title + page-number fragments separated by long whitespace).
    en_lines = merge_same_y_lines(en_lines)

    body_lines = [L for L in en_lines if not is_footnote_line(L, fn_y0)]
    fn_lines = [L for L in en_lines if is_footnote_line(L, fn_y0)]
    centered_footer_lines: list[dict] = []

    # A lone bottom citation can be missed by the font-size threshold when the
    # page has very little text. Treat obvious citation lines at the foot of the
    # page as notes instead of body content.
    if not fn_lines:
        citation_lines = [
            L for L in body_lines
            if L["y0"] > ph * 0.78
            and re.match(r"^\s*['’]?\s*(Quran|Hadith|Majlisi|Kulayni|Saduq)\b", L["text"])
        ]
        if citation_lines:
            fn_lines = citation_lines
            body_lines = [L for L in body_lines if L not in citation_lines]

    # A copyright/publisher imprint page is often entirely small centered text.
    # The font-size threshold can mistake that whole page for footnotes. If all
    # meaningful text fell into the footnote bucket, treat it as body instead.
    if not body_lines and len(fn_lines) >= 6:
        fn_y0 = ph * 1.1
        body_lines = list(en_lines)
        fn_lines = []
    centered_imprint = _looks_like_centered_imprint_page(body_lines, pw)
    centered_title_credits = False
    if centered_imprint:
        body_lines = _drop_margin_noise_on_centered_page(body_lines, pw)
        for L in body_lines:
            L["text"] = _frontmatter_clean(L["text"])
    elif _looks_like_centered_title_credits_page(body_lines, pw):
        centered_title_credits = True
        for L in body_lines:
            L["text"] = _frontmatter_clean(L["text"])
        if fn_lines and _looks_like_centered_publisher_footer(fn_lines, pw):
            centered_footer_lines = list(fn_lines)
            fn_lines = []
    elif fn_lines and _looks_like_centered_publisher_footer(fn_lines, pw):
        centered_footer_lines = list(fn_lines)
        fn_lines = []

    # Body left/right margins from body lines
    if body_lines:
        body_left = min(L["x0"] for L in body_lines)
        body_right = max(L["x1"] for L in body_lines)
    else:
        body_left, body_right = 60.0, pw - 60

    # Compute paragraphs
    paragraphs = [] if centered_imprint or centered_title_credits else vision_lines_to_paragraphs(
        body_lines, pdf_spans, body_size, body_left, body_right
    )

    # Arabic blocks split by zone
    ar_blocks = list(vision_page.get("arabic_blocks", []))
    body_ar = [b for b in ar_blocks if (b["y0"] + b["y1"]) / 2 < fn_y0]
    fn_ar = [b for b in ar_blocks if (b["y0"] + b["y1"]) / 2 >= fn_y0]

    elements: list[dict] = []
    if centered_imprint or centered_title_credits:
        for line in body_lines:
            text = _frontmatter_clean(_fold_apostrophes(line["text"]).strip())
            if not text:
                continue
            elements.append(
                {
                    "kind": "paragraph",
                    "text": text,
                    "y": line["y0"],
                    "conf": _line_conf_percent(line),
                    "indent": False,
                    "centered": True,
                }
            )

    for p in paragraphs:
        if p.heading:
            level = known_heading_level(p.text)
            level = 2 if (level == 1 or level is None and p.size >= body_size * 1.4) else 3
            if level is None:
                level = 3
            elements.append({"kind": "heading", "level": level, "text": p.text, "y": p.y0, "conf": p.conf})
        else:
            elements.append(
                {
                    "kind": "paragraph",
                    "text": p.text,
                    "y": p.y0,
                    "conf": p.conf,
                    "indent": p.indent,
                    "centered": p.centered,
                }
            )
    for line in centered_footer_lines:
        text = _frontmatter_clean(_fold_apostrophes(line["text"]).strip())
        if text:
            elements.append(
                {
                    "kind": "paragraph",
                    "text": text,
                    "y": line["y0"],
                    "conf": _line_conf_percent(line),
                    "indent": False,
                    "centered": True,
                }
            )
    for blk in body_ar:
        # is_quran: only when block is short, centered, AND looks like a verse
        # (bracketed with ASCII paren/dot or already decorated). Hadiths use
        # «» and don't get the special quran styling.
        text = _frontmatter_clean_ar(blk["text"].strip())
        starts_quran_like = bool(
            re.match(r"^[\(\{\u2026\.\u00ab\ufd3e]", text)
            or text.startswith("...")
        )
        is_quran = (
            len(blk.get("lines", [])) <= 4
            and (blk["x0"] > pw * 0.10)
            and (pw - blk["x1"] > pw * 0.05)
            and starts_quran_like
            and "\u00ab" not in text[:5]  # leading « = hadith
        )
        elements.append(
            {
                "kind": "arabic",
                "text": text,
                "y": blk["y0"],
                "conf": blk.get("conf", -1),
                "is_quran": is_quran,
            }
        )
    elements.sort(key=lambda e: e["y"])

    # Footnotes (each can contain inline Arabic via leading_arabic + embedded_arabic)
    notes = parse_footnotes_from_vision(
        fn_lines, all_pdf_spans, body_size, fn_ar, vision_page["pno"], fn_y_threshold=fn_y0
    )
    for n in notes:
        ftext = n["text"]
        for blk in n["embedded_arabic"]:
            blk_text = blk["text"].replace("\n", " \u2022 ")
            ftext += " \x10" + blk_text + "\x11"
        elements.append(
            {
                "kind": "footnote",
                "number": n["number"],
                "text": ftext,
                "y": n["y"],
            }
        )

    return elements


def _collect_page_citations(notes: list[dict]) -> list[tuple[int, int]]:
    """Scan footnote texts for ``Quran, <Surah> <s>:<v>`` citations."""
    from ..postprocess.canonical.quran import find_citations
    cits: list[tuple[int, int]] = []
    for n in notes:
        for c in find_citations(n["text"]):
            cits.append((c["surah"], c["ayah"]))
    return cits


# Markers that almost never appear in legitimate English body text but are
# common when a PDF Arabic-font glyph is misdecoded as a Latin codepoint.
_MOJIBAKE_CHARS = "\u2039\u203a\u00bf\u00a1\u00ab\u00bb\u00b0"
_MOJIBAKE_TOKEN_RE = re.compile(
    rf"\S*[{_MOJIBAKE_CHARS}]\S*(?:\s+\S+){{0,4}}?[{_MOJIBAKE_CHARS}\)\]]"
)

# Inline Arabic mojibake near a "verse" / "wording" / "name" cue.
# Examples: "the verse jes in this verse", "The wording f'es jo can demonstrate"
# We match a short cluster of 1-3 non-word tokens between two real English words
# preceded by one of the cue phrases.
_INLINE_CUE_RE = re.compile(
    r"(?P<cue>(?:\b(?:verse|wording|name|states?|saying|phrase|word|reads?|says?)\b|,)\s+)"
    r"(?P<gib>(?:[a-zA-Z]+'?[a-zA-Z]*[!?]?\s+){1,4})"
    r"(?P<after>\b(?:in this verse|in this passage|expresses|can demonstrate|states that|of [Aa]llah|here|means|is the|—)\b)",
    re.IGNORECASE,
)


def _replace_body_mojibake(text: str, citations: list[tuple[int, int]]) -> str:
    """Strip inline-Arabic mojibake clusters from body paragraphs.

    Two passes:
      1. Special-character mojibake (containing ‹ › ¿ ¡ ° « »).
      2. Cue-based mojibake: short token cluster between "verse" / "wording" etc.
         and a continuation word like "in", "can", "states".
    """
    out = text

    # Pass 1: special-char mojibake
    if any(c in out for c in _MOJIBAKE_CHARS):
        for _ in range(4):
            m = _MOJIBAKE_TOKEN_RE.search(out)
            if not m:
                break
            start = m.start()
            while start > 0:
                c = out[start - 1]
                if c == " ":
                    prev_word_match = re.search(r"[A-Za-z]{4,}\s*$", out[:start])
                    if prev_word_match:
                        break
                    start -= 1
                elif c.isalpha() or c.isdigit() or c in ".!?,_-…":
                    start -= 1
                else:
                    break
            placeholder = _quran_placeholder(citations)
            out = out[:start].rstrip() + " " + placeholder + out[m.end():]

    # Pass 2: cue-based mojibake. Every match is replaced with the cue + placeholder + after.
    # Tight whitelist: only common English short words can survive between
    # a "verse" / comma cue and an "in this verse" continuation. Anything
    # else is treated as Arabic mojibake.
    SAFE = {
        "the", "a", "an", "of", "is", "in", "at", "on", "by", "for", "to", "or",
        "and", "but", "yet", "so", "if", "as", "be", "do", "go", "he", "it",
        "we", "us", "no", "not", "all", "any", "one", "two", "three",
        "you", "his", "her", "its", "our", "are", "was", "had", "has", "may",
        "can", "old", "new", "see", "say", "men", "this",
        "very", "such", "from", "with", "into", "they", "them", "what", "when",
        "while", "where", "than", "then", "thus", "also", "even", "much",
        "more", "most", "many", "less",
    }
    def _gib_repl(m: re.Match) -> str:
        gib = m.group("gib").strip()
        toks = re.split(r"\s+", gib)
        clean = [re.sub(r"[!?,.;:]+$", "", t).lower() for t in toks if t]
        if not clean:
            return m.group(0)
        if any(len(t) > 6 for t in clean):
            return m.group(0)
        unknown = [t for t in clean if t not in SAFE]
        if not unknown:
            return m.group(0)
        # If majority of tokens are NOT in the safe whitelist, treat as mojibake.
        if len(unknown) >= max(1, (len(clean) + 1) // 2):
            placeholder = _quran_placeholder(citations)
            return f"{m.group('cue')}{placeholder} {m.group('after')}"
        return m.group(0)

    out = _INLINE_CUE_RE.sub(_gib_repl, out)
    return out


def _quran_placeholder(citations: list[tuple[int, int]]) -> str:
    if citations:
        s, a = citations[0]
        return f"[Quran {s}:{a}]"
    return "[Arabic]"


def _replace_with_canonical_quran(
    arabic_text: str, citations: list[tuple[int, int]]
) -> str | None:
    """If `arabic_text` matches a known Quran verse, return the canonical
    Arabic (wrapped in ﴿...﴾). Otherwise return None.
    """
    from ..postprocess.canonical.quran import find_best_verse, find_verse_by_anchor
    result = None
    if citations:
        result = find_best_verse(arabic_text, citations)
        if result and result[3] < 0.40:
            result = None
    if result is None:
        # Fall back to global anchor-based lookup (4-word exact match)
        result = find_verse_by_anchor(arabic_text)
        if result and result[3] < 0.55:
            result = None
    if result is None:
        return None
    surah, ayah, slice_text, score = result
    # Preserve any leading/trailing ellipses that the book uses to indicate
    # a fragment of a verse.
    has_lead = bool(re.match(r"^\s*[\(\[\{﴿]?\s*\.{2,}", arabic_text))
    has_trail = bool(re.search(r"\.{2,}\s*[\)\]\}﴾]?\s*$", arabic_text))
    canonical = slice_text.strip()
    if has_lead:
        canonical = "... " + canonical
    if has_trail:
        canonical = canonical + " ..."
    return "﴿" + canonical + "﴾"
