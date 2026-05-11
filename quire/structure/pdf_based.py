"""Per-page structuring: merge PDF English with OCR Arabic, fix hyphenation,
detect headings, footnote refs, and Quranic quote blocks.

The output of `structure_page` is a list of typed elements in reading order:

  Element kinds:
    - {'kind': 'heading', 'level': int, 'text': str, 'y': float}
    - {'kind': 'paragraph', 'text': str, 'y': float, 'indent': bool}
    - {'kind': 'arabic', 'text': str, 'y': float, 'conf': float, 'is_quran': bool}
    - {'kind': 'footnote', 'number': str, 'text': str, 'y': float}

Inline footnote markers in body paragraphs are marked as ``\u2020N\u2020``
placeholders. Renderers convert those placeholders to semantic note links.
"""

from __future__ import annotations

import re
import statistics

ARABIC_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")
SOFT_HYPHEN = "\u00ad"
FOOTNOTE_NUM_RE = re.compile(r"^([\d]+|[*])\s*[.,;:)\]]?\s*(.*)", re.S)
PLACEHOLDER_FN = "\u2020"  # used inline; cleaned to <sup> in emitter

# Book-specific chapter/section titles are loaded from `book.toml` or a book
# module at build time. Keeping this registry mutable avoids baking one book's
# table of contents into the core parser.
KNOWN_HEADINGS: list[tuple[str, int]] = []


def configure_known_headings(headings: list[tuple[str, int]] | None) -> None:
    KNOWN_HEADINGS[:] = [(str(title), int(level)) for title, level in (headings or [])]


def known_headings() -> list[tuple[str, int]]:
    return list(KNOWN_HEADINGS)


def _norm_heading(text: str) -> str:
    text = re.sub(r"[\x02-\x07]", "", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    # Fix common PDF text-extraction quirk: "The" sometimes comes out as "Hie"
    # because of an irregular ligature/kerning code path.
    text = re.sub(r"\bHie\b", "The", text)
    text = re.sub(r"\bhi\b", "the", text)
    return re.sub(r"\s+", " ", text).strip()


def known_heading_level(text: str) -> int | None:
    norm = _norm_heading(text).lower()
    for title, level in KNOWN_HEADINGS:
        if norm == title.lower():
            return level
    for title, level in KNOWN_HEADINGS:
        if norm == title.upper().lower():
            return level
    # Fuzzy: tolerate minor extraction errors in chapter titles.
    for title, level in KNOWN_HEADINGS:
        if _close_match(norm, title.lower()):
            return level
    return None


def _close_match(a: str, b: str, max_dist: int = 2) -> bool:
    if abs(len(a) - len(b)) > max_dist:
        return False
    if len(a) < 4 or len(b) < 4:
        return a == b
    # Cheap Levenshtein-with-cap implementation; suitable for short titles.
    n, m = len(a), len(b)
    if max_dist >= n + m:
        return True
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        best = curr[0]
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
            if curr[j] < best:
                best = curr[j]
        if best > max_dist:
            return False
        prev = curr
    return prev[m] <= max_dist


def normalize_known_title(text: str) -> str | None:
    """Return the canonical form of a known title if `text` matches (fuzzily),
    else None.
    """
    norm = _norm_heading(text).lower()
    for title, _ in KNOWN_HEADINGS:
        if norm == title.lower() or norm == title.upper().lower():
            return title
    for title, _ in KNOWN_HEADINGS:
        if _close_match(norm, title.lower()):
            return title
    return None


def looks_like_real_english(text: str) -> bool:
    """Reject obvious mojibake or formatting-marker-only strings."""
    norm = _norm_heading(text)
    if len(norm) < 3:
        return False
    words = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", norm)
    if not words:
        return False
    real_count = 0
    for w in words:
        if any(c.lower() in "aeiouy" for c in w) and not is_string_mojibake(w):
            real_count += 1
    return real_count >= 1


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def y_overlaps(a: tuple[float, float], b: tuple[float, float], pad: float = 2.0) -> bool:
    return not (a[1] + pad < b[0] or b[1] + pad < a[0])


def _block_arabic_chars(blk: dict) -> int:
    return sum(1 for c in blk.get("text", "") if ARABIC_RE.match(c))


def line_in_arabic_zone(line: dict, arabic_blocks: list[dict]) -> dict | None:
    """Return the Arabic OCR block whose Y range overlaps this line, if any.

    Padding is scaled to the size of the Arabic block: substantial blocks
    (multi-line Quran citations, bibliography entries) get a few points of
    padding so adjacent mojibake-only PDF lines are dropped too. Tiny noise
    blocks (1-3 characters) get zero padding and only drop a line if its
    center is squarely inside the block.
    """
    line_h = max(line["y_bottom"] - line["y"], 1)
    line_center = (line["y"] + line["y_bottom"]) / 2
    for blk in arabic_blocks:
        n_arabic = _block_arabic_chars(blk)
        if n_arabic < 4:
            pad = 0.0
        elif blk["y1"] - blk["y0"] >= 22:
            pad = 6.0
        else:
            pad = 2.0
        b0, b1 = blk["y0"] - pad, blk["y1"] + pad
        if line["y_bottom"] < b0 or line["y"] > b1:
            continue
        if n_arabic < 4:
            # Strict containment for tiny Arabic noise.
            if not (b0 - 0.5 <= line_center <= b1 + 0.5):
                continue
            # Furthermore: only drop if the line is also string-mojibake,
            # i.e. has no real English to begin with.
            if not is_string_mojibake(line["text"]):
                continue
        if b0 <= line_center <= b1:
            return blk
        overlap = max(0.0, min(line["y_bottom"], b1) - max(line["y"], b0))
        if overlap >= 0.5 * line_h:
            return blk
    return None


def _word_is_mojibake(word: str) -> bool:
    """True if a single token (no whitespace) looks like ABBYY-Arabic mojibake."""
    if len(word) < 3:
        return False
    if any(c in "^\\{}<>|~`" for c in word):
        return True
    if re.fullmatch(r"[A-Za-z\-']+", word) is None:
        # Word with embedded digits or stray symbols - mostly OK unless the
        # digit-letter sandwich pattern shows up.
        if re.search(r"[A-Za-z][0-9]+[A-Za-z]", word):
            return True
        if any(c in "^\\{}<>|~`" for c in word):
            return True
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return False
    vowels = sum(c.lower() in "aeiouy" for c in letters)
    if len(letters) >= 4 and vowels == 0:
        return True
    if re.search(r"[bcdfghjklmnpqrstvwxz]{4,}", word.lower()):
        return True
    upper = sum(c.isupper() for c in word)
    lower = sum(c.islower() for c in word)
    if upper >= 1 and lower >= 1:
        # Lower-then-Upper inside a word. Real names usually start with a
        # single capital and have all lowercase after; mojibake has interior
        # caps.
        inner_upper = sum(1 for c in word[1:] if c.isupper())
        if inner_upper >= 1:
            # Allow: word is uppercase-heavy (likely an abbreviation/acronym).
            if upper > len(word) / 2:
                return False
            # Allow: words like "McDonald" / "iPhone" are very rare in this
            # book, but keep them by requiring two interior caps OR very
            # short stems on either side.
            if inner_upper >= 1 and lower >= 2 and len(word) <= 7:
                return True
            if inner_upper >= 2:
                return True
    return False


def is_string_mojibake(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if has_arabic(t):
        return False
    if any(c in "^\\{}<>|~`" for c in t):
        return True
    if not any(c.isalpha() for c in t):
        return False
    words = t.split()
    if not words:
        return False
    moji = sum(1 for w in words if _word_is_mojibake(w))
    if moji and moji >= max(1, len(words) // 2):
        return True
    return False


_REAL_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")


def _real_english_word_count(text: str) -> int:
    n = 0
    for w in _REAL_WORD_RE.findall(text):
        if any(c.lower() in "aeiouy" for c in w) and not is_string_mojibake(w):
            n += 1
    return n


def filter_mojibake(lines: list[dict], arabic_blocks: list[dict]) -> list[dict]:
    """Drop lines that have no real English content after the span-level
    mojibake filter has already removed Arabic-glyph Arial spans.

    A line is dropped when ALL of these are true:
      - It overlaps an Arabic OCR zone (or is itself string-mojibake), AND
      - It contains fewer than 2 real English words.

    Otherwise, the line stays. This preserves footnotes that interleave
    bracketed transliterations with English commentary.
    """
    out: list[dict] = []
    for L in lines:
        text = L["text"].strip()
        real_words = _real_english_word_count(text)
        in_arabic_zone = line_in_arabic_zone(L, arabic_blocks) is not None
        if real_words >= 2:
            out.append(L)
            continue
        if in_arabic_zone:
            continue
        if is_string_mojibake(text):
            continue
        letters = sum(c.isalpha() for c in text)
        if letters < 2 and not any(c.isdigit() for c in text):
            continue
        if letters < 4 and len(text) > 0 and letters < len(text) * 0.4:
            continue
        out.append(L)
    return out


# ---------- paragraph reconstruction ----------


def merge_paragraphs(
    lines: list[dict],
    body_size: float,
    page_width: float,
    indent_threshold: float = 9.0,
) -> list[dict]:
    """Group cooked lines into paragraphs; preserve heading-likeness.

    A line starts a new paragraph if any of:
      - it is the first surviving line on the page,
      - the previous line ended with a sentence-final mark and this one is
        clearly indented (X start more than `indent_threshold` to the right of
        the page's left text margin),
      - vertical gap from previous line > 1.6 * line_height (paragraph break),
      - the line is a heading (large size or centered short title),
      - the line starts after a heading.
    """
    if not lines:
        return []
    left_margins = [L["x0"] for L in lines]
    page_left = min(left_margins)
    line_heights = [L["y_bottom"] - L["y"] for L in lines if L["y_bottom"] > L["y"]]
    avg_line_h = statistics.median(line_heights) if line_heights else 14.0

    def is_heading_line(L: dict) -> bool:
        if known_heading_level(L["text"]) is not None:
            return True
        if not looks_like_real_english(L["text"]):
            return False
        if L["median_size"] >= body_size * 1.25:
            return True
        if L["is_centered"] and L["median_size"] >= body_size * 1.08 and len(L["text"]) <= 60:
            words = L["text"].split()
            spans_bold = any(s.get("bold") for s in L.get("spans", []))
            if 1 <= len(words) <= 10 and spans_bold:
                return True
        return False

    paragraphs: list[dict] = []
    current: dict | None = None
    prev_y_bottom = None
    for _i, L in enumerate(lines):
        heading = is_heading_line(L)
        gap = (L["y"] - prev_y_bottom) if prev_y_bottom is not None else 0
        indented = (L["x0"] - page_left) > indent_threshold
        prev_text = current["lines"][-1]["text"] if current and current["lines"] else ""
        prev_ended_sentence = bool(re.search(r"[\.!?][\"'’\)\]]?\s*$", prev_text)) if prev_text else True
        # TOC and index entries: lines ending in a page number on their own
        # are stand-alone list items. If the previous line ends with a number
        # (no trailing sentence punctuation) and the current line begins with a
        # capital letter or italic transliteration marker, force a new para.
        prev_ends_with_number = bool(re.search(r"\d{1,3}\s*$", prev_text)) if prev_text else False
        cur_starts_capital = bool(re.match(r"\s*[\x02\x04\x06]?[A-Z(\u2018\u2019']", L["text"]))
        toc_break = prev_ends_with_number and cur_starts_capital
        new_para = (
            current is None
            or heading
            or (current and current.get("heading"))
            or gap > avg_line_h * 1.6
            or (indented and prev_ended_sentence)
            or toc_break
        )
        if new_para:
            if current is not None:
                paragraphs.append(current)
            current = {
                "lines": [L],
                "y": L["y"],
                "indent": indented and not heading,
                "heading": heading,
                "median_size": L["median_size"],
            }
        else:
            current["lines"].append(L)
        prev_y_bottom = L["y_bottom"]
    if current is not None:
        paragraphs.append(current)
    return paragraphs


def rejoin_text(text_lines: list[str]) -> str:
    out = ""
    for i, line in enumerate(text_lines):
        line = line.replace("\u00a0", " ")
        if not out:
            out = line
            continue
        if out.endswith(SOFT_HYPHEN):
            out = out[:-1] + line.lstrip()
        elif out.endswith("-") and i > 0 and line[:1].islower():
            # Soft-hyphenation rendered as ASCII '-': drop hyphen and join.
            out = out[:-1] + line.lstrip()
        else:
            out = out.rstrip() + " " + line.lstrip()
    return re.sub(r"\s+", " ", out).strip()


# ---------- inline footnote ref detection ----------


def annotate_footnote_refs(paragraph: dict, body_size: float, ref_size_max: float) -> str:
    """Walk spans of paragraph lines and emit text with inline footnote markers.

    Footnote refs are spans whose median size <= ref_size_max and whose text
    is a 1-2 char digit or '*' inserted between body words. We replace them in
    the joined text with PLACEHOLDER_FN + number + PLACEHOLDER_FN.
    """
    text_lines = []
    for L in paragraph["lines"]:
        parts: list[str] = []
        for s in L["spans"]:
            t = s["text"]
            if not t.strip():
                parts.append(t)
                continue
            stripped = t.strip()
            is_small = s["size"] <= ref_size_max
            is_digit_marker = (
                len(stripped) <= 2
                and (stripped.isdigit() or stripped == "*" or re.fullmatch(r"\d{1,2}\*?", stripped))
            )
            if is_small and is_digit_marker:
                parts.append(f" {PLACEHOLDER_FN}{stripped}{PLACEHOLDER_FN} ")
            elif s.get("italic") and not s.get("bold"):
                parts.append(f"\x02{t}\x03")  # italic markers
            elif s.get("bold") and not s.get("italic"):
                parts.append(f"\x04{t}\x05")  # bold markers
            elif s.get("bold") and s.get("italic"):
                parts.append(f"\x06{t}\x07")
            else:
                parts.append(t)
        text_lines.append("".join(parts))
    joined = rejoin_text(text_lines)
    # Collapse the placeholder spaces.
    joined = re.sub(r"\s+(\u2020[\d*]+\u2020)\s+", r"\1 ", joined)
    joined = re.sub(r"\s+(\u2020[\d*]+\u2020)$", r"\1", joined)
    joined = re.sub(r"^(\u2020[\d*]+\u2020)\s+", r"\1 ", joined)
    return joined


# ---------- footnote extraction ----------


def parse_footnotes(footnote_lines: list[dict], body_size: float) -> list[dict]:
    """Group footnote zone lines into individual footnotes by leading number."""
    if not footnote_lines:
        return []
    # Reconstruct numbered footnotes from line stream.
    notes: list[dict] = []
    current_text_lines: list[str] = []
    current_number: str | None = None
    current_y: float | None = None

    def flush():
        if current_number is None or not current_text_lines:
            return
        text = rejoin_text(current_text_lines)
        notes.append({"number": current_number, "text": text, "y": current_y})

    digit_start = re.compile(r"^([\d]{1,3})\s+(.*)$|^([*])\s+(.*)$")
    for L in footnote_lines:
        t = L["text"]
        if not t:
            continue
        m = digit_start.match(t)
        if m and (current_number is None or L["x0"] <= footnote_lines[0]["x0"] + 4):
            flush()
            num = m.group(1) or m.group(3)
            rest = m.group(2) or m.group(4) or ""
            current_number = num
            current_text_lines = [rest] if rest else []
            current_y = L["y"]
        else:
            if current_number is None:
                current_number = "•"
                current_y = L["y"]
            current_text_lines.append(t)
    flush()
    return notes


# ---------- main per-page structuring ----------


def structure_page(
    page_extract: dict,
    arabic_blocks: list[dict],
    body_size_global: float,
) -> list[dict]:
    body_lines = filter_mojibake(page_extract["body_lines"], arabic_blocks)
    footnote_lines = filter_mojibake(page_extract["footnote_lines"], arabic_blocks)
    body_size_local = page_extract["body_size"]
    body_size = body_size_global if body_size_global > 0 else body_size_local
    page_width = page_extract["page_size"][0]

    # Filter Arabic blocks: drop noise (very short blocks, header strays,
    # extremely low-confidence blocks). Keep blocks in the footnote zone -
    # bibliography pages and footnote citations all live there, and they
    # render inline at their Y position which preserves reading order.
    cleaned_arabic = []
    for blk in arabic_blocks:
        if blk["y1"] < 60:
            continue  # running header strays
        n_chars = sum(1 for c in blk["text"] if has_arabic(c))
        if n_chars < 4:
            continue  # noise
        if blk.get("conf", -1) < 35:
            continue
        cleaned_arabic.append(blk)

    paragraphs = merge_paragraphs(body_lines, body_size, page_width)

    elements: list[dict] = []
    # Mark headings versus body paragraphs.
    for p in paragraphs:
        ref_size_max = body_size * 0.78
        text = annotate_footnote_refs(p, body_size, ref_size_max)
        if not text:
            continue
        if p["heading"]:
            level = known_heading_level(text)
            if level is None:
                level = 2 if p["median_size"] >= body_size * 1.4 else 3
            else:
                # Map our 1/2 chapter/section into XHTML h2/h3.
                level = 2 if level == 1 else 3
            elements.append(
                {"kind": "heading", "level": level, "text": text, "y": p["y"]}
            )
        else:
            elements.append(
                {"kind": "paragraph", "text": text, "y": p["y"], "indent": p["indent"]}
            )
    for blk in cleaned_arabic:
        # Heuristic: short Arabic (1-2 lines, centered) inside body is likely
        # Quranic citation.
        is_quran = (
            len(blk["lines"]) <= 4
            and (blk["x0"] > page_width * 0.15)
            and (page_width - blk["x1"] > page_width * 0.05)
        )
        elements.append(
            {
                "kind": "arabic",
                "text": blk["text"],
                "y": blk["y0"],
                "conf": blk.get("conf", -1),
                "is_quran": is_quran,
            }
        )
    elements.sort(key=lambda e: e["y"])
    # Footnotes at end of page in order.
    footnotes = parse_footnotes(footnote_lines, body_size)
    for fn in footnotes:
        elements.append(
            {
                "kind": "footnote",
                "number": fn["number"],
                "text": fn["text"],
                "y": fn["y"],
            }
        )
    return elements


def estimate_global_body_size(pages: list[dict]) -> float:
    """Robust body font-size estimate using only clean English-heavy pages."""
    sizes: list[float] = []
    for p in pages:
        for L in p["body_lines"]:
            if not L["text"]:
                continue
            if has_arabic(L["text"]):
                continue
            if is_string_mojibake(L["text"]):
                continue
            if 9 <= L["median_size"] <= 14:
                sizes.append(L["median_size"])
    if not sizes:
        return 11.2
    return statistics.median(sizes)
