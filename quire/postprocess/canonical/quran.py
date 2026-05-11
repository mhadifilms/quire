"""Canonical Quran lookup using Tanzil Uthmani text.

The book references verses by surah name + verse range (e.g.
"Aal-'Imran 3:97" or "Bagarah 2:286"). Use this module to fetch the
canonical Arabic text and replace the OCR'd verse where confident.

The corpus path can be overridden via :func:`set_corpus_path`; by default
we look in ``data/canonical/quran/tanzil-uthmani.txt`` relative to the repo
root.
"""

from __future__ import annotations

import re
from pathlib import Path

# Default location: <repo>/data/canonical/quran/tanzil-uthmani.txt
_DEFAULT_TXT = (
    Path(__file__).resolve().parents[3]
    / "data" / "canonical" / "quran" / "tanzil-uthmani.txt"
)
QURAN_TXT: Path = _DEFAULT_TXT


def reset_caches() -> None:
    """Drop the in-memory verse and anchor caches.

    Call this between books in a sequential batch run so a corpus change
    via :func:`set_corpus_path` takes effect and so we release memory.
    """
    global _QURAN, _ANCHOR_INDEX, _ANCHOR_INDEX_3
    _QURAN = None
    _ANCHOR_INDEX = None
    _ANCHOR_INDEX_3 = None


def set_corpus_path(path: Path | str) -> None:
    """Override the corpus path (clears the cached load)."""
    global QURAN_TXT, _QURAN, _ANCHOR_INDEX, _ANCHOR_INDEX_3
    QURAN_TXT = Path(path)
    _QURAN = None
    _ANCHOR_INDEX = None
    _ANCHOR_INDEX_3 = None

# Surah aliases used in the book
SURAH_NAMES = {
    1: ["fatiha", "fatihah", "alfatiha", "al-fatihah"],
    2: ["baqarah", "albaqarah", "al-baqarah", "bagarah", "albagarah"],
    3: ["aalimran", "alimran", "aal-imran", "al-imran", "imran"],
    4: ["nisa", "nisaa", "alnisa"],
    5: ["maidah", "almaidah", "ma'idah"],
    6: ["anam", "alanam", "al-anam"],
    7: ["araf", "alaraf", "al-a'raf", "a'raf"],
    8: ["anfal", "alanfal", "al-anfal"],
    9: ["tawbah", "tauba", "altawbah", "al-tawbah"],
    10: ["yunus"],
    11: ["hud"],
    12: ["yusuf", "yousef"],
    13: ["rad", "alrad", "ra'd"],
    14: ["ibrahim"],
    15: ["hijr", "alhijr"],
    16: ["nahl", "alnahl"],
    17: ["isra", "alisra", "bani-israil", "baniisrail"],
    18: ["kahf", "alkahf"],
    19: ["maryam"],
    20: ["taha", "ta-ha"],
    21: ["anbiya", "alanbiya", "al-anbiya"],
    22: ["hajj", "alhajj", "al-hajj"],
    23: ["muminun", "almuminun", "al-mu'minun"],
    24: ["nur", "alnur"],
    25: ["furqan", "alfurqan"],
    26: ["shuara", "alshuara", "al-shu'ara"],
    27: ["naml", "alnaml"],
    28: ["qasas", "alqasas"],
    29: ["ankabut"],
    30: ["rum", "alrum"],
    31: ["luqman"],
    32: ["sajdah", "alsajdah"],
    33: ["ahzab", "alahzab"],
    34: ["saba"],
    35: ["fatir"],
    36: ["yasin", "yaseen", "ya-sin"],
    37: ["saffat", "alsaffat"],
    38: ["sad"],
    39: ["zumar", "alzumar"],
    40: ["ghafir", "mumin"],
    41: ["fussilat"],
    42: ["shura", "alshura"],
    43: ["zukhruf", "alzukhruf"],
    44: ["dukhan"],
    45: ["jathiyah", "aljathiyah"],
    46: ["ahqaf", "alahqaf"],
    47: ["muhammad"],
    48: ["fath", "alfath"],
    49: ["hujurat"],
    50: ["qaf"],
    51: ["dhariyat", "aldhariyat", "al-dhariyat"],
    52: ["tur", "altur"],
    53: ["najm", "alnajm"],
    54: ["qamar", "alqamar"],
    55: ["rahman", "alrahman"],
    56: ["waqiah", "alwaqiah"],
    57: ["hadid", "alhadid"],
    58: ["mujadilah"],
    59: ["hashr", "alhashr"],
    60: ["mumtahanah", "almumtahanah"],
    61: ["saff", "alsaff"],
    62: ["jumuah", "aljumuah"],
    63: ["munafiqun", "almunafiqun"],
    64: ["taghabun"],
    65: ["talaq", "altalaq"],
    66: ["tahrim", "altahrim"],
    67: ["mulk", "almulk"],
    68: ["qalam", "alqalam"],
    69: ["haqqah", "alhaqqah"],
    70: ["maarij", "almaarij"],
    71: ["nuh"],
    72: ["jinn", "aljinn"],
    73: ["muzzammil", "almuzzammil"],
    74: ["muddathir", "almuddathir"],
    75: ["qiyamah", "alqiyamah"],
    76: ["insan", "alinsan", "dahr"],
    77: ["mursalat", "almursalat"],
    78: ["naba", "alnaba"],
    79: ["naziat", "alnaziat"],
    80: ["abasa"],
    81: ["takwir", "altakwir"],
    82: ["infitar", "alinfitar"],
    83: ["mutaffifin", "almutaffifin"],
    84: ["inshiqaq", "alinshiqaq"],
    85: ["buruj", "alburuj"],
    86: ["tariq", "altariq"],
    87: ["ala", "alala", "al-a'la"],
    88: ["ghashiyah", "alghashiyah"],
    89: ["fajr", "alfajr"],
    90: ["balad", "albalad"],
    91: ["shams", "alshams"],
    92: ["layl", "allayl"],
    93: ["duha", "alduha"],
    94: ["sharh", "inshirah", "alinshirah"],
    95: ["tin", "altin"],
    96: ["alaq", "alalaq"],
    97: ["qadr", "alqadr"],
    98: ["bayyinah", "albayyinah"],
    99: ["zalzalah", "alzalzalah"],
    100: ["adiyat", "aladiyat"],
    101: ["qariah", "alqariah"],
    102: ["takathur", "altakathur"],
    103: ["asr", "alasr"],
    104: ["humazah", "alhumazah"],
    105: ["fil", "alfil"],
    106: ["quraysh"],
    107: ["maun", "almaun"],
    108: ["kawthar", "alkawthar"],
    109: ["kafirun", "alkafirun"],
    110: ["nasr", "alnasr"],
    111: ["masad", "lahab", "almasad"],
    112: ["ikhlas", "alikhlas"],
    113: ["falaq", "alfalaq"],
    114: ["nas", "alnas"],
}


_SURAH_LOOKUP: dict[str, int] = {}
for num, names in SURAH_NAMES.items():
    for n in names:
        _SURAH_LOOKUP[n.lower().replace("-", "").replace("'", "").replace(" ", "")] = num


_QURAN: dict[tuple[int, int], str] | None = None


def _load() -> dict[tuple[int, int], str]:
    global _QURAN
    if _QURAN is not None:
        return _QURAN
    out: dict[tuple[int, int], str] = {}
    if not QURAN_TXT.exists():
        _QURAN = out
        return out
    with open(QURAN_TXT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            try:
                s = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            out[(s, v)] = parts[2]
    _QURAN = out
    return out


def normalize_surah_name(name: str) -> int | None:
    key = re.sub(r"[^a-z]", "", name.lower())
    return _SURAH_LOOKUP.get(key)


def get_verse(surah: int, ayah: int) -> str | None:
    return _load().get((surah, ayah))


def get_verses(surah: int, start: int, end: int | None = None) -> str | None:
    end = end if end is not None else start
    parts = []
    for v in range(start, end + 1):
        text = get_verse(surah, v)
        if text:
            parts.append(text)
    return " ".join(parts) if parts else None


# Match a citation like "Quran, Aal-'Imran 3:97" or "Bagarah 2:286".
# We accept a SINGLE surah-name token (letters, hyphens, apostrophes, periods)
# followed by ``S:V`` so we don't accidentally scoop up surrounding prose.
CITATION_RE = re.compile(
    r"(?:Qur'?an[,\s]+)?"
    r"(?P<name>[A-Za-z\u00e0-\u017f][A-Za-z\u00e0-\u017f'\-\.]{2,})\s+"
    r"(?P<surah>\d{1,3}):(?P<verse>\d{1,3})(?:[\-\u2013](?P<verse_end>\d{1,3}))?",
    re.IGNORECASE,
)


def find_citations(text: str) -> list[dict]:
    """Find Quranic citations in a string. Returns dicts with surah/ayah keys."""
    out = []
    for m in CITATION_RE.finditer(text):
        name = m.group("name").strip()
        surah_num = int(m.group("surah"))
        if normalize_surah_name(name) != surah_num:
            continue
        ayah = int(m.group("verse"))
        end = int(m.group("verse_end")) if m.group("verse_end") else None
        out.append({
            "surah": surah_num,
            "ayah": ayah,
            "ayah_end": end,
            "raw": m.group(0),
            "span": (m.start(), m.end()),
        })
    return out


def _strip_diacritics(s: str) -> str:
    """Strip Arabic diacritics & normalize alif/yeh variants for fuzzy compare."""
    s = re.sub(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]", "", s)
    # Normalize alif variants (alif wasla, alif maqsura, etc.) to plain alif
    s = s.replace("\u0671", "\u0627")  # ٱ → ا (alif wasla)
    s = s.replace("\u0622", "\u0627")  # آ → ا (alif madda)
    s = s.replace("\u0623", "\u0627")  # أ → ا (alif hamza above)
    s = s.replace("\u0625", "\u0627")  # إ → ا (alif hamza below)
    s = s.replace("\u0649", "\u064a")  # ى → ي (alif maqsura)
    s = s.replace("\u0629", "\u0647")  # ة → ه (taa marbuta)
    return s


def _strip_brackets(s: str) -> str:
    """Strip leading/trailing decorative brackets and ellipses."""
    s = s.strip()
    s = re.sub(r"^[\(\[\{﴿﴾]+\s*", "", s)
    s = re.sub(r"\s*[\)\]\}﴿﴾]+$", "", s)
    s = re.sub(r"\.{2,}", " ", s)
    return s.strip()


def _word_set(text: str) -> set[str]:
    text = _strip_diacritics(text)
    return set(re.findall(r"[\u0600-\u06ff]+", text))


def _verse_words(text: str) -> list[str]:
    text = _strip_diacritics(text)
    return re.findall(r"[\u0600-\u06ff]+", text)


def _word_positions(canonical: str) -> list[tuple[int, int, str]]:
    return [
        (m.start(), m.end(), m.group(0))
        for m in re.finditer(r"[\u0600-\u06ff]+", canonical)
    ]


def _best_substring_match(ocr_words: list[str], canonical: str) -> tuple[float, str]:
    """Find the contiguous slice of `canonical` whose word sequence best
    aligns with `ocr_words`. Uses longest-common-subrun search.
    """
    pos = _word_positions(canonical)
    cand_lc = [_strip_diacritics(w).lower() for _, _, w in pos]
    if not ocr_words or not cand_lc:
        return 0.0, canonical
    ocr_lc = [_strip_diacritics(w).lower() for w in ocr_words]
    n = len(ocr_lc)
    m = len(cand_lc)

    # Find LCS-like contiguous block: for every starting cand index, count
    # how many of the next n+slack cand words appear in ocr_lc (in order
    # they would appear); take the run with the highest match count.
    ocr_set = set(ocr_lc)
    best_score = 0.0
    best_start = 0
    best_end = 0
    for start in range(m):
        # Find the longest run starting at `start` whose words are mostly
        # in ocr_set (allow up to 30% gaps).
        end = start
        misses = 0
        max_misses = max(2, n // 3)
        while end < m and misses <= max_misses:
            if cand_lc[end] in ocr_set:
                misses = 0
            else:
                misses += 1
            end += 1
        # `end` may have stepped past last hit; back up to last hit.
        last_hit = start - 1
        for j in range(start, min(end, m)):
            if cand_lc[j] in ocr_set:
                last_hit = j
        if last_hit < start:
            continue
        run_len = last_hit - start + 1
        hits = sum(1 for j in range(start, last_hit + 1) if cand_lc[j] in ocr_set)
        if run_len < 2:
            continue
        # Score: fraction of OCR words matched, normalized by run length
        score = hits / max(n, run_len)
        if score > best_score:
            best_score = score
            best_start = start
            best_end = last_hit
    if best_score == 0.0:
        return 0.0, canonical
    s_pos = pos[best_start][0]
    e_pos = pos[best_end][1]
    # Trim leading/trailing punctuation in slice
    slice_text = canonical[s_pos:e_pos].strip()
    return best_score, slice_text


def _has_anchor(ocr_words: list[str], canonical: str, anchor_len: int = 3) -> bool:
    """Return True if any anchor_len-gram of ocr_words appears verbatim
    (after diacritic stripping) in canonical."""
    cand = " ".join(_verse_words(canonical))
    cand_lc = " ".join(w.lower() for w in cand.split())
    n = len(ocr_words)
    for i in range(n - anchor_len + 1):
        gram = " ".join(w.lower() for w in ocr_words[i : i + anchor_len])
        if gram in cand_lc:
            return True
    return False


_ANCHOR_INDEX: dict[tuple[str, str, str, str], list[tuple[int, int]]] | None = None


def _build_anchor_index() -> dict[tuple[str, str, str, str], list[tuple[int, int]]]:
    """Index every 4-gram of every verse to its (surah, ayah) for global search."""
    global _ANCHOR_INDEX
    if _ANCHOR_INDEX is not None:
        return _ANCHOR_INDEX
    idx: dict = {}
    for (s, a), text in _load().items():
        words = [w.lower() for w in _verse_words(text)]
        for i in range(len(words) - 3):
            key = tuple(words[i : i + 4])
            idx.setdefault(key, []).append((s, a))
    _ANCHOR_INDEX = idx
    return idx


def find_verse_by_anchor(ocr_text: str) -> tuple[int, int, str, float] | None:
    """Find a Quran verse from OCR text alone, with no citation hint.

    Requires an exact 4-word anchor (or 3-word for short fragments).
    Returns the best match if found.
    """
    ocr_words = _verse_words(_strip_brackets(ocr_text))
    if len(ocr_words) < 3:
        return None
    ocr_lc = [w.lower() for w in ocr_words]
    idx = _build_anchor_index()
    candidates: set[tuple[int, int]] = set()
    # Try 4-grams first
    if len(ocr_lc) >= 4:
        for i in range(len(ocr_lc) - 3):
            key4 = (ocr_lc[i], ocr_lc[i + 1], ocr_lc[i + 2], ocr_lc[i + 3])
            for sa in idx.get(key4, []):
                candidates.add(sa)
    # If no 4-gram match, fall back to 3-grams (more candidates, less precise)
    if not candidates:
        idx3 = _build_anchor_index_3()
        for i in range(len(ocr_lc) - 2):
            key3 = (ocr_lc[i], ocr_lc[i + 1], ocr_lc[i + 2])
            for sa in idx3.get(key3, []):
                candidates.add(sa)
    if not candidates:
        return None
    quran = _load()
    best: tuple[int, int, str, float] | None = None
    for s, a in candidates:
        canonical = quran[s, a]
        score, slice_text = _best_substring_match(ocr_words, canonical)
        if best is None or score > best[3]:
            best = (s, a, slice_text, score)
    return best


_ANCHOR_INDEX_3: dict[tuple[str, str, str], list[tuple[int, int]]] | None = None


def _build_anchor_index_3():
    global _ANCHOR_INDEX_3
    if _ANCHOR_INDEX_3 is not None:
        return _ANCHOR_INDEX_3
    idx: dict = {}
    for (s, a), text in _load().items():
        words = [w.lower() for w in _verse_words(text)]
        for i in range(len(words) - 2):
            key = tuple(words[i : i + 3])
            idx.setdefault(key, []).append((s, a))
    _ANCHOR_INDEX_3 = idx
    return idx


def find_best_verse(
    ocr_text: str,
    citations_in_context: list[tuple[int, int]] | None = None,
) -> tuple[int, int, str, float] | None:
    """Find the canonical Quran verse (or substring) that matches OCR'd text.

    Returns (surah, ayah, canonical_substring, score) or None.
    Score is in [0, 1]; higher is better. Score = 0 if no 2-word anchor.
    """
    quran = _load()
    if not quran:
        return None
    ocr_clean = _strip_brackets(ocr_text)
    ocr_words_list = _verse_words(ocr_clean)
    if len(ocr_words_list) < 3:
        return None

    candidates: list[tuple[int, int]] = []
    if citations_in_context:
        seen = set()
        for s, a in citations_in_context:
            for da in range(-2, 3):
                key = (s, a + da)
                if key in quran and key not in seen:
                    candidates.append(key)
                    seen.add(key)
    else:
        candidates = list(quran.keys())

    best: tuple[int, int, str, float] | None = None
    for s, a in candidates:
        canonical = quran[s, a]
        score, slice_text = _best_substring_match(ocr_words_list, canonical)
        # Require a 2-gram exact-anchor, OR a 3-gram for higher confidence.
        if not _has_anchor(ocr_words_list, canonical, 2):
            continue
        if best is None or score > best[3]:
            best = (s, a, slice_text, score)
    return best
