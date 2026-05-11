"""Script and language utilities shared by extract/structure/render/postprocess.

The pipeline used to duplicate Arabic/Persian/Hebrew/Urdu detection logic in
several modules; this module centralizes the rules so renderers and
post-processors agree.

The detection rules are deliberately simple: count characters from each
script's Unicode block, then apply a small set of script-specific overrides
(e.g. Persian-only ``گ ژ پ چ`` letters, Urdu-only ``ے ٹ ڈ`` letters).
"""

from __future__ import annotations

import re

ARABIC_RANGE = "\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff"
HEBREW_RANGE = "\u0590-\u05ff\ufb1d-\ufb4f"
GREEK_RANGE = "\u0370-\u03ff\u1f00-\u1fff"
CYRILLIC_RANGE = "\u0400-\u04ff"
DEVANAGARI_RANGE = "\u0900-\u097f"
CJK_RANGE = "\u4e00-\u9fff"

ARABIC_RE = re.compile(f"[{ARABIC_RANGE}]")
HEBREW_RE = re.compile(f"[{HEBREW_RANGE}]")
GREEK_RE = re.compile(f"[{GREEK_RANGE}]")
CYRILLIC_RE = re.compile(f"[{CYRILLIC_RANGE}]")
DEVANAGARI_RE = re.compile(f"[{DEVANAGARI_RANGE}]")
CJK_RE = re.compile(f"[{CJK_RANGE}]")

# Persian uses ک (06A9) instead of Arabic ك (0643) and ی (06CC) instead of
# Arabic ي (064A). Combined with the truly Persian-only consonants پ چ ژ گ,
# these distinguish Persian from standard Arabic. Urdu also uses 06A9/06CC
# but URDU_ONLY checks Urdu-distinctive glyphs first.
PERSIAN_ONLY = re.compile(r"[\u067e\u0686\u0698\u06af\u06a9\u06cc]")  # پ چ ژ گ ک ی
URDU_ONLY = re.compile(r"[\u0679\u0688\u06d2\u0691\u06ba]")  # ٹ ڈ ے ڑ ں

RTL_LANGS = {"ar", "fa", "he", "ur", "yi", "ps", "sd"}


def detect_script(text: str) -> str | None:
    """Return a BCP-47 language tag for the dominant script in ``text``.

    Returns one of ``"ar"``, ``"fa"``, ``"ur"``, ``"he"``, ``"el"``, ``"hi"``,
    ``"zh"``, ``"ru"``, or ``None`` when no non-Latin script is detected.
    """
    if not text:
        return None
    if URDU_ONLY.search(text):
        return "ur"
    if PERSIAN_ONLY.search(text):
        return "fa"
    if ARABIC_RE.search(text):
        return "ar"
    if HEBREW_RE.search(text):
        return "he"
    if GREEK_RE.search(text):
        return "el"
    if DEVANAGARI_RE.search(text):
        return "hi"
    if CJK_RE.search(text):
        return "zh"
    if CYRILLIC_RE.search(text):
        return "ru"
    return None


def is_rtl(lang: str | None) -> bool:
    """Whether a BCP-47 language tag should render right-to-left."""
    if not lang:
        return False
    primary = lang.split("-", 1)[0].lower()
    return primary in RTL_LANGS


def css_class_for(lang: str | None) -> str | None:
    """CSS class hint matching the script. Returns ``None`` when no special
    treatment is needed.
    """
    if not lang:
        return None
    primary = lang.split("-", 1)[0].lower()
    if primary in {"ar"}:
        return "arabic"
    if primary in {"fa"}:
        return "persian"
    if primary in {"ur"}:
        return "urdu"
    if primary in {"he"}:
        return "hebrew"
    if primary in {"el"}:
        return "greek"
    return None
