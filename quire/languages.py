"""Deterministic language detection and script-aware text quality checks."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter

RTL = {"ar", "fa", "ur", "he", "ps", "yi", "sd"}
TESSERACT = {
    "en": "eng",
    "ar": "ara",
    "fa": "fas",
    "ur": "urd",
    "he": "heb",
    "zh": "chi_sim",
    "zh-cn": "chi_sim",
    "zh-tw": "chi_tra",
    "ja": "jpn",
    "ko": "kor",
    "ru": "rus",
    "uk": "ukr",
    "hi": "hin",
    "bn": "ben",
    "fr": "fra",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "pt": "por",
    "tr": "tur",
    "el": "ell",
    "nl": "nld",
    "pl": "pol",
    "sv": "swe",
    "id": "ind",
    "vi": "vie",
    "th": "tha",
    "ps": "pus",
    "sd": "snd",
}


def direction(language: str) -> str:
    return "rtl" if language.split("-")[0] in RTL else "ltr"


def valid_language(language: str) -> str:
    if not re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", language):
        raise ValueError("Use a language code such as en, fa, ar, or zh-TW")
    return language


def script_counts(text: str) -> Counter:
    counts: Counter = Counter()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        for script in (
            "ARABIC",
            "HEBREW",
            "CYRILLIC",
            "HIRAGANA",
            "KATAKANA",
            "HANGUL",
            "CJK",
            "DEVANAGARI",
            "BENGALI",
            "GREEK",
            "LATIN",
        ):
            if script in name:
                counts[script] += 1
                break
    return counts


def detect_language(text: str, *, hint: str | None = None) -> dict:
    """Return language + evidence, never silently call unknown text English.

    Language and script are different: short Arabic-script samples can remain
    uncertain. A user's explicit hint wins and is recorded as such.
    """
    if hint and hint != "auto":
        language = valid_language(hint)
        return {
            "language": language,
            "direction": direction(language),
            "confidence": 1.0,
            "method": "user",
            "scripts": dict(script_counts(text)),
        }
    counts = script_counts(text)
    language, confidence, method = "und", 0.0, "insufficient text"
    dominant = counts.most_common(1)[0][0] if counts else ""
    simple = {"HEBREW": "he", "HANGUL": "ko", "DEVANAGARI": "hi", "BENGALI": "bn", "GREEK": "el", "CJK": "zh"}
    if counts.get("HIRAGANA", 0) + counts.get("KATAKANA", 0) >= 2:
        language, confidence, method = "ja", 0.95, "script"
    elif dominant == "ARABIC":
        if re.search(r"[ٹڈڑںھہےۓ]", text):
            language, confidence, method = "ur", 0.9, "script"
        elif re.search(r"[پچژگ]", text) or re.search(r"\b(?:است|برای|را|می|این|خود)\b", text):
            language, confidence, method = "fa", 0.9, "script and words"
        elif re.search(r"\b(?:في|على|التي|الذي|هذا|هذه|إلى)\b", text):
            language, confidence, method = "ar", 0.9, "script and words"
        else:
            language, confidence, method = "ar", 0.45, "ambiguous Arabic script"
    elif dominant in simple:
        language, confidence, method = simple[dominant], 0.85, "script"
    if len(re.findall(r"\w", text)) >= 30 and confidence < 0.85:
        try:
            from langdetect import DetectorFactory, detect_langs

            DetectorFactory.seed = 0
            candidates = detect_langs(text[:12000])
            if candidates and candidates[0].prob >= 0.8:
                language, confidence, method = candidates[0].lang, candidates[0].prob, "language model"
        except (ImportError, ValueError):
            pass
        except Exception:  # langdetect raises its own exception on no features
            pass
    # Useful offline fallback when the optional detector is unavailable.
    if language == "und" and dominant == "LATIN":
        words = set(re.findall(r"[a-z]+", text.lower()))
        if len(words & {"the", "and", "of", "to", "is", "with", "this", "that", "for"}) >= 3:
            language, confidence, method = "en", 0.8, "common words"
    return {
        "language": language,
        "direction": direction(language),
        "confidence": round(float(confidence), 3),
        "method": method,
        "scripts": dict(counts),
    }


def text_quality(text: str) -> float:
    """Bounded extraction-health signal, explicitly not recognition accuracy."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    useful = sum(c.isalnum() or unicodedata.category(c).startswith("M") for c in chars)
    bad = sum(c == "\ufffd" or unicodedata.category(c) in {"Co", "Cc", "Cs"} for c in chars)
    repeated = sum(len(m.group()) for m in re.finditer(r"([^\W\d_])\1{4,}", text))
    return round(max(0.0, min(1.0, useful / len(chars) - 4 * bad / len(chars) - repeated / len(chars))), 3)
