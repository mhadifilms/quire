"""Conservative common OCR correction from book-local evidence.

This pass learns high-confidence/repeated multi-word proper names from the
book, then applies small edit-distance fixes only inside lower-confidence
paragraphs/headings. It is meant for OCR confusions like ``Ion Arabi`` where
the same book also confidently reads ``Ibn Arabi``.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass

from . import ocr_corrections

_FORMAT_MARKERS_RE = re.compile(r"[\x02-\x07\x10-\x13]")
_NAME_RE = re.compile(
    r"\b(?!See\b)[A-Z][A-Za-z'’-]{1,}"
    r"(?:\s+(?:al-|Al-)?[A-Z][A-Za-z'’-]{1,}){1,3}\b"
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")

_LEADING_STOPWORDS = {
    "A", "An", "And", "As", "At", "But", "By", "For", "From", "If", "In",
    "Into", "It", "Its", "Of", "On", "Or", "So", "That", "The", "Then",
    "This", "To", "When", "Where", "While", "With", "His", "Her", "My",
    "Our", "Their", "Your",
}
_PRONOUNS = {"his", "her", "my", "our", "their", "your"}
_NAME_MARKERS = {
    "ibn", "bin", "bint", "abu", "abi", "abd", "al", "ali", "muhammad",
    "mohammad", "ja'far", "jafar", "husayn", "hussain", "hasan",
}


@dataclass
class _Candidate:
    text: str
    words: tuple[str, ...]
    count: int
    max_conf: float


def _clean(text: str) -> str:
    text = _FORMAT_MARKERS_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm_word(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def _words(phrase: str) -> tuple[str, ...]:
    return tuple(_norm_word(m.group(0)) for m in _WORD_RE.finditer(_clean(phrase)) if _norm_word(m.group(0)))


def _has_name_marker(words: tuple[str, ...]) -> bool:
    return any(w in _NAME_MARKERS for w in words)


def _looks_like_name(phrase: str, words: tuple[str, ...]) -> bool:
    if len(words) < 2:
        return False
    first = _clean(phrase).split()[0]
    if first in _LEADING_STOPWORDS and not _has_name_marker(words):
        return False
    if all(len(w) <= 2 for w in words):
        return False
    return True


def _looks_like_possible_variant(words: tuple[str, ...]) -> bool:
    return len(words) >= 2 and not all(len(w) <= 2 for w in words)


def _edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cur.append(min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (ca != cb),
            ))
        prev = cur
    return prev[-1]


def _extract_names(text: str) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for m in _NAME_RE.finditer(_clean(text)):
        phrase = m.group(0)
        words = _words(phrase)
        if _looks_like_name(phrase, words):
            out.append((phrase, words))
    return out


def _element_conf(el: dict) -> float:
    try:
        return float(el.get("conf", -1))
    except (TypeError, ValueError):
        return -1.0


def _candidate_match(words: tuple[str, ...], candidate: _Candidate) -> tuple[int, str] | None:
    if len(words) != len(candidate.words) or words == candidate.words:
        return None
    distances = [_edit_distance(a, b) for a, b in zip(words, candidate.words, strict=False)]
    changed = [d for d in distances if d]
    if not changed or len(changed) > 1:
        return None
    idx = next(i for i, d in enumerate(distances) if d)
    # Keep this very conservative: all unchanged words must match exactly, and
    # the differing token can only be a one-character OCR slip.
    if distances[idx] > 1:
        return None
    if words[idx] in _PRONOUNS or candidate.words[idx] in _PRONOUNS:
        return None
    if words[idx].endswith("s") and words[idx][:-1] == candidate.words[idx]:
        return None
    short_ibn_variant = {words[idx], candidate.words[idx]} == {"in", "ibn"}
    if min(len(words[idx]), len(candidate.words[idx])) < 3 and not short_ibn_variant:
        return None
    return sum(distances), candidate.text


def _learn_candidates(
    ocr_pages: list[dict],
    *,
    high_conf: float,
    min_occurrences: int,
) -> dict[int, list[_Candidate]]:
    counts: Counter[tuple[str, ...]] = Counter()
    displays: dict[tuple[str, ...], Counter[str]] = {}
    max_conf: dict[tuple[str, ...], float] = {}
    for page in ocr_pages:
        for el in page.get("elements", []):
            if el.get("kind") not in {"paragraph", "heading"}:
                continue
            conf = _element_conf(el)
            if 0 <= conf < high_conf:
                continue
            for phrase, words in _extract_names(el.get("text", "")):
                counts[words] += 1
                displays.setdefault(words, Counter())[phrase] += 1
                max_conf[words] = max(max_conf.get(words, -1), conf)

    by_len: dict[int, list[_Candidate]] = {}
    for words, count in counts.items():
        if count < min_occurrences and not _has_name_marker(words):
            continue
        display = displays[words].most_common(1)[0][0]
        by_len.setdefault(len(words), []).append(
            _Candidate(display, words, count, max_conf.get(words, -1))
        )
    for candidates in by_len.values():
        candidates.sort(key=lambda c: (-c.count, -c.max_conf, c.text))
    return by_len


def _name_counts(ocr_pages: list[dict]) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for page in ocr_pages:
        for el in page.get("elements", []):
            if el.get("kind") not in {"paragraph", "heading", "footnote"}:
                continue
            for _phrase, words in _extract_names(el.get("text", "")):
                counts[words] += 1
    return counts


def _correct_text(
    text: str,
    candidates: dict[int, list[_Candidate]],
    seen_counts: Counter[tuple[str, ...]],
    *,
    low_confidence: bool,
    dominant_min_occurrences: int,
    max_variant_occurrences: int,
) -> tuple[str, list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    changes: list[tuple[str, str, str]] = []
    reviews: list[tuple[str, str, str]] = []

    def repl(m: re.Match) -> str:
        phrase = m.group(0)
        # Capture the original tokens (preserving casing/punctuation) so we
        # can rebuild the phrase if we only need to correct a trailing
        # sub-phrase like the last 2 tokens of a 3-token capture.
        token_re = re.compile(r"\S+")
        orig_tokens = token_re.findall(phrase)
        words = _words(phrase)
        if not _looks_like_possible_variant(words):
            return phrase
        # Try matching the full word tuple first, then progressively shorter
        # trailing subsequences (this catches cases like "See In Arabi" where
        # only the trailing "In Arabi" is a known-name OCR variant).
        for start in range(len(words)):
            sub_words = words[start:]
            if not _looks_like_possible_variant(sub_words):
                continue
            matches = [
                (c, match) for c in candidates.get(len(sub_words), [])
                if (match := _candidate_match(sub_words, c)) is not None
            ]
            if len(matches) != 1:
                continue
            candidate, match = matches[0]
            evidence = (
                f"canonical_count={candidate.count}; "
                f"canonical_max_conf={candidate.max_conf:.1f}; "
                f"variant_count={seen_counts.get(sub_words, 0)}"
            )
            if not low_confidence:
                if candidate.count < dominant_min_occurrences:
                    reviews.append((phrase, match[1], f"{evidence}; reason=below_dominance_threshold"))
                    continue
                if seen_counts.get(sub_words, 0) > max_variant_occurrences:
                    reviews.append((phrase, match[1], f"{evidence}; reason=variant_not_rare"))
                    continue
            prefix = " ".join(orig_tokens[:start])
            replacement = match[1]
            corrected = f"{prefix} {replacement}" if prefix else replacement
            changes.append((phrase, corrected, evidence))
            return corrected
        return phrase

    return _NAME_RE.sub(repl, text), changes, reviews


def post_structure(cfg, ocr_pages: list[dict]) -> None:
    settings = cfg.plugin_config("common_ocr") or {}
    high_conf = float(settings.get("high_confidence", 88))
    low_conf = float(settings.get("low_confidence", 82))
    min_occurrences = int(settings.get("min_occurrences", 2))
    dominant_min_occurrences = int(settings.get("dominant_min_occurrences", 5))
    max_variant_occurrences = int(settings.get("max_variant_occurrences", 1))
    candidates = _learn_candidates(
        ocr_pages,
        high_conf=high_conf,
        min_occurrences=min_occurrences,
    )
    if not candidates:
        return
    seen_counts = _name_counts(ocr_pages)
    total = 0
    for page in ocr_pages:
        for el in page.get("elements", []):
            if el.get("kind") not in {"paragraph", "heading", "footnote"}:
                continue
            conf = _element_conf(el)
            low_confidence = 0 <= conf <= low_conf
            before = el.get("text", "")
            corrected, changes, reviews = _correct_text(
                before,
                candidates,
                seen_counts,
                low_confidence=low_confidence,
                dominant_min_occurrences=dominant_min_occurrences,
                max_variant_occurrences=max_variant_occurrences,
            )
            if changes:
                el["text"] = corrected
                el["_common_ocr_corrections"] = len(changes)
                for original, replacement, evidence in changes:
                    ocr_corrections.record_correction(
                        cfg,
                        source="common_ocr",
                        rule="learned_proper_name",
                        page=page.get("pno", ""),
                        element=el,
                        before=original,
                        after=replacement,
                        evidence=evidence,
                    )
                total += len(changes)
            for original, suggested, evidence in reviews:
                ocr_corrections.record_review(
                    cfg,
                    kind="uncertain proper-name OCR variant",
                    page=page.get("pno", ""),
                    excerpt=f"{original} ({evidence})",
                    suggested_fix=suggested,
                )
    if total:
        print(f"[quire] common_ocr: corrected {total} proper-name OCR slips", file=sys.stderr)
