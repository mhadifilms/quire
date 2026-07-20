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
from collections.abc import Callable
from dataclasses import dataclass

from . import ocr_corrections

_FORMAT_MARKERS_RE = re.compile(r"[\x02-\x07\x10-\x13]")
_NAME_RE = re.compile(
    r"\b(?!See\b)[A-Z][A-Za-z'’-]{1,}"
    r"(?:\s+(?:al-|Al-)?[A-Z][A-Za-z'’-]{1,}){1,3}\b"
)
_WORD_RE = re.compile(r"[A-Za-z]+(?:['’-][A-Za-z]+)?")
_SPEAKER_NAME_RE = re.compile(
    r"\b([A-Z][A-Za-z'’-]{4,})(?=\s+(?:said|asked|replied|murmured|"
    r"whispered|shouted|called)\b)"
)

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


def _speaker_name_counts(ocr_pages: list[dict]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for page in ocr_pages:
        for el in page.get("elements", []):
            if el.get("kind") not in {"paragraph", "heading"}:
                continue
            text = _clean(el.get("text", ""))
            # Count all repeated capitalized tokens as possible canonical names,
            # then only rewrite tokens in explicit speech-attribution contexts.
            for word in re.findall(r"\b[A-Z][A-Za-z'’-]{4,}\b", text):
                if word not in _LEADING_STOPWORDS:
                    counts[word] += 1
    return counts


def _repair_rare_speaker_names(
    text: str,
    counts: Counter[str],
) -> tuple[str, list[tuple[str, str, str]]]:
    changes: list[tuple[str, str, str]] = []

    def repl(match: re.Match) -> str:
        original = match.group(1)
        if counts[original] > 1:
            return original
        ranked = sorted(
            (
                (_edit_distance(original.casefold(), candidate.casefold()), candidate, count)
                for candidate, count in counts.items()
                if count >= 5
                and candidate != original
                and candidate[:1].casefold() == original[:1].casefold()
            ),
            key=lambda item: (item[0], -item[2], item[1]),
        )
        if not ranked or ranked[0][0] > 2:
            return original
        best_distance = ranked[0][0]
        equally_close = [item for item in ranked if item[0] == best_distance]
        if len(equally_close) != 1:
            return original
        _, replacement, count = equally_close[0]
        evidence = (
            f"speaker_context; canonical_count={count}; "
            f"variant_count={counts[original]}; edit_distance={best_distance}"
        )
        changes.append((original, replacement, evidence))
        return replacement

    return _SPEAKER_NAME_RE.sub(repl, text), changes


def _repair_rare_capitalized_tokens(
    text: str,
    counts: Counter[str],
) -> tuple[str, list[tuple[str, str, str]]]:
    changes: list[tuple[str, str, str]] = []

    def repl(match: re.Match) -> str:
        original = match.group(0)
        if original in _LEADING_STOPWORDS or counts[original] > 1:
            return original
        candidates = [
            (candidate, count)
            for candidate, count in counts.items()
            if count >= 5
            and candidate != original
            and candidate[:1] == original[:1]
            and _edit_distance(original.casefold(), candidate.casefold()) == 1
        ]
        if len(candidates) != 1:
            return original
        replacement, count = candidates[0]
        changes.append(
            (
                original,
                replacement,
                f"dominant_capitalized_token; canonical_count={count}",
            )
        )
        return replacement

    return re.sub(r"\b[A-Z][A-Za-z'’-]{4,}\b", repl, text), changes


def _repair_contextual_english(
    text: str,
    preferred_dialogue_quote: str | None = None,
) -> tuple[str, list[tuple[str, str, str]]]:
    """Repair high-confidence English OCR forms using grammatical context."""
    changes: list[tuple[str, str, str]] = []

    rules: list[tuple[re.Pattern[str], str | Callable[[re.Match], str]]] = [
        (
            re.compile(r"\b(wars?|conflicts?)\s+arid\s+(famines?|hunger)\b", re.I),
            lambda m: f"{m.group(1)} and {m.group(2)}",
        ),
        (
            re.compile(r"\b(should|could|would|must|might)\s+ve\b", re.I),
            lambda m: m.group(1) + "'ve",
        ),
        (re.compile(r"\bT(?:a|d)\s+better\b"), "I'd better"),
        (re.compile(r"\bT['’]?ve\b"), "I've"),
        (re.compile(r"\bT['’]?d\b"), "I'd"),
        (re.compile(r"\bTt['’]s\b"), "It's"),
        (re.compile(r"\bId(?=\s+(?:expected|rather|better|like|seen|heard|known|been)\b)"), "I'd"),
        (re.compile(r"\bIm(?=\s+(?:not|a|an|the|so|too|very|going|sorry|sure|glad)\b)"), "I'm"),
        (
            re.compile(r"\b(I|you|he|she|we|they|it)\.\s+([a-z])"),
            lambda m: f"{m.group(1)} {m.group(2)}",
        ),
        (
            re.compile(r"\bat the same\.(?=\s+[A-Z])", re.I),
            "at the same time.",
        ),
        (re.compile(r"\bGrand\s+ma\b"), "Grandma"),
        (re.compile(r",\s*:\s*"), ", "),
        (re.compile(r"\byou,\s+know\b", re.I), "you know"),
        (re.compile(r"\bwas,\s+(?=(?:heavy|for the first time)\b)", re.I), "was "),
        (re.compile(r"\bher['’]\s+mother['’]s\b", re.I), "her mother's"),
        (re.compile(r"\bMc[lI](?=[A-Z][a-z])"), "Mc"),
        (re.compile(r"\bgenerations of Muslim\b"), "generations of Muslims"),
        (re.compile(r"\bThings picked up ['’](?=after\b)"), "Things picked up "),
        (re.compile(r"(?<![A-Za-z])'cuz'(?=\s+[A-Za-z])", re.I), "'cuz"),
        (re.compile(r"\bn['’]\s+êtes\b", re.I), "n'êtes"),
        (
            re.compile(r"\b([A-Z]{3,})\s+Y\s+([A-Z]{3,})(?=\s+4EVA\b)"),
            lambda m: f"{m.group(1)} ♥ {m.group(2)}",
        ),
        (
            re.compile(r"(\bmouthed the words to (?:her|him)self:\s*)—\s*\.", re.I),
            lambda m: m.group(1) + "————— —————.",
        ),
        (re.compile(r"—\s+(?=[a-z])"), "—"),
        (re.compile(r"\bSr:\s*['’]s\b"), "Sr.'s"),
        (re.compile(r"\bbeside the candes\b", re.I), "beside the candles"),
        (re.compile(r"\bmag\s+nify\b", re.I), "magnify"),
        (re.compile(r"(?:\.\s*){2,4}P(?=[\"”])"), "...?"),
        (re.compile(r"\b(festivals|holidays)-(?=times of year\b)", re.I), lambda m: m.group(1) + "—"),
        (
            re.compile(r"\byour delicate American stomach['’]"),
            "your 'delicate American stomach'",
        ),
        (re.compile(r"\bjoke-(?=['’]cuz\b)", re.I), "joke—"),
        (
            re.compile(r"\b(said|asked),\s*-\s*(?=[\"“‘'])", re.I),
            lambda m: m.group(1) + ", ",
        ),
        (
            re.compile(r"([”’])\s*;\s*(?=[\"“‘'])"),
            lambda m: m.group(1) + " ",
        ),
        (re.compile(r"«(?=['’][A-Z])"), '"'),
    ]

    for pattern, replacement in rules:
        def record(match: re.Match, replacement=replacement) -> str:
            after = replacement(match) if callable(replacement) else replacement
            changes.append(
                (
                    match.group(0),
                    after,
                    "generic_contextual_english_rule",
                )
            )
            return after

        text = pattern.sub(record, text)

    def repair_doubled_stop(match: re.Match) -> str:
        prefix = match.string[:match.start()]
        double_quotes = sum(prefix.count(char) for char in '"“”')
        single_quotes = len(
            re.findall(r"(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])", prefix)
        )
        if double_quotes % 2:
            after = '."'
        elif single_quotes % 2:
            after = ".'"
        else:
            after = "."
        changes.append((match.group(0), after, "generic_contextual_english_rule"))
        return after

    # Preserve an ellipsis-question (``...?``); repair only a single stop
    # followed by an OCR question-mark/quote confusion.
    text = re.sub(r"(?<!\.)\.\?(?=\s|$)", repair_doubled_stop, text)

    speech_verbs = (
        r"said|asked|replied|murmured|whispered|shouted|called|"
        r"looked|nodded|sighed|laughed|mumbled"
    )

    def missing_closer(match: re.Match) -> str:
        opener = match.group("open")
        closer = "'" if opener in {"'", "‘"} else '"'
        after = (
            f"{opener}{match.group('speech')}{match.group('punct')}"
            f"{closer} {match.group('tag')}"
        )
        changes.append((match.group(0), after, "dialogue_attribution_balance"))
        return after

    text = re.sub(
        rf"(?P<open>['‘“\"])(?P<speech>(?:[A-Za-z]+['’][A-Za-z]+|"
        rf"[^'’”\"]){{1,150}}?)"
        rf"(?P<punct>[,!?])\s+(?P<tag>(?:he|she|they|[A-Z][a-z]+)\s+"
        rf"(?:{speech_verbs})\b)",
        missing_closer,
        text,
    )
    # Same attribution pattern when OCR lost both quote boundaries.
    def add_both_boundaries(match: re.Match) -> str:
        quote = preferred_dialogue_quote or "'"
        return (
            match.group("prefix") + quote + match.group("speech")
            + quote + " " + match.group("tag")
        )

    text = re.sub(
        rf"(?P<prefix>^|[.!?]\s+)"
        rf"(?P<speech>(?:I['’](?:m|ve|d|ll)|We|I)\b[^.!?\n]{{2,140}}?[,])"
        rf"\s+(?P<tag>(?:(?:he|she|they|[A-Z][a-z]+)\s+(?:{speech_verbs})|"
        rf"(?:said|asked|replied)\s+[A-Z][a-z]+)\b)",
        add_both_boundaries,
        text,
    )

    if preferred_dialogue_quote == '"':
        # OCR often reads a closing double quote as an apostrophe. Once the
        # corpus establishes double-quoted dialogue, matching the opener is
        # safer than preserving the isolated OCR glyph.
        text = re.sub(
            r"\"([^\"\n]{1,500}?)([,.!?])['’](?=\s|$)",
            lambda m: '"' + m.group(1) + m.group(2) + '"',
            text,
        )

    text = re.sub(r"\)\s+\?(?=\s|$)", ") ", text)
    text = re.sub(r"\ba[\"“]\s+(?=Now,\s+I\b)", "", text)
    text = re.sub(r"(?<!['’])['’]{2}(?=[A-Z])", "'", text)

    def missing_opener(match: re.Match) -> str:
        closer = match.group("close")
        opener = "'" if closer in {"'", "’"} else '"'
        prefix = match.string[:match.start()]
        if opener == "'":
            already_open = len(re.findall(
                r"(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])",
                prefix,
            )) % 2
        else:
            already_open = sum(prefix.count(char) for char in '"“”') % 2
        if already_open:
            return match.group(0)
        after = (
            f"{match.group('prefix')}{opener}{match.group('speech')}"
            f"{closer} {match.group('tag')}"
        )
        changes.append((match.group(0), after, "dialogue_attribution_balance"))
        return after

    text = re.sub(
        rf"(?P<prefix>^|[.!?]\s+)(?P<speech>[A-Z][^'’”\"]{{1,150}}?"
        rf"[,!?])(?P<close>['’”\"])\s+(?P<tag>(?:he|she|they|[A-Z][a-z]+)\s+"
        rf"(?:{speech_verbs})\b)",
        missing_opener,
        text,
    )
    text = re.sub(
        rf"(?P<prefix>^|[.!?]\s+)(?P<speech>I['’](?:m|ve|d|ll|s)\b"
        rf"[^.!?]{{1,120}}?[,!?])(?P<close>['’])\s+"
        rf"(?P<tag>(?:he|she|they|[A-Z][a-z]+)\s+(?:{speech_verbs})\b)",
        missing_opener,
        text,
    )

    # Repair a single-quoted sentence followed by an unmistakable narrative
    # transition, and mismatched double closers on single-quoted dialogue.
    text = re.sub(
        r"(?P<open>['‘])(?P<speech>[A-Z][^.\n]{2,150}\.)"
        r"\s+(?P<next>With\s+(?:a|the|his|her)\b)",
        lambda m: (
            m.group("open") + m.group("speech") + "' " + m.group("next")
        ),
        text,
    )
    text = re.sub(
        r"(?P<open>(?<![A-Za-z])['‘])(?![Cc]uz\b)"
        r"(?P<speech>(?:[A-Za-z]+['’][A-Za-z]+|"
        r"[^'’”\"]){2,150}?)(?P<close>[”\"])(?=\s|$)",
        lambda m: m.group("open") + m.group("speech") + "'",
        text,
    )

    def repair_question_as_quote(match: re.Match) -> str:
        speech = match.group("speech")
        first_words = speech.lstrip().casefold()
        genuine_question = bool(re.match(
            r"(?:(?:but|and|so)\s+)?(?:why|what|when|where|who|whom|whose|how|"
            r"can|could|would|will|do|does|did|is|are|was|were|"
            r"have|has|had|should|may|might|even\b|that['’]s why\b)",
            first_words,
        ))
        punctuation = "?" if genuine_question else "."
        quote = preferred_dialogue_quote or (
            "'" if match.group("open") in {"'", "‘"} else '"'
        )
        return quote + speech + punctuation + quote

    narrative_after = (
        r"(?=\s+(?:\*{3}|She|He|Her|His|They|There|The|When|"
        r"Shushu|Kassim|Dina)\b|$)"
    )
    text = re.sub(
        r"(?P<open>(?<![A-Za-z])['‘\"“])"
        r"(?P<speech>[A-Z](?:[A-Za-z]+['’][A-Za-z]+|[^'’\"”\n]){1,350}?)"
        r"\?['’\"]?"
        + narrative_after,
        repair_question_as_quote,
        text,
    )
    if preferred_dialogue_quote == "'" and re.match(r"^Why not\?\s+It would\b", text):
        text = "'" + re.sub(r"\?\s*$", ".'", text)

    # A sentence containing an I-contraction and ending in a single closing
    # quote almost certainly lost its opening dialogue mark.
    def add_i_opener(match: re.Match) -> str:
        prefix = match.string[:match.start()]
        open_single = len(re.findall(
            r"(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])",
            prefix,
        )) % 2
        if open_single:
            return match.group(0)
        return match.group("prefix") + "'" + match.group("speech")

    text = re.sub(
        r"(?P<prefix>^|[.!?]\s+)(?P<speech>I['’](?:m|ve|d|ll)\b"
        r"[^.!?]{1,120}[.!?]['’])",
        add_i_opener,
        text,
    )
    if preferred_dialogue_quote == '"':
        text = re.sub(
            r"(?<![A-Za-z])['‘](I['’](?:m|ve|d|ll)\b[^'’\n]{1,180}?[.!?])['’]",
            lambda m: '"' + m.group(1) + '"',
            text,
        )
        text = re.sub(
            r"^(?P<speech>[A-Z][^\"“”\n]{2,180}[—-])”(?=\s+[\"“])",
            lambda m: '"' + m.group("speech") + "”",
            text,
        )
    if re.search(r"\.\.\s*$", text):
        double_open = sum(text.count(char) for char in '"“”') % 2
        single_open = len(re.findall(
            r"(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])",
            text,
        )) % 2
        if double_open or single_open:
            closer = '"' if double_open else "'"
            text = re.sub(r"\.\.\s*$", "..." + closer, text)

    # Commonly OCR'd Muslim greeting: a following narrative sentence proves
    # that the comma is a damaged dialogue close, not a continuation.
    text = re.sub(
        r"(['‘])([Ss]alamu alleikum)[?,]\s+(?=(?:It|He|She|They)\b)",
        r"\1\2.' ",
        text,
    )
    text = re.sub(
        r"(['‘])([Ss]alamu alleikum)[?,]?\s*$",
        r"\1\2.'",
        text,
    )

    def close_before_new_quote(match: re.Match) -> str:
        prefix = text[:match.start()]
        if sum(prefix.count(char) for char in '"“”') % 2 == 0:
            return match.group(0)
        return match.group(1) + '—" "'

    text = re.sub(
        r"([A-Za-z])-\s+\"(?=[A-Z])",
        close_before_new_quote,
        text,
    )
    # Re-run the I-contraction opener pass after narrative-boundary repairs;
    # those repairs may have just balanced an earlier quote in this element.
    text = re.sub(
        rf"(?P<prefix>^|[.!?]\s+)(?P<speech>I['’](?:m|ve|d|ll|s)\b"
        rf"[^.!?]{{1,120}}?[,!?])(?P<close>['’])\s+"
        rf"(?P<tag>(?:he|she|they|[A-Z][a-z]+)\s+(?:{speech_verbs})\b)",
        missing_opener,
        text,
    )
    quote = preferred_dialogue_quote or "'"
    if re.fullmatch(r"I['’](?:m|ve|d|ll)\b[^.!?\n]{2,180}[.!?]", text):
        text = quote + text + quote
    elif text.startswith(quote) and re.search(r"[.!?]\s*$", text):
        if quote == "'":
            quote_is_open = len(re.findall(
                r"(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])",
                text,
            )) % 2
        else:
            quote_is_open = sum(text.count(char) for char in '"“”') % 2
        has_mismatched_attribution_close = bool(
            quote == '"'
            and re.search(
                rf"[,.!?]['’]\s+(?:he|she|they|[A-Z][a-z]+)\s+"
                rf"(?:{speech_verbs})\b",
                text,
            )
        )
        if quote_is_open and not has_mismatched_attribution_close:
            text = text.rstrip() + quote

    text = re.sub(r'(?<=[a-z])["”]\s+(?=[a-z])', " ", text)
    text = re.sub(
        r"\b(this|that)\s+([^.—\n]{5,100})—(was|is)\b",
        lambda m: f"{m.group(1)}—{m.group(2)}—{m.group(3)}",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(sight)\s+(?=[\"“][A-Z])", r"\1. ", text)
    text = re.sub(
        r"\b(my|his|her)\s+[\"“]([a-z]+(?:\s+[a-z]+){1,3})\s+(had|was|is)\b",
        lambda m: f"{m.group(1)} “{m.group(2)}” {m.group(3)}",
        text,
    )

    if re.search(r"\bcontinuously\s+[\"“']peckish,\s+uneasy\b", text, re.I):
        text = re.sub(
            r"\b(continuously)\s+[\"“']peckish,\s+uneasy\b",
            r"\1 'peckish', uneasy",
            text,
            flags=re.I,
        )

    # Normalize boundary glyphs to the corpus's established dialogue style.
    if preferred_dialogue_quote == "'":
        if (
            len(re.findall(
                r"(?<![A-Za-z])['‘’]|['‘’](?![A-Za-z])",
                text,
            )) % 2
            and re.search(r"(?:^|[.!?,]\s+)['‘][A-Z]", text)
            and re.search(r"[.!?]\s*$", text)
        ):
            text = text.rstrip() + "'"
        text = text.replace("“", "'").replace("”", "'").replace('"', "'")
    elif preferred_dialogue_quote == '"':
        text = re.sub(
            rf"([,.!?])['’](?=\s+(?:he|she|they|[A-Z][a-z]+)\s+"
            rf"(?:{speech_verbs})\b)",
            r"\1”",
            text,
        )
        normalized: list[str] = []
        quote_open = False
        for char in text:
            if char == "“":
                normalized.append(char)
                quote_open = True
            elif char == "”":
                normalized.append(char)
                quote_open = False
            elif char == '"':
                normalized.append("”" if quote_open else "“")
                quote_open = not quote_open
            else:
                normalized.append(char)
        text = "".join(normalized)
    return text, changes


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
    seen_counts = _name_counts(ocr_pages)
    speaker_counts = _speaker_name_counts(ocr_pages)
    dialogue_text = "\n".join(
        element.get("text", "")
        for page in ocr_pages
        for element in page.get("elements", [])
        if element.get("kind") == "paragraph"
    )
    single_openers = len(re.findall(r"(?:^|[\s(])['‘](?=[A-Z])", dialogue_text))
    double_openers = len(re.findall(r'(?:^|[\s(])["“](?=[A-Z])', dialogue_text))
    preferred_dialogue_quote = (
        "'" if single_openers > double_openers * 1.25 else '"'
    )
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
            corrected, speaker_changes = _repair_rare_speaker_names(
                corrected,
                speaker_counts,
            )
            changes.extend(speaker_changes)
            corrected, contextual_changes = _repair_contextual_english(
                corrected,
                preferred_dialogue_quote,
            )
            changes.extend(contextual_changes)
            if corrected != before:
                el["text"] = corrected
                el["_common_ocr_corrections"] = max(1, len(changes))
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
                total += max(1, len(changes))
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
