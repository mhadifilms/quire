"""Body-text mojibake cleanup.

The PDF text layer often maps inline-Arabic glyphs to Latin codepoints. After
Vision OCR re-extracts the rendered glyphs we end up with gibberish like
``4...a! ¿Usin j›`` or ``jes`` interlaced with English. This plugin strips
those clusters out, replacing them with a discreet ``[Arabic]`` /
``[Quran s:a]`` placeholder so the prose still reads.
"""

from __future__ import annotations

import re

from . import ocr_corrections

_MOJIBAKE_CHARS = "\u2039\u203a\u00bf\u00a1\u00ab\u00bb\u00b0"
_MOJIBAKE_TOKEN_RE = re.compile(
    rf"\S*[{_MOJIBAKE_CHARS}]\S*(?:\s+\S+){{0,4}}?[{_MOJIBAKE_CHARS}\)\]]"
)
_INLINE_CUE_RE = re.compile(
    r"(?P<cue>(?:\b(?:verse|wording|name|states?|saying|phrase|word|reads?|says?)\b|,)\s+)"
    r"(?P<gib>(?:[a-zA-Z]+'?[a-zA-Z]*[!?]?\s+){1,4})"
    r"(?P<after>\b(?:in this verse|in this passage|expresses|can demonstrate|states that|of [Aa]llah|here|means|is the|—)\b)",
    re.IGNORECASE,
)

_SAFE_SHORT = {
    "the", "a", "an", "of", "is", "in", "at", "on", "by", "for", "to", "or",
    "and", "but", "yet", "so", "if", "as", "be", "do", "go", "he", "it",
    "we", "us", "no", "not", "all", "any", "one", "two", "three",
    "you", "his", "her", "its", "our", "are", "was", "had", "has", "may",
    "can", "old", "new", "see", "say", "this",
    "very", "such", "from", "with", "into", "they", "them", "what", "when",
    "while", "where", "than", "then", "thus", "also", "even", "much",
    "more", "most", "many", "less",
}

_KNOWN_TRANSLATED_INLINE = {
    "and for allah": "وَلِلَّهِ",
}

_TRANSLATED_INLINE_RE = re.compile(
    r"\b(?P<cue>phrase|word|term|wording)\s+"
    r"(?P<gib>[A-Za-z]{1,5}['’]?)\s+"
    r"\((?P<quote>['‘’\"]?)(?P<meaning>and for Allah)(?P=quote)\)",
    re.IGNORECASE,
)

_COMMON_OCR_WORD_FIXES = {
    "anyching": "anything",
    "Anyching": "Anything",
    "atig": "atiq",
    "Atig": "Atiq",
    "attribured": "attributed",
    "Attribured": "Attributed",
    "atrachments": "attachments",
    "Atrachments": "Attachments",
    "berween": "between",
    "Berween": "Between",
    "chastiry": "chastity",
    "Chastiry": "Chastity",
    "che": "the",
    "Che": "The",
    "cheir": "their",
    "Cheir": "Their",
    "chrowing": "throwing",
    "Chrowing": "Throwing",
    "cime": "time",
    "Cime": "Time",
    "circuir": "circuit",
    "Circuir": "Circuit",
    "concinues": "continues",
    "Concinues": "Continues",
    "conducr": "conduct",
    "Conducr": "Conduct",
    "curting": "cutting",
    "Curting": "Cutting",
    "bur": "but",
    "Bur": "But",
    "chat": "that",
    "Chat": "That",
    "chis": "this",
    "Chis": "This",
    "cherefore": "therefore",
    "Cherefore": "Therefore",
    "definicion": "definition",
    "Definicion": "Definition",
    "debr": "debt",
    "Debr": "Debt",
    "deceprive": "deceptive",
    "Deceprive": "Deceptive",
    "delegared": "delegated",
    "Delegared": "Delegated",
    "deliberarely": "deliberately",
    "Deliberarely": "Deliberately",
    "derails": "details",
    "Derails": "Details",
    "derermination": "determination",
    "Derermination": "Determination",
    "devorional": "devotional",
    "Devorional": "Devotional",
    "Darajar": "Darajat",
    "dissolure": "dissolute",
    "Dissolure": "Dissolute",
    "distribured": "distributed",
    "Distribured": "Distributed",
    "dury": "duty",
    "Dury": "Duty",
    "edired": "edited",
    "Edired": "Edited",
    "empry": "empty",
    "Empry": "Empty",
    "excepr": "except",
    "Excepr": "Except",
    "thoes": "those",
    "Thoes": "Those",
    "footnores": "footnotes",
    "Footnores": "Footnotes",
    "ourwardly": "outwardly",
    "Ourwardly": "Outwardly",
    "entirery": "entirety",
    "Entirery": "Entirety",
    "entiry": "entity",
    "Entiry": "Entity",
    "actribures": "attributes",
    "Actribures": "Attributes",
    "actributes": "attributes",
    "Actributes": "Attributes",
    "accurare": "accurate",
    "Accurare": "Accurate",
    "ream": "team",
    "Ream": "Team",
    "elsc": "else",
    "Elsc": "Else",
    "fachers": "fathers",
    "Fachers": "Fathers",
    "matrer": "matter",
    "Matrer": "Matter",
    "mencioned": "mentioned",
    "Mencioned": "Mentioned",
    "nighs": "night",
    "Nighs": "Night",
    "ninch": "ninth",
    "Ninch": "Ninth",
    "nore": "note",
    "Nore": "Note",
    "abour": "about",
    "Abour": "About",
    "banquer": "banquet",
    "Banquer": "Banquet",
    "Beirur": "Beirut",
    "Furuhat": "Futuhat",
    "Fusuhat": "Futuhat",
    "Futubat": "Futuhat",
    "Futuhas": "Futuhat",
    "grearest": "greatest",
    "Grearest": "Greatest",
    "gares": "gates",
    "Gares": "Gates",
    "ghust": "ghusl",
    "Ghust": "Ghusl",
    "indicares": "indicates",
    "Indicares": "Indicates",
    "inherens": "inherent",
    "Inherens": "Inherent",
    "innare": "innate",
    "Innare": "Innate",
    "Institure": "Institute",
    "Kirabchi": "Kitabchi",
    "kilomerers": "kilometers",
    "Kilomerers": "Kilometers",
    "Korob": "Kotob",
    "lirerally": "literally",
    "Lirerally": "Literally",
    "lierally": "literally",
    "Lierally": "Literally",
    "obligarory": "obligatory",
    "Obligarory": "Obligatory",
    "ourward": "outward",
    "Ourward": "Outward",
    "ourskirts": "outskirts",
    "Ourskirts": "Outskirts",
    "perfecr": "perfect",
    "Perfecr": "Perfect",
    "pracrice": "practice",
    "Pracrice": "Practice",
    "Propher": "Prophet",
    "prorected": "protected",
    "Prorected": "Protected",
    "prorection": "protection",
    "Prorection": "Protection",
    "realizarion": "realization",
    "Realizarion": "Realization",
    "rechnically": "technically",
    "Rechnically": "Technically",
    "relared": "related",
    "Relared": "Related",
    "rellect": "intellect",
    "Rellect": "Intellect",
    "rempts": "tempts",
    "Rempts": "Tempts",
    "representarive": "representative",
    "Representarive": "Representative",
    "restifying": "testifying",
    "Restifying": "Testifying",
    "restimony": "testimony",
    "Restimony": "Testimony",
    "righreous": "righteous",
    "Righreous": "Righteous",
    "rires": "rites",
    "Rires": "Rites",
    "rotal": "total",
    "Rotal": "Total",
    "rurn": "turn",
    "Rurn": "Turn",
    "roward": "toward",
    "Roward": "Toward",
    "rwo": "two",
    "Rwo": "Two",
    "safery": "safety",
    "Safery": "Safety",
    "self-blamng": "self-blaming",
    "Self-blamng": "Self-blaming",
    "ser": "set",
    "Ser": "Set",
    "Sociery": "Society",
    "stares": "states",
    "Stares": "States",
    "supplicares": "supplicates",
    "Supplicares": "Supplicates",
    "tempcations": "temptations",
    "Tempcations": "Temptations",
    "thar": "that",
    "Thar": "That",
    "thirsry": "thirsty",
    "Thirsry": "Thirsty",
    "tidles": "titles",
    "Tidles": "Titles",
    "translared": "translated",
    "Translared": "Translated",
    "throughour": "throughout",
    "Throughour": "Throughout",
    "unire": "unite",
    "Unire": "Unite",
    "utering": "uttering",
    "Utering": "Uttering",
    "utrerance": "utterance",
    "Utrerance": "Utterance",
    "wealch": "wealth",
    "Wealch": "Wealth",
    "whire": "white",
    "Whire": "White",
    "wich": "with",
    "Wich": "With",
    "withour": "without",
    "Withour": "Without",
    "welfth": "twelfth",
    "Welfth": "Twelfth",
}

_COMMON_OCR_WORD_RE = re.compile(
    r"\b(?:"
    + "|".join(re.escape(w) for w in sorted(_COMMON_OCR_WORD_FIXES, key=len, reverse=True))
    + r")\b"
)


_PROPER_NOUN_RE = re.compile(r"^[A-Z][a-z]{2,}$")
_LOWERCASE_WORD_RE = re.compile(r"^[a-z]+(?:'[a-z]+)?$")
_VOWEL_RE = re.compile(r"[aeiouy]")


def _token_is_plausible_english(token: str) -> bool:
    """Return True if ``token`` looks like real English text rather than
    OCR mojibake.

    Used by the cue-based mojibake replacement (Pass 2 in ``_clean_text``)
    to bail out when the candidate span is plausible English. The rule
    is intentionally permissive — we'd rather leave a fragment of
    gibberish in the body for human QC than corrupt a legitimate word
    like ``which``, ``Quran``, or ``Marwah``.

    Plausible English is one of:

    1. A short common word in ``_SAFE_SHORT`` (``the``, ``and``, ``of``…).
    2. A capitalized proper-noun-shaped token (``[A-Z][a-z]{2,}``) such
       as ``Kaaba``, ``Marwah``, ``Quran``, ``Allah``.
    3. A lowercase token of 4+ letters (optional internal apostrophe)
       containing at least one vowel — ``which``, ``while``, ``inward``,
       ``aspect``, ``heaven``, ``don't``.

    Pure-letter mojibake clusters like ``ji``, ``pal``, ``Eji`` fail
    rules 2 and 3 (2 letters, no proper-noun shape) and let the cue
    pass fire.
    """
    bare = re.sub(r"[!?,.;:]+$", "", token)
    if not bare:
        return True
    if bare.lower() in _SAFE_SHORT:
        return True
    if _PROPER_NOUN_RE.match(bare):
        return True
    if _LOWERCASE_WORD_RE.match(bare):
        letters = bare.replace("'", "")
        if len(letters) >= 4 and _VOWEL_RE.search(bare):
            return True
    return False


def _placeholder(citations: list[tuple[int, int]] | None) -> str:
    if citations:
        s, a = citations[0]
        return f"[Quran {s}:{a}]"
    return "[Arabic]"


def _clean_common_ocr_text(text: str, cfg, page: dict, el: dict) -> str:
    return ocr_corrections.apply_exact_fixes(
        text,
        cfg,
        page=page.get("pno", ""),
        element=el,
        source="mojibake_cleanup",
    )


def _clean_text(text: str, citations: list[tuple[int, int]] | None) -> str:
    out = text

    def _translated_inline_repl(m: re.Match) -> str:
        meaning = m.group("meaning").lower()
        canonical = _KNOWN_TRANSLATED_INLINE.get(meaning)
        if not canonical:
            return m.group(0)
        quote = m.group("quote") or "'"
        return (
            f"{m.group('cue')} \x10{canonical}\x11 "
            f"({quote}{m.group('meaning')}{quote})"
        )

    out = _TRANSLATED_INLINE_RE.sub(_translated_inline_repl, out)

    # Pass 1: special-character mojibake
    if any(c in out for c in _MOJIBAKE_CHARS):
        for _ in range(4):
            m = _MOJIBAKE_TOKEN_RE.search(out)
            if not m:
                break
            start = m.start()
            while start > 0:
                c = out[start - 1]
                if c == " ":
                    if re.search(r"[A-Za-z]{4,}\s*$", out[:start]):
                        break
                    start -= 1
                elif c.isalpha() or c.isdigit() or c in ".!?,_-…":
                    start -= 1
                else:
                    break
            out = out[:start].rstrip() + " " + _placeholder(citations) + out[m.end():]

    # Pass 2: cue-based mojibake.
    #
    # The cue pattern matches sentence shapes like ``, X is the …`` or
    # ``the wording X means …``. Without any further guard this catches
    # genuine OCR mojibake (``, jes is the …``), but it ALSO catches
    # ordinary English prose (``, the Quran states that …``,
    # ``, the Kaaba is the reflection of …``) and destroys it. We err
    # on the side of caution: a replacement only fires when at least one
    # token in the candidate span looks suspicious AND none of the tokens
    # look like real English. See ``_token_is_plausible_english``.
    def _gib_repl(m: re.Match) -> str:
        gib = m.group("gib").strip()
        toks = [t for t in re.split(r"\s+", gib) if t]
        clean = [re.sub(r"[!?,.;:]+$", "", t).lower() for t in toks]
        if not clean or any(len(t) > 6 for t in clean):
            return m.group(0)
        if all(_token_is_plausible_english(t) for t in toks):
            return m.group(0)
        unknown = [t for t in clean if t not in _SAFE_SHORT]
        if not unknown:
            return m.group(0)
        if len(unknown) >= max(1, (len(clean) + 1) // 2):
            return f"{m.group('cue')}{_placeholder(citations)} {m.group('after')}"
        return m.group(0)

    out = _INLINE_CUE_RE.sub(_gib_repl, out)
    return out


def post_structure(cfg, ocr_pages: list[dict]) -> None:
    """Apply mojibake cleanup to text-bearing elements."""
    for page in ocr_pages:
        cits: list = page.get("_quran_citations") or []
        for el in page.get("elements", []):
            if el.get("kind") in ("paragraph", "heading"):
                el["text"] = _clean_common_ocr_text(el["text"], cfg, page, el)
                el["text"] = _clean_text(el["text"], cits)
            elif el.get("kind") == "footnote":
                el["text"] = _clean_common_ocr_text(el["text"], cfg, page, el)
