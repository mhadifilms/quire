"""Post-render typography fixes for rendered chapter XHTML.

These fixes operate on the *rendered* chapter XHTML (after ``render_chapter``
and before ``build_epub``), correcting OCR / typesetting artifacts that
only surface at the rendered layer:

  - Hyphen-across-line-break stitching (``manu- script`` -> ``manuscript``).
    The OCR engine preserves the trailing hyphen of a soft-broken word as
    literal text. We rejoin if the joined token is already a word seen
    elsewhere in the corpus.

  - Loose footnote-digit conversion (``;3 rather`` -> ``;\u00b3 rather``,
    ``Smith1 has`` -> ``Smith\u00b9 has``). A superscript footnote marker
    misread as inline text becomes a Unicode superscript digit. The link
    target is lost in this conversion (we don't know which ``<aside>`` the
    digit pointed at), but the marker stops reading as inline content.

  - Footnote-misread quote stripping (``Smith" states`` -> ``Smith
    states``, ``essence." In`` -> ``essence. In``). A superscript footnote
    digit misread by OCR as ``"`` is removed when it sits between a word /
    sentence boundary and a clear continuation pattern.

  - Per-book ``qc_fixes.toml`` substitutions. Same TOML schema as
    ``ocr_fixes.toml`` but applied at the HTML layer where agent QC find
    strings match (rendered text with collapsed footnote markers). Safe
    replacements only: spans containing structural tags (``<p>``, ``<br>``,
    ``<em>``, etc.) are skipped to avoid corrupting markup.

Every transformation in this module is conservative: it must not break a
valid EPUB. The full test suite under ``tests/test_typography.py`` covers
the safety boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


# ---------- shared helpers ----------

_HTML_ENT: dict[str, str] = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&apos;": "'", "&#x27;": "'", "&nbsp;": " ",
}

_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")

_SUP_DIGITS = str.maketrans(
    "0123456789",
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079",
)


def html_to_plain(html: str) -> tuple[str, list[int]]:
    """Strip HTML tags. Return ``(plain_text, htmlpos_for_each_plain_char)``.

    The position map lets callers locate the HTML range that produced a
    given plain-text span: ``html[posmap[i] : posmap[j] + 1]`` covers the
    bytes that produced ``plain[i : j + 1]`` (modulo entity decoding).
    """
    out_chars: list[str] = []
    out_pos: list[int] = []
    i = 0
    in_tag = False
    while i < len(html):
        c = html[i]
        if c == "<":
            in_tag = True
            i += 1
            continue
        if c == ">":
            in_tag = False
            i += 1
            continue
        if in_tag:
            i += 1
            continue
        if c == "&":
            decoded: str | None = None
            ent_len = 0
            for ent, rep in _HTML_ENT.items():
                if html.startswith(ent, i):
                    decoded, ent_len = rep, len(ent)
                    break
            if decoded is None:
                m = re.match(r"&#x?[0-9a-fA-F]+;", html[i:])
                if m:
                    s = m.group()
                    try:
                        ch = chr(int(s[3:-1], 16)) if s.startswith("&#x") else chr(int(s[2:-1]))
                    except ValueError:
                        ch = "?"
                    decoded, ent_len = ch, len(s)
            if decoded is not None:
                for ch in decoded:
                    out_chars.append(ch)
                    out_pos.append(i)
                i += ent_len
                continue
        out_chars.append(c)
        out_pos.append(i)
        i += 1
    return "".join(out_chars), out_pos


def _operate_on_text_only(
    html: str, replacer: Callable[[str], tuple[str, int]],
) -> tuple[str, int]:
    """Run ``replacer(text)`` on each text-content segment of ``html``
    (between tags). Attribute values inside tags are never touched.
    """
    out: list[str] = []
    seg: list[str] = []
    total = 0
    i = 0
    in_tag = False

    def flush() -> None:
        nonlocal total
        if not seg:
            return
        text = "".join(seg)
        new_text, n = replacer(text)
        out.append(new_text)
        total += n
        seg.clear()

    while i < len(html):
        c = html[i]
        if in_tag:
            out.append(c)
            if c == ">":
                in_tag = False
        else:
            if c == "<":
                flush()
                out.append(c)
                in_tag = True
            else:
                seg.append(c)
        i += 1
    flush()
    return "".join(out), total


def build_vocab(text: str) -> set[str]:
    """Build a vocabulary of lowercase tokens from ``text``. Used as the
    membership oracle for hyphen-stitching decisions.
    """
    return {w.lower() for w in _WORD_RE.findall(text)}


# ---------- transform 1: hyphen stitch ----------

def stitch_hyphens(html: str, vocab: set[str]) -> tuple[str, list[str]]:
    """Join ``word- nextpart`` -> ``wordnextpart`` when ``wordnextpart`` is
    already a known token in ``vocab``. Returns ``(new_html, applied_fixes)``.

    Operates on text segments only; never inside tags. The vocab oracle
    prevents false positives like splitting compound words at hyphens.
    """
    fixes: list[str] = []

    def fix(text: str) -> tuple[str, int]:
        n_local = 0

        def repl(m: re.Match) -> str:
            nonlocal n_local
            a, b = m.group(1), m.group(2)
            joined = a + b
            if joined.lower() in vocab:
                n_local += 1
                fixes.append(f"{a}- {b} -> {joined}")
                return joined
            return m.group(0)

        new = re.sub(r"\b([A-Za-z]{3,})- ([a-z]{2,})\b", repl, text)
        return new, n_local

    new_html, _ = _operate_on_text_only(html, fix)
    return new_html, fixes


# ---------- transform 2: loose footnote digits ----------

def convert_loose_footnote_digits(html: str) -> tuple[str, int]:
    """Convert digit footnote markers attached to surrounding text into
    Unicode superscript characters.

    Patterns matched:
      - ``;\\d`` / ``,\\d`` followed by space + lowercase letter
        (mid-sentence continuation after a clause-final marker).
      - ``word\\d`` followed by space + lowercase letter
        (footnote on a noun like ``Smith1 has``).

    Conservative: requires the digit be followed by ``space + lowercase``
    so we don't catch real numerics like ``2026 was`` or ``vol 1 page``.
    """

    def fix(text: str) -> tuple[str, int]:
        n_local = 0

        def repl(m: re.Match) -> str:
            nonlocal n_local
            n_local += 1
            return m.group(1) + m.group(2).translate(_SUP_DIGITS)

        new = re.sub(r'([;,])(\d{1,2})(?=\s+[a-z])', repl, text)
        new = re.sub(r'([a-z])(\d{1,2})(?=\s+[a-z])', repl, new)
        return new, n_local

    return _operate_on_text_only(html, fix)


# ---------- transform 3: stray footnote-quote stripping ----------

_FOOTNOTE_QUOTE_VERBS = (
    "states|said|explains|narrates|narrated|wrote|declared|writes|"
    "asks|describes|explained|remarks|notes|argues|maintains|"
    "observes|comments|reports|relates|adds"
)


def strip_footnote_misread_quotes(html: str) -> tuple[str, int]:
    """Remove ``"`` characters that are misread superscript footnote
    digits, in two recognizable patterns:

      1. ``Name" verb`` -> ``Name verb`` where ``verb`` is a
         narration-style verb (``states``, ``said``, ``narrates``, etc.).
         Example: ``Smith" states`` -> ``Smith states``.

      2. ``word." Word`` (lowercase word + period + closing quote +
         space + Capital start) -> ``word. Word``, but only when the
         preceding 60 chars do NOT contain an unpaired ``"`` (i.e. the
         quote is not closing a real quoted phrase).

    Operates on text segments only.
    """

    def fix(text: str) -> tuple[str, int]:
        n_local = 0

        def r1(m: re.Match) -> str:
            nonlocal n_local
            n_local += 1
            return m.group(1) + " " + m.group(2)

        text = re.sub(
            rf'\b([A-Z][a-zA-Z]+)" ({_FOOTNOTE_QUOTE_VERBS})\b',
            r1, text,
        )

        def safe_r2(m: re.Match) -> str:
            nonlocal n_local
            start = m.start()
            preceding = text[max(0, start - 60):start]
            if preceding.count('"') % 2 == 1:
                return m.group(0)
            n_local += 1
            return m.group(1) + ". "

        text = re.sub(r'([a-z]+)\." (?=[A-Z][a-z])', safe_r2, text)
        return text, n_local

    return _operate_on_text_only(html, fix)


# ---------- transform 4: qc_fixes.toml HTML-layer substitutions ----------

_FN_NOTEREF_A_RE = re.compile(
    r'<a\b[^>]*epub:type="noteref"[^>]*>.*?</a>', re.S,
)
_FN_SUP_RE = re.compile(r'<sup\b[^>]*>.*?</sup>', re.S)


def apply_qc_fix(html: str, find: str, replace: str) -> tuple[str, int]:
    """Replace plain-text occurrences of ``find`` with ``replace`` in
    ``html``, preserving footnote anchors (``<a epub:type="noteref">``
    and bare ``<sup>...</sup>`` blocks) inside the matched span.

    Safety boundaries — a match is SKIPPED when the matched HTML span
    contains any inline tag other than the preserved footnote refs.
    This is what stops a substitution from clobbering ``<em>italic</em>``
    runs, ``<a>`` links, or any structural tag.

    Returns ``(new_html, count_of_applied_replacements)``.
    """
    plain, _ = html_to_plain(html)
    if find not in plain:
        return html, 0
    new_html = html
    count = 0
    safety = 0
    while True:
        safety += 1
        if safety > 50:
            break
        plain, posmap = html_to_plain(new_html)
        idx = plain.find(find)
        if idx < 0:
            break
        start_html = posmap[idx]
        end_plain = idx + len(find) - 1
        if end_plain >= len(posmap):
            break
        end_html_char = posmap[end_plain]
        end_html = end_html_char + 1
        if new_html[end_html_char] == "&":
            m = re.match(r"&[#a-zA-Z0-9]+;", new_html[end_html_char:])
            if m:
                end_html = end_html_char + len(m.group())
        span = new_html[start_html:end_html]

        # Pure-text match — trivial replace.
        if "<" not in span and ">" not in span:
            new_html = new_html[:start_html] + replace + new_html[end_html:]
            count += 1
            continue

        # Tag-bearing match — only acceptable if every tag in the span
        # is a preserved footnote ref (noteref <a> or bare <sup>).
        noterefs = _FN_NOTEREF_A_RE.findall(span)
        residue = _FN_NOTEREF_A_RE.sub("", span)
        sup_blocks = _FN_SUP_RE.findall(residue)
        residue = _FN_SUP_RE.sub("", residue)
        if "<" in residue or ">" in residue:
            # Other tags (em / strong / span / a-not-noteref) present.
            # Skip to avoid corrupting markup.
            break
        preserved = "".join(noterefs) + "".join(sup_blocks)
        new_html = new_html[:start_html] + replace + preserved + new_html[end_html:]
        count += 1
    return new_html, count


def load_qc_fixes(path: Path) -> dict[str, str]:
    """Load a ``qc_fixes.toml`` file. Expected schema::

        [phrase]
        "find text" = "replace text"
        "another find" = "another replace"

    Identical and empty entries are dropped. Returns an empty dict if the
    file does not exist or is malformed.
    """
    if not path.exists():
        return {}
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except (tomllib.TOMLDecodeError, OSError):
        return {}
    phrases = cfg.get("phrase", {})
    if not isinstance(phrases, dict):
        return {}
    out: dict[str, str] = {}
    for k, v in phrases.items():
        if not isinstance(k, str) or not isinstance(v, str):
            continue
        if not k or k == v:
            continue
        out[k] = v
    return out


# ---------- driver ----------

@dataclass
class TypographyReport:
    """Aggregated counts and details of typography fixes applied."""

    qc_fix_count: int = 0
    qc_fix_unique: int = 0
    hyphen_stitches: int = 0
    hyphen_examples: list[str] = field(default_factory=list)
    footnote_digits: int = 0
    quote_strips: int = 0

    def summary_line(self) -> str:
        parts = []
        if self.qc_fix_count:
            parts.append(f"{self.qc_fix_count} qc fix(es)")
        if self.hyphen_stitches:
            parts.append(f"{self.hyphen_stitches} hyphen stitch(es)")
        if self.footnote_digits:
            parts.append(f"{self.footnote_digits} footnote-digit normalization(s)")
        if self.quote_strips:
            parts.append(f"{self.quote_strips} stray-quote strip(s)")
        if not parts:
            return "no typography fixes applied"
        return "typography: " + ", ".join(parts)


def apply_typography_fixes(
    rendered: list[tuple[str, str]],
    *,
    vocab: set[str],
    qc_fixes: dict[str, str] | None = None,
) -> tuple[list[tuple[str, str]], TypographyReport]:
    """Run all post-render typography transforms on ``rendered``.

    Parameters
    ----------
    rendered :
        ``[(slug, xhtml), ...]`` -- output of ``render_chapter`` per chapter.
    vocab :
        Lowercase token set built from the markdown corpus. Used by
        ``stitch_hyphens`` to decide whether a candidate join is a real word.
    qc_fixes :
        Optional ``{find: replace}`` mapping (typically loaded from
        ``qc_fixes.toml``) applied to each chapter's XHTML.

    Returns ``(new_rendered, report)``.
    """
    qc_fixes = qc_fixes or {}
    rep = TypographyReport()
    qc_unique_matched: set[str] = set()

    out: list[tuple[str, str]] = []
    for slug, html in rendered:
        new_html = html

        for find, replace in qc_fixes.items():
            new_html, n = apply_qc_fix(new_html, find, replace)
            if n:
                rep.qc_fix_count += n
                qc_unique_matched.add(find)

        new_html, hfixes = stitch_hyphens(new_html, vocab)
        rep.hyphen_stitches += len(hfixes)
        rep.hyphen_examples.extend(hfixes[:5])

        new_html, n_quotes = strip_footnote_misread_quotes(new_html)
        rep.quote_strips += n_quotes

        new_html, n_digits = convert_loose_footnote_digits(new_html)
        rep.footnote_digits += n_digits

        out.append((slug, new_html))

    rep.qc_fix_unique = len(qc_unique_matched)
    return out, rep
