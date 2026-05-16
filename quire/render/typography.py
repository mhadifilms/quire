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

    Patterns matched (in order, conservative each time):

      1. ``;\\d`` / ``,\\d`` followed by space + lowercase letter
         — ``rare;3 rather`` -> ``rare;\u00b3 rather``.
         Mid-sentence continuation after a clause-final marker.

      2. ``word\\d`` followed by space + lowercase letter
         — ``Smith1 has`` -> ``Smith\u00b9 has``.
         Footnote glued to a noun before the next clause.

      3. ``word.\\s+\\d\\s+`` followed by a Capital-then-lowercase
         word — ``obligation. 1 This verse`` ->
         ``obligation.\u00b9 This verse``. The OCR engine consistently
         turns a small superscript footnote marker at a sentence
         boundary into a free-standing digit on the next baseline,
         producing ``... sentence end. 1 Next sentence``. This rule
         pulls that digit back onto the previous word as a superscript.

      Conservative across all three:
      - Digit must be 1-2 chars (footnote numbers don't exceed 99).
      - The character AFTER the digit determines safety: a lowercase
        letter (rules 1+2) or a capitalized word starting a sentence
        (rule 3). Real numerics like ``2026 was`` or ``vol 1 of``
        don't match because they lack the lower/Cap-lowercase shape.
    """

    def fix(text: str) -> tuple[str, int]:
        n_local = 0

        def repl(m: re.Match) -> str:
            nonlocal n_local
            n_local += 1
            return m.group(1) + m.group(2).translate(_SUP_DIGITS)

        # 1. ;1 or ,1 mid-sentence
        new = re.sub(r'([;,])(\d{1,2})(?=\s+[a-z])', repl, text)
        # 2. word1 has -> word¹ has
        new = re.sub(r'([a-z])(\d{1,2})(?=\s+[a-z])', repl, new)
        # 3. obligation. 1 This -> obligation.¹ This
        #    Detached digit at sentence boundary, lookbehind word ends
        #    in a clause/sentence punctuation and a single space.
        def boundary_repl(m: re.Match) -> str:
            nonlocal n_local
            n_local += 1
            return m.group(1) + m.group(2).translate(_SUP_DIGITS) + " "

        new = re.sub(
            r'([a-z][.!?\)\]])\s+(\d{1,2})\s+(?=[A-Z][a-z])',
            boundary_repl,
            new,
        )
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

# Tag-pair openings we treat as preserved footnote refs. Order matters:
# longer/more specific first so we don't shadow.
_FN_OPENINGS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r'<a\b[^>]*epub:type="noteref"[^>]*>'), "</a>"),
    (re.compile(r"<sup\b[^>]*>"), "</sup>"),
)


def _extend_left_into_footnote_tag(html: str, start: int) -> int:
    """If ``start`` lies inside a ``<sup>`` or ``<a noteref>`` opening
    tag's content, return the position of the tag's ``<`` so the matched
    span includes the whole tag pair. Iterates outward so a nested
    ``<a noteref><sup>1</sup></a>`` structure is captured in one call.

    Returns ``start`` unchanged when the position is not inside a
    footnote-ref tag.
    """
    cursor = start
    while True:
        new = _extend_left_one_level(html, cursor)
        if new >= cursor:
            return cursor
        cursor = new


def _extend_left_one_level(html: str, start: int) -> int:
    window_start = max(0, start - 200)
    chunk = html[window_start:start]
    last_open = chunk.rfind("<")
    if last_open < 0:
        return start
    tag_open_pos = window_start + last_open
    close_gt = html.find(">", tag_open_pos, start)
    if close_gt < 0:
        return start
    opening = html[tag_open_pos:close_gt + 1]
    for pattern, _ in _FN_OPENINGS:
        if pattern.fullmatch(opening):
            return tag_open_pos
    return start


def _extend_right_out_of_footnote_tag(html: str, end: int) -> int:
    """If ``end`` lies inside the content of a ``<sup>`` or ``<a noteref>``
    tag (i.e. before its closing tag), return the position just after the
    closing tag so the full pair is included in the matched span. Iterates
    outward so a nested footnote structure is captured.

    Returns ``end`` unchanged when not inside such a tag.
    """
    cursor = end
    while True:
        new = _extend_right_one_level(html, cursor)
        if new <= cursor:
            return cursor
        cursor = new


def _extend_right_one_level(html: str, end: int) -> int:
    for pattern, close_str in _FN_OPENINGS:
        last_open_match: re.Match[str] | None = None
        for m in pattern.finditer(html, 0, end):
            last_open_match = m
        if not last_open_match:
            continue
        open_end = last_open_match.end()
        if open_end >= end:
            continue
        close_idx = html.find(close_str, open_end, end)
        if close_idx < 0:
            close_after = html.find(close_str, end)
            if close_after >= 0:
                return close_after + len(close_str)
    return end


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

        # Extend boundaries outward if start/end falls INSIDE a footnote
        # ref tag (``<sup>...</sup>`` or ``<a epub:type="noteref">...</a>``).
        # Otherwise the span would split the tag, leaving an unbalanced
        # ``</sup>`` or ``</a>`` in the residue and tripping the tag-
        # safety guard. By widening to the whole tag pair we let the
        # preservation logic catch the noteref cleanly. This is critical
        # for find strings that begin with the digit collapsed out of a
        # leading ``<sup>1</sup>`` (e.g. Quran verse references in the
        # index where the first digit was OCR'd as a superscript). When
        # we extend INTO a tag, the digit it contained is part of the
        # text the user wants to overwrite -- so that tag is NOT
        # preserved (treated as consumed by the replacement).
        new_start = _extend_left_into_footnote_tag(new_html, start_html)
        extended_left = new_start < start_html
        start_html = new_start
        new_end = _extend_right_out_of_footnote_tag(new_html, end_html)
        extended_right = new_end > end_html
        end_html = new_end
        span = new_html[start_html:end_html]

        # Pure-text match — trivial replace.
        if "<" not in span and ">" not in span:
            candidate = new_html[:start_html] + replace + new_html[end_html:]
            if candidate == new_html:
                # No-op iteration (find == replace, no tags consumed):
                # break to avoid spinning until the safety cap.
                break
            new_html = candidate
            count += 1
            continue

        # Tag-bearing match — only acceptable if every tag in the span
        # is a preserved footnote ref (noteref <a> or bare <sup>).
        noterefs = list(_FN_NOTEREF_A_RE.findall(span))
        residue = _FN_NOTEREF_A_RE.sub("", span)
        sup_blocks = list(_FN_SUP_RE.findall(residue))
        residue = _FN_SUP_RE.sub("", residue)
        if "<" in residue or ">" in residue:
            # Other tags (em / strong / span / a-not-noteref) present.
            # Skip to avoid corrupting markup.
            break
        # Drop any preserved tag that was consumed by boundary extension.
        # The extension added the tag to the span specifically because
        # the user's find string overlapped its content -- preserving it
        # would re-insert the very bug we're trying to fix.
        if extended_left:
            if _FN_NOTEREF_A_RE.match(span):
                noterefs = noterefs[1:]
            elif _FN_SUP_RE.match(span):
                sup_blocks = sup_blocks[1:]
        if extended_right:
            if span.endswith("</a>") and noterefs:
                noterefs = noterefs[:-1]
            elif span.endswith("</sup>") and sup_blocks:
                sup_blocks = sup_blocks[:-1]
        preserved = "".join(noterefs) + "".join(sup_blocks)
        new_html = new_html[:start_html] + replace + preserved + new_html[end_html:]
        count += 1
    return new_html, count


def load_qc_fixes(path: Path) -> dict[str, str]:
    """Load a ``qc_fixes.toml`` file. Expected schema::

        [phrase]
        "find text" = "replace text"
        "another find" = "another replace"

    Empty keys are dropped. Identical ``find == replace`` entries are
    KEPT because they're meaningful when the find string starts inside
    a ``<sup>`` / ``<a noteref>`` opening tag and the user wants the tag
    pair stripped without changing the plain text (see
    ``_extend_left_into_footnote_tag`` in ``apply_qc_fix``). Returns an
    empty dict if the file does not exist or is malformed.
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
        if not k:
            continue
        out[k] = v
    return out


# ---------- driver ----------

_QC_FIXPOINT_MAX_PASSES = 4


@dataclass
class TypographyReport:
    """Aggregated counts and details of typography fixes applied."""

    qc_fix_count: int = 0
    qc_fix_unique: int = 0
    qc_no_op_entries: list[str] = field(default_factory=list)
    qc_tag_skipped_entries: list[str] = field(default_factory=list)
    hyphen_stitches: int = 0
    hyphen_examples: list[str] = field(default_factory=list)
    footnote_digits: int = 0
    quote_strips: int = 0
    run_collapses: int = 0
    run_collapse_examples: list[str] = field(default_factory=list)

    def summary_line(self) -> str:
        parts = []
        if self.qc_fix_count:
            parts.append(f"{self.qc_fix_count} qc fix(es)")
        if self.qc_tag_skipped_entries:
            parts.append(
                f"{len(self.qc_tag_skipped_entries)} qc fix(es) skipped (inline-tag span)"
            )
        if self.hyphen_stitches:
            parts.append(f"{self.hyphen_stitches} hyphen stitch(es)")
        if self.footnote_digits:
            parts.append(f"{self.footnote_digits} footnote-digit normalization(s)")
        if self.quote_strips:
            parts.append(f"{self.quote_strips} stray-quote strip(s)")
        if self.run_collapses:
            parts.append(f"{self.run_collapses} repetition-run collapse(s)")
        if not parts:
            return "no typography fixes applied"
        return "typography: " + ", ".join(parts)


# Thresholds chosen so that legitimate prose never trips the transform:
# - 10+ identical whole-word tokens in a row never appears in real
#   English / Arabic prose; the bar is set low because the OCR
#   pathology that creates these runs always overshoots into the
#   hundreds.
# - 8+ identical single non-word characters covers `((((((`, `''''''`,
#   `------`, `....` runs of length 8+ — anything shorter is
#   plausibly a normal ellipsis variant or a stylistic dash.
_RUN_WORD_MIN_REPS = 10
_RUN_CHAR_MIN_REPS = 8

# Runs of dots / commas collapse to a Unicode ellipsis (citation gap
# semantics preserved). Other character classes collapse to a single
# instance.
_RUN_TO_ELLIPSIS = set(".,")
_RUN_ALWAYS_DROP = set("'\"")

# Word-run regex: a non-whitespace, non-tag token followed by 10+ space-
# separated repetitions of itself. Anchored with `\B?` to also catch
# leading-space cases like ` term term term…`. The token must start with
# a word char so we don't capture punctuation streaks here — those are
# handled by the single-char regex below.
_RUN_WORD_RE = re.compile(
    rf"(?P<lead> ?)(?P<tok>[A-Za-z\u00c0-\u024f\u0600-\u06ff'\u2018\u2019\-]{{1,30}})"
    rf"(?:[ \t]+(?P=tok)){{{_RUN_WORD_MIN_REPS - 1},}}"
)

# Single-character run regex: any single non-word char (excluding space
# and HTML metacharacters `<` / `>` / `&`) repeated 8+ times.
_RUN_CHAR_RE = re.compile(
    rf"(?P<ch>[^A-Za-z0-9\s<>&])(?P=ch){{{_RUN_CHAR_MIN_REPS - 1},}}"
)


def collapse_repetition_runs(html: str) -> tuple[str, list[str]]:
    """Collapse pathological repetition runs left over from OCR / structuring.

    Some scanned PDFs surface a single token multiplied dozens or
    hundreds of times: ``term term term term …`` (200×),
    ``( ( ( ( ( …`` (200×), ``...........`` (200+ dots), etc. These
    runs are never the author's intent — they are a structural
    pathology — and they slip past the small find/replace layers
    because their exact length varies between rebuilds.

    Rules:

    * A 1- to 30-char word-shaped token (Latin / Arabic / hyphenated)
      separated by whitespace and repeated ``_RUN_WORD_MIN_REPS`` (10)+
      times collapses to a single instance.
    * A single non-word character (excluding whitespace and HTML
      metacharacters ``<``, ``>``, ``&``) repeated ``_RUN_CHAR_MIN_REPS``
      (8)+ times collapses to:

      - ``…`` (Unicode ellipsis) for ``.`` or ``,`` runs (citation
        gap semantics);
      - empty string for ``'``, ``"`` runs (stray apostrophe / quote
        garbage);
      - a single occurrence otherwise (``(``, ``-``, ``)`` etc.).

    The transform is intentionally conservative: it only fires on
    runs that are structurally impossible in well-formed prose, so
    legitimate em-dash patterns, ellipses, and stylistic punctuation
    are never touched.

    Returns ``(new_html, examples)`` where ``examples`` is a short
    list of human-readable descriptions of each collapse for audit
    logging (e.g. ``"200x ' term' -> ' term'"``).
    """
    examples: list[str] = []

    def _word_repl(m: re.Match[str]) -> str:
        tok = m.group("tok")
        lead = m.group("lead")
        full = m.group(0)
        reps = full.count(tok)
        examples.append(f"{reps}x {tok!r} -> single")
        return f"{lead}{tok}"

    def _char_repl(m: re.Match[str]) -> str:
        ch = m.group("ch")
        reps = len(m.group(0))
        if ch in _RUN_TO_ELLIPSIS:
            replacement = "\u2026"
        elif ch in _RUN_ALWAYS_DROP:
            replacement = ""
        else:
            replacement = ch
        examples.append(f"{reps}x {ch!r} -> {replacement!r}")
        return replacement

    new_html = _RUN_WORD_RE.sub(_word_repl, html)
    new_html = _RUN_CHAR_RE.sub(_char_repl, new_html)
    return new_html, examples


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

    qc_fixes are applied in a fixed-point loop (up to
    ``_QC_FIXPOINT_MAX_PASSES`` passes per chapter) so that an entry
    whose ``find`` only matches after some other entry has rewritten the
    text — e.g. a long contextual fix that depends on a prior
    short-word OCR correction — still applies regardless of the
    declaration order in ``qc_fixes.toml``.

    The returned report distinguishes three terminal states per
    ``qc_fixes`` entry:

    * applied at least once (counted in ``qc_fix_count`` / ``qc_fix_unique``);
    * never matched anywhere — likely stale / mistyped (``qc_no_op_entries``);
    * the ``find`` text exists in the final output but every match was
      skipped by the ``apply_qc_fix`` inline-tag safety guard
      (``qc_tag_skipped_entries``). This is almost always a bug in the
      ``find`` string: rewrite it to start/end outside of ``<em>``,
      ``<span>``, or other non-noteref markup.

    Returns ``(new_rendered, report)``.
    """
    qc_fixes = qc_fixes or {}
    rep = TypographyReport()
    qc_unique_matched: set[str] = set()

    out: list[tuple[str, str]] = []
    for slug, html in rendered:
        new_html = html

        for _ in range(_QC_FIXPOINT_MAX_PASSES):
            applied_this_pass = False
            for find, replace in qc_fixes.items():
                new_html, n = apply_qc_fix(new_html, find, replace)
                if n:
                    rep.qc_fix_count += n
                    qc_unique_matched.add(find)
                    applied_this_pass = True
            if not applied_this_pass:
                break

        new_html, run_examples = collapse_repetition_runs(new_html)
        if run_examples:
            rep.run_collapses += len(run_examples)
            rep.run_collapse_examples.extend(run_examples[:5])

        new_html, hfixes = stitch_hyphens(new_html, vocab)
        rep.hyphen_stitches += len(hfixes)
        rep.hyphen_examples.extend(hfixes[:5])

        new_html, n_quotes = strip_footnote_misread_quotes(new_html)
        rep.quote_strips += n_quotes

        new_html, n_digits = convert_loose_footnote_digits(new_html)
        rep.footnote_digits += n_digits

        out.append((slug, new_html))

    rep.qc_fix_unique = len(qc_unique_matched)

    # Surface qc_fixes entries that never applied. We split them into two
    # buckets by probing the final plain text across all chapters: a find
    # string still present in the rendered output that never matched
    # means the inline-tag safety guard skipped every candidate; a find
    # string completely absent is simply stale or mistyped.
    if qc_fixes:
        unmatched = [f for f in qc_fixes if f not in qc_unique_matched]
        if unmatched:
            final_plain_parts: list[str] = []
            for _slug, h in out:
                p, _ = html_to_plain(h)
                final_plain_parts.append(p)
            final_plain = "\n".join(final_plain_parts)
            tag_skipped = sorted(f for f in unmatched if f and f in final_plain)
            no_op = sorted(f for f in unmatched if f and f not in final_plain)
            rep.qc_tag_skipped_entries = tag_skipped
            rep.qc_no_op_entries = no_op

    return out, rep
