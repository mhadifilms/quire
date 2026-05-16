"""Tests for the post-render typography fixes.

These cover the contract documented in ``quire/render/typography.py`` —
each transform must be a strict no-op outside the patterns it targets,
must never break HTML structure, and must operate only on text content
(never inside tag attributes).
"""

from __future__ import annotations

import pathlib

import pytest

from quire.render.typography import (
    apply_qc_fix,
    apply_typography_fixes,
    build_vocab,
    collapse_repetition_runs,
    convert_loose_footnote_digits,
    html_to_plain,
    load_qc_fixes,
    stitch_hyphens,
    strip_footnote_misread_quotes,
)

# ---------- html_to_plain ----------

class TestHtmlToPlain:
    def test_strips_tags_and_maps_positions(self) -> None:
        html = "<p>Hello <em>world</em>!</p>"
        plain, posmap = html_to_plain(html)
        assert plain == "Hello world!"
        assert len(posmap) == len(plain)
        # 'H' is at position html.index('H')
        assert posmap[0] == html.index("H")
        # '!' is at the position just before </p>
        assert html[posmap[-1]] == "!"

    def test_decodes_entities(self) -> None:
        html = "<p>a &amp; b &lt;c&gt; &#x27;d&#x27;</p>"
        plain, _ = html_to_plain(html)
        assert plain == "a & b <c> 'd'"

    def test_attribute_text_not_included(self) -> None:
        html = '<a href="https://example.com/foo">click</a>'
        plain, _ = html_to_plain(html)
        assert plain == "click"


# ---------- stitch_hyphens ----------

class TestStitchHyphens:
    def test_joins_when_in_vocab(self) -> None:
        vocab = {"manuscript"}
        html = "<p>he copied the manu- script carefully</p>"
        new, fixes = stitch_hyphens(html, vocab)
        assert "manuscript" in new
        assert "manu- script" not in new
        assert fixes == ["manu- script -> manuscript"]

    def test_skips_when_not_in_vocab(self) -> None:
        vocab = {"unrelated"}
        html = "<p>some- nonsense word</p>"
        new, fixes = stitch_hyphens(html, vocab)
        assert "some- nonsense" in new
        assert fixes == []

    def test_does_not_touch_attribute_values(self) -> None:
        # The href contains "test- ing" (which would join to "testing"
        # if we operated on attribute text). It must NOT be touched.
        vocab = {"testing"}
        html = '<a href="https://x.example/test- ing">label</a>'
        new, _ = stitch_hyphens(html, vocab)
        assert 'href="https://x.example/test- ing"' in new


# ---------- convert_loose_footnote_digits ----------

class TestConvertLooseFootnoteDigits:
    def test_converts_semicolon_digit_before_lowercase(self) -> None:
        html = "<p>the term);3 rather</p>"
        new, n = convert_loose_footnote_digits(html)
        assert "the term);\u00b3 rather" in new
        assert n == 1

    def test_converts_word_digit_before_lowercase(self) -> None:
        html = "<p>The author Smith1 has stated</p>"
        new, n = convert_loose_footnote_digits(html)
        assert "Smith\u00b9 has" in new
        assert n == 1

    def test_does_not_touch_real_numbers(self) -> None:
        # "2026 was" - digit followed by lowercase, but starts a new
        # token (no leading letter / punct), so it should NOT match.
        html = "<p>In 2026 we built it</p>"
        new, n = convert_loose_footnote_digits(html)
        assert new == html
        assert n == 0

    def test_does_not_touch_attribute_digits(self) -> None:
        html = '<a href="x;3.html">click</a>'
        new, _ = convert_loose_footnote_digits(html)
        # The ;3 inside href should be preserved verbatim
        assert 'href="x;3.html"' in new

    def test_pulls_detached_digit_at_sentence_boundary(self) -> None:
        # A small superscript footnote at a sentence boundary is often
        # OCR'd as a detached digit on the next baseline: the rule
        # should pull it back onto the prior word.
        html = "<p>rites of Hajj. 1 This verse clearly</p>"
        new, n = convert_loose_footnote_digits(html)
        assert "rites of Hajj.\u00b9 This verse" in new
        assert n == 1

    def test_pulls_detached_digit_after_close_paren(self) -> None:
        html = "<p>the latter] 2 The next sentence begins.</p>"
        new, n = convert_loose_footnote_digits(html)
        assert "the latter]\u00b2 The next sentence" in new
        assert n == 1

    def test_does_not_touch_detached_digit_before_lowercase(self) -> None:
        # ``...obligation. 1 this`` -- lowercase ``this`` afterwards is
        # ambiguous (e.g. a literal page number), so leave it alone.
        html = "<p>obligation. 1 this verse clearly</p>"
        new, n = convert_loose_footnote_digits(html)
        assert new == html
        assert n == 0

    def test_does_not_touch_real_page_reference(self) -> None:
        # ``vol 1 of`` / ``page 1 of`` -- digit between Capitalized
        # tokens but not at a sentence boundary (no terminal punctuation).
        html = "<p>see vol 1 of the series</p>"
        new, n = convert_loose_footnote_digits(html)
        assert new == html
        assert n == 0


# ---------- strip_footnote_misread_quotes ----------

class TestStripFootnoteMisreadQuotes:
    def test_strips_name_quote_verb_pattern(self) -> None:
        html = '<p>Smith" states that ...</p>'
        new, n = strip_footnote_misread_quotes(html)
        assert "Smith states that" in new
        assert n == 1

    def test_strips_word_period_quote_at_sentence_boundary(self) -> None:
        # essence." starts a new sentence with "In" -- and there's no
        # unpaired opening " in the preceding text.
        html = '<p>due to the latter essence." In other words, He</p>'
        new, n = strip_footnote_misread_quotes(html)
        assert "essence. In other words" in new
        assert n == 1

    def test_keeps_real_closing_quotes(self) -> None:
        # The "..." spans a real quoted phrase. The closing " is paired.
        html = '<p>He said, "Yes." Then he left.</p>'
        new, n = strip_footnote_misread_quotes(html)
        assert new == html  # unchanged
        assert n == 0

    def test_does_not_strip_random_quotes(self) -> None:
        html = '<p>Word" not a verb</p>'
        new, _ = strip_footnote_misread_quotes(html)
        assert new == html


# ---------- apply_qc_fix (HTML-layer substitution) ----------

class TestApplyQcFix:
    def test_pure_text_replacement(self) -> None:
        html = "<p>The garbled word xyz ('original phrase')</p>"
        new, n = apply_qc_fix(
            html,
            "The garbled word xyz ('original phrase')",
            "The correct word abc ('original phrase')",
        )
        assert "correct word abc" in new
        assert n == 1

    def test_preserves_footnote_anchor_in_match(self) -> None:
        # find=word1 should match the rendered text (where <sup>1</sup>
        # collapses to '1'), and replacement preserves the <a> noteref.
        html = (
            '<p>The garbled word xyz'
            '<a epub:type="noteref" href="#fn-1"><sup>1</sup></a>'
            " ('original phrase')</p>"
        )
        new, n = apply_qc_fix(
            html,
            "The garbled word xyz1 ('original phrase')",
            "The correct word abc ('original phrase')",
        )
        assert n == 1
        assert "correct word abc" in new
        # The footnote anchor is preserved verbatim
        assert 'epub:type="noteref"' in new
        assert 'href="#fn-1"' in new

    def test_skips_span_with_em_tags(self) -> None:
        # If the match span crosses an <em> boundary, skip to avoid
        # corrupting markup. The original markup stays intact.
        html = "<p>not a collective rule (<em>some</em> thing)</p>"
        new, n = apply_qc_fix(
            html,
            "not a collective rule (some thing)",
            "REPLACED",
        )
        assert n == 0
        assert new == html

    def test_no_match_no_op(self) -> None:
        html = "<p>nothing relevant here</p>"
        new, n = apply_qc_fix(html, "missing string", "anything")
        assert n == 0
        assert new == html

    def test_identical_find_and_replace_in_pure_text_is_noop(self) -> None:
        # When find == replace and the matched span is pure text (no
        # consumed tags), the loop breaks early to avoid spinning.
        html = "<p>hello</p>"
        new, n = apply_qc_fix(html, "hello", "hello")
        assert n == 0
        assert new == html

    def test_identical_find_and_replace_strips_consumed_sup(self) -> None:
        # When the find string overlaps a leading ``<sup>`` and the
        # plain-text replace equals the find, the HTML transformation
        # still happens: the wrapper is consumed by boundary extension
        # and emitted as plain text. Subsequent loop iterations are
        # plain-text no-ops, so we run once and stop.
        html = '<p class="body"><sup>1</sup>1:75 Abraham was forbearing</p>'
        new, n = apply_qc_fix(
            html,
            "11:75 Abraham was forbearing",
            "11:75 Abraham was forbearing",
        )
        assert n == 1
        assert "<sup>" not in new
        assert "11:75 Abraham was forbearing" in new

    def test_nested_noteref_sup_is_extended_outward(self) -> None:
        # Real-world OCR pattern: the engine wrapped the wrong leading
        # digit in a complete <a noteref><sup>1</sup></a> pair. Plain
        # text reads "1nd edition, Beirur:..." -- the find begins inside
        # the sup, but the noteref encloses it. The implementation must
        # extend left iteratively all the way to <a> so the whole
        # spurious noteref is consumed.
        html = (
            '<p><a epub:type="noteref" href="#fn-1"><sup>1</sup></a>'
            "nd edition, Beirur: A'lami Institute for Publications, 1983.</p>"
        )
        new, n = apply_qc_fix(
            html,
            "1nd edition, Beirur: A'lami Institute",
            "2nd edition, Beirut: A'lami Institute",
        )
        assert n == 1
        assert "<a" not in new
        assert "<sup>" not in new
        assert "2nd edition, Beirut: A'lami Institute" in new

    def test_find_starting_inside_sup_extends_to_full_tag(self) -> None:
        # Quran verse references in the index get an OCR'd first digit
        # wrapped as ``<sup>1</sup>``. The plain text reads ``10:12 ...``
        # and the agent's find begins with ``10``. The span starts INSIDE
        # ``<sup>`` -- the implementation must extend left to capture the
        # entire ``<sup>...</sup>`` pair so the replacement drops it.
        html = '<p class="body"><sup>1</sup>0:12 Indeed I am your Lord!</p>'
        new, n = apply_qc_fix(
            html,
            "10:12 Indeed I am your Lord!",
            "20:12 Indeed I am your Lord!",
        )
        assert n == 1
        # The <sup> wrapper is gone, replaced by the new "20" verse number.
        assert "<sup>" not in new
        assert "20:12 Indeed I am your Lord" in new

    def test_find_ending_inside_sup_extends_past_close(self) -> None:
        # Symmetric case: find ends mid-sup. The implementation must
        # extend right past ``</sup>`` so the span covers the whole pair.
        html = '<p>see also Quran 2:1<sup>1</sup>5 for context</p>'
        new, n = apply_qc_fix(
            html,
            "see also Quran 2:11",
            "see also Quran 2:21",
        )
        # The "1" inside <sup> is the last char of the find. The span
        # extends to include </sup>, and the wrapper is dropped.
        assert n == 1
        assert "<sup>" not in new
        assert "see also Quran 2:21" in new


# ---------- load_qc_fixes ----------

class TestLoadQcFixes:
    def test_loads_phrase_table(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "qc_fixes.toml"
        p.write_text(
            '[phrase]\n"find me" = "replace me"\n"another" = "thing"\n',
            encoding="utf-8",
        )
        out = load_qc_fixes(p)
        assert out == {"find me": "replace me", "another": "thing"}

    def test_missing_file_returns_empty(self, tmp_path: pathlib.Path) -> None:
        assert load_qc_fixes(tmp_path / "nonexistent.toml") == {}

    def test_drops_empty_keys(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "qc_fixes.toml"
        p.write_text(
            '[phrase]\n"" = "x"\n"keep" = "keeper"\n',
            encoding="utf-8",
        )
        assert load_qc_fixes(p) == {"keep": "keeper"}

    def test_keeps_identical_find_and_replace(
        self, tmp_path: pathlib.Path,
    ) -> None:
        # When the find string starts inside a footnote-ref tag, the
        # plain-text find and replace can be identical -- the HTML-level
        # transformation strips the wrapper tag. These entries must be
        # preserved by the loader.
        p = tmp_path / "qc_fixes.toml"
        p.write_text(
            '[phrase]\n'
            '"11:75 Abraham was indeed" = "11:75 Abraham was indeed"\n',
            encoding="utf-8",
        )
        out = load_qc_fixes(p)
        assert "11:75 Abraham was indeed" in out
        assert out["11:75 Abraham was indeed"] == "11:75 Abraham was indeed"


# ---------- end-to-end driver ----------

class TestApplyTypographyFixes:
    def test_runs_all_transforms_in_order(self) -> None:
        rendered = [
            (
                "ch-01",
                '<p>The author says, the manu- script. '
                'Smith" states that the obligation;3 rather.</p>',
            ),
        ]
        vocab = build_vocab("the manuscript of obligation states")
        new, rep = apply_typography_fixes(rendered, vocab=vocab)
        out = new[0][1]
        assert "manuscript" in out
        assert "Smith states" in out
        assert ";\u00b3 rather" in out
        assert rep.hyphen_stitches == 1
        assert rep.quote_strips == 1
        assert rep.footnote_digits == 1

    def test_qc_fixes_applied_first(self) -> None:
        rendered = [("ch-01", "<p>garbled xyz example</p>")]
        vocab = build_vocab("garbled example")
        new, rep = apply_typography_fixes(
            rendered,
            vocab=vocab,
            qc_fixes={"garbled xyz example": "corrected abc example"},
        )
        assert "corrected abc" in new[0][1]
        assert rep.qc_fix_count == 1
        assert rep.qc_fix_unique == 1

    def test_empty_rendered_returns_empty(self) -> None:
        new, rep = apply_typography_fixes([], vocab=set())
        assert new == []
        assert rep.summary_line() == "no typography fixes applied"

    def test_report_summary_line_lists_fixes(self) -> None:
        rendered = [("ch", "<p>manu- script and</p>")]
        vocab = {"manuscript", "and"}
        _, rep = apply_typography_fixes(rendered, vocab=vocab)
        assert "hyphen" in rep.summary_line()

    def test_qc_fixes_apply_fixed_point_independent_of_order(self) -> None:
        # Regression: two qc_fixes entries where the long contextual one
        # only matches AFTER the short OCR correction runs. Both orderings
        # must produce the same final text under the fixed-point loop.
        rendered_a = [("ch", "<p>paying attention to chem, [Arabic] is the owner.</p>")]
        rendered_b = [("ch", "<p>paying attention to chem, [Arabic] is the owner.</p>")]
        vocab = build_vocab("attention owner")

        # Order 1: long contextual entry declared FIRST (used to silently
        # no-op because the find string didn't match the pre-fix text).
        new_a, rep_a = apply_typography_fixes(
            rendered_a,
            vocab=vocab,
            qc_fixes={
                "attention to them, [Arabic] is the owner": "attention to them, since Allah is the owner",
                "attention to chem,": "attention to them,",
            },
        )
        # Order 2: short OCR correction declared FIRST.
        new_b, rep_b = apply_typography_fixes(
            rendered_b,
            vocab=vocab,
            qc_fixes={
                "attention to chem,": "attention to them,",
                "attention to them, [Arabic] is the owner": "attention to them, since Allah is the owner",
            },
        )
        for new, rep in ((new_a, rep_a), (new_b, rep_b)):
            assert "chem" not in new[0][1]
            assert "[Arabic]" not in new[0][1]
            assert "since Allah is the owner" in new[0][1]
            assert rep.qc_fix_unique == 2
            assert rep.qc_tag_skipped_entries == []
            assert rep.qc_no_op_entries == []

    def test_qc_fixes_surfaces_tag_skipped_entries(self) -> None:
        # A find string that crosses an <em> boundary cannot apply; the
        # report must surface it so the author can fix the entry rather
        # than have it silently rot in qc_fixes.toml.
        rendered = [("ch", '<p>name <em>X</em>, [Arabic] is the rule.</p>')]
        vocab = build_vocab("rule name")
        new, rep = apply_typography_fixes(
            rendered,
            vocab=vocab,
            qc_fixes={"X, [Arabic] is the rule.": "X, something is the rule."},
        )
        # Original markup intact, no replacement happened.
        assert new[0][1] == rendered[0][1]
        # ...but the entry is surfaced for the author.
        assert rep.qc_fix_count == 0
        assert rep.qc_tag_skipped_entries == ["X, [Arabic] is the rule."]
        assert rep.qc_no_op_entries == []
        assert "skipped" in rep.summary_line()

    def test_qc_fixes_surfaces_no_op_entries(self) -> None:
        # A find string nowhere in any chapter ends up in qc_no_op_entries
        # rather than qc_tag_skipped_entries.
        rendered = [("ch", "<p>Normal prose with nothing to fix.</p>")]
        vocab = build_vocab("normal prose")
        _, rep = apply_typography_fixes(
            rendered,
            vocab=vocab,
            qc_fixes={"missing find string": "never applied"},
        )
        assert rep.qc_no_op_entries == ["missing find string"]
        assert rep.qc_tag_skipped_entries == []


# ---------- collapse_repetition_runs ----------

class TestCollapseRepetitionRuns:
    """The OCR / structuring pipeline occasionally multiplies a single
    token hundreds of times. This is structurally impossible in
    well-formed prose, so the transform can collapse aggressively
    without risking false positives. Test both the firing patterns and
    the no-op contract for things that *look* like runs but aren't."""

    def test_word_run_collapses_to_single(self) -> None:
        html = "as shown by the" + (" term" * 200) + " content"
        out, examples = collapse_repetition_runs(html)
        assert " term term" not in out
        assert "as shown by the term content" in out
        assert examples
        assert "200x" in examples[0]

    def test_paren_run_collapses_to_one(self) -> None:
        html = "before " + ("(" * 200) + " after"
        out, examples = collapse_repetition_runs(html)
        assert "((" not in out
        assert "before ( after" in out
        assert examples

    def test_dot_run_becomes_ellipsis(self) -> None:
        html = "greatness" + ("." * 203) + " 58"
        out, examples = collapse_repetition_runs(html)
        assert "..." not in out
        assert "greatness\u2026 58" in out
        assert examples

    def test_apostrophe_run_dropped(self) -> None:
        html = "Spirit" + ("'" * 200) + "<a>fn</a>"
        out, examples = collapse_repetition_runs(html)
        assert "''" not in out
        assert "Spirit<a>fn</a>" in out
        assert examples

    def test_hyphen_run_collapses(self) -> None:
        html = "before " + ("-" * 50) + " after"
        out, _ = collapse_repetition_runs(html)
        assert "--" not in out
        assert "before - after" in out

    def test_quote_run_dropped(self) -> None:
        html = 'word' + ('"' * 30) + "next"
        out, _ = collapse_repetition_runs(html)
        assert '""' not in out
        assert "wordnext" in out

    def test_short_word_repeats_left_alone(self) -> None:
        # Five repeats — below threshold (10) — must be preserved.
        html = "term term term term term"
        out, examples = collapse_repetition_runs(html)
        assert out == html
        assert examples == []

    def test_short_dot_run_preserved(self) -> None:
        # Three dots (intentional ellipsis) — below threshold (8).
        html = "fragment... continues"
        out, _ = collapse_repetition_runs(html)
        assert out == html

    def test_em_dash_pair_preserved(self) -> None:
        # Two em-dashes — below 8-rep threshold.
        html = "phrase -- continues"
        out, _ = collapse_repetition_runs(html)
        assert out == html

    def test_html_metachars_not_collapsed(self) -> None:
        # `<<<<<<<<` would be an HTML mistake but we don't touch it;
        # the threshold regex excludes `<`, `>`, `&` to avoid breaking
        # well-formed markup that happens to contain repeated metas
        # inside CDATA / scripts (none in EPUB output, but be safe).
        html = "<p>a</p>"
        out, _ = collapse_repetition_runs(html)
        assert out == html

    def test_arabic_word_repeats_collapse(self) -> None:
        # Arabic transliteration / Quranic words can hit the same
        # pathology. The token charset includes Latin diacritics +
        # Arabic Unicode block.
        html = "phrase" + (" tamattu" * 15) + " end"
        out, examples = collapse_repetition_runs(html)
        assert " tamattu tamattu" not in out
        assert "phrase tamattu end" in out
        assert examples

    def test_does_not_break_inline_tags(self) -> None:
        # Repeated tokens around an em-tag should not introduce
        # HTML breakage even when one repetition starts in plain text.
        # The transform is text-only; the em-tag is preserved verbatim.
        html = "<em>tamattu</em>" + ("'" * 200) + "<sup>1</sup>"
        out, _ = collapse_repetition_runs(html)
        # The 200 apostrophes between em and sup are gone; the tag pair
        # is intact.
        assert "<em>tamattu</em><sup>1</sup>" in out


# ---------- regression: must not break valid HTML ----------

class TestSafetyContract:
    def test_complex_chapter_html_remains_well_formed(self) -> None:
        # A small but realistic mini-chapter exercising tags we must
        # preserve verbatim through all transforms.
        html = (
            '<html xmlns="x"><body>'
            '<p class="body">Smith, <em>Book</em>'
            '<a epub:type="noteref" href="#fn-1"><sup>1</sup></a> '
            '<em>Volume One</em>, p. 2.</p>'
            '<p class="body">copy the manu- script.</p>'
            '<p class="body">Jones" states.</p>'
            "</body></html>"
        )
        rendered = [("ch-01", html)]
        vocab = build_vocab("manuscript copy states")
        new, _ = apply_typography_fixes(rendered, vocab=vocab)
        out = new[0][1]
        # All structural tags preserved
        assert out.count("<p ") == 3
        assert out.count("</p>") == 3
        assert "<em>Book</em>" in out  # italic preserved verbatim
        assert "<em>Volume One</em>" in out
        assert 'epub:type="noteref"' in out
        # Fixes applied
        assert "manuscript" in out
        assert "Jones states" in out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
