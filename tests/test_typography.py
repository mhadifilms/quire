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

    def test_skips_identical_replace(self) -> None:
        # apply_qc_fix doesn't filter identical -- caller (load_qc_fixes
        # / apply_typography_fixes) is expected to. But it should still
        # not enter an infinite loop.
        html = "<p>hello</p>"
        new, n = apply_qc_fix(html, "hello", "hello")
        # Replaces but loop safety caps it.
        assert n <= 50
        assert new.endswith("</p>")


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

    def test_drops_identical_and_empty(self, tmp_path: pathlib.Path) -> None:
        p = tmp_path / "qc_fixes.toml"
        p.write_text(
            '[phrase]\n"same" = "same"\n"" = "x"\n"keep" = "keeper"\n',
            encoding="utf-8",
        )
        assert load_qc_fixes(p) == {"keep": "keeper"}


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
