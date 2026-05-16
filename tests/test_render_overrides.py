"""Tests for the per-book chapter-override layer.

The override mechanism is the documented escape hatch for chapters
whose OCR pipeline output is too corrupt to repair with small
find/replace substitutions. These tests pin the contract that
overrides replace the rendered chapter wholesale, that missing or
empty override directories are no-ops, and that orphan override files
(no rendered chapter shares their slug) surface as a ``stale`` audit
entry without silently swallowing the file.
"""

from __future__ import annotations

from pathlib import Path

from quire.render.overrides import OverrideReport, apply_chapter_overrides


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_no_overrides_dir_is_noop(tmp_path: Path) -> None:
    rendered = [("ch-01", "<p>one</p>"), ("ch-02", "<p>two</p>")]
    out, rep = apply_chapter_overrides(rendered, tmp_path / "does-not-exist")
    assert out == rendered
    assert rep.applied == []
    assert rep.stale == []
    assert "no chapter overrides" in rep.summary_line()


def test_none_overrides_dir_is_noop() -> None:
    rendered = [("ch-01", "<p>one</p>")]
    out, rep = apply_chapter_overrides(rendered, None)
    assert out == rendered
    assert rep.applied == []


def test_empty_overrides_dir_is_noop(tmp_path: Path) -> None:
    overrides = tmp_path / "chapter_overrides"
    overrides.mkdir()
    rendered = [("ch-01", "<p>one</p>")]
    out, rep = apply_chapter_overrides(rendered, overrides)
    assert out == rendered
    assert rep.applied == []
    assert rep.stale == []


def test_override_replaces_matching_chapter(tmp_path: Path) -> None:
    overrides = tmp_path / "chapter_overrides"
    _write(overrides / "ch-02.xhtml", "<p>OVERRIDE</p>")
    rendered = [
        ("ch-01", "<p>one</p>"),
        ("ch-02", "<p>original two</p>"),
        ("ch-03", "<p>three</p>"),
    ]
    out, rep = apply_chapter_overrides(rendered, overrides)
    assert out == [
        ("ch-01", "<p>one</p>"),
        ("ch-02", "<p>OVERRIDE</p>"),
        ("ch-03", "<p>three</p>"),
    ]
    assert rep.applied == ["ch-02"]
    assert rep.stale == []


def test_override_preserves_chapter_order(tmp_path: Path) -> None:
    overrides = tmp_path / "chapter_overrides"
    # Write overrides in arbitrary order; output must match rendered order.
    _write(overrides / "ch-03.xhtml", "<p>three!</p>")
    _write(overrides / "ch-01.xhtml", "<p>one!</p>")
    rendered = [
        ("ch-01", "<p>one</p>"),
        ("ch-02", "<p>two</p>"),
        ("ch-03", "<p>three</p>"),
    ]
    out, _ = apply_chapter_overrides(rendered, overrides)
    assert [slug for slug, _ in out] == ["ch-01", "ch-02", "ch-03"]
    assert out[0][1] == "<p>one!</p>"
    assert out[2][1] == "<p>three!</p>"


def test_stale_override_is_reported_not_inserted(tmp_path: Path) -> None:
    overrides = tmp_path / "chapter_overrides"
    _write(overrides / "ch-99-typo.xhtml", "<p>oops</p>")
    rendered = [("ch-01", "<p>one</p>")]
    out, rep = apply_chapter_overrides(rendered, overrides)
    assert out == rendered, "stale override must not introduce new chapters"
    assert rep.applied == []
    assert rep.stale == ["ch-99-typo"]
    assert "stale" in rep.summary_line()


def test_override_can_replace_multiple_chapters(tmp_path: Path) -> None:
    overrides = tmp_path / "chapter_overrides"
    _write(overrides / "ch-01.xhtml", "<p>A</p>")
    _write(overrides / "ch-02.xhtml", "<p>B</p>")
    rendered = [("ch-01", "<p>1</p>"), ("ch-02", "<p>2</p>")]
    out, rep = apply_chapter_overrides(rendered, overrides)
    assert [html for _, html in out] == ["<p>A</p>", "<p>B</p>"]
    assert sorted(rep.applied) == ["ch-01", "ch-02"]


def test_override_read_with_utf8(tmp_path: Path) -> None:
    """ALA-LC diacritics and Arabic Unicode must round-trip."""
    overrides = tmp_path / "chapter_overrides"
    payload = "<p>ṭawāf الكعبة ʿumrah</p>"
    _write(overrides / "ch-01.xhtml", payload)
    rendered = [("ch-01", "<p>placeholder</p>")]
    out, rep = apply_chapter_overrides(rendered, overrides)
    assert out[0][1] == payload
    assert rep.applied == ["ch-01"]


def test_override_report_summary_line_combinations() -> None:
    assert OverrideReport().summary_line() == "no chapter overrides"
    rep = OverrideReport(applied=["ch-01"])
    assert "1 chapter override(s) applied" in rep.summary_line()
    rep = OverrideReport(applied=["a", "b"], stale=["c"])
    line = rep.summary_line()
    assert "2 chapter override(s) applied" in line
    assert "1 stale override(s) skipped" in line
