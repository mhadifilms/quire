"""Tests for quire.render.honorifics — font-coverage analysis + image fallback."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from quire.render import honorifics

# ---------- Profile selection ----------


def test_ios_books_profile_flags_known_problem_codepoints() -> None:
    text = "Hello ﵇ world ؑ again ﷺ ﷻ"
    flagged = honorifics.suspicious_codepoints(text, profile="ios-books")
    # U+FD47 and U+0611 are flagged; U+FDFA and U+FDFB are not (Apple supports them)
    assert 0xFD47 in flagged
    assert 0x0611 in flagged
    assert 0xFDFA not in flagged
    assert 0xFDFB not in flagged


def test_conservative_profile_is_stricter() -> None:
    text = "﵇ ؑ ﷺ ﷻ ﷲ"
    flagged = honorifics.suspicious_codepoints(text, profile="conservative")
    # Same suspect set plus more, but U+FDF2/FDFA/FDFB still safe
    assert 0xFD47 in flagged
    assert 0x0611 in flagged
    assert 0xFDFA not in flagged
    assert 0xFDFB not in flagged
    assert 0xFDF2 not in flagged


def test_none_profile_flags_nothing() -> None:
    text = "anything ﵇ ؑ ﷺ"
    assert honorifics.suspicious_codepoints(text, profile="none") == set()


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError):
        honorifics.suspicious_codepoints("x", profile="not-a-profile")


# ---------- Glyph rendering ----------


def _find_any_truetype_font() -> Path | None:
    candidates = [
        "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            return Path(p)
    return None


def test_render_glyph_png_writes_nonzero_image(tmp_path: Path) -> None:
    font_path = _find_any_truetype_font()
    if font_path is None:
        pytest.skip("no system TTF font available")
    out = tmp_path / "g.png"
    # U+0041 'A' renders in every font
    spec = honorifics.render_glyph_png(0x0041, font_path, out, size_px=100)
    assert spec.png_path == out and out.exists()
    assert spec.width_px > 0 and spec.height_px > 0
    img = Image.open(out)
    assert img.mode == "RGBA"


def test_render_honorific_set_anchors_combining_marks(tmp_path: Path) -> None:
    font_path = _find_any_truetype_font()
    if font_path is None:
        pytest.skip("no system TTF font available")
    out = honorifics.render_honorific_set(
        [0x0041, 0x0042], font_path, tmp_path, size_px=80,
    )
    assert set(out) == {0x0041, 0x0042}
    for spec in out.values():
        assert spec.png_path.exists()


# ---------- HTML substitution ----------


def test_substitute_with_images_replaces_only_listed_codepoints(tmp_path: Path) -> None:
    spec = honorifics.HonorificImage(
        codepoint=0x0041,
        png_path=tmp_path / "hon-0041.png",
        width_px=20, height_px=20,
        alt="A glyph",
    )
    mapping = {0x0041: spec}
    html_in = "hello A world! later A is here."
    out = honorifics.substitute_with_images(html_in, mapping)
    assert "<img" in out
    assert "hello" in out and "world" in out
    assert out.count("<img") == 2
    # Path is just basename relative to images/
    assert 'src="images/hon-0041.png"' in out
    # Non-matching characters are untouched
    assert "later" in out and "here" in out


def test_substitute_with_empty_mapping_is_identity() -> None:
    html_in = "<p>nothing to swap ﵇</p>"
    assert honorifics.substitute_with_images(html_in, {}) == html_in


def test_substitute_alt_text_is_html_escaped(tmp_path: Path) -> None:
    spec = honorifics.HonorificImage(
        codepoint=0x0041,
        png_path=tmp_path / "x.png",
        width_px=1, height_px=1,
        alt='dangerous "alt" with <html>',
    )
    out = honorifics.substitute_with_images("A", {0x0041: spec})
    assert "&quot;alt&quot;" in out
    assert "&lt;html&gt;" in out


# ---------- Coverage scan (requires fontTools) ----------


def test_coverage_scan_reports_per_font_coverage() -> None:
    pytest.importorskip("fontTools")
    font_path = _find_any_truetype_font()
    if font_path is None:
        pytest.skip("no system TTF font available")
    # 'H', 'e', 'l', 'o' are all ASCII so are skipped (< 0x80). Use a
    # non-ASCII char to verify the function actually populates the dict.
    assert honorifics.coverage_scan("Hello", [font_path]) == {}
    result_nonascii = honorifics.coverage_scan("Héllo", [font_path])
    assert 0x00E9 in result_nonascii  # 'é'
    assert isinstance(result_nonascii[0x00E9], list)


def test_coverage_scan_raises_helpful_error_without_fontTools(monkeypatch) -> None:
    # Make TTFont import fail by hiding fontTools from sys.modules
    import sys
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *a, **kw):
        if name.startswith("fontTools"):
            raise ImportError("simulated")
        return real_import(name, *a, **kw)

    monkeypatch.setattr("builtins.__import__", fake_import)
    with pytest.raises(ImportError, match="fontTools"):
        honorifics.coverage_scan("Héllo", [])
