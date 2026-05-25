"""Tests for quire.render.cover — programmatic cover generation."""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from quire.render.cover import PALETTES, CoverSpec, render_cover


def test_classic_template_renders_at_requested_size(tmp_path: Path) -> None:
    out = tmp_path / "cover.jpg"
    spec = CoverSpec(
        title="The Book",
        subtitle="A subtitle here",
        author="Author Name",
        section_list=("One", "Two", "Three"),
        template="classic",
        size=(800, 1200),
    )
    p = render_cover(spec, out)
    assert p == out and p.exists()
    img = Image.open(out)
    assert img.size == (800, 1200)
    assert img.mode == "RGB"


def test_minimal_template_works_with_only_title(tmp_path: Path) -> None:
    out = tmp_path / "cover.jpg"
    spec = CoverSpec(title="Minimal", template="minimal", size=(600, 900))
    render_cover(spec, out)
    img = Image.open(out)
    assert img.size == (600, 900)


def test_banded_template_paints_top_and_bottom_bands(tmp_path: Path) -> None:
    out = tmp_path / "cover.jpg"
    spec = CoverSpec(
        title="Banded",
        subtitle="With subtitle",
        author="Me",
        template="banded",
        size=(600, 900),
    )
    render_cover(spec, out)
    img = Image.open(out).convert("RGB")
    # The top-left and bottom-left should be dark band color (ink)
    ink = PALETTES["navy-gold"]["ink"]
    top_px = img.getpixel((10, 10))
    bot_px = img.getpixel((10, img.height - 10))
    # Allow JPEG compression rounding
    assert all(abs(a - b) < 20 for a, b in zip(top_px, ink, strict=True))
    assert all(abs(a - b) < 20 for a, b in zip(bot_px, ink, strict=True))


def test_palette_selection(tmp_path: Path) -> None:
    out = tmp_path / "cover.jpg"
    spec = CoverSpec(title="Sepia", palette="sepia", template="minimal",
                     size=(500, 700))
    render_cover(spec, out)
    img = Image.open(out).convert("RGB")
    # Background corner should match sepia bg roughly
    bg = PALETTES["sepia"]["bg"]
    corner = img.getpixel((5, 5))
    assert all(abs(a - b) < 15 for a, b in zip(corner, bg, strict=True))


def test_unknown_palette_falls_back_to_default(tmp_path: Path) -> None:
    out = tmp_path / "cover.jpg"
    spec = CoverSpec(title="X", palette="nope-not-real", template="minimal",
                     size=(400, 600))
    # Should not raise
    render_cover(spec, out)
    assert out.exists()


def test_unknown_template_falls_back_to_classic(tmp_path: Path) -> None:
    out = tmp_path / "cover.jpg"
    spec = CoverSpec(title="X", template="something-weird", size=(400, 600))
    render_cover(spec, out)
    assert out.exists()
