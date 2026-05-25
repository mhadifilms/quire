"""Tests for quire.extract.spreads — rotation, binding detection, split."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest
from PIL import Image, ImageDraw

from quire.extract import spreads


def _make_synthetic_spread(width: int = 1000, height: int = 500,
                            binding_at: int = 500, binding_w: int = 30,
                            text_lines: int = 8) -> Image.Image:
    """Build a synthetic upright two-page spread: white pages with text-like
    horizontal bands, separated by a dark vertical binding band in the middle.
    """
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    # Text-like bands on both pages: alternating dark strips
    band_h = height // (text_lines * 2)
    for page_x_range in [(20, binding_at - binding_w // 2 - 20),
                         (binding_at + binding_w // 2 + 20, width - 20)]:
        for i in range(text_lines):
            y0 = 30 + i * band_h * 2
            d.rectangle((page_x_range[0], y0, page_x_range[1], y0 + band_h),
                        fill=(40, 40, 40))
    # Dark binding band
    d.rectangle((binding_at - binding_w // 2, 0,
                 binding_at + binding_w // 2, height), fill=(20, 20, 20))
    return img


def _make_single_page(width: int = 500, height: int = 700) -> Image.Image:
    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)
    for i in range(10):
        y0 = 30 + i * 60
        d.rectangle((40, y0, width - 40, y0 + 20), fill=(40, 40, 40))
    return img


def test_detect_rotation_zero_for_upright_page() -> None:
    img = _make_single_page()
    deg = spreads.detect_rotation(img)
    # Upright: row variance dominates. Allow 0 (best) or 180 (also row-dom).
    assert deg in (0, 180)


def test_detect_rotation_finds_ninety_when_input_is_rotated() -> None:
    upright = _make_single_page()
    rotated_cw_90 = upright.rotate(-90, expand=True)  # PIL -90 = CW
    deg = spreads.detect_rotation(rotated_cw_90)
    # Should report 90 or 270 (both flip horizontal/vertical text axes back)
    assert deg in (90, 270)


def test_detect_binding_band_finds_middle_band() -> None:
    img = _make_synthetic_spread(width=1000, binding_at=500, binding_w=40)
    band = spreads.detect_binding_band(img, axis="x")
    assert band is not None
    assert band.axis == "x"
    # Center should be near 500
    assert abs(band.center - 500) < 60
    # Width should be roughly the binding width
    assert 10 < band.width < 200
    assert band.confidence > 0.05


def test_detect_binding_band_returns_none_on_single_page() -> None:
    img = _make_single_page()
    band = spreads.detect_binding_band(img, axis="x")
    assert band is None


def test_detect_binding_band_auto_picks_dominant_axis() -> None:
    # Spread with VERTICAL binding (split horizontally) → axis 'x' wins
    img = _make_synthetic_spread()
    band = spreads.detect_binding_band(img, axis="auto")
    assert band is not None
    assert band.axis == "x"


def test_split_spread_returns_two_pages() -> None:
    img = _make_synthetic_spread(width=1000, binding_at=500, binding_w=40)
    band = spreads.detect_binding_band(img, axis="x")
    assert band is not None
    left, right = spreads.split_spread(img, band)
    assert left.size[1] == img.size[1]
    assert right.size[1] == img.size[1]
    # Left and right should not overlap the binding
    assert left.size[0] + right.size[0] < img.size[0]


def test_split_spread_horizontal_binding() -> None:
    # Build a spread with HORIZONTAL binding (top page above, bottom below)
    upright = _make_synthetic_spread(width=1000, height=500, binding_at=500, binding_w=40)
    flipped = upright.rotate(90, expand=True)  # binding becomes horizontal
    band = spreads.detect_binding_band(flipped, axis="auto")
    assert band is not None
    assert band.axis == "y"
    top, bottom = spreads.split_spread(flipped, band)
    assert top.size[0] == flipped.size[0]
    assert bottom.size[0] == flipped.size[0]


def test_split_pdf_emits_one_image_per_page_when_no_spread(tmp_path: Path) -> None:
    pdf_path = tmp_path / "single.pdf"
    doc = fitz.open()
    for _ in range(3):
        p = doc.new_page(width=400, height=600)
        p.insert_text((60, 100), "Hello world.", fontname="helv", fontsize=14)
        for y in range(140, 540, 40):
            p.insert_text((60, y), "Line of text " * 4, fontname="helv", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()
    out_dir = tmp_path / "pages"
    paths = spreads.split_pdf(pdf_path, out_dir, dpi=72, rotate=False,
                              detect_spreads=False)
    assert len(paths) == 3
    for p in paths:
        assert p.exists() and p.suffix == ".jpg"


def test_split_pdf_splits_spread_pdf(tmp_path: Path) -> None:
    """End-to-end: build a PDF whose pages are spread-shaped (binding in middle),
    confirm split_pdf doubles the page count."""
    pdf_path = tmp_path / "spreads.pdf"
    # Build a PDF page from a synthetic spread PNG
    spread_img = _make_synthetic_spread(width=1000, height=500, binding_at=500, binding_w=40)
    img_bytes_path = tmp_path / "spread.png"
    spread_img.save(img_bytes_path)
    doc = fitz.open()
    for _ in range(2):
        page = doc.new_page(width=1000, height=500)
        page.insert_image(fitz.Rect(0, 0, 1000, 500), filename=str(img_bytes_path))
    doc.save(str(pdf_path))
    doc.close()
    out_dir = tmp_path / "out"
    paths = spreads.split_pdf(pdf_path, out_dir, dpi=72, rotate=False,
                              detect_spreads=True)
    # 2 PDF pages × 2 split pages = 4 output images
    assert len(paths) == 4
