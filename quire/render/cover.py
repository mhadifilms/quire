"""Auto-generate a book cover from BookConfig when no cover image is supplied.

Many books in the Quire pipeline arrive as a bare PDF + ``book.toml`` with no
cover artwork. This module produces a plain but presentable JPG cover from
just the metadata (title, subtitle, author, optional section list), in a
small set of selectable templates. It writes to ``artifacts/cover.jpg`` so
the packager picks it up via the normal cover-image path.

Templates are deliberately conservative: no decorative imagery, no scraped
fonts, no glyphs from the source PDF. The output is fully reproducible from
the inputs.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------- Public API ----------


@dataclass(frozen=True)
class CoverSpec:
    """Inputs to the cover generator. Strings should already be normalized."""

    title: str
    subtitle: str = ""
    author: str = ""
    section_list: tuple[str, ...] = ()
    template: str = "classic"  # 'classic' | 'minimal' | 'banded'
    size: tuple[int, int] = (1200, 1800)
    palette: str = "navy-gold"  # 'navy-gold' | 'sepia' | 'monochrome'


PALETTES: dict[str, dict[str, tuple[int, int, int]]] = {
    "navy-gold": {
        "bg": (246, 238, 215),
        "bg_dust": (240, 229, 198),
        "accent": (191, 142, 56),
        "accent_2": (212, 170, 84),
        "ink": (30, 44, 77),
        "ink_2": (70, 90, 130),
        "rust": (138, 63, 37),
    },
    "sepia": {
        "bg": (244, 234, 215),
        "bg_dust": (235, 223, 198),
        "accent": (132, 87, 41),
        "accent_2": (160, 117, 71),
        "ink": (62, 41, 22),
        "ink_2": (104, 78, 50),
        "rust": (122, 56, 32),
    },
    "monochrome": {
        "bg": (245, 245, 245),
        "bg_dust": (236, 236, 236),
        "accent": (60, 60, 60),
        "accent_2": (100, 100, 100),
        "ink": (20, 20, 20),
        "ink_2": (70, 70, 70),
        "rust": (90, 90, 90),
    },
}


def render_cover(spec: CoverSpec, out_path: Path, *, font_dir: Path | None = None) -> Path:
    """Render *spec* to *out_path* (.jpg). Returns the path written."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pal = PALETTES.get(spec.palette, PALETTES["navy-gold"])
    if spec.template == "minimal":
        img = _render_minimal(spec, pal, font_dir)
    elif spec.template == "banded":
        img = _render_banded(spec, pal, font_dir)
    else:
        img = _render_classic(spec, pal, font_dir)
    img.save(out_path, "JPEG", quality=92, optimize=True)
    return out_path


# ---------- Font loading (graceful fallback) ----------


_SERIF_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/Library/Fonts/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
)


def _find_serif(font_dir: Path | None, want_bold: bool) -> str | None:
    if font_dir and font_dir.is_dir():
        prefer = ["Serif-Bold", "Bold", "Serif", "Regular"] if want_bold else ["Serif", "Regular"]
        for stub in prefer:
            for path in font_dir.glob(f"*{stub}*.ttf"):
                return str(path)
            for path in font_dir.glob(f"*{stub}*.otf"):
                return str(path)
    for path in _SERIF_CANDIDATES:
        if Path(path).exists() and ("Bold" in path) == want_bold:
            return path
    for path in _SERIF_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _font(size: int, *, font_dir: Path | None, bold: bool = False) -> ImageFont.ImageFont:
    p = _find_serif(font_dir, want_bold=bold)
    if p:
        try:
            return ImageFont.truetype(p, size)
        except OSError:  # pragma: no cover
            pass
    return ImageFont.load_default()


def _text_w(d: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    """Compatible across Pillow versions (textlength returns float in 10+)."""
    if hasattr(d, "textlength"):
        return d.textlength(text, font=font)
    bbox = d.textbbox((0, 0), text, font=font)  # pragma: no cover
    return bbox[2] - bbox[0]


# ---------- Templates ----------


def _add_paper_texture(img: Image.Image, pal: dict[str, tuple[int, int, int]], seed: int = 7) -> None:
    """Sprinkle a subtle dot texture so the cover doesn't look flat."""
    rng = random.Random(seed)
    d = ImageDraw.Draw(img)
    w, h = img.size
    for _ in range(w * h // 240):
        x = rng.randrange(w)
        y = rng.randrange(h)
        d.point((x, y), fill=pal["bg_dust"])


def _draw_band(d: ImageDraw.ImageDraw, x0: int, y0: int, x1: int, y1: int,
               pal: dict[str, tuple[int, int, int]]) -> None:
    """Gold band with inscribed diamond pattern."""
    d.rectangle((x0, y0, x1, y1), fill=pal["accent"])
    step = 40
    diamond = 12
    cy = (y0 + y1) // 2
    for x in range(x0 + 30, x1 - 20, step):
        d.polygon(
            [(x, cy - diamond), (x + diamond, cy), (x, cy + diamond), (x - diamond, cy)],
            outline=pal["bg"], width=2,
        )


def _render_classic(spec: CoverSpec, pal: dict[str, tuple[int, int, int]],
                    font_dir: Path | None) -> Image.Image:
    W, H = spec.size
    img = Image.new("RGB", (W, H), pal["bg"])
    _add_paper_texture(img, pal)
    d = ImageDraw.Draw(img)

    margin = 50
    d.rectangle((margin, margin, W - margin, H - margin), outline=pal["accent"], width=3)
    d.rectangle((margin + 10, margin + 10, W - margin - 10, H - margin - 10),
                outline=pal["accent_2"], width=1)

    # Top band
    _draw_band(d, margin + 30, margin + 40, W - margin - 30, margin + 100, pal)

    # Title
    title_y = int(H * 0.18)
    f_title = _font(int(H * 0.12), font_dir=font_dir, bold=True)
    tw = _text_w(d, spec.title, f_title)
    d.text(((W - tw) / 2, title_y), spec.title, font=f_title, fill=pal["ink"])
    title_h = int(H * 0.13)

    # Decorative 8-point star
    if spec.subtitle or spec.section_list:
        star_y = title_y + title_h + 60
        cx = W // 2
        for i in range(8):
            angle = math.radians(i * 45)
            x1 = cx + 5 * math.cos(angle)
            y1 = star_y + 5 * math.sin(angle)
            x2 = cx + 26 * math.cos(angle)
            y2 = star_y + 26 * math.sin(angle)
            d.line((x1, y1, x2, y2), fill=pal["accent"], width=3)
        d.ellipse((cx - 7, star_y - 7, cx + 7, star_y + 7), fill=pal["accent"])
        sub_y = star_y + 60
    else:
        sub_y = title_y + title_h + 60

    # Subtitle
    if spec.subtitle:
        f_sub = _font(int(H * 0.028), font_dir=font_dir)
        sw = _text_w(d, spec.subtitle, f_sub)
        d.text(((W - sw) / 2, sub_y), spec.subtitle, font=f_sub, fill=pal["ink_2"])
        sub_y += int(H * 0.05)

    # Sections list
    if spec.section_list:
        f_sect = _font(int(H * 0.024), font_dir=font_dir)
        list_top = sub_y + 60
        spacing = int(H * 0.045)
        for i, s in enumerate(spec.section_list):
            y = list_top + i * spacing
            tw = _text_w(d, s, f_sect)
            d.text(((W - tw) / 2, y), s, font=f_sect, fill=pal["ink"])
            bx_left = (W - tw) / 2 - 40
            bx_right = (W + tw) / 2 + 40
            ydot = y + int(H * 0.013)
            d.ellipse((bx_left - 5, ydot, bx_left + 5, ydot + 10), fill=pal["accent"])
            d.ellipse((bx_right - 5, ydot, bx_right + 5, ydot + 10), fill=pal["accent"])

    # Bottom band
    bband_y1 = H - margin - 40
    bband_y0 = bband_y1 - 60
    _draw_band(d, margin + 30, bband_y0, W - margin - 30, bband_y1, pal)

    # Author above bottom band
    if spec.author:
        f_auth = _font(int(H * 0.022), font_dir=font_dir)
        aw = _text_w(d, spec.author, f_auth)
        d.text(((W - aw) / 2, bband_y0 - 100), spec.author, font=f_auth, fill=pal["ink"])

    return img


def _render_minimal(spec: CoverSpec, pal: dict[str, tuple[int, int, int]],
                    font_dir: Path | None) -> Image.Image:
    W, H = spec.size
    img = Image.new("RGB", (W, H), pal["bg"])
    d = ImageDraw.Draw(img)
    f_title = _font(int(H * 0.075), font_dir=font_dir, bold=True)
    tw = _text_w(d, spec.title, f_title)
    title_y = int(H * 0.42)
    d.text(((W - tw) / 2, title_y), spec.title, font=f_title, fill=pal["ink"])
    d.line(((W - tw) / 2 - 40, title_y + int(H * 0.085),
            (W + tw) / 2 + 40, title_y + int(H * 0.085)),
           fill=pal["accent"], width=2)
    if spec.subtitle:
        f_sub = _font(int(H * 0.025), font_dir=font_dir)
        sw = _text_w(d, spec.subtitle, f_sub)
        d.text(((W - sw) / 2, title_y + int(H * 0.11)),
               spec.subtitle, font=f_sub, fill=pal["ink_2"])
    if spec.author:
        f_auth = _font(int(H * 0.022), font_dir=font_dir)
        aw = _text_w(d, spec.author, f_auth)
        d.text(((W - aw) / 2, H - 200), spec.author, font=f_auth, fill=pal["ink"])
    return img


def _render_banded(spec: CoverSpec, pal: dict[str, tuple[int, int, int]],
                   font_dir: Path | None) -> Image.Image:
    W, H = spec.size
    img = Image.new("RGB", (W, H), pal["bg"])
    d = ImageDraw.Draw(img)
    band_h = int(H * 0.32)
    d.rectangle((0, 0, W, band_h), fill=pal["ink"])
    d.rectangle((0, H - band_h, W, H), fill=pal["ink"])
    f_title = _font(int(H * 0.07), font_dir=font_dir, bold=True)
    tw = _text_w(d, spec.title, f_title)
    d.text(((W - tw) / 2, band_h // 2 - int(H * 0.04)),
           spec.title, font=f_title, fill=pal["bg"])
    if spec.subtitle:
        f_sub = _font(int(H * 0.025), font_dir=font_dir)
        sw = _text_w(d, spec.subtitle, f_sub)
        d.text(((W - sw) / 2, band_h - int(H * 0.06)),
               spec.subtitle, font=f_sub, fill=pal["accent_2"])
    if spec.author:
        f_auth = _font(int(H * 0.022), font_dir=font_dir)
        aw = _text_w(d, spec.author, f_auth)
        d.text(((W - aw) / 2, H - band_h // 2 - 20),
               spec.author, font=f_auth, fill=pal["bg"])
    return img
