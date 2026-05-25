"""Font-coverage analysis and image-fallback for problematic glyphs.

EPUB readers vary enormously in font coverage for the Arabic Presentation
Forms-A block (U+FB50–FDFF). Apple Books in particular has ``ﷺ`` (U+FDFA)
and ``ﷻ`` (U+FDFB) but is missing most U+FD40–FDF9 honorific ligatures and
the combining marks U+0610–U+0615 (``ؐؑؒؓؔؕ``). When the reader's default
font lacks a glyph, it renders as a tofu box (``□``) — even when an
``@font-face`` font that has the glyph is embedded in the EPUB, because
some readers won't switch fonts for individual codepoints.

The robust workaround is to pre-render those glyphs to PNG using a font that
*does* have them (e.g. Scheherazade New, Amiri), embed the PNGs in the
EPUB, and substitute the text with inline ``<img>`` tags. This module
contains the pieces:

    coverage_scan(text, font_paths)        → {codepoint: [font_name, ...]}
    suspicious_codepoints(text, profile)   → {codepoint, ...}
    render_glyph_png(codepoint, font, out) → Path
    substitute_with_images(html, mapping)  → str

The ``profile`` selects a heuristic set of codepoints commonly broken in a
given reader family; today we ship ``ios-books`` and ``conservative``.

This module needs ``fontTools`` to read the font cmap. It's listed as a
soft dep; an ImportError is raised lazily only when ``coverage_scan`` is
called.
"""

from __future__ import annotations

import html as _html
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------- Reader-coverage profiles ----------

# Codepoints commonly missing from the default-font fallback chain on each
# reader family. These are reader heuristics, not authoritative lists.
_PROFILE_SUSPICIOUS: dict[str, frozenset[int]] = {
    # Apple Books on iOS / macOS: Iowan / Palatino / Athelas defaults lack
    # most U+FD40–FDF9 honorific ligatures and most combining marks above.
    "ios-books": frozenset(
        list(range(0x0610, 0x0616))                            # ؐؑؒؓؔؕ
        + [cp for cp in range(0xFD40, 0xFDFA) if cp != 0xFDF2]  # all U+FD40-U+FDF9 except U+FDF2 ﷲ
    ),
    # Be conservative: assume nothing in the Arabic-Presentation-Forms-A
    # block beyond U+FB50–FB99 and U+FDF2/FDFA/FDFB is reliable. Use this
    # when targeting unknown readers.
    "conservative": frozenset(
        list(range(0x0610, 0x0616))
        + [cp for cp in range(0xFD40, 0xFE00) if cp not in (0xFDF2, 0xFDFA, 0xFDFB)]
    ),
    # Empty: never substitute. Useful for tests.
    "none": frozenset(),
}


def suspicious_codepoints(text: str, profile: str = "ios-books") -> set[int]:
    """Return the subset of *text* codepoints likely to render as tofu in *profile*.

    The default profile (``ios-books``) is appropriate when the primary
    distribution target is Apple Books on iPhone/iPad.
    """
    if profile not in _PROFILE_SUSPICIOUS:
        raise ValueError(
            f"unknown profile {profile!r}; choose one of {sorted(_PROFILE_SUSPICIOUS)}"
        )
    bad = _PROFILE_SUSPICIOUS[profile]
    return {ord(ch) for ch in text if ord(ch) in bad}


# ---------- Font cmap reading ----------


def coverage_scan(text: str, font_paths: list[Path]) -> dict[int, list[str]]:
    """For every distinct codepoint in *text*, list which of *font_paths* contain it.

    Returns a dict from codepoint → list of font *names* (filename without
    extension) that contain that codepoint. A codepoint with an empty list
    has no coverage in any provided font.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError as e:
        raise ImportError(
            "quire.render.honorifics.coverage_scan requires fontTools. "
            "Install with: pip install fontTools"
        ) from e

    cmaps: list[tuple[str, dict[int, str]]] = []
    for fp in font_paths:
        try:
            cmaps.append((fp.stem, TTFont(str(fp)).getBestCmap() or {}))
        except Exception:
            cmaps.append((fp.stem, {}))

    out: dict[int, list[str]] = {}
    for ch in set(text):
        cp = ord(ch)
        if cp < 0x80:
            continue
        in_fonts = [name for name, cmap in cmaps if cp in cmap]
        out[cp] = in_fonts
    return out


# ---------- Glyph rendering ----------


@dataclass(frozen=True)
class HonorificImage:
    """A pre-rendered honorific image, ready to embed."""

    codepoint: int
    png_path: Path
    width_px: int
    height_px: int
    alt: str


def render_glyph_png(
    codepoint: int,
    font_path: Path,
    out_path: Path,
    *,
    size_px: int = 220,
    padding: int = 20,
    color: tuple[int, int, int, int] = (0, 0, 0, 255),
    base_char: str | None = None,
) -> HonorificImage:
    """Render a single glyph (or combining mark on a base) as a PNG.

    *base_char* is used for combining marks (e.g. U+0611 ؑ) that don't render
    standalone. Pass an Arabic letter like ``"ع"`` (U+0639) to anchor the
    mark above it.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    glyph = (base_char or "") + chr(codepoint)
    font = ImageFont.truetype(str(font_path), size_px)
    bbox = font.getbbox(glyph)
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    img = Image.new("RGBA", (w + padding * 2, h + padding * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((padding - bbox[0], padding - bbox[1]), glyph, font=font, fill=color)
    cropped_bbox = img.getbbox()
    if cropped_bbox is not None:
        img = img.crop(cropped_bbox)
    img.save(out_path, "PNG", optimize=True)
    return HonorificImage(
        codepoint=codepoint,
        png_path=out_path,
        width_px=img.size[0],
        height_px=img.size[1],
        alt=_default_alt(codepoint),
    )


# Friendly alt-text for the codepoints we ship images for. Falls back to
# Unicode name lookup at runtime when not in this table.
_ALT_TEXT: dict[int, str] = {
    0xFDFA: "ṣallallāhu ʿalayhi wa-ālihi wa-sallam",
    0xFDFB: "subḥānahu wa-taʿālā",
    0xFDF2: "Allah",
    0xFD47: "ʿalayhi as-salām",
    0x0611: "ʿalayhi as-salām",
    0x0610: "ṣallallāhu ʿalayhi wa-sallam",
    0x0612: "raḥmatu allāh ʿalayhi",
    0x0613: "raḍiya allāhu ʿanhu",
}


def _default_alt(cp: int) -> str:
    if cp in _ALT_TEXT:
        return _ALT_TEXT[cp]
    try:  # pragma: no cover
        import unicodedata
        return unicodedata.name(chr(cp), f"U+{cp:04X}")
    except ValueError:  # pragma: no cover
        return f"U+{cp:04X}"


def render_honorific_set(
    codepoints: list[int],
    font_path: Path,
    out_dir: Path,
    *,
    base_for_combining: str = "ع",
    size_px: int = 220,
) -> dict[int, HonorificImage]:
    """Render each requested *codepoints* as a PNG into *out_dir*.

    Combining marks (U+0610–U+0615) are anchored on *base_for_combining*
    (``"ع"`` ʿain by default, which is the closest visual analogue for the
    Imam honorific in Shia scholarly typography).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, HonorificImage] = {}
    for cp in codepoints:
        is_combining = 0x0610 <= cp <= 0x0615 or 0x064B <= cp <= 0x065F
        png_name = f"hon-{cp:04x}.png"
        png_path = out_dir / png_name
        out[cp] = render_glyph_png(
            cp,
            font_path,
            png_path,
            size_px=size_px,
            base_char=base_for_combining if is_combining else None,
        )
    return out


# ---------- HTML substitution ----------


def substitute_with_images(
    html_text: str,
    mapping: dict[int, HonorificImage],
    *,
    img_href_prefix: str = "images/",
    css_class: str = "hon-img",
) -> str:
    """Replace each codepoint in *mapping* with an ``<img>`` tag.

    *img_href_prefix* is prepended to the PNG basename to form the ``src``
    attribute (relative to the chapter XHTML location). The substitution
    is character-by-character; no other markup is touched.
    """
    if not mapping:
        return html_text

    chars_to_swap = {chr(cp) for cp in mapping}
    pattern = re.compile("[" + "".join(re.escape(c) for c in chars_to_swap) + "]")

    def _sub(m: re.Match[str]) -> str:
        cp = ord(m.group(0))
        spec = mapping[cp]
        src = f"{img_href_prefix}{spec.png_path.name}"
        alt = _html.escape(spec.alt)
        return f'<img src="{src}" alt="{alt}" class="{css_class}"/>'

    return pattern.sub(_sub, html_text)


# ---------- CSS snippet ----------


HONORIFIC_CSS = """\
/* Honorific glyphs pre-rendered as PNG to bypass reader font-coverage gaps.
   Black on transparent; CSS-inverted in dark mode so they read as white. */
.hon-img {
  height: 0.95em;
  vertical-align: -0.1em;
  margin: 0 0.05em;
  display: inline;
}
@media (prefers-color-scheme: dark) {
  .hon-img { filter: invert(1); }
}
"""
