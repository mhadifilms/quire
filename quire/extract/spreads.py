"""Two-page-spread detection and splitting for phone-scanned books.

A common source PDF for Quire is a book photographed on a phone with the
binding (spiral, comb, or saddle stitch) running horizontally across the
middle of each shot. Each scan therefore contains two book pages stacked
vertically, plus is rotated 90° from upright reading orientation.

This module turns those scans into a sequence of single upright pages:

    detect_rotation(img)      → 0 | 90 | 180 | 270
    detect_binding_band(img)  → (col_start, col_end) | None
    split_spread(img, band)   → (left_page, right_page)
    split_pdf(pdf, out_dir)   → list[Path]   (high-level orchestrator)

The binding detection works on any spiral / wire-bound book: the metal
rings drop the column-mean luminance well below the page mean for a thin
contiguous band. Saddle-stitched books with a deep gutter shadow work too.

No deps beyond Pillow and PyMuPDF (already required by Quire). NumPy is
used when available for a ~50× speedup on large images, with a pure-Pillow
fallback so the module keeps working without it.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import fitz
from PIL import Image

try:
    import numpy as _np

    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None
    _HAVE_NUMPY = False


# ---------- Brightness analysis ----------


def _column_means(img: Image.Image) -> list[float]:
    """Return mean luminance per column (length == img.width)."""
    L = img.convert("L")
    if _HAVE_NUMPY:
        arr = _np.asarray(L, dtype=_np.uint16)  # H x W
        return arr.mean(axis=0).tolist()
    w, h = L.size
    data = L.tobytes()
    out: list[float] = []
    for x in range(w):
        s = 0
        for y in range(h):
            s += data[y * w + x]
        out.append(s / h)
    return out


def _row_means(img: Image.Image) -> list[float]:
    """Return mean luminance per row (length == img.height)."""
    L = img.convert("L")
    if _HAVE_NUMPY:
        arr = _np.asarray(L, dtype=_np.uint16)
        return arr.mean(axis=1).tolist()
    w, h = L.size
    data = L.tobytes()
    out: list[float] = []
    for y in range(h):
        row = data[y * w : (y + 1) * w]
        out.append(sum(row) / w)
    return out


# ---------- Rotation ----------


def detect_rotation(img: Image.Image) -> int:
    """Best-guess upright rotation in {0, 90, 180, 270}.

    Heuristic: most book pages have more horizontal text-line variance than
    vertical (because text runs in horizontal strips with whitespace gaps
    between lines). When upright, the row-means show distinct alternating
    bright/dark bands; rotated 90°, the column-means do instead. We pick the
    orientation whose row-mean variance dominates.

    Returns the rotation (in degrees clockwise) that should be applied to
    *img* to make it upright. So if the input is rotated 90° CW from upright,
    we return 270 (rotate 270 CW == 90 CCW to undo).
    """
    candidates = [0, 90, 180, 270]
    scores: list[tuple[int, float]] = []
    for deg in candidates:
        rotated = img if deg == 0 else img.rotate(-deg, expand=True)
        rows = _row_means(rotated)
        cols = _column_means(rotated)
        if _HAVE_NUMPY:
            row_var = float(_np.var(_np.asarray(rows)))
            col_var = float(_np.var(_np.asarray(cols)))
        else:
            row_var = _variance(rows)
            col_var = _variance(cols)
        # Higher score = more "text-y" horizontal bands
        score = row_var / max(col_var, 1.0)
        scores.append((deg, score))
    best_deg, _ = max(scores, key=lambda t: t[1])
    return best_deg


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    m = sum(values) / len(values)
    return sum((v - m) ** 2 for v in values) / len(values)


# ---------- Binding detection ----------


@dataclass(frozen=True)
class BindingBand:
    """A detected binding band on an upright image.

    Coordinates are in pixels along the *splitting axis*. For a normal
    landscape spread (two pages side-by-side, binding vertical) the axis
    is the X axis. For a portrait spread (two pages top-and-bottom,
    binding horizontal) the axis is Y.
    """

    axis: str  # 'x' or 'y'
    start: int  # column/row where the binding band begins
    end: int  # column/row where it ends (exclusive)
    confidence: float  # 0..1, drop magnitude relative to background

    @property
    def center(self) -> int:
        return (self.start + self.end) // 2

    @property
    def width(self) -> int:
        return self.end - self.start


def detect_binding_band(
    img: Image.Image,
    axis: str = "auto",
    search_fraction: float = 0.4,
    drop_sigmas: float = 1.2,
    min_confidence: float = 0.05,
) -> BindingBand | None:
    """Detect a contiguous dark band near the middle that looks like a binding.

    Args:
        img: Pillow image (any mode; we convert to grayscale internally).
        axis: 'x' (vertical binding, split horizontally), 'y' (horizontal
            binding, split vertically), or 'auto' to pick the axis whose
            center-region brightness drop is sharpest.
        search_fraction: how far around the center to look. 0.4 means we
            look in the middle 40% (20% either side) of the image.
        drop_sigmas: a column/row counts as "dark" if it's this many
            std-devs below the overall mean.
        min_confidence: minimum drop magnitude / mean to accept a band.

    Returns:
        A BindingBand, or None if no clear binding is found (i.e. the page
        is probably a single page, not a spread).
    """
    if axis == "auto":
        x_band = detect_binding_band(img, axis="x", search_fraction=search_fraction,
                                      drop_sigmas=drop_sigmas, min_confidence=min_confidence)
        y_band = detect_binding_band(img, axis="y", search_fraction=search_fraction,
                                      drop_sigmas=drop_sigmas, min_confidence=min_confidence)
        if x_band is None and y_band is None:
            return None
        if x_band is None:
            return y_band
        if y_band is None:
            return x_band
        return x_band if x_band.confidence >= y_band.confidence else y_band

    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x', 'y', or 'auto', got {axis!r}")

    series = _column_means(img) if axis == "x" else _row_means(img)
    n = len(series)
    if n < 10:
        return None

    if _HAVE_NUMPY:
        arr = _np.asarray(series)
        mean = float(arr.mean())
        std = float(arr.std()) or 1.0
    else:
        mean = sum(series) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in series) / n) or 1.0

    center = n // 2
    half_window = int(n * search_fraction / 2)
    lo = max(0, center - half_window)
    hi = min(n, center + half_window)

    threshold = mean - drop_sigmas * std
    dark_positions = [i for i in range(lo, hi) if series[i] < threshold]
    if not dark_positions:
        return None

    # Find the contiguous (or near-contiguous) cluster nearest the center.
    cluster_start = min(dark_positions)
    cluster_end = max(dark_positions) + 1
    drop = (mean - min(series[cluster_start:cluster_end])) / max(mean, 1.0)
    if drop < min_confidence:
        return None

    return BindingBand(axis=axis, start=cluster_start, end=cluster_end, confidence=drop)


# ---------- Splitting ----------


def split_spread(
    img: Image.Image,
    band: BindingBand,
    outer_margin: int = 0,
    inner_trim: int = 0,
) -> tuple[Image.Image, Image.Image]:
    """Split *img* along *band* into (first_page, second_page).

    For axis='x' (vertical binding) → (left_page, right_page).
    For axis='y' (horizontal binding) → (top_page, bottom_page).

    *outer_margin* trims that many pixels off the outer (non-binding) edge of
    each page (useful for cropping page edges / colored corner triangles).
    *inner_trim* trims that many extra pixels off the binding edge to drop
    ring shadows.
    """
    w, h = img.size
    if band.axis == "x":
        first = img.crop((outer_margin, 0, band.start - inner_trim, h))
        second = img.crop((band.end + inner_trim, 0, w - outer_margin, h))
    else:
        first = img.crop((0, outer_margin, w, band.start - inner_trim))
        second = img.crop((0, band.end + inner_trim, w, h - outer_margin))
    return first, second


# ---------- High-level orchestrator ----------


def _render_pdf_pages(pdf_path: Path, dpi: int) -> list[Image.Image]:
    """Render every page of *pdf_path* to a Pillow Image at *dpi*."""
    doc = fitz.open(str(pdf_path))
    out: list[Image.Image] = []
    try:
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        for page in doc:
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            out.append(Image.open(io.BytesIO(pix.tobytes("png"))).copy())
    finally:
        doc.close()
    return out


def split_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int = 200,
    rotate: bool = True,
    detect_spreads: bool = True,
    outer_margin: int = 0,
    inner_trim: int = 0,
    image_format: str = "jpeg",
    image_quality: int = 88,
) -> list[Path]:
    """Render *pdf_path* to upright single-page images in *out_dir*.

    For each PDF page:
      1. Render at *dpi*.
      2. If *rotate*, auto-detect rotation and re-orient.
      3. If *detect_spreads*, attempt to detect a binding band and split
         into two pages; otherwise emit the page as-is.

    Returns a list of output image paths in reading order.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = "jpg" if image_format.lower() in ("jpeg", "jpg") else image_format.lower()
    save_kwargs: dict[str, object] = {}
    if image_format.lower() in ("jpeg", "jpg"):
        save_kwargs = {"quality": image_quality, "optimize": True}

    rendered: list[Image.Image] = _render_pdf_pages(pdf_path, dpi=dpi)
    pages: list[Image.Image] = []
    for img in rendered:
        if rotate:
            deg = detect_rotation(img)
            if deg:
                img = img.rotate(-deg, expand=True)
        if detect_spreads:
            band = detect_binding_band(img, axis="auto")
        else:
            band = None
        if band is not None:
            a, b = split_spread(img, band, outer_margin=outer_margin, inner_trim=inner_trim)
            pages.extend([a, b])
        else:
            pages.append(img)

    paths: list[Path] = []
    width = max(3, len(str(len(pages))))
    for i, p in enumerate(pages, 1):
        out_path = out_dir / f"p{i:0{width}d}.{ext}"
        p.save(out_path, **save_kwargs)
        paths.append(out_path)
    return paths
