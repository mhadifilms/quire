"""Render a single PDF page to a cached PNG suitable for VLM input.

The image cache lives under ``cfg.caches_dir / "qc_images"`` and is
keyed by ``(pdf_fingerprint_short, pdf_pno, dpi)`` so different builds
of the same book at the same DPI share the cache.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz

from ..config import BookConfig
from ..io_utils import atomic_write_bytes, content_fingerprint


def _cache_root(cfg: BookConfig) -> Path:
    return cfg.caches_dir / "qc_images"


def _cache_path(cfg: BookConfig, pdf_pno: int, dpi: int, fp: str) -> Path:
    return _cache_root(cfg) / f"{fp[:16]}_p{pdf_pno:04d}_dpi{dpi}.png"


def render_page_png(
    cfg: BookConfig,
    pdf_pno: int,
    *,
    dpi: int,
    doc: fitz.Document | None = None,
    fingerprint: str | None = None,
) -> bytes:
    """Return PNG bytes for ``pdf_pno`` (1-based). Cached on disk."""
    fp = fingerprint or content_fingerprint(cfg.pdf_path)
    cache = _cache_path(cfg, pdf_pno, dpi, fp)
    if cache.exists():
        return cache.read_bytes()
    owned = False
    if doc is None:
        doc = fitz.open(str(cfg.pdf_path))
        owned = True
    try:
        if pdf_pno < 1 or pdf_pno > len(doc):
            raise IndexError(f"pdf_pno {pdf_pno} out of range (1..{len(doc)})")
        pix = doc[pdf_pno - 1].get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
    finally:
        if owned:
            doc.close()
    atomic_write_bytes(cache, png_bytes)
    return png_bytes


def render_pages_for_qc(
    cfg: BookConfig,
    pdf_pnos: list[int],
    *,
    dpi: int,
) -> dict[int, bytes]:
    """Batch-render PNG bytes for many pdf pages. Shares a single fitz doc."""
    fp = content_fingerprint(cfg.pdf_path)
    doc = fitz.open(str(cfg.pdf_path))
    try:
        out: dict[int, bytes] = {}
        for pno in pdf_pnos:
            out[pno] = render_page_png(cfg, pno, dpi=dpi, doc=doc, fingerprint=fp)
        return out
    finally:
        doc.close()


def page_image_bytes_io(png_bytes: bytes) -> io.BytesIO:
    return io.BytesIO(png_bytes)
