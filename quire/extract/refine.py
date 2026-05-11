"""Refine RTL-script blocks with a tight-crop + script-only OCR pass.

The page-level OCR uses mixed languages which still allows the secondary
language to leak into RTL regions, occasionally producing mojibake. By
cropping just the RTL-dominant regions and re-running with the RTL language
preference only, we get much cleaner recognition.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import pickle
import re
import statistics
import tempfile

import fitz
from PIL import Image

# ``ocrmac`` is the macOS Vision binding and only installs on macOS. Import
# lazily so that ``quire.pipeline`` and ``quire.render.audit`` (which both
# import this module) remain usable on Linux / Windows with the Tesseract
# or text-layer engines. The actual ``ocrmac.OCR`` call inside
# :func:`_vision_pass` raises ``ImportError`` only if a build with
# ``engine = "vision"`` is started on a host without ocrmac installed.
try:
    from ocrmac import ocrmac  # type: ignore[import-not-found]
except ImportError:
    ocrmac = None  # type: ignore[assignment]

ARABIC_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def arabic_dominant(text: str) -> bool:
    ar = sum(1 for c in text if ARABIC_RE.match(c))
    if ar < 2:
        return False
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    return ar >= latin * 0.8


def _vision_pass(image_path: str, langs: list[str], *, retries: int) -> list:
    """Run Vision OCR with retry on transient failures."""
    if ocrmac is None:
        raise ImportError(
            "The Vision OCR engine requires the optional 'ocrmac' package, "
            "which is macOS-only. Install with `pip install quire[vision]` "
            "on macOS, or set `engine = \"tesseract\"` / `engine = \"text\"` "
            "in your book.toml."
        )
    last_exc: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            return ocrmac.OCR(
                image_path,
                recognition_level="accurate",
                language_preference=list(langs),
            ).recognize()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt >= retries:
                raise
    if last_exc is not None:
        raise last_exc
    return []


def crop_and_ocr(image: Image.Image, bbox_pt: tuple[float, float, float, float],
                 page_w_pt: float, page_h_pt: float, scale: int,
                 languages: list[str], *, retries: int = 1) -> list[dict]:
    """Crop a region and OCR it with the given language preference.
    bbox_pt is (x0, y0, x1, y1) in PDF points (top-left origin).
    """
    x0_pt, y0_pt, x1_pt, y1_pt = bbox_pt
    pad = 8  # PDF points padding
    x0_pt = max(0, x0_pt - pad)
    y0_pt = max(0, y0_pt - pad)
    x1_pt = min(page_w_pt, x1_pt + pad)
    y1_pt = min(page_h_pt, y1_pt + pad)
    px0 = int(x0_pt * scale)
    py0 = int(y0_pt * scale)
    px1 = int(x1_pt * scale)
    py1 = int(y1_pt * scale)
    cropped = image.crop((px0, py0, px1, py1))
    with tempfile.NamedTemporaryFile(prefix="quire_refine_", suffix=".png",
                                     delete=False) as ftmp:
        tmp = ftmp.name
    cropped.save(tmp)
    try:
        result = _vision_pass(tmp, languages, retries=retries)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
    out: list[dict] = []
    crop_w = px1 - px0
    crop_h = py1 - py0
    for txt, conf, bbox_n in result:
        bx, by, bw, bh = bbox_n
        # Convert from crop-relative normalized to PDF-point bbox (top-left)
        # Cocoa origin (bottom-left) within the crop
        rel_x0 = bx
        rel_y0 = (1 - by - bh)
        rel_x1 = rel_x0 + bw
        rel_y1 = rel_y0 + bh
        local_x0 = px0 + rel_x0 * crop_w
        local_y0 = py0 + rel_y0 * crop_h
        local_x1 = px0 + rel_x1 * crop_w
        local_y1 = py0 + rel_y1 * crop_h
        out.append({
            "text": txt,
            "conf": float(conf),
            "x0": local_x0 / scale,
            "y0": local_y0 / scale,
            "x1": local_x1 / scale,
            "y1": local_y1 / scale,
        })
    return out


def refine_page(pdf_path: str, pno: int, page_data: dict,
                scale: int = 4, languages: list[str] | None = None,
                retries: int = 1) -> dict:
    """For each arabic_block on a page, re-OCR the region with the given languages."""
    blocks = page_data.get("arabic_blocks", [])
    if not blocks:
        return {"pno": pno, "refined_blocks": []}
    langs = list(languages or ["ar-SA"])
    doc = fitz.open(pdf_path)
    try:
        page = doc[pno - 1]
        w_pt, h_pt = page.rect.width, page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    finally:
        doc.close()
    with tempfile.NamedTemporaryFile(prefix=f"quire_refinepage_p{pno:03d}_",
                                     suffix=".png", delete=False) as ftmp:
        img_path = ftmp.name
    pix.save(img_path)
    img = Image.open(img_path)
    refined: list[dict] = []
    try:
        for blk in blocks:
            x0, y0, x1, y1 = blk["x0"], blk["y0"], blk["x1"], blk["y1"]
            new_lines = crop_and_ocr(
                img, (x0, y0, x1, y1), w_pt, h_pt, scale, langs, retries=retries,
            )
            kept = [L for L in new_lines if arabic_dominant(L["text"])]
            kept.sort(key=lambda L: L["y0"])
            text = "\n".join(L["text"] for L in kept)
            confs = [L["conf"] for L in kept]
            refined.append({
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "text": text,
                "conf": (statistics.mean(confs) * 100) if confs else 0.0,
                "lines": kept,
            })
    finally:
        img.close()
        try:
            os.unlink(img_path)
        except FileNotFoundError:
            pass
    return {"pno": pno, "refined_blocks": refined}


def refine_all(pdf_path: str, vision_pages: list[dict], workers: int = 4,
               progress=None, languages: list[str] | None = None,
               scale: int = 4, retries: int = 1) -> list[dict]:
    n = len(vision_pages)
    out: list[dict | None] = [None] * n
    workers = max(1, int(workers))
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(refine_page, pdf_path, p["pno"], p, scale, languages, retries): p["pno"]
            for p in vision_pages
        }
        done = 0
        for fut in cf.as_completed(futs):
            pno = futs[fut]
            try:
                out[pno - 1] = fut.result()
            except Exception as e:  # noqa: BLE001
                out[pno - 1] = {"pno": pno, "refined_blocks": [], "error": str(e)}
            done += 1
            if progress:
                progress(done, n)
    return out


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("vision_cache")
    p.add_argument("--out", required=True)
    p.add_argument("--languages", nargs="+", default=["ar-SA"])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    with open(args.vision_cache, "rb") as f:
        vision_pages = pickle.load(f)

    def _p(d, t):
        print(f"  refine {d}/{t}", end="\r")

    refined = refine_all(args.pdf, vision_pages, workers=args.workers,
                         progress=_p, languages=args.languages)
    print()
    with open(args.out, "wb") as f:
        pickle.dump(refined, f)
    print(f"wrote {args.out}")
