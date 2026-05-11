"""macOS Vision Framework OCR.

Apple's Vision framework recognises Arabic with diacritics far better than
Tesseract. We render each PDF page to PNG and run Vision twice:
  - first-pass languages (e.g. ``["en-US", "ar-SA"]``) for the body text
  - second-pass languages flipped, biasing for the secondary script

Both passes are merged. Arabic-script lines are clustered into ``arabic_blocks``
by vertical proximity. Bboxes are returned in PDF points (top-left origin).

The ``ocr_all`` entry point is language-agnostic — pass ``languages=[...]``
to control the Vision recognition language preference.
"""

from __future__ import annotations

import concurrent.futures as cf
import os
import pickle
import re
import statistics
import tempfile

import fitz

# Lazy import of ``ocrmac`` — see ``quire/extract/refine.py`` for the
# rationale. Linux / Windows installs without ocrmac can still import
# this module and use the Tesseract or text engines.
try:
    from ocrmac import ocrmac  # type: ignore[import-not-found]
except ImportError:
    ocrmac = None  # type: ignore[assignment]

ARABIC_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")


def has_arabic(text: str) -> bool:
    return bool(ARABIC_RE.search(text))


def _bbox_to_pdf(bbox, img_w_pt: float, img_h_pt: float) -> tuple[float, float, float, float]:
    """Vision returns bbox as (x, y, w, h) normalised to image size, with the
    origin at the BOTTOM-LEFT of the image (Cocoa coords). Convert to top-left
    origin in PDF points using the page's actual point size.
    """
    x_n, y_n, w_n, h_n = bbox
    x0 = x_n * img_w_pt
    y0_top = (1 - y_n - h_n) * img_h_pt
    x1 = x0 + w_n * img_w_pt
    y1 = y0_top + h_n * img_h_pt
    return x0, y0_top, x1, y1


def vision_pass(image_path: str, langs: list[str]) -> list[dict]:
    if ocrmac is None:
        raise ImportError(
            "The Vision OCR engine requires the optional 'ocrmac' package, "
            "which is macOS-only. Install with `pip install quire[vision]` "
            "on macOS, or set `engine = \"tesseract\"` / `engine = \"text\"` "
            "in your book.toml."
        )
    result = ocrmac.OCR(
        image_path,
        recognition_level="accurate",
        language_preference=langs,
    ).recognize()
    out = []
    for txt, conf, bbox in result:
        out.append({"text": txt, "conf": float(conf), "bbox_norm": list(bbox)})
    return out


def _vision_with_retry(image_path: str, langs: list[str], *, retries: int) -> list[dict]:
    """Run :func:`vision_pass`, retrying on transient failures.

    Vision OCR sometimes returns an empty result for valid pages under load.
    A single retry is usually enough to recover; the caller can configure
    more aggressive retry budgets through ``ocr.retries`` in ``book.toml``.
    """
    last_exc: Exception | None = None
    for attempt in range(max(1, retries + 1)):
        try:
            return vision_pass(image_path, langs)
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt >= retries:
                raise
    if last_exc is not None:
        raise last_exc
    return []


def cluster_arabic_blocks(arabic_lines: list[dict], avg_h: float) -> list[dict]:
    if not arabic_lines:
        return []
    arabic_lines.sort(key=lambda L: L["y0"])
    blocks: list[dict] = []
    current: list[dict] = []
    for L in arabic_lines:
        if not current:
            current = [L]
            continue
        gap = L["y0"] - current[-1]["y1"]
        if gap <= avg_h * 1.5:
            current.append(L)
        else:
            blocks.append(_finalize_block(current))
            current = [L]
    if current:
        blocks.append(_finalize_block(current))
    return blocks


def _finalize_block(lines: list[dict]) -> dict:
    text = "\n".join(L["text"] for L in lines)
    confs = [L["conf"] for L in lines if L["conf"] >= 0]
    return {
        "lines": lines,
        "x0": min(L["x0"] for L in lines),
        "x1": max(L["x1"] for L in lines),
        "y0": min(L["y0"] for L in lines),
        "y1": max(L["y1"] for L in lines),
        "text": text,
        "conf": statistics.mean(confs) if confs else -1.0,
    }


def ocr_page(
    pdf_path: str,
    pno: int,
    *,
    scale: int = 4,
    languages: list[str] | None = None,
    retries: int = 1,
) -> dict:
    doc = fitz.open(pdf_path)
    try:
        page = doc[pno - 1]
        w_pt, h_pt = page.rect.width, page.rect.height
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    finally:
        doc.close()
    langs = list(languages or ["en-US", "ar-SA"])
    if len(langs) >= 2:
        primary, secondary = langs, [langs[1], langs[0]] + langs[2:]
    else:
        primary, secondary = langs, langs
    with tempfile.NamedTemporaryFile(prefix=f"quire_ocr_p{pno:03d}_", suffix=".png",
                                     delete=False) as tmp:
        image_path = tmp.name
    pix.save(image_path)
    try:
        en_pass = _vision_with_retry(image_path, primary, retries=retries)
        ar_pass = _vision_with_retry(image_path, secondary, retries=retries)
    finally:
        try:
            os.unlink(image_path)
        except FileNotFoundError:
            pass

    # Convert bboxes to PDF points
    def _to_lines(items: list[dict]) -> list[dict]:
        out = []
        for it in items:
            x0, y0, x1, y1 = _bbox_to_pdf(it["bbox_norm"], w_pt, h_pt)
            out.append(
                {
                    "text": it["text"],
                    "conf": it["conf"],
                    "x0": x0,
                    "y0": y0,
                    "x1": x1,
                    "y1": y1,
                }
            )
        return out

    en_lines = _to_lines(en_pass)
    ar_lines = _to_lines(ar_pass)

    # Arabic blocks: lines from the AR pass that contain Arabic glyphs.
    arabic_only = [L for L in ar_lines if has_arabic(L["text"])]
    heights = [L["y1"] - L["y0"] for L in arabic_only if L["y1"] > L["y0"]]
    avg_h = statistics.median(heights) if heights else 14.0
    arabic_blocks = cluster_arabic_blocks(arabic_only, avg_h)

    return {
        "pno": pno,
        "en_lines": en_lines,
        "ar_lines": ar_lines,
        "arabic_blocks": arabic_blocks,
        "page_size_pt": (w_pt, h_pt),
    }


def ocr_all(
    pdf_path: str,
    *,
    workers: int = 4,
    progress=None,
    languages: list[str] | None = None,
    scale: int = 4,
    retries: int = 1,
    page_numbers: list[int] | None = None,
) -> list[dict | None]:
    """Run Vision OCR over a PDF.

    If ``page_numbers`` is provided, only those pages (1-indexed) are OCR'd;
    the returned list still has length ``page_count`` with placeholders for
    untouched pages so callers can splice the result into an existing cache.
    """
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
    finally:
        doc.close()
    pages: list[dict | None] = [None] * n
    workers = max(1, int(workers))
    pno_list = list(page_numbers) if page_numbers is not None else list(range(1, n + 1))
    pno_list = [p for p in pno_list if 1 <= p <= n]
    if not pno_list:
        return pages
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(
                ocr_page,
                pdf_path,
                pno,
                languages=languages,
                scale=scale,
                retries=retries,
            ): pno
            for pno in pno_list
        }
        done = 0
        for fut in cf.as_completed(futs):
            pno = futs[fut]
            try:
                pages[pno - 1] = fut.result()
            except Exception as e:  # noqa: BLE001
                pages[pno - 1] = {
                    "pno": pno,
                    "en_lines": [],
                    "ar_lines": [],
                    "arabic_blocks": [],
                    "page_size_pt": (612, 792),
                    "error": str(e),
                }
            done += 1
            if progress:
                progress(done, len(pno_list))
    return pages


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("pdf")
    p.add_argument("--out", default="vision_cache.pkl")
    p.add_argument("--languages", nargs="+", default=["en-US", "ar-SA"])
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    def _progress(done: int, total: int) -> None:
        print(f"  {done}/{total}", end="\r")

    pages = ocr_all(args.pdf, workers=args.workers, progress=_progress,
                    languages=args.languages)
    print()
    with open(args.out, "wb") as f:
        pickle.dump(pages, f)
    print(f"wrote {args.out}: {len(pages)} pages")
