"""Page-local extraction with durable checkpoints and inspectable decisions."""

from __future__ import annotations

import io
import json
import statistics
import time
from pathlib import Path

import fitz

from ..io_utils import atomic_write_bytes
from ..languages import TESSERACT, detect_language, direction, text_quality
from .project import create, digest, load, locked, mutate, now, project_path


def _rect(box, page) -> list[float]:
    return [round(float(x), 2) for x in fitz.Rect(box) * page.rotation_matrix]


def _union(boxes: list[list]) -> list[float]:
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _overlap(a, b) -> float:
    ar, br = fitz.Rect(a), fitz.Rect(b)
    return (ar & br).get_area() / max(1, min(ar.get_area(), br.get_area()))


def _native(page) -> list[dict]:
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_PRESERVE_LIGATURES)[
        "blocks"
    ]
    sizes = [
        span["size"]
        for b in blocks
        if b["type"] == 0
        for line in b["lines"]
        for span in line["spans"]
        if span["text"].strip()
    ]
    body = statistics.median(sizes) if sizes else 12
    out = []
    for block in blocks:
        if block["type"] != 0:
            continue
        lines = block["lines"]
        text = "\n".join("".join(s["text"] for s in line["spans"]) for line in lines).strip()
        if not text:
            continue
        size = max((s["size"] for line in lines for s in line["spans"]), default=body)
        kind = "heading" if size >= body * 1.2 and len(text) < 220 else "paragraph"
        if size < body * 0.85 and block["bbox"][1] > page.mediabox.height * 0.7 and len(text) > 6:
            kind = "footnote"
        out.append(
            {
                "text": text,
                "bbox": _rect(block["bbox"], page),
                "kind": kind,
                "level": 1 if size >= body * 1.6 else 2,
                "confidence": None,
            }
        )
    return out


def _reading_order(nodes: list[dict], width: float, language: str) -> list[dict]:
    """Conservative two-column detection, preserving full-width headings."""
    if len(nodes) < 4:
        return sorted(nodes, key=lambda n: (n["bbox"][1], n["bbox"][0]))
    left = [n for n in nodes if n["bbox"][2] < width * 0.54]
    right = [n for n in nodes if n["bbox"][0] > width * 0.46]
    middle = [n for n in nodes if n not in left and n not in right]
    if len(left) >= 2 and len(right) >= 2:
        # Full-width content partitions the page into independent reading bands.
        result = []
        remaining = list(nodes)
        for divider in sorted(middle, key=lambda n: n["bbox"][1]) + [None]:
            boundary = divider["bbox"][1] if divider else float("inf")
            band = [n for n in remaining if n is not divider and n["bbox"][1] < boundary]
            columns = [right, left] if direction(language) == "rtl" else [left, right]
            for column in columns:
                result.extend(sorted([n for n in band if n in column], key=lambda n: n["bbox"][1]))
            result.extend(n for n in band if n not in result)
            if divider:
                result.append(divider)
            remaining = [n for n in remaining if n not in result]
        return result
    return sorted(nodes, key=lambda n: (n["bbox"][1], n["bbox"][0]))


def _tesseract(image, languages: list[str], *, psm: int = 3) -> list[dict]:
    import pytesseract

    installed = set(pytesseract.get_languages(config=""))
    codes = list(
        dict.fromkeys(
            TESSERACT.get(tag.lower(), TESSERACT.get(tag.split("-")[0], tag))
            for tag in languages
            if tag != "und"
        )
    )
    if not codes:
        # Without a text layer, detect among installed likely book languages.
        codes = [code for code in ("fas", "ara", "eng") if code in installed]
    missing = set(codes) - installed
    if missing:
        raise RuntimeError("Install Tesseract language data: " + ", ".join(sorted(missing)))
    if not codes:
        raise RuntimeError("No OCR language data installed; select a language when importing")
    rtl_codes = {"ara", "fas", "urd", "heb", "pus", "snd"}
    if any(c in rtl_codes for c in codes) and "eng" in installed and "eng" not in codes:
        codes.append("eng")
    passes = [[c for c in codes if c in rtl_codes], [c for c in codes if c not in rtl_codes]]
    result = []
    for selected in passes:
        if not selected:
            continue
        raw = pytesseract.image_to_data(
            image,
            lang="+".join(selected),
            config=f"--psm {psm}",
            output_type=pytesseract.Output.DICT,
            timeout=90,
        )
        groups: dict = {}
        for i, value in enumerate(raw["text"]):
            if not value.strip() or float(raw["conf"][i]) < 0:
                continue
            key = (raw["block_num"][i], raw["par_num"][i])
            group = groups.setdefault(key, {"words": [], "boxes": [], "scores": [], "line": None})
            if group["line"] is not None and group["line"] != raw["line_num"][i]:
                group["words"].append("\n")
            group["line"] = raw["line_num"][i]
            group["words"].append(value)
            x, y, w, h = (raw[k][i] for k in ("left", "top", "width", "height"))
            group["boxes"].append([x, y, x + w, y + h])
            group["scores"].append(float(raw["conf"][i]) / 100)
        for group in groups.values():
            text = " ".join(group["words"]).replace(" \n ", "\n")
            result.append(
                {
                    "text": text,
                    "bbox": _union(group["boxes"]),
                    "kind": "paragraph",
                    "confidence": round(statistics.mean(group["scores"]), 3),
                }
            )
    # Different direction passes may see the same physical text. Prefer the
    # stronger candidate at that location; retain both when they do not overlap.
    kept: list[dict] = []
    for item in sorted(result, key=lambda n: n["confidence"], reverse=True):
        if not any(_overlap(item["bbox"], other["bbox"]) > 0.65 for other in kept):
            kept.append(item)
    return kept


def _uncovered_regions(image, nodes: list[dict], width: float, height: float) -> list[tuple]:
    """Locate printed ink outside confident OCR regions (headings, last lines).

    A high mean OCR confidence cannot reveal text that was never recognized.
    This independent image check triggers a bounded set of small crop retries.
    """
    from PIL import ImageDraw

    mask = image.convert("L").point(lambda value: 255 if value < 190 else 0)
    drawing = ImageDraw.Draw(mask)
    for node in nodes:
        if node.get("confidence", 0) < 0.65:
            continue
        box = node["bbox"]
        drawing.rectangle(
            (
                max(0, box[0] / width * image.width - 3),
                max(0, box[1] / height * image.height - 3),
                box[2] / width * image.width + 3,
                box[3] / height * image.height + 3,
            ),
            fill=0,
        )
    # Row projection separates isolated headings from body continuations.
    projection = mask.resize((1, image.height))
    active = [i for i, value in enumerate(projection.getdata()) if value >= 1]
    bands: list[list[int]] = []
    for row in active:
        if not bands or row - bands[-1][-1] > 10:
            bands.append([])
        bands[-1].append(row)
    result = []
    for band in bands:
        if len(band) < 3:
            continue
        box = mask.crop((0, band[0], image.width, band[-1] + 1)).getbbox()
        if not box or box[2] - box[0] < 20:
            continue
        result.append(
            (
                max(0, box[0] - 12),
                max(0, band[0] - 12),
                min(image.width, box[2] + 12),
                min(image.height, band[-1] + 13),
            )
        )
    return result[:8]


def _vision(path: Path, languages: list[str], width: float, height: float) -> list[dict]:
    from ..extract.ocr import vision_pass

    items = vision_pass(str(path), [x for x in languages if x != "und"] or ["en-US"])
    return [
        {
            "text": item["text"],
            "confidence": item["conf"],
            "kind": "paragraph",
            "bbox": [
                item["bbox_norm"][0] * width,
                (1 - item["bbox_norm"][1] - item["bbox_norm"][3]) * height,
                (item["bbox_norm"][0] + item["bbox_norm"][2]) * width,
                (1 - item["bbox_norm"][1]) * height,
            ],
        }
        for item in items
    ]


def _score(nodes: list[dict]) -> float:
    text = "\n".join(n["text"] for n in nodes)
    if not text:
        return 0
    scores = [n["confidence"] for n in nodes if n.get("confidence") is not None]
    return text_quality(text) * 0.4 + (statistics.mean(scores) if scores else 0.7) * 0.6


def extract_page(
    page, root: Path, *, language: str = "auto", engine: str = "auto", dpi: int = 180
) -> tuple[dict, list[dict]]:
    from PIL import Image, ImageOps, ImageStat

    pno = page.number + 1
    image_path = f"pages/{pno:04d}.png"
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    atomic_write_bytes(root / image_path, pix.tobytes("png"))
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    native = _native(page)
    native_text = "\n".join(n["text"] for n in native)
    detection = detect_language(native_text, hint=language)
    reliable = len([c for c in native_text if c.isalpha()]) >= 25 and text_quality(native_text) >= 0.65
    record = {
        "number": pno,
        "width": page.rect.width,
        "height": page.rect.height,
        "image": image_path,
        "language": detection,
        "state": "complete",
        "reviewed": False,
        "warnings": [],
        "attempts": [],
    }
    if engine == "text" or (engine == "auto" and reliable):
        nodes, chosen = native, "text"
        if not reliable:
            record["warnings"].append("Sparse or damaged text layer; compare the scan")
    else:
        thumb = image.convert("L").resize((128, 128))
        if (
            not native_text.strip()
            and ImageStat.Stat(thumb).mean[0] > 253
            and ImageStat.Stat(thumb).stddev[0] < 4
        ):
            record.update(engine="blank", disposition="blank")
            return record, []
        languages = [detection["language"]]
        if detection["scripts"].get("LATIN", 0) > 20 and detection["language"] != "en":
            languages.append("en")
        chosen = "tesseract" if engine == "auto" else engine
        start = time.monotonic()
        if chosen == "vision":
            nodes = _vision(root / image_path, languages, page.rect.width, page.rect.height)
        else:
            nodes = _tesseract(image, languages)
            for n in nodes:
                n["bbox"] = [
                    v * (page.rect.width / image.width if i % 2 == 0 else page.rect.height / image.height)
                    for i, v in enumerate(n["bbox"])
                ]
        record["attempts"].append(
            {
                "engine": chosen,
                "health": round(_score(nodes), 3),
                "seconds": round(time.monotonic() - start, 3),
            }
        )
        if _score(nodes) < 0.72:
            try:
                improved = _tesseract(ImageOps.autocontrast(image.convert("L")), languages, psm=6)
                for n in improved:
                    n["bbox"] = [
                        v * (page.rect.width / image.width if i % 2 == 0 else page.rect.height / image.height)
                        for i, v in enumerate(n["bbox"])
                    ]
                record["attempts"].append(
                    {"engine": "tesseract-contrast-psm6", "health": round(_score(improved), 3)}
                )
                if _score(improved) > _score(nodes):
                    nodes, chosen = improved, "tesseract-contrast-psm6"
            except Exception as exc:
                record["warnings"].append(f"Fallback unavailable: {type(exc).__name__}")
        if not nodes:
            raise RuntimeError("No text recognized on a nonblank page; select another language or engine")
        regions = _uncovered_regions(image, nodes, page.rect.width, page.rect.height)
        for crop in regions:
            try:
                recovered = _tesseract(image.crop(crop), languages, psm=6)
                if not recovered and crop[3] - crop[1] < 120:
                    recovered = _tesseract(image.crop(crop), languages, psm=13)
                accepted = 0
                for candidate in recovered:
                    b = candidate["bbox"]
                    candidate["bbox"] = [
                        (b[0] + crop[0]) * page.rect.width / image.width,
                        (b[1] + crop[1]) * page.rect.height / image.height,
                        (b[2] + crop[0]) * page.rect.width / image.width,
                        (b[3] + crop[1]) * page.rect.height / image.height,
                    ]
                    overlaps = [n for n in nodes if _overlap(n["bbox"], candidate["bbox"]) > 0.65]
                    if candidate["confidence"] >= 0.55 and all(
                        candidate["confidence"] > n.get("confidence", 0) for n in overlaps
                    ):
                        nodes = [n for n in nodes if n not in overlaps] + [candidate]
                        accepted += 1
                record["attempts"].append(
                    {"engine": "tesseract-region", "region": list(crop), "recovered": accepted}
                )
            except Exception:
                record["warnings"].append("A source region still needs manual review")
        detection = detect_language("\n".join(n["text"] for n in nodes), hint=language)
        record["language"] = detection
    record["engine"] = chosen
    if detection["confidence"] < 0.75:
        record["warnings"].append("Language needs confirmation")
    # Preserve native figures. Full-page scan bitmaps are already the proof
    # image; they must not become duplicate illustrations in the edition.
    if chosen == "text":
        tables = []
        if hasattr(page, "find_tables"):
            try:
                tables = list(page.find_tables().tables)
            except Exception:
                record["warnings"].append("Automatic table detection unavailable")
        for table in tables:
            bbox = _rect(table.bbox, page)
            cells = [[str(c or "") for c in row] for row in table.extract()]
            nodes = [n for n in nodes if _overlap(n["bbox"], bbox) < 0.7]
            nodes.append(
                {
                    "text": "\n".join(" | ".join(r) for r in cells),
                    "cells": cells,
                    "bbox": bbox,
                    "kind": "table",
                    "confidence": None,
                }
            )
        for index, info in enumerate(page.get_image_info()):
            bbox = _rect(info["bbox"], page)
            if fitz.Rect(bbox).get_area() > page.rect.get_area() * 0.8:
                continue
            figure_crop = (
                bbox[0] / page.rect.width * image.width,
                bbox[1] / page.rect.height * image.height,
                bbox[2] / page.rect.width * image.width,
                bbox[3] / page.rect.height * image.height,
            )
            if figure_crop[2] <= figure_crop[0] or figure_crop[3] <= figure_crop[1]:
                continue
            asset = f"assets/p{pno:04d}-figure-{index + 1}.png"
            (root / "assets").mkdir(exist_ok=True)
            image.crop(figure_crop).save(root / asset)
            nodes.append(
                {"text": "", "kind": "figure", "bbox": bbox, "asset": asset, "alt": "", "confidence": None}
            )
    nodes = _reading_order(nodes, page.rect.width, detection["language"])
    for index, node in enumerate(nodes):
        original = node["text"]
        if node["kind"] == "paragraph":
            node["text"] = " ".join(node["text"].split())
        local_language = detect_language(original)
        node_language = (
            local_language["language"] if local_language["confidence"] >= 0.75 else detection["language"]
        )
        node.update(
            id=f"p{pno:04d}-b{index + 1:04d}",
            page=pno,
            order=index,
            original=original,
            status="pending",
            translations={},
            note_ids=[],
            language=node_language,
            review_seconds=0,
        )
        node.setdefault("level", 2)
        node["warnings"] = []
        if node.get("confidence") is not None and node["confidence"] < 0.8:
            node["warnings"].append("Low recognition confidence")
        if node["text"] and text_quality(node["text"]) < 0.6:
            node["warnings"].append("Unusual text; check against scan")
        if node["kind"] == "figure":
            node["warnings"].append("Describe the illustration")
    return record, nodes


def import_pdf(
    source: str | Path,
    root: str | Path,
    *,
    language: str = "auto",
    engine: str = "auto",
    title: str | None = None,
    author: str | None = None,
    dpi: int = 180,
    progress=None,
) -> dict:
    with locked(Path(root).expanduser().resolve(), ".extract.lock"):
        return _import_pdf(
            source,
            root,
            language=language,
            engine=engine,
            title=title,
            author=author,
            dpi=dpi,
            progress=progress,
        )


def _import_pdf(
    source: str | Path,
    root: str | Path,
    *,
    language: str = "auto",
    engine: str = "auto",
    title: str | None = None,
    author: str | None = None,
    dpi: int = 180,
    progress=None,
) -> dict:
    source, root = Path(source).expanduser().resolve(), Path(root).expanduser().resolve()
    if engine not in {"auto", "text", "tesseract", "vision"}:
        raise ValueError("Choose auto, text, tesseract, or vision extraction")
    if not 72 <= dpi <= 600:
        raise ValueError("DPI must be between 72 and 600")
    with fitz.open(source) as doc:
        if doc.needs_pass:
            raise ValueError("Unlock the PDF before importing")
        samples = "\n".join(doc[i].get_text() for i in range(min(12, len(doc))))
        detection = detect_language(samples, hint=language)
        data = create(
            source,
            root,
            title=title or doc.metadata.get("title") or source.stem,
            author=author or doc.metadata.get("author") or "",
            language=detection,
            pages=len(doc),
        )
    settings = {"language": language, "engine": engine, "dpi": dpi, "extractor_version": 1}
    previous = data["jobs"].get("extract", {})
    if previous.get("settings") not in (None, settings) and all(
        p["state"] == "complete" for p in data["pages"]
    ):
        raise ValueError("Every page is already extracted; use a new project to compare extraction settings")
    mutate(
        root,
        None,
        lambda d: d["jobs"].update(extract={"state": "running", "settings": settings, "started_at": now()}),
    )
    # Save after each page. A killed process resumes at the first uncommitted
    # page, and failed pages can be retried without touching successful ones.
    with fitz.open(project_path(root, data["source"]["path"])) as doc:
        done = {p["number"] for p in data["pages"] if p["state"] == "complete"}
        for page in doc:
            pno = page.number + 1
            if pno in done:
                continue
            start = time.monotonic()
            try:
                record, nodes = extract_page(page, root, language=language, engine=engine, dpi=dpi)
            except Exception as exc:
                record, nodes = (
                    {
                        "number": pno,
                        "state": "failed",
                        "error": str(exc)[:400],
                        "width": page.rect.width,
                        "height": page.rect.height,
                        "image": f"pages/{pno:04d}.png",
                        "reviewed": False,
                        "warnings": [],
                    },
                    [],
                )
            record["seconds"] = round(time.monotonic() - start, 3)

            def checkpoint(d, record=record, nodes=nodes, pno=pno):
                d["pages"] = sorted(
                    [p for p in d["pages"] if p["number"] != pno] + [record], key=lambda p: p["number"]
                )
                d["nodes"] = sorted(
                    [n for n in d["nodes"] if n["page"] != pno] + nodes, key=lambda n: (n["page"], n["order"])
                )

            mutate(root, None, checkpoint)
            if progress:
                progress(pno, len(doc), record)

    def finish(d):
        failed = sum(p["state"] == "failed" for p in d["pages"])
        d["jobs"]["extract"].update(
            state="partial" if failed else "complete", failed=failed, finished_at=now()
        )
        if language == "auto":
            d["language"] = detect_language("\n".join(n["text"] for n in d["nodes"][:100]))

    return mutate(root, None, finish)[0]


def convert_docling(root: Path) -> dict:
    """Optional layout engine; models may download on the first explicit run."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TesseractCliOcrOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError:
        raise RuntimeError("Install Quire's layout extra: pip install 'quire[layout]'") from None
    data = load(root, verify_source=True)
    if data["history"]:
        raise ValueError("Run layout analysis before review, or use a separate comparison project")
    options = PdfPipelineOptions()
    language = data["language"]["language"].split("-")[0]
    if language in TESSERACT:
        options.ocr_options = TesseractCliOcrOptions(lang=list(dict.fromkeys([TESSERACT[language], "eng"])))
    converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
    result = converter.convert(str(project_path(root, data["source"]["path"])))
    if str(result.status).lower().split(".")[-1] != "success":
        raise RuntimeError("Docling did not complete every page; the existing manuscript was preserved")
    document = result.document.export_to_dict()
    document_path = root / "layout/docling.json"
    atomic_write_bytes(document_path, json.dumps(document, ensure_ascii=False).encode())
    return import_docling(root, document_path, revision=data["revision"])


def import_docling(root: Path, document_path: Path, *, revision: int) -> dict:
    """Import a Docling JSON export as a reviewed layout proposal.

    Nothing is silently applied: only projects without human edits can replace
    their initial segmentation. Coordinates are transformed from Docling's
    explicit origin; raw JSON is preserved for future comparison.
    """
    raw = json.loads(document_path.read_text("utf-8"))
    data = load(root)
    if data["history"] or any(n["status"] != "pending" for n in data["nodes"]):
        raise ValueError("Import layout before review, or create a separate comparison project")
    labels = {
        "section_header": "heading",
        "title": "heading",
        "text": "paragraph",
        "list_item": "paragraph",
        "footnote": "footnote",
        "caption": "caption",
        "picture": "figure",
        "table": "table",
    }
    passages: list[dict] = []

    def walk(ref):
        item = raw
        for key in ref.lstrip("#/").split("/"):
            item = item[int(key)] if isinstance(item, list) else item[key]
        if item.get("label") in labels and item.get("prov"):
            for prov_index, prov in enumerate(item["prov"]):
                pno = int(prov["page_no"])
                page = next(p for p in data["pages"] if p["number"] == pno)
                box = prov["bbox"]
                y0, y1 = box["t"], box["b"]
                if box.get("coord_origin", "TOPLEFT") == "BOTTOMLEFT":
                    y0, y1 = page["height"] - y0, page["height"] - y1
                text = item.get("text", "")
                if prov.get("charspan"):
                    start, end = prov["charspan"]
                    text = text[start:end]
                node = {
                    "id": "d-" + digest([ref, prov_index])[:16],
                    "page": pno,
                    "order": len(passages),
                    "bbox": [box["l"], min(y0, y1), box["r"], max(y0, y1)],
                    "kind": labels[item["label"]],
                    "text": text,
                    "original": text,
                    "level": 2,
                    "language": data["language"]["language"],
                    "status": "pending",
                    "translations": {},
                    "note_ids": [],
                    "confidence": None,
                    "warnings": ["Imported layout needs source review"],
                    "review_seconds": 0,
                }
                if node["kind"] == "table":
                    table = item.get("data", {})
                    cells = [[""] * table.get("num_cols", 0) for _ in range(table.get("num_rows", 0))]
                    for cell in table.get("table_cells", []):
                        cells[cell["start_row_offset_idx"]][cell["start_col_offset_idx"]] = cell.get(
                            "text", ""
                        )
                    node["cells"] = cells
                    node["text"] = "\n".join(" | ".join(row) for row in cells)
                    node["original"] = node["text"]
                if node["kind"] == "figure":
                    from PIL import Image

                    with Image.open(root / page["image"]) as image:
                        b = node["bbox"]
                        crop = [
                            b[0] * image.width / page["width"],
                            b[1] * image.height / page["height"],
                            b[2] * image.width / page["width"],
                            b[3] * image.height / page["height"],
                        ]
                        node["asset"] = f"assets/{node['id']}.png"
                        (root / "assets").mkdir(exist_ok=True)
                        image.crop(tuple(crop)).save(root / node["asset"])
                    node["alt"] = ""
                passages.append(node)
        for child in item.get("children", []):
            walk(child["$ref"])

    walk("#/body")
    if not passages:
        raise ValueError("No supported passages in the Docling document")
    atomic_write_bytes(root / "layout/docling.json", document_path.read_bytes())

    def apply(d):
        if d["history"]:
            raise ValueError("Project acquired edits while importing layout")
        d["nodes"] = sorted(passages, key=lambda n: (n["page"], n["order"]))
        d["jobs"]["layout"] = {"engine": "docling", "state": "complete", "at": now()}

    return mutate(root, revision, apply)[0]
