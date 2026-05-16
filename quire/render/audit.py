"""Content audit for the reflowable EPUB.

Produces ``books/<slug>/artifacts/audit.txt`` with:
  - English word coverage (EPUB body vs PDF text-layer)
  - Arabic character coverage (EPUB vs OCR)
  - Per-chapter sanity (non-empty body)
  - Cross-file link resolution
  - Optional ``corrections.md`` for low-confidence Arabic blocks
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections.abc import Iterable
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz

from ..config import BookConfig
from ..io_utils import atomic_write_text

ARABIC_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufdff\ufe70-\ufeff]")
WORD_RE = re.compile(r"[A-Za-z']+")
NS_X = "http://www.w3.org/1999/xhtml"


def _existing_ocr_review_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cols = line.split("\t")
        row = {h: cols[i] if i < len(cols) else "" for i, h in enumerate(header)}
        if row.get("kind", "").startswith("uncertain "):
            rows.append(row)
    return rows


def _load_xhtml_files(epub_path: Path) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    with zipfile.ZipFile(epub_path) as zf:
        for name in zf.namelist():
            if name.endswith(".xhtml"):
                out[name] = zf.read(name)
    return out


def _tree_text(elem: ET.Element) -> str:
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_tree_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _visible_non_footnote_text(elem: ET.Element) -> str:
    tag = elem.tag.split("}")[-1]
    epub_type = elem.attrib.get("{http://www.idpf.org/2007/ops}type", "")
    cls = elem.attrib.get("class", "")
    if tag == "aside" or "footnote" in epub_type or "footnotes" in cls:
        return ""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(_visible_non_footnote_text(child))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def _count_english_words(html_bytes: bytes) -> int:
    try:
        root = ET.fromstring(html_bytes)
    except ET.ParseError:
        return 0
    body = root.find(f"{{{NS_X}}}body")
    if body is None:
        return 0
    total = 0
    for p in body.iter():
        tag = p.tag.split("}")[-1]
        if tag in {"p", "h1", "h2", "h3", "h4", "li", "td"}:
            text = _tree_text(p)
            for w in WORD_RE.findall(text):
                if w.isalpha():
                    total += 1
    return total


def _count_arabic_chars(html_bytes: bytes) -> int:
    try:
        root = ET.fromstring(html_bytes)
    except ET.ParseError:
        return 0
    body = root.find(f"{{{NS_X}}}body")
    if body is None:
        return 0
    return sum(1 for c in _tree_text(body) if ARABIC_RE.match(c))


def _collect_ids_and_links(html_bytes: bytes):
    ids: set[str] = set()
    links: list[tuple[str, str]] = []
    try:
        root = ET.fromstring(html_bytes)
    except ET.ParseError:
        return ids, links
    for el in root.iter():
        i = el.attrib.get("id")
        if i:
            ids.add(i)
        href = el.attrib.get("href") or el.attrib.get(f"{{{NS_X}}}href")
        if href and href.startswith("#"):
            links.append((href[1:], "_self"))
        elif href and "#" in href:
            file_part, frag = href.split("#", 1)
            links.append((frag, file_part))
    return ids, links


SUSPICIOUS_TEXT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("question/exclamation-shaped footnote marker", re.compile(r"\b[A-Za-z][A-Za-z\-]*(?:\))?(?:\.{1,3}|[,;:!])[?!](?:\s|$)|\b[A-Za-z][A-Za-z\-]*\?(?=\s+[a-z])")),
    ("common OCR typo: che/bur/chis/dury", re.compile(r"\b(?:che|bur|chis|dury)\b", re.I)),
    ("cursive transliteration OCR variant", re.compile(r"\b(?:migat|migar|migas|mubrim|ibram|tagsir)\b", re.I)),
    ("short inline-Arabic mojibake token", re.compile(r"\b(?:ug|jes|f'es|jo|jé)\b")),
    # ``word'`` directly after a letter (or after ``,``/``;``/``:``) is
    # almost always a misread superscript footnote marker. ``word.'``,
    # ``word?'``, ``word!'`` — closing single quote following sentence-end
    # punctuation — is legitimate dialogue/quotation and not a misread,
    # so we explicitly exclude those by requiring NO sentence-end char
    # immediately before the quote.
    ("quote-shaped footnote marker", re.compile(r"\b[A-Za-z][A-Za-z\-]*[,;:]?['’]\s+[A-Z]")),
]


def _suspicious_text_artifacts(epub_files: dict[str, bytes]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for name, data in epub_files.items():
        if not name.startswith("OEBPS/text/"):
            continue
        try:
            root = ET.fromstring(data)
        except ET.ParseError:
            continue
        body = root.find(f"{{{NS_X}}}body")
        if body is None:
            continue
        text = re.sub(r"\s+", " ", _visible_non_footnote_text(body))
        for label, pat in SUSPICIOUS_TEXT_PATTERNS:
            for m in pat.finditer(text):
                excerpt = text[max(0, m.start() - 90):m.end() + 120].strip()
                findings.append({
                    "kind": label,
                    "file": Path(name).name,
                    "excerpt": excerpt,
                })
                if len(findings) >= 80:
                    return findings
    return findings


def _filter_vision_blocks(
    blocks: Iterable[dict],
    *,
    strict_digits: bool = False,
) -> list[dict]:
    """Mirror :func:`quire.pipeline._filter_vision_blocks`.

    Imported lazily to avoid a circular import. ``strict_digits`` enables
    the Tesseract bibliography-hallucination rejects.
    """
    from ..pipeline import _filter_vision_blocks as _impl
    return _impl(list(blocks), strict_digits=strict_digits)


def _validate_opf(epub_path: Path) -> list[str]:
    """Check OPF manifest/spine integrity. Returns a list of issue strings."""
    import re as _re
    issues: list[str] = []
    try:
        with zipfile.ZipFile(epub_path) as zf:
            names = set(zf.namelist())
            try:
                opf = zf.read("OEBPS/content.opf").decode("utf-8", errors="replace")
            except KeyError:
                return ["OEBPS/content.opf missing from EPUB"]
    except zipfile.BadZipFile:
        return [f"{epub_path.name} is not a valid ZIP"]
    manifest_items: dict[str, str] = {}
    for m in _re.finditer(r'<item\s+id="([^"]+)"\s+href="([^"]+)"', opf):
        item_id, href = m.group(1), m.group(2)
        manifest_items[item_id] = href
        zip_path = href if href.startswith("OEBPS/") else f"OEBPS/{href}"
        if zip_path not in names:
            issues.append(f"manifest href '{href}' has no file in zip")
    if len(manifest_items) != len(set(manifest_items.keys())):
        issues.append("manifest contains duplicate ids")
    spine_refs = _re.findall(r'<itemref\s+idref="([^"]+)"', opf)
    for ref in spine_refs:
        if ref not in manifest_items:
            issues.append(f"spine itemref '{ref}' has no manifest item")
    return issues


def run_audit(
    cfg: BookConfig,
    *,
    ocr_pages: list[dict] | None = None,
    run_epubcheck: bool = True,
) -> dict:
    epub_path = cfg.epub_path
    if not epub_path.exists():
        raise SystemExit(f"missing EPUB: {epub_path}")

    files = _load_xhtml_files(epub_path)
    chapter_files = {
        n: d for n, d in files.items()
        if n.startswith("OEBPS/text/ch-") or n.endswith("front-matter.xhtml")
    }

    eng_words = sum(_count_english_words(d) for d in chapter_files.values())
    ar_chars = sum(_count_arabic_chars(d) for d in chapter_files.values())

    doc = fitz.open(str(cfg.pdf_path))
    pdf_words = 0
    pdf_real = 0
    for page in doc:
        text = page.get_text()
        for w in WORD_RE.findall(text):
            if not w.isalpha():
                continue
            pdf_words += 1
            vowels = sum(c.lower() in "aeiouy" for c in w)
            consonants = sum(1 for c in w.lower() if c.isalpha() and c not in "aeiouy")
            if consonants >= 4 and vowels == 0:
                continue
            inner_upper = sum(1 for c in w[1:] if c.isupper())
            upper_count = sum(1 for c in w if c.isupper())
            if inner_upper >= 1 and upper_count <= len(w) // 2 and len(w) <= 7:
                continue
            pdf_real += 1

    audit_engine = "vision"
    if ocr_pages is None:
        from ..pipeline import read_pages_cache
        # Try each engine's cache file in priority order; older Vision-only
        # builds had only ``vision_ocr.pkl``, Tesseract builds have
        # ``tesseract_ocr.pkl``.
        ocr_pages = []
        for cache_name, engine_label in (
            ("vision_ocr.pkl", "vision"),
            ("tesseract_ocr.pkl", "tesseract"),
        ):
            cache = cfg.caches_dir / cache_name
            pages = read_pages_cache(cache)
            if pages:
                ocr_pages = pages
                audit_engine = engine_label
                break
    else:
        audit_engine = (getattr(cfg, "ocr_engine", "vision") or "vision").lower()

    # Apply the same hallucination filters the pipeline does, so the audit
    # report does NOT silently inflate Tesseract's Arabic count with
    # bibliography hallucination — the line-level pre-cluster filter, the
    # per-block ``strict_digits`` filter, the block-level
    # embedded-text/diacritic guard, and the page-level
    # ``_page_is_english_citation_dense`` guard.
    import statistics as _stats

    from ..extract.ocr import cluster_arabic_blocks
    from ..pipeline import (
        _arabic_dominant,
        _embedded_english_words,
        _is_embedded_text_hallucination,
        _is_line_geometric_artifact,
        _is_line_hallucination,
        _page_is_english_citation_dense,
    )
    strict_digits = audit_engine == "tesseract"
    bib_suppress = audit_engine == "tesseract"
    embed_suppress = audit_engine == "tesseract"
    embed_doc = None
    if embed_suppress:
        try:
            embed_doc = fitz.open(str(cfg.pdf_path))
        except Exception:
            embed_doc = None
    ocr_chars_real = 0
    for p in ocr_pages:
        # For Tesseract: re-derive blocks from ar_lines using the same
        # line-level filter + cluster + block filter chain the pipeline
        # uses. For Vision: read the cached arabic_blocks directly.
        if embed_suppress and p.get("ar_lines"):
            pno = p.get("pno")
            page_idx = (pno - 1) if isinstance(pno, int) and pno >= 1 else None
            embedded_for_lines: list = []
            if (
                embed_doc is not None
                and page_idx is not None
                and 0 <= page_idx < len(embed_doc)
            ):
                embedded_for_lines = _embedded_english_words(embed_doc, page_idx)
            ar_lines = [
                L for L in p.get("ar_lines", []) if _arabic_dominant(L.get("text", ""))
            ]
            ar_lines = [
                L for L in ar_lines if not _is_line_geometric_artifact(L)
            ]
            if embedded_for_lines:
                ar_lines = [
                    L for L in ar_lines
                    if not _is_line_hallucination(L, embedded_for_lines)
                ]
            if ar_lines:
                heights = [
                    L["y1"] - L["y0"] for L in ar_lines if L["y1"] > L["y0"]
                ]
                avg_h = _stats.median(heights) if heights else 14.0
                blocks = cluster_arabic_blocks(ar_lines, avg_h)
                for b in blocks:
                    if 0 < b.get("conf", -1) <= 1.0:
                        b["conf"] *= 100
            else:
                blocks = []
        else:
            blocks = list(p.get("arabic_blocks", []))

        kept = _filter_vision_blocks(blocks, strict_digits=strict_digits)
        # Mirror the pipeline's block-level embedded-text filter.
        if embed_suppress and embed_doc is not None and kept:
            pno = p.get("pno")
            page_idx = (pno - 1) if isinstance(pno, int) and pno >= 1 else None
            if page_idx is not None and 0 <= page_idx < len(embed_doc):
                embedded = _embedded_english_words(embed_doc, page_idx)
                if embedded:
                    kept = [
                        b for b in kept
                        if not _is_embedded_text_hallucination(b, embedded)
                    ]
        # Mirror the pipeline: the page-level filter only fires after the
        # per-block filter has already applied (so we read it off the
        # filtered set, not the raw OCR cache).
        if bib_suppress and kept:
            virt_page = {**p, "arabic_blocks": kept}
            if _page_is_english_citation_dense(virt_page):
                kept = []
        for blk in kept:
            n = sum(1 for c in blk["text"] if ARABIC_RE.match(c))
            if n >= 4:
                ocr_chars_real += n
    if embed_doc is not None:
        embed_doc.close()

    eng_pct = (eng_words / pdf_real) * 100 if pdf_real else 0
    ar_pct = (ar_chars / ocr_chars_real) * 100 if ocr_chars_real else 0

    targets_by_file = {n: _collect_ids_and_links(d)[0] for n, d in files.items()}
    issues = 0
    link_findings: list[dict[str, str]] = []
    for name, data in files.items():
        _, links = _collect_ids_and_links(data)
        for frag, ref_file in links:
            if ref_file == "_self":
                if frag not in targets_by_file.get(name, set()):
                    issues += 1
                    link_findings.append({
                        "kind": "unresolved internal link",
                        "file": Path(name).name,
                        "excerpt": f"#{frag}",
                    })
            else:
                target_path = str(Path(name).parent / ref_file).replace("\\", "/")
                if not any(target_path == k or target_path == k.split("/", 1)[-1]
                           for k in targets_by_file):
                    candidates = [k for k in targets_by_file
                                  if k.endswith("/" + ref_file) or k.endswith(ref_file)]
                    if not candidates:
                        issues += 1
                        link_findings.append({
                            "kind": "missing linked file",
                            "file": Path(name).name,
                            "excerpt": f"{ref_file}#{frag}",
                        })
                        continue
                    target_path = candidates[0]
                if frag not in targets_by_file.get(target_path, set()):
                    issues += 1
                    link_findings.append({
                        "kind": "missing fragment",
                        "file": Path(name).name,
                        "excerpt": f"{ref_file}#{frag}",
                    })

    suspicious = link_findings + _suspicious_text_artifacts(files)

    report_lines = [
        f"chapter files: {len(chapter_files)}",
        f"EPUB English words (chapter body): {eng_words}",
        f"EPUB Arabic chars (chapter body):  {ar_chars}",
        f"PDF English words (text layer):    {pdf_words}",
        f"PDF English words (real, post-filter): {pdf_real}",
        f"OCR Arabic chars (real, blk>=4):   {ocr_chars_real}",
        f"English coverage: {eng_pct:.1f}%   (target >= 90%)",
        f"Arabic  coverage: {ar_pct:.1f}%   (target >= 95%)",
        f"unresolved internal links: {issues}",
        f"suspicious OCR/noteref artifacts: {len(suspicious)}",
        "",
        "per-chapter sanity:",
    ]
    for name, data in sorted(chapter_files.items()):
        w = _count_english_words(data)
        a = _count_arabic_chars(data)
        report_lines.append(f"  {Path(name).name:50}  en={w:>5}  ar={a:>4}")

    atomic_write_text(cfg.audit_path, "\n".join(report_lines) + "\n")
    print("\n".join(report_lines), file=sys.stderr)
    print(f"\nwrote {cfg.audit_path}", file=sys.stderr)

    opf_issues = _validate_opf(cfg.epub_path) if cfg.epub_path.exists() else []
    if opf_issues:
        for msg in opf_issues:
            print(f"[quire] opf-validate: {msg}", file=sys.stderr)

    epubcheck_result: dict | None = None
    if run_epubcheck and cfg.epub_path.exists():
        from ..epubcheck import run_epubcheck as _run_ec
        epubcheck_result = _run_ec(cfg.epub_path)
        if epubcheck_result.get("status") == "unavailable":
            print("[quire] epubcheck not installed; skipping validation", file=sys.stderr)
        else:
            ok_or_warn = epubcheck_result["status"] in {"ok", "warn"}
            tag = "OK" if ok_or_warn else "FAIL"
            print(
                f"[quire] epubcheck: {tag} ({len(epubcheck_result.get('messages', []))} messages)",
                file=sys.stderr,
            )

    audit_json = {
        "slug": cfg.slug,
        "title": cfg.title,
        "language": cfg.language,
        "ocr_engine": cfg.ocr_engine,
        "chapter_files": len(chapter_files),
        "english_words": eng_words,
        "arabic_chars": ar_chars,
        "pdf_english_words": pdf_words,
        "pdf_english_words_real": pdf_real,
        "ocr_arabic_chars_real": ocr_chars_real,
        "english_coverage_pct": eng_pct,
        "arabic_coverage_pct": ar_pct,
        "unresolved_links": issues,
        "suspicious_count": len(suspicious),
        "per_chapter": [
            {
                "file": Path(name).name,
                "english_words": _count_english_words(data),
                "arabic_chars": _count_arabic_chars(data),
            }
            for name, data in sorted(chapter_files.items())
        ],
        "findings": [
            {"kind": item["kind"], "file": item.get("file", ""), "excerpt": item["excerpt"]}
            for item in suspicious
        ],
    }
    if epubcheck_result is not None:
        audit_json["epubcheck_status"] = epubcheck_result["status"]
        audit_json["epubcheck_messages"] = epubcheck_result.get("messages", [])
    audit_json["opf_issues"] = opf_issues
    atomic_write_text(cfg.audit_json_path, json.dumps(audit_json, ensure_ascii=False, indent=2))

    review_rows: list[dict[str, str]] = _existing_ocr_review_rows(cfg.ocr_review_tsv_path)
    if ocr_pages:
        for p in ocr_pages:
            kept = _filter_vision_blocks(list(p.get("arabic_blocks", [])))
            for blk in kept:
                if blk.get("conf", -1) >= 0 and blk["conf"] < 70:
                    excerpt = blk["text"].replace("\n", " ⏎ ")[:160]
                    review_rows.append({
                        "kind": "low-confidence Arabic OCR",
                        "file": "",
                        "page": str(p.get("pno")),
                        "excerpt": excerpt,
                        "suggested_fix": "",
                    })
    for item in suspicious:
        review_rows.append({
            "kind": item["kind"],
            "file": item["file"],
            "page": "",
            "excerpt": item["excerpt"],
            "suggested_fix": "",
        })

    if review_rows:
        header = ["kind", "file", "page", "excerpt", "suggested_fix"]
        def cell(value: str) -> str:
            return str(value).replace("\t", " ").replace("\n", " ").strip()
        atomic_write_text(
            cfg.review_tsv_path,
            "\t".join(header) + "\n" +
            "\n".join("\t".join(cell(row.get(h, "")) for h in header) for row in review_rows) +
            "\n",
        )
        print(f"wrote {cfg.review_tsv_path} ({len(review_rows)} review rows)", file=sys.stderr)

    if cfg.include_corrections_md:
        log: list[str] = []
        if review_rows:
            log.extend([
                "# Corrections required",
                "",
                "Edit `manual_review.tsv` for spreadsheet-style review. The Markdown below is a readable copy.",
                "",
            ])
        low_conf = [r for r in review_rows if r["kind"] == "low-confidence Arabic OCR"]
        if low_conf:
            log.extend([
                "## Low-confidence Arabic OCR",
                "",
                "Review these against the printed source and add corrections to the book vocabulary or a post-processor.",
                "",
            ])
            log.extend(
                f"- p. {row['page']}: `{row['excerpt']}`"
                for row in low_conf
            )
        other_rows = [r for r in review_rows if r["kind"] != "low-confidence Arabic OCR"]
        if other_rows:
            log.extend(["", "## Suspicious OCR / note-marker patterns", ""])
            log.extend(
                f"- {row['file']}: {row['kind']}: `{row['excerpt']}`"
                for row in other_rows
            )
        if log:
            atomic_write_text(cfg.corrections_path, "\n".join(log) + "\n")
            print(f"wrote {cfg.corrections_path} ({len(review_rows)} review rows)",
                  file=sys.stderr)

    result = {
        "english_pct": eng_pct,
        "arabic_pct": ar_pct,
        "unresolved_links": issues,
        "suspicious_count": len(suspicious),
        "opf_issues": opf_issues,
        "audit_path": str(cfg.audit_path),
        "audit_json_path": str(cfg.audit_json_path),
    }
    if epubcheck_result is not None:
        result["epubcheck_status"] = epubcheck_result["status"]
    return result
