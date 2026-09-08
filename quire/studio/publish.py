"""Publish one corrected manuscript consistently across reading formats."""

from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import fitz

from ..epubcheck import epubcheck_executable, run_epubcheck
from ..io_utils import atomic_write_text
from ..languages import direction, valid_language
from .project import digest, file_hash, load, locked, now, save
from .quality import check

TEMPLATES = {
    "reading": {"size": (420, 595), "font": 11.5, "leading": 1.55, "margin": 42},
    "large-print": {"size": (595, 842), "font": 17, "leading": 1.65, "margin": 54},
    "study": {"size": (595, 842), "font": 12, "leading": 1.6, "margin": 56},
}
FONT_FILES = [
    "LiberationSerif-Regular.ttf",
    "LiberationSerif-Bold.ttf",
    "LiberationSerif-Italic.ttf",
    "NotoNaskhArabic.ttf",
    "NotoNastaliqUrdu.ttf",
    "OFL.txt",
]


def _fonts() -> Path:
    for base in (
        Path(__file__).resolve().parents[2] / "data/fonts",
        Path(sys.prefix) / "share/quire/fonts",
        Path(sys.base_prefix) / "share/quire/fonts",
    ):
        if all((base / name).is_file() for name in FONT_FILES):
            return base
    raise RuntimeError("Publication fonts are missing; reinstall Quire with its packaged font assets")


def _css(template: str) -> str:
    style = TEMPLATES[template]
    return f"""
@font-face {{font-family:Book;src:url('fonts/LiberationSerif-Regular.ttf')}}
@font-face {{font-family:Book;src:url('fonts/LiberationSerif-Bold.ttf');font-weight:bold}}
@font-face {{font-family:Book;src:url('fonts/LiberationSerif-Italic.ttf');font-style:italic}}
@font-face {{font-family:Arabic;src:url('fonts/NotoNaskhArabic.ttf')}}
@font-face {{font-family:Urdu;src:url('fonts/NotoNastaliqUrdu.ttf')}}
body {{font-family:Book,serif;font-size:{style["font"]}pt;line-height:{style["leading"]};color:#243343}}
[dir=rtl] {{font-family:Arabic,serif;text-align:right}}
[lang=ur] {{font-family:Urdu,Arabic,serif;line-height:2}}
h1,h2,h3,h4,h5,h6 {{line-height:1.2;page-break-after:avoid;color:#243343}}
h1 {{font-size:2.1em;margin-top:1.4em}} h2 {{font-size:1.5em;margin-top:1.3em}}
p {{margin:0 0 .85em}} blockquote {{margin:1em 1.5em;font-style:italic}}
.poetry {{white-space:pre-wrap;margin:1em 1.5em}} .cover {{text-align:center;page-break-after:always;padding-top:4em}}
.cover h1 {{font-size:2.6em}} .draft {{color:#995800}} .source-page {{font-size:.7em;color:#526577}}
.translation {{margin-top:.35em;padding-bottom:1em}} .original {{color:#526577}}
table {{border-collapse:collapse;width:100%;margin:1em 0}} th,td {{border:1px solid #bcc7d1;padding:.4em;text-align:start}}
figure {{margin:1em 0;page-break-inside:avoid}} img {{max-width:100%;height:auto}}
figcaption,.note {{font-size:.88em}} a {{color:#355ad7}} .notes {{page-break-before:always}}
.toc {{page-break-after:always}} .toc li {{margin:.4em 0}}
@media screen {{body {{max-width:46em;margin:3em auto;padding:0 1.5em}}}}
"""


def _inline(text: str, *, preserve_lines: bool = False) -> str:
    value = html.escape(text)
    return value.replace("\n", "<br />") if preserve_lines else value


def _block(node: dict, *, target: str | None, note_numbers: dict, variant: str = "") -> str:
    entry = node.get("translations", {}).get(target, {}) if target else {}
    text = entry.get("text", "[Translation missing]") if target else node["text"]
    language = target or node["language"]
    # MuPDF's CSS subset does not consistently apply attribute selectors;
    # explicit alignment also keeps bilingual PDF paragraphs independent.
    alignment = (
        "text-align:right;font-family:Arabic"
        if direction(language) == "rtl"
        else "text-align:left;font-family:Book"
    )
    attrs = f'lang="{html.escape(language)}" dir="{direction(language)}" style="{alignment}"'
    node_id = node["id"] + variant
    kind = node["kind"]
    references = " ".join(
        f'<a id="ref-{node_id}-{ref}" href="#{ref}{variant}" epub:type="noteref" role="doc-noteref"><sup>{note_numbers[ref]}</sup></a>'
        for ref in node.get("note_ids", [])
        if ref in note_numbers
    )
    content = _inline(text, preserve_lines=kind in {"poetry", "quote"}) + references
    uncertainty = entry.get("uncertainty_note") if target else node.get("uncertainty_note")
    editorial = f'<span class="note"> [Editorial note: {_inline(uncertainty)}]</span>' if uncertainty else ""
    content += editorial
    if kind == "heading":
        level = max(1, min(6, node.get("level", 2)))
        return f'<h{level} id="{node_id}" {attrs}>{content}</h{level}>'
    if kind == "table":
        cells = entry.get("cells") if target else node.get("cells")
        if not cells:
            return f'<p id="{node_id}" {attrs}>{content}</p>'
        rows = []
        for i, row in enumerate(cells):
            tag = "th" if i == 0 else "td"
            scope = ' scope="col"' if i == 0 else ""
            rows.append("<tr>" + "".join(f"<{tag}{scope}>{_inline(cell)}</{tag}>" for cell in row) + "</tr>")
        return f'<table id="{node_id}" {attrs}><tbody>{"".join(rows)}</tbody></table>{editorial}'
    if kind == "figure":
        alt = text if target else node.get("alt", "")
        return (
            f'<figure id="{node_id}" {attrs}><img src="{html.escape(node.get("asset", ""))}" '
            f'alt="{html.escape(alt)}" /><figcaption>{_inline(text)}{editorial}</figcaption></figure>'
        )
    if kind == "footnote":
        return f'<aside id="{node_id}" {attrs} epub:type="footnote" role="doc-footnote" class="note"><p>{note_numbers.get(node["id"], "")}. {content}</p></aside>'
    tag = "blockquote" if kind == "quote" else "p"
    return f'<{tag} id="{node_id}" {attrs} class="{kind}">{content}</{tag}>'


def manuscript(
    data: dict, *, target: str | None = None, bilingual: bool = False, draft: bool = False
) -> tuple[str, list]:
    nodes = [n for n in data["nodes"] if n["status"] != "excluded"]
    notes = {n["id"]: i + 1 for i, n in enumerate(n for n in nodes if n["kind"] == "footnote")}
    headings = [
        (n["id"], n.get("translations", {}).get(target, {}).get("text", n["text"]) if target else n["text"])
        for n in nodes
        if n["kind"] == "heading"
    ]
    toc = (
        '<nav class="toc" aria-label="Contents"><h2>Contents</h2><ol>'
        + "".join(f'<li><a href="#{node_id}">{_inline(title)}</a></li>' for node_id, title in headings)
        + "</ol></nav>"
    )
    body = [
        f'<section class="cover"><h1>{_inline(data["title"])}</h1><p>{_inline(data["author"])}</p>',
        '<p class="draft">Draft for review</p>' if draft else "",
        "</section>",
        toc,
    ]
    by_page: dict[int, list[dict]] = {}
    for node in nodes:
        by_page.setdefault(node["page"], []).append(node)
    for page in data["pages"]:
        pno = page["number"]
        body.append(
            f'<div id="page-{pno}" epub:type="pagebreak" role="doc-pagebreak" aria-label="{pno}" class="source-page">Source page {pno}</div>'
        )
        for node in by_page.get(pno, []):
            if node["kind"] == "footnote":
                continue
            if bilingual and target:
                body.append(
                    '<div class="original">'
                    + _block(node, target=None, note_numbers=notes, variant="-source")
                    + "</div>"
                )
            body.append(
                '<div class="translation">' + _block(node, target=target, note_numbers=notes) + "</div>"
            )
    if notes:
        body.append('<section class="notes" role="doc-endnotes"><h2>Notes</h2>')
        for node in nodes:
            if node["kind"] == "footnote":
                if bilingual and target:
                    body.append(_block(node, target=None, note_numbers=notes, variant="-source"))
                body.append(_block(node, target=target, note_numbers=notes))
        body.append("</section>")
    return "\n".join(body), headings


def _document(data, body, *, target, css=True):
    language = target or data["language"]["language"]
    return (
        f'<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        f'<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" '
        f'lang="{language}" xml:lang="{language}" dir="{direction(language)}"><head>'
        f'<meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />'
        f"<title>{_inline(data['title'])}</title>"
        + ('<link rel="stylesheet" href="book.css" />' if css else "")
        + f"</head><body>{body}</body></html>"
    )


def _pdf(data, html_text: str, output: Path, template: str, assets: Path) -> None:
    # MuPDF's HTML layout provides Unicode shaping and bidi using the exact
    # same manuscript as EPUB/HTML, including embedded Arabic/Urdu fonts.
    style = TEMPLATES[template]
    box = fitz.Rect(0, 0, *style["size"])
    margin = style["margin"]
    content = fitz.Rect(margin, margin, box.width - margin, box.height - margin)
    # MuPDF resolves left/right alignment relative to an RTL block, unlike
    # browsers. Keep semantic dir=rtl and adapt alignment only for this renderer.
    pdf_html = html_text.replace("text-align:right;font-family:Arabic", "text-align:left;font-family:Arabic")
    story = fitz.Story(html=pdf_html, user_css=_css(template), archive=str(assets))
    positions = []

    def rectfn(index, filled):
        if index > 10000:
            raise RuntimeError("PDF layout exceeded 10,000 pages; inspect an oversized table or figure")
        return box, content, None

    doc = story.write_with_links(rectfn, positionfn=lambda p: positions.append(p))
    expected = {n["id"] for n in data["nodes"] if n["status"] != "excluded"}
    placed = {p.id for p in positions if p.id and p.open_close & 1}
    if expected - placed:
        doc.close()
        raise RuntimeError("PDF layout omitted passages: " + ", ".join(sorted(expected - placed)[:10]))
    toc = []
    last_level = 0
    for p in positions:
        if p.heading and p.open_close & 1:
            level = min(p.heading, last_level + 1)
            toc.append([level, p.text, p.page_num])
            last_level = level
    if toc:
        doc.set_toc(toc)
    for page in doc:
        page.insert_text(
            (box.width / 2 - 5, box.height - 22), str(page.number + 1), fontsize=9, color=(0.32, 0.4, 0.47)
        )
    doc.set_metadata({"title": data["title"], "author": data["author"], "creator": "Quire"})
    doc.save(output, garbage=4, deflate=True)
    doc.close()


def _epub(data, body, headings, output: Path, *, target, folder: Path) -> None:
    language = target or data["language"]["language"]
    chapter = _document(data, body, target=target)
    ET.fromstring(chapter)
    nav_body = (
        '<nav epub:type="toc" role="doc-toc" aria-label="Contents" id="toc"><h1>Contents</h1><ol>'
        + (
            "".join(f'<li><a href="book.xhtml#{i}">{_inline(t)}</a></li>' for i, t in headings)
            or '<li><a href="book.xhtml">Book</a></li>'
        )
        + "</ol></nav>"
    )
    nav_body += (
        '<nav epub:type="page-list" role="doc-pagelist" aria-label="Source pages"><h2>Source pages</h2><ol>'
        + "".join(
            f'<li><a href="book.xhtml#page-{p["number"]}">{p["number"]}</a></li>' for p in data["pages"]
        )
        + "</ol></nav>"
    )
    assets = [
        p
        for directory in (folder / "assets", folder / "fonts")
        if directory.exists()
        for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".ttf"}
    ]
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".ttf": "font/ttf"}
    manifest = "".join(
        f'<item id="asset-{i}" href="{p.relative_to(folder).as_posix()}" media-type="{media[p.suffix.lower()]}" />'
        for i, p in enumerate(assets)
    )
    modified = now().replace("+00:00", "Z")
    has_figures = any(n["kind"] == "figure" and n["status"] != "excluded" for n in data["nodes"])
    visual_mode = '<meta property="schema:accessMode">visual</meta>' if has_figures else ""
    # Quire emits static text and still images, with no scripts, sound, or animation.
    # Do not claim textual sufficiency for unverified image descriptions.
    sufficient = "textual,visual" if has_figures else "textual"
    opf = f'''<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" xml:lang="{language}" unique-identifier="book-id" prefix="schema: http://schema.org/">
<metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
<dc:identifier id="book-id">urn:uuid:{data["id"]}</dc:identifier><dc:title>{_inline(data["title"])}</dc:title>
<dc:creator>{_inline(data["author"] or "Unknown")}</dc:creator><dc:language>{language}</dc:language>
<dc:source id="page-source">urn:sha256:{data["source"]["sha256"]}</dc:source>
<meta refines="#page-source" property="source-of">pagination</meta>
<meta property="dcterms:modified">{modified}</meta><meta property="schema:accessMode">textual</meta>
{visual_mode}<meta property="schema:accessModeSufficient">{sufficient}</meta>
<meta property="schema:accessibilityHazard">none</meta>
<meta property="schema:accessibilityFeature">structuralNavigation</meta>
<meta property="schema:accessibilityFeature">readingOrder</meta>
<meta property="schema:accessibilityFeature">displayTransformability</meta>
<meta property="schema:accessibilitySummary">Reflowable text with source-page navigation. See the accompanying validation report for checked accessibility results.</meta>
</metadata><manifest><item id="book" href="book.xhtml" media-type="application/xhtml+xml" />
<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav" />
<item id="css" href="book.css" media-type="text/css" />{manifest}</manifest>
<spine page-progression-direction="{direction(language)}"><itemref idref="book" /></spine></package>'''
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("mimetype", "application/epub+zip", compress_type=zipfile.ZIP_STORED)
        archive.writestr(
            "META-INF/container.xml",
            '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container" version="1.0"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" /></rootfiles></container>',
        )
        archive.writestr("OEBPS/content.opf", opf)
        archive.writestr("OEBPS/book.xhtml", chapter)
        archive.writestr("OEBPS/nav.xhtml", _document(data, nav_body, target=target))
        archive.write(folder / "book.css", "OEBPS/book.css")
        for asset in assets:
            archive.write(asset, "OEBPS/" + asset.relative_to(folder).as_posix())


def run_ace(epub: Path, folder: Path) -> dict:
    executable = shutil.which("ace")
    if not executable:
        return {"status": "unavailable", "reason": "Install @daisy/ace to run EPUB accessibility checks"}
    try:
        result = subprocess.run(
            [executable, "--silent", "--outdir", str(folder), str(epub)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        # EARL report aggregates assertions; rc alone is not sufficient.
        outcomes = []
        messages = []

        def walk(value):
            if isinstance(value, dict):
                if "earl:outcome" in value:
                    outcomes.append(value["earl:outcome"])
                assertion = value.get("earl:result", {})
                if assertion.get("earl:outcome") in {"fail", "failed", "earl:failed"} and value.get("earl:test"):
                    messages.append({"test": value["earl:test"].get("dct:title"), "message": assertion.get("dct:description", "Accessibility check failed")})
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(report)
        normalized = {str(o).rsplit(":", 1)[-1].rsplit("#", 1)[-1].lower() for o in outcomes}
        failed = bool(normalized & {"fail", "failed"})
        passed = bool(normalized & {"pass", "passed"}) and normalized <= {"pass", "passed", "inapplicable"}
        return {
            "status": "fail" if failed or result.returncode else "ok" if passed else "unavailable",
            "report": f"{folder.name}/{report_path.name}",
            "messages": messages,
            "stderr": result.stderr[-1000:],
        }
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        return {"status": "fail", "reason": str(exc)[:300]}


def _markdown(data, *, target, bilingual, draft):
    lines = [f"# {data['title']}", data["author"], "Draft for review" if draft else ""]
    for node in data["nodes"]:
        if node["status"] == "excluded":
            continue
        lines.extend(["", f"<!-- passage:{node['id']} source-page:{node['page']} -->"])
        selected = [None, target] if bilingual and target else [target]
        for tag in selected:
            entry = node.get("translations", {}).get(tag, {}) if tag else {}
            text = entry.get("text", "[Translation missing]") if tag else node["text"]
            if node["kind"] == "heading":
                text = "#" * node.get("level", 2) + " " + text
            elif node["kind"] == "figure":
                alt = entry.get("text", "") if tag else node.get("alt", "")
                text = f"![{alt.replace(']', '')}]({node.get('asset', '')})\n{text}"
            elif node["kind"] == "table":
                cells = entry.get("cells") if tag else node.get("cells")
                if cells:
                    rows = [
                        "| " + " | ".join(c.replace("|", "\\|").replace("\n", "<br>") for c in row) + " |"
                        for row in cells
                    ]
                    rows.insert(1, "| " + " | ".join("---" for _ in cells[0]) + " |")
                    text = "\n".join(rows)
            elif node["kind"] == "footnote":
                text = f"[^{node['id']}{'-source' if bilingual and tag is None else ''}]: {text}"
            elif node["kind"] == "quote":
                text = "\n".join("> " + line for line in text.splitlines())
            elif node["kind"] == "poetry":
                text = text.replace("\n", "  \n")
            suffix = "-source" if bilingual and tag is None else ""
            text += "".join(f"[^{ref}{suffix}]" for ref in node.get("note_ids", []))
            uncertainty = entry.get("uncertainty_note") if tag else node.get("uncertainty_note")
            if uncertainty:
                text += f" [Editorial note: {uncertainty}]"
            lines.extend([text, ""])
    return "\n".join(lines).strip() + "\n"


def publish(
    root: str | Path,
    *,
    formats=("pdf", "epub", "html", "markdown"),
    target: str | None = None,
    bilingual: bool = False,
    template: str = "reading",
    draft: bool = False,
) -> dict:
    root = Path(root).resolve()
    formats = set(formats)
    if target:
        valid_language(target)
    if not formats or formats - {"pdf", "epub", "html", "markdown", "text"}:
        raise ValueError("Choose pdf, epub, html, markdown, or text formats")
    if template not in TEMPLATES:
        raise ValueError("Unknown publication template")
    if bilingual and not target:
        raise ValueError("Choose a translation language for a bilingual edition")
    with locked(root):
        data = load(root, verify_source=True)
        quality = check(data, target=target, root=root)
        if not draft and not quality["ready"]:
            raise ValueError(
                f"Edition needs review: {quality['errors']} errors and {quality['review_items']} review items. Export a draft or resolve them in the workspace."
            )
        settings = {
            "formats": sorted(formats),
            "target": target,
            "bilingual": bilingual,
            "template": template,
            "draft": draft,
            "publisher_version": "0.2.0",
            "publication_schema": 2,
            "epubcheck_available": bool(epubcheck_executable()),
            "ace_available": bool(shutil.which("ace")),
        }
        release_id = f"r{data['revision']}-{digest(settings)[:8]}"
        destination = root / "editions" / release_id
        if (destination / "release.json").exists():
            prior = json.loads((destination / "release.json").read_text())
            if all(
                (destination / name).is_file() and file_hash(destination / name) == value
                for name, value in prior["checksums"].items()
            ):
                return prior
        (root / "editions").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".edition-", dir=root / "editions") as temporary:
            folder = Path(temporary)
            fonts = folder / "fonts"
            fonts.mkdir()
            for name in FONT_FILES:
                shutil.copy2(_fonts() / name, fonts / name)
            if (root / "assets").exists():
                shutil.copytree(root / "assets", folder / "assets")
            atomic_write_text(folder / "book.css", _css(template))
            body, headings = manuscript(data, target=target, bilingual=bilingual, draft=draft)
            document = _document(data, body, target=target)
            written = {}
            if "html" in formats:
                atomic_write_text(folder / "book.html", document)
                written["html"] = "book.html"
            if "markdown" in formats:
                atomic_write_text(
                    folder / "book.md", _markdown(data, target=target, bilingual=bilingual, draft=draft)
                )
                written["markdown"] = "book.md"
            if "text" in formats:
                tree = ET.fromstring(document)
                atomic_write_text(
                    folder / "book.txt", "\n".join(t.strip() for t in tree.itertext() if t.strip())
                )
                written["text"] = "book.txt"
            validation = {}
            if "pdf" in formats:
                _pdf(data, document, folder / "book.pdf", template, folder)
                written["pdf"] = "book.pdf"
                with fitz.open(folder / "book.pdf") as doc:
                    validation["pdf"] = {
                        "pages": len(doc),
                        "empty_pages": [
                            p.number + 1 for p in doc if not p.get_text().strip() and not p.get_images()
                        ],
                    }
            if "epub" in formats:
                _epub(data, body, headings, folder / "book.epub", target=target, folder=folder)
                written["epub"] = "book.epub"
                validation["epubcheck"] = run_epubcheck(folder / "book.epub")
                validation["ace"] = run_ace(folder / "book.epub", folder / "accessibility")
            failures = [name for name, result in validation.items() if result.get("status") == "fail"]
            if not draft and failures:
                atomic_write_text(root / "reports/publication-failure.json", json.dumps(validation, indent=2))
                raise ValueError("Publication validation failed: " + ", ".join(failures))
            incomplete = [
                name for name, result in validation.items() if result.get("status") == "unavailable"
            ]
            release = {
                "id": release_id,
                "project_id": data["id"],
                "revision": data["revision"],
                "at": now(),
                "state": "draft" if draft else "validation_pending" if incomplete else "ready",
                "settings": settings,
                "quality": quality,
                "validation": validation,
                "files": written,
                "directory": f"editions/{release_id}",
                "source_sha256": data["source"]["sha256"],
                "checksums": {
                    str(p.relative_to(folder)): file_hash(p) for p in folder.rglob("*") if p.is_file()
                },
            }
            atomic_write_text(folder / "release.json", json.dumps(release, ensure_ascii=False, indent=2))
            if destination.exists():
                raise ValueError(
                    "Existing edition was modified; preserve it and publish after another project revision"
                )
            folder.rename(destination)
        data["releases"].append(
            {k: release[k] for k in ("id", "at", "state", "directory", "revision", "settings")}
        )
        save(root, data)
        return release
