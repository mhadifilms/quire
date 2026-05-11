"""Build the EPUB3 container.

The packager reads :class:`BookConfig` for everything book-specific (title,
language, author, fonts) and produces an ``epubcheck``-clean reflowable EPUB.

EPUB layout:

    <book>.epub/
        mimetype                       (stored, no compression, first entry)
        META-INF/container.xml
        OEBPS/
            content.opf
            nav.xhtml
            text/{cover,titlepage,<chapters>}.xhtml
            styles/book.css
            fonts/<embedded fonts>
            images/cover.jpg
"""

from __future__ import annotations

import datetime as _dt
import html
import uuid
import zipfile
from pathlib import Path

import fitz

from ..config import REPO_ROOT, BookConfig
from ..io_utils import atomic_write_bytes, content_fingerprint

_QUIRE_NS = uuid.UUID("8b0f9c8a-2b8e-5d9f-9f1c-001100110011")


def _stable_book_id(cfg: BookConfig) -> str:
    """Deterministic urn-uuid for the book.

    Built from the slug and a content fingerprint of the source PDF, so
    rebuilding the same book produces the same EPUB id (diff-friendly,
    helpful for CDN caches and stable share URLs).
    """
    fingerprint = content_fingerprint(cfg.pdf_path)
    name = f"{cfg.slug}:{fingerprint}"
    return str(uuid.uuid5(_QUIRE_NS, name))

# ---------- font defaults ----------
# A sensible default font set for English + Arabic + Persian / Urdu books.
# Mapping of filename -> (family, weight, style).
DEFAULT_FONT_TABLE: dict[str, tuple[str, str, str]] = {
    "LiberationSerif-Regular.ttf": ("Liberation Serif Embedded", "normal", "normal"),
    "LiberationSerif-Bold.ttf": ("Liberation Serif Embedded", "bold", "normal"),
    "LiberationSerif-Italic.ttf": ("Liberation Serif Embedded", "normal", "italic"),
    "LiberationSerif-BoldItalic.ttf": ("Liberation Serif Embedded", "bold", "italic"),
    "NotoNaskhArabic.ttf": ("Noto Naskh Arabic Embedded", "normal", "normal"),
    "NotoNastaliqUrdu.ttf": ("Noto Nastaliq Urdu Embedded", "normal", "normal"),
}
USER_FONTS_DIR = REPO_ROOT / "data" / "fonts"


def _resolve_fonts(cfg: BookConfig) -> list[tuple[Path, str, str, str]]:
    """Return the list of ``(file_path, family, weight, style)`` fonts to embed.

    If ``cfg.embed_fonts`` is non-empty, embed exactly that set (with metadata
    looked up in :data:`DEFAULT_FONT_TABLE`, or inferred from the file stem).
    Otherwise embed any default font that exists on disk so the EPUB still
    builds when the user hasn't configured fonts explicitly.
    """
    out: list[tuple[Path, str, str, str]] = []
    if cfg.embed_fonts:
        sources = list(cfg.embed_fonts)
    else:
        sources = [USER_FONTS_DIR / fname for fname in DEFAULT_FONT_TABLE]
    for path in sources:
        if not path.exists():
            continue
        meta = DEFAULT_FONT_TABLE.get(path.name)
        if meta is None:
            family = f"{path.stem} Embedded"
            weight, style = "normal", "normal"
            lname = path.stem.lower()
            if "bold" in lname:
                weight = "bold"
            if "italic" in lname or "oblique" in lname:
                style = "italic"
        else:
            family, weight, style = meta
        out.append((path, family, weight, style))
    return out


def _base_dir(language: str) -> str:
    return "rtl" if language.split("-", 1)[0].lower() in {"ar", "fa", "he", "ur"} else "ltr"


def _font_face_block(fonts: list[tuple[Path, str, str, str]]) -> str:
    out = []
    for path, family, weight, style in fonts:
        out.append(
            f"@font-face {{\n"
            f"  font-family: \"{family}\";\n"
            f"  font-style: {style};\n"
            f"  font-weight: {weight};\n"
            f"  src: url(\"../fonts/{path.name}\") format(\"truetype\");\n"
            f"}}\n"
        )
    return "".join(out)


CSS_BODY = r"""
html, body {
  margin: 0;
  padding: 0;
}
body {
  font-family: "Liberation Serif Embedded", serif;
  line-height: 1.55;
  text-align: justify;
  hyphens: auto;
  -webkit-hyphens: auto;
  -epub-hyphens: auto;
}
section[epub|type~="chapter"] { margin: 0 0.5em; }
h1, h2, h3, h4 {
  font-family: "Liberation Serif Embedded", serif;
  page-break-after: avoid;
  break-after: avoid;
  text-align: center;
  font-weight: bold;
}
h1 { font-size: 1.75em; margin: 1.4em 0 1em; }
h2 { font-size: 1.35em; margin: 1.2em 0 0.8em; }
h3 { font-size: 1.15em; margin: 1em 0 0.6em; font-style: italic; }

p.body { margin: 0 0 0.4em; text-indent: 0; }
p.body.indent { text-indent: 1.4em; }
p.body.center { text-align: center; text-indent: 0; }
p.body + p.body { text-indent: 1.4em; }
p.body.center { text-align: center; text-indent: 0; }
p.body:first-child, h1 + p.body, h2 + p.body, h3 + p.body { text-indent: 0; }

p.arabic {
  font-family: "Noto Naskh Arabic Embedded", serif;
  text-align: center;
  line-height: 1.9;
  margin: 1.1em 0;
  font-size: 1.18em;
}
p.arabic.quran { margin: 1.3em 1em; }
p.persian {
  font-family: "Noto Nastaliq Urdu Embedded", "Noto Naskh Arabic Embedded", serif;
  text-align: center;
  line-height: 2.4;
  margin: 1.4em 0;
  font-size: 1.18em;
}
span.arabic-inline { font-family: "Noto Naskh Arabic Embedded", serif; }
span.persian-inline {
  font-family: "Noto Nastaliq Urdu Embedded", "Noto Naskh Arabic Embedded", serif;
}

aside.footnotes {
  margin-top: 2.5em;
  border-top: 1px solid #888;
  padding-top: 0.6em;
  font-size: 0.92em;
}
aside.footnotes h2.footnotes-title {
  font-size: 1em;
  font-style: italic;
  margin: 0.6em 0;
  text-align: left;
  font-weight: normal;
  color: #666;
}
aside[epub|type~="footnote"] { display: block; margin: 0.4em 0; }
aside[epub|type~="footnote"] p { margin: 0; text-indent: 0; text-align: left; }
aside[epub|type~="footnote"] .fn-num { font-weight: bold; margin-right: 0.3em; }
aside[epub|type~="footnote"] .fn-back {
  text-decoration: none; color: #666; margin-left: 0.4em;
}
a[epub|type~="noteref"] {
  font-size: 0.7em; vertical-align: super; line-height: 0;
  text-decoration: none; color: #336;
}
span[epub|type~="pagebreak"] { display: none; }
nav[epub|type~="toc"] ol, nav[epub|type~="page-list"] ol {
  list-style: none; padding-left: 0;
}
nav[epub|type~="toc"] li { margin: 0.2em 0; }
nav[epub|type~="page-list"] li { display: inline; }
nav[epub|type~="page-list"] li::after { content: " | "; }
nav h1 { text-align: left; }
"""


CONTAINER_XML = """<?xml version="1.0" encoding="utf-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml" />
  </rootfiles>
</container>
"""


def render_cover_xhtml(cfg: BookConfig) -> str:
    base_dir = _base_dir(cfg.language)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{html.escape(cfg.language)}" xml:lang="{html.escape(cfg.language)}" dir="{base_dir}">
  <head>
    <meta charset="utf-8" />
    <title>Cover</title>
    <link rel="stylesheet" type="text/css" href="../styles/book.css" />
    <style>
      body {{ margin: 0; padding: 0; text-align: center; }}
      img {{ max-width: 100%; height: auto; }}
    </style>
  </head>
  <body epub:type="cover" dir="{base_dir}">
    <section epub:type="cover" aria-label="Cover">
      <img src="../images/cover.jpg" alt="{html.escape(cfg.title)}" />
    </section>
  </body>
</html>
"""


def render_titlepage_xhtml(cfg: BookConfig) -> str:
    title = html.escape(cfg.title)
    author = html.escape(cfg.author)
    base_dir = _base_dir(cfg.language)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{html.escape(cfg.language)}" xml:lang="{html.escape(cfg.language)}" dir="{base_dir}">
  <head>
    <meta charset="utf-8" />
    <title>Title Page</title>
    <link rel="stylesheet" type="text/css" href="../styles/book.css" />
  </head>
  <body epub:type="titlepage" dir="{base_dir}">
    <section epub:type="titlepage">
      <h1>{title}</h1>
      <p style="text-align:center; margin-top: 2em;"><em>{author}</em></p>
    </section>
  </body>
</html>
"""


def render_nav(cfg: BookConfig, chapters, page_list) -> str:
    base_dir = _base_dir(cfg.language)
    toc_items = []
    first_chapter_slug = None
    for c in chapters:
        if c.title == "Front Matter" and not c.elements and not c.footnotes:
            continue
        href = f"text/{c.slug}.xhtml"
        if first_chapter_slug is None:
            first_chapter_slug = c.slug
        toc_items.append(f'      <li><a href="{href}">{html.escape(c.title)}</a></li>')

    page_items = [
        f'      <li><a href="{href}">{html.escape(label)}</a></li>'
        for (_pno, href, label) in page_list
    ]
    body_href = f"text/{first_chapter_slug}.xhtml" if first_chapter_slug else "text/cover.xhtml"

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops" lang="{html.escape(cfg.language)}" xml:lang="{html.escape(cfg.language)}" dir="{base_dir}">
  <head>
    <meta charset="utf-8" />
    <title>Navigation</title>
    <link rel="stylesheet" type="text/css" href="styles/book.css" />
  </head>
  <body dir="{base_dir}">
    <nav epub:type="toc" role="doc-toc" id="toc">
      <h1>Table of Contents</h1>
      <ol>
{chr(10).join(toc_items)}
      </ol>
    </nav>
    <nav epub:type="landmarks" hidden="">
      <h2>Landmarks</h2>
      <ol>
        <li><a epub:type="cover" href="text/cover.xhtml">Cover</a></li>
        <li><a epub:type="titlepage" href="text/titlepage.xhtml">Title Page</a></li>
        <li><a epub:type="bodymatter" href="{body_href}">Begin Reading</a></li>
      </ol>
    </nav>
    <nav epub:type="page-list" role="doc-pagelist" id="page-list">
      <h2>Page List</h2>
      <ol>
{chr(10).join(page_items)}
      </ol>
    </nav>
  </body>
</html>
"""


def render_opf(
    cfg: BookConfig,
    book_id: str,
    chapters,
    page_list_count: int,
    modified: str,
    fonts: list[tuple[Path, str, str, str]] | None = None,
) -> str:
    base_dir = _base_dir(cfg.language)
    fonts = fonts or []
    manifest_items = [
        '    <item id="cover-image" href="images/cover.jpg" media-type="image/jpeg" properties="cover-image" />',
        '    <item id="cover-html" href="text/cover.xhtml" media-type="application/xhtml+xml" />',
        '    <item id="titlepage" href="text/titlepage.xhtml" media-type="application/xhtml+xml" />',
        '    <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav" />',
        '    <item id="css" href="styles/book.css" media-type="text/css" />',
    ]
    spine_items = [
        '    <itemref idref="cover-html" />',
        '    <itemref idref="titlepage" />',
    ]

    for path, *_ in fonts:
        item_id = "font-" + path.name.replace(".", "-").lower()
        manifest_items.append(
            f'    <item id="{item_id}" href="fonts/{path.name}" media-type="application/font-sfnt" />'
        )

    for c in chapters:
        if c.title == "Front Matter" and not c.elements and not c.footnotes:
            continue
        manifest_items.append(
            f'    <item id="{c.slug}" href="text/{c.slug}.xhtml" media-type="application/xhtml+xml" />'
        )
        spine_items.append(f'    <itemref idref="{c.slug}" />')

    # Keep the package-level language to the primary reading language only.
    # Some EPUB readers interpret additional RTL languages (e.g. Arabic) as a
    # whole-book page progression hint, even though Arabic appears only in
    # tagged inline/block spans inside an English book.
    lang_tags = f"    <dc:language>{html.escape(cfg.language)}</dc:language>"

    title = html.escape(cfg.title)
    author = html.escape(cfg.author)
    return f"""<?xml version="1.0" encoding="utf-8"?>
<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid" xml:lang="{html.escape(cfg.language)}" dir="{base_dir}">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="bookid">urn:uuid:{book_id}</dc:identifier>
    <dc:title>{title}</dc:title>
    <dc:creator id="creator-1">{author}</dc:creator>
{lang_tags}
    <meta property="dcterms:modified">{modified}</meta>
    <meta property="schema:accessMode">textual</meta>
    <meta property="schema:accessMode">visual</meta>
    <meta property="schema:accessibilityFeature">tableOfContents</meta>
    <meta property="schema:accessibilityFeature">printPageNumbers</meta>
    <meta property="schema:accessibilityHazard">none</meta>
    <meta property="schema:accessibilitySummary">Reflowable EPUB built by Quire. Printed page numbers preserved as page-list anchors. Total page anchors: {page_list_count}.</meta>
  </metadata>
  <manifest>
{chr(10).join(manifest_items)}
  </manifest>
  <spine page-progression-direction="{base_dir}">
{chr(10).join(spine_items)}
  </spine>
</package>
"""


def _build_page_list(rendered: list[tuple[str, str]]) -> list[tuple[int, str, str]]:
    """Walk every chapter XHTML, find <span epub:type="pagebreak" id="page-NN" ...>
    anchors, and return ``(printed, href, label)`` tuples for the nav.
    """
    import re
    out: list[tuple[int, str, str]] = []
    seen: set[int] = set()
    for slug, xhtml in rendered:
        for m in re.finditer(
            r'<span epub:type="pagebreak"[^>]*id="(page-\d+)"[^>]*aria-label="(\d+)"',
            xhtml,
        ):
            pid, label = m.group(1), m.group(2)
            try:
                pno = int(label)
            except ValueError:
                continue
            if pno in seen:
                continue
            seen.add(pno)
            out.append((pno, f"text/{slug}.xhtml#{pid}", label))
    out.sort(key=lambda t: t[0])
    return out


def build_epub(
    *,
    cfg: BookConfig,
    chapters,
    rendered: list[tuple[str, str]],
    cover_jpeg: Path,
    emitted_printed_pages: set[int] | None = None,
) -> Path:
    """Assemble the EPUB ZIP, writing ``mimetype`` first, uncompressed.

    The output is written atomically: we build the ZIP at ``<final>.tmp`` and
    rename when complete, so partial builds never leave a corrupted EPUB.
    """
    epub_path = cfg.epub_path
    epub_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = epub_path.with_suffix(epub_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    book_id = _stable_book_id(cfg)
    modified = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0, tzinfo=None).isoformat() + "Z"
    page_list = _build_page_list(rendered)

    fonts = _resolve_fonts(cfg)
    css = _font_face_block(fonts) + CSS_BODY

    with zipfile.ZipFile(tmp_path, "w") as zf:
        zi = zipfile.ZipInfo("mimetype")
        zi.compress_type = zipfile.ZIP_STORED
        zf.writestr(zi, "application/epub+zip")

        zf.writestr("META-INF/container.xml", CONTAINER_XML, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/styles/book.css", css, zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/text/cover.xhtml", render_cover_xhtml(cfg), zipfile.ZIP_DEFLATED)
        zf.writestr("OEBPS/text/titlepage.xhtml", render_titlepage_xhtml(cfg), zipfile.ZIP_DEFLATED)

        cover_bytes = cover_jpeg.read_bytes() if isinstance(cover_jpeg, Path) else cover_jpeg
        zf.writestr("OEBPS/images/cover.jpg", cover_bytes, zipfile.ZIP_DEFLATED)

        for slug, xhtml in rendered:
            zf.writestr(f"OEBPS/text/{slug}.xhtml", xhtml, zipfile.ZIP_DEFLATED)

        zf.writestr("OEBPS/nav.xhtml", render_nav(cfg, chapters, page_list), zipfile.ZIP_DEFLATED)
        zf.writestr(
            "OEBPS/content.opf",
            render_opf(cfg, book_id, chapters, len(page_list), modified, fonts=fonts),
            zipfile.ZIP_DEFLATED,
        )

        for path, *_ in fonts:
            zf.writestr(f"OEBPS/fonts/{path.name}", path.read_bytes(), zipfile.ZIP_DEFLATED)

    import os
    os.replace(tmp_path, epub_path)
    return epub_path


def render_cover_jpeg(pdf_path: str, out_path: str, *, page_index: int = 0) -> Path:
    """Render a single PDF page to a JPEG suitable for the EPUB cover."""
    out = Path(out_path)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_index]
        pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), alpha=False, colorspace=fitz.csRGB)
        atomic_write_bytes(out, pix.tobytes("jpeg"))
    finally:
        doc.close()
    return out
