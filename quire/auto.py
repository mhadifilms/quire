"""Zero-configuration PDF detection and conversion."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import fitz

from .config import REPO_ROOT, BookConfig, load_book_config
from .languages import detect_language, text_quality
from .pipeline import _ocr_pages_for, build_book
from .render.audit import run_audit
from .render.chapters import slugify


@dataclass(frozen=True)
class AutoDetection:
    source: Path
    title: str
    author: str
    slug: str
    ocr_engine: str
    fallback_engine: str | None
    content_start_page: int
    language: str = "en"
    language_confidence: float = 1.0


def _clean_metadata(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "untitled", "unknown"} else text


def _filename_metadata(path: Path) -> tuple[str, str]:
    """Parse common ``Author, _Title_ from Collection.pdf`` filenames."""
    stem = path.stem.strip()
    italic_title = re.search(r"_([^_]+)_", stem)
    title = italic_title.group(1).strip() if italic_title else ""
    author = ""
    if "," in stem:
        author = stem.split(",", 1)[0].strip()
        if not title:
            title = stem.split(",", 1)[1].strip()
    elif not italic_title:
        # Scanner exports commonly retain only ``Author Name 0001-1``.
        # Preserve the stem as a neutral document title, but recover the
        # human-readable author rather than publishing it as ``Unknown``.
        numbered_scan = re.fullmatch(
            r"([A-Z][A-Za-z.'’-]+(?:\s+[A-Z][A-Za-z.'’-]+)+)\s+\d+(?:-\d+)*",
            stem,
        )
        if numbered_scan:
            author = numbered_scan.group(1)
    if not title:
        title = re.sub(r"\s+from\s+.+$", "", stem, flags=re.I)
    title = title.strip(" _.,")
    return title, author


def _normalized_words(text: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", text.lower(), re.UNICODE))


def _detect_content_start(texts: list[str], title: str) -> int:
    """Find a sparse standalone story-title page before collection content."""
    target = _normalized_words(title)
    if not target:
        return 1
    for index, text in enumerate(texts[:12]):
        normalized = _normalized_words(text)
        words = normalized.split()
        if (
            normalized == target
            or (
                normalized.startswith(target + " ")
                and len(words) <= len(target.split()) + 3
            )
        ):
            return index + 1
    return 1


def _vision_available() -> bool:
    return sys.platform == "darwin" and importlib.util.find_spec("ocrmac") is not None


def _tesseract_available() -> bool:
    return importlib.util.find_spec("pytesseract") is not None and shutil.which("tesseract") is not None


def detect_pdf(path: str | Path, *, language: str | None = None) -> AutoDetection:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"PDF not found: {source}")
    if source.suffix.lower() != ".pdf":
        raise ValueError(f"expected a PDF: {source}")

    filename_title, filename_author = _filename_metadata(source)
    with fitz.open(source) as doc:
        metadata = doc.metadata or {}
        title = filename_title or _clean_metadata(metadata.get("title")) or source.stem
        metadata_author = _clean_metadata(metadata.get("author"))
        # Reject application/user IDs accidentally stored as creator metadata
        # (e.g. camel-cased scanner account names).
        if (
            metadata_author
            and " " not in metadata_author
            and re.search(r"[a-z][A-Z]", metadata_author)
        ):
            metadata_author = ""
        author = filename_author or metadata_author or "Unknown"
        texts = [page.get_text() for page in doc]

    detection = detect_language("\n".join(texts)[:20000], hint=language)
    if detection["language"] == "und" and _tesseract_available():
        # Probe a few source images when a scan has no usable text layer.
        import io

        from PIL import Image

        from .studio.extract import _tesseract
        with fitz.open(source) as doc:
            for index in range(min(4, len(doc))):
                image = Image.open(io.BytesIO(doc[index].get_pixmap(dpi=140).tobytes("png")))
                candidates = _tesseract(image, [])
                sample = "\n".join(n["text"] for n in candidates if n["confidence"] >= .6)
                guessed = detect_language(sample)
                if guessed["confidence"] >= .75:
                    detection = guessed
                    break
    substantial_pages = sum(sum(c.isalpha() for c in text) >= 240 and text_quality(text) >= .65 for text in texts)
    word_count = sum(sum(c.isalpha() for c in text) / 3 for text in texts)
    reliable_text = bool(texts) and (
        substantial_pages >= max(1, round(len(texts) * 0.45))
        and word_count >= len(texts) * 80
    )
    if reliable_text:
        engine = "text"
        fallback = None
        content_start = _detect_content_start(texts, title)
    elif detection["language"] in {"fa", "ur", "ar"} and _tesseract_available():
        engine, fallback, content_start = "tesseract", None, 1
    elif _vision_available():
        engine = "vision"
        fallback = "tesseract" if _tesseract_available() else None
        content_start = 1
    elif _tesseract_available():
        engine = "tesseract"
        fallback = None
        content_start = 1
    else:
        raise RuntimeError("scanned PDF requires macOS Vision OCR or Tesseract")

    slug = slugify(title) or "book"
    return AutoDetection(
        source=source,
        title=title,
        author=author,
        slug=slug,
        ocr_engine=engine,
        fallback_engine=fallback,
        content_start_page=content_start,
        language=detection["language"],
        language_confidence=detection["confidence"],
    )


def _toml_string(value: str) -> str:
    # JSON string syntax is valid TOML basic-string syntax.
    return json.dumps(value, ensure_ascii=False)


def _prepare_workspace(detection: AutoDetection, workspace_root: Path) -> BookConfig:
    fingerprint = hashlib.sha256(detection.source.read_bytes()).hexdigest()[:12]
    book_dir = workspace_root / f"{detection.slug}-{fingerprint}"
    book_dir.mkdir(parents=True, exist_ok=True)
    source_pdf = book_dir / "source.pdf"

    with fitz.open(detection.source) as source:
        if detection.content_start_page > 1:
            trimmed = fitz.open()
            trimmed.insert_pdf(
                source,
                from_page=detection.content_start_page - 1,
                to_page=source.page_count - 1,
            )
            trimmed.save(source_pdf, garbage=4, deflate=True)
            trimmed.close()
        else:
            shutil.copy2(detection.source, source_pdf)

    fallback_line = (
        f"fallback_engine = {_toml_string(detection.fallback_engine)}\n"
        if detection.fallback_engine
        else ""
    )
    config = (
        "[book]\n"
        f"slug = {_toml_string(detection.slug)}\n"
        f"title = {_toml_string(detection.title)}\n"
        f"author = {_toml_string(detection.author)}\n"
        f'language = {_toml_string(detection.language)}\n\n'
        "[input]\n"
        'pdf = "source.pdf"\n'
        "cover_pdf_page = 1\n"
        "cover_is_content = true\n\n"
        "auto_detected = true\n\n"
        "[ocr]\n"
        f"engine = {_toml_string(detection.ocr_engine)}\n"
        f"{fallback_line}"
        f'languages = [{_toml_string(detection.language if detection.language != "en" else "en-US")}]\n'
        "workers = 4\n"
        "dpi_scale = 4\n"
        "retries = 1\n\n"
        "[structure]\n"
        f"headings = [{{ title = {_toml_string(detection.title)}, level = 1 }}]\n\n"
        "[postprocess]\n"
        "strict = true\n"
        'plugins = ["common_ocr"]\n\n'
        "[render]\n"
        'formats = ["epub", "markdown"]\n'
        "include_corrections_md = true\n"
        f"epub_filename = {_toml_string(detection.source.stem + '.epub')}\n"
    )
    (book_dir / "book.toml").write_text(config, encoding="utf-8")
    return load_book_config(book_dir, repo_root=REPO_ROOT)


def convert_pdf(
    path: str | Path,
    *,
    output_dir: str | Path | None = None,
    workspace_root: str | Path | None = None,
    force_ocr: bool = False,
    audit: bool = True,
    language: str | None = None,
) -> dict[str, Path]:
    """Auto-detect one PDF, build it, validate it, and map final outputs."""
    detection = detect_pdf(path, language=language)
    if detection.language == "und":
        raise ValueError("Language could not be identified; pass --language or use the review workspace import")
    workspace = Path(workspace_root or (REPO_ROOT / "books" / ".auto")).resolve()
    cfg = _prepare_workspace(detection, workspace)
    outputs = build_book(
        cfg,
        force_ocr=force_ocr,
        formats=["epub", "markdown"],
    )
    if audit:
        ocr_pages = (
            None
            if cfg.ocr_engine in {"text", "pdf", "pymupdf"}
            else _ocr_pages_for(cfg)
        )
        report = run_audit(cfg, ocr_pages=ocr_pages)
        if report.get("epubcheck_status") != "ok":
            raise RuntimeError(f"EPUBCheck failed for {detection.source.name}")
        if int(report.get("suspicious_count", 0)) > 0:
            raise RuntimeError(
                f"automatic audit found {report['suspicious_count']} "
                f"suspicious artifact(s) in {detection.source.name}"
            )

    destination = Path(output_dir).expanduser().resolve() if output_dir else detection.source.parent
    destination.mkdir(parents=True, exist_ok=True)
    mapped = {
        "epub": destination / f"{detection.source.stem}.epub",
        "markdown": destination / f"{detection.source.stem}.md",
    }
    shutil.copy2(outputs["epub"], mapped["epub"])
    shutil.copy2(outputs["markdown"], mapped["markdown"])
    return mapped
