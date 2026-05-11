"""Book configuration loader.

Each book in ``books/<slug>/`` has a ``book.toml`` describing its source PDF,
metadata, OCR setup, and the post-processing pipeline it wants to run. The
pipeline core stays language-agnostic; everything book-specific is injected
through this config.

The loader is intentionally side-effect-free: it never creates directories,
modifies global state, or imports unknown plugins. Callers that need output
directories should call :meth:`BookConfig.ensure_artifact_dirs` explicitly.
"""

from __future__ import annotations

import importlib.util
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib  # py3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


REPO_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_OCR_ENGINES = {"vision", "tesseract", "text", "pdf", "pymupdf"}
SUPPORTED_FORMATS = {"epub", "html", "markdown", "md", "text", "txt"}
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


class BookConfigError(ValueError):
    """Raised for malformed or invalid ``book.toml``."""


@dataclass
class BookConfig:
    """Resolved configuration for a single book build."""

    slug: str
    title: str
    author: str
    language: str
    book_dir: Path
    artifact_dir: Path
    pdf_path: Path
    cover_pdf_page: int
    ocr_engine: str
    ocr_languages: list[str]
    ocr_workers: int
    ocr_dpi_scale: int
    ocr_retries: int
    postprocess_plugins: list[str]
    plugin_settings: dict[str, dict[str, Any]]
    structure_headings: list[tuple[str, int]]
    book_heuristics: list[str]
    strict_plugins: bool
    embed_fonts: list[Path]
    missing_fonts: list[str]
    strict_fonts: bool
    include_corrections_md: bool
    epub_filename: str
    output_formats: list[str]
    raw: dict[str, Any]
    warnings: list[str] = field(default_factory=list)

    @property
    def caches_dir(self) -> Path:
        return self.artifact_dir / "caches"

    @property
    def page_images_dir(self) -> Path:
        return self.artifact_dir / "page_images"

    @property
    def epub_path(self) -> Path:
        return self.artifact_dir / self.epub_filename

    @property
    def html_dir(self) -> Path:
        return self.artifact_dir / "html"

    @property
    def markdown_path(self) -> Path:
        return self.artifact_dir / f"{self.slug}.md"

    @property
    def text_path(self) -> Path:
        return self.artifact_dir / f"{self.slug}.txt"

    @property
    def corrections_path(self) -> Path:
        return self.artifact_dir / "corrections.md"

    @property
    def review_tsv_path(self) -> Path:
        return self.artifact_dir / "manual_review.tsv"

    @property
    def ocr_corrections_path(self) -> Path:
        return self.artifact_dir / "ocr_corrections.tsv"

    @property
    def ocr_corrections_diff_path(self) -> Path:
        return self.artifact_dir / "ocr_corrections.diff"

    @property
    def ocr_review_tsv_path(self) -> Path:
        return self.artifact_dir / "ocr_review.tsv"

    @property
    def audit_path(self) -> Path:
        return self.artifact_dir / "audit.txt"

    @property
    def audit_json_path(self) -> Path:
        return self.artifact_dir / "audit.json"

    @property
    def lock_path(self) -> Path:
        return self.artifact_dir / ".build.lock"

    def plugin_config(self, name: str) -> dict[str, Any]:
        return self.plugin_settings.get(name, {})

    def ensure_artifact_dirs(self) -> None:
        """Create artifact + cache directories. Call from build/audit/export
        commands, never from the config loader."""
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.caches_dir.mkdir(parents=True, exist_ok=True)


def _resolve_under(base: Path, candidate: str) -> Path:
    p = Path(candidate)
    if p.is_absolute():
        return p
    return (base / p).resolve()


def load_book_config(book_dir: str | Path, *, repo_root: Path | None = None) -> BookConfig:
    """Load and validate a ``book.toml`` from ``books/<slug>/``.

    This function does not touch the filesystem outside of reading the book
    folder; in particular it does not create artifact directories.
    """
    repo_root = (repo_root or REPO_ROOT).resolve()
    book_dir = Path(book_dir).resolve()
    if not book_dir.is_dir():
        raise FileNotFoundError(f"book folder not found: {book_dir}")
    toml_path = book_dir / "book.toml"
    if not toml_path.exists():
        raise FileNotFoundError(f"missing book.toml in {book_dir}")
    with open(toml_path, "rb") as f:
        cfg = tomllib.load(f)

    warnings: list[str] = []
    book = cfg.get("book", {})
    inp = cfg.get("input", {})
    ocr = cfg.get("ocr", {})
    pp = cfg.get("postprocess", {})
    struct = cfg.get("structure", {})
    rd = cfg.get("render", {})

    slug = str(book.get("slug") or book_dir.name)
    if not _SLUG_RE.match(slug):
        raise BookConfigError(
            f"invalid slug {slug!r}: must match [a-z0-9][a-z0-9._-]*"
        )
    artifact_dir = (book_dir / "artifacts").resolve()

    pdf_path = _resolve_under(book_dir, inp.get("pdf", "source.pdf"))
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    cover_page = int(inp.get("cover_pdf_page", 1))
    if cover_page < 1:
        raise BookConfigError(f"cover_pdf_page must be >= 1, got {cover_page}")

    # New books default to Tesseract: cross-platform, 163 languages, native
    # RTL handling. Set ``[ocr] engine = "vision"`` to opt into macOS Vision
    # (faster on macOS, fewer languages), or ``"text"`` if the PDF already
    # contains a reliable text layer.
    ocr_engine = str(ocr.get("engine", "tesseract")).lower()
    if ocr_engine not in SUPPORTED_OCR_ENGINES:
        raise BookConfigError(
            f"unsupported ocr.engine {ocr_engine!r}. "
            f"Supported: {sorted(SUPPORTED_OCR_ENGINES)}"
        )

    raw_languages = ocr.get("languages", ["en-US"])
    if not isinstance(raw_languages, list) or not all(isinstance(x, str) for x in raw_languages):
        raise BookConfigError("ocr.languages must be a list of strings")
    ocr_languages = list(raw_languages) or ["en-US"]

    ocr_workers = int(ocr.get("workers", 4))
    if ocr_workers < 1:
        raise BookConfigError(f"ocr.workers must be >= 1, got {ocr_workers}")
    ocr_dpi_scale = int(ocr.get("dpi_scale", 4))
    if ocr_dpi_scale < 1:
        raise BookConfigError(f"ocr.dpi_scale must be >= 1, got {ocr_dpi_scale}")
    ocr_retries = int(ocr.get("retries", 1))
    if ocr_retries < 0:
        raise BookConfigError(f"ocr.retries must be >= 0, got {ocr_retries}")

    plugin_settings: dict[str, dict[str, Any]] = {}
    for k, v in pp.items():
        if k in {"plugins", "strict"}:
            continue
        if isinstance(v, dict):
            plugin_settings[k] = v
        else:
            warnings.append(f"postprocess.{k} ignored (expected table, got {type(v).__name__})")

    raw_plugins = pp.get("plugins", [])
    if not isinstance(raw_plugins, list) or not all(isinstance(p, str) for p in raw_plugins):
        raise BookConfigError("postprocess.plugins must be a list of strings")
    postprocess_plugins = list(raw_plugins)

    strict_plugins = bool(pp.get("strict", False))
    structure_headings = _load_headings(book_dir, struct)

    raw_heuristics = struct.get("book_heuristics", [])
    if not isinstance(raw_heuristics, list) or not all(isinstance(h, str) for h in raw_heuristics):
        raise BookConfigError("structure.book_heuristics must be a list of strings")
    book_heuristics: list[str] = [str(h) for h in raw_heuristics]
    # Per-book opt-in heuristics. Add new identifiers here and dispatch on
    # them from quire/structure/vision_based.py (or wherever applicable).
    _allowed_heuristics = {"imprint-fix"}
    unknown_heuristics = set(book_heuristics) - _allowed_heuristics
    if unknown_heuristics:
        raise BookConfigError(
            f"unknown book_heuristics: {sorted(unknown_heuristics)}. "
            f"Allowed: {sorted(_allowed_heuristics)}"
        )

    fonts_dir = repo_root / "data" / "fonts"
    embed_fonts: list[Path] = []
    missing_fonts: list[str] = []
    strict_fonts = bool(rd.get("strict_fonts", False))
    for font in rd.get("embed_fonts", []):
        font_path = fonts_dir / font
        if font_path.exists():
            embed_fonts.append(font_path)
        else:
            missing_fonts.append(str(font))
            warnings.append(f"font not found: {font_path}")
    if strict_fonts and missing_fonts:
        raise BookConfigError(
            f"strict_fonts=true but {len(missing_fonts)} font(s) missing under "
            f"{fonts_dir}: {missing_fonts}"
        )

    epub_filename = rd.get("epub_filename") or f"{slug}.epub"
    output_formats = [str(f).lower() for f in rd.get("formats", ["epub"])]
    invalid_formats = sorted(set(output_formats) - SUPPORTED_FORMATS)
    if invalid_formats:
        raise BookConfigError(
            f"unsupported render format(s) in {toml_path}: {', '.join(invalid_formats)}"
        )
    output_formats = [
        "markdown" if f == "md" else "text" if f == "txt" else f
        for f in output_formats
    ]

    return BookConfig(
        slug=slug,
        title=book.get("title", slug),
        author=book.get("author", "Unknown"),
        language=book.get("language", "en"),
        book_dir=book_dir,
        artifact_dir=artifact_dir,
        pdf_path=pdf_path,
        cover_pdf_page=cover_page,
        ocr_engine=ocr_engine,
        ocr_languages=ocr_languages,
        ocr_workers=ocr_workers,
        ocr_dpi_scale=ocr_dpi_scale,
        ocr_retries=ocr_retries,
        postprocess_plugins=postprocess_plugins,
        plugin_settings=plugin_settings,
        structure_headings=structure_headings,
        book_heuristics=book_heuristics,
        strict_plugins=strict_plugins,
        embed_fonts=embed_fonts,
        missing_fonts=missing_fonts,
        strict_fonts=strict_fonts,
        include_corrections_md=bool(rd.get("include_corrections_md", True)),
        epub_filename=epub_filename,
        output_formats=output_formats,
        raw=cfg,
        warnings=warnings,
    )


def _load_headings(book_dir: Path, struct: dict[str, Any]) -> list[tuple[str, int]]:
    """Load optional chapter/section headings from book configuration.

    Prefer a ``headings_module = "headings"`` entry for larger books; it should
    export ``HEADINGS = [("Title", 1), ...]``. Small books can use inline TOML:
    ``headings = [{ title = "Chapter 1", level = 1 }]``.

    SECURITY: ``headings_module`` executes arbitrary Python from the book
    folder. Only ingest book directories you trust.
    """
    out: list[tuple[str, int]] = []
    module_name = struct.get("headings_module")
    if module_name:
        module_path = (book_dir / f"{module_name}.py").resolve()
        if not module_path.exists():
            raise FileNotFoundError(f"headings module not found: {module_path}")
        # Refuse path escape (`..` or absolute) into other folders.
        if book_dir not in module_path.parents:
            raise BookConfigError(
                f"headings module {module_path} must live under {book_dir}"
            )
        spec = importlib.util.spec_from_file_location(
            f"quire_book_headings_{book_dir.name}", module_path
        )
        if spec is None or spec.loader is None:
            raise BookConfigError(f"could not load headings module: {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        candidate = getattr(mod, "HEADINGS", [])
        for item in candidate:
            title, level = item
            out.append((str(title), int(level)))
    for item in struct.get("headings", []):
        if isinstance(item, dict):
            title = item.get("title")
            level = item.get("level", 1)
        else:
            title, level = item
        if title:
            out.append((str(title), int(level)))
    return out


def find_book_dir(repo_root: Path, slug_or_path: str) -> Path:
    """Resolve a book reference to a folder.

    Accepts:
      - an absolute path to a book folder
      - a path relative to ``repo_root`` (e.g. ``books/<slug>``)
      - the slug of a folder under ``<repo>/books/``
    """
    p = Path(slug_or_path)
    if p.is_absolute() and p.is_dir():
        return p.resolve()
    candidates = [
        (repo_root / slug_or_path),
        (repo_root / "books" / slug_or_path),
        p,
    ]
    for c in candidates:
        c = c.resolve()
        if c.is_dir():
            return c
    raise FileNotFoundError(f"book not found: {slug_or_path}")
