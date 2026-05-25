"""Command-line entry point for ``quire``."""

from __future__ import annotations

import argparse
import sys

from .config import REPO_ROOT, find_book_dir, load_book_config


def _cmd_build(args: argparse.Namespace) -> int:
    from .pipeline import build_book

    book_dir = find_book_dir(REPO_ROOT, args.book)
    cfg = load_book_config(book_dir, repo_root=REPO_ROOT)
    if args.workers is not None:
        cfg.ocr_workers = max(1, int(args.workers))
    formats = args.format or None
    if args.all_formats:
        formats = ["epub", "html", "markdown", "text"]
    build_book(
        cfg,
        force_ocr=args.force_ocr,
        retry_failed=args.retry_failed,
        render_pages=args.render_pages,
        formats=formats,
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Validate a book.toml without running the pipeline."""
    book_dir = find_book_dir(REPO_ROOT, args.book)
    cfg = load_book_config(book_dir, repo_root=REPO_ROOT)
    print(f"book: {cfg.slug}", file=sys.stderr)
    print(f"  title:     {cfg.title}", file=sys.stderr)
    print(f"  author:    {cfg.author}", file=sys.stderr)
    print(f"  language:  {cfg.language}", file=sys.stderr)
    print(f"  pdf:       {cfg.pdf_path}", file=sys.stderr)
    print(f"  ocr:       engine={cfg.ocr_engine} langs={cfg.ocr_languages} "
          f"workers={cfg.ocr_workers}", file=sys.stderr)
    print(f"  plugins:   {cfg.postprocess_plugins}", file=sys.stderr)
    print(f"  formats:   {cfg.output_formats}", file=sys.stderr)
    print(f"  fonts:     {[p.name for p in cfg.embed_fonts]}", file=sys.stderr)
    if cfg.missing_fonts:
        print(f"  missing:   {cfg.missing_fonts}", file=sys.stderr)
    if cfg.warnings:
        for w in cfg.warnings:
            print(f"  warning:   {w}", file=sys.stderr)
    print("[quire] check: OK", file=sys.stderr)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    from .pipeline import _ocr_pages_for
    from .render.audit import run_audit

    book_dir = find_book_dir(REPO_ROOT, args.book)
    cfg = load_book_config(book_dir, repo_root=REPO_ROOT)
    cfg.ensure_artifact_dirs()
    pages = None if cfg.ocr_engine.lower() in {"text", "pdf", "pymupdf"} else _ocr_pages_for(cfg)
    run_audit(cfg, ocr_pages=pages)
    return 0


def _cmd_render_pages(args: argparse.Namespace) -> int:
    from .pipeline import render_page_images

    book_dir = find_book_dir(REPO_ROOT, args.book)
    cfg = load_book_config(book_dir, repo_root=REPO_ROOT)
    cfg.ensure_artifact_dirs()
    render_page_images(cfg, dpi=args.dpi)
    return 0


def _cmd_preprocess(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .extract.spreads import split_pdf

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"error: {pdf_path} does not exist", file=sys.stderr)
        return 2
    out_dir = Path(args.out)
    paths = split_pdf(
        pdf_path,
        out_dir,
        dpi=args.dpi,
        rotate=not args.no_rotate,
        detect_spreads=not args.no_split,
        outer_margin=args.outer_margin,
        inner_trim=args.inner_trim,
        image_format=args.format,
        image_quality=args.quality,
    )
    print(f"wrote {len(paths)} page(s) to {out_dir}")
    return 0


def _cmd_qc(args: argparse.Namespace) -> int:
    """Run AI-assisted page-by-page QC against the rendered EPUB content.

    Operates on the rendered chapter XHTML produced by the previous
    ``quire build`` run. If no rendered chapters exist yet, renders
    them in-memory without writing an EPUB.
    """
    import dataclasses as _dc

    from .config import QCSettings
    from .pipeline import (
        _build_rendered_chapters_for_qc,  # type: ignore[attr-defined]
    )
    from .qc import run_qc
    from .qc.page_text import build_page_map, extract_page_texts
    from .qc.runner import parse_page_spec

    book_dir = find_book_dir(REPO_ROOT, args.book)
    cfg = load_book_config(book_dir, repo_root=REPO_ROOT)
    cfg.ensure_artifact_dirs()

    settings = cfg.qc_settings or QCSettings(enabled=True)
    overrides: dict[str, object] = {"enabled": True}
    if args.engine:
        overrides["engine"] = args.engine
    if args.dpi is not None:
        overrides["dpi"] = args.dpi
    if args.concurrency is not None:
        overrides["concurrency"] = args.concurrency
    if args.max_cost_usd is not None:
        overrides["max_cost_usd"] = args.max_cost_usd
    if args.min_confidence is not None:
        overrides["min_confidence"] = args.min_confidence
    if args.pages is not None:
        overrides["pages"] = args.pages
    cfg.qc_settings = _dc.replace(settings, **overrides)  # type: ignore[arg-type]

    print(
        f"[quire qc] book={cfg.slug} model={cfg.qc_settings.engine} "
        f"dpi={cfg.qc_settings.dpi} concurrency={cfg.qc_settings.concurrency} "
        f"max_cost_usd={cfg.qc_settings.max_cost_usd}",
        file=sys.stderr,
    )

    rendered, chapters = _build_rendered_chapters_for_qc(cfg)
    page_map = build_page_map(chapters)
    page_texts = extract_page_texts(rendered, page_map)

    available = {pt.pdf_pno for pt in page_texts}
    try:
        pages_filter = parse_page_spec(cfg.qc_settings.pages, available=available)
    except ValueError as e:
        print(f"[quire qc] bad --pages: {e}", file=sys.stderr)
        return 2

    result = run_qc(
        cfg,
        page_texts=page_texts,
        pages_filter=pages_filter,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(f"[quire qc] {result.summary_line()}", file=sys.stderr)
    if result.aborted:
        return 1
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .batch import load_manifest, run_batch

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[quire] manifest not found: {manifest_path}", file=sys.stderr)
        return 2
    entries = load_manifest(manifest_path)
    status_path = Path(args.status) if args.status else None
    summary = run_batch(
        entries,
        workers=args.workers,
        status_path=status_path,
        fail_fast=args.fail_fast,
    )
    if status_path is not None:
        print(f"[quire batch] wrote {status_path}", file=sys.stderr)
    print(
        f"[quire batch] total={summary['total']} ok={summary['ok']} "
        f"failed={summary['failed']} elapsed={summary['elapsed_s']}s",
        file=sys.stderr,
    )
    return 0 if summary["failed"] == 0 else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="quire",
        description="Quire — reflowable EPUB pipeline for multi-script PDFs.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="Build EPUB from a book folder.")
    p_build.add_argument("book", help="Book slug or path to a books/<slug>/ folder.")
    p_build.add_argument(
        "--force-ocr",
        action="store_true",
        help="Re-run OCR even if a cache exists.",
    )
    p_build.add_argument(
        "--retry-failed",
        action="store_true",
        help="Resume from a partial OCR cache by re-OCRing only failed pages.",
    )
    p_build.add_argument(
        "--workers", type=int, default=None,
        help="Override OCR worker count (default: from [ocr].workers).",
    )
    p_build.add_argument(
        "--render-pages",
        action="store_true",
        help="Also render every PDF page as a PNG into books/<slug>/artifacts/page_images/.",
    )
    p_build.add_argument(
        "--format",
        action="append",
        choices=["epub", "html", "markdown", "md", "text", "txt"],
        help=(
            "Output format to write. Repeat for multiple formats. "
            "Defaults to [render].formats in book.toml, or epub."
        ),
    )
    p_build.add_argument(
        "--all-formats",
        action="store_true",
        help="Write epub, standalone HTML, Markdown, and plain text outputs.",
    )
    p_build.set_defaults(func=_cmd_build)

    p_check = sub.add_parser("check", help="Validate a book.toml without building.")
    p_check.add_argument("book", help="Book slug or folder path.")
    p_check.set_defaults(func=_cmd_check)

    p_audit = sub.add_parser("audit", help="Run audit & corrections report on built EPUB.")
    p_audit.add_argument("book", help="Book slug or folder path.")
    p_audit.set_defaults(func=_cmd_audit)

    p_render = sub.add_parser("render-pages", help="Render PDF pages as PNG images.")
    p_render.add_argument("book", help="Book slug or folder path.")
    p_render.add_argument("--dpi", type=int, default=160)
    p_render.set_defaults(func=_cmd_render_pages)

    p_pre = sub.add_parser(
        "preprocess",
        help="Detect rotation + two-page spreads in a PDF and split into single upright pages.",
    )
    p_pre.add_argument("pdf", help="Path to the source PDF.")
    p_pre.add_argument("--out", required=True,
                       help="Output directory for per-page images.")
    p_pre.add_argument("--dpi", type=int, default=200)
    p_pre.add_argument("--no-rotate", action="store_true",
                       help="Skip auto-rotation detection.")
    p_pre.add_argument("--no-split", action="store_true",
                       help="Skip spread-splitting (treat every PDF page as a single page).")
    p_pre.add_argument("--outer-margin", type=int, default=0,
                       help="Pixels to trim from the outer (non-binding) edge of each split page.")
    p_pre.add_argument("--inner-trim", type=int, default=0,
                       help="Pixels to trim from the binding edge to drop ring shadows.")
    p_pre.add_argument("--format", choices=["jpeg", "png"], default="jpeg")
    p_pre.add_argument("--quality", type=int, default=88,
                       help="JPEG quality 1-100 (ignored for PNG).")
    p_pre.set_defaults(func=_cmd_preprocess)

    p_qc = sub.add_parser(
        "qc",
        help="Run AI-assisted page-by-page QC and merge corrections into qc_fixes.toml.",
    )
    p_qc.add_argument("book", help="Book slug or folder path.")
    p_qc.add_argument(
        "--pages",
        default=None,
        help="Page range like '1-50' or '3,18,105' (default: all).",
    )
    p_qc.add_argument(
        "--engine", default=None,
        help="Override [qc].engine, e.g. gemini-2.5-flash-lite.",
    )
    p_qc.add_argument("--dpi", type=int, default=None)
    p_qc.add_argument("--concurrency", type=int, default=None)
    p_qc.add_argument("--max-cost-usd", type=float, default=None, dest="max_cost_usd")
    p_qc.add_argument(
        "--min-confidence",
        choices=["high", "medium", "low"],
        default=None,
        dest="min_confidence",
    )
    p_qc.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Estimate cost and exit without calling the model.",
    )
    p_qc.add_argument(
        "--force",
        action="store_true",
        help="Ignore the per-page response cache and re-call for every page.",
    )
    p_qc.set_defaults(func=_cmd_qc)

    p_batch = sub.add_parser("batch", help="Build many books from a manifest.")
    p_batch.add_argument("--manifest", required=True,
                         help="Path to a TOML manifest with [[book]] entries.")
    p_batch.add_argument("--workers", type=int, default=1,
                         help="Number of parallel book builds (default 1).")
    p_batch.add_argument("--status",
                         help="Optional path to write per-book status JSON.")
    p_batch.add_argument("--fail-fast", action="store_true",
                         help="Stop after the first failed book.")
    p_batch.set_defaults(func=_cmd_batch)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
