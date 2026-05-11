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
