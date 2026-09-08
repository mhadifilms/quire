"""Commands for the integrated book workspace."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def run(args) -> int:
    from . import project

    try:
        command = args.cmd
        if command == "import":
            from .extract import import_pdf

            result = import_pdf(
                args.pdf,
                args.project,
                language=args.language,
                engine=args.engine,
                dpi=args.dpi,
                title=args.title,
                author=args.author,
                progress=lambda p, total, state: print(
                    f"Page {p}/{total}: {state['state']}", file=sys.stderr
                ),
            )
            print(f"Project: {Path(args.project).resolve()}\nPassages: {len(result['nodes'])}")
            return 1 if result["jobs"]["extract"]["state"] != "complete" else 0
        if command in {"studio", "review"}:
            from .server import serve

            selected = Path(args.project).resolve() if command == "review" else None
            serve(
                selected.parent if selected else Path(args.library),
                selected=selected,
                port=args.port,
                open_browser=not args.no_open,
            )
            return 0
        if command == "project-check":
            from .quality import check

            root = Path(args.project).resolve()
            result = check(project.load(root), target=args.language, root=root)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["ready"] else 1
        if command == "translate":
            from .translate import apply_response, export_request, translate

            root = Path(args.project).resolve()
            if args.export_request:
                result = export_request(root, args.language, Path(args.export_request))
                print(f"Exported {len(result['passages'])} aligned passages")
            elif args.import_response:
                if not args.request:
                    raise ValueError("Provide --request with the original translation request JSON")
                request = json.loads(Path(args.request).read_text())
                if request["target_language"] != args.language:
                    raise ValueError("The requested language does not match the response")
                result = apply_response(root, request, json.loads(Path(args.import_response).read_text()))
                print("Translation imported for review")
            else:
                result = translate(
                    root,
                    args.language,
                    model=args.model,
                    token_budget=args.token_budget,
                    progress=lambda state: print(f"{state['translated']} passages saved", file=sys.stderr),
                )
                print(json.dumps(result, indent=2))
                return 0 if result["state"] == "complete" else 1
            return 0
        if command == "publish":
            from .publish import publish

            result = publish(
                args.project,
                formats=args.format or ["pdf", "epub", "html", "markdown"],
                target=args.language,
                bilingual=args.bilingual,
                template=args.template,
                draft=args.draft,
            )
            print(
                json.dumps(
                    {
                        "state": result["state"],
                        "directory": str(Path(args.project).resolve() / result["directory"]),
                        "files": result["files"],
                        "validation": result["validation"],
                    },
                    indent=2,
                )
            )
            return 0
        if command == "bundle":
            print(project.pack(Path(args.project), Path(args.output)))
            return 0
        if command == "restore":
            print(project.unpack(Path(args.bundle), Path(args.destination)))
            return 0
        if command == "layout-import":
            from .extract import import_docling

            root = Path(args.project).resolve()
            result = import_docling(root, Path(args.document), revision=project.load(root)["revision"])
            print(f"Imported {len(result['nodes'])} layout passages for review")
            return 0
        if command == "layout":
            from .extract import convert_docling

            result = convert_docling(Path(args.project).resolve())
            print(f"Recovered {len(result['nodes'])} layout passages for review")
            return 0
        if command == "benchmark":
            from .benchmark import run as benchmark

            result = benchmark(
                Path(args.manifest),
                Path(args.output),
                engine=args.engine,
                baseline=Path(args.baseline) if args.baseline else None,
            )
            print(
                f"Measured {result['measured']} cases; {result['failed']} failed. Report: {Path(args.output).resolve() / 'benchmark.md'}"
            )
            return 1 if result["failed"] or result["regressions"] else 0
        raise ValueError("Unknown workspace command")
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"[quire] {exc}", file=sys.stderr)
        return 1


def register(sub):
    p = sub.add_parser("import", help="Import a PDF into a portable review project; rerun to resume.")
    p.add_argument("pdf")
    p.add_argument("project")
    p.add_argument("--language", default="auto")
    p.add_argument("--engine", choices=["auto", "text", "tesseract", "vision"], default="auto")
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title")
    p.add_argument("--author")
    p.set_defaults(func=run)
    for command in ("studio", "review"):
        p = sub.add_parser(command, help="Open the local visual book workspace.")
        p.add_argument(
            "project" if command == "review" else "library",
            nargs="?" if command == "studio" else None,
            **({"default": "books/workspace"} if command == "studio" else {}),
        )
        p.add_argument("--port", type=int, default=8765)
        p.add_argument("--no-open", action="store_true")
        p.set_defaults(func=run)
    p = sub.add_parser("project-check", help="Check source completeness, review, translation, and links.")
    p.add_argument("project")
    p.add_argument("--language")
    p.set_defaults(func=run)
    p = sub.add_parser("translate", help="Translate aligned passages or exchange a translation request.")
    p.add_argument("project")
    p.add_argument("--language", required=True)
    p.add_argument("--model", default="gemini-2.5-flash")
    p.add_argument("--token-budget", type=int, default=100000)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--export-request")
    mode.add_argument("--import-response")
    p.add_argument("--request")
    p.set_defaults(func=run)
    p = sub.add_parser("publish", help="Publish a reviewed edition or an explicitly labeled draft.")
    p.add_argument("project")
    p.add_argument("--language")
    p.add_argument("--bilingual", action="store_true")
    p.add_argument("--template", choices=["reading", "study", "large-print"], default="reading")
    p.add_argument("--format", choices=["pdf", "epub", "html", "markdown", "text"], action="append")
    p.add_argument("--draft", action="store_true")
    p.set_defaults(func=run)
    p = sub.add_parser("bundle", help="Create a checksummed portable project archive.")
    p.add_argument("project")
    p.add_argument("output")
    p.set_defaults(func=run)
    p = sub.add_parser("restore", help="Verify and restore a project into a new folder.")
    p.add_argument("bundle")
    p.add_argument("destination")
    p.set_defaults(func=run)
    p = sub.add_parser("layout-import", help="Import a Docling JSON layout before human review.")
    p.add_argument("project")
    p.add_argument("document")
    p.set_defaults(func=run)
    p = sub.add_parser("layout", help="Analyze a project with optional Docling layout models before review.")
    p.add_argument("project")
    p.set_defaults(func=run)
    p = sub.add_parser("benchmark", help="Measure extraction against verified reference text.")
    p.add_argument("manifest")
    p.add_argument("--output", required=True)
    p.add_argument("--engine", choices=["auto", "text", "tesseract", "vision", "docling"], default="auto")
    p.add_argument("--baseline")
    p.set_defaults(func=run)
