"""Loopback-only review server; external translation requires an explicit action."""

from __future__ import annotations

import json
import mimetypes
import secrets
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from ..io_utils import atomic_write_bytes
from . import project
from .extract import import_pdf
from .publish import publish
from .quality import check
from .translate import translate

STATIC = Path(__file__).parent / "web"


class ReviewServer(ThreadingHTTPServer):
    executor: ThreadPoolExecutor


def make_server(library: Path, *, selected: Path | None = None, port: int = 0) -> ReviewServer:
    library = library.resolve()
    library.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(32)
    jobs: dict = {}
    executor = ThreadPoolExecutor(max_workers=1)
    job_lock = threading.Lock()

    def projects():
        roots = [selected] if selected else [p.parent for p in library.glob("*/project.json")]
        result = {}
        for root in roots:
            if root is None:
                continue
            try:
                data = project.load(root)
                result[data["id"]] = (root, data)
            except (OSError, ValueError, KeyError):
                continue
        return result

    def enqueue(action):
        with job_lock:
            if any(job["state"] in {"queued", "running"} for job in jobs.values()):
                raise ValueError("Another operation is running. Wait for it to finish.")
            job_id = str(uuid.uuid4())
            jobs[job_id] = {"id": job_id, "state": "queued"}

        def run():
            jobs[job_id]["state"] = "running"
            try:
                result = action(jobs[job_id])
                jobs[job_id].update(state="complete", result=result)
            except Exception as exc:
                jobs[job_id].update(state="failed", error=str(exc)[:500])

        executor.submit(run)
        return {"job_id": job_id}

    class Handler(BaseHTTPRequestHandler):
        server_version = "Quire"

        def log_message(self, fmt, *args):
            return

        def _headers(self, status=200, content_type="application/json", *, size=None):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; object-src 'none'; frame-ancestors 'none'",
            )
            if size is not None:
                self.send_header("Content-Length", str(size))
            self.end_headers()

        def _json(self, value, status=200):
            raw = json.dumps(value, ensure_ascii=False).encode()
            self._headers(status, size=len(raw))
            self.wfile.write(raw)

        def _safe_host(self):
            expected = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            if self.headers.get("Host") not in expected:
                raise PermissionError("Use the local Quire address printed in the terminal")
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in expected}:
                raise PermissionError("Cross-origin requests are not allowed")

        def _file(self, path, *, attachment=False):
            if not path.is_file():
                raise FileNotFoundError("File not found")
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(path)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cross-Origin-Resource-Policy", "same-origin")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'none'; style-src 'self' 'unsafe-inline'; img-src 'self'; font-src 'self'; sandbox allow-same-origin",
            )
            if attachment:
                self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as src:
                import shutil

                shutil.copyfileobj(src, self.wfile)

        def do_GET(self):
            try:
                self._safe_host()
                path = unquote(urlparse(self.path).path)
                parts = path.strip("/").split("/")
                if path == "/":
                    raw = (STATIC / "index.html").read_text().replace("__TOKEN__", token).encode()
                    self._headers(content_type="text/html; charset=utf-8", size=len(raw))
                    self.wfile.write(raw)
                elif path in {"/app.js", "/style.css"}:
                    self._file(STATIC / path[1:])
                elif path == "/api/projects":
                    self._json(
                        [
                            {k: data[k] for k in ("id", "title", "author", "language", "updated_at")}
                            | {"quality": check(data), "page_count": data["source"]["page_count"]}
                            for _, data in projects().values()
                        ]
                    )
                elif parts[:2] == ["api", "jobs"] and len(parts) == 3:
                    self._json(
                        jobs.get(
                            parts[2],
                            {
                                "state": "failed",
                                "error": "Operation is no longer running; reopen the project to resume",
                            },
                        )
                    )
                elif parts[:2] == ["api", "projects"] and len(parts) >= 3:
                    root, data = projects()[parts[2]]
                    if len(parts) == 3:
                        target = parse_qs(urlparse(self.path).query).get("target", [None])[0]
                        self._json({"project": data, "quality": check(data, target=target or None)})
                    elif len(parts) == 5 and parts[3] == "pages":
                        number = int(parts[4].removesuffix(".png"))
                        page = next(p for p in data["pages"] if p["number"] == number)
                        self._file(project.project_path(root, page["image"]))
                    elif len(parts) >= 6 and parts[3] == "editions":
                        release = next(r for r in data["releases"] if r["id"] == parts[4])
                        folder = project.project_path(root, release["directory"])
                        filename = "/".join(parts[5:])
                        manifest = json.loads((folder / "release.json").read_text())
                        if filename not in manifest["checksums"] and filename != "release.json":
                            raise FileNotFoundError("Edition file not found")
                        self._file(
                            project.project_path(folder, filename),
                            attachment=Path(filename).suffix in {".pdf", ".epub", ".md", ".txt"},
                        )
                    else:
                        raise FileNotFoundError("Unknown project resource")
                else:
                    raise FileNotFoundError("Not found")
            except PermissionError as exc:
                self._json({"error": str(exc)}, 403)
            except (KeyError, StopIteration, FileNotFoundError):
                self._json({"error": "Project or file not found"}, 404)
            except (ValueError, OSError) as exc:
                self._json({"error": str(exc)}, 400)

        def do_POST(self):
            try:
                self._safe_host()
                if not secrets.compare_digest(self.headers.get("X-Quire-Token", ""), token):
                    raise PermissionError("Refresh Quire to renew this window's editing session")
                size = int(self.headers.get("Content-Length", "0"))
                path = unquote(urlparse(self.path).path)
                if size <= 0 or size > (150 * 1024 * 1024 if path == "/api/import" else 2 * 1024 * 1024):
                    raise ValueError("Request is empty or too large")
                raw = self.rfile.read(size)
                if path == "/api/import":
                    if selected:
                        raise ValueError("Open the library view to import another book")
                    if not raw.startswith(b"%PDF-"):
                        raise ValueError("Choose a PDF file")
                    name = Path(unquote(self.headers.get("X-Filename", "book.pdf"))).stem[:120]
                    root = library / ("book-" + uuid.uuid4().hex[:12])
                    root.mkdir()
                    atomic_write_bytes(root / "source.pdf", raw)

                    def importing(job):
                        result = import_pdf(
                            root / "source.pdf",
                            root,
                            title=name,
                            language=self.headers.get("X-Language", "auto"),
                            progress=lambda p, total, state: job.update(progress=f"Page {p} of {total}"),
                        )
                        return {"project_id": result["id"]}

                    self._json(enqueue(importing), 202)
                    return
                parts = path.strip("/").split("/")
                if parts[:2] != ["api", "projects"] or len(parts) != 4:
                    raise FileNotFoundError("Unknown action")
                root, data = projects()[parts[2]]
                body = json.loads(raw)
                action = parts[3]
                revision = body.get("revision")
                if type(revision) is not int:
                    raise ValueError("A project revision is required")
                if action == "edit":
                    result = project.edit(
                        root, body["node_id"], body["changes"], revision=revision, target=body.get("target")
                    )
                elif action == "move":
                    result = project.move(root, body["node_id"], body["offset"], revision=revision)
                elif action == "join":
                    result = project.join_next(root, body["node_id"], revision=revision)
                elif action == "undo":
                    result = project.undo(root, revision=revision)
                elif action == "review-page":
                    result = project.review_page(root, body["page"], revision=revision)
                elif action == "add":
                    result = project.add_passage(
                        root, body["page"], body["bbox"], body["text"], body["kind"], revision=revision
                    )
                elif action == "metadata":
                    result = project.metadata(root, body["changes"], revision=revision)
                elif action == "glossary":
                    result = project.set_glossary(root, body["target"], body["terms"], revision=revision)
                elif action in {"publish", "translate", "retry"}:
                    if data["revision"] != revision:
                        raise ValueError("Project changed; refresh before starting this operation")

                    def operation(job):
                        if action == "retry":
                            settings = data["jobs"]["extract"]["settings"]
                            result = import_pdf(
                                root / data["source"]["path"],
                                root,
                                language=body.get("language", settings["language"]),
                                engine=settings["engine"],
                                dpi=settings["dpi"],
                                progress=lambda p, total, state: job.update(progress=f"Page {p} of {total}"),
                            )
                            return {"project_id": result["id"]}
                        if action == "translate":
                            return translate(
                                root,
                                body["target"],
                                token_budget=int(body.get("token_budget", 100000)),
                                progress=lambda state: job.update(
                                    progress=f"{state['translated']} passages translated"
                                ),
                            )
                        return publish(
                            root,
                            target=body.get("target"),
                            bilingual=bool(body.get("bilingual")),
                            template=body.get("template", "reading"),
                            draft=bool(body.get("draft")),
                            formats=body.get("formats", ["pdf", "epub", "html", "markdown"]),
                        )

                    self._json(enqueue(operation), 202)
                    return
                else:
                    raise FileNotFoundError("Unknown action")
                self._json({"project": result, "quality": check(result, target=body.get("target"))})
            except PermissionError as exc:
                self._json({"error": str(exc)}, 403)
            except (KeyError, StopIteration, FileNotFoundError):
                self._json({"error": "Project or required field not found"}, 404)
            except (ValueError, TypeError, OSError) as exc:
                self._json({"error": str(exc)}, 409 if "changed" in str(exc) else 400)

    server = ReviewServer(("127.0.0.1", port), Handler)
    server.daemon_threads = True
    server.executor = executor
    return server


def serve(library: Path, *, selected: Path | None = None, port: int = 8765, open_browser: bool = True):
    server = make_server(library, selected=selected, port=port)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Quire workspace: {url}", flush=True)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.executor.shutdown(wait=True)
