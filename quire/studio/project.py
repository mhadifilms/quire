"""A portable, versioned manuscript with atomic edits and source provenance."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import shutil
import tempfile
import uuid
import zipfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ..io_utils import atomic_write_text
from ..languages import valid_language

SCHEMA = 1
KINDS = {"paragraph", "heading", "footnote", "quote", "poetry", "table", "figure", "caption"}
STATUSES = {"pending", "reviewed", "excluded"}
EDITABLE = {
    "text",
    "kind",
    "level",
    "language",
    "status",
    "exclusion_reason",
    "note_ids",
    "alt",
    "cells",
    "review_seconds",
    "uncertainty_note",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def digest(value) -> str:
    raw = (
        value if isinstance(value, bytes) else json.dumps(value, sort_keys=True, ensure_ascii=False).encode()
    )
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_signature(node: dict) -> str:
    return digest({key: node.get(key) for key in ("text", "kind", "cells", "alt", "note_ids")})


def project_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Project paths must stay inside the project")
    return path


@contextmanager
def locked(root: Path, name: str = ".project.lock"):
    """Kernel-held lock: released on process exit, including crashes."""
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / name).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def load(root: str | Path, *, verify_source: bool = False) -> dict:
    root = Path(root).resolve()
    data = json.loads((root / "project.json").read_text("utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError("Unsupported project version")
    if len({n["id"] for n in data["nodes"]}) != len(data["nodes"]):
        raise ValueError("Duplicate passage identifiers")
    for node in data["nodes"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", node["id"]) or node["kind"] not in KINDS:
            raise ValueError("Invalid passage identifier or type")
        if not isinstance(node["text"], str) or node["status"] not in STATUSES:
            raise ValueError("Invalid passage text or review state")
        valid_language(node["language"])
        box = node.get("bbox", [])
        if len(box) != 4 or any(not isinstance(v, (float, int)) or not math.isfinite(v) for v in box):
            raise ValueError("Invalid source region")
        if not 1 <= node["page"] <= data["source"]["page_count"]:
            raise ValueError("Passage points outside the source PDF")
        if node.get("asset"):
            project_path(root, node["asset"])
    source = project_path(root, data["source"]["path"])
    if verify_source and file_hash(source) != data["source"]["sha256"]:
        raise ValueError("Source PDF changed; import it as a new project to preserve approved edits")
    return data


def save(root: Path, data: dict) -> None:
    data["updated_at"] = now()
    atomic_write_text(root / "project.json", json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def create(source: Path, root: Path, *, title: str, author: str, language: dict, pages: int) -> dict:
    root = root.resolve()
    fingerprint = file_hash(source)
    with locked(root):
        if (root / "project.json").exists():
            data = load(root, verify_source=True)
            if fingerprint != data["source"]["sha256"]:
                raise ValueError("This project belongs to a different PDF; choose a new folder")
            return data
        target = root / "source.pdf"
        if source.resolve() != target:
            if target.exists():
                raise ValueError("Destination already has a source.pdf; choose an empty project folder")
            shutil.copy2(source, target)
        data = {
            "schema": SCHEMA,
            "id": str(uuid.uuid4()),
            "revision": 0,
            "title": title,
            "author": author,
            "language": language,
            "created_at": now(),
            "source": {
                "path": "source.pdf",
                "sha256": fingerprint,
                "page_count": pages,
                "filename": source.name,
            },
            "pages": [],
            "nodes": [],
            "history": [],
            "glossaries": {},
            "jobs": {},
            "releases": [],
        }
        save(root, data)
        return data


def mutate(root: str | Path, revision: int | None, operation):
    root = Path(root).resolve()
    with locked(root):
        data = load(root)
        if revision is not None and data["revision"] != revision:
            raise ValueError("This project changed in another window. Refresh before saving.")
        result = operation(data)
        data["revision"] += 1
        save(root, data)
        return data, result


def _record(data: dict, action: str, before: dict, after: dict) -> str:
    event_id = str(uuid.uuid4())
    data["history"].append(
        {
            "id": event_id,
            "at": now(),
            "action": action,
            "before": copy.deepcopy(before),
            "after": copy.deepcopy(after),
            "undone": False,
        }
    )
    return event_id


def _node(data: dict, node_id: str) -> dict:
    for node in data["nodes"]:
        if node["id"] == node_id:
            return node
    raise ValueError("Passage no longer exists")


def edit(root, node_id: str, changes: dict, *, revision: int, target: str | None = None):
    if "cells" in changes:
        cells = changes["cells"]
        if (
            not isinstance(cells, list)
            or not cells
            or not isinstance(cells[0], list)
            or not cells[0]
            or any(
                not isinstance(row, list)
                or len(row) != len(cells[0])
                or any(not isinstance(c, str) for c in row)
                for row in cells
            )
        ):
            raise ValueError("Table cells must be rectangular rows of text")
        changes = {**changes, "text": "\n".join(" | ".join(row) for row in cells)}

    def apply(data):
        node = _node(data, node_id)
        before = copy.deepcopy(node)
        if target:
            valid_language(target)
            if set(changes) - {"text", "status", "review_seconds", "cells", "uncertainty_note"}:
                raise ValueError("Only translation text and review status can be edited here")
            entry = node.setdefault("translations", {}).setdefault(target, {})
            if "text" in changes:
                if not isinstance(changes["text"], str) or not changes["text"].strip():
                    raise ValueError("Translation text cannot be empty")
                entry.update(
                    text=changes["text"],
                    source_hash=source_signature(node),
                    status="pending",
                    glossary_hash=digest(data["glossaries"].get(target, {})),
                    engine="human",
                    warnings=[],
                )
            if "cells" in changes:
                entry["cells"] = changes["cells"]
            if "uncertainty_note" in changes:
                entry["uncertainty_note"] = str(changes["uncertainty_note"])
            if changes.get("status") == "reviewed":
                if not entry.get("text") or entry.get("source_hash") != source_signature(node):
                    raise ValueError("Update the translation to match the current source before approving it")
                entry["status"] = "reviewed"
                entry["glossary_hash"] = digest(data["glossaries"].get(target, {}))
            elif "status" in changes:
                entry["status"] = "pending"
        else:
            if set(changes) - EDITABLE:
                raise ValueError("Unknown passage field")
            if "kind" in changes and changes["kind"] not in KINDS:
                raise ValueError("Unknown passage type")
            if "status" in changes and changes["status"] not in STATUSES:
                raise ValueError("Unknown review state")
            if "language" in changes:
                valid_language(changes["language"])
            if "text" in changes and not isinstance(changes["text"], str):
                raise ValueError("Passage text must be a string")
            if "level" in changes and (type(changes["level"]) is not int or not 1 <= changes["level"] <= 6):
                raise ValueError("Heading level must be between 1 and 6")
            if "note_ids" in changes:
                if not isinstance(changes["note_ids"], list) or any(
                    _node(data, ref)["kind"] != "footnote" for ref in changes["note_ids"]
                ):
                    raise ValueError("Notes must point to footnote passages")
            content_changes = set(changes) - {"status", "review_seconds", "exclusion_reason"}
            if content_changes:
                node["status"] = "pending"
            node.update(changes)
            if node["status"] == "excluded" and not node.get("exclusion_reason", "").strip():
                raise ValueError("Record why this passage is excluded")
        seconds = changes.get("review_seconds", 0)
        if not isinstance(seconds, (int, float)) or not 0 <= seconds <= 3600:
            raise ValueError("Invalid review duration")
        node["review_seconds"] = before.get("review_seconds", 0) + seconds
        return _record(data, "translation" if target else "edit", {node_id: before}, {node_id: node})

    return mutate(root, revision, apply)[0]


def move(root, node_id: str, offset: int, *, revision: int):
    def apply(data):
        node = _node(data, node_id)
        same_page = [n for n in data["nodes"] if n["page"] == node["page"]]
        index = same_page.index(node)
        if offset not in {-1, 1} or not 0 <= index + offset < len(same_page):
            raise ValueError("Passage is already at the page boundary")
        other = same_page[index + offset]
        before = {n["id"]: copy.deepcopy(n) for n in (node, other)}
        node["order"], other["order"] = other["order"], node["order"]
        data["nodes"].sort(key=lambda n: (n["page"], n["order"]))
        _record(data, "move", before, {n["id"]: n for n in (node, other)})

    return mutate(root, revision, apply)[0]


def undo(root, *, revision: int):
    def apply(data):
        events = [
            e
            for e in data["history"]
            if not e["undone"]
            and e["action"] in {"edit", "translation", "move", "page_review", "add", "join"}
        ]
        if not events:
            raise ValueError("No edits to undo")
        event = events[-1]

        def get(key):
            if key.startswith("@page:"):
                return next(p for p in data["pages"] if p["number"] == int(key.split(":")[1]))
            return _node(data, key)

        for node_id, after in event["after"].items():
            if get(node_id) != after:
                raise ValueError("This passage has newer changes; undo those first")
        for node_id, before in event["before"].items():
            current = get(node_id)
            current.clear()
            current.update(copy.deepcopy(before))
        if event["action"] == "add":
            added_ids = set(event["after"]) - set(event["before"])
            data["nodes"] = [n for n in data["nodes"] if n["id"] not in added_ids]
        data["nodes"].sort(key=lambda n: (n["page"], n["order"]))
        event["undone"] = True
        event["undone_at"] = now()

    return mutate(root, revision, apply)[0]


def join_next(root, node_id: str, *, revision: int):
    """Join a split passage while retaining both original regions and history."""

    def apply(data):
        node = _node(data, node_id)
        nodes = [n for n in data["nodes"] if n["page"] == node["page"] and n["status"] != "excluded"]
        index = nodes.index(node)
        if index + 1 >= len(nodes):
            raise ValueError("There is no following passage on this page")
        following = nodes[index + 1]
        if node["kind"] not in {"paragraph", "poetry", "quote"} or following["kind"] != node["kind"]:
            raise ValueError("Join passages of the same text type")
        before = {n["id"]: copy.deepcopy(n) for n in (node, following)}

        def regions(n):
            return n.get("source_regions", [{"id": n["id"], "bbox": n["bbox"], "original": n["original"]}])

        node["source_regions"] = regions(node) + regions(following)
        node["text"] += ("\n" if node["kind"] == "poetry" else " ") + following["text"]
        node["note_ids"] = list(dict.fromkeys(node.get("note_ids", []) + following.get("note_ids", [])))
        if following.get("uncertainty_note"):
            node["uncertainty_note"] = " ".join(
                filter(None, [node.get("uncertainty_note"), following["uncertainty_note"]])
            )
        a, b = node["bbox"], following["bbox"]
        node["bbox"] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
        node["status"] = "pending"
        following.update(status="excluded", exclusion_reason=f"Continuation joined into {node_id}")
        _record(data, "join", before, {n["id"]: n for n in (node, following)})

    return mutate(root, revision, apply)[0]


def review_page(root, number: int, *, revision: int):
    def apply(data):
        page = next((p for p in data["pages"] if p["number"] == number), None)
        if not page or page["state"] != "complete":
            raise ValueError("Extract this page successfully before reviewing it")
        before = copy.deepcopy(page)
        page["reviewed"] = True
        _record(data, "page_review", {f"@page:{number}": before}, {f"@page:{number}": page})

    return mutate(root, revision, apply)[0]


def metadata(root, changes: dict, *, revision: int):
    if set(changes) - {"title", "author", "language"}:
        raise ValueError("Unknown book field")
    if any(not isinstance(value, str) for value in changes.values()):
        raise ValueError("Book details must be text")

    def apply(data):
        for key in ("title", "author"):
            if key in changes:
                data[key] = changes[key].strip()
        if not data["title"]:
            raise ValueError("Book title is required")
        if "language" in changes:
            from ..languages import detect_language

            data["language"] = detect_language("", hint=changes["language"])
            for node in data["nodes"]:
                if node["language"] == "und":
                    node["language"] = changes["language"]
        _record(data, "metadata", {}, changes)

    return mutate(root, revision, apply)[0]


def add_passage(root, number: int, bbox: list, text: str, kind: str, *, revision: int):
    """Recover an omitted source region without altering original extraction."""
    root = Path(root).resolve()
    if kind not in KINDS or not isinstance(text, str):
        raise ValueError("Choose a passage type and enter its text")
    if len(bbox) != 4 or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in bbox):
        raise ValueError("Select a valid region on the source page")

    def apply(data):
        page = next((p for p in data["pages"] if p["number"] == number), None)
        if page is None or page["state"] != "complete":
            raise ValueError("Extract this page successfully before adding a passage")
        before_page = copy.deepcopy(page)
        if not 0 <= bbox[0] < bbox[2] <= page["width"] or not 0 <= bbox[1] < bbox[3] <= page["height"]:
            raise ValueError("Source region lies outside the page")
        node_id = f"p{number:04d}-u{uuid.uuid4().hex[:12]}"
        node = {
            "id": node_id,
            "page": number,
            "order": max((n["order"] for n in data["nodes"] if n["page"] == number), default=-1) + 1,
            "bbox": bbox,
            "text": text,
            "original": "",
            "kind": kind,
            "level": 2,
            "language": data["language"]["language"],
            "status": "pending",
            "translations": {},
            "note_ids": [],
            "warnings": ["Manually recovered source region"],
            "confidence": None,
            "review_seconds": 0,
        }
        if kind == "figure":
            from PIL import Image

            with Image.open(project_path(root, page["image"])) as image:
                crop = tuple(
                    x
                    / (page["width"] if i % 2 == 0 else page["height"])
                    * (image.width if i % 2 == 0 else image.height)
                    for i, x in enumerate(bbox)
                )
                node["asset"] = f"assets/{node_id}.png"
                (root / "assets").mkdir(exist_ok=True)
                image.crop(crop).save(root / node["asset"])
            node["alt"] = text
        data["nodes"].append(node)
        data["nodes"].sort(key=lambda n: (n["page"], n["order"]))
        page["reviewed"] = False
        _record(data, "add", {f"@page:{number}": before_page}, {node_id: node, f"@page:{number}": page})

    return mutate(root, revision, apply)[0]


def set_glossary(root, target: str, terms: dict, *, revision: int):
    valid_language(target)
    if not isinstance(terms, dict) or any(
        not isinstance(k, str) or not isinstance(v, str) or not k.strip() or not v.strip()
        for k, v in terms.items()
    ):
        raise ValueError("A glossary maps source terms to non-empty translated terms")

    def apply(data):
        _record(data, "glossary", {target: data["glossaries"].get(target, {})}, {target: terms})
        data["glossaries"][target] = terms

    return mutate(root, revision, apply)[0]


def pack(root: Path, output: Path) -> Path:
    """Snapshot only project-owned files; checksums catch partial transfers."""
    root, output = root.resolve(), output.resolve()
    if output.is_relative_to(root):
        raise ValueError("Save the project bundle outside the project directory")
    with locked(root):
        load(root, verify_source=True)
        files = [
            p
            for p in root.rglob("*")
            if p.is_file()
            and not p.is_symlink()
            and not any(part.startswith(".") for part in p.relative_to(root).parts)
        ]
        manifest = {str(p.relative_to(root)): file_hash(p) for p in files}
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
            temporary = Path(tmp.name)
        try:
            with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in files:
                    archive.write(path, path.relative_to(root))
                archive.writestr("bundle-manifest.json", json.dumps(manifest, sort_keys=True))
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
    return output


def unpack(bundle: Path, destination: Path, *, max_bytes: int = 8 * 1024**3) -> Path:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Restore into a new folder")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        root = Path(temporary) / "project"
        root.mkdir()
        with zipfile.ZipFile(bundle) as archive:
            infos = archive.infolist()
            if len(infos) > 100000 or sum(i.file_size for i in infos) > max_bytes:
                raise ValueError("Project bundle exceeds extraction limits")
            names = [i.filename for i in infos]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate bundle entries")
            for info in infos:
                path = PurePosixPath(info.filename)
                if (
                    path.is_absolute()
                    or ".." in path.parts
                    or "\\" in info.filename
                    or (info.external_attr >> 16) & 0o170000 == 0o120000
                ):
                    raise ValueError("Unsafe bundle entry")
            manifest = json.loads(archive.read("bundle-manifest.json"))
            if set(manifest) != set(names) - {"bundle-manifest.json"}:
                raise ValueError("Bundle manifest does not match its contents")
            for name, expected in manifest.items():
                target = project_path(root, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(name) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                if file_hash(target) != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
        load(root, verify_source=True)
        root.rename(destination)
    return destination
