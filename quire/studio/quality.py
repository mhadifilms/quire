"""Publication readiness based on accounted content, not count ratios."""

from __future__ import annotations

from pathlib import Path

from .project import digest, file_hash, project_path, source_signature


def check(data: dict, *, target: str | None = None, root: Path | None = None) -> dict:
    findings = []

    def add(code, message, *, node=None, page=None, severity="review"):
        findings.append(
            {
                "code": code,
                "message": message,
                "node_id": node["id"] if node else None,
                "page": node["page"] if node else page,
                "severity": severity,
            }
        )

    expected = set(range(1, data["source"]["page_count"] + 1))
    actual = {p["number"] for p in data["pages"]}
    for pno in sorted(expected - actual):
        add("missing_page", "Page has not been extracted", page=pno, severity="error")
    if len(actual) != len(data["pages"]) or actual - expected:
        add("invalid_pages", "Page inventory does not match the source", severity="error")
    if root:
        source = project_path(root, data["source"]["path"])
        if not source.is_file() or file_hash(source) != data["source"]["sha256"]:
            add("source_changed", "Original PDF is missing or changed", severity="error")
    if data["language"]["language"] == "und" or data["language"]["confidence"] < 0.75:
        add("language", "Confirm the book's source language")
    by_id = {n["id"]: n for n in data["nodes"]}
    linked_notes = set()
    for page in data["pages"]:
        if page["state"] != "complete":
            add(
                "extraction_failed",
                page.get("error", "Page extraction is incomplete"),
                page=page["number"],
                severity="error",
            )
        if not page.get("reviewed"):
            add(
                "page_unchecked", "Check that this page's content is fully accounted for", page=page["number"]
            )
        if page["state"] == "complete" and not any(n["page"] == page["number"] for n in data["nodes"]):
            if page.get("disposition") != "blank":
                add(
                    "empty_page",
                    "Nonblank source page has no manuscript content",
                    page=page["number"],
                    severity="error",
                )
        if root and not project_path(root, page["image"]).is_file():
            add("missing_image", "Source proof image is missing", page=page["number"], severity="error")
    for node in data["nodes"]:
        if node["status"] == "excluded":
            if not node.get("exclusion_reason"):
                add(
                    "unexplained_exclusion", "An excluded passage needs a reason", node=node, severity="error"
                )
            continue
        if node["status"] != "reviewed":
            add("unreviewed", "Review this passage against the scan", node=node)
        if node["kind"] != "figure" and not node["text"].strip() and not node.get("cells"):
            add(
                "empty_passage",
                "Passage has no text; restore it or record an exclusion",
                node=node,
                severity="error",
            )
        if ("\ufffd" in node["text"] or "[unclear" in node["text"].lower()) and not node.get(
            "uncertainty_note"
        ):
            add("unclear_text", "Resolve or explicitly annotate the unclear source passage", node=node)
        if node["kind"] == "figure":
            if not node.get("alt", "").strip():
                add("figure_description", "Add an illustration description", node=node)
            if root and (not node.get("asset") or not project_path(root, node["asset"]).is_file()):
                add("figure_missing", "Illustration file is missing", node=node, severity="error")
        if node["kind"] == "table" and (not node.get("cells") or len({len(r) for r in node["cells"]}) != 1):
            add(
                "table_shape", "Table needs a consistent set of rows and columns", node=node, severity="error"
            )
        for ref in node.get("note_ids", []):
            note = by_id.get(ref)
            if not note or note["kind"] != "footnote" or note["status"] == "excluded":
                add("broken_note", "Footnote link has no included target", node=node, severity="error")
            else:
                linked_notes.add(ref)
        if target:
            entry = node.get("translations", {}).get(target)
            if not entry or not entry.get("text", "").strip():
                add("untranslated", "Translation is missing", node=node)
            elif entry.get("source_hash") != source_signature(node):
                add("stale_translation", "Source changed after this translation", node=node)
            elif entry.get("glossary_hash") != digest(data["glossaries"].get(target, {})):
                add("stale_terminology", "Terminology changed after this translation", node=node)
            elif entry.get("status") != "reviewed":
                add("translation_review", "Review this translation", node=node)
            if (
                entry
                and ("[unclear" in entry.get("text", "").lower() or entry.get("warnings"))
                and not entry.get("uncertainty_note")
            ):
                add("translation_uncertainty", "Translation contains unresolved uncertainty", node=node)
            if node["kind"] == "table" and entry:
                if [len(row) for row in entry.get("cells", [])] != [
                    len(row) for row in node.get("cells", [])
                ]:
                    add(
                        "translated_table_shape",
                        "Translated table must preserve every cell",
                        node=node,
                        severity="error",
                    )
    for node in data["nodes"]:
        if node["kind"] == "footnote" and node["status"] != "excluded" and node["id"] not in linked_notes:
            add("unlinked_note", "Connect this footnote to its passage", node=node)
    errors = sum(f["severity"] == "error" for f in findings)
    active = [n for n in data["nodes"] if n["status"] != "excluded"]
    return {
        "ready": not findings,
        "state": "blocked" if errors else "needs_review" if findings else "ready",
        "target": target,
        "errors": errors,
        "review_items": len(findings) - errors,
        "pages": len(actual & expected),
        "expected_pages": len(expected),
        "passages": len(active),
        "reviewed_passages": sum(n["status"] == "reviewed" for n in active),
        "review_seconds": round(sum(n.get("review_seconds", 0) for n in data["nodes"]), 1),
        "recognition_accuracy": None,
        "recognition_accuracy_reason": "Requires a verified reference transcription",
        "findings": findings,
    }
