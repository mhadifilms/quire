"""Apply unique QC matches inside their source page, with a review report."""

from __future__ import annotations

import copy
import json
import sys

from ..io_utils import atomic_write_text
from ..studio.project import file_hash

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


def apply_scoped_corrections(cfg, chapters):
    """Return corrected copies; ambiguous, stale, and merged regions stay intact."""
    if not cfg.qc_fixes_path.exists():
        return chapters, []
    with cfg.qc_fixes_path.open("rb") as stream:
        corrections = tomllib.load(stream).get("correction", [])
    if not corrections:
        return chapters, []
    fingerprint = file_hash(cfg.pdf_path)
    corrected = copy.deepcopy(chapters)
    items = [item for chapter in corrected for item in [*chapter.elements, *chapter.footnotes]]
    report = []
    for correction in corrections:
        result = {**correction, "state": "needs_review"}
        report.append(result)
        if correction.get("source_sha256") != fingerprint:
            result["reason"] = "Source PDF changed or source fingerprint is missing"
            continue
        find, replacement = correction.get("find"), correction.get("replace")
        if not isinstance(find, str) or not find or not isinstance(replacement, str):
            result["reason"] = "Invalid correction text"
            continue
        candidates = [
            item
            for item in items
            if str(
                item.get("_printed") if item.get("_printed") is not None else f"pdf-{item.get('_pdf_pno')}"
            )
            == str(correction.get("page"))
            and find in item.get("text", "")
        ]
        if sum(item["text"].count(find) for item in candidates) != 1:
            result["reason"] = "Expected one exact occurrence on this page"
            continue
        item = candidates[0]
        if item.get("_continuation_pagebreaks"):
            result["reason"] = "Passage spans source pages; review in the visual workspace"
            continue
        before = item["text"]
        item["text"] = before.replace(find, replacement, 1)
        result.update(state="applied", before=before, after=item["text"])
    atomic_write_text(
        cfg.artifact_dir / "qc_scope_report.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    return corrected, report
