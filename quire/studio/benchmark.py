"""Reference-based extraction benchmarks with transparent provenance."""

from __future__ import annotations

import json
import platform
import re
import time
import unicodedata
from pathlib import Path

from ..io_utils import atomic_write_text
from .extract import convert_docling, import_pdf
from .project import file_hash


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def errors(reference, hypothesis) -> dict:
    """Levenshtein alignment, O(hypothesis length) working memory."""
    previous = [(i, 0, i, 0) for i in range(len(hypothesis) + 1)]
    for r, expected in enumerate(reference, 1):
        current = [(r, r, 0, 0)]
        for h, observed in enumerate(hypothesis, 1):
            if expected == observed:
                current.append(previous[h - 1])
            else:
                sub, delete, insert = previous[h - 1], previous[h], current[h - 1]
                current.append(
                    min(
                        (sub[0] + 1, sub[1], sub[2], sub[3] + 1),
                        (delete[0] + 1, delete[1] + 1, delete[2], delete[3]),
                        (insert[0] + 1, insert[1], insert[2] + 1, insert[3]),
                    )
                )
        previous = current
    distance, deleted, inserted, substituted = previous[-1]
    return {
        "distance": distance,
        "deletions": deleted,
        "insertions": inserted,
        "substitutions": substituted,
        "reference_length": len(reference),
        "error_rate": distance / len(reference) if reference else None,
    }


def score(reference: str, hypothesis: str) -> dict:
    ref, hyp = normalize(reference), normalize(hypothesis)
    chars = errors(ref, hyp)
    words = errors(re.findall(r"\S+", ref), re.findall(r"\S+", hyp))
    return {
        "character_error_rate": chars["error_rate"],
        "word_error_rate": words["error_rate"],
        "omitted_words": words["deletions"],
        "reference_words": words["reference_length"],
        "word_alignment": words,
        "character_alignment": chars,
    }


def run(manifest: Path, output: Path, *, engine: str = "auto", baseline: Path | None = None) -> dict:
    config = json.loads(manifest.read_text("utf-8"))
    results = []
    output.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(config["cases"]):
        source = (manifest.parent / case["pdf"]).resolve()
        reference = case["reference"]
        provenance = case.get("reference_provenance", "unspecified")
        row = {"name": case["name"], "reference_provenance": provenance, "engine": engine}
        started = time.monotonic()
        try:
            actual_hash = file_hash(source)
            if case.get("source_sha256") and case["source_sha256"] != actual_hash:
                raise ValueError("Reference belongs to a different source PDF")
            root = output / "projects" / f"{index:03d}-{engine}"
            row["resumed_project"] = (root / "project.json").exists()
            data = import_pdf(
                source,
                root,
                engine="text" if engine == "docling" else engine,
                language=case.get("language", "auto"),
            )
            if engine == "docling":
                data = convert_docling(root)
            selected_pages = set(case.get("pages", range(1, data["source"]["page_count"] + 1)))
            hypothesis = "\n".join(n["text"] for n in data["nodes"] if n["page"] in selected_pages)
            row.update(score(reference, hypothesis))
            limits = case.get("limits", {})
            row["limits"] = limits
            row["regressions"] = [
                key for key, maximum in limits.items() if row.get(key) is not None and row[key] > maximum
            ]
            row.update(
                state="failed" if data["jobs"]["extract"]["state"] != "complete" else "measured",
                source_sha256=actual_hash,
                review_seconds=None,
                review_reason="No human review performed in an extraction benchmark",
                model_tokens=0,
                model_cost_usd=0,
                page_engines=[p.get("engine") for p in data["pages"]],
            )
        except Exception as exc:
            row.update(state="failed", error=str(exc)[:300])
        row["seconds"] = round(time.monotonic() - started, 3)
        results.append(row)
    report = {
        "schema": 1,
        "engine": engine,
        "cases": results,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "regressions": sum(bool(r.get("regressions")) for r in results),
        "measured": sum(r["state"] == "measured" for r in results),
        "failed": sum(r["state"] == "failed" for r in results),
    }
    if baseline:
        old = {r["name"]: r for r in json.loads(baseline.read_text())["cases"]}
        report["comparison"] = [
            {
                "name": r["name"],
                "character_error_change": r["character_error_rate"] - old[r["name"]]["character_error_rate"],
            }
            for r in results
            if r["name"] in old
            and r.get("character_error_rate") is not None
            and old[r["name"]].get("character_error_rate") is not None
        ]
    atomic_write_text(output / "benchmark.json", json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# Extraction benchmark",
        "",
        "Error rates compare extracted text to the supplied reference. Lower is better.",
        "",
        "| Case | Status | Character error | Word error | Omitted words | Seconds |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in results:

        def rate(key, row=row):
            return f"{100 * row[key]:.2f}%" if row.get(key) is not None else "unavailable"

        lines.append(
            f"| {row['name']} | {row['state']} | {rate('character_error_rate')} | {rate('word_error_rate')} | {row.get('omitted_words', 'unavailable')} | {row['seconds']} |"
        )
    atomic_write_text(output / "benchmark.md", "\n".join(lines) + "\n")
    return report
