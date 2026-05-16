"""End-to-end QC orchestrator.

Walks per-page text payloads, renders each PDF page to an image,
calls the configured engine, validates the response, and accumulates
corrections. Results are written to ``cfg.qc_fixes_path`` via
:mod:`quire.qc.writer`.

Concurrency is bounded by ``cfg.qc_settings.concurrency`` using a
thread pool (HTTP-bound calls). Each per-page result is cached on
disk so repeated runs skip pages that have already been processed.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import BookConfig
from ..io_utils import atomic_write_text
from ..logging_utils import log_event
from .engine import (
    GeminiQCEngine,
    correct_page_to_corrections,
)
from .models import Correction, CostInfo, PageText, QCResult
from .page_images import render_pages_for_qc
from .writer import merge_corrections


@dataclass
class QCRunInputs:
    """Inputs needed to run QC for one book."""

    page_texts: list[PageText]
    # callable: pdf_pno -> PNG bytes. Allows the pipeline to pass already-
    # rendered images; the CLI path defaults to rendering on demand.
    image_provider: Callable[[int], bytes]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _text_sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _cache_key(model: str, image_bytes: bytes, page_text: str) -> str:
    return f"{model}__{_sha256(image_bytes)[:32]}__{_text_sha256(page_text)[:32]}"


def _cache_dir(cfg: BookConfig) -> Path:
    p = cfg.caches_dir / "qc"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _cache_path(cfg: BookConfig, key: str) -> Path:
    return _cache_dir(cfg) / f"{key}.json"


def _serialize_corrections(corrections: Iterable[Correction]) -> list[dict[str, Any]]:
    return [
        {
            "find": c.find,
            "replace": c.replace,
            "confidence": c.confidence,
            "reason": c.reason,
            "page": c.page,
        }
        for c in corrections
    ]


def _deserialize_corrections(data: list[dict[str, Any]]) -> list[Correction]:
    out: list[Correction] = []
    for d in data:
        try:
            out.append(
                Correction(
                    find=str(d["find"]),
                    replace=str(d["replace"]),
                    confidence=d["confidence"],  # type: ignore[arg-type]
                    reason=str(d.get("reason", "")),
                    page=str(d.get("page", "")),
                )
            )
        except (KeyError, TypeError):
            continue
    return out


def _read_cache(path: Path) -> tuple[list[Correction], CostInfo] | None:
    if not path.exists():
        return None
    try:
        body = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    corrections = _deserialize_corrections(body.get("corrections", []))
    cost = CostInfo(
        input_tokens=int(body.get("cost", {}).get("input_tokens", 0) or 0),
        output_tokens=int(body.get("cost", {}).get("output_tokens", 0) or 0),
        usd=float(body.get("cost", {}).get("usd", 0.0) or 0.0),
    )
    return corrections, cost


def _write_cache(
    path: Path, corrections: list[Correction], cost: CostInfo
) -> None:
    body = {
        "corrections": _serialize_corrections(corrections),
        "cost": {
            "input_tokens": cost.input_tokens,
            "output_tokens": cost.output_tokens,
            "usd": cost.usd,
        },
    }
    atomic_write_text(path, json.dumps(body, ensure_ascii=False, indent=2))


# ---- page-range parsing ---------------------------------------------------

_RANGE_RE = re.compile(r"\s*(\d+)\s*(?:-\s*(\d+))?\s*")


def parse_page_spec(spec: str | None, *, available: set[int]) -> set[int]:
    """Parse a page spec like ``"1-50"`` or ``"3,18,105"`` or ``"all"``.

    Returns the subset of ``available`` PDF page numbers that match.
    ``None`` or ``"all"`` returns every available page.
    """
    if spec is None:
        return set(available)
    spec = spec.strip().lower()
    if spec in ("", "all", "*"):
        return set(available)
    selected: set[int] = set()
    for chunk in spec.split(","):
        m = _RANGE_RE.fullmatch(chunk)
        if not m:
            raise ValueError(f"bad page spec fragment: {chunk!r}")
        start = int(m.group(1))
        end = int(m.group(2)) if m.group(2) else start
        if end < start:
            start, end = end, start
        for n in range(start, end + 1):
            if n in available:
                selected.add(n)
    return selected


# ---- core orchestrator ----------------------------------------------------


def _estimate_image_tokens_from_size(image_bytes_size: int) -> int:
    """Cheap upper-bound estimate of vision tokens for cost projection.

    Gemini bills tile-based image input. ~258 tokens for a small image,
    up to ~3000 for very large. We use file size as a proxy.
    """
    if image_bytes_size < 50_000:
        return 600
    if image_bytes_size < 250_000:
        return 1500
    return 3000


def _estimate_text_tokens(text: str) -> int:
    # ~4 chars per token is the canonical English heuristic. Arabic and
    # other RTL scripts tokenize denser; we'll be slightly conservative.
    return max(64, len(text) // 3)


def estimate_cost(
    page_texts: list[PageText],
    *,
    model: str,
    avg_image_bytes: int = 150_000,
) -> tuple[int, float]:
    """Return ``(total_input_tokens, estimated_usd)`` for a dry run."""
    from .engine import MODEL_PRICING

    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        pricing = MODEL_PRICING["gemini-2.5-flash"]
    total_input = 0
    total_output = 0
    image_tokens = _estimate_image_tokens_from_size(avg_image_bytes)
    for pt in page_texts:
        total_input += _estimate_text_tokens(pt.plain_text) + image_tokens
        total_output += 200
    usd = (
        total_input / 1_000_000.0 * pricing.input_per_million
        + total_output / 1_000_000.0 * pricing.output_per_million
    )
    return total_input, usd


def run_qc(
    cfg: BookConfig,
    *,
    page_texts: list[PageText],
    image_provider: Callable[[int], bytes] | None = None,
    engine: GeminiQCEngine | None = None,
    pages_filter: set[int] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> QCResult:
    """Run QC for a single book.

    Parameters
    ----------
    cfg :
        Book configuration. ``cfg.qc_settings`` must be present (loader
        ensures this when ``[qc]`` is configured).
    page_texts :
        Per-page text payloads in PDF-page order, typically from
        :func:`quire.qc.page_text.extract_page_texts`.
    image_provider :
        Callable mapping ``pdf_pno`` -> PNG bytes. If ``None``, the
        runner renders images on demand from ``cfg.pdf_path``.
    engine :
        Optional pre-constructed engine. If ``None``, a
        :class:`GeminiQCEngine` is built from ``cfg.qc_settings``.
    pages_filter :
        Optional set of PDF pnos to restrict the run to.
    force :
        Ignore the response cache and re-call the model for every page.
    dry_run :
        If True, no API calls are made. Returns a result with the
        projected cost in ``cost.usd`` and zero corrections.
    """
    settings = cfg.qc_settings
    if settings is None or not settings.enabled:
        return QCResult(pages_skipped=len(page_texts))

    available = {pt.pdf_pno for pt in page_texts}
    if pages_filter is None:
        try:
            pages_filter = parse_page_spec(settings.pages, available=available)
        except ValueError as e:
            log_event("qc_bad_page_spec", error=str(e))
            pages_filter = available

    selected_pages = [pt for pt in page_texts if pt.pdf_pno in pages_filter]
    if not selected_pages:
        log_event("qc_empty_selection", pages=len(page_texts))
        return QCResult(pages_skipped=len(page_texts))

    result = QCResult()

    if dry_run:
        _, usd = estimate_cost(selected_pages, model=settings.engine)
        result.cost = CostInfo(usd=usd)
        result.pages_processed = 0
        result.pages_skipped = len(selected_pages)
        log_event(
            "qc_dry_run",
            pages=len(selected_pages),
            estimated_usd=round(usd, 4),
            model=settings.engine,
        )
        return result

    if image_provider is None:
        pnos = [pt.pdf_pno for pt in selected_pages]
        image_cache = render_pages_for_qc(cfg, pnos, dpi=settings.dpi)
        image_provider = lambda pno: image_cache[pno]  # noqa: E731

    if engine is None:
        engine = GeminiQCEngine(
            model=settings.engine,
            retries=settings.retry,
        )

    cost_lock = threading.Lock()
    abort_lock = threading.Lock()
    aborted: dict[str, Any] = {"flag": False, "reason": None}

    def process_page(pt: PageText) -> tuple[PageText, list[Correction], CostInfo, str | None, bool]:
        """Returns (page, corrections, cost, error, from_cache)."""
        try:
            image_bytes = image_provider(pt.pdf_pno)
        except Exception as e:
            return pt, [], CostInfo(), f"image render failed: {e}", False

        key = _cache_key(settings.engine, image_bytes, pt.plain_text)
        cache_path = _cache_path(cfg, key)
        if not force:
            cached = _read_cache(cache_path)
            if cached is not None:
                corrections, cost = cached
                return pt, corrections, cost, None, True

        with abort_lock:
            if aborted["flag"]:
                return pt, [], CostInfo(), "aborted", False

        corrections, eng_result = correct_page_to_corrections(
            engine,
            image_bytes=image_bytes,
            page_text=pt.plain_text,
            page_label=pt.label,
            min_confidence=settings.min_confidence,
        )
        if not eng_result.ok:
            return pt, [], eng_result.cost, eng_result.error, False

        try:
            _write_cache(cache_path, corrections, eng_result.cost)
        except OSError as e:
            log_event("qc_cache_write_failed", error=str(e))

        return pt, corrections, eng_result.cost, None, False

    log_event(
        "qc_start",
        slug=cfg.slug,
        pages=len(selected_pages),
        model=settings.engine,
        concurrency=settings.concurrency,
        max_cost_usd=settings.max_cost_usd,
    )

    try:
        with ThreadPoolExecutor(max_workers=max(1, settings.concurrency)) as pool:
            futures = [pool.submit(process_page, pt) for pt in selected_pages]
            for fut in as_completed(futures):
                pt, corrections, page_cost, error, from_cache = fut.result()
                if error is not None:
                    result.pages_failed += 1
                    log_event("qc_page_failed", page=pt.label, error=error)
                    continue
                result.corrections.extend(corrections)
                with cost_lock:
                    result.cost.add(page_cost)
                if from_cache:
                    result.pages_cached += 1
                else:
                    result.pages_processed += 1
                if (
                    settings.max_cost_usd > 0
                    and result.cost.usd >= settings.max_cost_usd
                    and not aborted["flag"]
                ):
                    with abort_lock:
                        aborted["flag"] = True
                        aborted["reason"] = (
                            f"max_cost_usd reached: ${result.cost.usd:.4f} >= "
                            f"${settings.max_cost_usd:.4f}"
                        )
                    log_event(
                        "qc_cost_cap_reached",
                        usd=round(result.cost.usd, 4),
                        cap=settings.max_cost_usd,
                    )
    finally:
        engine.close()

    if aborted["flag"]:
        result.aborted = True
        result.abort_reason = aborted["reason"]

    if result.corrections:
        write_result = merge_corrections(
            cfg.qc_fixes_path,
            result.corrections,
            preserve_human=settings.preserve_human,
        )
        result.fixes_written = write_result.written
        log_event(
            "qc_written",
            path=str(cfg.qc_fixes_path),
            total=write_result.written,
            added=write_result.added,
            preserved_human=write_result.preserved_human,
        )

    log_event(
        "qc_done",
        slug=cfg.slug,
        corrections=len(result.corrections),
        processed=result.pages_processed,
        cached=result.pages_cached,
        failed=result.pages_failed,
        usd=round(result.cost.usd, 4),
        aborted=result.aborted,
    )
    return result
