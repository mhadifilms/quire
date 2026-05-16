"""Typed result models for the QC stage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Confidence = Literal["high", "medium", "low"]

_CONFIDENCE_ORDER: dict[str, int] = {"low": 0, "medium": 1, "high": 2}


def confidence_at_least(value: str, threshold: str) -> bool:
    """Return ``True`` when ``value`` is at least ``threshold`` strict."""
    return _CONFIDENCE_ORDER.get(value, -1) >= _CONFIDENCE_ORDER.get(threshold, 3)


@dataclass(frozen=True)
class Correction:
    """One validated proofreading correction proposed by the model.

    ``find`` MUST be an exact substring of the page text that was sent
    to the model. ``replace`` is the proposed substitute (plain text;
    no HTML tags). ``confidence`` reflects the model's stated certainty.
    ``page`` is the printed-page label (or ``pdf-N`` fallback) the
    correction was discovered on; informational only.
    """

    find: str
    replace: str
    confidence: Confidence
    reason: str
    page: str


@dataclass
class PageText:
    """Per-page text payload sent to the QC engine."""

    pdf_pno: int
    printed: int | None
    plain_text: str

    @property
    def label(self) -> str:
        if self.printed is not None:
            return str(self.printed)
        return f"pdf-{self.pdf_pno}"


@dataclass
class CostInfo:
    """Token usage / cost accounting for one model call."""

    input_tokens: int = 0
    output_tokens: int = 0
    usd: float = 0.0

    def add(self, other: CostInfo) -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.usd += other.usd


@dataclass
class QCResult:
    """Summary of a QC run."""

    corrections: list[Correction] = field(default_factory=list)
    pages_processed: int = 0
    pages_cached: int = 0
    pages_skipped: int = 0
    pages_failed: int = 0
    cost: CostInfo = field(default_factory=CostInfo)
    aborted: bool = False
    abort_reason: str | None = None
    fixes_written: int = 0

    def summary_line(self) -> str:
        parts = [
            f"{len(self.corrections)} correction(s)",
            f"{self.pages_processed} page(s)",
        ]
        if self.pages_cached:
            parts.append(f"{self.pages_cached} cached")
        if self.pages_skipped:
            parts.append(f"{self.pages_skipped} skipped")
        if self.pages_failed:
            parts.append(f"{self.pages_failed} failed")
        if self.cost.usd:
            parts.append(f"~${self.cost.usd:.4f}")
        if self.aborted and self.abort_reason:
            parts.append(f"ABORTED: {self.abort_reason}")
        return "qc: " + ", ".join(parts)
