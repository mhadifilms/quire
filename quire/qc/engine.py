"""Gemini Vision QC engine.

A thin wrapper over the Gemini ``generateContent`` REST endpoint that
sends one (image, page_text) pair per call and returns parsed
:class:`Correction` objects with cost accounting.

The engine deliberately depends only on :mod:`httpx` (no Google SDK)
to keep the install footprint small and to avoid SDK churn.

Authentication is via the ``GEMINI_API_KEY`` (or ``GOOGLE_API_KEY``)
environment variable, consistent with Google's official examples.
"""

from __future__ import annotations

import base64
import dataclasses
import os
import time
from dataclasses import dataclass, field
from typing import Any

from .models import CostInfo
from .parser import QCParseError, parse_corrections
from .prompts import SYSTEM_PROMPT, build_user_prompt


@dataclass(frozen=True)
class ModelPricing:
    """Per-million-token USD pricing for input and output."""

    input_per_million: float
    output_per_million: float


# Pricing snapshot at time of implementation. Update via config if rates change.
MODEL_PRICING: dict[str, ModelPricing] = {
    "gemini-2.5-flash": ModelPricing(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash-lite": ModelPricing(input_per_million=0.10, output_per_million=0.40),
    "gemini-2.0-flash": ModelPricing(input_per_million=0.10, output_per_million=0.40),
}

DEFAULT_MODEL = "gemini-2.5-flash"

GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
DEFAULT_TIMEOUT_S = 90.0


class QCEngineError(RuntimeError):
    """Raised for non-recoverable engine failures."""


@dataclass
class EnginePageResult:
    """Outcome of a single :meth:`GeminiQCEngine.correct_page` call."""

    raw_response: dict[str, Any] | None
    cost: CostInfo
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.raw_response is not None


@dataclass
class GeminiQCEngine:
    """Synchronous Gemini REST client tailored to per-page QC.

    Parameters
    ----------
    model :
        Gemini model id, e.g. ``"gemini-2.5-flash"``.
    api_key :
        API key. If ``None``, read from ``GEMINI_API_KEY`` or
        ``GOOGLE_API_KEY``. Missing key raises :class:`QCEngineError`.
    retries :
        Total retry attempts on HTTP 429 / 5xx (default 2).
    timeout_s :
        Per-request timeout in seconds (default 90).
    pricing :
        Optional override for per-million-token pricing. Defaults to
        the rate table in :data:`MODEL_PRICING`.
    """

    model: str = DEFAULT_MODEL
    api_key: str | None = None
    retries: int = 2
    timeout_s: float = DEFAULT_TIMEOUT_S
    pricing: ModelPricing | None = None
    _http_client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get(
                "GOOGLE_API_KEY"
            )
        if not self.api_key:
            raise QCEngineError(
                "GEMINI_API_KEY (or GOOGLE_API_KEY) environment variable is not set"
            )
        if self.pricing is None:
            self.pricing = MODEL_PRICING.get(self.model, MODEL_PRICING[DEFAULT_MODEL])

    # ----- response cost accounting -----

    def _cost_from_usage(self, usage: dict[str, Any]) -> CostInfo:
        in_tok = int(usage.get("promptTokenCount", 0) or 0)
        out_tok = int(usage.get("candidatesTokenCount", 0) or 0)
        pricing = self.pricing or MODEL_PRICING[DEFAULT_MODEL]
        usd = (
            in_tok / 1_000_000.0 * pricing.input_per_million
            + out_tok / 1_000_000.0 * pricing.output_per_million
        )
        return CostInfo(input_tokens=in_tok, output_tokens=out_tok, usd=usd)

    # ----- request payload -----

    def _build_payload(
        self, *, image_bytes: bytes, image_mime: str, user_text: str
    ) -> dict[str, Any]:
        return {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": image_mime,
                                "data": base64.b64encode(image_bytes).decode("ascii"),
                            }
                        },
                        {"text": user_text},
                    ],
                }
            ],
            "generation_config": {
                "response_mime_type": "application/json",
                "temperature": 0.1,
                "max_output_tokens": 4096,
            },
        }

    # ----- HTTP -----

    def _http(self):
        """Return the configured httpx Client (lazily imported)."""
        if self._http_client is not None:
            return self._http_client
        import httpx  # local import keeps the dep optional at import time

        self._http_client = httpx.Client(timeout=self.timeout_s)
        return self._http_client

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._http()
        url = f"{GEMINI_API_BASE}/models/{self.model}:generateContent"
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = client.post(
                    url,
                    params={"key": self.api_key},
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
            except Exception as e:  # network / timeout
                last_exc = e
                if attempt < self.retries:
                    time.sleep(delay)
                    delay *= 2
                    continue
                raise QCEngineError(f"HTTP transport error: {e}") from e
            if resp.status_code == 200:
                try:
                    return resp.json()
                except ValueError as e:
                    raise QCEngineError(f"non-JSON response body: {e}") from e
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < self.retries:
                time.sleep(delay)
                delay *= 2
                continue
            body = (resp.text or "")[:500]
            raise QCEngineError(
                f"Gemini API error {resp.status_code}: {body}"
            )
        if last_exc is not None:
            raise QCEngineError(f"exhausted retries: {last_exc}")
        raise QCEngineError("exhausted retries with no response")

    # ----- public API -----

    def correct_page(
        self,
        *,
        image_bytes: bytes,
        page_text: str,
        page_label: str,
        image_mime: str = "image/png",
    ) -> EnginePageResult:
        """Call the model on one page. Returns raw response + cost.

        The caller is responsible for parsing the model's text output
        into validated corrections (see :mod:`quire.qc.parser`); this
        method only handles transport + cost.
        """
        user_text = build_user_prompt(page_text=page_text, page_label=page_label)
        payload = self._build_payload(
            image_bytes=image_bytes, image_mime=image_mime, user_text=user_text
        )
        try:
            response = self._post(payload)
        except QCEngineError as e:
            return EnginePageResult(raw_response=None, cost=CostInfo(), error=str(e))
        usage = response.get("usageMetadata") or {}
        cost = self._cost_from_usage(usage)
        return EnginePageResult(raw_response=response, cost=cost)

    def close(self) -> None:
        if self._http_client is not None:
            try:
                self._http_client.close()
            except Exception:  # pragma: no cover - cleanup best-effort
                pass
            self._http_client = None


def extract_response_text(response: dict[str, Any]) -> str:
    """Pull the model's text body out of a generateContent response.

    Returns an empty string if the response shape is missing the
    expected fields (some safety-blocked responses omit ``content``).
    """
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    content = (candidates[0] or {}).get("content") or {}
    parts = content.get("parts") or []
    texts = [p.get("text", "") for p in parts if isinstance(p, dict)]
    return "".join(t for t in texts if isinstance(t, str))


def correct_page_to_corrections(
    engine: GeminiQCEngine,
    *,
    image_bytes: bytes,
    page_text: str,
    page_label: str,
    min_confidence: str,
):
    """Convenience: call ``correct_page`` and parse the result.

    Returns ``(corrections, EnginePageResult)``. On engine error,
    corrections is an empty list and result.error is set.
    """
    result = engine.correct_page(
        image_bytes=image_bytes, page_text=page_text, page_label=page_label,
    )
    if not result.ok:
        return [], result
    body = extract_response_text(result.raw_response or {})
    try:
        corrections = parse_corrections(
            body, page_text=page_text, page_label=page_label, min_confidence=min_confidence,
        )
    except QCParseError as e:
        return [], dataclasses.replace(result, error=str(e))
    return corrections, result
