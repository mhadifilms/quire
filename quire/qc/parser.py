"""JSON parser and validator for Gemini QC responses.

The parser is intentionally strict: any deviation from the documented
shape (see :mod:`quire.qc.prompts`) results in the malformed
correction being dropped, not the whole batch being rejected. This is
the secondary defence against model hallucination; the system prompt
is the primary defence.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .models import Correction, confidence_at_least

_VALID_CONFIDENCE = {"high", "medium", "low"}
_HTML_TAG_RE = re.compile(r"<[A-Za-z!/][^>]*>")
_MIN_FIND_LEN = 3
_MAX_FIND_LEN = 300


class QCParseError(ValueError):
    """Raised when the response body is not parseable at all."""


def _strip_json_fences(raw: str) -> str:
    """Strip optional ```json ... ``` fences the model sometimes emits."""
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _parse_raw(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    text = _strip_json_fences(raw)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise QCParseError(f"response is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise QCParseError(f"response root must be an object, got {type(data).__name__}")
    return data


def parse_corrections(
    raw: str | dict[str, Any],
    *,
    page_text: str,
    page_label: str,
    min_confidence: str = "medium",
) -> list[Correction]:
    """Parse and validate corrections from a model response body.

    Returns the list of accepted :class:`Correction` items; silently
    drops malformed entries. Raises :class:`QCParseError` only when the
    body is unparseable JSON or has the wrong root shape.
    """
    data = _parse_raw(raw)
    items = data.get("corrections")
    if items is None:
        return []
    if not isinstance(items, list):
        raise QCParseError(
            f"`corrections` must be a list, got {type(items).__name__}"
        )

    accepted: list[Correction] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        find = item.get("find")
        replace = item.get("replace")
        confidence = item.get("confidence")
        reason = item.get("reason", "")
        if not isinstance(find, str) or not isinstance(replace, str):
            continue
        if not isinstance(confidence, str):
            continue
        if not isinstance(reason, str):
            reason = ""
        if confidence not in _VALID_CONFIDENCE:
            continue
        if not confidence_at_least(confidence, min_confidence):
            continue
        if not _validate_correction(find, replace, page_text):
            continue
        key = (find, replace)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(
            Correction(
                find=find,
                replace=replace,
                confidence=confidence,  # type: ignore[arg-type]
                reason=reason[:160],
                page=page_label,
            )
        )
    return accepted


def _validate_correction(find: str, replace: str, page_text: str) -> bool:
    """Return ``True`` only when the proposed correction is safe to apply."""
    if not find or not replace:
        return False
    if find == replace:
        return False
    if not (_MIN_FIND_LEN <= len(find) <= _MAX_FIND_LEN):
        return False
    if len(replace) > _MAX_FIND_LEN * 2:
        return False
    if find not in page_text:
        return False
    if _HTML_TAG_RE.search(replace):
        return False
    if "\u0000" in replace or "\u0000" in find:
        return False
    if "\n" in find or "\n" in replace:
        return False
    return True
