"""Structured logging for Quire.

Behaviour is controlled by the ``QUIRE_LOG`` environment variable:

- ``QUIRE_LOG=text`` (or unset): human-readable ``[quire] message`` lines on
  stderr (the legacy format).
- ``QUIRE_LOG=json``: one JSON object per line on stderr.

Either way, the public entry point is :func:`log_event` which takes a short
event name and a free-form ``**fields`` payload. Code that just wants a
human message can keep calling the existing :func:`quire.pipeline.log`.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any


def _format_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


def log_event(event: str, **fields: Any) -> None:
    """Emit a single structured log event.

    The ``event`` is a short snake_case label (e.g. ``"ocr_start"``,
    ``"epubcheck_done"``). Extra keyword arguments are recorded as fields
    on that event.
    """
    fmt = os.environ.get("QUIRE_LOG", "text").lower()
    record: dict[str, Any] = {
        "ts": time.time(),
        "event": event,
    }
    for k, v in fields.items():
        record[k] = _format_value(v)

    if fmt == "json":
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr, flush=True)
    else:
        extras = " ".join(f"{k}={v}" for k, v in fields.items())
        prefix = f"[quire] {event}"
        if extras:
            print(f"{prefix}: {extras}", file=sys.stderr, flush=True)
        else:
            print(prefix, file=sys.stderr, flush=True)
