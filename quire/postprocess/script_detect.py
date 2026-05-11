"""Script detection plugin.

Tags Arabic-script blocks (and optionally paragraphs/headings/footnotes) with
the right ``lang=`` attribute so renderers can pick fonts (Nastaliq for Urdu,
Naskh for Arabic, etc.) and set ``dir`` correctly.

This plugin does not transform text — it sets a ``script_lang`` field on each
relevant element which the renderer picks up.

Default detection lives in :mod:`quire.script_utils` and covers Arabic,
Persian, Urdu, Hebrew, Greek, Devanagari, CJK, Cyrillic. Book authors can
override the per-script glyph sets via ``book.toml``::

    [postprocess.script_detect.scripts]
    fa = ["\u067E", "\u0686", "\u0698", "\u06A9", "\u06AF"]
    ur = ["\u0679", "\u0688", "\u06D2"]

For an "arabic"-kind block we still default to ``ar`` if no other script is
detected (the kind name implies the Arabic script family).
"""

from __future__ import annotations

import re

from ..script_utils import detect_script

DEFAULT_PERSIAN = "\u067e\u0686\u0698\u06a9\u06af\u06cc\u06c0"

KINDS_TO_TAG = ("arabic", "footnote", "paragraph", "heading")


def _build_detector(scripts: dict[str, str]) -> dict[str, re.Pattern]:
    return {tag: re.compile(f"[{glyphs}]") for tag, glyphs in scripts.items()}


def post_structure(cfg, ocr_pages: list[dict]) -> None:
    settings = cfg.plugin_config("script_detect") or {}
    scripts_cfg: dict[str, str] = settings.get("scripts") or {}
    user_detectors = _build_detector(scripts_cfg) if scripts_cfg else {}

    for page in ocr_pages:
        for el in page.get("elements", []):
            kind = el.get("kind")
            if kind not in KINDS_TO_TAG:
                continue
            text = el.get("text") or ""
            if not text:
                continue
            # User-defined detectors win.
            tagged: str | None = None
            for tag, det in user_detectors.items():
                if det.search(text):
                    tagged = tag
                    break
            if tagged is None:
                tagged = detect_script(text)
            # For arabic-kind blocks default to "ar" when no script was found.
            if tagged is None and kind == "arabic":
                tagged = "ar"
            if tagged is not None:
                el["script_lang"] = tagged
