"""Plugin registry: maps plugin names from ``book.toml`` to callables."""

from __future__ import annotations

import importlib
import sys
from typing import Protocol


class PrePlugin(Protocol):
    def __call__(self, cfg, ocr_pages: list[dict]) -> None: ...


class PostPlugin(Protocol):
    def __call__(self, cfg, ocr_pages: list[dict]) -> None: ...


# Built-in plugins. Module path is relative to ``quire.postprocess``.
BUILTINS: dict[str, str] = {
    "vocabulary": ".vocabulary",
    "glossary_extract": ".glossary",
    "script_detect": ".script_detect",
    "mojibake_cleanup": ".mojibake",
    "common_ocr": ".common_ocr",
    "canonical_quran": ".canonical.quran_plugin",
}


def _load(name: str):
    module_path = BUILTINS.get(name, name)
    if module_path.startswith("."):
        return importlib.import_module(module_path, package="quire.postprocess")
    return importlib.import_module(module_path)


def run_pre_structure(cfg, ocr_pages: list[dict]) -> None:
    for name in cfg.postprocess_plugins:
        try:
            mod = _load(name)
        except ModuleNotFoundError as e:
            msg = f"plugin '{name}' not found: {e}"
            if getattr(cfg, "strict_plugins", False):
                raise ModuleNotFoundError(msg) from e
            print(f"[quire] WARN: {msg}", file=sys.stderr)
            continue
        fn = getattr(mod, "pre_structure", None)
        if fn:
            fn(cfg, ocr_pages)


def run_post_structure(cfg, ocr_pages: list[dict]) -> None:
    for name in cfg.postprocess_plugins:
        try:
            mod = _load(name)
        except ModuleNotFoundError as e:
            msg = f"plugin '{name}' not found: {e}"
            if getattr(cfg, "strict_plugins", False):
                raise ModuleNotFoundError(msg) from e
            print(f"[quire] WARN: {msg}", file=sys.stderr)
            continue
        fn = getattr(mod, "post_structure", None)
        if fn:
            fn(cfg, ocr_pages)
