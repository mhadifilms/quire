"""Vocabulary plugin: load a JSON map of transliteration -> canonical script.

Used by inline-Arabic / inline-Hebrew / etc. substitution code in
``quire.structure.vision_based``. The plugin doesn't *do* the substitution
itself — it loads the dictionary into a shared registry that the structure
layer reads from.

Configure in ``book.toml``:

  [postprocess.vocabulary]
  path = "vocabulary.json"   # relative to book folder
  # OR
  module = "vocabulary"      # python module under the book folder
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import unicodedata
from pathlib import Path

# Shared registry that the structure code reads from.
ARABIC_VOCAB: dict[str, str] = {}


_OCR_SUBSTITUTIONS: tuple[tuple[str, str], ...] = (
    ("g", "q"),  # migat/tagsir -> miqat/taqsir
    ("q", "g"),
    ("r", "t"),  # migar -> miqat
    ("t", "r"),
    ("b", "h"),  # mubrim/ibram -> muhrim/ihram
    ("h", "b"),
    ("c", "e"),
    ("e", "c"),
    ("j", "g"),
    ("g", "j"),
)

_OCR_VARIANTS: dict[str, str] = {
    "migat": "miqat",
    "migar": "miqat",
    "migas": "miqat",
    "mubrim": "muhrim",
    "ibram": "ihram",
    "tagsir": "taqsir",
    "ragat": "rakat",
}


def _normalize_translit(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    table = str.maketrans({
        "\u02bf": "",
        "\u02bb": "",
        "ḥ": "h", "ṣ": "s", "ṭ": "t", "ẓ": "z", "ḍ": "d",
        "ġ": "g", "ḏ": "d", "ṯ": "t", "š": "s", "ž": "z",
        "ş": "s", "ğ": "g", "ç": "c", "ı": "i",
    })
    s = s.translate(table)
    return re.sub(r"[^a-z']", "", s.lower())


def lookup(transliteration: str) -> str | None:
    norm = _normalize_translit(transliteration)
    if not norm:
        return None
    if norm in ARABIC_VOCAB:
        return ARABIC_VOCAB[norm]
    no_apos = norm.replace("'", "")
    if no_apos and no_apos in ARABIC_VOCAB:
        return ARABIC_VOCAB[no_apos]
    if no_apos.startswith("al") and no_apos[2:] in ARABIC_VOCAB:
        return ARABIC_VOCAB[no_apos[2:]]
    return None


def _case_like(source: str, target: str) -> str:
    if source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def _variant_norms(norm: str) -> set[str]:
    variants = {_OCR_VARIANTS.get(norm, "")}
    variants.discard("")
    one_step: set[str] = set()
    for old, new in _OCR_SUBSTITUTIONS:
        if old in norm:
            one_step.add(norm.replace(old, new))
    variants.update(one_step)
    for v in one_step:
        for old, new in _OCR_SUBSTITUTIONS:
            if old in v:
                variants.add(v.replace(old, new))
    return variants


def correct_ocr_transliteration(word: str, preferred: set[str] | None = None) -> str | None:
    """Return a canonical transliteration for common cursive OCR mistakes.

    This is intentionally vocabulary-gated: we only correct to a token already
    present in the configured transliteration map (or the current PDF italic
    overlay), so ordinary English words are not rewritten just because they are
    close to an Arabic/Persian term.
    """
    norm = _normalize_translit(word)
    if len(norm) < 4:
        return None
    known_variant = _OCR_VARIANTS.get(norm)
    if known_variant and known_variant in ARABIC_VOCAB:
        return _case_like(word, known_variant)
    if norm in ARABIC_VOCAB:
        return None

    preferred_norms = {
        _normalize_translit(p) for p in (preferred or set())
        if p and _normalize_translit(p) in ARABIC_VOCAB
    }
    variants = _variant_norms(norm)
    for candidate in sorted(preferred_norms, key=len):
        if candidate in variants:
            return _case_like(word, candidate)
    matches = [v for v in variants if v in ARABIC_VOCAB]
    if len(matches) == 1:
        return _case_like(word, matches[0])
    return None


def update_with_pairs(pairs: dict[str, str]) -> None:
    """Extend the shared dictionary with auto-extracted glossary pairs.

    Hand-curated entries take precedence: ``setdefault`` only fills holes.
    """
    for tr, ar in pairs.items():
        if not tr or not ar:
            continue
        norm = _normalize_translit(tr)
        if not norm:
            continue
        ARABIC_VOCAB.setdefault(norm, ar)


def _load_from_json(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def _load_from_module(book_dir: Path, module_name: str) -> dict[str, str]:
    # Resolve the candidate path and refuse to load anything outside the
    # book folder. Without this check, a malicious ``book.toml`` could
    # set ``[postprocess.vocabulary] module = "../../../tmp/evil"`` and
    # execute arbitrary Python from outside the book directory. The
    # ``headings_module`` loader in :mod:`quire.config` applies the same
    # guard; keep them in sync.
    book_root = book_dir.resolve()
    file = (book_dir / f"{module_name}.py").resolve()
    if book_root not in file.parents:
        return {}
    if not file.exists():
        return {}
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:
        return {}
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    candidate = (
        getattr(mod, "ARABIC_VOCAB", None)
        or getattr(mod, "VOCAB", None)
        or getattr(mod, "VOCABULARY", None)
    )
    if not isinstance(candidate, dict):
        return {}
    return {str(k): str(v) for k, v in candidate.items() if isinstance(v, str)}


def pre_structure(cfg, ocr_pages: list[dict]) -> None:
    settings = cfg.plugin_config("vocabulary")
    if not settings:
        return
    path = settings.get("path")
    module_name = settings.get("module")
    pairs: dict[str, str] = {}
    if path:
        candidate = (cfg.book_dir / path).resolve()
        if candidate.suffix.lower() == ".json":
            pairs.update(_load_from_json(candidate))
        elif candidate.suffix.lower() == ".py":
            mod_name = candidate.stem
            pairs.update(_load_from_module(candidate.parent, mod_name))
    elif module_name:
        pairs.update(_load_from_module(cfg.book_dir, module_name))
    if pairs:
        # Hand-curated entries should NOT be overridden, but they should fill ALL keys.
        for tr, ar in pairs.items():
            ARABIC_VOCAB[_normalize_translit(tr)] = ar
    print(f"[quire] vocabulary loaded: {len(ARABIC_VOCAB)} entries", file=sys.stderr)
