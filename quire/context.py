"""Per-book runtime context.

The pipeline historically used module-level globals to carry "configured
headings", "loaded vocabulary", and a cached Quran corpus across stages.
That works for one book per process but is unsafe for batch runs.

``BookContext`` collects this state per-build, and we provide a single entry
point (:func:`reset_for_build`) that the pipeline calls before each book so
sequential batch runs don't bleed state between books.

For *parallel* batch runs we use process-based workers (one fresh
interpreter per book), so we sidestep thread-safety entirely.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import BookConfig


@dataclass
class BookContext:
    """Scratch state held while one book is being built."""

    cfg: BookConfig
    headings: list[tuple[str, int]] = field(default_factory=list)
    vocabulary: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def reset_for_build(cfg: BookConfig) -> BookContext:
    """Reset all module-level globals to a clean state, then return a new
    :class:`BookContext` for this book.

    Modules that still hold globals (vocabulary, known headings, Quran
    corpus path) are reset explicitly here. New code should consume the
    returned :class:`BookContext` instead.
    """
    from .postprocess import vocabulary as vocab_mod
    from .postprocess.canonical import quran as quran_mod
    from .structure import pdf_based

    vocab_mod.ARABIC_VOCAB.clear()
    quran_mod.reset_caches()
    pdf_based.configure_known_headings(cfg.structure_headings)
    return BookContext(cfg=cfg, headings=list(cfg.structure_headings))
