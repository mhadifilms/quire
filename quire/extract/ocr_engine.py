"""OCR engine abstraction.

The pipeline used to hard-code the macOS Vision OCR engine via
:mod:`quire.extract.ocr`. This module defines a small protocol so additional
engines (text-layer-only, Tesseract, cloud OCR, …) can be added without
changing pipeline code.

Engines are registered by name in :data:`OCR_ENGINES` and selected by the
``[ocr] engine`` field in ``book.toml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class OCREngineOptions:
    """Runtime knobs an engine may consult."""

    languages: list[str]
    workers: int
    dpi_scale: int
    retries: int
    force: bool = False


class OCREngine(Protocol):
    """Minimal contract for any OCR back-end.

    The engine is expected to read the PDF at ``pdf_path`` and return one
    OCR-result dict per page, structurally compatible with the legacy
    Vision OCR output that the rest of the pipeline consumes.
    """

    name: str

    def ocr_pdf(
        self,
        pdf_path: str,
        options: OCREngineOptions,
    ) -> list[dict]: ...


class TextLayerEngine:
    """No-op engine that returns empty OCR pages.

    Used when the source PDF already contains a reliable text layer; the
    text-layer extractor in :mod:`quire.extract.pdf` provides the content
    and the structure stage uses the PDF text spans directly.
    """

    name = "text"

    def ocr_pdf(self, pdf_path: str, options: OCREngineOptions) -> list[dict]:
        import fitz
        doc = fitz.open(pdf_path)
        try:
            return [
                {"pno": i + 1, "blocks": [], "page_size_pt": (page.rect.width, page.rect.height)}
                for i, page in enumerate(doc)
            ]
        finally:
            doc.close()


class VisionEngine:
    """macOS Vision OCR via :mod:`quire.extract.ocr`."""

    name = "vision"

    def ocr_pdf(self, pdf_path: str, options: OCREngineOptions) -> list[dict]:
        from .ocr import ocr_all
        pages = ocr_all(
            pdf_path,
            languages=options.languages,
            workers=options.workers,
            scale=options.dpi_scale,
            retries=options.retries,
        )
        # Without ``page_numbers`` the underlying call fills every slot, so
        # we can safely strip the ``| None`` from the type for callers.
        return [p for p in pages if p is not None]


class TesseractEngine:
    """Cross-platform Tesseract 5 LSTM OCR (default).

    Requires the ``tesseract`` binary on ``$PATH`` plus language data packs
    for the scripts you intend to process (e.g. ``ara``, ``fas``, ``urd``,
    ``chi_sim``). On macOS::

        brew install tesseract tesseract-lang

    On Debian/Ubuntu::

        sudo apt-get install tesseract-ocr tesseract-ocr-ara tesseract-ocr-fas
    """

    name = "tesseract"

    def ocr_pdf(self, pdf_path: str, options: OCREngineOptions) -> list[dict]:
        from .tesseract_engine import ocr_pdf_tesseract
        pages = ocr_pdf_tesseract(
            pdf_path,
            languages=options.languages,
            workers=options.workers,
            scale=options.dpi_scale,
            retries=options.retries,
        )
        return [p for p in pages if p is not None]


OCR_ENGINES: dict[str, type[OCREngine]] = {
    "text": TextLayerEngine,
    "pdf": TextLayerEngine,
    "pymupdf": TextLayerEngine,
    "vision": VisionEngine,
    "tesseract": TesseractEngine,
}


def get_engine(name: str) -> OCREngine:
    """Look up an OCR engine by name. Raises ``KeyError`` if unknown."""
    try:
        cls = OCR_ENGINES[name.lower()]
    except KeyError as e:
        raise KeyError(
            f"unsupported OCR engine {name!r}. Available: {sorted(OCR_ENGINES)}"
        ) from e
    return cls()
