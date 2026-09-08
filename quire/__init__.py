"""Quire — pluggable, multi-script PDF -> reflowable EPUB pipeline."""

__version__ = "0.2.0"

from .config import BookConfig, load_book_config

__all__ = ["BookConfig", "load_book_config", "__version__"]
