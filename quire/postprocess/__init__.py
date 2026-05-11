"""Post-processing plugin system.

A post-processor implements one or both of:

  pre_structure(cfg, ocr_pages)        # runs before per-page structuring
  post_structure(cfg, ocr_pages)       # runs after structuring (sees elements)

Plugins are referenced by short names from ``book.toml`` ``postprocess.plugins``
list. Built-in plugins live alongside this file.
"""

from . import registry  # re-export the registry module

__all__ = ["registry"]
