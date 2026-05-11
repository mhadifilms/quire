"""Smoke tests: package imports, CLI parses."""

from __future__ import annotations

import importlib


def test_top_level_package_imports() -> None:
    import quire

    assert quire.__version__


def test_all_modules_import() -> None:
    for m in [
        "quire.cli",
        "quire.config",
        "quire.pipeline",
        "quire.extract.pdf",
        "quire.structure.pdf_based",
        "quire.render.chapters",
        "quire.render.export",
        "quire.render.package",
        "quire.render.audit",
        "quire.postprocess.registry",
        "quire.postprocess.ocr_corrections",
        "quire.postprocess.vocabulary",
        "quire.postprocess.glossary",
        "quire.postprocess.mojibake",
        "quire.postprocess.common_ocr",
        "quire.postprocess.script_detect",
        "quire.postprocess.canonical.quran",
        "quire.postprocess.canonical.quran_plugin",
    ]:
        importlib.import_module(m)


def test_cli_help_does_not_crash(capsys) -> None:
    from quire import cli

    try:
        cli.main(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
    out = capsys.readouterr().out
    assert "quire" in out.lower()
