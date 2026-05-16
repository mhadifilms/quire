"""Tests for the AI-assisted QC stage (``quire/qc/``).

All tests mock the HTTP transport; nothing here makes a live API call.

Coverage:
  - Parser: confidence filtering, hallucination guard, format validation,
    JSON-fence stripping, duplicate dedupe.
  - Engine: payload shape, retry on 5xx/429, fatal on 4xx, cost accounting.
  - Runner: cache hit/miss/force, max_cost_usd abort, page filter, dry run.
  - Writer: human entries preserved, auto-block roundtrip, dedup vs human.
  - Page text: pagebreak splitting, footnote ref normalization.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from quire.qc.engine import (
    DEFAULT_MODEL,
    MODEL_PRICING,
    GeminiQCEngine,
    QCEngineError,
    extract_response_text,
)
from quire.qc.models import Correction, QCResult, confidence_at_least
from quire.qc.page_text import (
    build_page_map,
    extract_page_texts,
)
from quire.qc.parser import QCParseError, parse_corrections
from quire.qc.runner import (
    estimate_cost,
    parse_page_spec,
    run_qc,
)
from quire.qc.writer import merge_corrections

# ---------- parser ---------------------------------------------------------


class TestParser:
    def test_accepts_well_formed_corrections(self) -> None:
        page_text = "The pilgrim recites the talbiyah at the Maqam."
        raw = json.dumps(
            {
                "corrections": [
                    {
                        "find": "talbiyah",
                        "replace": "talbiyah",  # identical -> dropped
                        "confidence": "high",
                        "reason": "no change",
                    },
                    {
                        "find": "Maqam",
                        "replace": "Maqam Ibrahim",
                        "confidence": "high",
                        "reason": "complete name",
                    },
                ]
            }
        )
        out = parse_corrections(raw, page_text=page_text, page_label="42", min_confidence="medium")
        assert len(out) == 1
        assert out[0].find == "Maqam"
        assert out[0].replace == "Maqam Ibrahim"
        assert out[0].page == "42"

    def test_drops_hallucinated_find(self) -> None:
        page_text = "The pilgrim recites the talbiyah."
        raw = json.dumps(
            {
                "corrections": [
                    {
                        "find": "NEVER APPEARED IN TEXT",
                        "replace": "something",
                        "confidence": "high",
                        "reason": "fabricated",
                    }
                ]
            }
        )
        out = parse_corrections(raw, page_text=page_text, page_label="1", min_confidence="medium")
        assert out == []

    def test_drops_low_confidence_when_threshold_medium(self) -> None:
        page_text = "The pilgrim recites the talbiyah."
        raw = json.dumps(
            {
                "corrections": [
                    {
                        "find": "talbiyah",
                        "replace": "talbiya",
                        "confidence": "low",
                        "reason": "uncertain",
                    }
                ]
            }
        )
        out = parse_corrections(raw, page_text=page_text, page_label="1", min_confidence="medium")
        assert out == []

    def test_strips_json_code_fences(self) -> None:
        page_text = "Bagarah is the second sura."
        raw = (
            "```json\n"
            + json.dumps(
                {
                    "corrections": [
                        {
                            "find": "Bagarah",
                            "replace": "Baqarah",
                            "confidence": "high",
                            "reason": "OCR g->q",
                        }
                    ]
                }
            )
            + "\n```"
        )
        out = parse_corrections(raw, page_text=page_text, page_label="2", min_confidence="medium")
        assert len(out) == 1
        assert out[0].replace == "Baqarah"

    def test_rejects_html_tag_in_replace(self) -> None:
        page_text = "essence of the matter"
        raw = json.dumps(
            {
                "corrections": [
                    {
                        "find": "essence",
                        "replace": "<em>essence</em>",
                        "confidence": "high",
                        "reason": "emphasis",
                    }
                ]
            }
        )
        out = parse_corrections(raw, page_text=page_text, page_label="3", min_confidence="medium")
        assert out == []

    def test_rejects_overshort_find(self) -> None:
        page_text = "x = 1 and y = 2"
        raw = json.dumps(
            {
                "corrections": [
                    {"find": "x", "replace": "z", "confidence": "high", "reason": ""},
                ]
            }
        )
        out = parse_corrections(raw, page_text=page_text, page_label="3", min_confidence="medium")
        assert out == []

    def test_dedupes_identical_pairs(self) -> None:
        page_text = "Bagarah is the second sura."
        raw = json.dumps(
            {
                "corrections": [
                    {"find": "Bagarah", "replace": "Baqarah", "confidence": "high", "reason": "a"},
                    {"find": "Bagarah", "replace": "Baqarah", "confidence": "high", "reason": "b"},
                ]
            }
        )
        out = parse_corrections(raw, page_text=page_text, page_label="3", min_confidence="medium")
        assert len(out) == 1

    def test_raises_on_unparseable_root(self) -> None:
        with pytest.raises(QCParseError):
            parse_corrections("not json at all", page_text="x", page_label="1", min_confidence="medium")


# ---------- writer ---------------------------------------------------------


class TestWriter:
    def test_writes_auto_block_when_file_missing(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "qc_fixes.toml"
        corrections = [
            Correction("Bagarah", "Baqarah", "high", "g->q", "12"),
            Correction("Magam", "Maqam", "high", "g->q", "13"),
        ]
        r = merge_corrections(target, corrections)
        assert r.added == 2
        assert r.written == 2
        body = target.read_text("utf-8")
        assert "AUTO QC START" in body
        assert '"Bagarah" = "Baqarah"' in body
        assert '"Magam" = "Maqam"' in body
        # Auto-generated section provides the [phrase] header when needed.
        assert "[phrase]" in body

    def test_preserves_human_entries(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "qc_fixes.toml"
        target.write_text(
            '[phrase]\n"manual" = "fixed"\n', "utf-8"
        )
        corrections = [Correction("Bagarah", "Baqarah", "high", "", "12")]
        merge_corrections(target, corrections)
        body = target.read_text("utf-8")
        assert '"manual" = "fixed"' in body
        assert '"Bagarah" = "Baqarah"' in body

    def test_human_entry_wins_over_auto(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "qc_fixes.toml"
        target.write_text(
            '[phrase]\n"Bagarah" = "human-wins"\n', "utf-8"
        )
        corrections = [Correction("Bagarah", "Baqarah", "high", "", "12")]
        merge_corrections(target, corrections, preserve_human=True)
        body = target.read_text("utf-8")
        assert '"Bagarah" = "human-wins"' in body
        assert '"Baqarah"' not in body

    def test_auto_block_replaced_on_rerun(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "qc_fixes.toml"
        merge_corrections(
            target, [Correction("OldOne", "Replacement", "high", "", "1")]
        )
        body1 = target.read_text("utf-8")
        assert '"OldOne"' in body1

        merge_corrections(
            target, [Correction("NewOne", "Replacement2", "high", "", "1")]
        )
        body2 = target.read_text("utf-8")
        # The previous auto entry is gone; only the new one remains.
        assert '"OldOne"' not in body2
        assert '"NewOne" = "Replacement2"' in body2

    def test_escapes_quotes_and_backslashes(self, tmp_path: pathlib.Path) -> None:
        target = tmp_path / "qc_fixes.toml"
        merge_corrections(
            target,
            [Correction('she said "hi"', "she said hi", "high", "", "1")],
        )
        body = target.read_text("utf-8")
        assert r'\"hi\"' in body


# ---------- engine ---------------------------------------------------------


class _FakeResp:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self) -> Any:
        if isinstance(self._body, dict):
            return self._body
        return json.loads(self._body)


class _FakeClient:
    """Replaces httpx.Client. Records calls; replays the queued responses."""

    def __init__(self, responses: list[_FakeResp | Exception]):
        self.responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **kwargs: Any):
        self.calls.append({"url": url, **kwargs})
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt

    def close(self) -> None:
        pass


def _engine_with_responses(monkeypatch, responses: list[Any]) -> tuple[GeminiQCEngine, _FakeClient]:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    engine = GeminiQCEngine(model=DEFAULT_MODEL, retries=2)
    fake = _FakeClient(responses)
    engine._http_client = fake
    return engine, fake


def _ok_response(text: str, *, in_tok: int = 100, out_tok: int = 50) -> dict[str, Any]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}]}}],
        "usageMetadata": {
            "promptTokenCount": in_tok,
            "candidatesTokenCount": out_tok,
        },
    }


class TestEngine:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with pytest.raises(QCEngineError):
            GeminiQCEngine()

    def test_payload_includes_inline_image(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok = _ok_response('{"corrections": []}')
        engine, fake = _engine_with_responses(monkeypatch, [_FakeResp(200, ok)])
        engine.correct_page(
            image_bytes=b"\x89PNG_fake",
            page_text="sample",
            page_label="1",
        )
        assert len(fake.calls) == 1
        payload = fake.calls[0]["json"]
        parts = payload["contents"][0]["parts"]
        assert "inline_data" in parts[0]
        assert parts[0]["inline_data"]["mime_type"] == "image/png"
        assert payload["generation_config"]["response_mime_type"] == "application/json"

    def test_retries_on_5xx_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok = _ok_response('{"corrections": []}')
        engine, fake = _engine_with_responses(
            monkeypatch,
            [
                _FakeResp(503, {"error": "unavailable"}),
                _FakeResp(200, ok),
            ],
        )
        monkeypatch.setattr("quire.qc.engine.time.sleep", lambda *_: None)
        result = engine.correct_page(
            image_bytes=b"x", page_text="sample", page_label="1"
        )
        assert result.ok
        assert len(fake.calls) == 2

    def test_fatal_on_400(self, monkeypatch: pytest.MonkeyPatch) -> None:
        engine, _ = _engine_with_responses(
            monkeypatch, [_FakeResp(400, {"error": "bad request"})]
        )
        result = engine.correct_page(
            image_bytes=b"x", page_text="s", page_label="1"
        )
        assert not result.ok
        assert "400" in (result.error or "")

    def test_cost_accounting_matches_pricing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        ok = _ok_response('{"corrections": []}', in_tok=1_000_000, out_tok=1_000_000)
        engine, _ = _engine_with_responses(monkeypatch, [_FakeResp(200, ok)])
        result = engine.correct_page(
            image_bytes=b"x", page_text="s", page_label="1"
        )
        pricing = MODEL_PRICING[DEFAULT_MODEL]
        expected = pricing.input_per_million + pricing.output_per_million
        assert result.cost.usd == pytest.approx(expected, rel=1e-6)


# ---------- runner ---------------------------------------------------------


def _settings(**overrides: Any):
    from quire.config import QCSettings

    base = dict(
        enabled=True,
        engine=DEFAULT_MODEL,
        dpi=150,
        concurrency=2,
        max_cost_usd=1.0,
        retry=0,
        min_confidence="medium",
        pages="all",
        preserve_human=True,
    )
    base.update(overrides)
    return QCSettings(**base)


def _make_cfg(tmp_path: pathlib.Path, settings) -> Any:
    """Build the smallest BookConfig-shaped object the runner needs."""
    book_dir = tmp_path / "book"
    book_dir.mkdir()
    artifact = book_dir / "artifacts"
    artifact.mkdir()
    caches = artifact / "caches"
    caches.mkdir()
    cfg = MagicMock()
    cfg.qc_settings = settings
    cfg.caches_dir = caches
    cfg.qc_fixes_path = book_dir / "qc_fixes.toml"
    cfg.book_dir = book_dir
    cfg.slug = "test-book"
    cfg.pdf_path = book_dir / "source.pdf"
    return cfg


def _page_text(pdf_pno: int, text: str, printed: int | None = None):
    from quire.qc.models import PageText

    return PageText(pdf_pno=pdf_pno, printed=printed, plain_text=text)


class TestRunner:
    def test_dry_run_makes_no_api_calls(self, tmp_path: pathlib.Path) -> None:
        cfg = _make_cfg(tmp_path, _settings())
        result = run_qc(
            cfg,
            page_texts=[_page_text(1, "Bagarah is the second sura.")],
            image_provider=lambda pno: b"img",
            engine=None,
            pages_filter={1},
            force=False,
            dry_run=True,
        )
        assert result.pages_processed == 0
        assert result.cost.usd > 0  # estimate non-zero

    def test_disabled_returns_empty(self, tmp_path: pathlib.Path) -> None:
        cfg = _make_cfg(tmp_path, _settings(enabled=False))
        result = run_qc(
            cfg,
            page_texts=[_page_text(1, "x")],
            image_provider=lambda pno: b"img",
        )
        assert result.pages_processed == 0
        assert result.pages_skipped == 1
        assert result.corrections == []

    def test_caches_per_page(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_cfg(tmp_path, _settings(concurrency=1))
        page_texts = [_page_text(1, "Bagarah is the second sura.")]
        ok = _ok_response(
            json.dumps(
                {
                    "corrections": [
                        {
                            "find": "Bagarah",
                            "replace": "Baqarah",
                            "confidence": "high",
                            "reason": "g->q",
                        }
                    ]
                }
            )
        )
        engine, fake = _engine_with_responses(monkeypatch, [_FakeResp(200, ok)])

        first = run_qc(
            cfg,
            page_texts=page_texts,
            image_provider=lambda pno: b"img",
            engine=engine,
            pages_filter={1},
        )
        assert first.pages_processed == 1
        assert len(first.corrections) == 1

        # Second call must hit the cache (engine is closed; no API call).
        engine2, _ = _engine_with_responses(monkeypatch, [])
        second = run_qc(
            cfg,
            page_texts=page_texts,
            image_provider=lambda pno: b"img",
            engine=engine2,
            pages_filter={1},
        )
        assert second.pages_cached == 1
        assert second.pages_processed == 0
        assert len(second.corrections) == 1

    def test_cost_cap_aborts(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_cfg(tmp_path, _settings(max_cost_usd=0.00001, concurrency=1))
        ok = _ok_response(
            json.dumps({"corrections": []}), in_tok=1_000_000, out_tok=100
        )
        # Three pages, but the cost cap should fire after the first.
        engine, _ = _engine_with_responses(
            monkeypatch, [_FakeResp(200, ok)] * 3
        )
        page_texts = [
            _page_text(1, "p1"),
            _page_text(2, "p2"),
            _page_text(3, "p3"),
        ]
        result = run_qc(
            cfg,
            page_texts=page_texts,
            image_provider=lambda pno: b"img",
            engine=engine,
            pages_filter={1, 2, 3},
        )
        assert result.aborted
        assert "max_cost_usd" in (result.abort_reason or "")

    def test_writes_qc_fixes_on_success(self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _make_cfg(tmp_path, _settings(concurrency=1))
        ok = _ok_response(
            json.dumps(
                {
                    "corrections": [
                        {
                            "find": "Magam",
                            "replace": "Maqam",
                            "confidence": "high",
                            "reason": "g->q",
                        }
                    ]
                }
            )
        )
        engine, _ = _engine_with_responses(monkeypatch, [_FakeResp(200, ok)])
        run_qc(
            cfg,
            page_texts=[_page_text(1, "Magam Ibrahim is near the Kaaba.")],
            image_provider=lambda pno: b"img",
            engine=engine,
            pages_filter={1},
        )
        body = cfg.qc_fixes_path.read_text("utf-8")
        assert '"Magam" = "Maqam"' in body


# ---------- page text ------------------------------------------------------


class TestPageText:
    def test_splits_on_pagebreaks(self) -> None:
        xhtml = (
            "<body>"
            '<span epub:type="pagebreak" id="page-pdf-1"></span>'
            "<p>Page one body.</p>"
            '<span epub:type="pagebreak" id="page-pdf-2"></span>'
            "<p>Page two body.</p>"
            "</body>"
        )
        page_map = [(1, None), (2, None)]
        out = extract_page_texts([("c1", xhtml)], page_map)
        assert len(out) == 2
        assert "Page one" in out[0].plain_text
        assert "Page two" in out[1].plain_text
        assert out[0].pdf_pno == 1
        assert out[1].pdf_pno == 2

    def test_normalizes_footnote_refs(self) -> None:
        xhtml = (
            "<body>"
            '<span epub:type="pagebreak" id="page-1"></span>'
            '<p>Smith<sup>3</sup> argues that</p>'
            "</body>"
        )
        out = extract_page_texts([("c1", xhtml)], [(1, 1)])
        assert "Smith[^3] argues" in out[0].plain_text

    def test_normalizes_noteref_anchor(self) -> None:
        xhtml = (
            "<body>"
            '<span epub:type="pagebreak" id="page-1"></span>'
            '<p>The author<a epub:type="noteref" href="#fn1"><sup>5</sup></a> wrote</p>'
            "</body>"
        )
        out = extract_page_texts([("c1", xhtml)], [(1, 1)])
        assert "author[^5] wrote" in out[0].plain_text

    def test_build_page_map_flattens_chapters(self) -> None:
        from quire.render.chapters import Chapter

        c1 = Chapter(title="c1", slug="c1", page_start=1)
        c1.add_pagebreak(1, None)
        c1.add_pagebreak(2, 5)
        c2 = Chapter(title="c2", slug="c2", page_start=3)
        c2.add_pagebreak(3, 6)
        out = build_page_map([c1, c2])
        assert out == [(1, None), (2, 5), (3, 6)]


# ---------- page-range parsing --------------------------------------------


class TestPageSpec:
    def test_all(self) -> None:
        assert parse_page_spec("all", available={1, 2, 3}) == {1, 2, 3}
        assert parse_page_spec(None, available={1, 2}) == {1, 2}

    def test_single_and_range(self) -> None:
        assert parse_page_spec("3", available={1, 2, 3, 4}) == {3}
        assert parse_page_spec("2-4", available={1, 2, 3, 4, 5}) == {2, 3, 4}

    def test_comma_combinations(self) -> None:
        assert parse_page_spec(
            "1, 3-5, 10", available={1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
        ) == {1, 3, 4, 5, 10}

    def test_intersects_with_available(self) -> None:
        # 99 isn't in the available set, so it's dropped silently.
        assert parse_page_spec("1, 99", available={1, 2}) == {1}

    def test_bad_spec_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_page_spec("not-a-range", available={1})


# ---------- response helpers ----------------------------------------------


class TestEngineHelpers:
    def test_extract_response_text_happy(self) -> None:
        r = {"candidates": [{"content": {"parts": [{"text": "abc"}, {"text": "def"}]}}]}
        assert extract_response_text(r) == "abcdef"

    def test_extract_response_text_safety_blocked(self) -> None:
        # When Gemini blocks output, candidates may exist without `content`.
        r = {"candidates": [{"finishReason": "SAFETY"}]}
        assert extract_response_text(r) == ""

    def test_confidence_at_least(self) -> None:
        assert confidence_at_least("high", "medium")
        assert confidence_at_least("medium", "medium")
        assert not confidence_at_least("low", "medium")


# ---------- cost estimator -------------------------------------------------


class TestEstimateCost:
    def test_scales_with_pages(self) -> None:
        from quire.qc.models import PageText

        pt = [PageText(pdf_pno=i, printed=None, plain_text="hello " * 100) for i in range(1, 11)]
        _, usd = estimate_cost(pt, model=DEFAULT_MODEL)
        assert usd > 0.0
