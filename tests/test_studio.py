"""Book workspace contracts, loss prevention, and real publication integration."""

from __future__ import annotations

import copy
import json
import threading
import zipfile
from pathlib import Path

import fitz
import httpx
import pytest

from quire.languages import detect_language, direction, text_quality
from quire.studio import project
from quire.studio.benchmark import score
from quire.studio.extract import import_pdf
from quire.studio.publish import publish
from quire.studio.quality import check
from quire.studio.server import make_server
from quire.studio.translate import apply_response, export_request, translate, validate_response


@pytest.fixture
def studio(tmp_path, tiny_pdf_path):
    root = tmp_path / "library" / "tiny"
    import_pdf(tiny_pdf_path, root, language="en")
    return root


@pytest.mark.parametrize(
    "text,language",
    [
        ("این کتاب برای پدران و مادران است. کودکان در کنار خانواده رشد می‌کنند.", "fa"),
        ("هذا الكتاب يتحدث عن القراءة في المدرسة وعن الأطفال الذين يتعلمون كل يوم.", "ar"),
        ("یہ کتاب بچوں کے بارے میں ہے اور ان کی تعلیم کے لیے لکھی گئی ہے۔", "ur"),
        ("זהו ספר על ילדים ועל הלמידה שלהם בבית הספר", "he"),
        ("The reader opens the book and begins to read the story with the family.", "en"),
        ("", "und"),
    ],
)
def test_detect_scripts_and_languages(text, language):
    detected = detect_language(text)
    assert detected["language"] == language
    assert detected["direction"] == direction(language)


def test_text_health_is_not_english_word_count():
    assert text_quality("این کتاب برای کودکان و خانواده است.") > 0.8
    assert text_quality("\ufffd\ufffd\ufffd abc") < 0.2
    assert detect_language("العلم")["confidence"] < 0.75


def test_import_keeps_every_page_and_native_region(studio):
    data = project.load(studio, verify_source=True)
    assert len(data["pages"]) == 2
    assert all(p["engine"] == "text" for p in data["pages"])
    assert all(
        n["bbox"] and " ".join(n["original"].split()) == " ".join(n["text"].split()) for n in data["nodes"]
    )
    assert all((studio / p["image"]).is_file() for p in data["pages"])
    assert not check(data)["ready"]
    assert check(data)["recognition_accuracy"] is None


def test_resume_does_not_rerun_or_overwrite_reviewed_page(studio, monkeypatch):
    from quire.studio import extract

    data = project.load(studio)
    first = data["nodes"][0]
    data = project.edit(
        studio,
        first["id"],
        {"text": "Approved custom title", "status": "reviewed"},
        revision=data["revision"],
    )

    def unexpected(*a, **kw):
        raise AssertionError("A completed page must not run again")

    monkeypatch.setattr(extract, "extract_page", unexpected)
    resumed = import_pdf(studio / "source.pdf", studio, language="en")
    assert resumed["nodes"][0]["text"] == "Approved custom title"
    assert resumed["nodes"][0]["status"] == "reviewed"


def test_failed_page_retries_without_losing_successful_page(tmp_path, tiny_pdf_path, monkeypatch):
    from quire.studio import extract

    real = extract.extract_page

    def fail_second(page, *args, **kwargs):
        if page.number == 1:
            raise RuntimeError("interrupted")
        return real(page, *args, **kwargs)

    monkeypatch.setattr(extract, "extract_page", fail_second)
    root = tmp_path / "partial"
    data = import_pdf(tiny_pdf_path, root, language="en")
    assert data["jobs"]["extract"]["state"] == "partial"
    first_ids = [n["id"] for n in data["nodes"]]
    monkeypatch.setattr(extract, "extract_page", real)
    data = import_pdf(tiny_pdf_path, root, language="en")
    assert data["jobs"]["extract"]["state"] == "complete"
    assert all(node_id in [n["id"] for n in data["nodes"]] for node_id in first_ids)


def test_same_phrase_edit_is_scoped_and_undo_is_exact(studio):
    data = project.load(studio)
    first, second = [n for n in data["nodes"] if n["text"] == "Tiny Book"]
    old = copy.deepcopy(data["nodes"])
    data = project.edit(studio, first["id"], {"text": "The edited title"}, revision=data["revision"])
    assert next(n for n in data["nodes"] if n["id"] == second["id"])["text"] == "Tiny Book"
    data = project.undo(studio, revision=data["revision"])
    assert data["nodes"] == old


def test_stale_edit_and_empty_exclusion_are_rejected_atomically(studio):
    data = project.load(studio)
    node = data["nodes"][0]
    project.edit(studio, node["id"], {"text": "First edit"}, revision=data["revision"])
    with pytest.raises(ValueError, match="changed"):
        project.edit(studio, node["id"], {"text": "Stale edit"}, revision=data["revision"])
    data = project.load(studio)
    with pytest.raises(ValueError, match="why"):
        project.edit(studio, node["id"], {"status": "excluded"}, revision=data["revision"])
    assert project.load(studio) == data


def test_page_review_is_separate_and_reversible(studio):
    data = project.load(studio)
    data = project.review_page(studio, 1, revision=data["revision"])
    assert data["pages"][0]["reviewed"]
    assert all(n["status"] == "pending" for n in data["nodes"])
    data = project.undo(studio, revision=data["revision"])
    assert not data["pages"][0]["reviewed"]


def test_translation_requires_exact_ids_and_source_binding(studio, tmp_path):
    request = export_request(studio, "fa", tmp_path / "request.json")
    response = {
        "translations": [
            {"id": n["id"], "text": "ترجمه کامل این بخش", "uncertainties": []} for n in request["passages"]
        ]
    }
    with pytest.raises(ValueError, match="every requested"):
        apply_response(studio, request, {"translations": response["translations"][:-1]})
    data = apply_response(studio, request, response)
    assert all(n["translations"]["fa"]["status"] == "pending" for n in data["nodes"])
    node = data["nodes"][0]
    data = project.edit(studio, node["id"], {"text": "Changed original"}, revision=data["revision"])
    assert "stale_translation" in {f["code"] for f in check(data, target="fa")["findings"]}


def test_table_translation_cannot_drop_a_cell():
    unit = {"id": "table", "kind": "table", "text": "a b", "cells": [["a", "b"]]}
    with pytest.raises(ValueError, match="shape"):
        validate_response([unit], [{"id": "table", "text": "الف", "cells": [["الف"]]}])
    with pytest.raises(ValueError, match="omitted"):
        validate_response([unit], [{"id": "table", "text": "الف", "cells": [["الف", ""]]}])


def test_translation_job_commits_batches_and_resumes(studio):
    calls = []

    def respond(request):
        body = json.loads(request.content)
        prompt = json.loads(body["contents"][0]["parts"][0]["text"])
        calls.append(prompt)
        translations = [
            {"id": u["id"], "text": "این متن ترجمه شده است.", "uncertainties": []} for u in prompt["passages"]
        ]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {"parts": [{"text": json.dumps({"translations": translations})}]},
                    }
                ],
                "usageMetadata": {"totalTokenCount": 300},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = translate(studio, "fa", client=client)
        assert result["state"] == "complete"
        count = len(calls)
        second = translate(studio, "fa", client=client)
        assert second["translated"] == 0
        assert len(calls) == count


def test_translation_budget_and_truncation_never_apply_partial_passages(studio):
    def no_call(request):
        raise AssertionError("Budget must stop the request first")

    with httpx.Client(transport=httpx.MockTransport(no_call)) as client:
        result = translate(studio, "fa", token_budget=100, client=client)
        assert result["state"] == "paused"

    def truncated(request):
        return httpx.Response(200, json={"candidates": [{"finishReason": "MAX_TOKENS"}]})

    with httpx.Client(transport=httpx.MockTransport(truncated)) as client:
        with pytest.raises(ValueError, match="truncated"):
            translate(studio, "fa", client=client)
    assert all(not n["translations"] for n in project.load(studio)["nodes"])


def test_approved_human_translation_is_never_overwritten(studio, tmp_path):
    request = export_request(studio, "fa", tmp_path / "req.json")
    data = project.load(studio)
    node = data["nodes"][0]
    project.edit(
        studio,
        node["id"],
        {"text": "ترجمه انسانی", "status": "reviewed"},
        revision=data["revision"],
        target="fa",
    )
    response = {"translations": [{"id": u["id"], "text": "جایگزین خودکار"} for u in request["passages"]]}
    with pytest.raises(ValueError, match="human"):
        apply_response(studio, request, response)


def test_missing_page_or_source_change_blocks_ready(studio):
    data = project.load(studio)
    data["pages"].pop()
    assert "missing_page" in {f["code"] for f in check(data)["findings"]}
    with (studio / "source.pdf").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="Source PDF changed"):
        project.load(studio, verify_source=True)


def test_portable_bundle_roundtrip_and_checksum_failure(studio, tmp_path):
    bundle = project.pack(studio, tmp_path / "book.quire.zip")
    restored = project.unpack(bundle, tmp_path / "restored")
    assert project.load(restored, verify_source=True) == project.load(studio)
    damaged = tmp_path / "damaged.zip"
    with zipfile.ZipFile(bundle) as src, zipfile.ZipFile(damaged, "w") as dst:
        for name in src.namelist():
            dst.writestr(name, b"bad" if name == "source.pdf" else src.read(name))
    with pytest.raises(ValueError, match="Checksum"):
        project.unpack(damaged, tmp_path / "bad-restoration")
    assert not (tmp_path / "bad-restoration").exists()


def test_zip_traversal_and_existing_destination_rejected(tmp_path):
    bundle = tmp_path / "evil.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("../outside", "bad")
    with pytest.raises(ValueError, match="Unsafe"):
        project.unpack(bundle, tmp_path / "restored")
    with pytest.raises(ValueError, match="new folder"):
        project.unpack(bundle, tmp_path)
    assert not (tmp_path / "outside").exists()


def test_publication_retains_manuscript_text_and_scoped_edits(studio, monkeypatch):
    import quire.studio.publish as publishing

    monkeypatch.setattr(publishing, "run_epubcheck", lambda p: {"status": "unavailable"})
    monkeypatch.setattr(publishing, "run_ace", lambda p, d: {"status": "unavailable"})
    data = project.load(studio)
    with pytest.raises(ValueError, match="needs review"):
        publish(studio)
    node = data["nodes"][1]
    project.edit(
        studio, node["id"], {"text": "A uniquely corrected passage for this page."}, revision=data["revision"]
    )
    release = publish(studio, formats=["pdf", "epub", "html", "markdown", "text"], draft=True)
    folder = studio / release["directory"]
    assert release["state"] == "draft"
    with fitz.open(folder / "book.pdf") as pdf:
        text = " ".join(p.get_text() for p in pdf)
        assert "uniquely corrected passage" in text
        assert "quick brown fox" in text  # second page remains intact
        assert len(pdf.get_toc()) >= 1
    with zipfile.ZipFile(folder / "book.epub") as epub:
        assert epub.infolist()[0].filename == "mimetype"
        assert epub.infolist()[0].compress_type == zipfile.ZIP_STORED
        text = epub.read("OEBPS/book.xhtml").decode()
        assert "uniquely corrected passage" in text
        assert "quick brown fox" in text
    for filename in ["book.html", "book.md", "book.txt"]:
        assert "uniquely corrected passage" in (folder / filename).read_text()


def test_verified_manuscript_does_not_claim_unavailable_validation(studio, monkeypatch):
    import quire.studio.publish as publishing

    monkeypatch.setattr(publishing, "run_epubcheck", lambda p: {"status": "unavailable"})
    monkeypatch.setattr(publishing, "run_ace", lambda p, d: {"status": "unavailable"})

    def approve(data):
        for page in data["pages"]:
            page["reviewed"] = True
        for node in data["nodes"]:
            node["status"] = "reviewed"

    project.mutate(studio, None, approve)
    release = publish(studio, formats=["epub"])
    assert release["state"] == "validation_pending"
    assert release["quality"]["ready"]


def test_benchmark_distinguishes_omissions_and_insertions():
    result = score("one two three four", "one three four extra")
    assert result["omitted_words"] == 1
    assert result["word_alignment"]["insertions"] == 1
    assert result["word_error_rate"] == 0.5
    assert score("", "unreferenced")["character_error_rate"] is None


def test_review_server_requires_local_host_token_and_current_revision(studio):
    server = make_server(studio.parent)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    try:
        with httpx.Client(base_url=url, trust_env=False) as client:
            assert client.get("/api/projects", headers={"Host": "attacker.example"}).status_code == 403
            response = client.get("/")
            import re

            token = re.search('name="quire-token" content="([^"]+)"', response.text)[1]
            data = project.load(studio)
            route = f"/api/projects/{data['id']}/edit"
            edit = {
                "revision": data["revision"],
                "node_id": data["nodes"][0]["id"],
                "changes": {"text": "API edit"},
            }
            assert client.post(route, json=edit).status_code == 403
            assert (
                client.post(
                    route, headers={"X-Quire-Token": token, "Origin": "https://attacker.example"}, json=edit
                ).status_code
                == 403
            )
            assert client.post(route, headers={"X-Quire-Token": token}, json=edit).status_code == 200
            assert client.post(route, headers={"X-Quire-Token": token}, json=edit).status_code == 409
            assert project.load(studio)["nodes"][0]["text"] == "API edit"
    finally:
        server.shutdown()
        server.server_close()
        server.executor.shutdown()


@pytest.mark.parametrize(
    "outcome,expected",
    [("pass", "ok"), ("fail", "fail"), ("earl:failed", "fail"), ("cantTell", "unavailable")],
)
def test_ace_uses_report_outcome_even_when_process_exits_zero(tmp_path, monkeypatch, outcome, expected):
    import subprocess

    import quire.studio.publish as publishing

    monkeypatch.setattr(publishing.shutil, "which", lambda _: "ace")
    monkeypatch.setattr(
        publishing.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess([], 0, "", "")
    )
    (tmp_path / "report.json").write_text(json.dumps({"earl:result": {"earl:outcome": outcome}}))
    assert publishing.run_ace(tmp_path / "book.epub", tmp_path)["status"] == expected


def test_recover_missing_figure_is_scoped_cropped_and_undoable(studio):
    from PIL import Image

    data = project.load(studio)
    data = project.review_page(studio, 1, revision=data["revision"])
    original = copy.deepcopy(data["nodes"])
    data = project.add_passage(
        studio, 1, [50, 60, 150, 160], "A source illustration", "figure", revision=data["revision"]
    )
    added = next(n for n in data["nodes"] if n["id"] not in {v["id"] for v in original})
    assert added["original"] == "" and added["status"] == "pending"
    with Image.open(studio / added["asset"]) as image:
        assert image.width > 100 and image.height > 100
    assert not data["pages"][0]["reviewed"]
    data = project.undo(studio, revision=data["revision"])
    assert data["nodes"] == original and data["pages"][0]["reviewed"]


def test_table_edits_keep_all_formats_and_translation_source_in_sync(studio):
    data = project.load(studio)
    node = data["nodes"][1]
    data = project.edit(
        studio,
        node["id"],
        {"kind": "table", "cells": [["Fruit", "Count"], ["Apple", "2"]]},
        revision=data["revision"],
    )
    edited = next(n for n in data["nodes"] if n["id"] == node["id"])
    assert "Apple | 2" in edited["text"]
    with pytest.raises(ValueError, match="rectangular"):
        project.edit(
            studio,
            node["id"],
            {"cells": [["One", "Two"], ["Missing"]]},
            revision=data["revision"],
            target="fa",
        )
    data = project.edit(
        studio,
        node["id"],
        {"cells": [["میوه", "تعداد"], ["سیب", "۲"]]},
        revision=data["revision"],
        target="fa",
    )
    edited = next(n for n in data["nodes"] if n["id"] == node["id"])
    assert "سیب | ۲" in edited["translations"]["fa"]["text"]
    assert edited["translations"]["fa"]["source_hash"] == project.source_signature(edited)


def test_docling_table_and_picture_preserve_regions_and_provenance(studio, tmp_path):
    from quire.studio.extract import import_docling

    data = project.load(studio)
    height = data["pages"][0]["height"]
    raw = {
        "body": {"children": [{"$ref": "#/tables/0"}, {"$ref": "#/pictures/0"}]},
        "tables": [
            {
                "label": "table",
                "prov": [
                    {
                        "page_no": 1,
                        "bbox": {
                            "l": 50,
                            "r": 200,
                            "t": height - 50,
                            "b": height - 100,
                            "coord_origin": "BOTTOMLEFT",
                        },
                    }
                ],
                "data": {
                    "num_rows": 2,
                    "num_cols": 2,
                    "table_cells": [
                        {"start_row_offset_idx": r, "start_col_offset_idx": c, "text": text}
                        for r, c, text in [(0, 0, "Fruit"), (0, 1, "Count"), (1, 0, "Apple"), (1, 1, "2")]
                    ],
                },
            }
        ],
        "pictures": [
            {
                "label": "picture",
                "prov": [
                    {"page_no": 2, "bbox": {"l": 50, "r": 150, "t": 100, "b": 200, "coord_origin": "TOPLEFT"}}
                ],
            }
        ],
    }
    path = tmp_path / "layout.json"
    path.write_text(json.dumps(raw))
    result = import_docling(studio, path, revision=data["revision"])
    assert result["nodes"][0]["bbox"] == [50, 50, 200, 100]
    assert result["nodes"][0]["text"] == result["nodes"][0]["original"] == "Fruit | Count\nApple | 2"
    assert (studio / result["nodes"][1]["asset"]).is_file()
    assert (studio / "layout/docling.json").is_file()
    with pytest.raises(ValueError, match="before review"):
        project.edit(studio, result["nodes"][0]["id"], {"status": "reviewed"}, revision=result["revision"])
        import_docling(studio, path, revision=project.load(studio)["revision"])


def test_join_preserves_regions_invalidates_translation_and_undoes(studio):
    data = project.load(studio)
    nodes = [n for n in data["nodes"] if n["page"] == 1]
    for n in nodes[:2]:
        data = project.edit(studio, n["id"], {"kind": "paragraph"}, revision=data["revision"])
    data = project.edit(
        studio,
        nodes[0]["id"],
        {"text": "ترجمه انسانی", "status": "reviewed"},
        revision=data["revision"],
        target="fa",
    )
    original = copy.deepcopy(data["nodes"])
    data = project.join_next(studio, nodes[0]["id"], revision=data["revision"])
    joined = data["nodes"][0]
    assert len(joined["source_regions"]) == 2
    assert joined["original"] == original[0]["original"]
    assert joined["translations"]["fa"]["text"] == "ترجمه انسانی"
    assert "stale_translation" in {f["code"] for f in check(data, target="fa")["findings"]}
    assert data["nodes"][1]["status"] == "excluded"
    assert project.undo(studio, revision=data["revision"])["nodes"] == original


def test_rtl_pdf_alignment_matches_the_browser_manuscript(studio):
    import unicodedata

    data = project.load(studio)
    project.edit(
        studio, data["nodes"][0]["id"], {"text": "متن فارسی", "language": "fa", "kind": "paragraph"}, revision=data["revision"]
    )
    release = publish(studio, formats=["pdf", "html"], draft=True)
    folder = studio / release["directory"]
    assert 'dir="rtl" style="text-align:right' in (folder / "book.html").read_text()
    regions = []
    with fitz.open(folder / "book.pdf") as pdf:
        for page in pdf:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    text = "".join(s["text"] for s in line["spans"])
                    if any("ARABIC" in unicodedata.name(c, "") for c in text):
                        regions.append((line["bbox"][0], page.rect.width))
    assert regions and all(left > width / 2 for left, width in regions)


def test_translation_numbers_are_review_signals_across_digit_scripts():
    unit = {"id": "passage", "kind": "paragraph", "text": "There are 12 chapters."}
    good = validate_response([unit], [{"id": "passage", "text": "این کتاب ۱۲ فصل دارد."}])
    assert not good["passage"]["uncertainties"]
    missing = validate_response([unit], [{"id": "passage", "text": "این کتاب فصل دارد."}])
    assert "numbers absent" in missing["passage"]["uncertainties"][0]
