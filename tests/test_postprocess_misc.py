"""Tests for the miscellaneous post-processors: mojibake, script_detect."""

from __future__ import annotations

from pathlib import Path

from quire.postprocess import common_ocr, mojibake, ocr_corrections, script_detect


class _Cfg:
    def __init__(self, settings=None, *, book_dir: Path | None = None, artifact_dir: Path | None = None):
        self._s = settings or {}
        self.book_dir = book_dir
        self.artifact_dir = artifact_dir

    def plugin_config(self, name):
        return self._s.get(name, {})

    @property
    def ocr_corrections_path(self):
        return self.artifact_dir / "ocr_corrections.tsv"

    @property
    def ocr_corrections_diff_path(self):
        return self.artifact_dir / "ocr_corrections.diff"

    @property
    def review_tsv_path(self):
        return self.artifact_dir / "manual_review.tsv"

    @property
    def ocr_review_tsv_path(self):
        return self.artifact_dir / "ocr_review.tsv"


def test_mojibake_cleanup_replaces_che_with_the() -> None:
    pages = [
        {"elements": [
            {"kind": "paragraph", "text": "che result is here", "y": 1},
        ]}
    ]
    mojibake.post_structure(_Cfg(), pages)
    text = pages[0]["elements"][0]["text"]
    assert "the result is here" in text


def test_mojibake_preserves_normal_text() -> None:
    pages = [{"elements": [
        {"kind": "paragraph", "text": "Perfectly fine English.", "y": 1}
    ]}]
    mojibake.post_structure(_Cfg(), pages)
    assert pages[0]["elements"][0]["text"] == "Perfectly fine English."


# ---------- Pass-2 cue regex conservatism (regression guards) ----------
#
# The cue regex matches sentence shapes like ``, X is the …``. Before the
# token-plausibility guard was added it nuked legitimate English prose,
# replacing words like ``the Quran`` / ``the Kaaba`` / ``which`` / ``but
# its inward aspect`` with the literal token ``[Arabic]``. These cases
# must round-trip unchanged.

def test_mojibake_pass2_preserves_proper_noun_after_comma() -> None:
    pages = [{"elements": [
        {
            "kind": "paragraph",
            "text": (
                "It declares that the Lord does not call His servants to "
                "carry out a task unless they are able to perform it. "
                "Elsewhere, the Quran states that:"
            ),
            "y": 1,
        }
    ]}]
    mojibake.post_structure(_Cfg(), pages)
    out = pages[0]["elements"][0]["text"]
    assert "Elsewhere, the Quran states that:" in out
    assert "[Arabic]" not in out


def test_mojibake_pass2_preserves_short_common_english_word() -> None:
    pages = [{"elements": [
        {
            "kind": "paragraph",
            "text": (
                "The inner meaning of the Yamani Corner is walayah, which "
                "is the divinely appointed authority of imamate."
            ),
            "y": 1,
        }
    ]}]
    mojibake.post_structure(_Cfg(), pages)
    out = pages[0]["elements"][0]["text"]
    assert "walayah, which is the divinely" in out
    assert "[Arabic]" not in out


def test_mojibake_pass2_preserves_multi_word_english_phrase() -> None:
    pages = [{"elements": [
        {
            "kind": "paragraph",
            "text": (
                "is the well-known House in Makkah, but its inward aspect "
                "is the heart of the perfect human being"
            ),
            "y": 1,
        }
    ]}]
    mojibake.post_structure(_Cfg(), pages)
    out = pages[0]["elements"][0]["text"]
    assert "but its inward aspect is the heart" in out
    assert "[Arabic]" not in out


def test_mojibake_pass2_preserves_known_proper_nouns() -> None:
    # Each phrase here was destroyed by the old cue regex. They must
    # survive verbatim under the conservative plausibility rule.
    samples = [
        "According to a tradition of Ali ibn Abi-Talib, the Kaaba is the reflection of al-Bayt al-Ma'mur upon the earth.",
        "Safa is the full scale, and Marwah is the empty one.",
        "Resurrection, and know that Safa is the scale of his good deeds and Marwah is the scale of his evil deeds.",
    ]
    pages = [
        {"elements": [{"kind": "paragraph", "text": s, "y": 1}]}
        for s in samples
    ]
    mojibake.post_structure(_Cfg(), pages)
    for original, page in zip(samples, pages, strict=False):
        out = page["elements"][0]["text"]
        assert "[Arabic]" not in out, f"corrupted: {out!r}"
        assert original == out, f"changed: {original!r} -> {out!r}"


def test_mojibake_pass2_still_catches_pure_letter_gibberish() -> None:
    # The cue pass is conservative but must still fire when there is a
    # clear mojibake signal: pure-lowercase 1-3 letter tokens that are
    # neither a known short word nor a real English word.
    pages = [{"elements": [
        {
            "kind": "paragraph",
            "text": "The verse jes ne ji here means something.",
            "y": 1,
        }
    ]}]
    mojibake.post_structure(_Cfg(), pages)
    out = pages[0]["elements"][0]["text"]
    assert "[Arabic]" in out
    assert "jes ne ji" not in out


def test_mojibake_cleanup_applies_common_fixes_to_footnotes() -> None:
    """The shared common_fixes.toml dictionary should apply to footnote
    elements just like body paragraphs. Uses only generic t->r scanner
    errors (no book-specific proper nouns)."""
    pages = [
        {"elements": [
            {
                "kind": "footnote",
                "text": "See nore 1 abour the matrer; bur verify chis.",
                "y": 1,
            },
        ]}
    ]
    mojibake.post_structure(_Cfg(), pages)
    assert pages[0]["elements"][0]["text"] == "See note 1 about the matter; but verify this."


def test_mojibake_cleanup_writes_shared_correction_audit(tmp_path: Path) -> None:
    cfg = _Cfg(book_dir=tmp_path, artifact_dir=tmp_path)
    ocr_corrections.reset_report()
    pages = [
        {"pno": 9, "elements": [
            {"kind": "paragraph", "text": "Cheir dury is mencioned.", "y": 12},
        ]}
    ]
    mojibake.post_structure(cfg, pages)
    ocr_corrections.write_report(cfg)

    assert pages[0]["elements"][0]["text"] == "Their duty is mentioned."
    audit = cfg.ocr_corrections_path.read_text()
    assert "word:Cheir" in audit
    assert "word:dury" in audit
    assert "word:mencioned" in audit


def test_book_local_ocr_fixes_extend_shared_rules(tmp_path: Path) -> None:
    (tmp_path / "ocr_fixes.toml").write_text('[word]\nCustorn = "Custom"\n', encoding="utf-8")
    cfg = _Cfg(book_dir=tmp_path, artifact_dir=tmp_path)
    ocr_corrections.reset_report()
    pages = [
        {"pno": 1, "elements": [
            {"kind": "paragraph", "text": "Custorn book fix.", "y": 1},
        ]}
    ]

    mojibake.post_structure(cfg, pages)

    assert pages[0]["elements"][0]["text"] == "Custom book fix."


def test_script_detect_marks_persian() -> None:
    pages = [{"elements": [
        {"kind": "arabic", "text": "کتاب گفت چه", "y": 1},
        {"kind": "arabic", "text": "السلام عليكم", "y": 2},
    ]}]
    script_detect.post_structure(_Cfg(), pages)
    elems = pages[0]["elements"]
    assert elems[0]["script_lang"] == "fa"
    assert elems[1]["script_lang"] == "ar"


def test_script_detect_marks_urdu_and_hebrew() -> None:
    pages = [{"elements": [
        {"kind": "paragraph", "text": "یہ اردو ہے", "y": 1},  # Urdu ے
        {"kind": "paragraph", "text": "שלום עליכם", "y": 2},  # Hebrew
        {"kind": "paragraph", "text": "Plain English only", "y": 3},
    ]}]
    script_detect.post_structure(_Cfg(), pages)
    elems = pages[0]["elements"]
    assert elems[0]["script_lang"] == "ur"
    assert elems[1]["script_lang"] == "he"
    # English paragraph: no script_lang set.
    assert "script_lang" not in elems[2]


def test_common_ocr_corrects_low_confidence_name_variant() -> None:
    pages = [
        {"elements": [
            {"kind": "paragraph", "text": "Ibn Arabi discusses the journey.", "conf": 96, "y": 1},
            {"kind": "paragraph", "text": "Ibn Arabi appears again.", "conf": 94, "y": 2},
            {"kind": "paragraph", "text": "Ion Arabi is mentioned here.", "conf": 55, "y": 3},
        ]}
    ]
    common_ocr.post_structure(_Cfg({"common_ocr": {}}), pages)
    assert pages[0]["elements"][2]["text"] == "Ibn Arabi is mentioned here."


def test_common_ocr_leaves_high_confidence_variant_alone() -> None:
    pages = [
        {"elements": [
            {"kind": "paragraph", "text": "Ibn Arabi discusses the journey.", "conf": 96, "y": 1},
            {"kind": "paragraph", "text": "Ibn Arabi appears again.", "conf": 94, "y": 2},
            {"kind": "paragraph", "text": "Ion Arabi is a high confidence reading.", "conf": 95, "y": 3},
        ]}
    ]
    common_ocr.post_structure(_Cfg({"common_ocr": {}}), pages)
    assert "Ion Arabi" in pages[0]["elements"][2]["text"]


def test_common_ocr_corrects_overconfident_one_off_variant_when_name_dominates() -> None:
    pages = [
        {"elements": [
            {"kind": "paragraph", "text": "Ibn Arabi discusses the journey.", "conf": 96, "y": 1},
            {"kind": "paragraph", "text": "Ibn Arabi appears again.", "conf": 94, "y": 2},
            {"kind": "paragraph", "text": "Ibn Arabi says more.", "conf": 95, "y": 3},
            {"kind": "paragraph", "text": "Ibn Arabi is cited.", "conf": 95, "y": 4},
            {"kind": "paragraph", "text": "Ibn Arabi closes it.", "conf": 95, "y": 5},
            {"kind": "paragraph", "text": "Ion Arabi is overconfident OCR.", "conf": 100, "y": 6},
        ]}
    ]
    common_ocr.post_structure(_Cfg({"common_ocr": {}}), pages)
    assert pages[0]["elements"][5]["text"] == "Ibn Arabi is overconfident OCR."


def test_common_ocr_corrects_dominant_name_variant_in_footnote() -> None:
    pages = [
        {"elements": [
            {"kind": "paragraph", "text": "Ibn Arabi discusses the journey.", "conf": 96, "y": 1},
            {"kind": "paragraph", "text": "Ibn Arabi appears again.", "conf": 94, "y": 2},
            {"kind": "paragraph", "text": "Ibn Arabi says more.", "conf": 95, "y": 3},
            {"kind": "paragraph", "text": "Ibn Arabi is cited.", "conf": 95, "y": 4},
            {"kind": "paragraph", "text": "Ibn Arabi closes it.", "conf": 95, "y": 5},
            {"kind": "footnote", "text": "See In Arabi, Al-Futuhat.", "y": 6},
        ]}
    ]
    common_ocr.post_structure(_Cfg({"common_ocr": {}}), pages)
    assert pages[0]["elements"][5]["text"] == "See Ibn Arabi, Al-Futuhat."


def test_common_ocr_does_not_rewrite_pronoun_phrases_or_possessives() -> None:
    pages = [
        {"elements": [
            {"kind": "paragraph", "text": "Our Lord is mentioned.", "conf": 96, "y": 1},
            {"kind": "paragraph", "text": "Our Lord appears again.", "conf": 94, "y": 2},
            {"kind": "paragraph", "text": "Our Lord says more.", "conf": 95, "y": 3},
            {"kind": "paragraph", "text": "Our Lord is cited.", "conf": 95, "y": 4},
            {"kind": "paragraph", "text": "Our Lord closes it.", "conf": 95, "y": 5},
            {"kind": "paragraph", "text": "Your Lord remains distinct.", "conf": 100, "y": 6},
            {"kind": "paragraph", "text": "Imam Sadiq speaks.", "conf": 96, "y": 7},
            {"kind": "paragraph", "text": "Imam Sadiq appears again.", "conf": 96, "y": 8},
            {"kind": "paragraph", "text": "Imam Sadiq says more.", "conf": 96, "y": 9},
            {"kind": "paragraph", "text": "Imam Sadiq is cited.", "conf": 96, "y": 10},
            {"kind": "paragraph", "text": "Imam Sadiq closes it.", "conf": 96, "y": 11},
            {"kind": "paragraph", "text": "Imam Sadiq's words remain possessive.", "conf": 100, "y": 12},
        ]}
    ]
    common_ocr.post_structure(_Cfg({"common_ocr": {}}), pages)
    assert pages[0]["elements"][5]["text"] == "Your Lord remains distinct."
    assert pages[0]["elements"][11]["text"] == "Imam Sadiq's words remain possessive."
