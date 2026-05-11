"""Tests for pure helper functions in ``quire.pipeline``."""

from __future__ import annotations

import pytest

from quire import pipeline


def test_filter_vision_blocks_drops_short() -> None:
    blocks = [
        {"text": "ال", "conf": 90},
        {"text": "السلام عليكم ورحمة الله", "conf": 90},
    ]
    out = pipeline._filter_vision_blocks(blocks)
    # The short one (2 chars) lacks 5 Arabic chars => dropped.
    assert len(out) == 1
    assert "السلام" in out[0]["text"]


def test_filter_vision_blocks_drops_low_confidence_short_runs() -> None:
    blocks = [{"text": "السلام عليكم", "conf": 40}]
    # Has a 4+ Arabic word (السلام), so should still pass.
    out = pipeline._filter_vision_blocks(blocks)
    assert len(out) == 1


def test_filter_vision_blocks_strict_digits_drops_bibliography_hallucination() -> None:
    """Tesseract's ``ara`` LSTM hallucinates Arabic on bibliography pages.

    The shape of the hallucination is a block dominated by ASCII digits
    (citation page numbers / years) interspersed with garbage Arabic
    glyphs. ``strict_digits=True`` must reject those, while leaving real
    Arabic prose untouched.
    """
    halluc = {
        "text": "زلهاط 08 2121800510135 جع لحل عب 12 لمم 14",
        "conf": 35,
    }
    real_prose = {
        "text": "السلام عليكم ورحمة الله وبركاته في هذا اليوم المبارك",
        "conf": 45,
    }
    real_with_some_digits = {
        # Bibliography line with a hijri year — must still pass.
        "text": "مؤسّسة آل البيت لإحياء التراث، قم، ١٤٢١ ق.",
        "conf": 70,
    }

    # Default behaviour preserves both — Vision build is unchanged.
    out = pipeline._filter_vision_blocks([halluc, real_prose, real_with_some_digits])
    assert len(out) == 3

    # Strict mode drops only the digit-dominated hallucination.
    out = pipeline._filter_vision_blocks(
        [halluc, real_prose, real_with_some_digits], strict_digits=True,
    )
    assert real_prose in out
    assert real_with_some_digits in out
    assert halluc not in out


def test_filter_vision_blocks_strict_drops_short_fragments_with_digits() -> None:
    """Short Arabic block + ASCII digit + low conf = index-entry noise."""
    short_noise = {"text": "5 ,16 صنه", "conf": 35}
    short_clean = {"text": "بَيتُ ٱلله", "conf": 70}
    out = pipeline._filter_vision_blocks(
        [short_noise, short_clean], strict_digits=True,
    )
    assert short_noise not in out


def test_block_horiz_coverage_zero_with_no_embedded() -> None:
    """Pure scans have no embedded text layer; coverage is 0 → filter is no-op."""
    block = {"x0": 50, "y0": 100, "x1": 350, "y1": 130, "text": "...", "conf": 20}
    assert pipeline._block_horiz_coverage_on_english(block, []) == 0.0
    assert not pipeline._is_embedded_text_hallucination(block, [])


def test_block_horiz_coverage_full_overlap() -> None:
    """Block fully covered by one embedded word at the same y → coverage ~1."""
    block = {"x0": 50, "y0": 100, "x1": 350, "y1": 130, "text": "...", "conf": 20}
    embedded = [(50, 110, 350, 122)]
    cov = pipeline._block_horiz_coverage_on_english(block, embedded)
    assert cov >= 0.99


def test_is_embedded_text_hallucination_drops_low_conf_no_diacritic() -> None:
    """The textbook hallucination shape: low conf, no diacritics, full overlap
    with embedded English."""
    halluc = {
        "x0": 50, "y0": 100, "x1": 350, "y1": 130,
        "text": "عط لعكتوعء لصة لعجوعاتهة طع4 ,معمسساه؟",
        "conf": 24,
    }
    embedded = [(50, 105, 220, 122), (230, 105, 350, 122)]
    assert pipeline._is_embedded_text_hallucination(halluc, embedded)


def test_is_embedded_text_hallucination_keeps_real_arabic_with_diacritics() -> None:
    """A block of real diacritised Arabic is never rejected — even if it
    happens to share a y-position with embedded English."""
    real_quran = {
        "x0": 50, "y0": 100, "x1": 350, "y1": 130,
        # Note the tashkeel (fatha, kasra, etc.)
        "text": "بِسْمِ ٱللَّهِ ٱلرَّحْمَـٰنِ ٱلرَّحِيمِ",
        "conf": 20,  # low conf but it's real
    }
    embedded = [(50, 105, 350, 122)]
    assert not pipeline._is_embedded_text_hallucination(real_quran, embedded)


def test_is_embedded_text_hallucination_keeps_high_confidence() -> None:
    """Even without diacritics, a high-confidence block is not hallucination."""
    high_conf_modern = {
        "x0": 50, "y0": 100, "x1": 350, "y1": 130,
        "text": "هذا نص عربي بدون تشكيل",
        "conf": 80,
    }
    embedded = [(50, 105, 350, 122)]
    assert not pipeline._is_embedded_text_hallucination(high_conf_modern, embedded)


def test_is_embedded_text_hallucination_keeps_blocks_off_embedded_english() -> None:
    """A low-conf no-diacritic block that DOESN'T overlap embedded English
    is left alone — it might be real Arabic Adobe missed."""
    arabic_no_diacritics = {
        "x0": 50, "y0": 100, "x1": 350, "y1": 130,
        "text": "هذا نص عربي بدون تشكيل",
        "conf": 25,
    }
    # English embedded text far away (different y entirely)
    embedded = [(50, 500, 350, 525)]
    assert not pipeline._is_embedded_text_hallucination(
        arabic_no_diacritics, embedded
    )


def test_page_is_english_citation_dense_flags_bibliography_page() -> None:
    """Citation-dense English page with no anchoring Arabic = hallucination."""
    bibliography_page = {
        "en_lines": [
            {"text": "Amuli, Sayyid Haydar. Nass al-Nusus, vol. 1, p. 234.", "conf": 90},
            {"text": "Beirut: Dar al-Kotob al-Ilmiyah, 1999.", "conf": 92},
            {"text": "Ibn Arabi. Al-Futuhat al-Makkiyah, 9 vols., Beirut 2002.", "conf": 88},
            {"text": "Qom: Alulbayt Institute, 1988. pp. 161-4.", "conf": 92},
        ] * 6,
        "arabic_blocks": [
            {"text": "عط لعكتوعء لصة لعجوعاتهة طع4 ,معمسساه؟",
             "conf": 28, "x0": 50, "y0": 100, "x1": 350, "y1": 130},
        ],
    }
    assert pipeline._page_is_english_citation_dense(bibliography_page)


def test_page_is_english_citation_dense_preserves_real_arabic_page() -> None:
    """A page with an anchoring Arabic block (long + high conf) is NOT bibliography,
    even if it has some English citations."""
    real_arabic_page = {
        "en_lines": [
            {"text": "The Quran says, see also al-Tabataba'i 1390 AH:", "conf": 90},
            {"text": "verse 2:127, vol. 1, p. 286, etc.", "conf": 92},
        ] * 30,  # lots of English with digits
        "arabic_blocks": [
            {
                "text": (
                    "بسم الله الرحمن الرحيم الحمد لله رب العالمين "
                    "الرحمن الرحيم مالك يوم الدين إياك نعبد وإياك "
                    "نستعين اهدنا الصراط المستقيم صراط الذين أنعمت "
                    "عليهم غير المغضوب عليهم ولا الضالين آمين"
                ),
                "conf": 75,
            },
        ],
    }
    assert not pipeline._page_is_english_citation_dense(real_arabic_page)


def test_page_is_english_citation_dense_skips_low_digit_pages() -> None:
    """Prose pages without citation-density of digits should NOT be flagged."""
    prose_page = {
        "en_lines": [
            {"text": "The pilgrim then enters the state of ihram before crossing the miqat.", "conf": 90},
            {"text": "He performs ablution and dons the two unsewn white garments.", "conf": 92},
        ] * 15,
        "arabic_blocks": [
            {"text": "إنا لله وإنا إليه راجعون", "conf": 40,
             "x0": 50, "y0": 100, "x1": 350, "y1": 130},
        ],
    }
    assert not pipeline._page_is_english_citation_dense(prose_page)


def test_page_is_english_citation_dense_skips_short_pages() -> None:
    """Pages with little English text (chapter headings) are not bibliography."""
    short_page = {
        "en_lines": [
            {"text": "Chapter 1, page 7", "conf": 95},
            {"text": "8. Section 2 of 5", "conf": 95},
        ],
        "arabic_blocks": [{"text": "هذا الكتاب", "conf": 50}],
    }
    assert not pipeline._page_is_english_citation_dense(short_page)


def test_arabic_dominant_true() -> None:
    assert pipeline._arabic_dominant("ا ا ا ا ا hi")


def test_arabic_dominant_false_on_english() -> None:
    assert not pipeline._arabic_dominant("hello world")


def test_normalize_formats_aliases_and_defaults() -> None:
    class Cfg:
        output_formats = ["md", "txt"]
    out = pipeline._normalize_formats(None, Cfg())
    assert out == {"markdown", "text"}


def test_normalize_formats_rejects_unknown() -> None:
    class Cfg:
        output_formats = ["epub"]
    with pytest.raises(ValueError):
        pipeline._normalize_formats(["docx"], Cfg())


def test_normalize_formats_defaults_to_epub() -> None:
    class Cfg:
        output_formats = []
    out = pipeline._normalize_formats(None, Cfg())
    assert out == {"epub"}


def test_cross_page_paragraph_merge() -> None:
    pages = [
        {"pno": 1, "elements": [
            {"kind": "paragraph", "text": "First page text continues without punctuation", "y": 100},
        ]},
        {"pno": 2, "elements": [
            {"kind": "paragraph", "text": "into the next page nicely.", "y": 50},
        ]},
    ]
    pipeline._merge_cross_page_paragraphs(pages)
    # Page 2's paragraph was merged into page 1.
    assert any("continues" in p["text"] and "next page" in p["text"]
               for p in pages[0]["elements"])
    # Page 2's first paragraph removed.
    assert not any(p["kind"] == "paragraph" and p["text"].startswith("into")
                   for p in pages[1]["elements"])


def test_cross_page_paragraph_no_merge_after_period() -> None:
    pages = [
        {"pno": 1, "elements": [
            {"kind": "paragraph", "text": "Sentence ends here.", "y": 100},
        ]},
        {"pno": 2, "elements": [
            {"kind": "paragraph", "text": "next page starts lowercase", "y": 50},
        ]},
    ]
    pipeline._merge_cross_page_paragraphs(pages)
    # No merge; both still present.
    assert pages[0]["elements"][0]["text"] == "Sentence ends here."
    assert pages[1]["elements"][0]["text"].startswith("next page")
