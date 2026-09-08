"""Render reproducible original-text references, including damaged RTL scans.

Run: python -m scripts.create_benchmark_corpus /tmp/quire-benchmark
"""

from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path

import fitz
from PIL import Image, ImageEnhance, ImageFilter

from quire.studio.project import file_hash
from quire.studio.publish import _css, _fonts


def create(output: Path) -> Path:
    import shutil
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_fonts(), output / "fonts", dirs_exist_ok=True)
    corpus = json.loads((Path(__file__).resolve().parents[1] / "benchmarks/corpus.json").read_text())
    cases = []
    for index, case in enumerate(corpus["cases"]):
        blocks = case["text"].split("\n\n")
        language = case["language"]
        body = f'<h1 lang="{language}" dir="{"rtl" if language in {"fa", "ar"} else "ltr"}">{html.escape(blocks[0])}</h1>'
        for block in blocks[1:]:
            is_latin = block[0].isascii()
            tag = "en" if is_latin else language
            body += f'<p lang="{tag}" dir="{"ltr" if is_latin else "rtl"}">{html.escape(block)}</p>'
        story = fitz.Story(html=body, user_css=_css("reading"), archive=str(output))
        pdf = story.write_with_links(lambda n, filled: (fitz.Rect(0, 0, 595, 842), fitz.Rect(55, 55, 540, 780), None))
        if case["scan"]:
            scanned = fitz.open()
            for page in pdf:
                image = Image.open(io.BytesIO(page.get_pixmap(dpi=170).tobytes("png")))
                if case.get("degrade"):
                    image = ImageEnhance.Contrast(image.convert("L")).enhance(.5).filter(ImageFilter.GaussianBlur(.45))
                if case.get("rotate"):
                    image = image.rotate(90, expand=True)
                    target = scanned.new_page(width=842, height=595)
                    target.set_rotation(90)
                else:
                    target = scanned.new_page(width=595, height=842)
                stream = io.BytesIO()
                image.save(stream, "PNG")
                target.insert_image(target.mediabox, stream=stream.getvalue())
            pdf.close()
            pdf = scanned
        path = output / f"case-{index + 1}.pdf"
        pdf.save(path)
        pdf.close()
        cases.append({"name": case["name"], "pdf": path.name, "language": case["language"],
                      "reference": case["text"], "reference_provenance": "authored source text; generated scan",
                      "limits": {"character_error_rate": .05 if case["scan"] else .01, "omitted_words": 1},
                      "source_sha256": file_hash(path)})
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"cases": cases}, ensure_ascii=False, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(create(args.output))
