"""
md_to_pdf.py - Render the report markdown (with tables and images) to PDF.

  pip install markdown xhtml2pdf
  python scripts/md_to_pdf.py report/Report_P1.md report/Report_P1.pdf

Pure-Python (works on Windows without pandoc / LaTeX). Image paths are resolved
relative to the markdown file.
"""
import os
import re
import sys

import markdown
from xhtml2pdf import pisa

FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "report", "fonts")

CSS_FONTS = """
@font-face { font-family: DejaVu; src: url("{d}/DejaVuSans.ttf"); }
@font-face { font-family: DejaVu; src: url("{d}/DejaVuSans-Bold.ttf"); font-weight: bold; }
@font-face { font-family: DejaVuMono; src: url("{d}/DejaVuSansMono.ttf"); }
"""

CSS = """
@page { size: A4; margin: 1.6cm 1.5cm 1.6cm 1.5cm; }
body { font-family: DejaVu; font-size: 9pt; line-height: 1.35; color: #111; }
h1 { font-size: 17pt; margin: 0 0 6pt 0; }
h2 { font-size: 13pt; margin: 12pt 0 4pt 0; border-bottom: 1px solid #999; padding-bottom: 2pt; }
h3 { font-size: 11pt; margin: 9pt 0 3pt 0; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 6pt 0; }
th, td { border: 0.6px solid #888; padding: 2.5pt 3pt; font-size: 8.2pt; vertical-align: top; }
th { background-color: #e8e8e8; }
code { font-family: DejaVuMono; font-size: 8.4pt; background-color: #f2f2f2; }
pre { font-family: DejaVuMono; font-size: 8pt; background-color: #f2f2f2; padding: 5pt; border: 0.5px solid #ccc; }
blockquote { margin: 4pt 0 6pt 0; padding: 4pt 8pt; background-color: #fff6db; border-left: 3px solid #e0a800; }
.caption { font-size: 8.2pt; color: #444; }
"""


def main(src, dst):
    base = os.path.dirname(os.path.abspath(src))
    with open(src, encoding="utf-8") as f:
        text = f.read()
    text = text.replace("✍️", "&#9998;")
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"])
    # give the first column of wide tables enough room (xhtml2pdf auto-sizing truncates it)
    html = html.replace("<table>\n<thead>\n<tr>\n<th>Variant</th>", "<table>\n<thead>\n<tr>\n<th width='24%'>Variant</th>")

    def _img(m):
        src = m.group(1)
        if not re.match(r"https?:|/|[A-Za-z]:", src):
            src = os.path.join(base, src)
        w = 480
        try:
            from PIL import Image
            iw, ih = Image.open(src).size
            if ih / iw > 1.0:            # tall figure: limit height
                w = int(480 * min(1.0, 1.0 / (ih / iw)))
        except Exception:
            pass
        return f'<img src="{src}" width="{w}"'
    html = re.sub(r'<img[^>]*?src="([^"]+)"', _img, html)
    fonts = CSS_FONTS.replace("{d}", FONT_DIR.replace("\\", "/"))
    doc = f"<html><head><meta charset='utf-8'><style>{fonts}{CSS}</style></head><body>{html}</body></html>"
    dst = os.path.abspath(dst)
    cwd = os.getcwd()
    os.chdir(os.path.commonpath([base, FONT_DIR]))   # xhtml2pdf only reads files below the cwd
    try:
        with open(dst, "wb") as out:
            res = pisa.CreatePDF(doc, dest=out, encoding="utf-8")
    finally:
        os.chdir(cwd)
    if res.err:
        sys.exit(f"PDF rendering failed with {res.err} errors")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(sys.argv[1])[0] + ".pdf")
