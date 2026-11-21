import argparse
import csv
import io
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
import pypdfium2 as pdfium


def draw(data, output, regular_font, bold_font):
    with data.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        image_diffs = [(int(i), float(random), float(pca)) for i, random, pca in reader]
    pdfmetrics.registerFont(TTFont("TR", str(regular_font)))
    pdfmetrics.registerFont(TTFont("TRBold", str(bold_font)))
    width, height = 465, 249
    stream = io.BytesIO()
    c = Canvas(stream, pagesize=(width, height))
    c.setFillColorRGB(1, 1, 1)
    c.rect(0, 0, width, height, fill=1, stroke=0)
    left, right, bottom, top = 85, 449, 42, 205
    xmin, xmax = -0.7, 5.15
    def x(value):
        return left + (value - xmin) * (right - left) / (xmax - xmin)
    def text_at(px, py, text, size, center=False, bold=False):
        c.setFont("TRBold" if bold else "TR", size)
        c.setFillColorRGB(0.13, 0.17, 0.20)
        if center:
            c.drawCentredString(px, py, text)
        else:
            c.drawString(px, py, text)
    def marker(px, py, style):
        if style == "random":
            c.setFillColorRGB(0.07, 0.34, 0.48)
            c.circle(px, py, 3.1, stroke=0, fill=1)
        else:
            c.setFillColorRGB(0.72, 0.27, 0.06)
            path = c.beginPath()
            path.moveTo(px, py + 4)
            path.lineTo(px + 3.4, py)
            path.lineTo(px, py - 4)
            path.lineTo(px - 3.4, py)
            path.close()
            c.drawPath(path, fill=1, stroke=0)
    marker(100, 236, "random")
    text_at(109, 232.6, "Öğrenilen − sabit rastgele", 9.5)
    marker(274, 236, "pca")
    text_at(283, 232.6, "Öğrenilen − PCA8", 9.5)
    text_at(7, 213, "Test görüntüsü", 8.75, bold=True)
    for k in range(6):
        shade = 0.42 if k == 0 else 0.85
        c.setLineWidth(1.15 if k == 0 else 0.45)
        c.setStrokeColorRGB(shade, shade, shade)
        c.line(x(k), bottom - 3, x(k), top + 5)
        text_at(x(k), bottom - 17, str(k), 9, center=True)
    for i, d_random, d_pca in image_diffs:
        y = top - (i - 1) * (top - bottom) / 9
        text_at(18, y - 3.2, f"Görüntü {i}", 9)
        c.setLineWidth(0.95)
        c.setStrokeColorRGB(0.62, 0.66, 0.68)
        c.line(x(d_random), y, x(d_pca), y)
        marker(x(d_random), y, "random")
        marker(x(d_pca), y, "pca")
    text_at((left + right) / 2, 6, "Görüntü başına makro-F1 farkı (yüzde puan)", 9.5, center=True)
    c.showPage()
    c.save()
    doc = pdfium.PdfDocument(stream.getvalue())
    page = doc[0]
    page.render(scale=4).to_pil().convert("RGB").save(output, dpi=(288, 288))
    page.close()
    doc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("regular_font", type=Path)
    parser.add_argument("bold_font", type=Path)
    args = parser.parse_args()
    draw(args.data, args.output, args.regular_font, args.bold_font)
