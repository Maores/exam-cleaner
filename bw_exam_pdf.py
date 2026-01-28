# bw_exam_pdf.py
# Convert PDFs to grayscale / pure black-and-white by rasterizing pages.
# This removes colored answers/markers (red highlights, etc.) reliably.

import argparse
import os
from io import BytesIO
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:
    raise SystemExit(
        "Missing dependency: PyMuPDF\n"
        "Install with: pip install pymupdf"
    )

try:
    from PIL import Image
except ImportError:
    raise SystemExit(
        "Missing dependency: Pillow\n"
        "Install with: pip install pillow"
    )


def convert_pdf_to_bw(input_pdf: Path, output_pdf: Path, dpi: int, threshold: int | None) -> None:
    doc = fitz.open(str(input_pdf))
    out = fitz.open()

    # Scale factor: 72 PDF points per inch, so dpi/72 is the zoom.
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc[page_index]

        # Render page to RGB pixmap (no alpha).
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB, alpha=False)

        # Convert to PIL Image.
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

        # Convert to grayscale.
        img = img.convert("L")

        # Optional: convert to pure black/white using a threshold.
        if threshold is not None:
            # 0..255; higher threshold -> more pixels become white.
            img = img.point(lambda p: 255 if p >= threshold else 0, mode="1")
            # Convert back to "L" so PDF embedding is consistent.
            img = img.convert("L")

        # Encode image as PNG in-memory.
        buf = BytesIO()
        img.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        # Create a new page sized like the original page (in points).
        rect = page.rect
        new_page = out.new_page(width=rect.width, height=rect.height)

        # Place the rendered image to cover the entire page.
        new_page.insert_image(rect, stream=img_bytes)

    out.save(str(output_pdf))
    out.close()
    doc.close()


def iter_pdfs(input_path: Path):
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        yield input_path
        return
    if input_path.is_dir():
        for p in sorted(input_path.glob("*.pdf")):
            yield p
        return
    raise FileNotFoundError(f"Input path not found or not a PDF/folder: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert exam PDFs to grayscale or pure B/W to hide colored answers."
    )
    parser.add_argument("--input", required=True, help="Input PDF file OR folder containing PDFs.")
    parser.add_argument("--output", required=True, help="Output PDF file OR output folder.")
    parser.add_argument("--dpi", type=int, default=220, help="Render DPI (quality). Typical: 200-300.")
    parser.add_argument(
        "--threshold",
        type=int,
        default=None,
        help="If set (0-255), converts to pure black/white using this threshold. "
             "Example: 180. If omitted, outputs grayscale."
    )

    args = parser.parse_args()
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.output).expanduser().resolve()

    pdfs = list(iter_pdfs(in_path))
    if not pdfs:
        raise SystemExit("No PDFs found.")

    # If multiple PDFs, output must be a folder.
    if len(pdfs) > 1:
        out_path.mkdir(parents=True, exist_ok=True)
        if not out_path.is_dir():
            raise SystemExit("When input is a folder (or multiple PDFs), --output must be a folder.")

        for pdf in pdfs:
            out_file = out_path / f"{pdf.stem}.bw.pdf"
            print(f"Converting: {pdf.name} -> {out_file.name}")
            convert_pdf_to_bw(pdf, out_file, dpi=args.dpi, threshold=args.threshold)
    else:
        # Single PDF: output can be a file path.
        if out_path.is_dir():
            out_file = out_path / f"{pdfs[0].stem}.bw.pdf"
        else:
            out_file = out_path
            out_file.parent.mkdir(parents=True, exist_ok=True)

        print(f"Converting: {pdfs[0].name} -> {out_file.name}")
        convert_pdf_to_bw(pdfs[0], out_file, dpi=args.dpi, threshold=args.threshold)

    print("Done.")


if __name__ == "__main__":
    main()
