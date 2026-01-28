"""
CLI wrapper for PDF to B/W converter.
Provides command-line interface for batch processing.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from models import (
    Settings, ProcessingMode, AutoThresholdMethod, OutputFormat,
    HighlightRemovalSettings, DocumentRedactions, ExportProgress
)
from processor import PDFDocument, PDFExporter, generate_default_output_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert PDF to B/W (study-safe) format with highlight removal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.pdf
  %(prog)s input.pdf -o output.pdf
  %(prog)s input.pdf --dpi 300 --threshold 200
  %(prog)s input.pdf --mode grayscale --auto-threshold otsu
  %(prog)s input.pdf --no-highlight-removal

Note: Output is rasterized (images). Text is NOT selectable/searchable.
"""
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input PDF file path"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Output PDF file path (default: <input>.bw.pdf)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=260,
        help="Export DPI (default: 260)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bw", "grayscale"],
        default="bw",
        help="Processing mode: 'bw' for pure black/white, 'grayscale' for grayscale (default: bw)"
    )
    
    parser.add_argument(
        "--threshold",
        type=int,
        default=210,
        help="Binarization threshold 0-255 (default: 210)"
    )
    
    parser.add_argument(
        "--auto-threshold",
        type=str,
        choices=["none", "otsu", "adaptive"],
        default="none",
        help="Auto-threshold method (default: none)"
    )
    
    parser.add_argument(
        "--auto-contrast",
        action="store_true",
        default=True,
        help="Enable auto contrast (default: enabled)"
    )
    
    parser.add_argument(
        "--no-auto-contrast",
        action="store_true",
        help="Disable auto contrast"
    )
    
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.3,
        help="Contrast factor 1.0-2.0 (default: 1.3)"
    )
    
    parser.add_argument(
        "--no-highlight-removal",
        action="store_true",
        help="Disable colored highlight removal"
    )
    
    parser.add_argument(
        "--sat-threshold",
        type=int,
        default=60,
        help="Saturation threshold for highlight detection 0-255 (default: 60)"
    )
    
    parser.add_argument(
        "--val-threshold",
        type=int,
        default=100,
        help="Value threshold for highlight detection 0-255 (default: 100)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpeg"],
        default="png",
        help="Output image format: 'png' (lossless) or 'jpeg' (smaller) (default: png)"
    )
    
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality 1-100 (default: 85)"
    )
    
    parser.add_argument(
        "--redactions",
        type=str,
        default=None,
        help="Path to redactions JSON file"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def create_settings(args: argparse.Namespace) -> Settings:
    """Create Settings from command-line arguments."""
    mode = ProcessingMode.PURE_BW if args.mode == "bw" else ProcessingMode.GRAYSCALE
    auto_thresh = AutoThresholdMethod(args.auto_threshold)
    output_fmt = OutputFormat(args.format)
    
    auto_contrast = args.auto_contrast and not args.no_auto_contrast
    
    return Settings(
        export_dpi=args.dpi,
        processing_mode=mode,
        threshold=args.threshold,
        auto_threshold_method=auto_thresh,
        auto_contrast=auto_contrast,
        contrast_factor=args.contrast,
        highlight_removal=HighlightRemovalSettings(
            enabled=not args.no_highlight_removal,
            saturation_threshold=args.sat_threshold,
            value_threshold=args.val_threshold
        ),
        output_format=output_fmt,
        jpeg_quality=args.jpeg_quality
    )


def progress_callback(progress: ExportProgress) -> None:
    """Print progress to console."""
    pct = int(progress.progress_fraction * 100)
    print(f"\rProcessing page {progress.current_page}/{progress.total_pages} ({pct}%)", end="", flush=True)


def main() -> int:
    """Main CLI entry point."""
    args = parse_args()
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1
    
    if not input_path.suffix.lower() == ".pdf":
        print(f"Warning: Input file may not be a PDF: {input_path}", file=sys.stderr)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = generate_default_output_path(input_path)
    
    # Create settings
    settings = create_settings(args)
    
    if args.verbose:
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        print(f"DPI:    {settings.export_dpi}")
        print(f"Mode:   {settings.processing_mode.value}")
        print(f"Threshold: {settings.threshold}")
        print(f"Auto-threshold: {settings.auto_threshold_method.value}")
        print(f"Auto-contrast: {settings.auto_contrast}")
        print(f"Contrast factor: {settings.contrast_factor}")
        print(f"Highlight removal: {settings.highlight_removal.enabled}")
        print(f"Output format: {settings.output_format.value}")
        print()
    
    # Load redactions if provided
    redactions: Optional[DocumentRedactions] = None
    if args.redactions:
        redactions_path = Path(args.redactions)
        if redactions_path.exists():
            try:
                redactions = DocumentRedactions.load_from_file(redactions_path)
                if args.verbose:
                    total_rects = sum(len(p.rectangles) for p in redactions.pages.values())
                    print(f"Loaded {total_rects} redaction rectangles from {redactions_path}")
            except Exception as e:
                print(f"Warning: Failed to load redactions: {e}", file=sys.stderr)
    
    # Open document and export
    try:
        doc = PDFDocument(input_path)
        doc.open()
        
        if args.verbose:
            print(f"Document has {doc.page_count} pages")
            print()
        
        print(f"Processing {input_path.name}...")
        
        exporter = PDFExporter(
            doc,
            settings,
            redactions,
            progress_callback if not args.verbose else None
        )
        
        success = exporter.export(output_path)
        
        print()  # New line after progress
        
        doc.close()
        
        if success:
            print(f"Successfully exported to: {output_path}")
            return 0
        else:
            error = exporter.progress.error or "Unknown error"
            print(f"Export failed: {error}", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
