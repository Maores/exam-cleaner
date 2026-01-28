# Exam Cleaner

A Windows desktop application that converts exam PDFs into "study-safe" PDFs where colored AND gray answer markings/highlights are removed by converting pages to pure black/white.

## ⚠️ Important Limitation

**Output PDFs are rasterized (image-based).** Text in the output PDF is NOT selectable or searchable. This is by design for the B/W conversion process.

## Features

- **Smart Page Analysis**: Automatically detects pages with colored highlights or gray shading
- **Preset-Based Workflow**: Simple Low/Medium/High strength options (no need to tune sliders)
- **Auto-Guardrails**: Protects dark text and auto-backoffs if removal is too aggressive
- **Interactive Redaction**: Draw, move, resize, and nudge redaction rectangles with keyboard/mouse
- **Modern UI**: Clean, themed interface with ttkbootstrap

## Quick Start

1. **Open PDF** → Load your exam/document
2. **Analyze** → Auto-detect pages with highlights or markings
3. **Review** → Check flagged pages, add manual redactions if needed
4. **Export** → Save the clean, study-safe PDF

## Installation

### Prerequisites
- Python 3.10 or higher
- Windows OS

### Setup

```bash
pip install -r requirements.txt
python app.py
```

## Usage

### Hide Answers Strength Presets

| Preset | Description | Best For |
|--------|-------------|----------|
| **Low** | Conservative removal, gray suppression OFF | Documents with light highlighting |
| **Medium** | Balanced removal with moderate gray suppression | Most exams and marked papers |
| **High** | Aggressive removal, heavy gray suppression | Heavily marked documents |

### Output Quality Options

- **High Quality**: PNG format (lossless, larger file)
- **Balanced**: JPEG 90% quality (smaller file, slight blur possible)

### Redaction Controls

When in Redaction Mode:
- **Click** on rectangle to select
- **Drag** to move selected rectangle
- **Drag corners** to resize
- **Arrow keys** to nudge (1px, Shift+Arrow = 10px)
- **Delete/Backspace** to remove selected

### Advanced Options

Expand "Advanced Options" to manually adjust:
- Threshold value
- Contrast factor
- Gray suppression toggle
- Export DPI

## Troubleshooting

### Highlights Not Removed
1. Increase strength to **Medium** or **High**
2. Check that highlight colors are saturated (not faded gray)

### Text Becoming Faint/Damaged
1. Decrease strength to **Low**
2. The app automatically protects dark pixels

### Large Output File Size
1. Switch to **Balanced** output quality
2. Lower Export DPI in Advanced Options

### Application Won't Start
1. Ensure ttkbootstrap is installed: `pip install ttkbootstrap`
2. Check Python version: `python --version` (need 3.10+)
3. Check `app.log` for errors

## Project Structure

```
├── app.py           # Main GUI application (ttkbootstrap)
├── processor.py     # PDF rendering, processing, page analysis
├── models.py        # Data classes with presets and settings
├── cli.py           # Command-line interface
├── requirements.txt # Python dependencies
├── config.json      # Auto-saved user settings
└── app.log          # Application log file
```

## Technical Notes

- **Dark Text Protection**: Never whitens pixels with V < 60 (HSV value)
- **Safety Cap**: Auto-reduces aggressiveness if > 25% of page affected
- **Gray Suppression**: Blur-based background estimation with ink preservation

## License

MIT License
