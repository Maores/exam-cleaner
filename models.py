"""
Data models for PDF to B/W converter application.
Contains dataclasses for Settings, Redactions, and related types.
"""

from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import json
from pathlib import Path


class ProcessingMode(Enum):
    """Image processing mode for output."""
    GRAYSCALE = "grayscale"
    PURE_BW = "pure_bw"


class AutoThresholdMethod(Enum):
    """Auto-threshold calculation method."""
    NONE = "none"
    OTSU = "otsu"
    ADAPTIVE = "adaptive"


class OutputFormat(Enum):
    """Output image encoding format."""
    PNG = "png"
    JPEG = "jpeg"


class StrengthPreset(Enum):
    """Hide Answers Strength preset levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class OutputPreset(Enum):
    """Output quality preset."""
    HIGH_QUALITY = "high_quality"  # PNG
    BALANCED = "balanced"  # JPEG 90


class AnalysisConfidence(Enum):
    """Confidence level for page analysis."""
    LIKELY_MARKED = "likely_marked"
    MAYBE_MARKED = "maybe_marked"
    CLEAN = "clean"


@dataclass
class AnalysisResult:
    """Result of analyzing a single page for markings."""
    page_number: int
    confidence: AnalysisConfidence = AnalysisConfidence.CLEAN
    color_score: float = 0.0  # Fraction of colored pixels
    gray_score: float = 0.0   # Fraction of gray shaded area
    ink_score: float = 0.0    # Ink density anomaly score
    
    @property
    def is_flagged(self) -> bool:
        """Check if this page should be flagged for review."""
        return self.confidence != AnalysisConfidence.CLEAN
    
    @property
    def flag_reasons(self) -> list[str]:
        """Get human-readable reasons why page was flagged."""
        reasons = []
        if self.color_score > 0.005:
            reasons.append("color highlights")
        if self.gray_score > 0.02:
            reasons.append("gray shading")
        if self.ink_score > 0.1:
            reasons.append("dense markings")
        return reasons


# Preset mappings: maps StrengthPreset to processing parameters
PRESET_MAPPINGS = {
    StrengthPreset.LOW: {
        "threshold": 200,
        "saturation_threshold": 80,
        "value_threshold": 140,
        "gray_suppression": 0.0,  # OFF for Low
        "gray_suppression_enabled": False,
    },
    StrengthPreset.MEDIUM: {
        "threshold": 210,
        "saturation_threshold": 60,
        "value_threshold": 100,
        "gray_suppression": 0.5,
        "gray_suppression_enabled": True,
    },
    StrengthPreset.HIGH: {
        "threshold": 220,
        "saturation_threshold": 40,
        "value_threshold": 80,
        "gray_suppression": 0.7,
        "gray_suppression_enabled": True,
    },
}

# Dark text protection threshold - never whiten pixels with V below this
DARK_TEXT_PROTECTION_V = 60

# Safety cap: if removal affects more than this fraction of pixels, auto-backoff
MAX_REMOVAL_FRACTION = 0.25


@dataclass
class RedactionRect:
    """
    A redaction rectangle in normalized coordinates (0..1).
    Coordinates are relative to the rendered page image dimensions.
    """
    x0: float  # Left edge (0..1)
    y0: float  # Top edge (0..1)
    x1: float  # Right edge (0..1)
    y1: float  # Bottom edge (0..1)
    id: str = field(default_factory=lambda: str(id(object())))  # Unique ID for selection

    def to_pixel_coords(self, width: int, height: int) -> tuple[int, int, int, int]:
        """Convert normalized coords to pixel coordinates for given image dimensions."""
        return (
            int(self.x0 * width),
            int(self.y0 * height),
            int(self.x1 * width),
            int(self.y1 * height)
        )
    
    def contains_point(self, x: float, y: float, tolerance: float = 0.02) -> bool:
        """Check if a normalized point is inside or near this rectangle."""
        return (self.x0 - tolerance <= x <= self.x1 + tolerance and
                self.y0 - tolerance <= y <= self.y1 + tolerance)
    
    def get_corner_at(self, x: float, y: float, tolerance: float = 0.03) -> Optional[str]:
        """
        Check if point is near a corner. Returns corner name or None.
        Corner names: 'tl', 'tr', 'bl', 'br' (top-left, top-right, etc.)
        """
        corners = {
            'tl': (self.x0, self.y0),
            'tr': (self.x1, self.y0),
            'bl': (self.x0, self.y1),
            'br': (self.x1, self.y1),
        }
        for corner_name, (cx, cy) in corners.items():
            if abs(x - cx) < tolerance and abs(y - cy) < tolerance:
                return corner_name
        return None
    
    def move(self, dx: float, dy: float) -> None:
        """Move rectangle by normalized delta."""
        self.x0 = max(0, min(1 - (self.x1 - self.x0), self.x0 + dx))
        self.y0 = max(0, min(1 - (self.y1 - self.y0), self.y0 + dy))
        width = self.x1 - self.x0 + dx if self.x0 + dx >= 0 else self.x1 - self.x0
        height = self.y1 - self.y0 + dy if self.y0 + dy >= 0 else self.y1 - self.y0
        # Recalculate based on clamped x0, y0
        old_w = self.x1 - self.x0 + dx
        old_h = self.y1 - self.y0 + dy
        self.x1 = self.x0 + (self.x1 - self.x0 + dx - (self.x0 + dx - max(0, min(1 - old_w, self.x0 + dx))))
        self.y1 = self.y0 + (self.y1 - self.y0 + dy - (self.y0 + dy - max(0, min(1 - old_h, self.y0 + dy))))
        # Simpler approach: just offset and clamp
        w = self.x1 - self.x0
        h = self.y1 - self.y0
        self.x0 = max(0, min(1 - w, self.x0))
        self.x1 = self.x0 + w
        self.y0 = max(0, min(1 - h, self.y0))
        self.y1 = self.y0 + h
    
    def resize_corner(self, corner: str, new_x: float, new_y: float) -> None:
        """Resize by moving a specific corner to new normalized coordinates."""
        new_x = max(0, min(1, new_x))
        new_y = max(0, min(1, new_y))
        
        if corner == 'tl':
            self.x0, self.y0 = min(new_x, self.x1 - 0.01), min(new_y, self.y1 - 0.01)
        elif corner == 'tr':
            self.x1, self.y0 = max(new_x, self.x0 + 0.01), min(new_y, self.y1 - 0.01)
        elif corner == 'bl':
            self.x0, self.y1 = min(new_x, self.x1 - 0.01), max(new_y, self.y0 + 0.01)
        elif corner == 'br':
            self.x1, self.y1 = max(new_x, self.x0 + 0.01), max(new_y, self.y0 + 0.01)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {"x0": self.x0, "y0": self.y0, "x1": self.x1, "y1": self.y1, "id": self.id}

    @classmethod
    def from_dict(cls, data: dict) -> "RedactionRect":
        """Deserialize from dictionary."""
        rect = cls(x0=data["x0"], y0=data["y0"], x1=data["x1"], y1=data["y1"])
        if "id" in data:
            rect.id = data["id"]
        return rect


@dataclass
class PageRedactions:
    """Redaction rectangles for a single page."""
    page_number: int
    rectangles: list[RedactionRect] = field(default_factory=list)

    def add_rect(self, rect: RedactionRect) -> None:
        """Add a redaction rectangle."""
        self.rectangles.append(rect)

    def remove_last(self) -> Optional[RedactionRect]:
        """Remove and return the last rectangle, or None if empty."""
        if self.rectangles:
            return self.rectangles.pop()
        return None

    def clear(self) -> None:
        """Remove all rectangles."""
        self.rectangles.clear()
    
    def find_by_id(self, rect_id: str) -> Optional[RedactionRect]:
        """Find a rectangle by its ID."""
        for rect in self.rectangles:
            if rect.id == rect_id:
                return rect
        return None
    
    def remove_by_id(self, rect_id: str) -> Optional[RedactionRect]:
        """Remove and return a rectangle by its ID."""
        for i, rect in enumerate(self.rectangles):
            if rect.id == rect_id:
                return self.rectangles.pop(i)
        return None
    
    def find_at_point(self, x: float, y: float) -> Optional[RedactionRect]:
        """Find the topmost rectangle containing the given normalized point."""
        # Search in reverse order (topmost = last added)
        for rect in reversed(self.rectangles):
            if rect.contains_point(x, y):
                return rect
        return None

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "page_number": self.page_number,
            "rectangles": [r.to_dict() for r in self.rectangles]
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PageRedactions":
        """Deserialize from dictionary."""
        return cls(
            page_number=data["page_number"],
            rectangles=[RedactionRect.from_dict(r) for r in data.get("rectangles", [])]
        )


@dataclass
class DocumentRedactions:
    """All redactions for a document."""
    file_path: str
    pages: dict[int, PageRedactions] = field(default_factory=dict)

    def get_page(self, page_num: int) -> PageRedactions:
        """Get or create redactions for a page."""
        if page_num not in self.pages:
            self.pages[page_num] = PageRedactions(page_number=page_num)
        return self.pages[page_num]

    def copy_to_all_pages(self, source_page: int, total_pages: int) -> None:
        """Copy redactions from source page to all other pages."""
        source = self.get_page(source_page)
        for page_num in range(total_pages):
            if page_num != source_page:
                target = self.get_page(page_num)
                target.rectangles = [
                    RedactionRect(r.x0, r.y0, r.x1, r.y1)
                    for r in source.rectangles
                ]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "file_path": self.file_path,
            "pages": {str(k): v.to_dict() for k, v in self.pages.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocumentRedactions":
        """Deserialize from dictionary."""
        doc = cls(file_path=data.get("file_path", ""))
        for k, v in data.get("pages", {}).items():
            doc.pages[int(k)] = PageRedactions.from_dict(v)
        return doc

    def save_to_file(self, path: Path) -> None:
        """Save redactions to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "DocumentRedactions":
        """Load redactions from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


@dataclass
class HighlightRemovalSettings:
    """Settings for colored highlight removal."""
    enabled: bool = True
    saturation_threshold: int = 60  # 0-255, pixels with S > this are considered colored
    value_threshold: int = 100  # 0-255, pixels with V > this (and S > sat) are highlights


@dataclass
class GraySuppressSettings:
    """Settings for gray shading suppression."""
    enabled: bool = True
    intensity: float = 0.5  # 0.0-1.0, how aggressive to suppress gray shading


@dataclass
class Settings:
    """Application settings for PDF processing."""
    # Preset selections (for new UI)
    strength_preset: StrengthPreset = StrengthPreset.MEDIUM
    output_preset: OutputPreset = OutputPreset.HIGH_QUALITY
    
    # Export settings
    export_dpi: int = 260
    processing_mode: ProcessingMode = ProcessingMode.PURE_BW
    
    # Threshold settings
    threshold: int = 210  # Manual threshold (0-255)
    auto_threshold_method: AutoThresholdMethod = AutoThresholdMethod.NONE
    
    # Contrast settings
    auto_contrast: bool = False  # Disabled by default - can cause issues with scanned docs
    contrast_factor: float = 1.3  # 1.0-2.0
    
    # Highlight removal
    highlight_removal: HighlightRemovalSettings = field(default_factory=HighlightRemovalSettings)
    
    # Gray shading suppression (NEW)
    gray_suppression: GraySuppressSettings = field(default_factory=GraySuppressSettings)
    
    # Output format
    output_format: OutputFormat = OutputFormat.PNG
    jpeg_quality: int = 90  # 1-100, increased default for better quality
    
    # Preview settings (internal)
    preview_dpi_cap: int = 200  # Max DPI for preview rendering
    
    def apply_preset(self, preset: StrengthPreset) -> None:
        """Apply a strength preset, updating all related settings."""
        self.strength_preset = preset
        mapping = PRESET_MAPPINGS[preset]
        
        self.threshold = mapping["threshold"]
        self.highlight_removal.saturation_threshold = mapping["saturation_threshold"]
        self.highlight_removal.value_threshold = mapping["value_threshold"]
        self.gray_suppression.enabled = mapping["gray_suppression_enabled"]
        self.gray_suppression.intensity = mapping["gray_suppression"]
    
    def apply_output_preset(self, preset: OutputPreset) -> None:
        """Apply an output quality preset."""
        self.output_preset = preset
        if preset == OutputPreset.HIGH_QUALITY:
            self.output_format = OutputFormat.PNG
        else:  # BALANCED
            self.output_format = OutputFormat.JPEG
            self.jpeg_quality = 90

    def get_preview_dpi(self) -> int:
        """Get the DPI to use for preview rendering."""
        return min(self.export_dpi, self.preview_dpi_cap)

    def to_dict(self) -> dict:
        """Serialize settings to dictionary."""
        return {
            "strength_preset": self.strength_preset.value,
            "output_preset": self.output_preset.value,
            "export_dpi": self.export_dpi,
            "processing_mode": self.processing_mode.value,
            "threshold": self.threshold,
            "auto_threshold_method": self.auto_threshold_method.value,
            "auto_contrast": self.auto_contrast,
            "contrast_factor": self.contrast_factor,
            "highlight_removal": {
                "enabled": self.highlight_removal.enabled,
                "saturation_threshold": self.highlight_removal.saturation_threshold,
                "value_threshold": self.highlight_removal.value_threshold
            },
            "gray_suppression": {
                "enabled": self.gray_suppression.enabled,
                "intensity": self.gray_suppression.intensity
            },
            "output_format": self.output_format.value,
            "jpeg_quality": self.jpeg_quality,
            "preview_dpi_cap": self.preview_dpi_cap
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        """Deserialize settings from dictionary."""
        settings = cls()
        
        if "strength_preset" in data:
            try:
                settings.strength_preset = StrengthPreset(data["strength_preset"])
            except ValueError:
                pass
        if "output_preset" in data:
            try:
                settings.output_preset = OutputPreset(data["output_preset"])
            except ValueError:
                pass
        if "export_dpi" in data:
            settings.export_dpi = int(data["export_dpi"])
        if "processing_mode" in data:
            settings.processing_mode = ProcessingMode(data["processing_mode"])
        if "threshold" in data:
            settings.threshold = int(data["threshold"])
        if "auto_threshold_method" in data:
            settings.auto_threshold_method = AutoThresholdMethod(data["auto_threshold_method"])
        if "auto_contrast" in data:
            settings.auto_contrast = bool(data["auto_contrast"])
        if "contrast_factor" in data:
            settings.contrast_factor = float(data["contrast_factor"])
        if "highlight_removal" in data:
            hr = data["highlight_removal"]
            settings.highlight_removal = HighlightRemovalSettings(
                enabled=hr.get("enabled", True),
                saturation_threshold=hr.get("saturation_threshold", 60),
                value_threshold=hr.get("value_threshold", 100)
            )
        if "gray_suppression" in data:
            gs = data["gray_suppression"]
            settings.gray_suppression = GraySuppressSettings(
                enabled=gs.get("enabled", True),
                intensity=gs.get("intensity", 0.5)
            )
        if "output_format" in data:
            settings.output_format = OutputFormat(data["output_format"])
        if "jpeg_quality" in data:
            settings.jpeg_quality = int(data["jpeg_quality"])
        if "preview_dpi_cap" in data:
            settings.preview_dpi_cap = int(data["preview_dpi_cap"])
        
        return settings

    def save_to_file(self, path: Path) -> None:
        """Save settings to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, path: Path) -> "Settings":
        """Load settings from JSON file, or return defaults if file doesn't exist."""
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return cls.from_dict(json.load(f))
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        return cls()


@dataclass
class ExportProgress:
    """Progress information for export operation."""
    current_page: int = 0
    total_pages: int = 0
    cancelled: bool = False
    error: Optional[str] = None
    completed: bool = False

    @property
    def progress_fraction(self) -> float:
        """Get progress as a fraction (0.0 to 1.0)."""
        if self.total_pages == 0:
            return 0.0
        return self.current_page / self.total_pages


@dataclass
class PageGeometry:
    """
    Geometry information for a PDF page.
    Uses CropBox as the canonical visible area and respects rotation.
    """
    page_number: int
    cropbox_width: float  # Width in points after rotation
    cropbox_height: float  # Height in points after rotation
    rotation: int  # Rotation in degrees (0, 90, 180, 270)

    def get_render_matrix(self, dpi: float) -> "fitz.Matrix":
        """
        Get the transformation matrix for rendering at the given DPI.
        This is used by the processor module.
        """
        import fitz
        scale = dpi / 72.0
        return fitz.Matrix(scale, scale)

    def get_image_dimensions(self, dpi: float) -> tuple[int, int]:
        """Get expected image dimensions when rendered at given DPI."""
        scale = dpi / 72.0
        return (
            int(self.cropbox_width * scale),
            int(self.cropbox_height * scale)
        )
