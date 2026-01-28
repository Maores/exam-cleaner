"""
PDF processing module for B/W conversion.
Handles PDF rendering, image processing, and export operations.
"""

import fitz  # PyMuPDF
import numpy as np
from PIL import Image, ImageEnhance
from pathlib import Path
from typing import Optional, Callable
import tempfile
import shutil
import logging
import io

from models import (
    Settings, ProcessingMode, AutoThresholdMethod, OutputFormat,
    DocumentRedactions, PageRedactions, RedactionRect, PageGeometry,
    ExportProgress, AnalysisResult, AnalysisConfidence,
    DARK_TEXT_PROTECTION_V, MAX_REMOVAL_FRACTION
)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PDFDocument:
    """Wrapper for PDF document handling with geometry caching."""
    
    def __init__(self, path: Path):
        self.path = path
        self._doc: Optional[fitz.Document] = None
        self._page_geometries: dict[int, PageGeometry] = {}
    
    def open(self) -> None:
        """Open the PDF document."""
        if self._doc is not None:
            self.close()
        self._doc = fitz.open(str(self.path))
        self._cache_geometries()
    
    def close(self) -> None:
        """Close the PDF document."""
        if self._doc:
            self._doc.close()
            self._doc = None
            self._page_geometries.clear()
    
    def __enter__(self) -> "PDFDocument":
        self.open()
        return self
    
    def __exit__(self, *args) -> None:
        self.close()
    
    @property
    def is_open(self) -> bool:
        """Check if document is open."""
        return self._doc is not None
    
    @property
    def page_count(self) -> int:
        """Get total number of pages."""
        if self._doc is None:
            return 0
        return len(self._doc)
    
    def _cache_geometries(self) -> None:
        """Cache page geometries for all pages."""
        if self._doc is None:
            return
        
        for page_num in range(len(self._doc)):
            page = self._doc[page_num]
            # Use CropBox as canonical visible area (falls back to MediaBox if not set)
            cropbox = page.cropbox
            rotation = page.rotation
            
            # After rotation, width/height may swap
            if rotation in (90, 270):
                width = cropbox.height
                height = cropbox.width
            else:
                width = cropbox.width
                height = cropbox.height
            
            self._page_geometries[page_num] = PageGeometry(
                page_number=page_num,
                cropbox_width=width,
                cropbox_height=height,
                rotation=rotation
            )
    
    def get_page_geometry(self, page_num: int) -> Optional[PageGeometry]:
        """Get cached geometry for a page."""
        return self._page_geometries.get(page_num)
    
    def render_page(self, page_num: int, dpi: float) -> Image.Image:
        """
        Render a page to RGB PIL Image at the given DPI.
        Uses CropBox and respects rotation.
        """
        if self._doc is None:
            raise RuntimeError("Document not open")
        
        page = self._doc[page_num]
        
        # Create transformation matrix for the given DPI
        scale = dpi / 72.0
        mat = fitz.Matrix(scale, scale)
        
        # Render with CropBox clip
        pix = page.get_pixmap(matrix=mat, clip=page.cropbox, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        
        return img


class PageAnalyzer:
    """
    Analyzes PDF pages to detect likely marked/highlighted content.
    Uses fast heuristics at low DPI for quick scanning.
    """
    
    ANALYSIS_DPI = 120  # Low DPI for fast analysis
    
    # Thresholds for scoring
    COLOR_SAT_THRESHOLD = 60   # Saturation threshold for color detection
    COLOR_VAL_THRESHOLD = 80   # Value threshold for color detection
    GRAY_LOW = 160             # Lower bound for gray shading detection
    GRAY_HIGH = 230            # Upper bound for gray shading detection
    
    # Confidence thresholds
    LIKELY_THRESHOLD = 0.02    # Above this = "likely marked"
    MAYBE_THRESHOLD = 0.005    # Above this = "maybe marked"
    
    def __init__(self, document: "PDFDocument"):
        self.document = document
        self._results: dict[int, AnalysisResult] = {}
        self._cancelled = False
    
    def cancel(self) -> None:
        """Cancel ongoing analysis."""
        self._cancelled = True
    
    def get_results(self) -> dict[int, AnalysisResult]:
        """Get cached analysis results."""
        return self._results
    
    def clear_results(self) -> None:
        """Clear cached results."""
        self._results.clear()
    
    def analyze_page(self, page_num: int) -> AnalysisResult:
        """
        Analyze a single page for markings.
        Returns cached result if already analyzed.
        """
        if page_num in self._results:
            return self._results[page_num]
        
        # Render at low DPI
        img = self.document.render_page(page_num, self.ANALYSIS_DPI)
        rgb_array = np.array(img, dtype=np.float32)
        
        # Calculate scores
        color_score = self._calc_color_score(rgb_array)
        gray_score = self._calc_gray_score(rgb_array)
        ink_score = self._calc_ink_score(rgb_array)
        
        # Determine confidence
        total_score = color_score + gray_score + ink_score * 0.5
        
        if total_score > self.LIKELY_THRESHOLD:
            confidence = AnalysisConfidence.LIKELY_MARKED
        elif total_score > self.MAYBE_THRESHOLD:
            confidence = AnalysisConfidence.MAYBE_MARKED
        else:
            confidence = AnalysisConfidence.CLEAN
        
        result = AnalysisResult(
            page_number=page_num,
            confidence=confidence,
            color_score=color_score,
            gray_score=gray_score,
            ink_score=ink_score
        )
        
        self._results[page_num] = result
        return result
    
    def analyze_all(
        self,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[AnalysisResult]:
        """
        Analyze all pages in the document.
        
        Args:
            progress_callback: Called with (current_page, total_pages)
        
        Returns:
            List of AnalysisResult for all pages
        """
        self._cancelled = False
        results = []
        total = self.document.page_count
        
        for page_num in range(total):
            if self._cancelled:
                break
            
            result = self.analyze_page(page_num)
            results.append(result)
            
            if progress_callback:
                progress_callback(page_num + 1, total)
        
        return results
    
    def get_flagged_pages(self) -> list[AnalysisResult]:
        """Get only the pages that were flagged (likely or maybe marked)."""
        return [r for r in self._results.values() if r.is_flagged]
    
    def _calc_color_score(self, rgb_array: np.ndarray) -> float:
        """Calculate fraction of colored pixels (potential highlights)."""
        # Convert to HSV space manually
        rgb_norm = rgb_array / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        
        v = np.maximum(np.maximum(r, g), b)
        min_rgb = np.minimum(np.minimum(r, g), b)
        diff = v - min_rgb
        
        s = np.zeros_like(v)
        nonzero_v = v > 0
        s[nonzero_v] = diff[nonzero_v] / v[nonzero_v]
        
        # Convert to 0-255 scale
        s_255 = (s * 255).astype(np.uint8)
        v_255 = (v * 255).astype(np.uint8)
        
        # Count colored pixels (high sat, high value)
        color_mask = (s_255 > self.COLOR_SAT_THRESHOLD) & (v_255 > self.COLOR_VAL_THRESHOLD)
        
        return np.sum(color_mask) / color_mask.size
    
    def _calc_gray_score(self, rgb_array: np.ndarray) -> float:
        """Calculate fraction of gray-shaded pixels."""
        # Convert to grayscale
        gray = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
        
        # Detect mid-gray regions
        gray_mask = (gray > self.GRAY_LOW) & (gray < self.GRAY_HIGH)
        
        # Calculate local variance to filter out texture
        # Use a simple approach: check if surrounding pixels are similar
        # For speed, just count the raw gray pixels
        return np.sum(gray_mask) / gray_mask.size
    
    def _calc_ink_score(self, rgb_array: np.ndarray) -> float:
        """Calculate ink density (fraction of dark pixels)."""
        gray = 0.299 * rgb_array[:, :, 0] + 0.587 * rgb_array[:, :, 1] + 0.114 * rgb_array[:, :, 2]
        
        # Count very dark pixels (ink)
        dark_mask = gray < 50
        ink_density = np.sum(dark_mask) / dark_mask.size
        
        # Return anomaly score: higher if above typical exam baseline (~5%)
        baseline = 0.05
        return max(0, ink_density - baseline)


class ImageProcessor:
    """Image processing operations for B/W conversion."""
    
    @staticmethod
    def apply_redactions(
        img: Image.Image,
        redactions: Optional[PageRedactions]
    ) -> Image.Image:
        """Apply redaction rectangles by painting them white."""
        if redactions is None or not redactions.rectangles:
            return img
        
        img = img.copy()
        width, height = img.size
        pixels = img.load()
        
        for rect in redactions.rectangles:
            x0, y0, x1, y1 = rect.to_pixel_coords(width, height)
            # Clamp to image bounds
            x0 = max(0, min(x0, width - 1))
            x1 = max(0, min(x1, width))
            y0 = max(0, min(y0, height - 1))
            y1 = max(0, min(y1, height))
            
            for y in range(y0, y1):
                for x in range(x0, x1):
                    pixels[x, y] = (255, 255, 255)
        
        return img
    
    @staticmethod
    def apply_redactions_numpy(
        img_array: np.ndarray,
        redactions: Optional[PageRedactions]
    ) -> np.ndarray:
        """Apply redaction rectangles using NumPy (faster for large areas)."""
        if redactions is None or not redactions.rectangles:
            return img_array
        
        img_array = img_array.copy()
        height, width = img_array.shape[:2]
        
        for rect in redactions.rectangles:
            x0, y0, x1, y1 = rect.to_pixel_coords(width, height)
            # Clamp to image bounds
            x0 = max(0, min(x0, width - 1))
            x1 = max(0, min(x1, width))
            y0 = max(0, min(y0, height - 1))
            y1 = max(0, min(y1, height))
            
            img_array[y0:y1, x0:x1] = 255
        
        return img_array
    
    @staticmethod
    def remove_colored_highlights(
        img: Image.Image,
        saturation_threshold: int = 60,
        value_threshold: int = 100,
        safety_cap: bool = True
    ) -> tuple[Image.Image, float]:
        """
        Remove colored highlights by converting them to white.
        Uses HSV color space to identify colored regions.
        
        GUARDRAILS:
        - Protects dark text pixels (V < DARK_TEXT_PROTECTION_V) regardless of saturation
        - Safety cap: if removal affects > MAX_REMOVAL_FRACTION, auto-backoff
        
        Args:
            img: RGB PIL Image
            saturation_threshold: Pixels with S > this are considered colored (0-255)
            value_threshold: Pixels with V > this (and S > sat_thresh) are highlights (0-255)
            safety_cap: If True, limit removal to MAX_REMOVAL_FRACTION of pixels
        
        Returns:
            Tuple of (processed image, fraction of pixels affected)
        """
        # Convert to numpy array
        rgb_array = np.array(img, dtype=np.float32)
        
        # Calculate grayscale luminance
        grayscale = (
            0.299 * rgb_array[:, :, 0] +
            0.587 * rgb_array[:, :, 1] +
            0.114 * rgb_array[:, :, 2]
        )
        
        # Convert RGB to HSV manually using NumPy
        rgb_norm = rgb_array / 255.0
        r, g, b = rgb_norm[:, :, 0], rgb_norm[:, :, 1], rgb_norm[:, :, 2]
        
        # Value is the max of RGB
        v = np.maximum(np.maximum(r, g), b)
        
        # Saturation
        min_rgb = np.minimum(np.minimum(r, g), b)
        diff = v - min_rgb
        
        s = np.zeros_like(v)
        nonzero_v = v > 0
        s[nonzero_v] = diff[nonzero_v] / v[nonzero_v]
        
        # Convert to 0-255 scale
        s_255 = (s * 255).astype(np.uint8)
        v_255 = (v * 255).astype(np.uint8)
        
        # GUARDRAIL 1: Dark text protection - never whiten dark pixels
        # Protect pixels with V < DARK_TEXT_PROTECTION_V
        dark_text_mask = v_255 < DARK_TEXT_PROTECTION_V
        
        # Bright enough in grayscale to be a highlight (not scanned text noise)
        grayscale_bright = grayscale > 180
        
        # Create highlight mask with all guardrails
        highlight_mask = (
            (s_255 > saturation_threshold) & 
            (v_255 > value_threshold) &
            grayscale_bright &
            ~dark_text_mask  # Never affect dark text
        )
        
        removal_fraction = np.sum(highlight_mask) / highlight_mask.size
        
        # GUARDRAIL 2: Safety cap - if too many pixels affected, backoff
        if safety_cap and removal_fraction > MAX_REMOVAL_FRACTION:
            logger.warning(f"Color removal would affect {removal_fraction:.1%} of pixels, "
                          f"exceeding {MAX_REMOVAL_FRACTION:.0%} cap. Reducing aggressiveness.")
            # Backoff: increase thresholds progressively
            for backoff_sat, backoff_val in [(80, 120), (100, 140), (120, 160)]:
                highlight_mask = (
                    (s_255 > backoff_sat) & 
                    (v_255 > backoff_val) &
                    grayscale_bright &
                    ~dark_text_mask
                )
                removal_fraction = np.sum(highlight_mask) / highlight_mask.size
                if removal_fraction <= MAX_REMOVAL_FRACTION:
                    break
        
        # Apply mask
        result = rgb_array.copy()
        result[highlight_mask] = 255
        
        return Image.fromarray(result.astype(np.uint8)), removal_fraction
    
    @staticmethod
    def convert_to_grayscale(img: Image.Image) -> Image.Image:
        """Convert RGB image to grayscale."""
        return img.convert("L")
    
    @staticmethod
    def apply_auto_contrast(img: Image.Image, clip_percent: float = 1.0) -> Image.Image:
        """
        Apply automatic contrast enhancement using percentile-based stretching.
        
        This is more robust than min/max stretching as it ignores outliers.
        
        Args:
            img: Grayscale PIL Image
            clip_percent: Percentage of pixels to clip at each end (default 1%)
        
        Returns:
            Contrast-enhanced image
        """
        img_array = np.array(img, dtype=np.float32)
        
        # Use percentile-based clipping instead of min/max
        # This is more robust to outliers (dust, artifacts, etc.)
        low_val = np.percentile(img_array, clip_percent)
        high_val = np.percentile(img_array, 100 - clip_percent)
        
        if high_val > low_val:
            # Clip and stretch
            stretched = np.clip(img_array, low_val, high_val)
            stretched = ((stretched - low_val) * 255 / (high_val - low_val))
            return Image.fromarray(stretched.astype(np.uint8))
        return img
    
    @staticmethod
    def apply_contrast_factor(img: Image.Image, factor: float) -> Image.Image:
        """Apply contrast enhancement with a given factor."""
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(factor)
    
    @staticmethod
    def calculate_otsu_threshold(img: Image.Image) -> int:
        """
        Calculate optimal threshold using Otsu's method with NumPy.
        
        Args:
            img: Grayscale PIL Image
        
        Returns:
            Optimal threshold value (0-255)
        """
        img_array = np.array(img).flatten()
        
        # Calculate histogram
        hist, _ = np.histogram(img_array, bins=256, range=(0, 256))
        hist = hist.astype(np.float64)
        
        # Normalize histogram
        total_pixels = img_array.size
        hist_norm = hist / total_pixels
        
        # Calculate cumulative sums
        cumsum = np.cumsum(hist_norm)
        cumsum_mean = np.cumsum(hist_norm * np.arange(256))
        
        # Global mean
        global_mean = cumsum_mean[-1]
        
        # Calculate between-class variance for all thresholds
        # Avoid division by zero
        w0 = cumsum
        w1 = 1 - cumsum
        
        # Mean of class 0 and class 1
        with np.errstate(divide='ignore', invalid='ignore'):
            mu0 = cumsum_mean / w0
            mu1 = (global_mean - cumsum_mean) / w1
        
        # Between-class variance
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        # Handle NaN values
        variance = np.nan_to_num(variance)
        
        # Find threshold that maximizes variance
        threshold = np.argmax(variance)
        
        return int(threshold)
    
    @staticmethod
    def binarize(img: Image.Image, threshold: int) -> Image.Image:
        """
        Binarize grayscale image using the given threshold.
        Pixels >= threshold become white, others become black.
        """
        img_array = np.array(img)
        binary = np.where(img_array >= threshold, 255, 0).astype(np.uint8)
        return Image.fromarray(binary)
    
    @staticmethod
    def suppress_gray_shading(
        img: Image.Image,
        intensity: float = 0.5,
        safety_check: bool = True
    ) -> Image.Image:
        """
        Suppress gray shading/backgrounds using adaptive local thresholding.
        
        This helps remove light gray answer boxes/shading while preserving text.
        Uses a blur-based background estimation approach.
        
        Args:
            img: Grayscale PIL Image
            intensity: 0.0-1.0, how aggressive to suppress (higher = more removal)
            safety_check: If True, reduce intensity if too much ink is lost
        
        Returns:
            Image with gray shading suppressed
        """
        img_array = np.array(img, dtype=np.float32)
        
        # Estimate background using a large blur
        # Higher intensity = smaller blur kernel = more local adaptation
        kernel_size = int(50 * (1.5 - intensity))
        kernel_size = max(11, kernel_size) | 1  # Ensure odd number >= 11
        
        # Simple box blur approximation for background estimation
        from PIL import ImageFilter
        
        blurred_img = img.filter(ImageFilter.BoxBlur(kernel_size))
        background = np.array(blurred_img, dtype=np.float32)
        
        # Calculate where background is light gray (potential shading)
        # and the original pixel is close to background
        gray_shading_mask = (background > 160) & (background < 240)
        pixel_near_background = np.abs(img_array - background) < (30 + 20 * intensity)
        
        shading_pixels = gray_shading_mask & pixel_near_background
        
        # Count ink before
        ink_before = np.sum(img_array < 100) / img_array.size
        
        # Push shading pixels toward white
        result = img_array.copy()
        boost = 30 + 50 * intensity  # How much to brighten
        result[shading_pixels] = np.minimum(255, result[shading_pixels] + boost)
        
        # Safety check: if we lost too much ink, reduce effect
        if safety_check:
            ink_after = np.sum(result < 100) / result.size
            if ink_after < ink_before * 0.7:  # Lost more than 30% of ink
                logger.warning("Gray suppression would remove too much content, reducing intensity")
                # Re-do with lower boost
                boost = 15 + 25 * intensity
                result = img_array.copy()
                result[shading_pixels] = np.minimum(255, result[shading_pixels] + boost)
        
        return Image.fromarray(result.astype(np.uint8))
    
    @classmethod
    def process_page_image(
        cls,
        img: Image.Image,
        settings: Settings,
        redactions: Optional[PageRedactions] = None
    ) -> Image.Image:
        """
        Apply full processing pipeline to a page image.
        
        Pipeline order:
        1. Apply manual redactions (paint white)
        2. Remove colored highlights (HSV mask → white) with guardrails
        3. Convert to grayscale
        4. Suppress gray shading (if enabled)
        5. Apply auto-contrast (if enabled) and contrast factor
        6. Binarize (manual threshold OR Otsu)
        
        Args:
            img: RGB PIL Image
            settings: Processing settings
            redactions: Optional page redactions
        
        Returns:
            Processed image (grayscale or binary)
        """
        # 1. Apply manual redactions
        img = cls.apply_redactions(img, redactions)
        
        # 2. Remove colored highlights (returns tuple now)
        if settings.highlight_removal.enabled:
            img, removal_fraction = cls.remove_colored_highlights(
                img,
                saturation_threshold=settings.highlight_removal.saturation_threshold,
                value_threshold=settings.highlight_removal.value_threshold,
                safety_cap=True
            )
            logger.debug(f"Color removal affected {removal_fraction:.1%} of pixels")
        
        # 3. Convert to grayscale
        img = cls.convert_to_grayscale(img)
        
        # 4. Suppress gray shading (if enabled)
        if settings.gray_suppression.enabled and settings.gray_suppression.intensity > 0:
            img = cls.suppress_gray_shading(
                img,
                intensity=settings.gray_suppression.intensity,
                safety_check=True
            )
        
        # 5. Apply contrast adjustments
        if settings.auto_contrast:
            img = cls.apply_auto_contrast(img)
        
        if settings.contrast_factor != 1.0:
            img = cls.apply_contrast_factor(img, settings.contrast_factor)
        
        # 6. Binarize (if pure B/W mode)
        if settings.processing_mode == ProcessingMode.PURE_BW:
            if settings.auto_threshold_method == AutoThresholdMethod.OTSU:
                threshold = cls.calculate_otsu_threshold(img)
                logger.debug(f"Otsu threshold calculated: {threshold}")
            elif settings.auto_threshold_method == AutoThresholdMethod.ADAPTIVE:
                threshold = cls.calculate_otsu_threshold(img)
                threshold = min(255, threshold + 10)
                logger.debug(f"Adaptive threshold: {threshold}")
            else:
                threshold = settings.threshold
            
            img = cls.binarize(img, threshold)
        
        return img


class PDFExporter:
    """Handles PDF export operations with stream processing."""
    
    def __init__(
        self,
        input_doc: PDFDocument,
        settings: Settings,
        redactions: Optional[DocumentRedactions] = None,
        progress_callback: Optional[Callable[[ExportProgress], None]] = None
    ):
        self.input_doc = input_doc
        self.settings = settings
        self.redactions = redactions
        self.progress_callback = progress_callback
        self.progress = ExportProgress()
        self._cancelled = False
    
    def cancel(self) -> None:
        """Request cancellation of the export."""
        self._cancelled = True
        self.progress.cancelled = True
    
    def export(self, output_path: Path) -> bool:
        """
        Export the PDF with B/W conversion.
        
        Uses stream processing: processes one page at a time and immediately
        inserts into output PDF to avoid keeping all images in RAM.
        
        Writes to temporary file first and renames on success.
        
        Args:
            output_path: Final output file path
        
        Returns:
            True if export completed successfully, False otherwise
        """
        # Check if output is locked
        try:
            if output_path.exists():
                # Try to open for writing to check if locked
                with open(output_path, "a"):
                    pass
        except PermissionError:
            self.progress.error = f"Output file is locked: {output_path}"
            logger.error(self.progress.error)
            return False
        
        # Create temp file in same directory (for atomic rename)
        temp_path = output_path.with_suffix(".tmp.pdf")
        
        try:
            self.progress.total_pages = self.input_doc.page_count
            self._update_progress()
            
            # Create output PDF
            output_doc = fitz.open()
            
            for page_num in range(self.input_doc.page_count):
                if self._cancelled:
                    output_doc.close()
                    self._cleanup_temp(temp_path)
                    return False
                
                # Process single page
                try:
                    self._process_and_insert_page(output_doc, page_num)
                except Exception as e:
                    logger.exception(f"Error processing page {page_num + 1}")
                    self.progress.error = f"Error on page {page_num + 1}: {str(e)}"
                    output_doc.close()
                    self._cleanup_temp(temp_path)
                    return False
                
                self.progress.current_page = page_num + 1
                self._update_progress()
            
            # Save to temp file
            output_doc.save(str(temp_path))
            output_doc.close()
            
            # Rename temp to final
            if output_path.exists():
                output_path.unlink()
            shutil.move(str(temp_path), str(output_path))
            
            self.progress.completed = True
            self._update_progress()
            
            logger.info(f"Export completed: {output_path}")
            return True
            
        except Exception as e:
            logger.exception("Export failed")
            self.progress.error = str(e)
            self._cleanup_temp(temp_path)
            return False
    
    def _process_and_insert_page(self, output_doc: fitz.Document, page_num: int) -> None:
        """Process a single page and insert into output document."""
        # Get page geometry
        geometry = self.input_doc.get_page_geometry(page_num)
        if geometry is None:
            raise RuntimeError(f"No geometry for page {page_num}")
        
        # Render at export DPI
        img = self.input_doc.render_page(page_num, self.settings.export_dpi)
        
        # Get redactions for this page
        page_redactions = None
        if self.redactions:
            page_redactions = self.redactions.pages.get(page_num)
        
        # Process the image
        processed = ImageProcessor.process_page_image(img, self.settings, page_redactions)
        
        # Convert to bytes for embedding
        img_bytes = self._image_to_bytes(processed)
        
        # Create new page with original dimensions (in points)
        page_rect = fitz.Rect(0, 0, geometry.cropbox_width, geometry.cropbox_height)
        new_page = output_doc.new_page(width=page_rect.width, height=page_rect.height)
        
        # Insert image to fill the entire page
        new_page.insert_image(page_rect, stream=img_bytes)
        
        # Free memory
        del img
        del processed
        del img_bytes
    
    def _image_to_bytes(self, img: Image.Image) -> bytes:
        """Convert PIL Image to bytes for PDF embedding."""
        buffer = io.BytesIO()
        
        if self.settings.output_format == OutputFormat.JPEG:
            # Convert to RGB if grayscale for JPEG
            if img.mode == "L":
                img = img.convert("RGB")
            img.save(buffer, format="JPEG", quality=self.settings.jpeg_quality)
        else:
            # PNG (default, lossless)
            img.save(buffer, format="PNG")
        
        return buffer.getvalue()
    
    def _update_progress(self) -> None:
        """Call progress callback if set."""
        if self.progress_callback:
            self.progress_callback(self.progress)
    
    def _cleanup_temp(self, temp_path: Path) -> None:
        """Clean up temporary file."""
        try:
            if temp_path.exists():
                temp_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")


class PreviewCache:
    """
    Manages cached preview images for efficient preview rendering.
    
    Renders RGB preview images at preview_dpi and caches them.
    When settings change, reprocesses from the cached image without re-rendering PDF.
    """
    
    def __init__(self, document: PDFDocument, settings: Settings):
        self.document = document
        self.settings = settings
        self._cache: dict[int, Image.Image] = {}  # page_num -> RGB image at preview DPI
        self._preview_dpi = settings.get_preview_dpi()
    
    def clear(self) -> None:
        """Clear all cached images."""
        self._cache.clear()
    
    def clear_page(self, page_num: int) -> None:
        """Clear cache for a specific page."""
        if page_num in self._cache:
            del self._cache[page_num]
    
    def update_preview_dpi(self, settings: Settings) -> None:
        """Update preview DPI if changed, clearing cache if necessary."""
        new_dpi = settings.get_preview_dpi()
        if new_dpi != self._preview_dpi:
            self._preview_dpi = new_dpi
            self.clear()
        self.settings = settings
    
    def get_raw_preview(self, page_num: int) -> Image.Image:
        """
        Get the raw (unprocessed) RGB preview image for a page.
        Renders and caches if not already cached.
        """
        if page_num not in self._cache:
            if not self.document.is_open:
                raise RuntimeError("Document not open")
            
            img = self.document.render_page(page_num, self._preview_dpi)
            self._cache[page_num] = img
        
        return self._cache[page_num]
    
    def get_processed_preview(
        self,
        page_num: int,
        settings: Settings,
        redactions: Optional[PageRedactions] = None
    ) -> Image.Image:
        """
        Get a processed preview image for a page.
        Uses cached raw image and applies processing pipeline.
        """
        # Get raw image (from cache or render)
        raw_img = self.get_raw_preview(page_num)
        
        # Apply processing pipeline
        return ImageProcessor.process_page_image(raw_img.copy(), settings, redactions)
    
    @property
    def preview_dpi(self) -> float:
        """Get the current preview DPI."""
        return self._preview_dpi


def check_output_writable(path: Path) -> tuple[bool, str]:
    """
    Check if output path is writable.
    
    Returns:
        Tuple of (is_writable, error_message)
    """
    try:
        if path.exists():
            # Try to open for append to check lock
            with open(path, "a"):
                pass
        else:
            # Try to create the file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            path.unlink()
        return True, ""
    except PermissionError:
        return False, f"File is locked or permission denied: {path}"
    except Exception as e:
        return False, str(e)


def generate_default_output_path(input_path: Path) -> Path:
    """Generate default output path from input path."""
    stem = input_path.stem
    if stem.endswith(".bw"):
        return input_path  # Already has .bw suffix
    return input_path.with_stem(f"{stem}.bw")
