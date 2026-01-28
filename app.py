"""
Exam Cleaner - Desktop Application
Modern Tkinter GUI with ttkbootstrap for converting exam PDFs to study-safe B/W PDFs.

Workflow: Open PDF → Analyze → Review flagged pages → Export
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from PIL import Image, ImageTk, ImageDraw
import fitz  # PyMuPDF
from pathlib import Path
import threading
import json
import logging
from typing import Optional
import time
try:
    from ctypes import windll
except ImportError:
    windll = None

from models import (
    Settings, ProcessingMode, AutoThresholdMethod, OutputFormat,
    DocumentRedactions, PageRedactions, RedactionRect, ExportProgress,
    StrengthPreset, OutputPreset, AnalysisResult, AnalysisConfidence
)
from processor import (
    PDFDocument, ImageProcessor, PDFExporter, PreviewCache, PageAnalyzer,
    check_output_writable, generate_default_output_path
)

# Configure logging
logging.basicConfig(
    filename='app.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration file path
CONFIG_PATH = Path("config.json")

# Debounce delay in milliseconds
DEBOUNCE_DELAY_MS = 250


class DebounceTimer:
    """Helper class for debouncing slider/setting changes."""
    
    def __init__(self, root: tk.Tk, delay_ms: int, callback):
        self.root = root
        self.delay_ms = delay_ms
        self.callback = callback
        self._timer_id: Optional[str] = None
    
    def trigger(self) -> None:
        """Trigger the debounced callback."""
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
        self._timer_id = self.root.after(self.delay_ms, self._execute)
    
    def _execute(self) -> None:
        """Execute the callback."""
        self._timer_id = None
        self.callback()
    
    def cancel(self) -> None:
        """Cancel any pending callback."""
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None


class PreviewCanvas(tk.Canvas):
    """
    Canvas widget for displaying PDF preview with zoom, pan, and redaction support.
    Enhanced with selection, move, resize, and nudge controls.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Image state
        self._original_image: Optional[Image.Image] = None
        self._display_image: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None
        
        # Zoom and pan state
        self._zoom_level: float = 1.0
        self._pan_offset: tuple[float, float] = (0, 0)
        self._fit_zoom: float = 1.0
        
        # Drag state
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_mode: str = "pan"  # "pan", "redact", "move", "resize"
        
        # Redaction state
        self._redaction_mode: bool = False
        self._current_redaction_rect: Optional[int] = None
        self._redaction_start: Optional[tuple[int, int]] = None
        self._redaction_callback: Optional[callable] = None
        self._update_callback: Optional[callable] = None
        
        # Selection state
        self._selected_rect_id: Optional[str] = None
        self._resize_corner: Optional[str] = None
        
        # Existing redaction rectangles (canvas ids -> rect ids)
        self._redaction_overlays: dict[int, str] = {}
        # Reverse mapping: model rect id -> canvas id (for efficient updates)
        self._rect_id_to_canvas_id: dict[str, int] = {}
        self._handle_ids: list[int] = []
        
        # Current page redactions reference
        self._page_redactions: Optional[PageRedactions] = None
        
        # Bind events
        self.bind("<ButtonPress-1>", self._on_button_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", self._on_button_release)
        self.bind("<MouseWheel>", self._on_mouse_wheel)
        self.bind("<Configure>", self._on_resize)
        self.bind("<Delete>", self._on_delete_key)
        self.bind("<BackSpace>", self._on_delete_key)
        
        # Arrow key nudge
        self.bind("<Up>", lambda e: self._nudge_selected(0, -1, e.state & 0x1))
        self.bind("<Down>", lambda e: self._nudge_selected(0, 1, e.state & 0x1))
        self.bind("<Left>", lambda e: self._nudge_selected(-1, 0, e.state & 0x1))
        self.bind("<Right>", lambda e: self._nudge_selected(1, 0, e.state & 0x1))
        
        # Focus for keyboard events
        self.bind("<Button-1>", lambda e: self.focus_set(), add="+")
    
    def set_image(self, img: Image.Image) -> None:
        """Set the image to display."""
        self._original_image = img
        self._update_display()
    
    def clear_image(self) -> None:
        """Clear the displayed image."""
        self._original_image = None
        self._display_image = None
        if self._image_id:
            self.delete(self._image_id)
            self._image_id = None
        self._clear_redaction_overlays()
    
    def set_redaction_mode(self, enabled: bool, callback: callable = None, update_callback: callable = None) -> None:
        """Enable or disable redaction drawing mode."""
        self._redaction_mode = enabled
        self._redaction_callback = callback
        self._update_callback = update_callback
        self.config(cursor="crosshair" if enabled else "")
        if not enabled:
            self._selected_rect_id = None
            self._clear_handles()
    
    def set_page_redactions(self, redactions: Optional[PageRedactions]) -> None:
        """Set reference to current page redactions for manipulation."""
        self._page_redactions = redactions
    
    def select_rect(self, rect_id: str) -> None:
        """Select a redaction rectangle by ID."""
        self._selected_rect_id = rect_id
        self._draw_handles()
    
    def deselect(self) -> None:
        """Deselect current rectangle."""
        self._selected_rect_id = None
        self._clear_handles()
    
    def fit_to_window(self) -> None:
        """Fit the image to the window."""
        if self._original_image is None:
            return
        
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        img_width, img_height = self._original_image.size
        
        zoom_x = canvas_width / img_width if img_width > 0 else 1
        zoom_y = canvas_height / img_height if img_height > 0 else 1
        self._fit_zoom = min(zoom_x, zoom_y, 1.0)
        self._zoom_level = self._fit_zoom
        self._pan_offset = (0, 0)
        
        self._update_display()
    
    def zoom_in(self) -> None:
        """Zoom in by 25%."""
        self._zoom_level = min(self._zoom_level * 1.25, 5.0)
        self._update_display()
    
    def zoom_out(self) -> None:
        """Zoom out by 25%."""
        self._zoom_level = max(self._zoom_level / 1.25, 0.1)
        self._update_display()
    
    def get_zoom_level(self) -> float:
        """Get current zoom level."""
        return self._zoom_level
    
    def draw_redaction_overlays(self, redactions: Optional[PageRedactions]) -> None:
        """Draw redaction rectangle overlays."""
        self._clear_redaction_overlays()
        self._page_redactions = redactions
        
        if redactions is None or self._original_image is None:
            return
        
        for rect in redactions.rectangles:
            canvas_rect = self._normalized_to_canvas(rect)
            if canvas_rect:
                is_selected = (rect.id == self._selected_rect_id)
                # Use outline-only styling (no alpha colors - Tkinter doesn't support them)
                if is_selected:
                    outline_color = "#0d6efd"
                    width = 3
                    dash = (6, 3)
                else:
                    outline_color = "#dc3545"
                    width = 2
                    dash = ()
                
                canvas_id = self.create_rectangle(
                    *canvas_rect,
                    outline=outline_color,
                    fill="",  # No fill - outline only
                    width=width,
                    dash=dash
                )
                self._redaction_overlays[canvas_id] = rect.id
                self._rect_id_to_canvas_id[rect.id] = canvas_id
        
        # Draw handles for selected
        if self._selected_rect_id:
            self._draw_handles()
    
    def add_single_redaction(self, rect: RedactionRect) -> None:
        """Add a single redaction overlay without clearing existing ones."""
        if self._original_image is None:
            return
        canvas_rect = self._normalized_to_canvas(rect)
        if canvas_rect:
            canvas_id = self.create_rectangle(
                *canvas_rect,
                outline="#dc3545",
                fill="",
                width=2
            )
            self._redaction_overlays[canvas_id] = rect.id
            self._rect_id_to_canvas_id[rect.id] = canvas_id
    
    def load_page_redactions(self, redactions: Optional[PageRedactions]) -> None:
        """Clear all overlays and load redactions for a specific page.
        
        This is the proper method to call on page navigation to ensure
        no overlays from previous pages remain visible.
        """
        # Always clear existing overlays first
        self._clear_redaction_overlays()
        # Clear selection when changing pages
        self._selected_rect_id = None
        self._page_redactions = redactions
        
        if redactions is None or self._original_image is None:
            return
        
        # Draw overlays for this page
        for rect in redactions.rectangles:
            canvas_rect = self._normalized_to_canvas(rect)
            if canvas_rect:
                canvas_id = self.create_rectangle(
                    *canvas_rect,
                    outline="#dc3545",
                    fill="",
                    width=2
                )
                self._redaction_overlays[canvas_id] = rect.id
                self._rect_id_to_canvas_id[rect.id] = canvas_id
    
    def remove_redaction(self, rect_id: str) -> bool:
        """Remove a redaction overlay by model rect ID. Returns True if found."""
        canvas_id = self._rect_id_to_canvas_id.get(rect_id)
        if canvas_id is not None:
            self.delete(canvas_id)
            del self._redaction_overlays[canvas_id]
            del self._rect_id_to_canvas_id[rect_id]
            # Clear selection if this was selected
            if self._selected_rect_id == rect_id:
                self._selected_rect_id = None
                self._clear_handles()
            return True
        return False
    
    def _draw_handles(self) -> None:
        """Draw resize handles for selected rectangle."""
        self._clear_handles()
        
        if not self._selected_rect_id or not self._page_redactions:
            return
        
        rect = self._page_redactions.find_by_id(self._selected_rect_id)
        if not rect:
            return
        
        canvas_rect = self._normalized_to_canvas(rect)
        if not canvas_rect:
            return
        
        x0, y0, x1, y1 = canvas_rect
        handle_size = 8
        corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
        
        for cx, cy in corners:
            handle = self.create_rectangle(
                cx - handle_size//2, cy - handle_size//2,
                cx + handle_size//2, cy + handle_size//2,
                fill="#0d6efd",
                outline="white",
                width=1
            )
            self._handle_ids.append(handle)
    
    def _clear_handles(self) -> None:
        """Clear resize handles."""
        for handle_id in self._handle_ids:
            self.delete(handle_id)
        self._handle_ids.clear()
    
    def _clear_redaction_overlays(self) -> None:
        """Clear all redaction overlays."""
        for canvas_id in self._redaction_overlays:
            self.delete(canvas_id)
        self._redaction_overlays.clear()
        self._rect_id_to_canvas_id.clear()
        self._clear_handles()
    
    def _normalized_to_canvas(self, rect: RedactionRect) -> Optional[tuple[int, int, int, int]]:
        """Convert normalized coordinates to canvas coordinates."""
        if self._original_image is None:
            return None
        
        img_w, img_h = self._original_image.size
        
        x0 = rect.x0 * img_w * self._zoom_level
        y0 = rect.y0 * img_h * self._zoom_level
        x1 = rect.x1 * img_w * self._zoom_level
        y1 = rect.y1 * img_h * self._zoom_level
        
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        display_w = img_w * self._zoom_level
        display_h = img_h * self._zoom_level
        
        offset_x = (canvas_w - display_w) / 2 + self._pan_offset[0]
        offset_y = (canvas_h - display_h) / 2 + self._pan_offset[1]
        
        return (
            int(x0 + offset_x),
            int(y0 + offset_y),
            int(x1 + offset_x),
            int(y1 + offset_y)
        )
    
    def _canvas_to_normalized(self, x: int, y: int) -> tuple[float, float]:
        """Convert canvas coordinates to normalized image coordinates."""
        if self._original_image is None:
            return (0, 0)
        
        img_w, img_h = self._original_image.size
        canvas_w = self.winfo_width()
        canvas_h = self.winfo_height()
        display_w = img_w * self._zoom_level
        display_h = img_h * self._zoom_level
        
        offset_x = (canvas_w - display_w) / 2 + self._pan_offset[0]
        offset_y = (canvas_h - display_h) / 2 + self._pan_offset[1]
        
        img_x = (x - offset_x) / self._zoom_level
        img_y = (y - offset_y) / self._zoom_level
        
        norm_x = max(0, min(1, img_x / img_w))
        norm_y = max(0, min(1, img_y / img_h))
        
        return (norm_x, norm_y)
    
    def _update_display(self) -> None:
        """Update the displayed image based on current zoom and pan."""
        if self._original_image is None:
            return
        
        new_width = int(self._original_image.width * self._zoom_level)
        new_height = int(self._original_image.height * self._zoom_level)
        
        if new_width < 1 or new_height < 1:
            return
        
        resized = self._original_image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        self._display_image = ImageTk.PhotoImage(resized)
        
        canvas_width = self.winfo_width()
        canvas_height = self.winfo_height()
        
        x = canvas_width // 2 + self._pan_offset[0]
        y = canvas_height // 2 + self._pan_offset[1]
        
        if self._image_id:
            self.coords(self._image_id, x, y)
            self.itemconfig(self._image_id, image=self._display_image)
        else:
            self._image_id = self.create_image(x, y, image=self._display_image, anchor="center")
    
    def _on_button_press(self, event: tk.Event) -> None:
        """Handle mouse button press."""
        # Use canvasx/canvasy for proper scroll support
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        norm_x, norm_y = self._canvas_to_normalized(cx, cy)
        
        if self._redaction_mode:
            # Check if clicking on a handle first (for resize)
            if self._selected_rect_id and self._page_redactions:
                rect = self._page_redactions.find_by_id(self._selected_rect_id)
                if rect:
                    corner = rect.get_corner_at(norm_x, norm_y, tolerance=0.03)
                    if corner:
                        self._drag_mode = "resize"
                        self._resize_corner = corner
                        self._drag_start = (cx, cy)
                        return
            
            # Check if clicking on existing rect (for selection/move)
            if self._page_redactions:
                clicked_rect = self._page_redactions.find_at_point(norm_x, norm_y)
                if clicked_rect:
                    self._selected_rect_id = clicked_rect.id
                    self._drag_mode = "move"
                    self._drag_start = (cx, cy)
                    # Update styling for selection without full redraw
                    self._update_selection_styling()
                    return
                else:
                    self._selected_rect_id = None
                    self._clear_handles()
            
            # Start new redaction
            self._redaction_start = (cx, cy)
            self._current_redaction_rect = self.create_rectangle(
                cx, cy, cx, cy,
                outline="#0d6efd",
                width=2,
                dash=(4, 4)
            )
        else:
            self._drag_start = (cx, cy)
            self._drag_mode = "pan"
    
    def _update_selection_styling(self) -> None:
        """Update canvas item styling to reflect current selection."""
        for canvas_id, rect_id in self._redaction_overlays.items():
            if rect_id == self._selected_rect_id:
                self.itemconfig(canvas_id, outline="#0d6efd", width=3, dash=(6, 3))
            else:
                self.itemconfig(canvas_id, outline="#dc3545", width=2, dash=())
        if self._selected_rect_id:
            self._draw_handles()
    
    def _on_drag(self, event: tk.Event) -> None:
        """Handle mouse drag."""
        # Use canvasx/canvasy for proper scroll support
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        
        if self._drag_mode == "resize" and self._selected_rect_id and self._page_redactions:
            rect = self._page_redactions.find_by_id(self._selected_rect_id)
            if rect and self._resize_corner:
                norm_x, norm_y = self._canvas_to_normalized(cx, cy)
                rect.resize_corner(self._resize_corner, norm_x, norm_y)
                # Update canvas item directly (O(1) operation)
                canvas_id = self._rect_id_to_canvas_id.get(self._selected_rect_id)
                if canvas_id:
                    new_coords = self._normalized_to_canvas(rect)
                    if new_coords:
                        self.coords(canvas_id, *new_coords)
                        self._draw_handles()  # Update handle positions
                
        elif self._drag_mode == "move" and self._selected_rect_id and self._page_redactions and self._drag_start:
            rect = self._page_redactions.find_by_id(self._selected_rect_id)
            if rect:
                old_norm = self._canvas_to_normalized(*self._drag_start)
                new_norm = self._canvas_to_normalized(cx, cy)
                dx = new_norm[0] - old_norm[0]
                dy = new_norm[1] - old_norm[1]
                
                # Apply move to model
                w = rect.x1 - rect.x0
                h = rect.y1 - rect.y0
                rect.x0 = max(0, min(1 - w, rect.x0 + dx))
                rect.y0 = max(0, min(1 - h, rect.y0 + dy))
                rect.x1 = rect.x0 + w
                rect.y1 = rect.y0 + h
                
                self._drag_start = (cx, cy)
                
                # Update canvas item directly (O(1) operation)
                canvas_id = self._rect_id_to_canvas_id.get(self._selected_rect_id)
                if canvas_id:
                    new_coords = self._normalized_to_canvas(rect)
                    if new_coords:
                        self.coords(canvas_id, *new_coords)
                        self._draw_handles()  # Update handle positions
                
        elif self._redaction_mode and self._redaction_start:
            if self._current_redaction_rect:
                self.coords(
                    self._current_redaction_rect,
                    self._redaction_start[0], self._redaction_start[1],
                    cx, cy
                )
        elif self._drag_start and self._drag_mode == "pan":
            dx = cx - self._drag_start[0]
            dy = cy - self._drag_start[1]
            self._pan_offset = (
                self._pan_offset[0] + dx,
                self._pan_offset[1] + dy
            )
            self._drag_start = (cx, cy)
            self._update_display()
    
    def _on_button_release(self, event: tk.Event) -> None:
        """Handle mouse button release."""
        # Use canvasx/canvasy for proper scroll support
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        
        if self._drag_mode in ("move", "resize"):
            # Just reset state - canvas coords were already updated during drag
            # No need for full redraw
            self._drag_mode = "pan"
            self._resize_corner = None
            self._drag_start = None
            return
        
        if self._redaction_mode and self._redaction_start:
            if self._current_redaction_rect:
                self.delete(self._current_redaction_rect)
                self._current_redaction_rect = None
            
            start_norm = self._canvas_to_normalized(*self._redaction_start)
            end_norm = self._canvas_to_normalized(cx, cy)
            
            x0 = min(start_norm[0], end_norm[0])
            y0 = min(start_norm[1], end_norm[1])
            x1 = max(start_norm[0], end_norm[0])
            y1 = max(start_norm[1], end_norm[1])
            
            if abs(x1 - x0) > 0.01 and abs(y1 - y0) > 0.01:
                rect = RedactionRect(x0, y0, x1, y1)
                if self._redaction_callback:
                    self._redaction_callback(rect)
            
            self._redaction_start = None
        else:
            self._drag_start = None
    
    def _on_mouse_wheel(self, event: tk.Event) -> None:
        """Handle mouse wheel for zooming."""
        if event.delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()
        
        if self._page_redactions:
            self.draw_redaction_overlays(self._page_redactions)
    
    def _on_resize(self, event: tk.Event) -> None:
        """Handle canvas resize."""
        if self._original_image:
            self._update_display()
            if self._page_redactions:
                self.draw_redaction_overlays(self._page_redactions)
    
    def _on_delete_key(self, event: tk.Event) -> None:
        """Handle delete key for removing selected rectangle."""
        if self._selected_rect_id and self._page_redactions:
            self._page_redactions.remove_by_id(self._selected_rect_id)
            self._selected_rect_id = None
            self.draw_redaction_overlays(self._page_redactions)
            if self._update_callback:
                self._update_callback()
    
    def _nudge_selected(self, dx: int, dy: int, shift: bool) -> None:
        """Nudge selected rectangle by pixels."""
        if not self._selected_rect_id or not self._page_redactions or not self._original_image:
            return
        
        rect = self._page_redactions.find_by_id(self._selected_rect_id)
        if not rect:
            return
        
        # Calculate normalized nudge amount
        img_w, img_h = self._original_image.size
        pixels = 10 if shift else 1
        norm_dx = (dx * pixels) / (img_w * self._zoom_level)
        norm_dy = (dy * pixels) / (img_h * self._zoom_level)
        
        # Apply nudge
        w = rect.x1 - rect.x0
        h = rect.y1 - rect.y0
        rect.x0 = max(0, min(1 - w, rect.x0 + norm_dx))
        rect.y0 = max(0, min(1 - h, rect.y0 + norm_dy))
        rect.x1 = rect.x0 + w
        rect.y1 = rect.y0 + h
        
        self.draw_redaction_overlays(self._page_redactions)
        if self._update_callback:
            self._update_callback()


class Application(TkinterDnD.Tk):
    """Main application window with modern ttkbootstrap theme."""
    
    def __init__(self):
        super().__init__()
        self.style = tb.Style()
        self.style.theme_use("flatly")
        
        self.title("Exam Cleaner")
        # Increased default size and set minimum size
        self.geometry("1280x850")
        self.minsize(1024, 768)
        
        # Configure grid weights for the main window
        self.grid_rowconfigure(1, weight=1)  # content expands
        self.grid_columnconfigure(0, weight=1)  # full width
        
        # Enable Drag and Drop
        self.drop_target_register(DND_FILES)
        self.dnd_bind('<<Drop>>', self._on_drop)
        
        # State
        self._document: Optional[PDFDocument] = None
        self._preview_cache: Optional[PreviewCache] = None
        self._current_page: int = 0
        self._settings = Settings.load_from_file(CONFIG_PATH)
        self._redactions: Optional[DocumentRedactions] = None
        self._exporter: Optional[PDFExporter] = None
        self._export_thread: Optional[threading.Thread] = None
        self._analyzer: Optional[PageAnalyzer] = None
        self._analysis_thread: Optional[threading.Thread] = None
        self._analysis_results: list[AnalysisResult] = []
        self._show_all_pages: bool = False
        
        # Debounce timer for preview updates
        self._preview_debounce = DebounceTimer(self, DEBOUNCE_DELAY_MS, self._update_preview)
        
        # Raw preview cache per page
        self._raw_preview_cache: dict[int, Image.Image] = {}
        self._cache_dpi: float = 0
        
        # Build UI
        self._build_ui()
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        
        # Bind keyboard shortcuts
        self.bind("<Control-z>", lambda e: self._undo_last_redaction())
        
        logger.info("Application started")
    
    def _build_ui(self) -> None:
        """Build the user interface."""
        # Top toolbar (row 0)
        self._build_toolbar()
        
        # Main content area (row 1)
        main_frame = ttk.Frame(self)
        main_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        
        # Configure grid for main_frame
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)  # Preview panel expands
        
        # Left panel: page list and settings
        self._build_left_panel(main_frame)
        
        # Right panel: preview
        self._build_preview_panel(main_frame)
        
        # Status bar
        self._build_status_bar()
    
    def _build_toolbar(self) -> None:
        """Build the top toolbar with primary actions."""
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        
        # Open button
        self._open_btn = ttk.Button(
            toolbar, text="📂 Open PDF", 
            bootstyle="primary",
            command=self._open_file
        )
        self._open_btn.pack(side=LEFT, padx=(0, 5))
        
        # Analyze button
        self._analyze_btn = ttk.Button(
            toolbar, text="🔍 Analyze",
            bootstyle="info",
            command=self._start_analysis,
            state=DISABLED
        )
        self._analyze_btn.pack(side=LEFT, padx=5)
        
        # Export button
        self._export_btn = ttk.Button(
            toolbar, text="💾 Export",
            bootstyle="success",
            command=self._start_export,
            state=DISABLED
        )
        self._export_btn.pack(side=LEFT, padx=5)
        
        # Cancel button (hidden initially)
        self._cancel_btn = ttk.Button(
            toolbar, text="⏹ Cancel",
            bootstyle="danger",
            command=self._cancel_operation
        )
        
        # Separator
        ttk.Separator(toolbar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=15)
        
        # Redaction mode toggle
        self._redaction_var = tk.BooleanVar(value=False)
        self._redaction_check = ttk.Checkbutton(
            toolbar, text="✏️ Redaction Mode",
            variable=self._redaction_var,
            bootstyle="warning-round-toggle",
            command=self._toggle_redaction_mode
        )
        self._redaction_check.pack(side=LEFT, padx=5)
        
        # File name label
        self._file_label = ttk.Label(toolbar, text="", font=("", 10, "italic"))
        self._file_label.pack(side=RIGHT, padx=10)
    
    def _build_left_panel(self, parent) -> None:
        """Build the left panel with page list and settings."""
        left_frame = ttk.Frame(parent, width=320)
        left_frame.grid(row=0, column=0, sticky="ns", padx=(0, 10))
        left_frame.grid_propagate(False)  # Enforce fixed width
        
        # Page list section
        self._build_page_list(left_frame)
        
        # Settings section
        self._build_settings_panel(left_frame)
    
    def _build_page_list(self, parent) -> None:
        """Build the page review list."""
        list_frame = ttk.LabelFrame(parent, text="Pages to Review")
        list_frame.pack(fill=BOTH, expand=True, pady=(0, 10))
        
        # Empty state label
        self._empty_label = ttk.Label(
            list_frame,
            text="📄 Open a PDF to begin",
            font=("", 11),
            foreground="gray"
        )
        self._empty_label.pack(expand=True)
        
        # Show all pages toggle
        toggle_frame = ttk.Frame(list_frame)
        self._show_all_var = tk.BooleanVar(value=False)
        self._show_all_check = ttk.Checkbutton(
            toggle_frame,
            text="Show all pages",
            variable=self._show_all_var,
            command=self._refresh_page_list,
            bootstyle="secondary-round-toggle"
        )
        
        # Page listbox with scrollbar
        self._page_list_frame = ttk.Frame(list_frame)
        
        self._page_listbox = tk.Listbox(
            self._page_list_frame,
            font=("", 10),
            selectmode=SINGLE,
            activestyle="none",
            highlightthickness=0,
            bd=0
        )
        self._page_listbox.bind("<<ListboxSelect>>", self._on_page_select)
        
        scrollbar = ttk.Scrollbar(self._page_list_frame, orient=VERTICAL, command=self._page_listbox.yview)
        self._page_listbox.config(yscrollcommand=scrollbar.set)
        
        self._page_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
        
        # Analysis progress (hidden initially)
        self._analysis_progress_frame = ttk.Frame(list_frame)
        self._analysis_progress = ttk.Progressbar(
            self._analysis_progress_frame,
            mode="determinate",
            bootstyle="info-striped"
        )
        self._analysis_label = ttk.Label(self._analysis_progress_frame, text="Analyzing...")
    
    def _build_settings_panel(self, parent) -> None:
        """Build the settings panel with presets and advanced options."""
        settings_frame = ttk.LabelFrame(parent, text="Settings")
        settings_frame.pack(fill=X)
        
        # Hide Answers Strength
        ttk.Label(settings_frame, text="Hide Answers Strength:", font=("", 10, "bold")).pack(anchor=W)
        
        strength_frame = ttk.Frame(settings_frame)
        strength_frame.pack(fill=X, pady=(5, 10))
        
        self._strength_var = tk.StringVar(value=self._settings.strength_preset.value)
        for preset in StrengthPreset:
            label = preset.value.capitalize()
            ttk.Radiobutton(
                strength_frame,
                text=label,
                value=preset.value,
                variable=self._strength_var,
                command=self._on_strength_change,
                bootstyle="info-toolbutton"
            ).pack(side=LEFT, padx=2, expand=True, fill=X)
        
        # Output Quality
        ttk.Label(settings_frame, text="Output Quality:", font=("", 10, "bold")).pack(anchor=W, pady=(5, 0))
        
        output_frame = ttk.Frame(settings_frame)
        output_frame.pack(fill=X, pady=(5, 10))
        
        self._output_var = tk.StringVar(value=self._settings.output_preset.value)
        ttk.Radiobutton(
            output_frame,
            text="High Quality",
            value=OutputPreset.HIGH_QUALITY.value,
            variable=self._output_var,
            command=self._on_output_change,
            bootstyle="success-toolbutton"
        ).pack(side=LEFT, padx=2, expand=True, fill=X)
        ttk.Radiobutton(
            output_frame,
            text="Balanced",
            value=OutputPreset.BALANCED.value,
            variable=self._output_var,
            command=self._on_output_change,
            bootstyle="success-toolbutton"
        ).pack(side=LEFT, padx=2, expand=True, fill=X)
        
        # Advanced expander
        self._advanced_var = tk.BooleanVar(value=False)
        advanced_toggle = ttk.Checkbutton(
            settings_frame,
            text="▶ Advanced Options",
            variable=self._advanced_var,
            command=self._toggle_advanced,
            bootstyle="secondary-outline-toolbutton"
        )
        advanced_toggle.pack(fill=X, pady=(5, 0))
        
        self._advanced_frame = ttk.Frame(settings_frame)
        self._build_advanced_options()
    
    def _build_advanced_options(self) -> None:
        """Build advanced options (hidden by default)."""
        frame = self._advanced_frame
        
        # Manual threshold
        ttk.Label(frame, text="Manual Threshold:").pack(anchor=W, pady=(5, 0))
        thresh_frame = ttk.Frame(frame)
        thresh_frame.pack(fill=X)
        self._threshold_var = tk.IntVar(value=self._settings.threshold)
        self._threshold_scale = ttk.Scale(
            thresh_frame, from_=0, to=255, variable=self._threshold_var,
            command=lambda v: self._on_advanced_change()
        )
        self._threshold_scale.pack(side=LEFT, fill=X, expand=True)
        self._threshold_label = ttk.Label(thresh_frame, text=str(self._settings.threshold), width=4)
        self._threshold_label.pack(side=RIGHT)
        self._threshold_var.trace_add("write", lambda *a: self._threshold_label.config(text=str(self._threshold_var.get())))
        
        # Contrast factor
        ttk.Label(frame, text="Contrast Factor:").pack(anchor=W, pady=(5, 0))
        contrast_frame = ttk.Frame(frame)
        contrast_frame.pack(fill=X)
        self._contrast_var = tk.DoubleVar(value=self._settings.contrast_factor)
        self._contrast_scale = ttk.Scale(
            contrast_frame, from_=1.0, to=2.0, variable=self._contrast_var,
            command=lambda v: self._on_advanced_change()
        )
        self._contrast_scale.pack(side=LEFT, fill=X, expand=True)
        self._contrast_label = ttk.Label(contrast_frame, text=f"{self._settings.contrast_factor:.1f}", width=4)
        self._contrast_label.pack(side=RIGHT)
        self._contrast_var.trace_add("write", lambda *a: self._contrast_label.config(text=f"{self._contrast_var.get():.1f}"))
        
        # Gray suppression toggle
        self._gray_suppress_var = tk.BooleanVar(value=self._settings.gray_suppression.enabled)
        ttk.Checkbutton(
            frame, text="Suppress Gray Shading",
            variable=self._gray_suppress_var,
            command=self._on_advanced_change
        ).pack(anchor=W, pady=(5, 0))
        
        # DPI
        ttk.Label(frame, text="Export DPI:").pack(anchor=W, pady=(5, 0))
        dpi_frame = ttk.Frame(frame)
        dpi_frame.pack(fill=X)
        self._dpi_var = tk.IntVar(value=self._settings.export_dpi)
        self._dpi_scale = ttk.Scale(
            dpi_frame, from_=72, to=400, variable=self._dpi_var,
            command=lambda v: self._on_advanced_change()
        )
        self._dpi_scale.pack(side=LEFT, fill=X, expand=True)
        self._dpi_label = ttk.Label(dpi_frame, text=str(self._settings.export_dpi), width=4)
        self._dpi_label.pack(side=RIGHT)
        self._dpi_var.trace_add("write", lambda *a: self._dpi_label.config(text=str(self._dpi_var.get())))
    
    def _toggle_advanced(self) -> None:
        """Toggle advanced options visibility."""
        if self._advanced_var.get():
            self._advanced_frame.pack(fill=X, pady=(5, 0))
        else:
            self._advanced_frame.pack_forget()
    
    def _build_preview_panel(self, parent) -> None:
        """Build the right preview panel."""
        preview_frame = ttk.Frame(parent)
        preview_frame.grid(row=0, column=1, sticky="nsew")
        
        # Navigation bar
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=X, pady=(0, 5))
        
        ttk.Button(nav_frame, text="◀", width=3, command=self._prev_page, bootstyle="secondary").pack(side=LEFT)
        ttk.Button(nav_frame, text="▶", width=3, command=self._next_page, bootstyle="secondary").pack(side=LEFT, padx=2)
        
        self._page_label = ttk.Label(nav_frame, text="Page 0 / 0", font=("", 10))
        self._page_label.pack(side=LEFT, padx=10)
        
        # Zoom controls
        ttk.Separator(nav_frame, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        ttk.Button(nav_frame, text="−", width=3, command=self._zoom_out, bootstyle="secondary").pack(side=LEFT)
        self._zoom_label = ttk.Label(nav_frame, text="100%", width=6)
        self._zoom_label.pack(side=LEFT)
        ttk.Button(nav_frame, text="+", width=3, command=self._zoom_in, bootstyle="secondary").pack(side=LEFT)
        ttk.Button(nav_frame, text="Fit", command=self._fit_to_window, bootstyle="secondary-outline").pack(side=LEFT, padx=5)
        
        # Redaction controls (right side)
        self._delete_rect_btn = ttk.Button(
            nav_frame, text="🗑 Delete Selected",
            command=self._delete_selected_rect,
            bootstyle="danger-outline",
            state=DISABLED
        )
        self._delete_rect_btn.pack(side=RIGHT, padx=5)
        
        # Undo button for redaction mode
        self._undo_btn = ttk.Button(
            nav_frame, text="↩ Undo",
            command=self._undo_last_redaction,
            bootstyle="secondary-outline",
            state=DISABLED
        )
        self._undo_btn.pack(side=RIGHT, padx=5)
        
        # Preview canvas
        canvas_frame = ttk.Frame(preview_frame, bootstyle="dark")
        canvas_frame.pack(fill=BOTH, expand=True)
        
        self._preview_canvas = PreviewCanvas(
            canvas_frame,
            bg="#f8f9fa",
            highlightthickness=0
        )
        self._preview_canvas.pack(fill=BOTH, expand=True, padx=2, pady=2)
    
    def _build_status_bar(self) -> None:
        """Build the status bar."""
        self._status_frame = ttk.Frame(self)
        self._status_frame.grid(row=2, column=0, sticky="ew")
        
        self._status_label = ttk.Label(self._status_frame, text="Ready", font=("", 9))
        self._status_label.pack(side=LEFT)
        
        # Progress bar (hidden initially)
        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            self._status_frame, 
            variable=self._progress_var, 
            maximum=100,
            bootstyle="success-striped",
            length=200
        )
    
    def _set_status(self, text: str) -> None:
        """Update status bar text."""
        self._status_label.config(text=text)
    
    def _open_file(self) -> None:
        """Open a PDF file."""
        filepath = filedialog.askopenfilename(
            title="Open PDF",
            filetypes=[("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if not filepath:
            return
        
        self._load_document(filepath)

    def _on_drop(self, event) -> None:
        """Handle file drop event."""
        filepath = event.data
        # Handle curly braces for paths with spaces (TkinterDnD quirk)
        if filepath.startswith('{') and filepath.endswith('}'):
            filepath = filepath[1:-1]
        
        self._load_document(filepath)

    def _load_document(self, filepath: str) -> None:
        """Load a document from the given path."""
        path = Path(filepath)
        
        try:
            if self._document:
                self._document.close()
            
            self._document = PDFDocument(path)
            self._document.open()
            
            self._preview_cache = PreviewCache(self._document, self._settings)
            self._redactions = DocumentRedactions(file_path=str(path))
            self._analyzer = PageAnalyzer(self._document)
            self._analysis_results = []
            self._raw_preview_cache.clear()
            
            self._current_page = 0
            
            # Update UI
            self._file_label.config(text=path.name)
            self._analyze_btn.config(state=NORMAL)
            self._export_btn.config(state=NORMAL)
            self._update_page_label()
            self._refresh_page_list()
            self._update_preview()
            
            self._set_status(f"Opened: {path.name} ({self._document.page_count} pages)")
            logger.info(f"Opened file: {path}")
            
        except Exception as e:
            logger.exception(f"Failed to open file: {path}")
            messagebox.showerror("Error", f"Failed to open file:\n{str(e)}")
    
    def _start_analysis(self) -> None:
        """Start page analysis in background thread."""
        if not self._document or not self._analyzer:
            return
        
        # Show progress UI
        self._analyze_btn.config(state=DISABLED)
        self._cancel_btn.pack(side=LEFT, padx=5)
        
        self._empty_label.pack_forget()
        self._page_list_frame.pack_forget()
        self._analysis_progress_frame.pack(fill=X, pady=10)
        self._analysis_progress.pack(fill=X)
        self._analysis_label.pack(pady=5)
        
        self._set_status("Analyzing pages...")
        
        # Clear previous results
        self._analyzer.clear_results()
        self._analysis_results = []
        
        # Start analysis thread
        self._analysis_thread = threading.Thread(target=self._run_analysis, daemon=True)
        self._analysis_thread.start()
    
    def _run_analysis(self) -> None:
        """Run analysis in background thread."""
        def progress_callback(current: int, total: int):
            self.after(0, lambda: self._update_analysis_progress(current, total))
        
        try:
            results = self._analyzer.analyze_all(progress_callback)
            self.after(0, lambda: self._on_analysis_complete(results))
        except Exception as e:
            logger.exception("Analysis failed")
            self.after(0, lambda: self._on_analysis_complete([], str(e)))
    
    def _update_analysis_progress(self, current: int, total: int) -> None:
        """Update analysis progress UI."""
        progress = (current / total) * 100 if total > 0 else 0
        self._analysis_progress.config(value=progress)
        self._analysis_label.config(text=f"Analyzing page {current} / {total}...")
    
    def _on_analysis_complete(self, results: list[AnalysisResult], error: str = None) -> None:
        """Handle analysis completion."""
        # Hide progress UI
        self._analysis_progress_frame.pack_forget()
        self._cancel_btn.pack_forget()
        self._analyze_btn.config(state=NORMAL)
        
        if error:
            messagebox.showerror("Analysis Error", f"Analysis failed:\n{error}")
            self._set_status("Analysis failed")
            return
        
        self._analysis_results = results
        flagged = [r for r in results if r.is_flagged]
        
        self._set_status(f"Analysis complete: {len(flagged)} pages flagged for review")
        self._refresh_page_list()
    
    def _refresh_page_list(self) -> None:
        """Refresh the page list based on analysis results."""
        if not self._document:
            self._empty_label.config(text="📄 Open a PDF to begin")
            self._empty_label.pack(expand=True)
            self._page_list_frame.pack_forget()
            return
        
        # Check if we have analysis results
        if not self._analysis_results:
            self._empty_label.config(text="🔍 Click Analyze to find marked pages")
            self._empty_label.pack(expand=True)
            self._page_list_frame.pack_forget()
            self._show_all_check.pack_forget()
            return
        
        self._empty_label.pack_forget()
        self._show_all_check.pack(anchor=W, pady=(0, 5))
        self._show_all_check.master.pack(fill=X, pady=(0, 5))
        self._page_list_frame.pack(fill=BOTH, expand=True)
        
        # Clear listbox
        self._page_listbox.delete(0, tk.END)
        
        show_all = self._show_all_var.get()
        
        for result in self._analysis_results:
            if show_all or result.is_flagged:
                # Format entry
                page_num = result.page_number + 1
                if result.confidence == AnalysisConfidence.LIKELY_MARKED:
                    prefix = "🔴"
                    suffix = " (Likely marked)"
                elif result.confidence == AnalysisConfidence.MAYBE_MARKED:
                    prefix = "🟡"
                    suffix = " (Maybe marked)"
                else:
                    prefix = "⚪"
                    suffix = ""
                
                entry = f"{prefix} Page {page_num}{suffix}"
                self._page_listbox.insert(tk.END, entry)
    
    def _on_page_select(self, event) -> None:
        """Handle page selection from list."""
        selection = self._page_listbox.curselection()
        if not selection:
            return
        
        # Map listbox index to actual page
        idx = selection[0]
        show_all = self._show_all_var.get()
        
        if show_all:
            self._current_page = idx
        else:
            flagged = [r for r in self._analysis_results if r.is_flagged]
            if idx < len(flagged):
                self._current_page = flagged[idx].page_number
        
        self._update_page_label()
        self._update_preview()
    
    def _collect_settings(self) -> Settings:
        """Collect current settings from UI controls."""
        from models import HighlightRemovalSettings, GraySuppressSettings
        
        settings = Settings(
            strength_preset=StrengthPreset(self._strength_var.get()),
            output_preset=OutputPreset(self._output_var.get()),
            export_dpi=self._dpi_var.get(),
            processing_mode=ProcessingMode.PURE_BW,
            threshold=self._threshold_var.get(),
            auto_threshold_method=AutoThresholdMethod.NONE,
            auto_contrast=False,
            contrast_factor=self._contrast_var.get(),
            highlight_removal=HighlightRemovalSettings(enabled=True),
            gray_suppression=GraySuppressSettings(
                enabled=self._gray_suppress_var.get(),
                intensity=0.5
            ),
            output_format=OutputFormat.PNG if self._output_var.get() == OutputPreset.HIGH_QUALITY.value else OutputFormat.JPEG,
            jpeg_quality=90
        )
        
        # Apply preset mappings
        settings.apply_preset(settings.strength_preset)
        settings.apply_output_preset(settings.output_preset)
        
        # Override with advanced settings if changed
        if self._advanced_var.get():
            settings.threshold = self._threshold_var.get()
            settings.contrast_factor = self._contrast_var.get()
            settings.gray_suppression.enabled = self._gray_suppress_var.get()
            settings.export_dpi = self._dpi_var.get()
        
        return settings
    
    def _on_strength_change(self) -> None:
        """Handle strength preset change."""
        self._settings.apply_preset(StrengthPreset(self._strength_var.get()))
        # Update advanced sliders to match
        self._threshold_var.set(self._settings.threshold)
        self._gray_suppress_var.set(self._settings.gray_suppression.enabled)
        self._preview_debounce.trigger()
    
    def _on_output_change(self) -> None:
        """Handle output preset change."""
        self._settings.apply_output_preset(OutputPreset(self._output_var.get()))
    
    def _on_advanced_change(self) -> None:
        """Called when any advanced setting changes."""
        self._preview_debounce.trigger()
    
    def _update_preview(self, preserve_view: bool = False) -> None:
        """Update the preview canvas with current settings.
        
        Args:
            preserve_view: If True, don't reset zoom/pan after updating.
        """
        if self._document is None:
            return
        
        try:
            self._settings = self._collect_settings()
            
            # Check if we need to re-render the raw preview
            preview_dpi = self._settings.get_preview_dpi()
            if self._cache_dpi != preview_dpi:
                self._raw_preview_cache.clear()
                self._cache_dpi = preview_dpi
            
            # Get raw preview (from cache or render)
            if self._current_page not in self._raw_preview_cache:
                raw_img = self._document.render_page(self._current_page, preview_dpi)
                self._raw_preview_cache[self._current_page] = raw_img
            
            raw_img = self._raw_preview_cache[self._current_page]
            
            # Get redactions for this page
            page_redactions = None
            if self._redactions:
                page_redactions = self._redactions.pages.get(self._current_page)
            
            # Process the image
            preview_img = ImageProcessor.process_page_image(
                raw_img.copy(),
                self._settings,
                page_redactions
            )
            
            # Convert for display
            if preview_img.mode == "L":
                preview_img = preview_img.convert("RGB")
            
            self._preview_canvas.set_image(preview_img)
            
            # Only fit to window if not preserving view
            if not preserve_view:
                self._preview_canvas.fit_to_window()
            
            # Always clear and reload overlays for current page (ensures no leaks)
            self._preview_canvas.load_page_redactions(page_redactions)
            
            self._update_zoom_label()
            self._sync_redaction_ui_state()
            
        except Exception as e:
            logger.exception("Failed to update preview")
            self._set_status(f"Preview error: {str(e)}")
    
    def _update_page_label(self) -> None:
        """Update the page navigation label."""
        if self._document:
            total = self._document.page_count
            self._page_label.config(text=f"Page {self._current_page + 1} / {total}")
        else:
            self._page_label.config(text="Page 0 / 0")
    
    def _update_zoom_label(self) -> None:
        """Update the zoom level label."""
        zoom = self._preview_canvas.get_zoom_level()
        self._zoom_label.config(text=f"{int(zoom * 100)}%")
    
    def _prev_page(self) -> None:
        """Go to previous page."""
        if self._document and self._current_page > 0:
            self._current_page -= 1
            self._update_page_label()
            self._update_preview()
    
    def _next_page(self) -> None:
        """Go to next page."""
        if self._document and self._current_page < self._document.page_count - 1:
            self._current_page += 1
            self._update_page_label()
            self._update_preview()
    
    def _zoom_in(self) -> None:
        """Zoom in on preview."""
        self._preview_canvas.zoom_in()
        self._update_zoom_label()
        if self._redactions:
            page_redactions = self._redactions.pages.get(self._current_page)
            self._preview_canvas.draw_redaction_overlays(page_redactions)
    
    def _zoom_out(self) -> None:
        """Zoom out on preview."""
        self._preview_canvas.zoom_out()
        self._update_zoom_label()
        if self._redactions:
            page_redactions = self._redactions.pages.get(self._current_page)
            self._preview_canvas.draw_redaction_overlays(page_redactions)
    
    def _fit_to_window(self) -> None:
        """Fit preview to window."""
        self._preview_canvas.fit_to_window()
        self._update_zoom_label()
        if self._redactions:
            page_redactions = self._redactions.pages.get(self._current_page)
            self._preview_canvas.draw_redaction_overlays(page_redactions)
    
    def _toggle_redaction_mode(self) -> None:
        """Toggle redaction drawing mode."""
        enabled = self._redaction_var.get()
        self._preview_canvas.set_redaction_mode(
            enabled, 
            self._on_redaction_drawn,
            None  # No update callback needed - we handle updates directly
        )
        self._sync_redaction_ui_state()
    
    def _on_redaction_drawn(self, rect: RedactionRect) -> None:
        """Called when a redaction rectangle is drawn."""
        if self._redactions is None:
            return
        
        page_red = self._redactions.get_page(self._current_page)
        page_red.add_rect(rect)
        # Update canvas reference to page redactions (may be new)
        self._preview_canvas.set_page_redactions(page_red)
        # Add overlay directly without full re-render (preserves zoom)
        self._preview_canvas.add_single_redaction(rect)
        self._preview_canvas.select_rect(rect.id)
        self._sync_redaction_ui_state()
    
    def _delete_selected_rect(self) -> None:
        """Delete the currently selected redaction rectangle."""
        if self._preview_canvas._selected_rect_id and self._redactions:
            page_red = self._redactions.pages.get(self._current_page)
            if page_red:
                rect_id = self._preview_canvas._selected_rect_id
                page_red.remove_by_id(rect_id)
                self._preview_canvas.remove_redaction(rect_id)
                self._sync_redaction_ui_state()
    
    def _undo_last_redaction(self) -> None:
        """Undo the last redaction rectangle on the current page."""
        if not self._redactions or not self._redaction_var.get():
            return
        page_red = self._redactions.pages.get(self._current_page)
        if page_red and page_red.rectangles:
            page_red.remove_last()
            # Force full resync of canvas with model to ensure visual matches
            self._preview_canvas.load_page_redactions(page_red)
        self._sync_redaction_ui_state()
    
    def _sync_redaction_ui_state(self) -> None:
        """Update Undo/Delete button states based on current context."""
        in_mode = self._redaction_var.get()
        page_red = self._redactions.pages.get(self._current_page) if self._redactions else None
        has_rects = bool(page_red and page_red.rectangles)
        has_selection = bool(self._preview_canvas._selected_rect_id)
        
        self._undo_btn.config(state=NORMAL if (in_mode and has_rects) else DISABLED)
        self._delete_rect_btn.config(state=NORMAL if (in_mode and has_selection) else DISABLED)
    
    def _cancel_operation(self) -> None:
        """Cancel current operation (analysis or export)."""
        if self._analyzer:
            self._analyzer.cancel()
        if self._exporter:
            self._exporter.cancel()
    
    def _start_export(self) -> None:
        """Start the export process."""
        if self._document is None:
            return
        
        input_path = self._document.path
        default_output = generate_default_output_path(input_path)
        
        output_path = filedialog.asksaveasfilename(
            title="Save As",
            defaultextension=".pdf",
            initialfile=default_output.name,
            initialdir=str(default_output.parent),
            filetypes=[("PDF files", "*.pdf")]
        )
        
        if not output_path:
            return
        
        output_path = Path(output_path)
        
        writable, error = check_output_writable(output_path)
        if not writable:
            messagebox.showerror("Error", error)
            return
        
        self._settings = self._collect_settings()
        
        # Show progress UI
        self._export_btn.config(state=DISABLED)
        self._cancel_btn.pack(side=LEFT, padx=5)
        self._progress_bar.pack(side=RIGHT, padx=10)
        self._progress_var.set(0)
        
        self._exporter = PDFExporter(
            self._document,
            self._settings,
            self._redactions,
            self._on_export_progress
        )
        
        self._export_thread = threading.Thread(
            target=self._run_export,
            args=(output_path,),
            daemon=True
        )
        self._export_thread.start()
    
    def _run_export(self, output_path: Path) -> None:
        """Run export in background thread."""
        try:
            success = self._exporter.export(output_path)
            self.after(0, lambda: self._on_export_complete(success, output_path))
        except Exception as e:
            logger.exception("Export failed")
            self.after(0, lambda: self._on_export_complete(False, output_path, str(e)))
    
    def _on_export_progress(self, progress: ExportProgress) -> None:
        """Called from export thread with progress updates."""
        self.after(0, lambda: self._update_export_progress(progress))
    
    def _update_export_progress(self, progress: ExportProgress) -> None:
        """Update UI with export progress (main thread)."""
        self._progress_var.set(progress.progress_fraction * 100)
        self._set_status(f"Exporting page {progress.current_page} / {progress.total_pages}...")
    
    def _on_export_complete(self, success: bool, output_path: Path, error: str = None) -> None:
        """Called when export completes."""
        self._cancel_btn.pack_forget()
        self._progress_bar.pack_forget()
        self._export_btn.config(state=NORMAL)
        
        if success:
            self._set_status(f"Export complete: {output_path.name}")
            messagebox.showinfo("Export Complete", f"PDF exported successfully:\n{output_path}")
        elif self._exporter and self._exporter.progress.cancelled:
            self._set_status("Export cancelled")
            messagebox.showinfo("Export Cancelled", "Export was cancelled.")
        else:
            err_msg = error or (self._exporter.progress.error if self._exporter else "Unknown error")
            self._set_status("Export failed")
            messagebox.showerror("Export Failed", f"Export failed:\n{err_msg}\n\nSee app.log for details.")
        
        self._exporter = None
        self._export_thread = None
    
    def _on_close(self) -> None:
        """Handle application close."""
        self._preview_debounce.cancel()
        
        if self._analyzer:
            self._analyzer.cancel()
        if self._exporter:
            self._exporter.cancel()
        
        try:
            self._settings = self._collect_settings()
            self._settings.save_to_file(CONFIG_PATH)
            logger.info("Settings saved")
        except Exception as e:
            logger.warning(f"Failed to save settings: {e}")
        
        if self._document:
            self._document.close()
        
        logger.info("Application closed")
        self.destroy()


def main():
    """Main entry point."""
    # Enable High DPI awareness on Windows
    if windll:
        try:
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass
            
    app = Application()
    app.mainloop()


if __name__ == "__main__":
    main()
