"""
Exam Cleaner - Desktop Application
Modern Tkinter GUI with ttkbootstrap for converting exam PDFs to study-safe B/W PDFs.

Workflow: Open PDF → Auto Clean → (Optional) Redaction → Export
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
    StrengthPreset, OutputPreset
)
from processor import (
    PDFDocument, ImageProcessor, PDFExporter, PreviewCache,
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
    Refactored to use native scan_mark/scan_dragto for panning.
    """
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Image state
        self._original_image: Optional[Image.Image] = None
        self._display_image: Optional[ImageTk.PhotoImage] = None
        self._image_id: Optional[int] = None
        
        # Zoom state
        self._zoom_level: float = 1.0
        self._fit_zoom: float = 1.0
        
        # Drag state
        self._drag_start: Optional[tuple[int, int]] = None
        self._drag_mode: str = "pan"  # "pan", "redact", "move", "resize"
        
        # Redaction state
        self._redaction_mode: bool = False
        self._current_redaction_rect: Optional[int] = None
        self._redaction_start_pos: Optional[tuple[int, int]] = None  # Original click pos
        self._redaction_active: bool = False  # True only after threshold passed
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
        
        # Right-click pan bindings
        self.bind("<ButtonPress-3>", self._on_right_button_press)
        self.bind("<B3-Motion>", self._on_right_drag)
        self.bind("<ButtonRelease-3>", self._on_right_release)
        
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
        
        # Calculate fit zoom
        zoom_x = canvas_width / img_width if img_width > 0 else 1
        zoom_y = canvas_height / img_height if img_height > 0 else 1
        self._fit_zoom = min(zoom_x, zoom_y, 1.0)
        self._zoom_level = self._fit_zoom
        
        # Reset view
        self.xview_moveto(0)
        self.yview_moveto(0)
        
        self._update_display()
        if self._page_redactions:
            self.draw_redaction_overlays(self._page_redactions)
    
    def zoom_in(self) -> None:
        """Zoom in by 25%."""
        self._zoom_level = min(self._zoom_level * 1.25, 5.0)
        self._update_display()
        if self._page_redactions:
            self.draw_redaction_overlays(self._page_redactions)
    
    def zoom_out(self) -> None:
        """Zoom out by 25%."""
        self._zoom_level = max(self._zoom_level / 1.25, 0.1)
        self._update_display()
        if self._page_redactions:
            self.draw_redaction_overlays(self._page_redactions)
    
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
                    fill="",
                    width=width,
                    dash=dash,
                    tags=("redaction", "overlay")
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
                width=2,
                tags=("redaction", "overlay")
            )
            self._redaction_overlays[canvas_id] = rect.id
            self._rect_id_to_canvas_id[rect.id] = canvas_id
    
    def load_page_redactions(self, redactions: Optional[PageRedactions]) -> None:
        """Clear all overlays and load redactions."""
        self._clear_redaction_overlays()
        self._selected_rect_id = None
        self._page_redactions = redactions
        
        if redactions is None or self._original_image is None:
            return
        
        for rect in redactions.rectangles:
            canvas_rect = self._normalized_to_canvas(rect)
            if canvas_rect:
                canvas_id = self.create_rectangle(
                    *canvas_rect,
                    outline="#dc3545",
                    fill="",
                    width=2,
                    tags=("redaction", "overlay")
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
                width=1,
                tags=("handle", "overlay")
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
    
    def _get_image_offset(self) -> tuple[int, int]:
        """Get the canvas origin (top-left) for shifting content."""
        if self._display_image is None:
            return 0, 0
        
        # Center the image in the scrollable area if it's smaller than the window
        # But we must respect the scrollregion.
        # Simple strategy: Always draw at (0,0) or (Margin, Margin).
        # Tkinter canvas allows negative coordinates, but scrollregion defines the limits.
        
        # Strategy:
        # 1. Image is always at (0, 0) relative to the scrollable canvas.
        # 2. If Window > Image, we center the view? No, we center the CONTENT in the window.
        #    This is done by offsetting the image placement.
        
        win_w = self.winfo_width()
        win_h = self.winfo_height()
        img_w = self._display_image.width()
        img_h = self._display_image.height()
        
        pad_x = max(0, (win_w - img_w) // 2)
        pad_y = max(0, (win_h - img_h) // 2)
        
        return pad_x, pad_y

    def _normalized_to_canvas(self, rect: RedactionRect) -> Optional[tuple[int, int, int, int]]:
        """Convert normalized coordinates to canvas coordinates."""
        if self._original_image is None:
            return None
        
        img_w, img_h = self._original_image.size
        # Use full floats for precision before int cast
        scale = self._zoom_level
        
        pad_x, pad_y = self._get_image_offset()
        
        x0 = rect.x0 * img_w * scale + pad_x
        y0 = rect.y0 * img_h * scale + pad_y
        x1 = rect.x1 * img_w * scale + pad_x
        y1 = rect.y1 * img_h * scale + pad_y
        
        return (int(x0), int(y0), int(x1), int(y1))
    
    def _canvas_to_normalized(self, x: int, y: int) -> tuple[float, float]:
        """Convert canvas coordinates to normalized image coordinates."""
        if self._original_image is None:
            return (0, 0)
        
        img_w, img_h = self._original_image.size
        scale = self._zoom_level
        
        pad_x, pad_y = self._get_image_offset()
        
        # Remove padding
        rel_x = x - pad_x
        rel_y = y - pad_y
        
        img_x = rel_x / scale
        img_y = rel_y / scale
        
        norm_x = img_x / img_w
        norm_y = img_y / img_h
        
        # No clamp - allow detection of out-of-bounds
        return (norm_x, norm_y)

    def _is_on_page(self, norm_x: float, norm_y: float) -> bool:
        """Check if normalized coordinates are within the page bounds."""
        return 0 <= norm_x <= 1 and 0 <= norm_y <= 1
    
    def _update_display(self) -> None:
        """Update the displayed image based on current zoom."""
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
        
        # Determine position (centered if smaller than window)
        pad_x, pad_y = self._get_image_offset()
        
        if self._image_id:
            self.coords(self._image_id, pad_x, pad_y)
            self.itemconfig(self._image_id, image=self._display_image)
        else:
            self._image_id = self.create_image(
                pad_x, pad_y, 
                image=self._display_image, 
                anchor="nw",
                tags="page_image"
            )
            
        # IMPORTANT: Update scrollregion so scan_dragto works properly
        # Make the scrollregion at least the window size, or the image size + padding
        win_w = self.winfo_width()
        win_h = self.winfo_height()
        req_w = max(win_w, pad_x + new_width)
        req_h = max(win_h, pad_y + new_height)
        
        self.config(scrollregion=(0, 0, req_w, req_h))
    
    def _on_button_press(self, event: tk.Event) -> None:
        """Handle mouse button press."""
        # Translate window coords (event.x) to canvas coords (canvasx)
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        
        norm_x, norm_y = self._canvas_to_normalized(cx, cy)
        
        # Identify what was clicked using tags/overlap
        # We use a small rectangle for hit testing to be robust
        hit_items = self.find_overlapping(cx-2, cy-2, cx+2, cy+2)
        tags = set()
        for item in hit_items:
            tags.update(self.gettags(item))
            
        is_handle_hit = "handle" in tags
        is_redaction_hit = "redaction" in tags
        is_page_hit = "page_image" in tags
        
        if self._redaction_mode:
            # 1. Resize (via Handle)
            if self._selected_rect_id and self._page_redactions and is_handle_hit:
                rect = self._page_redactions.find_by_id(self._selected_rect_id)
                if rect:
                    corner = rect.get_corner_at(norm_x, norm_y, tolerance=0.03)
                    if corner:
                        self._drag_mode = "resize"
                        self._resize_corner = corner
                        self._drag_start = (cx, cy)
                        return
            
            # 2. Select / Move (via Existing Rect) - Bounding-box based detection
            # Check if click is inside any redaction rectangle's bounding box
            clicked_rect_id = None
            for canvas_id, rect_id in self._redaction_overlays.items():
                coords = self.coords(canvas_id)  # Returns [x0, y0, x1, y1]
                if len(coords) == 4:
                    x0, y0, x1, y1 = coords
                    if x0 <= cx <= x1 and y0 <= cy <= y1:
                        clicked_rect_id = rect_id
                        break
            
            if clicked_rect_id and self._page_redactions:
                self._selected_rect_id = clicked_rect_id
                self._drag_mode = "move"
                self._drag_start = (cx, cy)
                self._update_selection_styling()
                if self._update_callback:
                    self._update_callback()
                return
            
            # 3. Create New Redaction (Drawing)
            # Only start drawing on empty page area (bounding-box check above handles rect hits)
            if is_page_hit and self._is_on_page(norm_x, norm_y):
                self._drag_mode = "redact"
                self._redaction_start_pos = (cx, cy)
                self._redaction_active = False
                self._current_redaction_rect = None
                
                # Deselect if clicking on empty page space
                if self._selected_rect_id:
                    self._selected_rect_id = None
                    self._clear_handles()
                    self._update_selection_styling() # To refresh previous selection
                return
        
        # Default: Pan
        # (If we clicked background, or neutral mode, or failed other checks)
        self._drag_mode = "pan"
        self.scan_mark(event.x, event.y)
        self.config(cursor="fleur")  # Visual feedback for pan
    
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
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        
        if self._drag_mode == "pan":
            self.scan_dragto(event.x, event.y, gain=1)
            return

        if self._drag_mode == "redact" and self._redaction_start_pos:
            start_x, start_y = self._redaction_start_pos
            
            # Check threshold
            if not self._redaction_active:
                dist = ((cx - start_x)**2 + (cy - start_y)**2)**0.5
                if dist > 5:
                    self._redaction_active = True
                    self._current_redaction_rect = self.create_rectangle(
                        start_x, start_y, cx, cy,
                        outline="#0d6efd", width=2, dash=(4, 4),
                        tags="temp_rect"
                    )
            
            if self._redaction_active and self._current_redaction_rect:
                self.coords(self._current_redaction_rect, start_x, start_y, cx, cy)
                
        elif self._drag_mode == "resize" and self._selected_rect_id and self._page_redactions:
            rect = self._page_redactions.find_by_id(self._selected_rect_id)
            if rect and self._resize_corner:
                norm_x, norm_y = self._canvas_to_normalized(cx, cy)
                rect.resize_corner(self._resize_corner, norm_x, norm_y)
                canvas_id = self._rect_id_to_canvas_id.get(self._selected_rect_id)
                if canvas_id:
                    new_coords = self._normalized_to_canvas(rect)
                    if new_coords:
                        self.coords(canvas_id, *new_coords)
                        self._draw_handles()
                
        elif self._drag_mode == "move" and self._selected_rect_id and self._page_redactions and self._drag_start:
            rect = self._page_redactions.find_by_id(self._selected_rect_id)
            if rect:
                old_norm = self._canvas_to_normalized(*self._drag_start)
                new_norm = self._canvas_to_normalized(cx, cy)
                dx = new_norm[0] - old_norm[0]
                dy = new_norm[1] - old_norm[1]
                
                w = rect.x1 - rect.x0
                h = rect.y1 - rect.y0
                rect.x0 = max(0, min(1 - w, rect.x0 + dx))
                rect.y0 = max(0, min(1 - h, rect.y0 + dy))
                rect.x1 = rect.x0 + w
                rect.y1 = rect.y0 + h
                
                self._drag_start = (cx, cy)
                
                canvas_id = self._rect_id_to_canvas_id.get(self._selected_rect_id)
                if canvas_id:
                    new_coords = self._normalized_to_canvas(rect)
                    if new_coords:
                        self.coords(canvas_id, *new_coords)
                        self._draw_handles()
    
    def _on_button_release(self, event: tk.Event) -> None:
        """Handle mouse button release."""
        cx = self.canvasx(event.x)
        cy = self.canvasy(event.y)
        
        # Reset cursor if we were panning
        if self._drag_mode == "pan":
             self.config(cursor="crosshair" if self._redaction_mode else "")
        
        if self._drag_mode == "redact":
            if self._current_redaction_rect:
                self.delete(self._current_redaction_rect)
                self._current_redaction_rect = None
            
            if self._redaction_active and self._redaction_start_pos:
                start_norm = self._canvas_to_normalized(*self._redaction_start_pos)
                end_norm = self._canvas_to_normalized(cx, cy)
                
                x0 = max(0, min(1, min(start_norm[0], end_norm[0])))
                y0 = max(0, min(1, min(start_norm[1], end_norm[1])))
                x1 = max(0, min(1, max(start_norm[0], end_norm[0])))
                y1 = max(0, min(1, max(start_norm[1], end_norm[1])))
                
                if abs(x1 - x0) > 0.01 and abs(y1 - y0) > 0.01:
                    rect = RedactionRect(x0, y0, x1, y1)
                    if self._redaction_callback:
                        self._redaction_callback(rect)

        self._drag_mode = "pan"
        self._resize_corner = None
        self._drag_start = None
        self._redaction_start_pos = None
        self._redaction_active = False

    def _on_right_button_press(self, event: tk.Event) -> None:
        """Handle right mouse button press for unconditional panning."""
        self._drag_mode = "pan_right"
        self.scan_mark(event.x, event.y)
        self.config(cursor="fleur")

    def _on_right_drag(self, event: tk.Event) -> None:
        """Handle right mouse drag."""
        if self._drag_mode == "pan_right":
            self.scan_dragto(event.x, event.y, gain=1)
    
    def _on_right_release(self, event: tk.Event) -> None:
        """Handle right mouse button release."""
        self._drag_mode = "pan"  # Reset
        # Restore cursor based on mode
        self.config(cursor="crosshair" if self._redaction_mode else "")
    
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
    
    # Modern Dashboard Color Palette
    BG_COLOR = "#F8FAFC"           # Very light gray background
    CARD_BG = "#FFFFFF"            # White card background
    ACCENT_COLOR = "#2563EB"       # Academic Blue
    PREVIEW_BG = "#334155"         # Dark slate gray for preview
    BORDER_SUBTLE = "#E2E8F0"      # Subtle borders
    
    def __init__(self):
        super().__init__()
        self.style = tb.Style()
        self.style.theme_use("litera")  # Clean, academic-modern theme
        
        # Configure main window
        self.title("Exam Cleaner")
        self.geometry("1300x1125")
        self.minsize(1024, 768)
        self.configure(bg=self.BG_COLOR)
        
        # Configure custom styles for Modern Dashboard
        self._configure_custom_styles()
        
        # Configure grid weights for the main window
        self.grid_rowconfigure(1, weight=1)  # Main content expands
        self.grid_columnconfigure(0, weight=1)  # Full width
        
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
    
    def _configure_custom_styles(self) -> None:
        """Configure custom ttk styles for Modern Dashboard look."""
        style = self.style
        
        # Card frame style (white background with subtle border simulation)
        style.configure("Card.TFrame", background=self.CARD_BG)
        
        # Header frame style
        style.configure("Header.TFrame", background=self.CARD_BG)
        
        # Section header labels
        style.configure(
            "SectionHeader.TLabel",
            font=("", 10, "bold"),
            background=self.CARD_BG
        )
        
        # Primary action button (solid blue)
        style.configure(
            "Primary.TButton",
            font=("", 10, "bold")
        )
        
        # Status badge style
        style.configure(
            "StatusBadge.TLabel",
            font=("", 9),
            foreground="#16A34A",  # Green for "Ready"
            background=self.CARD_BG
        )
        
        # Preview controls frame (for floating bar)
        style.configure("PreviewControls.TFrame", background=self.PREVIEW_BG)
        
        # Configure labelframe to have white background
        style.configure("Card.TLabelframe", background=self.CARD_BG)
        style.configure("Card.TLabelframe.Label", background=self.CARD_BG, font=("", 10, "bold"))
    
    def _build_ui(self) -> None:
        """Build the user interface with Modern Dashboard layout."""
        # Header bar (row 0)
        self._build_header()
        
        # Main content area (row 1) - Cards layout
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(10, 20))
        main_frame.configure(style="TFrame")
        
        # Configure grid for main_frame
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(1, weight=1)  # Preview card expands
        
        # Settings Card (Left)
        self._build_settings_card(main_frame)
        
        # Preview Card (Right)
        self._build_preview_card(main_frame)
        
        # Status bar (row 2)
        self._build_status_bar()
    
    def _build_header(self) -> None:
        """Build the slim professional header bar."""
        header = ttk.Frame(self, style="Header.TFrame")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        
        # Left side: Open PDF button
        self._open_btn = ttk.Button(
            header, 
            text="📂 Open PDF",
            bootstyle="secondary-outline",
            command=self._open_file
        )
        self._open_btn.pack(side=LEFT, padx=(0, 10))
        
        # Export button (Primary Action - solid blue)
        self._export_btn = ttk.Button(
            header,
            text="💾 Export",
            bootstyle="primary",
            command=self._start_export,
            state=DISABLED
        )
        self._export_btn.pack(side=LEFT, padx=(0, 5))
        
        # Status badge next to Export
        self._status_badge = ttk.Label(
            header,
            text="● Ready",
            style="StatusBadge.TLabel"
        )
        self._status_badge.pack(side=LEFT, padx=(5, 15))
        
        # Cancel button (hidden initially)
        self._cancel_btn = ttk.Button(
            header,
            text="⏹ Cancel",
            bootstyle="danger-outline",
            command=self._cancel_operation
        )
        
        # Separator
        ttk.Separator(header, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=15)
        
        # Redaction mode toggle
        self._redaction_var = tk.BooleanVar(value=False)
        self._redaction_check = ttk.Checkbutton(
            header,
            text="✏️ Redaction Mode",
            variable=self._redaction_var,
            bootstyle="warning-round-toggle",
            command=self._toggle_redaction_mode
        )
        self._redaction_check.pack(side=LEFT, padx=5)
        
        # Right side: File name label
        self._file_label = ttk.Label(
            header,
            text="",
            font=("", 10, "italic"),
            foreground="#64748B"
        )
        self._file_label.pack(side=RIGHT, padx=10)
    
    def _build_settings_card(self, parent) -> None:
        """Build the Settings Card (left panel)."""
        # Card container with padding
        card = ttk.Frame(parent, style="Card.TFrame", padding=15)
        card.grid(row=0, column=0, sticky="ns", padx=(0, 15))
        
        # Fixed width for settings card
        card.configure(width=320)
        card.grid_propagate(False)
        
        # Page list section
        self._build_page_list(card)
        
        # Settings section  
        self._build_settings_panel(card)
    
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

    def _build_nav_toolbar(self) -> None:
        """Build the global navigation toolbar (Page, Zoom, Edit)."""
        nav_frame = ttk.Frame(self)
        nav_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        
        # Page Navigation (Left)
        ttk.Button(nav_frame, text="◀", width=3, command=self._prev_page, bootstyle="secondary").pack(side=LEFT)
        ttk.Button(nav_frame, text="▶", width=3, command=self._next_page, bootstyle="secondary").pack(side=LEFT, padx=2)
        
        self._page_label = ttk.Label(nav_frame, text="Page 0 / 0", font=("", 10))
        self._page_label.pack(side=LEFT, padx=10)
        
        # Zoom controls (Left)
        ttk.Separator(nav_frame, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        ttk.Button(nav_frame, text="−", width=3, command=self._zoom_out, bootstyle="secondary").pack(side=LEFT)
        self._zoom_label = ttk.Label(nav_frame, text="100%", width=6)
        self._zoom_label.pack(side=LEFT)
        ttk.Button(nav_frame, text="+", width=3, command=self._zoom_in, bootstyle="secondary").pack(side=LEFT)
        ttk.Button(nav_frame, text="Fit", command=self._fit_to_window, bootstyle="secondary-outline").pack(side=LEFT, padx=5)
        
        # Redaction controls (Right)
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
        """Build the page list with card styling."""
        # Section header
        ttk.Label(
            parent,
            text="Document Pages",
            style="SectionHeader.TLabel"
        ).pack(anchor=W, pady=(0, 8))
        
        # List container
        list_frame = ttk.Frame(parent, style="Card.TFrame")
        list_frame.pack(fill=BOTH, expand=True, pady=(0, 15))
        
        # Empty state label
        self._empty_label = ttk.Label(
            list_frame,
            text="📄 Open a PDF to begin",
            font=("", 11),
            foreground="#94A3B8",
            background=self.CARD_BG
        )
        self._empty_label.pack(expand=True, pady=30)
        
        # Page listbox with scrollbar
        self._page_list_frame = ttk.Frame(list_frame, style="Card.TFrame")
        
        self._page_listbox = tk.Listbox(
            self._page_list_frame,
            font=("", 10),
            selectmode=SINGLE,
            activestyle="none",
            highlightthickness=0,
            bd=1,
            relief="flat",
            bg=self.CARD_BG,
            selectbackground=self.ACCENT_COLOR,
            selectforeground="white"
        )
        self._page_listbox.bind("<<ListboxSelect>>", self._on_page_select)
        
        scrollbar = ttk.Scrollbar(self._page_list_frame, orient=VERTICAL, command=self._page_listbox.yview)
        self._page_listbox.config(yscrollcommand=scrollbar.set)
        
        self._page_listbox.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y)
    
    def _build_settings_panel(self, parent) -> None:
        """Build the settings panel with presets and advanced options."""
        # Section header
        ttk.Label(
            parent,
            text="Settings",
            style="SectionHeader.TLabel"
        ).pack(anchor=W, pady=(0, 10))
        
        settings_frame = ttk.Frame(parent, style="Card.TFrame")
        settings_frame.pack(fill=X)
        
        # Hide Answers Strength
        ttk.Label(
            settings_frame,
            text="Hide Answers Strength",
            style="SectionHeader.TLabel"
        ).pack(anchor=W, pady=(0, 8))
        
        strength_frame = ttk.Frame(settings_frame, style="Card.TFrame")
        strength_frame.pack(fill=X, pady=(0, 15))
        
        self._strength_var = tk.StringVar(value=self._settings.strength_preset.value)
        for preset in StrengthPreset:
            label = preset.value.capitalize()
            ttk.Radiobutton(
                strength_frame,
                text=label,
                value=preset.value,
                variable=self._strength_var,
                command=self._on_strength_change,
                bootstyle="primary-toolbutton"  # Uses accent color when active
            ).pack(side=LEFT, padx=(0, 4), expand=True, fill=X)
        
        # Output Quality
        ttk.Label(
            settings_frame,
            text="Output Quality",
            style="SectionHeader.TLabel"
        ).pack(anchor=W, pady=(0, 8))
        
        output_frame = ttk.Frame(settings_frame, style="Card.TFrame")
        output_frame.pack(fill=X, pady=(0, 15))
        
        self._output_var = tk.StringVar(value=self._settings.output_preset.value)
        ttk.Radiobutton(
            output_frame,
            text="High Quality",
            value=OutputPreset.HIGH_QUALITY.value,
            variable=self._output_var,
            command=self._on_output_change,
            bootstyle="primary-toolbutton"
        ).pack(side=LEFT, padx=(0, 4), expand=True, fill=X)
        ttk.Radiobutton(
            output_frame,
            text="Balanced",
            value=OutputPreset.BALANCED.value,
            variable=self._output_var,
            command=self._on_output_change,
            bootstyle="primary-toolbutton"
        ).pack(side=LEFT, padx=(0, 4), expand=True, fill=X)
        
        # Advanced expander
        self._advanced_var = tk.BooleanVar(value=False)
        advanced_toggle = ttk.Checkbutton(
            settings_frame,
            text="▶ Advanced Options",
            variable=self._advanced_var,
            command=self._toggle_advanced,
            bootstyle="secondary-outline-toolbutton"
        )
        advanced_toggle.pack(fill=X, pady=(10, 0))
        
        self._advanced_frame = ttk.Frame(settings_frame, style="Card.TFrame")
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
        """Build the right preview panel (legacy, kept for compatibility)."""
        self._build_preview_card(parent)
    
    def _build_preview_card(self, parent) -> None:
        """Build the Preview Card with dark background and floating controls."""
        # Preview card container
        preview_card = ttk.Frame(parent, style="Card.TFrame", padding=0)
        preview_card.grid(row=0, column=1, sticky="nsew")
        
        # Configure grid for preview card
        preview_card.grid_rowconfigure(0, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)
        
        # Dark preview area
        preview_area = tk.Frame(preview_card, bg=self.PREVIEW_BG)
        preview_area.grid(row=0, column=0, sticky="nsew")
        preview_area.grid_rowconfigure(0, weight=1)
        preview_area.grid_columnconfigure(0, weight=1)
        
        # Preview canvas with dark background
        self._preview_canvas = PreviewCanvas(
            preview_area,
            bg=self.PREVIEW_BG,
            highlightthickness=0
        )
        self._preview_canvas.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Floating controls bar at bottom-center
        self._build_floating_controls(preview_area)
    
    def _build_floating_controls(self, parent) -> None:
        """Build the floating controls bar at the bottom of the preview."""
        # Container for floating bar (centered at bottom)
        controls_container = tk.Frame(parent, bg=self.PREVIEW_BG)
        controls_container.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        
        # Center the controls bar
        controls_container.grid_columnconfigure(0, weight=1)
        controls_container.grid_columnconfigure(2, weight=1)
        
        # Floating bar frame (semi-transparent look with card background)
        floating_bar = ttk.Frame(controls_container, style="Card.TFrame", padding=(15, 8))
        floating_bar.grid(row=0, column=1)
        
        # Page Navigation
        ttk.Button(
            floating_bar,
            text="◀",
            width=3,
            command=self._prev_page,
            bootstyle="secondary-outline"
        ).pack(side=LEFT, padx=(0, 5))
        
        self._page_label = ttk.Label(
            floating_bar,
            text="Page 0 / 0",
            font=("", 10),
            background=self.CARD_BG
        )
        self._page_label.pack(side=LEFT, padx=5)
        
        ttk.Button(
            floating_bar,
            text="▶",
            width=3,
            command=self._next_page,
            bootstyle="secondary-outline"
        ).pack(side=LEFT, padx=(5, 15))
        
        # Separator
        ttk.Separator(floating_bar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        
        # Zoom controls
        ttk.Button(
            floating_bar,
            text="−",
            width=3,
            command=self._zoom_out,
            bootstyle="secondary-outline"
        ).pack(side=LEFT, padx=(0, 5))
        
        self._zoom_label = ttk.Label(
            floating_bar,
            text="100%",
            width=6,
            background=self.CARD_BG
        )
        self._zoom_label.pack(side=LEFT)
        
        ttk.Button(
            floating_bar,
            text="+",
            width=3,
            command=self._zoom_in,
            bootstyle="secondary-outline"
        ).pack(side=LEFT, padx=(5, 10))
        
        ttk.Button(
            floating_bar,
            text="Fit",
            command=self._fit_to_window,
            bootstyle="secondary-outline"
        ).pack(side=LEFT, padx=(0, 15))
        
        # Separator
        ttk.Separator(floating_bar, orient=VERTICAL).pack(side=LEFT, fill=Y, padx=10)
        
        # Redaction controls
        self._undo_btn = ttk.Button(
            floating_bar,
            text="↩ Undo",
            command=self._undo_last_redaction,
            bootstyle="secondary-outline",
            state=DISABLED
        )
        self._undo_btn.pack(side=LEFT, padx=(0, 5))
        
        self._delete_rect_btn = ttk.Button(
            floating_bar,
            text="🗑 Delete",
            command=self._delete_selected_rect,
            bootstyle="danger-outline",
            state=DISABLED
        )
        self._delete_rect_btn.pack(side=LEFT)
    
    def _build_status_bar(self) -> None:
        """Build the status bar."""
        self._status_frame = ttk.Frame(self, style="Card.TFrame")
        self._status_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(0, 10))
        
        self._status_label = ttk.Label(
            self._status_frame,
            text="Ready",
            font=("", 9),
            foreground="#64748B"
        )
        self._status_label.pack(side=LEFT, padx=10, pady=5)
        
        # Progress bar (hidden initially)
        self._progress_var = tk.DoubleVar(value=0)
        self._progress_bar = ttk.Progressbar(
            self._status_frame, 
            variable=self._progress_var, 
            maximum=100,
            bootstyle="primary-striped",
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
            self._raw_preview_cache.clear()
            
            self._current_page = 0
            
            # Update UI
            self._file_label.config(text=path.name)
            self._export_btn.config(state=NORMAL)
            self._update_page_label()
            self._refresh_page_list()
            self._update_preview()
            
            self._set_status(f"Opened: {path.name} ({self._document.page_count} pages)")
            logger.info(f"Opened file: {path}")
            
        except Exception as e:
            logger.exception(f"Failed to open file: {path}")
            messagebox.showerror("Error", f"Failed to open file:\n{str(e)}")
    
    def _refresh_page_list(self) -> None:
        """Refresh the page list."""
        if not self._document:
            self._empty_label.config(text="📄 Open a PDF to begin")
            self._empty_label.pack(expand=True)
            self._page_list_frame.pack_forget()
            return
        
        self._empty_label.pack_forget()
        self._page_list_frame.pack(fill=BOTH, expand=True)
        
        # Clear listbox
        self._page_listbox.delete(0, tk.END)
        
        for i in range(self._document.page_count):
            entry = f"Page {i + 1}"
            self._page_listbox.insert(tk.END, entry)
    
    def _on_page_select(self, event) -> None:
        """Handle page selection from list."""
        selection = self._page_listbox.curselection()
        if not selection:
            return
        
        # Map listbox index to actual page
        idx = selection[0]
        if idx < self._document.page_count:
            self._current_page = idx
        
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
            self._sync_redaction_ui_state  # Callback to update button states on selection change
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
                # Update visual styling to reflect cleared selection
                self._preview_canvas._update_selection_styling()
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
        """Cancel current operation (export)."""
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
        # Print current window size for user reference
        print(f"Window geometry on close: {self.winfo_width()}x{self.winfo_height()}")
        
        self._preview_debounce.cancel()
        
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
