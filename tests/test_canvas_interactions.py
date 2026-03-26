import tkinter as tk
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import PreviewCanvas

def test_spacebar_panning_state():
    root = tk.Tk()
    canvas = PreviewCanvas(root)
    # Call handler directly for unit testing
    canvas._on_space_press(None)
    assert canvas._drag_mode == "pan_armed", f"Expected pan_armed, got {canvas._drag_mode}"
    root.destroy()
