import tkinter as tk
import sys
import os

# Add parent directory to path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import Application

def test_sidebar_presence():
    app = Application()
    # Ensure there is a sidebar frame created
    sidebar = getattr(app, '_sidebar_frame', None)
    assert sidebar is not None, "Sidebar frame should be initialized in app"
    app.destroy()
