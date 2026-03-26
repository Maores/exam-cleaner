# Exam Cleaner - Premium UI/UX Redesign
## Date: 2026-03-26

## Goal
The goal of this phase is to drastically improve the user experience (UX) and visual interface (UI) of the Exam Cleaner desktop application. It transforms the current basic layout into a sleek, premium experience while adding "fluid" controls to the core redaction workflow.

## 1. Dashboard Architecture (The Outer Shell)
**Current:** Standard single-window layout with exposed setting grids.
**New Design:** 
*   **Sidebar Navigation System:** A persistent left-aligned navigation bar containing large, clear buttons for distinct phases of the workflow: `Open Document`, `Analysis & Settings`, `Redaction Canvas`, and `Export`. 
*   **Card-based Content:** The center panel is a frame that swaps its contents based on the sidebar selection.
    *   Settings like "Preset Strength" and "Advanced Options" will be grouped into clean, rounded cards with `ttkbootstrap` elements (e.g. `LabelFrame` or custom styled frames) to ensure everything has breathing room.
    *   A prominent dark theme will be used to enhance the "premium" feel.

## 2. Fluid Redaction Experience (The Inner Editor)
**Current:** Basic interaction for drawing rectangles; panning requires scrollbars; zoom resets or feels clunky.
**New Design (Canvas Upgrades):**
*   **Panning:** Middle-click AND Spacebar+Left-click will natively bind to the Tkinter canvas `scan_mark` and `scan_dragto` features, allowing smooth, infinite panning of the document like Adobe Acrobat or Photoshop.
*   **Zooming:** `Ctrl + Scroll Wheel` will zoom in/out with respect to the mouse cursor position without resetting when actions are taken.
*   **Smart Selection:** We drop rigid "Modes" (Draw vs Select). Instead:
    *   Clicking an empty space creates a new box.
    *   Hovering over an existing box will highlight it or change the cursor (e.g., to a movement crosshair or sizing double-arrow).
    *   Clicking and dragging an existing box immediately moves it seamlessly.
*   **Undo/Redo:** Standard `Ctrl+Z` to undo the last redaction action.

## 3. Data Flow & State Management
*   The application currently uses `models.py` to hold state. We will extend state management so the currently selected menu node dictates the active `tk.Frame` visible in the content area.
*   The Redaction canvas state will hold a history stack (`undo_stack`) of actions for seamless reversing.

## Testing & Verification
*   **Manual UI Testing:** Verify that clicking sidebar items smoothly transitions the center panes without visual tearing.
*   **Redaction Testing:** Ensure panning and zooming do not cause the redaction rectangles to draw in incorrect offset positions (a common canvas bug).
