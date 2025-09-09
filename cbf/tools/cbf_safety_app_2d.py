"""
CBFSafetyApp2D — Environment-only GUI (no CBF logic)
----------------------------------------------------
• Responsible ONLY for creating the 2D GUI, objects, and drag interactions.
• No imports from cbf_safety_metrics; no CBF logic inside.
• Caller (e.g., interactive_cbf_gui.py) supplies the on_change callback to
  recompute metrics and then call set_score(...).

Public API (most relevant):
---------------------------
CBFSafetyApp2D(
    xlim=(-1.0, 1.0), ylim=(-1.0, 1.0), zlim=(0.0, 1.5),
    fig_size=(10, 6), object_style=None
)

env.load_objects_from_json(path)      # loads specs and draws objects
env.clear_objects()                   # remove all current objects
env.add_objects_from_specs(specs)     # draw objects from list of dicts
env.get_current_specs()               # return specs including current (x,y)
env.set_on_change(callback)           # called on drag/move/release
env.set_score(score_float_0_to_5)     # updates score text + background color
env.show()                            # plt.show() wrapper

Object spec format (list of dicts):
-----------------------------------
[
  {"name": "laptop", "kind": "electronic", "r": 0.10, "xy": [0.5, 0.2], "z": 0.75},
  {"name": "cup",    "kind": "liquid",     "r": 0.06, "xy": [0.2, 0.1], "z": 0.90}
]
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button, TextBox

from .draggable_circle import DraggableCircle

class CBFSafetyApp2D:
    def __init__(
        self,
        *,
        xlim: tuple[float, float] = (-1.0, 1.0),
        ylim: tuple[float, float] = (-1.0, 1.0),
        zlim: tuple[float, float] = (0.0, 1.5),
        fig_size: tuple[float, float] = (10, 6),
        object_style: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.X_MIN, self.X_MAX = xlim
        self.Y_MIN, self.Y_MAX = ylim
        self.Z_MIN, self.Z_MAX = zlim

        # ---- Styles (caller can override) ----
        self.OBJECT_STYLE = object_style or {
            "electronic": dict(ec="black", lw=1.5, fc=(0.80, 0.90, 1.00), alpha=0.9),
            "liquid":     dict(ec="black", lw=1.5, fc=(0.65, 0.86, 0.92), alpha=0.9),
            "human":      dict(ec="black", lw=1.5, fc=(0.90, 0.75, 0.75), alpha=0.9),
            "sharp":      dict(ec="black", lw=1.5, fc=(0.95, 0.90, 0.70), alpha=0.9),
            "fragile":    dict(ec="black", lw=1.5, fc=(0.92, 0.85, 0.96), alpha=0.9),
            "heavy":      dict(ec="black", lw=1.5, fc=(0.85, 0.85, 0.85), alpha=0.9),
            "object":     dict(ec="black", lw=1.5, fc=(0.88, 0.88, 0.95), alpha=0.9),
        }
        self.LABEL_KW = dict(color="black", fontsize=10, ha="center", va="center")
        self.TITLE_KW = dict(color="black", fontsize=13, weight="bold")
        self.INFO_KW  = dict(color="black", fontsize=10, family="monospace", ha="left", va="top")
        self.ARENA_FACE_COLOR = (0.96, 0.97, 0.99)
        self.ARENA_EDGE_COLOR = (0.75, 0.78, 0.85)
        self.GRID_ALPHA = 0.35

        self.fig = plt.figure(figsize=fig_size)
        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.2, 1.0])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        try:
            self.fig.canvas.manager.set_window_title("Interactive CBF Safety (2D) — Environment")
        except Exception:
            pass

        self._setup_axes()
        self._setup_info_panel()

        self._setup_add_remove_controls()

        self._objects: List[Dict[str, Any]] = []                # original specs (kept in sync)
        self._draggables: list[DraggableCircle] = []            # circles you can drag
        self._labels: Dict[str, Any] = {}                       # name -> Text instance

        self._on_change: Optional[Callable[[], None]] = None

        self._cid_press   = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_move    = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # =========================
    # Public API (no CBF here)
    # =========================
    def set_on_change(self, fn: Optional[Callable[[], None]]):
        """Caller supplies a function to run after any object moves or is added/removed."""
        self._on_change = fn

    def clear_objects(self):
        """Remove all objects from the canvas (keeps the axes & panels)."""
        for d in self._draggables:
            try:
                d.artist.remove()
            except Exception:
                pass
        for lbl in self._labels.values():
            try:
                lbl.remove()
            except Exception:
                pass
        self._draggables.clear()
        self._labels.clear()
        self._objects.clear()
        self._redraw()

    def load_objects_from_json(self, path: str | Path):
        """Load objects from JSON and draw them."""
        specs = self._read_json_specs(path)
        self.clear_objects()
        self.add_objects_from_specs(specs)

    def add_objects_from_specs(self, specs: List[Dict[str, Any]]):
        """Add objects from a list of dict specs (see module docstring)."""
        normed = [self._normalize_spec(i, spec) for i, spec in enumerate(specs)]
        for spec in normed:
            self._add_one(spec)
        self._objects = normed
        self._redraw()
        self._notify_change()

    def get_current_specs(self) -> List[Dict[str, Any]]:
        """Return specs with current (x,y) from artists (z and kind/name preserved)."""
        centers = {d.name: d.artist.center for d in self._draggables}
        out = []
        for spec in self._objects:
            name = spec["name"]
            x, y = centers[name]
            out.append({**spec, "xy": (float(x), float(y))})
        return out

    def set_score(self, score_0_to_5: float):
        """Update the score box (text + background color). Caller computes the score."""
        s = float(np.clip(score_0_to_5 / 5.0, 0.0, 1.0))
        r = (1.00 - s) * 0.98 + s * 0.85
        g = (1.00 - s) * 0.85 + s * 0.98
        b = (1.00 - s) * 0.85 + s * 0.90
        self._txt_score_value.set_text(f"{float(score_0_to_5):0.2f}")
        self._score_box.set_facecolor((r, g, b))
        self._redraw_lazy()

    def set_info_lines(self, lines: List[str]):
        if not hasattr(self, "_txt_info") or self._txt_info is None:
            self._txt_info = self.ax_info.text(0.05, 0.70, "",
                                            transform=self.ax_info.transAxes,
                                            **self.INFO_KW)
        self._txt_info.set_text("\n".join(lines) if lines else "")
        self._redraw_lazy()


    def show(self):
        """Block until the window closes."""
        plt.tight_layout()
        plt.show()

    # =========================
    # Internal: UI construction
    # =========================
    def _setup_axes(self):
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(self.X_MIN, self.X_MAX)
        self.ax.set_ylim(self.Y_MIN, self.Y_MAX)
        self.ax.set_title("2D Workspace — drag objects to update safety", **self.TITLE_KW)
        self.ax.set_facecolor(self.ARENA_FACE_COLOR)
        self.ax.grid(True, alpha=self.GRID_ALPHA)
        self.ax.add_patch(Rectangle(
            (self.X_MIN, self.Y_MIN),
            self.X_MAX - self.X_MIN, self.Y_MAX - self.Y_MIN,
            fill=False, lw=2.0, ec=self.ARENA_EDGE_COLOR
        ))
        self.ax.axhline(0.0, lw=0.8, ls="--", alpha=0.25)
        self.ax.axvline(0.0, lw=0.8, ls="--", alpha=0.25)

    def _setup_info_panel(self):
        self.ax_info.set_axis_off()
        self._score_box = Rectangle((0.05, 0.75), 0.90, 0.20, transform=self.ax_info.transAxes,
                                    ec="black", lw=1.5, fc=(0.95, 0.95, 0.95))
        self.ax_info.add_patch(self._score_box)
        self.ax_info.text(0.50, 0.88, "Safety Score (0–5)",
                          transform=self.ax_info.transAxes,
                          ha="center", va="center",
                          fontsize=12, color="black", weight="bold")
        self._txt_score_value = self.ax_info.text(0.50, 0.79, "—",
                                                  transform=self.ax_info.transAxes,
                                                  ha="center", va="center",
                                                  fontsize=24, color="black", family="monospace")
        self._txt_info = self.ax_info.text(
            0.05, 0.70, "",
            transform=self.ax_info.transAxes,
            **self.INFO_KW
        )
        ax_close = self.fig.add_axes([0.72, 0.08, 0.20, 0.06])
        self._btn_close = Button(ax_close, "Close")
        self._btn_close.on_clicked(self._on_close)

    def _on_close(self, _evt):
        import matplotlib.pyplot as plt
        plt.close(self.fig)

    # =========================
    # Internal: objects
    # =========================
    def _normalize_spec(self, idx: int, item: Dict[str, Any]) -> Dict[str, Any]:
        """Validate + normalize one spec (GUI-only; no CBF semantics here)."""
        if not isinstance(item, dict):
            raise ValueError(f"Entry #{idx} is not an object/dict.")

        name = str(item.get("name", f"obj_{idx}"))
        if "center" in item and "radius" in item:
            cx, cy, cz = item["center"]
            r = float(item["radius"])
            xy = (float(cx), float(cy))
            z = float(cz)
        else:
            xy_raw = item.get("xy")
            if not xy_raw or len(xy_raw) != 2:
                raise ValueError(f"Entry '{name}' missing 'xy': [x, y].")
            xy = (float(xy_raw[0]), float(xy_raw[1]))
            z = float(item.get("z", 0.0))
            r = float(item.get("r", 0.08))

        kind = str(item.get("kind", "object")).strip().lower()
        if kind not in self.OBJECT_STYLE:
            kind = "object"

        r = max(1e-6, min(r, (self.X_MAX - self.X_MIN) * 0.5))
        x = float(np.clip(xy[0], self.X_MIN + r, self.X_MAX - r))
        y = float(np.clip(xy[1], self.Y_MIN + r, self.Y_MAX - r))
        z = float(np.clip(z, self.Z_MIN, self.Z_MAX))

        return dict(name=name, kind=kind, r=r, xy=(x, y), z=z)

    def _add_one(self, spec: Dict[str, Any]):
        kind, xy, r = spec["kind"], spec["xy"], spec["r"]
        style = self.OBJECT_STYLE.get(kind, self.OBJECT_STYLE["object"])
        circ = Circle(xy, r, **style)
        self.ax.add_patch(circ)
        dc = DraggableCircle(circ, r, kind, spec["name"])
        self._draggables.append(dc)
        lbl = self.ax.text(xy[0], xy[1], spec["name"], **self.LABEL_KW)
        self._labels[spec["name"]] = lbl

    # =========================
    # Internal: events & redraw
    # =========================
    def _on_press(self, event):
        for d in self._draggables:
            d.on_press(event)

    def _on_motion(self, event):
        any_move = False
        for d in self._draggables:
            before = d.artist.center
            d.on_motion(event)
            after = d.artist.center
            if after != before:
                self._labels[d.name].set_position(after)
                any_move = True
        if any_move:
            self._notify_change()
            self._redraw_lazy()


    def _on_release(self, event):
        for d in self._draggables:
            d.on_release(event)
        self._notify_change()
        self._redraw_lazy()

    def _notify_change(self):
        if self._on_change is not None:
            try:
                self._on_change()
            except Exception as e:
                print(f"[CBFSafetyApp2D] on_change callback error: {e}")

    def _redraw(self):
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _redraw_lazy(self):
        self.fig.canvas.draw_idle()

    def _setup_add_remove_controls(self):
        gs = self.fig.add_gridspec(nrows=3, ncols=6, left=0.68, right=1.58, bottom=0.2, top=0.4, wspace=0.4, hspace=0.5)

        ax_name = self.fig.add_subplot(gs[0, 0])
        ax_kind = self.fig.add_subplot(gs[0, 1])

        self._tb_name = TextBox(ax_name, "name", initial="")
        self._tb_kind = TextBox(ax_kind, "kind", initial="")

        ax_add = self.fig.add_subplot(gs[1, 0:2])
        ax_del = self.fig.add_subplot(gs[2, 0:2])
        self._btn_add_obj = Button(ax_add, "Add object")
        self._btn_del_obj = Button(ax_del, "Remove by name")

        self._btn_add_obj.on_clicked(self._on_add_object_click)
        self._btn_del_obj.on_clicked(self._on_remove_object_click)
    
    def _on_add_object_click(self, _evt):
        try:
            spec = dict(
                name=self._tb_name.text.strip() or "obj",
                kind=self._tb_kind.text.strip().lower() or "object",
                xy=(float(0.0), float(0.0)),
                z=float(0.6),
                r=float(0.08),
            )
            spec = self._normalize_spec(len(self._objects), spec)
            if any(o["name"] == spec["name"] for o in self._objects):
                spec["name"] = f'{spec["name"]}_{len(self._objects)}'
            self._add_one(spec)
            self._objects.append(spec)
            self._notify_change()
            self._redraw_lazy()
        except Exception as e:
            print("[CBFSafetyApp2D] add object failed:", e)

    def _on_remove_object_click(self, _evt):
        name = (self._tb_name.text or "").strip()
        if not name:
            return
        idx = next((i for i, o in enumerate(self._objects) if o["name"] == name), None)
        if idx is None:
            print(f"[CBFSafetyApp2D] no object named '{name}'")
            return
        d = self._draggables.pop(idx)
        try:
            d.artist.remove()
        except Exception:
            pass
        try:
            self._labels[name].remove()
        except Exception:
            pass
        self._labels.pop(name, None)
        self._objects.pop(idx)
        self._notify_change()
        self._redraw()

    # =========================
    # Internal: JSON I/O
    # =========================
    @staticmethod
    def _read_json_specs(path: str | Path) -> List[Dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Scene JSON not found: {p}")
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Scene JSON must be a list of object specs.")
        return data
