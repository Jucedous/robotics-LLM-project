#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive CBF Safety Playground (2D Top-Down) — Objects Only
--------------------------------------------------------------
• Press 'a' to arm add-mode, then click in the canvas → a popup asks for a name; object is added there.
• Drag circles to move them; safety metrics update live.
• Press 'p' to force-refresh the hazard policy panel (usually auto-updates).

Requires:
- cbf_safety_metrics.py (providing ObjectState, Sphere, Workspace, Scene, evaluate_scene_metrics)
- hazard_policy.py (providing get_hazard_policy_via_llm; set HAZARD_POLICY_MOCK=1 for offline tests)
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

try:
    import tkinter as tk
    from tkinter import simpledialog
    _TK_AVAILABLE = True
except Exception:
    _TK_AVAILABLE = False

from cbf_safety_metrics import (
    ObjectState, Sphere, Workspace, Scene, evaluate_scene_metrics
)
from cbf.junks.hazard_policy import get_hazard_policy_via_llm

# -------------------------
# Config
# -------------------------
X_MIN, X_MAX = -1.0, 1.0
Y_MIN, Y_MAX = -1.0, 1.0
Z_MIN, Z_MAX = 0.0, 1.5

FIG_W, FIG_H = 10, 6
ARENA_FACE_COLOR = (0.96, 0.97, 0.99)
ARENA_EDGE_COLOR = (0.75, 0.78, 0.85)
GRID_ALPHA = 0.35

OBJECT_STYLE = {
    "electronic": dict(ec="black", lw=1.5, fc=(0.80, 0.90, 1.00), alpha=0.9),
    "liquid":     dict(ec="black", lw=1.5, fc=(0.65, 0.86, 0.92), alpha=0.9),
    "human":      dict(ec="black", lw=1.5, fc=(0.90, 0.75, 0.75), alpha=0.9),
    "sharp":      dict(ec="black", lw=1.5, fc=(0.95, 0.90, 0.70), alpha=0.9),
    "fragile":    dict(ec="black", lw=1.5, fc=(0.92, 0.85, 0.96), alpha=0.9),
    "heavy":      dict(ec="black", lw=1.5, fc=(0.85, 0.85, 0.85), alpha=0.9),
    "object":     dict(ec="black", lw=1.5, fc=(0.88, 0.88, 0.95), alpha=0.9),
}

LABEL_KW = dict(color="black", fontsize=10, ha="center", va="center")
TITLE_KW = dict(color="black", fontsize=13, weight="bold")
INFO_KW  = dict(color="black", fontsize=10, family="monospace", ha="left", va="top")

DEFAULT_NEW_KIND = "object"
DEFAULT_NEW_RADIUS = 0.07

# -------------------------
# Draggable circle helper
# -------------------------

class DraggableCircle:
    def __init__(self, artist: Circle, radius: float, kind: str, name: str):
        self.artist = artist
        self.radius = radius
        self.kind = kind
        self.name = name
        self.press_offset = None

    def contains(self, event) -> bool:
        if event.inaxes != self.artist.axes:
            return False
        contains, _ = self.artist.contains(event)
        return contains

    def on_press(self, event):
        if not self.contains(event):
            return
        x0, y0 = self.artist.center
        self.press_offset = (x0 - event.xdata, y0 - event.ydata)

    def on_motion(self, event):
        if self.press_offset is None or event.inaxes != self.artist.axes:
            return
        dx, dy = self.press_offset
        new_x = event.xdata + dx
        new_y = event.ydata + dy
        r = self.radius
        new_x = np.clip(new_x, X_MIN + r, X_MAX - r)
        new_y = np.clip(new_y, Y_MIN + r, Y_MAX - r)
        self.artist.center = (new_x, new_y)

    def on_release(self, event):
        self.press_offset = None


# -------------------------
# GUI Application
# -------------------------

class CBFSafetyApp2D:
    def __init__(self):
        self.fig = plt.figure(figsize=(FIG_W, FIG_H))
        gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.2, 1.0])
        self.ax = self.fig.add_subplot(gs[0, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.fig.canvas.manager.set_window_title("Interactive CBF Safety (2D Top-Down)")

        self._setup_axes()
        self._setup_info_panel()

        self.objects = []

        self.workspace = Workspace(bounds=np.array([
            [X_MIN, X_MAX],
            [Y_MIN, Y_MAX],
            [Z_MIN, Z_MAX],
        ]))

        self.draggables: list[DraggableCircle] = []
        self.labels = {}

        self._draw_arena()

        self.add_mode = False
        self.policy_cache = None

        self.cid_press   = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_move    = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self.cid_key     = self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._update_policy_panel(fetch=True)
        self.update_metrics()

    # --- Setup / drawing ---

    def _setup_axes(self):
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(X_MIN, X_MAX)
        self.ax.set_ylim(Y_MIN, Y_MAX)
        self.ax.set_title("2D Workspace — press 'a' then click to add; drag to move; 'p' to refresh policy", **TITLE_KW)
        self.ax.set_facecolor(ARENA_FACE_COLOR)
        self.ax.grid(True, alpha=GRID_ALPHA)
        self.ax.add_patch(Rectangle((X_MIN, Y_MIN),
                                    X_MAX - X_MIN, Y_MAX - Y_MIN,
                                    fill=False, lw=2.0, ec=ARENA_EDGE_COLOR))

    def _setup_info_panel(self):
        self.ax_info.set_axis_off()
        # Score box
        self.score_box = Rectangle((0.05, 0.75), 0.90, 0.20, transform=self.ax_info.transAxes,
                                   ec="black", lw=1.5, fc=(0.95, 0.95, 0.95))
        self.ax_info.add_patch(self.score_box)
        self.txt_score_title = self.ax_info.text(0.50, 0.88, "Safety Score (0–5)",
                                                 transform=self.ax_info.transAxes,
                                                 ha="center", va="center",
                                                 fontsize=12, color="black", weight="bold")
        self.txt_score_value = self.ax_info.text(0.50, 0.79, "—",
                                                 transform=self.ax_info.transAxes,
                                                 ha="center", va="center",
                                                 fontsize=24, color="black", family="monospace")
        self.txt_info = self.ax_info.text(0.05, 0.70,
                                          "Add objects to see metrics…",
                                          transform=self.ax_info.transAxes,
                                          **INFO_KW)
        self.txt_policy_header = self.ax_info.text(0.05, 0.64, "Active Hazard Policy",
                                                   transform=self.ax_info.transAxes,
                                                   fontsize=12, color="black", weight="bold", ha="left", va="top")
        self.txt_policy = self.ax_info.text(0.05, 0.62, "(none yet)",
                                            transform=self.ax_info.transAxes,
                                            **INFO_KW)

    def _draw_arena(self):
        pass

    # --- Object management ---

    def _add_object_at_click(self, x: float, y: float):
        """Popup ask for a name, then add an object at (x,y) with default kind/radius."""
        name = self._ask_for_name()
        if not name:
            return
        # clamp into workspace with radius padding
        r = DEFAULT_NEW_RADIUS
        x = float(np.clip(x, X_MIN + r, X_MAX - r))
        y = float(np.clip(y, Y_MIN + r, Y_MAX - r))
        obj = dict(name=name, kind=DEFAULT_NEW_KIND, r=r, xy=(x, y), z=0.75)
        self.objects.append(obj)

        # draw
        style = OBJECT_STYLE.get(obj["kind"], OBJECT_STYLE["object"])
        circ = Circle(obj["xy"], obj["r"], **style)
        self.ax.add_patch(circ)
        dc = DraggableCircle(circ, obj["r"], obj["kind"], obj["name"])
        self.draggables.append(dc)
        label = self.ax.text(obj["xy"][0], obj["xy"][1], obj["name"], **LABEL_KW)
        self.labels[obj["name"]] = label

        # refresh policy and metrics
        self._update_policy_panel(fetch=True)
        self.update_metrics()
        self.fig.canvas.draw_idle()

    def _ask_for_name(self) -> str | None:
        """Open a simple popup dialog to ask object name. Returns None if canceled."""
        if not _TK_AVAILABLE:
            # Fallback: simple input in console if Tk not available
            try:
                return input("Enter object name: ").strip() or None
            except EOFError:
                return None

        # Tk dialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            name = simpledialog.askstring("Add Object", "Enter object name:", parent=root)
            if name:
                name = name.strip()
            return name or None
        finally:
            root.destroy()

    # --- Scene marshaling ---

    def _scene_from_artists(self) -> Scene:
        name_to_pos = {d.name: d.artist.center for d in self.draggables}
        objs = []
        for spec in self.objects:
            name = spec["name"]
            kind = spec["kind"]
            r = float(spec["r"])
            x, y = name_to_pos.get(name, spec["xy"])
            z = float(spec["z"])
            sphere = Sphere(center=np.array([x, y, z], dtype=float), radius=r)
            objs.append(ObjectState(name=name, sphere=sphere, kind=kind))
        return Scene(objects=objs, workspace=self.workspace)

    # --- Event handlers ---

    def _on_press(self, event):
        if event.inaxes != self.ax:
            return
        # Add-mode click → add object with popup name
        if getattr(self, "add_mode", False) and event.button == 1:
            self._add_object_at_click(float(event.xdata), float(event.ydata))
            # stay in add mode so user can add multiple; press 'a' again to toggle off
            return

        # Otherwise, forward to draggables for potential dragging
        for d in self.draggables:
            d.on_press(event)

    def _on_motion(self, event):
        if event.inaxes != self.ax:
            return
        any_move = False
        for d in self.draggables:
            before = d.artist.center
            d.on_motion(event)
            after = d.artist.center
            if before != after:
                self.labels[d.name].set_position(after)
                any_move = True
        if any_move:
            # Update metrics; also update policy (object positions can affect CBF metrics,
            # but hazard policy depends only on names/kinds/aliases; still okay to refresh)
            self.update_metrics()
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        for d in self.draggables:
            d.on_release(event)
        self.update_metrics()
        self.fig.canvas.draw_idle()

    def _on_key(self, event):
        if not event.key:
            return
        k = event.key.lower()

        # Toggle add mode
        if k == 'a':
            self.add_mode = not getattr(self, "add_mode", False)
            suffix = "ON" if self.add_mode else "OFF"
            self.ax.set_title(f"2D Workspace — add mode [{suffix}] — press 'a' to toggle; click to add; drag to move; 'p' to refresh policy", **TITLE_KW)
            self.fig.canvas.draw_idle()
            return

        # Force refresh policy
        if k == 'p':
            self._update_policy_panel(fetch=True)
            self.update_metrics()
            self.fig.canvas.draw_idle()
            return

        # Quit helpers
        if k in ('q', 'escape'):
            plt.close(self.fig)

    # --- Policy + metrics UI ---

    def _score_to_color(self, score_0_to_5: float):
        s = np.clip(score_0_to_5 / 5.0, 0.0, 1.0)
        r = (1.00 - s) * 0.98 + s * 0.85
        g = (1.00 - s) * 0.85 + s * 0.98
        b = (1.00 - s) * 0.85 + s * 0.90
        return (r, g, b)

    def _update_policy_panel(self, fetch: bool = False):
        """Fetch and render active hazard policy (LLM or fallback)."""
        if fetch:
            try:
                # Build a temporary Scene objects list for the policy call (names/kinds/radii)
                scene_objs = []
                for spec in self.objects:
                    sphere = Sphere(center=np.array([spec["xy"][0], spec["xy"][1], spec["z"]], dtype=float),
                                    radius=float(spec["r"]))
                    scene_objs.append(ObjectState(name=spec["name"], sphere=sphere, kind=spec["kind"]))
                self.policy_cache = get_hazard_policy_via_llm(scene_objs)
            except Exception as e:
                self.policy_cache = {"error": str(e)}

        # Render panel
        y_lines = []
        pol = self.policy_cache
        if pol is None:
            y_lines.append("(policy fetching…)")
        elif "error" in pol:
            y_lines.append("LLM policy error:")
            y_lines.append(str(pol["error"])[:120] + "…")
        else:
            pairs = pol.get("pairs", [])
            aliases = pol.get("aliases", {})
            y_lines.append(f"Pairs: {len(pairs)}  |  Aliases: {len(aliases)}")
            max_show = 8
            for i, pr in enumerate(pairs[:max_show]):
                a = pr.get("A_kind", "?"); b = pr.get("B_kind", "?")
                sev = pr.get("severity", "notice")
                clr = pr.get("clearance_m", 0.0)
                over = pr.get("over_rule", {}) or {}
                over_frag = ""
                if over.get("enable", False):
                    over_frag = f" (over: xy≤{over.get('xy_margin_m',0):.2f}, dz≤{over.get('z_gap_max_m',0):.2f})"
                y_lines.append(f"• {a} → {b}  [{sev}]  clr={clr:.2f}{over_frag}")
            if len(pairs) > max_show:
                y_lines.append(f"… and {len(pairs)-max_show} more")
            if aliases:
                shown = 0
                y_lines.append("")
                y_lines.append("Aliases:")
                for name, kinds in list(aliases.items())[:6]:
                    y_lines.append(f"  {name}: {', '.join(kinds)}")
                    shown += 1
                    if shown >= 6:
                        break

        self.txt_policy.set_text("\n".join(y_lines) if y_lines else "(no policy)")

    def update_metrics(self):
        # Evaluate metrics on current scene (positions + kinds)
        scene = self._scene_from_artists()
        out = evaluate_scene_metrics(scene)

        score = float(out.get("safety_score_0_to_5", out.get("safety_score", 0.0)))
        self.txt_score_value.set_text(f"{score:0.2f}")
        self.score_box.set_facecolor(self._score_to_color(score))

        lines = []
        lines.append(f"Composite risk: {out.get('composite_risk', 0.0):0.3f}")
        lines.append("")
        metrics = out.get("metrics", [])
        for m in metrics:
            name = m.get("name", "metric")
            risk = m.get("risk", 0.0)
            if name == "object_collision_cbf":
                lines.append(f"[Collision] risk={risk:0.3f}")
            elif name == "workspace_cbf_objects":
                lines.append(f"[Workspace] risk={risk:0.3f}")
            elif name == "hazard_pairings_cbf":
                lines.append(f"[Hazards]   risk={risk:0.3f}")
                if m.get("critical_violation", False):
                    # keep generic; specific criticals are shown by policy pairs (critical)
                    lines.append("  CRITICAL hazard detected!")
            else:
                lines.append(f"[{name}] risk={risk:0.3f}")
        self.txt_info.set_text("\n".join(lines))

        # Update the policy panel (policy depends on names/kinds; re-fetch only when object set changed)
        # Here we refresh display only; fetching is done after add, or via 'p' key.
        self._update_policy_panel(fetch=False)


def main():
    app = CBFSafetyApp2D()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
