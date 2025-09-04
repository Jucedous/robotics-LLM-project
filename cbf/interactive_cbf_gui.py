#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive CBF Safety Playground (2D Top-Down) — Objects Only
--------------------------------------------------------------
Drag the circles around and watch the CBF-based safety score update live.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

from cbf_safety_metrics import (
    ObjectState, Sphere, Workspace, Scene, evaluate_scene_metrics
)

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

        self.objects = [
            dict(name="laptop",     kind="electronic", r=0.10, xy=(0.50, 0.20), z=0.75),
            dict(name="water_cup",  kind="liquid",     r=0.06, xy=(0.20, 0.10), z=0.90),
            dict(name="knife",      kind="sharp",      r=0.02, xy=(0.10,-0.20), z=0.50),
            dict(name="human_1",    kind="human",      r=0.15, xy=(0.40, 0.00), z=0.50),
            dict(name="glass_vase", kind="fragile",    r=0.08, xy=(0.00, 0.55), z=0.60),
            dict(name="cast_pan",   kind="heavy",      r=0.09, xy=(-0.50,0.00), z=0.60),
        ]

        self.workspace = Workspace(bounds=np.array([
            [X_MIN, X_MAX],
            [Y_MIN, Y_MAX],
            [Z_MIN, Z_MAX],
        ]))

        self.draggables: list[DraggableCircle] = []
        self.labels = {}

        self._draw_arena()
        self._add_objects()

        self.cid_press = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self.cid_move  = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)

        self.update_metrics()

    def _setup_axes(self):
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_xlim(X_MIN, X_MAX)
        self.ax.set_ylim(Y_MIN, Y_MAX)
        self.ax.set_title("2D Workspace — drag objects to update safety", **TITLE_KW)
        self.ax.set_facecolor(ARENA_FACE_COLOR)
        self.ax.grid(True, alpha=GRID_ALPHA)
        self.ax.add_patch(Rectangle((X_MIN, Y_MIN),
                                    X_MAX - X_MIN, Y_MAX - Y_MIN,
                                    fill=False, lw=2.0, ec=ARENA_EDGE_COLOR))

    def _setup_info_panel(self):
        self.ax_info.set_axis_off()
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
                                          "Move objects to see metrics…",
                                          transform=self.ax_info.transAxes,
                                          **INFO_KW)

    def _draw_arena(self):
        pass

    def _add_objects(self):
        for obj in self.objects:
            kind = obj["kind"]
            xy = obj["xy"]
            r = obj["r"]
            style = OBJECT_STYLE.get(kind, OBJECT_STYLE["object"])
            circ = Circle(xy, r, **style)
            self.ax.add_patch(circ)
            dc = DraggableCircle(circ, r, kind, obj["name"])
            self.draggables.append(dc)
            label = self.ax.text(xy[0], xy[1], obj["name"], **LABEL_KW)
            self.labels[obj["name"]] = label

    def _on_press(self, event):
        for d in self.draggables:
            d.on_press(event)

    def _on_motion(self, event):
        any_move = False
        for d in self.draggables:
            before = d.artist.center
            d.on_motion(event)
            after = d.artist.center
            if before != after:
                self.labels[d.name].set_position(after)
                any_move = True
        if any_move:
            self.update_metrics()
            self.fig.canvas.draw_idle()

    def _on_release(self, event):
        for d in self.draggables:
            d.on_release(event)
        self.update_metrics()
        self.fig.canvas.draw_idle()

    def _scene_from_artists(self) -> Scene:
        name_to_pos = {d.name: d.artist.center for d in self.draggables}
        objs = []
        for spec in self.objects:
            name = spec["name"]
            kind = spec["kind"]
            r = float(spec["r"])
            x, y = name_to_pos[name]
            z = float(spec["z"])
            sphere = Sphere(center=np.array([x, y, z], dtype=float), radius=r)
            objs.append(ObjectState(name=name, sphere=sphere, kind=kind))
        return Scene(objects=objs, workspace=self.workspace)

    @staticmethod
    def _score_to_color(score_0_to_5: float):
        s = np.clip(score_0_to_5 / 5.0, 0.0, 1.0)
        r = (1.00 - s) * 0.98 + s * 0.85
        g = (1.00 - s) * 0.85 + s * 0.98
        b = (1.00 - s) * 0.85 + s * 0.90
        return (r, g, b)

    def update_metrics(self):
        scene = self._scene_from_artists()
        out = evaluate_scene_metrics(scene)

        score = float(out.get("safety_score_0_to_5", 0.0))
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
                    lines.append("  CRITICAL: liquid–electronics hazard detected!")
            else:
                lines.append(f"[{name}] risk={risk:0.3f}")

        info_txt = "\n".join(lines)
        self.txt_info.set_text(info_txt)


def main():
    app = CBFSafetyApp2D()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
