from __future__ import annotations
from typing import List, Optional, Callable
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from cbf.cbf_safety_metrics import ObjectState, Sphere

# Try to use styling from tools; otherwise fallback to a default theme.
try:
    from cbf.tools.ui_theme import DEFAULT_OBJECT_STYLE
except Exception:
    DEFAULT_OBJECT_STYLE = {
        "object": dict(ec="black", lw=1.5, fc=(0.88, 0.88, 0.95), alpha=0.9)
    }


def style_for(obj: ObjectState):
    """Return a matplotlib patch style dict for the object kind (fallback to 'object')."""
    return DEFAULT_OBJECT_STYLE.get(
        obj.kind,
        DEFAULT_OBJECT_STYLE.get("object", dict(ec="black", lw=1.5, fc=(0.88, 0.88, 0.95), alpha=0.9)),
    )


def set_arena_bounds(ax: plt.Axes, objects: List[ObjectState]) -> None:
    """Set nice XY limits around current objects, with padding."""
    if not objects:
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        return
    xs = [o.sphere.center[0] for o in objects]
    ys = [o.sphere.center[1] for o in objects]
    rs = [o.sphere.radius for o in objects]
    pad = (max(rs) if rs else 0.1) * 3.0
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)


class DraggableCircle:
    """
    A draggable circle that keeps a text label locked to its center and
    calls 'on_change()' during drag & release (to recompute safety).
    """
    def __init__(self, ax: plt.Axes, obj: ObjectState, on_change: Optional[Callable[[], None]] = None):
        self.ax = ax
        self.obj = obj
        self.on_change = on_change

        x, y = float(obj.sphere.center[0]), float(obj.sphere.center[1])
        self.artist = Circle((x, y), radius=float(obj.sphere.radius), **style_for(obj))
        self.ax.add_patch(self.artist)

        self.label = self.ax.text(
            x, y, obj.name, ha="center", va="center", fontsize=9, clip_on=True
        )

        self._press = None
        fig = self.artist.figure
        self._cids = [
            fig.canvas.mpl_connect("button_press_event", self._on_press),
            fig.canvas.mpl_connect("button_release_event", self._on_release),
            fig.canvas.mpl_connect("motion_notify_event", self._on_motion),
        ]

    # -------- events --------
    def _contains(self, event) -> bool:
        return bool(self.artist.contains(event)[0])

    def _on_press(self, event):
        if event.inaxes is not self.ax or not self._contains(event):
            return
        x0, y0 = self.artist.center
        self._press = (x0, y0, event.xdata, event.ydata)

    def _on_motion(self, event):
        if self._press is None or event.inaxes is not self.ax:
            return
        x0, y0, xpress, ypress = self._press
        dx, dy = event.xdata - xpress, event.ydata - ypress
        nx, ny = x0 + dx, y0 + dy

        # visuals
        self.artist.center = (nx, ny)
        self.label.set_position((nx, ny))

        # state (z unchanged)
        self.obj.sphere.center[0] = float(nx)
        self.obj.sphere.center[1] = float(ny)

        if callable(self.on_change):
            self.on_change()
        self.artist.figure.canvas.draw_idle()

    def _on_release(self, event):
        if self._press is None:
            return
        self._press = None
        if callable(self.on_change):
            self.on_change()
        self.artist.figure.canvas.draw_idle()

    # -------- helpers --------
    def set_position(self, x: float, y: float):
        self.artist.center = (float(x), float(y))
        self.label.set_position((float(x), float(y)))

    def remove(self):
        fig = self.artist.figure
        for cid in self._cids:
            try:
                fig.canvas.mpl_disconnect(cid)
            except Exception:
                pass
        try:
            self.artist.remove()
        except Exception:
            pass
        try:
            self.label.remove()
        except Exception:
            pass
