"""
CBFSafetyApp2D — Environment-only GUI (no CBF logic)
Robust version: guards against bad specs (None / malformed) and logs skips.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from .ui_panels import setup_axes, setup_info_panel, setup_add_remove_controls
from .draggable_circle import DraggableCircle
from .ui_theme import LABEL_KW, TITLE_KW, INFO_KW, ARENA_FACE_COLOR, ARENA_EDGE_COLOR, GRID_ALPHA, DEFAULT_OBJECT_STYLE
from .ui_colors import get_score_color
from .scene_schema import normalize_spec
from .scene_io import load_specs


from .llm_kind_classifier import classify_kind_via_llm


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

        self.OBJECT_STYLE = object_style or DEFAULT_OBJECT_STYLE
        if "unknown" not in self.OBJECT_STYLE:
            self.OBJECT_STYLE["unknown"] = self.OBJECT_STYLE.get("object", {})
        self.LABEL_KW = dict(LABEL_KW)
        self.TITLE_KW = dict(TITLE_KW)
        self.INFO_KW = dict(INFO_KW)
        self.ARENA_FACE_COLOR = ARENA_FACE_COLOR
        self.ARENA_EDGE_COLOR = ARENA_EDGE_COLOR
        self.GRID_ALPHA = GRID_ALPHA

        # self.fig = plt.figure(figsize=fig_size)
        # self._toolbar_off()
        # gs = self.fig.add_gridspec(nrows=1, ncols=2, width_ratios=[2.2, 1.0])
        # self.ax = self.fig.add_subplot(gs[0, 0])
        # self.ax_info = self.fig.add_subplot(gs[0, 1])

        ui_scale = float(os.getenv("CBF_UI_SCALE", "1.25"))
        self.fig = plt.figure(figsize=(12 * ui_scale, 7 * ui_scale), constrained_layout=False)
        self._toolbar_off()

        gs = self.fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[2.4, 1.3],
            height_ratios=[1.0, 0.40],
            wspace=0.28,
            hspace=0.28,
            left=0.1,
            right=0.9,
            top=0.92,
            bottom=0.12,
        )

        self.ax = self.fig.add_subplot(gs[:, 0])
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_ctrl = self.fig.add_subplot(gs[1, 1])

        try:
            self.fig.canvas.manager.set_window_title("Interactive CBF Safety (2D) — Environment")
        except Exception:
            pass

        setup_axes(
            self.ax,
            xlim=(self.X_MIN, self.X_MAX),
            ylim=(self.Y_MIN, self.Y_MAX),
            title="2D Workspace — drag objects to update safety",
            arena_face_color=self.ARENA_FACE_COLOR,
            arena_edge_color=self.ARENA_EDGE_COLOR,
            grid_alpha=self.GRID_ALPHA,
        )

        info = setup_info_panel(self.fig, self.ax_info)
        self._score_box = info["score_box"]
        self._txt_score_value = info["txt_score_value"]
        self._txt_info = info["txt_info"]
        self._btn_close = info["btn_close"]
        self._btn_close.on_clicked(self._on_close)

        (self._tb_name, self._tb_kind,
         self._btn_add_obj, self._btn_del_obj) = setup_add_remove_controls(self.fig, self.ax_ctrl)
        self._btn_add_obj.on_clicked(self._on_add_object_click)
        self._btn_del_obj.on_clicked(self._on_remove_object_click)

        self._objects: List[Dict[str, Any]] = []
        self._draggables: list[DraggableCircle] = []
        self._labels: Dict[str, Any] = {}
        self._on_change: Optional[Callable[[], None]] = None

        self._cid_press   = self.fig.canvas.mpl_connect("button_press_event", self._on_press)
        self._cid_move    = self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self._cid_release = self.fig.canvas.mpl_connect("button_release_event", self._on_release)

    # Public API
    def set_on_change(self, fn: Optional[Callable[[], None]]):
        self._on_change = fn

    def clear_objects(self):
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
        specs = load_specs(path)
        self.clear_objects()
        self.add_objects_from_specs(specs)

    def add_objects_from_specs(self, specs: List[Dict[str, Any]]):
        normed: List[Dict[str, Any]] = []
        for i, spec in enumerate(specs):
            try:
                fixed = normalize_spec(
                    i, spec,
                    xlim=(self.X_MIN, self.X_MAX),
                    ylim=(self.Y_MIN, self.Y_MAX),
                    zlim=(self.Z_MIN, self.Z_MAX),
                    known_kinds=tuple(self.OBJECT_STYLE.keys()),
                    default_kind="object",
                )
                if fixed is None:
                    raise ValueError("normalize_spec returned None")
                normed.append(fixed)
            except Exception as e:
                print(f"[CBFSafetyApp2D] Skipping bad spec #{i}: {e}. Raw: {spec}")
                continue

        for spec in normed:
            try:
                self._add_one(spec)
            except Exception as e:
                print(f"[CBFSafetyApp2D] Failed to add spec {spec.get('name','?')}: {e}")

        self._objects = normed
        self._redraw()
        self._notify_change()

    def get_current_specs(self) -> List[Dict[str, Any]]:
        centers = {d.name: d.artist.center for d in self._draggables}
        out: List[Dict[str, Any]] = []
        for spec in self._objects:
            name = spec["name"]
            x, y = centers.get(name, (spec["xy"][0], spec["xy"][1]))
            out.append({**spec, "xy": (float(x), float(y))})
        return out

    def set_score(self, score_0_to_5: float):
        color = get_score_color(score_0_to_5)
        self._txt_score_value.set_text(f"{float(score_0_to_5):0.2f}")
        self._score_box.set_facecolor(color)
        self._redraw_lazy()

    def set_info_lines(self, lines: List[str]):
        if not hasattr(self, "_txt_info") or self._txt_info is None:
            self._txt_info = self.ax_info.text(0.05, 0.70, "",
                                               transform=self.ax_info.transAxes,
                                               **self.INFO_KW)
        self._txt_info.set_text("\n".join(lines) if lines else "")
        self._redraw_lazy()

    def show(self):
        plt.show()

    # Internals
    def _add_one(self, spec: Dict[str, Any]):
        for k in ("kind", "xy", "r", "name"):
            if k not in spec:
                raise KeyError(f"spec missing key '{k}': {spec}")
        kind, xy, r = spec["kind"], spec["xy"], spec["r"]
        style = self.OBJECT_STYLE.get(kind, self.OBJECT_STYLE["object"])
        circ = Circle(xy, r, **style)
        self.ax.add_patch(circ)
        dc = DraggableCircle(circ, r, kind, spec["name"])
        self._draggables.append(dc)
        lbl = self.ax.text(xy[0], xy[1], spec["name"], **self.LABEL_KW)
        self._labels[spec["name"]] = lbl

    def _on_press(self, event):
        self._toolbar_off()
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

    def _on_add_object_click(self, _evt):
        try:
            name_in = self._tb_name.text.strip() or "obj"
            kind_in = (self._tb_kind.text or "").strip().lower()

            if not kind_in:
                # what the LLM is allowed to pick from:
                all_kinds = tuple(self.OBJECT_STYLE.keys())
                allowed_for_llm = [k for k in all_kinds if k not in ("object",)]
                # ensure unknown renders, but don't *offer* "object" to the LLM
                if "unknown" not in self.OBJECT_STYLE:
                    self.OBJECT_STYLE["unknown"] = self.OBJECT_STYLE.get("object", {})
                kind_in = classify_kind_via_llm(
                    name=name_in,
                    allowed_kinds=allowed_for_llm + ["unknown"],  # keep unknown as explicit fallback
                    description="",  # (optional: add a Description field later)
            )


            spec = dict(
                name=name_in,
                kind=kind_in,
                xy=(float(0.0), float(0.0)),
                z=float(0.6),
                r=float(0.08),
            )

            # include 'unknown' in known_kinds so normalize_spec doesn't force 'object'
            known = tuple(set(self.OBJECT_STYLE.keys()) | {"unknown"})
            spec = normalize_spec(
                len(self._objects), spec,
                xlim=(self.X_MIN, self.X_MAX),
                ylim=(self.Y_MIN, self.Y_MAX),
                zlim=(self.Z_MIN, self.Z_MAX),
                known_kinds=known,
                default_kind="object",
            )

            if any(o["name"] == spec["name"] for o in self._objects):
                spec["name"] = f'{spec["name"]}_{len(self._objects)}'
            self._add_one(spec)
            self._objects.append(spec)
            self._notify_change()
            self._redraw_lazy()
        except Exception as e:
            print("[CBFSafetyApp2D] add object failed:", e)

        self._tb_name.set_val("")
        self._tb_kind.set_val("")

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

        self._tb_name.set_val("")
        self._tb_kind.set_val("")

    def _on_close(self, _evt):
        plt.close(self.fig)

    def _toolbar_off(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        if not tb:
            return
        try:
            tb._active = None
            tb.mode = ""
            self.fig.canvas.draw_idle()
        except Exception:
            pass

