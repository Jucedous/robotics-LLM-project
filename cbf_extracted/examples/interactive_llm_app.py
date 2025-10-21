import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

from cbf.semantics_runtime import (
    LLMConfig,
    analyze_scene_llm,
    instantiate_rules,
    classify_object_kind_llm,
)
from cbf.cbf_safety_metrics_llm import metric_hazard_pairings_cbf_objects_llm
from cbf.cbf_safety_metrics import ObjectState, Sphere
from cbf.tools.ui_helpers import DraggableCircle, set_arena_bounds

def load_scene(path: str):
    items = json.loads(Path(path).read_text())
    objs = []
    for it in items:
        x, y = it.get("xy", [0.0, 0.0])
        z = it.get("z", 0.0)
        r = float(it.get("r", 0.05))
        objs.append(
            ObjectState(
                name=str(it["name"]),
                kind=str(it.get("kind", "object")).strip(),
                sphere=Sphere(center=np.array([x, y, z], dtype=float), radius=r),
                tags=tuple(it.get("tags", [])),
            )
        )
    return objs


def to_llm_payload(objs):
    return [
        dict(
            name=o.name,
            kind=o.kind,
            tags=list(o.tags),
            xyz=[float(o.sphere.center[0]), float(o.sphere.center[1]), float(o.sphere.center[2])],
            r=float(o.sphere.radius),
        )
        for o in objs
    ]

class InteractiveLLMApp:
    def __init__(self, scene_path: str):
        self.objects = load_scene(scene_path)

        self.alpha_gain = 5.0
        self.scale_res = 0.05

        self.scene_version = 0
        self.rules_version = -1
        self._last_out = None

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            keyfile = _ROOT / "config" / "openai_key.txt"
            if keyfile.exists():
                api_key = keyfile.read_text().strip()
        if not api_key:
            raise SystemExit("Missing OPENAI_API_KEY (env) or config/openai_key.txt")
        self.cfg = LLMConfig(api_key=api_key)

        self.fig = plt.figure(figsize=(10.8, 8.2))
        gs = self.fig.add_gridspec(
            nrows=7, ncols=2,
            width_ratios=[3, 2],
            height_ratios=[24, 1, 1, 1, 1, 1, 1]
        )

        self.ax = self.fig.add_subplot(gs[0, 0])
        set_arena_bounds(self.ax, self.objects)
        self.ax.set_aspect("equal", adjustable="box")
        self.ax.set_title("Interactive LLM-Driven Safety (drag circles)")

        self.ax_info = self.fig.add_subplot(gs[0, 1]); self.ax_info.axis("off")

        self.ax_btn_requery = self.fig.add_subplot(gs[1, 1])
        self.ax_btn_reset   = self.fig.add_subplot(gs[2, 1])
        self.ax_add_box     = self.fig.add_subplot(gs[3, 1])
        self.ax_btn_add     = self.fig.add_subplot(gs[4, 1])
        self.ax_rm_box      = self.fig.add_subplot(gs[5, 1])
        self.ax_btn_close   = self.fig.add_subplot(gs[6, 1])

        self.btn_requery = Button(self.ax_btn_requery, label="Requery LLM")
        self.btn_reset   = Button(self.ax_btn_reset,   label="Reset Scene")
        self.btn_close   = Button(self.ax_btn_close,   label="Close Window")

        self.tb_add = TextBox(self.ax_add_box, label="Add name: ", initial="")
        self.btn_add = Button(self.ax_btn_add, label="Add")
        self.tb_rm = TextBox(self.ax_rm_box, label="Remove name (Enter): ", initial="")

        self.btn_requery.on_clicked(self.on_requery_llm)
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_close.on_clicked(self.on_close)
        self.btn_add.on_clicked(self.on_add_object)
        self.tb_add.on_submit(lambda _t: self.on_add_object(None))
        self.tb_rm.on_submit(lambda _t: self.on_remove_object())

        self.initial_positions = {o.name: o.sphere.center.copy() for o in self.objects}
        self.draggables: list[DraggableCircle] = []
        self._draw_all()

        self.update_info_panel()

    def _classify_all_kinds(self):
        """Classify kind for every object (name → kind), in-place."""
        for o in self.objects:
            try:
                k = classify_object_kind_llm(o.name, list(o.tags), self.cfg)
                o.kind = k or "object"
            except Exception as e:
                print(f"[Classify] {o.name}: {e} -> 'object'")
                o.kind = "object"

    def _rebuild_semantics(self):
        """Ask the LLM for semantic hazards for the CURRENT objects and bind rules."""
        risks = analyze_scene_llm(to_llm_payload(self.objects), self.cfg)
        self.rules, self.crit_map = instantiate_rules(self.objects, risks)

    def _compute_safety_now(self):
        """Compute metrics against current objects and current rules."""
        return metric_hazard_pairings_cbf_objects_llm(
            objects=self.objects,
            alpha_gain=self.alpha_gain,
            scale_res=self.scale_res,
            rules=self.rules,
            critical_by_pair=self.crit_map,
            include_collision_baseline=True,
            baseline_weight=0.25,
            baseline_clearance_m=0.0,
            show_collision_for_labeled_pairs=True,
            collision_visual_only=True,
            label_semantic_entries=True,
        )

    def _clear_draggables(self):
        for d in self.draggables:
            d.remove()
        self.draggables.clear()

    def _draw_all(self):
        self._clear_draggables()
        for o in self.objects:
            self.draggables.append(DraggableCircle(self.ax, o, on_change=self.update_info_panel))
        self.fig.canvas.draw_idle()


    def on_close(self, _evt):
        """Close the interactive window."""
        try:
            plt.close(self.fig)
        except Exception:
            pass

    def on_reset(self, _evt):
        for o in self.objects:
            if o.name in self.initial_positions:
                orig = self.initial_positions[o.name]
                o.sphere.center[0], o.sphere.center[1] = float(orig[0]), float(orig[1])
        for d in self.draggables:
            x, y = d.obj.sphere.center[0], d.obj.sphere.center[1]
            d.set_position(x, y)
        self.update_info_panel()

    def on_requery_llm(self, _evt):
        self._classify_all_kinds()
        self._rebuild_semantics()
        self.rules_version = self.scene_version
        self.update_info_panel()

    def on_add_object(self, _evt):
        name = self.tb_add.text.strip()
        if not name:
            return
        if any(o.name.lower() == name.lower() for o in self.objects):
            print(f"[Add] '{name}' already exists.")
            return

        obj = ObjectState(
            name=name,
            kind="object",
            sphere=Sphere(center=np.array([0.0, 0.0, 0.0], dtype=float), radius=0.05),
            tags=(),
        )
        self.objects.append(obj)
        self.initial_positions[name] = obj.sphere.center.copy()

        set_arena_bounds(self.ax, self.objects)
        self.draggables.append(DraggableCircle(self.ax, obj, on_change=self.update_info_panel))

        self.scene_version += 1
        self.update_info_panel()
        self.tb_add.set_val("")

    def on_remove_object(self):
        name = self.tb_rm.text.strip()
        if not name:
            return
        idx = next((i for i, o in enumerate(self.objects) if o.name.lower() == name.lower()), None)
        if idx is None:
            print(f"[Remove] No object named '{name}'.")
            return

        removed = self.objects.pop(idx)
        self.initial_positions.pop(removed.name, None)
        d_idx = next((i for i, d in enumerate(self.draggables) if d.obj.name.lower() == name.lower()), None)
        if d_idx is not None:
            self.draggables[d_idx].remove()
            self.draggables.pop(d_idx)

        set_arena_bounds(self.ax, self.objects)

        self.scene_version += 1
        self.update_info_panel()
        self.tb_rm.set_val("")

    def update_info_panel(self):
        scene_in_sync = (self.scene_version == self.rules_version)

        if scene_in_sync:
            out = self._compute_safety_now()
            self._last_out = out
        else:
            out = self._last_out or {
                "name": "hazard_pairings_llm",
                "metrics": [],
                "residual_min": float("inf"),
                "composite_risk": 0.0,
                "critical_violation": False,
                "critical_pairs": [],
                "safety_score_0_to_5": 5.0,
            }

        score = float(out.get("safety_score_0_to_5", 0.0))
        crit = bool(out.get("critical_violation", False))
        title = f"Safety Score: {score:.2f} / 5.00"
        if crit:
            title += "  —  CRITICAL ⚠️"
        if not scene_in_sync:
            title += "  —  (scene changed; press Requery LLM)"
        self.ax.set_title(title)

        for d in self.draggables:
            cx, cy = d.artist.center
            d.label.set_position((cx, cy))

        self.ax_info.clear()
        self.ax_info.axis("off")

        rules_lines = ["Rules (LLM):"]
        if scene_in_sync:
            if not getattr(self, "rules", []):
                rules_lines.append(" • (none)")
            else:
                for (A_kind, B_kind, clr, w, A_name, B_name) in self.rules:
                    rules_lines.append(f" • {A_name}({A_kind}) → {B_name}({B_kind}) | clr={clr:.3f}m, w={w:.2f}")
        else:
            rules_lines.append(" • (stale; press Requery LLM)")

        metrics = out.get("metrics", [])
        hazards = [m for m in metrics if m.get("channel") == "semantic"]
        collisions = [m for m in metrics if m.get("channel") == "collision" and not m.get("diagnostic_only", False)]
        collisions_visual = [m for m in metrics if m.get("channel") == "collision" and m.get("diagnostic_only", False)]

        haz_lines = [f"Hazard (semantic) pairs [{len(hazards)}]:"]
        for m in hazards:
            haz_lines.append(f" • {m['name']}: risk={m['risk']:.3g}, w={m.get('weight',1.0):.2f}")

        col_lines = [f"Collision baseline (aggregating) [{len(collisions)}]:"]
        for m in collisions:
            col_lines.append(f" • {m['name']}: risk={m['risk']:.3g}, w={m.get('weight',1.0):.2f}")

        vis_lines = []
        if collisions_visual:
            vis_lines = [f"Collision (visual for LLM-labeled) [{len(collisions_visual)}]:"]
            for m in collisions_visual:
                vis_lines.append(f" • {m['name']}: risk={m['risk']:.3g}, w={m.get('weight',1.0):.2f}")

        summary = (
            f"residual_min: {out.get('residual_min', float('inf')):.3f}\n"
            f"composite_risk: {out.get('composite_risk', 0.0):.3g}"
        )

        text = "\n".join(
            rules_lines + [""] + haz_lines + [""] + col_lines + ([""] + vis_lines if vis_lines else []) + ["", summary]
        )
        self.ax_info.text(0.02, 0.98, text, ha="left", va="top", family="monospace", fontsize=9)

        self.fig.canvas.draw_idle()


def main(scene_path: str):
    plt.show = plt.show
    app = InteractiveLLMApp(scene_path)
    plt.show(block=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive LLM-driven safety explorer")
    parser.add_argument("scene", nargs="?", default=str(_THIS.parent / "scene1.json"), help="Path to scene JSON")
    args = parser.parse_args()
    main(args.scene)
