import sys, os, json, re, getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cbf.semantics_runtime import (
    LLMConfig,
    analyze_scene_llm,
    instantiate_rules,
    classify_object_kind_llm,
)
from cbf.cbf_safety_metrics_llm import metric_hazard_pairings_cbf_objects_llm
from cbf.cbf_safety_metrics import ObjectState, Sphere
from cbf.tools.ui_helpers import DraggableCircle, set_arena_bounds

from cbf.preferences import PreferenceStore
from cbf.feedback_pipeline import label_and_store_feedback


def load_scene(path: str):
    items = json.loads(Path(path).read_text())
    objs = []
    for it in items:
        x, y = it.get("xy", [0.0, 0.0]); z = it.get("z", 0.0); r = float(it.get("r", 0.05))
        objs.append(ObjectState(
            name=str(it["name"]),
            kind=str(it.get("kind", "object")).strip(),
            sphere=Sphere(center=np.array([x, y, z], dtype=float), radius=r),
            tags=tuple(it.get("tags", [])),
        ))
    return objs

def to_llm_payload(objs):
    return [
        dict(
            name=o.name, kind=o.kind, tags=list(o.tags),
            xyz=[float(o.sphere.center[0]), float(o.sphere.center[1]), float(o.sphere.center[2])],
            r=float(o.sphere.radius)
        )
        for o in objs
    ]


def _match_selector(obj: ObjectState, sel: Dict[str, str]) -> bool:
    by, val = sel.get("by"), sel.get("value")
    if by == "kind": return obj.kind == val
    if by == "name": return obj.name == val
    if by == "tag":  return val in obj.tags
    return False

def _cond_ok(expr: Optional[str], A: ObjectState, B: ObjectState) -> bool:
    if not expr: return True
    Az = float(A.sphere.center[2]); Bz = float(B.sphere.center[2])
    if expr.strip() == "Az > Bz": return Az > Bz
    if expr.strip() == "Az < Bz": return Az < Bz
    return True

def enforce_user_preferences_on_instantiated_rules(
    objects: List[ObjectState],
    rules: List[Tuple[str, str, float, float, str, str]],
    critical_by_pair: Dict[Tuple[str, str], Any],
    user_rules: List[Dict[str, Any]],
) -> Tuple[List[Tuple[str, str, float, float, str, str]], Dict[Tuple[str, str], Any]]:
    if not user_rules:
        return rules, critical_by_pair
    name2obj = {o.name: o for o in objects}
    new_rules: List[Tuple[str, str, float, float, str, str]] = []
    for (Ak, Bk, clr, w, Aname, Bname) in rules:
        A, B = name2obj.get(Aname), name2obj.get(Bname)
        if A is None or B is None:
            new_rules.append((Ak,Bk,clr,w,Aname,Bname))
            continue
        drop = False; clr2, w2 = clr, w
        for R in user_rules:
            selA = R.get("selectors",{}).get("A"); selB = R.get("selectors",{}).get("B")
            if not selA or not selB: continue
            present = R.get("override",{}).get("present", None)
            directional = bool(R.get("directional", False))
            cond = R.get("condition_expr") or R.get("relation")
            def applies(X,Y):
                return _match_selector(X, selA) and _match_selector(Y, selB) and _cond_ok(cond, X, Y)
            matched = applies(A,B) or (not directional and applies(B,A))
            if not matched: continue
            if present is False:
                drop = True
                if isinstance(critical_by_pair, dict):
                    critical_by_pair.pop((Aname,Bname), None)
                    critical_by_pair.pop((Bname,Aname), None)
                break
            elif present is True:
                if "soft_clearance_m" in R.get("override", {}):
                    clr2 = float(R["override"]["soft_clearance_m"])
                if "weight" in R.get("override", {}):
                    w2 = float(R["override"]["weight"])
        if not drop:
            new_rules.append((Ak,Bk,clr2,w2,Aname,Bname))
    existing = {(Aname,Bname) for (_,_,_,_,Aname,Bname) in new_rules}
    for R in user_rules:
        if R.get("override",{}).get("present") is not True:
            continue
        selA = R.get("selectors",{}).get("A"); selB = R.get("selectors",{}).get("B")
        if not selA or not selB: continue
        directional = bool(R.get("directional", False))
        cond = R.get("condition_expr") or R.get("relation")
        clr_new = float(R.get("override",{}).get("soft_clearance_m", 0.0))
        w_new   = float(R.get("override",{}).get("weight", 1.0))
        for A in objects:
            if not _match_selector(A, selA): continue
            for B in objects:
                if A is B: continue
                if not _match_selector(B, selB): continue
                if directional and not _cond_ok(cond, A, B): continue
                if (A.name, B.name) not in existing:
                    new_rules.append((A.kind, B.kind, clr_new, w_new, A.name, B.name))
                    existing.add((A.name,B.name))
    return new_rules, critical_by_pair


class InteractiveLLMApp:
    def __init__(self, scene_path: str):
        self.objects = load_scene(scene_path)
        self.alpha_gain = 5.0
        self.scale_res  = 0.05
        self.scene_version = 0
        self.rules_version = -1
        self._last_out = None
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            keyfile = _ROOT / "config" / "openai_key.txt"
            if keyfile.exists():
                raw = keyfile.read_text().strip()
                api_key = raw.split("=",1)[1].strip().strip('"').strip("'") if "OPENAI_API_KEY" in raw else raw
        if not api_key:
            raise SystemExit("Missing OPENAI_API_KEY (env) or config/openai_key.txt")
        self.cfg = LLMConfig(api_key=api_key)
        self.fig = plt.figure(figsize=(15.2, 9.6))
        self.fig.subplots_adjust(left=0.06, right=0.97, bottom=0.08, top=0.86)
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
        self.user_id = getpass.getuser()
        self.store = PreferenceStore()
        self.ax_fb_box   = self.fig.add_axes([0.06, 0.90, 0.58, 0.055])
        self.tb_feedback = TextBox(self.ax_fb_box, label="Feedback: ", initial="")
        self.ax_fb_submit = self.fig.add_axes([0.66, 0.90, 0.16, 0.055])
        self.btn_fb_submit = Button(self.ax_fb_submit, label="Submit & Requery")
        self.ax_fb_clear  = self.fig.add_axes([0.84, 0.90, 0.12, 0.055])
        self.btn_fb_clear = Button(self.ax_fb_clear, label="Clear My Rules")
        self.btn_fb_submit.on_clicked(self.on_submit_feedback)
        self.tb_feedback.on_submit(lambda _t: self.on_submit_feedback(None))
        self.btn_fb_clear.on_clicked(self.on_clear_rules)
        self.initial_positions = {o.name: o.sphere.center.copy() for o in self.objects}
        self.draggables: List[DraggableCircle] = []
        self._draw_all()
        self.update_info_panel()

    def _classify_all_kinds(self):
        for o in self.objects:
            try:
                k = classify_object_kind_llm(o.name, list(o.tags), self.cfg)
                o.kind = k or "object"
            except Exception as e:
                print(f"[Classify] {o.name}: {e} -> 'object'")
                o.kind = "object"

    def _rebuild_semantics(self):
        risks = analyze_scene_llm(to_llm_payload(self.objects), self.cfg)
        self.rules, self.crit_map = instantiate_rules(self.objects, risks)
        user_rules = self.store.list_rules(self.user_id)
        self.rules, self.crit_map = enforce_user_preferences_on_instantiated_rules(
            self.objects, self.rules, self.crit_map, user_rules
        )

    def _compute_safety_now(self):
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

    def _draw_all(self):
        for d in getattr(self, "draggables", []):
            d.remove()
        self.draggables = []
        for o in self.objects:
            self.draggables.append(DraggableCircle(self.ax, o, on_change=self.update_info_panel))
        self.fig.canvas.draw_idle()

    def on_close(self, _evt):
        try: plt.close(self.fig)
        except Exception: pass

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
        if not name: return
        if any(o.name.lower() == name.lower() for o in self.objects):
            print(f"[Add] '{name}' already exists."); return
        obj = ObjectState(name=name, kind="object", sphere=Sphere(center=np.array([0,0,0],dtype=float), radius=0.05), tags=())
        self.objects.append(obj)
        self.initial_positions[name] = obj.sphere.center.copy()
        set_arena_bounds(self.ax, self.objects)
        self.draggables.append(DraggableCircle(self.ax, obj, on_change=self.update_info_panel))
        self.scene_version += 1
        self.update_info_panel()
        self.tb_add.set_val("")

    def on_remove_object(self):
        name = self.tb_rm.text.strip()
        if not name: return
        idx = next((i for i,o in enumerate(self.objects) if o.name.lower()==name.lower()), None)
        if idx is None:
            print(f"[Remove] No object named '{name}'."); return
        removed = self.objects.pop(idx)
        self.initial_positions.pop(removed.name, None)
        di = next((i for i,d in enumerate(self.draggables) if d.obj.name.lower()==name.lower()), None)
        if di is not None:
            self.draggables[di].remove(); self.draggables.pop(di)
        set_arena_bounds(self.ax, self.objects)
        self.scene_version += 1
        self.update_info_panel()
        self.tb_rm.set_val("")

    def on_submit_feedback(self, _evt):
        text = self.tb_feedback.text.strip()
        if not text:
            print("[Feedback] empty input -> ignored"); return
        try:
            n = label_and_store_feedback(text, self.objects, self.cfg, self.user_id, store=self.store)
            print(f"[Feedback] stored {n} rule(s) for user '{self.user_id}'.")
            self.tb_feedback.set_val("")
            self.on_requery_llm(None)
        except Exception as e:
            print(f"[Feedback] labeling failed: {e}")

    def on_clear_rules(self, _evt):
        n = self.store.clear_user(self.user_id)
        print(f"[Feedback] cleared {n} saved rule(s) for user '{self.user_id}'.")
        self.update_info_panel()

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
        if crit: title += "  —  CRITICAL ⚠️"
        if not scene_in_sync: title += "  —  (scene changed; press Requery LLM)"
        self.ax.set_title(title)
        for d in self.draggables:
            cx, cy = d.artist.center
            d.label.set_position((cx, cy))
        self.ax_info.clear(); self.ax_info.axis("off")
        rules_lines = ["Rules (LLM + prefs):"]
        if scene_in_sync:
            if not getattr(self, "rules", []):
                rules_lines.append(" • (none)")
            else:
                for (Ak,Bk,clr,w,Aname,Bname) in self.rules:
                    rules_lines.append(f" • {Aname}({Ak}) → {Bname}({Bk}) | clr={clr:.3f}m, w={w:.2f}")
        else:
            rules_lines.append(" • (stale; press Requery LLM)")
        metrics = out.get("metrics", [])
        hazards = [m for m in metrics if m.get("channel")=="semantic"]
        collisions = [m for m in metrics if m.get("channel")=="collision" and not m.get("diagnostic_only", False)]
        visual = [m for m in metrics if m.get("channel")=="collision" and m.get("diagnostic_only", False)]
        lines = []
        lines += [f"Hazard (semantic) [{len(hazards)}]:"] + [f" • {m['name']}: risk={m['risk']:.3g}" for m in hazards]
        lines += ["", f"Collision baseline [{len(collisions)}]:"] + [f" • {m['name']}: risk={m['risk']:.3g}" for m in collisions]
        if visual:
            lines += ["", f"Collision (visual) [{len(visual)}]:"] + [f" • {m['name']}: risk={m['risk']:.3g}" for m in visual]
        summary = f"\nresidual_min: {out.get('residual_min', float('inf')):.3f}\ncomposite_risk: {out.get('composite_risk',0.0):.3g}"
        text = "\n".join(rules_lines + [""] + lines + ["", summary])
        self.ax_info.text(0.02, 0.98, text, ha="left", va="top", family="monospace", fontsize=9)
        self.fig.canvas.draw_idle()


def main(scene_path: str):
    app = InteractiveLLMApp(scene_path)
    plt.show(block=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Interactive LLM safety (with LLM-based feedback pipeline)")
    parser.add_argument("scene", nargs="?", default=str(_THIS.parent / "scene1.json"))
    args = parser.parse_args()
    main(args.scene)
