"""
empty_test.py:
• Prefers cbf.tools.cbf_safety_app_3d.CBFSafetyApp3D, falls back to cbf.tools.cbf_safety_app_2d.CBFSafetyApp2D
• Wires UI to cbf.cbf_safety_metrics for risk/score
• Opens a scene JSON via --scene and saves changes back to the SAME file atomically

Run:
  python -m cbf.empty_test --scene scenes/scene1.json
"""

from __future__ import annotations
import argparse
import importlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import cbf_safety_metrics as metrics
from .cbf_safety_metrics import Sphere, ObjectState, Workspace, Scene, evaluate_scene_metrics

AppClass = None
APP_DIM = "2D"
mod2d = importlib.import_module(".tools.cbf_safety_app_2d", package=__package__)
AppClass = getattr(mod2d, "CBFSafetyApp2D")

CURRENT_SCENE_PATH: Path | None = None

KIND_MAP = {
    "water": "liquid",
    "electronics": "electronic",
}
def _normalize_kind(k: str) -> str:
    k2 = KIND_MAP.get(str(k).lower().strip(), str(k).lower().strip())
    return k2

def _serializable_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for s in specs:
        s2 = dict(s)
        x, y = s2["xy"]
        s2["xy"] = [float(x), float(y)]
        s2["z"] = float(s2["z"])
        s2["r"] = float(s2["r"])
        s2["name"] = str(s2["name"])
        s2["kind"] = str(s2["kind"])
        out.append(s2)
    return out

def _atomic_write_json(path: Path, data: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

def open_scene(app, path: Path): 
    """Load a scene file and remember it as the 'current' one."""
    global CURRENT_SCENE_PATH
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            specs = data
        elif isinstance(data, dict):
            specs = data.get("objects", [])
        else:
            specs = []
    except Exception:
        specs = []
    app.clear_objects()
    app.add_objects_from_specs(specs)
    CURRENT_SCENE_PATH = path

def save_current_scene(app):
    """Overwrite the currently opened scene JSON (if any)."""
    if CURRENT_SCENE_PATH is None:
        return
    specs = app.get_current_specs()
    _atomic_write_json(CURRENT_SCENE_PATH, _serializable_specs(specs))

def _specs_to_scene(specs: List[Dict[str, Any]], app) -> Scene:
    xmin, xmax = app.X_MIN, app.X_MAX
    ymin, ymax = app.Y_MIN, app.Y_MAX
    zmin = getattr(app, "Z_MIN", 0.0)
    zmax = getattr(app, "Z_MAX", 1.5)

    objs: List[ObjectState] = []
    for s in specs:
        name = str(s["name"])
        kind = _normalize_kind(s.get("kind", "object"))
        r = float(s.get("r", s.get("radius", 0.08)))

        if "center" in s:
            cx, cy, cz = s["center"]
            x, y, z = float(cx), float(cy), float(cz)
        else:
            x, y = float(s["xy"][0]), float(s["xy"][1])
            z = float(s.get("z", 0.6))

        sphere = Sphere(center=np.array([x, y, z], dtype=float), radius=r)
        objs.append(ObjectState(name=name, sphere=sphere, kind=kind))

    ws = Workspace(bounds=np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]], dtype=float))
    return Scene(objects=objs, workspace=ws)

def _format_info(out: Dict[str, Any]) -> List[str]:
    lines: List[str] = []
    lines.append(f"Composite risk: {out.get('composite_risk', 0.0):0.3f}")
    lines.append("")
    for m in out.get("metrics", []):
        nm = m.get("name", "metric")
        rk = m.get("risk", 0.0)
        if nm == "hazard_pairings_cbf" and m.get("critical_violation", False):
            lines.append(f"[{nm:<22}] risk={rk:0.3f}   **CRITICAL**")
        else:
            lines.append(f"[{nm:<22}] risk={rk:0.3f}")
    return lines

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="", help="Path to scene JSON to open")
    args, _ = parser.parse_known_args()

    app = AppClass()

    if not args.scene:
        print("Error: --scene argument is required.")
        exit(1)
    
    p = Path(args.scene)
    if not p.exists():
        print(f"Error: scene file not found: {p}")
        exit(1)
    
    open_scene(app, p)

    def on_change():
        try:
            specs = app.get_current_specs()
            if not specs:
                app.set_info_lines(["(no objects)"])
                app.set_score(5.0)
                save_current_scene(app)
                return

            scene = _specs_to_scene(specs, app)
            out = evaluate_scene_metrics(scene)

            score = out.get("safety_score_0_to_5", 0.0)
            try:
                score = float(score)
                if not np.isfinite(score):
                    score = 0.0
            except Exception:
                score = 0.0

            app.set_score(score)
            app.set_info_lines(_format_info(out))
            save_current_scene(app)

        except Exception as e:
            app.set_info_lines([f"Error in on_change: {e!r}"])

    if hasattr(app, "set_on_change"):
        app.set_on_change(on_change)
    on_change()

    if hasattr(app, "show"):
        app.show()
    else:
        import matplotlib.pyplot as plt
        plt.show()

if __name__ == "__main__":
    main()
