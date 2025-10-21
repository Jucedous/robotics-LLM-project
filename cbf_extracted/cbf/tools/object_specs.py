from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import json
import numpy as np


def read_json_specs(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scene JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Scene JSON must be a list of object specs.")
    return data

def normalize_spec(
    idx: int,
    item: Dict[str, Any],
    *,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    zlim: tuple[float, float],
    valid_kinds: set[str],
) -> Dict[str, Any]:
    """Validate + normalize one spec (GUI-agnostic)."""
    if not isinstance(item, dict):
        raise ValueError(f"Entry #{idx} is not an object/dict.")

    X_MIN, X_MAX = xlim
    Y_MIN, Y_MAX = ylim
    Z_MIN, Z_MAX = zlim

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
    if kind not in valid_kinds:
        kind = "object"

    r = max(1e-6, min(r, (X_MAX - X_MIN) * 0.5))
    x = float(np.clip(xy[0], X_MIN + r, X_MAX - r))
    y = float(np.clip(xy[1], Y_MIN + r, Y_MAX - r))
    z = float(np.clip(z, Z_MIN, Z_MAX))


    return dict(name=name, kind=kind, r=r, xy=(x, y), z=z)