from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

@dataclass
class ObjSpec:
    name: str
    kind: str
    r: float
    x: float
    y: float
    z: float = 0.0

    def to_dict(self) -> Dict:
        return dict(name=self.name, kind=self.kind, r=self.r, xy=(self.x, self.y), z=self.z)

def normalize_spec(
    idx: int,
    item: Dict,
    *,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    zlim: Tuple[float, float],
    known_kinds: Tuple[str, ...] = ("electronic", "liquid", "human", "sharp", "fragile", "heavy", "object"),
    default_kind: str = "object",
    default_r: float = 0.08,
) -> Dict:
    if not isinstance(item, dict):
        raise ValueError(f"Entry #{idx} is not an object/dict.")

    name = str(item.get("name", f"obj_{idx}"))

    if "center" in item and "radius" in item:
        cx, cy, cz = item["center"]
        r = float(item["radius"])
        x, y, z = float(cx), float(cy), float(cz)
    else:
        xy_raw = item.get("xy")
        if not xy_raw or len(xy_raw) != 2:
            raise ValueError(f"Entry '{name}' missing 'xy': [x, y].")
        x, y = float(xy_raw[0]), float(xy_raw[1])
        z = float(item.get("z", 0.0))
        r = float(item.get("r", default_r))

    kind = str(item.get("kind", default_kind)).strip().lower()
    if kind not in known_kinds:
        kind = default_kind

    X_MIN, X_MAX = xlim
    Y_MIN, Y_MAX = ylim
    Z_MIN, Z_MAX = zlim

    r = max(1e-6, min(r, (X_MAX - X_MIN) * 0.5))
    x = float(np.clip(x, X_MIN + r, X_MAX - r))
    y = float(np.clip(y, Y_MIN + r, Y_MAX - r))
    z = float(np.clip(z, Z_MIN, Z_MAX))

    return ObjSpec(name=name, kind=kind, r=r, x=x, y=y, z=z).to_dict()
