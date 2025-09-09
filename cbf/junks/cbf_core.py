from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np

# ---------- Utility ----------
def sigmoid_stable(x: np.ndarray | float) -> np.ndarray | float:
    if isinstance(x, np.ndarray):
        x = np.clip(x, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-x))
    else:
        x = float(np.clip(x, -60.0, 60.0))
        return 1.0 / (1.0 + np.exp(-x))

def alpha_linear(h: float, k: float = 1.0) -> float:
    return k * h

def pairwise_distance_cbf(A, B, buffer: float = 0.0) -> float:
    delta = A.center - B.center
    d2 = float(np.dot(delta, delta))
    rsum = A.radius + B.radius + buffer
    return d2 - rsum * rsum

def pairwise_dhdt(A_center, B_center, vA, vB) -> float:
    delta = A_center - B_center
    vrel = vA - vB
    return float(2.0 * np.dot(delta, vrel))

# ---------- Scene containers ----------
@dataclass
class Sphere:
    center: np.ndarray
    radius: float

@dataclass
class ObjectState:
    name: str
    sphere: Sphere
    kind: str = "object"
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tags: Tuple[str, ...] = field(default_factory=tuple)

@dataclass
class Workspace:
    bounds: np.ndarray  # shape (3,2)

@dataclass
class Scene:
    objects: List[ObjectState]
    workspace: Optional[Workspace] = None
