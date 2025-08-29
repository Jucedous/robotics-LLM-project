"""
CBF-based safety metrics for object-only manipulation scenes (no robot).

Updates (Aug 2025)
------------------
• Object-only rewrite: removed RobotState, LinkState, joint limits, human-speed, and
  link-vs-object checks. Metrics now consider only object–object relationships.
• Critical hazard override retained: any liquid–electronics overlap forces composite_risk=1.0
  and safety score to 0.0.
• Safety score reported as 0–5 (0 = very unsafe, 5 = very safe).

Core idea
---------
For each safety aspect, define a Control Barrier Function (CBF) h(x) ≥ 0 that encodes
forward-invariance of a safe set S = {x | h(x) ≥ 0}. Given state x and (optionally) xdot:
  • Safety margin:                   h(x)
  • CBF residual:                    ẋh(x) + α·h(x)   (should be ≥ 0)
  • Risk in [0, 1]:                  σ(−(h/scale)) and/or σ(−(residual/scale))

This file is dependency-light (NumPy only).
Call `evaluate_scene_metrics(scene)` to get per-metric reports and a composite score.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np

# -------------------------
# Utility math helpers
# -------------------------

def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def safe_norm(v: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.maximum(np.dot(v, v), eps)))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(x, hi))


# -------------------------
# Scene & state containers
# -------------------------

@dataclass
class Sphere:
    center: np.ndarray  # shape (3,)
    radius: float


@dataclass
class ObjectState:
    name: str
    sphere: Sphere
    kind: str = "object"               # e.g., 'human', 'fragile', 'liquid', 'electronic', 'sharp', 'heavy', 'hot', 'plastic'
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tags: Tuple[str, ...] = field(default_factory=tuple)  # arbitrary extra tags (e.g., ("hazard","knife"))


@dataclass
class Workspace:
    # Axis-aligned bounding box (AABB): [xmin, xmax], [ymin, ymax], [zmin, zmax]
    bounds: np.ndarray  # shape (3, 2)


@dataclass
class Scene:
    objects: List[ObjectState]
    workspace: Optional[Workspace] = None


# -------------------------
# CBF primitives (distance-based)
# -------------------------

def pairwise_distance_cbf(A: Sphere, B: Sphere, buffer: float = 0.0) -> float:
    """h(x) = d^2 - (rA + rB + buffer)^2 ≥ 0 is safe (squared-distance form)."""
    delta = A.center - B.center
    d2 = float(np.dot(delta, delta))
    rsum = A.radius + B.radius + buffer
    return d2 - rsum * rsum


def pairwise_dhdt(
    A_center: np.ndarray, B_center: np.ndarray,
    vA: np.ndarray, vB: np.ndarray,
) -> float:
    """Time derivative of h for the squared-distance barrier: ẋh = 2 (pA - pB)ᵀ (vA - vB)."""
    delta = A_center - B_center
    vrel = vA - vB
    return float(2.0 * np.dot(delta, vrel))


def alpha_linear(h: float, k: float = 1.0) -> float:
    return k * h


# -------------------------
# Object-only metrics
# -------------------------

def metric_object_collision_cbf(
    objects: List[ObjectState],
    buffer: float = 0.01,
    alpha_gain: float = 5.0,
    scale_h: float = 0.01,
    scale_res: float = 0.05,
) -> Dict:
    """
    Object–object collision avoidance CBF.
    For every unordered object pair (i<j), compute squared-distance barrier and residual.
    """
    per_pairs = []
    h_min = np.inf
    res_min = np.inf

    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            A = objects[i]
            B = objects[j]
            h = pairwise_distance_cbf(A.sphere, B.sphere, buffer=buffer)
            dh = pairwise_dhdt(A.sphere.center, B.sphere.center, A.velocity, B.velocity)
            residual = dh + alpha_linear(h, alpha_gain)
            h_min = min(h_min, h)
            res_min = min(res_min, residual)
            per_pairs.append({
                "A": A.name, "B": B.name,
                "h": h,
                "dh": dh,
                "residual": residual,
                "risk_h": float(sigmoid(-(h / scale_h))),
                "risk_residual": float(sigmoid(-(residual / scale_res))),
            })

    composite = float(np.mean([p["risk_h"] * 0.6 + p["risk_residual"] * 0.4 for p in per_pairs])) if per_pairs else 0.0
    return {
        "name": "object_collision_cbf",
        "h_min": float(h_min) if per_pairs else float("inf"),
        "residual_min": float(res_min) if per_pairs else float("inf"),
        "risk": composite,  # [0,1]
        "details": per_pairs,
        "explanation": "Distance-squared CBF between every object pair. Residual ≥ 0 is required for forward invariance."
    }


def metric_workspace_cbf_objects(
    objects: List[ObjectState],
    workspace: Optional[Workspace],
    alpha_gain: float = 5.0,
    scale_h: float = 0.01,
    scale_res: float = 0.05,
) -> Dict:
    """
    Workspace AABB constraint for each object's sphere center.
    h = min over face margins; residual uses the velocity component toward the active face.
    """
    if workspace is None:
        return {"name": "workspace_cbf_objects", "risk": 0.0, "details": [], "explanation": "No workspace bounds provided."}

    per = []
    hmins = []
    resmins = []

    bounds = workspace.bounds  # shape (3,2)
    for obj in objects:
        x = obj.sphere.center
        v = obj.velocity
        # margins to xmin,xmax,ymin,ymax,zmin,zmax
        h_axes = [float(x[d] - bounds[d, 0]) for d in range(3)] + [float(bounds[d, 1] - x[d]) for d in range(3)]
        idx = int(np.argmin(h_axes))
        h = h_axes[idx]
        if idx < 3:
            dh = float(v[idx])
        else:
            d = idx - 3
            dh = float(-v[d])
        residual = dh + alpha_linear(h, alpha_gain)
        per.append({
            "object": obj.name,
            "h": h,
            "dh": dh,
            "residual": residual,
            "risk_h": float(sigmoid(-(h / scale_h))),
            "risk_residual": float(sigmoid(-(residual / scale_res))),
        })
        hmins.append(h)
        resmins.append(residual)

    composite = float(np.mean([p["risk_h"] * 0.5 + p["risk_residual"] * 0.5 for p in per]))
    return {
        "name": "workspace_cbf_objects",
        "h_min": float(min(hmins)),
        "residual_min": float(min(resmins)),
        "risk": composite,
        "details": per,
        "explanation": "Workspace AABB CBF for object centers; smallest face margin is the active constraint."
    }


def metric_hazard_pairings_cbf_objects(
    objects: List[ObjectState],
    alpha_gain: float = 5.0,
    scale_res: float = 0.05,
) -> Dict:
    """
    Domain-specific hazard CBFs for risky object–object pairings (liquid–electronics, sharp–human, etc.).
    Also returns a **critical overlap** flag for liquid–electronics.
    """
    rules = [
        ("liquid", "electronic", 0.15),
        ("sharp", "human", 0.10),
        ("heavy", "fragile", 0.20),
        ("hot", "plastic", 0.12),
    ]

    per = []
    resmins = []
    critical_violation = False
    critical_pairs: List[Dict] = []

    # Build typed lists (allow tags to alias kinds)
    by_kind: Dict[str, List[ObjectState]] = {}
    for o in objects:
        by_kind.setdefault(o.kind, []).append(o)
        for t in o.tags:
            by_kind.setdefault(t, []).append(o)

    def centers_overlap(A: Sphere, B: Sphere) -> bool:
        d2 = float(np.dot(A.center - B.center, A.center - B.center))
        rsum_no_clearance = A.radius + B.radius
        return d2 <= (rsum_no_clearance * rsum_no_clearance)

    # Object-object hazards
    for A_kind, B_kind, safe_clearance in rules:
        As = by_kind.get(A_kind, [])
        Bs = by_kind.get(B_kind, [])
        for A in As:
            for B in Bs:
                if A is B:
                    continue
                h = pairwise_distance_cbf(A.sphere, B.sphere, buffer=safe_clearance)
                dh = pairwise_dhdt(A.sphere.center, B.sphere.center, A.velocity, B.velocity)
                residual = dh + alpha_linear(h, alpha_gain)
                overlap = centers_overlap(A.sphere, B.sphere)
                entry = {
                    "A": A.name, "B": B.name, "pair": f"{A_kind}-{B_kind}",
                    "clearance": safe_clearance, "h": h, "residual": residual,
                    "overlap": overlap,
                    "risk": float(sigmoid(-(residual / scale_res))),
                }
                per.append(entry)
                resmins.append(residual)
                # Critical override: liquid over electronics → hard fail
                if A_kind == "liquid" and B_kind == "electronic" and overlap:
                    critical_violation = True
                    critical_pairs.append(entry)

    composite = float(np.mean([p["risk"] for p in per])) if per else 0.0
    return {
        "name": "hazard_pairings_cbf",
        "residual_min": float(min(resmins)) if resmins else float("inf"),
        "risk": composite,
        "details": per,
        "critical_violation": critical_violation,
        "critical_pairs": critical_pairs,
        "explanation": "Distance-based CBFs with extra clearance for hazardous pairings. Liquid–electronics overlap triggers a hard fail."
    }


# -------------------------
# Top-level aggregation
# -------------------------

def evaluate_scene_metrics(scene: Scene) -> Dict:
    """
    Compute CBF-based safety metrics for an object-only scene and a composite score.

    Returns a dict with per-metric results and normalized scores in [0,1],
    plus a 0–5 safety score (5=safe, 0=unsafe).
    """
    # Individual metrics (object-only)
    m_collision = metric_object_collision_cbf(scene.objects)
    m_workspace = metric_workspace_cbf_objects(scene.objects, scene.workspace)
    m_hazard = metric_hazard_pairings_cbf_objects(scene.objects)

    metrics = [m_collision, m_workspace, m_hazard]

    # Composite risk: weighted average (tuneable)
    weights = {
        "object_collision_cbf": 0.40,
        "workspace_cbf_objects": 0.10,
        "hazard_pairings_cbf": 0.50,
    }
    score = 0.0
    wsum = 0.0
    for m in metrics:
        w = weights.get(m["name"], 0.0)
        score += w * m.get("risk", 0.0)
        wsum += w
    composite_risk = float(score / max(wsum, 1e-9))

    # *** Critical hazard override ***
    # If a liquid overlaps an electronic device, force overall risk to 1.0 (safety score 0.0).
    if m_hazard.get("critical_violation", False):
        composite_risk = 1.0

    # Map risk [0,1] → safety score [0,5] (5=safe, 0=unsafe)
    safety_score = float(np.clip(5.0 - 5.0 * composite_risk, 0.0, 5.0))

    return {
        "metrics": metrics,
        "composite_risk": composite_risk,      # 0 (safe) → 1 (risky)
        "safety_score_0_to_5": safety_score,   # 0 → 5
        "safety_grade_1_to_5": safety_score,   # kept for back-compat GUIs
    }


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Tiny demo with only objects
    objects = [
        ObjectState("laptop", Sphere(np.array([0.5, 0.2, 0.75]), 0.10), kind="electronic"),
        ObjectState("water_cup", Sphere(np.array([0.5, 0.2, 0.75]), 0.06), kind="liquid"),  # overlapping to demo hard fail
        ObjectState("knife", Sphere(np.array([0.1, -0.2, 0.5]), 0.02), kind="sharp"),
        ObjectState("human_1", Sphere(np.array([0.4, 0.0, 0.5]), 0.15), kind="human"),
        ObjectState("glass_vase", Sphere(np.array([0.2, 0.1, 0.6]), 0.08), kind="fragile"),
        ObjectState("cast_iron_pan", Sphere(np.array([0.3, 0.1, 0.6]), 0.09), kind="heavy"),
    ]

    workspace = Workspace(bounds=np.array([[-1.0, 1.0], [-1.0, 1.0], [0.0, 1.5]]))
    scene = Scene(objects=objects, workspace=workspace)

    out = evaluate_scene_metrics(scene)
    from pprint import pprint
    pprint(out)
