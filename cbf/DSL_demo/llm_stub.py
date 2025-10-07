from __future__ import annotations
import math
from typing import Dict, List, Tuple
from .types import Scene, Object
from .dsl.catalog import NAME_TO_CATS, CATEGORY_DEFAULTS

PLANAR_NEAR_THRESH = 0.35

def _planar_dist(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    ax, ay, _ = a
    bx, by, _ = b
    return math.hypot(ax - bx, ay - by)

def categorize_objects(objs: List[Object]) -> None:
    for o in objs:
        key = o.name.strip().lower()
        cats = NAME_TO_CATS.get(key, [])
        if 'water' in key and 'water' not in cats:
            cats = list(set(cats + ['water', 'liquid']))
        if 'card' in key and 'card' not in cats:
            cats = list(set(cats + ['card']))
        if 'glass' in key and 'fragile' not in cats:
            cats = list(set(cats + ['fragile']))
        o.categories = list(dict.fromkeys(cats))
        for c in o.categories:
            if c in CATEGORY_DEFAULTS:
                for k, v in CATEGORY_DEFAULTS[c].items():
                    setattr(o, k, v)

def infer_relations(scene: Scene) -> List[Dict]:
    relations: List[Dict] = []
    objs = scene.objects
    n = len(objs)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            A = objs[i]
            B = objs[j]
            if A.position[2] > B.position[2]:
                relations.append({
                    'type': 'above', 'a': A.name, 'b': B.name, 'score': 1.0
                })
            if _planar_dist(A.position, B.position) <= PLANAR_NEAR_THRESH:
                relations.append({
                    'type': 'near', 'a': A.name, 'b': B.name, 'score': 1.0
                })
    return relations