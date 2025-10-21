from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

Vec3 = Tuple[float, float, float]

@dataclass
class Object:
    name: str
    position: Vec3
    categories: List[str] = field(default_factory=list)
    mass: Optional[float] = None
    fragile: Optional[bool] = None
    velocity: Optional[Vec3] = None
    props: Dict[str, float] = field(default_factory=dict)

@dataclass
class Scene:
    objects: List[Object]

@dataclass
class Relation:
    type: str
    a: str
    b: str
    score: float
    meta: Dict[str, float] = field(default_factory=dict)

@dataclass
class HazardFinding:
    rule_id: str
    description: str
    involved: List[str]
    h_value: float
    risk: float

@dataclass
class EvaluationResult:
    safety_score: float
    findings: List[HazardFinding]