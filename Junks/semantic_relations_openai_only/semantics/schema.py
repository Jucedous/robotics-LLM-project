from dataclasses import dataclass, asdict
from typing import List, Dict, Any

@dataclass
class ObjectSpec:
    id: str
    kind: str

@dataclass
class SceneSpec:
    objects: List[ObjectSpec]

@dataclass
class RelationConstraint:
    lhs: str
    op: str
    rhs: str
    note: str

@dataclass
class PairwiseRelation:
    objects: List[str]
    rule: str
    severity: float
    confidence: float
    interpretation_3d: str
    constraints: List[RelationConstraint]

@dataclass
class RelationReport:
    scene_file: str
    pairwise_relations: List[PairwiseRelation]
    provider_meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_file": self.scene_file,
            "pairwise_relations": [
                {
                    "objects": pr.objects,
                    "rule": pr.rule,
                    "severity": pr.severity,
                    "confidence": pr.confidence,
                    "interpretation_3d": pr.interpretation_3d,
                    "constraints": [asdict(c) for c in pr.constraints],
                }
                for pr in self.pairwise_relations
            ],
            "provider_meta": self.provider_meta,
        }
