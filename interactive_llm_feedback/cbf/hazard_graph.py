from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Set, Tuple, List

from .cbf_safety_metrics import ObjectState


@dataclass
class ObjectHazardEdge:
    obj_a: str
    obj_b: str
    soft_clearance_m: float
    weight: float


@dataclass
class SceneHazardGraph:
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[Tuple[str, str], ObjectHazardEdge] = field(default_factory=dict)

    def add_node(self, name: str) -> None:
        self.nodes.add(name)

    def add_edge(self, a: str, b: str, *, soft_clearance_m: float, weight: float) -> None:
        key = tuple(sorted((a, b)))
        if key in self.edges:
            edge = self.edges[key]
            edge.soft_clearance_m = max(edge.soft_clearance_m, float(soft_clearance_m))
            edge.weight = max(edge.weight, float(weight))
        else:
            self.edges[key] = ObjectHazardEdge(
                obj_a=key[0],
                obj_b=key[1],
                soft_clearance_m=float(soft_clearance_m),
                weight=float(weight),
            )

    def as_text_lines(self) -> List[str]:
        """
        Convenience helper: human-readable adjacency-style lines.
        """
        if not self.edges:
            return [" • (no semantic hazard edges)"]
        lines: List[str] = []
        for (a, b), edge in sorted(self.edges.items()):
            lines.append(
                f" • {a} --(clr={edge.soft_clearance_m:.3f}m, w={edge.weight:.2f})--> {b}"
            )
        return lines


def build_scene_hazard_graph_from_rules(
    objects: List[ObjectState],
    rules: List[Tuple[str, str, float, float, str, str]],
) -> SceneHazardGraph:
    """
    Build an object-level semantic hazard graph for a scene.

    Args:
        objects: list of ObjectState for the current scene.
        rules:   list of instantiated LLM semantic rules produced by
                 cbf.semantics_runtime.instantiate_rules, BEFORE user preferences.
                 Each rule is (kindA, kindB, soft_clearance_m, weight, A_name, B_name).

    Returns:
        SceneHazardGraph with one node per object and undirected edges between
        object pairs that the LLM flagged as semantically hazardous.
    """
    graph = SceneHazardGraph()
    for o in objects:
        graph.add_node(o.name)

    for (_kindA, _kindB, clr, w, Aname, Bname) in rules:
        graph.add_edge(
            Aname,
            Bname,
            soft_clearance_m=clr,
            weight=w,
        )

    return graph
