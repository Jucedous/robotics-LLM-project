from __future__ import annotations
from typing import Dict, List
from .rules import RULES
from .cbf import barrier_to_risk
from ..types import Scene, HazardFinding, EvaluationResult

def aggregate_findings(findings: List[HazardFinding]) -> float:
    overall_risk = 0.0
    prod = 1.0
    for f in findings:
        r = max(0.0, min(1.0, f.risk))
        prod *= (1.0 - r)
    overall_risk = 1.0 - prod
    safety = 100.0 * (1.0 - overall_risk)
    return max(0.0, min(100.0, safety))

def evaluate(scene: Scene, relations: List[Dict]) -> EvaluationResult:
    findings: List[HazardFinding] = []

    name_to_obj = {o.name: o for o in scene.objects}


    for rel in relations:
        a = name_to_obj.get(rel['a'])
        b = name_to_obj.get(rel['b'])
        if not a or not b:
            continue
        ctx = {
            'objA': a,
            'objB': b,
            'posA': a.position,
            'posB': b.position,
            'velA': a.velocity,
            'velB': b.velocity,
            'catA': a.categories,
            'catB': b.categories,
            'rel': rel['type'],
            'score': rel.get('score', 1.0),
        }
        for rule in RULES:
            try:
                if rule.match(ctx):
                    h = rule.barrier(ctx)
                    risk = barrier_to_risk(h, scale=rule.risk_scale)
                    findings.append(HazardFinding(
                        rule_id=rule.rule_id,
                        description=rule.description,
                        involved=[a.name, b.name],
                        h_value=h,
                        risk=risk,
                ))
            except Exception as e:
                continue


    safety_score = aggregate_findings(findings)
    return EvaluationResult(safety_score=safety_score, findings=findings)