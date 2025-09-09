from typing import Dict, List
import numpy as np
from cbf.junks.cbf_core import (
    ObjectState, Workspace, Scene, pairwise_distance_cbf, pairwise_dhdt,
    alpha_linear, sigmoid_stable
)
from cbf.junks.hazard_policy import get_hazard_policy_via_llm, DEFAULT_FALLBACK_POLICY

def metric_object_collision_cbf(objects: List[ObjectState]) -> Dict:
    per_pairs, h_min, res_min = [], np.inf, np.inf
    for i in range(len(objects)):
        for j in range(i+1, len(objects)):
            A, B = objects[i], objects[j]
            h = pairwise_distance_cbf(A.sphere, B.sphere, buffer=0.01)
            dh = pairwise_dhdt(A.sphere.center, B.sphere.center, A.velocity, B.velocity)
            residual = dh + alpha_linear(h, 5.0)
            per_pairs.append({
                "A": A.name, "B": B.name,
                "h": h, "residual": residual,
                "risk": float(sigmoid_stable(-(residual / 0.05)))
            })
            h_min, res_min = min(h_min, h), min(res_min, residual)
    return {"name":"object_collision_cbf","risk":float(np.mean([p["risk"] for p in per_pairs])) if per_pairs else 0.0,"details":per_pairs}

def metric_workspace_cbf_objects(objects: List[ObjectState], workspace: Workspace) -> Dict:
    if not workspace: return {"name":"workspace_cbf_objects","risk":0.0,"details":[]}
    per, hmins = [], []
    for o in objects:
        x, r = o.sphere.center, o.sphere.radius
        margins = [
            (x[0]-r)-workspace.bounds[0,0],
            workspace.bounds[0,1]-(x[0]+r),
            (x[1]-r)-workspace.bounds[1,0],
            workspace.bounds[1,1]-(x[1]+r),
            (x[2]-r)-workspace.bounds[2,0],
            workspace.bounds[2,1]-(x[2]+r),
        ]
        h = min(margins); hmins.append(h)
        per.append({"object":o.name,"h":h,"risk":float(sigmoid_stable(-(h/0.01)))})
    return {"name":"workspace_cbf_objects","risk":float(np.mean([p["risk"] for p in per])),"details":per}

def metric_hazard_pairings_cbf_objects(objects: List[ObjectState]) -> Dict:
    policy = get_hazard_policy_via_llm(objects) or DEFAULT_FALLBACK_POLICY
    per, critical_pairs, critical_violation = [], [], False
    by_kind = {}
    def add_kind(k,o): by_kind.setdefault(k,{})[o.name]=o
    for o in objects:
        add_kind(o.kind,o)
        for t in o.tags: add_kind(t,o)
    for name, aliases in policy["aliases"].items():
        match = next((o for o in objects if o.name==name),None)
        if match:
            for k in aliases: add_kind(k,match)
    for pr in policy["pairs"]:
        A_kind,B_kind,clearance = pr["A_kind"],pr["B_kind"],float(pr["clearance_m"])
        As,Bs = by_kind.get(A_kind,{}).values(), by_kind.get(B_kind,{}).values()
        for A in As:
            for B in Bs:
                if A is B: continue
                h = pairwise_distance_cbf(A.sphere,B.sphere,buffer=clearance)
                dh = pairwise_dhdt(A.sphere.center,B.sphere.center,A.velocity,B.velocity)
                residual = dh + alpha_linear(h,5.0)
                risk = float(sigmoid_stable(-(residual/0.05)))
                entry = {"A":A.name,"B":B.name,"pair":f"{A_kind}-{B_kind}","risk":risk}
                per.append(entry)
                if pr["severity"]=="critical" and (entry["risk"]>0.9):
                    critical_violation=True
                    critical_pairs.append(entry)
    return {"name":"hazard_pairings_cbf","risk":float(np.mean([p["risk"] for p in per])) if per else 0.0,"details":per,"critical_violation":critical_violation,"critical_pairs":critical_pairs}

def evaluate_scene_metrics(scene: Scene) -> Dict:
    m1=metric_object_collision_cbf(scene.objects)
    m2=metric_workspace_cbf_objects(scene.objects,scene.workspace)
    m3=metric_hazard_pairings_cbf_objects(scene.objects)
    weights={"object_collision_cbf":0.4,"workspace_cbf_objects":0.1,"hazard_pairings_cbf":0.5}
    composite=sum(weights[m["name"]]*m["risk"] for m in [m1,m2,m3])
    if m3["critical_violation"]: composite=1.0
    safety=5.0-5.0*composite
    return {"metrics":[m1,m2,m3],"composite_risk":composite,"safety_score":max(0.0,min(5.0,safety))}
