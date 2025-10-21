import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import os
import json
import numpy as np
from cbf.semantics_runtime import LLMConfig, analyze_scene_llm, instantiate_rules
from cbf.cbf_safety_metrics import ObjectState, Sphere
from cbf.cbf_safety_metrics_llm import metric_hazard_pairings_cbf_objects_llm


def load_scene(path: str):
    items = json.loads(Path(path).read_text())
    objs = []
    for it in items:
        x, y = it["xy"]
        z = it.get("z", 0.0)
        r = it["r"]
        objs.append(
            ObjectState(
                name=it["name"],
                kind=it["kind"],
                sphere=Sphere(center=np.array([x, y, z], dtype=float), radius=float(r)),
                tags=tuple(it.get("tags", [])),
            )
        )
    return objs


def to_llm_payload(objs):
    return [
        dict(
            name=o.name,
            kind=o.kind,
            tags=list(o.tags),
            xyz=[
                float(o.sphere.center[0]),
                float(o.sphere.center[1]),
                float(o.sphere.center[2]),
            ],
            r=float(o.sphere.radius),
        )
        for o in objs
    ]


def main(scene_path: str):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        keyfile = Path(__file__).resolve().parent.parent / "config" / "openai_key.txt"
        if keyfile.exists():
            api_key = keyfile.read_text().strip()
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY (env) or config/openai_key.txt")

    cfg = LLMConfig(api_key=api_key)

    objects = load_scene(scene_path)
    risks = analyze_scene_llm(to_llm_payload(objects), cfg)
    rules, crit_map = instantiate_rules(objects, risks)

    out = metric_hazard_pairings_cbf_objects_llm(
        objects=objects,
        alpha_gain=5.0,
        scale_res=0.05,
        rules=rules,
        critical_by_pair=crit_map,
    )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    import sys

    scene = (
        sys.argv[1]
        if len(sys.argv) > 1
        else str(Path(__file__).resolve().parent / "scene1.json")
    )
    main(scene)
