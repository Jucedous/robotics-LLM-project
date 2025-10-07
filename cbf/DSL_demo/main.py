from __future__ import annotations
import argparse, json
from typing import List
from .types import Object, Scene
from .llm_stub import categorize_objects, infer_relations
from .dsl.engine import evaluate

# python3 -m DSL_demo.main --scene DSL_demo/scenes/scene_mixed.json

def load_scene(path: str) -> Scene:
    with open(path, 'r') as f:
        data = json.load(f)
    objs: List[Object] = []
    for item in data.get('objects', []):
        objs.append(Object(
            name=item['name'],
            position=tuple(item['position']),
            categories=item.get('categories', []),
            mass=item.get('mass'),
            fragile=item.get('fragile'),
            velocity=tuple(item['velocity']) if 'velocity' in item else None,
            props=item.get('props', {}),
        ))
    return Scene(objects=objs)


def main():
    parser = argparse.ArgumentParser(description='Pure DSL Semantic Safety Demo')
    parser.add_argument('--scene', type=str, required=True, help='Path to scene JSON')
    args = parser.parse_args()


    scene = load_scene(args.scene)
    categorize_objects(scene.objects)
    relations = infer_relations(scene)


    result = evaluate(scene, relations)


    print(f"Safety Score: {result.safety_score:.2f} / 100")
    if not result.findings:
        print("No hazards detected by DSL rules.")
    else:
        print("Hazard Findings (lower h = worse):")
        for f in sorted(result.findings, key=lambda x: x.risk, reverse=True):
            inv = ", ".join(f.involved)
            print(f"- [{f.rule_id}] {f.description} | objs=({inv}) | h={f.h_value:.3f} | risk={f.risk:.2f}")


if __name__ == '__main__':
    main()