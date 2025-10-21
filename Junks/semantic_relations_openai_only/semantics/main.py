import argparse, pathlib, json
from typing import Dict, Any
from .utils.io import read_json, write_json
from .schema import RelationReport, PairwiseRelation, RelationConstraint
from .providers.openai_provider import OpenAIProvider

def build_report(scene_path: str, model: str = "gpt-4.1-mini", timeout: int = 180) -> Dict[str, Any]:
    scene = read_json(scene_path)
    provider = OpenAIProvider(model=model, timeout=timeout)
    result = provider.analyze(scene)

    raw = result.get("pairwise_relations", []) or []
    pairwise = []
    for r in raw:
        objs = list(map(str, (r.get("objects") or [])[:2]))
        rule = str(r.get("rule", ""))
        severity = float(r.get("severity", 0.0))
        confidence = float(r.get("confidence", 0.0))
        interp = str(r.get("interpretation_3d", ""))

        cons = []
        for c in r.get("constraints", []) or []:
            lhs = str(c.get("lhs", "")).strip()
            op  = str(c.get("op", "")).strip()
            rhs = str(c.get("rhs", "")).strip()
            note= str(c.get("note", "")).strip()
            if lhs and op and rhs:
                cons.append(RelationConstraint(lhs=lhs, op=op, rhs=rhs, note=note))

        pairwise.append(
            PairwiseRelation(
                objects=objs,
                rule=rule,
                severity=max(0.0, min(5.0, severity)),
                confidence=max(0.0, min(1.0, confidence)),
                interpretation_3d=interp,
                constraints=cons,
            )
        )

    report = RelationReport(
        scene_file=str(scene_path),
        pairwise_relations=pairwise,
        provider_meta=result.get("provider_meta", {}),
    )
    return report.to_dict()

def main():
    ap = argparse.ArgumentParser(description="Semantic safety relations via OpenAI (Responses API).")
    ap.add_argument("--scene", required=True, help="Path to two-object scene JSON (id + kind only).")
    ap.add_argument("--model", default="gpt-4.1-mini", help="OpenAI model (default: gpt-4.1-mini).")
    ap.add_argument("--timeout", type=int, default=180, help="Request timeout seconds (default: 180).")
    ap.add_argument("--out", default=None, help="Optional output JSON path.")
    args = ap.parse_args()

    report = build_report(args.scene, model=args.model, timeout=args.timeout)

    out_path = args.out
    if out_path is None:
        out_dir = pathlib.Path(__file__).resolve().parents[1] / "semantics_out"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = pathlib.Path(args.scene).stem
        out_path = out_dir / f"{stem}_relations.json"

    write_json(report, str(out_path))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_path}")

if __name__ == "__main__":
    main()
