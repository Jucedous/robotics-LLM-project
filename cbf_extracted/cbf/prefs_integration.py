
from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
from .preferences import PreferenceStore

def _get_name(obj: Any, default: str) -> str:
    return getattr(obj, "name", getattr(obj, "id", default))

def _get_kind(obj: Any) -> Optional[str]:
    return getattr(obj, "kind", None)

def _get_tags(obj: Any) -> Tuple[str, ...]:
    return tuple(getattr(obj, "tags", ()))

def _get_center_radius(obj: Any) -> Tuple[Tuple[float, float, float], float]:
    if hasattr(obj, "sphere"):
        c = tuple(getattr(obj.sphere, "center", (0.0, 0.0, 0.0)))
        r = float(getattr(obj.sphere, "radius", getattr(obj, "radius", 0.05)))
        return c, r
    c = tuple(getattr(obj, "center", (0.0, 0.0, 0.0)))
    r = float(getattr(obj, "radius", getattr(obj, "r", 0.05)))
    return c, r

def _match_selector(obj: Any, sel: Dict[str, Any]) -> bool:
    by, val = sel.get("by"), sel.get("value")
    if by == "kind":
        return _get_kind(obj) == val
    if by == "name":
        return _get_name(obj, "") == val
    if by == "tag":
        return val in _get_tags(obj)
    return False

def _eval_simple_condition(expr: Optional[str], A: Any, B: Any) -> bool:
    if not expr:
        return True
    (Ac, Ar), (Bc, Br) = _get_center_radius(A), _get_center_radius(B)
    Ax, Ay, Az = Ac
    Bx, By, Bz = Bc
    if expr.strip() == "Az > Bz":
        return Az > Bz
    if expr.strip() == "Az < Bz":
        return Az < Bz
    return True

def _enumerate_pairs(objects: List[Any]) -> List[Tuple[Any, Any]]:
    out: List[Tuple[Any, Any]] = []
    n = len(objects)
    for i in range(n):
        for j in range(i + 1, n):
            out.append((objects[i], objects[j]))
    return out

def _resolve_pairs_from_assessments(
    compiled_assessments: List[Dict[str, Any]],
    objects: List[Any]
) -> List[Tuple[Any, Any]]:
    """
    If the assessment items already encode A_name/B_name, use that.
    Otherwise assume lexicographic pair order.
    """
    by_name = {_get_name(o, f"obj{i}"): o for i, o in enumerate(objects)}

    pairs: List[Tuple[Any, Any]] = []
    have_names = all(
        isinstance(a, dict) and ("A_name" in a and "B_name" in a)
        for a in compiled_assessments
    )
    if have_names:
        for a in compiled_assessments:
            A = by_name.get(a["A_name"])
            B = by_name.get(a["B_name"])
            if A is None or B is None:
                return _enumerate_pairs(objects)
            pairs.append((A, B))
        return pairs

    return _enumerate_pairs(objects)

def apply_user_preferences(
    compiled_assessments: List[Dict[str, Any]],
    objects: List[Any],
    user_id: str,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Mutates/returns the compiled assessments by enforcing user rules.
    Rules may set:
      - present: bool
      - weight: float
      - soft_clearance_m: float
      - critical_condition: None to remove
    """
    store = PreferenceStore(path=db_path) if db_path else PreferenceStore()
    rules = store.list_rules(user_id)
    if not rules:
        return compiled_assessments

    pairs = _resolve_pairs_from_assessments(compiled_assessments, objects)
    if len(pairs) != len(compiled_assessments):
        return compiled_assessments

    for i, assess in enumerate(compiled_assessments):
        A, B = pairs[i]

        for r in rules:
            selA = r.get("selectors", {}).get("A")
            selB = r.get("selectors", {}).get("B")
            if not (selA and selB):
                continue

            dir_rule = r.get("directional", False)
            cond = r.get("condition_expr")

            def _applies(X, Y):
                if not (_match_selector(X, selA) and _match_selector(Y, selB)):
                    return False
                return _eval_simple_condition(cond, X, Y)

            matched = _applies(A, B)
            if not matched and not dir_rule:
                matched = _applies(B, A)

            if not matched:
                continue

            override = r.get("override", {})
            if "present" in override:
                assess["present"] = bool(override["present"])

                if assess["present"] is False:
                    assess["critical_condition"] = None
            if "weight" in override:
                assess["weight"] = float(override["weight"])
            if "soft_clearance_m" in override:
                assess["soft_clearance_m"] = float(override["soft_clearance_m"])
            if "critical_condition" in override:
                assess["critical_condition"] = override["critical_condition"]

            reason = assess.get("reason", "")
            assess["reason"] = (reason + " | " if reason else "") + f"user_rule:{r.get('id','?')}"

    return compiled_assessments
