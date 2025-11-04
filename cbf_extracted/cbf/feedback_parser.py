
from __future__ import annotations
import re, getpass
from typing import Any, Dict, Optional

_KIND_SYNONYMS = {
    "water": "liquid", "liquid": "liquid", "juice": "liquid", "coffee": "liquid",
    "electronic": "electronic", "electronics": "electronic", "laptop": "electronic", "phone": "electronic",
    "tablet": "electronic", "camera": "electronic", "charger": "electronic",
    "human": "human", "person": "human", "hand": "human", "operator": "human",
    "sharp": "sharp", "knife": "sharp", "scissors": "sharp", "blade": "sharp",
}

_DIR_OVER   = re.compile(r"\b(over|above)\b", re.I)
_DIR_UNDER  = re.compile(r"\b(under|below)\b", re.I)
_NEAR       = re.compile(r"\b(near|next\s*to|close\s*to|adjacent)\b", re.I)

_NOT_DANGER = re.compile(r"\b(not\s+dangerous|not\s+hazard(?:ous)?|safe|harmless)\b", re.I)
_DANGER     = re.compile(r"\b(dangerous|hazard(?:ous)?)\b", re.I)


_CLEARANCE  = re.compile(r"\b(clearance|margin|gap)\s*(?:=|of|is)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", re.I)

_WEIGHT_NUM = re.compile(r"\b(weight|importance|priority)\s*(?:=|is)?\s*(\d+(?:\.\d+)?)\b", re.I)
_WEIGHT_WORD= re.compile(r"\b(weight|importance|priority)\s*(?:=|is)?\s*(low|medium|high)\b", re.I)

def _unit_to_m(val: float, unit: str) -> float:
    unit = unit.lower()
    if unit == "mm": return val / 1000.0
    if unit == "cm": return val / 100.0
    return val 

def _canon_kind(token: str) -> Optional[str]:
    return _KIND_SYNONYMS.get(token.lower())

def parse_feedback(text: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    t = text.strip()
    if not t:
        raise ValueError("Empty feedback.")

    user_id = user_id or getpass.getuser()

    tokens = re.findall(r"[A-Za-z]+", t)
    kinds = [_canon_kind(tok) for tok in tokens]
    kinds = [k for k in kinds if k]
    if len(kinds) < 2:
        raise ValueError("Could not find two kinds. Example: 'liquid over electronic not dangerous'.")

    A_kind, B_kind = kinds[0], kinds[1]

    directional = False
    condition_expr = None
    direction = None
    if _DIR_OVER.search(t):
        directional = True
        direction = "over"
        condition_expr = "Az > Bz"
    elif _DIR_UNDER.search(t):
        directional = True
        direction = "under"
        condition_expr = "Az < Bz"
    elif _NEAR.search(t):
        directional = False
        direction = "near" 

    present = None
    if _NOT_DANGER.search(t):
        present = False
    elif _DANGER.search(t):
        present = True

    if present is None:
        raise ValueError("Please specify whether it is 'dangerous' or 'not dangerous'.")

    soft_clearance_m = None
    m = _CLEARANCE.search(t)
    if m:
        soft_clearance_m = _unit_to_m(float(m.group(2)), m.group(3))

    weight = None
    n = _WEIGHT_NUM.search(t)
    if n:
        weight = float(n.group(2))
    else:
        w = _WEIGHT_WORD.search(t)
        if w:
            word = w.group(2).lower()
            weight = {"low": 0.25, "medium": 0.5, "high": 1.0}[word]

    rule = {
        "user_id": user_id,
        "selectors": {
            "A": {"by": "kind", "value": A_kind},
            "B": {"by": "kind", "value": B_kind},
        },
        "directional": bool(directional),
        "direction": direction, 
        "condition_expr": condition_expr, 
        "override": {
            "present": bool(present),
        },
        "reason": t,
    }
    if soft_clearance_m is not None:
        rule["override"]["soft_clearance_m"] = soft_clearance_m
    if weight is not None:
        rule["override"]["weight"] = weight

    return rule
