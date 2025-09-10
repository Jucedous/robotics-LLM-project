from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import json


def load_specs(path: str | Path) -> List[Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Scene JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Scene JSON must be a list of object specs.")
    return data




def save_specs(path: str | Path, specs: List[Dict]) -> None:
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        json.dump(specs, f, indent=2)