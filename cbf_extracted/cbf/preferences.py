# cbf/preferences.py
from __future__ import annotations
import json, time, uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

def _default_path() -> Path:
    here = Path(__file__).resolve().parent
    return (here.parent / "config" / "user_prefs.json").resolve()

class PreferenceStore:
    """
    JSON-backed store:
      { "rules": [ { ...rule... }, ... ] }
    A rule shape:
      {
        "id": "...",
        "user_id": "zilinzhao",
        "created_at": "...Z",
        "selectors": {
          "A": {"by":"name|kind","value":"..."},
          "B": {"by":"name|kind","value":"..."}
        },
        "directional": true|false,
        "relation": "over|under|near|contact|inside|touching|aligned|...",
        "condition_expr": "Az > Bz" | "Az < Bz" | null,
        "override": {"present": true|false, "soft_clearance_m": float?, "weight": float?},
        "reason": "original feedback sentence"
      }
    """
    def __init__(self, path: Optional[str|Path]=None):
        self.path = Path(path) if path else _default_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({"rules": []})

    def _read(self) -> Dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def _write(self, data: Dict[str, Any]) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def add_rule(self, rule: Dict[str, Any]) -> str:
        db = self._read()
        rule = dict(rule)
        rule.setdefault("id", uuid.uuid4().hex)
        rule.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        db.setdefault("rules", []).append(rule)
        self._write(db)
        return rule["id"]

    def add_rules(self, rules: List[Dict[str, Any]]) -> int:
        db = self._read()
        for r in rules:
            r.setdefault("id", uuid.uuid4().hex)
            r.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            db.setdefault("rules", []).append(r)
        self._write(db)
        return len(rules)

    def list_rules(self, user_id: str) -> List[Dict[str, Any]]:
        return [r for r in self._read().get("rules", []) if r.get("user_id") == user_id]

    def clear_user(self, user_id: str) -> int:
        db = self._read()
        before = len(db.get("rules", []))
        db["rules"] = [r for r in db.get("rules", []) if r.get("user_id") != user_id]
        self._write(db)
        return before - len(db["rules"])
