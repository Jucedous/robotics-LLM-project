from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Optional, Tuple
import requests
from cbf.preferences import PreferenceStore

def _norm_base_v1(base: Optional[str]) -> Optional[str]:
    if not base:
        return None
    b = base.strip().rstrip("/")
    b = re.sub(r"/chat/completions/?$", "", b)
    b = re.sub(r"/completions/?$", "", b)
    if not re.search(r"/v\d+($|/)", b):
        b += "/v1"
    return b.rstrip("/")

def _compose_completions_url(base: Optional[str]) -> str:
    b = _norm_base_v1(base)
    if not b:
        return "https://api.openai.com/v1/completions"
    return f"{b}/completions"

def _cfg_field(cfg: Any, name: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)

def _read_api_key(project_root: str) -> Optional[str]:
    for k in ("OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_TOKEN"):
        v = os.environ.get(k)
        if v:
            return v.strip()
    for p in (
        os.path.join(project_root, "config", "openai_key.txt"),
        os.path.join(project_root, "cbf_extracted", "config", "openai_key.txt"),
    ):
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s or s.startswith("#"):
                        continue
                    if "OPENAI_API_KEY" in s and "=" in s:
                        s = s.split("=", 1)[1].strip().strip('"').strip("'")
                    return s
    return None

def _objects_summary(objects: List[Any]) -> List[Dict[str, str]]:
    return [{"name": str(getattr(o, "name", "")), "kind": str(getattr(o, "kind", "object"))} for o in objects]

def _map_name_or_kind(s: str, object_names: List[str]) -> Tuple[str, str]:
    s_clean = (s or "").strip()
    for nm in object_names:
        if s_clean.lower() == nm.lower():
            return ("name", nm)
    return ("kind", s_clean.lower())

def _normalize_selector(sel: Dict[str, str], objects: List[Any]) -> Dict[str, str]:
    by, val = (sel or {}).get("by"), (sel or {}).get("value", "")
    if by == "name":
        for o in objects:
            if getattr(o, "name", "").lower() == val.lower():
                return {"by": "name", "value": o.name}
        if any(getattr(o, "kind", "").lower() == val.lower() for o in objects):
            return {"by": "kind", "value": val.lower()}
        return {"by": "name", "value": val}
    if by in ("kind", "tag"):
        return {"by": by, "value": val}
    return {"by": "kind", "value": val or "object"}

def _extract_json_block(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end   = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model did not return JSON.")
    return json.loads(text[start:end+1])

_PROMPT_TEMPLATE = """You are labeling a user's feedback sentence about safety relations between objects or kinds.

Return ONLY valid JSON with this exact schema:
{
  "verdict": "dangerous" | "not_dangerous",
  "pairs": [
    {"A": {"text": string}, "B": {"text": string}, "directional": boolean, "relation": string|null}
  ]
}

Guidelines:
- Choose the user's final intent if the sentence mixes negation.
- If the sentence implies direction (over/under/inside/on/below), set directional=true and relation to "over"|"under"|"inside"|"on"|"below".
- If it expresses proximity (near/close/next to), relation="near" and directional=false.
- If multiple pairs are referenced, include all.

User feedback: {feedback}

Available objects (names and kinds):
{objects}
"""

def label_and_store_feedback(
    text: str,
    objects: List[Any],
    cfg: Any,
    user_id: str,
    store: Optional[PreferenceStore] = None,
    model_fallback: str = "gpt-3.5-turbo-instruct"
) -> int:
    if not text or not text.strip():
        raise ValueError("Empty feedback.")
    project_root = os.getcwd()
    api_key = _cfg_field(cfg, "api_key") or _read_api_key(project_root)
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY (env) or config/openai_key.txt")
    mode = os.environ.get("FEEDBACK_API_MODE", "prompt").lower()
    if mode != "prompt":
        mode = "prompt"
    model = (
        os.environ.get("FEEDBACK_MODEL")
        or os.environ.get("LLM_FEEDBACK_MODEL")
        or _cfg_field(cfg, "model")
        or model_fallback
    )
    raw_base = (
        os.environ.get("FEEDBACK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("LLM_API_URL")
        or _cfg_field(cfg, "api_url")
        or _cfg_field(cfg, "base_url")
    )
    url = _compose_completions_url(raw_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    objects_ctx = _objects_summary(objects)
    prompt = _PROMPT_TEMPLATE.format(
        feedback=text.strip(),
        objects=json.dumps(objects_ctx, ensure_ascii=False, indent=2)
    )
    body = {
        "model": model,
        "prompt": prompt,
        "temperature": 0,
        "max_tokens": 512,
    }
    data_json: Dict[str, Any]
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        if resp.status_code >= 400:
            raise RuntimeError(f"{resp.status_code} {resp.text}")
        payload = resp.json()
        text_out = ""
        if isinstance(payload, dict) and "choices" in payload and payload["choices"]:
            c0 = payload["choices"][0]
            text_out = c0.get("text") or c0.get("message", {}).get("content", "")
        if not text_out:
            text_out = payload.get("output_text", "")
        if not text_out:
            raise RuntimeError(f"Unexpected completions payload: {payload}")
        data_json = _extract_json_block(text_out)
    except Exception as e:
        verdict = "not_dangerous" if re.search(r"\bnot\b.*\bdanger", text, flags=re.I) else ("dangerous" if re.search(r"\bdanger", text, flags=re.I) else "not_dangerous")
        toks = re.findall(r"[A-Za-z_]+", text)
        pairs = []
        if len(toks) >= 2:
            pairs.append({"A":{"text":toks[0]}, "B":{"text":toks[1]}, "directional": False, "relation": "near"})
        else:
            raise RuntimeError(f"Feedback labeling failed (HTTP/completions): {e}")
        data_json = {"verdict": verdict, "pairs": pairs}
    object_names = [o["name"] for o in objects_ctx]
    rules_to_write: List[Dict[str, Any]] = []
    for p in data_json.get("pairs") or []:
        Atext = (p.get("A") or {}).get("text", "")
        Btext = (p.get("B") or {}).get("text", "")
        A_type, A_val = _map_name_or_kind(Atext, object_names)
        B_type, B_val = _map_name_or_kind(Btext, object_names)
        directional = bool(p.get("directional", False))
        relation = (p.get("relation") or None)
        if isinstance(relation, str):
            relation = relation.strip().lower() or None
        cond = None
        if relation in ("over","above","on","on_top","on-top"):
            cond = "Az > Bz"
        elif relation in ("under","below","beneath"):
            cond = "Az < Bz"
        present = (str(data_json.get("verdict","")).strip().lower() == "dangerous")
        rule = {
            "user_id": user_id,
            "selectors": {
                "A": _normalize_selector({"by": A_type, "value": A_val}, objects),
                "B": _normalize_selector({"by": B_type, "value": B_val}, objects),
            },
            "directional": directional,
            "relation": relation,
            "condition_expr": cond,
            "override": {"present": present},
            "reason": text.strip(),
        }
        rules_to_write.append(rule)
    store = store or PreferenceStore()
    return store.add_rules(rules_to_write)
