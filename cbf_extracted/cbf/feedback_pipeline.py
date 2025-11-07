from __future__ import annotations
import json, os, re
from typing import Any, Dict, List, Optional, Tuple
import requests
from pathlib import Path
from cbf.preferences import PreferenceStore

def _cfg_field(cfg: Any, name: str, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)

def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _read_api_key_from_disk() -> Optional[str]:
    for env in ("OPENAI_API_KEY", "OPENAI_APIKEY", "OPENAI_TOKEN"):
        v = os.environ.get(env)
        if v:
            return v.strip()
    candidates = [
        _project_root() / "config" / "openai_key.txt",
        _project_root() / "cbf_extracted" / "config" / "openai_key.txt",
    ]
    for p in candidates:
        try:
            if p.exists():
                t = p.read_text(encoding="utf-8").strip()
                if t:
                    return t
        except Exception:
            pass
    return None

def _objects_summary(objects: List[Any]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for i, o in enumerate(objects):
        name = str(getattr(o, "name", getattr(o, "id", f"obj{i}")))
        kind = str(getattr(o, "kind", "object"))
        out.append({"name": name, "kind": kind})
    return out

def _canon(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.strip().lower())

def _normalize_selector(sel: Dict[str, str], objects: List[Any]) -> Dict[str, str]:
    by = (sel.get("by") or "").strip().lower()
    val = str(sel.get("value") or "").strip()
    if not val:
        return {"by": "kind", "value": "object"}
    names = { _canon(getattr(o, "name", "")): getattr(o, "name", "") for o in objects }
    kinds = { _canon(getattr(o, "kind", "")): getattr(o, "kind", "") for o in objects }
    cval = _canon(val)
    if cval in names:
        return {"by": "name", "value": names[cval]}
    if cval in kinds:
        return {"by": "kind", "value": kinds[cval]}
    for k, human in names.items():
        if cval and k and cval in k:
            return {"by": "name", "value": human}
    for k, human in kinds.items():
        if cval and k and cval in k:
            return {"by": "kind", "value": human}
    if by in ("name", "kind"):
        return {"by": by, "value": val}
    return {"by": "kind", "value": val}

def _iter_balanced_json_objects(text: str):
    stack = []
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    yield text[start:i+1]

def _clean_json_quiet(s: str) -> str:
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s

def _extract_json_block(text: str) -> Dict[str, Any]:
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", t)
    candidates: List[str] = []
    if m:
        candidates.append(m.group(1))
    candidates.extend(_iter_balanced_json_objects(t))
    if not candidates:
        raise ValueError("No JSON object found in model output.")
    def score(s: str):
        return (0 if ('"present"' in s) else 1, -len(s))
    candidates.sort(key=score)
    last_err: Optional[Exception] = None
    for c in candidates:
        try:
            return json.loads(_clean_json_quiet(c))
        except Exception as e:
            last_err = e
    raise ValueError(f"Could not parse JSON from model output. Last error: {last_err}")

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
    base = _norm_base_v1(base) or "https://api.openai.com/v1"
    return base + "/completions"

def _post_openai_chat_json(cfg: Any, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    api_key = _cfg_field(cfg, "api_key") or _read_api_key_from_disk()
    if not api_key:
        raise RuntimeError("Missing OpenAI API key.")
    api_url = _cfg_field(cfg, "api_url", "https://api.openai.com/v1/chat/completions")
    model   = _cfg_field(cfg, "model", "gpt-4o-mini")
    temperature = float(_cfg_field(cfg, "temperature", 0.0))
    timeout_s   = int(_cfg_field(cfg, "timeout_s", 60))
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    }
    r = requests.post(api_url, headers=headers, json=body, timeout=timeout_s)
    r.raise_for_status()
    content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)

_CHAT_SYSTEM = (
    "Return ONLY a single JSON object with keys:\n"
    '  "present": true or false\n'
    '  "pairs": [ { "A": {"text": string}, "B": {"text": string} } ]\n'
    "Decide from the sentence whether the relation between A and B is dangerous (present=true) or not (present=false).\n"
    "Use the user words to extract A and B as names or kinds that exist in the provided list."
)

_CHAT_USER_TEMPLATE = (
    "User feedback:\n{feedback}\n\n"
    "Available objects (names and kinds):\n{objects}\n"
)

_PROMPT_TEMPLATE = """Return ONLY JSON with fields:
{
  "present": true|false,
  "pairs": [
    { "A": {"text": "<name or kind>"}, "B": {"text": "<name or kind>"} }
  ]
}

User feedback:
{feedback}

Available objects (names and kinds):
{objects}
"""

def _guess_pairs_from_text(text: str, objects: List[Any]) -> List[Dict[str, Any]]:
    t = text.lower()
    names = [(getattr(o, "name", ""), getattr(o, "name", "").lower()) for o in objects]
    kinds = [(getattr(o, "kind", ""), getattr(o, "kind", "").lower()) for o in objects]
    mentions: List[str] = []
    for human, l in names:
        if human and l and l in t and human not in mentions:
            mentions.append(human)
    for human, l in kinds:
        if human and l and l in t and human not in mentions:
            mentions.append(human)
    pairs: List[Dict[str, Any]] = []
    if len(mentions) >= 2:
        pairs.append({"A": {"text": mentions[0]}, "B": {"text": mentions[1]}})
    return pairs

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
    objects_ctx = _objects_summary(objects)
    mode = os.environ.get("FEEDBACK_API_MODE", "prompt").lower()
    data_json: Dict[str, Any]
    if mode == "direct" or text.lstrip().startswith("{"):
        data_json = _extract_json_block(text)
    else:
        try:
            user_prompt = _CHAT_USER_TEMPLATE.format(
                feedback=text.strip(),
                objects=json.dumps(objects_ctx, ensure_ascii=False, indent=2),
            )
            data_json = _post_openai_chat_json(cfg, _CHAT_SYSTEM, user_prompt)
        except Exception as e_chat:
            try:
                base = (
                    os.environ.get("FEEDBACK_BASE_URL")
                    or os.environ.get("OPENAI_BASE_URL")
                    or os.environ.get("LLM_API_URL")
                    or _cfg_field(cfg, "api_url")
                    or _cfg_field(cfg, "base_url")
                )
                url = _compose_completions_url(base)
                api_key = _cfg_field(cfg, "api_key") or _read_api_key_from_disk()
                if not api_key:
                    raise RuntimeError("Missing OpenAI API key.")
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                prompt = _PROMPT_TEMPLATE.format(
                    feedback=text.strip(),
                    objects=json.dumps(objects_ctx, ensure_ascii=False, indent=2),
                )
                model = (
                    os.environ.get("FEEDBACK_MODEL")
                    or os.environ.get("LLM_FEEDBACK_MODEL")
                    or _cfg_field(cfg, "model")
                    or model_fallback
                )
                body = {"model": model, "prompt": prompt, "temperature": 0, "max_tokens": 512}
                resp = requests.post(url, headers=headers, json=body, timeout=int(_cfg_field(cfg, "timeout_s", 60)))
                resp.raise_for_status()
                payload = resp.json()
                text_out = ""
                if isinstance(payload, dict) and "choices" in payload and payload["choices"]:
                    c0 = payload["choices"][0]
                    text_out = (c0.get("text") or (c0.get("message") or {}).get("content") or "")
                if not text_out:
                    text_out = payload.get("output_text", "")
                if not text_out:
                    raise RuntimeError(f"Unexpected completions payload: {payload}")
                data_json = _extract_json_block(text_out)
            except Exception as e_comp:
                raise RuntimeError(f"Feedback labeling failed (chat:{e_chat}) (completions:{e_comp})")
    if "present" not in data_json:
        raise ValueError("Model output missing 'present'.")
    present = bool(data_json["present"])
    pairs = data_json.get("pairs") or []
    if not isinstance(pairs, list):
        raise ValueError("Model output 'pairs' must be a list.")
    if not pairs:
        pairs = _guess_pairs_from_text(text, objects)
        if not pairs:
            raise RuntimeError("No A/B pairs found.")
    rules_to_write: List[Dict[str, Any]] = []
    for p in pairs:
        A = p.get("A") or {}
        B = p.get("B") or {}
        A_text = str(A.get("text") or A.get("name") or A.get("kind") or "").strip()
        B_text = str(B.get("text") or B.get("name") or B.get("kind") or "").strip()
        if not A_text or not B_text:
            continue
        rule = {
            "user_id": user_id,
            "selectors": {
                "A": _normalize_selector({"by": "name", "value": A_text}, objects),
                "B": _normalize_selector({"by": "name", "value": B_text}, objects),
            },
            "directional": False,
            "relation": None,
            "condition_expr": None,
            "override": {"present": present},
            "reason": text.strip(),
        }
        rules_to_write.append(rule)
    if not rules_to_write:
        raise RuntimeError("No valid rules to save.")
    store = store or PreferenceStore()
    return store.add_rules(rules_to_write)
