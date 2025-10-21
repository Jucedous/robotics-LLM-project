from __future__ import annotations
from typing import Dict
import requests

# We reuse the LLMConfig shape from cbf.semantics_runtime.
# (No hard dependency here; we only access fields dynamically.)

_SYSTEM = """You classify an object NAME into a concise kind label for robot scene safety.
Return STRICT JSON only:

{"kind": "<lowercase kind, 1-2 words max>"}

Guidance (not rules): examples include "electronic", "liquid", "sharp", "human",
"tool", "container", "fragile", "furniture", "appliance", "food".
If ambiguous, return "object". No extra fields, no commentary.
"""

_USER_TMPL = """Name: {name}
Return JSON with only the "kind" field as instructed."""


def classify_kind_llm(name: str, cfg) -> str:
    """
    Ask the LLM to classify an object's kind from its NAME only.
    Returns a lowercase kind string; falls back to 'object' on error.
    """
    try:
        headers = {"Authorization": f"Bearer {cfg.api_key}", "Content-Type": "application/json"}
        body: Dict = {
            "model": getattr(cfg, "model", "gpt-4o-mini"),
            "temperature": getattr(cfg, "temperature", 0.0),
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": _USER_TMPL.format(name=name)},
            ],
        }
        api_url = getattr(cfg, "api_url", "https://api.openai.com/v1/chat/completions")
        timeout_s = int(getattr(cfg, "timeout_s", 60))
        r = requests.post(api_url, headers=headers, json=body, timeout=timeout_s)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        # Expect {"kind": "..."}
        import json as _json
        data = _json.loads(content)
        k = str(data.get("kind", "")).strip().lower()
        if not k:
            return "object"
        # keep it short
        return k[:40]
    except Exception as e:
        print(f"[kind_classifier] fallback to 'object' due to: {e}")
        return "object"
