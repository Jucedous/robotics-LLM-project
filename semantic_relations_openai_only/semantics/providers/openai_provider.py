import json, re, os
from typing import Dict, Any
from openai import OpenAI

JSON_PATTERN = r'\{(?:[^{}]|\{[^{}]*\})*\}'

def _extract_json(text: str) -> Dict[str, Any]:
    candidates = re.findall(JSON_PATTERN, text, flags=re.DOTALL)
    if not candidates:
        raise ValueError("No JSON object found in model output.")
    return json.loads(max(candidates, key=len))

class OpenAIProvider:
    """Pure OpenAI Chat Completions provider — no heuristics/fallbacks."""
    def __init__(self, model: str = "gpt-4.1-mini", timeout: float = 180.0):
        self.client = OpenAI(api_key = "[REDACTED_OPENAI_KEY]", timeout=timeout)
        self.model = model

        # Load external system prompt
        prompt_path = os.path.join(os.path.dirname(__file__), "../prompts/system_semantic_safety.md")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.system = f.read()

    def analyze(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        scene_json = json.dumps(scene, ensure_ascii=False, indent=2)
        messages = [
            {"role": "system", "content": self.system},
            {"role": "user",   "content": "Scene:\n" + scene_json},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
                max_tokens=800,
            )
            text = resp.choices[0].message.content or ""
        except Exception:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0,
                max_tokens=800,
            )
            text = resp.choices[0].message.content or ""

        parsed = json.loads(text) if text.strip().startswith("{") else _extract_json(text)
        return {
            "pairwise_relations": parsed.get("pairwise_relations", []),
            "provider_meta": {"name": "OpenAIProvider", "model": self.model},
        }
