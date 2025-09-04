# hazard_policy.py
import os, json, hashlib
from functools import lru_cache
from typing import List, Dict, Any, Tuple
from cbf_core import ObjectState

# ------------------------
# Public constants
# ------------------------
DEFAULT_FALLBACK_POLICY: Dict[str, Any] = {
    "version": "cbf-hazards-v1",
    "pairs": [
        {
            "A_kind": "liquid",
            "B_kind": "electronic",
            "clearance_m": 0.15,
            "over_rule": {"enable": True, "xy_margin_m": 0.05, "z_gap_max_m": 0.25},
            "severity": "critical",
            "rationale": "Spill risk into electronics.",
        },
        {
            "A_kind": "sharp",
            "B_kind": "human",
            "clearance_m": 0.10,
            "over_rule": {"enable": False, "xy_margin_m": 0.0, "z_gap_max_m": 0.0},
            "severity": "warning",
            "rationale": "Cut/puncture risk.",
        },
    ],
    "aliases": {}
}


# ------------------------
# LLM plumbing (replace _call_llm)
# ------------------------

def _build_prompt(scene_json: str) -> str:
    """System + task in one string; instruct STRICT JSON output."""
    return (
        "SYSTEM:\n"
        "You are a robotics safety assistant. Output STRICT JSON only (no markdown or prose).\n"
        "Infer hazard pairings for object-only manipulation scenes. Do not invent objects.\n"
        "All units are meters; use conservative but realistic thresholds.\n\n"
        "USER:\n"
        "Scene Objects (JSON):\n"
        f"{scene_json}\n\n"
        "Task:\n"
        "1) Infer semantic kinds/tags for listed objects when possible as aliases.\n"
        "2) Produce hazard pairings likely relevant to manipulation scenes.\n"
        "   For each pair object-kind A vs B, provide:\n"
        "     - A_kind, B_kind (strings from: liquid, electronic, sharp, human,\n"
        "       fragile, heavy, hot, plastic)\n"
        "     - clearance_m (float)\n"
        "     - over_rule: { enable(bool), xy_margin_m(float), z_gap_max_m(float) }\n"
        "       (use for 'A over B' hazards when relevant, e.g., liquid over electronics)\n"
        "     - severity in {\"notice\",\"warning\",\"critical\"}\n"
        "     - rationale (short string)\n"
        "3) Include an aliases map: object_name -> list of semantic kinds/tags.\n\n"
        "Output JSON ONLY with this exact schema keys:\n"
        "{ \"version\":\"cbf-hazards-v1\", \"pairs\":[ ... ], \"aliases\": { ... } }"
    )


def _call_llm(prompt: str) -> str:
    """
    Replace this with your LLM client call.
    MUST return a raw JSON string conforming to the schema (no extra text).
    Supports MOCK mode via env: HAZARD_POLICY_MOCK=1
    """
    # --- MOCK mode for development / offline runs ---
    if os.getenv("HAZARD_POLICY_MOCK", "0") == "1":
        mock = {
            "version": "cbf-hazards-v1",
            "pairs": [
                {
                    "A_kind": "liquid",
                    "B_kind": "electronic",
                    "clearance_m": 0.15,
                    "over_rule": {"enable": True, "xy_margin_m": 0.05, "z_gap_max_m": 0.25},
                    "severity": "critical",
                    "rationale": "Liquids can spill into electronics causing short-circuit.",
                },
                {
                    "A_kind": "heavy",
                    "B_kind": "fragile",
                    "clearance_m": 0.20,
                    "over_rule": {"enable": False, "xy_margin_m": 0.0, "z_gap_max_m": 0.0},
                    "severity": "warning",
                    "rationale": "Crush risk.",
                },
                {
                    "A_kind": "sharp",
                    "B_kind": "human",
                    "clearance_m": 0.10,
                    "over_rule": {"enable": False, "xy_margin_m": 0.0, "z_gap_max_m": 0.0},
                    "severity": "warning",
                    "rationale": "Cut risk.",
                },
            ],
            "aliases": {
                # Example: let the policy tag common names
                "mug": ["cup", "liquid"],
                "laptop": ["electronic", "fragile"]
            }
        }
        return json.dumps(mock)

    # --- Example: OpenAI (uncomment and fill api key/model) ---
    # import openai
    # openai.api_key = os.getenv("OPENAI_API_KEY")
    # resp = openai.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.0
    # )
    # return resp.choices[0].message.content

    # --- Example: Ollama (uncomment if running local model) ---
    # import requests
    # r = requests.post("http://localhost:11434/api/generate",
    #                   json={"model":"llama3", "prompt": prompt, "stream": False})
    # data = r.json()
    # return data.get("response","{}")

    raise NotImplementedError("Wire _call_llm to your provider or set HAZARD_POLICY_MOCK=1.")


# ------------------------
# Scene hashing & summary
# ------------------------

def _scene_summary(objects: List[ObjectState]) -> dict:
    return {
        "objects": [
            {
                "name": o.name,
                "kind": o.kind,
                "tags": list(o.tags),
                "radius_m": float(o.sphere.radius),
            }
            for o in objects
        ]
    }

def _stable_key(objects: List[ObjectState]) -> str:
    items = [(o.name, o.kind, tuple(sorted(o.tags))) for o in sorted(objects, key=lambda x: x.name)]
    raw = json.dumps(items, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ------------------------
# Policy validation & clamps
# ------------------------

_ALLOWED_SEVERITIES = {"notice", "warning", "critical"}
_ALLOWED_KINDS = {"liquid", "electronic", "sharp", "human", "fragile", "heavy", "hot", "plastic"}

def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(float(x), hi))

def _validate_pair(p: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Validate and sanitize one pair entry."""
    try:
        A = str(p.get("A_kind", "")).strip()
        B = str(p.get("B_kind", "")).strip()
        if A == "" or B == "":
            return False, {}
        # Accept any strings but prefer allowed set; keep user extensibility.
        # If you want to hard-enforce the set, uncomment:
        # if A not in _ALLOWED_KINDS or B not in _ALLOWED_KINDS:
        #     return False, {}

        clearance = float(p.get("clearance_m", 0.0))
        clearance = _clamp(clearance, 0.0, 2.0)  # 0–2 m sensible range

        over = p.get("over_rule", {}) or {}
        over_enable = bool(over.get("enable", False))
        xy_margin = _clamp(float(over.get("xy_margin_m", 0.0)), 0.0, 1.0)
        z_gap = _clamp(float(over.get("z_gap_max_m", 0.0)), 0.0, 1.0)

        sev = str(p.get("severity", "notice")).lower()
        if sev not in _ALLOWED_SEVERITIES:
            sev = "notice"

        rationale = str(p.get("rationale", "")).strip()

        return True, {
            "A_kind": A,
            "B_kind": B,
            "clearance_m": clearance,
            "over_rule": {"enable": over_enable, "xy_margin_m": xy_margin, "z_gap_max_m": z_gap},
            "severity": sev,
            "rationale": rationale,
        }
    except Exception:
        return False, {}

def _validate_policy(policy: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure schema correctness and safe numeric ranges; fall back if invalid."""
    if not isinstance(policy, dict):
        return DEFAULT_FALLBACK_POLICY
    if policy.get("version") != "cbf-hazards-v1":
        # accept missing version by injecting
        policy["version"] = "cbf-hazards-v1"

    pairs = policy.get("pairs", [])
    aliases = policy.get("aliases", {})

    out_pairs: List[Dict[str, Any]] = []
    if isinstance(pairs, list):
        for p in pairs:
            ok, pp = _validate_pair(p if isinstance(p, dict) else {})
            if ok:
                out_pairs.append(pp)

    # basic alias map sanitize
    out_aliases: Dict[str, List[str]] = {}
    if isinstance(aliases, dict):
        for name, kinds in aliases.items():
            try:
                klist = [str(k).strip() for k in kinds if str(k).strip()]
                out_aliases[str(name)] = klist[:8]  # cap per-object alias kinds
            except Exception:
                continue

    if not out_pairs:
        return DEFAULT_FALLBACK_POLICY

    return {"version": "cbf-hazards-v1", "pairs": out_pairs, "aliases": out_aliases}


# ------------------------
# Cached retrieval (public API)
# ------------------------

@lru_cache(maxsize=64)
def get_hazard_policy_via_llm_cached(key: str, scene_str: str) -> Dict[str, Any]:
    """
    Low-level cached fetch using a stable key and a JSON-serialized scene summary.
    """
    prompt = _build_prompt(scene_str)
    raw = _call_llm(prompt)
    try:
        policy = json.loads(raw)
    except Exception:
        policy = {}
    return _validate_policy(policy)

def get_hazard_policy_via_llm(objects: List[ObjectState]) -> Dict[str, Any]:
    """
    High-level API: builds a stable cache key from the objects, summarizes the scene,
    fetches policy from LLM (or mock), validates, and returns a policy dict.
    """
    key = _stable_key(objects)
    scene_json = json.dumps(_scene_summary(objects), sort_keys=True)
    return get_hazard_policy_via_llm_cached(key, scene_json)


# ------------------------
# Optional: quick CLI test
# ------------------------
if __name__ == "__main__":
    # Minimal smoke test (uses MOCK if HAZARD_POLICY_MOCK=1)
    class _S:  # tiny stand-in so this file can run standalone if needed
        def __init__(self, name, kind="object", tags=()):
            self.name = name
            self.kind = kind
            self.tags = tuple(tags)
            class _Sphere:
                radius = 0.05
            self.sphere = _Sphere()

    objs = [ _S("laptop", kind="electronic"), _S("mug", kind="object") ]
    policy = get_hazard_policy_via_llm(objs)
    print(json.dumps(policy, indent=2))
