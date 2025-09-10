# cbf/tools/llm_kind_classifier.py
from __future__ import annotations
import os
import re
import shlex
import subprocess
from typing import Iterable

PROMPT_TEMPLATE = """You are a strict labeler.
Given an object NAME and an optional DESCRIPTION, choose EXACTLY ONE label from the ALLOWED_KINDS list.

Selection rules (very important):
- Output the single label ONLY. No punctuation, no explanations.
- Prefer the MOST SPECIFIC matching label. Generic buckets (e.g., "object") are LAST RESORTS and should be avoided if any specific label fits.
- If none fits, output exactly: unknown

ALLOWED_KINDS: {allowed}
NAME: {name}
DESCRIPTION: {desc}

Examples:
- NAME: coffee -> liquid
- NAME: cup of water -> liquid
- NAME: MacBook -> electronic
- NAME: chef knife -> knife

Your answer (single token):"""


def _strip(s: str) -> str:
    return (s or "").strip()

def _first_token(s: str) -> str:
    m = re.search(r"[A-Za-z0-9_\-]+", s or "")
    return m.group(0) if m else ""

def _normalize_kind(ans: str, allowed_lower: set[str]) -> str:
    a = _first_token(ans).lower()
    return a if a in allowed_lower else "unknown"

def _call_ollama_cli(model: str, prompt: str, timeout: float = 20.0) -> str:
    bin_path = os.getenv("OLLAMA_BIN", "ollama")
    cmd = [bin_path, "run", model]
    try:
        p = subprocess.run(
            cmd, input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=timeout, check=False
        )
        out = p.stdout.decode("utf-8", errors="ignore")
        if not out.strip():
            out = (out + "\n" + p.stderr.decode("utf-8", errors="ignore")).strip()
        return out.strip()
    except Exception as e:
        return f"__ERR__ {type(e).__name__}: {e}"

def classify_kind_via_llm(
    name: str,
    allowed_kinds: Iterable[str],
    description: str | None = None,
    *,
    backend: str | None = None,
    model: str | None = None,
    timeout: float = 20.0,
) -> str:
    """
    Returns one of allowed_kinds (case-insensitive), or 'unknown' if no match / error.
    Backends:
      - 'cli'  : uses 'ollama run <model>' (no HTTP)
      - 'none' : skip LLM, always 'unknown'
    """
    allowed = [str(k).strip() for k in allowed_kinds if str(k).strip()]
    if not allowed:
        return "unknown"
    allowed_lower = {k.lower() for k in allowed}

    backend = (backend or os.getenv("CBF_LLM_BACKEND", "cli")).lower()
    model = model or os.getenv("CBF_LLM_MODEL", "llama3.1:8b")

    prompt = PROMPT_TEMPLATE.format(
        allowed=", ".join(sorted(allowed)),
        name=_strip(name),
        desc=_strip(description or ""),
    )

    if backend == "none":
        return "unknown"

    raw = _call_ollama_cli(model, prompt, timeout=timeout)
    if raw.startswith("__ERR__"):
        return "unknown"
    return _normalize_kind(raw, allowed_lower)
