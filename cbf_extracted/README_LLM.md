
# LLM-Driven Hazards (No Hard-Coded Rules)

This package removes all hard-coded semantic hazard rules. The LLM analyzes *any* scene objects and returns risks + optional critical boolean expressions. Those are compiled safely and fed into a CBF metric that has **no defaults**.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U requests numpy matplotlib

# Set your key (env takes precedence)
export OPENAI_API_KEY=YOUR_KEY_HERE

python examples/run_llm_semantics.py examples/scene1.json


## Files added
- `cbf/semantics_runtime.py` — LLM bridge, safe expression engine, scene → risks → concrete pairings.
- `cbf/cbf_safety_metrics_llm.py` — CBF hazard metric that only uses injected rules; **no built-in semantics**.
- `examples/run_llm_semantics.py` — runnable example using your scenes.

## Notes
- Your original `cbf_safety_metrics.py` is left intact; this flow calls the **LLM** version instead.
- The runner will read `OPENAI_API_KEY` from env or `config/openai_key.txt`.
