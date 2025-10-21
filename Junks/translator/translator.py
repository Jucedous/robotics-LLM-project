from dataclasses import dataclass
from typing import Dict, List, Any
import sympy as sp

def _safe_sympify(expr: str, symbols: Dict[str, sp.Symbol]) -> sp.Expr:
    if not set(expr) <= ALLOWED:
        raise ValueError("Disallowed characters in expression.")
    return sp.sympify(expr, locals={**ALLOWED_FUNCS, **symbols})

@dataclass
class CBFConstraint:
    id: str
    alpha: float
    symbols: List[sp.Symbol]
    h_expr: sp.Expr
    dh_dx_expr: List[sp.Expr]
    symbol_names: List[str]

    def h(self, x: Dict[str, float]) -> float:
        subs = {sp.Symbol(k): float(x[k]) for k in self.symbol_names}
        return float(self.h_expr.evalf(subs=subs))

    def dh_dx(self, x: Dict[str, float]) -> List[float]:
        subs = {sp.Symbol(k): float(x[k]) for k in self.symbol_names}
        return [float(e.evalf(subs=subs)) for e in self.dh_dx_expr]

def compile_hazards(spec: Dict[str, Any]) -> List[CBFConstraint]:
    constraints = []
    for hz in spec["hazards"]:
        cbf_syms = {name: sp.symbols(name) for name in hz["symbol_table"]}
        h_expr = _safe_sympify(hz["h_expression"], cbf_syms)
        state_syms = [name for name, bind in hz["state_bindings"].items()
                      if not isinstance(bind, (int, float))]
        symbols = [cbf_syms[n] for n in state_syms]
        dh_dx = [sp.diff(h_expr, s) for s in symbols]

        constraints.append(CBFConstraint(
            id=hz["id"],
            alpha=float(hz["alpha"]),
            symbols=symbols,
            h_expr=h_expr,
            dh_dx_expr=dh_dx,
            symbol_names=state_syms
        ))
    return constraints

def materialize_state(hz: Dict[str, Any], get_value) -> Dict[str, float]:
    """
    get_value(accessor: str) -> float
    Resolves accessors like 'pose(cup).z' using your sim/robot state.
    Numeric bindings (e.g., margin: 0.03) are passed through.
    """
    x = {}
    for name, binding in hz["state_bindings"].items():
        x[name] = float(binding) if isinstance(binding, (int, float)) else float(get_value(binding))
    return x
