from __future__ import annotations
import numpy as np
from typing import Tuple


def get_score_color(score_0_to_5: float) -> Tuple[float, float, float]:
    s = float(np.clip(score_0_to_5 / 5.0, 0.0, 1.0))
    r = (1.00 - s) * 0.98 + s * 0.85
    g = (1.00 - s) * 0.85 + s * 0.98
    b = (1.00 - s) * 0.85 + s * 0.90
    return (r, g, b)