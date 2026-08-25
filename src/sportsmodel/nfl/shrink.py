from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ShrinkParams:
    start: float = 0.75
    floor: float = 0.2
    decay: float = 0.25

def w_curve(week: int, params: ShrinkParams) -> float:
    if week > 18:
        return params.floor
    w = params.floor + (params.start - params.floor) * math.exp(-params.decay * (week - 1))
    return min(max(w, params.floor), params.start)

def shrink(model_value: float, market_value, week: int, params: ShrinkParams) -> float:
    if market_value is None:
        return model_value
    w = w_curve(week, params)
    return (1 - w) * model_value + w * market_value
