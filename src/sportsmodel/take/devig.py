"""N-way power-method de-vig and American price helpers."""
from __future__ import annotations

import math

_TOL = 1e-9


def american_to_decimal(price: int) -> float:
    a = float(price)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def novig_power_n(prices: list[int]) -> list[float]:
    """Power-method no-vig probabilities for an n-way market.

    Find k >= 1 with sum(p_i^k) = 1 (bisection on [1, 10], as serving.board.novig_power does) and return
    p_i^k. Falls back to proportional when there is no overround (sum <= 1) or the root lies beyond k = 10."""
    p = [1.0 / american_to_decimal(x) for x in prices]
    total = sum(p)
    proportional = [x / total for x in p]
    if total <= 1.0 or sum(x ** 10 for x in p) - 1.0 >= 0.0:
        return proportional
    lo, hi = 1.0, 10.0
    k = lo
    for _ in range(200):
        k = (lo + hi) / 2
        f = sum(x ** k for x in p) - 1.0
        if abs(f) < 1e-12:
            break
        if f > 0:      # still over-round: need a larger k
            lo = k
        else:
            hi = k
    return [x ** k for x in p]


def decimal_to_american_safe(dec: float) -> int:
    """The smallest-for-the-book American price whose decimal odds are >= `dec` (rounds toward the
    bettor-safe side). A tiny tolerance absorbs float noise so an exact price maps to itself."""
    if not dec > 1.0:
        raise ValueError(f"decimal odds must exceed 1: {dec}")
    if dec >= 2.0:
        x = 100.0 * (dec - 1.0)
        r = round(x)
        return int(r if abs(x - r) < _TOL else math.ceil(x))
    x = 100.0 / (dec - 1.0)
    r = round(x)
    return -int(r if abs(x - r) < _TOL else math.floor(x))
