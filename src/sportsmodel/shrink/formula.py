"""The shrink formula and the market probability it shrinks toward. PURE -- no IO.

From docs/superpowers/specs/2026-10-13-shrink-props-design.md section 1:

    shrunk = clip(m + k * (p - m), 0.01, 0.99)

p is the model's probability of the OVER (after MLB's calibration), m the market's no-vig probability of the over at
the same line, built exactly as the board builds `market_prob`: `board.novig(best over price, best under price)`.
The under side uses 1 - shrunk.
"""
from __future__ import annotations

import math

from ..serving.board import novig

CLIP_LO = 0.01
CLIP_HI = 0.99


def valid_american(price) -> bool:
    """True for a usable American price: a finite number with |price| >= 100 (so 0, +50 and NaN are not)."""
    if isinstance(price, bool):
        return False
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value) >= 100.0


def market_prob_over(over_price, under_price) -> float | None:
    """No-vig probability of the over from the best over and best under American prices; None if either is invalid."""
    if not (valid_american(over_price) and valid_american(under_price)):
        return None
    return novig(over_price, under_price)


def shrink_prob(p: float, m: float, k: float) -> float:
    """m + k * (p - m), clipped to [0.01, 0.99]. k = 0 gives the market (clipped); k = 1 gives the model (clipped)."""
    return min(max(m + k * (p - m), CLIP_LO), CLIP_HI)
