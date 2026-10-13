"""Line adjustment and EV: value a sportsbook's number using Pinnacle's fair probability at Pinnacle's number. PURE.

Everything is done in "table space" (see landing.py): a side wins when the integer result R beats a threshold c,
`R > c` for direction +1 (home spread, Over) and `R < c` for direction -1 (away spread, Under), pushes when R == c
(whole-number thresholds only). For a spread the threshold is minus the home team's betting line, for a total it is the
total, so both sides of one market at one book share one threshold and moving a bet from Pinnacle's number to a book's
is the same operation for every market and side.

Pinnacle's two-way no-vig probability is conditional on no push, so at Pinnacle's threshold
    P_win = fair * (1 - P_push),  P_loss = (1 - fair) * (1 - P_push),  P_push from the landing table.
Moving to the book's threshold re-classifies the table mass of every integer result whose outcome changes (the old push
value becomes a win or a loss, the new push value becomes the push); the three probabilities still sum to 1.
EV (1 unit staked) = P_win * (decimal - 1) - P_loss; a push returns the stake and contributes 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..serving.board import decimal_odds
from .constants import FAIR_MAX, FAIR_MIN, FLAG_EV_EPS, FLAG_EV_MIN, GRID_EPS, MAX_LINE_GAP
from .landing import Pmf, on_grid

DIRECTION = {("spread", "home"): 1, ("spread", "away"): -1, ("total", "over"): 1, ("total", "under"): -1}
NEGATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Valuation:
    p_win: float
    p_push: float
    p_loss: float
    ev: float


def direction(market: str, side: str) -> int | None:
    return DIRECTION.get((market, side))


def threshold(market: str, side: str, line) -> float | None:
    """Table-space threshold for a side's own line: minus the line for the home spread, the line itself for the away
    spread and for totals (the away team's line is minus the home team's). None for an unknown market / side."""
    if line is None or (market, side) not in DIRECTION:
        return None
    line = float(line)
    return (-line if side == "home" else line) if market == "spread" else line


def own_line(market: str, side: str, c: float) -> float:
    """Inverse of `threshold`: the side's own line for a table-space threshold."""
    return (-c if side == "home" else c) if market == "spread" else c


def _classify(r: int, d: int, c: float) -> int:
    x = (r - c) * d
    return 1 if x > 0 else (-1 if x < 0 else 0)


def ev_of(p_win: float, p_loss: float, price: int) -> float:
    """EV per unit staked at an American price; a push contributes 0."""
    return p_win * (decimal_odds(price) - 1.0) - p_loss


def is_flag(ev: float | None) -> bool:
    return ev is not None and ev + FLAG_EV_EPS >= FLAG_EV_MIN


def fair_in_range(fair: float) -> bool:
    return FAIR_MIN <= fair <= FAIR_MAX


def adjust(fair: float, d: int, cp: float, cb: float, pmf: Pmf, *, cap: float | None = MAX_LINE_GAP,
           strict: bool = True) -> tuple[float, float, float] | None:
    """(P_win, P_push, P_loss) at threshold `cb` for a side whose Pinnacle no-vig probability is `fair` at threshold `cp`.

    None when either threshold is off the half-point grid, when |cb - cp| exceeds `cap` (None = no cap), or, with
    strict=True, when the table says the move is bigger than the probability available (a negative component). With
    strict=False a negative component is clamped to 0 and the three are renormalised (used when grading a ticket against
    a close that moved a long way: that must count as a loss, not be dropped)."""
    if not on_grid(cp) or not on_grid(cb):
        return None
    if cap is not None and abs(cb - cp) > cap + GRID_EPS:
        return None
    push = pmf.mass(cp) if abs(cp - round(cp)) < GRID_EPS else 0.0
    bucket = {1: fair * (1.0 - push), 0: push, -1: (1.0 - fair) * (1.0 - push)}
    for r in range(math.floor(min(cp, cb)) - 1, math.ceil(max(cp, cb)) + 2):
        before, after = _classify(r, d, cp), _classify(r, d, cb)
        if before != after:
            mass = pmf.mass(r)
            bucket[before] -= mass
            bucket[after] += mass
    if min(bucket.values()) < -NEGATIVE_TOLERANCE:
        if strict:
            return None
    clamped = {k: max(v, 0.0) for k, v in bucket.items()}
    total = sum(clamped.values())
    if total <= 0:
        return None
    return clamped[1] / total, clamped[0] / total, clamped[-1] / total


def value_side(market: str, side: str, fair: float, pinnacle_line, book_line, price: int, pmf: Pmf | None, *,
               cap: float | None = MAX_LINE_GAP, strict: bool = True) -> Valuation | None:
    """Valuation of `side` at the book's line and price from Pinnacle's fair for that side at Pinnacle's line (both
    lines are the side's own). None when the table has no answer or the move is not allowed."""
    d = direction(market, side)
    cp, cb = threshold(market, side, pinnacle_line), threshold(market, side, book_line)
    if d is None or cp is None or cb is None or pmf is None or price is None:
        return None
    moved = adjust(fair, d, cp, cb, pmf, cap=cap, strict=strict)
    if moved is None:
        return None
    p_win, p_push, p_loss = moved
    return Valuation(p_win, p_push, p_loss, ev_of(p_win, p_loss, int(price)))
