"""Win/cover/over probabilities from margin/total + sigma, plus the clamped
desk-adjustment hook.

Sub-project 3 (+EV engine), task 2 -- see
.superpowers/sdd/2026-09-09-plus-ev-3-trueprob-model/task-2-brief.md.

PURE -- no IO, reuses `sportsmodel.model.distributions` for all probability
math (no hand-rolled normal CDFs here).
"""
from __future__ import annotations

from .distributions import (
    normal_to_margin_pmf,
    normal_to_pmf,
    prob_cover,
    prob_over_dist,
)

# Integer half-range for football margins fed to `normal_to_margin_pmf`; +-75
# comfortably covers realistic final-score margins.
MARGIN_OFFSET = 75

# Integer max for football totals fed to `normal_to_pmf`.
TOTAL_MAX = 120

# Desk scoring-time adjustment hook: points-flexible, probability-bounded.
DESK_MAX_PTS = 3.5
DESK_MAX_PROB_DELTA = 0.10

_TIER_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}


def win_prob(margin: float, sigma: float) -> float:
    """P(home wins) = P(home margin > 0) from a Normal(margin, sigma) margin dist."""
    dist = normal_to_margin_pmf(margin, sigma, MARGIN_OFFSET)
    return prob_cover(dist, 0.0)


def cover_prob(margin: float, sigma: float, home_line: float) -> float:
    """P(home covers `home_line`) from a Normal(margin, sigma) margin dist."""
    dist = normal_to_margin_pmf(margin, sigma, MARGIN_OFFSET)
    return prob_cover(dist, home_line)


def over_prob(total: float, sigma_total: float, line: float) -> float:
    """P(total > line) from a Normal(total, sigma_total) total dist."""
    dist = {"kind": "pmf", "pmf": normal_to_pmf(total, sigma_total, TOTAL_MAX)}
    return prob_over_dist(dist, line)


def desk_margin_shift(desk_pick: dict | None) -> float:
    """Nominal margin shift (points) implied by a desk pick, signed toward the
    desk's own side (home -> +, away -> -).

    `desk_pick` is `{"conviction_tier": "high|medium|low", "spread_side":
    "home"|"away"|None, "ml_pick": "home"|"away"|None}` (or `None`). If the
    desk has no `spread_side`, its `ml_pick` side is used instead. `None` -> 0.0.

    The shift magnitude is set by the desk's `conviction_tier` (high/medium/low
    -> _TIER_WEIGHTS). The desk's own numeric confidence is NOT used to size the
    nudge.
    """
    if not desk_pick:
        return 0.0
    weight = _TIER_WEIGHTS.get(desk_pick.get("conviction_tier"), 0.0)
    side = desk_pick.get("spread_side") or desk_pick.get("ml_pick")
    if side == "home":
        sign = 1.0
    elif side == "away":
        sign = -1.0
    else:
        return 0.0
    return sign * weight * DESK_MAX_PTS


def apply_desk(
    margin: float, sigma: float, desk_pick: dict | None
) -> tuple[float, float, float]:
    """Apply the clamped desk adjustment to a model margin.

    Returns `(shifted_margin, base_win_prob, adj_win_prob)` where
    `shifted_margin = margin + desk_margin_shift(desk_pick)` and
    `adj_win_prob` is `win_prob(shifted_margin, sigma)` clamped so it moves
    at most `DESK_MAX_PROB_DELTA` away from `base_win_prob`, in the direction
    of the raw shift.
    """
    base = win_prob(margin, sigma)
    shift = desk_margin_shift(desk_pick)
    shifted_margin = margin + shift
    raw_adj = win_prob(shifted_margin, sigma)
    lo = base - DESK_MAX_PROB_DELTA
    hi = base + DESK_MAX_PROB_DELTA
    clamped_adj = min(max(raw_adj, lo), hi)
    return shifted_margin, base, clamped_adj
