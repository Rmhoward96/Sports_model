"""Per-PA rate estimation and the odds-ratio matchup blend.

Implements docs/methodology.md Part A, steps 1-2 (and the normalize):
  - shrink():             empirical-Bayes regression toward a talent prior
  - odds_ratio_combine(): Tango log5 combination for a single outcome
  - matchup_vector():     per-PA outcome vector for a batter vs a pitcher
"""
from __future__ import annotations

from collections.abc import Mapping

# The 7 mutually-exclusive terminal PA outcomes (sum to 1).
OUTCOMES: tuple[str, ...] = ("p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out")

# Default regression constants (PA of prior weight), from reliability stabilization
# points. See docs/methodology.md §A.1 — tunable.
DEFAULT_K: dict[str, float] = {
    "p_k": 60,
    "p_bb": 120,
    "p_hr": 170,
    "p_1b": 290,
    "p_2b": 1100,
    "p_3b": 1100,
    "p_out": 70,
}

_EPS = 1e-9


def shrink(x: float, n: float, prior: float, k: float) -> float:
    """Empirical-Bayes rate: (observed + k*prior) / (n + k)."""
    denom = n + k
    return prior if denom <= 0 else (x + k * prior) / denom


def _odds(r: float) -> float:
    r = min(max(r, _EPS), 1 - _EPS)
    return r / (1 - r)


def odds_ratio_combine(batter: float, pitcher: float, league: float) -> float:
    """Tango odds-ratio (log5) for one outcome: O(b)*O(p)/O(l) -> probability."""
    o = _odds(batter) * _odds(pitcher) / _odds(league)
    return o / (1 + o)


def normalize(vec: Mapping[str, float]) -> dict[str, float]:
    """Scale a partial outcome vector so it sums to 1."""
    total = sum(vec.values())
    if total <= 0:
        raise ValueError("cannot normalize a non-positive vector")
    return {k: v / total for k, v in vec.items()}


def matchup_vector(
    batter: Mapping[str, float],
    pitcher: Mapping[str, float],
    league: Mapping[str, float],
) -> dict[str, float]:
    """Per-PA outcome vector for this batter vs this pitcher (odds-ratio + normalize).

    Each argument maps every key in OUTCOMES to a rate. Platoon lives inside these
    rates (pass the handedness-split rates), so there is no separate platoon step.
    """
    raw = {
        o: odds_ratio_combine(batter[o], pitcher[o], league[o]) for o in OUTCOMES
    }
    return normalize(raw)
