"""Batter prop projections from a per-PA outcome vector (docs/methodology.md §3b, Part C).

Given a batter's in-game per-PA vector and lineup slot (-> projected PA), produce the
mean and P(over standard line) for Hits, Total Bases, and Home Run.
"""
from __future__ import annotations

from .distributions import (
    hits_pmf,
    prob_at_least_one_hr,
    prob_over,
    total_bases_pmf,
)

# League expected plate appearances by batting-order slot (1-9). [tunable]
SLOT_PA = {1: 4.65, 2: 4.55, 3: 4.44, 4: 4.34, 5: 4.23, 6: 4.13, 7: 4.02, 8: 3.92, 9: 3.81}

# Standard line per market for the stored P(over); compare the mean to the book line.
DEFAULT_LINE = {"hits": 0.5, "total_bases": 1.5, "home_run": 0.5}


def batter_props(vec: dict[str, float], slot: int) -> dict:
    """Prop projections for one batter. Returns per-market {mean, line, prob_over}."""
    pa = SLOT_PA.get(slot, 4.1)
    n = max(1, round(pa))          # integer trials for the distributions
    vecs = [vec] * n
    p_hit = vec["p_1b"] + vec["p_2b"] + vec["p_3b"] + vec["p_hr"]
    tb1 = vec["p_1b"] + 2 * vec["p_2b"] + 3 * vec["p_3b"] + 4 * vec["p_hr"]

    hits = hits_pmf(vecs)
    tb = total_bases_pmf(vecs)
    return {
        "projected_pa": pa,
        "hits": {
            "mean": pa * p_hit,
            "line": DEFAULT_LINE["hits"],
            "prob_over": prob_over(hits, DEFAULT_LINE["hits"]),
        },
        "total_bases": {
            "mean": pa * tb1,
            "line": DEFAULT_LINE["total_bases"],
            "prob_over": prob_over(tb, DEFAULT_LINE["total_bases"]),
        },
        "home_run": {
            "mean": pa * vec["p_hr"],
            "line": DEFAULT_LINE["home_run"],
            "prob_over": prob_at_least_one_hr([vec["p_hr"]] * n),
        },
    }
