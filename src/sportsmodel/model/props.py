"""Batter prop projections from a per-PA outcome vector (docs/methodology.md §3b, Part C).

Given a batter's in-game per-PA vector and lineup slot (-> projected PA), produce the
mean and P(over standard line) for Hits, Total Bases, and Home Run.
"""
from __future__ import annotations

from statistics import fmean

from .distributions import (
    hits_pmf,
    nb_pmf,
    normal_sf,
    prob_at_least_one_hr,
    prob_over,
    total_bases_pmf,
)

# League expected plate appearances by batting-order slot (1-9). [tunable]
SLOT_PA = {1: 4.65, 2: 4.55, 3: 4.44, 4: 4.34, 5: 4.23, 6: 4.13, 7: 4.02, 8: 3.92, 9: 3.81}

# Default line per market, used only for the stored display `prob_over`. The real
# grading/board work evaluates each market's `dist` at the BOOK's actual line.
DEFAULT_LINE = {
    "hits": 0.5, "total_bases": 1.5, "home_run": 0.5, "hrr": 1.5,
    "pitcher_ks": 5.5, "outs_recorded": 17.5, "hits_allowed": 5.5,
}

# Rough marginal coefficients for the H+R+RBI combo (Phase-1 approximation). [tunable]
_RUN_FROM_HR = 1.0        # a HR scores the batter
_RUN_FROM_ON_BASE = 0.28  # a non-HR baserunner scores ~28% of the time
_RBI_HR = 1.5             # HR drives in self + ~0.5 runners
_RBI_XBH = 0.5            # double/triple RBI
_RBI_1B = 0.2             # single RBI


def batter_props(vec: dict[str, float], slot: int) -> dict:
    """Prop projections for one batter. Returns per-market {mean, line, prob_over}."""
    pa = SLOT_PA.get(slot, 4.1)
    n = max(1, round(pa))          # integer trials for the distributions
    vecs = [vec] * n
    p_hit = vec["p_1b"] + vec["p_2b"] + vec["p_3b"] + vec["p_hr"]
    tb1 = vec["p_1b"] + 2 * vec["p_2b"] + 3 * vec["p_3b"] + 4 * vec["p_hr"]

    hits = hits_pmf(vecs)
    tb = total_bases_pmf(vecs)
    p_hr1 = prob_at_least_one_hr([vec["p_hr"]] * n)
    return {
        "projected_pa": pa,
        "hits": {
            "mean": pa * p_hit,
            "line": DEFAULT_LINE["hits"],
            "prob_over": prob_over(hits, DEFAULT_LINE["hits"]),
            "dist": {"kind": "pmf", "pmf": list(hits)},
        },
        "total_bases": {
            "mean": pa * tb1,
            "line": DEFAULT_LINE["total_bases"],
            "prob_over": prob_over(tb, DEFAULT_LINE["total_bases"]),
            "dist": {"kind": "pmf", "pmf": list(tb)},
        },
        "home_run": {
            "mean": pa * vec["p_hr"],
            "line": DEFAULT_LINE["home_run"],
            "prob_over": p_hr1,
            # Y/N market: Bernoulli(P>=1 HR) -> pmf over {0,1}. P(over 0.5) = p_hr1.
            "dist": {"kind": "pmf", "pmf": [1.0 - p_hr1, p_hr1]},
        },
        "hrr": _hrr(vec, pa),
    }


def _hrr(vec: dict[str, float], pa: float) -> dict:
    """Hits+Runs+RBIs — rough marginal sum (Phase-1; Monte-Carlo is the real fix)."""
    p_hit = vec["p_1b"] + vec["p_2b"] + vec["p_3b"] + vec["p_hr"]
    on_base_nonhr = vec["p_1b"] + vec["p_2b"] + vec["p_3b"] + vec["p_bb"]
    e_h = pa * p_hit
    e_r = pa * (vec["p_hr"] * _RUN_FROM_HR + on_base_nonhr * _RUN_FROM_ON_BASE)
    e_rbi = pa * (vec["p_hr"] * _RBI_HR + (vec["p_2b"] + vec["p_3b"]) * _RBI_XBH
                  + vec["p_1b"] * _RBI_1B)
    mean = e_h + e_r + e_rbi
    # Crude count distribution for P(over): Poisson at the combined mean.
    pmf = nb_pmf(mean, mean)  # var<=mean -> Poisson inside nb_pmf
    return {"mean": mean, "line": DEFAULT_LINE["hrr"],
            "prob_over": prob_over(pmf, DEFAULT_LINE["hrr"]),
            "dist": {"kind": "pmf", "pmf": list(pmf)}}


def pitcher_props(opp_vecs: list[dict], avg_bf: float, var_bf: float,
                  avg_outs: float, sd_outs: float) -> dict:
    """Starter props vs the opposing lineup's per-PA matchup vectors.

    K and Hits Allowed: sum a per-PA rate over projected batters faced, with variance
    from both the Bernoullis and BF uncertainty (law of total variance) -> NB. Outs
    Recorded: a Normal from the workload profile (a light hook model).
    """
    kbar = fmean(v["p_k"] for v in opp_vecs)
    hbar = fmean(v["p_1b"] + v["p_2b"] + v["p_3b"] + v["p_hr"] for v in opp_vecs)

    def counting(rate):
        mean = avg_bf * rate
        var = avg_bf * rate * (1 - rate) + rate * rate * var_bf
        return mean, var

    e_k, v_k = counting(kbar)
    e_h, v_h = counting(hbar)
    k_pmf, h_pmf = nb_pmf(e_k, v_k), nb_pmf(e_h, v_h)
    return {
        "pitcher_ks": {
            "mean": e_k, "line": DEFAULT_LINE["pitcher_ks"],
            "prob_over": prob_over(k_pmf, DEFAULT_LINE["pitcher_ks"]),
            "dist": {"kind": "pmf", "pmf": list(k_pmf)},
        },
        "hits_allowed": {
            "mean": e_h, "line": DEFAULT_LINE["hits_allowed"],
            "prob_over": prob_over(h_pmf, DEFAULT_LINE["hits_allowed"]),
            "dist": {"kind": "pmf", "pmf": list(h_pmf)},
        },
        "outs_recorded": {
            "mean": avg_outs, "line": DEFAULT_LINE["outs_recorded"],
            "prob_over": normal_sf(DEFAULT_LINE["outs_recorded"], avg_outs, sd_outs),
            "dist": {"kind": "normal", "mean": float(avg_outs), "sd": float(sd_outs)},
        },
    }
