"""NFL player prop aggregation and disagreement metric.

Turns simulated player stat arrays into prop distributions and computes
the disagreement between analytic and simulated win probabilities.
"""
from __future__ import annotations

import numpy as np

from sportsmodel.sim.engine import _pmf_mean
from sportsmodel.sim.nfl.spec import NflGameSims


def nfl_player_prop_dists(sims: NflGameSims, market_max: dict) -> dict[str, dict[str, dict]]:
    """Aggregate per-player market statistics into probability distributions.

    For standard yard/reception markets, uses stat_pmf to build full distributions.
    For anytime_td, builds a binary distribution [P(no TD), P(at least 1 TD)].

    Args:
        sims: NflGameSims container with player_stats[pid][market] = np.ndarray.
        market_max: Dict mapping market names to their max values for binning
            (e.g., {"pass_yds": 400, "rush_yds": 200, ...}).

    Returns:
        Dict mapping player_id -> market -> {"kind": "pmf", "pmf": [...], "mean": float}.
    """
    out: dict[str, dict[str, dict]] = {}

    for player_id, stats in sims.player_stats.items():
        player_dists: dict[str, dict] = {}

        # Standard markets: pass_yds, rush_yds, rec_yds, receptions
        for market in ["pass_yds", "rush_yds", "rec_yds", "receptions"]:
            if market in stats:
                player_dists[market] = _pmf_mean(stats[market], market_max[market])

        # Binary market: anytime_td
        if "td" in stats:
            td_array = stats["td"]
            p_td_at_least_one = float(np.mean(td_array >= 1))
            player_dists["anytime_td"] = {
                "kind": "pmf",
                "pmf": [1 - p_td_at_least_one, p_td_at_least_one],
                "mean": float(np.mean(td_array)),
            }

        out[player_id] = player_dists

    return out


def disagreement(analytic_home_win_prob: float, sim_home_win_prob: float) -> float:
    """Compute absolute disagreement between two home win probabilities.

    Args:
        analytic_home_win_prob: Home win probability from analytic model.
        sim_home_win_prob: Home win probability from simulation.

    Returns:
        Absolute difference between the two probabilities.
    """
    return abs(analytic_home_win_prob - sim_home_win_prob)
