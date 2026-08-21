"""Sport-agnostic simulation result container and aggregation helpers.

A kernel fills a GameSims (raw per-sim arrays); these helpers turn it into the
stored outputs (win prob, total pmf, per-player pmfs). Nothing here is baseball-specific.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GameSims:
    home_score: np.ndarray
    away_score: np.ndarray
    batter_stats: dict[int, dict[str, np.ndarray]]
    pitcher_stats: dict[int, dict[str, np.ndarray]]


def home_win_prob(sims: GameSims) -> float:
    return float(np.mean(sims.home_score > sims.away_score))


def stat_pmf(arr: np.ndarray, max_k: int) -> list[float]:
    n = len(arr)
    counts = np.bincount(np.clip(arr, 0, max_k).astype(int), minlength=max_k + 1)[: max_k + 1]
    return (counts / n).tolist()


def total_pmf(sims: GameSims, max_total: int = 30) -> list[float]:
    return stat_pmf(sims.home_score + sims.away_score, max_total)


def pred_scores(sims: GameSims) -> dict:
    h = float(np.mean(sims.home_score))
    a = float(np.mean(sims.away_score))
    return {
        "pred_home_score": h,
        "pred_away_score": a,
        "pred_total": h + a,
        "pred_margin": h - a,
        "home_win_prob": home_win_prob(sims),
    }


_BATTER_MARKET_STAT = {"hits": "hits", "total_bases": "total_bases", "hrr": "hrr"}
_PITCHER_MARKET_STAT = {"pitcher_ks": "k", "hits_allowed": "hits", "outs_recorded": "outs"}


def _pmf_mean(arr: np.ndarray, max_k: int) -> dict:
    return {"kind": "pmf", "pmf": stat_pmf(arr, max_k), "mean": float(np.mean(arr))}


def player_prop_dists(sims: GameSims, market_max: dict) -> dict:
    out: dict = {}
    for pid, stats in sims.batter_stats.items():
        d = {}
        for market, stat in _BATTER_MARKET_STAT.items():
            d[market] = _pmf_mean(stats[stat], market_max[market])
        p_hr1 = float(np.mean(stats["hr"] >= 1))
        d["home_run"] = {"kind": "pmf", "pmf": [1 - p_hr1, p_hr1], "mean": float(np.mean(stats["hr"]))}
        out.setdefault(pid, {}).update(d)
    for pid, stats in sims.pitcher_stats.items():
        d = {}
        for market, stat in _PITCHER_MARKET_STAT.items():
            d[market] = _pmf_mean(stats[stat], market_max[market])
        out.setdefault(pid, {}).update(d)
    return out
