"""Kelly staking policy and bankroll simulator for the CFB profit model (PURE).

Conventions:
- ``d`` is a decimal price (stake returned on a win is ``d``); edge = p*d - 1.
- Stakes are fractions of the bankroll at the START of the slate day, after
  the per-bet cap and the per-day cap; a day's bets settle together at day end.
- A push returns the stake (P&L 0). Pushes count toward ``n_bets`` and
  ``staked`` (the money was at risk), so ROI is profit / all money staked.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

FLAT_STAKE = 10.0


def american_to_decimal(a: float) -> float:
    a = float(a)
    if not math.isfinite(a) or abs(a) < 100:
        raise ValueError(f"invalid American price: {a!r}")
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def choose_side(
    p_home: float,
    d_home: float,
    d_away: float,
    min_edge: float,
    sides: tuple[str, str] = ("home", "away"),
) -> tuple[str, float] | None:
    """Pick the side with the larger edge if it is >= min_edge, else None.

    ``p_home`` is P(first side wins); the other side's probability is
    1 - p_home. ``sides`` names them (``("over", "under")`` for totals). A tie
    goes to the first side.
    """
    edge_first = p_home * d_home - 1.0
    edge_second = (1.0 - p_home) * d_away - 1.0
    if edge_first >= edge_second:
        side, edge = sides[0], edge_first
    else:
        side, edge = sides[1], edge_second
    if edge >= min_edge and edge > 0:
        return side, edge
    return None


def stake_fraction(p: float, d: float, kelly_frac: float, cap: float = 0.03) -> float:
    if d <= 1.0:
        return 0.0
    return min(cap, kelly_frac * max(0.0, p * d - 1.0) / (d - 1.0))


def apply_day_cap(stakes: list[float], cap: float = 0.15) -> list[float]:
    total = sum(stakes)
    if total <= cap or total <= 0:
        return list(stakes)
    scale = cap / total
    return [s * scale for s in stakes]


def _pnl_unit(dec: float, result: str) -> float:
    """Profit per unit staked."""
    if result == "win":
        return dec - 1.0
    if result == "loss":
        return -1.0
    if result == "push":
        return 0.0
    raise ValueError(f"unknown result: {result!r}")


def simulate(bets: pd.DataFrame, start: float = 100.0) -> dict:
    """Bankroll simulation over bets in chronological order (columns: day,
    stake_frac, dec, result)."""
    bankroll = float(start)
    peak = bankroll
    max_dd = 0.0
    staked = 0.0
    n = len(bets)
    if n:
        for _, day in bets.groupby("day", sort=False):
            base = bankroll
            day_pnl = 0.0
            for frac, dec, res in zip(day["stake_frac"], day["dec"], day["result"]):
                stake = base * float(frac)
                staked += stake
                day_pnl += stake * _pnl_unit(float(dec), res)
            bankroll = base + day_pnl
            peak = max(peak, bankroll)
            if peak > 0:
                max_dd = max(max_dd, (peak - bankroll) / peak)
    profit = bankroll - start
    log_total = math.log(bankroll / start) if bankroll > 0 else float("-inf")
    return {
        "end_bankroll": bankroll,
        "log_growth_total": log_total if n else 0.0,
        "log_growth_per_bet": log_total / n if n else 0.0,
        "n_bets": n,
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked > 0 else 0.0,
        "max_drawdown": max_dd,
    }


def flat_pnl(bets: pd.DataFrame) -> dict:
    """Flat $10 per bet."""
    n = len(bets)
    profit = sum(
        FLAT_STAKE * _pnl_unit(float(d), r) for d, r in zip(bets["dec"], bets["result"])
    )
    staked = FLAT_STAKE * n
    return {
        "n_bets": n,
        "staked": staked,
        "profit": profit,
        "roi": profit / staked if staked > 0 else 0.0,
    }


def roi_ci(bets: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95 % percentile interval for ROI on money staked, bootstrapping
    (season, week) clusters. Bets are weighted by ``stake_frac``. Returns
    (nan, nan) with no bets."""
    if len(bets) == 0:
        return float("nan"), float("nan")
    df = pd.DataFrame({
        "season": bets["season"].to_numpy(),
        "week": bets["week"].to_numpy(),
        "stake": bets["stake_frac"].astype(float).to_numpy(),
        "unit": [_pnl_unit(float(d), r) for d, r in zip(bets["dec"], bets["result"])],
    })
    df["profit"] = df["stake"] * df["unit"]
    g = df.groupby(["season", "week"])[["profit", "stake"]].sum()
    profit = g["profit"].to_numpy()
    stake = g["stake"].to_numpy()
    k = len(g)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_boot, k))
    sp = profit[idx].sum(axis=1)
    ss = stake[idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        rois = np.where(ss > 0, sp / ss, np.nan)
    if np.all(np.isnan(rois)):
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(rois, [2.5, 97.5])
    return float(lo), float(hi)
