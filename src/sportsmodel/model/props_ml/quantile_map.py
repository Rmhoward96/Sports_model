"""Rank-preserving quantile mapping of sim draws (props ML, Props-2).
Pure functions, no IO.

After the per-player pmfs are blended and calibrated, serving needs the Monte
Carlo sim's per-player draws to follow those final pmfs *without* breaking the
cross-player / cross-stat dependence the sim encodes (parlays, team totals).
``map_draws`` does this by replacing each draw with the target quantile at the
draw's own rank: the i-th smallest draw becomes the value at mid-rank
probability ``u = (rank + 0.5) / n`` of the target CDF. The marginal becomes
the target (up to 1/n discretisation) while the ordering across sims — and so
every rank correlation with other arrays from the same sims — is kept, except
within tied draws, whose order is broken at random by ``rng``.

``map_game_sims`` applies this to an ``NflGameSims`` container. The
``anytime_td`` market is the aggregator's ``td >= 1`` indicator
(``sim.nfl.aggregate.nfl_player_prop_dists``); per Controller Ruling P1 the
indicator is mapped onto the 2-bin target and the ``td`` counts rewritten:
mapped 1 & original >= 1 keeps the original count, mapped 1 & original 0
becomes 1, mapped 0 becomes 0.
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from sportsmodel.sim.nfl.spec import NflGameSims

ANYTIME_TD = "anytime_td"
TD_STAT = "td"


def _check_pmf(target_pmf) -> np.ndarray:
    p = np.asarray(target_pmf, dtype=float)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"target pmf must be a non-empty 1-D array, got shape {p.shape}")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("target pmf must be finite and non-negative")
    total = p.sum()
    if total <= 0:
        raise ValueError("target pmf must have positive mass")
    return p / total


def map_draws(draws: np.ndarray, target_pmf: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Map ``draws`` onto ``target_pmf`` (support 0..len-1) preserving rank order.

    Draws are ranked ascending with ties broken uniformly at random by ``rng``
    (``np.lexsort((rng.random(n), draws))``); rank r maps to the smallest k
    with ``cumsum(target)[k] >= (r + 0.5) / n``. ``target_pmf`` is normalised
    (it need not sum to 1); zero-probability bins are never produced.

    Returns an int64 array of the same length. Deterministic for a fixed rng
    state; consumes ``n`` uniforms from ``rng``.
    """
    d = np.asarray(draws)
    if d.ndim != 1:
        raise ValueError(f"draws must be 1-D, got shape {d.shape}")
    p = _check_pmf(target_pmf)
    if np.issubdtype(d.dtype, np.floating) and np.any(np.isnan(d)):
        raise ValueError("draws must not contain NaN")
    n = d.size
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((rng.random(n), d))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)
    u = (rank + 0.5) / n
    cdf = np.cumsum(p)
    cdf[-1] = 1.0  # guard float round-off so every u < 1 lands in support
    k = np.searchsorted(cdf, u, side="left")
    return np.minimum(k, p.size - 1).astype(np.int64)


def _out_dtype(orig: np.ndarray, max_value: int) -> np.dtype:
    """The sim's dtype for this stat when it can hold 0..max_value, else int64."""
    dt = orig.dtype
    if np.issubdtype(dt, np.integer) and np.iinfo(dt).max >= max_value:
        return dt
    if np.issubdtype(dt, np.floating):
        return dt
    return np.dtype(np.int64)


def map_game_sims(
    sims: NflGameSims,
    targets: Mapping[str, Mapping[str, np.ndarray]],
    rng: np.random.Generator,
) -> NflGameSims:
    """Return a new ``NflGameSims`` whose targeted player stats follow ``targets``.

    ``targets[player_id][market]`` is a pmf on 0..K. Market names equal the sim
    stat names (``pass_yds``, ``rec_yds``, ...), except ``anytime_td`` which
    maps the ``td >= 1`` indicator onto a 2-bin pmf ``[P(0), P(>=1)]`` and
    rewrites ``td`` counts per Ruling P1 (see module docstring).

    The input is not mutated: the player-stats dict structure is copied and
    only mapped arrays are replaced; untouched players/stats (and the score
    arrays) are the caller's original array objects. A target whose player or
    stat is absent from ``sims`` is ignored (no error, no rng draw) — serving
    targets come from a projection slate that may list players the sim did not
    roster. An ``anytime_td`` target that is not 2 bins raises ``ValueError``.
    Mapped arrays keep the sim's dtype for that stat when it can hold the
    values. ``rng`` is consumed in ``targets`` iteration order, so the result
    is deterministic for a fixed rng state and targets ordering.
    """
    stats_out = {pid: dict(stats) for pid, stats in sims.player_stats.items()}
    for pid, markets in targets.items():
        for market, pmf in markets.items():
            if market == ANYTIME_TD and np.asarray(pmf).shape != (2,):
                raise ValueError(
                    f"{ANYTIME_TD} target must be a 2-bin pmf, got shape {np.asarray(pmf).shape}"
                )
            stats = stats_out.get(pid)
            stat = TD_STAT if market == ANYTIME_TD else market
            if stats is None or stat not in stats:
                continue
            orig = np.asarray(stats[stat])
            if market == ANYTIME_TD:
                ind = map_draws((orig >= 1).astype(np.int8), pmf, rng)
                new = np.where(ind == 1, np.maximum(orig, 1), 0)
                max_value = int(new.max()) if new.size else 0
            else:
                new = map_draws(orig, pmf, rng)
                max_value = len(pmf) - 1
            stats[stat] = new.astype(_out_dtype(orig, max_value), copy=False)
    return dataclasses.replace(sims, player_stats=stats_out)
