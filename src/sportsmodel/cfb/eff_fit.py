"""Walk-forward fit of the efficiency-rating hyperparameters (cfb-ratings-v3). PURE.

Objective: one-week-ahead, play-weighted squared error of PPA per play. For every week W of
every training season the state entering W (season-to-date ridge + previous-season prior)
predicts each team-game of week W:  mu + off[team] - def[opp] + hfa * home.  PPA alone drives
the fit; the chosen (ridge, half-life, floor, k0, k_ret, k_tal) are shared by all metrics.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sportsmodel.cfb.efficiency import EffConfig, season_prior, state_before

TRAIN_SEASONS = tuple(range(2016, 2023))      # 2016-2022 (2015 has no previous-season prior)
HOLDOUT_SEASONS = (2023, 2024, 2025)

GRID = {
    "ridge": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "half_life_games": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "prior_floor": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "k0": [round(0.3 + 0.1 * i, 2) for i in range(8)],            # 0.3 .. 1.0
    "k_ret": [round(-0.2 + 0.05 * i, 2) for i in range(9)],       # -0.2 .. 0.2
    "k_tal": [round(-0.2 + 0.05 * i, 2) for i in range(9)],
}
ORDER = ("ridge", "half_life_games", "prior_floor", "k0", "k_ret", "k_tal")


def ppa_holdout_loss(eff_games: pd.DataFrame, seasons, cfg: EffConfig, priors_rows_by_season: dict,
                     talent: pd.DataFrame | None, cache: dict | None = None) -> float:
    """Play-weighted one-week-ahead MSE of PPA per play over `seasons` (NaN if nothing scored)."""
    cache = {} if cache is None else cache
    sse = wsum = 0.0
    for season in seasons:
        prior = season_prior(eff_games, season, priors_rows_by_season.get(season, []), talent, cfg, cache)
        rows = eff_games[(eff_games["season"] == season) & eff_games["week"].notna()
                         & eff_games["y_ppa"].notna() & (eff_games["w_ppa"] > 0)]
        for week, wk in rows.groupby("week"):
            r = state_before(eff_games, season, week, prior, cfg, cache).ratings["ppa"]
            if r.mu != r.mu:                                  # no data and no prior yet
                continue
            pred = (r.mu + wk["team"].map(r.off).fillna(0.0).to_numpy()
                    - wk["opponent"].map(r.deff).fillna(0.0).to_numpy() + r.hfa * wk["home"].to_numpy())
            err = pred - wk["y_ppa"].to_numpy()
            w = wk["w_ppa"].to_numpy()
            sse += float((w * err ** 2).sum())
            wsum += float(w.sum())
    return sse / wsum if wsum else float("nan")


def fit_eff_config(eff_games: pd.DataFrame, priors_rows_by_season: dict, talent: pd.DataFrame | None,
                   train_seasons=TRAIN_SEASONS, grid: dict = GRID, start: EffConfig | None = None,
                   n_passes: int = 4) -> tuple[EffConfig, float]:
    """Coordinate search (one parameter at a time over `grid`, up to n_passes passes) on the
    training seasons only. Returns (config, train loss)."""
    cache: dict = {}
    seen: dict = {}

    def loss(cfg: EffConfig) -> float:
        key = tuple(getattr(cfg, p) for p in ORDER)
        if key not in seen:
            seen[key] = ppa_holdout_loss(eff_games, train_seasons, cfg, priors_rows_by_season, talent, cache)
        return seen[key]

    cur = start or EffConfig()
    for _ in range(n_passes):
        improved = False
        for p in ORDER:
            best, best_loss = cur, loss(cur)
            for v in grid[p]:
                cand = replace(cur, **{p: v})
                l = loss(cand)
                if l < best_loss - 1e-12:
                    best, best_loss = cand, l
            if best is not cur:
                improved, cur = True, best
        if not improved:
            break
    return cur, loss(cur)
