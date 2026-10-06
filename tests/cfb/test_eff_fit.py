import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import eff_fit, efficiency as eff


def persistent_league(seasons=(2020, 2021, 2022, 2023), weeks=6, n_teams=12, seed=11):
    """Team strength persists across seasons (AR(1)), so a previous-season prior is informative."""
    rng = np.random.default_rng(seed)
    off = rng.normal(0, 0.25, n_teams)
    dfn = rng.normal(0, 0.25, n_teams)
    rows, gid = [], 0
    for s in seasons:
        off, dfn = 0.8 * off + rng.normal(0, 0.08, n_teams), 0.8 * dfn + rng.normal(0, 0.08, n_teams)
        for w in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for i in range(0, n_teams, 2):
                h, a = int(order[i]), int(order[i + 1])
                gid += 1
                for team, opp, hs in ((h, a, 1), (a, h, -1)):
                    y = 0.1 + off[team] - dfn[opp] + 0.03 * hs + rng.normal(0, 0.15)
                    rows.append({"season": s, "game_id": gid, "team": str(team), "opponent": str(opp),
                                 "season_type": "regular", "week": float(w), "home": float(hs),
                                 "plays": 70.0, "rush_plays": 35.0, "pass_plays": 35.0,
                                 **{f"y_{m}": y for m in eff.METRICS}, **{f"w_{m}": 60.0 for m in eff.METRICS}})
    return pd.DataFrame(rows)


def test_loss_is_finite_and_a_prior_helps_early_weeks():
    g = persistent_league()
    no_prior = eff_fit.ppa_holdout_loss(g, (2022, 2023), eff.EffConfig(k0=0.0), {}, None)
    with_prior = eff_fit.ppa_holdout_loss(g, (2022, 2023), eff.EffConfig(k0=0.8), {}, None)
    assert np.isfinite(no_prior) and np.isfinite(with_prior)
    assert with_prior < no_prior


def test_loss_only_uses_games_before_each_week():
    g = persistent_league()
    base = eff_fit.ppa_holdout_loss(g, (2022,), eff.EffConfig(), {}, None)
    # corrupting the LAST week's observations changes that week's targets but never an earlier
    # week's prediction: losses differ, yet the prediction path up to week 5 is identical
    last = g[(g.season == 2022) & (g.week == 6)].index
    g2 = g.copy()
    g2.loc[last, "y_ppa"] += 1.0
    assert eff_fit.ppa_holdout_loss(g2, (2022,), eff.EffConfig(), {}, None) != base
    g3 = g[~((g.season == 2022) & (g.week == 6))]
    g4_future = g[(g.season == 2023)].copy()
    g4_future["y_ppa"] += 3.0                       # a later season must not change 2022's loss
    assert eff_fit.ppa_holdout_loss(pd.concat([g3, g4_future]), (2022,), eff.EffConfig(), {}, None) == \
        eff_fit.ppa_holdout_loss(g3, (2022,), eff.EffConfig(), {}, None)


def test_fit_never_worse_than_start_and_stays_on_grid():
    g = persistent_league()
    grid = {"ridge": [1.0, 4.0, 16.0], "half_life_games": [1.0, 3.0], "prior_floor": [0.0, 0.3],
            "k0": [0.2, 0.8], "k_ret": [0.0], "k_tal": [0.0]}
    start = eff.EffConfig(k0=0.2)
    cfg, loss = eff_fit.fit_eff_config(g, {}, None, train_seasons=(2021, 2022), grid=grid, start=start)
    assert loss <= eff_fit.ppa_holdout_loss(g, (2021, 2022), start, {}, None) + 1e-12
    assert cfg.ridge in grid["ridge"] and cfg.k0 in (0.2, 0.8)
    assert cfg.k0 == 0.8                                   # persistent strengths -> keep most of last year
