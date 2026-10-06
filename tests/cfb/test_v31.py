"""v3.1 rolling-origin fit (addendum A): season S is fitted on seasons first..S-1 ONLY. Network-free."""
import json
from dataclasses import replace

import numpy as np
import pytest

from sportsmodel.cfb import efficiency as eff, v3, v31
from tests.cfb.test_v3_table import inputs, world

SMALL_GRID = {"ridge": [1.0, 8.0], "half_life_games": [1.0, 4.0], "prior_floor": [0.0, 0.3],
              "k0": [0.3, 0.8], "k_ret": [0.0], "k_tal": [0.0]}
SEASONS = (2016, 2017, 2018, 2019, 2020)
S = 2019


def fit(frame, eg, season=S, **kw):
    return v31.fit_season(season, frame, inputs(eg), grid=SMALL_GRID, **kw)


def test_train_seasons_are_first_through_season_minus_one():
    assert v31.train_seasons_for(2023) == tuple(range(2016, 2023))
    assert v31.train_seasons_for(2025) == tuple(range(2016, 2025))
    assert v31.train_seasons_for(2019, first=2016) == (2016, 2017, 2018)


def test_fit_season_records_its_train_seasons_and_returns_only_season_s_rows():
    sched, eg = world(seasons=SEASONS)
    f = fit(sched, eg)
    assert f.weights.meta["train_seasons"] == [2016, 2017, 2018]
    assert set(f.table["season"]) == {S}
    assert f.gameline["sigma_margin"] == f.weights.meta["sigma_margin"]
    assert f.eff_cfg.ridge in SMALL_GRID["ridge"]


def test_poisoning_season_s_and_later_leaves_the_season_s_fit_unchanged():
    sched, eg = world(seasons=SEASONS)
    base = fit(sched, eg)
    ps, pe = sched.copy(), eg.copy()
    ms, me = ps["season"] >= S, pe["season"] >= S
    ps.loc[ms, "home_score"] = 777
    ps.loc[ms, "away_score"] = 3
    ps.loc[ms, "market_spread"] = -40.0
    for m in eff.METRICS:
        pe.loc[me, f"y_{m}"] = 9.0
        pe.loc[me, f"w_{m}"] = 500.0
    pois = fit(ps, pe)
    assert pois.eff_cfg == base.eff_cfg
    assert pois.weights.to_json() == base.weights.to_json()
    assert pois.gameline == base.gameline


def test_poisoning_the_last_training_season_does_change_the_fit():
    """Control for the test above: the poison is strong enough to move a fit that CAN see it."""
    sched, eg = world(seasons=SEASONS)
    base = fit(sched, eg)
    ps = sched.copy()
    ms = ps["season"] == S - 1
    ps.loc[ms, "home_score"] = 777
    ps.loc[ms, "away_score"] = 3
    assert fit(ps, eg).weights.to_json() != base.weights.to_json()


def test_prediction_rows_of_season_s_ignore_later_seasons():
    sched, eg = world(seasons=SEASONS)
    base = fit(sched, eg)
    later = fit(sched[sched["season"] <= S], eg[eg["season"] <= S])
    assert base.table.reset_index(drop=True).equals(later.table.reset_index(drop=True))


def test_write_season_outputs_roundtrips(tmp_path):
    sched, eg = world(seasons=SEASONS)
    f = fit(sched, eg)
    out = v31.write_season_outputs(f, tmp_path)
    assert out == tmp_path / str(S)
    assert v3.load_v3_weights(out / "v3_weights.json") == f.weights
    assert eff.load_eff_config(out / "eff_config.json") == f.eff_cfg
    assert v3.load_gameline_config(out / "gameline_v3.json").sigma_margin == f.gameline["sigma_margin"]
    json.loads((out / "gameline_v3.json").read_text())
