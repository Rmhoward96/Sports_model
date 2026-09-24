"""Tests for the learned volume/efficiency models (props ML, rung A).

All data is the synthetic `tbl` fixture from conftest.py; never the real
feature parquet, never the network.
"""
import dataclasses

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from sportsmodel.sim.nfl.learned import (
    RUNG_PREFIXES,
    apply_to_spec,
    feature_columns,
    fit_models,
    tune,
)

VOL = frozenset({"volume"})


def _spy_fit(monkeypatch):
    """Record (n_rows, columns, sample_weight) for every HGB fit."""
    calls = []
    real_fit = HistGradientBoostingRegressor.fit

    def spy(self, X, y, sample_weight=None):
        calls.append({"n": len(X), "cols": list(X.columns),
                      "w": None if sample_weight is None else np.asarray(sample_weight),
                      "loss": self.loss})
        return real_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    return calls


# ---- brief tests -----------------------------------------------------------

def test_feature_columns_by_toggle(tbl):
    cols = feature_columns(tbl.player, frozenset({"volume"}))
    assert all(c.startswith(("p_", "ngs_", "tm_", "op_", "st_")) for c in cols)
    assert not any(c.startswith(("cx_", "mk_", "y_")) for c in cols)
    assert any(c.startswith("mk_") for c in feature_columns(tbl.player, frozenset({"volume", "market"})))


def test_fit_uses_only_rows_before_upto(tbl, monkeypatch):
    seen = {}
    real_fit = HistGradientBoostingRegressor.fit

    def spy(self, X, y, sample_weight=None):
        seen.setdefault("n", len(X))
        return real_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    fit_models(tbl.player, tbl.team, frozenset({"volume"}), upto=(2024, 1), test_season=2024, decay=1.0, max_iter=20)
    assert seen["n"] == int(((tbl.player.season < 2024) & tbl.player.y_targets.notna()).sum())


def test_apply_renormalizes_shares_and_downweights_questionable(tbl, spec):
    m = fit_models(tbl.player, tbl.team, frozenset({"volume"}), upto=(2024, 1), test_season=2024, decay=1.0, max_iter=20)
    out = apply_to_spec(spec, m, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1), questionable={"h2"}, q_weight=0.5)
    shares = [p.target_share for p in out.home_players]
    assert abs(sum(shares) - 1.0) < 1e-9
    base = apply_to_spec(spec, m, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1), questionable=set(), q_weight=0.5)
    h2 = lambda s: next(p.target_share for p in s.home_players if p.player_id == "h2")  # noqa: E731
    assert h2(out) < h2(base)
    assert out.home.pass_att_pg > 0 and out.home.pass_att_pg != spec.home.pass_att_pg


def test_monotone_in_target_share(tbl):
    m = fit_models(tbl.player, tbl.team, frozenset({"volume"}), upto=(2024, 1), test_season=2024, decay=1.0, max_iter=50)
    row = tbl.player[tbl.player.season == 2023].iloc[[0]].copy()
    lo, hi = row.copy(), row.copy()
    lo["p_target_share_ewm"], hi["p_target_share_ewm"] = 0.05, 0.30
    assert m.targets.predict(hi[m.player_cols])[0] >= m.targets.predict(lo[m.player_cols])[0]


# ---- feature_columns ---------------------------------------------------------

def test_feature_columns_excludes_keys_labels_and_unprefixed(tbl):
    df = tbl.player.assign(random_extra=1.0)
    cols = feature_columns(df, frozenset({"volume", "context", "market"}))
    for bad in ("player_id", "season", "week", "team", "opponent", "position",
                "random_extra", "y_targets", "y_rec_yds"):
        assert bad not in cols
    assert "p_pos" in cols  # categorical stays
    assert "cx_rest" in cols and "mk_total" in cols
    # context toggle alone adds cx_ but not mk_
    ctx = feature_columns(df, frozenset({"volume", "context"}))
    assert "cx_rest" in ctx and "mk_total" not in ctx
    assert RUNG_PREFIXES["volume"] == ("p_", "ngs_", "tm_", "op_", "st_")


def test_feature_columns_volume_always_on(tbl):
    # "volume is always on when any learned model is used"
    assert feature_columns(tbl.player, frozenset({"efficiency"})) == feature_columns(tbl.player, VOL)


# ---- fit_models ---------------------------------------------------------------

def test_fit_filters_by_season_week_key_and_labels_for_every_model(tbl, monkeypatch):
    calls = _spy_fit(monkeypatch)
    fit_models(tbl.player, tbl.team, VOL, upto=(2023, 5), test_season=2023,
               decay=1.0, max_iter=10)
    p, t = tbl.player, tbl.team
    before_p = (p.season < 2023) | ((p.season == 2023) & (p.week < 5))
    before_t = (t.season < 2023) | ((t.season == 2023) & (t.week < 5))
    expected = sorted([
        int((before_p & p.y_targets.notna()).sum()),
        int((before_p & p.y_carries.notna()).sum()),
        int((before_t & t.y_team_pass_att.notna()).sum()),
        int((before_t & t.y_team_rush_att.notna()).sum()),
    ])
    assert sorted(c["n"] for c in calls) == expected
    assert all(c["loss"] == "poisson" for c in calls)


def test_fit_never_trains_on_stub_or_future_rows_even_with_poisoned_labels(tbl):
    # Poison every row at/after upto: a leak would move predictions massively.
    p, t = tbl.player.copy(), tbl.team.copy()
    fut = p.season >= 2024
    p.loc[fut, "y_targets"] = 500.0
    t.loc[t.season >= 2024, "y_team_pass_att"] = 900.0
    m = fit_models(p, t, VOL, upto=(2024, 1), test_season=2024, decay=1.0, max_iter=20)
    assert m.targets.predict(p[fut][m.player_cols]).max() < 50
    assert m.team_pass.predict(t[t.season >= 2024][m.team_cols]).max() < 60


def test_fit_sample_weight_decays_by_season(tbl, monkeypatch):
    calls = _spy_fit(monkeypatch)
    fit_models(tbl.player, tbl.team, VOL, upto=(2024, 1), test_season=2024,
               decay=0.5, max_iter=10)
    p = tbl.player
    train = p[(p.season < 2024) & p.y_targets.notna()]
    np.testing.assert_allclose(calls[0]["w"], 0.5 ** (2024 - train.season.to_numpy()))


def test_fit_drops_all_nan_columns_and_records_used_columns(tbl, monkeypatch):
    calls = _spy_fit(monkeypatch)
    m = fit_models(tbl.player, tbl.team, VOL, upto=(2024, 1), test_season=2024,
                   decay=1.0, max_iter=10)
    assert "ngs_empty" in m.player_cols          # candidate column...
    assert "ngs_empty" not in m.targets.cols     # ...but dropped for the fit
    assert all("ngs_empty" not in c["cols"] for c in calls)
    # prediction with the full candidate frame uses the fitted subset
    preds = m.targets.predict(tbl.rows_for(2024, 1)[m.player_cols])
    assert preds.shape == (8,) and np.all(preds > 0)


def test_fit_monotonic_constraint_only_when_column_present(tbl):
    m = fit_models(tbl.player, tbl.team, VOL, upto=(2024, 1), test_season=2024,
                   decay=1.0, max_iter=10)
    assert m.targets.estimator.monotonic_cst == {"p_target_share_ewm": 1}
    assert m.carries.estimator.monotonic_cst == {"p_carry_share_ewm": 1}
    p = tbl.player.drop(columns=["p_target_share_ewm", "p_carry_share_ewm"])
    m2 = fit_models(p, tbl.team, VOL, upto=(2024, 1), test_season=2024,
                    decay=1.0, max_iter=10)
    assert m2.targets.estimator.monotonic_cst is None
    assert m2.carries.estimator.monotonic_cst is None


def test_fit_hyperparameters_match_spec(tbl):
    m = fit_models(tbl.player, tbl.team, frozenset({"volume", "efficiency"}),
                   upto=(2024, 1), test_season=2024, decay=1.0, max_iter=7)
    ests = [m.team_pass, m.team_rush, m.targets, m.carries, *m.eff.values()]
    for f in ests:
        e = f.estimator
        assert (e.learning_rate, e.max_leaf_nodes, e.min_samples_leaf,
                e.l2_regularization, e.categorical_features, e.random_state,
                e.max_iter, e.early_stopping) == (0.05, 31, 50, 1.0, "from_dtype", 0, 7, False)
    assert {k: f.estimator.loss for k, f in m.eff.items()} == {
        "ypr": "squared_error", "catch_rate": "squared_error", "ypc": "squared_error"}


def test_fit_empty_training_set_gives_none_models(tbl):
    m = fit_models(tbl.player, tbl.team, frozenset({"volume", "efficiency"}),
                   upto=(2000, 1), test_season=2000, decay=1.0, max_iter=10)
    assert m.targets is None and m.carries is None
    assert m.team_pass is None and m.team_rush is None
    assert all(v is None for v in m.eff.values())


def test_fit_efficiency_rows_labels_and_weights(tbl, monkeypatch):
    calls = _spy_fit(monkeypatch)
    m = fit_models(tbl.player, tbl.team, frozenset({"volume", "efficiency"}),
                   upto=(2024, 1), test_season=2024, decay=1.0, max_iter=10)
    assert set(m.eff) == {"ypr", "catch_rate", "ypc"}
    p = tbl.player[tbl.player.season < 2024]
    eff_calls = [c for c in calls if c["loss"] == "squared_error"]
    got = sorted((c["n"], float(c["w"].sum())) for c in eff_calls)
    want = sorted([
        (int((p.y_receptions > 0).sum()), float(p.y_receptions[p.y_receptions > 0].sum())),
        (int((p.y_targets > 0).sum()), float(p.y_targets[p.y_targets > 0].sum())),
        (int((p.y_carries > 0).sum()), float(p.y_carries[p.y_carries > 0].sum())),
    ])
    assert got == want
    # no efficiency models without the toggle
    assert fit_models(tbl.player, tbl.team, VOL, upto=(2024, 1), test_season=2024,
                      decay=1.0, max_iter=10).eff is None


# ---- apply_to_spec -------------------------------------------------------------

@pytest.fixture(scope="module")
def eff_models(tbl):
    return fit_models(tbl.player, tbl.team, frozenset({"volume", "efficiency"}),
                      upto=(2024, 1), test_season=2024, decay=1.0, max_iter=20)


def test_apply_sets_both_sides_team_rates_and_shares(tbl, spec, eff_models):
    out = apply_to_spec(spec, eff_models, tbl.rows_for(2024, 1),
                        tbl.team_rows_for(2024, 1), questionable=set(), q_weight=0.75)
    for side, players in (("home", out.home_players), ("away", out.away_players)):
        rates = getattr(out, side)
        assert rates.pass_att_pg > 10 and rates.rush_att_pg > 10
        assert abs(sum(p.target_share for p in players) - 1.0) < 1e-9
        assert abs(sum(p.carry_share for p in players) - 1.0) < 1e-9
    # the WR out-targets the QB, the RB out-carries the WR
    hp = {p.player_id: p for p in out.home_players}
    assert hp["h2"].target_share > hp["h1"].target_share
    assert hp["h3"].carry_share > hp["h2"].carry_share
    # untouched fields
    assert out.home.drive_outcomes == spec.home.drive_outcomes
    assert [p.td_share for p in out.home_players] == [p.td_share for p in spec.home_players]
    assert [p.rec_td_share for p in out.home_players] == [p.rec_td_share for p in spec.home_players]


def test_apply_efficiency_replaces_ypr_catch_rate_ypc_and_derives_ypt(tbl, spec, eff_models):
    out = apply_to_spec(spec, eff_models, tbl.rows_for(2024, 1),
                        tbl.team_rows_for(2024, 1), questionable=set(), q_weight=0.75)
    for new, old in zip(out.home_players + out.away_players,
                        spec.home_players + spec.away_players):
        assert 0.05 <= new.catch_rate <= 1.0
        assert new.ypt == pytest.approx(new.catch_rate * new.ypr)
        assert (new.ypr, new.catch_rate, new.ypc) != (old.ypr, old.catch_rate, old.ypc)


def test_apply_without_efficiency_models_keeps_efficiency(tbl, spec):
    m = fit_models(tbl.player, tbl.team, VOL, upto=(2024, 1), test_season=2024,
                   decay=1.0, max_iter=10)
    out = apply_to_spec(spec, m, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1),
                        questionable=set(), q_weight=0.75)
    for new, old in zip(out.home_players, spec.home_players):
        assert (new.ypt, new.ypr, new.catch_rate, new.ypc) == (old.ypt, old.ypr, old.catch_rate, old.ypc)


def test_apply_missing_player_row_leaves_side_shares_and_counts(tbl, spec, eff_models):
    rows = tbl.rows_for(2024, 1)
    rows = rows[rows.player_id != "h4"]
    before = eff_models.share_fallbacks
    out = apply_to_spec(spec, eff_models, rows, tbl.team_rows_for(2024, 1),
                        questionable=set(), q_weight=0.75)
    assert [p.target_share for p in out.home_players] == [p.target_share for p in spec.home_players]
    assert [p.carry_share for p in out.home_players] == [p.carry_share for p in spec.home_players]
    assert eff_models.share_fallbacks == before + 1
    # away side still learned
    assert abs(sum(p.target_share for p in out.away_players) - 1.0) < 1e-9
    assert out.away_players[0].target_share != spec.away_players[0].target_share
    # efficiency still replaced for home players that DO have rows; h4 untouched
    h = {p.player_id: p for p in out.home_players}
    old_h4 = next(p for p in spec.home_players if p.player_id == "h4")
    assert dataclasses.astuple(h["h4"]) == dataclasses.astuple(old_h4)
    assert h["h2"].ypr != old_h4.ypr


def test_apply_missing_team_row_leaves_rates(tbl, spec, eff_models):
    trows = tbl.team_rows_for(2024, 1)
    out = apply_to_spec(spec, eff_models, tbl.rows_for(2024, 1),
                        trows[trows.team != "HOM"], questionable=set(), q_weight=0.75)
    assert out.home == spec.home
    assert out.away.pass_att_pg != spec.away.pass_att_pg


def test_apply_does_not_mutate_input_spec(tbl, spec, eff_models):
    snapshot = dataclasses.astuple(spec)
    apply_to_spec(spec, eff_models, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1),
                  questionable={"h2"}, q_weight=0.5)
    assert dataclasses.astuple(spec) == snapshot


# ---- tune ---------------------------------------------------------------------------

def test_tune_returns_grid_point_and_trains_only_before_validation_season(tbl, monkeypatch):
    calls = _spy_fit(monkeypatch)
    decay, max_iter = tune(tbl.player, 2024, VOL)
    assert decay in (1.0, 0.8, 0.6) and max_iter in (150, 300)
    p = tbl.player
    n_train = int(((p.season < 2023) & p.y_targets.notna()).sum())
    assert len(calls) == 6
    assert all(c["n"] == n_train for c in calls)
    # inner "test season" is 2023, training is 2022 only -> weight == decay
    assert sorted({round(float(c["w"].min()), 6) for c in calls}) == [0.6, 0.8, 1.0]


def test_tune_raises_without_validation_data(tbl):
    with pytest.raises(ValueError):
        tune(tbl.player, 2022, VOL)


class _ConstModel:
    """Stub fitted model: predicts the same value for every row."""

    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value, dtype=float)


def test_apply_floors_learned_ypr_and_ypc_at_half_a_yard(tbl, spec, eff_models):
    # A <= 0 learned ypr/ypc would give ypt <= 0 and can make the kernel fail;
    # the harness's per-game try/except would then silently drop the game.
    m = dataclasses.replace(eff_models, eff={"ypr": _ConstModel(-3.0),
                                             "catch_rate": _ConstModel(0.6),
                                             "ypc": _ConstModel(0.0)})
    out = apply_to_spec(spec, m, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1),
                        questionable=set(), q_weight=0.75)
    for p in out.home_players + out.away_players:
        assert p.ypr == 0.5 and p.ypc == 0.5
        assert p.ypt == pytest.approx(0.6 * 0.5)
    # values above the floor pass through untouched
    m2 = dataclasses.replace(eff_models, eff={"ypr": _ConstModel(11.0),
                                              "catch_rate": None,
                                              "ypc": _ConstModel(4.2)})
    out2 = apply_to_spec(spec, m2, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1),
                         questionable=set(), q_weight=0.75)
    assert {p.ypr for p in out2.home_players} == {11.0}
    assert {p.ypc for p in out2.home_players} == {4.2}


# ---- I2: count models train on the sim's population (stubs as zero) ---------------

def _spy_fit_y(monkeypatch):
    calls = []
    real_fit = HistGradientBoostingRegressor.fit

    def spy(self, X, y, sample_weight=None):
        calls.append({"n": len(X), "y": np.asarray(y, dtype=float), "loss": self.loss,
                      "cols": list(X.columns)})
        return real_fit(self, X, y, sample_weight=sample_weight)

    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    return calls


def _with_stub_flag(p):
    return p.assign(is_stub=p["y_targets"].isna())


def test_count_models_train_in_window_stubs_as_zero_efficiency_does_not(tbl, monkeypatch):
    p = _with_stub_flag(tbl.player)
    calls = _spy_fit_y(monkeypatch)
    fit_models(p, tbl.team, frozenset({"volume", "efficiency"}), upto=(2023, 5),
               test_season=2023, decay=1.0, max_iter=10)
    before = (p.season < 2023) | ((p.season == 2023) & (p.week < 5))
    n_played, n_stub = int((before & ~p.is_stub).sum()), int((before & p.is_stub).sum())
    assert n_stub > 0 and int((~before & p.is_stub).sum()) > 0     # stubs on both sides of upto
    targets, carries = calls[0], calls[1]                            # fit order: targets, carries, team...
    for c, label in ((targets, "y_targets"), (carries, "y_carries")):
        assert c["loss"] == "poisson" and c["n"] == n_played + n_stub   # future stubs never train
        want = np.sort(np.concatenate([p.loc[before & ~p.is_stub, label].to_numpy(), np.zeros(n_stub)]))
        np.testing.assert_array_equal(np.sort(c["y"]), want)
        assert "is_stub" not in c["cols"]
    eff = [c for c in calls if c["loss"] == "squared_error"]
    pb = p[before]
    assert sorted(c["n"] for c in eff) == sorted([int((pb.y_receptions > 0).sum()),
                                                   int((pb.y_targets > 0).sum()),
                                                   int((pb.y_carries > 0).sum())])


def test_feature_columns_never_include_the_stub_flag(tbl):
    p = _with_stub_flag(tbl.player)
    assert "is_stub" not in feature_columns(p, frozenset({"volume", "context", "market"}))


def test_tune_trains_and_validates_on_the_sim_population(tbl, monkeypatch):
    """With the flag, tune's targets fits and its validation slice both treat
    stub rows as 0-target rows (the population the model is applied to)."""
    p = _with_stub_flag(tbl.player)
    calls = _spy_fit_y(monkeypatch)
    scored = []
    import sportsmodel.sim.nfl.learned as L
    real_dev = L.mean_poisson_deviance
    monkeypatch.setattr(L, "mean_poisson_deviance",
                        lambda y, pred: scored.append(len(y)) or real_dev(y, pred))
    tune(p, 2024, VOL)
    assert all(c["n"] == int((p.season < 2023).sum()) for c in calls)
    assert set(scored) == {int((p.season == 2023).sum())}
