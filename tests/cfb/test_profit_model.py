"""Tests for sportsmodel.cfb.profit_model: per-market HGB classifiers with
monotone line/edge constraints, the season walk-forward and isotonic
calibration."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sportsmodel.cfb import profit_model as pm
from sportsmodel.cfb.profit_features import FEATURE_COLS


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def synth(market="spread", seasons=range(2015, 2023), n=1500, seed=0,
          logit=lambda d: 0.3 * d["f_edge_pts"]):
    """Synthetic feature table: every FEATURE_COLS[market] column exists
    (noise), edge = model - line, y ~ Bernoulli(sigmoid(logit(d)))."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in seasons:
        d = pd.DataFrame({c: rng.normal(size=n) for c in FEATURE_COLS[market]})
        d["season"] = s
        d["week"] = rng.integers(1, 15, size=n)
        d["market"] = market
        d["price_point"] = rng.choice(["open", "close"], size=n)
        if market in ("spread", "total"):
            base = 0.0 if market == "spread" else 55.0
            d["line"] = base + rng.normal(0, 10, size=n)
            d["f_model_margin"] = d["line"] + rng.normal(0, 3, size=n)
            d["f_edge_pts"] = d["f_model_margin"] - d["line"]
        else:
            d["f_edge_pts"] = rng.normal(0, 3, size=n)   # not a moneyline feature
        d["y"] = (rng.random(n) < _sigmoid(logit(d))).astype(float)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


# --- monotone constraints ---------------------------------------------------

def test_monotonic_cst_is_built_by_column_name():
    for market in ("spread", "total"):
        m = pm.fit_market(synth(market, seasons=[2015], n=800), market, max_iter=20)
        cst = dict(zip(m.cols, np.asarray(m.clf.monotonic_cst)))
        assert cst["line"] == -1
        assert cst["f_edge_pts"] == 1
        assert all(v == 0 for c, v in cst.items() if c not in ("line", "f_edge_pts"))
        assert m.cols == FEATURE_COLS[market]
    m = pm.fit_market(synth("moneyline", seasons=[2015], n=800), "moneyline", max_iter=20)
    assert m.cols == FEATURE_COLS["moneyline"]
    assert "line" not in m.cols and "f_edge_pts" not in m.cols
    assert m.clf.monotonic_cst is None or not np.any(np.asarray(m.clf.monotonic_cst))


@pytest.mark.parametrize("market", ["spread", "total"])
def test_predictions_fall_as_line_rises(market):
    # the natural relation (harder line -> lower p): predictions fall with line
    df = synth(market, seasons=[2015, 2016], n=3000,
               logit=lambda d: 0.3 * d["f_edge_pts"] - 0.08 * (d["line"] - d["line"].mean()))
    m = pm.fit_market(df, market, max_iter=100)
    row = df.iloc[[0]]
    grid = np.linspace(df["line"].quantile(0.02), df["line"].quantile(0.98), 60)
    probe = pd.concat([row] * len(grid), ignore_index=True)
    probe["line"] = grid
    p = m.predict(probe)
    assert np.all(np.diff(p) <= 1e-12)
    assert p[-1] < p[0] - 0.05


@pytest.mark.parametrize("market", ["spread", "total"])
def test_line_constraint_holds_against_contrary_data(market):
    # data where p RISES with line: the constraint still forces non-increasing
    df = synth(market, seasons=[2015, 2016], n=3000,
               logit=lambda d: 0.3 * d["f_edge_pts"] + 0.1 * (d["line"] - d["line"].mean()))
    m = pm.fit_market(df, market, max_iter=100)
    grid = np.linspace(df["line"].min(), df["line"].max(), 80)
    for i in range(5):
        probe = pd.concat([df.iloc[[i]]] * len(grid), ignore_index=True)
        probe["line"] = grid
        assert np.all(np.diff(m.predict(probe)) <= 1e-12)
        eprobe = pd.concat([df.iloc[[i]]] * len(grid), ignore_index=True)
        eprobe["f_edge_pts"] = np.linspace(-10, 10, len(grid))
        assert np.all(np.diff(m.predict(eprobe)) >= -1e-12)


def test_fit_ignores_unlabelled_rows():
    df = synth("spread", seasons=[2015], n=1000)
    df.loc[:99, "y"] = np.nan
    m = pm.fit_market(df, "spread", max_iter=20)
    p = m.predict(df)
    assert p.shape == (len(df),) and np.all((p > 0) & (p < 1))


# --- walk-forward -----------------------------------------------------------

def test_walk_forward_oof_auc_and_columns():
    df = synth("spread", seasons=range(2015, 2021), n=1500)
    oof = pm.walk_forward_oof(df, "spread", [2018, 2019, 2020], max_iter=100)
    assert sorted(oof["season"].unique()) == [2018, 2019, 2020]
    assert len(oof) == (df["season"] >= 2018).sum()
    assert {"p_raw", "p"} <= set(oof.columns)
    assert roc_auc_score(oof["y"], oof["p_raw"]) > 0.6
    assert roc_auc_score(oof["y"], oof["p"]) > 0.6


def test_calibrated_ece_small_on_large_synthetic():
    df = synth("total", seasons=range(2015, 2025), n=3000, seed=3)
    oof = pm.walk_forward_oof(df, "total", list(range(2019, 2025)), max_iter=100)
    assert pm.ece(oof["p"].to_numpy(), oof["y"].to_numpy()) < 0.03


def test_walk_forward_never_trains_on_predicted_season():
    # planted: in 2019 only, a feature equal to the label (NaN elsewhere). A
    # model that saw 2019 would score ~1.0 AUC on it; the walk-forward must not.
    df = synth("spread", seasons=range(2015, 2021), n=1500, seed=5,
               logit=lambda d: 0.0 * d["f_edge_pts"])
    df["f_rest_home"] = np.where(df["season"] == 2019, df["y"], np.nan)
    oof = pm.walk_forward_oof(df, "spread", [2019], max_iter=50)
    assert roc_auc_score(oof["y"], oof["p_raw"]) < 0.6
    # control: the same model fitted including 2019 does exploit it
    leaky = pm.fit_market(df[df["season"] <= 2019], "spread", max_iter=50)
    s19 = df[df["season"] == 2019]
    assert roc_auc_score(s19["y"], leaky.predict(s19)) > 0.95


def test_season_predictions_ignore_own_and_later_labels_and_pre_min_train():
    df = synth("spread", seasons=range(2013, 2021), n=800, seed=7)
    base = pm.walk_forward_oof(df, "spread", [2019], max_iter=40)
    flipped = df.copy()
    sel = (flipped["season"] >= 2019) | (flipped["season"] < 2015)
    flipped.loc[sel, "y"] = 1.0 - flipped.loc[sel, "y"]
    again = pm.walk_forward_oof(flipped, "spread", [2019], max_iter=40)
    np.testing.assert_allclose(base["p_raw"].to_numpy(), again["p_raw"].to_numpy())
    np.testing.assert_allclose(base["p"].to_numpy(), again["p"].to_numpy())


def test_identity_calibration_until_two_prior_oof_seasons():
    df = synth("spread", seasons=range(2015, 2020), n=800)
    oof = pm.walk_forward_oof(df, "spread", [2016, 2017, 2018, 2019], max_iter=30)
    for s in (2016, 2017):   # 0 and 1 prior OOF seasons -> identity
        o = oof[oof["season"] == s]
        np.testing.assert_allclose(o["p"].to_numpy(), o["p_raw"].to_numpy())
    o = oof[oof["season"] == 2019]  # 3 prior OOF seasons -> isotonic
    assert not np.allclose(o["p"].to_numpy(), o["p_raw"].to_numpy())


def test_calibration_uses_prior_oof_even_when_not_requested():
    df = synth("spread", seasons=range(2015, 2020), n=800)
    only = pm.walk_forward_oof(df, "spread", [2019], max_iter=30)
    full = pm.walk_forward_oof(df, "spread", [2016, 2017, 2018, 2019], max_iter=30)
    np.testing.assert_allclose(only["p"].to_numpy(),
                               full.loc[full["season"] == 2019, "p"].to_numpy())


def test_unlabelled_rows_of_predicted_season_are_scored():
    df = synth("spread", seasons=range(2015, 2019), n=800)
    idx = df.index[df["season"] == 2018][:50]
    df.loc[idx, "y"] = np.nan
    oof = pm.walk_forward_oof(df, "spread", [2018], max_iter=30)
    assert len(oof) == (df["season"] == 2018).sum()
    assert oof["p"].notna().all()


def test_walk_forward_rejects_season_without_training_rows():
    df = synth("spread", seasons=range(2015, 2018), n=500)
    with pytest.raises(ValueError):
        pm.walk_forward_oof(df, "spread", [2015], max_iter=10)


def test_walk_forward_filters_to_market():
    df = pd.concat([synth("spread", seasons=[2015, 2016], n=300),
                    synth("total", seasons=[2015, 2016], n=300)], ignore_index=True)
    oof = pm.walk_forward_oof(df, "total", [2016], max_iter=10)
    assert (oof["market"] == "total").all() and len(oof) == 300


# --- calibration / metrics --------------------------------------------------

def test_ece_values():
    assert pm.ece(np.array([0.5, 0.5]), np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert pm.ece(np.array([0.9, 0.9, 0.9, 0.9]), np.array([1, 1, 0, 0])) == pytest.approx(0.4)
    # two bins, weighted by count: |0.15-0|*2/4 + |0.85-1|*2/4 = 0.15
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert pm.ece(p, np.array([0, 0, 1, 1])) == pytest.approx(0.15)
    assert pm.ece(np.array([0.0, 1.0]), np.array([0, 1])) == pytest.approx(0.0)


def test_calibrator_isotonic_clip_and_identity():
    rng = np.random.default_rng(0)
    p_raw = rng.uniform(0.2, 0.8, 5000)
    y = (rng.random(5000) < np.clip(p_raw * 1.5 - 0.25, 0, 1)).astype(float)
    cal = pm.fit_calibrator(p_raw, y)
    out = cal.predict(np.array([0.0, 0.2, 0.5, 0.8, 1.0]))
    assert np.all(np.diff(out) >= 0)
    assert out[0] == out[1] and out[-1] == out[-2]       # clipped outside range
    assert pm.ece(cal.predict(p_raw), y) < pm.ece(p_raw, y)
    ident = pm.Calibrator()
    np.testing.assert_allclose(ident.predict(np.array([0.1, 0.7])), [0.1, 0.7])


def test_fit_tolerates_all_nan_training_column():
    # f_move is all-NaN before openers exist (pre-2021); it must be ignored
    df = synth("spread", seasons=[2015], n=800)
    df["f_move"] = np.nan
    m = pm.fit_market(df, "spread", max_iter=20)
    probe = df.iloc[:50].copy()
    base = m.predict(probe)
    probe["f_move"] = np.linspace(-5, 5, 50)
    np.testing.assert_allclose(m.predict(probe), base)
