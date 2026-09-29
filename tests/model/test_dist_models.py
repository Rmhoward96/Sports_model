"""Tests for the B per-market distribution models (props ML, Props-2).

All data is the synthetic `small_tbl` fixture below; never the real feature
parquet, never the network.
"""
from __future__ import annotations

import importlib.util
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits  # scikit-learn's own dependency

from sportsmodel.model.props_ml.dist_models import (
    LABELS,
    TAUS,
    ROLE_SUBSETS,
    MarketModel,
    bernoulli_pmf,
    fit_market,
    in_role,
    nb_pmf,
    predict_pmfs,
    quantiles_to_pmf,
)
from sportsmodel.model.props_ml.dist_models import _tier_dispersion, _tiers
from sportsmodel.sim.nfl.learned import feature_columns

# Support from the backtest's single MARKET_MAX constant (+ anytime_td, 2 bins).
_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_sim_nfl.py"
_spec = importlib.util.spec_from_file_location("backtest_sim_nfl", _p)
_bsn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bsn)
KMAX = {**_bsn.MARKET_MAX, "anytime_td": 1}
SEASONS = (2021, 2022, 2023)
WEEKS = range(1, 11)
UPTO = (2023, 4)
FIT = dict(upto=UPTO, test_season=2023, decay=0.8, max_iter=10)


@pytest.fixture(scope="module", autouse=True)
def _one_thread():
    """HGB's OpenMP threads cost more than they save on these tiny fits."""
    with threadpool_limits(1):
        yield


@pytest.fixture(scope="module")
def small_tbl() -> pd.DataFrame:
    """3 seasons x 10 weeks x 40 players; all seven labels; ~10% stub rows
    (NaN labels); ``p_all_nan`` is 100% NaN; rush yards can be negative."""
    rng = np.random.default_rng(11)
    positions = ["QB"] * 6 + ["RB"] * 10 + ["WR"] * 14 + ["TE"] * 10
    rows = []
    for season in SEASONS:
        for week in WEEKS:
            for i, pos in enumerate(positions):
                tshare = float(np.clip({"QB": .01, "RB": .12, "WR": .22, "TE": .15}[pos]
                                       + rng.normal(0, .05), 0, .5))
                cshare = float(np.clip({"QB": .1, "RB": .5, "WR": .02, "TE": .01}[pos]
                                       + rng.normal(0, .08), 0, .9))
                tgts = rng.poisson(0.5 + 35 * tshare)
                cars = rng.poisson(0.3 + 25 * cshare)
                rec = rng.binomial(tgts, 0.65)
                patt = rng.poisson(33) if pos == "QB" else 0
                ptds = rng.poisson(1.5) if pos == "QB" else 0
                row = {
                    "player_id": f"p{i}", "season": season, "week": week,
                    "position": pos, "is_stub": False,
                    "y_targets": float(tgts), "y_carries": float(cars),
                    "y_pass_att": float(patt), "y_receptions": float(rec),
                    "y_rec_yds": float(rec * 11 + (rng.normal(0, 4) if rec else 0)),
                    "y_rush_yds": float(cars * 4.2 + (rng.normal(0, 6) if cars else 0)),
                    "y_pass_yds": float(patt * 6.8 + (rng.normal(0, 25) if patt else 0)),
                    "y_pass_tds": float(ptds),
                    "y_anytime_td": float(rng.random() < 0.05 + 0.4 * (tshare + cshare)),
                    "p_target_share_ewm": tshare, "p_carry_share_ewm": cshare,
                    # role EWMs: QB p5 is a backup (< 10 pass att); week 1 has no
                    # history (NaN -> out of every usage role)
                    "p_y_pass_att_ewm": ((3.0 if i == 5 else 33.0) if pos == "QB" else 0.0)
                    if week > 1 else np.nan,
                    "p_y_carries_ewm": 25 * cshare if week > 1 else np.nan,
                    "p_y_targets_ewm": 0.5 + 35 * tshare if week > 1 else np.nan,
                    "p_all_nan": np.nan, "p_pos": pos,
                    "tm_pass_att_ewm": rng.uniform(28, 40),
                }
                if rng.random() < 0.10:
                    row["is_stub"] = True
                    for k in row:
                        if k.startswith("y_"):
                            row[k] = np.nan
                rows.append(row)
    df = pd.DataFrame(rows)
    df["p_pos"] = df["p_pos"].astype(pd.CategoricalDtype(["QB", "RB", "WR", "TE"]))
    return df


@pytest.fixture(scope="module")
def cols(small_tbl) -> list[str]:
    return feature_columns(small_tbl, frozenset())


@pytest.fixture(scope="module")
def fitted(small_tbl, cols):
    """All seven markets fitted once with ``FIT``, plus the spied fit calls per
    market (n rows, columns, labels, sample weights)."""
    models, calls = {}, {}
    with pytest.MonkeyPatch.context() as mp:
        log = _spy_fits(mp)
        for market in LABELS:
            log.clear()
            models[market] = fit_market(small_tbl, market, cols, **FIT)
            calls[market] = list(log)
    return models, calls


def _role_mask(df: pd.DataFrame, market: str) -> pd.Series:
    """The controller's role subsets, written out independently of ROLE_SUBSETS."""
    if market in ("pass_yds", "pass_tds"):
        return (df.position == "QB") & (df.p_y_pass_att_ewm >= 10)
    if market in ("rush_yds", "rush_att"):
        return df.p_y_carries_ewm >= 3
    if market in ("rec_yds", "receptions"):
        return df.position.isin(["WR", "TE", "RB"]) & (df.p_y_targets_ewm >= 2)
    return pd.Series(True, index=df.index)


def _train_mask(df: pd.DataFrame, label: str) -> pd.Series:
    s, w = UPTO
    market = {v: k for k, v in LABELS.items()}[label]
    before = (df.season < s) | ((df.season == s) & (df.week < w))
    return before & df[label].notna() & ~df["is_stub"] & _role_mask(df, market)


def _spy_fits(monkeypatch):
    calls = []
    for cls in (HistGradientBoostingRegressor, HistGradientBoostingClassifier):
        real = cls.fit

        def spy(self, X, y, sample_weight=None, _real=real):
            calls.append({"n": len(X), "cols": list(X.columns), "y": np.asarray(y),
                          "w": None if sample_weight is None else np.asarray(sample_weight)})
            return _real(self, X, y, sample_weight=sample_weight)

        monkeypatch.setattr(cls, "fit", spy)
    return calls


# ---- brief tests -------------------------------------------------------------

def test_quantiles_to_pmf_point_mass_and_normalization():
    q = np.zeros(19); q[10:] = [5, 8, 12, 15, 20, 25, 30, 40, 55]
    p = quantiles_to_pmf(q, 200)
    assert abs(p.sum() - 1) < 1e-9 and p[0] >= 0.5 - 1e-9 and p.argmax() == 0


def test_quantiles_to_pmf_unsorted_input_is_rearranged():
    q = np.linspace(10, 100, 19)
    assert np.allclose(quantiles_to_pmf(q[::-1], 200), quantiles_to_pmf(q, 200))


def test_nb_pmf_mean_and_poisson_limit():
    p = nb_pmf(4.0, 3.0, 60); k = np.arange(61)
    assert abs((k * p).sum() - 4.0) < 1e-3 and abs(p.sum() - 1) < 1e-9
    assert np.allclose(nb_pmf(4.0, None, 60), nb_pmf(4.0, 1e9, 60), atol=1e-9)


def test_nb_pmf_large_r_approaches_poisson():
    # r = 1e5 takes the NB branch (not the > 1e6 Poisson shortcut)
    assert np.allclose(nb_pmf(4.0, 1e5, 60), nb_pmf(4.0, None, 60), atol=1e-3)
    assert not np.array_equal(nb_pmf(4.0, 1e5, 60), nb_pmf(4.0, None, 60))


def test_nb_pmf_nan_mu_raises():
    with pytest.raises(ValueError):
        nb_pmf(float("nan"), 2.0, 10)


def test_fit_market_respects_upto(small_tbl, fitted):
    _, calls = fitted  # rows at/after upto never train
    for market, label in LABELS.items():
        assert calls[market], market
        expected = int(_train_mask(small_tbl, label).sum())
        assert {c["n"] for c in calls[market]} == {expected}, market


def test_predict_pmfs_shapes(small_tbl, fitted):
    rows = small_tbl[(small_tbl.season == 2023) & (small_tbl.week == 5)].head(12)
    for market in LABELS:
        m = fitted[0][market]
        pmfs = predict_pmfs(m, rows, KMAX[market])
        assert len(pmfs) == len(rows)
        n_bins = 2 if market == "anytime_td" else KMAX[market] + 1
        for p in pmfs:
            assert p.shape == (n_bins,)
            assert abs(p.sum() - 1) < 1e-9 and (p >= 0).all()


# ---- pure helpers ---------------------------------------------------------------

def test_taus():
    assert len(TAUS) == 19 and TAUS[0] == 0.05 and TAUS[-1] == 0.95 and TAUS[9] == 0.5


def test_quantiles_to_pmf_cdf_rule():
    # q_i = 10 * (i + 1): F linear between knots (-0.5, 0), (10, .05), ..., (190, .95), (200.5, 1)
    q = 10.0 * np.arange(1, 20)
    p = quantiles_to_pmf(q, 200)
    F = np.concatenate([[0.0], np.cumsum(p)])  # F(k + 0.5) for k = -1..200
    assert F[1 + 10] == pytest.approx(0.05 + 0.005 * 0.5)  # F(10.5)
    assert F[1 + 0] == pytest.approx(0.05 * 1.0 / 10.5)    # F(0.5)
    assert F[1 + 195] == pytest.approx(0.95 + 0.05 * 5.5 / 10.5)  # F(195.5)


def test_quantiles_to_pmf_clips_to_support():
    hi = quantiles_to_pmf(np.full(19, 500.0), 200)
    assert hi.shape == (201,) and abs(hi.sum() - 1) < 1e-9 and hi[200] >= 0.95 - 1e-9
    lo = quantiles_to_pmf(np.full(19, -30.0), 200)
    assert abs(lo.sum() - 1) < 1e-9 and lo[0] >= 0.95 - 1e-9


def test_quantiles_to_pmf_mid_range_tie_is_a_cdf_step():
    # taus 0.40..0.60 (indices 7..11) all at 20 -> F jumps .40 -> .60 at 20
    q = np.concatenate([2.0 * np.arange(1, 8), np.full(5, 20.0), 10.0 * np.arange(3, 10)])
    assert (np.diff(q) >= 0).all() and list(TAUS[7:12]) == [0.4, 0.45, 0.5, 0.55, 0.6]
    p = quantiles_to_pmf(q, 200)
    assert abs(p.sum() - 1) < 1e-9
    assert p[20] == pytest.approx(0.20, abs=0.01)
    assert p.argmax() == 20


def test_nb_pmf_tail_folded_and_zero_mean():
    p = nb_pmf(30.0, 2.0, 10)
    assert p.shape == (11,) and abs(p.sum() - 1) < 1e-9 and p[10] > 0.5
    z = nb_pmf(0.0, 2.0, 6)
    assert z[0] == pytest.approx(1.0) and abs(z.sum() - 1) < 1e-12


def test_nb_pmf_matches_closed_form():
    mu, r = 2.5, 1.7
    pr = r / (r + mu)
    k = 3
    want = math.gamma(k + r) / (math.gamma(r) * math.factorial(k)) * pr ** r * (1 - pr) ** k
    assert nb_pmf(mu, r, 40)[k] == pytest.approx(want, rel=1e-10)
    lam = 2.5
    assert nb_pmf(lam, None, 40)[k] == pytest.approx(math.exp(-lam) * lam ** k / 6, rel=1e-10)


def test_bernoulli_pmf():
    assert np.allclose(bernoulli_pmf(0.3), [0.7, 0.3])


# ---- fit/predict details -----------------------------------------------------------

def test_fit_market_excludes_stub_rows_even_with_labels(small_tbl, cols, monkeypatch):
    df = small_tbl.copy()
    df.loc[df["is_stub"], "y_receptions"] = 0.0  # a labelled stub must still not train
    calls = _spy_fits(monkeypatch)
    fit_market(df, "receptions", cols, **FIT)
    assert calls[0]["n"] == int(_train_mask(small_tbl, "y_receptions").sum())


def test_fit_market_future_rows_do_not_change_predictions(small_tbl, cols, fitted):
    s, w = UPTO
    past = small_tbl[(small_tbl.season < s) | ((small_tbl.season == s) & (small_tbl.week < w))]
    rows = small_tbl[(small_tbl.season == 2023) & (small_tbl.week == 6)]
    for market in ("rush_yds", "receptions", "anytime_td"):
        a = predict_pmfs(fitted[0][market], rows, KMAX[market])
        b = predict_pmfs(fit_market(past, market, cols, **FIT), rows, KMAX[market])
        assert all(np.array_equal(x, y) for x, y in zip(a, b)), market


def test_fit_market_weights_clip_and_drop_all_nan(small_tbl, fitted):
    m, calls = fitted[0]["rush_yds"], fitted[1]["rush_yds"]
    tr = small_tbl[_train_mask(small_tbl, "y_rush_yds")]
    assert (tr["y_rush_yds"] < 0).any()  # fixture really has negative yards
    assert len(calls) == len(TAUS)
    for c in calls:
        assert np.allclose(c["w"], 0.8 ** (2023 - tr["season"].to_numpy(dtype=float)))
        assert c["y"].min() >= 0
        assert "p_all_nan" not in c["cols"]
    assert "p_all_nan" not in m.cols and "p_pos" in m.cols
    assert m.kind == "quantile" and len(m.models) == len(TAUS)


def test_market_kinds_and_settings(fitted):
    kinds = {"rec_yds": "quantile", "rush_yds": "quantile", "pass_yds": "quantile",
             "receptions": "count", "rush_att": "count", "pass_tds": "count",
             "anytime_td": "binary"}
    for market, kind in kinds.items():
        m = fitted[0][market]
        assert isinstance(m, MarketModel) and m.market == market and m.kind == kind
        for est in m.models:
            p = est.get_params()
            assert (p["learning_rate"], p["max_leaf_nodes"], p["min_samples_leaf"],
                    p["l2_regularization"], p["categorical_features"], p["random_state"],
                    p["early_stopping"], p["max_iter"]) == (0.05, 31, 50, 1.0, "from_dtype", 0, False, 10)
        if kind == "quantile":
            assert [e.quantile for e in m.models] == list(TAUS)
            assert all(e.loss == "quantile" for e in m.models) and m.dispersion is None
        elif kind == "count":
            assert len(m.models) == 1 and m.models[0].loss == "poisson"
            assert len(m.dispersion["cuts"]) == 2 and len(m.dispersion["r"]) == 3
        else:
            assert len(m.models) == 1 and isinstance(m.models[0], HistGradientBoostingClassifier)
            assert m.dispersion is None


def test_count_dispersion_tiers_used_at_prediction(small_tbl, fitted):
    m = fitted[0]["rush_att"]
    rows = small_tbl[(small_tbl.season == 2023) & (small_tbl.week == 7)]
    mu = np.clip(m.models[0].predict(rows[m.cols]), 0, None)
    c1, c2 = m.dispersion["cuts"]
    tiers = np.where(mu <= c1, 0, np.where(mu <= c2, 1, 2))
    pmfs = predict_pmfs(m, rows, 40)
    for p, mu_i, t in zip(pmfs, mu, tiers):
        assert np.allclose(p, nb_pmf(float(mu_i), m.dispersion["r"][t], 40))


def test_count_dispersion_method_of_moments(small_tbl, fitted):
    m = fitted[0]["rush_att"]
    tr = small_tbl[_train_mask(small_tbl, "y_carries")]
    mu = m.models[0].predict(tr[m.cols])
    cuts = np.quantile(mu, [1 / 3, 2 / 3])
    assert np.allclose(m.dispersion["cuts"], cuts)
    tier = np.where(mu <= cuts[0], 0, np.where(mu <= cuts[1], 1, 2))
    y = tr["y_carries"].to_numpy(dtype=float)
    for t in range(3):
        mt, yt = mu[tier == t], y[tier == t]
        den = ((yt - mt) ** 2 - mt).sum()
        want = (mt ** 2).sum() / den if den > 0 else None
        got = m.dispersion["r"][t]
        assert (got is None) if want is None else got == pytest.approx(want)


def test_tier_dispersion_constant_mu_matches_unconditional_form():
    rng = np.random.default_rng(3)
    mu = np.repeat([1.0, 4.0, 9.0], 20000)
    y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu)).astype(float)
    d = _tier_dispersion(mu, y)
    for t, m in enumerate((1.0, 4.0, 9.0)):
        yt = y[mu == m]
        # equal up to (mean(y) - mu)^2, i.e. sampling noise
        assert d["r"][t] == pytest.approx(m ** 2 / (yt.var() - m), rel=2e-3)
        assert d["r"][t] == pytest.approx(2.0, rel=0.15)


def test_tier_dispersion_ignores_spread_of_mu_within_tier():
    rng = np.random.default_rng(4)
    mu = rng.uniform(0.5, 30.0, 60000)
    y_pois = rng.poisson(mu).astype(float)
    assert all(r is None or r > 100 for r in _tier_dispersion(mu, y_pois)["r"])
    y_nb = rng.negative_binomial(3.0, 3.0 / (3.0 + mu)).astype(float)
    for r in _tier_dispersion(mu, y_nb)["r"]:
        assert r == pytest.approx(3.0, rel=0.15)


def test_tier_ties_stay_in_lower_tier():
    mu = np.array([0.03] * 70 + [1.5] * 30)
    y = np.array([0.0] * 70 + [1.0, 2.0, 1.0] * 10)
    d = _tier_dispersion(mu, y)
    assert d["cuts"] == [pytest.approx(0.03), pytest.approx(0.03)]
    assert list(_tiers(d["cuts"], np.array([0.03, 1.5]))) == [0, 2]
    assert d["r"][1] is None  # empty middle tier


def test_fit_market_unweighted_when_decay_is_one(small_tbl, cols, monkeypatch):
    calls = _spy_fits(monkeypatch)
    for market in ("receptions", "anytime_td", "rush_yds"):
        calls.clear()
        fit_market(small_tbl, market, cols, **{**FIT, "decay": 1.0})
        assert calls and all(c["w"] is None for c in calls), market
        calls.clear()
        fit_market(small_tbl, market, cols, **{**FIT, "decay": 0.9})
        assert calls and all(isinstance(c["w"], np.ndarray) for c in calls), market


def test_fit_market_no_training_rows_raises(small_tbl, cols):
    with pytest.raises(ValueError):
        fit_market(small_tbl, "receptions", cols, upto=(2000, 1), test_season=2021,
                   decay=1.0, max_iter=10)


def test_predict_pmfs_empty_rows(small_tbl, fitted):
    m = fitted[0]["anytime_td"]
    assert predict_pmfs(m, small_tbl.iloc[:0], 1) == []


# ---- role subsets (controller fix round 1) -------------------------------------------

def test_role_subsets_definition():
    assert ROLE_SUBSETS["pass_yds"] == ROLE_SUBSETS["pass_tds"] == {
        "positions": ("QB",), "col": "p_y_pass_att_ewm", "min": 10.0}
    assert ROLE_SUBSETS["rush_yds"] == ROLE_SUBSETS["rush_att"] == {
        "positions": None, "col": "p_y_carries_ewm", "min": 3.0}
    assert ROLE_SUBSETS["rec_yds"] == ROLE_SUBSETS["receptions"] == {
        "positions": ("WR", "TE", "RB"), "col": "p_y_targets_ewm", "min": 2.0}
    assert ROLE_SUBSETS["anytime_td"] is None


def test_in_role_thresholds_positions_and_nan():
    df = pd.DataFrame({
        "position": ["QB", "QB", "WR", "QB", "RB", "TE", "QB", "WR"],
        "p_y_pass_att_ewm": [10.0, 9.9, 30.0, np.nan, 0, 0, 0, 0],
        "p_y_carries_ewm": [0, 0, 0, 0, 3.0, 2.9, 5.0, np.nan],
        "p_y_targets_ewm": [0, 0, 2.0, 0, 1.9, 2.5, 6.0, np.nan],
    })
    assert in_role(df, "pass_yds").tolist() == [True, False, False, False, False, False, False, False]
    assert in_role(df, "rush_att").tolist() == [False, False, False, False, True, False, True, False]
    assert in_role(df, "receptions").tolist() == [False, False, True, False, False, True, False, False]
    assert in_role(df, "anytime_td").all()
    assert in_role(df, "rec_yds").dtype == bool


def test_fit_market_trains_only_in_role_rows(small_tbl, cols, monkeypatch):
    calls = _spy_fits(monkeypatch)
    for market in ("pass_yds", "rush_att", "receptions", "anytime_td"):
        calls.clear()
        fit_market(small_tbl, market, cols, **FIT)
        label = LABELS[market]
        want = _train_mask(small_tbl, label)
        without_role = (_train_mask(small_tbl.assign(position="QB", p_y_pass_att_ewm=99.0,
                                                     p_y_carries_ewm=99.0), label)
                        if market != "receptions" else want)
        assert {c["n"] for c in calls} == {int(want.sum())}, market
        if market in ("pass_yds", "rush_att"):
            assert int(want.sum()) < int(without_role.sum())  # the role really excludes rows
    # a QB backup (p5, pass-att EWM 3) and week-1 (NaN EWM) rows never train pass_yds
    calls.clear()
    fit_market(small_tbl, "pass_yds", cols, **FIT)
    tr = small_tbl[_train_mask(small_tbl, "y_pass_yds")]
    assert "p5" not in set(tr.player_id) and 1 not in set(tr.week)
    assert set(tr.position) == {"QB"}


# ---- stale-QB role (v2 role table) --------------------------------------------------

from sportsmodel.model.props_ml import dist_models  # noqa: E402


def test_stale_qb_is_out_of_role():
    df = pd.DataFrame({"position": ["QB", "QB"], "p_y_pass_att_ewm": [15.7, 30.0],
                       "p_y_pass_att_r10": [np.nan, 28.0]})
    assert dist_models.in_role(df, "pass_yds", subsets=dist_models.ROLE_SUBSETS_V2
                               ).tolist() == [False, True]
    assert dist_models.in_role(df, "pass_tds", subsets=dist_models.ROLE_SUBSETS_V2
                               ).tolist() == [False, True]


def test_stale_qb_stays_in_role_under_default_v1_table():
    df = pd.DataFrame({"position": ["QB", "QB"], "p_y_pass_att_ewm": [15.7, 30.0],
                       "p_y_pass_att_r10": [np.nan, 28.0]})
    assert dist_models.in_role(df, "pass_yds").tolist() == [True, True]
    assert dist_models.in_role(df, "pass_yds", subsets=ROLE_SUBSETS).tolist() == [True, True]


def test_role_subsets_v2_only_adds_the_recent_attempts_condition():
    v2 = dist_models.ROLE_SUBSETS_V2
    assert set(v2) == set(ROLE_SUBSETS)
    for m in ROLE_SUBSETS:
        if m not in ("pass_yds", "pass_tds"):
            assert v2[m] == ROLE_SUBSETS[m]
    df = pd.DataFrame({"position": ["QB", "QB", "QB", "WR"],
                       "p_y_pass_att_ewm": [9.0, 12.0, 12.0, 30.0],
                       "p_y_pass_att_r10": [5.0, 0.0, 1.0, 30.0]})
    # ewm >= 10 (inclusive) AND r10 > 0 (strict), QB only
    assert dist_models.in_role(df, "pass_yds", subsets=v2).tolist() == [False, False, True, False]
    assert dist_models.in_role(df, "pass_yds", subsets=v2).dtype == bool


def test_fit_market_threads_role_subsets(small_tbl, cols, monkeypatch):
    stale = small_tbl.assign(p_y_pass_att_r10=np.where(small_tbl.week % 2 == 0, np.nan, 20.0))
    calls = _spy_fits(monkeypatch)
    fit_market(stale, "pass_yds", cols, **FIT)
    n_v1 = {c["n"] for c in calls}
    calls.clear()
    fit_market(stale, "pass_yds", cols, **FIT, subsets=dist_models.ROLE_SUBSETS_V2)
    n_v2 = {c["n"] for c in calls}
    want_v2 = _train_mask(stale, "y_pass_yds") & (stale.p_y_pass_att_r10 > 0)
    assert n_v1 == {int(_train_mask(stale, "y_pass_yds").sum())}
    assert n_v2 == {int(want_v2.sum())} and int(want_v2.sum()) < min(n_v1)
