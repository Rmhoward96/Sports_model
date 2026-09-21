"""Tests for the pure helpers in scripts/train_cover_ensemble.py (Task 6).

Covers `fit_gbm`, `residual_sigma`, `fit_meta`, and `calibration_curve` --
the pure-ish wrappers around HistGradientBoostingRegressor / LogisticRegression
/ a reliability-diagram binner that `evaluate()`'s walk-forward loop
(IO-adjacent, not unit-tested end-to-end here; see the module docstring)
composes into the shipped cover/total ensemble -- plus a regression test
pinning `_walk_forward_market`'s leakage invariant (a test season is scored
by a model fit ONLY on strictly earlier seasons; review finding, fix round 1).

`evaluate()` itself, the season walk-forward + SHIP GATE + artifact writes,
is exercised end-to-end by running `scripts/train_cover_ensemble.py` against
the real `assets/nfl/cover_dataset.parquet` (see task-6-report.md for that
run's output), not by a synthetic unit test -- the walk-forward's whole
point is real out-of-sample performance on the real data.
"""
import importlib.util
import pathlib

import numpy as np
import pandas as pd

from sportsmodel.serving.ensemble import ensemble_prob

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "train_cover_ensemble.py"
_spec = importlib.util.spec_from_file_location("train_cover_ensemble", _p)
tce = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tce)


def _synthetic_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    eff_diff = rng.normal(0, 5, n)
    week = rng.integers(1, 18, n)
    home_field = np.ones(n)
    line = rng.normal(0, 3, n)
    X = np.column_stack([eff_diff, week, home_field, line])
    noise = rng.normal(0, 4, n)
    y = 0.6 * eff_diff - 0.3 * line + noise  # margin-like target
    return X, y


def test_fit_gbm_trains_and_predicts_finite_values():
    X, y = _synthetic_frame()
    model = tce.fit_gbm(X, y)
    preds = model.predict(X)
    assert np.isfinite(preds).all()
    assert len(preds) == len(y)


def test_residual_sigma_is_positive_float():
    X, y = _synthetic_frame()
    model = tce.fit_gbm(X, y)
    sigma = tce.residual_sigma(model, X, y)
    assert isinstance(sigma, float)
    assert sigma > 0.0


def test_fit_meta_on_separable_synthetic_probs_tracks_label():
    # Two base learners that agree and are both informative -- separable by
    # construction. fit_meta must learn coefficients such that
    # ensemble_prob's predicted side matches the label on (nearly) every row.
    rng = np.random.default_rng(1)
    n = 300
    labels = rng.integers(0, 2, n)
    base = []
    for y in labels:
        if y == 1:
            p1, p2 = rng.uniform(0.6, 0.95), rng.uniform(0.6, 0.95)
        else:
            p1, p2 = rng.uniform(0.05, 0.4), rng.uniform(0.05, 0.4)
        base.append((p1, p2))

    coef, intercept = tce.fit_meta(base, labels.tolist())
    assert len(coef) == 2
    assert isinstance(intercept, float)

    preds = [ensemble_prob(row, coef, intercept) for row in base]
    pred_labels = [int(p > 0.5) for p in preds]
    accuracy = sum(int(a == b) for a, b in zip(pred_labels, labels)) / n
    assert accuracy > 0.9


def test_fit_meta_coefficients_are_positive_for_agreeing_informative_bases():
    # Both base probs move with the label in the same direction -> both
    # meta coefficients should end up positive (the stack trusts each base).
    rng = np.random.default_rng(2)
    n = 300
    labels = rng.integers(0, 2, n)
    base = [(0.85, 0.8) if y == 1 else (0.15, 0.2) for y in labels]
    coef, intercept = tce.fit_meta(base, labels.tolist())
    assert coef[0] > 0
    assert coef[1] > 0


def test_calibration_curve_reports_near_zero_ece_when_well_calibrated():
    # Predictions generated as true Bernoulli(p) draws -- by construction,
    # p should track the empirical rate in every bin, so ECE should be small.
    rng = np.random.default_rng(3)
    probs = rng.uniform(0.05, 0.95, 3000)
    ys = (rng.uniform(0, 1, 3000) < probs).astype(int)

    result = tce.calibration_curve(probs, ys, n_bins=10)

    assert result["ece"] < 0.05
    assert len(result["bins"]) > 0
    total_count = sum(b["count"] for b in result["bins"])
    assert total_count == 3000
    for b in result["bins"]:
        assert b["count"] > 0
        assert 0.0 <= b["mean_pred"] <= 1.0
        assert 0.0 <= b["empirical_rate"] <= 1.0


def test_calibration_curve_flags_a_badly_miscalibrated_model():
    # A model that always predicts 0.9 when the true rate is 0.1 should
    # show a large gap in its one occupied bin and a large ECE.
    probs = [0.9] * 200
    ys = [0] * 180 + [1] * 20  # empirical rate 0.1
    result = tce.calibration_curve(probs, ys, n_bins=10)
    assert result["ece"] > 0.5
    assert len(result["bins"]) == 1
    assert result["bins"][0]["empirical_rate"] == 0.1


def test_calibration_curve_empty_input_is_safe():
    result = tce.calibration_curve([], [], n_bins=10)
    assert result == {"bins": [], "ece": 0.0}


def _leakage_probe_frame(seasons):
    """Small synthetic multi-season frame with the columns
    `_walk_forward_market` needs. `margin_actual` is set to the season
    number itself so a spy on `fit_gbm` can read off, from `y_train`'s max
    value, the latest season a given walk-forward iteration trained on --
    a direct, non-statistical check of the leakage invariant (as opposed to
    inferring it from noisy accuracy)."""
    rng = np.random.default_rng(7)
    rows = []
    for season in seasons:
        for i in range(20):
            rows.append({
                "season": season,
                "eff_diff": rng.normal(),
                "home_off_adj": rng.normal(),
                "home_def_adj": rng.normal(),
                "away_off_adj": rng.normal(),
                "away_def_adj": rng.normal(),
                "total_off": rng.normal(),
                "week": (i % 17) + 1,
                "home_field": 1,
                "spread_line": rng.normal(0, 3),
                "margin_actual": float(season),
                "home_cover": int(rng.integers(0, 2)),
                "ratings_cover_p": float(rng.uniform(0.3, 0.7)),
            })
    df = pd.DataFrame(rows)
    # Backfill any newer feature columns (e.g. Phase-2 market features) with
    # random values so the frame has the current feature set. (Real values are
    # irrelevant to this leakage test; using random floats rather than NaN
    # avoids an all-NaN column degenerating HistGradientBoosting on this tiny
    # synthetic frame.)
    for col in tce.MARGIN_FEATURES:
        if col not in df.columns:
            df[col] = rng.normal(size=len(df))
    return df


def test_walk_forward_market_never_trains_on_the_test_season_or_later(monkeypatch):
    seasons = [2015, 2016, 2017, 2018]
    df = _leakage_probe_frame(seasons)

    max_train_season_per_call = []
    real_fit_gbm = tce.fit_gbm

    def spy_fit_gbm(X, y, *args, **kwargs):
        # margin_actual (== y here) was set to the season number itself, so
        # its max over the training rows IS the latest season trained on.
        max_train_season_per_call.append(float(np.max(y)))
        return real_fit_gbm(X, y, *args, **kwargs)

    monkeypatch.setattr(tce, "fit_gbm", spy_fit_gbm)

    oof = tce._walk_forward_market(
        df, tce.MARGIN_FEATURES, "spread_line", "margin_actual", "home_cover",
        "ratings_cover_p", "margin", flip_line=True,
    )

    test_seasons = seasons[1:]  # 2015 has no prior season to train on
    assert len(max_train_season_per_call) == len(test_seasons)
    for test_season, max_train_season in zip(test_seasons, max_train_season_per_call):
        # The whole invariant: nothing from the test season (or later) ever
        # reaches fit_gbm's training rows.
        assert max_train_season < test_season

    assert sorted(oof["season"].unique().tolist()) == test_seasons
