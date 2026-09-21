"""Tests for the pure helpers in scripts/train_cover_ensemble.py (Task 6).

Covers `fit_gbm`, `residual_sigma`, and `fit_meta` -- the pure-ish wrappers
around HistGradientBoostingRegressor / LogisticRegression that
`evaluate()`'s walk-forward loop (IO-adjacent, not unit-tested here; see the
module docstring) composes into the shipped cover/total ensemble.

`evaluate()` itself, the season walk-forward + SHIP GATE + artifact writes,
is exercised end-to-end by running `scripts/train_cover_ensemble.py` against
the real `assets/nfl/cover_dataset.parquet` (see task-6-report.md for that
run's output), not by a synthetic unit test -- the walk-forward's whole
point is real out-of-sample performance on the real data.
"""
import importlib.util
import pathlib

import numpy as np

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
