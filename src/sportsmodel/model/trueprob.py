"""True-probability feature model: pure ridge-regression margin/total fit + predict.

Sub-project 3 (+EV engine), task 1 -- see
docs/superpowers/plans/2026-09-09-plus-ev-3-trueprob-model.md.

PURE numpy/pandas only -- no IO. Callers pass in a `features.parquet`-shaped
DataFrame (per-game rows with home/away `*_diff` feature columns) and get back
a plain-dict "Model" that `predict()` can apply to any compatible DataFrame.

No leakage: `feature_matrix()` computes the imputation mean and standardization
mu/sd from the TRAINING matrix only; at inference time the caller passes the
train-fit mu/sd back in so the test/holdout set never contributes its own
statistics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Ordered `*_diff` feature columns from assets/nfl/features.parquet (home - away).
# Home-field is implicit in every home-away diff, so no separate HFA term is needed
# here; `fit_ridge`'s unpenalized intercept absorbs any residual home-field constant.
# CFB (or any other sport) passes its own `features=` list to `fit_model` --
# this constant is NFL's default, not a hardcoded assumption baked into the fit.
FEATURES: list[str] = [
    "elo_diff",
    "last10_diff",
    "sos_diff",
    "sov_diff",
    "rest_diff",
    "off_epa_diff",
    "def_epa_diff",
]


def feature_matrix(
    df: pd.DataFrame,
    features: list[str],
    mu: np.ndarray | None = None,
    sd: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select `features` from `df`, impute NaN, and standardize.

    Train call (`mu`/`sd` omitted): the imputation value and the standardization
    mu/sd are all computed from THIS matrix.

    Test/inference call (`mu`/`sd` passed in, from a prior train call): NaNs are
    imputed to the TRAIN mean (`mu`), and standardization reuses the train mu/sd.
    Nothing about the test data itself is estimated, so there is no leakage.

    Returns (X_std, mu, sd) -- X_std has no intercept column (`fit_ridge` adds one).
    """
    raw = df.loc[:, features].to_numpy(dtype=float)

    if mu is None:
        # Train: impute to this matrix's own column mean (computed ignoring NaN),
        # then derive mu/sd from the imputed matrix.
        col_mean = np.nanmean(raw, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean)
        nan_mask = np.isnan(raw)
        imputed = np.where(nan_mask, col_mean, raw)
        mu = imputed.mean(axis=0)
        sd = imputed.std(axis=0)
    else:
        # Test: impute to the train mean; do not touch mu/sd.
        nan_mask = np.isnan(raw)
        imputed = np.where(nan_mask, mu, raw)

    sd_safe = np.where(sd == 0, 1.0, sd)
    X = (imputed - mu) / sd_safe
    return X, mu, sd


def fit_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Closed-form ridge regression with an unpenalized intercept.

    Prepends a ones column for the intercept, then solves
    `coef = (XtX + P)^-1 Xty` where `P` is `alpha * I` with the intercept's
    penalty zeroed out (row/col 0), via `np.linalg.solve` (not a matrix inverse).

    Returns the full coefficient vector, intercept first.
    """
    n = X.shape[0]
    Xd = np.hstack([np.ones((n, 1)), X])
    p = Xd.shape[1]
    penalty = np.eye(p) * alpha
    penalty[0, 0] = 0.0
    XtX = Xd.T @ Xd
    Xty = Xd.T @ y
    coef = np.linalg.solve(XtX + penalty, Xty)
    return coef


def fit_model(
    train_df: pd.DataFrame,
    target: str,
    features: list[str],
    alpha: float = 1.0,
) -> dict:
    """Fit a ridge margin/total model on `train_df`.

    Rows with a NaN target are dropped before fitting (feature NaNs are imputed
    by `feature_matrix`, not dropped).

    Returns a Model dict: {"coef", "mu", "sd", "features", "target"}.
    """
    clean = train_df.loc[train_df[target].notna()].reset_index(drop=True)
    X, mu, sd = feature_matrix(clean, features)
    y = clean[target].to_numpy(dtype=float)
    coef = fit_ridge(X, y, alpha)
    return {
        "coef": coef,
        "mu": mu,
        "sd": sd,
        "features": list(features),
        "target": target,
    }


def predict(model: dict, df: pd.DataFrame) -> np.ndarray:
    """Apply a fitted Model to `df`, reusing the model's train mu/sd (no leakage)."""
    X, _, _ = feature_matrix(df, model["features"], model["mu"], model["sd"])
    coef = model["coef"]
    return X @ coef[1:] + coef[0]


def residual_sigma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Std of residuals (y_true - y_pred) over finite pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    finite = np.isfinite(resid)
    return float(np.std(resid[finite]))
