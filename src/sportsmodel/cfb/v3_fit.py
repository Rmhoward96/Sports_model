"""Fit the v3 weights on the training seasons of the v3 feature table (2016-2022). PURE.

1. points map: ridge-regularised least squares of each offense's actual points on its efficiency
   side features (both sides of every game stacked).
2. blends: stage 1 = least squares on the core terms (v2 / efficiency / prior / home field);
   stage 2 = lasso on the stage-1 residual for the context terms, the penalty chosen by
   leave-one-season-out MAE with the one-standard-error rule (SE over the fold MAEs), so a context term that does not
   help is fitted to EXACTLY 0 (the kept terms are refit unpenalised, "relaxed lasso").
   `weather_missing` is a nuisance column: fitted (so the other weather terms are estimated from
   games that have weather) but dropped from the served model.
3. sigmas: RMSE of the fitted v3 margin / total on the training seasons (moneyline mapping).

Cross-validation scope: stage 1 (the core least squares) and the points map are fitted ONCE on all
training seasons; only the context penalty (the lasso alpha) is cross-validated, leave-one-season-out.
The CV MAEs therefore do not re-fit stage 1 or the points map inside each fold.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from .context import MARGIN_CTX, NUISANCE, TOTAL_CTX
from .efficiency import POINT_FEATURES
from .v3 import (MARGIN_CORE, TOTAL_CORE, LinearBlend, V3Weights, eff_margin_total, predict_v3)

TRAIN_SEASONS = tuple(range(2016, 2023))
LASSO_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
POINTS_RIDGE = 1.0


def _design(df: pd.DataFrame, cols) -> np.ndarray:
    return np.nan_to_num(df[list(cols)].to_numpy(float), nan=0.0)


def fit_points_map(table: pd.DataFrame, train_seasons=TRAIN_SEASONS, ridge: float = POINTS_RIDGE) -> LinearBlend:
    """Ridge LS of a side's actual points on POINT_FEATURES (intercept unpenalised)."""
    t = table[table["season"].isin(train_seasons) & table["actual_margin"].notna()]
    home_pts, away_pts = (t["actual_total"] + t["actual_margin"]) / 2, (t["actual_total"] - t["actual_margin"]) / 2
    x = np.vstack([_design(t.rename(columns={f"h_{f}": f for f in POINT_FEATURES}), POINT_FEATURES),
                   _design(t.rename(columns={f"a_{f}": f for f in POINT_FEATURES}), POINT_FEATURES)])
    y = np.concatenate([home_pts.to_numpy(float), away_pts.to_numpy(float)])
    x1 = np.c_[np.ones(len(x)), x]
    pen = np.eye(x1.shape[1]) * ridge
    pen[0, 0] = 0.0
    beta = np.linalg.solve(x1.T @ x1 + pen, x1.T @ y)
    return LinearBlend(float(beta[0]), {f: float(b) for f, b in zip(POINT_FEATURES, beta[1:])})


def add_eff(table: pd.DataFrame, pm: LinearBlend) -> pd.DataFrame:
    """table + margin_eff / total_eff columns from the points map."""
    out = table.copy()
    mt = [eff_margin_total(pm, r) for r in table.to_dict("records")]
    out["margin_eff"] = [m for m, _ in mt]
    out["total_eff"] = [t for _, t in mt]
    return out


def fit_blend(df: pd.DataFrame, y: np.ndarray, core, ctx, nuisance=(), alphas=LASSO_ALPHAS) -> tuple[LinearBlend, dict]:
    """Two-stage fit (see module docstring). Returns (blend, info)."""
    x1 = np.c_[np.ones(len(df)), _design(df, core)]
    beta = np.linalg.lstsq(x1, y, rcond=None)[0]
    resid = y - x1 @ beta
    zc = list(ctx) + list(nuisance)
    z = _design(df, zc)
    sd = z.std(axis=0)
    sd[sd == 0] = 1.0
    zs = z / sd
    seasons = df["season"].to_numpy()

    def lasso(a, tr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return Lasso(alpha=a, max_iter=20000).fit(zs[tr], resid[tr])

    # leave-one-season-out: one fold per training season; the CV error of an alpha is its fold MAEs
    folds = np.unique(seasons)
    fold_mae = {None: np.array([np.abs(resid[seasons == s]).mean() for s in folds])}
    for a in alphas:
        maes = []
        for s in folds:
            te, tr = seasons == s, seasons != s
            maes.append(np.abs(resid[te] - lasso(a, tr).predict(zs[te])).mean())
        fold_mae[a] = np.array(maes)
    cv = {a: float(m.mean()) for a, m in fold_mae.items()}
    best_a = min(cv, key=cv.get)
    # one-standard-error rule, glmnet style: the SE is the spread of the best alpha's FOLD MAEs over
    # sqrt(n_folds) (NOT the per-game spread of |error|, which is game-to-game noise and swamps any
    # real gain). Take the most penalised alpha within 1 SE of the best CV MAE, so a context term
    # has to earn its place (None = no context terms at all = the largest penalty)
    se = float(fold_mae[best_a].std(ddof=1) / np.sqrt(len(folds)))
    ok = [a for a in cv if cv[a] <= cv[best_a] + se]
    best = None if None in ok else max(ok)
    coefs = {c: 0.0 for c in ctx}
    intercept = float(beta[0])
    if best is not None:
        m = lasso(best, np.ones(len(df), bool))
        # relaxed lasso: the penalty only SELECTS the context terms; the kept ones (plus the
        # nuisance columns) are refit by plain least squares so their sizes are not shrunk
        keep = [i for i, c in enumerate(zc) if m.coef_[i] != 0.0 or c in nuisance]
        if keep:
            xs = np.c_[np.ones(len(df)), zs[:, keep]]
            b = np.linalg.lstsq(xs, resid, rcond=None)[0]
            coefs.update({zc[i]: float(b[1 + j] / sd[i]) for j, i in enumerate(keep) if zc[i] not in nuisance})
            intercept += float(b[0])
    core_coefs = {c: float(b) for c, b in zip(core, beta[1:])}
    return (LinearBlend(intercept, {**core_coefs, **coefs}),
            {"lasso_alpha": best, "cv_se": se, "cv_mae": {str(k): v for k, v in cv.items()},
             "fold_mae": {str(k): [float(x) for x in m] for k, m in fold_mae.items()}})


def fit_v3(table: pd.DataFrame, train_seasons=TRAIN_SEASONS) -> V3Weights:
    """Fit the whole of v3 on `train_seasons` of `table` (build_table output)."""
    pm = fit_points_map(table, train_seasons)
    tr = add_eff(table[table["season"].isin(train_seasons) & table["actual_margin"].notna()], pm)
    margin, mi = fit_blend(tr, tr["actual_margin"].to_numpy(float), MARGIN_CORE, MARGIN_CTX)
    total, ti = fit_blend(tr, tr["actual_total"].to_numpy(float), TOTAL_CORE, TOTAL_CTX, NUISANCE)
    w = V3Weights(pm, margin, total, {})
    preds = np.array([predict_v3(r, w) for r in tr.to_dict("records")])
    sig_m = float(np.sqrt(np.mean((preds[:, 0] - tr["actual_margin"].to_numpy()) ** 2)))
    sig_t = float(np.sqrt(np.mean((preds[:, 1] - tr["actual_total"].to_numpy()) ** 2)))
    meta = {"train_seasons": [int(s) for s in train_seasons], "n_train": int(len(tr)),
            "sigma_margin": sig_m, "sigma_total": sig_t, "margin": mi, "total": ti,
            "train_mae": {"margin": float(np.abs(preds[:, 0] - tr["actual_margin"]).mean()),
                          "total": float(np.abs(preds[:, 1] - tr["actual_total"]).mean())}}
    return V3Weights(pm, margin, total, meta)
