"""CFB profit model: per-market calibrated probability models.

One `HistGradientBoostingClassifier` per market (spread / total / moneyline)
on `profit_features.FEATURE_COLS[market]`, predicting the feature table's
label `y` (home covers / over / home wins). Monotone constraints (plan ruling
P3), built by column name: -1 on `line` for spread and total (a higher home
line / higher total => lower P(home covers) / P(over)) and +1 on `f_edge_pts`
wherever it is a feature. Moneyline has neither.

`walk_forward_oof` is the season walk-forward: season S is predicted by a
model trained only on labelled rows of seasons [min_train_season, S) -- both
price points -- and its isotonic calibrator is fitted on the out-of-fold raw
predictions of the seasons strictly before S (identity until 2 such seasons
exist). The calibration pool is "mature" out-of-fold seasons only -- seasons
s >= min_train_season + CAL_MIN_TRAIN_SEASONS, i.e. predicted by models trained
on >= 3 seasons -- produced for every such season before S whether or not it
was requested, so a season's calibrator does not depend on the request list.
Calibrated probabilities are clipped to [P_MIN, P_MAX] so Kelly never sees 0/1.
Rows with y = NaN (unscored games) are never trained or calibrated on; they
are still scored.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from sportsmodel.cfb.profit_features import FEATURE_COLS

# per-market monotone constraints by feature name (P3)
MONOTONE: dict[str, dict[str, int]] = {
    "spread": {"line": -1, "f_edge_pts": 1},
    "total": {"line": -1, "f_edge_pts": 1},
    "moneyline": {"f_edge_pts": 1},   # f_edge_pts is not a moneyline feature
}
MIN_CAL_SEASONS = 2          # calibrate only with >= 2 pooled OOF seasons
CAL_MIN_TRAIN_SEASONS = 3    # pooled OOF seasons come from models trained on >= 3
P_MIN, P_MAX = 0.01, 0.99    # calibrated-probability clip

# HGB defaults (tunable per market by the gate on 2019-2022 OOF log-loss)
HGB_DEFAULTS = {"learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 100,
                "l2_regularization": 1.0}


def monotonic_cst(market: str, cols: list[str]) -> list[int]:
    """Constraint array aligned to `cols` (0 = unconstrained)."""
    cst = MONOTONE[market]
    return [int(cst.get(c, 0)) for c in cols]


def _X(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.array(df[cols].to_numpy(dtype=np.float64), copy=True)


@dataclass
class MarketModel:
    market: str
    cols: list[str]
    clf: HistGradientBoostingClassifier

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Raw (uncalibrated) P(y = 1) per row."""
        if len(df) == 0:
            return np.empty(0)
        return self.clf.predict_proba(_X(df, self.cols))[:, 1]


def fit_market(df: pd.DataFrame, market: str, *, learning_rate: float = 0.05,
               max_iter: int = 300, min_samples_leaf: int = 100,
               l2_regularization: float = 1.0, seed: int = 0) -> MarketModel:
    """Fit the market's classifier on the labelled rows of `df` (rows of that
    market; y-NaN rows dropped). early_stopping is off explicitly: sklearn's
    'auto' switches it on above 10k rows, which the live all-seasons fit would
    cross but the gate's walk-forward fits do not."""
    cols = list(FEATURE_COLS[market])
    train = df[df["y"].notna()]
    if "market" in train.columns:
        train = train[train["market"] == market]
    if train.empty:
        raise ValueError(f"no labelled {market} rows to train on")
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=31,
        min_samples_leaf=min_samples_leaf, l2_regularization=l2_regularization,
        max_iter=max_iter, early_stopping=False, random_state=seed,
        monotonic_cst=monotonic_cst(market, cols))
    X = _X(train, cols)
    # sklearn cannot bin an all-NaN column (e.g. f_move before openers exist,
    # 2021+): a constant stand-in gives it no split, so the model ignores it.
    X[:, np.isnan(X).all(axis=0)] = 0.0
    clf.fit(X, train["y"].to_numpy(dtype=int))
    return MarketModel(market=market, cols=cols, clf=clf)


@dataclass
class Calibrator:
    """Isotonic map raw p -> calibrated p (clipped outside the fitted range);
    identity when unfitted. Output is clipped to [P_MIN, P_MAX] either way."""
    iso: IsotonicRegression | None = field(default=None)

    def predict(self, p) -> np.ndarray:
        p = np.asarray(p, dtype=np.float64)
        out = p.copy() if (self.iso is None or p.size == 0) else self.iso.predict(p)
        return np.clip(out, P_MIN, P_MAX)


def fit_calibrator(p_raw, y) -> Calibrator:
    p_raw = np.asarray(p_raw, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = ~(np.isnan(p_raw) | np.isnan(y))
    if not ok.any():
        return Calibrator()
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(p_raw[ok], y[ok])
    return Calibrator(iso=iso)


def ece(p, y, bins: int = 10) -> float:
    """Expected calibration error over `bins` equal-width bins on [0, 1]:
    sum_b (n_b / N) * |mean p_b - mean y_b|. NaN pairs are ignored."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = ~(np.isnan(p) | np.isnan(y))
    p, y = p[ok], y[ok]
    if p.size == 0:
        return float("nan")
    idx = np.clip(np.floor(p * bins).astype(int), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() * abs(p[m].mean() - y[m].mean())
    return float(total / p.size)


def walk_forward_oof(df: pd.DataFrame, market: str, seasons: list[int],
                     min_train_season: int = 2015, *, learning_rate: float = 0.05,
                     max_iter: int = 300, min_samples_leaf: int = 100,
                     l2_regularization: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Out-of-fold predictions for the requested seasons (all rows of the
    market in those seasons, labelled or not) with `p_raw` (model trained on
    labelled rows of seasons [min_train_season, S)) and `p` (isotonic
    calibrator fitted on the OOF p_raw / y of the mature seasons
    [min_train_season + 3, S); identity with fewer than 2 such seasons; clipped
    to [P_MIN, P_MAX]). HGB hyperparameters pass through to fit_market."""
    if not seasons:
        raise ValueError("no seasons requested")
    d = df[df["market"] == market] if "market" in df.columns else df
    requested = sorted({int(s) for s in seasons})
    labelled = d["y"].notna()
    for s in requested:
        n_train = (labelled & (d["season"] >= min_train_season) & (d["season"] < s)).sum()
        if n_train == 0:
            raise ValueError(f"season {s}: no labelled {market} training rows "
                             f"in [{min_train_season}, {s})")

    hgb = {"learning_rate": learning_rate, "max_iter": max_iter,
           "min_samples_leaf": min_samples_leaf,
           "l2_regularization": l2_regularization, "seed": seed}
    # raw OOF for the requested seasons plus every mature pool season before
    # the last requested one
    last = requested[-1]
    first_pool = min_train_season + CAL_MIN_TRAIN_SEASONS
    pool_seasons = {int(s) for s in d["season"].unique() if first_pool <= s < last}
    oof_seasons = sorted(pool_seasons | set(requested))
    raw: dict[int, pd.DataFrame] = {}
    for s in oof_seasons:
        train = d[labelled & (d["season"] >= min_train_season) & (d["season"] < s)]
        test = d[d["season"] == s]
        if train.empty or test.empty:
            continue
        model = fit_market(train, market, **hgb)
        out = test.copy()
        out["p_raw"] = model.predict(test)
        raw[s] = out

    parts = []
    for s in requested:
        if s not in raw:
            continue
        prior = [raw[t] for t in sorted(raw) if first_pool <= t < s]
        if len(prior) >= MIN_CAL_SEASONS:
            hist = pd.concat(prior)
            cal = fit_calibrator(hist["p_raw"], hist["y"])
        else:
            cal = Calibrator()
        out = raw[s]
        out["p"] = cal.predict(out["p_raw"].to_numpy())
        parts.append(out)
    if not parts:
        raise ValueError(f"no {market} rows in any requested season {requested}")
    return pd.concat(parts, ignore_index=True)
