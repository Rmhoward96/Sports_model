"""B per-market distribution models (props ML, Props-2).

Each prop market gets a model that predicts a FULL pmf per player-game
directly from the player feature table (later blended with the sim's A pmf,
calibrated, and served):

* ``quantile`` markets (``rec_yds``, ``rush_yds``, ``pass_yds``) -- one
  quantile-loss HGB per tau in ``TAUS``; the 19 predicted quantiles become a
  pmf on ``0..kmax`` via ``quantiles_to_pmf``. Yards labels are clipped at 0
  before fitting (the sim's support starts at 0).
* ``count`` markets (``receptions``, ``rush_att``, ``pass_tds``) -- a
  Poisson-loss HGB for the mean plus a negative-binomial size per usage tier
  (terciles of the predicted training mean; cut points stored so prediction
  assigns tiers the same way; size by method of moments conditional on the
  predicted mean, see ``_tier_dispersion``); pmf via ``nb_pmf``.
* ``binary`` (``anytime_td``) -- an HGB classifier; pmf ``[P(0), P(1)]``.

Leakage contract (same as ``sim.nfl.learned``): ``fit_market`` itself keeps
only rows with ``(season, week) < upto``, the label present and ``is_stub``
not set (B models E[stat | played]), so callers cannot leak by passing the
full table. Walk-forward only; no random splits, no early stopping. A column
that is 100% NaN in the training slice is dropped for that fit and the used
columns are recorded on the ``MarketModel``. Model hyper-parameters come from
``learned._hgb`` so A and B share one definition.

scipy is deliberately not imported: the negative binomial uses ``math.lgamma``.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import lgamma, log
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from sportsmodel.sim.nfl.learned import STUB_COL, _before, _hgb

TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)

# market -> label column in the player feature table
LABELS: dict[str, str] = {
    "rec_yds": "y_rec_yds",
    "rush_yds": "y_rush_yds",
    "pass_yds": "y_pass_yds",
    "receptions": "y_receptions",
    "rush_att": "y_carries",
    "pass_tds": "y_pass_tds",
    "anytime_td": "y_anytime_td",
}

Kind = Literal["quantile", "count", "binary"]
KINDS: dict[str, Kind] = {
    "rec_yds": "quantile", "rush_yds": "quantile", "pass_yds": "quantile",
    "receptions": "count", "rush_att": "count", "pass_tds": "count",
    "anytime_td": "binary",
}

# NB size above this is treated as Poisson.
POISSON_R = 1e6


# ---- pmf builders --------------------------------------------------------------

def quantiles_to_pmf(qvals: np.ndarray, kmax: int) -> np.ndarray:
    """pmf on ``0..kmax`` from the 19 quantiles at ``TAUS``.

    Quantiles are sorted (monotone rearrangement) and clipped to
    ``[0, kmax]``. The CDF is piecewise linear through the knots
    ``(-0.5, 0)``, ``(q_i, tau_i)``, ``(kmax + 0.5, 1)``; knots sharing an x
    keep both the smallest and the largest tau there, i.e. F steps at that x
    (right-continuous) and the tied taus' mass is a point mass on it.
    ``pmf[k] = F(k + 0.5) - F(k - 0.5)``, renormalized.
    """
    q = np.asarray(qvals, dtype=float)
    if q.shape != TAUS.shape:
        raise ValueError(f"expected {len(TAUS)} quantiles, got shape {q.shape}")
    q = np.clip(np.sort(q), 0.0, float(kmax))
    x = np.concatenate([[-0.5], q, [kmax + 0.5]])
    y = np.concatenate([[0.0], TAUS, [1.0]])
    # x is sorted and y increasing: keep the FIRST (smallest tau) and LAST
    # (largest tau) knot of each equal-x run. np.interp takes the later knot
    # at a duplicated x, so F is right-continuous at the step.
    first = np.insert(x[1:] != x[:-1], 0, True)
    last = np.append(x[1:] != x[:-1], True)
    keep = first | last
    F = np.interp(np.arange(kmax + 2) - 0.5, x[keep], y[keep])
    pmf = np.clip(np.diff(F), 0.0, None)
    return pmf / pmf.sum()


def nb_pmf(mu: float, r: float | None, kmax: int) -> np.ndarray:
    """Negative-binomial pmf on ``0..kmax`` with mean ``mu`` and size ``r``.

    ``r is None`` or ``r > 1e6`` gives the Poisson limit. Computed in log
    space via ``math.lgamma``; the tail mass beyond ``kmax`` is folded into
    ``pmf[kmax]``. ``mu <= 0`` is a point mass at 0; NaN ``mu`` raises.
    """
    if mu != mu:
        raise ValueError("nb_pmf: mu is NaN")
    pmf = np.zeros(kmax + 1)
    if not mu > 0:
        pmf[0] = 1.0
        return pmf
    lfact = [lgamma(k + 1) for k in range(kmax + 1)]
    if r is None or r > POISSON_R:
        lm = log(mu)
        logp = [-mu + k * lm - lfact[k] for k in range(kmax + 1)]
    else:
        if not r > 0:
            raise ValueError(f"NB size must be > 0, got {r}")
        a, b, lr = r * log(r / (r + mu)), log(mu / (r + mu)), lgamma(r)
        logp = [lgamma(k + r) - lr - lfact[k] + a + k * b for k in range(kmax + 1)]
    pmf = np.exp(np.asarray(logp))
    pmf[kmax] = max(1.0 - pmf[:kmax].sum(), 0.0)
    return pmf / pmf.sum()


def bernoulli_pmf(p: float) -> np.ndarray:
    """``[P(0), P(1)]``."""
    return np.array([1.0 - p, p], dtype=float)


# ---- models ----------------------------------------------------------------------

@dataclass
class MarketModel:
    """Fitted B model for one market.

    ``models``: quantile -> one regressor per tau in ``TAUS`` (same order);
    count -> ``[poisson regressor]``; binary -> ``[classifier]``. ``cols`` are
    the columns actually fitted on. ``dispersion`` (count only):
    ``{"cuts": [c1, c2], "r": [r_low, r_mid, r_high]}`` with ``None`` = Poisson.
    """

    market: str
    kind: Kind
    cols: list[str]
    models: list
    dispersion: dict | None


def _classifier(max_iter: int) -> HistGradientBoostingClassifier:
    """Classifier with the regressors' settings (``learned._hgb``)."""
    reg = _hgb("poisson", max_iter, None).get_params()
    allowed = HistGradientBoostingClassifier().get_params().keys()
    return HistGradientBoostingClassifier(
        **{k: v for k, v in reg.items() if k in allowed and k != "loss"})


def _training_slice(df: pd.DataFrame, label: str, upto: tuple[int, int]) -> pd.DataFrame:
    m = _before(df, upto) & df[label].notna()
    if STUB_COL in df.columns:
        m &= ~df[STUB_COL].fillna(False).astype(bool)
    return df[m]


def _tiers(cuts, mu: np.ndarray) -> np.ndarray:
    """Usage tier 0/1/2: ``mu <= c1`` -> 0, ``c1 < mu <= c2`` -> 1, else 2
    (a mass point at a cut stays in the lower tier)."""
    return np.searchsorted(np.asarray(cuts, dtype=float), mu, side="left")


def _tier_dispersion(mu: np.ndarray, y: np.ndarray) -> dict:
    """NB size per usage tier (terciles of the predicted mean ``mu``).

    Method of moments conditional on ``mu`` (NB2: ``Var(y|mu) = mu + mu^2/r``):
    ``r = sum(mu^2) / sum((y - mu)^2 - mu)`` over the tier, ``None``
    (Poisson) when the denominator is <= 0. With ``mu`` constant in a tier (and
    equal to the tier's mean of ``y``) this is exactly the unconditional
    ``mean(mu)^2 / (var(y) - mean(mu))``; unlike that form it does not count
    the spread of ``mu`` within a tier as overdispersion (which, on the real
    table, gave e.g. rush_att top-tier r ~= 0.9 and pass_tds r ~= 0.1).
    """
    cuts = np.quantile(mu, [1 / 3, 2 / 3])
    tier = _tiers(cuts, mu)
    rs: list[float | None] = []
    for t in range(3):
        sel = tier == t
        num = float((mu[sel] ** 2).sum())
        den = float(((y[sel] - mu[sel]) ** 2 - mu[sel]).sum())
        rs.append(num / den if num > 0 and den > 0 else None)
    return {"cuts": [float(c) for c in cuts], "r": rs}


def fit_market(df: pd.DataFrame, market: str, cols: list[str], *,
               upto: tuple[int, int], test_season: int, decay: float,
               max_iter: int) -> MarketModel:
    """Fit the B model for ``market`` on played rows before ``upto`` with the
    label present; sample weight ``decay ** (test_season - season)``.
    ``decay == 1.0`` fits UNWEIGHTED (``sample_weight=None``): scikit-learn's
    weighted binning path costs ~12 s per fit on the real table even when all
    weights are equal.

    Raises ``ValueError`` for an unknown market, an empty training slice, or
    no usable (non-all-NaN) feature column.
    """
    if market not in LABELS:
        raise ValueError(f"unknown market {market!r}")
    label, kind = LABELS[market], KINDS[market]
    tr = _training_slice(df, label, upto)
    if tr.empty:
        raise ValueError(f"fit_market({market}): no training rows before {upto}")
    used = [c for c in cols if tr[c].notna().any()]
    if not used:
        raise ValueError(f"fit_market({market}): every feature column is all-NaN")
    X = tr[used]
    y = tr[label].to_numpy(dtype=float)
    w = (None if decay == 1.0
         else decay ** (test_season - tr["season"].to_numpy(dtype=float)))

    if kind == "quantile":
        y = np.clip(y, 0.0, None)
        models = [_hgb("quantile", max_iter, None).set_params(quantile=float(t))
                  .fit(X, y, sample_weight=w) for t in TAUS]
        return MarketModel(market, kind, used, models, None)
    if kind == "count":
        est = _hgb("poisson", max_iter, None).fit(X, y, sample_weight=w)
        disp = _tier_dispersion(est.predict(X), y)
        return MarketModel(market, kind, used, [est], disp)
    yb = (y > 0).astype(int)
    if len(np.unique(yb)) < 2:
        raise ValueError(f"fit_market({market}): training labels have one class")
    clf = _classifier(max_iter).fit(X, yb, sample_weight=w)
    return MarketModel(market, kind, used, [clf], None)


def predict_pmfs(model: MarketModel, rows: pd.DataFrame, kmax: int) -> list[np.ndarray]:
    """One pmf per row: length ``kmax + 1`` (binary: 2, ``kmax`` ignored)."""
    if len(rows) == 0:
        return []
    X = rows[model.cols]
    if model.kind == "quantile":
        Q = np.column_stack([m.predict(X) for m in model.models])
        return [quantiles_to_pmf(q, kmax) for q in Q]
    if model.kind == "count":
        mu = np.clip(model.models[0].predict(X), 0.0, None)
        tiers = _tiers(model.dispersion["cuts"], mu)
        rs = model.dispersion["r"]
        return [nb_pmf(float(m), rs[t], kmax) for m, t in zip(mu, tiers)]
    clf = model.models[0]
    p1 = clf.predict_proba(X)[:, list(clf.classes_).index(1)]
    return [bernoulli_pmf(float(p)) for p in p1]
