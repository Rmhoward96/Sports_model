"""PIT and Platt calibration maps (props ML, Props-2). Pure functions, no IO.

PIT map (count/yards markets): if forecasts are calibrated, PITs are
U(0,1). ``fit_pit_map`` estimates G, the empirical CDF of observed PITs, as
monotone knots ``(u, G(u))`` on an even grid; ``apply_pit_map`` recalibrates
a pmf's CDF as ``F' = G(F)`` (linear interpolation between knots). PITs piled
near 0 and 1 (overconfident forecasts) give a G that is steep at the edges,
which moves mass into the tails -- the pmf widens.

Platt (binary ``anytime_td``): ``p' = sigmoid(a*logit(p) + b)``, fitted by
logistic regression on ``logit(p)``; identity is ``(1, 0)``.

Serialization: knots are a ``(2, K)`` float array (``.tolist()`` for JSON;
``apply_pit_map`` accepts lists), Platt params a tuple of two Python floats.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from sportsmodel.model.props_ml.blend import _logit, _scalar_or_array, _sigmoid

N_KNOTS = 201
IDENTITY_KNOTS = np.array([[0.0, 1.0], [0.0, 1.0]])
IDENTITY_PLATT = (1.0, 0.0)


def fit_pit_map(pits) -> np.ndarray:
    """Knots ``(2, K)`` of the empirical CDF of the finite PITs (clipped to
    [0, 1]) on ``K = 201`` evenly spaced u in [0, 1], anchored at (0, 0) and
    (1, 1) and made monotone. No finite PITs -> identity knots."""
    p = np.asarray(pits, dtype=float).ravel()
    p = np.sort(np.clip(p[np.isfinite(p)], 0.0, 1.0))
    if p.size == 0:
        return IDENTITY_KNOTS.copy()
    u = np.linspace(0.0, 1.0, N_KNOTS)
    g = np.searchsorted(p, u, side="right") / p.size
    g[0], g[-1] = 0.0, 1.0
    g = np.clip(np.maximum.accumulate(g), 0.0, 1.0)
    return np.vstack([u, g])


def apply_pit_map(pmf, knots) -> np.ndarray:
    """Recalibrate a pmf: ``F = cumsum(pmf)``, ``F' = interp(F, knots)``,
    ``pmf' = diff([0, F'])``, clipped at 0 and renormalized."""
    p = np.asarray(pmf, dtype=float)
    u, g = np.asarray(knots, dtype=float)
    f = np.clip(np.cumsum(p), 0.0, 1.0)
    f_new = np.interp(f, u, g)
    out = np.clip(np.diff(np.concatenate([[0.0], f_new])), 0.0, None)
    return out / out.sum()


def fit_platt(p, y) -> tuple[float, float]:
    """Unpenalized logistic regression of ``y`` on ``logit(p)`` (p clipped to
    [1e-4, 1-1e-4]); returns ``(a, b)``. A single observed class cannot
    identify the map -> identity ``(1.0, 0.0)``."""
    x = _logit(np.asarray(p, dtype=float).ravel()).reshape(-1, 1)
    y = np.asarray(y).ravel().astype(int)
    if np.unique(y).size < 2:
        return IDENTITY_PLATT
    lr = LogisticRegression(C=np.inf, max_iter=1000).fit(x, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def apply_platt(p, ab):
    """``sigmoid(a*logit(p) + b)``; scalars in -> float out."""
    a, b = ab
    return _scalar_or_array(_sigmoid(float(a) * _logit(p) + float(b)))
