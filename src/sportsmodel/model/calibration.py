"""Probability calibration (Platt scaling) for win prob and prop P(over).

Backtests show small, systematic biases (e.g. Hits over-predicts ~5pts). A Platt map
`calibrated = sigmoid(a*logit(raw) + b)`, fit per target from historical backtest data,
corrects them. Params live in assets/calibration.json; production applies them.
"""
from __future__ import annotations

import json
from math import exp, log
from pathlib import Path

from .. import config

_EPS = 1e-6
_CALIB_PATH = config.PROJECT_ROOT / "assets" / "calibration.json"
_cache: dict | None = None


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return log(p / (1 - p))


def apply(raw: float, params) -> float:
    """calibrated = sigmoid(a*logit(raw) + b). params = (a, b) or None (identity)."""
    if params is None:
        return raw
    a, b = params
    return 1.0 / (1.0 + exp(-(a * _logit(raw) + b)))


def fit(probs, ys) -> tuple[float, float]:
    """Fit (a, b) minimizing log-loss of sigmoid(a*logit(p)+b) vs binary ys."""
    import numpy as np
    from scipy.optimize import minimize

    lg = np.array([_logit(p) for p in probs])
    y = np.asarray(ys, dtype=float)

    def nll(theta):
        a, b = theta
        p = 1.0 / (1.0 + np.exp(-(a * lg + b)))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize(nll, [1.0, 0.0], method="Nelder-Mead")
    return float(res.x[0]), float(res.x[1])


def load() -> dict:
    """{target: [a, b]} from assets/calibration.json (cached); {} if absent."""
    global _cache
    if _cache is None:
        _cache = json.loads(_CALIB_PATH.read_text()) if _CALIB_PATH.exists() else {}
    return _cache


def calibrate(target: str, raw: float) -> float:
    """Apply the stored calibration for `target` (identity if none)."""
    return apply(raw, load().get(target))
