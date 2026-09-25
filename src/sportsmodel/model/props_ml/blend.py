"""A/B pmf blending (props ML, Props-2). Pure functions, no IO.

A is the learned sim's per-player pmf, B the direct per-market model's
(``dist_models``). Count/yards markets use a linear pool of the two pmfs;
the binary ``anytime_td`` market uses a logit blend of P(1). ``choose_weight``
picks the pool weight on A from a fixed grid by walk-forward score (mean RPS;
for a 2-bin pmf RPS equals the Brier score), breaking ties toward larger w
(prefer the gated A).
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd

from sportsmodel.model.props_eval import rps_pmf

BINARY_MARKET = "anytime_td"
P_CLIP = 1e-4
DEFAULT_GRID = np.round(np.arange(0, 1.01, 0.1), 1)
_TIE_TOL = 1e-12


def _check_w(w: float) -> float:
    w = float(w)
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1], got {w}")
    return w


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), P_CLIP, 1.0 - P_CLIP)
    return np.log(p) - np.log1p(-p)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def _scalar_or_array(x):
    x = np.asarray(x, dtype=float)
    return float(x) if x.ndim == 0 else x


def blend_pmf(pmf_a, pmf_b, w: float) -> np.ndarray:
    """Linear pool ``w*a + (1-w)*b``, renormalized. Pmfs of unequal length are
    zero-padded to the longer support (both start at 0)."""
    w = _check_w(w)
    a = np.asarray(pmf_a, dtype=float)
    b = np.asarray(pmf_b, dtype=float)
    n = max(len(a), len(b))
    a = np.pad(a, (0, n - len(a)))
    b = np.pad(b, (0, n - len(b)))
    out = w * a + (1.0 - w) * b
    return out / out.sum()


def blend_binary(p_a, p_b, w: float):
    """Logit blend ``sigmoid(w*logit(p_a) + (1-w)*logit(p_b))`` with both
    probabilities clipped to [1e-4, 1-1e-4]. Scalars in -> float out."""
    w = _check_w(w)
    return _scalar_or_array(_sigmoid(w * _logit(p_a) + (1.0 - w) * _logit(p_b)))


def _blend_row(pmf_a, pmf_b, w: float, market: str) -> np.ndarray:
    if market == BINARY_MARKET:
        p1 = blend_binary(np.asarray(pmf_a, dtype=float)[1], np.asarray(pmf_b, dtype=float)[1], w)
        return np.array([1.0 - p1, p1])
    return blend_pmf(pmf_a, pmf_b, w)


def choose_weight(rows: Iterable[Mapping] | pd.DataFrame, market: str,
                  grid=DEFAULT_GRID) -> float:
    """Grid weight on A minimising the mean score of the blend over ``rows``
    (dicts or a DataFrame with ``pmf_a``, ``pmf_b``, ``actual``).

    ``anytime_td``: 2-bin pmfs ``[P(0), P(1)]``, logit blend of P(1), scored by
    Brier (= RPS of the 2-bin pmf). Other markets: linear pool, mean RPS.
    Scores within 1e-12 of the minimum count as ties; ties go to the larger w.
    """
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict("records")
    rows = list(rows)
    if not rows:
        raise ValueError("choose_weight needs at least one row")
    grid = np.asarray(grid, dtype=float)
    scores = np.array([
        np.mean([rps_pmf(_blend_row(r["pmf_a"], r["pmf_b"], g, market), r["actual"])
                 for r in rows])
        for g in grid
    ])
    best = scores.min()
    return float(grid[scores <= best + _TIE_TOL].max())
