"""The per-record props-ML pipeline step: A -> blend with B -> calibrate.
Pure, no IO.

Shared by the offline ladder / quick gate (``scripts/train_props_ml_b.py``
``apply_pipeline``, which also re-scores against the actual) and live serving
(``sim.nfl.ml_serving``, no actual), so both compute the served pmf the same way.
"""
from __future__ import annotations

import numpy as np

from sportsmodel.model.props_ml.blend import (
    BINARY_MARKET,
    blend_binary,
    blend_pmf,
    prob_at_least_one,
)
from sportsmodel.model.props_ml.pit_calibration import apply_pit_map, apply_platt


def blend_calibrate(pmf_a, pmf_b, w: float, market: str, cal=None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """``(pre, post)``: the pre-calibration pmf and the final one.

    ``pre``: ``anytime_td`` -> ``[1-p, p]`` with p the logit blend (weight
    ``w`` on A) of P(>=1) = ``1 - pmf[0]`` of A and B; other markets -> the
    linear pool ``blend_pmf(A, B, w)``. ``pmf_b`` None keeps A (``w`` unused;
    ``anytime_td`` still leaves as ``[1-p, p]`` of A).
    ``post``: ``cal`` None -> ``pre``; else Platt ``(a, b)`` for anytime_td,
    PIT knots otherwise.
    """
    pa = np.asarray(pmf_a, dtype=float)
    if market == BINARY_MARKET:
        p = (blend_binary(prob_at_least_one(pa), prob_at_least_one(pmf_b), w)
             if pmf_b is not None else prob_at_least_one(pa))
        pre = np.array([1.0 - p, p])
    else:
        pre = blend_pmf(pa, pmf_b, w) if pmf_b is not None else pa
    if cal is None:
        return pre, pre
    if market == BINARY_MARKET:
        q = apply_platt(pre[1], cal)
        return pre, np.array([1.0 - q, q])
    return pre, apply_pit_map(pre, cal)
