"""Cover-ensemble serving core: meta-learner apply + agreement gate.

Task 3 of the NFL cover/total stacked-ensemble (see
docs/superpowers/specs/2026-09-21-nfl-cover-ensemble-design.md). Consumes
base-model probabilities per market (e.g. gameline sim, GBM point estimate,
market-implied) and produces a single calibrated ensemble probability plus an
agreement gate that flags when the base models disagree on the side.

Pure module: the only IO is `load_ensemble` reading the committed
`assets/nfl/cover_ensemble.json` artifact (written by Task 6's fit script),
and that load is graceful -- it returns None when the artifact is absent
(e.g. before Task 6 has run) rather than raising, so callers can fall back to
a single base model.

Reuses `model.distributions` for all distribution math (`prob_cover`,
`prob_over_dist`, `normal_to_margin_pmf`, `normal_to_pmf`) rather than
duplicating it -- `gbm_prob` is a thin adapter from a GBM point prediction
(mean margin/total) + assumed sigma onto those existing helpers.
"""
from __future__ import annotations

import json
import pathlib
from collections.abc import Sequence
from math import exp, log

from ..model.distributions import (
    normal_to_margin_pmf,
    normal_to_pmf,
    prob_cover,
    prob_over_dist,
)

_ASSETS = pathlib.Path(__file__).resolve().parents[3] / "assets" / "nfl"

# Matches nfl.gameline.GameLineConfig defaults: margin support is -75..+75,
# total support is 0..120. Callers modeling a market whose committed config
# uses different values should pass offset/xmax explicitly.
_DEFAULT_MARGIN_OFFSET = 75
_DEFAULT_TOTAL_XMAX = 120

_EPS = 1e-6


def _clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def _logit(p: float) -> float:
    p = _clip(p)
    return log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = exp(-x)
        return 1.0 / (1.0 + z)
    z = exp(x)
    return z / (1.0 + z)


def ensemble_prob(base_probs: Sequence[float], coef: Sequence[float], intercept: float) -> float:
    """Logistic-regression meta-learner over base probabilities' log-odds.

    `p = sigmoid(intercept + sum(coef_i * logit(base_probs_i)))`. Each base
    probability is clipped to [1e-6, 1-1e-6] before taking its logit so a
    base model's 0/1 certainty can't blow up the stack.
    """
    z = intercept + sum(c * _logit(p) for c, p in zip(coef, base_probs))
    return _sigmoid(z)


def agreement(base_probs: Sequence[float], ref: float = 0.5) -> bool:
    """True iff every base probability sits on the same side of `ref`.

    Used as a confidence gate: base models split across the reference point
    disagree on which side to take, independent of what the blended
    ensemble probability says.
    """
    sides = {p > ref for p in base_probs}
    return len(sides) <= 1


def gbm_prob(pred: float, sigma: float, dist_kind: str, line: float,
             offset: int | None = None, xmax: int | None = None) -> float:
    """Convert a GBM point prediction (mean) + sigma into a cover/over prob.

    `dist_kind == "margin"`: discretizes Normal(pred, sigma) onto the margin
    support via `normal_to_margin_pmf` and returns `prob_cover(..., line)`
    where `line` is the home spread (e.g. -2.5).

    `dist_kind == "total"`: discretizes Normal(pred, sigma) onto 0..xmax via
    `normal_to_pmf` and returns `prob_over_dist(..., line)` where `line` is
    the total line.

    `offset`/`xmax` default to the values `nfl.gameline.GameLineConfig` uses
    (75 / 120) so callers scoring the same market get consistent support
    without needing to know those constants.
    """
    if dist_kind == "margin":
        margin_dist = normal_to_margin_pmf(pred, sigma, offset if offset is not None else _DEFAULT_MARGIN_OFFSET)
        return prob_cover(margin_dist, line)
    if dist_kind == "total":
        pmf = normal_to_pmf(pred, sigma, xmax if xmax is not None else _DEFAULT_TOTAL_XMAX)
        return prob_over_dist({"kind": "pmf", "pmf": pmf}, line)
    raise ValueError(f"gbm_prob: unknown dist_kind {dist_kind!r} (expected 'margin' or 'total')")


def load_ensemble() -> dict | None:
    """Read the committed cover-ensemble fit (coef/intercept/sigma per market).

    Returns None if `assets/nfl/cover_ensemble.json` is absent -- Task 6
    writes this artifact from the fit script, so it will not exist until
    that task lands. Callers must handle None (fall back to a single base
    model) rather than treating its absence as an error.
    """
    path = _ASSETS / "cover_ensemble.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
