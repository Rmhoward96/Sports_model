"""Tests for the cover-ensemble serving core (sportsmodel.serving.ensemble).

Covers the meta-learner apply (`ensemble_prob`), the agreement gate
(`agreement`), the GBM-point -> cover/over probability conversion
(`gbm_prob`), and the graceful-absent loader (`load_ensemble`)."""
import math

from sportsmodel.serving.ensemble import ensemble_prob, agreement, gbm_prob, load_ensemble


def test_ensemble_prob_is_calibrated_logistic_blend():
    # equal 0.6 inputs, coef summing with intercept 0 -> monotone in inputs
    p = ensemble_prob([0.6,0.6,0.6], coef=[1.0,1.0,1.0], intercept=-1.5)
    assert 0.0 < p < 1.0
    hi = ensemble_prob([0.8,0.8,0.8], coef=[1.0,1.0,1.0], intercept=-1.5)
    assert hi > p                         # higher base probs -> higher ensemble prob


def test_agreement_requires_same_side():
    assert agreement([0.6,0.58,0.61]) is True     # all favor the side
    assert agreement([0.6,0.48,0.61]) is False    # one disagrees


def test_gbm_prob_margin_monotone_in_prediction():
    a = gbm_prob(3.0, sigma=13.2, dist_kind="margin", line=-2.5)
    b = gbm_prob(7.0, sigma=13.2, dist_kind="margin", line=-2.5)
    assert b > a                          # bigger predicted home margin -> more cover prob


def test_load_ensemble_returns_none_when_asset_absent():
    # Task 6 writes assets/nfl/cover_ensemble.json; it does not exist yet, so
    # this must fail gracefully rather than raising.
    assert load_ensemble() is None
