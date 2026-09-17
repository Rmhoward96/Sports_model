"""Bias fitting/eval for the game-line recalibration (model.recalibration)."""
import pytest

from sportsmodel.model.recalibration import bias_eval, fit_bias


def test_fit_bias_empty_is_zero():
    assert fit_bias([]) == 0.0


def test_fit_bias_shrinks_mean_residual():
    # mean residual +8, default shrink 0.5 -> +4, but capped at +3.0
    assert fit_bias([8, 8, 8, 8], shrink_factor=0.5, cap=3.0) == 3.0
    # within the cap: 0.5 * 4 = 2.0
    assert fit_bias([4, 4], shrink_factor=0.5, cap=3.0) == 2.0


def test_fit_bias_preserves_sign_and_caps_negative():
    assert fit_bias([-10, -10], shrink_factor=0.5, cap=3.0) == -3.0
    assert fit_bias([-2, -2], shrink_factor=0.5, cap=3.0) == -1.0


def test_fit_bias_no_shrink_no_cap():
    assert fit_bias([4, 6], shrink_factor=1.0, cap=100.0) == 5.0


def test_bias_eval_zero_bias_matches_raw_error():
    preds = [{"pred_margin": 3, "pred_total": 50, "actual_margin": 0, "actual_total": 45}]
    r = bias_eval(preds, 0.0, 0.0)
    assert r["margin_mae"] == 3.0        # |3 - 0|
    assert r["total_mae"] == 5.0         # |50 - 45|
    assert r["ou_acc"] == 0.0            # actual_total 45 > pred 50? no


def test_bias_eval_total_bias_improves_biased_total():
    # model runs +5 high on totals; correcting by +5 zeroes the error
    preds = [
        {"pred_margin": 0, "pred_total": 55, "actual_margin": 0, "actual_total": 50},
        {"pred_margin": 0, "pred_total": 45, "actual_margin": 0, "actual_total": 40},
    ]
    assert bias_eval(preds, 0.0, 0.0)["total_mae"] == 5.0
    assert bias_eval(preds, 0.0, 5.0)["total_mae"] == 0.0
    # ou_acc: with no correction both actual < pred -> 0.0 (biased high); after
    # subtracting 5, pred==actual so actual_total > corrected is false too, but
    # the MAE improvement is the signal here.


def test_bias_eval_cover_uses_corrected_margin_sign():
    # model says home by 2, actual home lost by 1. A +3 margin bias correction
    # flips the predicted side to the away/underdog, matching the actual outcome.
    preds = [{"pred_margin": 2, "pred_total": 40, "actual_margin": -1, "actual_total": 40}]
    assert bias_eval(preds, 0.0, 0.0)["cover_acc"] == 0.0   # pred home, actual away
    assert bias_eval(preds, 3.0, 0.0)["cover_acc"] == 1.0   # corrected -1 -> away, correct
