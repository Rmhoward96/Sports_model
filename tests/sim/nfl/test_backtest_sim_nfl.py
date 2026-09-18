"""Tests for the PURE metric helpers in backtest_sim_nfl.py (the NFL sim
walk-forward ship gate). run_backtest()/report()/main() are heavy IO and NOT
unit-tested here -- see the module docstring in scripts/backtest_sim_nfl.py."""
import importlib.util
import math
import pathlib

_p = pathlib.Path(__file__).resolve().parents[3] / "scripts" / "backtest_sim_nfl.py"
_spec = importlib.util.spec_from_file_location("backtest_sim_nfl", _p)
bsn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bsn)


# =============================================================================
# brier
# =============================================================================

def test_brier_perfect_predictions_scores_zero():
    assert bsn.brier([1.0, 0.0, 1.0], [1.0, 0.0, 1.0]) == 0.0


def test_brier_worst_predictions_scores_one():
    assert bsn.brier([1.0, 0.0], [0.0, 1.0]) == 1.0


def test_brier_coin_flip_on_certain_outcomes_scores_quarter():
    assert bsn.brier([0.5, 0.5], [1.0, 0.0]) == 0.25


def test_brier_matches_known_hand_computation():
    # (0.7-1)^2=0.09, (0.4-0)^2=0.16, (0.6-1)^2=0.16 -> mean = 0.41/3
    result = bsn.brier([0.7, 0.4, 0.6], [1.0, 0.0, 1.0])
    assert math.isclose(result, (0.09 + 0.16 + 0.16) / 3, abs_tol=1e-9)


def test_brier_empty_input_is_nan():
    assert math.isnan(bsn.brier([], []))


# =============================================================================
# mae
# =============================================================================

def test_mae_identical_sequences_is_zero():
    assert bsn.mae([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_mae_matches_known_hand_computation():
    # |3-1| + |5-8| + |0-1| = 2 + 3 + 1 = 6 -> mean 2.0
    assert bsn.mae([3.0, 5.0, 0.0], [1.0, 8.0, 1.0]) == 2.0


def test_mae_is_symmetric_in_sign():
    assert bsn.mae([10.0], [7.0]) == bsn.mae([7.0], [10.0])


def test_mae_empty_input_is_nan():
    assert math.isnan(bsn.mae([], []))


# =============================================================================
# pearson_corr
# =============================================================================

def test_pearson_corr_perfect_positive_correlation():
    result = bsn.pearson_corr([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert math.isclose(result, 1.0, abs_tol=1e-9)


def test_pearson_corr_perfect_negative_correlation():
    result = bsn.pearson_corr([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
    assert math.isclose(result, -1.0, abs_tol=1e-9)


def test_pearson_corr_no_correlation_is_near_zero():
    # Symmetric zigzag: deviation products cancel exactly -> corr == 0.
    result = bsn.pearson_corr([1.0, 2.0, 3.0, 4.0], [3.0, 1.0, 4.0, 2.0])
    assert math.isclose(result, 0.0, abs_tol=1e-9)


def test_pearson_corr_zero_variance_series_is_nan():
    assert math.isnan(bsn.pearson_corr([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_pearson_corr_single_point_is_nan():
    assert math.isnan(bsn.pearson_corr([1.0], [2.0]))


# =============================================================================
# quantile_from_pmf
# =============================================================================

def _uniform_pmf(n: int) -> dict:
    """A pmf uniform over integers 0..n-1, as {"kind","pmf","mean"}."""
    p = 1.0 / n
    return {"kind": "pmf", "pmf": [p] * n, "mean": (n - 1) / 2.0}


def test_quantile_from_pmf_median_of_uniform_ten_bin_pmf():
    # Uniform over 0..9: cumsum reaches 0.5 exactly at k=4 (5*0.1=0.5).
    dist = _uniform_pmf(10)
    assert bsn.quantile_from_pmf(dist, 0.50) == 4.0


def test_quantile_from_pmf_p90_of_uniform_ten_bin_pmf():
    dist = _uniform_pmf(10)
    assert bsn.quantile_from_pmf(dist, 0.90) == 8.0


def test_quantile_from_pmf_point_mass_returns_that_point():
    dist = {"kind": "pmf", "pmf": [0.0, 0.0, 1.0, 0.0], "mean": 2.0}
    assert bsn.quantile_from_pmf(dist, 0.50) == 2.0
    assert bsn.quantile_from_pmf(dist, 0.99) == 2.0


def test_quantile_from_pmf_q_beyond_total_mass_falls_back_to_top_bin():
    dist = {"kind": "pmf", "pmf": [0.5, 0.3], "mean": 0.3}
    assert bsn.quantile_from_pmf(dist, 0.999) == 1.0


# =============================================================================
# share_below
# =============================================================================

def test_share_below_known_pmf_p50_matches_target_quantile():
    """Build actuals sampled exactly at the deciles of a known uniform pmf's
    quantile function; share_below at p50 should land at 0.5 for actuals
    drawn symmetrically around the median."""
    dist = _uniform_pmf(10)
    p50 = bsn.quantile_from_pmf(dist, 0.50)  # 4.0
    # 10 actuals: 5 at/below the p50 value, 5 above -> coverage should be 0.5
    actuals = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    quantile_values = [p50] * len(actuals)
    result = bsn.share_below(quantile_values, actuals)
    assert math.isclose(result, 0.5, abs_tol=1e-9)


def test_share_below_all_actuals_under_quantile_is_one():
    result = bsn.share_below([10.0, 10.0, 10.0], [1.0, 2.0, 3.0])
    assert result == 1.0


def test_share_below_all_actuals_over_quantile_is_zero():
    result = bsn.share_below([1.0, 1.0, 1.0], [10.0, 20.0, 30.0])
    assert result == 0.0


def test_share_below_boundary_equal_values_counts_as_covered():
    result = bsn.share_below([5.0], [5.0])
    assert result == 1.0


def test_share_below_empty_input_is_nan():
    assert math.isnan(bsn.share_below([], []))
