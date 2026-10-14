"""The shrink formula, the clip and the market probability: exact values from the spec. No IO."""
import math

import pytest

from sportsmodel.serving.board import novig
from sportsmodel.shrink import formula


def test_the_clip_bounds_are_the_specs():
    assert (formula.CLIP_LO, formula.CLIP_HI) == (0.01, 0.99)


def test_k_zero_is_the_market_and_k_one_is_the_model():
    assert formula.shrink_prob(0.70, 0.50, 0.0) == pytest.approx(0.50)
    assert formula.shrink_prob(0.70, 0.50, 1.0) == pytest.approx(0.70)
    assert formula.shrink_prob(0.70, 0.50, 0.25) == pytest.approx(0.55)          # m + k (p - m)
    assert formula.shrink_prob(0.30, 0.50, 0.25) == pytest.approx(0.45)          # works below the market too


def test_the_result_is_clipped_to_one_and_ninety_nine_percent():
    assert formula.shrink_prob(1.0, 0.5, 1.0) == 0.99
    assert formula.shrink_prob(0.0, 0.5, 1.0) == 0.01
    assert formula.shrink_prob(0.5, 0.995, 0.0) == 0.99          # a market the book prices at 99.5% is clipped too
    assert formula.shrink_prob(0.5, 0.004, 0.0) == 0.01


@pytest.mark.parametrize("price", [100, -100, 150, -110, 100.0, -250, 10000, "-110"])
def test_valid_american_prices(price):
    assert formula.valid_american(price) is True


@pytest.mark.parametrize("price", [0, 99, -99, 50, -50, None, "x", "", float("nan"), float("inf"), -float("inf"), True])
def test_invalid_american_prices(price):
    assert formula.valid_american(price) is False


def test_market_prob_over_is_the_boards_novig_of_best_over_and_best_under():
    assert formula.market_prob_over(-110, -110) == pytest.approx(0.5)
    assert formula.market_prob_over(150, -170) == pytest.approx(novig(150, -170))
    assert formula.market_prob_over(-130, 110) == pytest.approx(novig(-130, 110))
    assert 0 < formula.market_prob_over(-130, 110) < 1


def test_market_prob_over_is_none_when_either_price_is_unusable():
    assert formula.market_prob_over(None, -110) is None
    assert formula.market_prob_over(-110, 0) is None
    assert formula.market_prob_over(-99, -110) is None
    assert formula.market_prob_over(float("nan"), -110) is None
    assert not math.isnan(formula.market_prob_over(100, 100))
