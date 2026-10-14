"""Power-method de-vig used by the +EV game-line board (board.novig_power)."""
from __future__ import annotations

import pytest

from sportsmodel.serving.board import implied_prob, novig, novig_power


@pytest.mark.parametrize("a,b", [(-110, -110), (-170, 150), (-900, 605), (105, -125), (-105, -115), (300, -400)])
def test_two_sides_sum_to_one(a, b):
    assert novig_power(a, b) + novig_power(b, a) == pytest.approx(1.0, abs=1e-9)


def test_equal_prices_give_half():
    assert novig_power(-110, -110) == pytest.approx(0.5, abs=1e-9)
    assert novig_power(150, 150) == pytest.approx(0.5, abs=1e-9)


@pytest.mark.parametrize("fav,dog", [(-900, 605), (-170, 150), (-300, 250), (-150, 125)])
def test_longshot_lower_and_favourite_higher_than_proportional(fav, dog):
    assert novig_power(dog, fav) < novig(dog, fav)
    assert novig_power(fav, dog) > novig(fav, dog)


def test_pinnacle_minus900_plus605_pinned_value():
    # a = implied(-900) = 900/1000 = 0.9 ; b = implied(+605) = 100/705 = 0.141844 ; a + b = 1.041844.
    # Solve a^k + b^k = 1 for k >= 1: k = 1.122889 -> a^k = 0.888422, b^k = 0.111578.
    # (proportional would give b / (a + b) = 0.136147 for the long shot.)
    assert novig_power(605, -900) == pytest.approx(0.111578, abs=1e-6)
    assert novig_power(-900, 605) == pytest.approx(0.888422, abs=1e-6)
    assert novig(605, -900) == pytest.approx(0.136147, abs=1e-6)


@pytest.mark.parametrize("a,b", [(-105, 110), (105, 105), (110, 110), (100, 100)])
def test_no_overround_falls_back_to_proportional(a, b):
    # a + b <= 1 (no vig, or a negative-overround arb pair): k would be <= 1, so we return novig.
    assert implied_prob(a) + implied_prob(b) <= 1 + 1e-12
    assert novig_power(a, b) == novig(a, b)


def test_invalid_price_behaves_like_novig():
    # implied_prob divides by decimal_odds; american 0 -> 100/-0 raises ZeroDivisionError in novig, so here too.
    with pytest.raises(ZeroDivisionError):
        novig(0, -110)
    with pytest.raises(ZeroDivisionError):
        novig_power(0, -110)
