import pytest

from sportsmodel.serving import board as b


def test_decimal_and_implied():
    assert b.decimal_odds(100) == pytest.approx(2.0)
    assert b.decimal_odds(-110) == pytest.approx(1.9090909, rel=1e-4)
    assert b.implied_prob(100) == pytest.approx(0.5)


def test_best_price_picks_highest_decimal():
    # +120 (dec 2.2) beats +110 (2.1) and -105 (1.952) for the bettor
    assert b.best_price([("DK", 110), ("FD", 120), ("MGM", -105)]) == ("FD", 120)
    # among negatives, -105 (1.952) beats -120 (1.833)
    assert b.best_price([("DK", -120), ("FD", -105)]) == ("FD", -105)
    assert b.best_price([]) is None


def test_novig_removes_hold():
    assert b.novig(-110, -110) == pytest.approx(0.5)
    assert b.novig(-200, 170) > 0.5


def test_ev_sign():
    assert b.ev(0.6, 100) == pytest.approx(0.2)
    assert b.ev(0.4, 100) == pytest.approx(-0.2)
