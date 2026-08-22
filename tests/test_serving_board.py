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


def test_moneyline_row_favors_and_prices_best_book():
    row = b.moneyline_row(0.60, [("DK", -130), ("FD", -120)], [("MGM", 110)], "Home", "Away")
    assert row["side"] == "home" and row["pick_label"] == "Home ML"
    assert row["book"] == "FD" and row["odds"] == -120   # best (highest decimal) home price
    assert row["model_prob"] == pytest.approx(0.60)
    assert row["is_pick"] is True   # 0.60 * dec(-120) - 1 > 0


def test_moneyline_row_favors_away_when_model_leans_away():
    row = b.moneyline_row(0.40, [("DK", -130)], [("MGM", 115)], "Home", "Away")
    assert row["side"] == "away" and row["pick_label"] == "Away ML"
    assert row["model_prob"] == pytest.approx(0.60)


def test_total_row_picks_ev_side_not_mean():
    # right-skewed total: mass at 8 and 12 -> P(over 8.5) = 0.4 (<0.5) -> under side
    pmf = [0.0] * 21
    pmf[8], pmf[12] = 0.6, 0.4
    dist = {"kind": "pmf", "pmf": pmf}
    row = b.total_row(dist, (0.0, 1.0), 8.5, [("DK", -110)], [("FD", -105)])
    assert row["side"] == "under" and row["pick_label"] == "Under 8.5"
    assert row["model_prob"] == pytest.approx(0.6)   # P(under) = 1 - 0.4


def test_spread_row_uses_cover_prob():
    md = {"kind": "margin", "offset": 10, "pmf": [0.0] * 21}
    md["pmf"][3 + 10] = 1.0   # all mass at margin +3 -> home -1.5 always covers
    row = b.spread_row(md, (0.0, 1.0), -1.5, [("DK", -140)], [("FD", 120)], "Home", "Away")
    assert row["side"] == "home" and row["pick_label"] == "Home -1.5"
    assert row["model_prob"] == pytest.approx(1.0)
