import pytest

from sportsmodel.serving import board as b


def test_decimal_and_implied():
    assert b.decimal_odds(100) == pytest.approx(2.0)
    assert b.decimal_odds(-110) == pytest.approx(1.9090909, rel=1e-4)
    assert b.implied_prob(100) == pytest.approx(0.5)


def test_best_price_picks_highest_decimal_major_books_only():
    # +120 (dec 2.2) beats +110 and -105 for the bettor
    assert b.best_price([("draftkings", 110), ("fanduel", 120), ("betmgm", -105)]) == ("fanduel", 120)
    assert b.best_price([("draftkings", -120), ("fanduel", -105)]) == ("fanduel", -105)
    assert b.best_price([]) is None
    # offshore/soft books are excluded even if they post the best number
    assert b.best_price([("bovada", 400), ("lowvig", 350)]) is None


def test_novig_removes_hold():
    assert b.novig(-110, -110) == pytest.approx(0.5)
    assert b.novig(-200, 170) > 0.5


def test_ev_sign():
    assert b.ev(0.6, 100) == pytest.approx(0.2)
    assert b.ev(0.4, 100) == pytest.approx(-0.2)


def test_moneyline_row_favors_and_prices_best_book_with_display_name():
    row = b.moneyline_row(0.60, [("draftkings", -130), ("fanduel", -120)], [("betmgm", 110)], "Home", "Away")
    assert row["side"] == "home" and row["pick_label"] == "Home ML"
    assert row["book"] == "FanDuel" and row["odds"] == -120  # display name; best home price
    assert row["model_prob"] == pytest.approx(0.60)
    assert row["is_pick"] is True   # EV ~+0.10, within (0, 0.25]


def test_moneyline_row_favors_away_when_model_leans_away():
    row = b.moneyline_row(0.40, [("draftkings", -130)], [("betmgm", 115)], "Home", "Away")
    assert row["side"] == "away" and row["pick_label"] == "Away ML"
    assert row["model_prob"] == pytest.approx(0.60)


def test_moneyline_row_none_when_only_offshore_books():
    assert b.moneyline_row(0.60, [("bovada", -120)], [("lowvig", 110)], "Home", "Away") is None


def test_total_row_picks_ev_side_not_mean():
    pmf = [0.0] * 21
    pmf[8], pmf[12] = 0.6, 0.4   # P(over 8.5) = 0.4 (<0.5) -> under
    dist = {"kind": "pmf", "pmf": pmf}
    row = b.total_row(dist, (0.0, 1.0), 8.5, [("draftkings", -110)], [("fanduel", -105)])
    assert row["side"] == "under" and row["pick_label"] == "Under 8.5"
    assert row["model_prob"] == pytest.approx(0.6)


def test_spread_row_uses_cover_prob():
    md = {"kind": "margin", "offset": 10, "pmf": [0.0] * 21}
    md["pmf"][3 + 10] = 1.0
    row = b.spread_row(md, (0.0, 1.0), -1.5, [("draftkings", -140)], [("fanduel", 120)], "Home", "Away")
    assert row["side"] == "home" and row["pick_label"] == "Home -1.5"
    assert row["model_prob"] == pytest.approx(1.0)


def test_ev_ceiling_excludes_absurd_plays():
    # P(over)=0.9 at +100 -> EV +0.8, way above the ceiling -> is_pick False
    pmf = [0.0] * 21
    pmf[12] = 0.9
    pmf[4] = 0.1
    dist = {"kind": "pmf", "pmf": pmf}
    row = b.total_row(dist, (0.0, 1.0), 8.5, [("draftkings", 100)], [("fanduel", 100)])
    assert row["ev"] > b.EV_CEILING and row["is_pick"] is False


def test_prop_row_over_only_home_run_passes_when_negative_ev():
    row = b.prop_row("home_run", {"kind": "pmf", "pmf": [0.88, 0.12]}, "home_run", 0.5,
                     [("draftkings", 650)], [])
    assert row["side"] == "over" and row["is_pick"] is False


def test_prop_row_picks_over_when_plus_ev_within_ceiling():
    dist = {"kind": "pmf", "pmf": [0.2, 0.2, 0.6]}  # P(over 1.5) = 0.6
    row = b.prop_row("hits", dist, "hits", 1.5, [("fanduel", -110)], [("draftkings", -120)])
    assert row["side"] == "over" and row["is_pick"] is True
