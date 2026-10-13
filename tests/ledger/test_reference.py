"""Close reference selection and CLV maths. Pure."""
import pytest

from sportsmodel.ledger import reference as ref


def row(market, side, book, price, line=None, player_name=""):
    return {"market": market, "side": side, "book": book, "price": price, "line": line, "player_name": player_name}


def two_way(market, book, p_a, p_b, line_a=None, line_b=None, sides=("home", "away"), player=""):
    return [row(market, sides[0], book, p_a, line_a, player), row(market, sides[1], book, p_b, line_b, player)]


def spread_pick(side="home", line=-3.5):
    return {"sport": "nfl", "market": "spread", "side": side, "line": line, "player_id": None, "player_name": None}


def prop_pick(side="over", line=64.5):
    return {"sport": "nfl", "market": "rec_yds", "side": side, "line": line, "player_id": "00-1", "player_name": "A.J. Brown Jr."}


# ---- reference selection --------------------------------------------------------------------------------

def test_pinnacle_at_the_exact_line_wins_and_is_no_vigged():
    rows = (two_way("spread", "pinnacle", -110, -110, -3.5, 3.5)
            + two_way("spread", "draftkings", -120, 100, -3.5, 3.5))
    out = ref.close_reference(spread_pick(), rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(0.5)
    assert out["same_line"] is True and out["close_line"] == -3.5 and out["books"] == 1


def test_away_side_reads_the_mirrored_line():
    rows = two_way("spread", "pinnacle", -115, -105, -3.5, 3.5)
    out = ref.close_reference(spread_pick("away", 3.5), rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(1 - ref.novig(-115, -105))


def test_without_pinnacle_three_us_books_at_the_exact_line_give_the_median():
    rows = (two_way("total", "draftkings", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "fanduel", -115, -105, 47.5, 47.5, ("over", "under"))
            + two_way("total", "betmgm", -120, 100, 47.5, 47.5, ("over", "under")))
    pick = {"sport": "nfl", "market": "total", "side": "over", "line": 47.5, "player_id": None, "player_name": None}
    out = ref.close_reference(pick, rows)
    fairs = sorted([0.5, ref.novig(-115, -105), ref.novig(-120, 100)])
    assert out["close_kind"] == "us_consensus_novig" and out["fair_prob"] == pytest.approx(fairs[1])
    assert out["books"] == 3 and out["same_line"] is True


def test_two_us_books_are_not_a_consensus():
    rows = (two_way("total", "draftkings", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "fanduel", -115, -105, 47.5, 47.5, ("over", "under")))
    pick = {"sport": "nfl", "market": "total", "side": "over", "line": 47.5, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "missing"


def test_a_book_at_another_line_does_not_count_toward_the_consensus():
    rows = (two_way("total", "draftkings", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "fanduel", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "betmgm", -110, -110, 48.5, 48.5, ("over", "under")))
    pick = {"sport": "nfl", "market": "total", "side": "over", "line": 47.5, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "missing"


def test_caesars_and_williamhill_are_one_book_and_offshore_books_do_not_count():
    rows = []
    for book in ("williamhill_us", "caesars", "bovada", "draftkings"):
        rows += two_way("moneyline", book, -150, 130)
    pick = {"sport": "nfl", "market": "moneyline", "side": "home", "line": None, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "missing"            # draftkings + one Caesars = 2 books
    rows += two_way("moneyline", "fanduel", -150, 130)
    out = ref.close_reference(pick, rows)
    assert out["close_kind"] == "us_consensus_novig" and out["books"] == 3


def test_moneyline_has_no_line_to_match():
    rows = two_way("moneyline", "pinnacle", -200, 170)
    pick = {"sport": "cfb", "market": "moneyline", "side": "away", "line": None, "player_id": None, "player_name": None}
    out = ref.close_reference(pick, rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(1 - ref.novig(-200, 170))
    assert out["same_line"] is True and out["close_line"] is None


def test_pinnacle_on_another_line_gives_the_line_move_but_no_probability():
    rows = two_way("spread", "pinnacle", -110, -110, -4.5, 4.5)
    out = ref.close_reference(spread_pick("home", -3.5), rows)
    assert out == {"close_kind": "pinnacle_novig", "fair_prob": None, "close_line": -4.5, "same_line": False, "books": 1}


def test_consensus_at_the_exact_line_beats_a_moved_pinnacle_but_reports_the_moved_line():
    rows = two_way("spread", "pinnacle", -110, -110, -4.5, 4.5)
    for b in ("draftkings", "fanduel", "betmgm"):
        rows += two_way("spread", b, -110, -110, -3.5, 3.5)
    out = ref.close_reference(spread_pick("home", -3.5), rows)
    assert out["close_kind"] == "us_consensus_novig" and out["fair_prob"] == pytest.approx(0.5)
    assert out["close_line"] == -4.5 and out["same_line"] is False


def test_nothing_quoted_is_missing():
    assert ref.close_reference(spread_pick(), []) == {
        "close_kind": "missing", "fair_prob": None, "close_line": None, "same_line": None, "books": 0}
    assert ref.close_reference(spread_pick(), two_way("total", "pinnacle", -110, -110, 47.5, 47.5, ("over", "under")))["close_kind"] == "missing"


def test_a_one_sided_quote_is_not_usable():
    rows = [row("spread", "pinnacle", -110, -110, -3.5)]
    assert ref.close_reference(spread_pick(), rows)["close_kind"] == "missing"


def test_props_match_the_player_by_normalised_name_and_the_exact_line():
    rows = (two_way("reception_yds", "pinnacle", -120, 100, 64.5, 64.5, ("over", "under"), player="AJ Brown")
            + two_way("reception_yds", "pinnacle", -110, -110, 64.5, 64.5, ("over", "under"), player="Someone Else"))
    out = ref.close_reference(prop_pick(), rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(ref.novig(-120, 100))


def test_prop_consensus_ignores_other_players_and_other_lines():
    rows = []
    for b in ("draftkings", "fanduel"):
        rows += two_way("reception_yds", b, -110, -110, 64.5, 64.5, ("over", "under"), player="A.J. Brown")
    rows += two_way("reception_yds", "betmgm", -110, -110, 66.5, 66.5, ("over", "under"), player="A.J. Brown")
    rows += two_way("reception_yds", "caesars", -110, -110, 64.5, 64.5, ("over", "under"), player="Other Guy")
    assert ref.close_reference(prop_pick(), rows)["close_kind"] == "missing"
    rows += two_way("reception_yds", "betmgm", -110, -110, 64.5, 64.5, ("over", "under"), player="A.J. Brown")
    out = ref.close_reference(prop_pick(), rows)
    assert out["close_kind"] == "us_consensus_novig" and out["books"] == 3


def test_game_market_rows_are_not_confused_with_prop_rows_and_unknown_markets_are_missing():
    rows = two_way("spread", "pinnacle", -110, -110, -3.5, 3.5, player="Some Player")
    assert ref.close_reference(spread_pick(), rows)["close_kind"] == "missing"        # player rows are props
    odd = {"sport": "nfl", "market": "anytime_td", "side": "over", "line": 0.5, "player_id": "1", "player_name": "x"}
    assert ref.close_reference(odd, [])["close_kind"] == "missing"


def test_float32_lines_still_match():
    rows = two_way("total", "pinnacle", -110, -110, 4.099999904632568, 4.099999904632568, ("over", "under"))
    pick = {"sport": "mlb", "market": "total", "side": "over", "line": 4.1, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "pinnacle_novig"


# ---- CLV maths -----------------------------------------------------------------------------------------

def test_clv_prob_is_close_fair_minus_the_implied_price_taken():
    assert ref.clv_prob(0.55, -110) == pytest.approx(0.55 - 110 / 210)
    assert ref.clv_prob(0.50, 105) == pytest.approx(0.5 - 100 / 205)
    assert ref.clv_prob(None, -110) is None and ref.clv_prob(0.5, None) is None


@pytest.mark.parametrize("market, side, pick, close, expected", [
    ("spread", "home", -3.5, -4.5, 1.0),      # took -3.5, market now -4.5: a point better
    ("spread", "home", -3.5, -2.5, -1.0),
    ("spread", "away", 3.5, 2.5, 1.0),
    ("spread", "away", 3.5, 4.5, -1.0),
    ("total", "over", 45.5, 47.0, 1.5),
    ("total", "over", 45.5, 44.5, -1.0),
    ("total", "under", 45.5, 44.5, 1.0),
    ("total", "under", 45.5, 47.0, -1.5),
    ("spread", "home", -3.5, -3.5, 0.0),
    ("moneyline", "home", None, None, None),
    ("spread", "home", -3.5, None, None),
])
def test_clv_pts_signs(market, side, pick, close, expected):
    got = ref.clv_pts(market, side, pick, close)
    assert got == expected if expected is None else got == pytest.approx(expected)


def test_prop_clv_pts_uses_the_total_convention():
    assert ref.clv_pts("rec_yds", "over", 64.5, 66.5) == pytest.approx(2.0)
    assert ref.clv_pts("rec_yds", "under", 64.5, 66.5) == pytest.approx(-2.0)


@pytest.mark.parametrize("prob, pts, expected", [
    (0.02, 0.0, True), (-0.01, 0.0, False), (None, 0.0, None), (0.03, None, True), (None, None, None),
    (-0.05, 1.0, True), (0.05, -1.0, False),            # a moved line is judged in points, not by price
])
def test_beat_close(prob, pts, expected):
    assert ref.beat_close(prob, pts) is expected


# ---- settlement and units ------------------------------------------------------------------------------

@pytest.mark.parametrize("market, side, line, margin, total, expected", [
    ("moneyline", "home", None, 7, 40, "W"), ("moneyline", "away", None, 7, 40, "L"), ("moneyline", "home", None, 0, 40, "P"),
    ("spread", "home", -3.5, 7, 40, "W"), ("spread", "home", -7.0, 7, 40, "P"), ("spread", "home", -10.5, 7, 40, "L"),
    ("spread", "away", 3.5, 7, 40, "L"), ("spread", "away", 7.5, 7, 40, "W"), ("spread", "away", 7.0, 7, 40, "P"),
    ("total", "over", 38.5, 7, 40, "W"), ("total", "under", 40.0, 7, 40, "P"), ("total", "under", 38.5, 7, 40, "L"),
    ("spread", "home", None, 7, 40, None), ("total", "over", 40.5, 7, None, None), ("moneyline", "home", None, None, 40, None),
    ("prop", "home", 1, 1, 1, None),
])
def test_settle_game(market, side, line, margin, total, expected):
    assert ref.settle_game(market, side, line, margin, total) == expected


@pytest.mark.parametrize("side, line, actual, expected", [
    ("over", 64.5, 70, "W"), ("over", 64.5, 60, "L"), ("over", 64.0, 64, "P"),
    ("under", 64.5, 60, "W"), ("under", 64.5, 70, "L"), ("under", 64.0, 64, "P"),
    ("over", 64.5, None, None), ("over", None, 70, None), ("home", 64.5, 70, None)])
def test_settle_prop(side, line, actual, expected):
    assert ref.settle_prop(side, line, actual) == expected


def test_units_at_pick():
    assert ref.units_at_pick("W", -110) == pytest.approx(100 / 110)
    assert ref.units_at_pick("W", 150) == pytest.approx(1.5)
    assert ref.units_at_pick("L", 150) == -1.0 and ref.units_at_pick("P", -110) == 0.0
    assert ref.units_at_pick("W", None) is None and ref.units_at_pick(None, -110) is None
