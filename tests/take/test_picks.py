from datetime import UTC, datetime, timedelta

import pytest

from sportsmodel.take.picks import find_picks, good_until, pinnacle_fair, same_line

PULLED = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)
EV = {"event_id": "e1", "league": "nfl", "home_team": "H", "away_team": "A",
      "commence_time": PULLED + timedelta(hours=2)}
SOCCER_EV = {**EV, "event_id": "s1", "league": "epl"}


def row(book, market, side, line, price, age_min=1, event_id="e1", league="nfl"):
    return {"event_id": event_id, "league": league, "market": market, "side": side, "book": book,
            "line": line, "price": price, "last_update": PULLED - timedelta(minutes=age_min), "pulled_at": PULLED}


def pin_spread(line, home, away, age_min=1):
    return [row("pinnacle", "spread", "home", line, home, age_min),
            row("pinnacle", "spread", "away", -line, away, age_min)]


def pin_ml(home, away, age_min=1):
    return [row("pinnacle", "ml", "home", None, home, age_min), row("pinnacle", "ml", "away", None, away, age_min)]


def pin_total(line, over, under, age_min=1):
    return [row("pinnacle", "total", "over", line, over, age_min), row("pinnacle", "total", "under", line, under, age_min)]


def pin_soccer(home, draw, away, age_min=1):
    return [row("pinnacle", "ml", s, None, p, age_min, "s1", "epl") for s, p in
            (("home", home), ("draw", draw), ("away", away))]


def test_same_line_and_good_until_helpers():
    assert same_line(None, None) and not same_line(None, 1.0) and same_line(-3.0, -3.0000001)
    assert not same_line(-3.0, -2.5)
    assert good_until(0.5) == 103


def test_same_line_pick():          # pin -110/-110 at -3; DK home -3 at +105 -> ev = 0.5*2.05-1 = 0.025
    p = find_picks([*pin_spread(-3, -110, -110), row("draftkings", "spread", "home", -3, 105)], [EV], PULLED)
    assert len(p) == 1 and p[0]["book"] == "draftkings" and not p[0]["better_line"]
    assert p[0]["ev"] == pytest.approx(0.025) and p[0]["ref_line"] == -3 and p[0]["good_until"] == 103
    assert p[0]["fair_prob"] == pytest.approx(0.5) and p[0]["others"] == []
    assert p[0]["pulled_at"] == PULLED and p[0]["commence_time"] == EV["commence_time"]
    assert p[0]["pin_prices"] == {"home": {"line": -3, "price": -110}, "away": {"line": 3, "price": -110}}


def test_better_line_pick_marked_and_uses_pinnacle_fair():
    p = find_picks([*pin_spread(-3, -110, -110), row("fanduel", "spread", "away", 3.5, 103)], [EV], PULLED)
    assert p[0]["better_line"] and p[0]["line"] == 3.5 and p[0]["ref_line"] == 3
    assert p[0]["fair_prob"] == pytest.approx(0.5) and p[0]["ev"] == pytest.approx(0.015)


def test_worse_line_no_pick():
    assert find_picks([*pin_spread(-3, -110, -110), row("fanduel", "spread", "away", 2.5, 120)], [EV], PULLED) == []


def test_better_line_direction_for_totals_and_home_spread():
    base = pin_total(47.5, -110, -110)
    assert len(find_picks([*base, row("fanduel", "total", "over", 47, 105)], [EV], PULLED)) == 1
    assert find_picks([*base, row("fanduel", "total", "over", 48, 120)], [EV], PULLED) == []
    assert len(find_picks([*base, row("fanduel", "total", "under", 48, 105)], [EV], PULLED)) == 1
    assert find_picks([*base, row("fanduel", "total", "under", 47, 120)], [EV], PULLED) == []
    sp = pin_spread(-3, -110, -110)
    assert len(find_picks([*sp, row("fanduel", "spread", "home", -2.5, 105)], [EV], PULLED)) == 1
    assert find_picks([*sp, row("fanduel", "spread", "home", -3.5, 120)], [EV], PULLED) == []


def test_threshold_and_price_range():
    ml = pin_ml(-110, -110)      # fair .5 each
    assert find_picks([*ml, row("draftkings", "ml", "home", None, 102)], [EV], PULLED) == []     # ev .01
    assert len(find_picks([*ml, row("draftkings", "ml", "home", None, 103)], [EV], PULLED)) == 1  # ev .015 passes
    lopsided = pin_ml(-300, 250)
    assert find_picks([*lopsided, row("draftkings", "ml", "home", None, -260)], [EV], PULLED) == []
    longshot = pin_ml(420, -500)
    assert find_picks([*longshot, row("draftkings", "ml", "home", None, 550)], [EV], PULLED) == []
    assert len(find_picks([*longshot, row("draftkings", "ml", "home", None, 500)], [EV], PULLED)) == 1


def test_stale_pinnacle_no_picks():
    assert find_picks([*pin_spread(-3, -110, -110, age_min=31), row("draftkings", "spread", "home", -3, 120)], [EV], PULLED) == []
    assert len(find_picks([*pin_spread(-3, -110, -110, age_min=30), row("draftkings", "spread", "home", -3, 120)], [EV], PULLED)) == 1
    no_update = pin_spread(-3, -110, -110)
    no_update[0]["last_update"] = None
    assert find_picks([*no_update, row("draftkings", "spread", "home", -3, 120)], [EV], PULLED) == []


def test_started_game_no_picks():
    started = {**EV, "commence_time": PULLED}
    assert find_picks([*pin_spread(-3, -110, -110), row("draftkings", "spread", "home", -3, 120)], [started], PULLED) == []


def test_partial_pinnacle_market_makes_no_picks():
    soccer_partial = [row("pinnacle", "ml", s, None, p, 1, "s1", "epl") for s, p in (("home", 150), ("away", 190))]
    book = row("fanduel", "ml", "draw", None, 400, 1, "s1", "epl")
    assert find_picks([*soccer_partial, book], [SOCCER_EV], PULLED) == []
    assert find_picks([row("pinnacle", "spread", "home", -3, -110), row("draftkings", "spread", "home", -3, 120)],
                      [EV], PULLED) == []
    assert find_picks([row("pinnacle", "total", "over", 47.5, -110), row("draftkings", "total", "over", 47.5, 120)],
                      [EV], PULLED) == []
    assert find_picks([row("pinnacle", "ml", "home", None, -110), row("draftkings", "ml", "home", None, 120)],
                      [EV], PULLED) == []


def test_incomplete_or_inconsistent_pinnacle_lines():
    assert pinnacle_fair(pin_spread(-3, -110, -110)[:1] + [row("pinnacle", "spread", "away", 2.5, -110)],
                         "nfl", "spread", PULLED) is None
    assert pinnacle_fair([row("pinnacle", "total", "over", 47.5, -110), row("pinnacle", "total", "under", 48, -110)],
                         "nfl", "total", PULLED) is None
    fair = pinnacle_fair(pin_total(47.5, -110, -110), "nfl", "total", PULLED)
    assert set(fair) == {"over", "under"} and fair["over"]["fair"] == pytest.approx(0.5)
    assert fair["over"]["line"] == 47.5 and fair["over"]["price"] == -110


def test_soccer_draw_pickable():
    p = find_picks([*pin_soccer(150, 230, 190), row("fanduel", "ml", "draw", None, 260, 1, "s1", "epl")],
                   [SOCCER_EV], PULLED)
    assert len(p) == 1 and p[0]["side"] == "draw" and p[0]["ref_line"] is None
    assert set(p[0]["pin_prices"]) == {"home", "draw", "away"}


def test_line_float_tolerance():
    p = find_picks([*pin_spread(-3.0, -110, -110), row("draftkings", "spread", "home", -3.0000001, 105)], [EV], PULLED)
    assert len(p) == 1 and not p[0]["better_line"]
    t = find_picks([*pin_total(2.25, -110, -110), row("draftkings", "total", "over", 2.25, 105)], [EV], PULLED)
    assert len(t) == 1 and not t[0]["better_line"]


def test_best_book_and_others():
    prices = [*pin_spread(-3, -110, -110), row("draftkings", "spread", "home", -3, 105),
              row("fanduel", "spread", "home", -3, 110)]
    p = find_picks(prices, [EV], PULLED)
    assert len(p) == 1 and p[0]["book"] == "fanduel"
    assert p[0]["others"] == [{"book": "draftkings", "line": -3, "price": 105, "ev": pytest.approx(0.025)}]


def test_tie_prefers_same_line_then_book_order():
    sp = pin_spread(-3, -110, -110)
    same_first = find_picks([*sp, row("fanduel", "spread", "home", -3, 105), row("draftkings", "spread", "home", -2.5, 105)],
                            [EV], PULLED)
    assert same_first[0]["book"] == "fanduel" and not same_first[0]["better_line"]
    assert [o["book"] for o in same_first[0]["others"]] == ["draftkings"]
    order = find_picks([*sp, row("betmgm", "spread", "home", -3, 105), row("draftkings", "spread", "home", -3, 105)],
                       [EV], PULLED)
    assert order[0]["book"] == "draftkings" and order[0]["others"][0]["book"] == "betmgm"


def test_non_list_book_ignored():
    assert find_picks([*pin_spread(-3, -110, -110), row("betrivers", "spread", "home", -3, 150)], [EV], PULLED) == []


def test_null_line_spread_and_total_rows_ignored():
    sp = [*pin_spread(-3, -110, -110), row("draftkings", "spread", "home", None, 150)]
    assert find_picks(sp, [EV], PULLED) == []
    assert find_picks([row("pinnacle", "total", "over", None, -110), row("pinnacle", "total", "under", None, -110),
                       row("draftkings", "total", "over", 47.5, 150)], [EV], PULLED) == []


def test_duplicate_pinnacle_side_uses_first():
    dup = [*pin_spread(-3, -110, -110), row("pinnacle", "spread", "home", -3, -150)]
    p = find_picks([*dup, row("draftkings", "spread", "home", -3, 105)], [EV], PULLED)
    assert len(p) == 1 and p[0]["fair_prob"] == pytest.approx(0.5)


def test_one_pick_per_event_market_side():
    prices = [*pin_ml(-110, -110), row("draftkings", "ml", "home", None, 110), row("draftkings", "ml", "away", None, 110),
              *pin_total(47.5, -110, -110), row("fanduel", "total", "over", 47.5, 110)]
    p = find_picks(prices, [EV], PULLED)
    assert sorted((x["market"], x["side"]) for x in p) == [("ml", "away"), ("ml", "home"), ("total", "over")]


def test_unknown_event_ignored():
    assert find_picks([*pin_spread(-3, -110, -110), row("draftkings", "spread", "home", -3, 120)], [], PULLED) == []
