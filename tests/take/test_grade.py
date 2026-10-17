from datetime import UTC, datetime

import pytest

from sportsmodel.take.devig import novig_power_n
from sportsmodel.take.grade import close_fair, close_valid, grade_pick, settle

T = datetime(2026, 11, 2, 18, 0, tzinfo=UTC)


def pick(market, side, line, price, ref_line=None, league="nfl", pick_id=7):
    return {"pick_id": pick_id, "market": market, "side": side, "line": line, "price": price,
            "ref_line": ref_line, "league": league}


def close(prices, minutes=10.0):
    return {"event_id": "e1", "market": "x", "prices": prices, "captured_at": T, "minutes_to_start": minutes}


def ml(home, away):
    return {"home": {"line": None, "price": home}, "away": {"line": None, "price": away}}


def spread(home_line, home, away):
    return {"home": {"line": home_line, "price": home}, "away": {"line": -home_line, "price": away}}


def total(line, over, under):
    return {"over": {"line": line, "price": over}, "under": {"line": line, "price": under}}


def test_close_valid_window():
    assert close_valid(close(ml(-110, -110), 0)) and close_valid(close(ml(-110, -110), 30))
    assert not close_valid(close(ml(-110, -110), -1)) and not close_valid(close(ml(-110, -110), 31))
    assert not close_valid(None)


def test_close_fair_requires_complete_market():
    assert close_fair(ml(-120, 100), "nfl", "ml", "home") == pytest.approx(novig_power_n([-120, 100])[0])
    assert close_fair({"home": ml(-120, 100)["home"]}, "nfl", "ml", "home") is None
    assert close_fair(ml(-120, 100), "epl", "ml", "home") is None          # soccer needs the draw too
    three = {**ml(-120, 100), "draw": {"line": None, "price": 250}}
    assert close_fair(three, "epl", "ml", "draw") == pytest.approx(novig_power_n([-120, 250, 100])[1])
    assert close_fair(spread(-3, -110, -110), "nfl", "spread", "away") == pytest.approx(0.5)
    assert close_fair(total(47.5, -110, -110), "nfl", "total", "over") == pytest.approx(0.5)


def test_ml_clv():
    r = grade_pick(pick("ml", "home", None, 105), close(ml(-120, 100), 8), T)
    fair = novig_power_n([-120, 100])[0]
    assert r["graded_by"] == "clv" and r["pick_id"] == 7
    assert r["close_fair"] == pytest.approx(fair)
    assert r["clv"] == pytest.approx(fair * 2.05 - 1) and r["beat"] is True
    assert r["close_line"] is None and r["close_minutes"] == 8 and r["graded_at"] is T


def test_ml_clv_negative_is_not_beat():
    r = grade_pick(pick("ml", "home", None, -130), close(ml(-120, 100), 8), T)
    assert r["graded_by"] == "clv" and r["clv"] < 0 and r["beat"] is False


def test_spread_close_at_R_uses_clv():
    # better-line pick: away +3.5 vs Pinnacle's +3; the close sits at the reference line
    r = grade_pick(pick("spread", "away", 3.5, -110, ref_line=3.0), close(spread(-3.0, -110, -110), 12), T)
    assert r["graded_by"] == "clv" and r["close_line"] == 3.0
    assert r["close_fair"] == pytest.approx(0.5)
    assert r["clv"] == pytest.approx(0.5 * (1 + 100 / 110) - 1) and r["beat"] is False


def test_spread_moved_toward_pick_is_beat():
    r = grade_pick(pick("spread", "home", -3.0, -110, ref_line=-3.0), close(spread(-3.5, -110, -110), 12), T)
    assert r["graded_by"] == "direction" and r["beat"] is True
    assert r["clv"] is None and r["close_fair"] is None and r["close_line"] == -3.5


def test_spread_moved_away_is_miss():
    r = grade_pick(pick("spread", "home", -3.0, -110, ref_line=-3.0), close(spread(-2.5, -110, -110), 12), T)
    assert r["graded_by"] == "direction" and r["beat"] is False and r["clv"] is None and r["close_line"] == -2.5


def test_total_under_direction():
    r = grade_pick(pick("total", "under", 47.5, -110, ref_line=47.5), close(total(46.5, -110, -110), 12), T)
    assert r["graded_by"] == "direction" and r["beat"] is True and r["close_line"] == 46.5


def test_total_over_direction():
    up = grade_pick(pick("total", "over", 47.5, -110, ref_line=47.5), close(total(48.5, -110, -110), 12), T)
    down = grade_pick(pick("total", "over", 47.5, -110, ref_line=47.5), close(total(46.5, -110, -110), 12), T)
    assert up["beat"] is True and down["beat"] is False


def test_total_close_at_R_uses_clv():
    r = grade_pick(pick("total", "over", 47.0, -105, ref_line=47.5), close(total(47.5, -115, -105), 12), T)
    assert r["graded_by"] == "clv" and r["close_line"] == 47.5
    assert r["close_fair"] == pytest.approx(novig_power_n([-115, -105])[0])


def test_quarter_move_is_miss():
    r = grade_pick(pick("total", "over", 2.5, -110, ref_line=2.5, league="epl"), close(total(2.75, -110, -110), 12), T)
    assert r["graded_by"] == "direction" and r["beat"] is False


def test_half_move_boundary_is_beat():
    # away +2.5 closes at +2.0 (home -2.0): the pick's line is half a point better than the close
    r = grade_pick(pick("spread", "away", 2.5, -110, ref_line=2.5), close(spread(-2.0, -110, -110), 12), T)
    assert r["graded_by"] == "direction" and r["beat"] is True and r["close_line"] == 2.0


def test_no_close_when_late_or_early_or_missing():
    p = pick("ml", "home", None, 105)
    for c in (close(ml(-120, 100), -1), close(ml(-120, 100), 31), None):
        r = grade_pick(p, c, T)
        assert r["graded_by"] == "no_close"
        assert r["clv"] is None and r["beat"] is None and r["close_fair"] is None and r["close_line"] is None
        assert r["graded_at"] is T and r["pick_id"] == 7


def test_invalid_close_still_reports_minutes():
    r = grade_pick(pick("ml", "home", None, 105), close(ml(-120, 100), 31), T)
    assert r["close_minutes"] == 31


def test_no_close_when_a_side_is_missing():
    one_side = {"home": {"line": None, "price": -120}}
    r = grade_pick(pick("ml", "home", None, 105), close(one_side, 8), T)
    assert r["graded_by"] == "no_close" and r["clv"] is None and r["beat"] is None
    soccer = grade_pick(pick("ml", "draw", None, 250, league="epl"), close(ml(-120, 100), 8), T)
    assert soccer["graded_by"] == "no_close"
    sp = grade_pick(pick("spread", "home", -3.0, -110, ref_line=-3.0),
                    close({"home": {"line": -3.0, "price": -110}}, 8), T)
    assert sp["graded_by"] == "no_close"


def test_settle():
    assert settle(pick("spread", "away", 3.5, 110), 27, 24) == ("win", pytest.approx(1.1))
    assert settle(pick("spread", "home", -3.0, -110), 27, 24) == ("push", 0.0)
    assert settle(pick("spread", "home", -3.5, -110), 27, 24) == ("loss", -1.0)
    assert settle(pick("ml", "draw", None, 250, league="epl"), 1, 1) == ("win", pytest.approx(2.5))
    assert settle(pick("ml", "home", None, -150, league="epl"), 1, 1) == ("loss", -1.0)
    assert settle(pick("ml", "away", None, 200, league="epl"), 0, 1) == ("win", pytest.approx(2.0))
    assert settle(pick("ml", "draw", None, 250, league="epl"), 2, 1) == ("loss", -1.0)
    assert settle(pick("ml", "home", None, -150), 20, 20) == ("push", 0.0)
    assert settle(pick("ml", "home", None, -150), 24, 20) == ("win", pytest.approx(100 / 150))
    assert settle(pick("total", "over", 47.0, -110), 27, 20)[0] == "push"
    assert settle(pick("total", "over", 46.5, 100), 27, 20) == ("win", pytest.approx(1.0))
    assert settle(pick("total", "under", 46.5, 100), 27, 20) == ("loss", -1.0)
    assert settle(pick("total", "under", 47.5, -110), 27, 20) == ("win", pytest.approx(100 / 110))


def test_grade_pick_with_a_none_close_line_is_no_close():
    half = {"home": {"line": None, "price": -110}, "away": {"line": -3.0, "price": -110}}
    assert grade_pick(pick("spread", "home", -3.0, -105, ref_line=-3.0), close(half), T)["graded_by"] == "no_close"
    no_pt = {"over": {"line": 47.5, "price": -110}, "under": {"line": None, "price": -110}}
    assert grade_pick(pick("total", "over", 47.5, 105, ref_line=47.5), close(no_pt), T)["graded_by"] == "no_close"
    assert close_fair(half, "nfl", "spread", "home") is None and close_fair(no_pt, "nfl", "total", "over") is None


def test_grade_pick_with_no_ref_line_is_no_close():
    row = grade_pick(pick("total", "over", 47.5, 105, ref_line=None), close(total(47.5, -110, -110)), T)
    assert row["graded_by"] == "no_close" and row["clv"] is None and row["pick_id"] == 7
