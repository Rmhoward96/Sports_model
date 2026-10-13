"""early/grading.py: close EV, missing closes, line CLV signs, pushes and available_next_poll. Pure."""
from datetime import timedelta

import pytest

from sportsmodel.early import grading
from sportsmodel.early.constants import PINNACLE
from sportsmodel.ledger import reference

from .builders import NOW, StubTable, mass_at, row

KICK = NOW + timedelta(hours=60)
CLOSE_AT = KICK - timedelta(minutes=15)
M = 0.0318
TABLES = {("cfb", "total"): StubTable(mass_at((55, 56), M)), ("nfl", "spread"): StubTable(mass_at(3, M)),
          ("cfb", "spread"): StubTable(mass_at(3, M))}


def ticket(**over):
    base = {"id": 11, "sport": "cfb", "game_pk": 1, "market": "total", "side": "over", "book": "draftkings",
            "book_line": 54.5, "price": -110, "observed_at": NOW, "commence_time": KICK}
    return base | over


def close_total(line=55.5, over=-110, under=-110, at=CLOSE_AT):
    return [row(1, "total", "over", PINNACLE, line, over, KICK, captured_at=at),
            row(1, "total", "under", PINNACLE, line, under, KICK, captured_at=at)]


FINAL = {"actual_margin": 3.0, "actual_total": 58.0}


def grade(t=None, final=FINAL, close=None, later=(), tables=TABLES):
    return grading.grade_ticket(t or ticket(), final, close_total() if close is None else close, list(later), tables)


# ---- close EV and its companions -----------------------------------------------------------------------------

def test_close_ev_values_the_ticket_at_pinnacles_closing_fair():
    g = grade()
    assert g["close_kind"] == "pinnacle_novig" and g["close_fair"] == pytest.approx(0.5) and g["close_line"] == 55.5
    assert g["close_ev"] == pytest.approx(-0.04545454 + 1.9090909 * M, abs=1e-6)       # Over 54.5 vs a 55.5 close
    assert g["beat_close"] is True and g["close_minutes_to_kick"] == pytest.approx(15.0)


def test_close_ev_uses_the_closing_fair_not_the_flag_time_fair():
    sharp = grade(close=close_total(over=-130, under=110))          # the market moved toward the Over: fair above 50%
    assert sharp["close_fair"] > 0.5 and sharp["close_ev"] > grade()["close_ev"]
    soft = grade(close=close_total(over=110, under=-130))
    assert soft["close_ev"] < 0 and soft["beat_close"] is False


def test_no_pinnacle_quote_at_the_close_is_missing_and_never_beats_it():
    no_pin = [dict(r, book="draftkings") for r in close_total()]
    for rows in (no_pin, close_total()[:1], []):
        g = grade(close=rows)
        assert g["close_kind"] == "missing" and g["close_ev"] is None and g["beat_close"] is None
        assert g["close_fair"] is None and g["clv_pts"] is None and g["result"] == "W"     # still graded W/L/units


def test_a_quarter_point_closing_line_is_missing_not_rounded():
    assert grade(close=close_total(line=55.75))["close_kind"] == "missing"


def test_a_close_that_moved_more_than_three_points_still_counts_and_counts_against_the_ticket():
    g = grade(close=close_total(line=49.5))                           # the total fell 5 points; the Over 54.5 got worse
    assert g["close_kind"] == "pinnacle_novig" and g["close_ev"] is not None
    assert g["clv_pts"] == pytest.approx(-5.0)


def test_close_ev_exactly_zero_is_not_a_beat():
    flat = {("cfb", "total"): StubTable(mass_at((55, 56), 0.0))}
    g = grade(close=close_total(line=54.5), tables=flat)               # same line, fair 50%, price -110: EV = -4.5%
    assert g["close_ev"] < 0 and g["beat_close"] is False
    even = grade(t=ticket(price=100), close=close_total(line=54.5, over=100, under=100), tables=flat)
    assert even["close_ev"] == pytest.approx(0.0) and even["beat_close"] is False


# ---- clv_pts ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("market, side, book_line, close_line, expected", [
    ("total", "over", 54.5, 55.5, 1.0),         # we bet Over lower than the close: gained a point
    ("total", "over", 56.5, 55.5, -1.0),
    ("total", "under", 56.5, 55.5, 1.0),
    ("total", "under", 54.5, 55.5, -1.0),
    ("spread", "home", -2.5, -3.5, 1.0),        # home -2.5 vs a -3.5 close: gained a point
    ("spread", "away", 3.5, 2.5, 1.0),          # away +3.5 vs a +2.5 close
    ("spread", "away", 2.5, 3.5, -1.0),
])
def test_clv_pts_is_signed_in_the_ticket_sides_favour(market, side, book_line, close_line, expected):
    assert reference.clv_pts(market, side, book_line, close_line) == expected


def test_spread_ticket_clv_and_close_use_the_tickets_own_side_lines():
    t = ticket(sport="nfl", market="spread", side="away", book_line=3.5)
    close = [row(1, "spread", "home", PINNACLE, -2.5, -110, KICK, captured_at=CLOSE_AT),
             row(1, "spread", "away", PINNACLE, 2.5, -110, KICK, captured_at=CLOSE_AT)]
    g = grade(t=t, final={"actual_margin": 3.0, "actual_total": 40.0}, close=close)
    assert g["close_line"] == 2.5 and g["clv_pts"] == pytest.approx(1.0) and g["beat_close"] is True
    assert g["result"] == "W"                                           # home won by 3, so the away side +3.5 covers


# ---- result, units, pushes -------------------------------------------------------------------------------------

@pytest.mark.parametrize("total, result, units", [(60.0, "W", 100 / 110), (50.0, "L", -1.0)])
def test_result_and_units_at_the_tickets_price(total, result, units):
    g = grade(final={"actual_margin": 3.0, "actual_total": total})
    assert g["result"] == result and g["units"] == pytest.approx(units)


def test_a_push_pays_the_stake_back_and_is_still_graded():
    g = grade(t=ticket(book_line=58.0), final=FINAL)                   # total 58 on a 58 line
    assert g["result"] == "P" and g["units"] == 0.0 and g["close_kind"] == "pinnacle_novig"


def test_a_game_without_a_result_stays_ungraded():
    assert grade(final=None) is None and grade(final={}) is None
    assert grading.grade_tickets([ticket()], {}, {1: close_total()}, {}, TABLES) == []


def test_result_columns_match_the_table_columns():
    from sportsmodel import db
    assert set(grade()) == set(db.EARLY_COLUMNS["early_ticket_results"])


def test_grade_tickets_routes_finals_closes_and_later_polls_by_game():
    t2 = ticket(id=12, game_pk=2, sport="cfb")
    out = grading.grade_tickets([ticket(), t2], {("cfb", 1): FINAL}, {1: close_total()}, {}, TABLES)
    assert [g["ticket_id"] for g in out] == [11] and out[0]["close_kind"] == "pinnacle_novig"


# ---- available_next_poll ---------------------------------------------------------------------------------------

def later(minutes, *, price=-110, line=54.5, book="draftkings", side="over", pinnacle=True):
    at = NOW + timedelta(minutes=minutes)
    rows = [row(1, "total", side, book, line, price, KICK, captured_at=at)]
    if pinnacle:
        rows += [row(1, "total", "over", PINNACLE, 55.5, -110, KICK, captured_at=at),
                 row(1, "total", "under", PINNACLE, 55.5, -110, KICK, captured_at=at)]
    return rows


def avail(rows, **over):
    return grading.available_next_poll(ticket(**over), rows)


def test_same_line_same_or_better_price_one_poll_later_is_available():
    assert avail(later(60)) is True
    assert avail(later(60, price=-105)) is True
    assert avail(later(60, price=120)) is True                          # across the +/-100 boundary: better


def test_a_worse_price_a_moved_line_or_a_pulled_book_is_not_available():
    assert avail(later(60, price=-115)) is False
    assert avail(later(60, line=55.5)) is False
    assert avail(later(60, line=53.5)) is False
    assert avail(later(60, book="fanduel")) is False
    assert avail(later(60, side="under")) is False


def test_the_next_poll_must_be_30_minutes_to_3_hours_after_the_ticket():
    assert avail(later(30)) is True and avail(later(180)) is True
    assert avail(later(29)) is None                                      # too soon to be a new look, and nothing else
    assert avail(later(181)) is None
    assert avail([]) is None
    assert avail(later(20) + later(75, price=-115)) is False             # a 20-minute snapshot is skipped; the 75-minute poll decides


def test_the_first_qualifying_poll_decides_not_a_later_one():
    assert avail(later(60, price=-115) + later(120)) is False
    assert avail(later(60) + later(120, price=-150)) is True


def test_a_snapshot_without_a_pinnacle_quote_is_not_a_poll():
    assert avail(later(60, pinnacle=False)) is None
    assert avail(later(60, pinnacle=False) + later(90)) is True


def test_the_tickets_book_key_is_matched_exactly_and_line_with_a_float32_tolerance():
    assert avail(later(60, line=54.5000001)) is True
    assert avail(later(60), book="fanduel") is False


def test_available_next_poll_is_stored_on_the_result():
    assert grade(later=later(60))["available_next_poll"] is True
    assert grade(later=[])["available_next_poll"] is None
