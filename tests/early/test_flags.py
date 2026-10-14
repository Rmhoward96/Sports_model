"""flags.py: Pinnacle's reference, the best book per side, and every filter of the flag rule. Pure."""
from datetime import timedelta

import pytest

from sportsmodel.early import flags
from sportsmodel.early.constants import FLAG_EV_MIN

from .builders import NOW, StubTable, mass_at, row, total_rows, two_way

# A game 72 hours away; Pinnacle totals 55.5 at -110/-110 (fair 50%); a book at 54.5 gains the table mass on 55.
KICK = NOW + timedelta(hours=72)
M_FLAGS = 0.0318          # EV of Over 54.5 -110 = -0.04545 + 1.909 * m = +1.525%: just above the 1.5% line
M_SHORT = 0.0314          # ... and +1.449%: just below


def tables(m=M_FLAGS):
    return {("cfb", "total"): StubTable(mass_at((55, 56), m)), ("cfb", "spread"): StubTable(mass_at(3, m)),
            ("nfl", "spread"): StubTable(mass_at(3, m))}


def poll(rows, *, sport="cfb", existing=None, m=M_FLAGS, observed_at=NOW):
    return flags.flag_poll(rows, sport=sport, observed_at=observed_at, tables=tables(m), existing=existing or set(),
                           poll_run_id="run-1")


def good_rows(**books):
    return total_rows(1, KICK, books=books or {"draftkings": (54.5, -110, -110)})


def test_the_main_case_makes_one_over_ticket_with_every_stored_field():
    (t,) = poll(good_rows())
    assert (t["sport"], t["game_pk"], t["market"], t["side"], t["book"]) == ("cfb", 1, "total", "over", "draftkings")
    assert (t["book_line"], t["price"], t["pinnacle_line"]) == (54.5, -110, 55.5)
    assert t["pinnacle_fair"] == pytest.approx(0.5)
    assert t["p_win"] == pytest.approx(0.5 + M_FLAGS) and t["p_push"] == 0.0 and t["p_loss"] == pytest.approx(0.5 - M_FLAGS)
    assert t["ev"] == pytest.approx(-0.04545454 + 1.9090909 * M_FLAGS, abs=1e-6) and t["ev"] >= FLAG_EV_MIN
    assert t["observed_at"] == NOW and t["commence_time"] == KICK and t["hours_to_kick"] == pytest.approx(72.0)
    assert t["poll_run_id"] == "run-1"


def test_ticket_fields_are_exactly_the_early_tickets_columns():
    from sportsmodel import db
    (t,) = poll(good_rows())
    assert set(t) == set(db.EARLY_COLUMNS["early_tickets"])


def test_the_flag_threshold_is_one_and_a_half_percent_ev():
    assert len(poll(good_rows(), m=M_FLAGS)) == 1
    assert poll(good_rows(), m=M_SHORT) == []


def test_the_under_side_of_a_higher_book_total_flags_too():
    (t,) = poll(total_rows(1, KICK, books={"fanduel": (56.5, -110, -110)}))
    assert (t["side"], t["book"], t["book_line"]) == ("under", "fanduel", 56.5) and t["pinnacle_line"] == 55.5


def test_kickoff_must_be_at_least_24_hours_after_the_poll():
    assert len(poll(total_rows(1, NOW + timedelta(hours=24), books={"draftkings": (54.5, -110, -110)}))) == 1
    assert poll(total_rows(1, NOW + timedelta(hours=24) - timedelta(seconds=1),
                           books={"draftkings": (54.5, -110, -110)})) == []


def test_no_pinnacle_two_way_in_this_poll_means_no_ticket():
    rows = [r for r in good_rows() if r["book"] != "pinnacle"]
    assert poll(rows) == []
    one_leg = [r for r in good_rows() if not (r["book"] == "pinnacle" and r["side"] == "under")]
    assert poll(one_leg) == []
    wrong_line = [dict(r, line=56.5) if (r["book"] == "pinnacle" and r["side"] == "under") else r for r in good_rows()]
    assert poll(wrong_line) == []


def test_pinnacle_fair_outside_15_to_85_percent_means_no_ticket_for_either_side():
    lopsided = total_rows(1, KICK, pin=(-800, 550), books={"draftkings": (54.5, -110, -110), "fanduel": (56.5, 300, -110)})
    assert poll(lopsided) == []
    inside = total_rows(1, KICK, pin=(-600, 450), books={"draftkings": (54.5, -110, -110)})
    assert flags.pinnacle_reference(inside, "total").fair["over"] == pytest.approx(0.8251, abs=1e-3)
    assert poll(inside)                                   # 0.825 is inside the range: tickets are allowed


def test_the_highest_ev_book_wins_and_ties_go_to_the_first_book_name():
    rows = good_rows(fanduel=(54.5, -105, -110), draftkings=(54.5, -110, -110), betmgm=(54.5, -105, -110))
    (t,) = poll(rows)
    assert t["book"] == "betmgm" and t["price"] == -105           # better price, and betmgm < fanduel alphabetically
    tie = good_rows(fanduel=(54.5, -110, -110), draftkings=(54.5, -110, -110))
    assert poll(tie)[0]["book"] == "draftkings"


def test_only_the_users_nine_books_can_hold_a_ticket():
    rows = good_rows(bovada=(53.5, 150, -110), betonlineag=(53.5, 150, -110))
    assert poll(rows) == []                                      # offshore books: never
    pinnacle_only = total_rows(1, KICK)
    assert poll(pinnacle_only) == []                              # Pinnacle itself is the reference, not a bet


def test_a_book_more_than_three_points_from_pinnacle_is_a_data_error_not_a_ticket():
    assert poll(good_rows(draftkings=(51.5, -110, -110))) == []
    assert len(poll(good_rows(draftkings=(52.5, -110, -110)))) == 1        # exactly 3.0 points is allowed


def test_one_ticket_per_side_ever_later_polls_never_add_a_second():
    first = poll(good_rows())
    assert len(first) == 1
    existing = {(t["sport"], t["game_pk"], t["market"], t["side"]) for t in first}
    assert poll(good_rows(), existing=existing) == []
    # a different side of the same market is still free to flag
    both = good_rows(draftkings=(54.5, -110, -110), fanduel=(56.5, -110, -110))
    assert {t["side"] for t in poll(both)} == {"over", "under"}
    assert {t["side"] for t in poll(both, existing=existing)} == {"under"}


def test_only_the_tested_markets_are_flagged():
    nfl_totals = [dict(r, game_pk=2) for r in good_rows()]
    assert poll(nfl_totals, sport="nfl") == []                     # NFL totals are excluded
    moneyline = [row(3, "moneyline", "home", "pinnacle", None, -150, KICK), row(3, "moneyline", "away", "pinnacle", None, 130, KICK),
                 row(3, "moneyline", "home", "draftkings", None, -120, KICK), row(3, "moneyline", "away", "draftkings", None, 130, KICK)]
    assert poll(moneyline) == [] and poll(moneyline, sport="nfl") == []
    props = [dict(r, player_name="A. Player") for r in good_rows()]
    assert poll(props) == []


def test_spread_sides_use_each_sides_own_lines():
    rows = (two_way(7, "spread", "pinnacle", -3.5, -110, 3.5, -110, KICK, ("home", "away"))
            + two_way(7, "spread", "betmgm", -2.5, -110, 2.5, -110, KICK, ("home", "away")))
    (t,) = poll(rows, sport="nfl")                                   # home -2.5 vs Pinnacle -3.5 gains the margin-3 mass
    assert (t["side"], t["book"], t["book_line"], t["pinnacle_line"]) == ("home", "betmgm", -2.5, -3.5)
    away = (two_way(7, "spread", "pinnacle", -2.5, -110, 2.5, -110, KICK, ("home", "away"))
            + two_way(7, "spread", "betmgm", -3.5, -110, 3.5, -110, KICK, ("home", "away")))
    (t,) = poll(away, sport="nfl")                                   # away +3.5 vs +2.5 gains the margin-3 mass
    assert (t["side"], t["book_line"], t["pinnacle_line"]) == ("away", 3.5, 2.5)


def test_pinnacle_with_several_quoted_pairs_uses_the_one_closest_to_even():
    rows = (two_way(1, "total", "pinnacle", 55.5, -110, 55.5, -110, KICK, ("over", "under"))
            + two_way(1, "total", "pinnacle", 52.5, -180, 52.5, 150, KICK, ("over", "under")))
    ref = flags.pinnacle_reference(rows, "total")
    assert ref.threshold == 55.5 and ref.fair["over"] == pytest.approx(0.5)


def test_quarter_point_lines_are_never_rounded_into_a_ticket():
    assert poll(good_rows(draftkings=(54.25, -110, -110))) == []
    quarter_pin = total_rows(1, KICK, pin_line=55.75, books={"draftkings": (54.5, -110, -110)})
    assert poll(quarter_pin) == []


def test_a_game_with_no_parseable_kickoff_or_no_table_is_skipped():
    rows = [dict(r, commence_time=None) for r in good_rows()]
    assert poll(rows) == []
    assert flags.flag_poll(good_rows(), sport="cfb", observed_at=NOW, tables={}, existing=set(), poll_run_id="r") == []


def test_output_is_deterministic_and_ordered_by_game():
    rows = [dict(r, game_pk=pk) for pk in (9, 4) for r in good_rows()]
    assert [t["game_pk"] for t in poll(rows)] == [4, 9]
    assert poll(rows) == poll(list(reversed(rows)))


def test_a_book_posting_alternate_lines_is_valued_at_each_line_and_the_best_one_wins():
    rows = good_rows(draftkings=(54.5, -110, -110)) + two_way(1, "total", "draftkings", 53.5, -150, 53.5, 125, KICK, ("over", "under"))
    (t,) = [t for t in poll(rows) if t["side"] == "over"]
    assert (t["book_line"], t["price"]) == (54.5, -110)               # 53.5 at -150 is worse than 54.5 at -110 here
    cheap_alt = good_rows(draftkings=(54.5, -110, -110)) + two_way(1, "total", "draftkings", 53.5, 120, 53.5, -140, KICK, ("over", "under"))
    (t,) = [t for t in poll(cheap_alt) if t["side"] == "over"]
    assert (t["book_line"], t["price"]) == (53.5, 120)


def test_a_line_outside_the_committed_landing_tables_makes_no_ticket():
    from sportsmodel.early import landing
    real = landing.load_tables()
    heavy = (two_way(5, "spread", "pinnacle", -38.5, -110, 38.5, -110, KICK, ("home", "away"))
             + two_way(5, "spread", "draftkings", -37.5, -110, 37.5, -110, KICK, ("home", "away")))
    assert flags.flag_poll(heavy, sport="cfb", observed_at=NOW, tables=real, existing=set(), poll_run_id="r") == []
    ordinary = (two_way(5, "spread", "pinnacle", -3.5, -110, 3.5, -110, KICK, ("home", "away"))
                + two_way(5, "spread", "draftkings", -2.5, -110, 2.5, -110, KICK, ("home", "away")))
    got = flags.flag_poll(ordinary, sport="cfb", observed_at=NOW, tables=real, existing=set(), poll_run_id="r")
    assert all(t["market"] == "spread" for t in got)               # an ordinary CFB line is inside the table (it may or may not flag)
    assert real[("cfb", "spread")].pmf(3.5) is not None and real[("cfb", "spread")].pmf(38.5) is None


def test_a_junk_price_is_skipped_never_a_crash_and_never_a_ticket():
    # a book offer priced 0 (and one at -99) next to a valid one: the valid one is still considered and wins
    junk = good_rows() + [row(1, "total", "over", "fanduel", 53.5, 0, KICK), row(1, "total", "under", "fanduel", 53.5, -110, KICK),
                          row(1, "total", "over", "betmgm", 53.5, -99, KICK), row(1, "total", "under", "betmgm", 53.5, 99, KICK)]
    (t,) = poll(junk)
    assert (t["book"], t["book_line"], t["price"]) == ("draftkings", 54.5, -110)
    # junk alone makes no ticket
    only_junk = total_rows(1, KICK, books={"fanduel": (54.5, 0, -110), "betmgm": (54.5, 50, -50)})
    assert poll(only_junk) == []
    # a junk Pinnacle side leaves no usable two-way: no ticket, no crash
    bad_pin = [dict(r, price=0) if (r["book"] == "pinnacle" and r["side"] == "under") else r for r in good_rows()]
    assert poll(bad_pin) == []
    # a missing or non-numeric price is skipped the same way
    odd = good_rows() + [row(1, "total", "over", "fanduel", 53.5, None, KICK), row(1, "total", "over", "betmgm", 53.5, "n/a", KICK)]
    assert [t["book"] for t in poll(odd)] == ["draftkings"]
