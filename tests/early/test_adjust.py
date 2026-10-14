import pytest

from sportsmodel.early import adjust
from sportsmodel.early.landing import Pmf


def pmf(masses):
    total = sum(masses.values())
    return Pmf(n=1000, window=1.0, probs={k: v / total for k, v in masses.items()})


# NFL-like home-margin table: 10% of the mass sits on the key number 3, 5% on 2 and 4, the rest elsewhere.
NFL_MASSES = {0: 0.04, 1: 0.03, 2: 0.05, 3: 0.10, 4: 0.05, 5: 0.03, 6: 0.04, 7: 0.07, 8: 0.03, -3: 0.10, -7: 0.07,
              10: 0.04, -1: 0.03, -2: 0.05, -4: 0.05, -5: 0.03, -6: 0.04, -8: 0.03, 14: 0.03, -10: 0.04, 20: 0.05}
assert sum(NFL_MASSES.values()) == pytest.approx(1.0)
NFL = pmf(NFL_MASSES)
FAIR = 0.55


def moved(side_fair, d, cp, cb, table=NFL, **kw):
    return adjust.adjust(side_fair, d, cp, cb, table, **kw)


def test_threshold_and_direction_by_market_and_side():
    assert adjust.threshold("spread", "home", -2.5) == 2.5        # home laying 2.5: home must win by 3+, result > 2.5
    assert adjust.threshold("spread", "away", 2.5) == 2.5        # the away side's own line is +2.5
    assert adjust.threshold("total", "over", 54.5) == 54.5 == adjust.threshold("total", "under", 54.5)
    assert adjust.threshold("spread", "home", None) is None and adjust.threshold("moneyline", "home", 1) is None
    assert [adjust.direction("spread", "home"), adjust.direction("spread", "away"),
            adjust.direction("total", "over"), adjust.direction("total", "under")] == [1, -1, 1, -1]
    assert adjust.own_line("spread", "home", 2.5) == -2.5 and adjust.own_line("spread", "away", 2.5) == 2.5
    assert adjust.own_line("total", "under", 54.5) == 54.5


def test_same_line_is_an_identity_on_a_half_point_line():
    assert moved(FAIR, 1, 2.5, 2.5) == pytest.approx((FAIR, 0.0, 1 - FAIR))


def test_same_whole_number_line_keeps_the_table_push_mass_and_the_conditional_fair():
    win, push, loss = moved(FAIR, 1, 3.0, 3.0)
    assert push == pytest.approx(0.10) and win == pytest.approx(FAIR * 0.90) and loss == pytest.approx(0.45 * 0.90)


def test_a_half_point_across_the_key_number_moves_win_probability_by_the_table_mass():
    # Pinnacle home -2.5 (c = 2.5: home wins on margin >= 3); the book hangs -3.5 (needs margin >= 4)
    win, push, loss = moved(FAIR, 1, 2.5, 3.5)
    assert win == pytest.approx(FAIR - 0.10) and loss == pytest.approx(1 - FAIR + 0.10) and push == 0.0


def test_moving_to_a_whole_number_turns_the_crossed_mass_into_a_push():
    win, push, loss = moved(FAIR, 1, 2.5, 3.0)          # -2.5 -> -3: margin 3 stops winning and pushes
    assert (win, push, loss) == pytest.approx((FAIR - 0.10, 0.10, 1 - FAIR))


def test_pinnacle_push_mass_becomes_a_win_when_the_book_line_is_better():
    # Pinnacle -3 (push on 3, fair is conditional on no push); the book offers -2.5: the push becomes a win
    win, push, loss = moved(FAIR, 1, 3.0, 2.5)
    assert win == pytest.approx(FAIR * 0.9 + 0.10) and push == 0.0 and loss == pytest.approx(0.45 * 0.9)


def test_pinnacle_push_mass_becomes_a_loss_when_the_book_line_is_worse():
    win, push, loss = moved(FAIR, 1, 3.0, 3.5)
    assert win == pytest.approx(FAIR * 0.9) and push == 0.0 and loss == pytest.approx(0.45 * 0.9 + 0.10)


def test_the_down_direction_mirrors_the_up_direction():
    # away side at the same thresholds: it wins on margin < c
    win, push, loss = moved(0.45, -1, 3.5, 2.5)         # away +3.5 -> +2.5: margin 3 now beats the away side
    assert win == pytest.approx(0.45 - 0.10) and loss == pytest.approx(0.55 + 0.10)


def test_probabilities_always_sum_to_one():
    for cp, cb in [(2.5, 3.5), (3.0, 2.5), (-3.0, -2.0), (7.0, 6.5), (0.0, 1.5), (3.0, 3.0)]:
        for d in (1, -1):
            assert sum(moved(0.5, d, cp, cb)) == pytest.approx(1.0)


def test_three_point_cap_is_inclusive():
    assert moved(FAIR, 1, 2.5, 5.5) is not None                    # exactly 3.0 points
    assert moved(FAIR, 1, 2.5, 5.5 + 0.5) is None                  # 3.5 points: no ticket
    assert moved(FAIR, 1, 5.5 + 0.5, 2.5) is None
    assert moved(FAIR, 1, 2.5, 6.5, cap=None) is not None          # grading removes the cap


def test_off_grid_lines_are_refused_never_rounded():
    assert moved(FAIR, 1, 2.75, 2.5) is None and moved(FAIR, 1, 2.5, 2.25) is None


def test_a_move_bigger_than_the_available_probability_is_refused_strictly_and_clamped_when_grading():
    wide = pmf({r: 1.0 for r in range(-10, 11)})
    assert moved(0.15, 1, 0.5, 6.5, table=wide) is None           # the table moves 6 / 21 of the mass out of a 0.15 win
    win, push, loss = moved(0.15, 1, 0.5, 6.5, table=wide, strict=False, cap=None)
    assert win == 0.0 and push == 0.0 and loss == pytest.approx(1.0)


def test_ev_formula_with_a_push_contributing_zero():
    assert adjust.ev_of(0.5, 0.5, 100) == pytest.approx(0.0)
    assert adjust.ev_of(0.55, 0.45, -110) == pytest.approx(0.55 * (100 / 110) - 0.45)
    assert adjust.ev_of(0.40, 0.40, 120) == pytest.approx(0.40 * 1.2 - 0.40)       # push mass 0.20 is simply absent


def test_flag_threshold_boundary_is_inclusive():
    assert adjust.is_flag(0.015) and adjust.is_flag(0.0150001) and adjust.is_flag(0.015 - 1e-13)
    assert not adjust.is_flag(0.0149) and not adjust.is_flag(None) and not adjust.is_flag(-0.2)


def test_fair_range_bounds_are_inclusive():
    assert adjust.fair_in_range(0.15) and adjust.fair_in_range(0.85) and adjust.fair_in_range(0.5)
    assert not adjust.fair_in_range(0.1499) and not adjust.fair_in_range(0.8501)


def test_value_side_end_to_end_for_a_home_spread():
    v = adjust.value_side("spread", "home", FAIR, -2.5, -2.5, -105, NFL)
    assert v.p_win == pytest.approx(FAIR) and v.p_push == 0.0
    assert v.ev == pytest.approx(FAIR * (100 / 105) - (1 - FAIR))
    worse = adjust.value_side("spread", "home", FAIR, -2.5, -3.5, -105, NFL)
    assert worse.ev < v.ev and worse.p_win == pytest.approx(FAIR - 0.10)


def test_value_side_for_an_away_spread_uses_the_side_own_lines():
    # away +2.5 at Pinnacle, away +3.5 at the book: the book gives the underdog an extra point (better)
    v = adjust.value_side("spread", "away", 1 - FAIR, 2.5, 3.5, -110, NFL)
    assert v.p_win == pytest.approx(1 - FAIR + 0.10)


def test_value_side_for_totals_under_and_over():
    totals = pmf({50: 0.1, 51: 0.1, 52: 0.1, 53: 0.1, 54: 0.1, 55: 0.1, 56: 0.1, 57: 0.1, 58: 0.1, 59: 0.1})
    over = adjust.value_side("total", "over", 0.5, 54.5, 53.5, -110, totals)        # book total is lower: better Over
    assert over.p_win == pytest.approx(0.5 + 0.1)
    under = adjust.value_side("total", "under", 0.5, 54.5, 55.5, -110, totals)      # book total is higher: better Under
    assert under.p_win == pytest.approx(0.5 + 0.1)


def test_value_side_returns_none_without_a_table_price_or_known_side():
    assert adjust.value_side("spread", "home", FAIR, -2.5, -2.5, -110, None) is None
    assert adjust.value_side("spread", "home", FAIR, -2.5, -2.5, None, NFL) is None
    assert adjust.value_side("spread", "sideways", FAIR, -2.5, -2.5, -110, NFL) is None


def test_sign_convention_end_to_end_with_the_committed_nfl_table():
    from sportsmodel.early.landing import load_tables
    table = load_tables()[("nfl", "spread")]
    # Pinnacle home -3 (expected home margin +3: threshold 3), fair 0.5; the book hangs home -2.5 (easier to cover)
    pmf3 = table.pmf(adjust.threshold("spread", "home", -3.0))
    assert pmf3 is not None and pmf3.mass(3) > 0.05           # key number 3 carries real mass at a -3 line
    better = adjust.value_side("spread", "home", 0.5, -3.0, -2.5, -110, pmf3)
    worse = adjust.value_side("spread", "home", 0.5, -3.0, -3.5, -110, pmf3)
    same = adjust.value_side("spread", "home", 0.5, -3.0, -3.0, -110, pmf3)
    assert better.p_win > same.p_win and worse.p_win == pytest.approx(same.p_win)   # -3.5: the push becomes a loss
    assert worse.p_loss > same.p_loss and better.ev > same.ev > worse.ev
    assert better.p_win == pytest.approx(0.5 * (1 - pmf3.mass(3)) + pmf3.mass(3))
    # the away side of the same game: +3 at Pinnacle is a push on 3; +2.5 turns that push into a loss, +3.5 into a win
    away = [adjust.value_side("spread", "away", 0.5, 3.0, line, -110, pmf3) for line in (2.5, 3.0, 3.5)]
    assert away[0].ev < away[1].ev < away[2].ev
    assert away[0].p_loss > away[1].p_loss and away[2].p_win > away[1].p_win
