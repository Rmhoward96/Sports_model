"""Ledger grading: first / last basis rows, results, units, close reference, CLV. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel import db
from sportsmodel.ledger import grading

KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)


def lrow(pick_key="k1", id=1, hours_before=48, **kw):
    base = dict(id=id, pick_key=pick_key, source="ev_game", sport="nfl", game_pk=7, market="spread", side="home",
                line=-3.5, player_id=None, player_name=None, best_price=-105, observed_at=KICK - timedelta(hours=hours_before),
                commence_time=KICK, minutes_to_kick=hours_before * 60.0)
    return {**base, **kw}


def snap(market, side, book, price, line=None, player="", at=KICK - timedelta(minutes=18)):
    return {"game_pk": 7, "market": market, "side": side, "book": book, "price": price, "line": line,
            "player_name": player, "captured_at": at}


PIN_SPREAD = [snap("spread", "home", "pinnacle", -110, -3.5), snap("spread", "away", "pinnacle", -110, 3.5)]
FINAL_HOME_BY_7 = {7: {"actual_margin": 7.0, "actual_total": 44.0}}


def close_of(rows):
    return grading.merge_close(rows, [])


# ---- first / last basis -----------------------------------------------------------------------------------

def test_first_and_last_are_the_earliest_and_latest_pre_kick_rows():
    rows = [lrow(id=2, hours_before=20), lrow(id=1, hours_before=48), lrow(id=3, hours_before=2)]
    b = grading.basis_rows(rows)
    assert b["first"]["id"] == 1 and b["last"]["id"] == 3


def test_a_single_row_is_both_first_and_last():
    b = grading.basis_rows([lrow()])
    assert b["first"] is b["last"]


def test_rows_at_or_after_kickoff_are_never_bettable():
    late = lrow(id=9, hours_before=-1)
    assert grading.basis_rows([late]) == {}
    b = grading.basis_rows([lrow(id=1, hours_before=5), late])
    assert b["last"]["id"] == 1


# ---- grading ---------------------------------------------------------------------------------------------

def test_game_pick_result_units_and_clv_against_pinnacle():
    (r1,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD)) if g["basis"] == "first"]
    assert r1["result"] == "W" and r1["units_at_pick"] == pytest.approx(100 / 105)
    assert r1["close_kind"] == "pinnacle_novig" and r1["close_fair_prob"] == pytest.approx(0.5)
    assert r1["clv_prob"] == pytest.approx(0.5 - 105 / 205) and r1["clv_pts"] == 0.0
    assert r1["beat_close"] is False and r1["close_same_line"] is True and r1["close_books"] == 1
    assert r1["close_minutes_to_kick"] == pytest.approx(18) and r1["ledger_id"] == 1
    assert set(r1) == set(db.LEDGER_COLUMNS["pick_ledger_results"])


def test_both_bases_are_written_and_can_differ_by_line():
    rows = [lrow(id=1, hours_before=60, line=-3.5, best_price=-110), lrow(id=2, hours_before=3, line=-7.5, best_price=-110)]
    out = {g["basis"]: g for g in grading.grade_rows(rows, FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD))}
    assert out["first"]["result"] == "W" and out["last"]["result"] == "L"          # won by 7: -3.5 covers, -7.5 does not
    assert out["first"]["clv_pts"] == 0.0 and out["last"]["clv_pts"] == pytest.approx(-4.0)
    assert out["last"]["close_same_line"] is False and out["last"]["clv_prob"] is None
    assert out["last"]["beat_close"] is False


def test_line_move_in_the_bettors_favor_beats_the_close_on_points():
    close = close_of([snap("spread", "home", "pinnacle", -110, -4.5), snap("spread", "away", "pinnacle", -110, 4.5)])
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, close) if g["basis"] == "first"]
    assert g["close_kind"] == "pinnacle_novig" and g["close_fair_prob"] is None
    assert g["clv_pts"] == pytest.approx(1.0) and g["beat_close"] is True and g["clv_prob"] is None


def test_missing_close_is_recorded_not_dropped():
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, {}) if g["basis"] == "first"]
    assert g["close_kind"] == "missing" and g["clv_prob"] is None and g["clv_pts"] is None
    assert g["beat_close"] is None and g["close_minutes_to_kick"] is None and g["result"] == "W"


def test_ungradable_picks_are_left_pending():
    assert grading.grade_rows([lrow()], {}, {}, close_of(PIN_SPREAD)) == []                   # no final yet
    prop = lrow(source="ev_prop", market="rec_yds", side="over", line=64.5, player_id="00-1", player_name="A. Brown")
    assert grading.grade_rows([prop], FINAL_HOME_BY_7, {}, {}) == []                          # no player actual yet


def test_model_game_picks_are_graded_at_flat_minus_110_without_a_price():
    row = lrow(source="model_game", best_price=None, market="spread")
    (g,) = [g for g in grading.grade_rows([row], FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD)) if g["basis"] == "first"]
    assert g["units_at_pick"] == pytest.approx(100 / 110) and g["best_price"] is None and g["clv_prob"] is None
    ml = lrow(source="model_game", best_price=None, market="moneyline", line=None)
    (g,) = [g for g in grading.grade_rows([ml], FINAL_HOME_BY_7, {}, {}) if g["basis"] == "first"]
    assert g["result"] == "W" and g["units_at_pick"] is None                                  # no price, no units


def test_prop_pick_uses_player_actuals_and_the_props_close():
    prop = lrow(source="ev_prop", market="rec_yds", side="over", line=64.5, player_id="00-1", player_name="A. Brown",
                best_price=-105)
    rows = [snap("reception_yds", "over", "pinnacle", -120, 64.5, "A. Brown"),
            snap("reception_yds", "under", "pinnacle", 100, 64.5, "A. Brown")]
    close = grading.merge_close(rows, [])
    out = grading.grade_rows([prop], {}, {(7, "00-1", "rec_yds"): 80.0}, close)
    g = [g for g in out if g["basis"] == "first"][0]
    assert g["result"] == "W" and g["close_kind"] == "pinnacle_novig" and g["close_same_line"] is True
    assert g["clv_prob"] == pytest.approx(grading.ref.novig(-120, 100) - 105 / 205)
    push = grading.grade_rows([prop], {}, {(7, "00-1", "rec_yds"): 64.5}, close)
    assert push[0]["result"] == "P" and push[0]["units_at_pick"] == 0.0


def test_props_and_lines_closes_are_kept_apart_per_game():
    both = PIN_SPREAD + [snap("reception_yds", "over", "pinnacle", -110, 64.5, "A. Brown")]
    close = grading.merge_close(both, [])
    assert set(close) == {(7, False), (7, True)}


def test_capture_rows_beat_the_stale_fallback_per_game_and_kind():
    old = KICK - timedelta(minutes=170)
    fb = [snap("spread", "home", "pinnacle", -130, -3.5, at=old), snap("spread", "away", "pinnacle", 110, 3.5, at=old)]
    merged = grading.merge_close(PIN_SPREAD, fb)
    assert merged[(7, False)]["captured_at"] == KICK - timedelta(minutes=18) and len(merged[(7, False)]["rows"]) == 2
    only_fb = grading.merge_close([], fb)
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, only_fb) if g["basis"] == "first"]
    assert g["close_minutes_to_kick"] == pytest.approx(170)          # the gate reads this: an old close is visible as one


# ---- rulings: same_line gates CLV, graded at the row's own line, model moneyline has no CLV ---------------

def test_consensus_close_with_a_moved_pinnacle_line_does_not_compute_clv_prob():
    # Pinnacle quotes -4.5 (moved); four US books price the pick's own -3.5. fair_prob exists at -3.5, but the close
    # line moved, so same_line is False and clv_prob must stay NULL (the move is measured in points instead).
    rows = [snap("spread", "home", "pinnacle", -110, -4.5), snap("spread", "away", "pinnacle", -110, 4.5)]
    for b in ("draftkings", "fanduel", "betmgm", "caesars"):
        rows += [snap("spread", "home", b, -110, -3.5), snap("spread", "away", b, -110, 3.5)]
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, close_of(rows)) if g["basis"] == "first"]
    assert g["close_kind"] == "us_consensus_novig" and g["close_same_line"] is False
    assert g["close_fair_prob"] == pytest.approx(0.5) and g["clv_prob"] is None
    assert g["clv_pts"] == pytest.approx(1.0) and g["beat_close"] is True


def test_graded_at_the_ledger_rows_own_line_not_the_close_line():
    close = close_of([snap("spread", "home", "pinnacle", -110, -9.5), snap("spread", "away", "pinnacle", -110, 9.5)])
    (g,) = [g for g in grading.grade_rows([lrow(line=-3.5)], FINAL_HOME_BY_7, {}, close) if g["basis"] == "first"]
    assert g["result"] == "W" and g["line"] == -3.5


def test_model_moneyline_has_no_price_clv_or_beat_close():
    ml = lrow(source="model_game", best_price=None, market="moneyline", line=None)
    close = close_of([snap("moneyline", "home", "pinnacle", -150), snap("moneyline", "away", "pinnacle", 130)])
    (g,) = [g for g in grading.grade_rows([ml], FINAL_HOME_BY_7, {}, close) if g["basis"] == "first"]
    assert g["close_kind"] == "pinnacle_novig" and g["close_fair_prob"] is not None
    assert g["units_at_pick"] is None and g["clv_prob"] is None and g["beat_close"] is None


def test_model_game_clv_never_uses_a_stored_price():
    row = lrow(source="model_game", best_price=-105, market="spread")           # even if a price were present
    (g,) = [g for g in grading.grade_rows([row], FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD)) if g["basis"] == "first"]
    assert g["clv_prob"] is None and g["units_at_pick"] == pytest.approx(100 / 110)


def test_postponed_or_unfinished_game_is_skipped_not_graded():
    assert grading.grade_rows([lrow()], {7: {"actual_margin": None, "actual_total": None}}, {}, close_of(PIN_SPREAD)) == []
