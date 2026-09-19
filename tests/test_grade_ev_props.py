"""Unit tests for `sportsmodel.serving.props_ev.grade_prop_pick`'s pure
forward-CLV grading of NFL player-prop picks.

Mirrors tests/test_grade_ev.py's structure for the game-level pilot, but for
one `ev_prop_picks` row -> `ev_prop_results` row. No network, no DB --
grade_prop_pick is a pure function of (pick dict, actual float, closing
price int|None).
"""
from sportsmodel.serving.board import decimal_odds
from sportsmodel.serving.props_ev import grade_prop_pick


def _pick(**overrides) -> dict:
    base = {
        "sport": "nfl",
        "game_pk": 401671789,
        "player_id": "00-0034796",
        "player_name": "A.J. Brown",
        "market": "rec_yds",
        "side": "over",
        "line": 65.5,
        "model_version": "props-sim-v1",
        "commence_time": "2026-09-14T17:00:00+00:00",
        "model_prob": 0.55,
        "best_price": -110,
        "pinnacle_price": -120,
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------
# result / profit
# --------------------------------------------------------------------------

def test_over_pick_hits_line_wins_and_profits_at_best_price():
    row = grade_prop_pick(_pick(side="over", line=65.5, best_price=-110), actual=80.0,
                           closing_price=-150)
    assert row["result"] == "win"
    assert row["profit"] == decimal_odds(-110) - 1


def test_under_pick_hits_line_wins():
    row = grade_prop_pick(_pick(side="under", line=65.5, best_price=+100), actual=40.0,
                           closing_price=-150)
    assert row["result"] == "win"
    assert row["profit"] == decimal_odds(100) - 1


def test_actual_equals_line_is_push_with_zero_profit():
    row = grade_prop_pick(_pick(side="over", line=65.5, best_price=-110), actual=65.5,
                           closing_price=-150)
    assert row["result"] == "push"
    assert row["profit"] == 0.0


def test_over_pick_misses_line_loses_with_unit_loss():
    row = grade_prop_pick(_pick(side="over", line=65.5, best_price=-110), actual=40.0,
                           closing_price=-150)
    assert row["result"] == "loss"
    assert row["profit"] == -1.0


def test_under_pick_misses_line_loses():
    row = grade_prop_pick(_pick(side="under", line=65.5, best_price=-110), actual=80.0,
                           closing_price=-150)
    assert row["result"] == "loss"
    assert row["profit"] == -1.0


def test_result_and_profit_none_when_actual_missing():
    row = grade_prop_pick(_pick(), actual=None, closing_price=-150)
    assert row["result"] is None
    assert row["profit"] is None


def test_result_and_profit_none_when_line_missing():
    row = grade_prop_pick(_pick(line=None), actual=80.0, closing_price=-150)
    assert row["result"] is None
    assert row["profit"] is None


# --------------------------------------------------------------------------
# clv -- signed closing-line value from the pick-time Pinnacle price
# --------------------------------------------------------------------------

def test_clv_positive_when_closing_implied_prob_higher_than_pick_time():
    # took -120 (implied .5455), closed -150 (implied .6) -> market moved
    # toward this side -> pick-time price was better -> positive clv.
    row = grade_prop_pick(_pick(pinnacle_price=-120), actual=80.0, closing_price=-150)
    assert row["clv"] is not None
    assert row["clv"] > 0


def test_clv_negative_when_closing_implied_prob_lower_than_pick_time():
    # took -150 (implied .6), closed -120 (implied .5455) -> market moved
    # away from this side -> negative clv.
    row = grade_prop_pick(_pick(pinnacle_price=-150), actual=80.0, closing_price=-120)
    assert row["clv"] is not None
    assert row["clv"] < 0


def test_clv_none_when_closing_price_missing():
    row = grade_prop_pick(_pick(pinnacle_price=-120), actual=80.0, closing_price=None)
    assert row["clv"] is None
    # result/profit still computed even though CLV falls back to None.
    assert row["result"] == "win"
    assert row["profit"] == decimal_odds(-110) - 1


def test_novig_close_is_none_when_closing_price_missing():
    row = grade_prop_pick(_pick(), actual=80.0, closing_price=None)
    assert row["novig_close"] is None


def test_novig_close_set_when_closing_price_present():
    row = grade_prop_pick(_pick(), actual=80.0, closing_price=-150)
    assert row["novig_close"] is not None


# --------------------------------------------------------------------------
# row shape -- keys match db._EV_PROP_RESULTS_COLS
# --------------------------------------------------------------------------

def test_row_keys_match_ev_prop_results_columns():
    row = grade_prop_pick(_pick(), actual=80.0, closing_price=-150)
    expected = {
        "sport", "game_pk", "player_id", "player_name", "market", "side", "line",
        "model_version", "commence_time", "model_prob", "novig_close", "actual",
        "result", "clv", "profit",
    }
    assert set(row.keys()) == expected
