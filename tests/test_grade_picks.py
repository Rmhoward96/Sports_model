import importlib.util
import pathlib

import pytest

from sportsmodel import db

_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "grade_results.py"
_s = importlib.util.spec_from_file_location("grade_results", _p)
gr = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gr)


def test_board_and_picks_helpers_exist():
    for fn in ("upsert_board_picks", "insert_new_picks", "update_graded_picks"):
        assert hasattr(db, fn), f"db.{fn} missing"


def test_grade_pick_total_under_win_profit_and_clv():
    pick = {"game_pk": 1, "market": "total", "player_id": 0, "side": "under", "line": 8.5,
            "bet_odds": -105, "novig_bet": 0.50}
    out = gr.grade_pick(pick, actual=7.0, novig_close=0.54)  # total 7 -> under wins
    assert out["result"] == "win"
    assert out["profit"] == pytest.approx(gr._decimal(-105) - 1)
    assert out["clv"] == pytest.approx(0.04)   # 0.54 - 0.50


def test_grade_pick_moneyline_loss_is_minus_one_unit():
    pick = {"game_pk": 1, "market": "moneyline", "player_id": 0, "side": "home", "line": None,
            "bet_odds": 120, "novig_bet": 0.45}
    out = gr.grade_pick(pick, actual=-1.0, novig_close=0.47)  # margin<0 -> home lost
    assert out["result"] == "loss" and out["profit"] == -1.0
    assert out["clv"] == pytest.approx(0.02)


def test_grade_pick_spread_away_covers():
    # away +1.5 (line=+1.5); actual margin (home-away) = -3 -> away won by 3 -> covers
    pick = {"game_pk": 1, "market": "spread", "player_id": 0, "side": "away", "line": 1.5,
            "bet_odds": -140, "novig_bet": 0.60}
    out = gr.grade_pick(pick, actual=-3.0, novig_close=0.60)
    assert out["result"] == "win"


def test_grade_pick_total_push():
    pick = {"game_pk": 1, "market": "total", "player_id": 0, "side": "over", "line": 9.0,
            "bet_odds": -110, "novig_bet": 0.5}
    out = gr.grade_pick(pick, actual=9.0, novig_close=0.5)
    assert out["result"] == "push" and out["profit"] == 0.0
