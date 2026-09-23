"""Tests for scripts/grade_best_parlays.py.

Tests the pure `grade_ticket` function for pending, loss, win, void, and prop scenarios.
"""
import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "grade_best_parlays.py"
_spec = importlib.util.spec_from_file_location("grade_best_parlays", _SCRIPT_PATH)
grade_best_parlays = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grade_best_parlays)


def _ticket(legs):
    """Helper to create a test ticket with the given legs."""
    return {
        "parlay_id": "dk|g:123:moneyline:home|p:456:789:rec_yds:over",
        "sport": "nfl",
        "book": "dk",
        "parlay_price": 250,
        "first_commence": "2025-01-01T20:00:00Z",
        "legs": legs,
    }


def _game_leg(game_pk, market, side, line=None, price=-110):
    """Helper to create a game leg."""
    return {
        "key": f"g:{game_pk}:{market}:{side}",
        "kind": "game",
        "sport": "nfl",
        "game_pk": game_pk,
        "market": market,
        "side": side,
        "line": line,
        "prob": 0.55,
        "label": f"Team {side}",
        "matchup": "NYY @ BOS",
        "commence_time": "2025-01-01T20:00:00Z",
        "player_id": None,
        "player_name": None,
        "price": price,
    }


def _prop_leg(game_pk, player_id, market, side, line, prob=0.60, price=-110):
    """Helper to create a prop leg."""
    return {
        "key": f"p:{game_pk}:{player_id}:{market}:{side}",
        "kind": "prop",
        "sport": "nfl",
        "game_pk": game_pk,
        "market": market,
        "side": side,
        "line": line,
        "prob": prob,
        "label": f"Player {market} {side.upper()} {line}",
        "matchup": "NYY @ BOS",
        "commence_time": "2025-01-01T20:00:00Z",
        "player_id": player_id,
        "player_name": "Test Player",
        "price": price,
    }


def test_grade_ticket_pending_returns_none():
    """When all legs are pending (no finals/actuals), grade_ticket returns None."""
    legs = [
        _game_leg(123, "moneyline", "home"),
        _prop_leg(456, 789, "rec_yds", "over", 45.5),
    ]
    ticket = _ticket(legs)

    # No finals, no actuals, no captured games
    result = grade_best_parlays.grade_ticket(ticket, {}, {}, set())
    assert result is None


def test_grade_ticket_one_leg_lost_while_others_pending():
    """When one leg loses and others are pending, result is a loss row immediately
    (loss takes precedence). The pending leg's result is None in the returned ticket."""
    legs = [
        _game_leg(123, "moneyline", "home"),  # Will be a loss
        _game_leg(124, "moneyline", "home"),  # Will be pending (no final)
    ]
    ticket = _ticket(legs)

    # Only game 123 is graded (loses); game 124 is pending
    finals = {
        123: {"actual_margin": -5.0, "actual_total": None},   # Away wins by 5 (home loses)
    }

    result = grade_best_parlays.grade_ticket(ticket, finals, {}, set())

    assert result is not None
    assert result["result"] == "loss"
    assert result["pnl"] == -10.0
    assert result["payout_dec"] is None
    # Loss leg should have result="loss", pending leg should have result=None
    assert result["legs"][0]["result"] == "loss"
    assert result["legs"][1]["result"] is None


def test_grade_ticket_all_legs_win():
    """When all game legs win, result is a win row with exact payout calculation.
    2 legs at -110 each: 1.909090... decimal, product = 3.6446..., pnl = 10 * 2.6446..."""
    legs = [
        _game_leg(123, "moneyline", "home", price=-110),
        _game_leg(124, "moneyline", "away", price=-110),
    ]
    ticket = _ticket(legs)

    # Both games win for the picked sides
    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},   # Home wins
        124: {"actual_margin": -3.0, "actual_total": None},  # Away wins
    }

    result = grade_best_parlays.grade_ticket(ticket, finals, {}, set())

    assert result is not None
    assert result["result"] == "win"
    # -110 odds = 1.909090... decimal; product = 3.644628...; pnl = 10 * 2.644628...
    expected_pnl = 10 * (1.909090909 * 1.909090909 - 1)
    assert result["pnl"] == pytest.approx(expected_pnl, rel=0.001)
    assert result["payout_dec"] is not None
    assert result["payout_dec"] == pytest.approx(1.909090909 * 1.909090909, rel=0.001)


def test_grade_ticket_prop_void_captured_no_actual():
    """When a prop's game is captured but has no actual, the leg is a push and
    correctly dropped from payout calculation: only the winning leg (1.909 dec) is used."""
    legs = [
        _game_leg(123, "moneyline", "home", price=-110),       # Win leg: 1.909 dec
        _prop_leg(123, "789", "rec_yds", "over", 45.5, price=-110),  # Void (push)
    ]
    ticket = _ticket(legs)

    # Game leg wins, prop game is captured but no actual for player
    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},
    }
    actuals = {}  # No actual for the prop
    captured_games = {123}

    result = grade_best_parlays.grade_ticket(ticket, finals, actuals, captured_games)

    assert result is not None
    assert result["result"] == "win"
    # One win leg (1.909 dec), one push leg (dropped) => payout = 1.909
    # pnl = 10 * (1.909 - 1) = 9.09
    assert result["payout_dec"] == pytest.approx(1.909090909, rel=0.001)
    assert result["pnl"] == pytest.approx(9.090909, rel=0.001)
    assert result["legs"][0]["result"] == "win"
    assert result["legs"][1]["result"] == "push"


def test_grade_ticket_all_pushes():
    """When all legs push, result is a push with pnl=0 and payout_dec=1."""
    legs = [
        _game_leg(123, "spread", "home", line=-3.0, price=-110),
        _game_leg(124, "spread", "away", line=3.0, price=-110),
    ]
    ticket = _ticket(legs)

    # Both spreads push:
    # Game 123: home -3.0, actual_margin=3.0 => side_margin=3.0 => 3.0 + (-3.0) = 0 (push)
    # Game 124: away +3.0, actual_margin=3.0 => side_margin=-3.0 => -3.0 + 3.0 = 0 (push)
    finals = {
        123: {"actual_margin": 3.0, "actual_total": None},
        124: {"actual_margin": 3.0, "actual_total": None},
    }

    result = grade_best_parlays.grade_ticket(ticket, finals, {}, set())

    assert result is not None
    assert result["result"] == "push"
    assert result["pnl"] == 0.0
    assert result["payout_dec"] == 1.0


def test_grade_ticket_legs_json_string():
    """When legs is a JSON string (from DB), it is decoded before grading."""
    legs = [
        _game_leg(123, "moneyline", "home"),
    ]
    ticket = _ticket(legs)
    # Simulate DB returning legs as a JSON string
    ticket["legs"] = json.dumps(legs)

    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},
    }

    result = grade_best_parlays.grade_ticket(ticket, finals, {}, set())

    assert result is not None
    assert result["result"] == "win"


def test_grade_ticket_result_includes_legs_with_results():
    """Returned ticket has legs array with per-leg result attached."""
    legs = [
        _game_leg(123, "moneyline", "home"),
        _game_leg(124, "moneyline", "away"),
    ]
    ticket = _ticket(legs)

    # Game 123: home, margin=5.0 => side_margin=5.0 => win
    # Game 124: away, margin=5.0 => side_margin=-5.0 => loss
    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},
        124: {"actual_margin": 5.0, "actual_total": None},
    }

    result = grade_best_parlays.grade_ticket(ticket, finals, {}, set())

    assert result is not None
    assert len(result["legs"]) == 2
    assert result["legs"][0]["result"] == "win"
    assert result["legs"][1]["result"] == "loss"


def test_grade_ticket_prop_with_actual_over_wins():
    """Prop leg settles via actual lookup: player actual 60.0 vs line 45.5 over wins."""
    legs = [
        _game_leg(123, "moneyline", "home", price=-110),
        _prop_leg(123, "789", "rec_yds", "over", 45.5, price=-110),
    ]
    ticket = _ticket(legs)

    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},
    }
    # player_id is TEXT (string) in the DB
    actuals = {(123, "789", "rec_yds"): 60.0}  # Actual 60 > line 45.5

    result = grade_best_parlays.grade_ticket(ticket, finals, actuals, set())

    assert result is not None
    assert result["result"] == "win"
    # Both legs win: 1.909 * 1.909 parlay
    assert result["payout_dec"] == pytest.approx(1.909090909 * 1.909090909, rel=0.001)
    assert result["legs"][1]["result"] == "win"


def test_grade_ticket_prop_with_actual_under_loses():
    """Prop leg settles via actual lookup: player actual 50.0 vs line 45.5 under loses."""
    legs = [
        _game_leg(123, "moneyline", "home", price=-110),
        _prop_leg(123, "789", "rec_yds", "under", 45.5, price=-110),
    ]
    ticket = _ticket(legs)

    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},
    }
    # Actual 50 > line 45.5, so under loses
    actuals = {(123, "789", "rec_yds"): 50.0}

    result = grade_best_parlays.grade_ticket(ticket, finals, actuals, set())

    assert result is not None
    assert result["result"] == "loss"
    assert result["pnl"] == -10.0
    assert result["legs"][1]["result"] == "loss"
