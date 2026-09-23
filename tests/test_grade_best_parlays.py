"""Tests for scripts/grade_best_parlays.py.

Tests the pure `grade_ticket` function for pending, loss, win, and void scenarios.
"""
import importlib.util
from pathlib import Path
from decimal import Decimal

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
    """When one leg loses and others are pending, result is a loss row."""
    legs = [
        _game_leg(123, "moneyline", "home"),  # Will be a win
        _game_leg(124, "moneyline", "home"),  # Will be a loss
    ]
    ticket = _ticket(legs)

    # Fixture: game 123 wins (margin favors home), game 124 loses (margin favors away)
    finals = {
        123: {"actual_margin": 5.0, "actual_total": None},    # Home wins by 5
        124: {"actual_margin": -3.0, "actual_total": None},   # Away wins by 3
    }

    result = grade_best_parlays.grade_ticket(ticket, finals, {}, set())

    assert result is not None
    assert result["result"] == "loss"
    assert result["pnl"] == -10.0
    assert result["payout_dec"] is None


def test_grade_ticket_all_legs_win():
    """When all game legs win, result is a win row with correct payout."""
    legs = [
        _game_leg(123, "moneyline", "home", price=-110),    # 1.909 dec
        _game_leg(124, "moneyline", "away", price=-110),    # 1.909 dec
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
    # 1.909 * 1.909 = 3.6442... => pnl = 10 * (3.6442 - 1) = 26.442
    assert result["pnl"] > 0
    assert result["payout_dec"] is not None
    assert result["payout_dec"] > 1


def test_grade_ticket_prop_void_captured_no_actual():
    """When a prop's game is captured but has no actual, the leg is a push and
    dropped from the payout calculation."""
    legs = [
        _game_leg(123, "moneyline", "home", price=-110),
        _prop_leg(123, 789, "rec_yds", "over", 45.5, price=-110),  # Will be void (push)
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
    # One win leg, one push leg -> payout is just the winning leg (1.909 dec)
    assert result["result"] == "win"
    assert result["payout_dec"] is not None


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
    import json
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
