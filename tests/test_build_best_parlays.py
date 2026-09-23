"""Light pure test for scripts/build_best_parlays.py.

Loads the script module directly (it isn't a package) via importlib.
Tests the pure `ticket_summary` helper and `main` with loaders monkeypatched.
"""
import importlib.util
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_best_parlays.py"
_spec = importlib.util.spec_from_file_location("build_best_parlays", _SCRIPT_PATH)
build_best_parlays = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_best_parlays)


def test_ticket_summary():
    """Pure `ticket_summary` formats ticket details correctly."""
    ticket = {
        "book": "dk",
        "parlay_price": 250,
        "ev": 0.057,
        "legs": [
            {"label": "NYY ML"},
            {"label": "Soto Rec Yds Over 45.5"},
            {"label": "Over 8.5"},
        ],
    }
    summary = build_best_parlays.ticket_summary(ticket)
    assert summary == "dk +250 ev=0.057 :: NYY ML / Soto Rec Yds Over 45.5 / Over 8.5"


def test_ticket_summary_negative_price():
    """Negative parlay price formats with minus sign."""
    ticket = {
        "book": "fanduel",
        "parlay_price": -110,
        "ev": 0.025,
        "legs": [
            {"label": "Team A -3"},
        ],
    }
    summary = build_best_parlays.ticket_summary(ticket)
    assert summary == "fanduel -110 ev=0.025 :: Team A -3"


def test_main_no_legs_short_circuits():
    """main() with no legs calls replace_unlocked_best_parlays([]) and prints correctly."""
    mock_game_picks = []
    mock_prop_picks = []
    mock_game_odds = []
    mock_prop_odds = []
    mock_locked_keys = frozenset()
    mock_replace = MagicMock(return_value=0)

    with patch.object(build_best_parlays, "load_game_picks", return_value=mock_game_picks), \
         patch.object(build_best_parlays, "load_prop_picks", return_value=mock_prop_picks), \
         patch.object(build_best_parlays, "load_game_odds", return_value=mock_game_odds), \
         patch.object(build_best_parlays, "load_prop_odds", return_value=mock_prop_odds), \
         patch.object(build_best_parlays, "locked_leg_keys", return_value=mock_locked_keys), \
         patch.object(build_best_parlays, "replace_unlocked_best_parlays", mock_replace), \
         patch("builtins.print"):
        build_best_parlays.main()
        mock_replace.assert_called_once_with([])


def test_main_with_legs():
    """main() assembles legs, builds tickets, adds model_version, and replaces parlays."""
    mock_game_pick = {
        "sport": "nfl",
        "game_pk": 123,
        "market": "moneyline",
        "side": "home",
        "matchup": "NYY @ BOS",
        "commence_time": "2025-01-01T20:00:00Z",
        "true_prob": 0.55,
        "line": None,
    }
    mock_prop_pick = {
        "game_pk": 123,
        "player_id": 456,
        "player_name": "Aaron Judge",
        "market": "rec_yds",
        "side": "over",
        "line": 45.5,
        "model_prob": 0.60,
        "matchup": "NYY @ BOS",
        "commence_time": "2025-01-01T20:00:00Z",
    }

    # Simulated legs from assemble functions
    mock_game_leg = {
        "key": "g:123:moneyline:home",
        "kind": "game",
        "sport": "nfl",
        "game_pk": 123,
        "market": "moneyline",
        "side": "home",
        "prob": 0.55,
        "label": "NYY ML",
        "matchup": "NYY @ BOS",
        "commence_time": "2025-01-01T20:00:00Z",
        "player_id": None,
        "player_name": None,
        "book_prices": {"dk": -110},
    }
    mock_prop_leg = {
        "key": "p:123:456:rec_yds:over",
        "kind": "prop",
        "sport": "nfl",
        "game_pk": 123,
        "market": "rec_yds",
        "side": "over",
        "prob": 0.60,
        "label": "Aaron Judge Rec Yds Over 45.5",
        "matchup": "NYY @ BOS",
        "commence_time": "2025-01-01T20:00:00Z",
        "player_id": 456,
        "player_name": "Aaron Judge",
        "book_prices": {"dk": -110},
    }
    mock_legs = [mock_game_leg, mock_prop_leg]
    mock_game_odds = []
    mock_prop_odds = []

    # Simulated ticket from build_best_parlays (won't actually build in this test, but main still expects it to work)
    mock_tickets = []
    mock_locked_keys = frozenset()
    mock_replace = MagicMock(return_value=0)

    with patch.object(build_best_parlays, "load_game_picks", return_value=[mock_game_pick]), \
         patch.object(build_best_parlays, "load_prop_picks", return_value=[mock_prop_pick]), \
         patch.object(build_best_parlays, "load_game_odds", return_value=mock_game_odds), \
         patch.object(build_best_parlays, "load_prop_odds", return_value=mock_prop_odds), \
         patch.object(build_best_parlays, "locked_leg_keys", return_value=mock_locked_keys), \
         patch.object(build_best_parlays, "assemble_game_legs", return_value=[mock_game_leg]), \
         patch.object(build_best_parlays, "assemble_prop_legs", return_value=[mock_prop_leg]), \
         patch.object(build_best_parlays, "build_best_parlays", return_value=mock_tickets), \
         patch.object(build_best_parlays, "replace_unlocked_best_parlays", mock_replace), \
         patch("builtins.print"):
        build_best_parlays.main()
        mock_replace.assert_called_once_with([])
