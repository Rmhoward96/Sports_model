"""Light pure test for scripts/build_best_parlays.py.

Loads the script module directly (it isn't a package) via importlib.
Tests the pure `ticket_summary` helper and `main` with loaders monkeypatched.
"""
import importlib.util
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

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
    future_time = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    mock_game_pick = {
        "sport": "nfl",
        "game_pk": 123,
        "market": "moneyline",
        "side": "home",
        "matchup": "NYY @ BOS",
        "commence_time": future_time,
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
        "commence_time": future_time,
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
        "commence_time": future_time,
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
        "commence_time": future_time,
        "player_id": 456,
        "player_name": "Aaron Judge",
        "book_prices": {"dk": -110},
    }
    mock_game_odds = []
    mock_prop_odds = []

    # Real ticket from build_best_parlays to verify model_version is added
    mock_ticket = {
        "parlay_id": "dk|g:123:moneyline:home|p:123:456:rec_yds:over",
        "book": "dk",
        "legs": [mock_game_leg, mock_prop_leg],
        "parlay_price": 250,
        "ev": 0.057,
        "sport": "nfl",
        "first_commence": future_time,
    }
    mock_tickets = [mock_ticket]
    mock_locked_keys = frozenset()
    mock_replace = MagicMock(return_value=1)

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
        # Verify replace was called with the ticket list
        mock_replace.assert_called_once()
        called_tickets = mock_replace.call_args[0][0]
        assert len(called_tickets) == 1
        # Verify model_version was added
        assert called_tickets[0]["model_version"] == "ev-parlays-v1"


def test_main_drops_locked_tickets_with_past_first_commence():
    """main() drops tickets whose first_commence is at/before now (game started)."""
    now = datetime.now(timezone.utc)
    past_time = (now - timedelta(hours=1)).isoformat().replace("+00:00", "Z")
    future_time = (now + timedelta(hours=2)).isoformat().replace("+00:00", "Z")

    mock_game_pick = {
        "sport": "nfl",
        "game_pk": 123,
        "market": "moneyline",
        "side": "home",
        "matchup": "NYY @ BOS",
        "commence_time": future_time,
        "true_prob": 0.55,
        "line": None,
    }
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
        "commence_time": future_time,
        "player_id": None,
        "player_name": None,
        "book_prices": {"dk": -110},
    }

    # Ticket with past first_commence (leg kicked off)
    locked_ticket = {
        "parlay_id": "dk|g:123:moneyline:home",
        "book": "dk",
        "legs": [mock_game_leg],
        "parlay_price": -110,
        "ev": 0.01,
        "sport": "nfl",
        "first_commence": past_time,
    }
    # Ticket with future first_commence (good to lock in)
    good_ticket = {
        "parlay_id": "dk|g:123:moneyline:home_2",
        "book": "dk",
        "legs": [mock_game_leg],
        "parlay_price": -110,
        "ev": 0.01,
        "sport": "nfl",
        "first_commence": future_time,
    }
    mock_replace = MagicMock(return_value=1)

    with patch.object(build_best_parlays, "load_game_picks", return_value=[mock_game_pick]), \
         patch.object(build_best_parlays, "load_prop_picks", return_value=[]), \
         patch.object(build_best_parlays, "load_game_odds", return_value=[]), \
         patch.object(build_best_parlays, "load_prop_odds", return_value=[]), \
         patch.object(build_best_parlays, "locked_leg_keys", return_value=frozenset()), \
         patch.object(build_best_parlays, "assemble_game_legs", return_value=[mock_game_leg]), \
         patch.object(build_best_parlays, "assemble_prop_legs", return_value=[]), \
         patch.object(build_best_parlays, "build_best_parlays", return_value=[locked_ticket, good_ticket]), \
         patch.object(build_best_parlays, "replace_unlocked_best_parlays", mock_replace), \
         patch("builtins.print"):
        build_best_parlays.main()
        # Only the good_ticket should be passed to replace
        mock_replace.assert_called_once()
        called_tickets = mock_replace.call_args[0][0]
        assert len(called_tickets) == 1
        assert called_tickets[0] == good_ticket


WINDOW_48H = re.compile(r"captured_at\s*>\s*now\(\)\s*-\s*interval\s*'48 hours'")


def _capture_q(monkeypatch):
    """Monkeypatch get_postgres with a fake conn whose cursor records SQL."""
    sink = []

    class _Cur:
        description = []

        def execute(self, sql, params=None):
            sink.append((sql, params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(build_best_parlays, "get_postgres", lambda: _Conn())
    return sink


def test_load_game_odds_ignores_book_prices_older_than_48h(monkeypatch):
    sink = _capture_q(monkeypatch)
    assert build_best_parlays.load_game_odds([7]) == []
    assert len(sink) == 1
    sql, params = sink[0]
    assert WINDOW_48H.search(sql)
    assert "captured_at <= commence_time" in sql
    assert params == [[7]]


def test_load_prop_odds_ignores_book_prices_older_than_48h(monkeypatch):
    sink = _capture_q(monkeypatch)
    assert build_best_parlays.load_prop_odds([7]) == []
    assert len(sink) == 1
    sql, params = sink[0]
    assert WINDOW_48H.search(sql)
    assert "captured_at <= commence_time" in sql
    assert params[0] == [7]


def test_odds_loaders_empty_game_pks_short_circuit_no_db():
    assert build_best_parlays.load_game_odds([]) == []
    assert build_best_parlays.load_prop_odds([]) == []
