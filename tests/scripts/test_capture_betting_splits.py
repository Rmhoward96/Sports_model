"""Tests for the pure seams in capture_betting_splits.py (slate_index +
splits_enabled). The actor/DB IO in main() is thin and not unit-tested."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "capture_betting_splits.py"
_spec = importlib.util.spec_from_file_location("capture_betting_splits", _p)
cap = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cap)


def _game(pk, home, away, when):
    return {"game_pk": pk, "home_team": home, "away_team": away, "commence_time": when}


def test_slate_index_keys_on_team_and_utc_date():
    games = [
        _game(401772000, "LA", "NYG", "2026-09-22T00:15Z"),
        _game(401772001, "KC", "BUF", "2026-09-21T20:25Z"),
    ]
    idx = cap.slate_index(games)
    assert idx[("LA", "NYG", "2026-09-22")] == 401772000
    assert idx[("KC", "BUF", "2026-09-21")] == 401772001


def test_slate_index_skips_games_missing_pk_or_time():
    games = [
        _game(None, "LA", "NYG", "2026-09-22T00:15Z"),
        _game(401772001, "KC", "BUF", None),
        _game(401772002, "SF", "SEA", "2026-09-21T20:25Z"),
    ]
    idx = cap.slate_index(games)
    assert idx == {("SF", "SEA", "2026-09-21"): 401772002}


def test_splits_enabled_requires_flag_token_and_window():
    base = {"INGEST_SPLITS": "true", "APIFY_TOKEN": "abc"}
    assert cap.splits_enabled(base, 180) is True
    assert cap.splits_enabled(base, 0) is False                       # window must be positive
    assert cap.splits_enabled({"INGEST_SPLITS": "true"}, 180) is False  # no token
    assert cap.splits_enabled({"APIFY_TOKEN": "abc"}, 180) is False     # flag off (default)
    assert cap.splits_enabled({}, 180) is False
