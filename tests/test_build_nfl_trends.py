"""Light pure test for scripts/build_nfl_trends.py.

Tests build_upcoming (PURE function): predicted pk kept, market_spread used,
NaN spread fallback flips sign, non-predicted/NaN espn dropped.
"""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_nfl_trends.py"
_spec = importlib.util.spec_from_file_location("build_nfl_trends", _SCRIPT_PATH)
build_nfl_trends = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_nfl_trends)


def _sched_row(espn, home_team, away_team, gameday, gametime, spread_line):
    """Helper to build a schedule row (as namedtuple from itertuples)."""
    from collections import namedtuple
    Row = namedtuple("Row", ["espn", "home_team", "away_team", "gameday", "gametime", "spread_line"])
    return Row(espn, home_team, away_team, gameday, gametime, spread_line)


class FakeSched:
    """Fake schedule DataFrame-like object."""
    def __init__(self, rows):
        self.rows = rows

    def itertuples(self, index=False):
        return iter(self.rows)


def test_build_upcoming_predicted_game_pk_kept():
    """Predicted game_pk is kept and included in output."""
    preds = {123: -5.5}
    sched = FakeSched([
        _sched_row(123, "BUF", "KC", "2026-01-15", "20:30", 5.5)
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert len(result) == 1
    assert result[0]["game_pk"] == 123
    assert result[0]["home_team"] == "BUF"
    assert result[0]["away_team"] == "KC"
    assert result[0]["gameday"] == "2026-01-15"
    assert result[0]["gametime"] == "20:30"


def test_build_upcoming_market_spread_used():
    """market_spread (from preds) is used directly."""
    preds = {456: -3.0}
    sched = FakeSched([
        _sched_row(456, "LAR", "SF", "2026-01-20", "18:00", 4.0)
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert result[0]["home_line"] == -3.0


def test_build_upcoming_nan_spread_fallback_flips_sign():
    """When market_spread is None, use -(schedule spread_line)."""
    preds = {789: None}
    sched = FakeSched([
        _sched_row(789, "DAL", "PHI", "2026-01-22", "19:00", 2.5)
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    # schedule spread_line is 2.5 (+ = home favored), nflverse convention
    # -> home_line = -(2.5) = -2.5
    assert result[0]["home_line"] == -2.5


def test_build_upcoming_nan_spread_line_fallback_does_not_apply():
    """When both market_spread and schedule spread_line are NaN, home_line is None."""
    preds = {111: None}
    sched = FakeSched([
        _sched_row(111, "BUF", "KC", "2026-01-25", "16:00", float("nan"))
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert result[0]["home_line"] is None


def test_build_upcoming_non_predicted_dropped():
    """Games not in preds dict are dropped."""
    preds = {123: -5.5}
    sched = FakeSched([
        _sched_row(123, "BUF", "KC", "2026-01-15", "20:30", 5.5),
        _sched_row(456, "LAR", "SF", "2026-01-20", "18:00", 4.0),  # not predicted
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert len(result) == 1
    assert result[0]["game_pk"] == 123


def test_build_upcoming_nan_espn_dropped():
    """Rows with non-numeric espn are dropped."""
    preds = {123: -5.5, 789: None}
    sched = FakeSched([
        _sched_row(123, "BUF", "KC", "2026-01-15", "20:30", 5.5),
        _sched_row(None, "DAL", "PHI", "2026-01-22", "19:00", 2.5),
        _sched_row("not_a_number", "LAR", "SF", "2026-01-20", "18:00", 4.0),
        _sched_row(789, "TB", "NO", "2026-01-28", "13:00", 1.5),
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert len(result) == 2
    pks = {r["game_pk"] for r in result}
    assert pks == {123, 789}


def test_build_upcoming_gametime_nan_becomes_empty_string():
    """When gametime is NaN, it becomes empty string (not 'nan')."""
    preds = {222: -4.0}
    sched = FakeSched([
        _sched_row(222, "BUF", "KC", "2026-01-15", float("nan"), 4.0)
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert result[0]["gametime"] == ""


def test_build_upcoming_multiple_mixed():
    """Mixed scenario: predicted and non-predicted, valid and NaN espn, with and without market spread."""
    preds = {100: -1.5, 200: None, 300: -2.5}
    sched = FakeSched([
        _sched_row(100, "A", "B", "2026-01-01", "12:00", 1.0),
        _sched_row(200, "C", "D", "2026-01-02", "14:00", 2.0),
        _sched_row(400, "E", "F", "2026-01-03", "16:00", 3.0),  # not predicted
        _sched_row(None, "G", "H", "2026-01-04", "18:00", 4.0),  # NaN espn
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert len(result) == 2

    game_100 = [r for r in result if r["game_pk"] == 100][0]
    assert game_100["home_line"] == -1.5

    game_200 = [r for r in result if r["game_pk"] == 200][0]
    assert game_200["home_line"] == -2.0  # -(2.0) from spread_line


def test_build_upcoming_none_spread_line_fallback_does_not_apply():
    """A None (object-dtype missing) schedule spread_line is also treated as missing."""
    preds = {333: None}
    sched = FakeSched([
        _sched_row(333, "BUF", "KC", "2026-01-25", "16:00", None)
    ])
    result = build_nfl_trends.build_upcoming(sched, preds)
    assert result[0]["home_line"] is None
