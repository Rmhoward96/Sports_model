"""Tests for nflverse NFL injury parsing (sportsmodel.nfl.injuries_nflverse).

The desk sources NFL injuries from nflverse (SportsDataIO's NFL injury feed is
scrambled); these cover the PURE parse -- status filtering, latest-week
selection, the note join, and the {player, position, status, note} shape that
must match the CFB adapter's output.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from sportsmodel.nfl.injuries_nflverse import nfl_season, parse_injuries


def _df(rows):
    cols = [
        "season", "week", "team", "full_name", "position",
        "report_status", "report_primary_injury", "report_secondary_injury",
        "practice_status",
    ]
    return pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])


def test_nfl_season_september_is_current_year():
    assert nfl_season(datetime(2026, 9, 10, tzinfo=timezone.utc)) == 2026


def test_nfl_season_january_is_prior_year():
    # January playoffs belong to the prior calendar year's season.
    assert nfl_season(datetime(2027, 1, 15, tzinfo=timezone.utc)) == 2026


def test_nfl_season_march_rolls_to_new_season():
    assert nfl_season(datetime(2026, 3, 1, tzinfo=timezone.utc)) == 2026


def test_empty_df_yields_empty():
    assert parse_injuries(_df([])) == {}
    assert parse_injuries(None) == {}


def test_parses_reported_row_into_desk_shape():
    df = _df([{
        "week": 2, "team": "PHI", "full_name": "A.J. Brown", "position": "WR",
        "report_status": "Questionable", "report_primary_injury": "Hamstring",
        "report_secondary_injury": None, "practice_status": "Limited Participation",
    }])
    out = parse_injuries(df)
    assert out == {"PHI": [{
        "player": "A.J. Brown", "position": "WR", "status": "Questionable",
        "note": "Hamstring - Limited Participation",
    }]}


def test_note_joins_primary_and_secondary_and_practice():
    df = _df([{
        "week": 1, "team": "DAL", "full_name": "Example Player", "position": "CB",
        "report_status": "Out", "report_primary_injury": "Knee",
        "report_secondary_injury": "Ankle", "practice_status": "Did Not Participate In Practice",
    }])
    note = parse_injuries(df)["DAL"][0]["note"]
    assert note == "Knee - Ankle - Did Not Participate In Practice"


def test_note_is_none_when_all_blank():
    df = _df([{
        "week": 1, "team": "GB", "full_name": "No Detail", "position": "T",
        "report_status": "Doubtful", "report_primary_injury": None,
        "report_secondary_injury": None, "practice_status": None,
    }])
    assert parse_injuries(df)["GB"][0]["note"] is None


def test_drops_blank_and_non_report_statuses():
    # Only Out/Doubtful/Questionable are real injury-report designations;
    # NaN-status roster rows (and any other status) are dropped.
    df = _df([
        {"week": 1, "team": "SF", "full_name": "Rostered Guy", "position": "WR",
         "report_status": None, "report_primary_injury": None,
         "report_secondary_injury": None, "practice_status": None},
        {"week": 1, "team": "SF", "full_name": "Hurt Guy", "position": "RB",
         "report_status": "Out", "report_primary_injury": "Foot",
         "report_secondary_injury": None, "practice_status": None},
    ])
    out = parse_injuries(df)
    assert list(out.keys()) == ["SF"]
    assert [p["player"] for p in out["SF"]] == ["Hurt Guy"]


def test_selects_latest_reported_week_by_default():
    df = _df([
        {"week": 1, "team": "BUF", "full_name": "Old Report", "position": "LB",
         "report_status": "Questionable", "report_primary_injury": "Wrist",
         "report_secondary_injury": None, "practice_status": None},
        {"week": 2, "team": "BUF", "full_name": "New Report", "position": "S",
         "report_status": "Out", "report_primary_injury": "Concussion",
         "report_secondary_injury": None, "practice_status": None},
    ])
    out = parse_injuries(df)
    assert [p["player"] for p in out["BUF"]] == ["New Report"]


def test_explicit_week_pins_selection():
    df = _df([
        {"week": 1, "team": "BUF", "full_name": "Week1", "position": "LB",
         "report_status": "Out", "report_primary_injury": "Wrist",
         "report_secondary_injury": None, "practice_status": None},
        {"week": 2, "team": "BUF", "full_name": "Week2", "position": "S",
         "report_status": "Out", "report_primary_injury": "Concussion",
         "report_secondary_injury": None, "practice_status": None},
    ])
    out = parse_injuries(df, week=1)
    assert [p["player"] for p in out["BUF"]] == ["Week1"]


def test_skips_rows_with_null_team():
    df = _df([
        {"week": 1, "team": None, "full_name": "No Team", "position": "WR",
         "report_status": "Out", "report_primary_injury": "Hip",
         "report_secondary_injury": None, "practice_status": None},
        {"week": 1, "team": "MIN", "full_name": "Has Team", "position": "WR",
         "report_status": "Out", "report_primary_injury": "Hip",
         "report_secondary_injury": None, "practice_status": None},
    ])
    out = parse_injuries(df)
    assert list(out.keys()) == ["MIN"]
