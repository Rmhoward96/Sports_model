"""compute_recent_form must reflect the CURRENT season only.

Regression: the desk showed prior-season records (e.g. CFB Ole Miss "5-0",
LSU "2-3") for teams that had played only 2 games in the current season,
because compute_recent_form took the trailing N games ACROSS season
boundaries (and the schedule asset can lag a season). It must filter to the
current season: a full current-season W-L record, and recent-form metrics
over the current season's trailing N games; a team with no current-season
games has no form (absent), never a prior-season record.
"""
from datetime import datetime, timezone

import pandas as pd

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
from desk_inputs import compute_recent_form, _current_season  # noqa: E402


def _sched(rows):
    return pd.DataFrame(rows, columns=["season", "week", "home_team", "away_team",
                                       "home_score", "away_score", "game_type"])


def test_current_season_from_now():
    assert _current_season(datetime(2026, 9, 19, tzinfo=timezone.utc)) == 2026
    assert _current_season(datetime(2026, 1, 5, tzinfo=timezone.utc)) == 2025   # bowls/playoff belong to prior season
    assert _current_season(datetime(2026, 6, 1, tzinfo=timezone.utc)) == 2025   # offseason -> last completed season


def test_record_is_current_season_only_not_cross_season():
    # LSU: 5 wins in 2025, then 1-1 in 2026. Cross-season tail(5) would say ~5-1
    # or "5-0"; current-season-only must say 1-1.
    rows = []
    for w in range(1, 6):
        rows.append([2025, w, "LSU", "Opp", 30, 10, "REG"])          # 5 wins in 2025
    rows.append([2026, 1, "LSU", "OppA", 20, 17, "REG"])             # 2026 W
    rows.append([2026, 2, "OppB", "LSU", 24, 20, "REG"])             # 2026 L (LSU away)
    form = compute_recent_form(_sched(rows), {"LSU"}, current_season=2026)
    assert form["LSU"]["record"] == "1-1"
    assert form["LSU"]["last_n"] == ["L", "W"]  # most-recent first


def test_full_current_season_record_not_capped_at_n():
    # 7 current-season games (6-1): the record counts ALL of them, not just tail(n=5).
    rows = [[2026, w, "Bama", "O", 40, 10, "REG"] for w in range(1, 7)]   # 6 wins
    rows.append([2026, 7, "O2", "Bama", 30, 20, "REG"])                   # 1 loss
    form = compute_recent_form(_sched(rows), {"Bama"}, current_season=2026, n=5)
    assert form["Bama"]["record"] == "6-1"          # full season, not tail(5)
    assert len(form["Bama"]["last_n"]) == 5         # form metrics still trailing-N


def test_no_current_season_games_means_no_form():
    # Only prior-season games -> team absent (no misleading prior-season record).
    rows = [[2025, w, "OleMiss", "Opp", 35, 14, "REG"] for w in range(1, 6)]
    form = compute_recent_form(_sched(rows), {"OleMiss"}, current_season=2026)
    assert "OleMiss" not in form
