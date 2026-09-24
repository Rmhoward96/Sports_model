"""Tests for the shared NFL injury report (pure merge + IO).

The desk and sim both need the same view of which players are designated Out/Doubtful/Questionable:
nflverse's official weekly report, verified per player against ESPN's live injury list, and never
trusted when stale. nflverse's newest report is LAST week's until the Wed-Fri report posts; a
player Out last week (Penix, Week 2) would otherwise stay Out in this week's sim/desk. Solution:
if the report's week is older than the target week and ESPN is reachable, the report is dropped
and ESPN's current designations are used; ESPN also overrides the current report per player.
"""
from __future__ import annotations

import pandas as pd

from sportsmodel.nfl.injury_report import latest_report_week, merge_report, normalize_status


N2A = {"Atlanta Falcons": "ATL", "Green Bay Packers": "GB"}
NFLV = {
    "ATL": [
        {"player": "Michael Penix Jr.", "position": "QB", "status": "Out", "note": None},
        {"player": "Samson Ebukam", "position": "DE", "status": "Out", "note": None},
    ]
}


def test_normalize_status():
    assert normalize_status("Injured Reserve") == "Out" and normalize_status("Suspension") == "Out"
    assert normalize_status("doubtful") == "Doubtful" and normalize_status("Questionable") == "Questionable"
    assert normalize_status("Active") is None and normalize_status("Day-To-Day") is None


def test_normalize_status_hyphenated_and_slashed():
    """NFI and PUP should normalize to Out; punctuation should not break matching."""
    assert normalize_status("Non-Football Injury") == "Out"
    assert normalize_status("Reserve/Non-Football Injury") == "Out"
    assert normalize_status("PUP") == "Out"
    assert normalize_status("Physically Unable to Perform") == "Out"


def test_normalize_status_whole_word_match():
    """Doubtful/Questionable should match as whole words, even in parenthetical."""
    assert normalize_status("Doubtful (knee)") == "Doubtful"
    assert normalize_status("Questionable (ankle)") == "Questionable"


def test_normalize_status_negative_cases():
    """Statuses that should NOT normalize to Out."""
    assert normalize_status("Active") is None
    assert normalize_status("Day-To-Day") is None
    assert normalize_status("Probable") is None
    assert normalize_status("Commissioner Exempt") is None


def test_stale_report_dropped_when_espn_available():
    espn = [{"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Active"}]
    r = merge_report(NFLV, report_week=2, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    assert r["stale"] is True
    assert r["by_team"].get("ATL", []) == []  # Ebukam (stale-only) dropped, Penix cleared
    assert {"team": "ATL", "player": "Michael Penix Jr.", "nflverse": "Out", "espn": None} in r["conflicts"]


def test_current_report_kept_and_espn_overrides_per_player():
    espn = [
        {"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Questionable"},
        {"team": "Green Bay Packers", "player": "Jordan Love", "status": "Out"},
    ]
    r = merge_report(NFLV, report_week=3, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    atl = {x["player"]: (x["status"], x["source"]) for x in r["by_team"]["ATL"]}
    assert atl == {"Michael Penix Jr.": ("Questionable", "espn"), "Samson Ebukam": ("Out", "nflverse")}
    assert [(x["player"], x["status"]) for x in r["by_team"]["GB"]] == [("Jordan Love", "Out")]
    assert r["stale"] is False


def test_stale_without_espn_uses_nflverse_flagged():
    r = merge_report(NFLV, report_week=2, target_week=3, espn_rows=[], name_to_abbr=N2A, espn_available=False)
    assert r["stale"] is True and r["espn_available"] is False
    assert len(r["by_team"]["ATL"]) == 2


def test_report_week_none_with_espn_available():
    """report_week=None is stale; only ESPN rows used."""
    espn = [{"team": "Atlanta Falcons", "player": "Test Player", "status": "Out"}]
    r = merge_report(NFLV, report_week=None, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    assert r["stale"] is True
    assert r["report_week"] is None
    # nflverse rows (stale) should be dropped; only ESPN row should be present
    assert len(r["by_team"].get("ATL", [])) == 1
    assert r["by_team"]["ATL"][0]["player"] == "Test Player"


def test_target_week_none_is_not_stale():
    """target_week=None means no target; report is not stale."""
    espn = [{"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Doubtful"}]
    r = merge_report(NFLV, report_week=2, target_week=None, espn_rows=espn, name_to_abbr=N2A)
    assert r["stale"] is False
    # Both nflverse and ESPN should be present
    atl = {x["player"]: (x["status"], x["source"]) for x in r["by_team"]["ATL"]}
    assert atl == {"Michael Penix Jr.": ("Doubtful", "espn"), "Samson Ebukam": ("Out", "nflverse")}


def test_name_matching_normalized_and_unknown_team_skipped():
    espn = [
        {"team": "Atlanta Falcons", "player": "michael penix jr", "status": "Active"},
        {"team": "Nowhere", "player": "X", "status": "Out"},
    ]
    r = merge_report(NFLV, report_week=3, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    assert [x["player"] for x in r["by_team"]["ATL"]] == ["Samson Ebukam"]


def test_curly_apostrophe_matching():
    """Ja'Marr Chase (nflverse) Out + Ja'Marr Chase (ESPN, curly apostrophe) Active → cleared + conflict."""
    nflv = {"CIN": [{"player": "Ja'Marr Chase", "position": "WR", "status": "Out", "note": None}]}
    espn = [{"team": "Cincinnati Bengals", "player": "Ja'Marr Chase", "status": "Active"}]
    n2a = {"Cincinnati Bengals": "CIN"}
    r = merge_report(nflv, report_week=3, target_week=3, espn_rows=espn, name_to_abbr=n2a)
    # Ja'Marr Chase should be cleared (ESPN Active) and not in by_team
    assert r["by_team"].get("CIN", []) == []
    # Conflict: nflverse says Out, ESPN says None (cleared)
    assert {"team": "CIN", "player": "Ja'Marr Chase", "nflverse": "Out", "espn": None} in r["conflicts"]


def test_latest_report_week_with_designations():
    """Test latest_report_week with a tiny DataFrame (weeks 1-3, only 1-2 with designations)."""
    df = pd.DataFrame([
        {"week": 1, "report_status": "Out", "player": "Player1"},
        {"week": 2, "report_status": "Doubtful", "player": "Player2"},
        {"week": 3, "report_status": None, "player": "Player3"},
    ])
    assert latest_report_week(df) == 2


def test_latest_report_week_empty_returns_none():
    """Test latest_report_week with empty DataFrame."""
    df = pd.DataFrame({"week": [], "report_status": []})
    assert latest_report_week(df) is None


def test_latest_report_week_no_designations_returns_none():
    """Test latest_report_week with no real designations."""
    df = pd.DataFrame([
        {"week": 1, "report_status": None, "player": "Player1"},
        {"week": 2, "report_status": "Active", "player": "Player2"},
    ])
    assert latest_report_week(df) is None
