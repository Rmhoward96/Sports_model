import pandas as pd

from sportsmodel.sim.nfl.usage import depth_charts_asof

SCHED = pd.DataFrame({
    "season": [2025, 2025], "week": [1, 2], "game_type": ["REG", "REG"],
    "gameday": ["2025-09-07", "2025-09-14"], "gametime": ["13:00", "13:00"],
    "home_team": ["KC", "LAC"], "away_team": ["LAC", "KC"],
})


def _snap(dt, team, gsis, pos, rank, name):
    return {"dt": dt, "team": team, "gsis_id": gsis, "pos_abb": pos, "pos_rank": rank, "player_name": name}


def test_snapshot_rows_take_latest_chart_at_or_before_kickoff():
    raw = pd.DataFrame([
        _snap("2025-09-01T10:00:00Z", "KC", "q1", "QB", 1, "Old Starter"),
        _snap("2025-09-06T10:00:00Z", "KC", "q2", "QB", 1, "Week1 Starter"),
        _snap("2025-09-12T10:00:00Z", "KC", "q3", "QB", 1, "Week2 Starter"),
        _snap("2025-09-20T10:00:00Z", "KC", "q4", "QB", 1, "Future"),
    ])
    out = depth_charts_asof(raw, SCHED)
    kc = out[out["club_code"] == "KC"].set_index("week")["gsis_id"]
    assert kc.loc[1] == "q2" and kc.loc[2] == "q3"      # never the future snapshot
    assert set(out.columns) >= {"season", "week", "club_code", "depth_team", "position",
                                "gsis_id", "full_name", "football_name"}


def test_old_schema_rows_pass_through_and_mix_with_snapshots():
    old = pd.DataFrame({"season": [2024], "week": [3], "club_code": ["KC"], "depth_team": ["1"],
                        "gsis_id": ["p"], "position": ["QB"], "football_name": ["Pat"],
                        "first_name": ["Pat"], "last_name": ["M"]})
    new = pd.DataFrame([_snap("2025-09-06T10:00:00Z", "KC", "q2", "QB", 1, "X")])
    out = depth_charts_asof(pd.concat([old, new], ignore_index=True), SCHED)
    assert ((out["season"] == 2024) & (out["week"] == 3) & (out["gsis_id"] == "p")).any()
    assert ((out["season"] == 2025) & (out["week"] == 1) & (out["gsis_id"] == "q2")).any()
    assert out.loc[out["gsis_id"] == "p", "full_name"].iloc[0] == "Pat M"


def test_team_with_no_snapshot_before_kickoff_gets_no_rows():
    raw = pd.DataFrame([_snap("2025-09-20T10:00:00Z", "LAC", "z", "QB", 1, "Late")])
    out = depth_charts_asof(raw, SCHED)
    assert out[(out["club_code"] == "LAC") & (out["week"] == 1)].empty


def test_snapshot_team_code_is_normalized_before_matching_schedule():
    # Ruling R1: the schedule frame's teams are normalize_team-normalized
    # (LA/LV/LAC/WAS); an alias snapshot code ("LAR") must still match, and
    # an unknown code is dropped rather than raising.
    sched = pd.DataFrame({
        "season": [2025], "week": [1], "game_type": ["REG"],
        "gameday": ["2025-09-07"], "gametime": ["16:25"],
        "home_team": ["LA"], "away_team": ["HOU"],
    })
    raw = pd.DataFrame([
        _snap("2025-09-06T10:00:00Z", "LAR", "s1", "QB", 1, "Rams QB"),
        _snap("2025-09-06T10:00:00Z", "XXX", "bad", "QB", 1, "Nobody"),
    ])
    out = depth_charts_asof(raw, sched)
    la = out[(out["club_code"] == "LA") & (out["week"] == 1)]
    assert list(la["gsis_id"]) == ["s1"]
    assert "bad" not in set(out["gsis_id"])
    assert "LAR" not in set(out["club_code"])


def test_old_schema_rows_without_a_week_are_dropped():
    # Real nflverse 2016+ depth charts carry game_type "SBBYE" rows (the
    # Super Bowl bye week) with a null `week`; they key to no game and must
    # be dropped rather than crash the int cast.
    old = pd.DataFrame({"season": [2016, 2016], "week": [5, None], "club_code": ["ATL", "ATL"],
                        "game_type": ["REG", "SBBYE"], "depth_team": ["1", "1"], "gsis_id": ["a", "b"],
                        "position": ["QB", "QB"], "football_name": ["A", "B"],
                        "first_name": ["A", "B"], "last_name": ["X", "Y"]})
    out = depth_charts_asof(old, SCHED)
    assert out["gsis_id"].tolist() == ["a"] and out["week"].tolist() == [5]


def test_formation_passes_through_for_old_rows_and_is_missing_for_snapshots():
    """_depth_rank needs the old schema's formation (Offense / Special Teams);
    snapshot rows have none (their KR/PR slots are separate pos_abb values)."""
    old = pd.DataFrame({"season": [2024, 2024], "week": [3, 3], "club_code": ["KC", "KC"],
                        "depth_team": ["1", "3"], "gsis_id": ["kr", "kr"], "position": ["WR", "WR"],
                        "formation": ["Special Teams", "Offense"], "football_name": ["K", "K"],
                        "first_name": ["K", "K"], "last_name": ["R", "R"]})
    new = pd.DataFrame([_snap("2025-09-06T10:00:00Z", "KC", "q2", "QB", 1, "X")])
    out = depth_charts_asof(pd.concat([old, new], ignore_index=True), SCHED)
    assert out.loc[out["gsis_id"] == "kr", "formation"].tolist() == ["Special Teams", "Offense"]
    assert out.loc[out["gsis_id"] == "q2", "formation"].isna().all()


# ---- chart_weeks_asof: the (season, week) chart active_usage uses per team-week ------

def test_chart_weeks_asof_matches_active_usage_fallback_rule():
    """Exact week when the team has rows that week, else its latest earlier
    chart (across seasons), else none -- the same answer as
    active_usage's _latest_depth_week for every team-week."""
    from sportsmodel.sim.nfl.usage import _latest_depth_week, chart_weeks_asof
    depth = pd.DataFrame({
        "season": [2023, 2024, 2024, 2024, 2024],
        "week": [17, 1, 3, 1, 5],
        "club_code": ["KC", "KC", "KC", "BAL", "BAL"],
        "position": ["QB", "WR", "LT", "QB", "QB"],   # any position counts as a chart
    })
    tw = pd.DataFrame({"team": ["KC", "KC", "KC", "KC", "BAL", "BAL", "MIA"],
                       "season": [2024, 2024, 2024, 2025, 2024, 2024, 2024],
                       "week": [1, 2, 4, 1, 1, 4, 1]})
    got = chart_weeks_asof(depth, tw)
    assert list(got.columns) == ["team", "season", "week", "chart_season", "chart_week"]
    for r in got.itertuples(index=False):
        want = _latest_depth_week(depth, r.team, r.season, r.week)
        have = None if pd.isna(r.chart_season) else (int(r.chart_season), int(r.chart_week))
        assert have == want, (r, want)
    assert got["chart_week"].tolist()[:3] == [1, 1, 3]
