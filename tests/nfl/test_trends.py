import pandas as pd
from sportsmodel.nfl.trends import (NFL_DIVISIONS, compute_game_trends, situations_for, team_game_log)


def _g(season, day, home, away, hs, as_, spread, tot, time="13:00", espn=None):
    return {"season": season, "week": 1, "game_type": "REG", "gameday": day, "gametime": time,
            "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
            "result": None if hs is None else hs - as_, "total": None if hs is None else hs + as_,
            "spread_line": spread, "total_line": tot, "espn": espn}


def test_team_game_log_cover_and_ou_from_each_side():
    sched = pd.DataFrame([_g(2025, "2025-09-07", "BUF", "NYJ", 27, 20, 6.5, 44.5)])
    log = team_game_log(sched)
    buf = log[log.team == "BUF"].iloc[0]; nyj = log[log.team == "NYJ"].iloc[0]
    assert buf.is_home and buf.team_line == -6.5 and buf.ats == "W"      # won by 7, laid 6.5
    assert (not nyj.is_home) and nyj.team_line == 6.5 and nyj.ats == "L"
    assert buf.ou == "O" and nyj.ou == "O"                                # 47 > 44.5


def test_push_and_missing_line_rows():
    sched = pd.DataFrame([_g(2025, "2025-09-07", "BUF", "NYJ", 27, 20, 7.0, 47.0),
                          _g(2025, "2025-09-14", "BUF", "MIA", 20, 17, None, None)])
    log = team_game_log(sched)
    assert log[(log.team == "BUF") & (log.gameday == "2025-09-07")].iloc[0].ats == "P"
    assert pd.isna(log[(log.team == "BUF") & (log.gameday == "2025-09-14")].iloc[0].ats)


def test_situations_for():
    prev = pd.DataFrame([{"gameday": "2026-09-13", "is_home": False, "su": "W", "season": 2026}])
    s = situations_for("BUF", True, -3.0, "2026-09-27 20:20", prev, opp="MIA", season=2026, gameday="2026-09-27")
    assert set(s) >= {"home", "favorite", "off_road", "off_win", "off_bye", "division", "primetime"}
    assert "underdog" not in s and "road" not in s


def test_compute_game_trends_counts_matching_past_games_and_min_n():
    rows = []
    # BUF: 5 past road games each followed by a home game (to build "off a road game" history)
    # Then one final road game to set up off_road for upcoming
    days = pd.date_range("2025-09-07", periods=11, freq="7D").strftime("%Y-%m-%d")
    for i, d in enumerate(days):
        if i % 2 == 0:
            # Even indices (0,2,4,6,8,10): BUF road
            rows.append(_g(2025, d, "NYJ", "BUF", 10, 20, 3.0, 40.0))
        else:
            # Odd indices (1,3,5,7,9): BUF home after road
            rows.append(_g(2025, d, "BUF", "NE", 24, 21, 1.0, 40.0))
    
    sched = pd.DataFrame(rows)
    upcoming = [{"game_pk": 401, "home_team": "BUF", "away_team": "MIA", "home_line": -2.5,
                 "gameday": "2026-09-27", "gametime": "13:00"}]
    out = compute_game_trends(sched, upcoming, current_season=2026, min_n=5)
    
    # Test for off_road situation: BUF's last game (2025-11-16) is road, so off_road applies
    off_road = [r for r in out if r["team"] == "BUF" and r["situation"] == "off_road"]
    
    # Past games with off_road (games 1,3,5,7,9 where BUF is home after road game):
    # Each is: BUF home vs NE, spread 1.0, result 24-21=3, ats=W (3 > -1.0)
    # total 45 vs 40, ou=O
    # Exactly 5 games
    assert len(off_road) > 0, "off_road situation should exist for BUF"
    for r in off_road:
        assert r["n"] == 5, f"off_road should have exactly 5 games, got {r['n']}"
        assert r["ats_w"] == 5 and r["ats_l"] == 0 and r["ats_p"] == 0, \
            f"off_road ats should be 5W-0L-0P, got {r['ats_w']}W-{r['ats_l']}L-{r['ats_p']}P"
        assert r["ou_o"] == 5 and r["ou_u"] == 0 and r["ou_p"] == 0, \
            f"off_road ou should be 5O-0U-0P, got {r['ou_o']}O-{r['ou_u']}U-{r['ou_p']}P"
        assert r["ats_w"] + r["ats_l"] + r["ats_p"] == r["n"]
    
    # Test min_n: off_home should be omitted (only 4 road games before any home games initially)
    off_home = [r for r in out if r["team"] == "BUF" and r["situation"] == "off_home"]
    assert len(off_home) == 0, "off_home should be omitted (fewer than 5 games)"
    
    # Verify other assertions
    assert all(r["n"] >= 5 for r in out), "All situations should have n >= 5"
    assert all(r["since_season"] == 2023 for r in out), "since_season should be 2023 (2026 - 3)"
