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
    # BUF: alternating road/home games, all wins, total lines create mixed over/under
    days = pd.date_range("2025-09-07", periods=11, freq="7D").strftime("%Y-%m-%d")
    for i, d in enumerate(days):
        if i % 2 == 0:
            # Even indices (0,2,4,6,8,10): BUF road vs NYJ (division)
            # NYJ home 10, BUF away 20, total 30 < 40 → Under
            rows.append(_g(2025, d, "NYJ", "BUF", 10, 20, 3.0, 40.0))
        else:
            # Odd indices (1,3,5,7,9): BUF home vs NE (division)
            # BUF home 24, NE away 21, total 45 > 40 → Over
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

    # Assert full expected set of BUF rows: home, favorite, off_road, off_win, division
    expected_situations = {"home", "favorite", "off_road", "off_win", "division"}
    buf_situations = {r["situation"] for r in out if r["team"] == "BUF"}
    assert buf_situations == expected_situations, \
        f"Expected {expected_situations}, got {buf_situations}"

    # Assert full expected counts for each situation
    # home: 5 games (all Over, all cover)
    # favorite: 5 games (all home games, Over, all cover)
    # off_road: 5 games (home games after road, Over, all cover)
    # off_win: 10 games (5 road Under + 5 home Over, all cover)
    # division: 11 games (6 road Under + 5 home Over, all cover)
    expected_counts = {
        "home": {"ats_w": 5, "ats_l": 0, "ats_p": 0, "ou_o": 5, "ou_u": 0, "ou_p": 0, "n": 5},
        "favorite": {"ats_w": 5, "ats_l": 0, "ats_p": 0, "ou_o": 5, "ou_u": 0, "ou_p": 0, "n": 5},
        "off_road": {"ats_w": 5, "ats_l": 0, "ats_p": 0, "ou_o": 5, "ou_u": 0, "ou_p": 0, "n": 5},
        "off_win": {"ats_w": 10, "ats_l": 0, "ats_p": 0, "ou_o": 5, "ou_u": 5, "ou_p": 0, "n": 10},
        "division": {"ats_w": 11, "ats_l": 0, "ats_p": 0, "ou_o": 5, "ou_u": 6, "ou_p": 0, "n": 11},
    }
    for r in out:
        if r["team"] == "BUF":
            exp = expected_counts[r["situation"]]
            assert r["ats_w"] == exp["ats_w"] and r["ats_l"] == exp["ats_l"] and r["ats_p"] == exp["ats_p"], \
                f"{r['situation']}: ats {r['ats_w']}W-{r['ats_l']}L-{r['ats_p']}P != expected {exp['ats_w']}W-{exp['ats_l']}L-{exp['ats_p']}P"
            assert r["ou_o"] == exp["ou_o"] and r["ou_u"] == exp["ou_u"] and r["ou_p"] == exp["ou_p"], \
                f"{r['situation']}: ou {r['ou_o']}O-{r['ou_u']}U-{r['ou_p']}P != expected {exp['ou_o']}O-{exp['ou_u']}U-{exp['ou_p']}P"
            assert r["n"] == exp["n"], f"{r['situation']}: n {r['n']} != expected {exp['n']}"

    # Verify other assertions
    assert all(r["n"] >= 5 for r in out), "All situations should have n >= 5"
    assert all(r["since_season"] == 2023 for r in out), "since_season should be 2023 (2026 - 3)"

    # Verify NFL_DIVISIONS usage: assert 32 unique teams, 8 divisions
    assert len(NFL_DIVISIONS) == 32, f"Expected 32 teams, got {len(NFL_DIVISIONS)}"
    divisions = set(NFL_DIVISIONS.values())
    assert len(divisions) == 8, f"Expected 8 divisions, got {len(divisions)}"


def test_compute_game_trends_min_n_filter_with_primetime():
    """Test that situations with fewer than min_n games are omitted.
    Using primetime: exactly 4 past primetime games should be excluded."""
    rows = []
    days = pd.date_range("2025-09-07", periods=4, freq="7D").strftime("%Y-%m-%d")
    # Create 4 primetime games (19:00 or later)
    for i, d in enumerate(days):
        rows.append(_g(2025, d, "BUF", "NE", 24, 21, 1.0, 40.0, time="20:00"))
    # Upcoming game is primetime too
    sched = pd.DataFrame(rows)
    upcoming = [{"game_pk": 401, "home_team": "BUF", "away_team": "MIA", "home_line": -2.5,
                 "gameday": "2026-09-27", "gametime": "20:00"}]
    out = compute_game_trends(sched, upcoming, current_season=2026, min_n=5)

    # Primetime should be omitted (only 4 past games < min_n=5)
    primetime = [r for r in out if r["team"] == "BUF" and r["situation"] == "primetime"]
    assert len(primetime) == 0, f"primetime should be omitted (4 < min_n=5), but got {len(primetime)} rows"


def test_compute_game_trends_season_filtering():
    """Test that games from season 2022 (since-1) influence situations but aren't counted.
    2022 road game is 'previous' for first 2023 game (off_road applies), but 2022 is
    excluded from n count. Division includes only 2023+ games (excludes 2022)."""
    rows = []
    # 2022 road game: serves as previous for 2023 games
    rows.append(_g(2022, "2022-12-25", "NYJ", "BUF", 10, 20, 3.0, 40.0))
    # 2023-2025: alternating road/home, 5 games per year
    for year in [2023, 2024, 2025]:
        dates = pd.date_range(f"{year}-09-10", periods=5, freq="7D").strftime("%Y-%m-%d")
        for i, d in enumerate(dates):
            if i % 2 == 0:
                rows.append(_g(year, d, "NYJ", "BUF", 10, 20, 3.0, 40.0))  # road
            else:
                rows.append(_g(year, d, "BUF", "NE", 24, 21, 1.0, 40.0))   # home (after road)

    sched = pd.DataFrame(rows)
    # Upcoming: BUF road (last game before this is home, so off_road applies)
    upcoming = [{"game_pk": 401, "home_team": "NYJ", "away_team": "BUF", "home_line": 3.0,
                 "gameday": "2026-09-27", "gametime": "13:00"}]
    out = compute_game_trends(sched, upcoming, current_season=2026, min_n=5)

    # off_road: games after road (indices 1,3,5,7,9,11,13,15 = 8 games from 2023-2025)
    off_road = [r for r in out if r["team"] == "BUF" and r["situation"] == "off_road"]
    assert len(off_road) > 0, "off_road should exist"
    for r in off_road:
        # First 2023 game has 2022 road as previous, influencing off_road situation
        # off_road applies to: indices 1,3,5,7,9,11,13,15 (8 games total)
        assert r["n"] >= 5, f"off_road needs n >= min_n=5, got {r['n']}"

    # division: NYJ vs BUF (both AFC East)
    # All games from 2023-2025 (15 total), 2022 game excluded (season < since=2023)
    division = [r for r in out if r["team"] == "BUF" and r["situation"] == "division"]
    assert len(division) > 0, "division should exist"
    for r in division:
        # Verify that 2022 game is NOT counted: 15 games (2023-2025) not 16 (with 2022)
        assert r["n"] == 15, f"division should have 15 games (2023-2025, not 2022), got {r['n']}"


def test_compute_game_trends_mixed_line_no_line_with_nan_upcoming():
    """Test that NaN home_line in upcoming game doesn't create favorite/underdog."""
    rows = []
    # Create 5 home games with lines (to meet min_n for home situations)
    days = pd.date_range("2025-09-01", periods=5, freq="7D").strftime("%Y-%m-%d")
    for i, d in enumerate(days):
        rows.append(_g(2025, d, "BUF", "NE", 24, 21, 1.0, 40.0))

    sched = pd.DataFrame(rows)
    # Upcoming game with NaN home_line (will convert to None)
    upcoming = [{"game_pk": 401, "home_team": "BUF", "away_team": "MIA", "home_line": float("nan"),
                 "gameday": "2026-09-27", "gametime": "13:00"}]
    out = compute_game_trends(sched, upcoming, current_season=2026, min_n=5)

    # Verify no row has "favorite" or "underdog" (NaN home_line should convert to None)
    for r in out:
        if r["team"] == "BUF":
            assert r["situation"] not in ("favorite", "underdog"), \
                f"NaN home_line should not create favorite/underdog, but got {r['situation']}"

    # Verify ats_w + ats_l + ats_p == n for every row (no missing games)
    for r in out:
        total = r["ats_w"] + r["ats_l"] + r["ats_p"]
        assert total == r["n"], \
            f"Situation {r['situation']}: ats total {total} != n {r['n']}"
