import pandas as pd
from sportsmodel.nfl.universe import active_universe, match_book_player

def test_active_universe_filters_out_and_inactive():
    rosters = pd.DataFrame([
        {"player_id":"a","player_name":"Alice Back","team":"KC","position":"RB","depth_chart_position":"RB","status":"ACT"},
        {"player_id":"b","player_name":"Bob Wide","team":"KC","position":"WR","depth_chart_position":"WR","status":"ACT"},
        {"player_id":"c","player_name":"Cy End","team":"KC","position":"TE","depth_chart_position":"TE","status":"ACT"},
        {"player_id":"d","player_name":"Deep Snap","team":"KC","position":"LS","depth_chart_position":"LS","status":"ACT"},
    ])
    injuries = pd.DataFrame([{"gsis_id":"b","team":"KC","season":2024,"week":1,"report_status":"Out"}])
    uni = active_universe(rosters, injuries, espn_inactives=["Cy End"], season=2024, week=1)
    ids = {p["player_id"] for p in uni}
    assert "a" in ids          # active skill player
    assert "b" not in ids      # report_status Out
    assert "c" not in ids      # ESPN inactive by name
    assert "d" not in ids      # non-skill position (LS)

def test_match_book_player_normalizes():
    uni = [{"player_id":"a","player_name":"Patrick Mahomes"}]
    assert match_book_player("patrick  mahomes", uni) == "a"
    assert match_book_player("Nobody Here", uni) is None

def test_active_universe_week_discrimination():
    rosters = pd.DataFrame([
        {"player_id":"a","player_name":"Alice Back","team":"KC","position":"RB","depth_chart_position":"RB","status":"ACT"},
    ])
    injuries = pd.DataFrame([{"gsis_id":"a","team":"KC","season":2024,"week":2,"report_status":"Out"}])
    uni_week1 = active_universe(rosters, injuries, espn_inactives=[], season=2024, week=1)
    ids_week1 = {p["player_id"] for p in uni_week1}
    assert "a" in ids_week1    # injury reported for week 2 must not exclude week-1 universe

    uni_week2 = active_universe(rosters, injuries, espn_inactives=[], season=2024, week=2)
    ids_week2 = {p["player_id"] for p in uni_week2}
    assert "a" not in ids_week2  # same player IS excluded in the matching week

def test_active_universe_only_out_excludes():
    rosters = pd.DataFrame([
        {"player_id":"a","player_name":"Alice Back","team":"KC","position":"RB","depth_chart_position":"RB","status":"ACT"},
        {"player_id":"b","player_name":"Bob Wide","team":"KC","position":"WR","depth_chart_position":"WR","status":"ACT"},
    ])
    injuries = pd.DataFrame([
        {"gsis_id":"a","team":"KC","season":2024,"week":1,"report_status":"Questionable"},
        {"gsis_id":"b","team":"KC","season":2024,"week":1,"report_status":"Doubtful"},
    ])
    uni = active_universe(rosters, injuries, espn_inactives=[], season=2024, week=1)
    ids = {p["player_id"] for p in uni}
    assert "a" in ids   # Questionable players still get props
    assert "b" in ids   # Doubtful players still get props
