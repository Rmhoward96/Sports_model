import json, pathlib
from sportsmodel.nfl.espn import parse_schedule, parse_final, parse_current_week

FIX = json.loads((pathlib.Path(__file__).parent.parent
                  / "fixtures/nfl/espn_scoreboard.json").read_text())
CURRENT_WEEK_FIX = json.loads((pathlib.Path(__file__).parent.parent
                               / "fixtures/nfl/espn_current_week.json").read_text())

def test_parse_schedule_normalizes_and_types():
    games = parse_schedule(FIX)
    assert len(games) == 2
    g0 = games[0]
    assert g0["game_pk"] == 401671789 and isinstance(g0["game_pk"], int)
    assert g0["home_team"] == "KC" and g0["away_team"] == "BAL"
    assert g0["status"] == "STATUS_FINAL"
    g1 = games[1]
    assert g1["home_team"] == "WAS" and g1["away_team"] == "LA"   # WSH/LAR normalized

def test_parse_final_gates_on_status():
    assert parse_final(FIX["events"][0]) == {"home_score": 27, "away_score": 20, "final": True}
    assert parse_final(FIX["events"][1]) is None   # not STATUS_FINAL

def test_parse_schedule_emits_display_names():
    g = parse_schedule(FIX)[0]
    assert g["home_name"] and g["away_name"]        # full display names present
    assert g["game_pk"] == 401671789

def test_parse_current_week():
    assert parse_current_week(CURRENT_WEEK_FIX) == {"season": 2024, "week": 3, "season_type": 2}
