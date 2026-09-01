import copy
import json
import pathlib

from sportsmodel.cfb.espn import parse_schedule, parse_final

FIX = json.loads((pathlib.Path(__file__).parent.parent
                  / "fixtures/cfb/espn_scoreboard.json").read_text())


def test_parse_schedule_normalizes_fbs_and_fcs_teams():
    games = parse_schedule(FIX)
    assert len(games) == 2

    g0 = games[0]  # Georgia @ Kentucky -- FBS vs FBS
    assert g0["game_pk"] == 401628354 and isinstance(g0["game_pk"], int)
    assert g0["home_team"] == "96"   # Kentucky ESPN id, FBS passthrough
    assert g0["away_team"] == "61"   # Georgia ESPN id, FBS passthrough
    assert g0["home_name"] == "Kentucky Wildcats"
    assert g0["away_name"] == "Georgia Bulldogs"
    assert g0["status"] == "STATUS_FINAL"

    g1 = games[1]  # Northern Iowa @ Nebraska -- FBS vs FCS
    assert g1["home_team"] == "158"   # Nebraska ESPN id, FBS passthrough
    assert g1["away_team"] == "FCS"   # Northern Iowa collapses to FCS anchor
    assert g1["away_name"] == "Northern Iowa Panthers"


def test_parse_schedule_types_scores_and_ids():
    g0 = parse_schedule(FIX)[0]
    assert g0["home_score"] == 12 and isinstance(g0["home_score"], int)
    assert g0["away_score"] == 13 and isinstance(g0["away_score"], int)


def test_parse_schedule_populates_week_and_season():
    for g in parse_schedule(FIX):
        assert g["week"] == 3
        assert g["season"] == 2024


def test_parse_final_gates_on_status():
    assert parse_final(FIX["events"][0]) == {"home_score": 12, "away_score": 13, "final": True}
    assert parse_final(FIX["events"][1]) == {"home_score": 34, "away_score": 3, "final": True}

    not_final = copy.deepcopy(FIX["events"][0])
    not_final["status"]["type"]["name"] = "STATUS_SCHEDULED"
    assert parse_final(not_final) is None
