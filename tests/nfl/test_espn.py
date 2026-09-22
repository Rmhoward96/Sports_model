import json, pathlib
from sportsmodel.nfl.espn import (parse_schedule, parse_final, parse_current_week,
                                  target_week, advance_if_complete)

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

def test_target_week_regular_season_passthrough():
    assert target_week({"season": 2026, "week": 3, "season_type": 2}) == \
        {"season": 2026, "week": 3, "season_type": 2}

def test_target_week_postseason_passthrough():
    assert target_week({"season": 2026, "week": 2, "season_type": 3}) == \
        {"season": 2026, "week": 2, "season_type": 3}

def test_target_week_preseason_looks_ahead_to_regular_week1():
    # preseason (type 1) -> regular-season (type 2) Week 1, the games the market prices
    assert target_week({"season": 2026, "week": 3, "season_type": 1}) == \
        {"season": 2026, "week": 1, "season_type": 2}

def test_target_week_offseason_targets_regular_week1():
    assert target_week({"season": 2026, "week": 1, "season_type": 4}) == \
        {"season": 2026, "week": 1, "season_type": 2}


def _g(status):
    return {"status": status}


def test_advance_if_complete_advances_when_all_final():
    tw = {"season": 2026, "week": 2, "season_type": 2}
    games = [_g("STATUS_FINAL"), _g("STATUS_FINAL")]
    assert advance_if_complete(tw, games) == {"season": 2026, "week": 3, "season_type": 2}


def test_advance_if_complete_holds_when_a_game_pending():
    tw = {"season": 2026, "week": 2, "season_type": 2}
    games = [_g("STATUS_FINAL"), _g("STATUS_SCHEDULED")]
    assert advance_if_complete(tw, games) == tw


def test_advance_if_complete_holds_on_empty_or_postseason():
    tw = {"season": 2026, "week": 2, "season_type": 2}
    assert advance_if_complete(tw, []) == tw            # schedule not posted -> no move
    post = {"season": 2026, "week": 2, "season_type": 3}
    assert advance_if_complete(post, [_g("STATUS_FINAL")]) == post


def test_advance_if_complete_caps_at_week_18():
    tw = {"season": 2026, "week": 18, "season_type": 2}
    assert advance_if_complete(tw, [_g("STATUS_FINAL")]) == tw  # let ESPN roll to postseason
