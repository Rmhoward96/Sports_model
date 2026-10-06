import json, pathlib
from sportsmodel.nfl.espn import (parse_schedule, parse_final, parse_current_week,
                                  target_week, advance_if_complete, parse_injuries)

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


def test_parse_injuries_flattens_team_player_status():
    payload = {"injuries": [
        {"id": "19", "displayName": "New York Giants", "injuries": [
            {"status": "Doubtful", "athlete": {"displayName": "Jaxson Dart"}},
            {"status": "Active", "athlete": {"displayName": "Malik Nabers"}},
            {"status": "Out", "athlete": {}},  # no athlete name -> skipped
        ]},
        {"displayName": None, "injuries": [  # no team name -> skipped
            {"status": "Out", "athlete": {"displayName": "Nobody"}}]},
    ]}
    rows = parse_injuries(payload)
    assert {"team": "New York Giants", "player": "Jaxson Dart", "status": "Doubtful"} in rows
    assert any(r["player"] == "Malik Nabers" for r in rows)
    assert all(r["player"] != "Nobody" for r in rows)   # team-less entry dropped
    assert len(rows) == 2                                 # the no-athlete row dropped
    assert parse_injuries({}) == []


# ---------------------------------------------------------------- game info --

# Venue block copied from a LIVE finished-game summary (2026-10-06, event 401872971); it has no weather block.
LIVE_VENUE = {"id": "11938", "fullName": "Highmark Stadium",
              "address": {"city": "Orchard Park", "state": "NY", "zipCode": "14127", "country": "USA"},
              "grass": True, "images": []}


def test_parse_game_info_live_venue_shape_without_weather():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "attendance": 60606}})
    assert got == {"venue_name": "Highmark Stadium", "city": "Orchard Park", "state": "NY", "indoor": False,
                   "temp_f": None, "wind_mph": None, "precip_chance": None, "conditions": None}


# gameInfo.weather copied from a LIVE upcoming-game summary (2026-10-06, event 401872984, Nissan Stadium, week 5).
# Real keys: temperature, highTemperature, lowTemperature, conditionId (a code, no text), gust, precipitation
# (a 0-100 chance), link. There is NO sustained-wind key and NO condition text.
LIVE_WEATHER = {"temperature": 74, "highTemperature": 74, "lowTemperature": 74, "conditionId": "18", "gust": 30,
                "precipitation": 65, "link": {"language": "en-US", "rel": ["37213"], "text": "Weather", "isExternal": True}}


def test_parse_game_info_live_weather_block():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "weather": LIVE_WEATHER}})
    assert got["temp_f"] == 74.0 and got["precip_chance"] == 65.0
    assert got["wind_mph"] is None and got["conditions"] is None  # gust is not sustained wind; conditionId has no text
    assert got["indoor"] is False


def test_parse_game_info_known_dome_venues_are_indoor_without_an_indoor_key():
    from sportsmodel.nfl.espn import parse_game_info
    venue = {"id": "5239", "fullName": "U.S. Bank Stadium", "address": {"city": "Minneapolis", "state": "MN"}}
    got = parse_game_info({"gameInfo": {"venue": venue, "weather": LIVE_WEATHER}})
    assert got["indoor"] is True and got["temp_f"] is None and got["precip_chance"] is None
    assert got["venue_name"] == "U.S. Bank Stadium" and got["city"] == "Minneapolis"


def test_parse_game_info_gust_is_not_wind_and_bad_values_are_none():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "weather": {"temperature": "n/a", "gust": 30,
                                                                            "precipitation": 250}}})
    assert (got["temp_f"], got["wind_mph"], got["precip_chance"], got["conditions"]) == (None, None, None, None)


def test_parse_game_info_indoor_venue_drops_weather():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": {**LIVE_VENUE, "indoor": True},
                                        "weather": {"temperature": 72, "displayValue": "Clear"}}})
    assert got["indoor"] is True and got["temp_f"] is None and got["conditions"] is None
    assert got["venue_name"] == "Highmark Stadium"


def test_parse_game_info_missing_blocks_never_raise():
    from sportsmodel.nfl.espn import parse_game_info
    for payload in ({}, {"gameInfo": None}, {"gameInfo": {"venue": None, "weather": []}}):
        got = parse_game_info(payload)
        assert got["venue_name"] is None and got["indoor"] is False and got["temp_f"] is None


def test_fetch_game_info_reads_the_summary_of_one_event(monkeypatch):
    from sportsmodel.nfl import espn
    seen = {}
    monkeypatch.setattr(espn, "_get", lambda path, params=None: seen.update(path=path, params=params) or {"gameInfo": {"venue": LIVE_VENUE}})
    assert espn.fetch_game_info(401872971)["venue_name"] == "Highmark Stadium"
    assert seen == {"path": "/summary", "params": {"event": 401872971}}


def test_parse_game_info_dome_match_is_case_and_whitespace_insensitive_and_sofi_is_open():
    from sportsmodel.nfl.espn import parse_game_info
    wx = {"temperature": 70, "precipitation": 10}
    dome = parse_game_info({"gameInfo": {"venue": {"fullName": "  u.s. bank STADIUM "}, "weather": wx}})
    assert dome["indoor"] is True and dome["temp_f"] is None
    sofi = parse_game_info({"gameInfo": {"venue": {"fullName": "SoFi Stadium"}, "weather": wx}})
    assert sofi["indoor"] is False and sofi["temp_f"] == 70.0


def test_parse_game_info_non_finite_numbers_are_none():
    from sportsmodel.nfl.espn import parse_game_info
    for bad in ("nan", "inf", "-inf", float("nan"), float("inf")):
        got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "weather": {"temperature": bad, "precipitation": bad}}})
        assert got["temp_f"] is None and got["precip_chance"] is None
