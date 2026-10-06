import pandas as pd
import pytest

from sportsmodel import game_info as gi
from sportsmodel.cfb.cfbd_games import parse_weather_window

NOW = pd.Timestamp("2026-10-07T14:00:00Z")


def _g(pk, kick):
    return {"game_pk": pk, "commence_time": kick}


def test_in_window_bounds_and_junk():
    assert gi.in_window("2026-10-05T14:00:00Z", NOW) and gi.in_window("2026-10-17T14:00:00Z", NOW)
    assert not gi.in_window("2026-10-05T13:59:00Z", NOW) and not gi.in_window("2026-10-17T14:01:00Z", NOW)
    assert gi.in_window("2026-10-10T00:00", NOW)                     # naive -> UTC
    assert not gi.in_window("garbage", NOW) and not gi.in_window(None, NOW)


WEATHER = parse_weather_window([
    {"id": 1, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z", "gameIndoors": False,
     "homeTeam": "Alabama", "awayTeam": "Georgia", "venueId": 3657, "venue": "Bryant-Denny Stadium",
     "temperature": 41.2, "windSpeed": 14.0, "precipitation": 0.04, "weatherCondition": "Cloudy"},
    {"id": 2, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-08T23:30:00.000Z", "gameIndoors": True,
     "homeTeam": "UTSA", "awayTeam": "South Florida", "venueId": 3604, "venue": "Alamodome",
     "temperature": 82.8, "windSpeed": 9.2, "precipitation": 0, "weatherCondition": "Fair"},
    {"id": 3, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-06T00:00:00.000Z", "gameIndoors": False,
     "homeTeam": "Alabama", "awayTeam": "Mercer", "venueId": 9999, "venue": "Unlisted Field",
     "temperature": 70, "windSpeed": 5, "precipitation": 0},
    {"id": 4, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z", "gameIndoors": False,
     "homeTeam": "Bridgewater State", "awayTeam": "Framingham State", "venueId": 5808, "venue": "Swenson",
     "temperature": 66, "windSpeed": 10.1, "precipitation": 0}])
VENUES = pd.DataFrame({"venue_id": [3657, 3604], "name": ["Bryant-Denny Stadium", "Alamodome"],
                       "city": ["Tuscaloosa", "San Antonio"], "state": ["AL", "TX"], "dome": [0.0, 1.0]})
GAMES = [_g(1, "2026-10-09T22:00:00Z"), _g(2, "2026-10-08T23:30:00Z"), _g(3, "2026-10-06T00:00:00Z"),
         _g(4, "2026-10-09T22:00:00Z"), _g(5, "2026-10-09T22:00:00Z"), _g(6, "2026-12-01T22:00:00Z")]


def test_cfb_rows_forecast_indoor_observed_window_and_skips():
    rows = {r["game_pk"]: r for r in gi.cfb_game_info(GAMES, {3: {"home": [7, 7, 7, 7], "away": [0, 0, 0, 3]}}, WEATHER, VENUES, NOW)}
    assert set(rows) == {1, 2, 3}                       # 4: no FBS side, 5: no weather and no line score, 6: outside the window
    a = rows[1]
    assert (a["sport"], a["source"], a["venue_name"], a["city"], a["state"]) == ("cfb", "cfbd", "Bryant-Denny Stadium", "Tuscaloosa", "AL")
    assert (a["indoor"], a["temp_f"], a["wind_mph"], a["precip_in"], a["conditions"], a["weather_kind"]) == (False, 41.2, 14.0, 0.04, "Cloudy", "forecast")
    assert a["precip_chance"] is None and a["line_score"] is None
    b = rows[2]                                          # indoors: weather columns stay NULL
    assert b["indoor"] is True and b["temp_f"] is None and b["wind_mph"] is None and b["conditions"] is None and b["weather_kind"] is None
    assert b["city"] == "San Antonio"
    c = rows[3]                                          # kicked off already; venue not in venues.parquet -> CFBD's own name, no city
    assert c["weather_kind"] == "observed" and c["venue_name"] == "Unlisted Field" and c["city"] is None and c["state"] is None
    assert c["line_score"] == {"home": [7, 7, 7, 7], "away": [0, 0, 0, 3]} and c["indoor"] is False
    assert all(r["captured_at"].isoformat().startswith("2026-10-07T14:00:00") for r in rows.values())


def test_cfb_line_score_only_row_and_no_inputs():
    only = gi.cfb_game_info([_g(9, "2026-10-06T00:00:00Z")], {9: {"home": [1, 2, 3, 4], "away": [4, 3, 2, 1]}},
                            WEATHER.iloc[0:0], None, NOW)
    assert len(only) == 1 and only[0]["line_score"]["home"] == [1, 2, 3, 4] and only[0]["venue_name"] is None and only[0]["indoor"] is None
    assert gi.cfb_game_info([], {}, WEATHER, VENUES, NOW) == []


def test_cfb_line_score_drops_ot_keys():
    ls = {5: {"home": [7, 0, 7, 3], "away": [3, 7, 0, 7], "home_ot": 6, "away_ot": 0}}
    (row,) = gi.cfb_game_info([_g(5, "2026-10-06T00:00:00Z")], ls, WEATHER.iloc[0:0], None, NOW)
    assert row["line_score"] == {"home": [7, 0, 7, 3], "away": [3, 7, 0, 7]}


def _info(**kw):
    base = {"venue_name": "Highmark Stadium", "city": "Orchard Park", "state": "NY", "indoor": False,
            "temp_f": None, "wind_mph": None, "precip_chance": None, "conditions": None}
    return {**base, **kw}


def test_nfl_rows_window_dedupe_weather_kind_and_errors():
    calls, warns = [], []

    def fetch(pk):
        calls.append(pk)
        if pk == 13:
            raise RuntimeError("boom")
        return {10: _info(temp_f=41, wind_mph=14, precip_chance=20, conditions="Cloudy"), 11: _info(), 12: _info(indoor=True, venue_name="Dome")}[pk]

    games = [_g(10, "2026-10-11T17:00:00Z"), _g(10, "2026-10-11T17:00:00Z"), _g(11, "2026-10-12T00:20:00Z"),
             _g(12, "2026-10-11T20:25:00Z"), _g(13, "2026-10-11T20:25:00Z"), _g(14, "2027-01-01T00:00:00Z")]
    rows = {r["game_pk"]: r for r in gi.nfl_game_info(games, fetch, NOW, warns.append)}
    assert sorted(calls) == [10, 11, 12, 13] and set(rows) == {10, 11, 12}      # duplicate and out-of-window not fetched
    assert len(warns) == 1 and "13" in warns[0]
    a = rows[10]
    assert (a["sport"], a["source"], a["indoor"], a["temp_f"], a["wind_mph"], a["precip_chance"], a["weather_kind"]) == ("nfl", "espn", False, 41.0, 14.0, 20.0, "forecast")
    assert rows[11]["weather_kind"] is None and rows[11]["temp_f"] is None and rows[11]["city"] == "Orchard Park"
    assert rows[12]["indoor"] is True and rows[12]["venue_name"] == "Dome"
