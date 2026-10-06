"""Pure-parser tests for cfbd_games + the CfbdClient. No network: payloads are
inline dicts shaped like the CFBD v2 OpenAPI schema."""
import math

import httpx
import pandas as pd
import pytest
from tenacity import wait_none

from sportsmodel.cfb import cfbd, cfbd_games as cg
from sportsmodel.cfb.teams import cfbd_to_espn

BAMA, UGA = cfbd_to_espn("Alabama"), cfbd_to_espn("Georgia")


# ---------------------------------------------------------------- the client --

def _client(handler, **kw):
    http = httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler))
    return cfbd.CfbdClient("sekret", http=http, wait=wait_none(), **kw)


def test_client_sends_bearer_counts_calls_and_hides_key():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["Authorization"]
        seen["url"] = str(req.url)
        return httpx.Response(200, json=[{"a": 1}])

    c = _client(handler)
    assert c.get("/talent", {"year": 2024}) == [{"a": 1}]
    assert seen["auth"] == "Bearer sekret" and "sekret" not in seen["url"]
    assert c.calls == 1 and "1" in c.summary()
    assert "sekret" not in repr(c)


def test_client_retries_5xx_then_succeeds_and_counts_attempts():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(503) if n["i"] < 3 else httpx.Response(200, json={"ok": True})

    c = _client(handler)
    assert c.get("/x") == {"ok": True}
    assert c.calls == 3


def test_client_does_not_retry_4xx():
    c = _client(lambda req: httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        c.get("/x")
    assert c.calls == 1


def test_client_gives_up_after_attempts():
    c = _client(lambda req: httpx.Response(500), attempts=2)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("/x")
    assert c.calls == 2


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cfbd.CfbdClient.from_env()


# ---------------------------------------------------------------- games meta --

GAMES = [
    {"id": 401, "season": 2023, "week": 3, "seasonType": "regular", "startDate": "2023-09-16T16:00:00.000Z",
     "neutralSite": False, "venueId": 3657, "homeTeam": "Alabama", "awayTeam": "Georgia",
     "homePoints": 27, "awayPoints": 24, "homePregameElo": 2100, "awayPregameElo": None},
    {"id": 402, "season": 2023, "week": 3, "seasonType": "regular", "homeTeam": "Alabama",
     "awayTeam": "Nowhere State Fighting Pickles"},
    {"id": 403, "season": 2023, "week": 1, "seasonType": "postseason", "neutralSite": True,
     "homeTeam": "Georgia", "awayTeam": "Alabama", "homePoints": None, "awayPoints": None},
]


def test_games_meta_maps_teams_drops_fcs_and_nans():
    df = cg.parse_games_meta(GAMES)
    assert len(df) == 2 and df.attrs["dropped"] == 1
    r = df.iloc[0]
    assert (r["game_id"], r["home_team"], r["away_team"]) == (401, BAMA, UGA)
    assert r["season_type"] == "regular" and r["venue_id"] == 3657 and not r["neutral_site"]
    assert r["home_pregame_elo"] == 2100 and math.isnan(r["away_pregame_elo"])
    p = df.iloc[1]
    assert p["season_type"] == "postseason" and p["neutral_site"] and math.isnan(p["home_points"])
    assert math.isnan(p["venue_id"])


def test_games_meta_empty_keeps_schema():
    df = cg.parse_games_meta([])
    assert list(df.columns) == cg.GAMES_COLUMNS and len(df) == 0


# -------------------------------------------------------------------- havoc --

def _hv(rate, ev=10, plays=60):
    return {"havocRate": rate, "frontSevenHavocRate": rate / 2, "dbHavocRate": rate / 2,
            "totalHavocEvents": ev, "totalPlays": plays, "frontSevenHavocEvents": 4, "dbHavocEvents": 6}


HAVOC = [
    {"gameId": 401, "season": 2023, "seasonType": "regular", "week": 3, "team": "Alabama",
     "opponent": "Georgia", "offense": _hv(0.12), "defense": _hv(0.20)},
    {"gameId": 402, "season": 2023, "seasonType": "regular", "week": 3, "team": "Alabama",
     "opponent": "Nowhere State Fighting Pickles", "offense": _hv(0.1), "defense": _hv(0.1)},
    {"gameId": 404, "season": 2023, "seasonType": "regular", "week": 4, "team": "Georgia",
     "opponent": "Alabama", "offense": None, "defense": {"havocRate": None}},
]


def test_havoc_maps_units_and_drops():
    df = cg.parse_havoc_games(HAVOC)
    assert len(df) == 2 and df.attrs["dropped"] == 1
    r = df.iloc[0]
    assert (r["team"], r["opponent"], r["season_type"]) == (BAMA, UGA, "regular")
    assert r["off_havoc"] == 0.12 and r["def_havoc"] == 0.20
    assert r["off_front7_havoc"] == 0.06 and r["off_havoc_events"] == 10 and r["off_plays"] == 60


def test_havoc_missing_units_are_nan_not_zero():
    r = cg.parse_havoc_games(HAVOC).iloc[1]
    for c in ("off_havoc", "off_plays", "def_havoc", "def_front7_havoc"):
        assert math.isnan(r[c]), c


# ------------------------------------------------------------------- drives --

META = cg.parse_games_meta(GAMES)


def _drive(gid, off, dfn, s_ytg, e_ytg, s_pts, e_pts, result="PUNT", opp_score=0):
    return {"gameId": gid, "offense": off, "defense": dfn, "startYardsToGoal": s_ytg,
            "endYardsToGoal": e_ytg, "startOffenseScore": s_pts, "endOffenseScore": e_pts,
            "driveResult": result, "plays": 6, "yards": 40}


DRIVES = [
    _drive(401, "Alabama", "Georgia", 75, 0, 0, 7, "TD"),          # scores from own 25: opp, 7 pts
    _drive(401, "Alabama", "Georgia", 70, 45, 7, 7, "PUNT"),       # stalled at the 45: no opp
    _drive(401, "Alabama", "Georgia", 60, 22, 7, 10, "FG"),        # FG: opp, 3 pts
    _drive(401, "Georgia", "Alabama", 80, 35, 0, 0, "FUMBLE"),     # reached the 35: opp, 0 pts
    _drive(401, "Alabama", "Georgia", 90, 90, 10, 10, "END OF HALF"),  # excluded
    _drive(401, "Alabama", "Nowhere State Fighting Pickles", 70, 0, 0, 7, "TD"),  # unmapped -> dropped
    _drive(999, "Alabama", "Georgia", 70, 0, 0, 7, "TD"),          # game not in meta -> dropped
    {"gameId": 401, "offense": "Georgia", "defense": "Alabama", "startYardsToGoal": 70,
     "endYardsToGoal": 60, "startOffenseScore": None, "endOffenseScore": None, "driveResult": "PUNT"},
]


def test_drive_games_aggregate_points_and_opps():
    df = cg.parse_drive_games(DRIVES, META)
    assert df.attrs["dropped"] == 2
    a = df[df["team"] == BAMA].iloc[0]
    assert (a["season"], a["week"], a["season_type"], a["game_id"]) == (2023, 3, "regular", 401)
    assert a["off_drives"] == 3 and a["off_points"] == 10
    assert a["off_opps"] == 2 and a["off_points_per_opp"] == 5.0
    assert a["off_ppd"] == pytest.approx(10 / 3)
    assert a["off_start_yd"] == pytest.approx(((100 - 75) + (100 - 70) + (100 - 60)) / 3)


def test_drive_games_defense_is_opponents_offense_and_unreadable_scores_skipped():
    df = cg.parse_drive_games(DRIVES, META)
    a, g = df[df["team"] == BAMA].iloc[0], df[df["team"] == UGA].iloc[0]
    assert a["def_drives"] == g["off_drives"] == 1      # the None-score Georgia drive was skipped
    assert g["off_opps"] == 1 and g["off_points_per_opp"] == 0.0
    assert g["def_points"] == a["off_points"]


def test_drive_games_no_opps_gives_nan_not_zero():
    d = [_drive(401, "Alabama", "Georgia", 80, 70, 0, 0, "PUNT")]
    r = cg.parse_drive_games(d, META).iloc[0]
    assert r["off_opps"] == 0 and math.isnan(r["off_points_per_opp"])


# ------------------------------------------------------------------ weather --

def _w(gid, home, away, **kw):
    base = {"id": gid, "season": 2023, "week": 3, "seasonType": "regular",
            "startTime": "2023-09-16T16:00:00.000Z", "gameIndoors": False, "homeTeam": home,
            "awayTeam": away, "venueId": 3657, "venue": "Bryant-Denny", "temperature": 71.0,
            "windSpeed": 9.5, "precipitation": 0.0, "snowfall": None, "humidity": 55.0}
    base.update(kw)
    return base


def test_weather_parse_and_nan_for_missing():
    df = cg.parse_weather_games([_w(401, "Alabama", "Georgia"),
                                 _w(402, "Alabama", "Georgia", temperature=None, windSpeed=None, gameIndoors=True),
                                 _w(403, "Alabama", "Nowhere State Fighting Pickles")])
    assert len(df) == 2 and df.attrs["dropped"] == 1
    r0, r1 = df.iloc[0], df.iloc[1]
    assert (r0["home_team"], r0["away_team"], r0["venue_id"]) == (BAMA, UGA, 3657)
    assert r0["temperature"] == 71.0 and r0["wind_speed"] == 9.5 and not r0["game_indoors"]
    assert math.isnan(r0["snowfall"])
    assert r1["game_indoors"] and math.isnan(r1["temperature"]) and math.isnan(r1["wind_speed"])


# ------------------------------------------------- talent / venues / ratings --

def test_talent_maps_and_drops_unmapped_or_null():
    df = cg.parse_talent([{"year": 2024, "team": "Georgia", "talent": 988.2},
                          {"year": 2024, "team": "Nowhere State Fighting Pickles", "talent": 200.0},
                          {"year": 2024, "team": "Alabama", "talent": None}])
    assert len(df) == 1 and df.attrs["dropped"] == 2
    assert df.iloc[0]["team"] == UGA and df.iloc[0]["talent"] == 988.2 and df.iloc[0]["season"] == 2024


def test_venues_parse_coords_elevation_string_and_dome():
    df = cg.parse_venues([
        {"id": 1, "name": "Dome", "timezone": "America/Chicago", "latitude": 29.7, "longitude": -95.4,
         "elevation": "12.3", "dome": True},
        {"id": 2, "name": "Open", "timezone": None, "latitude": None, "longitude": None,
         "elevation": None, "dome": None},
        {"id": None, "name": "no id"}])
    assert len(df) == 2
    d, o = df.iloc[0], df.iloc[1]
    assert d["timezone"] == "America/Chicago" and d["elevation"] == 12.3 and d["dome"] == 1.0
    assert math.isnan(o["latitude"]) and math.isnan(o["elevation"]) and math.isnan(o["dome"])


def test_prior_ratings_join_fpi_and_srs_by_team():
    df = cg.parse_prior_ratings([{"year": 2023, "team": "Georgia", "fpi": 28.1},
                                 {"year": 2023, "team": "Alabama", "fpi": None}],
                                [{"year": 2023, "team": "Georgia", "rating": 20.5},
                                 {"year": 2023, "team": "Nowhere State Fighting Pickles", "rating": 1.0}], 2023)
    u = df[df["team"] == UGA].iloc[0]
    assert (u["season"], u["fpi"], u["srs"]) == (2023, 28.1, 20.5)
    b = df[df["team"] == BAMA].iloc[0]
    assert math.isnan(b["fpi"]) and math.isnan(b["srs"])
    assert df.attrs["dropped"] == 1
