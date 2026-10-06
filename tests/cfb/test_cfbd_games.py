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


# ------------------------------------------- missing identity keys are dropped --

def _without(row, key):
    return {k: v for k, v in row.items() if k != key}


@pytest.mark.parametrize("key", ["id", "season", "week"])
@pytest.mark.parametrize("how", ["missing", "null"])
def test_games_and_weather_drop_rows_missing_identity_keys(key, how):
    for parse, row in ((cg.parse_games_meta, GAMES[0]), (cg.parse_weather_games, _w(401, "Alabama", "Georgia"))):
        row = dict(row, id=401)
        bad = _without(row, key) if how == "missing" else dict(row, **{key: None})
        df = parse([bad, row])
        assert len(df) == 1 and df.attrs["dropped"] == 1, (parse.__name__, key, how)


@pytest.mark.parametrize("key", ["gameId", "season", "week"])
@pytest.mark.parametrize("how", ["missing", "null"])
def test_havoc_drops_rows_missing_identity_keys(key, how):
    row = HAVOC[0]
    bad = _without(row, key) if how == "missing" else dict(row, **{key: None})
    df = cg.parse_havoc_games([bad, row])
    assert len(df) == 1 and df.attrs["dropped"] == 1


@pytest.mark.parametrize("how", ["missing", "null"])
def test_drives_drop_rows_missing_game_id(how):
    row = DRIVES[0]
    bad = _without(row, "gameId") if how == "missing" else dict(row, gameId=None)
    df = cg.parse_drive_games([bad, row], META)
    assert df.attrs["dropped"] == 1 and len(df) == 1


@pytest.mark.parametrize("how", ["missing", "null"])
def test_talent_drops_rows_missing_year(how):
    ok = {"year": 2024, "team": "Georgia", "talent": 988.2}
    bad = _without(ok, "year") if how == "missing" else dict(ok, year=None)
    df = cg.parse_talent([bad, ok])
    assert len(df) == 1 and df.attrs["dropped"] == 1


def test_prior_ratings_count_unmapped_fpi_rows_as_dropped():
    df = cg.parse_prior_ratings([{"team": "Georgia", "fpi": 28.1},
                                 {"team": "Nowhere State Fighting Pickles", "fpi": 1.0}],
                                [{"team": "Georgia", "rating": 20.5}], 2023)
    assert len(df) == 1 and df.attrs["dropped"] == 1


def test_cfbd_module_docstring_describes_the_client():
    assert "CfbdClient" in cfbd.__doc__


AZ, OKST = cfbd_to_espn("Arizona"), cfbd_to_espn("Oklahoma State")

def test_line_scores():
    assert cg.line_scores([0, 10, 3, 24]) == [0.0, 10.0, 3.0, 24.0, 0.0]
    assert cg.line_scores([7, 7, 7, 3, 7, 6]) == [7.0, 7.0, 7.0, 3.0, 13.0]
    for bad in (None, [], [1, 2, 3], [1, 2, None, 4], "x"):
        assert all(math.isnan(x) for x in cg.line_scores(bad))

def test_games_line_scores():
    df = cg.parse_games_meta([
        {"id": 1, "season": 2025, "week": 6, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
         "homePoints": 37, "awayPoints": 10, "homeLineScores": [0, 10, 3, 24], "awayLineScores": [0, 3, 7, 0]},
        {"id": 2, "season": 2025, "week": 6, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
         "homePoints": 31, "awayPoints": 27, "homeLineScores": [7, 7, 7, 3, 7], "awayLineScores": [7, 7, 7, 3, 3]},
        {"id": 3, "season": 2025, "week": 7, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
         "homeLineScores": None, "awayLineScores": None}])
    a, b, c = df.iloc[0], df.iloc[1], df.iloc[2]
    assert [a["home_q1"], a["home_q2"], a["home_q3"], a["home_q4"], a["home_ot"]] == [0, 10, 3, 24, 0]
    assert (b["home_ot"], b["away_ot"]) == (7.0, 3.0)
    assert math.isnan(c["home_q1"]) and math.isnan(c["away_ot"])
    assert list(df.columns) == cg.GAMES_COLUMNS

def test_venue_city_state():
    df = cg.parse_venues([{"id": 3657, "name": "Bryant-Denny Stadium", "city": "Tuscaloosa", "state": "AL"},
                          {"id": 9, "name": "Somewhere", "city": "Dublin", "state": None}])
    assert list(df["city"]) == ["Tuscaloosa", "Dublin"] and list(df["state"]) == ["AL", ""]

def _t(name, ha, stats): return {"teamId": 1, "team": name, "homeAway": ha, "points": 1, "stats": [{"category": k, "stat": str(v)} for k, v in stats.items()]}

def test_team_stats():
    pay = [{"id": 401756910, "teams": [_t("Arizona", "home", {"turnovers": 3, "fumblesLost": 1, "interceptions": 2}),
                                       _t("Oklahoma State", "away", {"turnovers": 1, "fumblesLost": 1})]},
           {"id": 5, "teams": [_t("Arizona", "home", {}), _t("Nowhere State Fighting Pickles", "away", {"turnovers": 2})]},
           {"id": 6, "teams": [_t("Arizona", "home", {"fumblesLost": 1, "interceptions": 0}), _t("Oklahoma State", "away", {})]}]
    df = cg.parse_team_game_stats(pay, 2025, 6)
    assert df.attrs["dropped"] == 1 and len(df) == 4
    az = df[(df.game_id == 401756910) & (df.team == AZ)].iloc[0]
    ok = df[(df.game_id == 401756910) & (df.team == OKST)].iloc[0]
    assert (az["giveaways"], az["takeaways"], az["fumbles_lost"], az["interceptions_thrown"]) == (3, 1, 1, 2)
    assert (ok["giveaways"], ok["takeaways"]) == (1, 3) and math.isnan(ok["interceptions_thrown"])
    g6 = df[(df.game_id == 6) & (df.team == AZ)].iloc[0]
    assert g6["giveaways"] == 1 and math.isnan(g6["takeaways"])
    assert set(df["season"]) == {2025} and set(df["week"]) == {6} and set(df["season_type"]) == {"regular"}
    assert list(cg.parse_team_game_stats([], 2025, 1).columns) == cg.TEAM_STAT_COLUMNS

def test_weather_window():
    pay = [{"id": 401862794, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-08T23:30:00.000Z",
            "gameIndoors": True, "homeTeam": "UTSA", "awayTeam": "South Florida", "venueId": 3604, "venue": "Alamodome",
            "temperature": 82.8, "windSpeed": 9.2, "precipitation": 0, "weatherCondition": "Fair"},
           {"id": 401908581, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z",
            "gameIndoors": False, "homeTeam": "Bridgewater State", "awayTeam": "Framingham State", "venueId": 5808,
            "venue": "Swenson", "temperature": 66, "windSpeed": 10.1, "precipitation": 0.035, "weatherCondition": None},
           {"id": 7, "season": 2026, "week": 6, "homeTeam": "Alabama", "awayTeam": "Mercer", "windSpeed": None},
           {"season": 2026, "week": 6}]
    df = cg.parse_weather_window(pay)
    assert len(df) == 3 and df.attrs["dropped"] == 1
    assert list(df["has_fbs"]) == [True, False, True]
    r = df.iloc[0]; assert (r["venue_id"], r["venue"], r["game_indoors"], r["condition"]) == (3604, "Alamodome", True, "Fair")
    assert df.iloc[1]["condition"] == "" and df.iloc[1]["precipitation"] == 0.035
    assert math.isnan(df.iloc[2]["temperature"]) and math.isnan(df.iloc[2]["venue_id"])


def test_games_line_scores_are_kept_only_when_verified_against_the_final_score():
    def g(i, hp, ap, hl, al, **kw):
        return {"id": i, "season": 2025, "week": 6, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
                "homePoints": hp, "awayPoints": ap, "homeLineScores": hl, "awayLineScores": al, **kw}
    df = cg.parse_games_meta([
        g(1, 31, 27, [7, 7, 7, 3, 7], [7, 7, 7, 3, 3], completed=True),      # valid OT game: kept
        g(2, 38, 10, [0, 10, 3, 24], [0, 3, 7, 0], completed=True),          # home periods sum to 37: dropped
        g(3, 14, 7, [7, 7, 0, 0], [0, 7, 0, 0], completed=False),            # in progress: dropped
        g(4, None, None, [7, 7, 0, 0], [0, 7, 0, 0]),                        # no final points: dropped
        g(5, 31, 27, [7, 7, 7, 3, None], [7, 7, 7, 3, 3], completed=True)])  # null OT period: sum unverifiable
    cols = cg.LINE_SCORE_COLUMNS
    assert list(df.iloc[0][cols]) == [7, 7, 7, 3, 7, 7, 7, 7, 3, 3]
    for i in range(1, 5):
        assert all(math.isnan(x) for x in df.iloc[i][cols]), i


def test_team_stats_giveaways_prefer_the_component_sum_over_turnovers():
    pay = [{"id": 8, "teams": [_t("Arizona", "home", {"turnovers": 5, "fumblesLost": 1, "interceptions": 2}),
                               _t("Oklahoma State", "away", {"turnovers": 1, "fumblesLost": 1, "interceptions": 0})]}]
    df = cg.parse_team_game_stats(pay, 2025, 6)
    az, ok = df[df.team == AZ].iloc[0], df[df.team == OKST].iloc[0]
    assert (az["giveaways"], az["takeaways"]) == (3, 1) and (ok["giveaways"], ok["takeaways"]) == (1, 3)
