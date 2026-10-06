"""Pure parsers for the per-game CFBD endpoints behind cfb-ratings-v3.

Every parser takes an already-decoded JSON payload (no network, no file reads)
and returns a pandas DataFrame with one row per team-game (or game / team /
venue). CFBD school names are mapped to ESPN ids with `cfbd_to_espn`; a row
whose team (or opponent) does not map to an FBS id is dropped and counted in
`df.attrs["dropped"]`. Missing / null numbers are NaN, never 0. Every
game-level frame carries season, week, season_type and game_id.

Response shapes follow the CFBD v2 OpenAPI document (api-docs.json). They have
NOT been verified against live responses with the project key; the parsers read
every field defensively (`.get`, NaN on anything non-numeric) and
`scripts/build_cfb_game_data.py --probe` prints the real key structure.
"""
from __future__ import annotations

import math

import pandas as pd

from sportsmodel.cfb.teams import cfbd_to_espn

NAN = float("nan")


def num(x) -> float:
    """float(x) for real numbers; NaN for None / bool / strings / anything else."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return NAN
    return float(x)


def _ints(g: dict, *keys: str):
    """Required identity keys of one CFBD record as ints, or None when any is missing /
    null / non-integer (the caller drops and counts the row instead of raising)."""
    out = []
    for k in keys:
        v = g.get(k)
        if v is None or isinstance(v, bool):
            return None
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            return None
    return out


def season_type(x) -> str:
    """CFBD seasonType -> 'regular' | 'postseason' (lower-cased pass-through otherwise)."""
    return str(x or "regular").lower()


def _frame(rows: list[dict], columns: list[str], ints=(), strs=(), bools=(), dropped: int = 0):
    df = pd.DataFrame(rows, columns=columns)
    for c in columns:
        if c in ints:
            df[c] = df[c].astype("int64")
        elif c in strs:
            df[c] = df[c].astype(str)
        elif c in bools:
            df[c] = df[c].astype(bool)
        else:
            df[c] = df[c].astype("float64")
    df.attrs["dropped"] = dropped
    return df


# --------------------------------------------------------------- games meta --

LINE_SCORE_PERIODS = ("q1", "q2", "q3", "q4", "ot")
LINE_SCORE_COLUMNS = [f"{side}_{p}" for side in ("home", "away") for p in LINE_SCORE_PERIODS]
GAMES_COLUMNS = ["game_id", "season", "week", "season_type", "start_date", "neutral_site",
                 "venue_id", "home_team", "away_team", "home_points", "away_points",
                 "home_pregame_elo", "away_pregame_elo"] + LINE_SCORE_COLUMNS


def line_scores(v) -> list[float]:
    """CFBD `homeLineScores` / `awayLineScores` -> [q1, q2, q3, q4, ot]. `ot` is the sum of every period
    after the fourth (0.0 when the game ended in regulation). All five are NaN unless the list holds at
    least four numeric entries (an unplayed game, or a completed one CFBD has no line score for)."""
    nan5 = [NAN] * 5
    if not isinstance(v, list) or len(v) < 4:
        return nan5
    vals = [num(x) for x in v]
    if any(math.isnan(x) for x in vals):
        return nan5
    return vals[:4] + [float(sum(vals[4:]))]


def _verified_line_scores(g: dict) -> list[float]:
    """The 10 `LINE_SCORE_COLUMNS` values of one /games record, or ten NaN. A line score is kept only for a
    completed game with both final points present, and only when each side's q1+q2+q3+q4+ot equals that
    side's points (CFBD's list can be partial, stale or mid-game); otherwise it is dropped whole."""
    hp, ap = num(g.get("homePoints")), num(g.get("awayPoints"))
    home, away = line_scores(g.get("homeLineScores")), line_scores(g.get("awayLineScores"))
    if (g.get("completed") is False or math.isnan(hp) or math.isnan(ap)
            or sum(home) != hp or sum(away) != ap):  # a NaN anywhere makes the sum compare False
        return [NAN] * 10
    return home + away


def parse_games_meta(payload) -> pd.DataFrame:
    """CFBD `/games` -> one row per FBS-vs-FBS game: ids, week, season_type, venue,
    neutral flag, final points (NaN until played) and CFBD's PRE-game Elo (leak-free;
    the post-game fields are deliberately not read)."""
    rows, dropped = [], 0
    for g in payload:
        h, a = cfbd_to_espn(g.get("homeTeam") or ""), cfbd_to_espn(g.get("awayTeam") or "")
        ids = _ints(g, "id", "season", "week")
        if not h or not a or ids is None:
            dropped += 1
            continue
        rows.append({"game_id": ids[0], "season": ids[1], "week": ids[2],
                     "season_type": season_type(g.get("seasonType")),
                     "start_date": str(g.get("startDate") or ""),
                     "neutral_site": bool(g.get("neutralSite")),
                     "venue_id": num(g.get("venueId")), "home_team": h, "away_team": a,
                     "home_points": num(g.get("homePoints")), "away_points": num(g.get("awayPoints")),
                     "home_pregame_elo": num(g.get("homePregameElo")),
                     "away_pregame_elo": num(g.get("awayPregameElo")),
                     **dict(zip(LINE_SCORE_COLUMNS, _verified_line_scores(g)))})
    return _frame(rows, GAMES_COLUMNS, ints=("game_id", "season", "week"),
                  strs=("season_type", "start_date", "home_team", "away_team"),
                  bools=("neutral_site",), dropped=dropped)


# -------------------------------------------------------------------- havoc --

_HAVOC_FIELDS = (("havoc", "havocRate"), ("front7_havoc", "frontSevenHavocRate"),
                 ("db_havoc", "dbHavocRate"), ("havoc_events", "totalHavocEvents"),
                 ("plays", "totalPlays"))
HAVOC_COLUMNS = (["season", "week", "season_type", "game_id", "team", "opponent"]
                 + [f"{u}_{k}" for u in ("off", "def") for k, _ in _HAVOC_FIELDS])


def parse_havoc_games(payload) -> pd.DataFrame:
    """CFBD `/stats/game/havoc` -> one row per team-game. `off_*` is the havoc the
    team's OFFENSE suffered, `def_*` the havoc its DEFENSE created, with rates per
    play (the unit objects also carry event counts and total plays)."""
    rows, dropped = [], 0
    for g in payload:
        team, opp = cfbd_to_espn(g.get("team") or ""), cfbd_to_espn(g.get("opponent") or "")
        ids = _ints(g, "season", "week", "gameId")
        if not team or not opp or ids is None:
            dropped += 1
            continue
        row = {"season": ids[0], "week": ids[1],
               "season_type": season_type(g.get("seasonType")),
               "game_id": ids[2], "team": team, "opponent": opp}
        for unit, key in (("off", "offense"), ("def", "defense")):
            obj = g.get(key) if isinstance(g.get(key), dict) else {}
            for k, src in _HAVOC_FIELDS:
                row[f"{unit}_{k}"] = num(obj.get(src))
        rows.append(row)
    return _frame(rows, HAVOC_COLUMNS, ints=("season", "week", "game_id"),
                  strs=("season_type", "team", "opponent"), dropped=dropped)


# ------------------------------------------------------------------- drives --

# CFBD driveResult values that are not real possessions (matched upper-case).
EXCLUDED_DRIVE_RESULTS = frozenset({"END OF HALF", "END OF GAME", "END OF 4TH QUARTER"})
SCORING_OPP_YARDS = 40          # a "scoring opportunity" = the drive reached the opp's 40
DRIVE_COLUMNS = (["season", "week", "season_type", "game_id", "team", "opponent"]
                 + [f"{u}_{k}" for u in ("off", "def")
                    for k in ("drives", "points", "ppd", "opps", "points_per_opp", "start_yd")])


def _drive_points(d: dict) -> float:
    s, e = num(d.get("startOffenseScore")), num(d.get("endOffenseScore"))
    return NAN if math.isnan(s) or math.isnan(e) else max(0.0, e - s)


def parse_drive_games(payload, games_meta: pd.DataFrame) -> pd.DataFrame:
    """CFBD `/drives` -> one row per team-game of drive efficiency.

    points = offense score at drive end minus at drive start; a scoring
    opportunity is a drive whose closest point to the opponent's goal line
    (min of start/end yards-to-goal) was inside the 40 (CFBD has no per-play
    drive progress, so this is an approximation); start_yd is the average own-
    yard-line start (100 - startYardsToGoal). Non-possession results and drives
    with unreadable scores are skipped. `games_meta` (parse_games_meta output)
    supplies season / week / season_type; drives of unknown games are dropped
    and counted. `def_*` is the opponent offense's value in the same game.
    """
    meta = games_meta.set_index("game_id")[["season", "week", "season_type"]].to_dict("index")
    agg: dict[tuple[int, str], dict] = {}
    opp_of: dict[tuple[int, str], str] = {}
    dropped = 0
    for d in payload:
        ids = _ints(d, "gameId")
        off, dfn = cfbd_to_espn(d.get("offense") or ""), cfbd_to_espn(d.get("defense") or "")
        if ids is None or not off or not dfn or ids[0] not in meta:
            dropped += 1
            continue
        gid = ids[0]
        if str(d.get("driveResult") or "").upper() in EXCLUDED_DRIVE_RESULTS:
            continue
        pts, ytg_s, ytg_e = _drive_points(d), num(d.get("startYardsToGoal")), num(d.get("endYardsToGoal"))
        if math.isnan(pts) or math.isnan(ytg_s) or math.isnan(ytg_e):
            continue
        a = agg.setdefault((gid, off), {"drives": 0, "points": 0.0, "opps": 0, "start": 0.0})
        a["drives"] += 1
        a["points"] += pts
        a["opps"] += int(min(ytg_s, ytg_e) <= SCORING_OPP_YARDS)
        a["start"] += 100.0 - ytg_s
        opp_of[(gid, off)] = dfn

    def unit(a: dict | None) -> dict:
        if not a:
            return {k: NAN for k in ("drives", "points", "ppd", "opps", "points_per_opp", "start_yd")}
        return {"drives": float(a["drives"]), "points": a["points"], "ppd": a["points"] / a["drives"],
                "opps": float(a["opps"]),
                "points_per_opp": a["points"] / a["opps"] if a["opps"] else NAN,
                "start_yd": a["start"] / a["drives"]}

    rows = []
    for (gid, team), a in sorted(agg.items()):
        opp = opp_of[(gid, team)]
        m = meta[gid]
        row = {"season": int(m["season"]), "week": int(m["week"]), "season_type": m["season_type"],
               "game_id": gid, "team": team, "opponent": opp}
        row.update({f"off_{k}": v for k, v in unit(a).items()})
        row.update({f"def_{k}": v for k, v in unit(agg.get((gid, opp))).items()})
        rows.append(row)
    return _frame(rows, DRIVE_COLUMNS, ints=("season", "week", "game_id"),
                  strs=("season_type", "team", "opponent"), dropped=dropped)


# ------------------------------------------------------------------ weather --

WEATHER_COLUMNS = ["game_id", "season", "week", "season_type", "start_time", "home_team",
                   "away_team", "venue_id", "game_indoors", "temperature", "wind_speed",
                   "precipitation", "snowfall", "humidity"]


def parse_weather_games(payload) -> pd.DataFrame:
    """CFBD `/games/weather` -> one row per FBS-vs-FBS game (historical observation
    or forecast). `game_indoors` is CFBD's dome flag; every numeric field is NaN when
    CFBD has no reading. Units are CFBD's own (expected: temperature F, wind mph,
    precipitation inches; checked by the backfill sanity step, not assumed here)."""
    rows, dropped = [], 0
    for w in payload:
        h, a = cfbd_to_espn(w.get("homeTeam") or ""), cfbd_to_espn(w.get("awayTeam") or "")
        ids = _ints(w, "id", "season", "week")
        if not h or not a or ids is None:
            dropped += 1
            continue
        rows.append({"game_id": ids[0], "season": ids[1], "week": ids[2],
                     "season_type": season_type(w.get("seasonType")),
                     "start_time": str(w.get("startTime") or ""), "home_team": h, "away_team": a,
                     "venue_id": num(w.get("venueId")), "game_indoors": bool(w.get("gameIndoors")),
                     "temperature": num(w.get("temperature")), "wind_speed": num(w.get("windSpeed")),
                     "precipitation": num(w.get("precipitation")), "snowfall": num(w.get("snowfall")),
                     "humidity": num(w.get("humidity"))})
    return _frame(rows, WEATHER_COLUMNS, ints=("game_id", "season", "week"),
                  strs=("season_type", "start_time", "home_team", "away_team"),
                  bools=("game_indoors",), dropped=dropped)


# ------------------------------------------------------------------- talent --

def parse_talent(payload) -> pd.DataFrame:
    """CFBD `/talent?year=Y` (247 team talent composite) -> season, team, talent."""
    rows, dropped = [], 0
    for t in payload:
        team = cfbd_to_espn(t.get("team") or "")
        ids = _ints(t, "year")
        if not team or ids is None or math.isnan(num(t.get("talent"))):
            dropped += 1
            continue
        rows.append({"season": ids[0], "team": team, "talent": num(t["talent"])})
    return _frame(rows, ["season", "team", "talent"], ints=("season",), strs=("team",), dropped=dropped)


# ------------------------------------------------------------------- venues --

VENUE_COLUMNS = ["venue_id", "name", "timezone", "latitude", "longitude", "elevation", "dome", "city", "state"]


def parse_venues(payload) -> pd.DataFrame:
    """CFBD `/venues` -> venue_id, name, IANA timezone, lat/lon, elevation (CFBD sends it
    as a string), dome (NaN when unknown), city and state ("" when CFBD has none; a few venues have no
    state). Venues without an id are skipped."""
    rows = []
    for v in payload:
        if v.get("id") is None:
            continue
        try:
            elev = float(v.get("elevation"))
        except (TypeError, ValueError):
            elev = NAN
        dome = v.get("dome")
        rows.append({"venue_id": int(v["id"]), "name": str(v.get("name") or ""),
                     "timezone": str(v.get("timezone") or ""), "latitude": num(v.get("latitude")),
                     "longitude": num(v.get("longitude")), "elevation": elev,
                     "dome": float(dome) if isinstance(dome, bool) else NAN,
                     "city": str(v.get("city") or ""), "state": str(v.get("state") or "")})
    return _frame(rows, VENUE_COLUMNS, ints=("venue_id",), strs=("name", "timezone", "city", "state"))


# ------------------------------------------------------------ prior ratings --

def parse_prior_ratings(fpi_payload, srs_payload, season: int) -> pd.DataFrame:
    """CFBD `/ratings/fpi` + `/ratings/srs` for `season` -> season, team, fpi, srs.
    These are END-of-season ratings: the residual check only ever joins them to the
    NEXT season's games (season + 1), never to in-season games."""
    fpi, dropped = {}, 0
    for r in fpi_payload:
        team = cfbd_to_espn(r.get("team") or "")
        if team:
            fpi[team] = num(r.get("fpi"))
        else:
            dropped += 1
    srs = {}
    for r in srs_payload:
        team = cfbd_to_espn(r.get("team") or "")
        if team:
            srs[team] = num(r.get("rating"))
        else:
            dropped += 1
    rows = [{"season": int(season), "team": t, "fpi": fpi.get(t, NAN), "srs": srs.get(t, NAN)}
            for t in sorted(set(fpi) | set(srs))]
    return _frame(rows, ["season", "team", "fpi", "srs"], ints=("season",), strs=("team",),
                  dropped=dropped)


# -------------------------------------------------------------- team stats --

TEAM_STAT_COLUMNS = ["season", "week", "season_type", "game_id", "team", "opponent",
                     "fumbles_lost", "interceptions_thrown", "giveaways", "takeaways"]


def _stat(stats, category: str) -> float:
    """One entry of a CFBD `/games/teams` stats list ([{"category": "turnovers", "stat": "3"}, ...]) as a
    float; NaN when the category is absent or its value is not a number (the values arrive as strings)."""
    for s in stats if isinstance(stats, list) else []:
        if isinstance(s, dict) and s.get("category") == category:
            try:
                return float(s.get("stat"))
            except (TypeError, ValueError):
                return NAN
    return NAN


def parse_team_game_stats(payload, season: int, week: int, stype: str = "regular") -> pd.DataFrame:
    """CFBD `/games/teams?year=&week=&seasonType=` -> one row per team-game of ball security.

    The payload is [{"id": gameId, "teams": [{"team", "homeAway", "stats": [{"category", "stat"}]}, x2]}] and
    carries no season / week, so the caller stamps `season`, `week` and `stype` (the request's own params).
    `giveaways` = fumblesLost + interceptions THROWN when both are present, else the `turnovers` stat (verified live that CFBD's
    `interceptions` is the offense's, `passesIntercepted` the defense's); `takeaways` = the opponent's
    giveaways in the same game. A missing stat is NaN, never 0. A game whose two sides do not both map to
    FBS ids (an FCS opponent) is dropped and counted in `df.attrs["dropped"]`, matching havoc / advanced."""
    rows, dropped = [], 0
    for g in payload:
        gid = _ints(g, "id")
        teams = g.get("teams") if isinstance(g.get("teams"), list) else []
        ids = [cfbd_to_espn((t or {}).get("team") or "") for t in teams]
        if gid is None or len(teams) != 2 or not all(ids):
            dropped += 1
            continue
        per = []
        for t in teams:
            st = t.get("stats")
            fl, ints, tot = _stat(st, "fumblesLost"), _stat(st, "interceptions"), _stat(st, "turnovers")
            if not (math.isnan(fl) or math.isnan(ints)):
                tot = fl + ints  # the components win over CFBD's `turnovers` when the two disagree
            per.append((fl, ints, tot))
        for i in (0, 1):
            rows.append({"season": int(season), "week": int(week), "season_type": season_type(stype),
                         "game_id": gid[0], "team": ids[i], "opponent": ids[1 - i],
                         "fumbles_lost": per[i][0], "interceptions_thrown": per[i][1],
                         "giveaways": per[i][2], "takeaways": per[1 - i][2]})
    return _frame(rows, TEAM_STAT_COLUMNS, ints=("season", "week", "game_id"),
                  strs=("season_type", "team", "opponent"), dropped=dropped)


# ---------------------------------------------------------- weather window --

WEATHER_WINDOW_COLUMNS = ["game_id", "season", "week", "season_type", "start_time", "venue_id", "venue",
                          "game_indoors", "temperature", "wind_speed", "precipitation", "condition", "has_fbs"]


def parse_weather_window(payload) -> pd.DataFrame:
    """CFBD `/games/weather` (forecast before kickoff, observation after) -> one row per game for the
    site's game_info table. Unlike `parse_weather_games` this keeps FBS-vs-FCS games (the site has pages
    for them): `has_fbs` says at least one side maps to an FBS id, and the caller drops the rest (CFBD returns
    every division). `venue_id` / `venue` come straight from the weather row (verified live: it carries both),
    `condition` is CFBD's weatherCondition text ("" when absent, which is most rows). CFBD sends no
    precipitation probability, only `precipitation` (inches), so no chance is derived here."""
    rows, dropped = [], 0
    for w in payload:
        ids = _ints(w, "id", "season", "week")
        if ids is None:
            dropped += 1
            continue
        h, a = cfbd_to_espn(w.get("homeTeam") or ""), cfbd_to_espn(w.get("awayTeam") or "")
        rows.append({"game_id": ids[0], "season": ids[1], "week": ids[2],
                     "season_type": season_type(w.get("seasonType")),
                     "start_time": str(w.get("startTime") or ""), "venue_id": num(w.get("venueId")),
                     "venue": str(w.get("venue") or ""), "game_indoors": bool(w.get("gameIndoors")),
                     "temperature": num(w.get("temperature")), "wind_speed": num(w.get("windSpeed")),
                     "precipitation": num(w.get("precipitation")),
                     "condition": str(w.get("weatherCondition") or ""), "has_fbs": bool(h or a)})
    return _frame(rows, WEATHER_WINDOW_COLUMNS, ints=("game_id", "season", "week"),
                  strs=("season_type", "start_time", "venue", "condition"),
                  bools=("game_indoors", "has_fbs"), dropped=dropped)
