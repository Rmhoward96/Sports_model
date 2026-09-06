"""CollegeFootballData (CFBD) adapter: forward-looking preseason signals.

Pure response parsers for the CFBD endpoints that feed the preseason priors
(SP+ rating, returning production, recruiting, transfer portal, coaching
changes, schedule for strength-of-schedule). Every parser is a pure function
over already-decoded JSON -- no network, no DB, no file reads -- and keys its
output by CFBD school display name (the same "team"/"school" strings CFBD
returns). Mapping those names to ESPN team ids happens later, in
`cfb.teams.cfbd_to_espn` (a downstream task's job, not this module's).

`_get` is the only network-touching piece here, and it is used solely by a
later ingest step's main() -- never by the parse_* functions above. It
mirrors `cfb.espn._get` / `nfl.espn._get`'s retry policy (tenacity: 3
attempts, exponential backoff) but adds the CFBD Bearer-token auth header.
"""
from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_BASE = "https://api.collegefootballdata.com"

# v1 proxy threshold for "QB production is returning": fraction of a team's
# passing-game PPA production (CFBD's percentPassingPPA field on
# /player/returning) attributable to players who are back this season. True
# starter-level QB continuity tracking (e.g. matching the specific returning
# passer to last year's starts) is a later increment; >=0.5 is a documented
# stand-in for "the guy who threw most of the passes is still here".
QB_RETURNING_THRESHOLD = 0.5


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, params: dict | None = None) -> Any:
    """GET {_BASE}{path} with CFBD Bearer auth and return parsed JSON, retrying
    transient failures.

    Reads the API key from `CFBD_API_KEY` on every call (not cached at import
    time) so tests can monkeypatch the environment. tenacity retries any
    raised exception (connect/read timeouts and raise_for_status errors) 3
    times with exponential backoff -- same retry policy as cfb.espn._get /
    nfl.espn._get."""
    headers = {"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"}
    r = httpx.get(f"{_BASE}{path}", params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_sp(payload) -> dict[str, float]:
    """SP+ overall rating by team, from CFBD `/ratings/sp` (list of
    {"team": ..., "rating": ..., ...})."""
    return {row["team"]: row["rating"] for row in payload}


def parse_returning(payload) -> dict[str, dict]:
    """Returning-production signals by team, from CFBD `/player/returning`
    (list of {"team", "percentPPA", "usage", "percentPassingPPA", ...}).

    Per team, returns:
      - "returning_pct": overall percent production returning (percentPPA) --
        the headline "returning production" figure most CFB analysis quotes.
      - "returning_starters": overall returning usage (usage) -- CFBD's
        snap-share-weighted measure of returning-player usage, used as the
        numeric returning-starters proxy the downstream rating consumes.
      - "qb_returning": bool proxy for "the passing game's production is
        back" = percentPassingPPA >= QB_RETURNING_THRESHOLD (0.5). This is a
        team-level production proxy, not confirmed returning-starter
        identity; true QB-starter tracking is a later increment.
    """
    out = {}
    for row in payload:
        out[row["team"]] = {
            "returning_pct": row["percentPPA"],
            "returning_starters": row["usage"],
            "qb_returning": row["percentPassingPPA"] >= QB_RETURNING_THRESHOLD,
        }
    return out


def parse_recruiting(payload) -> dict[str, float]:
    """Recruiting class strength by team, from CFBD `/recruiting/teams` (list
    of {"team", "points", "rank", ...}). Uses `points` (the composite class
    score) rather than `rank`, since points is continuous and comparable
    across classes/years while rank is not."""
    return {row["team"]: row["points"] for row in payload}


def parse_portal(payload) -> dict[str, dict]:
    """Transfer-portal net rating by team, from CFBD `/player/portal` (list
    of {"origin", "destination", "rating", ...}). Sums `rating` into "in" for
    each destination school and "out" for each origin school; "net" is
    in - out (rounded to 4 decimals). Every school appearing as either an
    origin or a destination gets an entry, defaulting the side it never
    appears on to 0."""
    out: dict[str, dict] = {}

    def _entry(team: str) -> dict:
        return out.setdefault(team, {"in": 0, "out": 0})

    for row in payload:
        rating = row["rating"]
        _entry(row["destination"])["in"] += rating
        _entry(row["origin"])["out"] += rating

    for team, sides in out.items():
        sides["net"] = round(sides["in"] - sides["out"], 4)
    return out


def parse_coaches(payload, season: int) -> dict[str, bool]:
    """First-year-head-coach flag by school for `season`, from CFBD
    `/coaches` (list of {"seasons": [{"school", "year", ...}, ...]} per
    coach -- a coach's `seasons` list spans every school/year they've
    coached, so the school is per-season, not per-coach).

    For each school with a season entry in the target `season`, the flag is
    True iff that season is the earliest year that coach's record shows at
    that school (i.e. this is their first season there)."""
    out: dict[str, bool] = {}
    for coach in payload:
        years_by_school: dict[str, list[int]] = {}
        for s in coach.get("seasons", []):
            school, year = s.get("school"), s.get("year")
            if school is None or year is None:
                continue
            years_by_school.setdefault(school, []).append(year)
        for school, years in years_by_school.items():
            if season in years:
                out[school] = min(years) == season
    return out


def parse_games(payload) -> list[dict]:
    """Matchups for strength-of-schedule, from CFBD `/games` (list of
    {"homeTeam", "awayTeam", "season", ...}). Returns one dict per game with
    just the fields SoS needs: home_team, away_team, season."""
    return [
        {"home_team": g["homeTeam"], "away_team": g["awayTeam"], "season": g["season"]}
        for g in payload
    ]
