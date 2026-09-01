"""ESPN college-football schedule/score adapter.

Mirrors sportsmodel.nfl.espn's structure and STATUS_FINAL gate. The only
differences: the base path is college-football, and teams are normalized via
cfb.teams.normalize (FBS ESPN team id passthrough, everything else -> "FCS").
"""
from __future__ import annotations

import httpx

from .teams import normalize

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"


def _competitors(event) -> dict:
    comp = event["competitions"][0]["competitors"]
    return {c["homeAway"]: c for c in comp}


def _score(competitor: dict) -> int | None:
    raw = competitor.get("score")
    if raw is None or raw == "":
        return None
    return int(raw)


def parse_schedule(payload) -> list[dict]:
    default_season = payload.get("season", {}).get("year")
    default_week = payload.get("week", {}).get("number")
    out = []
    for ev in payload.get("events", []):
        c = _competitors(ev)
        season = ev.get("season", {}).get("year", default_season)
        week = ev.get("week", {}).get("number", default_week)
        out.append({
            "game_pk": int(ev["id"]),
            "home_team": normalize(c["home"]["team"]["id"]),
            "away_team": normalize(c["away"]["team"]["id"]),
            "home_name": c["home"]["team"].get("displayName"),
            "away_name": c["away"]["team"].get("displayName"),
            "home_score": _score(c["home"]),
            "away_score": _score(c["away"]),
            "commence_time": ev["date"],
            "status": ev["status"]["type"]["name"],
            "week": week,
            "season": season,
        })
    return out


def parse_final(event) -> dict | None:
    if event["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    c = _competitors(event)
    return {"home_score": int(c["home"]["score"]),
            "away_score": int(c["away"]["score"]), "final": True}


def fetch_schedule(season: int, week: int, season_type: int = 2) -> list[dict]:
    r = httpx.get(f"{_BASE}/scoreboard",
                  params={"dates": season, "seasontype": season_type,
                          "week": week, "groups": 80},
                  timeout=20)
    r.raise_for_status()
    return parse_schedule(r.json())
