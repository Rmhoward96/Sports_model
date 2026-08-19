"""MLB StatsAPI (statsapi.mlb.com) — free, no key. Schedules + probable pitchers.

Used for the daily current-data pull: today's / tomorrow's slate, probable pitchers,
venue, and status. See docs — this is the official-internal API powering MLB.com.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

BASE = "https://statsapi.mlb.com/api/v1"


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    with httpx.Client(timeout=20) as client:
        resp = client.get(f"{BASE}{path}", params=params)
        resp.raise_for_status()
        return resp.json()


def _parse_game(g: dict[str, Any]) -> dict[str, Any]:
    home = g["teams"]["home"]
    away = g["teams"]["away"]

    def probable(side: dict[str, Any]) -> tuple[int | None, str | None]:
        pp = side.get("probablePitcher") or {}
        return pp.get("id"), pp.get("fullName")

    home_pp_id, home_pp_name = probable(home)
    away_pp_id, away_pp_name = probable(away)
    return {
        "game_pk": g["gamePk"],
        "game_date": g["officialDate"],
        "status": g["status"]["detailedState"],
        "venue_id": g.get("venue", {}).get("id"),
        "venue_name": g.get("venue", {}).get("name"),
        "home_team_id": home["team"]["id"],
        "home_team_name": home["team"]["name"],
        "away_team_id": away["team"]["id"],
        "away_team_name": away["team"]["name"],
        "home_probable_pitcher_id": home_pp_id,
        "home_probable_pitcher_name": home_pp_name,
        "away_probable_pitcher_id": away_pp_id,
        "away_probable_pitcher_name": away_pp_name,
    }


def fetch_schedule(date: str) -> list[dict[str, Any]]:
    """Return one flat record per game on `date` (YYYY-MM-DD), incl. probable pitchers."""
    data = _get(
        "/schedule",
        {"sportId": 1, "date": date, "hydrate": "probablePitcher,team,venue"},
    )
    games: list[dict[str, Any]] = []
    for day in data.get("dates", []):
        for g in day.get("games", []):
            games.append(_parse_game(g))
    return games
