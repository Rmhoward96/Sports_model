"""Batting lineups from MLB StatsAPI — confirmed when posted, projected otherwise.

Confirmed: a game's boxscore `battingOrder` (player ids in slot order) populates a
few hours before first pitch. Projected: fall back to the team's most recent final
game's batting order. Props read whichever is available (confirmed preferred).
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .mlb_statsapi import _get


def fetch_confirmed_batting(game_pk: int) -> dict[str, list[tuple[int, str]]]:
    """{'home': [(player_id, name), ...], 'away': [...]} in order; empty if not posted."""
    data = _get(f"/game/{game_pk}/boxscore", {})
    out: dict[str, list[tuple[int, str]]] = {}
    for side in ("home", "away"):
        team = data.get("teams", {}).get(side, {})
        order = team.get("battingOrder", []) or []
        players = team.get("players", {})
        seq = []
        for pid in order:
            name = players.get(f"ID{pid}", {}).get("person", {}).get("fullName", "")
            seq.append((int(pid), name))
        out[side] = seq
    return out


def fetch_last_batting(team_id: int, before: date, lookback_days: int = 14) -> list[tuple[int, str]]:
    """The team's batting order from its most recent final game before `before`."""
    data = _get("/schedule", {
        "sportId": 1, "teamId": team_id,
        "startDate": (before - timedelta(days=lookback_days)).isoformat(),
        "endDate": (before - timedelta(days=1)).isoformat(),
    })
    finals: list[tuple[str, int, str]] = []  # (date, game_pk, side)
    for day in data.get("dates", []):
        for g in day.get("games", []):
            if g.get("status", {}).get("abstractGameState") != "Final":
                continue
            side = "home" if g["teams"]["home"]["team"]["id"] == team_id else "away"
            finals.append((g["officialDate"], g["gamePk"], side))
    if not finals:
        return []
    finals.sort()
    _, pk, side = finals[-1]
    return fetch_confirmed_batting(pk).get(side, [])


def lineups_for_game(game_pk: int, home_team_id: int, away_team_id: int,
                     game_date: str) -> dict[str, Any]:
    """Best available lineup per side: confirmed if posted, else projected.

    Returns {'home': {...}, 'away': {...}} where each is
    {'batting_order': [player_id,...], 'source': 'confirmed'|'projected'}.
    """
    confirmed = fetch_confirmed_batting(game_pk)
    gd = date.fromisoformat(str(game_date))
    result: dict[str, Any] = {}
    for side, team_id in (("home", home_team_id), ("away", away_team_id)):
        order = confirmed.get(side, [])
        if order:
            result[side] = {"batting_order": order, "source": "confirmed"}
        else:
            result[side] = {
                "batting_order": fetch_last_batting(team_id, gd),
                "source": "projected",
            }
    return result
