"""Actual game + player results from MLB StatsAPI boxscores (for grading).

Returns final runs and each player's realized stats in our prop-market terms, so
predictions can be graded against reality. Available once a game is final.
"""
from __future__ import annotations

from typing import Any

from .mlb_statsapi import _get


def _ip_to_outs(ip) -> int:
    """MLB innings-pitched like '5.2' -> outs (5*3 + 2)."""
    if ip is None:
        return 0
    whole, _, frac = str(ip).partition(".")
    return int(whole or 0) * 3 + int(frac or 0)


def fetch_results(game_pk: int) -> dict[str, Any] | None:
    """Final scores + per-player actuals, or None if the game isn't final yet.

    batters[player_id]  = {name, hits, total_bases, home_run, hrr}
    pitchers[player_id] = {name, pitcher_ks, hits_allowed, outs_recorded}
    """
    data = _get(f"/game/{game_pk}/boxscore", {})
    teams = data.get("teams", {})
    home, away = teams.get("home", {}), teams.get("away", {})
    home_runs = home.get("teamStats", {}).get("batting", {}).get("runs")
    away_runs = away.get("teamStats", {}).get("batting", {}).get("runs")
    if home_runs is None or away_runs is None:
        return None  # not final / no box yet

    batters: dict[int, dict] = {}
    pitchers: dict[int, dict] = {}
    for side in (home, away):
        for p in side.get("players", {}).values():
            pid = p.get("person", {}).get("id")
            name = p.get("person", {}).get("fullName", "")
            stats = p.get("stats", {})
            b = stats.get("batting", {})
            if b and (b.get("plateAppearances") or b.get("atBats")):
                h = b.get("hits", 0); d = b.get("doubles", 0)
                t = b.get("triples", 0); hr = b.get("homeRuns", 0)
                batters[pid] = {
                    "name": name,
                    "hits": h,
                    "total_bases": h + d + 2 * t + 3 * hr,  # TB = H + 2B + 2*3B + 3*HR
                    "home_run": hr,
                    "hrr": h + b.get("runs", 0) + b.get("rbi", 0),
                }
            pit = stats.get("pitching", {})
            if pit and (pit.get("outs") or pit.get("inningsPitched")):
                outs = pit.get("outs")
                if outs is None:
                    outs = _ip_to_outs(pit.get("inningsPitched"))
                pitchers[pid] = {
                    "name": name,
                    "pitcher_ks": pit.get("strikeOuts", 0),
                    "hits_allowed": pit.get("hits", 0),
                    "outs_recorded": outs,
                }
    return {"home_runs": home_runs, "away_runs": away_runs,
            "batters": batters, "pitchers": pitchers}
