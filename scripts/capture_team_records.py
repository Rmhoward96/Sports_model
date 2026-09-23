"""Capture Action Network season betting records (ATS / O-U / units by split)
for NFL + CFB into team_betting_records. One actor call per league with
includeStandings (maxGames 1 keeps it cheap). Daily, before the desk.

Usage: APIFY_TOKEN=... DATABASE_URL=... uv run python scripts/capture_team_records.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import upsert_team_betting_records
from sportsmodel.nfl.action_network import fetch_splits, parse_team_records
from sportsmodel.serving.team_names import match_team_names

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {"nfl": "nfl", "cfb": "ncaaf"}


def our_team_names(sport: str) -> list[str]:
    path = ROOT / "assets" / ("nfl/nfl_teams.json" if sport == "nfl" else "cfb/fbs_teams.json")
    return list(json.loads(path.read_text()).values())


def current_season(sport: str) -> int:
    if sport == "nfl":
        from sportsmodel.nfl import espn
    else:
        from sportsmodel.cfb import espn
    return int(espn.fetch_current_week()["season"])


def resolve_names(rows: list[dict], ours: list[str]) -> tuple[list[dict], list[str]]:
    """Attach team_name (our name when matched, else the AN name); return the
    unmatched AN names for logging. PURE."""
    m = match_team_names([r["an_team_name"] for r in rows], ours)
    unmatched = [r["an_team_name"] for r in rows if r["an_team_name"] not in m]
    return [{**r, "team_name": m.get(r["an_team_name"], r["an_team_name"])} for r in rows], unmatched


def main() -> None:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise SystemExit("APIFY_TOKEN required")
    for sport, league in LEAGUES.items():
        try:
            season = current_season(sport)
            items = fetch_splits(token, leagues=(league,), game_status=("scheduled",),
                                 extra_input={"season": season, "maxGames": 1, "includeStandings": True})
            rows, unmatched = resolve_names(parse_team_records(items), our_team_names(sport))
            n = upsert_team_betting_records(rows)
            print(f"[{sport}] season={season} teams={n} unmatched={len(unmatched)} {unmatched[:10]}")
        except Exception as exc:  # noqa: BLE001 -- one league failing must not abort the other
            print(f"[{sport}] team records capture failed: {exc}")


if __name__ == "__main__":
    main()
