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
from sportsmodel.nfl.teams import normalize_team
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


def _nfl_abbr_names() -> dict[str, str]:
    """{nflverse abbr: our full name} from assets/nfl/nfl_teams.json."""
    return json.loads((ROOT / "assets" / "nfl" / "nfl_teams.json").read_text())


def _nfl_name_from_abbr(abbr, by_abbr: dict[str, str]) -> str | None:
    """Our NFL name for an AN abbreviation (normalized via normalize_team, e.g.
    LAR -> LA), or None when it can't be resolved."""
    try:
        return by_abbr.get(normalize_team(abbr))
    except Exception:  # noqa: BLE001 -- unknown/missing abbr -> no fallback
        return None


# Action Network CFB names that are the same school under a different name in
# assets/cfb/fbs_teams.json (from the first live capture's unmatched list,
# 2026-09-23). Applied before normalized matching.
CFB_ALIASES = {
    "Miami (FL) Hurricanes": "Miami Hurricanes",
    "North Carolina State Wolfpack": "NC State Wolfpack",
    "UMass Minutemen": "Massachusetts Minutemen",
    "Hawai'i Warriors": "Hawai'i Rainbow Warriors",
    "UL-Monroe Warhawks": "UL Monroe Warhawks",
    "Appalachian State Mountaineers": "App State Mountaineers",
    "Delaware Fightin Blue Hens": "Delaware Blue Hens",
    "Sam Houston State Bearkats": "Sam Houston Bearkats",
}


def resolve_names(rows: list[dict], ours: list[str]) -> tuple[list[dict], list[str]]:
    """Attach team_name (our name when matched -- CFB via CFB_ALIASES first; for
    NFL rows whose name doesn't match, our name via the row's `abbr`; else the AN
    name); return the unmatched AN names for logging. PURE (reads the NFL
    crosswalk asset)."""
    def key(r: dict) -> str:
        a = r["an_team_name"]
        return CFB_ALIASES.get(a, a) if r.get("sport") == "cfb" else a

    m = match_team_names([key(r) for r in rows], ours)
    by_abbr = _nfl_abbr_names() if any(r.get("sport") == "nfl" for r in rows) else {}
    out, unmatched = [], []
    for r in rows:
        name = m.get(key(r))
        if name is None and r.get("sport") == "nfl":
            name = _nfl_name_from_abbr(r.get("abbr"), by_abbr)
        if name is None:
            unmatched.append(r["an_team_name"])
            name = r["an_team_name"]
        out.append({**r, "team_name": name})
    return out, unmatched


def stamp_season(rows: list[dict], season: int) -> list[dict]:
    """Every row gets the season we requested from ESPN (the desk filters on its
    own current season), overriding whatever the actor item carried. PURE."""
    return [{**r, "season": season} for r in rows]


def main() -> None:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise SystemExit("APIFY_TOKEN required")
    failed: list[str] = []
    for sport, league in LEAGUES.items():
        try:
            season = current_season(sport)
            items = fetch_splits(token, leagues=(league,), game_status=("scheduled",),
                                 extra_input={"season": season, "maxGames": 1, "includeStandings": True})
            rows, unmatched = resolve_names(parse_team_records(items), our_team_names(sport))
            n = upsert_team_betting_records(stamp_season(rows, season))
            print(f"[{sport}] season={season} teams={n} unmatched={len(unmatched)} {unmatched[:10]}")
            if not n:
                failed.append(f"{sport} (0 rows)")
        except Exception as exc:  # noqa: BLE001 -- one league failing must not abort the other
            print(f"[{sport}] team records capture failed: {exc}")
            failed.append(f"{sport} (error)")
    if failed:
        print(f"team records capture FAILED for: {', '.join(failed)}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
