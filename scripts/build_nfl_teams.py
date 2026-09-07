"""Regenerate assets/nfl/nfl_teams.json: {abbreviation -> ESPN displayName}.

The NFL analog of assets/cfb/fbs_teams.json. desk_inputs.py uses it to
translate assets/nfl/schedules.parquet's abbreviation-keyed teams onto the
ESPN displayNames that predictions_current (and thus the desk bundle) use,
so recent-form joins line up. Keys are normalized via nfl.teams.normalize_team
so historical/alternate codes collapse onto the current franchise code.

Run on demand (not in CI):
    PYTHONPATH=src uv run python scripts/build_nfl_teams.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from sportsmodel.nfl.teams import normalize_team

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
OUT = Path(__file__).resolve().parents[1] / "assets" / "nfl" / "nfl_teams.json"


def main() -> None:
    data = httpx.get(URL, timeout=20).json()
    teams = data["sports"][0]["leagues"][0]["teams"]
    out = {}
    for t in teams:
        tm = t["team"]
        out[normalize_team(tm["abbreviation"])] = tm["displayName"]
    OUT.write_text(json.dumps(dict(sorted(out.items())), indent=2) + "\n")
    print(f"Wrote {OUT} ({len(out)} teams)")


if __name__ == "__main__":
    main()
