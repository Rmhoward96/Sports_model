"""Compute NFL situational trends for the upcoming slate -> nfl_game_trends.
Daily (build-trends.yml), before the desk. See sportsmodel/nfl/trends.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import get_postgres, replace_nfl_game_trends
from sportsmodel.nfl.data import load_schedules
from sportsmodel.nfl.injuries_nflverse import nfl_season
from sportsmodel.nfl.trends import compute_game_trends

ROOT = Path(__file__).resolve().parents[1]


def upcoming_games(sched) -> list[dict]:
    """Upcoming NFL games from predictions_current, mapped to nflverse abbrs +
    schedule date/time via the schedule's `espn` id. PURE-ish helper kept small."""
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute("SELECT game_pk, market_spread FROM predictions_current WHERE sport = 'nfl' AND commence_time > now()")
        preds = {int(pk): spread for pk, spread in cur.fetchall()}
    return build_upcoming(sched, preds)


def build_upcoming(sched, preds: dict[int, float | None]) -> list[dict]:
    """PURE: schedule rows whose `espn` id is an upcoming predicted game_pk."""
    out = []
    for g in sched.itertuples(index=False):
        try:
            pk = int(g.espn)
        except (TypeError, ValueError):
            continue
        if pk not in preds:
            continue
        line = preds[pk]
        if line is None and not pd.isna(g.spread_line):              # not NaN/None
            line = -float(g.spread_line)                              # nflverse: + = home favored
        out.append({"game_pk": pk, "home_team": g.home_team, "away_team": g.away_team,
                    "home_line": None if line is None else float(line),
                    "gameday": str(g.gameday)[:10], "gametime": "" if pd.isna(g.gametime) else str(g.gametime)})
    return out


def main() -> None:
    from datetime import datetime, timezone
    season = nfl_season(datetime.now(timezone.utc))
    sched = load_schedules(list(range(season - 4, season + 1)))
    crosswalk = json.loads((ROOT / "assets/nfl/nfl_teams.json").read_text())
    games = upcoming_games(sched)
    rows = compute_game_trends(sched, games, current_season=season)
    for r in rows:
        abbr = r.pop("team")
        r["team_name"] = crosswalk.get(abbr, abbr)
    n = replace_nfl_game_trends([g["game_pk"] for g in games], rows)
    print(f"[build_nfl_trends] season={season} games={len(games)} trend_rows={n}")


if __name__ == "__main__":
    main()
