"""Capture realized player box stats for finished NFL games into Supabase
`nfl_player_actuals`, so the game page can show actuals next to the sim's
projected props.

For each finished game that still has projections in `nfl_player_sim` but no
captured actuals, this resolves the game's NFL week (nflverse `import_schedules`
`espn` join), fetches that season's `import_weekly_data`, and writes each
projected player's realized value for the six projected markets
(pass_yds/pass_tds/rush_yds/rec_yds/receptions/anytime_td). Only players the
sim projected are captured, so the table lines up 1:1 with the props table.

Reads Supabase + nflverse only -- NO Odds API calls. Rolling window (like
grade_ev_props.py); idempotent -- re-running re-upserts the same rows via
(game_pk, player_id, market) (db.upsert_nfl_player_actuals). A game whose week
or weekly data can't be resolved yet (nflverse lag) is skipped and retried next
run, never crashing the batch.

The pure `assemble_actual_rows` seam is unit-tested; main()'s DB/nflverse IO is
thin and not unit-tested.

Usage:
    uv run python scripts/capture_nfl_player_actuals.py [--days 8]
"""
from __future__ import annotations

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import nfl_data_py as nfl_data_py_import
import pandas as pd

from sportsmodel import config
from sportsmodel.db import get_postgres, upsert_nfl_player_actuals
from sportsmodel.nfl.data import load_schedules
from sportsmodel.nfl.injuries_nflverse import nfl_season
from sportsmodel.nfl.nflverse import import_by_season
from sportsmodel.nfl.weekly_actuals import (
    WEEKLY_STAT_COLS,
    player_market_actual,
    week_for_game,
)


def _window_start(days: int, today: date | None = None) -> str:
    d = today or date.today()
    return (d - timedelta(days=days)).isoformat()


def assemble_actual_rows(projected, weekly_frame, game_pk, season: int, week: int) -> list[dict]:
    """PURE. `projected` = [(player_id, player_name), ...] the sim projected for
    this game; `weekly_frame` = that season's nflverse weekly data. Returns one
    `nfl_player_actuals` row (dict, columns matching db._NFL_PLAYER_ACTUALS_COLS)
    per (player, market) that resolves to a non-None actual for the game's week.
    Players absent from the week's data, and markets with no realized value, are
    skipped."""
    if weekly_frame is None or getattr(weekly_frame, "empty", True) or "week" not in weekly_frame.columns:
        return []
    wk = weekly_frame[weekly_frame["week"] == week]
    if wk.empty:
        return []
    out: list[dict] = []
    for pid, pname in projected:
        prow = wk[wk["player_id"] == pid]
        if prow.empty:
            continue
        row = prow.iloc[0]
        for market in WEEKLY_STAT_COLS:
            actual = player_market_actual(row, market)
            if actual is None:
                continue
            out.append({
                "game_pk": game_pk, "player_id": pid, "player_name": pname,
                "market": market, "actual": actual, "season": season, "week": week,
            })
    return out


# Per-run caches so a slate sharing a season doesn't refetch schedules/weekly.
_schedule_cache: dict[int, pd.DataFrame | None] = {}
_weekly_cache: dict[int, pd.DataFrame | None] = {}


def _schedule_for(season: int) -> pd.DataFrame | None:
    if season not in _schedule_cache:
        try:
            _schedule_cache[season] = load_schedules([season])
        except Exception as exc:  # noqa: BLE001 -- schedule fetch is best-effort
            print(f"  schedule fetch failed for season {season}: {exc}")
            _schedule_cache[season] = None
    return _schedule_cache[season]


def _weekly_for(season: int) -> pd.DataFrame | None:
    if season not in _weekly_cache:
        _weekly_cache[season] = import_by_season(
            nfl_data_py_import.import_weekly_data, [season], "weekly", required=False)
    return _weekly_cache[season]


def _pending_games(cur, start: str) -> list[tuple]:
    """Finished games (commence_time <= now) within the window that have sim
    projections but no captured actuals yet."""
    cur.execute("""
        SELECT DISTINCT s.game_pk, s.commence_time
        FROM nfl_player_sim s
        WHERE s.commence_time <= now() AND s.commence_time >= %(start)s
          AND NOT EXISTS (
              SELECT 1 FROM nfl_player_actuals a WHERE a.game_pk = s.game_pk
          )
        ORDER BY s.commence_time
    """, {"start": start})
    return list(cur.fetchall())


def _projected_players(cur, game_pk) -> list[tuple]:
    """Distinct (player_id, name) the sim projected for `game_pk`."""
    cur.execute("""
        SELECT DISTINCT player_id, name FROM nfl_player_sim WHERE game_pk = %s
    """, (game_pk,))
    return [(pid, name) for pid, name in cur.fetchall()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=8)
    args = ap.parse_args()
    if not config.DATABASE_URL:
        raise SystemExit("DATABASE_URL required (capture reads/writes Supabase).")

    start = _window_start(args.days)
    all_rows: list[dict] = []
    n_games = 0
    n_skipped = 0

    with get_postgres() as conn, conn.cursor() as cur:
        pending = _pending_games(cur, start)
        for game_pk, commence in pending:
            try:
                season = nfl_season(commence)
                week = week_for_game(_schedule_for(season), game_pk)
                if week is None:
                    n_skipped += 1
                    continue  # week not resolvable yet -- retry next run
                weekly = _weekly_for(season)
                projected = _projected_players(cur, game_pk)
                rows = assemble_actual_rows(projected, weekly, game_pk, season, week)
                if rows:
                    all_rows.extend(rows)
                    n_games += 1
                else:
                    n_skipped += 1
            except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the batch
                print(f"  nfl {game_pk}: capture failed ({exc}); skipping")
                n_skipped += 1
                continue

    n_players = len({(r["game_pk"], r["player_id"]) for r in all_rows})
    print(f"nfl: {len(pending)} pending games, {n_games} captured, {n_skipped} skipped")
    if all_rows:
        written = upsert_nfl_player_actuals(all_rows)
        print(f"Upserted {written} nfl_player_actuals rows "
              f"(players={n_players}, actuals={len(all_rows)}).")


if __name__ == "__main__":
    main()
