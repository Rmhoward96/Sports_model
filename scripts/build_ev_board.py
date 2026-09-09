"""Build the +EV desk-driven pilot board: read the desk's upcoming picks +
latest odds, compute edge/EV via `sportsmodel.serving.ev_pilot`, and land the
rows in `ev_picks`.

Sub-project 4 (+EV desk-driven pilot), task 1 -- see
.superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/task-1-brief.md.

TODO(task 2): `ev_picks` (the table + `db.upsert_ev_picks`) lands in sub-project 4
task 2. Until that upsert exists, this script prints the computed rows and writes
them to a JSON file instead of upserting -- swap `_write_rows` below for
`db.upsert_ev_picks(rows)` once that function exists, per the task-1 brief
("if `db.upsert_ev_picks` doesn't exist yet, write the rows to a JSON/print ...
do NOT block on Task 2").

Reads:
  - `desk_current` (view over desk_picks): the desk's latest pick per upcoming
    game for `--sport`.
  - `odds_snapshot`: latest per-book snapshot per (game_pk, market, side, book)
    captured before commence_time, for those games' Pinnacle + soft-book prices.

Usage:
    DATABASE_URL=... PYTHONPATH=src uv run python scripts/build_ev_board.py --sport nfl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import get_postgres
from sportsmodel.serving.ev_pilot import assemble_games, ev_rows_for_game

DESK_CURRENT_COLS = [
    "sport", "game_pk", "matchup", "commence_time",
    "ml_pick", "spread_side", "total_side", "conviction_tier",
]

OUT_PATH = Path(__file__).resolve().parents[1] / "tmp" / "ev_board.json"


def load_desk_current(sport: str) -> list[dict]:
    """The desk's current (upcoming) picks for `sport`, from the `desk_current`
    view (already deduped to the latest model_version per game)."""
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            f"SELECT {', '.join(DESK_CURRENT_COLS)} FROM desk_current WHERE sport = %s",
            [sport],
        )
        rows = cur.fetchall()
    return [dict(zip(DESK_CURRENT_COLS, r)) for r in rows]


def load_latest_odds(game_pks: list[int]) -> list[dict]:
    """Latest odds_snapshot row per (game_pk, market, side, book) -- Pinnacle +
    MAJOR_BOOKS -- for the given games, captured before kickoff. Empty list of
    game_pks -> empty result (no games to query)."""
    if not game_pks:
        return []
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT game_pk, market, side, book, line, price
            FROM (
                SELECT DISTINCT ON (game_pk, market, side, book)
                       game_pk, market, side, book, line, price
                FROM odds_snapshot
                WHERE game_pk = ANY(%s) AND captured_at <= commence_time
                ORDER BY game_pk, market, side, book, captured_at DESC
            ) t
            """,
            [list(game_pks)],
        )
        rows = cur.fetchall()
    cols = ["game_pk", "market", "side", "book", "line", "price"]
    return [dict(zip(cols, r)) for r in rows]


def _write_rows(rows: list[dict]) -> None:
    """TODO(task 2): replace with db.upsert_ev_picks(rows) once the `ev_picks`
    table + upsert exist. For now, print + dump to JSON so the pilot's edge/EV
    engine is inspectable end-to-end without blocking on the task-2 table."""
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(rows, indent=2, default=str))
    print(f"[build_ev_board] TODO(task 2): wrote {len(rows)} row(s) to {OUT_PATH} "
          f"(no ev_picks table/upsert yet -- see db.upsert_ev_picks TODO).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="nfl", help="Sport key (default: nfl)")
    args = parser.parse_args()

    desk_rows = load_desk_current(args.sport)
    odds_rows = load_latest_odds([d["game_pk"] for d in desk_rows])
    games = assemble_games(desk_rows, odds_rows)

    all_rows = []
    for game in games:
        all_rows.extend(ev_rows_for_game(game))

    picks = [r for r in all_rows if r["is_pick"]]
    passes = [r for r in all_rows if not r["is_pick"]]

    _write_rows(all_rows)

    print(f"[build_ev_board] sport={args.sport} games={len(games)} rows={len(all_rows)} "
          f"picks={len(picks)} passes={len(passes)}")


if __name__ == "__main__":
    main()
