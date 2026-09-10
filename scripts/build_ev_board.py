"""Build the +EV desk-driven pilot board: read the desk's upcoming picks +
latest odds, compute edge/EV via `sportsmodel.serving.ev_pilot`, and land the
rows in `ev_picks`.

Sub-project 4 (+EV desk-driven pilot), task 1 & 2 -- see
.superpowers/sdd/2026-09-09-plus-ev-4-desk-driven-pilot/task-1-brief.md and
task-2-brief.md.

Reads:
  - `predictions_current`: the full upcoming slate for `--sport` (game_pk, team
    names -> matchup, commence_time). The board is built over EVERY upcoming
    game, so LINE-SHOPPING (best soft price vs the sharp Pinnacle fair) is
    surfaced everywhere, not only on games the desk picked.
  - `desk_current` (view over desk_picks): LEFT-joined as an OPTIONAL overlay --
    a game with no desk pick still gets a line-shopping row (desk_delta 0).
  - `odds_snapshot`: latest per-book snapshot per (game_pk, market, side, book)
    captured before commence_time, for those games' Pinnacle + soft-book prices.

Usage:
    DATABASE_URL=... PYTHONPATH=src uv run python scripts/build_ev_board.py --sport nfl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import get_postgres, upsert_ev_picks
from sportsmodel.serving.ev_pilot import assemble_games, ev_rows_for_game

MODEL_VERSION = "ev-pilot-v1"

# One row per upcoming game: identity/matchup from predictions_current, the
# desk's pick fields LEFT-joined (NULL when the desk didn't touch that game).
# assemble_games sets desk=None when ml_pick/spread_side/total_side are all
# NULL, so a non-desk game yields a pure line-shopping row.
GAME_COLS = [
    "sport", "game_pk", "matchup", "commence_time",
    "ml_pick", "spread_side", "total_side", "conviction_tier",
]


def load_upcoming_games(sport: str) -> list[dict]:
    """Every upcoming game for `sport` from `predictions_current`, with the
    desk's current pick (if any) LEFT-joined from `desk_current`. matchup is
    built as "<away> @ <home>" to match the desk convention."""
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT p.sport,
                   p.game_pk,
                   p.away_team_name || ' @ ' || p.home_team_name AS matchup,
                   p.commence_time,
                   d.ml_pick, d.spread_side, d.total_side, d.conviction_tier
            FROM predictions_current p
            LEFT JOIN desk_current d
              ON d.sport = p.sport AND d.game_pk = p.game_pk
            WHERE p.sport = %s AND p.commence_time > now()
            """,
            [sport],
        )
        rows = cur.fetchall()
    return [dict(zip(GAME_COLS, r)) for r in rows]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="nfl", help="Sport key (default: nfl)")
    args = parser.parse_args()

    game_rows = load_upcoming_games(args.sport)
    odds_rows = load_latest_odds([g["game_pk"] for g in game_rows])
    games = assemble_games(game_rows, odds_rows)

    all_rows = []
    for game in games:
        all_rows.extend(ev_rows_for_game(game))

    for row in all_rows:
        row["model_version"] = MODEL_VERSION

    picks = [r for r in all_rows if r["is_pick"]]
    passes = [r for r in all_rows if not r["is_pick"]]

    upsert_ev_picks(all_rows)

    print(f"[build_ev_board] sport={args.sport} games={len(games)} rows={len(all_rows)} "
          f"picks={len(picks)} passes={len(passes)}")


if __name__ == "__main__":
    main()
