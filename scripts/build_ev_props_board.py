"""Build the NFL +EV player-prop board: join the sim's per-player prop
distributions (`nfl_player_sim_current`) to the latest book prop odds
(`odds_snapshot`), compute EV via `sportsmodel.serving.props_ev`, and land
the rows in `ev_prop_picks`.

Sub-project C (NFL player-props productization), Task 4 -- see
.superpowers/sdd/2026-09-19-nfl-props-productization-C/task-4-brief.md.
Mirrors `scripts/build_ev_board.py`'s IO/DB structure for the game-level
+EV pilot board.

Reads:
  - `nfl_player_sim_current`: the upcoming player-prop slate (already
    filtered to commence_time > now() by the view) for every (game, player,
    market) the sim engine has scored.
  - `odds_snapshot`: latest per-(game_pk, market, side, player_name, book,
    line) prop-odds row captured before kickoff, restricted to this sim
    slate's games and to the sport's prop market codes
    (`SportConfig[sport].prop_market_map` keys).

Usage:
    DATABASE_URL=... PYTHONPATH=src uv run python scripts/build_ev_props_board.py --sport nfl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import sports
from sportsmodel.db import clear_stale_prop_line_picks, get_postgres, upsert_ev_prop_picks
from sportsmodel.serving.props_ev import assemble_prop_rows

MODEL_VERSION = "props-sim-v1"

SIM_COLS = [
    "game_pk", "player_id", "model_version", "name", "pos", "team",
    "market", "mean", "dist", "commence_time", "matchup",
]

ODDS_COLS = ["game_pk", "market", "side", "player_name", "book", "line", "price"]


def load_sim_rows(sport: str) -> list[dict]:
    """The upcoming player-prop slate from `nfl_player_sim_current`, with the
    game's matchup LEFT-joined from `nfl_sim_current` (best-effort -- NULL if
    the game-level sim hasn't been built for that game_pk). NFL only for C v1."""
    if sport != "nfl":
        return []
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT s.game_pk, s.player_id, s.model_version, s.name, s.pos, s.team,
                   s.market, s.mean, s.dist, s.commence_time, g.matchup
            FROM nfl_player_sim_current s
            LEFT JOIN nfl_sim_current g ON g.game_pk = s.game_pk
            """
        )
        rows = cur.fetchall()
    return [dict(zip(SIM_COLS, r)) for r in rows]


def load_latest_prop_odds(sport: str, game_pks: list[int]) -> list[dict]:
    """Latest odds_snapshot row per (game_pk, market, side, player_name, book,
    line) for this sport's prop markets, restricted to `game_pks` and
    captured before kickoff. Empty list of game_pks -> empty result."""
    if not game_pks:
        return []
    prop_markets = list(sports.get(sport).prop_market_map.keys())
    if not prop_markets:
        return []
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT game_pk, market, side, player_name, book, line, price
            FROM (
                SELECT DISTINCT ON (game_pk, market, side, player_name, book, line)
                       game_pk, market, side, player_name, book, line, price
                FROM odds_snapshot
                WHERE game_pk = ANY(%s) AND market = ANY(%s)
                  AND captured_at <= commence_time
                ORDER BY game_pk, market, side, player_name, book, line, captured_at DESC
            ) t
            """,
            [list(game_pks), prop_markets],
        )
        rows = cur.fetchall()
    return [dict(zip(ODDS_COLS, r)) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="nfl", help="Sport key (default: nfl)")
    args = parser.parse_args()

    sim_rows = load_sim_rows(args.sport)
    game_pks = sorted({r["game_pk"] for r in sim_rows})
    odds_rows = load_latest_prop_odds(args.sport, game_pks)

    rows = assemble_prop_rows(sim_rows, odds_rows, MODEL_VERSION)
    upsert_ev_prop_picks(rows)

    picks = [r for r in rows if r["is_pick"]]
    # A player's main line can shift between builds (books move the number);
    # since `line` is part of ev_prop_picks' primary key, the OLD line's row
    # would otherwise persist with is_pick=true and surface as a phantom
    # pick alongside the new line. Demote every other line for each player
    # just picked.
    cleared = clear_stale_prop_line_picks(picks)

    print(f"[build_ev_props_board] sport={args.sport} games={len(game_pks)} "
          f"sim_rows={len(sim_rows)} prop_picks={len(rows)} plus_ev={len(picks)} "
          f"cleared_stale_lines={cleared}")


if __name__ == "__main__":
    main()
