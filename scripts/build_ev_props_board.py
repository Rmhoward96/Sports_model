"""Build the +EV player-prop board (NFL and MLB): join the model's per-player prop
distributions (NFL: `nfl_player_sim_current`; MLB: the latest `prop_predictions` for
upcoming games) to the latest book prop odds (`odds_snapshot`), compute EV via
`sportsmodel.serving.props_ev`, and land the rows in `ev_prop_picks`.

MLB differences: markets are the four published pre-pause (total_bases, pitcher_ks,
hits_allowed, outs_recorded -- hits/hrr stay dropped), no projected-usage gate, the
model's P(over) at the book line is Platt-calibrated per market (as the pre-pause board
did), and there is no `nfl_prop_lines` accuracy table (`--lines-only` is a no-op).

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
from sportsmodel.model import calibration
from sportsmodel.db import (
    clear_stale_prop_line_picks,
    get_postgres,
    upsert_ev_prop_picks,
    upsert_nfl_prop_lines,
)
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    always_propable,
    assemble_prop_line_rows,
    assemble_prop_rows,
    drop_pulled_lines,
    latest_capture_only,
)

MODEL_VERSION = "props-sim-v1"
# ev_prop_picks' key is (game_pk, player_id, market, line, model_version) and game_pks
# never collide across sports, so a separate MLB version string is for readability only.
MODEL_VERSION_BY_SPORT = {"nfl": MODEL_VERSION, "mlb": "props-mlb-v1"}

SIM_COLS = [
    "game_pk", "player_id", "model_version", "name", "pos", "team",
    "market", "mean", "dist", "commence_time", "matchup",
]

# Same keys as SIM_COLS (minus the NFL-only `pos`), in _load_mlb_sim_rows' SELECT order.
MLB_SIM_COLS = [
    "game_pk", "player_id", "model_version", "name", "team",
    "market", "mean", "dist", "commence_time", "matchup",
]

ODDS_COLS = ["game_pk", "market", "side", "player_name", "book", "line", "price", "captured_at"]


def _load_mlb_sim_rows() -> list[dict]:
    """MLB's upcoming prop slate in the same shape as the NFL sim rows: the latest
    prop_predictions row per (game, player, market) for games that have not started
    (commence_time and matchup come from the game's latest game_predictions row, which
    generate_sim stamps with the StatsAPI first-pitch time). player_id is a string
    (ev_prop_picks.player_id is TEXT)."""
    markets = list(MLB_SIM_TO_ODDS_MARKET)
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT ON (pp.game_pk, pp.player_id, pp.market)
                   pp.game_pk, pp.player_id::text, pp.model_version, pp.player_name,
                   pp.team_name, pp.market, pp.projected_mean, pp.dist,
                   g.commence_time, g.matchup
            FROM prop_predictions pp
            JOIN (
                SELECT DISTINCT ON (game_pk) game_pk, commence_time,
                       away_team_name || ' @ ' || home_team_name AS matchup
                FROM game_predictions
                WHERE sport = 'mlb' AND commence_time > now()
                ORDER BY game_pk, generated_at DESC
            ) g ON g.game_pk = pp.game_pk
            WHERE pp.sport = 'mlb' AND pp.market = ANY(%s)
            ORDER BY pp.game_pk, pp.player_id, pp.market, pp.generated_at DESC
            """,
            [markets],
        )
        rows = cur.fetchall()
    return [dict(zip(MLB_SIM_COLS, r)) for r in rows]


def load_sim_rows(sport: str) -> list[dict]:
    """The upcoming player-prop slate. NFL: `nfl_player_sim_current`, with the
    game's matchup LEFT-joined from `nfl_sim_current` (best-effort -- NULL if
    the game-level sim hasn't been built for that game_pk). MLB: see
    `_load_mlb_sim_rows`. Other sports have no prop model."""
    if sport == "mlb":
        return _load_mlb_sim_rows()
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
    captured before kickoff and within the last 48h (a stale book drops out,
    same window as the site's price views), further filtered to each book's MOST RECENT
    capture per (game_pk, market, player_name, book) via `latest_capture_only`
    -- a book's superseded line (moved since) is dropped, not just its
    superseded (game_pk, market, side, player_name, book, line) row -- and to
    the latest pull of each (game_pk, market) via `drop_pulled_lines`, so a
    prop the books took down (a benched QB's) stops counting. Empty list of
    game_pks -> empty result."""
    if not game_pks:
        return []
    prop_markets = list(sports.get(sport).prop_market_map.keys())
    if not prop_markets:
        return []
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute(
            """
            SELECT game_pk, market, side, player_name, book, line, price, captured_at
            FROM (
                SELECT DISTINCT ON (game_pk, market, side, player_name, book, line)
                       game_pk, market, side, player_name, book, line, price, captured_at
                FROM odds_snapshot
                WHERE game_pk = ANY(%s) AND market = ANY(%s)
                  AND captured_at <= commence_time
                  AND captured_at > now() - interval '48 hours'
                ORDER BY game_pk, market, side, player_name, book, line, captured_at DESC
            ) t
            """,
            [list(game_pks), prop_markets],
        )
        rows = cur.fetchall()
    return drop_pulled_lines(latest_capture_only([dict(zip(ODDS_COLS, r)) for r in rows]))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="nfl", help="Sport key (default: nfl)")
    parser.add_argument(
        "--lines-only", action="store_true",
        help="Write nfl_prop_lines only (skip the +EV board)",
    )
    args = parser.parse_args(argv)

    is_mlb = args.sport == "mlb"
    model_version = MODEL_VERSION_BY_SPORT.get(args.sport, MODEL_VERSION)
    if is_mlb and args.lines_only:
        # nfl_prop_lines is the NFL sim-accuracy table; MLB has no counterpart.
        print("[build_ev_props_board] sport=mlb: --lines-only is a no-op (no MLB prop-lines table)")
        return

    sim_rows = load_sim_rows(args.sport)
    game_pks = sorted({r["game_pk"] for r in sim_rows})
    odds_rows = load_latest_prop_odds(args.sport, game_pks)

    # The lines write is a separate concern from the +EV board below -- a
    # missing nfl_prop_lines table (migration not yet applied) or any other
    # lines-write error must not take down the game-day +EV build. Only
    # --lines-only (whose sole job IS the lines write) re-raises, so that
    # run still exits non-zero.
    if not is_mlb:
        try:
            line_rows = assemble_prop_line_rows(sim_rows, odds_rows)
            n_lines = upsert_nfl_prop_lines(line_rows)
            print(f"[build_ev_props_board] prop_lines={n_lines}")
        except Exception as e:
            print(f"[build_ev_props_board] prop_lines FAILED: {type(e).__name__}: {e}")
            if args.lines_only:
                raise

    if args.lines_only:
        return

    if is_mlb:
        rows = assemble_prop_rows(
            sim_rows, odds_rows, model_version, sport="mlb",
            market_map=MLB_SIM_TO_ODDS_MARKET, gate=always_propable,
            calibrate_fn=calibration.calibrate)
    else:
        rows = assemble_prop_rows(sim_rows, odds_rows, model_version)
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
