"""Generate game-line predictions for the current NFL slate (LIVE producer).

Analog of `generate_cfb.py`: assembles per-game predictions from the P1
Elo/SoS margin model + P2 opponent-adjusted points/gameline model and writes
`game_predictions` rows tagged `sport='nfl'`. NFL has no props/odds/market-
shrink stage: predictions here are MODEL-ONLY (this is a game-PREDICTION
producer, not a betting tool), so `build_gameline` is always called with an
empty market (`{"spread_line": None, "total_line": None}`), which makes
`shrink()` fall straight through to the model value regardless of week or the
fitted shrink-curve weights -- exactly how generate_cfb.py runs.

    P1 (elo/srs/ratings):     pre-game Elo + season-to-date SRS -> model_margin
    P2 (points/gameline):     opponent-adjusted points -> model_total; wrapped
                              (unshrunk) into serving dists via build_gameline

`build_game_row` is pure (all inputs injected) and unit tested directly;
`main()`'s ESPN/DB I/O is the thin live wrapper that feeds it from the
committed P1/P2 assets + a live ESPN schedule pull.

Data sources (all committed snapshots so the pure assembly is testable/CI-safe):
  - ratings/gameline configs: assets/nfl/{rating,gameline}.json
  - historical schedule: assets/nfl/schedules.parquet
  - live schedule: ESPN scoreboard (nfl/espn.py)
  - output: Supabase game_predictions (sport='nfl'), only if DATABASE_URL is
            set (mirrors generate_cfb.py)

Usage:
    uv run python scripts/generate_nfl.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel import config
from sportsmodel.db import upsert_game_predictions
from sportsmodel.nfl import config as nfl_config
from sportsmodel.nfl import espn, gameline, points, ratings, srs
from sportsmodel.nfl.elo import run_elo

GAME_MODEL_VERSION = "nfl-elo-v1"

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "nfl"


def _load_committed(name: str) -> pd.DataFrame:
    return pd.read_parquet(_ASSETS / name)


def _game_date_from_commence(commence_iso: str) -> str:
    """US game date from an ESPN UTC `commence_time`.

    NFL kickoffs run from ~13:00 UTC (early Sunday/int'l window) to ~00:20-
    01:20 UTC the NEXT day (SNF/MNF). An 8h shift maps every real kickoff
    time back into its true US game day without needing a timezone lookup
    (identical trick to generate_cfb.py's version, tuned for NFL's window).
    """
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    return (dt - timedelta(hours=8)).date().isoformat()


def build_game_row(game: dict, ctx: dict, gl_cfg: "gameline.GameLineConfig") -> dict:
    """Pure: model margin/total -> a `game_predictions`-shaped row.

    `ctx` = {"model_margin", "model_total", "week"}. No market line is ever
    passed to `build_gameline` -- NFL predictions are model-only, so the market
    dict is always empty and `shrink()` falls straight through to the model
    value. `margin_dist`/`total_dist` are returned as plain dicts (not JSON) so
    callers can inspect them directly (see the unit test); the live `main()`
    JSON-encodes them right before the DB upsert, same boundary generate_cfb.py
    uses.
    """
    empty_market = {"spread_line": None, "total_line": None}
    row = gameline.build_gameline(ctx["model_margin"], ctx["model_total"],
                                  empty_market, ctx["week"], gl_cfg)
    return {
        **row,
        "sport": "nfl",
        "model_version": GAME_MODEL_VERSION,
        "game_pk": game["game_pk"],
        "game_date": game["game_date"],
        "home_team_name": game["home_name"],
        "away_team_name": game["away_name"],
    }


def _resolve_season_week() -> tuple[int, int, int]:
    tw = espn.resolve_target_week()
    return int(tw["season"]), int(tw["week"]), int(tw["season_type"])


def main() -> None:
    season, week, season_type = _resolve_season_week()

    elo_cfg, blend_cfg = nfl_config.load_rating()
    gl_cfg = nfl_config.load_gameline()

    sched = _load_committed("schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    played = reg[(reg["season"] < season) | ((reg["season"] == season) & (reg["week"] < week))].copy()
    played = played.dropna(subset=["home_score", "away_score"])

    elo_final = run_elo(played, elo_cfg).final if len(played) else {}
    srs_now = srs.compute_srs(played) if len(played) else {}
    points_ratings, lg_avg = points.compute_points_ratings(played) if len(played) else ({}, 44.0)
    games_played: dict[str, int] = {}
    for _, g in played.iterrows():
        games_played[g["home_team"]] = games_played.get(g["home_team"], 0) + 1
        games_played[g["away_team"]] = games_played.get(g["away_team"], 0) + 1

    espn_games = espn.fetch_schedule(season, week, season_type=season_type)

    game_rows = []
    for g in espn_games:
        model_margin = ratings.expected_margin(
            elo_final.get(g["home_team"], elo_cfg.base),
            elo_final.get(g["away_team"], elo_cfg.base),
            srs_now.get(g["home_team"]), srs_now.get(g["away_team"]),
            games_played.get(g["home_team"], 0), games_played.get(g["away_team"], 0),
            elo_cfg, blend_cfg)
        model_total = points.expected_total(points_ratings, lg_avg, g["home_team"], g["away_team"])
        ctx = {"model_margin": model_margin, "model_total": model_total, "week": week}
        game_for_row = {**g, "game_date": _game_date_from_commence(g["commence_time"])}
        row = build_game_row(game_for_row, ctx, gl_cfg)
        game_rows.append({
            **row,
            "margin_dist": json.dumps(row["margin_dist"]),
            "total_dist": json.dumps(row["total_dist"]),
        })

    if config.DATABASE_URL:
        upsert_game_predictions(game_rows)

    print(f"predicted {len(game_rows)} games")


if __name__ == "__main__":
    main()
