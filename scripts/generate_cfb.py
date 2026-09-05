"""Generate game-line predictions for the current CFB slate (LIVE producer).

Analog of `generate_nfl.py`'s GAME-LINE half for CFB -- assembles per-game
predictions from the P1 Elo/SoS margin model + P2 opponent-adjusted points
model and writes `game_predictions` rows tagged `sport='cfb'`. CFB has no
props/odds/market-shrink stage: predictions here are MODEL-ONLY (this is a
game-PREDICTION producer, not a betting tool), so `build_gameline` is always
called with an empty market (`{"spread_line": None, "total_line": None}`),
which makes `shrink()` fall straight through to the model value regardless of
week or the fitted shrink-curve weights.

    P1 (elo/srs/ratings):     pre-game Elo + season-to-date SRS -> model_margin
    P2 (points/gameline):     opponent-adjusted points -> model_total; wrapped
                              (unshrunk) into serving dists via build_gameline

`build_game_row`/`build_game_rows` are pure (all inputs injected) and unit
tested directly; `main()`'s ESPN/DB I/O is the thin live wrapper that feeds
them from the committed P1/P2 assets + a live ESPN schedule pull.

Data sources (all committed snapshots so the pure assembly is testable/CI-safe):
  - ratings/gameline configs: assets/cfb/{rating,gameline}.json
  - historical + season-to-date schedule: assets/cfb/schedules.parquet
  - live schedule: ESPN scoreboard (cfb/espn.py)
  - output: Supabase game_predictions (sport='cfb'), only if DATABASE_URL is
            set (mirrors generate_nfl.py/generate_sim.py)

Reuses the exact same league-agnostic rating/gameline engine as generate_nfl.py
(sportsmodel.nfl.elo/.srs/.ratings/.points/.gameline), the same engine
scripts/backtest_cfb_gameline.py fits assets/cfb/{rating,gameline}.json
against -- only the CFB-specific team-id normalization (cfb.teams, via
cfb.espn) and the FCS-pseudo-team skip below are CFB-specific.

Usage:
    uv run python scripts/generate_cfb.py
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel import config
from sportsmodel.cfb import espn
from sportsmodel.cfb.teams import FCS
from sportsmodel.db import upsert_game_predictions
from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.shrink import ShrinkParams
from sportsmodel.nfl.srs import compute_srs

GAME_MODEL_VERSION = "cfb-ratings-v1"

_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "cfb"

# Schedule-wide mean/median total, used only as a fallback before any
# points-ratings history exists (mirrors backtest_cfb_gameline.py's
# _DEFAULT_TOTAL_SEED); in practice `played` always has prior-season history
# so this branch is not expected to trigger in the live producer.
_DEFAULT_TOTAL_SEED = 55.0


def _load_committed(name: str) -> pd.DataFrame:
    return pd.read_parquet(_ASSETS / name)


def load_rating() -> tuple[EloConfig, BlendConfig]:
    j = json.loads((_ASSETS / "rating.json").read_text())
    return (EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"],
                      base=j.get("base", 1500.0)),
            BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"]))


def load_gameline() -> GameLineConfig:
    j = json.loads((_ASSETS / "gameline.json").read_text())
    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"],
                          offset=j["offset"], total_max=j["total_max"],
                          w_margin=ShrinkParams(**j["w_margin"]),
                          w_total=ShrinkParams(**j["w_total"]))


def _game_date_from_commence(commence_iso: str) -> str:
    """US game date from an ESPN UTC `commence_time`.

    CFB kickoffs run from ~15:00 UTC (late-morning ET games) to ~04:00-07:00
    UTC the NEXT day (late West Coast/Hawaii night games). An 8h shift maps
    every real kickoff time back into its true US game day without needing a
    timezone lookup (identical trick to generate_nfl.py's version of this
    function, retuned for CFB's kickoff window -- both windows are wide
    enough that a single 8h shift covers them the same way).
    """
    dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
    return (dt - timedelta(hours=8)).date().isoformat()


def build_game_row(game: dict, ctx: dict, gl_cfg: GameLineConfig) -> dict:
    """Pure: model margin/total -> a `game_predictions`-shaped row.

    `ctx` = {"model_margin", "model_total", "week"}. No market line is ever
    passed to `build_gameline` -- CFB predictions are model-only, so the
    market dict is always empty and `shrink()` falls straight through to the
    model value. `margin_dist`/`total_dist` are returned as plain dicts (not
    JSON) so callers can inspect them directly (see the unit test); the live
    `main()` JSON-encodes them right before the DB upsert, same boundary
    generate_nfl.py uses.
    """
    empty_market = {"spread_line": None, "total_line": None}
    row = build_gameline(ctx["model_margin"], ctx["model_total"], empty_market,
                         ctx["week"], gl_cfg)
    return {
        **row,
        "sport": "cfb",
        "model_version": GAME_MODEL_VERSION,
        "game_pk": game["game_pk"],
        "game_date": game["game_date"],
        "commence_time": game.get("commence_time"),
        "market_spread": game.get("market_spread"),
        "market_total": game.get("market_total"),
        "home_team_name": game["home_name"],
        "away_team_name": game["away_name"],
    }


def build_game_rows(games: list[dict], ratings: dict, week: int,
                    gl_cfg: GameLineConfig) -> list[dict]:
    """Pure: FBS schedule rows (each carrying `home_team`/`away_team` ESPN-id-
    or-"FCS" team keys, `home_name`/`away_name` display names, and a
    `game_date`) + season ratings state -> `game_predictions`-shaped rows.

    Games where either side is the CFB rating engine's "FCS" pseudo-team
    (cfb.teams.normalize collapses every non-FBS opponent to this one anchor
    -- see cfb/teams.py) are SKIPPED: there is no meaningful individual
    rating for the large, undifferentiated pool of non-FBS teams that anchor
    represents, so no game-line prediction is served for those games.

    `ratings` = {"elo_final", "srs_now", "points_ratings", "lg_avg",
    "games_played", "elo_cfg", "blend_cfg"} -- the season-to-date state
    `main()` computes once (via run_elo/compute_srs/compute_points_ratings
    over `assets/cfb/schedules.parquet`) before this is called.
    """
    elo_final = ratings["elo_final"]
    srs_now = ratings["srs_now"]
    points_ratings = ratings["points_ratings"]
    lg_avg = ratings["lg_avg"]
    games_played = ratings["games_played"]
    elo_cfg = ratings["elo_cfg"]
    blend_cfg = ratings["blend_cfg"]

    rows = []
    for g in games:
        h, a = g["home_team"], g["away_team"]
        if h == FCS or a == FCS:
            continue
        elo_h = elo_final.get(h, elo_cfg.base)
        elo_a = elo_final.get(a, elo_cfg.base)
        model_margin = expected_margin(
            elo_h, elo_a, srs_now.get(h), srs_now.get(a),
            games_played.get(h, 0), games_played.get(a, 0), elo_cfg, blend_cfg)
        model_total = expected_total(points_ratings, lg_avg, h, a)
        ctx = {"model_margin": model_margin, "model_total": model_total, "week": week}
        rows.append(build_game_row(g, ctx, gl_cfg))
    return rows


def _season_to_date_ratings(sched: pd.DataFrame, season: int, week: int,
                            elo_cfg: EloConfig, blend_cfg: BlendConfig) -> dict:
    """Leak-free ratings state as of (season, week): historical prior-season
    games + this season's already-completed games, exactly the same
    (run_elo/compute_srs/compute_points_ratings) engine calls
    backtest_cfb_gameline.py's walk-forward core fits assets/cfb/*.json
    against -- but as a single as-of snapshot (the live producer only ever
    needs ratings for the ONE upcoming week, not a per-week backtest replay).

    Rows with `home_team == away_team` (a handful of CFBD aggregation
    artifacts where two different unmatched/FCS opponents both collapse to
    the "FCS" pseudo-team on both sides) are dropped first, same as
    `backtest_cfb_gameline.load_merged_schedule` -- left in, the "FCS" team
    would face itself and corrupt every real FBS team's SRS/points rating
    via its games against "FCS".
    """
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()
    played = reg[(reg["season"] < season) | ((reg["season"] == season) & (reg["week"] < week))].copy()
    played = played.dropna(subset=["home_score", "away_score"])

    elo_final = run_elo(played, elo_cfg).final if len(played) else {}
    srs_now = compute_srs(played) if len(played) else {}
    if len(played):
        points_ratings, lg_avg = compute_points_ratings(played)
    else:
        points_ratings, lg_avg = {}, _DEFAULT_TOTAL_SEED
    games_played: dict[str, int] = {}
    for _, g in played.iterrows():
        games_played[g["home_team"]] = games_played.get(g["home_team"], 0) + 1
        games_played[g["away_team"]] = games_played.get(g["away_team"], 0) + 1

    return {"elo_final": elo_final, "srs_now": srs_now, "points_ratings": points_ratings,
            "lg_avg": lg_avg, "games_played": games_played, "elo_cfg": elo_cfg,
            "blend_cfg": blend_cfg}


def main() -> None:
    cur = espn.fetch_current_week()
    season, week, season_type = int(cur["season"]), int(cur["week"]), int(cur["season_type"])

    elo_cfg, blend_cfg = load_rating()
    gl_cfg = load_gameline()

    sched = _load_committed("schedules.parquet")
    ratings = _season_to_date_ratings(sched, season, week, elo_cfg, blend_cfg)

    espn_games = espn.fetch_schedule(season, week, season_type=season_type)
    games_for_rows = [{**g, "game_date": _game_date_from_commence(g["commence_time"])}
                      for g in espn_games]

    game_rows_raw = build_game_rows(games_for_rows, ratings, week, gl_cfg)
    game_rows = [{**row, "margin_dist": json.dumps(row["margin_dist"]),
                 "total_dist": json.dumps(row["total_dist"])} for row in game_rows_raw]

    if config.DATABASE_URL:
        upsert_game_predictions(game_rows)

    print(f"predicted {len(game_rows)} games")


if __name__ == "__main__":
    main()
