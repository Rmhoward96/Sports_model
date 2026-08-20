"""Generate game-level predictions for the current slate and write them out.

v1 model (`mlb-game-v1-teamoff`): each team's per-PA outcome vector is the odds-ratio
blend of its own OFFENSE and the OPPOSING starter's run-prevention, relative to league
(docs/methodology.md Part A step 2 + Part B). Park/weather/bullpen are later upgrades.

Data sources (all committed snapshots so this runs in CI without the ~1GB raw backfill):
  - profiles: assets/profiles/{feat_pitcher_profile,feat_team_offense,ref_league_rates}.parquet
  - schedule: Supabase daily_schedule if DATABASE_URL is set, else local DuckDB
  - output: Supabase game_predictions if DATABASE_URL is set, else local DuckDB

Usage:
    uv run python scripts/generate_predictions.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config, profiles, teams, venues, weather
from sportsmodel.db import get_duckdb, upsert_game_predictions
from sportsmodel.model import calibration, game, rates

MODEL_VERSION = "mlb-game-v2-defense"
OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
_MIN_PA = 150       # below this in the season window, prefer 'career' (pitchers)
_MIN_TEAM_PA = 1000  # teams accrue PAs fast; season window is usually rich enough
_STARTER_SHARE = 0.62  # fraction of a team's PAs faced by the opposing starter [tunable]
PROFILE_DIR = config.PROJECT_ROOT / "assets" / "profiles"

SCHED_COLS = [
    "game_pk", "game_date",
    "home_team_id", "home_team_name", "away_team_id", "away_team_name",
    "home_probable_pitcher_id", "home_probable_pitcher_name",
    "away_probable_pitcher_id", "away_probable_pitcher_name",
]


def load_schedule() -> list[dict]:
    today = date.today().isoformat()
    if config.DATABASE_URL:
        from sportsmodel.db import get_postgres
        with get_postgres() as pg, pg.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(SCHED_COLS)} FROM daily_schedule WHERE game_date = %s",
                [today],
            )
            rows = cur.fetchall()
    else:
        con = get_duckdb(read_only=True)
        rows = con.execute(
            f"SELECT {', '.join(SCHED_COLS)} FROM stg_schedule_raw WHERE game_date = ?",
            [today],
        ).fetchall()
        con.close()
    return [dict(zip(SCHED_COLS, r)) for r in rows]


def blend(vec_sp: dict, vec_bp: dict, sp_share: float) -> dict:
    """Weighted average of two per-PA vectors (both sum to 1, so does the result)."""
    return {o: sp_share * vec_sp[o] + (1 - sp_share) * vec_bp[o] for o in OUTCOMES}


def write_predictions(preds: list[dict]) -> None:
    if config.DATABASE_URL:
        n = upsert_game_predictions(preds)
        print(f"Upserted {n} rows into Supabase game_predictions.")
        return
    con = get_duckdb()
    con.execute("""
        CREATE TABLE IF NOT EXISTS game_predictions (
            game_pk BIGINT, model_version TEXT, game_date DATE,
            home_team_name TEXT, away_team_name TEXT,
            home_probable_pitcher_name TEXT, away_probable_pitcher_name TEXT,
            pred_home_score REAL, pred_away_score REAL, pred_total REAL,
            pred_margin REAL, home_win_prob REAL,
            PRIMARY KEY (game_pk, model_version))
    """)
    for p in preds:
        con.execute(
            "INSERT OR REPLACE INTO game_predictions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [p["game_pk"], p["model_version"], p["game_date"],
             p["home_team_name"], p["away_team_name"],
             p["home_probable_pitcher_name"], p["away_probable_pitcher_name"],
             p["pred_home_score"], p["pred_away_score"], p["pred_total"],
             p["pred_margin"], p["home_win_prob"]],
        )
    con.close()
    print(f"Wrote {len(preds)} rows into local DuckDB game_predictions.")


def main() -> None:
    games = load_schedule()
    print(f"{len(games)} games on today's slate")

    ids = [g["home_probable_pitcher_id"] for g in games] + \
          [g["away_probable_pitcher_id"] for g in games]
    pitchers = profiles.load_pitcher_vectors(ids)
    team_off = profiles.load_team_offense_vectors()
    bullpen = profiles.load_team_bullpen_vectors()
    league = profiles.load_league_vector()
    park = profiles.load_park_factors()
    defense = profiles.load_team_defense()

    def team_runs(off, opp_sp, opp_bp, pf, hr_mult, opp_def):
        """Offense vs opposing starter (~62%)+bullpen, then defense, park, weather."""
        vec_sp = rates.matchup_vector(off, opp_sp, league)
        vec_bp = rates.matchup_vector(off, opp_bp, league) if opp_bp else vec_sp
        vec = blend(vec_sp, vec_bp, _STARTER_SHARE)
        if opp_def != 1.0:
            vec = game.apply_bip_defense(vec, opp_def)  # opposing fielders convert BIP
        if hr_mult != 1.0:
            vec = game.apply_hr_multiplier(vec, hr_mult)
        return game.expected_runs(vec, park_factor=pf)

    preds, skipped = [], 0
    for g in games:
        home_abbrev = teams.statcast_abbrev(g["home_team_id"])
        away_abbrev = teams.statcast_abbrev(g["away_team_id"])
        home_sp = pitchers.get(g["home_probable_pitcher_id"])
        away_sp = pitchers.get(g["away_probable_pitcher_id"])
        home_off = team_off.get(home_abbrev)
        away_off = team_off.get(away_abbrev)
        if None in (home_sp, away_sp, home_off, away_off):
            skipped += 1
            continue
        pf = park.get(home_abbrev, 1.0)  # both teams bat in the home park
        # Weather HR nudge for outdoor parks (both teams share the environment).
        p = venues.park(home_abbrev)
        hr_mult = 1.0
        if p and p[2]:
            temp = weather.fetch_game_temp(p[0], p[1], g["game_date"])
            if temp is not None:
                hr_mult = game.weather_hr_multiplier(temp)
        # Each offense is converted by the OPPOSING team's defense.
        home_runs = team_runs(home_off, away_sp, bullpen.get(away_abbrev), pf, hr_mult,
                              defense.get(away_abbrev, 1.0))
        away_runs = team_runs(away_off, home_sp, bullpen.get(home_abbrev), pf, hr_mult,
                              defense.get(home_abbrev, 1.0))
        res = game.win_total_probabilities(home_runs, away_runs)
        res["home_win_prob"] = calibration.calibrate("win_prob", res["home_win_prob"])
        preds.append({
            "game_pk": g["game_pk"], "model_version": MODEL_VERSION,
            "game_date": g["game_date"],
            "home_team_name": g["home_team_name"], "away_team_name": g["away_team_name"],
            "home_probable_pitcher_name": g["home_probable_pitcher_name"],
            "away_probable_pitcher_name": g["away_probable_pitcher_name"],
            **res,
        })

    print(f"predicted {len(preds)}, skipped {skipped} (missing starter or profile)")
    if preds:
        write_predictions(preds)


if __name__ == "__main__":
    main()
