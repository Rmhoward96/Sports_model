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

import duckdb

from sportsmodel import config, teams, venues, weather
from sportsmodel.db import get_duckdb, upsert_game_predictions
from sportsmodel.model import game, rates

MODEL_VERSION = "mlb-game-v1-context"
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


def load_pitcher_vectors(pitcher_ids) -> dict[int, dict]:
    """Best available allowed-rate vector (vs_hand=ALL) per pitcher, from the snapshot."""
    ids = sorted({int(p) for p in pitcher_ids if p is not None})
    if not ids:
        return {}
    path = PROFILE_DIR / "feat_pitcher_profile.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"""
        SELECT player_id, window_name, pa, {', '.join(OUTCOMES)}
        FROM read_parquet('{path}')
        WHERE vs_hand = 'ALL' AND player_id IN ({','.join(map(str, ids))})
        """
    ).fetchall()
    con.close()

    best: dict[int, tuple[int, dict]] = {}
    for r in rows:
        pid, win, pa = int(r[0]), r[1], r[2]
        vec = {o: r[3 + i] for i, o in enumerate(OUTCOMES)}
        rank = 0 if (win == "season" and pa >= _MIN_PA) else (1 if win == "career" else 2)
        if pid not in best or rank < best[pid][0]:
            best[pid] = (rank, vec)
    return {pid: v for pid, (_, v) in best.items()}


def load_team_offense_vectors() -> dict[str, dict]:
    """Best offensive vector (vs_hand=ALL) per team abbreviation, from the snapshot."""
    path = PROFILE_DIR / "feat_team_offense.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT team, window_name, pa, {', '.join(OUTCOMES)} "
        f"FROM read_parquet('{path}') WHERE vs_hand = 'ALL'"
    ).fetchall()
    con.close()
    best: dict[str, tuple[int, dict]] = {}
    for r in rows:
        team, win, pa = r[0], r[1], r[2]
        vec = {o: r[3 + i] for i, o in enumerate(OUTCOMES)}
        rank = 0 if (win == "season" and pa >= _MIN_TEAM_PA) else (1 if win == "career" else 2)
        if team not in best or rank < best[team][0]:
            best[team] = (rank, vec)
    return {t: v for t, (_, v) in best.items()}


def load_team_bullpen_vectors() -> dict[str, dict]:
    """Best bullpen allowed-rate vector (vs_hand=ALL) per team abbreviation."""
    path = PROFILE_DIR / "feat_team_bullpen.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(
        f"SELECT team, window_name, pa, {', '.join(OUTCOMES)} "
        f"FROM read_parquet('{path}') WHERE vs_hand = 'ALL'"
    ).fetchall()
    con.close()
    best: dict[str, tuple[int, dict]] = {}
    for r in rows:
        team, win, pa = r[0], r[1], r[2]
        vec = {o: r[3 + i] for i, o in enumerate(OUTCOMES)}
        rank = 0 if (win == "season" and pa >= _MIN_TEAM_PA) else (1 if win == "career" else 2)
        if team not in best or rank < best[team][0]:
            best[team] = (rank, vec)
    return {t: v for t, (_, v) in best.items()}


def blend(vec_sp: dict, vec_bp: dict, sp_share: float) -> dict:
    """Weighted average of two per-PA vectors (both sum to 1, so does the result)."""
    return {o: sp_share * vec_sp[o] + (1 - sp_share) * vec_bp[o] for o in OUTCOMES}


def load_park_factors() -> dict[str, float]:
    """{team abbrev: run park factor} for the team's home park."""
    path = PROFILE_DIR / "park_factors.parquet"
    con = duckdb.connect(":memory:")
    rows = con.execute(f"SELECT team, pf_runs FROM read_parquet('{path}')").fetchall()
    con.close()
    return {t: pf for t, pf in rows}


def load_league_vector() -> dict:
    """League-average per-PA vector (vs_hand=ALL, career) — the odds-ratio baseline."""
    path = PROFILE_DIR / "ref_league_rates.parquet"
    con = duckdb.connect(":memory:")
    r = con.execute(
        f"SELECT {', '.join(OUTCOMES)} FROM read_parquet('{path}') "
        f"WHERE vs_hand = 'ALL' AND window_name = 'career' LIMIT 1"
    ).fetchone()
    con.close()
    return {o: r[i] for i, o in enumerate(OUTCOMES)}


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
    pitchers = load_pitcher_vectors(ids)
    team_off = load_team_offense_vectors()
    bullpen = load_team_bullpen_vectors()
    league = load_league_vector()
    park = load_park_factors()

    def team_runs(off, opp_sp, opp_bp, pf, hr_mult):
        """Blend opposing starter (~62% of PAs) with bullpen; apply park + weather."""
        vec_sp = rates.matchup_vector(off, opp_sp, league)
        vec_bp = rates.matchup_vector(off, opp_bp, league) if opp_bp else vec_sp
        vec = blend(vec_sp, vec_bp, _STARTER_SHARE)
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
        home_runs = team_runs(home_off, away_sp, bullpen.get(away_abbrev), pf, hr_mult)
        away_runs = team_runs(away_off, home_sp, bullpen.get(home_abbrev), pf, hr_mult)
        res = game.win_total_probabilities(home_runs, away_runs)
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
