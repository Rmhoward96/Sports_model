"""Generate game-level predictions for the current slate and write them out.

v0 model (`mlb-game-v0-pitching`): each team's expected runs is driven by the
OPPOSING starter's run-prevention profile (docs/methodology.md Part B). Team-offense
differentiation is the v1 upgrade.

Data sources:
  - pitcher profiles: committed snapshot assets/profiles/feat_pitcher_profile.parquet
    (so this runs in CI without the ~1GB raw backfill)
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

from sportsmodel import config
from sportsmodel.db import get_duckdb, upsert_game_predictions
from sportsmodel.model import game

MODEL_VERSION = "mlb-game-v0-pitching"
OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
_MIN_PA = 150  # below this in the season window, prefer 'career'
PROFILE_DIR = config.PROJECT_ROOT / "assets" / "profiles"

SCHED_COLS = [
    "game_pk", "game_date", "home_team_name", "away_team_name",
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
    vecs = load_pitcher_vectors(ids)

    preds, skipped = [], 0
    for g in games:
        home_vec = vecs.get(g["home_probable_pitcher_id"])
        away_vec = vecs.get(g["away_probable_pitcher_id"])
        if home_vec is None or away_vec is None:
            skipped += 1
            continue
        home_runs = game.expected_runs(away_vec)   # home offense limited by away SP
        away_runs = game.expected_runs(home_vec)
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
