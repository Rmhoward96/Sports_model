"""Generate game-level predictions for the current slate and write them out.

v0 model (`mlb-game-v0-pitching`): each team's expected runs is driven by the
OPPOSING starter's run-prevention profile (docs/methodology.md Part B). Team-offense
differentiation is the v1 upgrade. Reads the schedule (Supabase daily_schedule if
DATABASE_URL is set, else the local DuckDB stg_schedule_raw) and pitcher profiles
from the local DuckDB warehouse; writes to game_predictions (Supabase or DuckDB).

Usage:
    uv run python scripts/generate_predictions.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import get_duckdb, upsert_game_predictions
from sportsmodel.model import game

MODEL_VERSION = "mlb-game-v0-pitching"
OUTCOMES = ["p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out"]
_MIN_PA = 150  # below this, prefer the 'career' window over 'season'


def _pitcher_vec(con, pitcher_id: int) -> dict | None:
    """Best available allowed-rate vector for a pitcher (vs_hand=ALL)."""
    if pitcher_id is None:
        return None
    row = con.execute(
        f"""
        SELECT {', '.join(OUTCOMES)}, pa, window_name FROM feat_pitcher_profile
        WHERE player_id = ? AND vs_hand = 'ALL'
        ORDER BY CASE WHEN window_name='season' AND pa >= ? THEN 0
                      WHEN window_name='career' THEN 1 ELSE 2 END
        LIMIT 1
        """,
        [pitcher_id, _MIN_PA],
    ).fetchone()
    if not row:
        return None
    return {o: row[i] for i, o in enumerate(OUTCOMES)}


def _load_schedule(con) -> list[dict]:
    today = date.today().isoformat()
    cols = [
        "game_pk", "game_date", "home_team_name", "away_team_name",
        "home_probable_pitcher_id", "home_probable_pitcher_name",
        "away_probable_pitcher_id", "away_probable_pitcher_name",
    ]
    if config.DATABASE_URL:
        from sportsmodel.db import get_postgres
        with get_postgres() as pg, pg.cursor() as cur:
            cur.execute(
                f"SELECT {', '.join(cols)} FROM daily_schedule WHERE game_date = %s",
                [today],
            )
            rows = cur.fetchall()
    else:
        rows = con.execute(
            f"SELECT {', '.join(cols)} FROM stg_schedule_raw WHERE game_date = ?",
            [today],
        ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def main() -> None:
    con = get_duckdb(read_only=True)
    games = _load_schedule(con)
    print(f"{len(games)} games on today's slate")

    preds: list[dict] = []
    skipped = 0
    for g in games:
        home_vec = _pitcher_vec(con, g["home_probable_pitcher_id"])
        away_vec = _pitcher_vec(con, g["away_probable_pitcher_id"])
        if home_vec is None or away_vec is None:
            skipped += 1
            continue
        # Home offense is limited by the AWAY starter, and vice versa.
        home_runs = game.expected_runs(away_vec)
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
    con.close()

    print(f"predicted {len(preds)}, skipped {skipped} (missing starter or profile)")
    if not preds:
        return

    if config.DATABASE_URL:
        n = upsert_game_predictions(preds)
        print(f"Upserted {n} rows into Supabase game_predictions.")
    else:
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
                """INSERT OR REPLACE INTO game_predictions VALUES
                   (?,?,?,?,?,?,?,?,?,?,?,?)""",
                [p["game_pk"], p["model_version"], p["game_date"],
                 p["home_team_name"], p["away_team_name"],
                 p["home_probable_pitcher_name"], p["away_probable_pitcher_name"],
                 p["pred_home_score"], p["pred_away_score"], p["pred_total"],
                 p["pred_margin"], p["home_win_prob"]],
            )
        con.close()
        print(f"Wrote {len(preds)} rows into local DuckDB game_predictions.")


if __name__ == "__main__":
    main()
