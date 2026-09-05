"""Warehouse connections: DuckDB (local crunch) and optional Supabase Postgres (serving)."""
from __future__ import annotations

from pathlib import Path

import duckdb

from . import config


def get_duckdb(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the local DuckDB warehouse."""
    config.DUCKDB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(config.DUCKDB_PATH), read_only=read_only)
    con.execute("INSTALL json; LOAD json;")
    return con


def get_postgres():
    """Open a Supabase/Postgres connection, or raise if DATABASE_URL is unset."""
    if not config.DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Copy .env.example to .env and add your "
            "Supabase connection string, or run local-only against DuckDB."
        )
    import psycopg  # imported lazily so local-only workflows need no driver

    return psycopg.connect(config.DATABASE_URL)


def read_parquet_glob(con: duckdb.DuckDBPyConnection, pattern: str | Path):
    """Register a parquet glob as a queryable relation."""
    return con.read_parquet(str(pattern))


def upsert_daily_schedule(records: list[dict]) -> int:
    """Upsert daily-slate records into Supabase `daily_schedule` (idempotent on game_pk).

    Returns rows written. Requires DATABASE_URL and the daily_schedule table
    (db/serving_bootstrap.sql). Safe to call every run — re-pulls overwrite in place.
    """
    if not records:
        return 0
    cols = [
        "game_pk", "game_date", "status", "venue_id", "venue_name",
        "home_team_id", "home_team_name", "away_team_id", "away_team_name",
        "home_probable_pitcher_id", "home_probable_pitcher_name",
        "away_probable_pitcher_id", "away_probable_pitcher_name",
    ]
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "game_pk")
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO daily_schedule ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk) DO UPDATE SET {updates}, updated_at = now()"
    )
    rows = [tuple(r.get(c) for c in cols) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def upsert_prop_predictions(records: list[dict]) -> int:
    """Upsert player-prop projections into Supabase `prop_predictions`.

    Idempotent on (game_pk, player_id, market, model_version) — so a later confirmed
    lineup run overwrites the earlier projected-lineup rows in place.
    """
    if not records:
        return 0
    cols = [
        "game_pk", "player_id", "market", "model_version", "game_date",
        "player_name", "team_name", "batting_slot", "projected_pa",
        "lineup_source", "projected_mean", "line", "prob_over", "dist",
        "sport",
    ]
    key = ("game_pk", "player_id", "market", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO prop_predictions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market, model_version) "
        f"DO UPDATE SET {updates}, generated_at = now()"
    )
    rows = [tuple(r.get(c, "mlb") if c == "sport" else r.get(c) for c in cols) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


_PREDICTION_ACCURACY_COLS = [
    "sport", "game_pk", "game_date", "home_team_name", "away_team_name",
    "win_prob", "predicted_winner", "actual_winner", "winner_correct",
    "pred_margin", "actual_margin", "margin_error", "spread_covered",
    "pred_total", "actual_total", "total_error", "total_over",
    "market_spread", "market_total", "spread_pick_correct", "total_pick_correct",
]


def upsert_prediction_accuracy(records: list[dict]) -> int:
    """Upsert graded prediction-accuracy rows into Supabase `prediction_accuracy`.

    Idempotent on (sport, game_pk) -- re-grading a game overwrites in place.
    Requires DATABASE_URL and the prediction_accuracy table
    (db/migration_prediction_tool.sql).
    """
    if not records:
        return 0
    key = ("sport", "game_pk")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _PREDICTION_ACCURACY_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_PREDICTION_ACCURACY_COLS))
    sql = (
        f"INSERT INTO prediction_accuracy ({', '.join(_PREDICTION_ACCURACY_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, game_pk) DO UPDATE SET {updates}, graded_at = now()"
    )
    rows = [tuple(r.get(c) for c in _PREDICTION_ACCURACY_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def upsert_game_predictions(records: list[dict]) -> int:
    """Upsert game-level predictions into Supabase `game_predictions`.

    Idempotent on (game_pk, model_version) — re-running a model version overwrites.
    """
    if not records:
        return 0
    cols = [
        "game_pk", "model_version", "game_date", "commence_time",
        "home_team_name", "away_team_name",
        "home_probable_pitcher_name", "away_probable_pitcher_name",
        "pred_home_score", "pred_away_score", "pred_total", "pred_margin",
        "home_win_prob", "total_dist", "margin_dist",
        "market_spread", "market_total",
        "sport",
    ]
    key = ("game_pk", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO game_predictions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, model_version) DO UPDATE SET {updates}, generated_at = now()"
    )
    rows = [tuple(r.get(c, "mlb") if c == "sport" else r.get(c) for c in cols) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
