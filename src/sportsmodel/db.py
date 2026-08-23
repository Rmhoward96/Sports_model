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
    ]
    key = ("game_pk", "player_id", "market", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO prop_predictions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market, model_version) "
        f"DO UPDATE SET {updates}, generated_at = now()"
    )
    rows = [tuple(r.get(c) for c in cols) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def upsert_prediction_results(records: list[dict]) -> int:
    """Upsert graded predictions into Supabase `prediction_results` (idempotent)."""
    if not records:
        return 0
    cols = ["game_pk", "market", "player_id", "player_name", "model_version",
            "game_date", "model_number", "closing_line", "closing_price", "lean",
            "actual", "result", "profit", "edge",
            "model_prob", "market_prob", "ev"]
    key = ("game_pk", "market", "player_id", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO prediction_results ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, market, player_id, model_version) "
        f"DO UPDATE SET {updates}, graded_at = now()"
    )
    rows = [tuple(r.get(c) for c in cols) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


_BOARD_COLS = ["sport", "game_pk", "game_date", "commence_time", "matchup",
               "market", "market_label", "player_id", "player_name", "team",
               "pick_label", "side", "line", "odds", "book",
               "model_prob", "implied_prob", "ev", "is_pick"]
_PICK_INSERT_COLS = ["game_pk", "market", "player_id", "sport", "game_date",
                     "commence_time", "matchup", "market_label", "player_name", "team",
                     "pick_label", "side", "line", "bet_odds", "bet_book",
                     "model_prob", "novig_bet", "ev_bet"]


def clear_board_picks() -> None:
    """Empty `board_picks` so a regeneration is a true full refresh -- an upsert alone
    leaves orphaned rows for picks the current run no longer produces (e.g. a market
    that got excluded, or a line that dropped off the slate)."""
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM board_picks")
        conn.commit()


def upsert_board_picks(records: list[dict]) -> int:
    """Full-refresh the live `board_picks` table (upsert on game/market/player)."""
    if not records:
        return 0
    key = ("game_pk", "market", "player_id")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _BOARD_COLS if c not in key)
    ph = ", ".join(["%s"] * len(_BOARD_COLS))
    sql = (f"INSERT INTO board_picks ({', '.join(_BOARD_COLS)}) VALUES ({ph}) "
           f"ON CONFLICT (game_pk, market, player_id) "
           f"DO UPDATE SET {updates}, generated_at = now()")
    rows = [tuple(r.get(c) for c in _BOARD_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def insert_new_picks(records: list[dict]) -> int:
    """Insert-once into `picks` (locks the first-+EV bet; existing rows untouched)."""
    if not records:
        return 0
    ph = ", ".join(["%s"] * len(_PICK_INSERT_COLS))
    sql = (f"INSERT INTO picks ({', '.join(_PICK_INSERT_COLS)}) VALUES ({ph}) "
           f"ON CONFLICT (game_pk, market, player_id) DO NOTHING")
    rows = [tuple(r.get(c) for c in _PICK_INSERT_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def update_graded_picks(records: list[dict]) -> int:
    """Fill graded outcome + CLV on existing `picks` rows (by game/market/player)."""
    if not records:
        return 0
    sql = ("UPDATE picks SET status = 'graded', actual = %s, result = %s, profit = %s, "
           "novig_close = %s, clv = %s, home_score = %s, away_score = %s, graded_at = now() "
           "WHERE game_pk = %s AND market = %s AND player_id = %s")
    rows = [(r["actual"], r["result"], r["profit"], r["novig_close"], r["clv"],
             r.get("home_score"), r.get("away_score"),
             r["game_pk"], r["market"], r["player_id"]) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def upsert_odds_snapshot(records: list[dict]) -> int:
    """Insert odds snapshots into Supabase `odds_snapshot` (idempotent per capture)."""
    if not records:
        return 0
    cols = ["game_pk", "market", "side", "player_name", "book", "line",
            "price", "commence_time", "captured_at"]
    placeholders = ", ".join(["%s"] * len(cols))
    # captured_at is in the PK, so a re-run of the same pull is a no-op.
    sql = (
        f"INSERT INTO odds_snapshot ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT DO NOTHING"
    )
    rows = [tuple(r.get(c) for c in cols) for r in records]
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
        "game_pk", "model_version", "game_date",
        "home_team_name", "away_team_name",
        "home_probable_pitcher_name", "away_probable_pitcher_name",
        "pred_home_score", "pred_away_score", "pred_total", "pred_margin",
        "home_win_prob", "total_dist", "margin_dist",
    ]
    key = ("game_pk", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    placeholders = ", ".join(["%s"] * len(cols))
    sql = (
        f"INSERT INTO game_predictions ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, model_version) DO UPDATE SET {updates}, generated_at = now()"
    )
    rows = [tuple(r.get(c) for c in cols) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
