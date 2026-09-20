"""Warehouse connections: DuckDB (local crunch) and optional Supabase Postgres (serving)."""
from __future__ import annotations

import json
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


_DESK_PICKS_COLS = [
    "sport", "game_pk", "model_version", "game_date", "commence_time",
    "matchup", "ml_pick", "spread_side", "spread_line", "total_side",
    "total_line", "confidence", "conviction_tier", "rationale", "agent_notes",
]


def upsert_desk_picks(records: list[dict]) -> int:
    """Upsert decision-desk picks into Supabase `desk_picks`.

    Idempotent on (sport, game_pk, model_version) -- a re-run of the same
    model_version for the same game overwrites the pick fields in place.
    `created_at` is intentionally excluded from both the column list and the
    DO UPDATE SET clause: it is set once by the table's DEFAULT now() on the
    first INSERT and must NOT change on a later update, so a pick's original
    "posted at" time survives edits (contrast with prediction_accuracy's
    graded_at, which IS bumped on every update -- here we want the opposite:
    never touch it after the first write).
    Requires DATABASE_URL and desk_picks (db/migration_decision_desk.sql).
    """
    if not records:
        return 0
    key = ("sport", "game_pk", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _DESK_PICKS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_DESK_PICKS_COLS))
    sql = (
        f"INSERT INTO desk_picks ({', '.join(_DESK_PICKS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, game_pk, model_version) DO UPDATE SET {updates}"
    )
    rows = [tuple(r.get(c) for c in _DESK_PICKS_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


_DESK_PICK_RESULTS_COLS = [
    "sport", "game_pk", "ml_correct", "spread_cover", "total_result",
    "clv_spread", "clv_total",
]


def upsert_desk_pick_results(records: list[dict]) -> int:
    """Upsert graded decision-desk results into Supabase `desk_pick_results`.

    Idempotent on (sport, game_pk) -- re-grading a game overwrites in place.
    Requires DATABASE_URL and desk_pick_results (db/migration_decision_desk.sql).
    """
    if not records:
        return 0
    key = ("sport", "game_pk")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _DESK_PICK_RESULTS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_DESK_PICK_RESULTS_COLS))
    sql = (
        f"INSERT INTO desk_pick_results ({', '.join(_DESK_PICK_RESULTS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, game_pk) DO UPDATE SET {updates}, graded_at = now()"
    )
    rows = [tuple(r.get(c) for c in _DESK_PICK_RESULTS_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


_EV_PICKS_COLS = [
    "sport", "game_pk", "market", "side", "model_version", "matchup",
    "commence_time", "base_prob", "true_prob", "edge", "desk_delta",
    "conviction_tier", "pinnacle_price", "open_pinnacle_price", "ev_pinnacle",
    "best_book", "best_price", "ev_best", "best_line_implied",
    "soft_vs_sharp_gap", "is_pick",
]

# Columns set once on first INSERT and NEVER overwritten on a later upsert --
# the pick-time (opening) Pinnacle price, so CLV grading measures the price the
# pick was first surfaced at, not the refreshed near-closing price. (created_at
# gets the same treatment via the table DEFAULT; this one is app-supplied so it
# must be excluded from DO UPDATE explicitly.)
_EV_PICKS_IMMUTABLE = {"open_pinnacle_price"}

_EV_PILOT_DEFAULT_MODEL_VERSION = "ev-pilot-v1"


def upsert_ev_picks(records: list[dict]) -> int:
    """Upsert +EV pilot board rows into Supabase `ev_picks`.

    Idempotent on (sport, game_pk, market, side, model_version) -- a re-run of
    the same model_version for the same (game, market, side) overwrites the
    row's fields in place. `created_at` is intentionally excluded from both
    the column list and the DO UPDATE SET clause: it is set once by the
    table's DEFAULT now() on the first INSERT and must NOT change on a later
    update, so a row's original "first posted" time survives recomputation
    (same pattern as upsert_desk_picks). A record that omits `model_version`
    defaults to "ev-pilot-v1".
    Requires DATABASE_URL and ev_picks (db/migration_ev_pilot.sql).
    """
    if not records:
        return 0
    key = ("sport", "game_pk", "market", "side", "model_version")
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in _EV_PICKS_COLS
        if c not in key and c not in _EV_PICKS_IMMUTABLE
    )
    placeholders = ", ".join(["%s"] * len(_EV_PICKS_COLS))
    sql = (
        f"INSERT INTO ev_picks ({', '.join(_EV_PICKS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, game_pk, market, side, model_version) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(
            r.get(c, _EV_PILOT_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c)
            for c in _EV_PICKS_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def clear_other_side_picks(records: list[dict]) -> int:
    """Demote the OPPOSITE side of every market just written, so a two-way
    market never shows both sides as picks.

    The engine writes exactly one side per (game, market) per build -- the
    desk's side, or the market-favored side. When that side flips between builds
    (e.g. the desk's ml_pick changes), the previous side's row persists with
    is_pick=true because `side` is part of the primary key, so BOTH sides end up
    surfaced and graded -- which locks in a guaranteed net loss. After each
    board build, this sets is_pick=false on any row for the same
    (sport, game_pk, market, model_version) whose side differs from the one just
    written. Idempotent; only touches rows still flagged is_pick=true."""
    if not records:
        return 0
    sql = (
        "UPDATE ev_picks SET is_pick = false "
        "WHERE sport = %s AND game_pk = %s AND market = %s "
        "AND model_version = %s AND side <> %s AND is_pick = true"
    )
    params = [
        (r.get("sport"), r.get("game_pk"), r.get("market"),
         r.get("model_version", _EV_PILOT_DEFAULT_MODEL_VERSION), r.get("side"))
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, params)
        cleared = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
    return cleared


_EV_RESULTS_COLS = ["sport", "game_pk", "market", "side", "won", "clv"]


def upsert_ev_results(records: list[dict]) -> int:
    """Upsert graded +EV pilot results into Supabase `ev_results`.

    Idempotent on (sport, game_pk, market, side) -- re-grading a row
    overwrites in place and bumps graded_at (same pattern as
    upsert_desk_pick_results).
    Requires DATABASE_URL and ev_results (db/migration_ev_pilot.sql).
    """
    if not records:
        return 0
    key = ("sport", "game_pk", "market", "side")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _EV_RESULTS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_EV_RESULTS_COLS))
    sql = (
        f"INSERT INTO ev_results ({', '.join(_EV_RESULTS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, game_pk, market, side) DO UPDATE SET {updates}, graded_at = now()"
    )
    rows = [tuple(r.get(c) for c in _EV_RESULTS_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


# =============================================================================
# +EV prop board (NFL player props) -- see db/migration_ev_prop_picks.sql
# =============================================================================

_EV_PROP_PICKS_COLS = [
    "sport", "game_pk", "player_id", "player_name", "market", "side", "line",
    "model_version", "matchup", "commence_time", "model_prob", "market_prob",
    "edge", "ev_best", "best_book", "best_price", "pinnacle_price",
    "open_pinnacle_price", "is_pick",
]

# Set once on first INSERT and NEVER overwritten on a later upsert -- the
# pick-time (opening) Pinnacle price, so CLV grading measures the price the
# pick was first surfaced at, not the refreshed near-closing price (same
# pattern as _EV_PICKS_IMMUTABLE). created_at gets the same treatment via the
# table DEFAULT and is excluded from the column list entirely.
_EV_PROP_PICKS_IMMUTABLE = frozenset({"open_pinnacle_price"})

_EV_PROP_DEFAULT_MODEL_VERSION = "props-sim-v1"


def upsert_ev_prop_picks(records: list[dict]) -> int:
    """Upsert +EV player-prop board rows into Supabase `ev_prop_picks`.

    Idempotent on (game_pk, player_id, market, line, model_version) -- a
    re-run of the same model_version for the same (game, player, market,
    line) overwrites the row's fields in place. `created_at` is intentionally
    excluded from both the column list and the DO UPDATE SET clause: it is
    set once by the table's DEFAULT now() on the first INSERT and must NOT
    change on a later update (same pattern as upsert_ev_picks).
    `open_pinnacle_price` is excluded from DO UPDATE (but not the column
    list) so it stays frozen at the pick-time price for CLV grading. A record
    that omits `model_version` defaults to "props-sim-v1".
    Requires DATABASE_URL and ev_prop_picks (db/migration_ev_prop_picks.sql).
    """
    if not records:
        return 0
    key = ("game_pk", "player_id", "market", "line", "model_version")
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in _EV_PROP_PICKS_COLS
        if c not in key and c not in _EV_PROP_PICKS_IMMUTABLE
    )
    placeholders = ", ".join(["%s"] * len(_EV_PROP_PICKS_COLS))
    sql = (
        f"INSERT INTO ev_prop_picks ({', '.join(_EV_PROP_PICKS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market, line, model_version) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(
            r.get(c, _EV_PROP_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c)
            for c in _EV_PROP_PICKS_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


def clear_stale_prop_line_picks(records: list[dict]) -> int:
    """Demote every OTHER line of the same (game, player, market) just
    written, so a shifted main line never leaves a stale pick behind.

    `ev_prop_picks`' primary key is (game_pk, player_id, market, line,
    model_version) -- line, not side -- so when a player's main line shifts
    between builds (e.g. the book moves from 68.5 to 69.5), the OLD line's
    row persists with is_pick=true because `line` is part of the primary key
    rather than the conflict target changing it in place. `ev_prop_picks_current`
    (DISTINCT ON game_pk, player_id, market, line) would then surface BOTH
    lines as live picks -- a phantom pick that was never actually the current
    board's line. After each board build, this sets is_pick=false on any row
    for the same (game_pk, player_id, market, model_version) whose line
    differs from the one just written. Idempotent; only touches rows still
    flagged is_pick=true (same pattern as clear_other_side_picks)."""
    if not records:
        return 0
    sql = (
        "UPDATE ev_prop_picks SET is_pick = false "
        "WHERE game_pk = %s AND player_id = %s AND market = %s "
        "AND model_version = %s AND line <> %s AND is_pick = true"
    )
    params = [
        (r.get("game_pk"), r.get("player_id"), r.get("market"),
         r.get("model_version", _EV_PROP_DEFAULT_MODEL_VERSION), r.get("line"))
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, params)
        cleared = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
    return cleared


_EV_PROP_RESULTS_COLS = [
    "sport", "game_pk", "player_id", "player_name", "market", "side", "line",
    "model_version", "commence_time", "model_prob", "novig_close", "actual",
    "result", "clv", "profit",
]


def upsert_ev_prop_results(records: list[dict]) -> int:
    """Upsert graded player-prop results into Supabase `ev_prop_results`.

    Idempotent on (game_pk, player_id, market, line, model_version) -- a
    re-grade overwrites in place. `created_at` is intentionally excluded from
    both the column list and the DO UPDATE SET clause: it is set once by the
    table's DEFAULT now() on the first INSERT and must NOT change on a later
    update (same pattern as upsert_ev_prop_picks). A record that omits
    `model_version` defaults to "props-sim-v1".
    Requires DATABASE_URL and ev_prop_results (db/migration_ev_prop_picks.sql).
    """
    if not records:
        return 0
    key = ("game_pk", "player_id", "market", "line", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _EV_PROP_RESULTS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_EV_PROP_RESULTS_COLS))
    sql = (
        f"INSERT INTO ev_prop_results ({', '.join(_EV_PROP_RESULTS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market, line, model_version) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(
            r.get(c, _EV_PROP_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c)
            for c in _EV_PROP_RESULTS_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


# =============================================================================
# +EV parlays (auto-parlay of juiced-favorite legs) -- see serving/parlay.py
# =============================================================================

_EV_PARLAYS_COLS = [
    "sport", "parlay_id", "model_version", "legs", "book", "parlay_price",
    "true_prob", "ev", "n_legs", "commence_time", "is_pick",
]


def upsert_ev_parlays(records: list[dict]) -> int:
    """Upsert +EV parlay tickets into Supabase `ev_parlays`.

    Idempotent on (sport, parlay_id, model_version) -- parlay_id is a stable
    hash of the leg set, so a rebuild with the same legs overwrites in place.
    `legs` is written as JSON. `created_at` is DEFAULT-set on first insert and
    never updated. Requires DATABASE_URL and ev_parlays (db/migration_ev_parlays.sql)."""
    if not records:
        return 0
    key = ("sport", "parlay_id", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _EV_PARLAYS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_EV_PARLAYS_COLS))
    sql = (
        f"INSERT INTO ev_parlays ({', '.join(_EV_PARLAYS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, parlay_id, model_version) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(
            json.dumps(r.get("legs")) if c == "legs"
            else (r.get(c, _EV_PILOT_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c))
            for c in _EV_PARLAYS_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


_NFL_SIM_DEFAULT_MODEL_VERSION = "sim-nfl-v1"

_NFL_SIM_COLS = [
    "game_pk", "model_version", "matchup", "commence_time",
    "sim_home_win_prob", "sim_margin", "sim_total", "disagreement",
]


def upsert_nfl_sim(records: list[dict]) -> int:
    """Upsert NFL sim-engine game outputs into Supabase `nfl_sim`.

    Idempotent on (game_pk, model_version) -- a re-run of the same
    model_version for the same game overwrites the row's fields in place.
    `created_at` is intentionally excluded from both the column list and the
    DO UPDATE SET clause: it is set once by the table's DEFAULT now() on the
    first INSERT and must NOT change on a later update (same pattern as
    upsert_ev_picks). A record that omits `model_version` defaults to
    "sim-nfl-v1".
    Requires DATABASE_URL and nfl_sim (db/migration_nfl_sim.sql).
    """
    if not records:
        return 0
    key = ("game_pk", "model_version")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _NFL_SIM_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_NFL_SIM_COLS))
    sql = (
        f"INSERT INTO nfl_sim ({', '.join(_NFL_SIM_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, model_version) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(
            r.get(c, _NFL_SIM_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c)
            for c in _NFL_SIM_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)


_NFL_PLAYER_SIM_COLS = [
    "game_pk", "player_id", "model_version", "name", "pos", "team",
    "market", "mean", "dist", "commence_time",
]


def upsert_nfl_player_sim(records: list[dict]) -> int:
    """Replace-per-game write of NFL sim player-prop outputs into `nfl_player_sim`.

    For each (game_pk, model_version) present in `records`, this first DELETEs
    that game's existing rows for that model_version, then inserts the fresh
    roster -- all in one transaction. A plain upsert-in-place would leave behind
    ORPHAN rows for players who were on the roster in an earlier run but have
    since dropped off (injury / depth change): those stale rows never get
    rewritten and then surface on the game page with old (often 0) projections
    for a player who isn't even active. Replacing the game's roster each run
    keeps `nfl_player_sim` in sync with the current active set. The generator
    only processes UPCOMING games, so a kicked-off game's stored pre-game sim is
    never touched (it must persist for the sim-vs-actual view).

    Idempotent per (game_pk, model_version). `dist` is written as JSON. A record
    that omits `model_version` defaults to "sim-nfl-v1". `created_at` now
    reflects the latest run for the game (the whole roster is rewritten
    together, so it stays consistent within a game). Requires DATABASE_URL and
    nfl_player_sim (db/migration_nfl_sim.sql)."""
    if not records:
        return 0
    game_versions = {
        (r.get("game_pk"), r.get("model_version", _NFL_SIM_DEFAULT_MODEL_VERSION))
        for r in records
    }
    placeholders = ", ".join(["%s"] * len(_NFL_PLAYER_SIM_COLS))
    insert_sql = (
        f"INSERT INTO nfl_player_sim ({', '.join(_NFL_PLAYER_SIM_COLS)}) "
        f"VALUES ({placeholders})"
    )
    rows = [
        tuple(
            json.dumps(r.get("dist")) if c == "dist"
            else (r.get(c, _NFL_SIM_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c))
            for c in _NFL_PLAYER_SIM_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        for game_pk, model_version in game_versions:
            cur.execute(
                "DELETE FROM nfl_player_sim WHERE game_pk = %s AND model_version = %s",
                (game_pk, model_version),
            )
        cur.executemany(insert_sql, rows)
        conn.commit()
    return len(rows)


def demote_stale_parlays(sport: str, keep_parlay_id: str | None,
                          model_version: str = _EV_PILOT_DEFAULT_MODEL_VERSION) -> int:
    """Set is_pick=false on any still-upcoming parlay for `sport` whose
    parlay_id isn't `keep_parlay_id` (None keeps nothing). The leg set can
    change between builds -> a new parlay_id -> the old parlay must be demoted so
    only the current ticket shows. Only touches commence_time > now() rows so a
    parlay whose games already kicked off (and may be graded) is left intact."""
    sql = (
        "UPDATE ev_parlays SET is_pick = false "
        "WHERE sport = %s AND model_version = %s AND is_pick = true "
        "AND commence_time > now() AND parlay_id <> %s"
    )
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute(sql, (sport, model_version, keep_parlay_id or ""))
        demoted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0
        conn.commit()
    return demoted


_EV_PARLAY_RESULTS_COLS = ["sport", "parlay_id", "won"]


def upsert_ev_parlay_results(records: list[dict]) -> int:
    """Upsert graded parlay results into Supabase `ev_parlay_results`.
    Idempotent on (sport, parlay_id); bumps graded_at on update."""
    if not records:
        return 0
    key = ("sport", "parlay_id")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _EV_PARLAY_RESULTS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_EV_PARLAY_RESULTS_COLS))
    sql = (
        f"INSERT INTO ev_parlay_results ({', '.join(_EV_PARLAY_RESULTS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (sport, parlay_id) DO UPDATE SET {updates}, graded_at = now()"
    )
    rows = [tuple(r.get(c) for c in _EV_PARLAY_RESULTS_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
