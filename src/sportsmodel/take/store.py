"""Postgres store for the 'take these' list (db/migration_take_these.sql).

take_picks and take_verdict are APPEND-ONLY (trigger + revoked privileges): insert_picks / insert_verdict only ever
INSERT ... ON CONFLICT DO NOTHING. Every function takes `connect_timeout` / `statement_timeout_ms` keywords (default:
the pick ledger's short ones, so a dead database cannot hang a job) and returns the number of rows ATTEMPTED unless
stated otherwise. Nothing here reads or writes until the migration has run: callers check `missing_tables()` first."""
from __future__ import annotations

import json
from datetime import datetime

from .. import db

TABLES: tuple[str, ...] = ("take_events", "take_prices", "take_picks", "take_pick_state", "take_closes",
                           "take_results", "take_api_calls", "take_verdict")

EVENT_COLUMNS = ("event_id", "league", "home_team", "away_team", "commence_time", "game_pk")
PRICE_COLUMNS = ("pulled_at", "event_id", "market", "side", "book", "league", "line", "price", "last_update")
# take_picks insert order; first_seen is the pick's pulled_at, run_id is the argument (see pick_tuple).
PICK_COLUMNS = ("event_id", "market", "side", "league", "book", "line", "price", "ev", "fair_prob", "good_until",
                "better_line", "ref_line", "pin_prices", "first_seen", "run_id")
STATE_COLUMNS = ("event_id", "market", "side", "league", "book", "line", "price", "ev", "fair_prob", "good_until",
                 "better_line", "ref_line", "others", "first_seen", "last_seen")
CLOSE_COLUMNS = ("event_id", "market", "prices", "captured_at", "minutes_to_start")
# insert_results writes the grade; outcome / units / settled_at arrive later through set_outcomes.
RESULT_COLUMNS = ("pick_id", "graded_by", "close_fair", "clv", "beat", "close_line", "close_minutes", "graded_at")
CALL_COLUMNS = ("job", "league", "endpoint", "credits_used", "credits_remaining", "books_missing", "run_id")
VERDICT_COLUMNS = ("id", "judged_at", "n", "beat_rate", "mean_clv", "lower", "passed", "criteria", "pick_ids")
TEST_COLUMNS = ("id", "event_id", "league", "market", "side", "book", "line", "price", "ev", "fair_prob",
                "better_line", "ref_line", "first_seen", "home_team", "away_team", "commence_time", "graded_by",
                "clv", "beat", "close_fair", "close_line", "close_minutes", "graded_at", "outcome", "units")

_JSONB = {"pin_prices", "others", "prices", "criteria"}


def _values(cols: tuple[str, ...]) -> str:
    return ", ".join("%s::jsonb" if c in _JSONB else "%s::bigint[]" if c == "pick_ids" else "%s" for c in cols)


def _insert(table: str, cols: tuple[str, ...]) -> str:
    return f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({_values(cols)})"


def _set_excluded(cols: tuple[str, ...], skip: tuple[str, ...]) -> str:
    return ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in skip)


UPSERT_EVENTS_SQL = (
    _insert("take_events", EVENT_COLUMNS) + " ON CONFLICT (event_id) DO UPDATE SET "
    "commence_time = EXCLUDED.commence_time, home_team = EXCLUDED.home_team, away_team = EXCLUDED.away_team, "
    "game_pk = COALESCE(EXCLUDED.game_pk, take_events.game_pk), updated_at = now()")
INSERT_PRICES_SQL = (_insert("take_prices", PRICE_COLUMNS)
                     + " ON CONFLICT (pulled_at, event_id, market, side, book) DO NOTHING")
INSERT_PICKS_SQL = _insert("take_picks", PICK_COLUMNS) + " ON CONFLICT (event_id, market, side) DO NOTHING"
# first_seen is deliberately not updated: a pick that drops and returns keeps its original sighting.
UPSERT_STATE_SQL = (
    f"INSERT INTO take_pick_state ({', '.join(STATE_COLUMNS)}, active, dropped_at) "
    f"VALUES ({_values(STATE_COLUMNS)}, true, NULL) "
    "ON CONFLICT (event_id, market, side) DO UPDATE SET "
    + _set_excluded(STATE_COLUMNS, ("event_id", "market", "side", "first_seen"))
    + ", active = true, dropped_at = NULL")
# Active rows of the league whose event was in the pull but whose key was not among the picks.
DROP_STATE_SQL = (
    "UPDATE take_pick_state SET active = false, dropped_at = %s "
    "WHERE active AND league = %s AND event_id = ANY(%s) AND NOT EXISTS ("
    "SELECT 1 FROM unnest(%s::text[], %s::text[], %s::text[]) AS k(e, m, s) "
    "WHERE k.e = take_pick_state.event_id AND k.m = take_pick_state.market AND k.s = take_pick_state.side)")
# A pick on a game that has started is over, whatever the pull saw (or whether the league was pulled at all).
EXPIRE_STATE_SQL = (
    "UPDATE take_pick_state s SET active = false, dropped_at = e.commence_time FROM take_events e "
    "WHERE e.event_id = s.event_id AND s.active AND e.commence_time <= %s")
UPSERT_CLOSES_SQL = (
    _insert("take_closes", CLOSE_COLUMNS) + " ON CONFLICT (event_id, market) DO UPDATE SET "
    + _set_excluded(CLOSE_COLUMNS, ("event_id", "market"))
    + " WHERE EXCLUDED.captured_at > take_closes.captured_at")
INSERT_RESULTS_SQL = _insert("take_results", RESULT_COLUMNS) + " ON CONFLICT (pick_id) DO NOTHING"
INSERT_VERDICT_SQL = _insert("take_verdict", VERDICT_COLUMNS) + " ON CONFLICT (id) DO NOTHING"

CLOSE_TARGETS_SQL = (
    "SELECT DISTINCT e.league, p.market FROM take_picks p JOIN take_events e ON e.event_id = p.event_id "
    "LEFT JOIN take_results r ON r.pick_id = p.id WHERE r.pick_id IS NULL "
    "AND e.commence_time BETWEEN %s + %s * interval '1 minute' AND %s + %s * interval '1 minute'")
UNGRADED_SQL = (
    "SELECT p.id, p.event_id, p.league, p.market, p.side, p.line, p.price, p.ref_line, "
    "c.prices, c.captured_at, c.minutes_to_start FROM take_picks p "
    "JOIN take_events e ON e.event_id = p.event_id LEFT JOIN take_results r ON r.pick_id = p.id "
    "LEFT JOIN take_closes c ON c.event_id = p.event_id AND c.market = p.market "
    "WHERE r.pick_id IS NULL AND e.commence_time <= %s ORDER BY p.id")
_UNSETTLED_FROM = (
    "FROM take_results r JOIN take_picks p ON p.id = r.pick_id JOIN take_events e ON e.event_id = p.event_id "
    "WHERE r.outcome IS NULL AND e.commence_time < now() - interval '3 hours'")
UNSETTLED_LEAGUES_SQL = f"SELECT DISTINCT e.league {_UNSETTLED_FROM} ORDER BY e.league"
UNSETTLED_SQL = (f"SELECT p.id, p.event_id, p.market, p.side, p.line, p.price, p.league {_UNSETTLED_FROM} "
                 "AND p.league = %s ORDER BY p.id")


def pick_tuple(pick: dict, run_id: str) -> tuple:
    """A pick as a PICK_COLUMNS-ordered tuple (pin_prices JSON-encoded; first_seen is the pick's pulled_at)."""
    vals = dict(pick, first_seen=pick["pulled_at"], run_id=run_id, pin_prices=json.dumps(pick["pin_prices"]))
    return tuple(vals.get(c) for c in PICK_COLUMNS)


def _state_tuple(pick: dict, pulled_at: datetime) -> tuple:
    vals = dict(pick, others=json.dumps(pick.get("others") or []), first_seen=pulled_at, last_seen=pulled_at)
    return tuple(vals.get(c) for c in STATE_COLUMNS)


def _json(v):
    return json.loads(v) if isinstance(v, str) else v


def _tuples(rows: list[dict], cols: tuple[str, ...]) -> list[tuple]:
    return [tuple(json.dumps(r.get(c)) if c in _JSONB else r.get(c) for c in cols) for r in rows]


def _run_many(sql: str, vals: list[tuple], connect_timeout, statement_timeout_ms) -> int:
    if not vals:
        return 0
    with db._pg(connect_timeout) as conn, conn.cursor() as cur:
        db._limit_statements(cur, statement_timeout_ms)
        cur.executemany(sql, vals)
        conn.commit()
    return len(vals)


def _read(sql: str, params, connect_timeout, statement_timeout_ms) -> list[tuple]:
    with db._pg(connect_timeout) as conn, conn.cursor() as cur:
        db._limit_statements(cur, statement_timeout_ms)
        cur.execute(sql, params)
        return cur.fetchall()


def missing_tables(*, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                   statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> list[str]:
    """The take tables that do not exist in `public` yet, in TABLES order (jobs warn and exit 0 until the
    migration has run)."""
    return db._missing_public_tables(list(TABLES), connect_timeout, statement_timeout_ms)


def upsert_events(rows: list[dict], *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                  statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Upsert events; refreshes start time and teams, and keeps an existing game_pk when the new one is None."""
    return _run_many(UPSERT_EVENTS_SQL, _tuples(rows, EVENT_COLUMNS), connect_timeout, statement_timeout_ms)


def insert_prices(rows: list[dict], *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                  statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    return _run_many(INSERT_PRICES_SQL, _tuples(rows, PRICE_COLUMNS), connect_timeout, statement_timeout_ms)


def prune_prices(before: datetime, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                 statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Delete raw prices pulled before `before`; returns the number deleted."""
    with db._pg(connect_timeout) as conn, conn.cursor() as cur:
        db._limit_statements(cur, statement_timeout_ms)
        cur.execute("DELETE FROM take_prices WHERE pulled_at < %s", (before,))
        deleted = cur.rowcount
        conn.commit()
    return deleted


def insert_picks(picks: list[dict], run_id: str, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                 statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Append picks. INSERT only; a pick for an (event, market, side) that already has one is a no-op."""
    return _run_many(INSERT_PICKS_SQL, [pick_tuple(p, run_id) for p in picks], connect_timeout,
                     statement_timeout_ms)


def sync_state(league: str, event_ids: list[str], picks: list[dict], pulled_at: datetime, *,
               connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
               statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> None:
    """Make take_pick_state match this pull: upsert every pick as active (keeping first_seen), then deactivate
    the league's active rows for the pulled `event_ids` whose key is not among `picks`. Games that have started are
    ended separately, for every league, by `expire_started`."""
    if not picks and not event_ids:
        return
    with db._pg(connect_timeout) as conn, conn.cursor() as cur:
        db._limit_statements(cur, statement_timeout_ms)
        if picks:
            cur.executemany(UPSERT_STATE_SQL, [_state_tuple(p, pulled_at) for p in picks])
        cur.execute(DROP_STATE_SQL, (pulled_at, league, list(event_ids), [p["event_id"] for p in picks],
                                     [p["market"] for p in picks], [p["side"] for p in picks]))
        conn.commit()


def expire_started(now: datetime, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                   statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """End every active pick (any league) whose event has started by `now`: active=false, dropped_at=commence_time.
    Independent of the odds pull, so a skipped or floor-stopped league still expires. Returns the rows ended."""
    with db._pg(connect_timeout) as conn, conn.cursor() as cur:
        db._limit_statements(cur, statement_timeout_ms)
        cur.execute(EXPIRE_STATE_SQL, (now,))
        ended = cur.rowcount
        conn.commit()
    return ended


def log_call(job: str, league: str | None, endpoint: str | None, credits_used: int | None,
             credits_remaining: int | None, books_missing: list[str] | None, run_id: str | None, *,
             connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
             statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> None:
    row = (job, league, endpoint, credits_used, credits_remaining,
           list(books_missing) if books_missing is not None else None, run_id)
    _run_many(_insert("take_api_calls", CALL_COLUMNS), [row], connect_timeout, statement_timeout_ms)


def recent_calls(since: datetime, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                 statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> list[dict]:
    """The paid-call log since `since` (credits used and books missing per call), oldest first."""
    keys = ("job", "league", "credits_used", "books_missing", "called_at")
    sql = f"SELECT {', '.join(keys)} FROM take_api_calls WHERE called_at >= %s ORDER BY called_at, id"
    return [dict(zip(keys, r)) for r in _read(sql, (since,), connect_timeout, statement_timeout_ms)]


def close_targets(now: datetime, lo_min: float, hi_min: float, *,
                  connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                  statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> dict[str, set[str]]:
    """league -> markets holding ungraded picks whose event starts between `lo_min` and `hi_min` minutes from now."""
    out: dict[str, set[str]] = {}
    for league, market in _read(CLOSE_TARGETS_SQL, (now, lo_min, now, hi_min), connect_timeout, statement_timeout_ms):
        out.setdefault(league, set()).add(market)
    return out


def upsert_closes(rows: list[dict], *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                  statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Upsert closes; an existing close is overwritten only by a later `captured_at`."""
    return _run_many(UPSERT_CLOSES_SQL, _tuples(rows, CLOSE_COLUMNS), connect_timeout, statement_timeout_ms)


def ungraded_started(now: datetime, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                     statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> list[dict]:
    """Picks whose event has started and that have no result yet, each with its close row (or None) under "close"."""
    out = []
    for (pick_id, event_id, league, market, side, line, price, ref_line,
         prices, captured_at, minutes) in _read(UNGRADED_SQL, (now,), connect_timeout, statement_timeout_ms):
        close = None if captured_at is None else {"event_id": event_id, "market": market, "prices": _json(prices),
                                                  "captured_at": captured_at, "minutes_to_start": minutes}
        out.append({"pick_id": pick_id, "event_id": event_id, "league": league, "market": market, "side": side,
                    "line": line, "price": price, "ref_line": ref_line, "close": close})
    return out


def insert_results(rows: list[dict], *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                   statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Insert grades (RESULT_COLUMNS); a pick that already has a result is left alone."""
    return _run_many(INSERT_RESULTS_SQL, _tuples(rows, RESULT_COLUMNS), connect_timeout, statement_timeout_ms)


def unsettled_leagues(*, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                      statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> list[str]:
    """Leagues with a graded pick still missing its outcome whose event started over 3 hours ago."""
    return [r[0] for r in _read(UNSETTLED_LEAGUES_SQL, None, connect_timeout, statement_timeout_ms)]


def unsettled(league: str, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
              statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> list[dict]:
    keys = ("pick_id", "event_id", "market", "side", "line", "price", "league")
    return [dict(zip(keys, r)) for r in _read(UNSETTLED_SQL, (league,), connect_timeout, statement_timeout_ms)]


def store_scores(finals: dict[str, tuple[int, int]], *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                 statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Record final scores {event_id: (home, away)} on take_events and mark the events completed."""
    sql = ("UPDATE take_events SET home_score = %s, away_score = %s, completed = true, updated_at = now() "
           "WHERE event_id = %s")
    return _run_many(sql, [(h, a, eid) for eid, (h, a) in finals.items()], connect_timeout, statement_timeout_ms)


def set_outcomes(rows: list[dict], *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                 statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> int:
    """Settle graded picks: rows of {pick_id, outcome, units}."""
    sql = "UPDATE take_results SET outcome = %s, units = %s, settled_at = now() WHERE pick_id = %s"
    return _run_many(sql, [(r["outcome"], r["units"], r["pick_id"]) for r in rows], connect_timeout,
                     statement_timeout_ms)


def test_rows(*, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
              statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> list[dict]:
    """Every row of take_test_v (TEST_COLUMNS), oldest pick first."""
    sql = f"SELECT {', '.join(TEST_COLUMNS)} FROM take_test_v ORDER BY first_seen, id"
    return [dict(zip(TEST_COLUMNS, r)) for r in _read(sql, None, connect_timeout, statement_timeout_ms)]


def get_verdict(*, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> dict | None:
    rows = _read(f"SELECT {', '.join(VERDICT_COLUMNS)} FROM take_verdict WHERE id = 1", None, connect_timeout,
                 statement_timeout_ms)
    return dict(zip(VERDICT_COLUMNS, rows[0])) if rows else None


def insert_verdict(row: dict, *, connect_timeout: int | None = db.LEDGER_CONNECT_TIMEOUT,
                   statement_timeout_ms: int | None = db.LEDGER_STATEMENT_TIMEOUT_MS) -> bool:
    """Append the one verdict row (id = 1). Returns whether it was inserted (False: a verdict already exists)."""
    vals = tuple(1 if c == "id" else json.dumps(row[c]) if c in _JSONB else row[c] for c in VERDICT_COLUMNS)
    with db._pg(connect_timeout) as conn, conn.cursor() as cur:
        db._limit_statements(cur, statement_timeout_ms)
        cur.execute(INSERT_VERDICT_SQL, vals)
        inserted = cur.rowcount == 1
        conn.commit()
    return inserted
