"""db/migration_take_these.sql and take/store.py: SQL shape, tuple building, column agreement, the append-only
guarantees and the missing-table guard. No live DB (fake connections)."""
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sportsmodel import db
from sportsmodel.take import store
from sportsmodel.take.constants import LEAGUES

MIGRATION = (Path(__file__).resolve().parents[2] / "db" / "migration_take_these.sql").read_text()
NOW = datetime(2026, 11, 2, 18, 0, tzinfo=UTC)

SAMPLE_PICK = {
    "event_id": "ev1", "league": "nfl", "market": "spread", "side": "home", "book": "fanduel", "line": -2.5,
    "price": -105, "ev": 0.021, "fair_prob": 0.52, "good_until": -112, "better_line": True, "ref_line": -3.0,
    "pin_prices": {"home": {"line": -3.0, "price": -110}, "away": {"line": 3.0, "price": -110}},
    "others": [{"book": "betmgm", "line": -2.5, "price": -108, "ev": 0.016}],
    "pulled_at": NOW, "commence_time": datetime(2026, 11, 3, tzinfo=UTC),
}


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


def _view(name):
    m = re.search(rf"CREATE OR REPLACE VIEW {name} WITH \(security_invoker = true\) AS\n(.+?);", MIGRATION, re.DOTALL)
    assert m, f"no security_invoker view {name}"
    return m.group(1)


# ---- migration --------------------------------------------------------------------------------

def test_migration_idempotent_shape():
    for t in store.TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {t} " in MIGRATION
    assert "DROP POLICY IF EXISTS" in MIGRATION and "CREATE OR REPLACE VIEW take_list_v" in MIGRATION
    assert "'soccer'" not in MIGRATION and "'epl'" in MIGRATION and "'mls'" in MIGRATION
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == len(store.TABLES) == 8
    assert MIGRATION.count("CREATE TRIGGER") == MIGRATION.count("DROP TRIGGER IF EXISTS") == 4
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE)\b", MIGRATION, re.MULTILINE | re.IGNORECASE)


def test_league_check_lists_exactly_the_eight_constants_leagues():
    m = re.search(r"CREATE TABLE IF NOT EXISTS take_events .*?CHECK \(league IN \(([^)]+)\)\)", MIGRATION, re.DOTALL)
    assert m and re.findall(r"'(\w+)'", m.group(1)) == list(LEAGUES)


@pytest.mark.parametrize("table,cols", [
    ("take_events", store.EVENT_COLUMNS), ("take_prices", store.PRICE_COLUMNS), ("take_picks", store.PICK_COLUMNS),
    ("take_pick_state", store.STATE_COLUMNS), ("take_closes", store.CLOSE_COLUMNS),
    ("take_results", store.RESULT_COLUMNS), ("take_api_calls", store.CALL_COLUMNS),
    ("take_verdict", store.VERDICT_COLUMNS),
])
def test_every_written_column_exists_in_the_migration(table, cols):
    have = set(re.findall(r"^\s{2}(\w+)\s", _create_table(table), re.MULTILINE))
    assert set(cols) <= have, set(cols) - have


def test_every_selected_view_column_exists_in_the_view():
    assert set(store.TEST_COLUMNS) <= {c.split(".")[-1] for c in re.findall(r"[\w.]+(?=,|\s+FROM)", _view("take_test_v"))}


def test_keys_and_references():
    assert "UNIQUE (event_id, market, side)" in _create_table("take_picks")
    assert "id           bigserial PRIMARY KEY" in _create_table("take_picks")
    assert "PRIMARY KEY (pulled_at, event_id, market, side, book)" in _create_table("take_prices")
    assert "PRIMARY KEY (event_id, market, side)" in _create_table("take_pick_state")
    assert "PRIMARY KEY (event_id, market)" in _create_table("take_closes")
    assert "pick_id       bigint PRIMARY KEY REFERENCES take_picks (id)" in _create_table("take_results")
    assert "CHECK (id = 1)" in _create_table("take_verdict")
    assert "CREATE INDEX IF NOT EXISTS take_prices_pulled_idx ON take_prices (pulled_at)" in MIGRATION


def test_check_constraints_pin_the_vocabularies():
    assert "CHECK (graded_by IN ('clv', 'direction', 'no_close'))" in MIGRATION
    assert "CHECK (outcome IN ('win', 'loss', 'push'))" in MIGRATION
    assert "CHECK (job IN ('pull', 'close', 'scores'))" in MIGRATION
    assert "CHECK (market IN ('ml', 'spread', 'total'))" in MIGRATION


def test_ref_line_is_stored_on_picks_and_state_and_exposed_in_the_list_view():
    assert re.search(r"^\s{2}ref_line\s+double precision", _create_table("take_pick_state"), re.MULTILINE)
    assert re.search(r"^\s{2}ref_line\s+double precision", _create_table("take_picks"), re.MULTILINE)
    assert "s.ref_line" in _view("take_list_v")


def test_take_picks_and_verdict_append_only():
    assert "BEFORE UPDATE OR DELETE ON take_picks" in MIGRATION and "BEFORE UPDATE OR DELETE ON take_verdict" in MIGRATION
    assert "BEFORE TRUNCATE ON take_picks" in MIGRATION and "BEFORE TRUNCATE ON take_verdict" in MIGRATION
    assert "REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON take_picks" in MIGRATION
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON take_picks, take_verdict FROM PUBLIC;" in MIGRATION
    assert "rolname = 'service_role'" in MIGRATION
    assert re.search(r"CREATE OR REPLACE FUNCTION take_append_only\(\) RETURNS trigger\s+"
                     r"LANGUAGE plpgsql SET search_path = ''", MIGRATION)
    assert "RAISE EXCEPTION" in MIGRATION


def test_rls_select_policy_on_all_eight_tables_and_write_revokes():
    for t in store.TABLES:
        assert f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;" in MIGRATION
        assert f'CREATE POLICY "public read {t}" ON {t} FOR SELECT USING (true);' in MIGRATION
    assert MIGRATION.count("FOR SELECT USING (true)") == 8
    m = re.search(r"REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ([^;]+) FROM anon, authenticated;", MIGRATION)
    g = re.search(r"GRANT SELECT ON ([^;]+) TO anon, authenticated;", MIGRATION)
    assert m and g
    for t in store.TABLES:
        assert t in re.split(r",\s*", m.group(1)) and t in re.split(r",\s*", g.group(1))
    assert "GRANT SELECT ON take_list_v, take_test_v TO anon, authenticated;" in MIGRATION
    assert MIGRATION.index("REVOKE INSERT") < MIGRATION.index("GRANT SELECT ON take_picks")


def test_views_are_security_invoker_and_expose_the_columns_the_site_needs():
    for v in ("take_list_v", "take_test_v"):
        assert f"CREATE OR REPLACE VIEW {v} WITH (security_invoker = true) AS" in MIGRATION
    lst = _view("take_list_v")
    for c in ("league", "book", "line", "price", "ev", "fair_prob", "good_until", "better_line", "ref_line", "others",
              "first_seen", "last_seen", "active", "dropped_at", "event_id", "market", "side"):
        assert f"s.{c}" in lst, c
    for c in ("home_team", "away_team", "commence_time", "game_pk"):
        assert f"e.{c}" in lst, c
    test = _view("take_test_v")
    for c in ("id", "event_id", "league", "market", "side", "book", "line", "price", "ev", "better_line",
              "first_seen", "graded_by", "clv", "beat", "graded_at", "outcome", "units"):
        assert re.search(rf"\b[pr]\.{c}\b", test), c
    assert "e.commence_time" in test and "LEFT JOIN take_results r ON r.pick_id = p.id" in test


def test_take_list_v_keeps_active_or_recently_dropped_for_24_hours():
    assert ("WHERE s.active OR coalesce(s.dropped_at, e.commence_time) > now() - interval '24 hours'"
            in _view("take_list_v"))


def test_header_documents_go_live_and_rollback():
    head = MIGRATION.split("CREATE TABLE")[0]
    assert "TAKE_THESE=on" in head and "Rollback" in head and "DROP VIEW take_list_v, take_test_v" in head


# ---- pure builders ----------------------------------------------------------------------------

def test_pick_tuple_order_matches_columns():
    t = store.pick_tuple(SAMPLE_PICK, "run1")
    assert len(t) == len(store.PICK_COLUMNS)
    assert t[store.PICK_COLUMNS.index("pin_prices")] == json.dumps(SAMPLE_PICK["pin_prices"])
    assert t[store.PICK_COLUMNS.index("first_seen")] == NOW and t[store.PICK_COLUMNS.index("run_id")] == "run1"
    assert t[store.PICK_COLUMNS.index("ref_line")] == -3.0 and t[store.PICK_COLUMNS.index("event_id")] == "ev1"
    assert "others" not in store.PICK_COLUMNS and "pulled_at" not in store.PICK_COLUMNS


def test_closes_upsert_keeps_latest():
    assert "WHERE EXCLUDED.captured_at > take_closes.captured_at" in store.UPSERT_CLOSES_SQL
    assert "ON CONFLICT (event_id, market) DO UPDATE SET" in store.UPSERT_CLOSES_SQL


def test_sql_constants_have_the_expected_guards():
    assert "ON CONFLICT (event_id, market, side) DO NOTHING" in store.INSERT_PICKS_SQL
    assert "UPDATE" not in store.INSERT_PICKS_SQL.replace("DO NOTHING", "")
    assert "COALESCE(EXCLUDED.game_pk, take_events.game_pk)" in store.UPSERT_EVENTS_SQL
    assert "ON CONFLICT (pick_id) DO NOTHING" in store.INSERT_RESULTS_SQL
    assert "ON CONFLICT (id) DO NOTHING" in store.INSERT_VERDICT_SQL
    assert "ON CONFLICT (pulled_at, event_id, market, side, book) DO NOTHING" in store.INSERT_PRICES_SQL
    up = store.UPSERT_STATE_SQL
    assert "first_seen = EXCLUDED.first_seen" not in up and "last_seen = EXCLUDED.last_seen" in up
    assert "ref_line = EXCLUDED.ref_line" in up and "active = true" in up and "dropped_at = NULL" in up
    assert "active = false" in store.DROP_STATE_SQL and "dropped_at = %s" in store.DROP_STATE_SQL


# ---- fake connections -------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, log, fetch=None, rowcount=1):
        self.log, self.fetch, self.rowcount = log, fetch or [], rowcount

    def executemany(self, sql, rows):
        self.log.append(("many", sql, list(rows)))

    def execute(self, sql, params=None):
        self.log.append(("one", sql, params))

    def fetchall(self):
        return self.fetch

    def fetchone(self):
        return self.fetch[0] if self.fetch else None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, log, fetch=None, rowcount=1):
        self.log, self.fetch, self.rowcount = log, fetch, rowcount

    def cursor(self):
        return FakeCursor(self.log, self.fetch, self.rowcount)

    def commit(self):
        self.log.append(("commit",))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake(monkeypatch, fetch=None, rowcount=1):
    log, seen = [], {}
    monkeypatch.setattr(db, "get_postgres",
                        lambda connect_timeout=None: seen.update(t=connect_timeout) or FakeConn(log, fetch, rowcount))
    return log, seen


def _boom(*a, **k):
    raise AssertionError("get_postgres should not be called")


def _stmts(log):
    return [e for e in log if e[0] in ("one", "many") and "statement_timeout" not in e[1]]


def test_functions_default_to_the_ledger_timeouts(monkeypatch):
    log, seen = _fake(monkeypatch)
    store.log_call("pull", "nfl", "odds", 3, 480, ["bet365"], "run1")
    assert seen["t"] == db.LEDGER_CONNECT_TIMEOUT
    assert log[0] == ("one", f"SET LOCAL statement_timeout = {db.LEDGER_STATEMENT_TIMEOUT_MS}", None)
    log.clear()
    store.log_call("pull", "nfl", "odds", 3, 480, None, "run1", connect_timeout=2, statement_timeout_ms=500)
    assert seen["t"] == 2 and log[0][1] == "SET LOCAL statement_timeout = 500"


@pytest.mark.parametrize("fn,args", [
    ("upsert_events", ([],)), ("insert_prices", ([],)), ("insert_picks", ([], "r")), ("upsert_closes", ([],)),
    ("insert_results", ([],)), ("store_scores", ({},)), ("set_outcomes", ([],)),
])
def test_empty_input_never_connects(monkeypatch, fn, args):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(store, fn)(*args) == 0
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert store.sync_state("nfl", [], [], NOW) is None


def test_insert_picks_is_insert_only_and_ordered_by_column(monkeypatch):
    log, _ = _fake(monkeypatch)
    assert store.insert_picks([SAMPLE_PICK], "run1") == 1
    (_, sql, (tup,)), = _stmts(log)
    assert sql.startswith("INSERT INTO take_picks (") and "%s::jsonb" in sql
    assert tup == store.pick_tuple(SAMPLE_PICK, "run1") and ("commit",) in log


def test_sync_state_upserts_each_pick_then_drops_the_vanished(monkeypatch):
    log, _ = _fake(monkeypatch)
    store.sync_state("nfl", ["ev1", "ev2"], [SAMPLE_PICK], NOW)
    ups, drop = _stmts(log)
    assert ups[0] == "many" and ups[1] == store.UPSERT_STATE_SQL
    (row,) = ups[2]
    assert row[store.STATE_COLUMNS.index("others")] == json.dumps(SAMPLE_PICK["others"])
    assert row[store.STATE_COLUMNS.index("ref_line")] == -3.0
    assert row[store.STATE_COLUMNS.index("first_seen")] == NOW == row[store.STATE_COLUMNS.index("last_seen")]
    assert drop[0] == "one" and drop[1] == store.DROP_STATE_SQL
    assert drop[2] == (NOW, "nfl", ["ev1", "ev2"], ["ev1"], ["spread"], ["home"])
    assert log.index(("commit",)) > log.index(drop)


def test_sync_state_with_no_picks_still_drops_everything_for_the_pulled_events(monkeypatch):
    log, _ = _fake(monkeypatch)
    store.sync_state("nfl", ["ev1"], [], NOW)
    (drop,) = _stmts(log)
    assert drop[2] == (NOW, "nfl", ["ev1"], [], [], [])


def test_expire_started_ends_every_leagues_started_picks(monkeypatch):
    sql = store.EXPIRE_STATE_SQL
    assert "active = false" in sql and "dropped_at = e.commence_time" in sql
    assert "s.active" in sql and "e.commence_time <= %s" in sql and "league" not in sql
    log, _ = _fake(monkeypatch, rowcount=4)
    assert store.expire_started(NOW) == 4
    (_, ran, params), = _stmts(log)
    assert ran == store.EXPIRE_STATE_SQL and params == (NOW,) and ("commit",) in log


def test_close_targets_groups_markets_by_league(monkeypatch):
    log, _ = _fake(monkeypatch, fetch=[("nfl", "spread"), ("nfl", "total"), ("epl", "ml")])
    assert store.close_targets(NOW, 5, 20) == {"nfl": {"spread", "total"}, "epl": {"ml"}}
    (_, sql, params), = _stmts(log)
    assert "r.pick_id IS NULL" in sql and params == (NOW, 5, NOW, 20)


def test_upsert_closes_json_encodes_prices(monkeypatch):
    log, _ = _fake(monkeypatch)
    prices = {"home": {"line": -3.0, "price": -110}}
    row = {"event_id": "ev1", "market": "spread", "prices": prices, "captured_at": NOW, "minutes_to_start": 12.0}
    assert store.upsert_closes([row]) == 1
    (_, sql, (tup,)), = _stmts(log)
    assert sql == store.UPSERT_CLOSES_SQL and tup == ("ev1", "spread", json.dumps(prices), NOW, 12.0)


def test_ungraded_started_attaches_the_close_or_none(monkeypatch):
    prices = {"home": {"line": -3.0, "price": -110}}
    fetch = [(1, "ev1", "nfl", "spread", "home", -2.5, -105, -3.0, prices, NOW, 12.0),
             (2, "ev2", "nfl", "ml", "away", None, 150, None, None, None, None)]
    _fake(monkeypatch, fetch=fetch)
    a, b = store.ungraded_started(NOW)
    assert a["pick_id"] == 1 and a["ref_line"] == -3.0 and a["league"] == "nfl"
    assert a["close"] == {"event_id": "ev1", "market": "spread", "prices": prices, "captured_at": NOW,
                          "minutes_to_start": 12.0}
    assert b["close"] is None and b["pick_id"] == 2


def test_insert_results_is_do_nothing_on_pick_id(monkeypatch):
    log, _ = _fake(monkeypatch)
    row = {"pick_id": 1, "graded_by": "clv", "close_fair": 0.5, "clv": 0.02, "beat": True, "close_line": None,
           "close_minutes": 10.0, "graded_at": NOW}
    assert store.insert_results([row]) == 1
    (_, sql, (tup,)), = _stmts(log)
    assert sql == store.INSERT_RESULTS_SQL and tup == (1, "clv", 0.5, 0.02, True, None, 10.0, NOW)


def test_unsettled_leagues_and_unsettled(monkeypatch):
    log, _ = _fake(monkeypatch, fetch=[("nfl",), ("mlb",)])
    assert store.unsettled_leagues() == ["nfl", "mlb"]
    assert "r.outcome IS NULL" in log[-1][1] and "interval '3 hours'" in log[-1][1]
    _fake(monkeypatch, fetch=[(7, "ev1", "spread", "home", -2.5, -105, "nfl")])
    assert store.unsettled("nfl") == [{"pick_id": 7, "event_id": "ev1", "market": "spread", "side": "home",
                                       "line": -2.5, "price": -105, "league": "nfl"}]


def test_store_scores_and_set_outcomes(monkeypatch):
    log, _ = _fake(monkeypatch)
    assert store.store_scores({"ev1": (24, 17)}) == 1
    (_, sql, (tup,)), = _stmts(log)
    assert "completed = true" in sql and "updated_at = now()" in sql and tup == (24, 17, "ev1")
    log.clear()
    assert store.set_outcomes([{"pick_id": 7, "outcome": "win", "units": 0.9}]) == 1
    (_, sql, (tup,)), = _stmts(log)
    assert "settled_at = now()" in sql and tup == ("win", 0.9, 7)


def test_verdict_roundtrip(monkeypatch):
    log, _ = _fake(monkeypatch, rowcount=1)
    row = {"judged_at": NOW, "n": 300, "beat_rate": 0.56, "mean_clv": 0.01, "lower": 0.002, "passed": True,
           "criteria": {"n": True}, "pick_ids": [1, 2]}
    assert store.insert_verdict(row) is True
    (_, sql, params), = _stmts(log)
    assert sql == store.INSERT_VERDICT_SQL and params[0] == 1 and json.dumps({"n": True}) in params
    _fake(monkeypatch, rowcount=0)
    assert store.insert_verdict(row) is False
    _fake(monkeypatch, fetch=[])
    assert store.get_verdict() is None
    vals = (1, NOW, 300, 0.56, 0.01, 0.002, True, {"n": True}, [1, 2])
    _fake(monkeypatch, fetch=[vals])
    assert store.get_verdict() == dict(zip(store.VERDICT_COLUMNS, vals))


def test_prune_prices_uses_before_and_reports_deleted(monkeypatch):
    log, _ = _fake(monkeypatch, rowcount=42)
    assert store.prune_prices(NOW) == 42
    assert log[1] == ("one", "DELETE FROM take_prices WHERE pulled_at < %s", (NOW,))


def test_missing_tables_checks_to_regclass_in_order(monkeypatch):
    calls = []

    def fake(tables, connect_timeout=None, statement_timeout_ms=None):
        calls.append((tables, connect_timeout, statement_timeout_ms))
        return ["take_verdict"]

    monkeypatch.setattr(db, "_missing_public_tables", fake)
    assert store.missing_tables() == ["take_verdict"]
    assert calls == [(list(store.TABLES), db.LEDGER_CONNECT_TIMEOUT, db.LEDGER_STATEMENT_TIMEOUT_MS)]


def test_recent_calls_reads_the_log_since(monkeypatch):
    called = NOW - timedelta(days=1)
    log, _ = _fake(monkeypatch, fetch=[("pull", "nfl", 3, ["fanatics"], called), ("scores", None, 2, None, called)])
    assert store.recent_calls(NOW - timedelta(days=7)) == [
        {"job": "pull", "league": "nfl", "credits_used": 3, "books_missing": ["fanatics"], "called_at": called},
        {"job": "scores", "league": None, "credits_used": 2, "books_missing": None, "called_at": called}]
    (_, sql, params), = _stmts(log)
    assert "FROM take_api_calls" in sql and "called_at >= %s" in sql and params == (NOW - timedelta(days=7),)


def test_unsettled_window_matches_the_scores_days_from():
    window = "e.commence_time > now() - interval '3 days' AND e.commence_time < now() - interval '3 hours'"
    assert window in store.UNSETTLED_LEAGUES_SQL and window in store.UNSETTLED_SQL


def test_verdict_mean_clv_and_lower_are_nullable():
    table = _create_table("take_verdict")
    assert re.search(r"mean_clv\s+double precision,", table) and re.search(r"\blower\s+double precision,", table)
    assert re.search(r"beat_rate\s+double precision NOT NULL", table)
