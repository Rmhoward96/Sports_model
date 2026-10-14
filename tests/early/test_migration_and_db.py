"""db/migration_early_tickets.sql and the early-monitor helpers in db.py: SQL shape, tuple building, column
agreement, the append-only guarantees and the missing-table guard. No live DB (fake connections)."""
import re
from pathlib import Path

import pytest

from sportsmodel import db

MIGRATION = (Path(__file__).resolve().parents[2] / "db" / "migration_early_tickets.sql").read_text()
TABLES = ["early_tickets", "early_ticket_results"]


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


# ---- migration --------------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_every_written_column_exists_in_the_migration(table):
    cols = re.findall(r"^\s{2}(\w+)\s", _create_table(table), re.MULTILINE)
    assert set(db.EARLY_COLUMNS[table]) <= set(cols), set(db.EARLY_COLUMNS[table]) - set(cols)


def test_tickets_have_a_bigserial_id_and_the_unique_ticket_key():
    body = _create_table("early_tickets")
    assert "id              bigserial PRIMARY KEY" in body
    assert "UNIQUE (sport, game_pk, market, side)" in body
    assert db.EARLY_KEYS["early_tickets"] == ("sport", "game_pk", "market", "side")


def test_results_are_keyed_by_ticket_id():
    body = _create_table("early_ticket_results")
    assert "ticket_id              bigint PRIMARY KEY REFERENCES early_tickets (id)" in body
    assert db.EARLY_KEYS["early_ticket_results"] == ("ticket_id",)


def test_tickets_are_append_only_by_trigger_for_update_delete_and_truncate():
    assert "CREATE OR REPLACE FUNCTION early_tickets_append_only() RETURNS trigger" in MIGRATION
    assert "RAISE EXCEPTION" in MIGRATION
    assert re.search(r"CREATE TRIGGER early_tickets_no_update_delete\s+BEFORE UPDATE OR DELETE ON early_tickets\s+"
                     r"FOR EACH ROW EXECUTE FUNCTION early_tickets_append_only\(\);", MIGRATION)
    assert re.search(r"CREATE TRIGGER early_tickets_no_truncate\s+BEFORE TRUNCATE ON early_tickets\s+"
                     r"FOR EACH STATEMENT EXECUTE FUNCTION early_tickets_append_only\(\);", MIGRATION)
    assert MIGRATION.count("DROP TRIGGER IF EXISTS") == 2          # re-runnable
    assert re.search(r"CREATE OR REPLACE FUNCTION early_tickets_append_only\(\) RETURNS trigger\s+"
                     r"LANGUAGE plpgsql SET search_path = ''", MIGRATION)


def test_rls_select_policy_and_write_revokes_for_both_tables():
    for t in TABLES:
        assert f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;" in MIGRATION
        assert f'CREATE POLICY "public read {t}" ON {t} FOR SELECT USING (true);' in MIGRATION
    m = re.search(r"REVOKE ([A-Z, ]+) ON ([^;]+) FROM anon, authenticated;", MIGRATION)
    assert m and set(re.split(r",\s*", m.group(1))) == {"INSERT", "UPDATE", "DELETE", "TRUNCATE"}
    for t in TABLES:
        assert f"public.{t}" in m.group(2)
    assert "GRANT SELECT ON early_tickets, early_ticket_results TO anon, authenticated;" in MIGRATION
    assert MIGRATION.index("REVOKE INSERT") < MIGRATION.index("GRANT SELECT ON")


def test_tickets_revoke_update_delete_truncate_from_public_and_service_role():
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM PUBLIC;" in MIGRATION
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM service_role" in MIGRATION
    assert "rolname = 'service_role'" in MIGRATION                  # guarded: the role may not exist locally


def test_migration_is_idempotent_additive_and_writes_no_data():
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == 2
    assert MIGRATION.count("DROP POLICY IF EXISTS") == 2
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE|ALTER TABLE \w+ DROP)\b", MIGRATION,
                         re.MULTILINE | re.IGNORECASE)
    assert "CREATE INDEX IF NOT EXISTS" in MIGRATION and "CREATE INDEX early" not in MIGRATION


def test_check_constraints_pin_the_vocabularies():
    assert "CHECK (sport IN ('cfb', 'nfl'))" in MIGRATION
    assert "CHECK (market IN ('spread', 'total'))" in MIGRATION
    assert "CHECK (side IN ('home', 'away', 'over', 'under'))" in MIGRATION
    assert "CHECK (result IN ('W', 'L', 'P'))" in MIGRATION
    assert "CHECK (close_kind IN ('pinnacle_novig', 'missing'))" in MIGRATION


# ---- db helpers ---------------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, sink, fetch=None):
        self.sink, self.fetch = sink, fetch or []

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def execute(self, sql, params=None):
        self.sink.setdefault("executed", []).append(sql)
        self.sink["sql"], self.sink["params"] = sql, params

    def fetchall(self):
        return self.fetch

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink, fetch=None):
        self.sink, self.fetch = sink, fetch

    def cursor(self):
        return FakeCursor(self.sink, self.fetch)

    def commit(self):
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _boom(*a, **k):
    raise AssertionError("get_postgres should not be called")


@pytest.mark.parametrize("fn", ["insert_early_tickets", "upsert_early_ticket_results"])
def test_empty_rows_never_connect(monkeypatch, fn):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(db, fn)([]) == 0


def test_insert_early_tickets_is_insert_only_idempotent_and_ordered_by_column(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.EARLY_COLUMNS["early_tickets"]} | {"sport": "cfb", "game_pk": 7, "side": "over"}
    assert db.insert_early_tickets([row]) == 1
    sql, (tup,) = sink["sql"], sink["rows"]
    assert sql.startswith("INSERT INTO early_tickets (")
    assert "ON CONFLICT (sport, game_pk, market, side) DO NOTHING" in sql
    assert "UPDATE" not in sql.replace("DO NOTHING", "") and "DELETE" not in sql
    cols = db.EARLY_COLUMNS["early_tickets"]
    assert len(tup) == len(cols) and tup[cols.index("sport")] == "cfb" and tup[cols.index("game_pk")] == 7
    assert sink["committed"] is True


def test_insert_early_tickets_passes_the_timeouts_through(monkeypatch):
    sink, seen = {}, {}
    monkeypatch.setattr(db, "get_postgres", lambda connect_timeout=None: seen.update(t=connect_timeout) or FakeConn(sink))
    row = {c: None for c in db.EARLY_COLUMNS["early_tickets"]}
    db.insert_early_tickets([row], connect_timeout=10, statement_timeout_ms=15_000)
    assert seen["t"] == 10 and sink["executed"][0] == "SET LOCAL statement_timeout = 15000"


def test_results_upsert_keys_on_ticket_id_and_stamps_graded_at(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.EARLY_COLUMNS["early_ticket_results"]} | {"ticket_id": 5, "result": "W"}
    assert db.upsert_early_ticket_results([row]) == 1
    sql = sink["sql"]
    assert "ON CONFLICT (ticket_id) DO UPDATE SET" in sql and "graded_at = now()" in sql
    assert "result = EXCLUDED.result" in sql and "ticket_id = EXCLUDED.ticket_id" not in sql


class _RegCursor(FakeCursor):
    def __init__(self, existing, log):
        self.existing, self.log, self.last = existing, log, None

    def execute(self, sql, params=None):
        if params is None:                                  # SET LOCAL statement_timeout
            return
        self.log.append((sql, params))
        name = params[0].split(".", 1)[1]
        self.last = (params[0],) if name in self.existing else (None,)

    def fetchone(self):
        return self.last


class _RegConn(FakeConn):
    def __init__(self, existing, log):
        self.existing, self.log = existing, log

    def cursor(self):
        return _RegCursor(self.existing, self.log)


def test_missing_early_tables_checks_to_regclass_in_public_in_order(monkeypatch):
    log = []
    monkeypatch.setattr(db, "get_postgres", lambda: _RegConn({"early_ticket_results"}, log))
    assert db.missing_early_tables() == ["early_tickets"]
    assert [p[0] for _, p in log] == [f"public.{t}" for t in TABLES]
    assert db.missing_early_tables(["early_ticket_results"]) == []


def test_missing_early_tables_rejects_unknown_names_and_propagates_connection_errors(monkeypatch):
    with pytest.raises(KeyError):
        db.missing_early_tables(["pick_ledger"])

    def down(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(db, "get_postgres", down)
    with pytest.raises(ConnectionError):
        db.missing_early_tables(["early_tickets"])


def test_the_ledger_guard_does_not_know_the_early_tables():
    with pytest.raises(KeyError):
        db.missing_ledger_tables(["early_tickets"])


def test_early_ticket_keys_returns_the_logged_sides_of_upcoming_games(monkeypatch):
    sink = {}
    fetched = [("cfb", 7, "total", "over"), ("cfb", 8, "spread", "home")]
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink, fetched))
    assert db.early_ticket_keys("cfb", "2026-10-13T00:00:00Z") == set(fetched)
    assert "FROM early_tickets WHERE sport = %s AND commence_time > %s" in sink["sql"]
    assert sink["params"] == ("cfb", "2026-10-13T00:00:00Z")
