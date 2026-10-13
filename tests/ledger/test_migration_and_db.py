"""db/migration_pick_ledger.sql and the ledger helpers in db.py: SQL shape, tuple building, column agreement,
the append-only guarantees and the missing-table guard. No live DB (fake connections)."""
import re
from pathlib import Path

import pytest

from sportsmodel import db

MIGRATION = (Path(__file__).resolve().parents[2] / "db" / "migration_pick_ledger.sql").read_text()
TABLES = ["pick_ledger", "close_capture", "pick_ledger_results"]


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


# ---- migration --------------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_every_written_column_exists_in_the_migration_and_keys_are_declared(table):
    body = _create_table(table)
    cols = re.findall(r"^\s{2}(\w+)\s", body, re.MULTILINE)
    assert set(db.LEDGER_COLUMNS[table]) <= set(cols), set(db.LEDGER_COLUMNS[table]) - set(cols)
    key = ", ".join(db.LEDGER_KEYS[table])
    if table == "pick_ledger":
        assert "id               bigserial PRIMARY KEY" in body
        assert f"CREATE UNIQUE INDEX IF NOT EXISTS pick_ledger_key_observed_idx ON pick_ledger ({key});" in MIGRATION
    else:
        assert f"PRIMARY KEY ({key})" in body


def test_pick_ledger_is_append_only_by_trigger_for_update_delete_and_truncate():
    assert "CREATE OR REPLACE FUNCTION pick_ledger_append_only() RETURNS trigger" in MIGRATION
    assert "RAISE EXCEPTION" in MIGRATION
    assert re.search(r"CREATE TRIGGER pick_ledger_no_update_delete\s+BEFORE UPDATE OR DELETE ON pick_ledger\s+"
                     r"FOR EACH ROW EXECUTE FUNCTION pick_ledger_append_only\(\);", MIGRATION)
    assert re.search(r"CREATE TRIGGER pick_ledger_no_truncate\s+BEFORE TRUNCATE ON pick_ledger\s+"
                     r"FOR EACH STATEMENT EXECUTE FUNCTION pick_ledger_append_only\(\);", MIGRATION)
    assert MIGRATION.count("DROP TRIGGER IF EXISTS") == 2          # re-runnable


def test_rls_select_policy_and_write_revokes_for_every_table():
    for t in TABLES:
        assert f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;" in MIGRATION
        assert f'CREATE POLICY "public read {t}" ON {t} FOR SELECT USING (true);' in MIGRATION
    m = re.search(r"REVOKE ([A-Z, ]+) ON ([^;]+) FROM anon, authenticated;", MIGRATION)
    assert m and set(re.split(r",\s*", m.group(1))) == {"INSERT", "UPDATE", "DELETE", "TRUNCATE"}
    for t in TABLES:
        assert f"public.{t}" in m.group(2)
    assert re.search(r"GRANT SELECT ON pick_ledger, close_capture, pick_ledger_results TO anon, authenticated;", MIGRATION)
    assert MIGRATION.index("REVOKE INSERT") < MIGRATION.index("GRANT SELECT ON")


def test_ledger_revokes_update_delete_truncate_from_public_and_service_role():
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM PUBLIC;" in MIGRATION
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM service_role" in MIGRATION
    assert "rolname = 'service_role'" in MIGRATION                  # guarded: the role may not exist locally


def test_migration_is_idempotent_additive_and_writes_no_data():
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == 3
    assert MIGRATION.count("DROP POLICY IF EXISTS") == 3
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE|ALTER TABLE \w+ DROP)\b", MIGRATION,
                         re.MULTILINE | re.IGNORECASE)
    assert "CREATE INDEX IF NOT EXISTS" in MIGRATION and "CREATE INDEX pick" not in MIGRATION


def test_check_constraints_pin_the_vocabularies():
    assert "CHECK (source IN ('model_game', 'ev_game', 'ev_prop', 'parlay_leg'))" in MIGRATION
    assert "ref_kind IN ('pinnacle_novig', 'us_consensus_novig')" in MIGRATION
    assert "CHECK (kind IN ('lines', 'props'))" in MIGRATION
    assert "CHECK (status IN ('ok', 'empty', 'failed'))" in MIGRATION
    assert "CHECK (basis IN ('first', 'last'))" in MIGRATION
    assert "CHECK (result IN ('W', 'L', 'P'))" in MIGRATION
    assert "close_kind IN ('pinnacle_novig', 'us_consensus_novig', 'missing')" in MIGRATION


# ---- db helpers ---------------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, sink, fetch=None):
        self.sink, self.fetch = sink, fetch or []

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def execute(self, sql, params=None):
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


def _boom():
    raise AssertionError("get_postgres should not be called")


@pytest.mark.parametrize("fn", ["insert_pick_ledger", "upsert_close_capture", "upsert_pick_ledger_results"])
def test_empty_rows_never_connect(monkeypatch, fn):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(db, fn)([]) == 0


def test_latest_rows_with_no_keys_never_connects(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert db.latest_pick_ledger_rows([]) == {}


def test_insert_pick_ledger_is_insert_only_idempotent_and_ordered_by_column(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.LEDGER_COLUMNS["pick_ledger"]} | {"pick_key": "k", "sport": "nfl", "game_pk": 7}
    assert db.insert_pick_ledger([row]) == 1
    sql, (tup,) = sink["sql"], sink["rows"]
    assert sql.startswith("INSERT INTO pick_ledger (") and "ON CONFLICT (pick_key, observed_at) DO NOTHING" in sql
    assert "UPDATE" not in sql.replace("DO NOTHING", "") and "DELETE" not in sql
    cols = db.LEDGER_COLUMNS["pick_ledger"]
    assert len(tup) == len(cols) and tup[cols.index("pick_key")] == "k" and tup[cols.index("game_pk")] == 7
    assert sink["committed"] is True


def test_latest_pick_ledger_rows_maps_by_key(monkeypatch):
    sink = {}
    fetched = [("k1", -110, 3.5, 0.55, 0.52, "2026-10-13T00:00:00Z")]
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink, fetched))
    out = db.latest_pick_ledger_rows(["k1", "k2"])
    assert out == {"k1": dict(zip(db.LEDGER_LAST_COLS, fetched[0]))}
    assert "DISTINCT ON (pick_key)" in sink["sql"] and "ORDER BY pick_key, observed_at DESC, id DESC" in sink["sql"]
    assert sink["params"] == (["k1", "k2"],)


def test_close_capture_upsert_only_supersedes_a_failed_row(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.LEDGER_COLUMNS["close_capture"]} | {"sport": "nfl", "game_pk": 1, "kind": "lines"}
    assert db.upsert_close_capture([row]) == 1
    sql = sink["sql"]
    assert "ON CONFLICT (sport, game_pk, kind) DO UPDATE SET" in sql
    assert sql.rstrip().endswith("WHERE close_capture.status = 'failed'")
    assert "status = EXCLUDED.status" in sql and "sport = EXCLUDED.sport" not in sql


def test_results_upsert_keys_on_pick_key_and_basis(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.LEDGER_COLUMNS["pick_ledger_results"]} | {"pick_key": "k", "basis": "first"}
    assert db.upsert_pick_ledger_results([row]) == 1
    sql = sink["sql"]
    assert "ON CONFLICT (pick_key, basis) DO UPDATE SET" in sql and "graded_at = now()" in sql
    assert "result = EXCLUDED.result" in sql and "pick_key = EXCLUDED.pick_key" not in sql


class _RegCursor(FakeCursor):
    def __init__(self, existing, log):
        self.existing, self.log, self.last = existing, log, None

    def execute(self, sql, params):
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


def test_missing_ledger_tables_checks_to_regclass_in_public_in_order(monkeypatch):
    log = []
    monkeypatch.setattr(db, "get_postgres", lambda: _RegConn({"pick_ledger"}, log))
    assert db.missing_ledger_tables() == ["close_capture", "pick_ledger_results"]
    assert [p[0] for _, p in log] == [f"public.{t}" for t in TABLES]
    assert db.missing_ledger_tables(["pick_ledger"]) == []


def test_missing_ledger_tables_rejects_unknown_names_and_propagates_connection_errors(monkeypatch):
    with pytest.raises(KeyError):
        db.missing_ledger_tables(["users; drop table x"])

    def down():
        raise ConnectionError("db down")

    monkeypatch.setattr(db, "get_postgres", down)
    with pytest.raises(ConnectionError):
        db.missing_ledger_tables(["pick_ledger"])


def test_site_panel_guard_still_works_after_the_shared_helper_refactor(monkeypatch):
    log = []
    monkeypatch.setattr(db, "get_postgres", lambda: _RegConn({"game_info"}, log))
    assert db.missing_site_panel_tables(["game_info", "cfb_team_insights"]) == ["cfb_team_insights"]
    with pytest.raises(KeyError):
        db.missing_site_panel_tables(["pick_ledger"])
