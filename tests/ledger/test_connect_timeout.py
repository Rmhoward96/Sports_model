"""The ledger's database connections carry a short connect timeout so a dead database can never hang a pick
writer, without changing get_postgres for any other caller. No test here connects to anything."""
import sys
import types

import pytest

from sportsmodel import config, db
from sportsmodel.ledger import record


class _Cur:
    def __init__(self):
        self.rowcount = 0

    def execute(self, *a, **k):
        pass

    executemany = execute

    def fetchone(self):
        return ("public.x",)

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def cursor(self):
        return _Cur()

    def commit(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_psycopg(monkeypatch):
    calls = []
    mod = types.SimpleNamespace(connect=lambda *a, **k: calls.append((a, k)) or _Conn())
    monkeypatch.setitem(sys.modules, "psycopg", mod)
    monkeypatch.setattr(config, "DATABASE_URL", "postgresql://example.invalid/db")
    return calls


def test_get_postgres_default_call_is_unchanged(monkeypatch):
    calls = _fake_psycopg(monkeypatch)
    db.get_postgres()
    assert calls == [(("postgresql://example.invalid/db",), {})]


def test_get_postgres_passes_an_optional_connect_timeout(monkeypatch):
    calls = _fake_psycopg(monkeypatch)
    db.get_postgres(connect_timeout=7)
    assert calls == [(("postgresql://example.invalid/db",), {"connect_timeout": 7})]


def test_the_ledger_timeout_is_short():
    assert 1 <= db.LEDGER_CONNECT_TIMEOUT <= 15


def test_ledger_db_helpers_connect_with_the_given_timeout(monkeypatch):
    seen = []
    monkeypatch.setattr(db, "get_postgres", lambda **kw: seen.append(kw) or _Conn())
    db.missing_ledger_tables(["pick_ledger"], connect_timeout=9)
    db.latest_pick_ledger_rows(["k"], connect_timeout=9)
    db.insert_pick_ledger([{"pick_key": "k"}], connect_timeout=9)
    assert seen == [{"connect_timeout": 9}] * 3


def test_ledger_db_helpers_without_a_timeout_keep_the_plain_call(monkeypatch):
    seen = []
    monkeypatch.setattr(db, "get_postgres", lambda *a, **kw: seen.append((a, kw)) or _Conn())
    db.missing_ledger_tables(["pick_ledger"])
    db.latest_pick_ledger_rows(["k"])
    db.insert_pick_ledger([{"pick_key": "k"}])
    assert seen == [((), {})] * 3


def test_db_store_uses_the_ledger_timeout_for_every_call(monkeypatch):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    seen = []
    monkeypatch.setattr(db, "missing_ledger_tables", lambda t, **kw: seen.append(("missing", kw)) or [])
    monkeypatch.setattr(db, "latest_pick_ledger_rows", lambda k, **kw: seen.append(("latest", kw)) or {})
    monkeypatch.setattr(db, "insert_pick_ledger", lambda r, **kw: seen.append(("insert", kw)) or 0)
    store = record.DbStore()
    store.missing_tables(), store.latest(["k"]), store.insert([{}])
    assert seen == [(n, {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT}) for n in ("missing", "latest", "insert")]


def test_other_ledger_helpers_are_not_given_a_timeout_they_do_not_take():
    import inspect
    for fn in (db.upsert_close_capture, db.upsert_pick_ledger_results):
        assert "connect_timeout" not in inspect.signature(fn).parameters
