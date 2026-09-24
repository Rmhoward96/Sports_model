"""load_injury_snapshots / upsert_injury_snapshots: SQL shape + tuple building,
no live DB (FakeConn pattern from tests/test_db_nfl_player_actuals.py)."""
import json

from sportsmodel import db
from sportsmodel import db as db_module


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def executemany(self, sql, rows):
        self.sink["sql"] = sql
        self.sink["rows"] = list(rows)

    def execute(self, sql, params=None):
        self.sink["sql"] = sql
        self.sink["params"] = params

    def fetchall(self):
        return self.sink.get("fetch", [])

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink):
        self.sink = sink
        self.committed = False

    def cursor(self):
        return FakeCursor(self.sink)

    def commit(self):
        self.committed = True
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _boom():
    raise AssertionError("get_postgres should not be called")


def test_upsert_empty_returns_zero(monkeypatch):
    monkeypatch.setattr(db_module, "get_postgres", _boom)
    assert db.upsert_injury_snapshots([]) == 0


def test_upsert_sql_and_tuples(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    statuses = [{"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Out"}]
    n = db.upsert_injury_snapshots([
        {"sport": "nfl", "game_pk": 401, "fingerprint": "abc", "statuses": statuses},
    ])
    assert n == 1
    (tup,) = sink["rows"]
    assert tup[:3] == ("nfl", 401, "abc")
    assert json.loads(tup[3]) == statuses
    sql = sink["sql"]
    assert "INSERT INTO injury_snapshots" in sql
    assert "ON CONFLICT (sport, game_pk) DO UPDATE" in sql
    assert "captured_at = now()" in sql
    assert "fingerprint = EXCLUDED.fingerprint" in sql
    assert "statuses = EXCLUDED.statuses" in sql
    assert sink["committed"] is True


def test_load_empty_pks_returns_empty(monkeypatch):
    monkeypatch.setattr(db_module, "get_postgres", _boom)
    assert db.load_injury_snapshots("nfl", []) == {}


def test_load_maps_rows_by_game_pk(monkeypatch):
    statuses = [{"team": "T", "player": "P", "status": "Out"}]
    sink = {"fetch": [(401, "fp1", statuses, "ts1"), (402, "fp2", json.dumps(statuses), "ts2")]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    got = db.load_injury_snapshots("nfl", [401, 402, 403])
    assert set(got) == {401, 402}
    assert got[401] == {"fingerprint": "fp1", "statuses": statuses, "captured_at": "ts1"}
    assert got[402]["statuses"] == statuses  # JSON text decoded
    assert "FROM injury_snapshots" in sink["sql"]
    assert "game_pk = ANY(%s)" in sink["sql"]
    assert sink["params"] == ("nfl", [401, 402, 403])
