"""upsert_nfl_betting_splits: column-order tuple building + SQL shape, no live
DB -- mirrors tests/test_db_nfl_sim.py's FakeConn pattern."""
from sportsmodel import db
from sportsmodel import db as db_module


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink
        self.rowcount = 0

    def executemany(self, sql, rows):
        self.sink["sql"] = sql
        self.sink["rows"] = list(rows)
        self.rowcount = len(self.sink["rows"])

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

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_helper_exists():
    assert hasattr(db, "upsert_nfl_betting_splits")


def test_empty_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_nfl_betting_splits([]) == 0


def test_tuple_and_sql(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    row = {"game_pk": 401872934, "market": "spread", "side": "home",
           "cash_pct": 62.0, "ticket_pct": 48.0,
           "commence_time": "2026-09-20T17:00:00+00:00",
           "captured_at": "2026-09-20T15:30:00+00:00"}
    assert db.upsert_nfl_betting_splits([row]) == 1
    (tup,) = sink["rows"]
    cols = db._NFL_BETTING_SPLITS_COLS
    assert tup == tuple(row.get(c) for c in cols)
    assert tup[cols.index("cash_pct")] == 62.0
    assert tup[cols.index("ticket_pct")] == 48.0
    assert "INSERT INTO nfl_betting_splits" in sink["sql"]
    assert "ON CONFLICT (game_pk, market, side, captured_at) DO UPDATE" in sink["sql"]
    # PK columns must not be reassigned in the update clause
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "captured_at = EXCLUDED.captured_at" not in sink["sql"]
    # non-key columns ARE updated
    assert "cash_pct = EXCLUDED.cash_pct" in sink["sql"]


def test_commit_called(monkeypatch):
    sink = {}
    holder = {}

    def fake():
        c = FakeConn(sink)
        holder["c"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake)
    db.upsert_nfl_betting_splits([
        {"game_pk": 1, "market": "total", "side": "over", "cash_pct": 55.0,
         "ticket_pct": 51.0, "commence_time": None, "captured_at": "2026-09-20T15:30:00+00:00"},
    ])
    assert len(sink["rows"]) == 1
    assert holder["c"].committed is True
