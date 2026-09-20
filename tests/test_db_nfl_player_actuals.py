"""upsert_nfl_player_actuals: column-order tuple building + SQL shape, no live
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


def test_upsert_nfl_player_actuals_helper_exists():
    assert hasattr(db, "upsert_nfl_player_actuals")


def test_upsert_nfl_player_actuals_empty_returns_zero(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_nfl_player_actuals([]) == 0


def test_upsert_nfl_player_actuals_tuple_and_sql(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    row = {"game_pk": 401872934, "player_id": "00-0030431",
           "player_name": "Robert Woods", "market": "rec_yds", "actual": 37.0,
           "season": 2026, "week": 3}
    assert db.upsert_nfl_player_actuals([row]) == 1
    (tup,) = sink["rows"]
    cols = db._NFL_PLAYER_ACTUALS_COLS
    assert tup[cols.index("actual")] == 37.0
    assert tup[cols.index("market")] == "rec_yds"
    assert tup[cols.index("game_pk")] == 401872934
    assert "INSERT INTO nfl_player_actuals" in sink["sql"]
    assert "ON CONFLICT (game_pk, player_id, market) DO UPDATE" in sink["sql"]
    # captured_at is DEFAULT/now()-managed, never a bound column value
    assert tup == tuple(row.get(c) for c in cols)
    assert "captured_at = now()" in sink["sql"]


def test_upsert_nfl_player_actuals_commit_called(monkeypatch):
    sink = {}
    holder = {}

    def fake():
        c = FakeConn(sink)
        holder["c"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake)
    db.upsert_nfl_player_actuals([
        {"game_pk": 1, "player_id": "a", "market": "rush_yds", "actual": 12.0},
        {"game_pk": 1, "player_id": "b", "market": "anytime_td", "actual": 1.0},
    ])
    assert len(sink["rows"]) == 2
    assert holder["c"].committed is True
