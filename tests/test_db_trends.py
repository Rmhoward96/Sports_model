"""upsert_team_betting_records: column-order tuple building + SQL shape, no live
DB -- mirrors tests/test_db_nfl_player_actuals.py's FakeConn pattern."""
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


def test_upsert_team_betting_records_helper_exists():
    assert hasattr(db, "upsert_team_betting_records")


def test_upsert_team_betting_records_empty_returns_zero(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_team_betting_records([]) == 0


def test_upsert_team_betting_records_tuple_and_sql(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    row = {"sport": "nfl", "season": 2026, "team_name": "Buffalo Bills",
           "an_team_name": "Buffalo Bills", "abbr": "BUF",
           "records": {"ats": {"w": 2, "l": 1, "p": 1, "o": None, "u": None}}}
    assert db.upsert_team_betting_records([row]) == 1
    (tup,) = sink["rows"]
    cols = db._TEAM_RECORD_COLS
    assert tup[cols.index("sport")] == "nfl"
    assert tup[cols.index("season")] == 2026
    assert tup[cols.index("team_name")] == "Buffalo Bills"
    assert tup[cols.index("records")] == '{"ats": {"w": 2, "l": 1, "p": 1, "o": null, "u": null}}'
    assert "INSERT INTO team_betting_records" in sink["sql"]
    assert "ON CONFLICT (sport, season, team_name) DO UPDATE" in sink["sql"]
    assert "captured_at = now()" in sink["sql"]


def test_upsert_team_betting_records_commit_called(monkeypatch):
    sink = {}
    holder = {}

    def fake():
        c = FakeConn(sink)
        holder["c"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake)
    rows = [
        {"sport": "nfl", "season": 2026, "team_name": "Buffalo Bills",
         "an_team_name": "Buffalo Bills", "abbr": "BUF", "records": {}},
        {"sport": "cfb", "season": 2026, "team_name": "Alabama Crimson Tide",
         "an_team_name": "Alabama Crimson Tide", "abbr": None, "records": {}},
    ]
    db.upsert_team_betting_records(rows)
    assert len(sink["rows"]) == 2
    assert holder["c"].committed is True
