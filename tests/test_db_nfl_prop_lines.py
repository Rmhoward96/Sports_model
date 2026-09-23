"""upsert_nfl_prop_lines: column-order tuple building + SQL shape, no live
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


def test_upsert_nfl_prop_lines_shape(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    rec = {c: None for c in db._NFL_PROP_LINES_COLS}
    rec.update(game_pk=1, player_id="00-1", market="rec_yds", line=54.5)
    assert db.upsert_nfl_prop_lines([rec]) == 1
    assert "ON CONFLICT (game_pk, player_id, market) DO UPDATE" in sink["sql"]
    assert "updated_at = now()" in sink["sql"]
    assert sink["rows"][0][db._NFL_PROP_LINES_COLS.index("line")] == 54.5
    assert db.upsert_nfl_prop_lines([]) == 0
