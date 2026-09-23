"""refresh_prediction_pnl: issues the materialized-view refresh and commits,
no live DB -- FakeConn pattern as tests/test_db_nfl_player_actuals.py."""
from sportsmodel import db as db_module


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.setdefault("sql", []).append(sql)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return FakeCursor(self.sink)

    def commit(self):
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_refresh_prediction_pnl_refreshes_matview_and_commits(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    db_module.refresh_prediction_pnl()
    assert sink["sql"] == ["REFRESH MATERIALIZED VIEW prediction_pnl_daily"]
    assert sink["committed"] is True
