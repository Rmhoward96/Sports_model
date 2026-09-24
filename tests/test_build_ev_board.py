"""Light pure test for scripts/build_ev_board.py.

Loads the script module directly (it isn't a package) via importlib. No
network, no DB -- `get_postgres` is monkeypatched with a fake connection whose
cursor records the executed SQL.
"""
import importlib.util
import re
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_ev_board.py"
_spec = importlib.util.spec_from_file_location("build_ev_board", _SCRIPT_PATH)
build_ev_board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_ev_board)

WINDOW_48H = re.compile(r"captured_at\s*>\s*now\(\)\s*-\s*interval\s*'48 hours'")


class _FakeCursor:
    description = []

    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.append((sql, params))

    def fetchall(self):
        return []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return _FakeCursor(self.sink)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_load_latest_odds_empty_game_pks_short_circuits_no_db():
    assert build_ev_board.load_latest_odds([]) == []


def test_load_latest_odds_ignores_book_prices_older_than_48h(monkeypatch):
    # A book that stopped updating must not keep its last (stale) capture as
    # the "current" price -- same 48h window as the site's price views.
    sink = []
    monkeypatch.setattr(build_ev_board, "get_postgres", lambda: _FakeConn(sink))
    assert build_ev_board.load_latest_odds([1, 2]) == []
    assert len(sink) == 1
    sql, params = sink[0]
    assert WINDOW_48H.search(sql)
    assert "captured_at <= commence_time" in sql
    assert params == [[1, 2]]
