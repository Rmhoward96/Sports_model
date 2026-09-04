"""upsert_prediction_accuracy: column-order tuple building, no live DB."""
import pytest

from sportsmodel import db as db_module
from sportsmodel import db


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def executemany(self, sql, rows):
        self.sink["sql"] = sql
        self.sink["rows"] = list(rows)

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


def test_upsert_prediction_accuracy_helper_exists():
    assert hasattr(db, "upsert_prediction_accuracy")


def test_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_prediction_accuracy([]) == 0


def test_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "mlb",
        "game_pk": 12345,
        "game_date": "2026-09-01",
        "home_team_name": "Yankees",
        "away_team_name": "Red Sox",
        "win_prob": 0.62,
        "predicted_winner": "Yankees",
        "actual_winner": "Yankees",
        "winner_correct": True,
        "pred_margin": 1.4,
        "actual_margin": 3.0,
        "margin_error": 1.6,
        "spread_covered": True,
        "pred_total": 8.7,
        "actual_total": 9.0,
        "total_error": 0.3,
        "total_over": False,
        "market_spread": -1.5,
        "market_total": 8.5,
        "spread_pick_correct": True,
        "total_pick_correct": False,
    }
    n = db.upsert_prediction_accuracy([row])

    assert n == 1
    assert sink["rows"] == [(
        "mlb", 12345, "2026-09-01", "Yankees", "Red Sox",
        0.62, "Yankees", "Yankees", True,
        1.4, 3.0, 1.6, True,
        8.7, 9.0, 0.3, False,
        -1.5, 8.5, True, False,
    )]
    # column list, table name, and conflict target all present in the emitted SQL
    assert "INSERT INTO prediction_accuracy" in sink["sql"]
    assert "ON CONFLICT (sport, game_pk) DO UPDATE" in sink["sql"]
    assert "graded_at = now()" in sink["sql"]
    # PK columns must not be reassigned in the DO UPDATE SET clause
    assert "sport = EXCLUDED.sport" not in sink["sql"]
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]


def test_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    # Only the PK + a couple of fields supplied -- everything else should
    # come through as None rather than raising KeyError.
    partial = {"sport": "nfl", "game_pk": 999, "predicted_winner": "Chiefs"}
    n = db.upsert_prediction_accuracy([partial])

    assert n == 1
    (tup,) = sink["rows"]
    assert tup[0] == "nfl"
    assert tup[1] == 999
    # predicted_winner is column index 6 in _PREDICTION_ACCURACY_COLS
    assert tup[6] == "Chiefs"
    # every other field defaulted to None
    other_positions = [i for i in range(len(tup)) if i not in (0, 1, 6)]
    assert all(tup[i] is None for i in other_positions)


def test_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"sport": "mlb", "game_pk": 1, "winner_correct": True},
        {"sport": "mlb", "game_pk": 2, "winner_correct": False},
    ]
    n = db.upsert_prediction_accuracy(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True
