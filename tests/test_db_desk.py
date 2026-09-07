"""upsert_desk_picks / upsert_desk_pick_results: column-order tuple building,
no live DB -- mirrors tests/test_db_prediction_accuracy.py's FakeConn pattern."""
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


# ---------------------------------------------------------------------------
# upsert_desk_picks
# ---------------------------------------------------------------------------

def test_upsert_desk_picks_helper_exists():
    assert hasattr(db, "upsert_desk_picks")


def test_desk_picks_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_desk_picks([]) == 0


def test_desk_picks_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "cfb",
        "game_pk": 12345,
        "model_version": "cfb-desk-v1",
        "game_date": "2026-09-05",
        "commence_time": "2026-09-05T19:00:00+00:00",
        "matchup": "Ohio State @ Michigan",
        "ml_pick": "Ohio State",
        "spread_side": "Ohio State -7",
        "spread_line": -7.0,
        "total_side": "Over",
        "total_line": 54.5,
        "confidence": 0.71,
        "conviction_tier": "A",
        "rationale": "Elo edge + healthy roster",
        "agent_notes": '{"statistics": "..."}',
    }
    n = db.upsert_desk_picks([row])

    assert n == 1
    assert sink["rows"] == [(
        "cfb", 12345, "cfb-desk-v1", "2026-09-05", "2026-09-05T19:00:00+00:00",
        "Ohio State @ Michigan", "Ohio State", "Ohio State -7", -7.0,
        "Over", 54.5, 0.71, "A", "Elo edge + healthy roster",
        '{"statistics": "..."}',
    )]
    assert "INSERT INTO desk_picks" in sink["sql"]
    assert "ON CONFLICT (sport, game_pk, model_version) DO UPDATE" in sink["sql"]
    # PK columns must not be reassigned in the DO UPDATE SET clause
    assert "sport = EXCLUDED.sport" not in sink["sql"]
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "model_version = EXCLUDED.model_version" not in sink["sql"]
    # created_at must NOT be touched on conflict -- keeps the original insert time
    assert "created_at" not in sink["sql"]


def test_desk_picks_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"sport": "cfb", "game_pk": 999, "model_version": "cfb-desk-v1"}
    n = db.upsert_desk_picks([partial])

    assert n == 1
    (tup,) = sink["rows"]
    assert tup[0] == "cfb"
    assert tup[1] == 999
    assert tup[2] == "cfb-desk-v1"
    other_positions = [i for i in range(len(tup)) if i not in (0, 1, 2)]
    assert all(tup[i] is None for i in other_positions)


def test_desk_picks_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"sport": "cfb", "game_pk": 1, "model_version": "cfb-desk-v1"},
        {"sport": "cfb", "game_pk": 2, "model_version": "cfb-desk-v1"},
    ]
    n = db.upsert_desk_picks(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True


# ---------------------------------------------------------------------------
# upsert_desk_pick_results
# ---------------------------------------------------------------------------

def test_upsert_desk_pick_results_helper_exists():
    assert hasattr(db, "upsert_desk_pick_results")


def test_desk_pick_results_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_desk_pick_results([]) == 0


def test_desk_pick_results_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "cfb",
        "game_pk": 12345,
        "ml_correct": True,
        "spread_cover": False,
        "total_result": True,
        "clv_spread": 1.5,
        "clv_total": -0.5,
    }
    n = db.upsert_desk_pick_results([row])

    assert n == 1
    assert sink["rows"] == [(
        "cfb", 12345, True, False, True, 1.5, -0.5,
    )]
    assert "INSERT INTO desk_pick_results" in sink["sql"]
    assert "ON CONFLICT (sport, game_pk) DO UPDATE" in sink["sql"]
    assert "graded_at = now()" in sink["sql"]
    assert "sport = EXCLUDED.sport" not in sink["sql"]
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]


def test_desk_pick_results_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"sport": "cfb", "game_pk": 999, "ml_correct": True}
    n = db.upsert_desk_pick_results([partial])

    assert n == 1
    (tup,) = sink["rows"]
    assert tup[0] == "cfb"
    assert tup[1] == 999
    assert tup[2] is True
    other_positions = [i for i in range(len(tup)) if i not in (0, 1, 2)]
    assert all(tup[i] is None for i in other_positions)


def test_desk_pick_results_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"sport": "cfb", "game_pk": 1, "ml_correct": True},
        {"sport": "cfb", "game_pk": 2, "ml_correct": False},
    ]
    n = db.upsert_desk_pick_results(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True
