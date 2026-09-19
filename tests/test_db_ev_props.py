"""upsert_ev_prop_picks / upsert_ev_prop_results: column-order tuple building,
ON CONFLICT key + DO UPDATE shape, no live DB -- mirrors
tests/test_db_nfl_sim.py's FakeConn pattern."""
import pytest

from sportsmodel import db as db_module
from sportsmodel import db


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


# ---------------------------------------------------------------------------
# upsert_ev_prop_picks
# ---------------------------------------------------------------------------

def test_upsert_ev_prop_picks_helper_exists():
    assert hasattr(db, "upsert_ev_prop_picks")


def test_ev_prop_picks_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_ev_prop_picks([]) == 0


def test_ev_prop_picks_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "nfl",
        "game_pk": 12345,
        "player_id": "00-0033873",
        "player_name": "Josh Allen",
        "market": "passing_yards",
        "side": "over",
        "line": 250.5,
        "model_version": "props-sim-v1",
        "matchup": "Bills @ Jets",
        "commence_time": "2026-09-14T17:00:00+00:00",
        "model_prob": 0.55,
        "market_prob": 0.5,
        "edge": 0.05,
        "ev_best": 0.04,
        "best_book": "DraftKings",
        "best_price": -110,
        "pinnacle_price": -105,
        "open_pinnacle_price": -108,
        "is_pick": True,
    }
    n = db.upsert_ev_prop_picks([row])

    assert n == 1
    cols = db._EV_PROP_PICKS_COLS
    (tup,) = sink["rows"]
    assert tup == tuple(row[c] for c in cols)
    assert "INSERT INTO ev_prop_picks" in sink["sql"]
    assert (
        "ON CONFLICT (game_pk, player_id, market, line, model_version) DO UPDATE"
        in sink["sql"]
    )
    # key columns must not be reassigned in the DO UPDATE SET clause
    for key_col in ("game_pk", "player_id", "market", "line", "model_version"):
        assert f"{key_col} = EXCLUDED.{key_col}" not in sink["sql"]
    # created_at must NOT be touched -- keeps the original insert time
    assert "created_at" not in sink["sql"]
    # open_pinnacle_price is the CLV anchor -- immutable, never reassigned
    assert "open_pinnacle_price = EXCLUDED.open_pinnacle_price" not in sink["sql"]
    # a mutable field IS reassigned on conflict
    assert "edge = EXCLUDED.edge" in sink["sql"]


def test_ev_prop_picks_missing_model_version_defaults_to_props_sim_v1(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {
        "game_pk": 999, "player_id": "abc", "market": "rushing_yards",
        "line": 55.5,
    }
    n = db.upsert_ev_prop_picks([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._EV_PROP_PICKS_COLS
    assert tup[cols.index("model_version")] == db._EV_PROP_DEFAULT_MODEL_VERSION
    assert db._EV_PROP_DEFAULT_MODEL_VERSION == "props-sim-v1"


def test_ev_prop_picks_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"game_pk": 999, "player_id": "abc", "market": "rushing_yards", "line": 55.5}
    n = db.upsert_ev_prop_picks([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._EV_PROP_PICKS_COLS
    other_positions = [
        i for i, c in enumerate(cols)
        if c not in ("game_pk", "player_id", "market", "line", "model_version")
    ]
    assert all(tup[i] is None for i in other_positions)


def test_ev_prop_picks_immutable_set_matches_open_pinnacle_price():
    assert db._EV_PROP_PICKS_IMMUTABLE == frozenset({"open_pinnacle_price"})


def test_ev_prop_picks_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"game_pk": 1, "player_id": "a", "market": "passing_yards", "line": 250.5,
         "model_version": "props-sim-v1"},
        {"game_pk": 2, "player_id": "b", "market": "rushing_yards", "line": 55.5,
         "model_version": "props-sim-v1"},
    ]
    n = db.upsert_ev_prop_picks(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True


# ---------------------------------------------------------------------------
# upsert_ev_prop_results
# ---------------------------------------------------------------------------

def test_upsert_ev_prop_results_helper_exists():
    assert hasattr(db, "upsert_ev_prop_results")


def test_ev_prop_results_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_ev_prop_results([]) == 0


def test_ev_prop_results_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "nfl",
        "game_pk": 12345,
        "player_id": "00-0033873",
        "player_name": "Josh Allen",
        "market": "passing_yards",
        "side": "over",
        "line": 250.5,
        "model_version": "props-sim-v1",
        "commence_time": "2026-09-14T17:00:00+00:00",
        "model_prob": 0.55,
        "novig_close": 0.5,
        "actual": 275.0,
        "result": "win",
        "clv": 0.05,
        "profit": 0.91,
    }
    n = db.upsert_ev_prop_results([row])

    assert n == 1
    cols = db._EV_PROP_RESULTS_COLS
    (tup,) = sink["rows"]
    assert tup == tuple(row[c] for c in cols)
    assert "INSERT INTO ev_prop_results" in sink["sql"]
    assert (
        "ON CONFLICT (game_pk, player_id, market, line, model_version) DO UPDATE"
        in sink["sql"]
    )
    for key_col in ("game_pk", "player_id", "market", "line", "model_version"):
        assert f"{key_col} = EXCLUDED.{key_col}" not in sink["sql"]
    assert "created_at" not in sink["sql"]
    assert "result = EXCLUDED.result" in sink["sql"]


def test_ev_prop_results_missing_model_version_defaults_to_props_sim_v1(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {
        "game_pk": 999, "player_id": "abc", "market": "rushing_yards", "line": 55.5,
    }
    n = db.upsert_ev_prop_results([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._EV_PROP_RESULTS_COLS
    assert tup[cols.index("model_version")] == db._EV_PROP_DEFAULT_MODEL_VERSION


def test_ev_prop_results_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"game_pk": 1, "player_id": "a", "market": "passing_yards", "line": 250.5,
         "model_version": "props-sim-v1"},
        {"game_pk": 2, "player_id": "b", "market": "rushing_yards", "line": 55.5,
         "model_version": "props-sim-v1"},
    ]
    n = db.upsert_ev_prop_results(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True
