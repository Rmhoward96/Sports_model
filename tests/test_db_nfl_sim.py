"""upsert_nfl_sim / upsert_nfl_player_sim: column-order tuple building,
no live DB -- mirrors tests/test_db_ev.py's FakeConn pattern."""
import json
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
# upsert_nfl_sim
# ---------------------------------------------------------------------------

def test_upsert_nfl_sim_helper_exists():
    assert hasattr(db, "upsert_nfl_sim")


def test_nfl_sim_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_nfl_sim([]) == 0


def test_nfl_sim_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "game_pk": 12345,
        "model_version": "sim-nfl-v1",
        "matchup": "Bills @ Jets",
        "commence_time": "2026-09-14T17:00:00+00:00",
        "sim_home_win_prob": 0.57,
        "sim_margin": 3.2,
        "sim_total": 44.5,
        "disagreement": 0.08,
    }
    n = db.upsert_nfl_sim([row])

    assert n == 1
    assert sink["rows"] == [(
        12345, "sim-nfl-v1", "Bills @ Jets", "2026-09-14T17:00:00+00:00",
        0.57, 3.2, 44.5, 0.08,
    )]
    assert "INSERT INTO nfl_sim" in sink["sql"]
    assert "ON CONFLICT (game_pk, model_version) DO UPDATE" in sink["sql"]
    # PK columns must not be reassigned in the DO UPDATE SET clause
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "model_version = EXCLUDED.model_version" not in sink["sql"]
    # created_at must NOT be touched on conflict -- keeps the original insert time
    assert "created_at" not in sink["sql"]


def test_nfl_sim_missing_model_version_defaults_to_sim_nfl_v1(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"game_pk": 999, "matchup": "Chiefs @ Broncos"}
    n = db.upsert_nfl_sim([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._NFL_SIM_COLS
    assert tup[cols.index("model_version")] == "sim-nfl-v1"


def test_nfl_sim_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"game_pk": 999}
    n = db.upsert_nfl_sim([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._NFL_SIM_COLS
    assert tup[cols.index("game_pk")] == 999
    assert tup[cols.index("model_version")] == "sim-nfl-v1"
    other_positions = [
        i for i, c in enumerate(cols) if c not in ("game_pk", "model_version")
    ]
    assert all(tup[i] is None for i in other_positions)


def test_nfl_sim_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"game_pk": 1, "model_version": "sim-nfl-v1"},
        {"game_pk": 2, "model_version": "sim-nfl-v1"},
    ]
    n = db.upsert_nfl_sim(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True


# ---------------------------------------------------------------------------
# upsert_nfl_player_sim -- dist serialized to JSON
# ---------------------------------------------------------------------------

def test_upsert_nfl_player_sim_helper_exists():
    assert hasattr(db, "upsert_nfl_player_sim")


def test_nfl_player_sim_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_nfl_player_sim([]) == 0


def test_nfl_player_sim_tuple_built_in_column_order_with_dist_json(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "game_pk": 12345,
        "player_id": "00-0033873",
        "model_version": "sim-nfl-v1",
        "name": "Josh Allen",
        "pos": "QB",
        "team": "BUF",
        "market": "passing_yards",
        "mean": 255.4,
        "dist": {"250": 0.1, "275": 0.2},
        "commence_time": "2026-09-14T17:00:00+00:00",
    }
    n = db.upsert_nfl_player_sim([row])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._NFL_PLAYER_SIM_COLS
    # dist column must be a JSON STRING, not a Python dict
    dist_val = tup[cols.index("dist")]
    assert isinstance(dist_val, str)
    assert json.loads(dist_val) == row["dist"]
    assert tup[cols.index("game_pk")] == 12345
    assert tup[cols.index("player_id")] == "00-0033873"
    assert tup[cols.index("mean")] == 255.4
    assert "INSERT INTO nfl_player_sim" in sink["sql"]
    assert (
        "ON CONFLICT (game_pk, player_id, market, model_version) DO UPDATE"
        in sink["sql"]
    )
    assert "created_at" not in sink["sql"]
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "player_id = EXCLUDED.player_id" not in sink["sql"]
    assert "market = EXCLUDED.market" not in sink["sql"]
    assert "model_version = EXCLUDED.model_version" not in sink["sql"]


def test_nfl_player_sim_missing_model_version_defaults_to_sim_nfl_v1(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"game_pk": 999, "player_id": "abc", "market": "rushing_yards"}
    n = db.upsert_nfl_player_sim([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._NFL_PLAYER_SIM_COLS
    assert tup[cols.index("model_version")] == "sim-nfl-v1"


def test_nfl_player_sim_missing_dist_serializes_none(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"game_pk": 999, "player_id": "abc", "market": "rushing_yards"}
    n = db.upsert_nfl_player_sim([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._NFL_PLAYER_SIM_COLS
    dist_val = tup[cols.index("dist")]
    assert dist_val == json.dumps(None)


def test_nfl_player_sim_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"game_pk": 1, "player_id": "a", "market": "passing_yards",
         "model_version": "sim-nfl-v1"},
        {"game_pk": 2, "player_id": "b", "market": "rushing_yards",
         "model_version": "sim-nfl-v1"},
    ]
    n = db.upsert_nfl_player_sim(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True
