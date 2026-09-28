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

    def execute(self, sql, params=None):
        # Captures the per-game DELETE(s) that upsert_nfl_player_sim issues
        # before its bulk INSERT (replace-per-game semantics).
        self.sink.setdefault("execs", []).append((sql, params))

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
        "margin_dist": {"kind": "margin", "offset": 2, "pmf": [0.5, 0.5]},
        "total_dist": {"kind": "pmf", "pmf": [0.3, 0.7]},
        "away_score_dist": {"kind": "pmf", "pmf": [1.0]},
        "home_score_dist": {"kind": "pmf", "pmf": [1.0]},
    }
    n = db.upsert_nfl_sim([row])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._NFL_SIM_COLS
    assert tup[cols.index("game_pk")] == 12345
    assert tup[cols.index("matchup")] == "Bills @ Jets"
    assert tup[cols.index("sim_total")] == 44.5
    # dist columns are JSON STRINGS, not Python dicts
    md = tup[cols.index("margin_dist")]
    assert isinstance(md, str)
    assert json.loads(md) == row["margin_dist"]
    assert json.loads(tup[cols.index("total_dist")]) == row["total_dist"]
    assert "INSERT INTO nfl_sim" in sink["sql"]
    assert "ON CONFLICT (game_pk, model_version) DO UPDATE" in sink["sql"]
    # PK columns must not be reassigned in the DO UPDATE SET clause
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "model_version = EXCLUDED.model_version" not in sink["sql"]
    # created_at must NOT be touched on conflict -- keeps the original insert time
    assert "created_at" not in sink["sql"]


def test_nfl_sim_dist_columns_present_and_default_to_json_null(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    n = db.upsert_nfl_sim([{"game_pk": 7}])
    assert n == 1
    cols = db._NFL_SIM_COLS
    for c in ("margin_dist", "total_dist", "away_score_dist", "home_score_dist"):
        assert c in cols
        # a missing dist serializes to JSON null (the string "null"), not None
        assert sink["rows"][0][cols.index(c)] == json.dumps(None)


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
    dist_cols = {"margin_dist", "total_dist", "away_score_dist", "home_score_dist"}
    scalar_positions = [
        i for i, c in enumerate(cols)
        if c not in ("game_pk", "model_version") and c not in dist_cols
    ]
    assert all(tup[i] is None for i in scalar_positions)
    # missing dist cols serialize to JSON null, not Python None
    for c in dist_cols:
        assert tup[cols.index(c)] == json.dumps(None)


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
    # Replace-per-game semantics: a plain INSERT (no upsert), preceded by a
    # DELETE of this game's rows for the model_version, so orphaned players
    # from a prior run don't linger.
    assert "ON CONFLICT" not in sink["sql"]
    assert "created_at" not in sink["sql"]
    deletes = sink.get("execs", [])
    assert len(deletes) == 1
    del_sql, del_params = deletes[0]
    assert "DELETE FROM nfl_player_sim" in del_sql
    assert "game_pk = %s" in del_sql and "model_version = %s" in del_sql
    assert del_params == (12345, "sim-nfl-v1")


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
    # One DELETE per distinct (game_pk, model_version) before the bulk insert.
    del_params = {p for (_sql, p) in sink.get("execs", [])}
    assert del_params == {(1, "sim-nfl-v1"), (2, "sim-nfl-v1")}
    assert conn_holder["conn"].committed is True


# ---------------------------------------------------------------------------
# upsert_nfl_sim_slate -- game rows + player rows in ONE transaction
# served_nfl_sim_version -- nfl_sim_serving read (None when the table is absent)
# ---------------------------------------------------------------------------

class _LogCursor:
    def __init__(self, log, fetch=None, raise_on_execute=None):
        self.log = log
        self.fetch = fetch
        self.raise_on_execute = raise_on_execute

    def execute(self, sql, params=None):
        if self.raise_on_execute is not None:
            raise self.raise_on_execute
        self.log.append(("execute", sql, params))

    def executemany(self, sql, rows):
        self.log.append(("executemany", sql, list(rows)))

    def fetchone(self):
        return self.fetch

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _LogConn:
    def __init__(self, log, **cursor_kw):
        self.log = log
        self.cursor_kw = cursor_kw

    def cursor(self):
        return _LogCursor(self.log, **self.cursor_kw)

    def commit(self):
        self.log.append(("commit",))

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _conns(monkeypatch, **cursor_kw):
    log, opened = [], []

    def fake():
        opened.append(1)
        return _LogConn(log, **cursor_kw)

    monkeypatch.setattr(db_module, "get_postgres", fake)
    return log, opened


def test_nfl_sim_slate_one_connection_one_commit(monkeypatch):
    log, opened = _conns(monkeypatch)
    sim = [{"game_pk": 1, "model_version": "nfl-sim-ml-v1", "margin_dist": {"pmf": [1.0]}},
           {"game_pk": 2, "model_version": "nfl-sim-ml-v1"}]
    players = [{"game_pk": 1, "player_id": "a", "market": "rec_yds", "model_version": "nfl-sim-ml-v1",
                "dist": {"pmf": [1.0]}},
               {"game_pk": 2, "player_id": "b", "market": "rec_yds", "model_version": "nfl-sim-ml-v1"}]
    assert db.upsert_nfl_sim_slate(sim, players) == (2, 2)
    assert len(opened) == 1                               # one connection
    assert [e[0] for e in log].count("commit") == 1 and log[-1] == ("commit",)   # one transaction
    kinds = [(e[0], e[1].split()[0] + " " + e[1].split()[2]) for e in log if e[0] != "commit"]
    # game rows upserted, then each game's player rows replaced, then inserted
    assert kinds[0] == ("executemany", "INSERT nfl_sim")
    assert sorted(e[2] for e in log if e[0] == "execute") == [(1, "nfl-sim-ml-v1"), (2, "nfl-sim-ml-v1")]
    assert kinds[-1] == ("executemany", "INSERT nfl_player_sim")
    sim_sql = log[0][1]
    assert "ON CONFLICT (game_pk, model_version) DO UPDATE" in sim_sql
    assert json.loads(log[0][2][0][db._NFL_SIM_COLS.index("margin_dist")]) == {"pmf": [1.0]}
    ins = log[-2]
    assert ins[0] == "executemany" and "INSERT INTO nfl_player_sim" in ins[1]
    assert json.loads(ins[2][0][db._NFL_PLAYER_SIM_COLS.index("dist")]) == {"pmf": [1.0]}


def test_nfl_sim_slate_matches_the_two_upserts_statements(monkeypatch):
    """Same SQL + row tuples as upsert_nfl_sim followed by upsert_nfl_player_sim."""
    sim = [{"game_pk": 1, "model_version": "nfl-sim-ml-v1", "matchup": "A @ B"}]
    players = [{"game_pk": 1, "player_id": "a", "market": "rec_yds", "model_version": "nfl-sim-ml-v1",
                "mean": 3.0}]
    log, _ = _conns(monkeypatch)
    db.upsert_nfl_sim_slate(sim, players)
    slate = [e for e in log if e[0] != "commit"]
    log2, _ = _conns(monkeypatch)
    db.upsert_nfl_sim(sim)
    db.upsert_nfl_player_sim(players)
    assert slate == [e for e in log2 if e[0] != "commit"]


def test_nfl_sim_slate_empty_touches_nothing(monkeypatch):
    def boom():
        raise AssertionError("no DB for an empty slate")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_nfl_sim_slate([], []) == (0, 0)


def test_served_nfl_sim_version_reads_the_row(monkeypatch):
    log, _ = _conns(monkeypatch, fetch=("nfl-sim-ml-v1",))
    assert db.served_nfl_sim_version() == "nfl-sim-ml-v1"
    ((op, sql, _),) = log
    assert op == "execute" and "SELECT model_version FROM nfl_sim_serving WHERE id = 1" in sql


def test_served_nfl_sim_version_none_when_table_missing_or_row_missing(monkeypatch):
    import psycopg
    _conns(monkeypatch, raise_on_execute=psycopg.errors.UndefinedTable("relation does not exist"))
    assert db.served_nfl_sim_version() is None
    _conns(monkeypatch, fetch=None)
    assert db.served_nfl_sim_version() is None


def test_served_nfl_sim_version_other_errors_propagate(monkeypatch):
    import psycopg
    _conns(monkeypatch, raise_on_execute=psycopg.OperationalError("connection lost"))
    with pytest.raises(psycopg.OperationalError):
        db.served_nfl_sim_version()
