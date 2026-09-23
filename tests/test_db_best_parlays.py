"""ev_best_parlays helpers: replace_unlocked_best_parlays, locked_leg_keys,
ungraded_locked_parlays, upsert_best_parlay_results -- column-order tuple
building + SQL shape, no live DB. FakeConn pattern extends
tests/test_db_nfl_player_actuals.py with execute()/fetchall()."""
import json
from datetime import datetime, timezone

from sportsmodel import db
from sportsmodel import db as db_module


class FakeCursor:
    def __init__(self, sink, fetch_rows=None):
        self.sink = sink
        self.rowcount = 0
        self._fetch_rows = fetch_rows if fetch_rows is not None else []

    def execute(self, sql, params=None):
        self.sink.setdefault("calls", []).append(("execute", sql, params))

    def executemany(self, sql, rows):
        rows = list(rows)
        self.sink.setdefault("calls", []).append(("executemany", sql, rows))
        self.rowcount = len(rows)

    def fetchall(self):
        return self._fetch_rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink, fetch_rows=None):
        self.sink = sink
        self.committed = False
        self._fetch_rows = fetch_rows if fetch_rows is not None else []

    def cursor(self):
        return FakeCursor(self.sink, self._fetch_rows)

    def commit(self):
        self.committed = True
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ticket(parlay_id="dk|a|b|c"):
    return {
        "parlay_id": parlay_id,
        "sport": "nfl",
        "book": "draftkings",
        "legs": [
            {"key": "g:1:moneyline:home", "kind": "game", "sport": "nfl", "game_pk": 1,
             "market": "moneyline", "side": "home", "line": None, "prob": 0.6, "price": -150,
             "label": "Home ML", "matchup": "AWAY @ HOME",
             "commence_time": datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
             "player_id": None, "player_name": None},
        ],
        "n_legs": 1,
        "parlay_dec": 1.66,
        "parlay_price": -150,
        "true_prob": 0.6,
        "ev": 0.0,
        "first_commence": datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
        "last_commence": datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
        "model_version": "parlay-v1",
    }


# ---------------------------------------------------------------------------
# _BEST_PARLAY_COLS
# ---------------------------------------------------------------------------

def test_best_parlay_cols_shape():
    assert db._BEST_PARLAY_COLS == [
        "parlay_id", "sport", "book", "legs", "n_legs", "parlay_dec", "parlay_price",
        "true_prob", "ev", "first_commence", "last_commence", "model_version",
    ]


# ---------------------------------------------------------------------------
# replace_unlocked_best_parlays
# ---------------------------------------------------------------------------

def test_replace_unlocked_best_parlays_deletes_then_inserts_and_commits(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    ticket = _ticket()
    assert db.replace_unlocked_best_parlays([ticket]) == 1

    calls = sink["calls"]
    assert calls[0][0] == "execute"
    assert "DELETE FROM ev_best_parlays" in calls[0][1]
    assert "first_commence > now()" in calls[0][1]

    assert calls[1][0] == "executemany"
    assert "INSERT INTO ev_best_parlays" in calls[1][1]
    (row,) = calls[1][2]
    cols = db._BEST_PARLAY_COLS
    legs_json = row[cols.index("legs")]
    assert isinstance(legs_json, str)
    decoded = json.loads(legs_json)
    assert decoded[0]["key"] == "g:1:moneyline:home"
    # commence_time (a datetime) survives json.dumps via default=str
    assert isinstance(decoded[0]["commence_time"], str)
    assert row[cols.index("parlay_id")] == ticket["parlay_id"]
    assert row[cols.index("book")] == "draftkings"

    assert sink["committed"] is True


def test_replace_unlocked_best_parlays_empty_still_deletes_no_insert(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    assert db.replace_unlocked_best_parlays([]) == 0
    calls = sink["calls"]
    assert len(calls) == 1
    assert calls[0][0] == "execute"
    assert "DELETE FROM ev_best_parlays" in calls[0][1]
    assert sink["committed"] is True


# ---------------------------------------------------------------------------
# locked_leg_keys
# ---------------------------------------------------------------------------

def test_locked_leg_keys_flattens_keys_from_list_legs(monkeypatch):
    sink = {}
    rows = [
        ([{"key": "g:1:moneyline:home"}, {"key": "p:2:00-1:rec_yds:over"}],),
        ([{"key": "g:3:total:over"}],),
    ]
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink, rows))
    keys = db.locked_leg_keys()
    assert keys == frozenset({"g:1:moneyline:home", "p:2:00-1:rec_yds:over", "g:3:total:over"})
    call = sink["calls"][0]
    assert call[0] == "execute"
    assert "first_commence <= now()" in call[1]
    assert "NOT EXISTS" in call[1]


def test_locked_leg_keys_flattens_keys_from_json_string_legs(monkeypatch):
    sink = {}
    rows = [(json.dumps([{"key": "g:9:spread:away"}]),)]
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink, rows))
    assert db.locked_leg_keys() == frozenset({"g:9:spread:away"})


def test_locked_leg_keys_empty(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink, []))
    assert db.locked_leg_keys() == frozenset()


# ---------------------------------------------------------------------------
# ungraded_locked_parlays
# ---------------------------------------------------------------------------

def test_ungraded_locked_parlays_returns_dicts(monkeypatch):
    sink = {}
    cols = ["parlay_id", "sport", "book", "legs", "parlay_price", "first_commence"]
    rows = [
        ("dk|a|b|c", "nfl", "draftkings", [{"key": "g:1:moneyline:home"}], -150,
         datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc)),
    ]
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink, rows))
    result = db.ungraded_locked_parlays()
    assert result == [dict(zip(cols, rows[0]))]
    call = sink["calls"][0]
    assert call[0] == "execute"
    assert "first_commence <= now()" in call[1]
    assert "NOT EXISTS" in call[1]


def test_ungraded_locked_parlays_empty(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink, []))
    assert db.ungraded_locked_parlays() == []


# ---------------------------------------------------------------------------
# upsert_best_parlay_results
# ---------------------------------------------------------------------------

def test_upsert_best_parlay_results_empty_returns_zero_no_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_best_parlay_results([]) == 0


def test_upsert_best_parlay_results_upsert_sql_and_json(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    record = {
        "parlay_id": "dk|a|b|c", "sport": "nfl", "book": "draftkings",
        "parlay_price": -150, "first_commence": datetime(2026, 9, 27, 17, 0, tzinfo=timezone.utc),
        "result": "win", "pnl": 6.67, "payout_dec": 1.667,
        "legs": [{"key": "g:1:moneyline:home", "result": "win"}],
    }
    assert db.upsert_best_parlay_results([record]) == 1

    call = sink["calls"][0]
    assert call[0] == "executemany"
    assert "INSERT INTO ev_best_parlay_results" in call[1]
    assert "ON CONFLICT (parlay_id) DO UPDATE" in call[1]
    (row,) = call[2]
    cols = db._BEST_PARLAY_RESULT_COLS
    legs_json = row[cols.index("legs")]
    assert isinstance(legs_json, str)
    assert json.loads(legs_json)[0]["result"] == "win"
    assert row[cols.index("parlay_id")] == "dk|a|b|c"
    assert sink["committed"] is True
