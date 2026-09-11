"""upsert_ev_picks / upsert_ev_results: column-order tuple building,
no live DB -- mirrors tests/test_db_desk.py's FakeConn pattern."""
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
# upsert_ev_picks
# ---------------------------------------------------------------------------

def test_upsert_ev_picks_helper_exists():
    assert hasattr(db, "upsert_ev_picks")


def test_ev_picks_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_ev_picks([]) == 0


def test_ev_picks_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "nfl",
        "game_pk": 12345,
        "market": "spread",
        "side": "home",
        "model_version": "ev-pilot-v1",
        "matchup": "Bills @ Jets",
        "commence_time": "2026-09-14T17:00:00+00:00",
        "base_prob": 0.52,
        "true_prob": 0.57,
        "edge": 0.05,
        "desk_delta": 0.05,
        "conviction_tier": "A",
        "pinnacle_price": -110,
        "open_pinnacle_price": -108,
        "ev_pinnacle": 0.04,
        "best_book": "draftkings",
        "best_price": -105,
        "ev_best": 0.06,
        "best_line_implied": 0.512,
        "soft_vs_sharp_gap": 0.008,
        "is_pick": True,
    }
    n = db.upsert_ev_picks([row])

    assert n == 1
    assert sink["rows"] == [(
        "nfl", 12345, "spread", "home", "ev-pilot-v1",
        "Bills @ Jets", "2026-09-14T17:00:00+00:00",
        0.52, 0.57, 0.05, 0.05, "A", -110, -108, 0.04,
        "draftkings", -105, 0.06, 0.512, 0.008, True,
    )]
    assert "INSERT INTO ev_picks" in sink["sql"]
    assert "ON CONFLICT (sport, game_pk, market, side, model_version) DO UPDATE" in sink["sql"]
    # PK columns must not be reassigned in the DO UPDATE SET clause
    assert "sport = EXCLUDED.sport" not in sink["sql"]
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "market = EXCLUDED.market" not in sink["sql"]
    assert "side = EXCLUDED.side" not in sink["sql"]
    assert "model_version = EXCLUDED.model_version" not in sink["sql"]
    # created_at must NOT be touched on conflict -- keeps the original insert time
    assert "created_at" not in sink["sql"]
    # open_pinnacle_price is INSERTed but frozen -- never reassigned on conflict,
    # so the pick-time price survives every later board rebuild (for CLV).
    assert "open_pinnacle_price" in sink["sql"]
    assert "open_pinnacle_price = EXCLUDED.open_pinnacle_price" not in sink["sql"]


def test_ev_picks_missing_model_version_defaults_to_ev_pilot_v1(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"sport": "nfl", "game_pk": 999, "market": "moneyline", "side": "away"}
    n = db.upsert_ev_picks([partial])

    assert n == 1
    (tup,) = sink["rows"]
    cols = db._EV_PICKS_COLS
    assert tup[cols.index("model_version")] == "ev-pilot-v1"


def test_ev_picks_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"sport": "nfl", "game_pk": 999, "market": "moneyline", "side": "away",
               "model_version": "ev-pilot-v1"}
    n = db.upsert_ev_picks([partial])

    assert n == 1
    (tup,) = sink["rows"]
    key_positions = {0, 1, 2, 3, 4}
    assert tup[0] == "nfl"
    assert tup[1] == 999
    assert tup[2] == "moneyline"
    assert tup[3] == "away"
    assert tup[4] == "ev-pilot-v1"
    other_positions = [i for i in range(len(tup)) if i not in key_positions]
    assert all(tup[i] is None for i in other_positions)


def test_ev_picks_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"sport": "nfl", "game_pk": 1, "market": "spread", "side": "home",
         "model_version": "ev-pilot-v1"},
        {"sport": "nfl", "game_pk": 2, "market": "total", "side": "over",
         "model_version": "ev-pilot-v1"},
    ]
    n = db.upsert_ev_picks(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True


# ---------------------------------------------------------------------------
# upsert_ev_results
# ---------------------------------------------------------------------------

def test_upsert_ev_results_helper_exists():
    assert hasattr(db, "upsert_ev_results")


def test_ev_results_empty_records_returns_zero_without_touching_db(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.upsert_ev_results([]) == 0


def test_ev_results_tuple_built_in_column_order_with_all_fields(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    row = {
        "sport": "nfl",
        "game_pk": 12345,
        "market": "spread",
        "side": "home",
        "won": True,
        "clv": 1.5,
    }
    n = db.upsert_ev_results([row])

    assert n == 1
    assert sink["rows"] == [(
        "nfl", 12345, "spread", "home", True, 1.5,
    )]
    assert "INSERT INTO ev_results" in sink["sql"]
    assert "ON CONFLICT (sport, game_pk, market, side) DO UPDATE" in sink["sql"]
    assert "graded_at = now()" in sink["sql"]
    assert "sport = EXCLUDED.sport" not in sink["sql"]
    assert "game_pk = EXCLUDED.game_pk" not in sink["sql"]
    assert "market = EXCLUDED.market" not in sink["sql"]
    assert "side = EXCLUDED.side" not in sink["sql"]


def test_ev_results_missing_keys_default_to_none_not_keyerror(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))

    partial = {"sport": "nfl", "game_pk": 999, "market": "total", "side": "over", "won": True}
    n = db.upsert_ev_results([partial])

    assert n == 1
    (tup,) = sink["rows"]
    assert tup[0] == "nfl"
    assert tup[1] == 999
    assert tup[2] == "total"
    assert tup[3] == "over"
    assert tup[4] is True
    other_positions = [i for i in range(len(tup)) if i not in (0, 1, 2, 3, 4)]
    assert all(tup[i] is None for i in other_positions)


def test_ev_results_multiple_records_all_converted_and_commit_called(monkeypatch):
    sink = {}
    conn_holder = {}

    def fake_get_postgres():
        c = FakeConn(sink)
        conn_holder["conn"] = c
        return c

    monkeypatch.setattr(db_module, "get_postgres", fake_get_postgres)

    rows = [
        {"sport": "nfl", "game_pk": 1, "market": "spread", "side": "home", "won": True},
        {"sport": "nfl", "game_pk": 2, "market": "total", "side": "over", "won": False},
    ]
    n = db.upsert_ev_results(rows)

    assert n == 2
    assert len(sink["rows"]) == 2
    assert conn_holder["conn"].committed is True


# ---------------------------------------------------------------------------
# clear_other_side_picks -- demote the stale opposite side of a market
# ---------------------------------------------------------------------------

def test_clear_other_side_picks_empty_is_noop(monkeypatch):
    def boom():
        raise AssertionError("get_postgres should not be called for an empty list")
    monkeypatch.setattr(db_module, "get_postgres", boom)
    assert db.clear_other_side_picks([]) == 0


def test_clear_other_side_picks_demotes_the_opposite_side(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    n = db.clear_other_side_picks([
        {"sport": "nfl", "game_pk": 42, "market": "moneyline", "side": "away",
         "model_version": "ev-pilot-v1"},
    ])
    assert n == 1
    # demotes only the OTHER side of this exact market, and only if still a pick
    assert "UPDATE ev_picks SET is_pick = false" in sink["sql"]
    assert "side <> %s" in sink["sql"]
    assert "is_pick = true" in sink["sql"]
    assert sink["rows"] == [("nfl", 42, "moneyline", "ev-pilot-v1", "away")]


def test_clear_other_side_picks_defaults_model_version(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    db.clear_other_side_picks([
        {"sport": "cfb", "game_pk": 7, "market": "spread", "side": "home"},
    ])
    assert sink["rows"] == [("cfb", 7, "spread", "ev-pilot-v1", "home")]


# ---------------------------------------------------------------------------
# _EV_PICKS_COLS sanity
# ---------------------------------------------------------------------------

def test_ev_picks_cols_matches_ev_rows_for_game_output_shape():
    """_EV_PICKS_COLS must cover every field ev_rows_for_game emits (plus
    model_version, which the board-builder script stamps on)."""
    ev_row_fields = {
        "sport", "game_pk", "matchup", "commence_time", "market", "side",
        "base_prob", "true_prob", "edge", "desk_delta", "conviction_tier",
        "pinnacle_price", "open_pinnacle_price", "ev_pinnacle", "best_book",
        "best_price", "ev_best", "best_line_implied", "soft_vs_sharp_gap",
        "is_pick",
    }
    cols = set(db._EV_PICKS_COLS)
    assert ev_row_fields <= cols
    assert cols - ev_row_fields == {"model_version"}
