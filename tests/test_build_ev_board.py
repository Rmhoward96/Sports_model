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


# ---- ML-only NFL Task 3: the Decision Desk is retired from the board ---------------------

class _RowsCursor(_FakeCursor):
    def __init__(self, sink, rows):
        super().__init__(sink)
        self.rows = rows

    def fetchall(self):
        return self.rows


class _RowsConn(_FakeConn):
    def __init__(self, sink, rows):
        super().__init__(sink)
        self.rows = rows

    def cursor(self):
        return _RowsCursor(self.sink, self.rows)


DESK_FIELDS = {"ml_pick", "spread_side", "total_side", "conviction_tier", "confidence"}


def test_load_upcoming_games_sql_does_not_join_desk(monkeypatch):
    sink = []
    monkeypatch.setattr(build_ev_board, "get_postgres", lambda: _FakeConn(sink))
    assert build_ev_board.load_upcoming_games("nfl") == []
    assert len(sink) == 1
    sql, params = sink[0]
    assert "desk_current" not in sql
    assert "desk" not in sql.lower()
    assert "predictions_current" in sql
    assert params == ["nfl"]


def test_game_cols_drop_desk_fields():
    assert build_ev_board.GAME_COLS == ["sport", "game_pk", "matchup", "commence_time"]
    assert not DESK_FIELDS & set(build_ev_board.GAME_COLS)


def test_board_true_prob_is_pinnacle_novig_everywhere(monkeypatch):
    """No desk overlay: every row's desk_delta is 0, edge is 0 and true_prob
    equals Pinnacle's no-vig base_prob -- the board is pure line shopping."""
    from sportsmodel.serving.ev_pilot import assemble_games, ev_rows_for_game

    sink = []
    rows = [("nfl", 101, "Carolina Panthers @ Atlanta Falcons", "2026-10-04T17:00:00Z")]
    monkeypatch.setattr(build_ev_board, "get_postgres", lambda: _RowsConn(sink, rows))
    games_in = build_ev_board.load_upcoming_games("nfl")
    assert games_in == [{"sport": "nfl", "game_pk": 101,
                         "matchup": "Carolina Panthers @ Atlanta Falcons",
                         "commence_time": "2026-10-04T17:00:00Z"}]
    odds = [
        {"game_pk": 101, "market": "moneyline", "side": "home", "book": "pinnacle", "line": None, "price": -150},
        {"game_pk": 101, "market": "moneyline", "side": "away", "book": "pinnacle", "line": None, "price": 130},
        {"game_pk": 101, "market": "spread", "side": "home", "book": "pinnacle", "line": -3.0, "price": -110},
        {"game_pk": 101, "market": "spread", "side": "away", "book": "pinnacle", "line": 3.0, "price": -110},
        {"game_pk": 101, "market": "total", "side": "over", "book": "pinnacle", "line": 44.5, "price": -105},
        {"game_pk": 101, "market": "total", "side": "under", "book": "pinnacle", "line": 44.5, "price": -115},
    ]
    games = assemble_games(games_in, odds)
    assert [g["desk"] for g in games] == [None]
    out = ev_rows_for_game(games[0])
    assert {r["market"] for r in out} == {"moneyline", "spread", "total"}
    for r in out:
        assert r["desk_delta"] == 0
        assert r["edge"] == 0
        assert r["true_prob"] == r["base_prob"]
        assert r["conviction_tier"] is None


def test_main_clears_other_side_only_for_pick_rows(monkeypatch):
    """The engine now writes BOTH sides of every market; clear_other_side_picks demotes the
    opposite side of each PICK, so it must be handed only the is_pick rows -- passing every
    row would demote each side by the other."""
    import sys

    game = {"sport": "nfl", "game_pk": 101, "matchup": "A @ B", "commence_time": "2099-01-01T00:00:00Z"}
    odds = []
    for market, side, line, price in [
            ("moneyline", "home", None, -170), ("moneyline", "away", None, 150),
            ("spread", "home", -3.5, -110), ("spread", "away", 3.5, -110),
            ("total", "over", 47.5, -110), ("total", "under", 47.5, -110)]:
        odds.append({"game_pk": 101, "market": market, "side": side, "book": "pinnacle", "line": line, "price": price})
    odds.append({"game_pk": 101, "market": "moneyline", "side": "away", "book": "betrivers", "line": None, "price": 175})
    seen = {}
    monkeypatch.setattr(build_ev_board, "load_upcoming_games", lambda sport: [game])
    monkeypatch.setattr(build_ev_board, "load_latest_odds", lambda pks: odds)
    monkeypatch.setattr(build_ev_board, "upsert_ev_picks", lambda rows: seen.setdefault("upsert", list(rows)) and len(rows))
    monkeypatch.setattr(build_ev_board, "clear_other_side_picks", lambda rows: seen.setdefault("clear", list(rows)) and 0)
    monkeypatch.setattr(build_ev_board, "build_parlay", lambda cands: None)
    monkeypatch.setattr(build_ev_board, "demote_stale_parlays", lambda *a: 0)
    monkeypatch.setattr(build_ev_board, "upsert_ev_parlays", lambda rows: 0)
    monkeypatch.setattr(build_ev_board, "safe_record", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", ["build_ev_board.py", "--sport", "nfl"])
    build_ev_board.main()
    assert len(seen["upsert"]) == 6                        # both sides of all three markets written
    assert len(seen["clear"]) == 1                         # ...but only the pick is handed to the demoter
    assert all(r["is_pick"] for r in seen["clear"])
    assert (seen["clear"][0]["market"], seen["clear"][0]["side"], seen["clear"][0]["best_book"]) == ("moneyline", "away", "betrivers")
