"""Light pure test for scripts/build_ev_props_board.py.

Loads the script module directly (it isn't a package) via importlib, same
pattern as tests/test_grade_ev.py. No network, no DB -- only checks the
prop-market-codes list `load_latest_prop_odds` restricts its query to, and
that a sport with no known games/prop markets short-circuits without
touching the DB.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_ev_props_board.py"
_spec = importlib.util.spec_from_file_location("build_ev_props_board", _SCRIPT_PATH)
build_ev_props_board = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build_ev_props_board)

from sportsmodel import sports  # noqa: E402


def test_nfl_prop_market_codes_match_sport_config_prop_market_map_keys():
    # The script queries odds_snapshot restricted to this sport's prop market
    # CODES (SportConfig["nfl"].prop_market_map keys, e.g. "reception_yds"),
    # not the Odds API's own market keys (the map's values).
    expected = set(sports.get("nfl").prop_market_map.keys())
    assert expected == {
        "pass_yds", "pass_tds", "reception_yds", "receptions",
        "rush_yds", "rush_reception_yds", "rush_att", "anytime_td",
    }


def test_load_latest_prop_odds_empty_game_pks_short_circuits_no_db():
    # No game_pks -> no query issued (would raise if it tried to open a DB
    # connection without DATABASE_URL set).
    assert build_ev_props_board.load_latest_prop_odds("nfl", []) == []


def test_load_sim_rows_non_nfl_sport_short_circuits_no_db():
    # C v1 is NFL-only; other sports short-circuit rather than querying
    # NFL-only tables/views.
    assert build_ev_props_board.load_sim_rows("mlb") == []


def test_load_latest_prop_odds_sport_with_no_prop_markets_short_circuits_no_db():
    # cfb's prop_market_map is empty in C v1 -- no prop markets to query for.
    assert build_ev_props_board.load_latest_prop_odds("cfb", [123]) == []


def _odds_row(game_pk=1, market="rush_yds", side="over", player_name="A", book="dk",
              line=54.5, price=-110, captured_at="2026-09-17T00:00:00+00:00"):
    return {
        "game_pk": game_pk, "market": market, "side": side, "player_name": player_name,
        "book": book, "line": line, "price": price, "captured_at": captured_at,
    }


def test_latest_capture_only_keeps_only_newest_capture_both_sides():
    # A book's superseded line (Wednesday 54.5) must not survive alongside its
    # newer capture (Sunday 58.5) -- both sides (over/under) of the OLD
    # capture are dropped, both sides of the NEW capture are kept.
    rows = [
        _odds_row(side="over", line=54.5, price=-110, captured_at="2026-09-17T00:00:00+00:00"),
        _odds_row(side="under", line=54.5, price=-110, captured_at="2026-09-17T00:00:00+00:00"),
        _odds_row(side="over", line=58.5, price=-115, captured_at="2026-09-21T00:00:00+00:00"),
        _odds_row(side="under", line=58.5, price=-105, captured_at="2026-09-21T00:00:00+00:00"),
    ]
    kept = build_ev_props_board.latest_capture_only(rows)
    assert len(kept) == 2
    assert all(r["line"] == 58.5 for r in kept)
    assert {r["side"] for r in kept} == {"over", "under"}


def test_latest_capture_only_two_books_each_keep_their_own_latest():
    rows = [
        _odds_row(book="dk", line=54.5, captured_at="2026-09-17T00:00:00+00:00"),
        _odds_row(book="dk", line=58.5, captured_at="2026-09-21T00:00:00+00:00"),
        _odds_row(book="fd", line=55.5, captured_at="2026-09-18T00:00:00+00:00"),
        _odds_row(book="fd", line=57.5, captured_at="2026-09-20T00:00:00+00:00"),
    ]
    kept = build_ev_props_board.latest_capture_only(rows)
    by_book = {r["book"]: r["line"] for r in kept}
    assert by_book == {"dk": 58.5, "fd": 57.5}


def test_latest_capture_only_different_players_and_markets_independent():
    rows = [
        _odds_row(player_name="A", market="rush_yds", line=54.5, captured_at="2026-09-17T00:00:00+00:00"),
        _odds_row(player_name="A", market="rush_yds", line=58.5, captured_at="2026-09-21T00:00:00+00:00"),
        _odds_row(player_name="B", market="rush_yds", line=40.5, captured_at="2026-09-19T00:00:00+00:00"),
        _odds_row(player_name="A", market="receptions", line=4.5, captured_at="2026-09-19T00:00:00+00:00"),
    ]
    kept = build_ev_props_board.latest_capture_only(rows)
    lines = {(r["player_name"], r["market"]): r["line"] for r in kept}
    assert lines == {
        ("A", "rush_yds"): 58.5,
        ("B", "rush_yds"): 40.5,
        ("A", "receptions"): 4.5,
    }


def _stub_loaders(monkeypatch):
    # No DB touched: sim/odds loaders return small fixed lists so
    # assemble_prop_rows/assemble_prop_line_rows run on real (pure) code with
    # no network/DB access anywhere in main().
    monkeypatch.setattr(build_ev_props_board, "load_sim_rows", lambda sport: [])
    monkeypatch.setattr(build_ev_props_board, "load_latest_prop_odds", lambda sport, pks: [])


def test_main_lines_upsert_failure_still_runs_ev_board_when_not_lines_only(monkeypatch, capsys):
    # A missing nfl_prop_lines table (migration not applied) or any other
    # lines-write error must not abort the game-day +EV board.
    _stub_loaders(monkeypatch)

    def boom(rows):
        raise RuntimeError('relation "nfl_prop_lines" does not exist')

    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines", boom)
    calls = {}
    monkeypatch.setattr(build_ev_props_board, "upsert_ev_prop_picks", lambda rows: calls.__setitem__("ev", rows))
    monkeypatch.setattr(build_ev_props_board, "clear_stale_prop_line_picks", lambda picks: 0)

    build_ev_props_board.main([])

    assert "ev" in calls  # the +EV upsert ran despite the lines failure
    out = capsys.readouterr().out
    assert "[build_ev_props_board] prop_lines FAILED: RuntimeError:" in out


def test_main_lines_upsert_failure_reraises_when_lines_only(monkeypatch):
    # --lines-only's only job is the lines write; if that fails the run's
    # only job failed, so it must exit non-zero (propagate the exception).
    _stub_loaders(monkeypatch)

    def boom(rows):
        raise RuntimeError("boom")

    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines", boom)
    ev_called = []
    monkeypatch.setattr(build_ev_props_board, "upsert_ev_prop_picks", lambda rows: ev_called.append(rows))

    with pytest.raises(RuntimeError, match="boom"):
        build_ev_props_board.main(["--lines-only"])

    assert not ev_called  # +EV path never reached


def test_main_lines_upsert_success_with_lines_only_returns_before_ev_board(monkeypatch):
    _stub_loaders(monkeypatch)
    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines", lambda rows: 0)
    ev_called = []
    monkeypatch.setattr(build_ev_props_board, "upsert_ev_prop_picks", lambda rows: ev_called.append(rows))

    build_ev_props_board.main(["--lines-only"])

    assert not ev_called


def test_load_latest_prop_odds_ignores_book_prices_older_than_48h(monkeypatch):
    # A book that stopped updating must not keep a days-old capture as its
    # "current" prop price -- same 48h window as the site's price views.
    import re

    sink = []

    class _Cur:
        def execute(self, sql, params=None):
            sink.append((sql, params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Conn:
        def cursor(self):
            return _Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(build_ev_props_board, "get_postgres", lambda: _Conn())
    assert build_ev_props_board.load_latest_prop_odds("nfl", [5]) == []
    assert len(sink) == 1
    sql, params = sink[0]
    assert re.search(r"captured_at\s*>\s*now\(\)\s*-\s*interval\s*'48 hours'", sql)
    assert "captured_at <= commence_time" in sql
    assert params[0] == [5]
