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


def test_load_sim_rows_sport_without_a_prop_model_short_circuits_no_db():
    # Only NFL (sim tables) and MLB (prop_predictions) have a prop model; CFB short-circuits
    # rather than querying NFL-only tables/views.
    assert build_ev_props_board.load_sim_rows("cfb") == []


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


def test_drop_pulled_lines_keeps_only_the_latest_pull_per_game_market():
    from datetime import datetime, timedelta, timezone

    from sportsmodel.serving.props_ev import drop_pulled_lines

    t = datetime(2026, 9, 28, 20, 56, tzinfo=timezone.utc)
    row = lambda name, market, at, book="dk": {
        "game_pk": 1, "market": market, "side": "over", "player_name": name,
        "book": book, "line": 200.5, "price": -110, "captured_at": at}
    rows = [
        row("Tyson Bagent", "pass_yds", t - timedelta(days=2)),       # pulled by the books
        row("Case Keenum", "pass_yds", t),
        row("Jalen Hurts", "pass_yds", t - timedelta(minutes=15), "fd"),  # same pull, a bit earlier
        row("DJ Moore", "rec_yds", t - timedelta(hours=5)),          # rec_yds' own latest pull
    ]
    kept = {(r["player_name"], r["market"]) for r in drop_pulled_lines(rows)}
    assert kept == {("Case Keenum", "pass_yds"), ("Jalen Hurts", "pass_yds"), ("DJ Moore", "rec_yds")}


# -- MLB (reactivated 2026-10-05) ---------------------------------------------------

def test_mlb_prop_market_codes_match_sport_config_and_the_board_map():
    from sportsmodel.serving.props_ev import MLB_SIM_TO_ODDS_MARKET
    assert set(sports.get("mlb").prop_market_map) == set(MLB_SIM_TO_ODDS_MARKET) == {
        "total_bases", "pitcher_ks", "hits_allowed", "outs_recorded"}


def test_mlb_sim_cols_have_the_keys_assemble_prop_rows_reads():
    assert {"game_pk", "player_id", "name", "market", "mean", "dist", "commence_time", "matchup"} <= set(
        build_ev_props_board.MLB_SIM_COLS)
    # the MLB SELECT in _load_mlb_sim_rows yields exactly one value per column
    assert len(build_ev_props_board.MLB_SIM_COLS) == 10


def test_mlb_lines_only_is_a_noop_before_any_db_access(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("must not touch the DB")

    monkeypatch.setattr(build_ev_props_board, "load_sim_rows", boom)
    monkeypatch.setattr(build_ev_props_board, "load_latest_prop_odds", boom)
    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines", boom)
    build_ev_props_board.main(["--sport", "mlb", "--lines-only"])
    assert "no-op" in capsys.readouterr().out


def test_mlb_main_builds_mlb_rows_without_touching_nfl_prop_lines(monkeypatch):
    pmf = [0.0] * 11
    pmf[0], pmf[1], pmf[2], pmf[3], pmf[4] = 0.25, 0.2, 0.2, 0.2, 0.15   # P(>1.5) = 0.55
    sim = [{"game_pk": 849819, "player_id": "660271", "model_version": "mlb-hybrid-props-v1",
            "name": "Shohei Ohtani", "team": "Los Angeles Dodgers", "market": "total_bases", "mean": 1.6,
            "dist": {"kind": "pmf", "pmf": pmf}, "commence_time": "2026-10-06T22:00:00Z",
            "matchup": "Los Angeles Dodgers @ Atlanta Braves"}]
    odds_rows = [
        {"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani",
         "book": "draftkings", "line": 1.5, "price": 110, "captured_at": "2026-10-06T21:00:00+00:00"},
        {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani",
         "book": "fanduel", "line": 1.5, "price": -130, "captured_at": "2026-10-06T21:00:00+00:00"},
    ]
    seen = {}
    monkeypatch.setattr(build_ev_props_board, "load_sim_rows", lambda sport: seen.setdefault("sport", sport) and sim)
    monkeypatch.setattr(build_ev_props_board, "load_latest_prop_odds", lambda sport, pks: odds_rows)
    monkeypatch.setattr(build_ev_props_board, "calibration", type("C", (), {"calibrate": staticmethod(lambda m, p: p)}))
    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines",
                        lambda rows: (_ for _ in ()).throw(AssertionError("NFL lines table touched")))
    written, cleared = {}, {}
    monkeypatch.setattr(build_ev_props_board, "upsert_ev_prop_picks", lambda rows: written.setdefault("rows", rows))
    monkeypatch.setattr(build_ev_props_board, "clear_stale_prop_line_picks",
                        lambda picks: cleared.setdefault("picks", picks) and 0)

    build_ev_props_board.main(["--sport", "mlb"])

    assert seen["sport"] == "mlb"
    [row] = written["rows"]
    assert row["sport"] == "mlb" and row["model_version"] == "props-mlb-v1"
    assert row["market"] == "total_bases" and row["side"] == "over" and row["is_pick"] is True
    assert row["player_id"] == "660271"
    assert cleared["picks"] == [row]
