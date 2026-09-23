"""Light pure test for scripts/build_ev_props_board.py.

Loads the script module directly (it isn't a package) via importlib, same
pattern as tests/test_grade_ev.py. No network, no DB -- only checks the
prop-market-codes list `load_latest_prop_odds` restricts its query to, and
that a sport with no known games/prop markets short-circuits without
touching the DB.
"""
import importlib.util
from pathlib import Path

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
