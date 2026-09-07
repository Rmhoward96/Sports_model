"""Pure-assembly tests for scripts/desk_inputs.py's build_bundle.

No network/DB/file access: build_bundle takes already-fetched/parsed inputs
(games, model_rows, form_rows, injuries, news, weather) plus an injected
`now`, and returns one bundle entry per UPCOMING game (commence_time strictly
after `now`). Live-input gathering (SPORTSDATA_API_KEY, Supabase,
schedules.parquet, ESPN) lives in main(), not here.
"""
import importlib.util
import pathlib
from datetime import datetime, timezone

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "desk_inputs.py"
_spec = importlib.util.spec_from_file_location("desk_inputs", _p)
desk_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(desk_inputs)

build_bundle = desk_inputs.build_bundle

NOW = datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc)

# -- fixture: three games -------------------------------------------------
# 401: Boone @ Ames -- upcoming, full model row + full market line
# 402: Dover @ Coralville -- ALREADY STARTED relative to NOW -> must be excluded
# 403: Coralville @ Ames -- upcoming, but NO model_rows entry at all -> still
#      appears in the bundle, with model/market fields all None.
GAMES = [
    {"game_pk": 401, "home_team": "Ames", "away_team": "Boone",
     "commence_time": "2026-09-10T18:00:00Z"},
    {"game_pk": 402, "home_team": "Coralville", "away_team": "Dover",
     "commence_time": "2026-09-01T18:00:00Z"},
    {"game_pk": 403, "home_team": "Ames", "away_team": "Coralville",
     "commence_time": "2026-09-12T18:00:00Z"},
]

MODEL_ROWS = [
    {"game_pk": 401, "home_win_prob": 0.65, "pred_home_score": 28.0,
     "pred_away_score": 24.0, "market_spread": -3.5, "market_total": 51.5},
    {"game_pk": 402, "home_win_prob": 0.55, "pred_home_score": 21.0,
     "pred_away_score": 17.0, "market_spread": -2.5, "market_total": 44.0},
    # 403 deliberately absent -- no model row yet for this game.
]

FORM_ROWS = {
    "Ames": {"record": "3-1", "last_n": ["W", "W", "L", "W"], "avg_margin": 6.5, "pace": 71.0},
    "Boone": {"record": "2-2", "last_n": ["L", "W", "L", "W"], "avg_margin": -1.0, "pace": 65.0},
    "Coralville": {"record": "1-3", "last_n": ["L", "L", "W", "L"], "avg_margin": -9.0, "pace": 60.0},
    # Dover deliberately absent -- game 402 is excluded anyway.
}

INJURIES = {
    "Ames": [{"player": "John Doe", "position": "QB", "status": "Questionable", "note": "Ankle - Limited"}],
    "Boone": [{"player": "Jane Roe", "position": "RB", "status": "Out", "note": None}],
    "Coralville": [],
}

NEWS = [
    {"headline": "Ames QB battle heats up", "teams": ["Ames"],
     "published": "2026-09-06T10:00:00Z", "summary": "..."},
    {"headline": "Coralville shakes up O-line", "teams": ["Coralville"],
     "published": "2026-09-05T10:00:00Z", "summary": "..."},
    {"headline": "Unrelated Dover story", "teams": ["Dover"],
     "published": "2026-09-04T10:00:00Z", "summary": "..."},
]

WEATHER = {
    401: {"temp": 70.0, "wind": 5.0, "precip": False, "dome": False},
    # 403 deliberately absent -- no forecast populated yet.
}


def _by_pk(bundle, pk):
    return next(e for e in bundle if e["game_pk"] == pk)


def test_only_upcoming_games_included():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, NEWS, WEATHER, NOW)
    pks = {e["game_pk"] for e in bundle}
    assert pks == {401, 403}


def test_full_entry_has_model_form_news_and_market_line():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, NEWS, WEATHER, NOW)
    entry = _by_pk(bundle, 401)

    assert entry["matchup"] == "Boone @ Ames"
    assert entry["commence_time"] == "2026-09-10T18:00:00Z"

    # pick-time market line
    assert entry["market_spread"] == -3.5
    assert entry["market_total"] == 51.5

    # model block
    assert entry["model"]["margin"] == 4.0
    assert entry["model"]["total"] == 52.0
    assert entry["model"]["win_prob"] == 0.65

    # recent-form block, both teams
    assert entry["form"]["home"] == FORM_ROWS["Ames"]
    assert entry["form"]["away"] == FORM_ROWS["Boone"]

    # news block: injuries for BOTH teams, weather, relevant headlines only
    assert entry["news"]["injuries"]["home"] == INJURIES["Ames"]
    assert entry["news"]["injuries"]["away"] == INJURIES["Boone"]
    assert entry["news"]["weather"] == WEATHER[401]
    headlines = {h["headline"] for h in entry["news"]["headlines"]}
    assert headlines == {"Ames QB battle heats up"}


def test_game_with_no_line_or_model_row_still_appears_with_none_fields():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, NEWS, WEATHER, NOW)
    entry = _by_pk(bundle, 403)

    assert entry["market_spread"] is None
    assert entry["market_total"] is None
    assert entry["model"]["margin"] is None
    assert entry["model"]["total"] is None
    assert entry["model"]["win_prob"] is None

    # form/news still populated from the team-keyed inputs even with no model row
    assert entry["form"]["home"] == FORM_ROWS["Ames"]
    assert entry["form"]["away"] == FORM_ROWS["Coralville"]
    assert entry["news"]["injuries"]["away"] == INJURIES["Coralville"]
    assert entry["news"]["weather"] is None
    # both Ames (home) and Coralville (away) headlines attach; Dover's does not
    headlines = {h["headline"] for h in entry["news"]["headlines"]}
    assert headlines == {"Ames QB battle heats up", "Coralville shakes up O-line"}
