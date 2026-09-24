"""Pure-assembly tests for scripts/desk_inputs.py's build_bundle.

No network/DB/file access: build_bundle takes already-fetched/parsed inputs
(games, model_rows, form_rows, injuries, weather) plus an injected `now`,
and returns one bundle entry per UPCOMING game (commence_time strictly after
`now`). Live-input gathering (SPORTSDATA_API_KEY, Supabase,
schedules.parquet, ESPN) lives in main(), not here. There is no news
endpoint in SportsDataIO's CFB API, so build_bundle no longer takes a `news`
argument or produces a `headlines` field.
"""
import importlib.util
import pathlib
from datetime import datetime, timezone

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "desk_inputs.py"
_spec = importlib.util.spec_from_file_location("desk_inputs", _p)
desk_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(desk_inputs)

build_bundle = desk_inputs.build_bundle
trend_block = desk_inputs.trend_block
game_trends = desk_inputs.game_trends

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

WEATHER = {
    401: {"temp": 70.0, "wind": 5.0, "precip": False, "dome": False},
    # 403 deliberately absent -- no forecast populated yet.
}


def _by_pk(bundle, pk):
    return next(e for e in bundle if e["game_pk"] == pk)


def test_only_upcoming_games_included():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW)
    pks = {e["game_pk"] for e in bundle}
    assert pks == {401, 403}


def test_full_entry_has_model_form_news_and_market_line():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW)
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

    # news block: injuries for BOTH teams, weather
    assert entry["news"]["injuries"]["home"] == INJURIES["Ames"]
    assert entry["news"]["injuries"]["away"] == INJURIES["Boone"]
    assert entry["news"]["weather"] == WEATHER[401]


def test_game_with_no_line_or_model_row_still_appears_with_none_fields():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW)
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


def test_trend_block_formats_records_and_situational():
    records = {"ats": {"w": 2, "l": 1, "p": 0}, "ats_road": {"w": 1, "l": 0, "p": 0},
               "ats_dog": {"w": 0, "l": 0, "p": 0}, "ats_last_5": {"w": 2, "l": 1, "p": 0},
               "over_under": {"o": 2, "u": 1, "p": 0}, "over_under_road": {"o": 1, "u": 0, "p": 0},
               "units": {"w": 1.2, "l": 0}}
    sit = [{"label": "off a road game", "ats_w": 7, "ats_l": 2, "ats_p": 0, "ou_o": 4, "ou_u": 5, "ou_p": 0, "since_season": 2023}]
    b = trend_block(records, sit, is_home=False, is_fav=False)
    assert b["records"]["ATS"] == "2-1-0" and b["records"]["ATS on the road"] == "1-0-0"
    assert b["records"]["ATS as underdog"] == "0-0-0" and b["records"]["ATS last 5"] == "2-1-0"
    assert b["records"]["O/U"] == "2-1-0" and b["records"]["O/U on the road"] == "1-0-0"
    assert b["records"]["Units"] == "+1.2"
    assert b["situational"] == ["7-2-0 ATS, O/U 4-5-0 off a road game since 2023"]


def test_trend_block_none_when_no_data():
    assert trend_block(None, [], is_home=True, is_fav=True) is None


def test_build_bundle_attaches_trends_when_given():
    trends = {GAMES[0]["game_pk"]: {"home": {"records": {"ATS": "1-0-0"}, "situational": []}, "away": None}}
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW, trends=trends)
    assert bundle[0]["trends"]["home"]["records"]["ATS"] == "1-0-0"
    # default: no trends arg -> key present with both sides None
    assert build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW)[0]["trends"] == {"home": None, "away": None}


def test_trend_block_is_home_true_uses_at_home_labels():
    records = {"ats": {"w": 2, "l": 1, "p": 0}, "ats_home": {"w": 1, "l": 0, "p": 0},
               "over_under": {"o": 2, "u": 1, "p": 0}, "over_under_home": {"o": 1, "u": 0, "p": 0}}
    b = trend_block(records, [], is_home=True, is_fav=None)
    assert "ATS at home" in b["records"]
    assert "O/U at home" in b["records"]
    assert "ATS on the road" not in b["records"]


def test_trend_block_is_fav_none_omits_role_keys():
    records = {"ats": {"w": 2, "l": 1, "p": 0}, "ats_fav": {"w": 1, "l": 1, "p": 0},
               "ats_dog": {"w": 1, "l": 0, "p": 0}}
    b = trend_block(records, [], is_home=True, is_fav=None)
    assert "ATS as favorite" not in b["records"]
    assert "ATS as underdog" not in b["records"]
    assert b["records"]["ATS"] == "2-1-0"


def test_trend_block_is_fav_true_includes_favorite_key():
    records = {"ats": {"w": 2, "l": 1, "p": 0}, "ats_fav": {"w": 3, "l": 0, "p": 0}}
    b = trend_block(records, [], is_home=True, is_fav=True)
    assert b["records"]["ATS as favorite"] == "3-0-0"


def test_trend_block_situational_only_records_none():
    sit = [{"label": "primetime", "ats_w": 5, "ats_l": 2, "ats_p": 0, "ou_o": 3, "ou_u": 4, "ou_p": 0, "since_season": 2024}]
    b = trend_block(None, sit, is_home=True, is_fav=None)
    assert b["records"] == {}
    assert b["situational"] == ["5-2-0 ATS, O/U 3-4-0 primetime since 2024"]


def test_game_trends_negative_spread_home_favored():
    records = {
        "Home Team": {"ats": {"w": 2, "l": 1, "p": 0}, "ats_fav": {"w": 2, "l": 0, "p": 0}},
        "Away Team": {"ats": {"w": 1, "l": 2, "p": 0}, "ats_dog": {"w": 1, "l": 2, "p": 0}},
    }
    sit = {}
    result = game_trends("Home Team", "Away Team", -3.5, records, sit, 101)
    # Home is favored: home has is_fav=True, away has is_fav=False
    assert "ATS as favorite" in result["home"]["records"]
    assert "ATS as underdog" in result["away"]["records"]


def test_game_trends_positive_spread_away_favored():
    records = {
        "Home Team": {"ats": {"w": 1, "l": 2, "p": 0}, "ats_dog": {"w": 0, "l": 2, "p": 0}},
        "Away Team": {"ats": {"w": 3, "l": 0, "p": 0}, "ats_fav": {"w": 3, "l": 0, "p": 0}},
    }
    sit = {}
    result = game_trends("Home Team", "Away Team", 2.5, records, sit, 102)
    # Away is favored: home has is_fav=False, away has is_fav=True
    assert "ATS as underdog" in result["home"]["records"]
    assert "ATS as favorite" in result["away"]["records"]


def test_game_trends_zero_spread_pickem_no_role_trends():
    records = {
        "Home Team": {"ats": {"w": 2, "l": 1, "p": 0}, "ats_fav": {"w": 1, "l": 0, "p": 0}, "ats_dog": {"w": 1, "l": 1, "p": 0}},
        "Away Team": {"ats": {"w": 2, "l": 2, "p": 0}, "ats_fav": {"w": 1, "l": 1, "p": 0}, "ats_dog": {"w": 1, "l": 1, "p": 0}},
    }
    sit = {}
    result = game_trends("Home Team", "Away Team", 0.0, records, sit, 103)
    # Pick'em: both teams have is_fav=None (no role trends)
    assert "ATS as favorite" not in result["home"]["records"]
    assert "ATS as underdog" not in result["home"]["records"]
    assert "ATS as favorite" not in result["away"]["records"]
    assert "ATS as underdog" not in result["away"]["records"]
    assert result["home"]["records"]["ATS"] == "2-1-0"
    assert result["away"]["records"]["ATS"] == "2-2-0"


def test_game_trends_none_spread_no_role_trends():
    records = {
        "Home Team": {"ats": {"w": 2, "l": 1, "p": 0}},
        "Away Team": {"ats": {"w": 1, "l": 2, "p": 0}},
    }
    sit = {}
    result = game_trends("Home Team", "Away Team", None, records, sit, 104)
    # No spread: both teams have is_fav=None (no role trends)
    assert "ATS as favorite" not in (result["home"]["records"] if result["home"] else {})
    assert "ATS as underdog" not in (result["away"]["records"] if result["away"] else {})


def test_trend_block_units_losses_stored_negative():
    # Action Network stores units lost as a NEGATIVE number (live 2026-09-23:
    # Falcons 0-2 SU -> {"w": 0, "l": -2}; Browns won once as a big dog ->
    # {"w": 3.3, "l": -1}). Net must be w - |l|, not w - l.
    assert trend_block({"units": {"w": 0, "l": -2}}, [], is_home=False, is_fav=None)["records"]["Units"] == "-2.0"
    assert trend_block({"units": {"w": 3.3, "l": -1}}, [], is_home=False, is_fav=None)["records"]["Units"] == "+2.3"
    assert trend_block({"units": {"w": 1.2, "l": 0}}, [], is_home=False, is_fav=None)["records"]["Units"] == "+1.2"
    # Robust if a positive loss magnitude is ever sent.
    assert trend_block({"units": {"w": 0, "l": 2}}, [], is_home=False, is_fav=None)["records"]["Units"] == "-2.0"


# -- injury freshness block (news.injury_report) ------------------------------

INJURY_META = {
    "source": "nflverse+espn", "stale": True, "report_week": 2, "target_week": 3,
    "espn_available": True,
    "conflicts": [
        {"team": "Ames", "player": "John Doe", "nflverse": "Out", "espn": "Questionable"},
        {"team": "Boone", "player": "Jane Roe", "nflverse": None, "espn": "Out"},
        {"team": "Elsewhere", "player": "Not Here", "nflverse": "Out", "espn": None},
    ],
}


def test_build_bundle_injury_report_default_none():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW)
    assert all(g["news"]["injury_report"] is None for g in bundle)


def test_build_bundle_injury_report_conflicts_filtered_to_game_teams():
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW,
                          trends=None, injury_meta=INJURY_META)
    g401 = _by_pk(bundle, 401)["news"]["injury_report"]  # Boone @ Ames
    assert g401["source"] == "nflverse+espn"
    assert g401["espn_available"] is True
    assert g401["stale"] is True
    assert g401["report_week"] == 2 and g401["target_week"] == 3
    assert [c["player"] for c in g401["conflicts"]] == ["John Doe", "Jane Roe"]
    g403 = _by_pk(bundle, 403)["news"]["injury_report"]  # Coralville @ Ames
    assert [c["player"] for c in g403["conflicts"]] == ["John Doe"]


def test_build_bundle_injury_report_cfb_meta_no_week_keys():
    meta = {"source": "sportsdata", "stale": False, "conflicts": []}
    g = _by_pk(build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW,
                            injury_meta=meta), 401)["news"]["injury_report"]
    assert g == {"source": "sportsdata", "stale": False, "espn_available": None,
                 "report_week": None, "target_week": None, "conflicts": []}


# -- injury sources return (by_name, meta) ------------------------------------

def test_build_bundle_injury_report_espn_down_is_nflverse_only():
    meta = dict(INJURY_META, espn_available=False, stale=True, conflicts=[])
    g = _by_pk(build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW,
                            injury_meta=meta), 401)["news"]["injury_report"]
    assert g["espn_available"] is False
    assert g["source"] == "nflverse"


def test_nfl_injuries_by_name_uses_shared_report(monkeypatch):
    calls = {}

    def fake_report(now, target_week, name_to_abbr):
        calls["args"] = (now, target_week, name_to_abbr)
        return {
            "by_team": {
                "ATL": [{"player": "Michael Penix Jr.", "position": "QB", "status": "Questionable",
                         "note": None, "source": "espn"}],
                "ZZZ": [{"player": "Nobody", "position": "QB", "status": "Out", "note": None, "source": "espn"}],
            },
            "stale": True, "report_week": 2, "target_week": 3, "espn_available": True,
            "conflicts": [{"team": "ATL", "player": "Michael Penix Jr.", "nflverse": "Out", "espn": "Questionable"}],
        }

    monkeypatch.setattr(desk_inputs.injury_report, "current_report", fake_report)
    monkeypatch.setattr(desk_inputs.injury_report, "resolve_target_week", lambda now: 3)
    crosswalk = {"ATL": "Atlanta Falcons", "GB": "Green Bay Packers"}
    by_name, meta = desk_inputs._nfl_injuries_by_name(None, None, crosswalk, NOW)

    assert calls["args"] == (NOW, 3, {"Atlanta Falcons": "ATL", "Green Bay Packers": "GB"})
    assert list(by_name) == ["Atlanta Falcons"]  # unknown abbr dropped
    assert by_name["Atlanta Falcons"][0]["status"] == "Questionable"
    assert "by_team" not in meta
    assert meta["source"] == "nflverse+espn"
    assert meta["stale"] is True and meta["report_week"] == 2 and meta["target_week"] == 3
    assert meta["conflicts"] == [{"team": "Atlanta Falcons", "player": "Michael Penix Jr.",
                                  "nflverse": "Out", "espn": "Questionable"}]


def test_nfl_injuries_by_name_target_week_unresolved_is_none(monkeypatch):
    seen = {}

    def fake_report(now, target_week, name_to_abbr):
        seen["tw"] = target_week
        return {"by_team": {}, "stale": False, "report_week": 3, "target_week": None,
                "espn_available": True, "conflicts": []}

    monkeypatch.setattr(desk_inputs.injury_report, "current_report", fake_report)
    monkeypatch.setattr(desk_inputs.injury_report, "resolve_target_week", lambda now: None)
    by_name, meta = desk_inputs._nfl_injuries_by_name(None, None, {"ATL": "Atlanta Falcons"}, NOW)
    assert seen["tw"] is None
    assert by_name == {} and meta["stale"] is False


def test_cfb_injuries_by_name_returns_live_meta():
    class FakeAdapter:
        INJURED_PLAYERS_PATH = "inj"
        TEAMS_PATH = "teams"

        @staticmethod
        def _get(path, key):
            return path

        @staticmethod
        def parse_injuries(payload):
            return {"AMES": [{"player": "John Doe", "status": "Out"}], "UNK": []}

        @staticmethod
        def parse_teams(payload):
            return {"AMES": "Ames"}

    by_name, meta = desk_inputs._cfb_injuries_by_name(FakeAdapter, "k", {}, NOW)
    assert by_name == {"Ames": [{"player": "John Doe", "status": "Out"}]}
    assert meta == {"source": "sportsdata", "stale": False, "conflicts": []}
