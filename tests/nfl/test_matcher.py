from sportsmodel.nfl.matcher import match_odds_event

ESPN = [
    {"game_pk": 401671789, "home_name": "Kansas City Chiefs",
     "away_name": "Baltimore Ravens", "commence_time": "2024-09-06T00:20Z"},
    {"game_pk": 401671790, "home_name": "Washington Commanders",
     "away_name": "Los Angeles Rams", "commence_time": "2024-09-08T17:00Z"},
]

def test_matches_by_names_and_date():
    ev = {"home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
          "commence_time": "2024-09-06T00:20:00Z"}
    assert match_odds_event(ev, ESPN) == 401671789

def test_relocated_team_name():
    ev = {"home_team": "Washington Commanders", "away_team": "Los Angeles Rams",
          "commence_time": "2024-09-08T17:05:00Z"}   # slightly different minute
    assert match_odds_event(ev, ESPN) == 401671790

def test_no_match_returns_none():
    ev = {"home_team": "Dallas Cowboys", "away_team": "New York Giants",
          "commence_time": "2024-09-06T00:20:00Z"}
    assert match_odds_event(ev, ESPN) is None

def test_match_over_parse_schedule_output():
    import json, pathlib
    from sportsmodel.nfl.espn import parse_schedule
    fix = json.loads((pathlib.Path(__file__).parent.parent
                      / "fixtures/nfl/espn_scoreboard.json").read_text())
    espn_games = parse_schedule(fix)
    ev = {"home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
          "commence_time": "2024-09-06T00:20:00Z"}
    assert match_odds_event(ev, espn_games) == 401671789
