from sportsmodel.cfb.matcher import match_odds_event

ESPN = [
    {"game_pk": 401628318, "home_name": "Alabama Crimson Tide",
     "away_name": "Georgia Bulldogs", "commence_time": "2024-09-28T19:30Z"},
    {"game_pk": 401628319, "home_name": "Ohio State Buckeyes",
     "away_name": "Michigan Wolverines", "commence_time": "2024-11-30T17:00Z"},
]

def test_matches_by_names_and_date():
    ev = {"home_team": "Alabama Crimson Tide", "away_team": "Georgia Bulldogs",
          "commence_time": "2024-09-28T19:30:00Z"}
    assert match_odds_event(ev, ESPN) == 401628318

def test_matches_with_slightly_different_time():
    ev = {"home_team": "Ohio State Buckeyes", "away_team": "Michigan Wolverines",
          "commence_time": "2024-11-30T17:05:00Z"}   # slightly different minute
    assert match_odds_event(ev, ESPN) == 401628319

def test_no_match_returns_none():
    ev = {"home_team": "Clemson Tigers", "away_team": "Florida State Seminoles",
          "commence_time": "2024-09-28T19:30:00Z"}
    assert match_odds_event(ev, ESPN) is None
