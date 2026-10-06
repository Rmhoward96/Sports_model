"""MLB Odds-event -> StatsAPI game_pk matcher (network-free)."""
from sportsmodel.mlb.matcher import match_odds_event


def _game(pk, home="Los Angeles Dodgers", away="Atlanta Braves", day="2026-10-05"):
    return {"game_pk": pk, "home_name": home, "away_name": away, "game_date": day}


def _event(commence, home="Los Angeles Dodgers", away="Atlanta Braves"):
    return {"id": "e", "home_team": home, "away_team": away, "commence_time": commence}


def test_matches_on_teams_and_us_game_date():
    assert match_odds_event(_event("2026-10-05T23:00:00Z"), [_game(1)]) == 1


def test_night_game_after_utc_midnight_resolves_to_the_us_day():
    # 00:00Z on Oct 6 is the 8pm ET game of Oct 5: the schedule's officialDate is 10-05.
    assert match_odds_event(_event("2026-10-06T00:00:00Z"), [_game(2, day="2026-10-05")]) == 2
    assert match_odds_event(_event("2026-10-06T00:00:00Z"), [_game(3, day="2026-10-06")]) is None


def test_postseason_games_match_like_any_other():
    # Nothing keys on game type: the StatsAPI schedule feed carries F/D/L/W games unfiltered.
    games = [{**_game(849819, day="2026-10-06"), "game_type": "D"}]
    assert match_odds_event(_event("2026-10-06T22:00:00Z"), games) == 849819


def test_name_match_is_case_and_whitespace_insensitive_and_folds_oakland():
    g = _game(4, home="Athletics", away="Seattle Mariners")
    assert match_odds_event(_event("2026-10-05T23:00:00Z", home=" athletics ", away="SEATTLE MARINERS"), [g]) == 4
    assert match_odds_event(_event("2026-10-05T23:00:00Z", home="Oakland Athletics", away="Seattle Mariners"), [g]) == 4


def test_unknown_matchup_is_none():
    assert match_odds_event(_event("2026-10-05T23:00:00Z", home="Nope"), [_game(1)]) is None
    assert match_odds_event(_event("2026-10-05T23:00:00Z"), []) is None


def test_doubleheader_is_left_unmatched_not_misfiled():
    # Same teams, same day: the (home, away, date) lookup downstream cannot tell the games
    # apart, so neither event is matched rather than risk filing one game's prices under the other.
    games = [_game(10), _game(11)]
    assert match_odds_event(_event("2026-10-05T17:00:00Z"), games) is None
    assert match_odds_event(_event("2026-10-05T23:30:00Z"), games) is None


def test_missing_commence_time_is_none():
    assert match_odds_event({"home_team": "A", "away_team": "B"}, [_game(1)]) is None
