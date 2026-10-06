"""mlb_results.actual_for_pick: boxscore payload -> a prop pick's realized stat (pure)."""
from sportsmodel.ingest.mlb_results import actual_for_pick

RES = {
    "home_runs": 4, "away_runs": 3,
    "batters": {660271: {"name": "Shohei Ohtani", "hits": 2, "total_bases": 5, "home_run": 1, "hrr": 4}},
    "pitchers": {808967: {"name": "Yoshinobu Yamamoto", "pitcher_ks": 7, "hits_allowed": 3, "outs_recorded": 18}},
}


def test_batter_and_pitcher_markets_resolve():
    assert actual_for_pick(RES, 660271, "total_bases") == 5.0
    assert actual_for_pick(RES, 808967, "pitcher_ks") == 7.0
    assert actual_for_pick(RES, 808967, "hits_allowed") == 3.0
    assert actual_for_pick(RES, 808967, "outs_recorded") == 18.0


def test_player_id_may_be_a_string_ev_prop_picks_stores_text():
    assert actual_for_pick(RES, "660271", "total_bases") == 5.0


def test_unresolvable_picks_are_none_not_zero():
    assert actual_for_pick(None, 660271, "total_bases") is None          # game not final
    assert actual_for_pick(RES, 1, "total_bases") is None                # never appeared (DNP): book voids it
    assert actual_for_pick(RES, 660271, "pitcher_ks") is None            # a batter has no pitcher stats
    assert actual_for_pick(RES, 660271, "rush_yds") is None              # unknown market
    assert actual_for_pick(RES, "not-an-id", "total_bases") is None
