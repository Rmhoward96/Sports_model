from sportsmodel.sports import get, SPORTS

def test_mlb_config_matches_legacy_constants():
    from sportsmodel.ingest import odds
    m = get("mlb")
    assert m.odds_sport == "baseball_mlb"
    assert m.game_markets == ["h2h", "totals", "spreads"]
    assert m.prop_market_map == odds.PROP_MARKET_MAP
    assert m.commence_shift_hours == 10

def test_nfl_config_present_with_seven_prop_markets():
    n = get("nfl")
    assert n.odds_sport == "americanfootball_nfl"
    assert set(n.prop_market_map.values()) == {
        "player_pass_yds", "player_pass_tds", "player_reception_yds",
        "player_receptions", "player_rush_yds", "player_rush_reception_yds",
        "player_anytime_td"}

def test_unknown_sport_raises():
    import pytest
    with pytest.raises(KeyError):
        get("cricket")
