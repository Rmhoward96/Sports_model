from sportsmodel.sports import get, SPORTS
from sportsmodel.ingest.odds import GAME_MARKETS

def test_mlb_config_matches_legacy_constants():
    from sportsmodel.ingest import odds
    m = get("mlb")
    assert m.odds_sport == "baseball_mlb"
    assert m.game_markets == ["h2h", "totals", "spreads"]
    # The ingester requests only the LIVE props (one credit per market per event); the full
    # PROP_MARKET_MAP stays the parse vocabulary.
    assert m.prop_market_map == odds.LIVE_PROP_MARKET_MAP
    assert m.commence_shift_hours == 10


def test_mlb_live_props_keep_hits_hrr_and_home_run_dropped():
    # Commit 206941b: hits (-32U) and hrr (-35U) were the two worst markets over ~1,970 live
    # graded picks; home_run was never published. Reactivation must not re-buy their credits.
    from sportsmodel.ingest import odds
    assert set(get("mlb").prop_market_map) == {"total_bases", "pitcher_ks", "hits_allowed", "outs_recorded"}
    assert set(get("mlb").prop_market_map.values()) == {
        "batter_total_bases", "pitcher_strikeouts", "pitcher_hits_allowed", "pitcher_outs"}
    for dropped in ("hits", "hrr", "home_run"):
        assert dropped not in odds.LIVE_PROP_MARKET_MAP
        assert dropped in odds.PROP_MARKET_MAP  # still parseable if a stray row arrives

def test_nfl_config_present_with_eight_prop_markets():
    n = get("nfl")
    assert n.odds_sport == "americanfootball_nfl"
    assert set(n.prop_market_map.values()) == {
        "player_pass_yds", "player_pass_tds", "player_reception_yds",
        "player_receptions", "player_rush_yds", "player_rush_reception_yds",
        "player_rush_attempts", "player_anytime_td"}

def test_cfb_config_present():
    c = get("cfb")
    assert c.odds_sport == "americanfootball_ncaaf"
    assert c.game_markets == GAME_MARKETS
    assert c.commence_shift_hours == 8

def test_unknown_sport_raises():
    import pytest
    with pytest.raises(KeyError):
        get("cricket")
