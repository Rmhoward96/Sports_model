import pytest

from sportsmodel.model import odds


def test_american_to_prob_minus_110():
    # -110 implies 110/210 = 0.5238 (the classic breakeven).
    assert odds.american_to_prob(-110) == pytest.approx(110 / 210, abs=1e-6)


def test_american_to_prob_plus_money():
    assert odds.american_to_prob(150) == pytest.approx(100 / 250)


def test_remove_vig_two_way_sums_to_one():
    a, b = odds.no_vig_two_way(-110, -110)
    assert a + b == pytest.approx(1.0)
    assert a == pytest.approx(0.5)


def test_prob_to_american_roundtrip():
    for p in (0.25, 0.4, 0.5, 0.62, 0.8):
        line = odds.prob_to_american(p)
        assert odds.american_to_prob(line) == pytest.approx(p, abs=0.01)


def test_fetch_functions_use_sport_config(monkeypatch):
    from sportsmodel.ingest import odds
    from sportsmodel.sports import get
    calls = {}

    def _fake_get(path, params):
        calls["last"] = (path, params)
        return []

    monkeypatch.setattr(odds, "_get", _fake_get)
    odds.fetch_game_odds(get("nfl"))
    assert calls["last"][0] == "/sports/americanfootball_nfl/odds"
    odds.fetch_game_odds()  # default MLB
    assert calls["last"][0] == "/sports/baseball_mlb/odds"


def _event_props_payload(market_key: str, outcome_name: str, player: str):
    return {
        "commence_time": "2026-09-11T00:20:00Z",
        "bookmakers": [{
            "key": "draftkings",
            "markets": [{
                "key": market_key,
                "outcomes": [
                    {"name": outcome_name, "description": player, "point": 275.5, "price": -110},
                ],
            }],
        }],
    }


def test_parse_prop_odds_maps_nfl_market_keys_with_cfg():
    # NFL odds-api keys (player_pass_yds/player_anytime_td) aren't in MLB's
    # PROP_MARKET_MAP -- without cfg they'd resolve to None and be dropped,
    # silently zeroing out every NFL prop board row.
    from sportsmodel.ingest import odds
    from sportsmodel.sports import get

    nfl_cfg = get("nfl")
    payload = _event_props_payload("player_pass_yds", "Over", "Patrick Mahomes")
    rows = odds.parse_prop_odds(payload, 401, "2026-09-10T12:00:00Z", cfg=nfl_cfg)
    assert len(rows) == 1
    assert rows[0]["market"] == "pass_yds"
    assert rows[0]["player_name"] == "Patrick Mahomes"

    payload2 = _event_props_payload("player_anytime_td", "Yes", "Travis Kelce")
    rows2 = odds.parse_prop_odds(payload2, 401, "2026-09-10T12:00:00Z", cfg=nfl_cfg)
    assert len(rows2) == 1
    assert rows2[0]["market"] == "anytime_td"


def test_parse_prop_odds_default_cfg_none_keeps_mlb_behavior_unchanged():
    from sportsmodel.ingest import odds

    payload = _event_props_payload("batter_hits", "Over", "Aaron Judge")
    rows = odds.parse_prop_odds(payload, 1001, "2026-06-01T12:00:00Z")
    assert len(rows) == 1
    assert rows[0]["market"] == "hits"
    assert rows[0]["player_name"] == "Aaron Judge"

    # an NFL-only key must still be dropped on the default (MLB) path
    nfl_payload = _event_props_payload("player_pass_yds", "Over", "Patrick Mahomes")
    rows2 = odds.parse_prop_odds(nfl_payload, 1001, "2026-06-01T12:00:00Z")
    assert rows2 == []
