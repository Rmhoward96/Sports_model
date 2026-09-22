"""Pure-parser tests for the Action Network (Apify) betting-splits adapter.
No network; a mock dataset item matching the actor's documented output shape."""
from sportsmodel.nfl import action_network as an


def _mock_item():
    return {
        "gameId": 12345,
        "league": "nfl",
        "startTime": "2026-09-22T00:15:00.000Z",
        "homeTeam": {"abbreviation": "LA"},
        "awayTeam": {"abbreviation": "NYG"},
        "consensus": {
            "spread": {"sides": [
                {"side": "home", "ticketPercent": 46, "moneyPercent": 35, "sharpGap": -11},
                {"side": "away", "ticketPercent": 54, "moneyPercent": 65, "sharpGap": 11},
            ]},
            "total": {"sides": [
                {"side": "over", "ticketPercent": 60, "moneyPercent": 52},
                {"side": "under", "ticketPercent": 40, "moneyPercent": 48},
            ]},
            "moneyline": {"sides": [
                {"side": "home", "ticketPercent": 70, "moneyPercent": 55},
                {"side": "away", "ticketPercent": 30, "moneyPercent": 45},
            ]},
        },
    }


def test_parse_maps_markets_sides_and_percentages():
    rows = an.parse_action_network_splits([_mock_item()])
    by = {(r["market"], r["side"]): r for r in rows}
    assert len(rows) == 6  # 2 spread + 2 total + 2 moneyline
    sp_home = by[("spread", "home")]
    assert sp_home["cash_pct"] == 35 and sp_home["ticket_pct"] == 46   # money=cash, ticket=ticket
    assert sp_home["home_abbr"] == "LA" and sp_home["away_abbr"] == "NYG"
    assert sp_home["start_time"] == "2026-09-22T00:15:00.000Z"
    assert by[("total", "over")]["cash_pct"] == 52
    assert set(by) == {("spread", "home"), ("spread", "away"),
                       ("total", "over"), ("total", "under"),
                       ("moneyline", "home"), ("moneyline", "away")}


def test_parse_normalizes_team_abbreviations():
    item = _mock_item()
    item["homeTeam"]["abbreviation"] = "WSH"   # alias -> WAS
    item["awayTeam"]["abbreviation"] = "JAC"   # alias -> JAX
    rows = an.parse_action_network_splits([item])
    assert rows[0]["home_abbr"] == "WAS" and rows[0]["away_abbr"] == "JAX"


def test_parse_skips_unknown_markets_and_empty_sides():
    item = _mock_item()
    item["consensus"]["firstHalf"] = {"sides": [{"side": "home", "ticketPercent": 50, "moneyPercent": 50}]}
    item["consensus"]["spread"]["sides"].append(
        {"side": "home", "ticketPercent": None, "moneyPercent": None})  # both None -> skipped
    rows = an.parse_action_network_splits([item])
    assert not any(r["market"] == "firstHalf" for r in rows)   # unknown market dropped
    assert len([r for r in rows if r["market"] == "spread"]) == 2  # the empty side dropped


def test_parse_empty_input():
    assert an.parse_action_network_splits([]) == []
    assert an.parse_action_network_splits(None) == []


def test_attach_game_pks_matches_on_team_and_utc_date():
    rows = an.parse_action_network_splits([_mock_item()])  # LA vs NYG, 2026-09-22
    index = {("LA", "NYG", "2026-09-22"): 401772000}
    out = an.attach_game_pks(rows, index)
    assert len(out) == len(rows)
    assert all(r["game_pk"] == 401772000 for r in out)


def test_attach_game_pks_drops_unmatched_games():
    rows = an.parse_action_network_splits([_mock_item()])
    # wrong date -> no slate game -> every row dropped
    assert an.attach_game_pks(rows, {("LA", "NYG", "2026-09-23"): 1}) == []
    # empty index -> nothing matches
    assert an.attach_game_pks(rows, {}) == []


def _cfb_item():
    return {
        "startTime": "2026-10-03T16:00:00.000Z",
        "homeTeam": {"abbreviation": "UGA", "displayName": "Georgia Bulldogs"},
        "awayTeam": {"abbreviation": "BAMA", "displayName": "Alabama Crimson Tide"},
        "consensus": {"spread": {"sides": [
            {"side": "home", "ticketPercent": 58, "moneyPercent": 62},
            {"side": "away", "ticketPercent": 42, "moneyPercent": 38},
        ]}},
    }


def test_parse_emits_team_names():
    rows = an.parse_action_network_splits([_cfb_item()])
    assert rows and rows[0]["home_name"] == "Georgia Bulldogs"
    assert rows[0]["away_name"] == "Alabama Crimson Tide"


def test_attach_game_pks_by_name_matches_on_normalized_name_and_date():
    rows = an.parse_action_network_splits([_cfb_item()])
    idx = {("georgia bulldogs", "alabama crimson tide", "2026-10-03"): 401800001}
    out = an.attach_game_pks_by_name(rows, idx)
    assert out and all(r["game_pk"] == 401800001 for r in out)
    # wrong date / empty -> dropped
    assert an.attach_game_pks_by_name(rows, {("georgia bulldogs", "alabama crimson tide", "2026-10-04"): 1}) == []
    assert an.attach_game_pks_by_name(rows, {}) == []
