"""polling.py and ingest.odds.fetch_bookmaker_odds: the bookmakers list, cost per poll and the credit floor. No network."""
import pytest

from sportsmodel import sports
from sportsmodel.early import polling
from sportsmodel.early.constants import API_MARKETS, BOOKMAKERS, MONITOR_FLOOR, US_BOOKS
from sportsmodel.ingest import odds


def test_the_ten_bookmaker_keys():
    assert BOOKMAKERS == ("draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "hardrockbet",
                          "ballybet", "betrivers", "pinnacle")
    assert len(BOOKMAKERS) == 10 == len(set(BOOKMAKERS)) and set(BOOKMAKERS) - set(US_BOOKS) == {"pinnacle"}
    assert polling.bookmakers_param() == ",".join(BOOKMAKERS)


def test_cost_per_poll_is_one_credit_per_market_because_ten_books_bill_as_one_region():
    assert polling.regions_billed() == 1 and polling.regions_billed(11) == 2 and polling.regions_billed(1) == 1
    assert polling.api_markets("cfb") == ["spreads", "totals"] and polling.poll_cost("cfb") == 2
    assert polling.api_markets("nfl") == ["spreads"] and polling.poll_cost("nfl") == 1
    assert polling.poll_cost("cfb") + polling.poll_cost("nfl") == 3           # about 3 credits a poll
    assert 60 * 3 == 180                                                       # 60 polls a week: about 180 credits


def test_the_credit_floor_boundary_is_1500():
    assert MONITOR_FLOOR == 1500
    assert polling.budget_decision(1500).ok is True and polling.budget_decision("1500").ok is True
    low = polling.budget_decision(1499)
    assert low.ok is False and "1499" in low.warning and "1500" in low.warning
    assert polling.budget_decision("1499.0").ok is False and polling.budget_decision(0).ok is False


@pytest.mark.parametrize("unknown", [None, "", "n/a", float("nan"), []])
def test_an_unknown_remaining_value_runs_the_first_call(unknown):
    assert polling.budget_decision(unknown) == polling.BudgetDecision(True)


def test_fetch_bookmaker_odds_sends_bookmakers_and_never_regions(monkeypatch):
    seen = {}
    monkeypatch.setattr(odds, "_get_once", lambda path, params: seen.update(path=path, params=params) or [])
    assert odds.fetch_bookmaker_odds(sports.get("cfb"), list(BOOKMAKERS), ["spreads", "totals"]) == []
    assert seen["path"] == "/sports/americanfootball_ncaaf/odds"
    assert seen["params"] == {"bookmakers": polling.bookmakers_param(), "markets": "spreads,totals",
                              "oddsFormat": "american", "dateFormat": "iso"}
    assert "regions" not in seen["params"]
    odds.fetch_bookmaker_odds(sports.get("nfl"), list(BOOKMAKERS), polling.api_markets("nfl"))
    assert seen["path"] == "/sports/americanfootball_nfl/odds" and seen["params"]["markets"] == "spreads"


def test_every_polled_sport_has_a_market_list_and_a_sport_config():
    for sport in API_MARKETS:
        assert sports.get(sport).odds_sport
