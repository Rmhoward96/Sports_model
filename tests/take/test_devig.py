import pytest

from sportsmodel.take.constants import BOOKMAKERS, LEAGUES, budget_ok
from sportsmodel.take.devig import decimal_to_american_safe, novig_power_n


def test_two_way_matches_board():
    from sportsmodel.serving.board import novig_power
    for a, b in [(-110, -110), (-150, 130), (-300, 240), (105, -125)]:
        p = novig_power_n([a, b])
        assert abs(p[0] - novig_power(a, b)) < 1e-9 and abs(sum(p) - 1) < 1e-9


def test_three_way_sums_to_one_and_keeps_order():
    p = novig_power_n([150, 230, 190])          # home, draw, away
    assert abs(sum(p) - 1) < 1e-9 and p[0] > p[2] > p[1]


def test_no_vig_market_is_proportional():
    p = novig_power_n([100, 100])
    assert p == pytest.approx([0.5, 0.5])


def test_decimal_to_american_safe():
    assert decimal_to_american_safe(1.015 / 0.5) == 103
    assert decimal_to_american_safe(1.015 / 0.6) == -144
    assert decimal_to_american_safe(1.015 / 0.25) == 306
    assert decimal_to_american_safe(2.0) == 100


def test_budget_ok():
    assert budget_ok(None, 1500) == (True, None)
    assert budget_ok("1499", 1500)[0] is False
    assert budget_ok("1500", 1500) == (True, None)


def test_league_keys():
    assert LEAGUES["mls"] == "soccer_usa_mls" and len(LEAGUES) == 8
    assert "bet365" not in BOOKMAKERS and len(BOOKMAKERS) == 8
