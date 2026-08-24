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
