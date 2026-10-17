import json
from datetime import UTC, datetime
from pathlib import Path

from sportsmodel.take.parse import parse_events, parse_prices, parse_scores

FIXTURES = Path(__file__).parent / "fixtures"
PULLED = datetime(2026, 11, 2, 17, 0, tzinfo=UTC)


def load(name):
    return json.loads((FIXTURES / name).read_text())


def test_parse_prices_sides_and_lines():
    rows = parse_prices(load("odds_nfl.json"), "nfl", PULLED)
    dk = {(r["market"], r["side"]): r for r in rows if r["book"] == "draftkings"}
    assert dk[("spread", "away")]["line"] == 3.5 and dk[("total", "over")]["line"] == 47.5
    assert all(r["league"] == "nfl" and r["pulled_at"] == PULLED for r in rows)


def test_unknown_outcome_name_dropped():
    rows = parse_prices(load("odds_nfl.json"), "nfl", PULLED)
    assert not [r for r in rows if r["book"] == "draftkings" and r["market"] == "ml" and r["side"] not in ("home", "away")]
    assert len([r for r in rows if r["book"] == "draftkings" and r["market"] == "ml"]) == 1


def test_soccer_draw_side():
    rows = parse_prices(load("odds_epl.json"), "epl", PULLED)
    assert {r["side"] for r in rows if r["book"] == "pinnacle"} == {"home", "draw", "away"}


def test_draw_dropped_outside_soccer():
    ev = load("odds_epl.json")
    rows = parse_prices(ev, "nfl", PULLED)
    assert "draw" not in {r["side"] for r in rows}


def test_row_shape_and_last_update():
    rows = parse_prices(load("odds_nfl.json"), "nfl", PULLED)
    pin_ml = [r for r in rows if r["book"] == "pinnacle" and r["market"] == "ml" and r["side"] == "home"][0]
    assert pin_ml == {
        "event_id": "evt_nfl_1", "league": "nfl", "market": "ml", "side": "home", "book": "pinnacle",
        "line": None, "price": -160, "last_update": datetime(2026, 11, 2, 16, 55, tzinfo=UTC), "pulled_at": PULLED,
    }
    pin_spread = {r["side"]: r["line"] for r in rows if r["book"] == "pinnacle" and r["market"] == "spread"}
    assert pin_spread == {"home": -3.0, "away": 3.0}


def test_unknown_market_ignored():
    ev = load("odds_nfl.json")
    ev[0]["bookmakers"][0]["markets"].append(
        {"key": "h2h_lay", "last_update": "2026-11-02T16:55:00Z", "outcomes": [{"name": "Over", "price": 100}]})
    assert len(parse_prices(ev, "nfl", PULLED)) == len(parse_prices(load("odds_nfl.json"), "nfl", PULLED))


def test_parse_events_and_scores():
    ev = parse_events(load("odds_nfl.json"), "nfl")[0]
    assert ev["commence_time"] == datetime(2026, 11, 2, 18, tzinfo=UTC)
    assert ev == {"event_id": "evt_nfl_1", "league": "nfl", "home_team": "Kansas City Chiefs",
                  "away_team": "Buffalo Bills", "commence_time": datetime(2026, 11, 2, 18, tzinfo=UTC)}
    assert list(parse_scores(load("scores_nfl.json")).values()) == [(27, 24)]


def test_parse_scores_maps_by_team_name_and_skips_incomplete():
    scores = load("scores_nfl.json")
    assert parse_scores(scores) == {"evt_nfl_1": (27, 24)}      # away listed first in the fixture
    scores[0]["scores"] = None
    assert parse_scores(scores) == {}


def test_fetchers_use_expected_paths_and_params(monkeypatch):
    from sportsmodel.ingest import odds
    from sportsmodel.take.constants import league
    calls = []
    monkeypatch.setattr(odds, "_get_once", lambda path, params: calls.append(("once", path, params)) or [])
    monkeypatch.setattr(odds, "_get", lambda path, params: calls.append(("get", path, params)) or [])
    odds.fetch_scores(league("nfl"))
    odds.fetch_sports()
    assert calls == [
        ("once", "/sports/americanfootball_nfl/scores", {"daysFrom": 3, "dateFormat": "iso"}),
        ("get", "/sports", {}),
    ]
