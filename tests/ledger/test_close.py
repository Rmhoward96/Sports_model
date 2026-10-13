"""Close-capture planning: window selection, budget guard, credit maths, prop-market planning. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel import sports
from sportsmodel.ledger import close

NOW = datetime(2026, 10, 18, 16, 40, tzinfo=timezone.utc)


def _g(pk, minutes, **kw):
    return {"game_pk": pk, "commence_time": (NOW + timedelta(minutes=minutes)).isoformat(), **kw}


def test_window_is_five_to_twenty_five_minutes_inclusive():
    games = [_g(1, 4.9), _g(2, 5), _g(3, 15), _g(4, 25), _g(5, 25.1), _g(6, -3), {"game_pk": 7, "commence_time": None}]
    assert [g["game_pk"] for g in close.select_targets(games, set(), "nfl", "lines", NOW)] == [2, 3, 4]


def test_the_window_closes_inside_the_gates_thirty_minute_bound():
    assert close.WINDOW_MAX <= 30 and close.WINDOW_MIN == 5


def test_already_captured_events_are_skipped_per_kind_and_sport():
    games = [_g(1, 12), _g(2, 12)]
    captured = {("nfl", 1, "lines"), ("cfb", 2, "lines"), ("nfl", 2, "props")}
    assert [g["game_pk"] for g in close.select_targets(games, captured, "nfl", "lines", NOW)] == [2]
    assert [g["game_pk"] for g in close.select_targets(games, captured, "nfl", "props", NOW)] == [1]


def test_targets_carry_their_minutes_to_kick():
    (t,) = close.select_targets([_g(1, 12)], set(), "nfl", "lines", NOW)
    assert t["minutes_to_kick"] == pytest.approx(12)


def test_prefilter_lists_only_sports_with_a_kickoff_in_the_next_hour_in_fixed_order():
    cands = [{"sport": "mlb", "game_pk": 1, "commence_time": NOW + timedelta(minutes=50)},
             {"sport": "nfl", "game_pk": 2, "commence_time": (NOW + timedelta(minutes=59)).isoformat()},
             {"sport": "cfb", "game_pk": 3, "commence_time": NOW + timedelta(minutes=61)},
             {"sport": "cfb", "game_pk": 4, "commence_time": NOW - timedelta(minutes=1)},
             {"sport": "cfb", "game_pk": 5, "commence_time": None}]
    assert close.prefilter_sports(cands, NOW) == ["nfl", "mlb"]
    assert close.prefilter_sports([], NOW) == []


def test_budget_floors_are_below_the_regular_ingest_floors():
    # the close capture is the highest-priority credit use: the regular ingest must stop first
    from sportsmodel.ingest import odds
    assert (close.BUDGET_PROPS_FLOOR, close.BUDGET_ALL_FLOOR) == (600, 200)
    assert close.BUDGET_ALL_FLOOR < close.BUDGET_PROPS_FLOOR < odds.INGEST_LINES_FLOOR < odds.INGEST_PROPS_FLOOR


@pytest.mark.parametrize("remaining, lines, props, warns", [
    (30000, True, True, False), (600, True, True, False), (599, True, False, True), (200, True, False, True),
    (199, False, False, True), (0, False, False, True), (None, True, True, True)])
def test_budget_thresholds(remaining, lines, props, warns):
    d = close.budget_decision(remaining)
    assert (d.lines, d.props, d.warning is not None) == (lines, props, warns)


def test_parse_credits_and_estimated_cost():
    assert close.parse_credits("19000") == 19000 and close.parse_credits("8.0") == 8
    assert close.parse_credits(None) is None and close.parse_credits("n/a") is None
    assert close.estimated_cost(3, "us,eu") == 6 and close.estimated_cost(4, "us") == 4


@pytest.mark.parametrize("total, n", [(6, 1), (6, 4), (7, 3), (0, 2), (5, 5)])
def test_split_credits_sums_exactly(total, n):
    parts = close.split_credits(total, n)
    assert len(parts) == n and sum(parts) == total and max(parts) - min(parts) <= 1
    assert close.split_credits(5, 0) == []


@pytest.mark.parametrize("env, expected", [({}, "us,eu"), ({"CLOSE_PROP_REGIONS": ""}, "us,eu"),
                                           ({"CLOSE_PROP_REGIONS": "us"}, "us"), ({"CLOSE_PROP_REGIONS": " US , EU "}, "us,eu"),
                                           ({"CLOSE_PROP_REGIONS": "uk"}, "us,eu")])
def test_prop_regions_default_and_override(env, expected):
    assert close.prop_regions(env) == expected


def test_api_markets_cover_only_the_markets_with_picks():
    nfl = sports.get("nfl").prop_market_map
    assert close.api_markets_for("nfl", {"rec_yds", "pass_tds"}, nfl) == ["player_pass_tds", "player_reception_yds"]
    assert close.api_markets_for("nfl", {"nonsense"}, nfl) == [] and close.api_markets_for("nfl", set(), nfl) == []
    mlb = sports.get("mlb").prop_market_map
    assert close.api_markets_for("mlb", {"total_bases", "pitcher_ks"}, mlb) == ["batter_total_bases", "pitcher_strikeouts"]
    assert close.api_markets_for("cfb", {"rec_yds"}, {}) == []


def test_capture_row_shape_matches_the_table_columns():
    from sportsmodel import db
    row = close.capture_row(sport="nfl", game=_g(1, 12), event_id="e1", kind="lines", now=NOW, credits=6,
                            status="ok", regions="us,eu", markets=["h2h", "spreads"])
    assert set(row) == set(db.LEDGER_COLUMNS["close_capture"])
    assert row["minutes_to_kick"] == pytest.approx(12) and row["markets"] == "h2h,spreads"
    assert row["captured_at"] == NOW.isoformat()
