"""Pure/network-free tests for the NFL+CFB odds ingester's assembly logic.

`build_game_lookup`, `parse_game_odds`, and `events_in_prop_window` are all pure
functions of plain dicts, so these tests exercise the sport-loop's join and
prop-window logic without hitting the Odds API or ESPN -- no monkeypatching of
network calls needed.
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "ingest_odds", Path(__file__).resolve().parents[1] / "scripts" / "ingest_odds.py"
)
ingest_odds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_odds)

from sportsmodel import sports
from sportsmodel.ingest import odds


def _fake_resolved_game_date(commence_iso):
    # deterministic stand-in: just take the date portion, no UTC-shift math
    return (commence_iso or "")[:10]


def test_build_game_lookup_maps_matched_and_drops_unmatched():
    events = [
        {"id": "e1", "home_team": "Home A", "away_team": "Away A",
         "commence_time": "2026-09-10T00:20:00Z"},
        {"id": "e2", "home_team": "Home B", "away_team": "Away B",
         "commence_time": "2026-09-11T00:20:00Z"},
    ]
    espn_games = [
        {"game_pk": 501, "home_name": "Home A", "away_name": "Away A",
         "commence_time": "2026-09-10T00:20:00Z"},
        # no ESPN game for e2 -> should be dropped
    ]

    def fake_matcher(ev, games):
        for g in games:
            if g["home_name"] == ev["home_team"] and g["away_name"] == ev["away_team"]:
                return g["game_pk"]
        return None

    lookup = ingest_odds.build_game_lookup(
        events, espn_games, fake_matcher, _fake_resolved_game_date
    )

    assert lookup == {("Home A", "Away A", "2026-09-10"): 501}
    assert ("Home B", "Away B", "2026-09-11") not in lookup
    assert len(lookup) == 1


def test_build_game_lookup_empty_events_or_no_matches():
    assert ingest_odds.build_game_lookup([], [], lambda ev, g: None, _fake_resolved_game_date) == {}

    events = [{"id": "e1", "home_team": "X", "away_team": "Y", "commence_time": "2026-09-10T00:00:00Z"}]
    lookup = ingest_odds.build_game_lookup(events, [], lambda ev, g: None, _fake_resolved_game_date)
    assert lookup == {}


def _pinnacle_game_event(home: str, away: str, commence: str) -> dict:
    return {
        "id": "evt-1",
        "home_team": home,
        "away_team": away,
        "commence_time": commence,
        "bookmakers": [{
            "key": "pinnacle",
            "markets": [
                {"key": "h2h", "outcomes": [
                    {"name": home, "price": -150},
                    {"name": away, "price": 130},
                ]},
                {"key": "spreads", "outcomes": [
                    {"name": home, "point": -3.5, "price": -110},
                    {"name": away, "point": 3.5, "price": -110},
                ]},
                {"key": "totals", "outcomes": [
                    {"name": "Over", "point": 44.5, "price": -105},
                    {"name": "Under", "point": 44.5, "price": -115},
                ]},
            ],
        }],
    }


def test_parse_game_odds_pinnacle_bookmaker_maps_all_three_markets():
    home, away, commence = "Kansas City Chiefs", "Buffalo Bills", "2026-09-11T00:20:00Z"
    event = _pinnacle_game_event(home, away, commence)
    game_lookup = {(home, away, odds.resolved_game_date(commence)): 999}

    rows = odds.parse_game_odds([event], game_lookup, "2026-09-10T12:00:00Z")

    assert rows, "expected rows for a matched pinnacle event"
    assert all(r["game_pk"] == 999 for r in rows)

    pinnacle_rows = [r for r in rows if r["book"] == "pinnacle"]
    assert pinnacle_rows == rows, "every row should be attributed to the pinnacle book"

    markets = {r["market"] for r in pinnacle_rows}
    assert markets == {"moneyline", "spread", "total"}

    ml = {r["side"]: r for r in pinnacle_rows if r["market"] == "moneyline"}
    assert ml["home"]["price"] == -150
    assert ml["away"]["price"] == 130

    sp = {r["side"]: r for r in pinnacle_rows if r["market"] == "spread"}
    assert sp["home"]["line"] == -3.5
    assert sp["away"]["line"] == 3.5

    tot = {r["side"]: r for r in pinnacle_rows if r["market"] == "total"}
    assert tot["over"]["line"] == 44.5
    assert tot["under"]["line"] == 44.5


def test_parse_game_odds_drops_events_with_no_game_lookup_match():
    event = _pinnacle_game_event("A", "B", "2026-09-11T00:20:00Z")
    rows = odds.parse_game_odds([event], {}, "2026-09-10T12:00:00Z")
    assert rows == []


# -- events_in_prop_window -----------------------------------------------------

_NOW = datetime(2026, 9, 11, 0, 0, 0, tzinfo=timezone.utc)


def _matcher_all_match(ev, games):
    return {"e-in-window": 501, "e-too-far": 502, "e-started": 503}.get(ev["id"])


def _iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def test_events_in_prop_window_includes_event_inside_window():
    events = [{"id": "e-in-window", "commence_time": _iso(_NOW + timedelta(minutes=60))}]
    result = ingest_odds.events_in_prop_window(events, _matcher_all_match, [], _NOW, 150)
    assert result == [(events[0], 501)]


def test_events_in_prop_window_excludes_event_beyond_window():
    events = [{"id": "e-too-far", "commence_time": _iso(_NOW + timedelta(minutes=300))}]
    result = ingest_odds.events_in_prop_window(events, _matcher_all_match, [], _NOW, 150)
    assert result == []


def test_events_in_prop_window_excludes_already_started_event():
    events = [{"id": "e-started", "commence_time": _iso(_NOW - timedelta(minutes=5))}]
    result = ingest_odds.events_in_prop_window(events, _matcher_all_match, [], _NOW, 150)
    assert result == []


def test_events_in_prop_window_drops_unmatched_event():
    events = [{"id": "e-unmatched", "commence_time": _iso(_NOW + timedelta(minutes=60))}]
    result = ingest_odds.events_in_prop_window(events, _matcher_all_match, [], _NOW, 150)
    assert result == []


def test_events_in_prop_window_mixed_batch():
    events = [
        {"id": "e-in-window", "commence_time": _iso(_NOW + timedelta(minutes=60))},
        {"id": "e-too-far", "commence_time": _iso(_NOW + timedelta(minutes=300))},
        {"id": "e-started", "commence_time": _iso(_NOW - timedelta(minutes=5))},
        {"id": "e-unmatched", "commence_time": _iso(_NOW + timedelta(minutes=60))},
    ]
    result = ingest_odds.events_in_prop_window(events, _matcher_all_match, [], _NOW, 150)
    assert result == [(events[0], 501)]


def test_nfl_prop_markets_match_prop_market_map_values():
    cfg = sports.get("nfl")
    assert list(cfg.prop_market_map.values()) == [
        "player_pass_yds", "player_pass_tds", "player_reception_yds",
        "player_receptions", "player_rush_yds", "player_rush_reception_yds",
        "player_rush_attempts", "player_anytime_td",
    ]


def test_prop_window_minutes_default_and_env():
    assert ingest_odds.prop_window_minutes({}) == 150
    assert ingest_odds.prop_window_minutes({"PROP_WINDOW_MIN": "90"}) == 90
    assert ingest_odds.prop_window_minutes({"PROP_WINDOW_MIN": ""}) == 0


def test_prop_window_minutes_slate_scope_covers_a_week():
    assert ingest_odds.prop_window_minutes({"PROP_SCOPE": "slate"}) == 7 * 24 * 60
    assert ingest_odds.prop_window_minutes({"PROP_SCOPE": "SLATE", "PROP_WINDOW_MIN": "90"}) == 7 * 24 * 60


# -- prop-capture enable predicate ----------------------------------------------

def test_props_enabled_default_true():
    assert ingest_odds.props_enabled({}, 150) is True


def test_props_enabled_false_when_ingest_props_false():
    assert ingest_odds.props_enabled({"INGEST_PROPS": "false"}, 150) is False


def test_props_enabled_false_when_ingest_props_false_mixed_case():
    assert ingest_odds.props_enabled({"INGEST_PROPS": "False"}, 150) is False


def test_props_enabled_false_when_window_zero():
    assert ingest_odds.props_enabled({}, 0) is False


# -- MLB (reactivated 2026-10-05) ---------------------------------------------------

def test_mlb_is_an_ingested_sport_with_schedule_fetcher_and_matcher():
    from sportsmodel.mlb import matcher
    assert "mlb" in ingest_odds.SPORTS
    assert "mlb" in ingest_odds._ESPN_FETCHERS
    assert ingest_odds._matcher_for("mlb") is matcher.match_odds_event
    # football wiring is untouched
    assert ingest_odds._matcher_for("nfl").__module__.endswith("nfl.matcher")
    assert ingest_odds._matcher_for("cfb").__module__.endswith("cfb.matcher")


def test_sports_to_run_defaults_to_all_and_honors_odds_sports():
    assert ingest_odds.sports_to_run({}) == ("nfl", "cfb", "mlb")
    assert ingest_odds.sports_to_run({"ODDS_SPORTS": ""}) == ("nfl", "cfb", "mlb")
    assert ingest_odds.sports_to_run({"ODDS_SPORTS": "mlb"}) == ("mlb",)
    assert ingest_odds.sports_to_run({"ODDS_SPORTS": "NFL, cfb"}) == ("nfl", "cfb")
    assert ingest_odds.sports_to_run({"ODDS_SPORTS": "mlb,cricket"}) == ("mlb",)   # unknown dropped
    # a typo must never fall back to "all sports" (credits): nothing valid -> nothing runs
    assert ingest_odds.sports_to_run({"ODDS_SPORTS": "cricket"}) == ()


def test_mlb_prop_window_is_capped_but_football_is_not():
    slate = {"PROP_SCOPE": "slate"}
    assert ingest_odds.sport_prop_window_minutes("nfl", slate) == 7 * 24 * 60
    assert ingest_odds.sport_prop_window_minutes("mlb", slate) == ingest_odds.MLB_PROP_WINDOW_CAP_MIN == 240
    # a shorter configured window is respected, never widened to the cap
    assert ingest_odds.sport_prop_window_minutes("mlb", {"PROP_WINDOW_MIN": "90"}) == 90
    assert ingest_odds.sport_prop_window_minutes("mlb", {"PROP_WINDOW_MIN": ""}) == 0
    assert ingest_odds.sport_prop_window_minutes("nfl", {"PROP_WINDOW_MIN": "90"}) == 90


def test_mlb_prop_markets_requested_are_only_the_live_four():
    cfg = sports.get("mlb")
    assert sorted(cfg.prop_market_map.values()) == [
        "batter_total_bases", "pitcher_hits_allowed", "pitcher_outs", "pitcher_strikeouts"]
    assert not {"batter_hits", "batter_hits_runs_rbis", "batter_home_runs"} & set(cfg.prop_market_map.values())


def test_fetch_games_mlb_keeps_every_game_type(monkeypatch):
    from sportsmodel.ingest import mlb_statsapi

    def fake_schedule(day):
        return [{"game_pk": 1, "game_date": day, "home_team_name": "H", "away_team_name": "A",
                 "commence_time": f"{day}T23:00:00Z", "game_type": "D"},
                {"game_pk": 2, "game_date": day, "home_team_name": "H2", "away_team_name": "A2",
                 "commence_time": None, "game_type": "R"}]

    monkeypatch.setattr(mlb_statsapi, "fetch_schedule", fake_schedule)
    games = ingest_odds._fetch_games_mlb()
    assert len(games) == 5 * 2          # yesterday .. +3 days, both game types kept
    assert {"game_pk", "home_name", "away_name", "commence_time", "game_date"} <= set(games[0])


def test_mlb_event_flows_through_lookup_and_parse_to_game_pk():
    from sportsmodel.mlb import matcher
    events = [{"id": "e1", "home_team": "Los Angeles Dodgers", "away_team": "Atlanta Braves",
               "commence_time": "2026-10-07T01:30:00Z",
               "bookmakers": [{"key": "pinnacle", "markets": [{"key": "h2h", "outcomes": [
                   {"name": "Los Angeles Dodgers", "price": -140}, {"name": "Atlanta Braves", "price": 120}]}]}]}]
    games = [{"game_pk": 849826, "home_name": "Los Angeles Dodgers", "away_name": "Atlanta Braves",
              "game_date": "2026-10-06", "commence_time": "2026-10-07T01:30:00Z"}]
    lookup = ingest_odds.build_game_lookup(events, games, matcher.match_odds_event)
    rows = odds.parse_game_odds(events, lookup, "2026-10-06T20:00:00+00:00")
    assert {(r["game_pk"], r["market"], r["side"], r["price"]) for r in rows} == {
        (849826, "moneyline", "home", -140), (849826, "moneyline", "away", 120)}
