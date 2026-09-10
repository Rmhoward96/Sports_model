"""Pure/network-free tests for the NFL+CFB game-lines ingester's assembly logic.

`build_game_lookup` and `parse_game_odds` are both pure functions of plain dicts,
so these tests exercise the sport-loop's join logic without hitting the Odds API
or ESPN -- no monkeypatching of network calls needed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_SPEC = importlib.util.spec_from_file_location(
    "ingest_odds", Path(__file__).resolve().parents[1] / "scripts" / "ingest_odds.py"
)
ingest_odds = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ingest_odds)

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
