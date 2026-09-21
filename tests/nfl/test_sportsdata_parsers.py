"""Pure-parser tests for the NFL SportsDataIO adapter (injuries + team
crosswalk). No network; fixtures under tests/fixtures/nfl/ match
SportsDataIO's REAL NFL schema field names (Player[]: Team/FirstName/
LastName/Position/InjuryStatus/InjuryBodyPart/InjuryNotes; Team[]: Key/
FullName/City/Name -- no School field, unlike CFB)."""
import json
import pathlib

from sportsmodel.nfl import sportsdata

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "nfl"


def _load(name: str):
    return json.loads((FIX / name).read_text())


def test_endpoint_path_constants():
    # NFL injuries live under /projections/, not /scores/ like CFB.
    assert sportsdata.INJURED_PLAYERS_PATH == "/projections/json/InjuredPlayers"
    assert sportsdata.TEAMS_PATH == "/scores/json/Teams"


def test_parse_injuries_groups_by_team_abbreviation_with_status():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    assert "PHI" in out
    qb = next(p for p in out["PHI"] if p["position"] == "QB")
    assert qb["player"] == "John Smith"
    assert qb["status"] == "Questionable"
    assert qb["note"] == "Shoulder - Limited in practice"


def test_parse_injuries_note_none_when_bodypart_and_notes_null():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    rb = next(p for p in out["PHI"] if p["position"] == "RB")
    assert rb["note"] is None
    assert rb["status"] == "Out"


def test_parse_injuries_skips_rows_with_null_team():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    assert None not in out
    assert set(out) == {"PHI", "DAL"}


def test_parse_teams_maps_key_to_fullname_skipping_nulls():
    out = sportsdata.parse_teams(_load("sportsdata_teams.json"))
    assert out == {"PHI": "Philadelphia Eagles", "DAL": "Dallas Cowboys"}


# --- betting splits (Phase 2) -------------------------------------------------
# Mock payload following SportsDataIO's documented Betting swagger. Field names
# are UNVERIFIED against a live tier (see parse_betting_splits' caveat); this
# test pins the mapping logic (market/side/percentages), not the live schema.
def _mock_splits_payload():
    return {
        "BettingMarketSplits": [
            {"BettingMarketType": "Point Spread", "BettingBetSplits": [
                {"BettingOutcomeType": "Home", "MoneyPercentage": 62.0, "BetPercentage": 48.0},
                {"BettingOutcomeType": "Away", "MoneyPercentage": 38.0, "BetPercentage": 52.0},
            ]},
            {"BettingMarketType": "Total Points", "BettingBetSplits": [
                {"BettingOutcomeType": "Over", "MoneyPercentage": 55.0, "BetPercentage": 51.0},
                {"BettingOutcomeType": "Under", "MoneyPercentage": 45.0, "BetPercentage": 49.0},
            ]},
            {"BettingMarketType": "Player Props", "BettingBetSplits": [  # unmapped -> skipped
                {"BettingOutcomeType": "Over", "MoneyPercentage": 10.0, "BetPercentage": 90.0},
            ]},
        ]
    }


def test_parse_betting_splits_maps_markets_sides_and_percentages():
    rows = sportsdata.parse_betting_splits(_mock_splits_payload())
    by = {(r["market"], r["side"]): r for r in rows}
    assert by[("spread", "home")]["cash_pct"] == 62.0
    assert by[("spread", "home")]["ticket_pct"] == 48.0
    assert by[("total", "over")]["cash_pct"] == 55.0
    assert set(by) == {("spread", "home"), ("spread", "away"),
                       ("total", "over"), ("total", "under")}  # Player Props dropped


def test_parse_betting_splits_accepts_bare_list_and_skips_empty():
    bare = _mock_splits_payload()["BettingMarketSplits"]
    assert len(sportsdata.parse_betting_splits(bare)) == 4
    assert sportsdata.parse_betting_splits([]) == []
    assert sportsdata.parse_betting_splits({"BettingMarketSplits": []}) == []
    # a split with neither percentage is skipped
    none_pct = [{"BettingMarketType": "Moneyline", "BettingBetSplits": [
        {"BettingOutcomeType": "Home", "MoneyPercentage": None, "BetPercentage": None}]}]
    assert sportsdata.parse_betting_splits(none_pct) == []
