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
