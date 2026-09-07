"""Pure-parser tests for the SportsDataIO adapter (CFB injuries + team
crosswalk). No network; all fixtures are committed sample JSON under
tests/fixtures/cfb/, shaped to match SportsDataIO's REAL CFB schema field
names verified from their OpenAPI swagger:
  - InjuredPlayers (`Player[]`): `Team`/`TeamID`/`FirstName`/`LastName`/
    `Position`/`InjuryStatus`/`InjuryBodyPart`/`InjuryNotes`/
    `InjuryStartDate`.
  - Teams (`Team[]`): `TeamID`/`Key`/`School`/`Name`/`TeamLogoUrl`.

There is no News or weather endpoint in the CFB API, so no parsers/fixtures
exist for those here.
"""
import json
import pathlib

from sportsmodel.cfb import sportsdata

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "cfb"


def _load(name: str):
    return json.loads((FIX / name).read_text())


# ------------------------------------------------------------ parse_injuries --

def test_parse_injuries_groups_by_team_abbreviation_with_status():
    payload = _load("sportsdata_injuries.json")
    out = sportsdata.parse_injuries(payload)
    assert "ALA" in out
    qb = next(p for p in out["ALA"] if p["position"] == "QB")
    assert qb["status"] in {"Out", "Questionable", "Doubtful", "Probable"}
    assert qb["player"] == "John Smith"
    assert qb["note"] == "Shoulder - Limited in practice"


def test_parse_injuries_skips_rows_with_null_team():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    assert None not in out
    assert set(out) == {"ALA", "UGA"}


def test_parse_injuries_null_bodypart_and_notes_does_not_crash():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    rb = next(p for p in out["ALA"] if p["position"] == "RB")
    assert rb["note"] is None


def test_parse_injuries_single_null_note_field_falls_back_to_the_other():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    wr = next(p for p in out["UGA"] if p["position"] == "WR")
    assert wr["note"] == "Ankle"


# --------------------------------------------------------------- parse_teams --

def test_parse_teams_maps_key_to_school():
    payload = _load("sportsdata_teams.json")
    out = sportsdata.parse_teams(payload)
    assert out["ALA"] == "Alabama"
    assert out["UGA"] == "Georgia"


def test_parse_teams_skips_rows_with_null_key_or_school():
    out = sportsdata.parse_teams(_load("sportsdata_teams.json"))
    assert None not in out
    assert "Unknown" not in out.values()
    assert "XXX" not in out
    assert set(out) == {"ALA", "UGA"}
