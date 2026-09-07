"""The committed NFL team crosswalk asset is well-formed and complete."""
import json
import pathlib

from sportsmodel.nfl.teams import TEAMS

ASSET = pathlib.Path(__file__).parents[2] / "assets" / "nfl" / "nfl_teams.json"


def test_asset_covers_all_32_current_franchises():
    crosswalk = json.loads(ASSET.read_text())
    assert set(crosswalk) == set(TEAMS)  # every current franchise, no extras
    assert len(crosswalk) == 32


def test_known_mappings():
    crosswalk = json.loads(ASSET.read_text())
    assert crosswalk["PHI"] == "Philadelphia Eagles"
    assert crosswalk["WAS"] == "Washington Commanders"
    assert crosswalk["LA"] == "Los Angeles Rams"
    assert all(isinstance(v, str) and v for v in crosswalk.values())
