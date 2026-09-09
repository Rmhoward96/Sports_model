"""Pure-parser tests for `cfbd.parse_team_ppa` (CFBD `/ppa/teams`). No
network; fixture is committed sample JSON under tests/fixtures/cfb/, shaped
to match the real CFBD OpenAPI schema (per-team offense/defense PPA
blocks)."""
import json
import pathlib

from sportsmodel.cfb import cfbd

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "cfb"


def _load(name: str):
    return json.loads((FIX / name).read_text())


def test_parse_team_ppa_maps_team_to_off_def_overall():
    payload = _load("cfbd_ppa.json")
    out = cfbd.parse_team_ppa(payload)

    assert out["Alabama"] == {"off_ppa": 0.23, "def_ppa": 0.05}
    assert out["Georgia"] == {"off_ppa": 0.31, "def_ppa": -0.09}
    assert out["Kent State"] == {"off_ppa": -0.05, "def_ppa": 0.18}


def test_parse_team_ppa_missing_offense_defense_blocks_is_none():
    out = cfbd.parse_team_ppa(_load("cfbd_ppa.json"))
    assert out["New Program"] == {"off_ppa": None, "def_ppa": None}


def test_parse_team_ppa_skips_null_team():
    out = cfbd.parse_team_ppa(_load("cfbd_ppa.json"))
    assert None not in out
    assert set(out) == {"Alabama", "Georgia", "Kent State", "New Program"}
