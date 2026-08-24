import pytest
from sportsmodel.nfl.teams import normalize_team, TEAMS


def test_len_is_32():
    assert len(TEAMS) == 32


def test_aliases_and_relocations():
    assert normalize_team("LAR") == "LA"
    assert normalize_team("WSH") == "WAS"
    assert normalize_team("OAK") == "LV"
    assert normalize_team("SD") == "LAC"
    assert normalize_team("STL") == "LA"


def test_idempotent_and_case_insensitive():
    for t in TEAMS:
        assert normalize_team(t) == t
    assert normalize_team("kc") == "KC"
    assert normalize_team(normalize_team("OAK")) == "LV"


def test_unknown_raises():
    with pytest.raises(ValueError):
        normalize_team("XYZ")
