from sportsmodel import teams

# The 30 Statcast abbreviations validated against the 2026 data.
STATCAST_ABBREVS = {
    "ATH", "ATL", "AZ", "BAL", "BOS", "CHC", "CIN", "CLE", "COL", "CWS",
    "DET", "HOU", "KC", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY",
    "PHI", "PIT", "SD", "SEA", "SF", "STL", "TB", "TEX", "TOR", "WSH",
}


def test_all_30_teams_mapped():
    assert len(teams.MLBAM_TO_STATCAST) == 30


def test_mapping_targets_match_statcast_exactly():
    assert set(teams.MLBAM_TO_STATCAST.values()) == STATCAST_ABBREVS


def test_lookup_and_missing():
    assert teams.statcast_abbrev(147) == "NYY"
    assert teams.statcast_abbrev(133) == "ATH"  # the relocated Athletics
    assert teams.statcast_abbrev(None) is None
    assert teams.statcast_abbrev(9999) is None
