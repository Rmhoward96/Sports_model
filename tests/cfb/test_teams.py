from sportsmodel.cfb.teams import FCS, cfbd_to_espn, load_fbs_ids, normalize


def test_fbs_passthrough_and_fcs_collapse():
    fbs = load_fbs_ids()
    assert len(fbs) > 120  # ~130 FBS programs
    some_fbs = next(iter(fbs))
    assert normalize(int(some_fbs)) == some_fbs
    assert normalize("99999999") == FCS  # unknown id -> FCS anchor
    assert normalize(some_fbs) == some_fbs  # str or int in


def test_cfbd_to_espn_matches_known_schools():
    cases = {
        # plain (unique prefix)
        "Alabama": "333", "Ohio State": "194", "Georgia Tech": "59",
        "Arizona State": "9", "Louisiana Tech": "2348", "Texas A&M": "245",
        # ambiguous bare names -> flagship (not the longer sibling)
        "Texas": "251", "Ohio": "195", "Miami": "2390", "Miami (OH)": "193",
        "Louisiana": "309", "Washington": "264", "Washington State": "265",
        "Georgia": "61", "Michigan": "130",
        # CFBD spelling differs from ESPN displayName
        "Connecticut": "41", "Appalachian State": "2026",
        "Southern Mississippi": "2572", "Louisiana Monroe": "2433",
        # acronyms + accents
        "LSU": "99", "TCU": "2628", "UCF": "2116",
        "San José State": "23", "Hawai'i": "62",
    }
    for cfbd_name, espn_id in cases.items():
        got = cfbd_to_espn(cfbd_name)
        assert got == espn_id, f"{cfbd_name!r} -> {got!r}, expected {espn_id!r}"
        assert got in load_fbs_ids()


def test_cfbd_to_espn_unknown_returns_none():
    assert cfbd_to_espn("North Dakota State") is None  # FCS
    assert cfbd_to_espn("Not A Team") is None
    assert cfbd_to_espn("") is None
