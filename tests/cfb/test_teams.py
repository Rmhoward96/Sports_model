from sportsmodel.cfb.teams import FCS, load_fbs_ids, normalize


def test_fbs_passthrough_and_fcs_collapse():
    fbs = load_fbs_ids()
    assert len(fbs) > 120  # ~130 FBS programs
    some_fbs = next(iter(fbs))
    assert normalize(int(some_fbs)) == some_fbs
    assert normalize("99999999") == FCS  # unknown id -> FCS anchor
    assert normalize(some_fbs) == some_fbs  # str or int in
