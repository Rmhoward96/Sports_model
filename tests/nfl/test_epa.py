import pandas as pd

from sportsmodel.nfl.epa import team_epa_from_pbp


def test_off_def_epa_means_with_nan_and_null_team_dropped():
    # KC offense plays: epa 1.0, 2.0 (mean off_epa = 1.5)
    # BUF offense plays: epa 0.5, NaN (dropped) -> mean off_epa = 0.5
    # KC defense plays (BUF on offense): epa 0.5, NaN (dropped) -> mean def_epa = 0.5
    # BUF defense plays (KC on offense): epa 1.0, 2.0 -> mean def_epa = 1.5
    # One row has null posteam but valid defteam=BUF and epa=3.0 -> dropped from
    # offense aggregation entirely, but still counts toward BUF's def_epa.
    pbp = pd.DataFrame(
        [
            {"posteam": "KC", "defteam": "BUF", "epa": 1.0},
            {"posteam": "KC", "defteam": "BUF", "epa": 2.0},
            {"posteam": "BUF", "defteam": "KC", "epa": 0.5},
            {"posteam": "BUF", "defteam": "KC", "epa": float("nan")},
            {"posteam": None, "defteam": "BUF", "epa": 3.0},
        ]
    )

    result = team_epa_from_pbp(pbp)

    assert set(result.keys()) == {"KC", "BUF"}

    # KC offense: rows 1,2 -> mean(1.0, 2.0) = 1.5
    assert result["KC"]["off_epa"] == 1.5
    # KC defense: rows 3 (0.5), row 4 dropped (NaN epa) -> mean = 0.5
    assert result["KC"]["def_epa"] == 0.5

    # BUF offense: row 3 (0.5), row 4 dropped (NaN epa) -> mean = 0.5
    assert result["BUF"]["off_epa"] == 0.5
    # BUF defense: rows 1,2 (1.0, 2.0) plus row 5 (3.0, null posteam still counts
    # on defense) -> mean(1.0, 2.0, 3.0) = 2.0
    assert result["BUF"]["def_epa"] == 2.0


def test_team_only_on_one_side_gets_none_for_other():
    # NE only ever appears as posteam (offense); never as defteam.
    pbp = pd.DataFrame(
        [
            {"posteam": "NE", "defteam": "MIA", "epa": 1.0},
        ]
    )

    result = team_epa_from_pbp(pbp)

    assert result["NE"]["off_epa"] == 1.0
    assert result["NE"]["def_epa"] is None

    assert result["MIA"]["off_epa"] is None
    assert result["MIA"]["def_epa"] == 1.0
