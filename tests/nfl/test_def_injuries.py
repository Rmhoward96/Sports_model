import numpy as np
import pandas as pd

from sportsmodel.nfl.def_injuries import DEF_GROUPS, defender_snaps, vacated_by_defense


def _snaps():
    rows = []
    for wk in (1, 2, 3):
        rows += [
            {"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "cb1", "position": "CB",
             "team": "BUF", "opponent": "KC", "defense_snaps": 60, "defense_pct": 1.0},
            {"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "de1", "position": "DE",
             "team": "BUF", "opponent": "KC", "defense_snaps": 30, "defense_pct": 0.5},
            {"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "lb1", "position": "LB",
             "team": "BUF", "opponent": "KC", "defense_snaps": 45, "defense_pct": 0.75},
        ]
    return pd.DataFrame(rows)


IDS = {"cb1": "00-CB", "de1": "00-DE", "lb1": "00-LB"}


def test_groups():
    assert DEF_GROUPS["CB"] == "cov" and DEF_GROUPS["DE"] == "rush" and DEF_GROUPS["LB"] == "run"


def test_vacated_sums_out_defenders_prior_share_by_group():
    ds = defender_snaps(_snaps(), IDS)
    inj = pd.DataFrame({"season": [2024, 2024], "week": [4, 4], "team": ["BUF", "BUF"],
                        "gsis_id": ["00-CB", "00-DE"], "report_status": ["Out", "Questionable"]})
    tw = pd.DataFrame({"season": [2024], "week": [4], "team": ["BUF"]})
    v = vacated_by_defense(ds, inj, tw).iloc[0]
    assert np.isclose(v.di_vacated_cov, 1.0) and v.di_vacated_rush == 0.0 and v.di_vacated_run == 0.0


def test_no_report_week_is_nan_and_own_week_snaps_never_count():
    ds = defender_snaps(_snaps(), IDS)
    tw = pd.DataFrame({"season": [2024, 2024], "week": [2, 5], "team": ["BUF", "BUF"]})
    inj = pd.DataFrame({"season": [2024], "week": [2], "team": ["BUF"],
                        "gsis_id": ["00-LB"], "report_status": ["Doubtful"]})
    v = vacated_by_defense(ds, inj, tw)
    w2 = v[v.week == 2].iloc[0]
    assert np.isclose(w2.di_vacated_run, 0.75)      # only week-1 snaps are before week 2
    assert np.isnan(v[v.week == 5].iloc[0].di_vacated_cov)
