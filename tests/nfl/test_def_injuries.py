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


def test_group_is_as_of_prior_game_not_all_time_position():
    """A defender who was a safety (S) in 2024 and moved to LB in 2025 must be
    scored against his 2024-as-of group (cov) for a 2024 wk4 report, and his
    2025-as-of group (run) for a 2025 wk4 report -- never using games from a
    later season to decide the earlier season's group (no look-ahead)."""
    rows = []
    for wk in (1, 2, 3):
        rows.append({"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "sw1",
                     "position": "S", "team": "BUF", "opponent": "KC",
                     "defense_snaps": 50, "defense_pct": 0.8})
    for wk in (1, 2, 3):
        rows.append({"season": 2025, "week": wk, "game_type": "REG", "pfr_player_id": "sw1",
                     "position": "LB", "team": "BUF", "opponent": "KC",
                     "defense_snaps": 50, "defense_pct": 0.8})
    ds = defender_snaps(pd.DataFrame(rows), {"sw1": "00-SW"})

    inj2024 = pd.DataFrame({"season": [2024], "week": [4], "team": ["BUF"],
                            "gsis_id": ["00-SW"], "report_status": ["Out"]})
    tw2024 = pd.DataFrame({"season": [2024], "week": [4], "team": ["BUF"]})
    v2024 = vacated_by_defense(ds, inj2024, tw2024).iloc[0]
    assert v2024.di_vacated_cov > 0.0
    assert v2024.di_vacated_run == 0.0

    inj2025 = pd.DataFrame({"season": [2025], "week": [4], "team": ["BUF"],
                            "gsis_id": ["00-SW"], "report_status": ["Out"]})
    tw2025 = pd.DataFrame({"season": [2025], "week": [4], "team": ["BUF"]})
    v2025 = vacated_by_defense(ds, inj2025, tw2025).iloc[0]
    assert v2025.di_vacated_run > 0.0
    assert v2025.di_vacated_cov == 0.0


def test_defense_pct_scale_is_detected_per_row():
    snaps = pd.DataFrame([
        {"season": 2024, "week": 1, "game_type": "REG", "pfr_player_id": "cb1", "position": "CB",
         "team": "BUF", "opponent": "KC", "defense_snaps": 60, "defense_pct": 100},
        {"season": 2024, "week": 1, "game_type": "REG", "pfr_player_id": "de1", "position": "DE",
         "team": "BUF", "opponent": "KC", "defense_snaps": 30, "defense_pct": 0.5},
    ])
    ds = defender_snaps(snaps, {"cb1": "00-CB", "de1": "00-DE"})
    shares = ds.set_index("player_id")["share"]
    assert np.isclose(shares["00-CB"], 1.0)
    assert np.isclose(shares["00-DE"], 0.5)
