import pandas as pd
from sportsmodel.nfl.efficiency import (
    team_game_epa, adjusted_efficiency, efficiency_features)


def _pbp(rows):
    return pd.DataFrame(rows, columns=["season","week","posteam","defteam","epa"])


def test_team_game_epa_offense_and_defense():
    pbp = _pbp([
        [2023,1,"KC","DET",0.2],[2023,1,"KC","DET",0.4],
        [2023,1,"DET","KC",-0.1],
    ])
    g = team_game_epa(pbp)
    assert g[(2023,1,"KC")]["off"] == 0.3           # mean of KC offensive plays
    assert g[(2023,1,"KC")]["def"] == -0.1          # DET's epa vs KC == KC defense
    assert g[(2023,1,"DET")]["off"] == -0.1


def test_adjusted_efficiency_is_leakage_free():
    pbp = _pbp([
        [2023,1,"KC","DET",0.3],[2023,1,"DET","KC",-0.1],
        [2023,2,"KC","CHI",0.5],[2023,2,"CHI","KC",0.0],
    ])
    g = team_game_epa(pbp)
    adj = adjusted_efficiency(g, 2023, upto_week=2)   # uses only week 1
    assert "KC" in adj and "off_adj" in adj["KC"]
    # week-2 data must NOT leak: KC off_adj reflects only its 0.3 week-1 game
    assert adj["KC"]["off_adj"] == 0.3 or abs(adj["KC"]["off_adj"] - 0.3) < 0.5


def test_team_game_epa_normalizes_team_codes():
    # WSH is an alternate/historical code for WAS; LA plays under its
    # canonical code already. Without normalization these would fragment
    # into separate keys instead of aggregating as one team.
    pbp = _pbp([
        [2023,1,"WSH","LA",0.2],[2023,1,"WAS","LA",0.4],
        [2023,1,"LA","WSH",-0.1],[2023,1,"LA","WAS",-0.3],
    ])
    g = team_game_epa(pbp)
    assert set(k[2] for k in g) == {"WAS", "LA"}
    assert g[(2023,1,"WAS")]["off"] == 0.3
    assert g[(2023,1,"WAS")]["opp"] == "LA"
    assert g[(2023,1,"LA")]["def"] == 0.3


def test_efficiency_features_differential_sign():
    adj = {"KC":{"off_adj":0.3,"def_adj":-0.2}, "DET":{"off_adj":0.0,"def_adj":0.1}}
    f = efficiency_features(adj, home="KC", away="DET")
    assert f["eff_diff"] > 0     # KC clearly better -> positive home edge
    assert "home_off_adj" in f and "away_def_adj" in f
