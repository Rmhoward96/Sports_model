import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import residual as rs
from sportsmodel.cfb.efficiency import POINT_FEATURES


def test_prior_game_means_are_expanding_means_of_earlier_games_only():
    adv = pd.DataFrame({"season": [2023] * 4, "season_type": ["regular"] * 3 + ["postseason"],
                        "week": [1, 2, 3, 1], "game_id": [1, 2, 3, 9], "team": ["a"] * 4,
                        "off_std_down_success": [0.4, 0.6, 0.8, 0.1]})
    sched = pd.DataFrame({"game_pk": [1, 2, 3], "week": [1, 2, 3], "game_type": ["REG"] * 3})
    m = rs.prior_game_means(adv, sched).set_index("game_id")
    assert np.isnan(m.loc[1, "std_down_success"])                 # opener: nothing earlier
    assert m.loc[2, "std_down_success"] == pytest.approx(0.4)
    assert m.loc[3, "std_down_success"] == pytest.approx(0.5)
    assert 9 not in m.index                                       # postseason never feeds in-season means
    assert np.isnan(m.loc[2, "line_yards"])                       # column absent from the pull -> NaN


def test_residual_features_are_home_minus_away_zero_filled_and_leak_scoped():
    table = pd.DataFrame({"season": [2023, 2023], "game_pk": [1, 2], "home_team": ["a", "a"],
                          "away_team": ["b", "c"], "week": [2, 3], "margin_v2": [0.0, 0.0],
                          "talent_gap": [0.3, 0.1], "travel_far_diff": [1.0, 0.0]})
    for f in POINT_FEATURES:
        table[f"h_{f}"], table[f"a_{f}"] = 0.0, 0.0
    table["h_havoc"], table["a_havoc"] = [0.2, 0.1], [0.05, 0.0]
    table["h_ppa_plays"] = [1.0, 1.0]
    means = pd.DataFrame({"game_id": [1, 1, 2, 2], "team": ["a", "b", "a", "c"],
                          "std_down_success": [0.6, 0.5, 0.7, np.nan]})
    for c in rs.DOWN_COLS:
        if c not in means:
            means[c] = np.nan
    meta = pd.DataFrame({"game_id": [1, 2], "home_pregame_elo": [1700.0, 1650.0],
                         "away_pregame_elo": [1600.0, np.nan]})
    ratings = pd.DataFrame({"season": [2022, 2022, 2023], "team": ["a", "b", "a"],
                            "fpi": [10.0, 4.0, 99.0], "srs": [8.0, 3.0, 99.0]})
    priors = {2023: [{"team_espn_id": "a", "returning_starters": 0.7},
                     {"team_espn_id": "b", "returning_starters": 0.5}]}
    pts = {f: 1.0 for f in POINT_FEATURES}
    pts["havoc"] = 0.0
    f = rs.residual_features(table, means, meta, priors, ratings,
                             {"talent_gap": 0.4, "travel_far_diff": 0.0}, {}, pts)
    assert f.loc[0, "cfbd_elo_diff"] == 100.0 and f.loc[1, "cfbd_elo_diff"] == 0.0     # NaN -> 0
    assert f.loc[0, "fpi_diff_prev"] == 6.0                       # 2022 ratings for 2023 games, not the 2023 row
    assert f.loc[1, "fpi_diff_prev"] == 0.0                       # team c unrated
    assert f.loc[0, "returning_usage_diff"] == pytest.approx(0.2)
    assert f.loc[0, "std_down_success_diff"] == pytest.approx(0.1)
    assert "unused_travel_far_diff" in f and "unused_talent_gap" not in f   # only 0-weight terms
    assert f.loc[0, "unused_pt_havoc"] == pytest.approx(0.15) and "unused_pt_ppa_plays" not in f
    assert not f.isna().any().any()


def synth(n=3000, seed=4, signal=True):
    rng = np.random.default_rng(seed)
    feat = pd.DataFrame({"signal": rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
    seasons = rng.choice(np.arange(2016, 2026), n)
    base = rng.normal(0, 3, n)
    resid_m = base + (0.8 * feat["signal"].to_numpy() if signal else 0.0)
    return feat, resid_m, base.copy(), seasons


def test_check_finds_a_real_signal_in_all_held_out_seasons():
    feat, rm, rt, seasons = synth()
    out = rs.run_residual_check(feat, rm, rt, seasons, range(2016, 2023), (2023, 2024, 2025))
    m = out["margin"]
    assert m["ridge"]["r2_oos"] > 0.03 and m["ridge"]["mae_change"] < 0
    assert m["ridge"]["top_features"][0]["feature"] == "signal"
    assert m["hgb"]["top_features"][0]["feature"] == "signal"
    assert {c["feature"] for c in m["v4_candidates"]} == {"signal"}
    assert set(m["ridge"]["mae_change_by_season"]) == {2023, 2024, 2025}
    assert out["total"]["v4_candidates"] == [] and out["total"]["ridge"]["r2_oos"] < 0.01
    assert "report only" in out["note"]


def test_pure_noise_gives_no_edge():
    feat, rm, rt, seasons = synth(signal=False)
    out = rs.run_residual_check(feat, rm, rt, seasons, range(2016, 2023), (2023, 2024, 2025))
    assert out["margin"]["ridge"]["r2_oos"] < 0.01
    assert out["margin"]["hgb"]["r2_oos"] < 0.01
