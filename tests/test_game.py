import pytest

from sportsmodel.model import game

# A league-average-ish per-PA vector.
LG_VEC = {
    "p_bb": 0.085, "p_k": 0.225, "p_1b": 0.145, "p_2b": 0.045,
    "p_3b": 0.004, "p_hr": 0.033, "p_out": 0.463,
}


def test_score_distribution_normalized_and_shaped():
    pmf = game.score_distribution(4.5)
    assert sum(pmf) == pytest.approx(1.0, abs=1e-9)
    # Mean of the PMF should recover the input (minus tiny truncation error).
    mean = sum(k * p for k, p in enumerate(pmf))
    assert mean == pytest.approx(4.5, abs=0.05)
    # Overdispersed: variance > mean.
    var = sum((k - mean) ** 2 * p for k, p in enumerate(pmf))
    assert var > mean


def test_expected_runs_reasonable():
    r = game.expected_runs(LG_VEC)
    # A league-average lineup should score roughly a normal MLB total (~3.5-5.5).
    assert 3.0 < r < 6.0


def test_better_offense_scores_more():
    strong = {**LG_VEC, "p_hr": 0.06, "p_1b": 0.17, "p_out": 0.411}
    assert game.expected_runs(strong) > game.expected_runs(LG_VEC)


def test_equal_teams_home_edge():
    res = game.win_total_probabilities(4.5, 4.5)
    # Equal expected runs -> home just above 50% from the extra-innings edge.
    assert 0.50 < res["home_win_prob"] < 0.55
    assert res["pred_total"] == pytest.approx(9.0)


def test_stronger_home_team_favored():
    res = game.win_total_probabilities(5.5, 3.5)
    assert res["home_win_prob"] > 0.6


def test_total_over_monotonic():
    # Higher line -> lower P(over).
    p_low = game.prob_total_over(4.5, 4.5, 7.5)
    p_high = game.prob_total_over(4.5, 4.5, 9.5)
    assert p_low > p_high
    assert 0.0 < p_high < p_low < 1.0


def test_weather_hr_multiplier():
    assert game.weather_hr_multiplier(70) == pytest.approx(1.0)
    assert game.weather_hr_multiplier(90) > 1.0   # hot -> more HR
    assert game.weather_hr_multiplier(45) < 1.0   # cold -> fewer HR
    assert game.weather_hr_multiplier(200) == pytest.approx(1.20)  # clamped


def test_apply_hr_multiplier_renormalizes_and_lifts_runs():
    v = game.apply_hr_multiplier(LG_VEC, 1.15)
    assert sum(v.values()) == pytest.approx(1.0)
    assert v["p_hr"] > LG_VEC["p_hr"]
    assert game.expected_runs(v) > game.expected_runs(LG_VEC)


def test_apply_bip_defense_good_defense_lowers_runs():
    good = game.apply_bip_defense(LG_VEC, 0.95)   # better defense (fewer BIP hits)
    bad = game.apply_bip_defense(LG_VEC, 1.05)    # worse defense
    assert sum(good.values()) == pytest.approx(1.0)
    assert sum(bad.values()) == pytest.approx(1.0)
    assert good["p_1b"] < LG_VEC["p_1b"] < bad["p_1b"]
    assert game.expected_runs(good) < game.expected_runs(LG_VEC) < game.expected_runs(bad)
