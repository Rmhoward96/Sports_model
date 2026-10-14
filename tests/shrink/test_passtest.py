"""The frozen shrink pass test: walk-forward scoring, the 300 earlier-leg rule, the 2,000 judgement point, both pass
conditions at their boundaries, the seeded game-clustered bootstrap and the frozen verdict. Pure."""
import random

import numpy as np
import pytest
from scipy.optimize import minimize

from sportsmodel.shrink import fit, passtest
from sportsmodel.shrink.formula import shrink_prob


def leg(week, game_pk, p=0.6, m=0.5, y=1, market="rush_yds", season=2026, version="v2"):
    return {"season": season, "week": week, "game_pk": game_pk, "market": market, "p": p, "m": m, "y": y, "version": version}


def informative(week, n, game0=0, k=0.4, market="rush_yds", season=2026):
    """n legs whose true k is exactly `k` (expected outcomes): p - m = 0.1, y = m + k * 0.1."""
    return [leg(week, game0 + i // 2, p=0.6, m=0.5, y=0.5 + k * 0.1, market=market, season=season) for i in range(n)]


def scored_leg(gp, m, y, shrunk):
    return {"season": 2026, "week": 9, "game_pk": gp, "market": "rush_yds", "p": shrunk, "m": m, "y": y, "k": 1.0, "shrunk": shrunk}


def test_the_frozen_constants_are_the_specs():
    assert passtest.JUDGEMENT_LEGS == 2000 and passtest.MIN_BRIER_GAIN == 0.001
    assert (passtest.SLOPE_LO, passtest.SLOPE_HI) == (0.8, 1.2)
    assert (passtest.BOOTSTRAP_RESAMPLES, passtest.BOOTSTRAP_SEED, passtest.LOWER_PERCENTILE) == (10_000, 20261013, 2.5)


# ---- walk-forward scoring ---------------------------------------------------------------------------------------

def test_a_week_is_scored_with_the_k_fitted_on_earlier_weeks_only():
    legs = informative(1, 200, k=0.2) + informative(2, 150, game0=1000, k=0.2) + informative(3, 40, game0=2000, k=0.9)
    scored = passtest.score_walk_forward(legs, "v2")
    assert {s["week"] for s in scored} == {3}                       # weeks 1 and 2 have < 300 earlier legs
    expected_k = fit.fit_table(legs[:350], "v2")["markets"]["nfl"]["rush_yds"]["k"]
    assert expected_k == pytest.approx(0.2) and all(s["k"] == pytest.approx(0.2) for s in scored)    # not 0.9: week 3 never trains itself
    assert scored[0]["shrunk"] == pytest.approx(shrink_prob(0.6, 0.5, expected_k))


def test_later_weeks_never_change_an_earlier_weeks_score():
    base = informative(1, 300, k=0.2) + informative(2, 50, game0=1000, k=0.2)
    before = passtest.score_walk_forward(base, "v2")
    after = passtest.score_walk_forward(base + informative(3, 400, game0=2000, k=0.95), "v2")
    assert [s for s in after if s["week"] == 2] == before and len(before) == 50


def test_the_300_earlier_leg_threshold_is_exact():
    short = passtest.score_walk_forward(informative(1, 299) + informative(2, 10, game0=900), "v2")
    assert short == []
    enough = passtest.score_walk_forward(informative(1, 300) + informative(2, 10, game0=900), "v2")
    assert len(enough) == 10 and {s["week"] for s in enough} == {2}


def test_every_week_after_the_threshold_is_scored_and_earlier_weeks_accumulate():
    legs = informative(1, 300) + informative(2, 20, game0=1000) + informative(3, 20, game0=2000)
    scored = passtest.score_walk_forward(legs, "v2")
    assert sorted({s["week"] for s in scored}) == [2, 3] and len(scored) == 40


def test_weeks_are_ordered_by_season_then_week_and_legs_may_arrive_out_of_order():
    old = informative(18, 300, season=2025)
    new = informative(1, 10, game0=900, season=2026)
    scored = passtest.score_walk_forward(new + old, "v2")                  # 2026 week 1 comes after 2025 week 18
    assert len(scored) == 10 and {(s["season"], s["week"]) for s in scored} == {(2026, 1)}
    assert passtest.score_walk_forward(old + new, "v2") == scored


def test_the_version_rule_applies_inside_each_walk_forward_fit():
    v1 = informative(1, 400, k=0.8)
    for l in v1:
        l["version"] = "v1"
    v2 = informative(1, 300, game0=5000, k=0.1)
    scored = passtest.score_walk_forward(v1 + v2 + informative(2, 5, game0=9000), "v2")
    assert {round(s["k"], 6) for s in scored} == {0.1}                     # 300 current-version legs exist, so v1 is ignored
    only_v1 = passtest.score_walk_forward(v1 + informative(2, 5, game0=9000), "v2")
    assert {round(s["k"], 6) for s in only_v1} == {0.8}                    # fewer than 300 current-version legs: all versions pooled


def test_a_market_with_no_earlier_legs_is_scored_at_the_market_probability():
    legs = informative(1, 400, market="rush_yds") + [leg(2, 9000, p=0.7, m=0.5, y=1, market="receptions")]
    [s] = passtest.score_walk_forward(legs, "v2")
    assert s["market"] == "receptions" and s["k"] == 0.0 and s["shrunk"] == pytest.approx(0.5)


def test_no_legs_and_too_few_legs_score_nothing():
    assert passtest.score_walk_forward([], "v2") == [] and passtest.score_walk_forward(informative(1, 50), "v2") == []


# ---- the paired Brier gain ----------------------------------------------------------------------------------------

def test_the_gain_is_market_brier_minus_shrunk_brier():
    sc = [scored_leg(1, m=0.5, y=1, shrunk=0.6), scored_leg(2, m=0.5, y=0, shrunk=0.4)]
    assert passtest.brier_gain(sc) == pytest.approx(0.25 - 0.16)
    worse = [scored_leg(1, m=0.5, y=1, shrunk=0.4)]
    assert passtest.brier_gain(worse) == pytest.approx(0.25 - 0.36) and passtest.brier_gain([]) is None


def test_k_zero_scores_exactly_zero_gain_so_it_can_never_pass():
    sc = [scored_leg(i, m=0.55, y=i % 2, shrunk=0.55) for i in range(50)]
    assert passtest.brier_gain(sc) == 0.0
    assert passtest.verdict(0.0, 0.0, 1.0)["verdict"] == "FAIL"


# ---- the clustered bootstrap --------------------------------------------------------------------------------------

def _mixed(n_games, per_game, seed=3):
    rng = random.Random(seed)
    out = []
    for g in range(n_games):
        bias = rng.uniform(-0.05, 0.08)                                    # one shared shock per game: the legs inside are correlated
        for _ in range(per_game):
            out.append(scored_leg(g, m=0.5, y=1 if rng.random() < 0.5 + bias else 0, shrunk=0.5 + 0.3 * bias))
    return out


def test_the_bootstrap_is_deterministic_for_the_frozen_seed():
    sc = _mixed(60, 4)
    assert passtest.cluster_bootstrap_lower(sc) == passtest.cluster_bootstrap_lower(list(reversed(sc)))
    assert passtest.cluster_bootstrap_lower(sc, seed=1) != passtest.cluster_bootstrap_lower(sc)


def test_the_bootstrap_matches_a_hand_rolled_resample_of_games_with_the_frozen_seed():
    sc = _mixed(30, 3)
    gains = {}
    for s, g in zip(sc, passtest._gains(sc)):
        gains.setdefault(s["game_pk"], []).append(g)
    games = sorted(gains, key=str)
    rng = np.random.default_rng(20261013)
    means = []
    for _ in range(10):
        idx = rng.integers(0, len(games), size=(1000, len(games)))
        for row in idx:
            vals = [x for i in row for x in gains[games[i]]]
            means.append(sum(vals) / len(vals))
    assert passtest.cluster_bootstrap_lower(sc) == pytest.approx(float(np.percentile(means, 2.5)), abs=1e-12)


def test_resampling_whole_games_widens_the_interval_versus_resampling_legs():
    rng = random.Random(9)
    sc = []
    for g in range(40):
        y = 1 if rng.random() < 0.5 else 0                                  # every leg of a game shares one outcome
        sc += [scored_leg(g, m=0.5, y=y, shrunk=0.56) for _ in range(25)]
    clustered = passtest.cluster_bootstrap_lower(sc)
    one_leg_per_game = [dict(s, game_pk=i) for i, s in enumerate(sc)]       # pretend every leg is its own game
    assert clustered < passtest.cluster_bootstrap_lower(one_leg_per_game) - 0.01


def test_a_single_game_has_a_degenerate_interval_equal_to_its_mean_gain():
    sc = [scored_leg(7, m=0.5, y=1, shrunk=0.6), scored_leg(7, m=0.5, y=0, shrunk=0.45)]
    assert passtest.cluster_bootstrap_lower(sc) == pytest.approx(passtest.brier_gain(sc))
    assert passtest.cluster_bootstrap_lower([]) is None


# ---- the calibration slope ----------------------------------------------------------------------------------------

def _scipy_slope(sc):
    x = np.log(np.array([s["shrunk"] for s in sc]) / (1 - np.array([s["shrunk"] for s in sc])))
    y = np.array([s["y"] for s in sc], dtype=float)

    def nll(b):
        z = b[0] + b[1] * x
        return float(np.sum(np.logaddexp(0, z) - y * z))
    return float(minimize(nll, [0.0, 1.0], method="BFGS", options={"gtol": 1e-10}).x[1])


def test_the_slope_matches_an_independent_maximum_likelihood_fit():
    rng = random.Random(5)
    sc = []
    for i in range(3000):
        shrunk = rng.uniform(0.3, 0.7)
        true = 1 / (1 + np.exp(-(0.1 + 1.4 * np.log(shrunk / (1 - shrunk)))))
        sc.append(scored_leg(i, m=0.5, y=1 if rng.random() < true else 0, shrunk=shrunk))
    assert passtest.calibration_slope(sc) == pytest.approx(_scipy_slope(sc), abs=1e-4)
    assert 1.0 < passtest.calibration_slope(sc) < 1.9


def test_a_calibrated_forecast_has_slope_near_one():
    rng = random.Random(11)
    sc = [scored_leg(i, 0.5, 0, 0.5) | {"shrunk": s} for i, s in enumerate(rng.uniform(0.2, 0.8) for _ in range(6000))]
    for s in sc:
        s["y"] = 1 if rng.random() < s["shrunk"] else 0
    assert passtest.calibration_slope(sc) == pytest.approx(1.0, abs=0.2)


def test_the_slope_is_none_when_it_is_undefined():
    assert passtest.calibration_slope([scored_leg(i, 0.5, i % 2, 0.5) for i in range(20)]) is None          # no variance
    assert passtest.calibration_slope([scored_leg(1, 0.5, 1, 0.6)]) is None                                  # too few
    separated = [scored_leg(i, 0.5, 1 if i >= 10 else 0, 0.2 if i < 10 else 0.8) for i in range(20)]
    assert passtest.calibration_slope(separated) is None                                                     # perfect separation


# ---- the two pass conditions at their boundaries --------------------------------------------------------------------

@pytest.mark.parametrize("gain,lower,slope,expected", [
    (0.001, 1e-9, 1.0, "PASS"),              # a gain of exactly 0.001 passes
    (0.00099, 1e-9, 1.0, "FAIL"),
    (0.01, 0.0, 1.0, "FAIL"),                # the lower bound must be strictly above 0
    (0.01, -1e-9, 1.0, "FAIL"),
    (0.01, 1e-9, 0.8, "PASS"),               # slope bounds are inclusive
    (0.01, 1e-9, 1.2, "PASS"),
    (0.01, 1e-9, 0.7999, "FAIL"),
    (0.01, 1e-9, 1.2001, "FAIL"),
    (0.01, 1e-9, None, "FAIL"),              # an undefined slope cannot pass
    (None, None, None, "FAIL"),
])
def test_verdict_boundaries(gain, lower, slope, expected):
    assert passtest.verdict(gain, lower, slope)["verdict"] == expected


def test_a_gain_under_the_threshold_fails_even_with_perfect_calibration_and_a_positive_bound():
    v = passtest.verdict(0.0005, 0.0001, 1.0)
    assert v["verdict"] == "FAIL" and [c["passed"] for c in v["criteria"]] == [False, True, True]


# ---- the judgement point and the frozen verdict ---------------------------------------------------------------------

def _scored(n, per_game=2, edge=0.03):
    """n scored legs where the shrunk probability genuinely beats the market (a clear PASS if judged)."""
    rng = random.Random(21)
    out = []
    for i in range(n):
        true = rng.uniform(0.4, 0.6)
        shrunk = true
        out.append(scored_leg(i // per_game, m=true - edge * (1 if i % 2 else -1), y=1 if rng.random() < true else 0, shrunk=shrunk))
    return out


def test_1999_scored_legs_are_still_collecting_and_2000_are_judged():
    d = passtest.decide(_scored(1999), None, "2026-12-01", resamples=200)
    assert d["status"] == "collecting" and d["scored"] == 1999 and d["need_scored"] == 2000 and "verdict" not in d
    j = passtest.decide(_scored(2000), None, "2026-12-08", resamples=200)
    assert j["status"] == "judged" and j["judged_on"] == "2026-12-08" and j["verdict"] in ("PASS", "FAIL")
    assert [c["name"] for c in j["criteria"]] == list(passtest.CRITERIA)


def test_collecting_reports_running_numbers_but_cannot_pass_or_fail():
    d = passtest.decide(_scored(300), None, "2026-12-01", resamples=200)
    assert d["status"] == "collecting" and d["running"]["n"] == 300 and d["running"]["gain"] is not None
    assert passtest.decide([], None, "2026-12-01")["running"] is None


def test_a_judged_verdict_is_carried_forward_unchanged_whatever_the_new_data_say():
    frozen = {"status": "judged", "judged_on": "2026-11-17", "scored": 2100, "need_scored": 2000, "running": None,
              "verdict": "FAIL", "criteria": []}
    assert passtest.decide(_scored(4000), frozen, "2027-01-05", resamples=200) is frozen
    assert passtest.decide([], frozen, "2027-01-05") is frozen


def test_a_previous_collecting_decision_does_not_freeze_anything():
    prev = {"status": "collecting", "scored": 10, "need_scored": 2000, "running": None}
    assert passtest.decide(_scored(2000), prev, "2026-12-08", resamples=200)["status"] == "judged"


def test_evaluate_returns_both_briers_the_gain_the_bound_and_the_slope():
    ev = passtest.evaluate(_scored(2000), resamples=300)
    assert ev["n"] == 2000 and ev["games"] == 1000
    assert ev["gain"] == pytest.approx(ev["brier_market"] - ev["brier_shrunk"])
    assert ev["lower"] < ev["gain"] and ev["slope"] is not None
