import math
from sportsmodel.model.distributions import (
    normal_to_pmf, normal_to_margin_pmf, prob_cover, prob_over_dist, normal_sf,
)

def test_normal_to_pmf_sums_and_centers():
    pmf = normal_to_pmf(45.0, 10.0, 120)
    assert math.isclose(sum(pmf), 1.0, abs_tol=1e-6)
    mean = sum(k * p for k, p in enumerate(pmf))
    assert abs(mean - 45.0) < 0.2
    # P(total > 44) should match the normal survival within discretization error
    assert abs(prob_over_dist({"kind": "pmf", "pmf": pmf}, 44) - normal_sf(44, 45, 10)) < 0.02

def test_normal_to_margin_pmf_winprob_matches_normal():
    d = normal_to_margin_pmf(3.0, 13.2, 75)
    assert d["kind"] == "margin" and d["offset"] == 75
    assert math.isclose(sum(d["pmf"]), 1.0, abs_tol=1e-6)
    # win_prob = P(margin>0) via prob_cover(dist,0) ~ normal_sf(0, mean, sd)
    assert abs(prob_cover(d, 0.0) - normal_sf(0.0, 3.0, 13.2)) < 0.02

def test_margin_pmf_symmetric_at_zero_mean():
    d = normal_to_margin_pmf(0.0, 13.2, 75)
    # NOTE: prob_cover excludes an exact push (margin==0) per its own strict->
    # comparison contract, and the discretized bin at margin=0 carries real
    # mass for a symmetric zero-mean distribution, so P(margin>0) is
    # (1 - mass_at_0)/2, not exactly 0.5 (the brief's 1e-6 tolerance against
    # 0.5 is mathematically unreachable here). What IS exactly true for a
    # zero-mean symmetric distribution is P(margin>0) == P(margin<0), so
    # assert that symmetry directly and tightly instead.
    p_over = prob_cover(d, 0.0)            # P(margin > 0)
    p_zero = d["pmf"][d["offset"]]         # P(margin == 0)
    p_under = 1.0 - p_over - p_zero        # P(margin < 0)
    assert math.isclose(p_over, p_under, abs_tol=1e-9)   # true symmetry, exact
