import math

import pytest

from sportsmodel.model import rates

LEAGUE = {
    "p_bb": 0.085,
    "p_k": 0.225,
    "p_1b": 0.145,
    "p_2b": 0.045,
    "p_3b": 0.004,
    "p_hr": 0.033,
    "p_out": 0.463,
}


def test_league_vs_league_returns_league():
    # A league-average batter vs a league-average pitcher must reproduce league rates.
    vec = rates.matchup_vector(LEAGUE, LEAGUE, LEAGUE)
    for o in rates.OUTCOMES:
        assert vec[o] == pytest.approx(LEAGUE[o], abs=1e-9)


def test_matchup_vector_sums_to_one():
    batter = {**LEAGUE, "p_hr": 0.06, "p_k": 0.18}
    pitcher = {**LEAGUE, "p_k": 0.28}
    vec = rates.matchup_vector(batter, pitcher, LEAGUE)
    assert sum(vec.values()) == pytest.approx(1.0, abs=1e-9)


def test_odds_ratio_elevates_shared_strength():
    # A HR-prone batter vs a HR-prone pitcher should exceed league HR rate.
    batter = {**LEAGUE, "p_hr": 0.07}
    pitcher = {**LEAGUE, "p_hr": 0.05}
    vec = rates.matchup_vector(batter, pitcher, LEAGUE)
    assert vec["p_hr"] > LEAGUE["p_hr"]


def test_shrink_regresses_small_samples():
    # 1 HR in 10 PA should regress hard toward a 0.03 prior, not read 0.10.
    est = rates.shrink(x=1, n=10, prior=0.03, k=rates.DEFAULT_K["p_hr"])
    assert est < 0.05
    # Large samples barely move.
    est_big = rates.shrink(x=60, n=600, prior=0.03, k=rates.DEFAULT_K["p_hr"])
    assert math.isclose(est_big, 60 / 600, abs_tol=0.02)
