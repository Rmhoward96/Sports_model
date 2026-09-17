"""Tests for NFL player prop aggregation and disagreement metric."""
import numpy as np
import pytest

from sportsmodel.sim.nfl.aggregate import disagreement, nfl_player_prop_dists
from sportsmodel.sim.nfl.spec import NflGameSims


def test_nfl_player_prop_dists_pass_yds_pmf_sums_to_one():
    """PMF for pass_yds should sum to ~1."""
    sims = NflGameSims(
        home_score=np.array([20, 21, 19]),
        away_score=np.array([17, 18, 20]),
        player_stats={
            "qb1": {
                "pass_yds": np.array([250, 280, 200]),
                "rush_yds": np.array([0, 0, 0]),
                "rec_yds": np.array([0, 0, 0]),
                "receptions": np.array([0, 0, 0]),
                "td": np.array([0, 1, 0]),
            }
        },
    )
    market_max = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 300, "receptions": 20}

    dists = nfl_player_prop_dists(sims, market_max)

    pmf = dists["qb1"]["pass_yds"]["pmf"]
    assert abs(sum(pmf) - 1.0) < 0.01


def test_nfl_player_prop_dists_pass_yds_mean_matches():
    """Mean from _pmf_mean should match array mean."""
    arr = np.array([250, 280, 200])
    sims = NflGameSims(
        home_score=np.array([20, 21, 19]),
        away_score=np.array([17, 18, 20]),
        player_stats={
            "qb1": {
                "pass_yds": arr,
                "rush_yds": np.array([0, 0, 0]),
                "rec_yds": np.array([0, 0, 0]),
                "receptions": np.array([0, 0, 0]),
                "td": np.array([0, 1, 0]),
            }
        },
    )
    market_max = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 300, "receptions": 20}

    dists = nfl_player_prop_dists(sims, market_max)

    mean_from_dist = dists["qb1"]["pass_yds"]["mean"]
    expected_mean = float(np.mean(arr))
    assert abs(mean_from_dist - expected_mean) < 1e-6


def test_nfl_player_prop_dists_anytime_td_pmf_form():
    """anytime_td should have pmf=[1-p, p] where p=P(td>=1)."""
    arr = np.array([0, 1, 2, 0, 1])
    sims = NflGameSims(
        home_score=np.array([20, 21, 19, 17, 18]),
        away_score=np.array([17, 18, 20, 16, 19]),
        player_stats={
            "wr1": {
                "pass_yds": np.array([0, 0, 0, 0, 0]),
                "rush_yds": np.array([0, 0, 0, 0, 0]),
                "rec_yds": np.array([100, 120, 90, 110, 95]),
                "receptions": np.array([5, 6, 4, 5, 5]),
                "td": arr,
            }
        },
    )
    market_max = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 300, "receptions": 20}

    dists = nfl_player_prop_dists(sims, market_max)

    p_td_at_least_one = float(np.mean(arr >= 1))
    pmf = dists["wr1"]["anytime_td"]["pmf"]

    assert len(pmf) == 2
    assert pmf[0] == pytest.approx(1 - p_td_at_least_one)
    assert pmf[1] == pytest.approx(p_td_at_least_one)
    assert 0 <= p_td_at_least_one <= 1


def test_nfl_player_prop_dists_anytime_td_mean():
    """anytime_td mean should match array mean."""
    arr = np.array([0, 1, 2, 0, 1])
    sims = NflGameSims(
        home_score=np.array([20, 21, 19, 17, 18]),
        away_score=np.array([17, 18, 20, 16, 19]),
        player_stats={
            "wr1": {
                "pass_yds": np.array([0, 0, 0, 0, 0]),
                "rush_yds": np.array([0, 0, 0, 0, 0]),
                "rec_yds": np.array([100, 120, 90, 110, 95]),
                "receptions": np.array([5, 6, 4, 5, 5]),
                "td": arr,
            }
        },
    )
    market_max = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 300, "receptions": 20}

    dists = nfl_player_prop_dists(sims, market_max)

    mean_from_dist = dists["wr1"]["anytime_td"]["mean"]
    expected_mean = float(np.mean(arr))
    assert abs(mean_from_dist - expected_mean) < 1e-6


def test_nfl_player_prop_dists_multiple_markets():
    """All markets should be present in output."""
    sims = NflGameSims(
        home_score=np.array([20, 21, 19]),
        away_score=np.array([17, 18, 20]),
        player_stats={
            "rb1": {
                "pass_yds": np.array([0, 0, 0]),
                "rush_yds": np.array([100, 120, 90]),
                "rec_yds": np.array([50, 60, 40]),
                "receptions": np.array([3, 4, 2]),
                "td": np.array([0, 1, 0]),
            }
        },
    )
    market_max = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 300, "receptions": 20}

    dists = nfl_player_prop_dists(sims, market_max)

    assert "rb1" in dists
    assert "pass_yds" in dists["rb1"]
    assert "rush_yds" in dists["rb1"]
    assert "rec_yds" in dists["rb1"]
    assert "receptions" in dists["rb1"]
    assert "anytime_td" in dists["rb1"]


def test_nfl_player_prop_dists_multiple_players():
    """Function should iterate over all players in player_stats."""
    sims = NflGameSims(
        home_score=np.array([20, 21, 19]),
        away_score=np.array([17, 18, 20]),
        player_stats={
            "qb1": {
                "pass_yds": np.array([250, 280, 200]),
                "rush_yds": np.array([10, 15, 5]),
                "rec_yds": np.array([0, 0, 0]),
                "receptions": np.array([0, 0, 0]),
                "td": np.array([2, 2, 1]),
            },
            "wr1": {
                "pass_yds": np.array([0, 0, 0]),
                "rush_yds": np.array([0, 0, 0]),
                "rec_yds": np.array([100, 120, 90]),
                "receptions": np.array([5, 6, 4]),
                "td": np.array([1, 0, 1]),
            },
        },
    )
    market_max = {"pass_yds": 400, "rush_yds": 200, "rec_yds": 300, "receptions": 20}

    dists = nfl_player_prop_dists(sims, market_max)

    assert len(dists) == 2
    assert "qb1" in dists
    assert "wr1" in dists


def test_disagreement_basic():
    """disagreement(0.6, 0.5) should equal 0.1."""
    assert disagreement(0.6, 0.5) == pytest.approx(0.1)


def test_disagreement_commutative():
    """disagreement should be symmetric."""
    assert disagreement(0.6, 0.5) == pytest.approx(disagreement(0.5, 0.6))


def test_disagreement_identical():
    """disagreement between identical values should be 0."""
    assert disagreement(0.5, 0.5) == pytest.approx(0.0)


def test_disagreement_range():
    """disagreement should handle probabilities in [0, 1]."""
    assert disagreement(0.0, 1.0) == pytest.approx(1.0)
    assert disagreement(0.0, 0.0) == pytest.approx(0.0)
    assert disagreement(1.0, 1.0) == pytest.approx(0.0)
    assert disagreement(0.25, 0.75) == pytest.approx(0.5)
