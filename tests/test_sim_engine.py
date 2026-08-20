import numpy as np
from sportsmodel.sim.engine import GameSims, home_win_prob, total_pmf, pred_scores, stat_pmf


def _toy():
    return GameSims(
        home_score=np.array([3, 5, 2, 4]),
        away_score=np.array([2, 5, 4, 1]),
        batter_stats={}, pitcher_stats={},
    )


def test_home_win_prob_counts_wins_only():
    # ties (5-5) do not count as home wins; sim resolves ties via extra innings,
    # but the aggregator must not credit an equal-score row as a win.
    assert home_win_prob(_toy()) == 0.5  # wins: 3>2, 4>1 ; losses: 2<4 ; tie: 5=5 -> excluded


def test_total_pmf_sums_to_one():
    p = total_pmf(_toy(), max_total=12)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[5] == 0.5  # totals: 5,10,6,5 -> total 5 appears twice of four


def test_stat_pmf():
    p = stat_pmf(np.array([0, 1, 1, 2]), max_k=3)
    assert p == [0.25, 0.5, 0.25, 0.0]
