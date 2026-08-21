import numpy as np
from sportsmodel.sim.engine import GameSims, home_win_prob, total_pmf, pred_scores, stat_pmf, player_prop_dists


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


def test_player_prop_dists_shapes():
    sims = GameSims(
        home_score=np.array([1, 2]), away_score=np.array([0, 1]),
        batter_stats={100: {"hits": np.array([0, 2]), "total_bases": np.array([0, 3]),
                            "hr": np.array([0, 1]), "runs": np.array([0, 1]),
                            "rbi": np.array([0, 2]), "hrr": np.array([0, 5])}},
        pitcher_stats={1: {"k": np.array([5, 7]), "hits": np.array([4, 6]),
                           "outs": np.array([15, 18])}},
    )
    out = player_prop_dists(sims, {"hits": 5, "total_bases": 8, "hrr": 12,
                                   "pitcher_ks": 12, "hits_allowed": 12, "outs_recorded": 27})
    hr = out[100]["home_run"]["pmf"]
    assert len(hr) == 2 and abs(sum(hr) - 1.0) < 1e-9   # [P(0), P(>=1)]
    assert out[100]["hits"]["pmf"][0] == 0.5             # one sim had 0 hits
    assert out[1]["outs_recorded"]["mean"] == 16.5
