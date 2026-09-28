"""Tests for rank-preserving quantile mapping of sim draws (props ML,
Props-2). Pure synthetic data; no IO, no network."""
from __future__ import annotations

import numpy as np
import pytest

from sportsmodel.model.props_ml.quantile_map import map_draws, map_game_sims
from sportsmodel.sim.nfl.aggregate import nfl_player_prop_dists
from sportsmodel.sim.nfl.spec import NflGameSims


def _avg_ranks(x: np.ndarray) -> np.ndarray:
    """Average (mid) ranks, 0-based, ties share their mean rank."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.corrcoef(_avg_ranks(x), _avg_ranks(y))[0, 1])


def _discrete_normal(mean: float, sd: float, kmax: int) -> np.ndarray:
    k = np.arange(kmax + 1)
    p = np.exp(-0.5 * ((k - mean) / sd) ** 2)
    return p / p.sum()


def _empirical_pmf(v: np.ndarray, size: int) -> np.ndarray:
    return np.bincount(v, minlength=size)[:size] / len(v)


def _corr_normals(n: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    z1 = rng.standard_normal(n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return z1, z2


# ----------------------------------------------------------------- map_draws


def test_mapped_draws_match_target_pmf():
    n = 20000
    draws = np.random.default_rng(1).gamma(2.0, 30.0, n).round()  # tied int-like
    target = _discrete_normal(62.0, 25.0, 200)
    out = map_draws(draws, target, np.random.default_rng(2))
    tv = 0.5 * np.abs(_empirical_pmf(out, len(target)) - target).sum()
    assert tv < 0.02


def test_output_is_int_same_length_and_in_support():
    draws = np.random.default_rng(3).poisson(4.0, 5000)
    target = np.array([0.1, 0.0, 0.3, 0.6])
    out = map_draws(draws, target, np.random.default_rng(4))
    assert out.shape == draws.shape
    assert np.issubdtype(out.dtype, np.integer)
    assert out.min() >= 0 and out.max() <= len(target) - 1
    assert not np.any(out == 1)  # zero-probability bin never produced


def test_mapping_is_monotone_in_draws():
    draws = np.random.default_rng(5).poisson(3.0, 4000)
    out = map_draws(draws, _discrete_normal(40.0, 12.0, 100), np.random.default_rng(6))
    # a strictly larger draw never maps to a smaller value
    for lo in np.unique(draws)[:-1]:
        assert out[draws == lo].max() <= out[draws > lo].min()


def test_spearman_between_correlated_vectors_is_preserved():
    n = 20000
    z1, z2 = _corr_normals(n, 0.6, seed=7)
    a = np.clip(np.round(60 + 25 * z1), 0, None)  # sim-like yards (ties)
    b = np.clip(np.round(45 + 20 * z2), 0, None)
    before = _spearman(a, b)
    rng = np.random.default_rng(8)
    ma = map_draws(a, _discrete_normal(75.0, 30.0, 250), rng)
    mb = map_draws(b, _discrete_normal(38.0, 18.0, 200), rng)
    after = _spearman(ma, mb)
    assert abs(after - before) < 0.02


def test_ties_broken_deterministically_for_fixed_seed():
    draws = np.random.default_rng(9).poisson(1.5, 3000)  # heavy ties
    target = _discrete_normal(3.0, 1.5, 10)
    o1 = map_draws(draws, target, np.random.default_rng(42))
    o2 = map_draws(draws, target, np.random.default_rng(42))
    o3 = map_draws(draws, target, np.random.default_rng(43))
    np.testing.assert_array_equal(o1, o2)
    assert not np.array_equal(o1, o3)  # tie-break genuinely uses the rng


def test_binary_target_mean():
    draws = np.random.default_rng(10).poisson(0.4, 20000)
    out = map_draws(draws, np.array([0.7, 0.3]), np.random.default_rng(11))
    assert abs(out.mean() - 0.3) < 0.01


def test_unnormalized_target_is_normalized():
    draws = np.arange(1000)
    out = map_draws(draws, np.array([2.0, 2.0]), np.random.default_rng(0))
    assert out.sum() == 500
    assert np.all(out[:500] == 0) and np.all(out[500:] == 1)


def test_empty_draws():
    out = map_draws(np.array([], dtype=np.int64), np.array([0.5, 0.5]), np.random.default_rng(0))
    assert out.shape == (0,) and np.issubdtype(out.dtype, np.integer)


@pytest.mark.parametrize(
    "bad",
    [np.array([]), np.array([0.5, -0.1, 0.6]), np.array([0.0, 0.0]),
     np.array([0.5, np.nan]), np.array([[0.5, 0.5]])],
)
def test_invalid_target_raises(bad):
    with pytest.raises(ValueError):
        map_draws(np.arange(10), bad, np.random.default_rng(0))


# ------------------------------------------------------------- map_game_sims


def _sims(n: int = 20000, seed: int = 20) -> NflGameSims:
    rng = np.random.default_rng(seed)
    z1, z2 = _corr_normals(n, 0.5, seed=seed + 1)
    qb_yds = np.clip(np.round(230 + 60 * z1), 0, None).astype(np.int64)
    wr_yds = np.clip(np.round(60 + 25 * z2), 0, None).astype(np.int64)
    return NflGameSims(
        home_score=rng.poisson(23, n).astype(np.int64),
        away_score=rng.poisson(20, n).astype(np.int64),
        player_stats={
            "qb": {"pass_yds": qb_yds, "pass_tds": rng.poisson(1.6, n).astype(np.int64),
                   "td": rng.poisson(0.1, n).astype(np.int64)},
            "wr": {"rec_yds": wr_yds, "receptions": rng.poisson(5, n).astype(np.int64),
                   "td": rng.poisson(0.45, n).astype(np.int64)},
            "rb": {"rush_yds": rng.poisson(55, n).astype(np.int64),
                   "td": rng.poisson(0.5, n).astype(np.int64)},
        },
    )


def test_map_game_sims_returns_new_object_and_leaves_input_untouched():
    sims = _sims()
    snapshot = {p: {m: a.copy() for m, a in s.items()} for p, s in sims.player_stats.items()}
    targets = {"wr": {"rec_yds": _discrete_normal(80.0, 30.0, 250)}}
    out = map_game_sims(sims, targets, np.random.default_rng(0))
    assert out is not sims
    assert out.player_stats is not sims.player_stats
    assert out.player_stats["wr"] is not sims.player_stats["wr"]
    for p, s in snapshot.items():
        for m, a in s.items():
            np.testing.assert_array_equal(sims.player_stats[p][m], a)
    # untouched players/stats keep their arrays; scores carried through
    assert out.player_stats["qb"]["pass_yds"] is sims.player_stats["qb"]["pass_yds"]
    assert out.player_stats["wr"]["receptions"] is sims.player_stats["wr"]["receptions"]
    assert out.home_score is sims.home_score and out.away_score is sims.away_score
    # mapped stat changed, dtype kept
    mapped = out.player_stats["wr"]["rec_yds"]
    assert mapped.dtype == sims.player_stats["wr"]["rec_yds"].dtype
    assert abs(mapped.mean() - 80.0) < 1.0


def test_map_game_sims_targets_match_and_cross_player_correlation_kept():
    sims = _sims()
    before = _spearman(sims.player_stats["qb"]["pass_yds"], sims.player_stats["wr"]["rec_yds"])
    tq = _discrete_normal(260.0, 70.0, 500)
    tw = _discrete_normal(70.0, 28.0, 250)
    out = map_game_sims(sims, {"qb": {"pass_yds": tq}, "wr": {"rec_yds": tw}},
                        np.random.default_rng(1))
    q, w = out.player_stats["qb"]["pass_yds"], out.player_stats["wr"]["rec_yds"]
    assert 0.5 * np.abs(_empirical_pmf(q, len(tq)) - tq).sum() < 0.02
    assert 0.5 * np.abs(_empirical_pmf(w, len(tw)) - tw).sum() < 0.02
    assert abs(_spearman(q, w) - before) < 0.02


def test_map_game_sims_anytime_td_ruling():
    n = 8
    td = np.array([0, 0, 0, 0, 2, 1, 3, 0], dtype=np.int64)
    sims = NflGameSims(np.zeros(n, np.int64), np.zeros(n, np.int64),
                       {"rb": {"td": td.copy(), "rush_yds": np.arange(n, dtype=np.int64)}})
    # raise P(>=1) from 3/8 to 5/8: two original zeros become 1, originals kept
    up = map_game_sims(sims, {"rb": {"anytime_td": np.array([3 / 8, 5 / 8])}},
                       np.random.default_rng(0)).player_stats["rb"]["td"]
    assert up.dtype == np.int64
    assert np.sum(up >= 1) == 5
    np.testing.assert_array_equal(up[[4, 5, 6]], [2, 1, 3])  # mapped 1 & orig>=1: keep count
    assert set(up[[0, 1, 2, 3, 7]].tolist()) <= {0, 1}  # mapped 1 & orig 0 -> 1
    # lower P(>=1) to 1/8: mapped-0 rows become 0, the survivor keeps its count
    down = map_game_sims(sims, {"rb": {"anytime_td": np.array([7 / 8, 1 / 8])}},
                         np.random.default_rng(0)).player_stats["rb"]["td"]
    assert np.sum(down >= 1) == 1
    assert np.all(down[[0, 1, 2, 3, 7]] == 0)
    assert down[down >= 1][0] in (1, 2, 3)
    np.testing.assert_array_equal(sims.player_stats["rb"]["td"], td)  # input untouched


def test_map_game_sims_anytime_td_feeds_aggregator():
    sims = _sims()
    out = map_game_sims(sims, {"wr": {"anytime_td": np.array([0.55, 0.45])},
                               "rb": {"anytime_td": np.array([0.8, 0.2])}},
                        np.random.default_rng(2))
    dists = nfl_player_prop_dists(out, market_max={})
    assert abs(dists["wr"]["anytime_td"]["pmf"][1] - 0.45) < 0.001
    assert abs(dists["rb"]["anytime_td"]["pmf"][1] - 0.2) < 0.001
    # mean TD count stays >= P(>=1) (multi-TD games preserved)
    assert dists["wr"]["anytime_td"]["mean"] >= 0.45


def test_map_game_sims_anytime_td_requires_two_bins():
    with pytest.raises(ValueError):
        map_game_sims(_sims(100), {"wr": {"anytime_td": np.array([0.5, 0.3, 0.2])}},
                      np.random.default_rng(0))


def test_map_game_sims_ignores_absent_player_and_stat():
    sims = _sims(500)
    targets = {"ghost": {"rec_yds": np.array([0.5, 0.5])},
               "rb": {"pass_yds": np.array([0.5, 0.5])},  # rb has no pass_yds
               "qb": {"rush_att": np.array([0.5, 0.5])}}
    out = map_game_sims(sims, targets, np.random.default_rng(0))
    assert "ghost" not in out.player_stats
    assert "pass_yds" not in out.player_stats["rb"]
    assert "rush_att" not in out.player_stats["qb"]
    for p, s in sims.player_stats.items():
        for m, a in s.items():
            assert out.player_stats[p][m] is a


def test_map_game_sims_deterministic_for_fixed_rng():
    sims = _sims(3000)
    targets = {"qb": {"pass_tds": _discrete_normal(2.0, 1.0, 6)},
               "wr": {"anytime_td": np.array([0.6, 0.4])}}
    a = map_game_sims(sims, targets, np.random.default_rng(5))
    b = map_game_sims(sims, targets, np.random.default_rng(5))
    for p in ("qb", "wr"):
        for m in sims.player_stats[p]:
            np.testing.assert_array_equal(a.player_stats[p][m], b.player_stats[p][m])


# ------------------------------------- Task 9 rulings: td ranking, name checks


def test_anytime_td_demotion_takes_the_lowest_counts_first():
    # 6 scoring sims (counts 1,1,1,2,3,3); P(>=1) 6/8 -> 2/8: the survivors must
    # be the two 3-TD sims (ranked on the ORIGINAL counts, not the indicator)
    td = np.array([0, 3, 1, 0, 2, 1, 3, 1], dtype=np.int64)
    sims = NflGameSims(np.zeros(8, np.int64), np.zeros(8, np.int64), {"wr": {"td": td.copy()}})
    for seed in range(20):
        out = map_game_sims(sims, {"wr": {"anytime_td": np.array([6 / 8, 2 / 8])}},
                            np.random.default_rng(seed)).player_stats["wr"]["td"]
        np.testing.assert_array_equal(out, [0, 3, 0, 0, 0, 0, 3, 0])
    # P(>=1) 6/8 -> 3/8: both 3s survive plus the 2 (never a 1 over the 2)
    out = map_game_sims(sims, {"wr": {"anytime_td": np.array([5 / 8, 3 / 8])}},
                        np.random.default_rng(0)).player_stats["wr"]["td"]
    np.testing.assert_array_equal(out, [0, 3, 0, 0, 2, 0, 3, 0])


def test_unknown_market_name_raises_even_for_an_absent_player():
    sims = _sims(200)
    with pytest.raises(ValueError, match="rec_yards"):
        map_game_sims(sims, {"wr": {"rec_yards": np.array([0.5, 0.5])}},
                      np.random.default_rng(0))
    with pytest.raises(ValueError, match="bogus"):
        map_game_sims(sims, {"ghost": {"bogus": np.array([0.5, 0.5])}},
                      np.random.default_rng(0))


def test_td_and_anytime_td_targets_for_one_player_raise():
    with pytest.raises(ValueError, match="anytime_td"):
        map_game_sims(_sims(200), {"wr": {"td": np.array([0.5, 0.3, 0.2]),
                                          "anytime_td": np.array([0.6, 0.4])}},
                      np.random.default_rng(0))


def test_td_count_target_maps_the_counts_directly():
    sims = _sims(20000)
    target = np.array([0.5, 0.3, 0.15, 0.05])
    out = map_game_sims(sims, {"rb": {"td": target}}, np.random.default_rng(1))
    td = out.player_stats["rb"]["td"]
    assert 0.5 * np.abs(_empirical_pmf(td, 4) - target).sum() < 0.01


def test_absent_targets_consume_no_rng():
    sims = _sims(300)
    rng = np.random.default_rng(9)
    map_game_sims(sims, {"ghost": {"rec_yds": np.array([0.5, 0.5])},
                         "rb": {"pass_yds": np.array([0.5, 0.5])}}, rng)
    assert rng.random() == np.random.default_rng(9).random()
