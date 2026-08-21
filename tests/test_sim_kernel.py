import glob

import duckdb
import numpy as np
import pytest

from sportsmodel import transforms
from sportsmodel.model import game as gamemodel
from sportsmodel.sim.engine import GameSims
from sportsmodel.sim.mlb import kernel as K
from sportsmodel.sim.mlb.advancement import AdvancementTable
from sportsmodel.sim.mlb.build_advancement import build_advancement_table

_EMPTY_ADV = AdvancementTable.from_rows([])


def test_home_run_bases_loaded_scores_four():
    st = K.BaseState(first=3, second=4, third=5)
    runs, outs = K.resolve_pa(st, batter_idx=6, outcome=K.HR, adv=_EMPTY_ADV, u=0.0)
    assert runs == 4 and outs == 0
    assert st.occ() == 0  # bases cleared


def test_walk_forces_only():
    st = K.BaseState(first=1, second=-1, third=-1)
    runs, outs = K.resolve_pa(st, batter_idx=2, outcome=K.BB, adv=_EMPTY_ADV, u=0.0)
    assert runs == 0 and outs == 0
    assert st.first == 2 and st.second == 1  # batter to 1st, forced runner to 2nd


def test_strikeout_is_pure_out():
    st = K.BaseState(first=1, second=-1, third=-1)
    runs, outs = K.resolve_pa(st, batter_idx=2, outcome=K.K, adv=_EMPTY_ADV, u=0.0)
    assert runs == 0 and outs == 1
    assert st.first == 1  # unchanged


def test_single_from_table_empty_bases():
    adv = AdvancementTable.from_rows(
        [{"outcome": "p_1b", "occ": 0, "end_occ": 1, "runs": 0, "prob": 1.0}])
    st = K.BaseState(-1, -1, -1)
    runs, outs = K.resolve_pa(st, batter_idx=7, outcome=K.S, adv=adv, u=0.5)
    assert runs == 0 and outs == 0
    assert st.first == 7 and st.second == -1  # batter on first


def test_sample_outcome_cumulative():
    vec = {"p_bb": 0.1, "p_k": 0.2, "p_1b": 0.2, "p_2b": 0.05,
           "p_3b": 0.02, "p_hr": 0.03, "p_out": 0.4}
    assert K.sample_outcome(vec, 0.05) == K.BB      # first 0.10
    assert K.sample_outcome(vec, 0.25) == K.K       # 0.10-0.30
    assert K.sample_outcome(vec, 0.35) == K.S       # 0.30-0.50
    assert K.sample_outcome(vec, 0.99) == K.OUT_INPLAY  # last bucket


def test_outcome_codes_distinct():
    codes = {K.BB, K.K, K.S, K.D, K.T, K.HR, K.OUT_INPLAY}
    assert len(codes) == 7


def _flat_vec(**over):
    v = {"p_bb": 0.08, "p_k": 0.22, "p_1b": 0.15, "p_2b": 0.04,
         "p_3b": 0.005, "p_hr": 0.03, "p_out": 0.475 - 0.03}
    v.update(over)
    s = sum(v.values())
    return {k: x / s for k, x in v.items()}


def _spec():
    bs = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(100, 109)]
    aw = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(200, 209)]
    return K.GameSpec(bs, aw, K.Pitcher(1, 16.0, 5.0), K.Pitcher(2, 16.0, 5.0))


def test_simulate_returns_gamesims_with_right_shapes():
    sims = K.simulate_scalar(_spec(), n_sims=50, rng=np.random.default_rng(0))
    assert isinstance(sims, GameSims)
    assert sims.home_score.shape == (50,) and sims.away_score.shape == (50,)
    assert set(sims.batter_stats.keys()) == set(range(100, 109)) | set(range(200, 209))
    assert "hits" in sims.batter_stats[100] and "hrr" in sims.batter_stats[100]
    assert sims.pitcher_stats[1]["outs"].shape == (50,)


def test_no_ties_scores_are_decided():
    sims = K.simulate_scalar(_spec(), n_sims=200, rng=np.random.default_rng(1))
    assert not np.any(sims.home_score == sims.away_score)  # extras resolve ties


def test_reproducible_with_seed():
    a = K.simulate_scalar(_spec(), 30, np.random.default_rng(7))
    b = K.simulate_scalar(_spec(), 30, np.random.default_rng(7))
    assert np.array_equal(a.home_score, b.home_score)


def test_strikeout_outs_not_double_counted():
    """Regression test: strikeouts should count as exactly 1 out, not 2.

    Before the fix, resolve_pa returned outs_added=1 for K, but the game loop
    then added another out via the buggy term (1 if outcome == K else 0),
    resulting in 2 outs per strikeout and half-innings ending early.

    This test uses an all-strikeout pitcher (p_k=1.0) with a starter whose
    outs-recorded hook is requested at avg_outs=999, sd_outs=0 -- the kernel's
    top clamp (min(27.0, ...)) pins the sampled hook at exactly 27 outs, i.e.
    a full 9-inning outing with no early exit. In 9 innings each with 3 outs,
    exactly 27 strikeouts should happen (since no baserunners via strikeouts).
    The starter should record exactly 27 outs, not 54.
    """
    # Create batters with p_k = 1.0 (100% strikeout rate)
    all_k_vec = {"p_bb": 0.0, "p_k": 1.0, "p_1b": 0.0, "p_2b": 0.0,
                 "p_3b": 0.0, "p_hr": 0.0, "p_out": 0.0}
    bs = [K.Batter(pid, all_k_vec, all_k_vec) for pid in range(100, 109)]
    aw = [K.Batter(pid, all_k_vec, all_k_vec) for pid in range(200, 209)]
    # avg_outs=999, sd_outs=0 clamps to a 27-out (9-inning) hook -- starter never
    # leaves early, matching the original avg_bf=999,sd_bf=0 "never leaves" intent.
    spec = K.GameSpec(bs, aw, K.Pitcher(1, avg_outs=999, sd_outs=0), K.Pitcher(2, avg_outs=999, sd_outs=0))

    sims = K.simulate_scalar(spec, n_sims=5, rng=np.random.default_rng(42))

    # Each sim should have exactly 27 outs per starter in 9 complete innings
    # (or more if extras, but always K == outs for an all-strikeout pitcher)
    for i in range(5):
        home_starter_k = sims.pitcher_stats[1]["k"][i]
        home_starter_outs = sims.pitcher_stats[1]["outs"][i]
        away_starter_k = sims.pitcher_stats[2]["k"][i]
        away_starter_outs = sims.pitcher_stats[2]["outs"][i]
        # Each strikeout should contribute exactly 1 out (not 2)
        assert home_starter_outs == home_starter_k, \
            f"Home starter sim {i}: outs={home_starter_outs} but k={home_starter_k} (double-count bug?)"
        assert away_starter_outs == away_starter_k, \
            f"Away starter sim {i}: outs={away_starter_outs} but k={away_starter_k} (double-count bug?)"
        # In a 9-inning game with all strikeouts, each team should have exactly 27 outs
        # (9 innings * 3 outs per inning). If there are extras, outs will be higher.
        # The important check is that outs == k, which this assertion covers above.
        assert home_starter_k >= 27, \
            f"Home starter sim {i}: fewer than 27 strikeouts, expected >= 27"
        assert away_starter_k >= 27, \
            f"Away starter sim {i}: fewer than 27 strikeouts, expected >= 27"


def test_vectorized_matches_scalar_distribution():
    spec = _spec()
    a = K.simulate_scalar(spec, 4000, np.random.default_rng(3))
    b = K.simulate(spec, 4000, np.random.default_rng(3))
    # aggregate means should agree within Monte Carlo noise (~0.1 run)
    assert abs(a.home_score.mean() - b.home_score.mean()) < 0.15
    assert abs(a.away_score.mean() - b.away_score.mean()) < 0.15
    # win prob within a couple points
    from sportsmodel.sim.engine import home_win_prob
    assert abs(home_win_prob(a) - home_win_prob(b)) < 0.03


def test_sim_mean_runs_near_analytic():
    """Cross-check sim mean total runs vs the analytic `game.expected_runs` model,
    using the REAL empirically-derived advancement table (not the empty-table
    fallback).

    Rework rationale (controller-confirmed PLAN DEFECT, not a kernel bug): the
    original version of this test ran `K.simulate` on a spec with no `adv` table
    attached, so the kernel used `AdvancementTable.from_rows([])` -- a degenerate
    fallback where a single with a runner already on first is a no-op (`occ|1`
    doesn't advance) and doubles/triples do nothing, so only HRs and bases-loaded
    walks score. That produced sim_total ~3.75 vs analytic ~8.98 (delta ~5.23),
    which could never satisfy any reasonable tolerance. Wiring in the real table
    (built from the local Statcast backfill) makes the comparison meaningful.
    """
    if not glob.glob(transforms._PARQUET_GLOB, recursive=True):
        pytest.skip("requires local Statcast backfill (parquet not found)")

    transforms.set_cutoff(None)  # use all local data
    con = duckdb.connect(":memory:")
    con.execute("INSTALL json; LOAD json;")
    rows = build_advancement_table(con)
    con.close()
    if not rows:
        pytest.skip("advancement table build produced no rows")
    adv = AdvancementTable.from_rows(rows)

    spec = _spec()  # flat vectors, no park/def; advancement table is player-agnostic
    spec.adv = adv
    sims = K.simulate(spec, 6000, np.random.default_rng(11))
    sim_total = (sims.home_score.mean() + sims.away_score.mean())
    # analytic expected runs for the same flat offense vector, ~38 PA/team
    v = _flat_vec()
    analytic = 2 * gamemodel.expected_runs(v)

    # MEASURED (real table, seed=11): sim_total=7.7275, analytic=8.9806,
    # delta=1.253 (~3.86 runs/team -- baseball-plausible). Re-checked across
    # seeds {1,2,3,11,42,99}: delta ranged 1.14-1.30, so this is a stable result,
    # not a lucky seed. A night-and-day improvement over the degenerate
    # empty-table case (sim_total ~3.75, delta ~5.23). Tolerance of 2.0 gives
    # comfortable headroom over MC noise + legitimate base-out-realism slack,
    # while still failing on the empty-table fallback (~5.23 delta) or a gross
    # scoring regression (e.g. adv table not wired, advancement logic broken).
    assert abs(sim_total - analytic) < 2.0
