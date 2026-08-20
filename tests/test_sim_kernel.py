import numpy as np

from sportsmodel.sim.engine import GameSims
from sportsmodel.sim.mlb import kernel as K
from sportsmodel.sim.mlb.advancement import AdvancementTable

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
    return K.GameSpec(bs, aw, K.Pitcher(1, 24.0, 3.0), K.Pitcher(2, 24.0, 3.0))


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
