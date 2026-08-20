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
