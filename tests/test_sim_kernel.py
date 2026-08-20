from sportsmodel.sim.mlb import kernel as K


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
