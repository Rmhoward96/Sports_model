from sportsmodel.cfb.priors import (
    PriorWeights,
    load_weights,
    preseason_rating,
    season_features_z,
    zscore,
)


def test_zscore_centers_and_scales():
    z = zscore({"A": 10.0, "B": 20.0, "C": 30.0})
    assert abs(z["B"]) < 1e-9                       # mean maps to 0
    assert z["C"] > 0 and z["A"] < 0


def test_zscore_zero_std_returns_zeros():
    z = zscore({"A": 5.0, "B": 5.0, "C": 5.0})
    assert z == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_preseason_rating_is_sp_base_plus_weighted_adjustments():
    w = PriorWeights(sp_scale=25.0, sp_offset=1500.0, w_portal=10.0, w_coach=-8.0,
                     w_qb=12.0, w_starters=6.0, w_sos_prior=0.0, w_sos_shift=0.0)
    row = {"sp_rating": 1.0, "coach_first_year": True, "qb_returning": True}
    z = {"portal_net": 2.0, "returning_starters": 1.0, "prior_sos": 0.0, "forward_sos_shift": 0.0}
    r = preseason_rating(row, z, w)
    # 1500 + 25*1.0 + 10*2.0 + (-8)*1 + 12*(+1) + 6*1.0
    assert r == 1500.0 + 25.0 + 20.0 - 8.0 + 12.0 + 6.0


def test_qb_returning_none_is_neutral_not_a_penalty():
    # Missing /player/returning data (qb_returning=None) must contribute 0,
    # not be treated the same as a confirmed-departed QB (-w_qb).
    w = PriorWeights(sp_scale=25.0, sp_offset=1500.0, w_qb=12.0)
    z = {"portal_net": 0.0, "returning_starters": 0.0, "prior_sos": 0.0, "forward_sos_shift": 0.0}
    base = {"sp_rating": 0.0, "coach_first_year": False}

    r_true = preseason_rating({**base, "qb_returning": True}, z, w)
    r_false = preseason_rating({**base, "qb_returning": False}, z, w)
    r_none = preseason_rating({**base, "qb_returning": None}, z, w)

    assert r_true == 1500.0 + 12.0
    assert r_false == 1500.0 - 12.0
    assert r_none == 1500.0


def test_zero_weights_reduce_to_sp_only():
    w = PriorWeights(sp_scale=25.0, sp_offset=1500.0)   # rest default 0
    r = preseason_rating({"sp_rating": 2.0, "coach_first_year": False, "qb_returning": True},
                         {"portal_net": 5.0, "returning_starters": 5.0, "prior_sos": 5.0, "forward_sos_shift": 5.0}, w)
    assert r == 1500.0 + 50.0


def test_load_weights_defaults_to_identity_sp_map_when_missing(tmp_path):
    w = load_weights(tmp_path / "does_not_exist.json")
    assert w == PriorWeights()
    assert w.sp_scale == 1.0
    assert w.sp_offset == 1500.0
    assert w.w_portal == 0.0


def test_load_weights_reads_json(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text('{"sp_scale": 25.0, "sp_offset": 1500.0, "w_portal": 10.0}')
    w = load_weights(p)
    assert w.sp_scale == 25.0
    assert w.w_portal == 10.0
    assert w.w_coach == 0.0  # unspecified -> default


def test_season_features_z_middle_team_is_zero_and_none_maps_to_zero():
    rows = [
        {"team_espn_id": "1", "portal_net": -5.0, "returning_starters": 3, "prior_sos": 1.0, "forward_sos_shift": 0.5},
        {"team_espn_id": "2", "portal_net": 0.0, "returning_starters": 5, "prior_sos": 2.0, "forward_sos_shift": 1.0},
        {"team_espn_id": "3", "portal_net": 5.0, "returning_starters": 7, "prior_sos": 3.0, "forward_sos_shift": 1.5},
    ]
    z = season_features_z(rows)
    assert abs(z["2"]["portal_net"]) < 1e-9
    assert z["3"]["portal_net"] > 0 and z["1"]["portal_net"] < 0

    # None values are excluded from mean/std and map to 0.0 for that team.
    rows_with_none = [
        {"team_espn_id": "1", "portal_net": -5.0, "returning_starters": 3, "prior_sos": None, "forward_sos_shift": None},
        {"team_espn_id": "2", "portal_net": None, "returning_starters": 5, "prior_sos": 2.0, "forward_sos_shift": 1.0},
        {"team_espn_id": "3", "portal_net": 5.0, "returning_starters": 7, "prior_sos": 3.0, "forward_sos_shift": 1.5},
    ]
    z2 = season_features_z(rows_with_none)
    assert z2["2"]["portal_net"] == 0.0
    assert z2["1"]["prior_sos"] == 0.0
    assert z2["1"]["forward_sos_shift"] == 0.0
