"""Leak-free preseason prior (fix/cfb-live-ratings, part B).

R_pre = sp_offset + sp_scale * prev_sp + sum(w * z) + w_qb * qb_flag, where
prev_sp is the team's PREVIOUS season's final SP+ (season-1 sp_rating;
missing -> that season's league mean) and z are the season's FBS-wide
z-scored PRESEASON features (returning_pct, recruiting_points, portal_net,
prior_sos). Same-season sp_rating (end-of-season SP+), coach_first_year and
forward_sos_shift carry post-season information and must never reach it.

(The old tests here pinned the leaky definition -- R_pre from the SAME
season's sp_rating plus coach/starters/forward-SoS terms -- and were
rewritten for the new definition.)
"""
import math

import pytest

from sportsmodel.cfb.priors import (
    PRIOR_Z_FEATURES,
    PriorWeights,
    load_weights,
    preseason_rating,
    season_prior_inputs,
    season_priors,
    season_features_z,
    zscore,
)

NAN = float("nan")


def _row(team, season, sp, **kw):
    base = {"season": season, "team_espn_id": team, "sp_rating": sp,
            "returning_pct": 0.5, "recruiting_points": 200.0, "portal_net": NAN,
            "prior_sos": 0.0, "qb_returning": True, "coach_first_year": False,
            "returning_starters": 0.5, "forward_sos_shift": 0.0}
    return {**base, **kw}


def _by_season(rows):
    out = {}
    for r in rows:
        out.setdefault(r["season"], []).append(r)
    return out


def test_zscore_centers_and_scales():
    z = zscore({"A": 10.0, "B": 20.0, "C": 30.0})
    assert abs(z["B"]) < 1e-9
    assert z["C"] > 0 and z["A"] < 0


def test_zscore_accepts_numpy_floats():
    np = __import__("numpy")
    z = zscore({"A": np.float64(10.0), "B": np.float64(20.0), "C": np.float64(30.0)})
    assert abs(z["B"]) < 1e-9 and z["C"] > 0 and z["A"] < 0


def test_zscore_zero_std_returns_zeros():
    assert zscore({"A": 5.0, "B": 5.0, "C": 5.0}) == {"A": 0.0, "B": 0.0, "C": 0.0}


def test_prior_features_are_the_leak_free_set():
    assert set(PRIOR_Z_FEATURES) == {"returning_pct", "recruiting_points", "portal_net",
                                     "prior_sos"}


def test_season_features_z_excludes_nan_and_none():
    rows = [
        {"team_espn_id": "1", "returning_pct": 0.2, "recruiting_points": 100.0,
         "portal_net": 10.0, "prior_sos": None},
        {"team_espn_id": "2", "returning_pct": NAN, "recruiting_points": 200.0,
         "portal_net": 30.0, "prior_sos": 7.0},
        {"team_espn_id": "3", "returning_pct": 0.6, "recruiting_points": 300.0,
         "portal_net": 20.0, "prior_sos": 9.0},
    ]
    z = season_features_z(rows)
    assert z["2"]["returning_pct"] == 0.0 and z["1"]["prior_sos"] == 0.0
    assert abs(z["2"]["recruiting_points"]) < 1e-9
    assert abs(z["3"]["portal_net"]) < 1e-9
    assert z["1"]["portal_net"] < 0 < z["2"]["portal_net"]
    assert set(z["1"]) == set(PRIOR_Z_FEATURES)


def test_preseason_rating_is_prev_sp_plus_weighted_z():
    w = PriorWeights(sp_scale=20.0, sp_offset=1500.0, w_returning=10.0, w_recruiting=5.0,
                     w_portal=3.0, w_qb=12.0, w_sos_prior=-2.0)
    inp = {"prev_sp": 1.5, "qb_flag": 1.0,
           "z": {"returning_pct": 2.0, "recruiting_points": 1.0, "portal_net": -1.0,
                 "prior_sos": 1.0}}
    assert preseason_rating(inp, w) == pytest.approx(
        1500.0 + 30.0 + 20.0 + 5.0 - 3.0 + 12.0 - 2.0)


def test_prior_uses_previous_season_final_sp():
    rows = [_row("A", 2022, 10.0), _row("B", 2022, -10.0),
            _row("A", 2023, -30.0), _row("B", 2023, 30.0)]
    r = season_priors(_by_season(rows), 2023, PriorWeights(sp_scale=20.0))
    assert r["A"] == pytest.approx(1500.0 + 200.0)
    assert r["B"] == pytest.approx(1500.0 - 200.0)


def test_planted_same_season_sp_coach_and_forward_sos_never_reach_the_prior():
    w = PriorWeights(sp_scale=20.0, w_returning=10.0, w_recruiting=5.0, w_portal=3.0,
                     w_qb=12.0, w_sos_prior=4.0)
    prev = [_row("A", 2022, 10.0), _row("B", 2022, -10.0), _row("C", 2022, 0.0)]
    cur = [_row("A", 2023, 1.0, returning_pct=0.7), _row("B", 2023, 2.0),
           _row("C", 2023, 3.0, recruiting_points=250.0)]
    clean = season_priors(_by_season(prev + cur), 2023, w)
    planted = [{**r, "sp_rating": 999.0 * (i + 1), "coach_first_year": True,
                "forward_sos_shift": 50.0 * (i + 1), "returning_starters": 0.01 * i}
               for i, r in enumerate(cur)]
    assert season_priors(_by_season(prev + planted), 2023, w) == clean


def test_missing_prev_sp_maps_to_previous_season_league_mean():
    rows = [_row("A", 2022, 10.0), _row("B", 2022, 20.0), _row("C", 2022, NAN),
            _row("A", 2023, 0.0), _row("B", 2023, 0.0), _row("C", 2023, 0.0),
            _row("NEW", 2023, 0.0)]
    inp = season_prior_inputs(_by_season(rows), 2023)
    assert inp["C"]["prev_sp"] == pytest.approx(15.0)      # NaN last year -> mean
    assert inp["NEW"]["prev_sp"] == pytest.approx(15.0)    # absent last year -> mean
    # no previous season at all -> 0.0 (SP+ is centred near 0)
    first = season_prior_inputs(_by_season([_row("A", 2015, 5.0)]), 2015)
    assert first["A"]["prev_sp"] == 0.0


def test_portal_all_nan_pre_2021_is_neutral():
    rows = [_row("A", 2019, 1.0), _row("B", 2019, 2.0),
            _row("A", 2020, 0.0), _row("B", 2020, 0.0)]
    inp = season_prior_inputs(_by_season(rows), 2020)
    assert inp["A"]["z"]["portal_net"] == 0.0 and inp["B"]["z"]["portal_net"] == 0.0


def test_qb_returning_signed_and_none_is_neutral():
    rows = [_row("A", 2023, 0.0, qb_returning=True), _row("B", 2023, 0.0, qb_returning=False),
            _row("C", 2023, 0.0, qb_returning=None), _row("D", 2023, 0.0, qb_returning=NAN)]
    inp = season_prior_inputs(_by_season(rows), 2023)
    assert [inp[t]["qb_flag"] for t in "ABCD"] == [1.0, -1.0, 0.0, 0.0]


def test_load_weights_defaults_to_identity_prev_sp_map_when_missing(tmp_path):
    w = load_weights(tmp_path / "does_not_exist.json")
    assert w == PriorWeights() and w.sp_scale == 1.0 and w.sp_offset == 1500.0


def test_load_weights_reads_new_keys(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text('{"sp_scale": 20.0, "w_returning": 10.0, "w_recruiting": 4.0}')
    w = load_weights(p)
    assert (w.sp_scale, w.w_returning, w.w_recruiting, w.w_portal) == (20.0, 10.0, 4.0, 0.0)


def test_load_weights_rejects_nonzero_leaky_legacy_weights(tmp_path):
    p = tmp_path / "weights.json"
    p.write_text('{"sp_scale": 1.0, "w_coach": 0.0, "w_starters": 0.0, "w_sos_shift": 0.0}')
    assert load_weights(p) == PriorWeights()          # zero legacy keys are ignored
    p.write_text('{"sp_scale": 1.0, "w_coach": 25.0}')
    with pytest.raises(ValueError, match="w_coach"):
        load_weights(p)
    p.write_text('{"sp_scale": 1.0, "w_typo": 1.0}')
    with pytest.raises(TypeError):
        load_weights(p)


def test_committed_weights_load_and_are_leak_free():
    from sportsmodel import config
    w = load_weights(config.PROJECT_ROOT / "assets" / "cfb" / "priors_weights.json")
    assert isinstance(w, PriorWeights) and math.isfinite(w.sp_scale)
