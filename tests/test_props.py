import pytest

from sportsmodel.model import props

# A decent hitter's per-PA vector.
VEC = {
    "p_bb": 0.09, "p_k": 0.20, "p_1b": 0.15, "p_2b": 0.05,
    "p_3b": 0.005, "p_hr": 0.04, "p_out": 0.465,
}


def test_slot_pa_ordering():
    # Leadoff gets more PAs than the 9-hole.
    assert props.SLOT_PA[1] > props.SLOT_PA[9]


def test_batter_props_shape_and_ranges():
    p = props.batter_props(VEC, slot=3)
    assert p["projected_pa"] == props.SLOT_PA[3]
    for market in ("hits", "total_bases", "home_run"):
        assert 0.0 <= p[market]["prob_over"] <= 1.0
        assert p[market]["mean"] > 0
    # Total bases mean exceeds hits mean (extra-base hits count multiple bases).
    assert p["total_bases"]["mean"] > p["hits"]["mean"]
    # HR mean is small; P(1+ HR) modest.
    assert p["home_run"]["mean"] < 0.3
    assert p["home_run"]["prob_over"] < 0.3


def test_better_slot_more_hits():
    top = props.batter_props(VEC, slot=1)
    bottom = props.batter_props(VEC, slot=9)
    assert top["hits"]["mean"] > bottom["hits"]["mean"]


def test_hrr_exceeds_hits():
    p = props.batter_props(VEC, slot=3)
    # H+R+RBI sums three components, so its mean exceeds hits alone.
    assert p["hrr"]["mean"] > p["hits"]["mean"]
    assert 0.0 <= p["hrr"]["prob_over"] <= 1.0


def test_pitcher_props_ranges_and_strikeouts():
    # A high-K, low-contact lineup of 9 identical batters.
    ky = {"p_bb": 0.08, "p_k": 0.30, "p_1b": 0.12, "p_2b": 0.04,
          "p_3b": 0.004, "p_hr": 0.03, "p_out": 0.426}
    lo = {**ky, "p_k": 0.15, "p_out": 0.576}
    pp_hi = props.pitcher_props([ky] * 9, avg_bf=24, var_bf=9, avg_outs=17, sd_outs=6)
    pp_lo = props.pitcher_props([lo] * 9, avg_bf=24, var_bf=9, avg_outs=17, sd_outs=6)
    # More Ks against the high-strikeout lineup.
    assert pp_hi["pitcher_ks"]["mean"] > pp_lo["pitcher_ks"]["mean"]
    for m in ("pitcher_ks", "hits_allowed", "outs_recorded"):
        assert 0.0 <= pp_hi[m]["prob_over"] <= 1.0
    # ~24 BF at 30% K -> ~7 Ks.
    assert 6 < pp_hi["pitcher_ks"]["mean"] < 8
    assert pp_hi["outs_recorded"]["mean"] == 17
