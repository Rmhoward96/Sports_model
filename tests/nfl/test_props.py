import math
from sportsmodel.nfl.props import PropConfig, build_prop
from sportsmodel.model.distributions import prob_over_dist, normal_sf

CFG = PropConfig()
VOL = {"pass_att": 34.0, "carries": 15.0, "targets": 8.0}
EFF = {"ypa": 7.5, "catch_rate": 0.65, "ypr": 11.0, "ypc": 4.3,
       "pass_td_rate": 0.05, "rec_td_rate": 0.06, "rush_td_rate": 0.03}

def test_pass_yds_normal():
    p = build_prop("pass_yds", VOL, EFF, CFG)
    assert math.isclose(p["projected_mean"], 34.0 * 7.5)
    assert p["dist"]["kind"] == "normal" and p["dist"]["mean"] == 34.0 * 7.5
    # P(over book line 250.5) matches the normal
    assert abs(prob_over_dist(p["dist"], 250.5) - normal_sf(250.5, 255.0, CFG.sigma["pass_yds"])) < 1e-9

def test_rush_reception_yds_sums_components():
    p = build_prop("rush_reception_yds", VOL, EFF, CFG)
    rush = 15.0 * 4.3
    rec = (8.0 * 0.65) * 11.0
    assert math.isclose(p["projected_mean"], rush + rec)

def test_receptions_negbin_mean():
    p = build_prop("receptions", VOL, EFF, CFG)
    mean = 8.0 * 0.65
    assert math.isclose(p["projected_mean"], mean)
    assert p["dist"]["kind"] == "pmf"
    got = sum(k * v for k, v in enumerate(p["dist"]["pmf"]))
    assert abs(got - mean) < 0.2   # NB pmf mean ~ input mean
