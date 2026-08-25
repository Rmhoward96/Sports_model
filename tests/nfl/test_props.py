import json
import math
import pathlib
from sportsmodel.nfl.props import PropConfig, build_prop
from sportsmodel.model.distributions import prob_over_dist, normal_sf

ALL_SEVEN_MARKETS = ("pass_yds", "reception_yds", "rush_yds", "rush_reception_yds",
                    "receptions", "pass_tds", "anytime_td")
PROPS_JSON_PATH = pathlib.Path(__file__).resolve().parents[2] / "assets" / "nfl" / "props.json"

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

def test_anytime_td_prob_at_least_one():
    p = build_prop("anytime_td", VOL, EFF, CFG)
    lam = 15.0*0.03 + 8.0*0.06
    assert abs(prob_over_dist(p["dist"], 0.5) - (1 - math.exp(-lam))) < 1e-6

def test_pass_tds_poisson_mean():
    p = build_prop("pass_tds", {"pass_att":34.0,"carries":0,"targets":0},
                   {"pass_td_rate":0.05,"catch_rate":0,"ypr":0,"ypc":0,"ypa":0,
                    "rec_td_rate":0,"rush_td_rate":0}, CFG)
    assert abs(p["projected_mean"] - 34.0*0.05) < 1e-9

def test_default_mean_mult_is_unity_for_all_seven_markets():
    # CFG = PropConfig() must default mean_mult to 1.0 everywhere, TD markets
    # included: the backtest's walk-forward fits mean_mult FROM predictions
    # made with these defaults, so a non-unity default would make the fit
    # circular.
    for market in ("pass_yds", "reception_yds", "rush_yds", "rush_reception_yds",
                   "receptions", "pass_tds", "anytime_td"):
        assert CFG.mean_mult[market] == 1.0

def test_mean_mult_scales_pass_yds_projected_mean():
    cfg = PropConfig(mean_mult={**CFG.mean_mult, "pass_yds": 1.5})
    p = build_prop("pass_yds", VOL, EFF, cfg)
    expected = 34.0 * 7.5 * 1.5
    assert math.isclose(p["projected_mean"], expected)
    assert p["dist"]["kind"] == "normal" and math.isclose(p["dist"]["mean"], expected)

def test_mean_mult_scales_rush_reception_yds_combined_total():
    # mean_mult is applied to the COMBINED total, not to each component
    # separately -- this is the "just scale the result" contract.
    cfg = PropConfig(mean_mult={**CFG.mean_mult, "rush_reception_yds": 1.2})
    p = build_prop("rush_reception_yds", VOL, EFF, cfg)
    raw = 15.0 * 4.3 + (8.0 * 0.65) * 11.0
    assert math.isclose(p["projected_mean"], raw * 1.2)

def test_mean_mult_scales_receptions_negbin_mean():
    cfg = PropConfig(mean_mult={**CFG.mean_mult, "receptions": 1.3})
    p = build_prop("receptions", VOL, EFF, cfg)
    raw_mean = 8.0 * 0.65
    assert math.isclose(p["projected_mean"], raw_mean * 1.3)

def test_mean_mult_scales_pass_tds_lambda():
    # Fix round 2: pass_tds gets the same mean_mult treatment as the yardage
    # markets, applied to the Poisson lambda before poisson_pmf.
    cfg = PropConfig(mean_mult={**CFG.mean_mult, "pass_tds": 1.6})
    p = build_prop("pass_tds", VOL, EFF, cfg)
    raw_lambda = 34.0 * 0.05
    assert math.isclose(p["projected_mean"], raw_lambda * 1.6)

def test_mean_mult_scales_anytime_td_lambda_and_prob_identity():
    # With a non-unity mult applied, prob_over_dist(dist, 0.5) must still
    # equal 1 - exp(-(lambda * mult)) -- the P(>=1) identity is preserved
    # against the CALIBRATED lambda, not the raw one.
    cfg = PropConfig(mean_mult={**CFG.mean_mult, "anytime_td": 1.4})
    p = build_prop("anytime_td", VOL, EFF, cfg)
    raw_lam = 15.0 * 0.03 + 8.0 * 0.06
    calibrated_lam = raw_lam * 1.4
    assert math.isclose(p["projected_mean"], calibrated_lam)
    assert abs(prob_over_dist(p["dist"], 0.5) - (1 - math.exp(-calibrated_lam))) < 1e-6

def test_props_json_round_trips_into_propconfig_for_all_seven_markets():
    # Final-review fix: locks the assets/nfl/props.json <-> PropConfig contract
    # that P4's producer is expected to rely on -- loading the COMMITTED
    # calibration file the natural way (sigma/nb_var_mult/mean_mult passed
    # straight through) must build a valid prop for every market, with no
    # KeyError/TypeError from a schema mismatch (e.g. nb_var_mult shipped as a
    # bare float instead of a dict indexed by market, which is what broke
    # `build_prop("receptions", ...)`'s `cfg.nb_var_mult["receptions"]` lookup
    # before this fix).
    j = json.loads(PROPS_JSON_PATH.read_text())
    cfg = PropConfig(sigma=j["sigma"], nb_var_mult=j["nb_var_mult"], mean_mult=j["mean_mult"])

    for market in ALL_SEVEN_MARKETS:
        p = build_prop(market, VOL, EFF, cfg)
        assert "projected_mean" in p and isinstance(p["projected_mean"], float)
        assert p["projected_mean"] >= 0.0
        dist = p["dist"]
        assert dist["kind"] in ("normal", "pmf")
        if dist["kind"] == "normal":
            assert dist["sd"] > 0
        else:
            assert abs(sum(dist["pmf"]) - 1.0) < 1e-6
