from sportsmodel.nfl.config import load_rating, load_gameline, load_props
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.props import PropConfig

def test_load_rating():
    elo, blend = load_rating()
    assert isinstance(elo, EloConfig) and isinstance(blend, BlendConfig)
    assert elo.k > 0 and 0 <= blend.w_sos <= 1

def test_load_gameline_uses_fitted_sigma_not_default():
    gl = load_gameline()
    assert isinstance(gl, GameLineConfig)
    assert gl.sigma_total != 10.0        # fitted (~13.58), not the illustrative default
    assert gl.offset == 75

def test_load_props_builds_all_seven():
    from sportsmodel.nfl.props import build_prop
    cfg = load_props()
    assert isinstance(cfg, PropConfig)
    vol = {"pass_att": 34.0, "carries": 15.0, "targets": 8.0}
    eff = {"ypa": 7.5, "catch_rate": 0.65, "ypr": 11.0, "ypc": 4.3,
           "pass_td_rate": 0.05, "rec_td_rate": 0.06, "rush_td_rate": 0.03}
    for m in ("pass_yds","reception_yds","rush_yds","rush_reception_yds","receptions","pass_tds","anytime_td"):
        assert "dist" in build_prop(m, vol, eff, cfg)
