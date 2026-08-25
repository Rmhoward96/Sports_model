import math
from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
from sportsmodel.nfl.shrink import ShrinkParams
from sportsmodel.model.distributions import prob_cover, normal_sf

CFG = GameLineConfig(sigma_margin=13.2, sigma_total=10.0, offset=75, total_max=120)

def test_build_gameline_is_valid_serving_row():
    row = build_gameline(model_margin=3.0, model_total=45.0,
                         market={"spread_line": None, "total_line": None},
                         week=1, cfg=CFG)  # no market -> model-only
    md, td = row["margin_dist"], row["total_dist"]
    assert md["kind"] == "margin" and md["offset"] == 75
    assert td["kind"] == "pmf"
    assert math.isclose(sum(md["pmf"]), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(td["pmf"]), 1.0, abs_tol=1e-6)
    # win_prob = P(margin>0) from the margin dist
    assert math.isclose(row["home_win_prob"], prob_cover(md, 0.0))
    assert abs(row["home_win_prob"] - normal_sf(0.0, 3.0, 13.2)) < 0.02
    # scores reconstruct total/margin
    assert math.isclose(row["pred_home_score"] + row["pred_away_score"], row["pred_total"])
    assert math.isclose(row["pred_home_score"] - row["pred_away_score"], row["pred_margin"])

def test_full_market_weight_reproduces_market_line():
    cfg = GameLineConfig(sigma_margin=13.2, sigma_total=10.0, offset=75, total_max=120,
                         w_margin=ShrinkParams(1.0, 1.0, 0.0),
                         w_total=ShrinkParams(1.0, 1.0, 0.0))
    row = build_gameline(model_margin=10.0, model_total=60.0,
                         market={"spread_line": 3.0, "total_line": 44.0}, week=1, cfg=cfg)
    assert math.isclose(row["pred_margin"], 3.0)   # w=1 -> market spread
    assert math.isclose(row["pred_total"], 44.0)   # w=1 -> market total
