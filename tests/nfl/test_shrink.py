import math
from sportsmodel.nfl.shrink import ShrinkParams, w_curve, shrink

P = ShrinkParams(start=0.75, floor=0.2, decay=0.25)

def test_w_curve_decays_and_clamps():
    assert math.isclose(w_curve(1, P), 0.75, abs_tol=1e-9)   # week 1 = start
    assert w_curve(1, P) > w_curve(8, P) > P.floor           # decays
    assert w_curve(50, P) == P.floor                          # deep = floor
    assert w_curve(19, P) == P.floor                          # playoffs clamp

def test_shrink_endpoints():
    assert shrink(10.0, 3.0, 1, ShrinkParams(1.0, 1.0, 0.0)) == 3.0   # w=1 -> market
    assert shrink(10.0, 3.0, 1, ShrinkParams(0.0, 0.0, 0.0)) == 10.0  # w=0 -> model

def test_shrink_missing_market_returns_model():
    assert shrink(10.0, None, 1, P) == 10.0

def test_shrink_blends():
    w = w_curve(1, P)  # 0.75
    assert math.isclose(shrink(10.0, 2.0, 1, P), (1 - w) * 10.0 + w * 2.0)
