import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3, v3_gate
from sportsmodel.cfb.context import CONTEXT_COLS
from sportsmodel.cfb.efficiency import POINT_FEATURES
from sportsmodel.nfl.gameline import GameLineConfig

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gate_cfb_v3.py"
_s = importlib.util.spec_from_file_location("gate_cfb_v3", _p)
gate = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gate)

GL = GameLineConfig(sigma_margin=15.0, sigma_total=15.0, offset=110, total_max=150)


def raw_table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"season": np.repeat([2023, 2024, 2025], n // 3), "week": 5, "game_pk": np.arange(n)})
    t["margin_v2"], t["total_v2"] = rng.normal(3, 10, n), rng.normal(55, 6, n)
    t["actual_margin"] = t["margin_v2"] + rng.normal(0, 12, n)
    t["actual_total"] = t["total_v2"] + rng.normal(0, 12, n)
    t["market_spread"] = t["margin_v2"].round() + 0.5
    t["market_total"] = t["total_v2"].round() + 0.5
    t["prior_margin"], t["non_neutral"] = 0.0, 1.0
    for f in POINT_FEATURES:
        t[f"h_{f}"], t[f"a_{f}"] = 0.0, 0.0
    for c in CONTEXT_COLS:
        t[c] = 0.0
    return t


W = v3.V3Weights(v3.LinearBlend(28.0, {}), v3.LinearBlend(0.0, {"margin_v2": 1.0}),
                 v3.LinearBlend(0.0, {"total_v2": 1.0}))


def test_add_predictions_serves_v2_through_its_gameline_and_v3_through_the_blend():
    t = raw_table()
    gl2 = GameLineConfig(sigma_margin=15.0, sigma_total=15.0, offset=110, total_max=150, bias_margin=-0.5)
    out = gate.add_predictions(t, W, gl2, GL)
    assert np.allclose(out["v2_margin"], t["margin_v2"] + 0.5)             # served = model - bias
    assert np.allclose(out["v3_margin"], t["margin_v2"])                   # identity blend, no bias
    assert ((out["v3_wp"] > 0) & (out["v3_wp"] < 1)).all()
    assert (out["v3_wp"] > 0.5).eq(out["v3_margin"] > 0).all()


def test_evaluate_gate_reproduces_baseline_then_compares_v3():
    out = gate.add_predictions(raw_table(), W, GL, GL)       # v3 == v2 here: ties
    ev = v3_gate.eval_set(out)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    expected = {k: round(v2c[k], 4) for k in ("margin_mae", "total_mae", "ats")}
    res = gate.evaluate_gate(out, expected=expected)
    assert res["baseline"]["reproduced"] and res["n_games"] == len(ev)
    assert res["ship"] is False                              # strict < on the MAEs: a tie does not ship
    assert res["criteria"]["margin_mae"]["pass"] is False and res["criteria"]["ats"]["pass"] is True
    assert set(res["v3"]["by_season"]) == {2023, 2024, 2025}
    json.dumps(v3_gate.clean(res), allow_nan=False)


def test_evaluate_gate_stops_on_baseline_mismatch_before_scoring_v3():
    out = gate.add_predictions(raw_table(), W, GL, GL)
    with pytest.raises(v3_gate.BaselineMismatch):
        gate.evaluate_gate(out)                              # the real published numbers do not match noise data


def test_better_v3_ships_and_report_renders():
    t = raw_table()
    t2 = t.copy()
    w_better = v3.V3Weights(v3.LinearBlend(28.0, {}), v3.LinearBlend(0.0, {"margin_v2": 0.5}),
                            v3.LinearBlend(0.0, {"total_v2": 1.0}))
    out = gate.add_predictions(t2, w_better, GL, GL)
    # make v3's margin strictly better on the eval set by moving it toward the actual margin
    out["v3_margin"] = 0.5 * out["v2_margin"] + 0.5 * out["actual_margin"]
    out["v3_total"] = 0.5 * out["v2_total"] + 0.5 * out["actual_total"]
    ev = v3_gate.eval_set(out)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    res = gate.evaluate_gate(out, expected={k: round(v2c[k], 4) for k in ("margin_mae", "total_mae", "ats")})
    res.update({"generated": "2026-10-07", "residual_check": None})
    assert res["criteria"]["margin_mae"]["pass"] and res["criteria"]["total_mae"]["pass"]
    md = gate.render_report(res)
    assert "Live model: v2 (unchanged)" in md and "Held-out [2023, 2024, 2025]" in md
    assert "| 2024 | v3 |" in md


def test_report_shows_n_ats_n_ou_and_the_decided_games_note_without_changing_the_verdict():
    out = gate.add_predictions(raw_table(), W, GL, GL)
    ev = v3_gate.eval_set(out)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    res = gate.evaluate_gate(out, expected={k: round(v2c[k], 4) for k in ("margin_mae", "total_mae", "ats")})
    ship_before = res["ship"]
    res.update({"generated": "2026-10-07", "residual_check": None})
    md = gate.render_report(res)
    assert "n ATS" in md and "n O/U" in md
    assert f"| 2023 | v3 | {res['v3']['by_season'][2023]['n']} | {res['v3']['by_season'][2023]['n_ats']} |" in md
    assert "decided games only" in md
    assert res["ship"] is ship_before
