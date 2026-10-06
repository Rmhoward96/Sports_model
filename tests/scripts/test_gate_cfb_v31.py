import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3, v3_gate, v31
from sportsmodel.cfb.efficiency import EffConfig
from sportsmodel.nfl.gameline import GameLineConfig

_scripts = pathlib.Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(_scripts))
_s = importlib.util.spec_from_file_location("gate_cfb_v31", _scripts / "gate_cfb_v31.py")
g31 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(g31)
from tests.scripts.test_gate_cfb_v3 import GL, W, raw_table  # noqa: E402


def fake_fits(table):
    gl = v3.gameline_v3_dict(15.0, 15.0)
    meta = {"train_seasons": [2016, 2017], "n_train": 100, "sigma_margin": 15.0, "sigma_total": 15.0,
            "margin": {"lasso_alpha": None}, "total": {"lasso_alpha": 0.1}}
    w = v3.V3Weights(W.points_map, v3.LinearBlend(0.1, {"margin_v2": 0.7, "margin_eff": 0.3, "prior_margin": 0.1,
                                                         "non_neutral": -0.5, "talent_gap": 0.2}),
                     v3.LinearBlend(5.0, {"total_eff": 0.5, "total_v2": 0.4, "wind_excess": 0.0}), meta)
    return [v31.SeasonFit(s, EffConfig(), w, gl, table[table["season"] == s].reset_index(drop=True))
            for s in (2023, 2024, 2025)]


def test_score_fits_scores_each_season_with_its_own_weights_and_concatenates():
    t = raw_table()
    fits = fake_fits(t)
    # season 2024 gets an identity blend; the others the blend from fake_fits
    ident = v3.V3Weights(W.points_map, v3.LinearBlend(0.0, {"margin_v2": 1.0}), v3.LinearBlend(0.0, {"total_v2": 1.0}),
                         fits[1].weights.meta)
    fits[1] = v31.SeasonFit(2024, EffConfig(), ident, fits[1].gameline, fits[1].table)
    out = g31.score_fits(fits, GL)
    assert len(out) == len(t) and list(out["season"].unique()) == [2023, 2024, 2025]
    m24 = out["season"] == 2024
    assert np.allclose(out.loc[m24, "v3_margin"], out.loc[m24, "margin_v2"])
    assert not np.allclose(out.loc[~m24, "v3_margin"], out.loc[~m24, "margin_v2"])


def make_res():
    scored = g31.score_fits(fake_fits(raw_table()), GL)
    ev = v3_gate.eval_set(scored)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    res = g31.evaluate_v31(scored, expected={k: round(v2c[k], 4) for k in ("margin_mae", "total_mae", "ats")})
    res.update({"generated": "2026-10-07", "fits": {f.season: g31.fit_summary(f) for f in fake_fits(raw_table())}})
    return res


def test_evaluate_v31_relabels_adds_paired_ses_and_is_json_safe():
    res = make_res()
    assert "v3" not in res and set(res["v31"]) == {"combined", "by_season"}
    assert all("v31" in c and "v3" not in c and c["label"].startswith(("v3.1", "ATS", "O/U", "ML")) for c in res["criteria"].values())
    assert res["criteria"]["margin_mae"]["label"] == "v3.1 margin MAE < v2"
    assert set(res["paired"]["combined"]) == {"ats", "ou", "ml_logloss"}
    assert set(res["paired"]["by_season"]) == {2023, 2024, 2025}
    assert res["v31"]["combined"]["n_ats"] > 0 and "n_ou" in res["v31"]["combined"]
    assert set(res["bias"]) == {"convention", "v2", "v31"}
    json.dumps(v3_gate.clean(res), allow_nan=False)


def test_evaluate_v31_stops_on_baseline_mismatch():
    scored = g31.score_fits(fake_fits(raw_table()), GL)
    with pytest.raises(v3_gate.BaselineMismatch):
        g31.evaluate_v31(scored)                      # published v2 numbers do not match noise data


def test_report_has_n_paired_se_bias_and_per_season_fit_tables():
    md = g31.render_report(make_res())
    assert "# cfb-ratings-v3.1 gate" in md and "Live model: v2 (unchanged)" in md
    assert "## Paired differences (v3.1 - v2)" in md and "paired SE" in md
    assert "n ATS:" in md and "| n ATS | n O/U |" in md
    assert "| 2024 | v3.1 |" in md and "## Bias" in md and "| all | v3.1 |" in md
    assert "## Per-season fits" in md and "talent_gap=+0.200" in md and "wind_excess" not in md.split("context kept")[1]
