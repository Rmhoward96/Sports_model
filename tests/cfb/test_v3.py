"""v3 blend math, JSON round trip, gameline config."""
import json

import pytest

from sportsmodel.cfb import v3


def test_linear_blend_treats_nan_and_missing_as_zero():
    b = v3.LinearBlend(1.0, {"a": 2.0, "b": 3.0, "c": 4.0})
    assert b.predict({"a": 1.0, "b": float("nan"), "c": None}) == 3.0
    assert b.predict({}) == 1.0


def test_predict_v3_combines_points_map_and_blend():
    pm = v3.LinearBlend(20.0, {"success": 10.0})
    w = v3.V3Weights(pm, v3.LinearBlend(0.5, {"margin_v2": 0.8, "margin_eff": 0.2, "wind_excess": 9.9}),
                     v3.LinearBlend(2.0, {"total_eff": 0.5, "total_v2": 0.4, "wind_excess": -0.3}))
    row = {"margin_v2": 6.0, "total_v2": 50.0, "h_success": 0.1, "a_success": -0.1, "wind_excess": 10.0}
    m, t = v3.predict_v3(row, w)
    eff_m, eff_t = (21.0 - 19.0), (21.0 + 19.0)
    assert m == pytest.approx(0.5 + 0.8 * 6.0 + 0.2 * eff_m + 9.9 * 10.0)   # margin blend reads the margin ctx it has
    assert t == pytest.approx(2.0 + 0.5 * eff_t + 0.4 * 50.0 - 0.3 * 10.0)


def test_weights_json_roundtrip_and_strict_load(tmp_path):
    w = v3.V3Weights(v3.LinearBlend(1.0, {"ppa_plays": 2.0}), v3.LinearBlend(0.1, {"margin_v2": 1.0}),
                     v3.LinearBlend(3.0, {"total_v2": 0.9}), {"n_train": 5})
    p = tmp_path / "v3_weights.json"
    p.write_text(w.to_json())
    assert v3.load_v3_weights(p) == w
    assert json.loads(p.read_text())["version"] == "cfb-ratings-v3"
    with pytest.raises(FileNotFoundError):
        v3.load_v3_weights(tmp_path / "nope.json")


def test_gameline_v3_dict_has_every_key_generate_cfb_reads():
    d = v3.gameline_v3_dict(15.5, 16.0)
    assert set(d) >= {"sigma_margin", "sigma_total", "offset", "total_max", "w_margin", "w_total",
                      "bias_margin", "bias_total"}
    assert d["w_margin"] == {"start": 0.0, "floor": 0.0, "decay": 0.0} and d["bias_margin"] == 0.0


def test_load_gameline_config_reads_gameline_shaped_json_for_v2_and_v3(tmp_path):
    p = tmp_path / "gameline_v3.json"
    p.write_text(json.dumps(v3.gameline_v3_dict(15.5, 16.0)))
    cfg = v3.load_gameline_config(p)
    assert (cfg.sigma_margin, cfg.sigma_total, cfg.offset, cfg.total_max) == (15.5, 16.0, 110, 150)
    assert cfg.bias_margin == 0.0 and cfg.w_margin.start == 0.0
    q = tmp_path / "old.json"                                    # a pre-bias gameline.json still loads
    d = v3.gameline_v3_dict(1.0, 2.0)
    del d["bias_margin"], d["bias_total"]
    q.write_text(json.dumps(d))
    assert v3.load_gameline_config(q).bias_total == 0.0


def _weights_json(**over):
    d = json.loads(v3.V3Weights(v3.LinearBlend(1.0, {"ppa_plays": 2.0}), v3.LinearBlend(0.1, {"margin_v2": 1.0}),
                                v3.LinearBlend(3.0, {"total_v2": 0.9})).to_json())
    d.update(over)
    return json.dumps(d)


@pytest.mark.parametrize("section,bad", [("margin", "margin_v3"), ("margin", "wind_excess"),
                                         ("total", "total_v3"), ("total", "short_week_diff"),
                                         ("points_map", "ppa_play")])
def test_load_v3_weights_rejects_unknown_feature_keys(tmp_path, section, bad):
    d = json.loads(_weights_json())
    d[section]["coefs"][bad] = 1.0                    # a typo / wrong-model key must not be silently dropped
    p = tmp_path / "w.json"
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match=bad):
        v3.load_v3_weights(p)


def test_load_v3_weights_rejects_a_wrong_or_missing_version(tmp_path):
    p = tmp_path / "w.json"
    p.write_text(_weights_json(version="cfb-ratings-v2"))
    with pytest.raises(ValueError, match="version"):
        v3.load_v3_weights(p)
    d = json.loads(_weights_json())
    del d["version"]
    p.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="version"):
        v3.load_v3_weights(p)


def test_load_v3_weights_accepts_every_declared_feature(tmp_path):
    w = v3.V3Weights(v3.LinearBlend(1.0, {f: 0.1 for f in v3.POINT_FEATURES}),
                     v3.LinearBlend(0.0, {f: 0.1 for f in v3.ALL_MARGIN_FEATURES}),
                     v3.LinearBlend(0.0, {f: 0.1 for f in v3.ALL_TOTAL_FEATURES}))
    p = tmp_path / "w.json"
    p.write_text(w.to_json())
    assert v3.load_v3_weights(p) == w
