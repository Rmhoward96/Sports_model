import importlib.util
import json
import pathlib

from sportsmodel.cfb import v3

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fit_cfb_v3.py"
_s = importlib.util.spec_from_file_location("fit_cfb_v3", _p)
fit_script = importlib.util.module_from_spec(_s)
_s.loader.exec_module(fit_script)


def test_write_outputs_roundtrips_weights_and_emits_a_loadable_gameline(tmp_path):
    w = v3.V3Weights(v3.LinearBlend(20.0, {"success": 3.0}), v3.LinearBlend(0.0, {"margin_v2": 1.0}),
                     v3.LinearBlend(1.0, {"total_v2": 1.0}), {"sigma_margin": 15.5, "sigma_total": 16.5})
    fit_script.write_outputs(w, tmp_path)
    assert v3.load_v3_weights(tmp_path / "v3_weights.json") == w
    gl = json.loads((tmp_path / "gameline_v3.json").read_text())
    assert gl["sigma_margin"] == 15.5 and gl["sigma_total"] == 16.5 and gl["bias_margin"] == 0.0
