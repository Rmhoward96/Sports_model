"""The props-ML artifact load / verify module (``model.props_ml.artifacts``).

Its behaviour (round trip, column / role / B-set checks) is exercised through
``scripts/fit_props_ml_final.py`` in tests/scripts/test_fit_props_ml_final.py
and through ``sim.nfl.ml_serving`` in tests/sim/nfl/test_ml_serving.py; here:
the script re-uses (does not duplicate) it, and the pure helpers."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

from sportsmodel.model.props_ml import artifacts

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fit_props_ml_final.py"
_spec = importlib.util.spec_from_file_location("fit_props_ml_final_art", _p)
fpf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fpf)


@pytest.mark.parametrize("name", ["Artifacts", "load_artifacts", "required_columns",
                                  "verify_feature_columns", "calib_for_apply", "b_markets",
                                  "CONFIG_FILE", "LEARNED_FILE", "CALIB_FILE", "FORMAT_VERSION"])
def test_fit_script_uses_the_src_module(name):
    assert getattr(fpf, name) is getattr(artifacts, name)


def test_b_markets_and_calib_for_apply():
    cfg = {"markets": {"rec_yds": {"source": "ml", "w_final": 0.4},
                       "rush_yds": {"source": "ml", "w_final": 1.0},
                       "pass_yds": {"source": "baseline", "w_final": 0.2}}}
    assert artifacts.b_markets(cfg) == ["rec_yds"]
    cal = {"markets": {"rec_yds": {"calibrate": True, "kind": "pit", "map": [[0, 1], [0, 1]]},
                       "anytime_td": {"calibrate": True, "kind": "platt", "map": [0.9, 0.1]}}}
    got = artifacts.calib_for_apply(cal)
    assert got["anytime_td"] == (0.9, 0.1)
    np.testing.assert_array_equal(got["rec_yds"], [[0, 1], [0, 1]])
    for v in cal["markets"].values():
        v["calibrate"] = False
    assert artifacts.calib_for_apply(cal) is None
