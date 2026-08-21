"""Data-free smoke test for scripts/fit_calibration_sim.py.

Tests the merge logic in isolation -- no backtest, no duckdb/parquet access.
The real script's whole job is producing the right split between sim-fit and
analytic-kept calibration params; these tests exercise `merge_calibration`
directly against small synthetic dicts to pin that split down.
"""
import importlib.util
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "fit_calibration_sim.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("fit_calibration_sim", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_mod = _load_module()


def test_script_imports_cleanly():
    """Catches a syntax error or bad import in the script itself, same as the
    sibling backtest_sim / backtest_sim_props smoke tests."""
    assert hasattr(_mod, "merge_calibration")
    assert hasattr(_mod, "main")
    assert _mod.SIM_FIT_TARGETS == ("win_prob", "hits", "total_bases", "home_run", "outs_recorded")
    assert _mod.ANALYTIC_KEEP_TARGETS == ("pitcher_ks", "hits_allowed")


def _existing():
    """A stand-in for the current assets/calibration.json: analytic-fit params
    for every market, including the two that must survive the sim refit."""
    return {
        "win_prob": [0.64, 0.09],
        "hits": [0.76, -0.07],
        "total_bases": [0.84, -0.27],
        "home_run": [0.89, -0.35],
        "pitcher_ks": [1.08, 0.10],
        "hits_allowed": [0.78, 0.00],
        "outs_recorded": [0.35, -0.17],
    }


def _sim_params():
    """Stand-in for freshly sim-fit params -- deliberately different numbers
    from `_existing()` so overrides are unambiguous in assertions."""
    return {
        "win_prob": [1.10, 0.02],
        "hits": [0.95, -0.01],
        "total_bases": [0.90, -0.05],
        "home_run": [1.05, 0.03],
        "outs_recorded": [0.80, 0.10],
    }


def test_sim_fit_targets_are_overridden():
    merged = _mod.merge_calibration(_sim_params(), _existing())
    for t in _mod.SIM_FIT_TARGETS:
        assert merged[t] == _sim_params()[t], f"{t} should take the sim-fit value"


def test_analytic_keep_targets_are_unchanged():
    existing = _existing()
    merged = _mod.merge_calibration(_sim_params(), existing)
    for t in _mod.ANALYTIC_KEEP_TARGETS:
        assert merged[t] == existing[t], f"{t} must stay analytic-fit, not be overwritten"
        # sim_params never even carries these keys -- confirm that invariant too.
        assert t not in _sim_params()


def test_hrr_passes_through_unchanged_when_present():
    existing = _existing()
    existing["hrr"] = [1.0, 0.0]
    merged = _mod.merge_calibration(_sim_params(), existing)
    assert merged["hrr"] == [1.0, 0.0]


def test_hrr_stays_absent_when_not_present():
    merged = _mod.merge_calibration(_sim_params(), _existing())
    assert "hrr" not in merged


def test_merge_result_has_exactly_the_expected_keys():
    existing = _existing()
    merged = _mod.merge_calibration(_sim_params(), existing)
    assert set(merged) == set(existing)  # sim_params introduces no new keys here


def test_merge_does_not_mutate_inputs():
    existing = _existing()
    sim_params = _sim_params()
    existing_copy = dict(existing)
    sim_params_copy = dict(sim_params)
    _mod.merge_calibration(sim_params, existing)
    assert existing == existing_copy
    assert sim_params == sim_params_copy


def test_merge_survives_empty_existing():
    """First-ever run with no assets/calibration.json yet: sim-fit targets show
    up, analytic-kept targets are simply absent (nothing to keep)."""
    merged = _mod.merge_calibration(_sim_params(), {})
    for t in _mod.SIM_FIT_TARGETS:
        assert t in merged
    for t in _mod.ANALYTIC_KEEP_TARGETS:
        assert t not in merged


def test_fit_dist_affine_moment_match():
    import numpy as np
    fcs = _load_module()
    sim = np.random.default_rng(0).normal(7.9, 3.96, 20000)
    loc, scale = fcs.fit_dist_affine(sim, emp_mean=8.85, emp_sd=4.60)
    assert abs(loc - (8.85 - sim.mean())) < 1e-9
    assert abs(scale - (4.60 / sim.std())) < 1e-9
    # after applying, the sim marginal mean lands on the empirical mean
    assert abs((sim.mean() + loc) - 8.85) < 1e-9
