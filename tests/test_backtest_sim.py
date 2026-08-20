"""Data-free smoke test for scripts/backtest_sim.py's sim path.

Builds a tiny synthetic GameSpec and runs the vectorized kernel at low n_sims,
asserting engine.pred_scores returns the expected keys. Must pass in CI without
the Statcast backfill (no parquet/duckdb access).
"""
import importlib.util
from pathlib import Path

import numpy as np

from sportsmodel.sim.mlb import kernel
from sportsmodel.sim.engine import pred_scores

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_sim.py"


def test_backtest_sim_script_imports_cleanly():
    """Actually loads scripts/backtest_sim.py's own module code (not just the
    kernel/engine it calls), so a syntax error or bad import in the script
    itself is caught here rather than only surfacing on a real backtest run."""
    spec = importlib.util.spec_from_file_location("backtest_sim", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "run_sim_backtest")
    assert hasattr(module, "main")


def _v():
    v = {"p_bb": .08, "p_k": .22, "p_1b": .15, "p_2b": .045,
         "p_3b": .004, "p_hr": .03, "p_out": .471}
    s = sum(v.values())
    return {k: x / s for k, x in v.items()}


def test_pred_scores_from_sim_have_expected_keys():
    bs = [kernel.Batter(100 + i, _v(), _v()) for i in range(9)]
    aw = [kernel.Batter(200 + i, _v(), _v()) for i in range(9)]
    spec = kernel.GameSpec(bs, aw, kernel.Pitcher(1, 24, 3), kernel.Pitcher(2, 24, 3))
    sims = kernel.simulate(spec, 200, np.random.default_rng(0))
    d = pred_scores(sims)
    assert {"pred_total", "home_win_prob", "pred_margin"} <= set(d)
