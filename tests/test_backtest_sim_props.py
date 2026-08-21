"""Data-free smoke test for scripts/backtest_sim_props.py's prop-scoring path.

Builds a tiny synthetic GameSpec, runs the vectorized kernel at low n_sims,
turns the result into per-player prop dists (engine.player_prop_dists), and
scores one market end-to-end via prob_over_dist -- exactly the sequence
backtest_sim_props.run_sim_props_backtest uses per game. Must pass in CI
without the Statcast backfill (no parquet/duckdb access).
"""
import importlib.util
from pathlib import Path

import numpy as np

from sportsmodel.model.distributions import prob_over_dist
from sportsmodel.model.props import DEFAULT_LINE
from sportsmodel.sim.engine import player_prop_dists
from sportsmodel.sim.mlb import kernel

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_sim_props.py"


def test_backtest_sim_props_script_imports_cleanly():
    """Actually loads scripts/backtest_sim_props.py's own module code (not just
    the kernel/engine it calls), so a syntax error or bad import in the script
    itself is caught here rather than only surfacing on a real backtest run."""
    spec = importlib.util.spec_from_file_location("backtest_sim_props", _SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert hasattr(module, "run_sim_props_backtest")
    assert hasattr(module, "main")
    assert set(module.MARKET_MAX) == {
        "hits", "total_bases", "hrr", "pitcher_ks", "hits_allowed", "outs_recorded",
    }


def _v():
    v = {"p_bb": .08, "p_k": .22, "p_1b": .15, "p_2b": .045,
         "p_3b": .004, "p_hr": .03, "p_out": .471}
    s = sum(v.values())
    return {k: x / s for k, x in v.items()}


def test_player_prop_dists_score_without_error():
    bs = [kernel.Batter(100 + i, _v(), _v()) for i in range(9)]
    aw = [kernel.Batter(200 + i, _v(), _v()) for i in range(9)]
    spec = kernel.GameSpec(bs, aw, kernel.Pitcher(1, 16, 5), kernel.Pitcher(2, 16, 5))
    sims = kernel.simulate(spec, 200, np.random.default_rng(0))

    market_max = {"hits": 6, "total_bases": 10, "hrr": 15,
                  "pitcher_ks": 15, "hits_allowed": 15, "outs_recorded": 30}
    dists = player_prop_dists(sims, market_max)

    # Batter market: score a synthetic actual (2 hits) against the sim's dist for player 100.
    d = dists[100]["hits"]
    prob_over = prob_over_dist(d, DEFAULT_LINE["hits"])
    assert 0.0 <= prob_over <= 1.0
    assert isinstance(d["mean"], float)

    # Pitcher market: score a synthetic actual (5 Ks) against the sim's dist for starter 1.
    pd = dists[1]["pitcher_ks"]
    p_over_k = prob_over_dist(pd, DEFAULT_LINE["pitcher_ks"])
    assert 0.0 <= p_over_k <= 1.0
