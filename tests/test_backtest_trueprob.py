"""Unit tests for scripts/backtest_trueprob.py's PURE metric helpers, plus a
small synthetic end-to-end test of run_walk_forward.

Loads the script module directly (it isn't a package) via importlib, the
same pattern tests/cfb/test_backtest_cfb_priors.py uses. No parquet reads,
no network -- the pure helpers operate on plain lists/dicts and the
end-to-end test builds its own tiny synthetic DataFrame.
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "backtest_trueprob.py"
_spec = importlib.util.spec_from_file_location("backtest_trueprob", _SCRIPT_PATH)
backtest_trueprob = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(backtest_trueprob)

mae = backtest_trueprob.mae
mae_vs_line = backtest_trueprob.mae_vs_line
brier = backtest_trueprob.brier
reliability = backtest_trueprob.reliability
clv_proxy = backtest_trueprob.clv_proxy
disagreement_rate = backtest_trueprob.disagreement_rate
run_walk_forward = backtest_trueprob.run_walk_forward
CFB_FEATURES = backtest_trueprob.CFB_FEATURES


# ---------------------------------------------------------------------------
# ast.parse sanity check
# ---------------------------------------------------------------------------

def test_script_parses_as_valid_python():
    import ast

    ast.parse(_SCRIPT_PATH.read_text())


# ---------------------------------------------------------------------------
# mae
# ---------------------------------------------------------------------------

def test_mae_known_value():
    pred = [1.0, 2.0, 3.0, 4.0]
    actual = [1.0, 4.0, 3.0, 0.0]
    # abs diffs: 0, 2, 0, 4 -> mean 1.5
    assert mae(pred, actual) == pytest.approx(1.5)


def test_mae_empty_is_nan():
    assert math.isnan(mae([], []))


# ---------------------------------------------------------------------------
# mae_vs_line
# ---------------------------------------------------------------------------

def test_mae_vs_line_known_value():
    rows = [
        {"actual_margin": 3.0, "spread_line": 1.0},   # |1-3| = 2
        {"actual_margin": -7.0, "spread_line": -3.0},  # |-3 - -7| = 4
        {"actual_margin": 10.0, "spread_line": 10.0},  # 0
    ]
    assert mae_vs_line(rows, "actual_margin", "spread_line") == pytest.approx(2.0)


def test_mae_vs_line_skips_missing_line():
    rows = [
        {"actual_margin": 3.0, "spread_line": 1.0},    # |diff| = 2
        {"actual_margin": -7.0, "spread_line": None},  # skipped
        {"actual_margin": 5.0, "spread_line": float("nan")},  # skipped
    ]
    assert mae_vs_line(rows, "actual_margin", "spread_line") == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# brier
# ---------------------------------------------------------------------------

def test_brier_known_value():
    probs = [0.9, 0.1, 0.5]
    outcomes = [1, 0, 1]
    # (0.9-1)^2=0.01, (0.1-0)^2=0.01, (0.5-1)^2=0.25 -> mean 0.09
    assert brier(probs, outcomes) == pytest.approx(0.09)


def test_brier_perfect_predictions_is_zero():
    assert brier([1.0, 0.0, 1.0], [1, 0, 1]) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# reliability
# ---------------------------------------------------------------------------

def test_reliability_single_bucket():
    # All probs land in [0.7, 0.8); mean_pred=0.75, mean_actual = 2/3
    probs = [0.71, 0.75, 0.79]
    outcomes = [1, 1, 0]
    table = reliability(probs, outcomes, n_bins=10)
    assert len(table) == 1
    bucket = table[0]
    assert bucket["bin_lo"] == pytest.approx(0.7)
    assert bucket["bin_hi"] == pytest.approx(0.8)
    assert bucket["n"] == 3
    assert bucket["mean_pred"] == pytest.approx((0.71 + 0.75 + 0.79) / 3)
    assert bucket["mean_actual"] == pytest.approx(2 / 3)


def test_reliability_empty_bins_are_omitted():
    probs = [0.05, 0.95]
    outcomes = [0, 1]
    table = reliability(probs, outcomes, n_bins=10)
    assert len(table) == 2
    los = {round(b["bin_lo"], 2) for b in table}
    assert los == {0.0, 0.9}


# ---------------------------------------------------------------------------
# clv_proxy
# ---------------------------------------------------------------------------

def test_clv_proxy_sign_model_likes_home_more_than_market():
    # model margin consistently above the line's implied home margin ->
    # positive clv_proxy (model favors home relative to the market)
    rows = [
        {"pred_margin": 5.0, "spread_line": 2.0},
        {"pred_margin": 3.0, "spread_line": 1.0},
    ]
    # edges: (5-2)=3, (3-1)=2 -> mean 2.5
    assert clv_proxy(rows) == pytest.approx(2.5)


def test_clv_proxy_sign_model_likes_away_more_than_market():
    rows = [
        {"pred_margin": -6.0, "spread_line": -2.0},
        {"pred_margin": -4.0, "spread_line": -2.0},
    ]
    assert clv_proxy(rows) == pytest.approx(-3.0)


def test_clv_proxy_skips_missing_line():
    rows = [
        {"pred_margin": 5.0, "spread_line": 2.0},
        {"pred_margin": 99.0, "spread_line": None},
    ]
    assert clv_proxy(rows) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# disagreement_rate
# ---------------------------------------------------------------------------

def test_disagreement_rate_known_value():
    rows = [
        {"pred_margin": 3.0, "spread_line": 1.0},    # agree (both home)
        {"pred_margin": -3.0, "spread_line": 1.0},   # disagree
        {"pred_margin": -1.0, "spread_line": -4.0},  # agree (both away)
        {"pred_margin": 2.0, "spread_line": -1.0},   # disagree
    ]
    assert disagreement_rate(rows) == pytest.approx(0.5)


def test_disagreement_rate_excludes_zero_side():
    rows = [
        {"pred_margin": 0.0, "spread_line": 1.0},   # no side -> excluded
        {"pred_margin": 3.0, "spread_line": 1.0},   # agree
    ]
    assert disagreement_rate(rows) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# run_walk_forward -- synthetic end-to-end
# ---------------------------------------------------------------------------

def _synthetic_features_df(seasons=(2021, 2022, 2023), n_per_season=80, seed=11) -> pd.DataFrame:
    """3-season synthetic frame: margin is a clean linear function of
    elo_diff plus small noise; the market line is independent random noise
    with a larger spread, so it is DELIBERATELY a worse predictor of the
    actual margin than the fitted model -- this is what lets the test assert
    model_margin_mae < line_margin_mae without depending on real data."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(n_per_season):
            elo_diff = rng.normal(scale=10.0)
            noise = rng.normal(scale=3.0)
            margin = 0.5 * elo_diff + noise
            total = 45.0 + rng.normal(scale=8.0)
            spread_line = rng.normal(scale=15.0)  # independent of margin -> worse than the model
            total_line = 45.0 + rng.normal(scale=20.0)  # independent of total -> worse than the model
            rows.append({
                "season": season,
                "week": (i % 17) + 1,
                "home_team": f"H{i % 8}",
                "away_team": f"A{i % 8}",
                "elo_diff": elo_diff,
                "margin": margin,
                "total": total,
                "spread_line": spread_line,
                "total_line": total_line,
            })
    return pd.DataFrame(rows)


def test_run_walk_forward_synthetic_beats_the_deliberately_worse_line():
    df = _synthetic_features_df()
    result = run_walk_forward(df, sport="nfl", features=["elo_diff"], alpha=1.0)

    assert result["n_games"] > 0
    # two scored seasons (2022 trained on 2021; 2023 trained on 2021+2022)
    assert result["seasons_scored"] == [2022, 2023]

    assert result["model_margin_mae"] < result["line_margin_mae"]
    assert math.isfinite(result["brier_raw"])
    assert math.isfinite(result["brier_calibrated"])
    assert 0.0 <= result["brier_raw"] <= 1.0
    assert 0.0 <= result["brier_calibrated"] <= 1.0
    assert isinstance(result["reliability"], list)
    assert math.isfinite(result["mean_clv_proxy"])
    assert 0.0 <= result["disagreement_rate"] <= 1.0


def test_run_walk_forward_never_trains_on_current_or_future_season():
    """Leak-free invariant: appending a LATER season's rows must not change
    an EARLIER scored season's per-game predictions (those only ever depend
    on seasons strictly before them)."""
    df = _synthetic_features_df(seasons=(2021, 2022, 2023))
    df_truncated = df[df["season"] != 2023].reset_index(drop=True)

    full = run_walk_forward(df, sport="nfl", features=["elo_diff"], alpha=1.0)
    truncated = run_walk_forward(df_truncated, sport="nfl", features=["elo_diff"], alpha=1.0)

    full_2022 = sorted(
        (r["home_team"], r["away_team"], r["week"], round(r["pred_margin"], 8))
        for r in full["rows"] if r["season"] == 2022
    )
    truncated_2022 = sorted(
        (r["home_team"], r["away_team"], r["week"], round(r["pred_margin"], 8))
        for r in truncated["rows"] if r["season"] == 2022
    )
    assert full_2022 == truncated_2022


def test_cfb_features_list_has_no_nfl_only_rest_diff():
    assert "rest_diff" not in CFB_FEATURES
    assert "off_ppa_diff" in CFB_FEATURES
    assert "def_ppa_diff" in CFB_FEATURES
