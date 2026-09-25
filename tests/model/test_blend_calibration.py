"""Tests for the A/B pmf blend and the PIT / Platt calibration maps (props ML,
Props-2). Pure synthetic data; no IO, no network."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sportsmodel.model.props_eval import pit_pmf, rps_pmf
from sportsmodel.model.props_ml.blend import blend_binary, blend_pmf, choose_weight
from sportsmodel.model.props_ml.pit_calibration import (
    apply_pit_map,
    apply_platt,
    fit_pit_map,
    fit_platt,
)


def _pmf_var(p) -> float:
    p = np.asarray(p, dtype=float)
    k = np.arange(len(p))
    m = float(np.sum(k * p))
    return float(np.sum((k - m) ** 2 * p))


def _discrete_normal(mean: float, sd: float, kmax: int = 40) -> np.ndarray:
    k = np.arange(kmax + 1)
    p = np.exp(-0.5 * ((k - mean) / sd) ** 2)
    return p / p.sum()


# --- blend_pmf ------------------------------------------------------------------


def test_blend_pmf_endpoints():
    a = np.array([0.1, 0.2, 0.7])
    b = np.array([0.5, 0.3, 0.2])
    np.testing.assert_allclose(blend_pmf(a, b, 1.0), a, atol=1e-15)
    np.testing.assert_allclose(blend_pmf(a, b, 0.0), b, atol=1e-15)


def test_blend_pmf_is_linear_pool_and_a_pmf():
    a = np.array([0.1, 0.2, 0.7])
    b = np.array([0.5, 0.3, 0.2])
    out = blend_pmf(a, b, 0.3)
    np.testing.assert_allclose(out, 0.3 * a + 0.7 * b, atol=1e-15)
    assert out.min() >= 0.0
    assert out.sum() == pytest.approx(1.0, abs=1e-12)


def test_blend_pmf_pads_unequal_supports():
    a = np.array([0.5, 0.5])
    b = np.array([0.2, 0.3, 0.5])
    out = blend_pmf(a, b, 0.5)
    np.testing.assert_allclose(out, [0.35, 0.4, 0.25], atol=1e-15)


def test_blend_pmf_rejects_weight_outside_unit_interval():
    with pytest.raises(ValueError):
        blend_pmf([0.5, 0.5], [0.5, 0.5], 1.5)


# --- blend_binary ---------------------------------------------------------------


def test_blend_binary_endpoints():
    assert blend_binary(0.3, 0.6, 1.0) == pytest.approx(0.3, abs=1e-12)
    assert blend_binary(0.3, 0.6, 0.0) == pytest.approx(0.6, abs=1e-12)


def test_blend_binary_is_logit_blend():
    lg = lambda p: np.log(p / (1 - p))  # noqa: E731
    expect = 1.0 / (1.0 + np.exp(-(0.4 * lg(0.2) + 0.6 * lg(0.7))))
    assert blend_binary(0.2, 0.7, 0.4) == pytest.approx(expect, abs=1e-12)


def test_blend_binary_monotone_in_w():
    ws = np.linspace(0, 1, 21)
    up = [blend_binary(0.8, 0.1, w) for w in ws]  # p_a > p_b: increasing in w
    down = [blend_binary(0.1, 0.8, w) for w in ws]
    assert np.all(np.diff(up) > 0)
    assert np.all(np.diff(down) < 0)


def test_blend_binary_clips_extremes():
    assert blend_binary(0.0, 0.0, 0.5) == pytest.approx(1e-4, rel=1e-9)
    assert blend_binary(1.0, 1.0, 0.5) == pytest.approx(1 - 1e-4, rel=1e-9)


def test_blend_binary_vectorised():
    out = blend_binary(np.array([0.2, 0.5]), np.array([0.2, 0.5]), 0.5)
    np.testing.assert_allclose(out, [0.2, 0.5], atol=1e-12)


# --- choose_weight --------------------------------------------------------------


def _rows_b_better(n: int = 50) -> list[dict]:
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(n):
        y = int(rng.integers(0, 6))
        b = np.full(6, 0.02)
        b[y] = 0.9
        b /= b.sum()
        a = np.full(6, 1 / 6)  # uninformative A
        rows.append({"pmf_a": a, "pmf_b": b, "actual": y})
    return rows


def test_choose_weight_picks_zero_when_b_strictly_better():
    assert choose_weight(_rows_b_better(), "receptions") == 0.0


def test_choose_weight_prefers_larger_w_on_ties():
    rows = [
        {"pmf_a": [0.2, 0.5, 0.3], "pmf_b": [0.2, 0.5, 0.3], "actual": y}
        for y in (0, 1, 2, 1)
    ]
    assert choose_weight(rows, "rush_att") == 1.0


def test_choose_weight_picks_one_when_a_strictly_better():
    rows = [
        {"pmf_a": r["pmf_b"], "pmf_b": r["pmf_a"], "actual": r["actual"]}
        for r in _rows_b_better()
    ]
    assert choose_weight(rows, "rec_yds") == 1.0


def test_choose_weight_interior_optimum_and_grid_minimum():
    # A says 0, B says 2, truth is 1 every time -> the 50/50 pool wins.
    rows = [{"pmf_a": [1.0, 0.0, 0.0], "pmf_b": [0.0, 0.0, 1.0], "actual": 1}] * 5
    w = choose_weight(rows, "pass_tds")
    assert w == 0.5
    grid = np.round(np.arange(0, 1.01, 0.1), 1)
    scores = [np.mean([rps_pmf(blend_pmf(r["pmf_a"], r["pmf_b"], g), 1) for r in rows])
              for g in grid]
    assert min(scores) == pytest.approx(
        np.mean([rps_pmf(blend_pmf(r["pmf_a"], r["pmf_b"], w), 1) for r in rows]))


def test_choose_weight_binary_uses_logit_blend_brier():
    rng = np.random.default_rng(1)
    rows = []
    for _ in range(200):
        p_true = rng.uniform(0.05, 0.6)
        y = int(rng.random() < p_true)
        rows.append({"pmf_a": [0.7, 0.3], "pmf_b": [1 - p_true, p_true], "actual": y})
    assert choose_weight(rows, "anytime_td") == 0.0
    tied = [{"pmf_a": [0.6, 0.4], "pmf_b": [0.6, 0.4], "actual": y} for y in (0, 1, 1)]
    assert choose_weight(tied, "anytime_td") == 1.0


def test_choose_weight_binary_differs_from_linear_pool():
    # Logit blend vs linear pool disagree on the optimum here; the binary path
    # must use the logit blend. A = 0.5, B = 0.01, truth rate 0.2.
    rows = [{"pmf_a": [0.5, 0.5], "pmf_b": [0.99, 0.01], "actual": int(i < 20)}
            for i in range(100)]
    grid = np.round(np.arange(0, 1.01, 0.1), 1)

    def brier(g):
        return np.mean([(blend_binary(0.5, 0.01, g) - r["actual"]) ** 2 for r in rows])

    expect = max(g for g in grid if brier(g) <= min(brier(x) for x in grid) + 1e-12)
    assert choose_weight(rows, "anytime_td") == pytest.approx(expect)


def test_choose_weight_accepts_dataframe():
    df = pd.DataFrame(_rows_b_better())
    assert choose_weight(df, "receptions") == 0.0


def test_choose_weight_rejects_empty_rows():
    with pytest.raises(ValueError):
        choose_weight([], "receptions")


# --- PIT map --------------------------------------------------------------------


def test_fit_pit_map_uniform_is_near_identity():
    rng = np.random.default_rng(2)
    knots = fit_pit_map(rng.random(20_000))
    assert knots.shape[0] == 2
    assert knots.shape[1] <= 201
    u, g = knots
    assert u[0] == 0.0 and g[0] == 0.0 and u[-1] == 1.0 and g[-1] == 1.0
    assert np.all(np.diff(u) > 0)
    assert np.all(np.diff(g) >= 0)
    assert np.max(np.abs(g - u)) < 0.02


def test_fit_pit_map_ignores_nonfinite_and_clips():
    pits = np.tile([np.nan, np.inf, -np.inf, -0.5, 1.5, 0.25, 0.75], 100)  # 400 finite
    u, g = fit_pit_map(pits)
    # finite: -0.5 -> 0, 1.5 -> 1, 0.25, 0.75; half of them are <= 0.5
    assert g[np.searchsorted(u, 0.5)] == pytest.approx(0.5)
    assert np.all(np.isfinite(g))
    assert g.min() >= 0.0 and g.max() <= 1.0
    assert np.all(np.diff(g) >= 0)


def test_fit_pit_map_is_json_serialisable():
    knots = fit_pit_map(np.linspace(0, 1, 500))
    assert knots.shape == (2, 201)
    back = np.asarray(json.loads(json.dumps(knots.tolist())))
    np.testing.assert_array_equal(back, knots)


def test_fit_pit_map_empty_is_identity():
    u, g = fit_pit_map([np.nan])
    np.testing.assert_array_equal(u, g)


def test_apply_pit_map_identity_leaves_pmf_unchanged():
    pmf = _discrete_normal(8.0, 2.5, 30)
    ident = np.array([[0.0, 1.0], [0.0, 1.0]])
    np.testing.assert_allclose(apply_pit_map(pmf, ident), pmf, atol=1e-12)
    grid = np.linspace(0, 1, 201)
    np.testing.assert_allclose(apply_pit_map(pmf, np.vstack([grid, grid])), pmf, atol=1e-12)
    # list-of-lists knots (as read back from calibration.json) work too
    np.testing.assert_allclose(apply_pit_map(pmf, [[0.0, 1.0], [0.0, 1.0]]), pmf, atol=1e-12)


def test_pit_map_from_overconfident_forecasts_widens_pmf():
    rng = np.random.default_rng(3)
    forecast = _discrete_normal(15.0, 1.5)  # too narrow
    actuals = np.clip(np.round(rng.normal(15.0, 5.0, 20_000)), 0, 40)  # truth is wide
    pits = np.array([pit_pmf(forecast, a, u) for a, u in zip(actuals, rng.random(len(actuals)))])
    knots = fit_pit_map(pits)
    # PITs pile at the edges: G rises steeply near 0 and flattens in the middle
    assert np.interp(0.1, *knots) > 0.2
    widened = apply_pit_map(forecast, knots)
    assert _pmf_var(widened) > 2.0 * _pmf_var(forecast)


def test_apply_pit_map_preserves_sum_and_nonnegativity():
    rng = np.random.default_rng(4)
    knots = fit_pit_map(rng.beta(0.4, 0.4, 5_000))
    for pmf in (_discrete_normal(3.0, 1.0, 10), np.array([0.0, 1.0, 0.0]),
                np.array([0.3, 0.7]), rng.dirichlet(np.ones(25))):
        out = apply_pit_map(pmf, knots)
        assert out.shape == np.asarray(pmf).shape
        assert out.min() >= 0.0
        assert out.sum() == pytest.approx(1.0, abs=1e-12)


# --- Platt ----------------------------------------------------------------------


def test_fit_platt_on_calibrated_data_is_near_identity():
    rng = np.random.default_rng(5)
    p = rng.uniform(0.02, 0.9, 20_000)
    y = (rng.random(len(p)) < p).astype(int)
    a, b = fit_platt(p, y)
    assert isinstance(a, float) and isinstance(b, float)
    assert abs(a - 1.0) < 0.1
    assert abs(b) < 0.1


def test_fit_platt_corrects_overconfidence():
    rng = np.random.default_rng(6)
    true_p = rng.uniform(0.1, 0.5, 20_000)
    lg = np.log(true_p / (1 - true_p))
    stated = 1.0 / (1.0 + np.exp(-2.0 * lg))  # logit stretched x2
    y = (rng.random(len(true_p)) < true_p).astype(int)
    a, _ = fit_platt(stated, y)
    assert a == pytest.approx(0.5, abs=0.1)


def test_apply_platt_identity_and_shape():
    p = np.array([0.05, 0.3, 0.8])
    np.testing.assert_allclose(apply_platt(p, (1.0, 0.0)), p, atol=1e-12)
    assert apply_platt(0.3, (1.0, 0.0)) == pytest.approx(0.3, abs=1e-12)
    assert apply_platt(0.3, (1.0, 1.0)) > 0.3


def test_fit_platt_single_class_falls_back_to_identity():
    assert fit_platt([0.2, 0.3, 0.4], [0, 0, 0]) == (1.0, 0.0)


# --- Props-2 Task 3 rulings: robustness guards ------------------------------------


def test_fit_pit_map_needs_min_pits_else_identity():
    from sportsmodel.model.props_ml.pit_calibration import IDENTITY_KNOTS, MIN_PITS
    assert MIN_PITS == 300
    rng = np.random.default_rng(7)
    few = np.concatenate([rng.beta(0.3, 0.3, MIN_PITS - 1), [np.nan] * 50])
    np.testing.assert_array_equal(fit_pit_map(few), IDENTITY_KNOTS)
    enough = fit_pit_map(rng.beta(0.3, 0.3, MIN_PITS))
    assert enough.shape == (2, 201)
    assert np.interp(0.1, *enough) > 0.15  # a real (non-identity) map


def test_fit_platt_needs_min_per_class_else_identity():
    from sportsmodel.model.props_ml.pit_calibration import MIN_CLASS
    assert MIN_CLASS == 100
    rng = np.random.default_rng(8)
    p = rng.uniform(0.05, 0.6, 5_000)
    y = np.zeros(len(p), dtype=int)
    y[:MIN_CLASS - 1] = 1                     # 99 positives: too few to identify
    assert fit_platt(p, y) == (1.0, 0.0)
    y[:MIN_CLASS] = 1                         # 100 of each: fitted
    assert fit_platt(p, y) != (1.0, 0.0)


def test_fit_platt_separable_data_stays_finite():
    p = np.concatenate([np.full(200, 0.2), np.full(200, 0.8)])
    y = np.concatenate([np.zeros(200, dtype=int), np.ones(200, dtype=int)])
    a, b = fit_platt(p, y)                    # finite penalty (C=1e4): no divergence
    assert np.isfinite(a) and np.isfinite(b) and 1.0 < a < 20.0


def test_fit_platt_uses_finite_penalty(monkeypatch):
    import sportsmodel.model.props_ml.pit_calibration as pc
    seen = {}
    real = pc.LogisticRegression

    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(pc, "LogisticRegression", spy)
    rng = np.random.default_rng(10)
    p = rng.uniform(0.05, 0.9, 2_000)
    fit_platt(p, (rng.random(len(p)) < p).astype(int))
    assert seen["C"] == 1e4


def test_choose_weight_binary_reads_p_any_as_one_minus_p0():
    from sportsmodel.model.props_ml.blend import prob_at_least_one
    assert prob_at_least_one([0.6, 0.3, 0.1]) == pytest.approx(0.4)
    assert prob_at_least_one(np.array([0.25, 0.75], dtype=np.float32)) == pytest.approx(0.75)
    # a TD-count pmf (3 bins) scores exactly like its collapsed [P(0), P(>=1)] form
    rng = np.random.default_rng(9)
    rows3, rows2 = [], []
    for _ in range(60):
        a, b = rng.dirichlet(np.ones(3)), rng.dirichlet(np.ones(3))
        y = int(rng.random() < 0.4)
        rows3.append({"pmf_a": a, "pmf_b": b, "actual": y})
        rows2.append({"pmf_a": [a[0], 1 - a[0]], "pmf_b": [b[0], 1 - b[0]], "actual": y})
    assert choose_weight(rows3, "anytime_td") == choose_weight(rows2, "anytime_td")
