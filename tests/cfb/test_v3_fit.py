"""The v3 fit on a synthetic table with known structure."""
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3_fit
from sportsmodel.cfb.context import CONTEXT_COLS
from sportsmodel.cfb.efficiency import POINT_FEATURES


def synthetic_table(n=2400, seed=2):
    """Known truth: margin = 0.7*v2 + 0.4*eff - 1 + 1.5*non_neutral + 2*short_week_diff + noise;
    total = 0.5*eff_total + 0.5*v2 - 0.4*wind_excess + noise. Everything else has zero effect."""
    rng = np.random.default_rng(seed)
    seasons = rng.choice(np.arange(2016, 2023), n)
    t = pd.DataFrame({"season": seasons, "week": rng.integers(1, 14, n), "game_pk": np.arange(n)})
    for f in POINT_FEATURES:
        t[f"h_{f}"], t[f"a_{f}"] = rng.normal(0, 1, n), rng.normal(0, 1, n)
    pm_true = {"intercept": 28.0, "ppa_plays": 0.8, "success": 1.5}
    hp = pm_true["intercept"] + 0.8 * t["h_ppa_plays"] + 1.5 * t["h_success"]
    ap = pm_true["intercept"] + 0.8 * t["a_ppa_plays"] + 1.5 * t["a_success"]
    t["margin_v2"], t["total_v2"] = rng.normal(3, 12, n), rng.normal(55, 8, n)
    t["prior_margin"] = rng.normal(0, 3, n)
    t["non_neutral"] = (rng.random(n) > 0.1).astype(float)
    for c in CONTEXT_COLS:
        t[c] = 0.0
    t["short_week_diff"] = rng.choice([-1.0, 0.0, 1.0], n, p=[0.15, 0.7, 0.15])
    t["wind_excess"] = np.where(rng.random(n) < 0.3, rng.uniform(0, 15, n), 0.0)
    t["weather_missing"] = (rng.random(n) < 0.25).astype(float)
    t["travel_far_diff"] = rng.choice([-1.0, 0.0, 1.0], n)               # pure noise columns
    t["talent_gap"] = rng.normal(0, 1, n)
    eff_m, eff_t = hp - ap, hp + ap
    t["actual_margin"] = (0.7 * t["margin_v2"] + 0.4 * eff_m - 1.0 + 1.5 * t["non_neutral"]
                          + 2.0 * t["short_week_diff"] + rng.normal(0, 3, n))
    t["actual_total"] = (0.5 * eff_t + 0.5 * t["total_v2"] - 0.4 * t["wind_excess"]
                         + 5 * t["weather_missing"] + rng.normal(0, 3, n))
    return t


def test_points_map_recovers_known_coefficients():
    t = synthetic_table()
    # make actual points consistent with the true points map for this check
    hp = 28.0 + 0.8 * t["h_ppa_plays"] + 1.5 * t["h_success"]
    ap = 28.0 + 0.8 * t["a_ppa_plays"] + 1.5 * t["a_success"]
    t["actual_margin"], t["actual_total"] = hp - ap, hp + ap
    pm = v3_fit.fit_points_map(t)
    assert pm.intercept == pytest.approx(28.0, abs=0.05)
    assert pm.coefs["ppa_plays"] == pytest.approx(0.8, abs=0.02) and pm.coefs["success"] == pytest.approx(1.5, abs=0.02)
    assert abs(pm.coefs["havoc"]) < 0.02


def test_fit_v3_recovers_blend_zeroes_irrelevant_context_and_drops_nuisance():
    t = synthetic_table()
    w = v3_fit.fit_v3(t)
    m, tot = w.margin.coefs, w.total.coefs
    assert m["margin_v2"] == pytest.approx(0.7, abs=0.05) and m["non_neutral"] == pytest.approx(1.5, abs=0.3)
    assert m["short_week_diff"] == pytest.approx(2.0, abs=0.3)
    assert m["travel_far_diff"] == 0.0 and m["talent_gap"] == 0.0 and m["bye_diff"] == 0.0   # can be fitted to 0
    assert tot["wind_excess"] == pytest.approx(-0.4, abs=0.05)
    assert tot["cold_excess"] == 0.0 and tot["precip"] == 0.0
    assert "weather_missing" not in tot                       # nuisance dropped at serve time
    assert w.meta["sigma_margin"] == pytest.approx(3.0, abs=0.4) and w.meta["n_train"] == len(t)
    assert w.meta["train_mae"]["margin"] < 3.0


def test_fit_v3_only_reads_training_seasons():
    t = synthetic_table()
    base = v3_fit.fit_v3(t)
    poisoned = t.copy()
    mask = poisoned["season"] == 2022
    poisoned.loc[mask, "actual_margin"] = 999.0
    held = v3_fit.fit_v3(poisoned, train_seasons=tuple(range(2016, 2022)))
    ref = v3_fit.fit_v3(t[t["season"] != 2022], train_seasons=tuple(range(2016, 2022)))
    assert held.margin == ref.margin and held.total == ref.total
    assert base.margin != held.margin
