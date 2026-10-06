import json
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3_gate as g


def frame(**cols):
    base = {"season": [2023, 2023, 2024, 2025], "actual_margin": [7.0, -3.0, 10.0, -14.0],
            "actual_total": [50.0, 41.0, 63.0, 38.0], "market_spread": [3.5, -6.5, 14.0, -3.0],
            "market_total": [47.5, 44.5, np.nan, 38.0]}
    base.update(cols)
    return pd.DataFrame(base)


def test_mae_and_side_accuracy_skip_pushes_and_no_pick_games():
    assert g.mae([1, 5], [3, 1]) == 3.0
    # lines 3.5 / -6.5 / 14 / -3 ; model margins 6 (home) / -10 (away) / 14 (no pick) / 0 (home)
    acc, n = g.side_accuracy([6.0, -10.0, 14.0, 0.0], [3.5, -6.5, 14.0, -3.0], [7.0, -3.0, 10.0, -14.0])
    # g1: model home, actual 7>3.5 home -> hit ; g2: model away(-10<-6.5), actual -3>-6.5 home -> miss
    # g3: pred == line -> skipped ; g4: model home (0>-3), actual -14<-3 away -> miss
    assert n == 3 and acc == pytest.approx(1 / 3)
    acc2, n2 = g.side_accuracy([1.0], [1.0], [1.0])
    assert math.isnan(acc2) and n2 == 0
    _, n3 = g.side_accuracy([5.0], [4.0], [4.0])                 # actual == line: a push
    assert n3 == 0


def test_log_loss_and_ece():
    assert g.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert g.log_loss([1.0], [0.0]) > 10                          # clipped, finite
    perfect = g.ece([0.1] * 10 + [0.9] * 10, [0] * 9 + [1] + [1] * 9 + [0])
    assert perfect == pytest.approx(0.0)
    assert g.ece([0.9] * 10, [0] * 10) == pytest.approx(0.9)


def test_metrics_on_the_eval_frame():
    df = frame(pm=[6.0, -10.0, 14.0, 0.0], pt=[49.0, 40.0, 60.0, 40.0], pw=[0.8, 0.2, 0.9, 0.4])
    m = g.metrics(df, "pm", "pt", "pw")
    assert m["n"] == 4 and m["margin_mae"] == pytest.approx((1 + 7 + 4 + 14) / 4)
    assert m["total_mae"] == pytest.approx((1 + 1 + 3 + 2) / 4)
    assert m["n_ou"] == 2 and m["ou"] == 1.0   # game 3 has no closing total, game 4 pushes (38 vs 38)
    assert 0 < m["ml_logloss"] < 2 and 0 <= m["ml_ece"] <= 1


def test_eval_set_filters_season_scored_and_priced():
    t = frame()
    t.loc[1, "market_spread"] = np.nan
    t.loc[2, "actual_margin"] = np.nan
    assert list(g.eval_set(t)["season"]) == [2023, 2025]
    assert len(g.eval_set(t, seasons=(2024,))) == 0


def test_check_baseline_passes_within_rounding_and_stops_otherwise():
    ok = {"margin_mae": 12.6069, "total_mae": 13.0980, "ats": 0.48964}
    assert g.check_baseline(ok)["margin_mae"] == (12.61, 12.6069)
    for bad in ({**ok, "margin_mae": 12.70}, {**ok, "total_mae": 13.2}, {**ok, "ats": 0.50}):
        with pytest.raises(g.BaselineMismatch, match="NOT reproduced"):
            g.check_baseline(bad)


V2 = {"margin_mae": 12.61, "total_mae": 13.10, "ats": 0.49, "ou": 0.50, "ml_logloss": 0.60, "ml_ece": 0.020}


def test_verdict_every_criterion_decides_ship():
    better = {"margin_mae": 12.5, "total_mae": 13.0, "ats": 0.49, "ou": 0.50, "ml_logloss": 0.60, "ml_ece": 0.024}
    v = g.verdict(V2, better)
    assert v["ship"] and all(c["pass"] for c in v["criteria"].values())     # ties pass ats/ou/logloss; ece within slack
    for key, bad in (("margin_mae", 12.61), ("total_mae", 13.10), ("ats", 0.4899), ("ou", 0.4999),
                     ("ml_logloss", 0.6001), ("ml_ece", 0.0251)):
        worse = {**better, key: bad}
        out = g.verdict(V2, worse)
        assert not out["ship"] and not out["criteria"][key]["pass"], key


def test_verdict_flags_a_worse_season_without_changing_the_combined_verdict():
    better = {"margin_mae": 12.5, "total_mae": 13.0, "ats": 0.5, "ou": 0.5, "ml_logloss": 0.59, "ml_ece": 0.02}
    s2 = {2023: dict(V2), 2024: dict(V2)}
    s3 = {2023: {**better}, 2024: {**better, "margin_mae": 12.9}}
    v = g.verdict(V2, better, s2, s3)
    assert v["ship"] is True
    assert v["season_flags"] == [{"season": 2024, "criterion": "margin_mae", "v2": 12.61, "v3": 12.9}]


def test_clean_makes_json_safe():
    out = g.clean({2023: {"a": np.float64("nan"), "b": np.int64(3), "c": [np.float64(1.5), np.bool_(True)]}})
    assert out == {"2023": {"a": None, "b": 3, "c": [1.5, True]}}
    json.dumps(out, allow_nan=False)


def test_bias_is_mean_signed_error_pred_minus_actual():
    df = frame(pm=[10.0, -3.0, 12.0, -10.0], pt=[52.0, 45.0, 60.0, 40.0])
    b = g.bias(df, "pm", "pt")
    assert b["margin_bias"] == pytest.approx((3 + 0 + 2 + 4) / 4)
    assert b["total_bias"] == pytest.approx((2 + 4 - 3 + 2) / 4)
    bs = g.bias_by_season(df, "pm", "pt")
    assert set(bs) == {2023, 2024, 2025} and bs[2023]["n"] == 2
    assert bs[2023]["margin_bias"] == pytest.approx(1.5)


def test_paired_side_diff_is_the_per_game_difference_of_hits_on_the_commonly_decided_games():
    line = np.array([3.5, -6.5, 14.0, -3.0, 0.5])
    actual = np.array([7.0, -3.0, 10.0, -14.0, 3.0])
    a = np.array([6.0, -10.0, 14.0, 0.0, 2.0])         # game 3 is a no-pick for a (pred == line)
    b = np.array([6.0, 0.0, 20.0, 0.0, -2.0])
    d = g.paired_side_diff(a, b, line, actual)
    # decided for both: games 1, 2, 4, 5.  a hits: 1,0,0,1  b hits: 1,1,0,0  -> diffs 0,-1,0,1
    assert d["n"] == 4 and d["diff"] == pytest.approx(0.0)
    assert d["se"] == pytest.approx(np.std([0, -1, 0, 1], ddof=1) / 2)
    assert d["z"] == pytest.approx(0.0)
    same = g.paired_side_diff(b, b, line, actual)
    assert same["diff"] == 0.0 and same["se"] == 0.0 and same["z"] is None


def test_paired_logloss_diff_uses_per_game_loss_differences_on_decided_games():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    pa = np.array([0.8, 0.3, 0.6, 0.9])
    pb = np.array([0.7, 0.4, 0.5, 0.6])
    d = g.paired_logloss_diff(pa, pb, y)
    la = -(y * np.log(pa) + (1 - y) * np.log(1 - pa))
    lb = -(y * np.log(pb) + (1 - y) * np.log(1 - pb))
    assert d["n"] == 4 and d["diff"] == pytest.approx(np.mean(la - lb))
    assert d["se"] == pytest.approx(np.std(la - lb, ddof=1) / 2)
    assert d["diff"] == pytest.approx(g.log_loss(pa, y) - g.log_loss(pb, y))


def test_paired_diffs_wires_ats_ou_and_logloss_from_model_columns():
    df = frame(a_margin=[6.0, -10.0, 15.0, 0.0], b_margin=[6.0, 0.0, 20.0, 0.0],
               a_total=[50.0, 40.0, 70.0, 40.0], b_total=[49.0, 46.0, 70.0, 36.0],
               a_wp=[0.8, 0.3, 0.6, 0.9], b_wp=[0.7, 0.4, 0.5, 0.6])
    out = g.paired_diffs(df, "a", "b")
    assert set(out) == {"ats", "ou", "ml_logloss"}
    assert out["ats"]["n"] == 4 and out["ou"]["n"] == 2          # game 3 has no closing total
    assert out["ml_logloss"]["n"] == 4
    json.dumps(g.clean(out), allow_nan=False)
