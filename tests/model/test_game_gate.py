import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.model import props_eval
from sportsmodel.model.game_gate import (
    cover_prob_normal,
    cover_prob_pmf,
    game_metrics,
    gate_decision,
    over_prob_normal,
    over_prob_pmf,
    paired_diffs,
    win_prob_pmf,
)


def _phi(z):
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


# ---------------------------------------------------------------- pmf probs

def test_win_prob_symmetric_margin_pmf_is_half():
    # support -2..2 (offset = -2 is the margin value at index 0)
    pmf = [0.1, 0.2, 0.4, 0.2, 0.1]
    assert win_prob_pmf(pmf, -2) == pytest.approx(0.5)


def test_win_prob_ties_count_half():
    # margins -1, 0, 1, 2 with probs 0.2, 0.2, 0.3, 0.3
    pmf = [0.2, 0.2, 0.3, 0.3]
    assert win_prob_pmf(pmf, -1) == pytest.approx(0.6 + 0.5 * 0.2)


def test_win_prob_engine_convention_via_negated_offset():
    # engine.margin_pmf stores offset=half_range with pmf[i] = P(margin == i - offset)
    engine_out = {"offset": 2, "pmf": [0.0, 0.0, 0.0, 0.25, 0.75]}
    assert win_prob_pmf(engine_out["pmf"], -engine_out["offset"]) == pytest.approx(1.0)


def test_cover_prob_integer_line_excludes_push_and_renormalizes():
    # margins -2..2: 0.1, 0.2, 0.4, 0.2, 0.1 ; line = 1 -> push mass 0.2
    pmf = [0.1, 0.2, 0.4, 0.2, 0.1]
    # P(margin > 1) = 0.1 ; renormalize by 1 - 0.2
    assert cover_prob_pmf(pmf, -2, 1.0) == pytest.approx(0.1 / 0.8)


def test_cover_prob_half_point_line_never_pushes():
    pmf = [0.1, 0.2, 0.4, 0.2, 0.1]
    # P(margin > 0.5) = 0.3, no renormalization
    assert cover_prob_pmf(pmf, -2, 0.5) == pytest.approx(0.3)
    # negative line: home is an underdog of 1.5 -> P(margin > -1.5) = 0.9
    assert cover_prob_pmf(pmf, -2, -1.5) == pytest.approx(0.9)


def test_cover_prob_missing_line_or_all_push_is_none():
    pmf = [0.1, 0.2, 0.4, 0.2, 0.1]
    assert cover_prob_pmf(pmf, -2, None) is None
    assert cover_prob_pmf(pmf, -2, float("nan")) is None
    assert cover_prob_pmf([0.0, 1.0, 0.0], -1, 0.0) is None


def test_cover_prob_line_outside_support():
    pmf = [0.5, 0.5]  # margins 3, 4
    assert cover_prob_pmf(pmf, 3, 10.0) == pytest.approx(0.0)
    assert cover_prob_pmf(pmf, 3, -10.0) == pytest.approx(1.0)


def test_over_prob_pmf_push_excluded_and_half_point():
    # totals 0..4
    pmf = [0.1, 0.1, 0.3, 0.3, 0.2]
    assert over_prob_pmf(pmf, 2.0) == pytest.approx(0.5 / 0.7)
    assert over_prob_pmf(pmf, 2.5) == pytest.approx(0.5)
    assert over_prob_pmf(pmf, None) is None
    assert over_prob_pmf(pmf, float("nan")) is None
    assert over_prob_pmf([0.0, 0.0, 1.0], 2.0) is None


# ---------------------------------------------------------------- normal probs

def test_normal_probs_match_closed_form():
    assert cover_prob_normal(3.0, 13.5, 3.0) == pytest.approx(0.5)
    assert cover_prob_normal(3.0, 13.5, -2.5) == pytest.approx(1.0 - _phi((-2.5 - 3.0) / 13.5))
    assert cover_prob_normal(-1.0, 10.0, 6.5) == pytest.approx(1.0 - _phi(7.5 / 10.0))
    assert over_prob_normal(44.0, 10.0, 47.5) == pytest.approx(1.0 - _phi(0.35))
    assert over_prob_normal(44.0, 10.0, 44.0) == pytest.approx(0.5)


def test_normal_probs_missing_line_is_none():
    assert cover_prob_normal(3.0, 13.5, None) is None
    assert over_prob_normal(44.0, 10.0, float("nan")) is None


def test_normal_probs_reject_nonpositive_sigma():
    with pytest.raises(ValueError):
        cover_prob_normal(0.0, 0.0, 1.5)


# ---------------------------------------------------------------- game_metrics

def _rec(season=2024, week=1, home="KC", win=0.6, cover=0.5, over=0.5,
         pm=3.0, pt=45.0, am=7, at=41, sl=3.0, tl=44.5):
    return {"season": season, "week": week, "home": home,
            "win_prob": win, "cover_prob": cover, "over_prob": over,
            "pred_margin": pm, "pred_total": pt,
            "actual_margin": am, "actual_total": at,
            "spread_line": sl, "total_line": tl}


def test_game_metrics_win_brier_tie_is_half():
    recs = [_rec(home="A", win=0.8, am=7), _rec(home="B", win=0.8, am=0),
            _rec(home="C", win=0.8, am=-3)]
    m = game_metrics(recs)
    # (0.8-1)^2=0.04, (0.8-0.5)^2=0.09, (0.8-0)^2=0.64
    assert m["win_brier"] == pytest.approx((0.04 + 0.09 + 0.64) / 3)
    assert m["n_win"] == 3


def test_game_metrics_cover_brier_excludes_push_and_missing_line():
    recs = [
        _rec(home="A", cover=0.7, am=7, sl=3.0),     # covers -> 0.09
        _rec(home="B", cover=0.7, am=3, sl=3.0),     # push -> excluded
        _rec(home="C", cover=0.7, am=1, sl=3.0),     # fails -> 0.49
        _rec(home="D", cover=None, am=1, sl=None),   # no line -> excluded
        _rec(home="E", cover=0.4, am=-3, sl=-2.5),   # -3 > -2.5 false -> 0.16
    ]
    m = game_metrics(recs)
    assert m["n_cover"] == 3
    assert m["cover_brier"] == pytest.approx((0.09 + 0.49 + 0.16) / 3)
    # missing line only drops the cover market, not win/margin
    assert m["n_win"] == 5
    assert m["n_margin"] == 5


def test_game_metrics_over_brier_excludes_push_and_mae():
    recs = [
        _rec(home="A", over=0.6, at=50, tl=44.0, pm=3.0, am=7, pt=45.0),   # over -> 0.16
        _rec(home="B", over=0.6, at=44, tl=44.0, pm=-2.0, am=-5, pt=40.0),  # push
        _rec(home="C", over=0.6, at=30, tl=float("nan"), pm=1.0, am=1, pt=30.0),  # no line
        _rec(home="D", over=0.3, at=20, tl=41.5, pm=0.0, am=10, pt=48.0),  # under -> 0.09
    ]
    m = game_metrics(recs)
    assert m["n_over"] == 2
    assert m["over_brier"] == pytest.approx((0.16 + 0.09) / 2)
    assert m["margin_mae"] == pytest.approx((4 + 3 + 0 + 10) / 4)
    assert m["total_mae"] == pytest.approx((5 + 4 + 0 + 28) / 4)
    assert m["n_margin"] == 4 and m["n_total"] == 4


def test_game_metrics_empty_market_is_nan_with_zero_n():
    m = game_metrics([_rec(cover=None, sl=None)])
    assert m["n_cover"] == 0
    assert math.isnan(m["cover_brier"])


# ---------------------------------------------------------------- paired_diffs

def test_paired_diffs_inner_join_and_errors():
    ml = [_rec(home="A", win=0.8, am=7, pm=5.0), _rec(home="B", win=0.3, am=-4),
          _rec(week=2, home="A")]
    elo = [_rec(home="B", win=0.4, am=-4), _rec(home="A", win=0.6, am=7, pm=10.0),
           _rec(week=3, home="Z")]
    df = paired_diffs(ml, elo)
    assert len(df) == 2
    assert list(df["home"]) == ["A", "B"]
    a = df.iloc[0]
    assert a["win_sq_ml"] == pytest.approx(0.04)
    assert a["win_sq_elo"] == pytest.approx(0.16)
    assert a["margin_ae_ml"] == pytest.approx(2.0)
    assert a["margin_ae_elo"] == pytest.approx(3.0)
    assert set(df["cluster"]) == {"2024-1"}


def test_paired_diffs_push_rows_are_nan_for_that_market():
    df = paired_diffs([_rec(am=3, sl=3.0)], [_rec(am=3, sl=3.0)])
    assert math.isnan(df.iloc[0]["cover_sq_ml"])
    assert math.isnan(df.iloc[0]["cover_sq_elo"])
    assert not math.isnan(df.iloc[0]["win_sq_ml"])


def test_paired_diffs_rejects_duplicate_keys():
    with pytest.raises(ValueError):
        paired_diffs([_rec(), _rec()], [_rec()])


def test_paired_metric_means_agree_with_game_metrics():
    rng = np.random.default_rng(3)
    recs = [_rec(week=w, home=h, win=float(rng.uniform()), cover=float(rng.uniform()),
                 over=float(rng.uniform()), am=int(rng.integers(-14, 15)),
                 at=int(rng.integers(20, 60)), pm=float(rng.normal()), pt=44.0)
            for w in range(1, 5) for h in ("A", "B", "C")]
    df = paired_diffs(recs, recs)
    m = game_metrics(recs)
    assert df["win_sq_ml"].mean() == pytest.approx(m["win_brier"])
    assert df["cover_sq_ml"].mean() == pytest.approx(m["cover_brier"])
    assert df["total_ae_elo"].mean() == pytest.approx(m["total_mae"])


# ---------------------------------------------------------------- gate_decision

def _paired_games(ml_win, elo_win, ml_pm, elo_pm, n_weeks=6, games=3):
    ml, elo = [], []
    for w in range(1, n_weeks + 1):
        for g in range(games):
            h = f"T{g}"
            am = 7 if (w + g) % 2 else -3
            ml.append(_rec(week=w, home=h, win=ml_win if am > 0 else 1 - ml_win,
                           cover=ml_win if am > 3 else 1 - ml_win,
                           over=ml_win, at=50, pm=am + ml_pm, am=am, pt=50.0))
            elo.append(_rec(week=w, home=h, win=elo_win if am > 0 else 1 - elo_win,
                            cover=elo_win if am > 3 else 1 - elo_win,
                            over=elo_win, at=50, pm=am + elo_pm, am=am, pt=50.0))
    return paired_diffs(ml, elo)


def test_gate_passes_when_ml_better_everywhere():
    df = _paired_games(ml_win=0.8, elo_win=0.6, ml_pm=2.0, elo_pm=4.0)
    d = gate_decision(df, n_boot=50, seed=0)
    assert d["pass"] is True
    assert d["reasons"] == []
    wb = d["metrics"]["win_brier"]
    assert wb["ml"] == pytest.approx(0.04)
    assert wb["elo"] == pytest.approx(0.16)
    assert wb["diff"] == pytest.approx(-0.12)
    assert wb["lo"] <= wb["diff"] <= wb["hi"]
    assert d["metrics"]["margin_mae"]["n"] == 18


def test_gate_fails_on_brier_and_names_metrics():
    df = _paired_games(ml_win=0.6, elo_win=0.8, ml_pm=2.0, elo_pm=4.0)
    d = gate_decision(df, n_boot=50, seed=0)
    assert d["pass"] is False
    joined = " ".join(d["reasons"])
    for name in ("win_brier", "cover_brier", "over_brier"):
        assert name in joined
    assert "margin_mae" not in joined
    assert "total_mae" not in joined


def test_gate_mae_tolerance_is_one_percent():
    # ML margin error 101 vs Elo 100 -> exactly 1.01x -> pass (worse, but tolerated)
    ok = gate_decision(_paired_games(0.8, 0.8, 101.0, 100.0), n_boot=20, seed=0)
    assert ok["pass"] is True, ok["reasons"]
    bad = gate_decision(_paired_games(0.8, 0.8, 102.0, 100.0), n_boot=20, seed=0)
    assert bad["pass"] is False
    assert any("margin_mae" in r for r in bad["reasons"])
    assert not any("total_mae" in r for r in bad["reasons"])


def test_gate_empty_frame_fails():
    d = gate_decision(paired_diffs([], []), n_boot=10, seed=0)
    assert d["pass"] is False
    assert d["reasons"]


def test_gate_bootstrap_uses_props_eval_cluster_bootstrap_by_season_week(monkeypatch):
    seen = []
    real = props_eval.cluster_bootstrap

    def spy(df, stat, n_boot=1000, seed=0):
        seen.append(sorted(df["cluster"].unique()))
        return real(df, stat, n_boot=n_boot, seed=seed)

    monkeypatch.setattr(props_eval, "cluster_bootstrap", spy)
    df = _paired_games(0.8, 0.6, 2.0, 4.0, n_weeks=3, games=4)
    gate_decision(df, n_boot=5, seed=0)
    assert len(seen) == 5
    assert seen[0] == ["2024-1", "2024-2", "2024-3"]


def test_gate_bootstrap_resamples_clusters_with_replacement_and_is_seeded():
    # One heavy week where ML is far worse; the rest ML is slightly better.
    ml, elo = [], []
    for w in range(1, 9):
        for g in range(3):
            am = 7
            ml_win = 0.1 if w == 1 else 0.8 + 0.02 * w  # per-week diffs vary
            ml.append(_rec(week=w, home=f"T{g}", win=ml_win, am=am))
            elo.append(_rec(week=w, home=f"T{g}", win=0.8, am=am))
    df = paired_diffs(ml, elo)
    a = gate_decision(df, n_boot=300, seed=11)
    b = gate_decision(df, n_boot=300, seed=11)
    c = gate_decision(df, n_boot=300, seed=12)
    wa, wb, wc = (x["metrics"]["win_brier"] for x in (a, b, c))
    assert (wa["lo"], wa["hi"]) == (wb["lo"], wb["hi"])
    assert (wa["lo"], wa["hi"]) != (wc["lo"], wc["hi"])
    # with replacement the heavy week can be drawn 0..k times -> CI straddles point
    assert wa["lo"] < wa["diff"] < wa["hi"]
