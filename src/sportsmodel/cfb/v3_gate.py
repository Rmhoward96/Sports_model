"""The v3 ship gate (spec section 4): pure metrics, the v2-baseline reproduction check and the
pass/fail verdict. No IO.

Eval set (identical to the one v2's 12.61 / 13.10 / 49.0 % were measured on): held-out 2023-2025
REG FBS-vs-FBS games that have a closing spread. O/U is scored on the subset with a closing total.
`market_spread` is in HOME-MARGIN convention (positive = home favored): the model picks home when
its margin is above the number. Pushes and no-pick games are excluded from ATS / O/U.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOLDOUT_SEASONS = (2023, 2024, 2025)
# v2's held-out numbers (compare_cfb_live_fix.py "fixed", 2026-09-30) -- reproduced FIRST
V2_EXPECTED = {"margin_mae": 12.61, "total_mae": 13.10, "ats": 0.490}
BASELINE_TOL = {"margin_mae": 0.006, "total_mae": 0.006, "ats": 0.0006}   # rounding of the published figures
ECE_SLACK = 0.005
ECE_BINS = 10


class BaselineMismatch(RuntimeError):
    """v2 run through the gate harness did not reproduce its published held-out numbers."""


def eval_set(table: pd.DataFrame, seasons=HOLDOUT_SEASONS) -> pd.DataFrame:
    return table[table["season"].isin(seasons) & table["actual_margin"].notna()
                 & table["market_spread"].notna()].reset_index(drop=True)


def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(actual, float))))


def side_accuracy(pred, line, actual) -> tuple[float, int]:
    """Share of decided games where the model's side of `line` matches the actual side
    (pushes and games with pred == line are skipped). Returns (accuracy, n decided)."""
    pred, line, actual = (np.asarray(x, float) for x in (pred, line, actual))
    ok = ~np.isnan(line) & (actual != line) & (pred != line)
    if not ok.any():
        return float("nan"), 0
    return float(np.mean((pred[ok] > line[ok]) == (actual[ok] > line[ok]))), int(ok.sum())


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins: int = ECE_BINS) -> float:
    """Expected calibration error: bin-size-weighted mean |mean(p) - mean(y)| over equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(p))


def metrics(df: pd.DataFrame, margin_col: str, total_col: str, prob_col: str) -> dict:
    """Gate metrics of one model on an eval-set frame (needs actual_*, market_*, the three columns)."""
    ats, n_ats = side_accuracy(df[margin_col], df["market_spread"], df["actual_margin"])
    has_t = df["market_total"].notna()
    ou, n_ou = side_accuracy(df.loc[has_t, total_col], df.loc[has_t, "market_total"], df.loc[has_t, "actual_total"])
    dec = df["actual_margin"] != 0
    y = (df.loc[dec, "actual_margin"] > 0).astype(float)
    return {"n": int(len(df)), "margin_mae": mae(df[margin_col], df["actual_margin"]),
            "total_mae": mae(df[total_col], df["actual_total"]),
            "ats": ats, "n_ats": n_ats, "ou": ou, "n_ou": n_ou,
            "ml_logloss": log_loss(df.loc[dec, prob_col], y), "ml_ece": ece(df.loc[dec, prob_col], y)}


def by_season(df: pd.DataFrame, margin_col: str, total_col: str, prob_col: str) -> dict:
    return {int(s): metrics(g, margin_col, total_col, prob_col) for s, g in df.groupby("season")}


def bias(df: pd.DataFrame, margin_col: str, total_col: str) -> dict:
    """Mean SIGNED error (prediction - actual; positive = over-predicts) of margin and total.
    Reported for every model/season (ruling F2); never part of the verdict and never corrected."""
    return {"n": int(len(df)),
            "margin_bias": float(np.mean(np.asarray(df[margin_col], float) - np.asarray(df["actual_margin"], float))),
            "total_bias": float(np.mean(np.asarray(df[total_col], float) - np.asarray(df["actual_total"], float)))}


def bias_by_season(df: pd.DataFrame, margin_col: str, total_col: str) -> dict:
    return {int(s): bias(g, margin_col, total_col) for s, g in df.groupby("season")}


def _paired(d: np.ndarray) -> dict:
    """Mean and standard error (sd / sqrt(n), ddof=1) of a vector of per-game differences."""
    n = int(len(d))
    diff = float(np.mean(d)) if n else float("nan")
    se = float(np.std(d, ddof=1) / math.sqrt(n)) if n > 1 else float("nan")
    z = diff / se if n > 1 and se > 0 else None
    return {"n": n, "diff": diff, "se": se, "z": z}


def paired_side_diff(pred_a, pred_b, line, actual) -> dict:
    """Paired comparison of two models' side accuracy (A - B) against `line`: per-game hit
    differences on the games decided for BOTH (no push, neither model on the line). The headline
    accuracies in `metrics` use each model's own decided set; this is the matched-pairs version."""
    pa, pb, line, actual = (np.asarray(x, float) for x in (pred_a, pred_b, line, actual))
    ok = ~np.isnan(line) & (actual != line) & (pa != line) & (pb != line)
    hit_a = (pa[ok] > line[ok]) == (actual[ok] > line[ok])
    hit_b = (pb[ok] > line[ok]) == (actual[ok] > line[ok])
    return _paired(hit_a.astype(float) - hit_b.astype(float))


def paired_logloss_diff(p_a, p_b, y) -> dict:
    """Paired comparison of two win-probability models' log-loss (A - B): per-game loss differences."""
    y = np.asarray(y, float)

    def loss(p):
        p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))
    return _paired(loss(p_a) - loss(p_b))


def paired_diffs(df: pd.DataFrame, a: str, b: str) -> dict:
    """Paired (A - B) differences, with standard errors, for ATS, O/U and ML log-loss on an
    eval-set frame holding `<a>_margin/_total/_wp` and `<b>_...` columns."""
    has_t = df["market_total"].notna()
    dec = df["actual_margin"] != 0
    y = (df.loc[dec, "actual_margin"] > 0).astype(float)
    return {"ats": paired_side_diff(df[f"{a}_margin"], df[f"{b}_margin"], df["market_spread"], df["actual_margin"]),
            "ou": paired_side_diff(df.loc[has_t, f"{a}_total"], df.loc[has_t, f"{b}_total"],
                                   df.loc[has_t, "market_total"], df.loc[has_t, "actual_total"]),
            "ml_logloss": paired_logloss_diff(df.loc[dec, f"{a}_wp"], df.loc[dec, f"{b}_wp"], y)}


def check_baseline(measured: dict, expected: dict = V2_EXPECTED, tol: dict = BASELINE_TOL) -> dict:
    """Raise BaselineMismatch unless v2's measured combined numbers equal the published ones within
    rounding; returns {metric: (expected, measured)} on success."""
    diffs = {k: (expected[k], measured[k]) for k in expected}
    bad = {k: v for k, v in diffs.items() if abs(v[0] - v[1]) > tol[k]}
    if bad:
        raise BaselineMismatch("v2 baseline NOT reproduced (expected vs measured): "
                               + ", ".join(f"{k} {e} vs {m:.4f}" for k, (e, m) in bad.items()))
    return diffs


CRITERIA = (
    ("margin_mae", "v3 margin MAE < v2", lambda v2, v3: v3 < v2),
    ("total_mae", "v3 total MAE < v2", lambda v2, v3: v3 < v2),
    ("ats", "ATS vs closing spread >= v2", lambda v2, v3: v3 >= v2),
    ("ou", "O/U vs closing total >= v2", lambda v2, v3: v3 >= v2),
    ("ml_logloss", "ML log-loss <= v2", lambda v2, v3: v3 <= v2),
    ("ml_ece", f"ML ECE <= v2 + {ECE_SLACK}", lambda v2, v3: v3 <= v2 + ECE_SLACK),
)


def verdict(v2: dict, v3: dict, v2_seasons: dict | None = None, v3_seasons: dict | None = None) -> dict:
    """Pass/fail per criterion on the COMBINED numbers (the verdict) plus a list of (season,
    criterion) pairs where v3 is worse than v2 (flagged, never part of the verdict)."""
    crit = {k: {"label": label, "v2": v2[k], "v3": v3[k], "pass": bool(rule(v2[k], v3[k]))}
            for k, label, rule in CRITERIA}
    flags = []
    for s in sorted(v3_seasons or {}):
        for k, label, rule in CRITERIA:
            if not rule(v2_seasons[s][k], v3_seasons[s][k]):
                flags.append({"season": int(s), "criterion": k, "v2": v2_seasons[s][k], "v3": v3_seasons[s][k]})
    return {"criteria": crit, "ship": all(c["pass"] for c in crit.values()), "season_flags": flags}


def clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> python, int keys -> str."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
