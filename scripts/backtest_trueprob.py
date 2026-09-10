"""Leak-free walk-forward backtest for the true-probability feature model
(margin/total ridge regression + win-prob calibration) -> GO/NO-GO report.

Sub-project 3 (+EV engine), task 3 -- see
.superpowers/sdd/2026-09-09-plus-ev-3-trueprob-model/task-3-brief.md.

Reuses, untouched:
- `sportsmodel.model.trueprob`: FEATURES, feature_matrix, fit_ridge,
  fit_model, predict, residual_sigma (Task 1).
- `sportsmodel.model.trueprob_prob`: win_prob (Task 2).
- `sportsmodel.model.calibration`: fit, apply (Platt scaling).

=== Leak-free walk-forward ===
For each season S with >=1 prior season: margin/total models are fit ONLY
on `df[df.season < S]` (`fit_model` drops NaN-target rows itself); sigma_margin/
sigma_total come from the TRAIN residuals (never test); the win-prob Platt
calibrator is fit on the prior seasons' OWN (win_prob, home_win) pairs --
computed from that same seasons-<S model -- and applied to season S. Season
S never contributes anything to its own model, sigma, or calibration.

=== Closing-line convention (READ BEFORE TRUSTING THE NUMBERS) ===
Every metric here compares model output to ACTUAL home margin
(`home_score - away_score`) and to the closing line's IMPLIED home margin,
in the SAME sign convention: positive = home favored / home did better than
an even game.

- CFB (`assets/cfb/lines.parquet`, column `market_spread`): already stored
  in home-margin convention by `build_cfb_lines.py` ("CFBD's spread is home
  spread in book convention (home favored => negative); we store
  market_spread in home-margin convention (home favored => positive) =
  -spread"). Used AS-IS, no negation, by `_join_cfb_lines`.

- NFL (`assets/nfl/schedules.parquet`, column `spread_line`): the initial
  assumption for this task was that nflverse's `spread_line` is in "book
  convention" (home favored => negative), which would require negating it
  to get an implied home margin. **That assumption was checked against this
  repo's actual asset and is WRONG for it.**

  Verification (2026-09-09, `assets/nfl/schedules.parquet`, all completed
  games, n=6688): regressing `actual_margin` (home_score - away_score) on
  `spread_line` gives slope ~1.05 / intercept ~-0.05 (i.e. `spread_line`
  tracks `actual_margin` almost 1:1, not -1:1); `mean(|actual_margin -
  spread_line|)` ~= 10.25 vs `mean(|actual_margin - (-spread_line)|)` ~= 14.64.
  A ~10-point MAE is the well-known ballpark for how well an NFL closing
  spread predicts the final margin; ~14.6 is not (worse than guessing "home
  wins by 0" for many seasons). This is confirmed by the codebase itself:
  `sportsmodel.nfl.gameline.build_gameline` (used by
  `scripts/backtest_nfl_gameline.py`, the P2 gate) calls
  `shrink(model_margin, market["spread_line"], ...)`, blending `spread_line`
  directly against `model_margin` with NO sign flip -- i.e. the existing,
  already-tested NFL gameline path treats `spread_line` as home-margin
  convention already. So `spread_line` IS the implied home margin
  (positive = home favored) here -- used AS-IS, no negation, by
  `_join_nfl_lines`.

Both conventions converge on the same output shape: a `spread_line` column
holding the CLOSING LINE'S IMPLIED HOME MARGIN, directly comparable to
`home_score - away_score`. `total_line`/`market_total` need no conversion
(a total has no "side").
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.model import trueprob
from sportsmodel.model.calibration import apply as calib_apply
from sportsmodel.model.calibration import fit as calib_fit
from sportsmodel.model.trueprob import fit_model, predict, residual_sigma
from sportsmodel.model.trueprob_prob import win_prob

ASSETS = Path(__file__).resolve().parents[1] / "assets"
REPORT_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs" / "superpowers" / "reports" / "2026-09-09-plus-ev-trueprob-backtest.md"
)

# CFB's build_features.assemble() never emits an NFL-style "rest_diff"
# column -- CFB rest is home_rest_weeks/away_rest_weeks/home_off_bye/
# away_off_bye (a week-gap/bye proxy, not a day-count diff), so there is no
# CFB column to substitute in for it. CFB substitutes off_ppa_diff/
# def_ppa_diff for NFL's off_epa_diff/def_epa_diff and simply omits rest.
CFB_FEATURES: list[str] = [
    "elo_diff",
    "last10_diff",
    "sos_diff",
    "sov_diff",
    "off_ppa_diff",
    "def_ppa_diff",
]

# Minimum prior-seasons sample size to fit a Platt calibrator; below this,
# calibration is skipped and the raw win_prob is used unchanged (a
# handful of games produces a wildly overfit (a, b)).
MIN_CALIB_SAMPLES = 30

DEFAULT_N_BINS = 10


# ---------------------------------------------------------------------------
# PURE metric helpers (unit-tested in tests/test_backtest_trueprob.py)
# ---------------------------------------------------------------------------

def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def mae(pred: list[float], actual: list[float]) -> float:
    """Mean absolute error between two equal-length plain lists of floats.
    NaN on an empty input (rather than a ZeroDivisionError)."""
    pairs = list(zip(pred, actual))
    if not pairs:
        return float("nan")
    return sum(abs(float(p) - float(a)) for p, a in pairs) / len(pairs)


def mae_vs_line(preds: list[dict], actual_key: str, line_key: str) -> float:
    """Mean |line - actual| -- the MARKET's own error against the same
    target `mae()` scores the model against, so the two are directly
    comparable (lower is better for both). Rows whose `line_key` is
    missing/None/NaN are skipped (no market line for that game)."""
    pairs = [
        (row[line_key], row[actual_key])
        for row in preds
        if _finite(row.get(line_key))
    ]
    if not pairs:
        return float("nan")
    return sum(abs(float(l) - float(a)) for l, a in pairs) / len(pairs)


def brier(probs: list[float], outcomes: list[int]) -> float:
    """Mean squared error between predicted probabilities and binary
    outcomes (0/1). NaN on an empty input."""
    pairs = list(zip(probs, outcomes))
    if not pairs:
        return float("nan")
    return sum((float(p) - float(o)) ** 2 for p, o in pairs) / len(pairs)


def reliability(probs: list[float], outcomes: list[int], n_bins: int = DEFAULT_N_BINS) -> list[dict]:
    """Calibration table: equal-width bins over [0, 1]. Bin i covers
    `[i/n_bins, (i+1)/n_bins)`, except the last bin which is closed at 1.0
    (so a prob of exactly 1.0 lands somewhere). Empty bins are omitted
    entirely rather than returned with n=0. Each entry is
    `{bin_lo, bin_hi, n, mean_pred, mean_actual}` -- a well-calibrated model
    has `mean_pred` ~= `mean_actual` in every bin."""
    table = []
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        if i == n_bins - 1:
            members = [(p, o) for p, o in zip(probs, outcomes) if lo <= p <= hi]
        else:
            members = [(p, o) for p, o in zip(probs, outcomes) if lo <= p < hi]
        if not members:
            continue
        n = len(members)
        table.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "n": n,
            "mean_pred": sum(p for p, _ in members) / n,
            "mean_actual": sum(o for _, o in members) / n,
        })
    return table


def clv_proxy(preds: list[dict], pred_key: str = "pred_margin", line_key: str = "spread_line") -> float:
    """Signed, PRE-GAME model-vs-closing-line edge, averaged:
    `mean(pred_margin - spread_line)` over rows with a line present.

    Sign convention: positive => on average the model expects HOME to do
    better than the closing line implies (model_margin > line's implied
    home margin); negative => the model favors AWAY relative to the market.
    A value near 0 means the model tracks the market's side on average with
    no systematic home/away bias -- it says nothing about accuracy (that is
    `mae_vs_line`'s job).

    This is a *disagreement-direction* proxy, not a realized closing-line-
    value metric: true CLV needs an OPENING line to compare the close
    against, which this repo does not retain for either sport (only closing
    lines are stored) -- see `scripts/backtest_cfb_priors.py`'s design note
    #4 for the same caveat on the CFB side.
    """
    pairs = [
        (row[pred_key], row[line_key])
        for row in preds
        if _finite(row.get(line_key)) and _finite(row.get(pred_key))
    ]
    if not pairs:
        return float("nan")
    return sum(float(p) - float(l) for p, l in pairs) / len(pairs)


def disagreement_rate(preds: list[dict], pred_key: str = "pred_margin", line_key: str = "spread_line") -> float:
    """Share of games (with a line present) where the model's side (sign of
    `pred_margin`) differs from the line's side (sign of the line's implied
    home margin). Both are already in the SAME home-margin convention (see
    module docstring), so this is a direct sign comparison -- no negation.
    A `pred_margin`/`spread_line` of exactly 0.0 has no side to agree or
    disagree with, so such rows are excluded from the denominator. NaN if
    no row qualifies."""
    considered = [
        (row[pred_key], row[line_key])
        for row in preds
        if _finite(row.get(line_key)) and _finite(row.get(pred_key))
    ]
    considered = [(p, l) for p, l in considered if p != 0.0 and l != 0.0]
    if not considered:
        return float("nan")
    disagree = sum(1 for p, l in considered if (p > 0) != (l > 0))
    return disagree / len(considered)


# ---------------------------------------------------------------------------
# Leak-free walk-forward
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: pd.DataFrame,
    sport: str,
    features: list[str],
    alpha: float = 1.0,
    min_calib_samples: int = MIN_CALIB_SAMPLES,
) -> dict:
    """Leak-free walk-forward over `df`'s seasons.

    `df` is a features-parquet-shaped frame: one row per completed game,
    with `season`, `week`, `home_team`, `away_team`, the feature columns
    named in `features`, `margin` (home_score - away_score), `total`
    (home_score + away_score), and the closing line already joined in and
    sign-converted by the caller (`main()` / `_join_nfl_lines` /
    `_join_cfb_lines`) as `spread_line` (implied home margin) and
    `total_line`.

    `sport` is accepted for labeling only (it does not change any modeling
    logic here -- `features` already encodes the sport-specific column set,
    and the win-prob math in `trueprob_prob.win_prob` is sport-agnostic).

    For every season S that has at least one prior season in `df`: fit
    margin/total ridge models on `df[df.season < S]` only (`fit_model` drops
    NaN-target rows itself); take `sigma_margin`/`sigma_total` from the
    TRAINING residuals; Platt-calibrate the win probability using the
    (win_prob, home_win) pairs of THOSE SAME PRIOR SEASONS (computed from
    the season-<S model), falling back to the raw win_prob if there are
    fewer than `min_calib_samples` prior games; then predict and score
    season S. Season S's own rows never influence its model, sigma, or
    calibrator -- the leak-free invariant.

    Returns a dict of aggregated metrics plus the full per-game `rows` list
    (each row is a plain dict, suitable input to the pure helpers above).
    """
    seasons = sorted(int(s) for s in df["season"].dropna().unique())
    rows: list[dict] = []

    for season in seasons[1:]:  # first season has no prior season to train on
        train_df = df.loc[df["season"] < season].reset_index(drop=True)
        test_df = df.loc[df["season"] == season].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        margin_model = fit_model(train_df, "margin", features, alpha)
        total_model = fit_model(train_df, "total", features, alpha)

        train_margin_clean = train_df.loc[train_df["margin"].notna()].reset_index(drop=True)
        train_pred_margin = predict(margin_model, train_margin_clean)
        sigma_margin = residual_sigma(
            train_margin_clean["margin"].to_numpy(dtype=float), train_pred_margin
        )

        train_total_clean = train_df.loc[train_df["total"].notna()].reset_index(drop=True)
        train_pred_total = predict(total_model, train_total_clean)
        sigma_total = residual_sigma(
            train_total_clean["total"].to_numpy(dtype=float), train_pred_total
        )

        # Platt calibration, fit on the PRIOR seasons' own (win_prob,
        # home_win) pairs (from the season-<S model) -- never on season S.
        train_win_probs = [win_prob(float(m), sigma_margin) for m in train_pred_margin]
        train_home_win = (train_margin_clean["margin"].to_numpy(dtype=float) > 0).astype(int).tolist()
        if len(train_win_probs) >= min_calib_samples:
            calib_params = calib_fit(train_win_probs, train_home_win)
        else:
            calib_params = None

        test_clean = test_df.loc[
            test_df["margin"].notna() & test_df["total"].notna()
        ].reset_index(drop=True)
        if test_clean.empty:
            continue

        pred_margin = predict(margin_model, test_clean)
        pred_total = predict(total_model, test_clean)
        win_probs_raw = [win_prob(float(m), sigma_margin) for m in pred_margin]
        win_probs_cal = [calib_apply(p, calib_params) for p in win_probs_raw]

        for i in range(len(test_clean)):
            r = test_clean.iloc[i]
            actual_margin = float(r["margin"])
            rows.append({
                "season": int(r["season"]),
                "week": int(r["week"]) if pd.notna(r.get("week")) else None,
                "home_team": r.get("home_team"),
                "away_team": r.get("away_team"),
                "pred_margin": float(pred_margin[i]),
                "actual_margin": actual_margin,
                "pred_total": float(pred_total[i]),
                "actual_total": float(r["total"]),
                "spread_line": float(r["spread_line"]) if _finite(r.get("spread_line")) else None,
                "total_line": float(r["total_line"]) if _finite(r.get("total_line")) else None,
                "win_prob_raw": float(win_probs_raw[i]),
                "win_prob_calibrated": float(win_probs_cal[i]),
                "home_win": 1 if actual_margin > 0 else 0,
                "sigma_margin": sigma_margin,
                "sigma_total": sigma_total,
                "calibrated": calib_params is not None,
            })

    if not rows:
        return {
            "sport": sport,
            "n_games": 0,
            "seasons_scored": [],
            "model_margin_mae": float("nan"),
            "line_margin_mae": float("nan"),
            "margin_mae_delta": float("nan"),
            "model_total_mae": float("nan"),
            "line_total_mae": float("nan"),
            "total_mae_delta": float("nan"),
            "brier_raw": float("nan"),
            "brier_calibrated": float("nan"),
            "reliability": [],
            "mean_clv_proxy": float("nan"),
            "disagreement_rate": float("nan"),
            "rows": [],
        }

    model_margin_mae = mae([r["pred_margin"] for r in rows], [r["actual_margin"] for r in rows])
    line_margin_mae = mae_vs_line(rows, "actual_margin", "spread_line")
    model_total_mae = mae([r["pred_total"] for r in rows], [r["actual_total"] for r in rows])
    line_total_mae = mae_vs_line(rows, "actual_total", "total_line")
    brier_raw = brier([r["win_prob_raw"] for r in rows], [r["home_win"] for r in rows])
    brier_calibrated = brier([r["win_prob_calibrated"] for r in rows], [r["home_win"] for r in rows])
    reliability_table = reliability(
        [r["win_prob_calibrated"] for r in rows], [r["home_win"] for r in rows]
    )
    mean_clv = clv_proxy(rows, "pred_margin", "spread_line")
    dis_rate = disagreement_rate(rows, "pred_margin", "spread_line")

    return {
        "sport": sport,
        "n_games": len(rows),
        "seasons_scored": sorted({r["season"] for r in rows}),
        "model_margin_mae": model_margin_mae,
        "line_margin_mae": line_margin_mae,
        "margin_mae_delta": model_margin_mae - line_margin_mae,
        "model_total_mae": model_total_mae,
        "line_total_mae": line_total_mae,
        "total_mae_delta": model_total_mae - line_total_mae,
        "brier_raw": brier_raw,
        "brier_calibrated": brier_calibrated,
        "reliability": reliability_table,
        "mean_clv_proxy": mean_clv,
        "disagreement_rate": dis_rate,
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Data loading / line joins (thin IO; main()'s responsibility)
# ---------------------------------------------------------------------------

def _join_nfl_lines(features_df: pd.DataFrame, schedules_path: Path) -> pd.DataFrame:
    """Join `assets/nfl/schedules.parquet`'s `spread_line`/`total_line` onto
    `features_df` by `game_id` (falling back to season+week+home_team+
    away_team for any row the game_id join misses). See the module
    docstring for why `spread_line` is used AS-IS (already home-margin
    convention in this asset)."""
    sched = pd.read_parquet(schedules_path)
    key_cols = ["season", "week", "home_team", "away_team"]

    if "game_id" in features_df.columns and "game_id" in sched.columns:
        lines = sched[["game_id", "spread_line", "total_line"]].drop_duplicates(subset="game_id")
        merged = features_df.merge(lines, on="game_id", how="left")

        missing = merged["spread_line"].isna() & merged["total_line"].isna()
        if missing.any():
            fb_lines = sched[key_cols + ["spread_line", "total_line"]].drop_duplicates(subset=key_cols)
            fb = merged.loc[missing, key_cols].merge(fb_lines, on=key_cols, how="left")
            merged.loc[missing, "spread_line"] = fb["spread_line"].to_numpy()
            merged.loc[missing, "total_line"] = fb["total_line"].to_numpy()
        return merged

    lines = sched[key_cols + ["spread_line", "total_line"]].drop_duplicates(subset=key_cols)
    return features_df.merge(lines, on=key_cols, how="left")


def _join_cfb_lines(features_df: pd.DataFrame, lines_path: Path) -> pd.DataFrame:
    """Join `assets/cfb/lines.parquet`'s `market_spread`/`market_total` onto
    `features_df`, renamed to `spread_line`/`total_line` for a common shape
    with the NFL path. `market_spread` is already home-margin convention
    (build_cfb_lines.py) -- used AS-IS, no negation."""
    key_cols = ["season", "week", "home_team", "away_team"]
    lines = pd.read_parquet(lines_path)[key_cols + ["market_spread", "market_total"]]
    lines = lines.drop_duplicates(subset=key_cols)
    merged = features_df.merge(lines, on=key_cols, how="left")
    return merged.rename(columns={"market_spread": "spread_line", "market_total": "total_line"})


# ---------------------------------------------------------------------------
# GO/NO-GO verdict + reporting
# ---------------------------------------------------------------------------

# "within noise" tolerance (points of MAE) for the margin/total GO check --
# a model that is very slightly worse than the closing line by less than
# this is not distinguishable from the line "by noise" at typical NFL/CFB
# season sample sizes; anything worse than this is a genuine NO-GO signal.
NOISE_TOLERANCE_PTS = 0.25

# Mean absolute (mean_pred - mean_actual) across reliability bins, above
# which calibration is considered NOT reasonable for a GO.
CALIBRATION_TOLERANCE = 0.10


def go_no_go_verdict(result: dict, noise_tol: float = NOISE_TOLERANCE_PTS,
                      calib_tol: float = CALIBRATION_TOLERANCE) -> tuple[str, list[str]]:
    """GO iff the model's margin AND total MAE are each <= the closing
    line's MAE within `noise_tol` points, AND the (calibrated) reliability
    table's bins average within `calib_tol` of the diagonal. Returns
    `(verdict, reasons)` where `reasons` is a list of one line per check
    (for the printed table / report)."""
    reasons = []

    margin_ok = result["model_margin_mae"] <= result["line_margin_mae"] + noise_tol
    reasons.append(
        f"margin MAE: model={result['model_margin_mae']:.3f} "
        f"line={result['line_margin_mae']:.3f} delta={result['margin_mae_delta']:+.3f} "
        f"({'OK' if margin_ok else 'FAIL'} vs tolerance {noise_tol})"
    )

    total_ok = result["model_total_mae"] <= result["line_total_mae"] + noise_tol
    reasons.append(
        f"total MAE: model={result['model_total_mae']:.3f} "
        f"line={result['line_total_mae']:.3f} delta={result['total_mae_delta']:+.3f} "
        f"({'OK' if total_ok else 'FAIL'} vs tolerance {noise_tol})"
    )

    rel = result["reliability"]
    if rel:
        mean_gap = sum(abs(b["mean_pred"] - b["mean_actual"]) for b in rel) / len(rel)
    else:
        mean_gap = float("nan")
    calib_ok = _finite(mean_gap) and mean_gap <= calib_tol
    reasons.append(
        f"calibration: mean|mean_pred-mean_actual| across bins={mean_gap:.3f} "
        f"({'OK' if calib_ok else 'FAIL'} vs tolerance {calib_tol})"
    )

    verdict = "GO" if (margin_ok and total_ok and calib_ok) else "NO-GO"
    return verdict, reasons


def _print_report(result: dict, sport: str, alpha: float) -> None:
    print(f"\n=== +EV true-prob walk-forward backtest: {sport} (alpha={alpha}) ===")
    print(f"games scored: {result['n_games']}  seasons: {result['seasons_scored']}")
    print(
        f"margin MAE -- model: {result['model_margin_mae']:.3f}  "
        f"line: {result['line_margin_mae']:.3f}  delta: {result['margin_mae_delta']:+.3f}"
    )
    print(
        f"total  MAE -- model: {result['model_total_mae']:.3f}  "
        f"line: {result['line_total_mae']:.3f}  delta: {result['total_mae_delta']:+.3f}"
    )
    print(f"Brier -- raw: {result['brier_raw']:.4f}  calibrated: {result['brier_calibrated']:.4f}")
    print(f"mean CLV proxy (model - line, signed toward home): {result['mean_clv_proxy']:+.3f}")
    print(f"disagreement rate (model side != line side): {result['disagreement_rate']:.3f}")
    print("\nreliability table (calibrated win prob):")
    print(f"  {'bin':>12} {'n':>6} {'mean_pred':>10} {'mean_actual':>12}")
    for b in result["reliability"]:
        print(
            f"  [{b['bin_lo']:.1f},{b['bin_hi']:.1f}) {b['n']:>6} "
            f"{b['mean_pred']:>10.3f} {b['mean_actual']:>12.3f}"
        )

    verdict, reasons = go_no_go_verdict(result)
    print(f"\n=== GO/NO-GO: {verdict} ===")
    for reason in reasons:
        print(f"  - {reason}")


def _write_report(result: dict, sport: str, alpha: float, features: list[str]) -> None:
    verdict, reasons = go_no_go_verdict(result)

    lines = [
        "# +EV true-prob model -- walk-forward backtest / GO-NO-GO",
        "",
        f"**Sport:** {sport}  **alpha:** {alpha}",
        f"**Features:** {', '.join(features)}",
        f"**Games scored:** {result['n_games']}  **Seasons scored:** {result['seasons_scored']}",
        "",
        "Leak-free walk-forward: each season S is scored by a model fit only on "
        "seasons < S, with sigma from TRAIN residuals and Platt calibration fit "
        "only on the prior seasons' own (win_prob, home_win) pairs. See "
        "scripts/backtest_trueprob.py's module docstring for the closing-line "
        "sign convention (verified empirically for NFL's spread_line).",
        "",
        "## Margin / total accuracy vs the closing line",
        "",
        "| metric | model MAE | line MAE | delta (model - line) |",
        "|---|---|---|---|",
        f"| margin | {result['model_margin_mae']:.3f} | {result['line_margin_mae']:.3f} | {result['margin_mae_delta']:+.3f} |",
        f"| total | {result['model_total_mae']:.3f} | {result['line_total_mae']:.3f} | {result['total_mae_delta']:+.3f} |",
        "",
        "(delta < 0 means the model beats the closing line on this metric.)",
        "",
        "## Win-probability calibration",
        "",
        f"Brier (raw): {result['brier_raw']:.4f}",
        f"Brier (calibrated): {result['brier_calibrated']:.4f}",
        "",
        "| bin | n | mean_pred | mean_actual |",
        "|---|---|---|---|",
    ]
    for b in result["reliability"]:
        lines.append(
            f"| [{b['bin_lo']:.1f}, {b['bin_hi']:.1f}) | {b['n']} | "
            f"{b['mean_pred']:.3f} | {b['mean_actual']:.3f} |"
        )
    lines += [
        "",
        "## Edge proxies",
        "",
        f"- mean CLV proxy (signed model-minus-line margin edge, positive = model favors home more than the market): {result['mean_clv_proxy']:+.3f}",
        f"- disagreement rate (model's side != the line's side): {result['disagreement_rate']:.3f}",
        "",
        f"## Verdict: {verdict}",
        "",
    ]
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    if verdict == "GO":
        lines.append(
            "GO: the model's margin/total MAE beat (or are within noise of) the "
            "closing line, and the calibrated win probabilities track the "
            "diagonal closely enough to trust. Proceed to sub-project 4 "
            "(edge/EV engine)."
        )
    else:
        lines.append(
            "NO-GO / partial: at least one of the margin/total MAE deltas or the "
            "calibration check failed the threshold above. Per the plan, STOP "
            "before building the +EV UI -- iterate on features/regularization, "
            "or conclude the +EV view would surface only noise and pause."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nwritten {REPORT_PATH}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", choices=["nfl", "cfb"], required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args(argv)

    assets_dir = ASSETS / args.sport
    features_path = assets_dir / "features.parquet"
    if not features_path.exists():
        sys.exit(
            f"{features_path} not found. Run scripts/build_features.py "
            f"--sport {args.sport} first (needs the sport's schedule/EPA-PPA "
            f"assets already built)."
        )

    df = pd.read_parquet(features_path)

    if args.sport == "nfl":
        df = _join_nfl_lines(df, ASSETS / "nfl" / "schedules.parquet")
        features = list(trueprob.FEATURES)
    else:
        lines_path = ASSETS / "cfb" / "lines.parquet"
        if not lines_path.exists():
            sys.exit(f"{lines_path} not found. Run scripts/build_cfb_lines.py first.")
        df = _join_cfb_lines(df, lines_path)
        features = list(CFB_FEATURES)

    result = run_walk_forward(df, args.sport, features, alpha=args.alpha)
    _print_report(result, args.sport, args.alpha)
    _write_report(result, args.sport, args.alpha, features)


if __name__ == "__main__":
    main()
