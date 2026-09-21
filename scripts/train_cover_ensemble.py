"""Task 6 of docs/superpowers/specs/2026-09-21-nfl-cover-ensemble-design.md:
train the GBM + meta-learner (season walk-forward) for the NFL cover/total
stacked ensemble, and write the shipped ensemble artifacts -- gated on
out-of-sample performance.

CONTROLLER RULINGS THIS SCRIPT IMPLEMENTS (see
.superpowers/sdd/2026-09-21-nfl-cover-ensemble/progress.md, Task 6):

1. Meta shape: `sim_cover_p`/`sim_over_p` in `assets/nfl/cover_dataset.parquet`
   are only populated for the 2024 season (~9.5% of rows; Task 5's bounded
   backfill). The SHIPPED meta-learner is therefore a 2-way stack per
   market -- cover: [ratings_cover_p, gbm_cover_p]; total: [ratings_over_p,
   gbm_over_p] -- trained walk-forward on ALL seasons. A 3-way
   [ratings, sim, gbm] meta is ALSO fit and reported on the 2024 subset for
   reference only (`three_way_reference_2024`) -- it is never written to the
   shipped artifact. Sim enters serving via the agreement gate (a later
   task), not this meta, until sim history is backfilled deeper.

2. SHIP GATE: artifacts are written ONLY IF the 2-way ensemble beats BOTH
   base learners (ratings-only, gbm-only) AND the naive p=0.5, out-of-sample,
   on BOTH Brier and log-loss, for BOTH markets. If the gate fails, `main()`
   prints the full metrics table and exits non-zero WITHOUT writing any
   artifact -- the gate is never weakened to force a pass.

Pure/IO split (mirrors build_cover_dataset.py's own docstring convention):
`fit_gbm`, `residual_sigma`, `fit_meta`, `_oof_predict`, `_brier_logloss`,
`check_gate`, and `three_way_reference_2024` are pure (arrays/frames in,
values out) and unit-tested directly in
tests/scripts/test_train_cover_ensemble.py. `evaluate()` and `main()` do the
season walk-forward over the real dataset plus the artifact/asset IO
(parquet reads, joblib dumps, JSON writes) and are exercised by running this
script against the real `assets/nfl/cover_dataset.parquet`, not by a
synthetic unit test -- the walk-forward's whole point is real out-of-sample
performance.

Leakage: for each test season, the GBM and meta are fit ONLY on strictly
earlier seasons (`train_df = df[df.season < test_season]`); a test season's
rows are only ever scored by models that never saw that season (or any
later one). Within a training split, the GBM's own contribution to the meta's
training features uses out-of-fold predictions (`_oof_predict`, a K-fold CV
over the training split only) rather than the GBM's in-sample fit, so the
meta doesn't learn to over-trust an overfit GBM.

Usage:
    uv run python scripts/train_cover_ensemble.py
"""
from __future__ import annotations

import json
import pathlib
import sys
from math import exp, log

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

from sportsmodel.model import calibration
from sportsmodel.serving.ensemble import ensemble_prob, gbm_prob

_ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets" / "nfl"
_CALIBRATION_PATH = pathlib.Path(__file__).resolve().parents[1] / "assets" / "calibration.json"

_EPS = 1e-6

_GBM_KWARGS = dict(max_depth=3, min_samples_leaf=40, learning_rate=0.05, early_stopping=True)

_EFF_FEATURES = ["eff_diff", "home_off_adj", "home_def_adj", "away_off_adj", "away_def_adj", "total_off"]
MARGIN_FEATURES = _EFF_FEATURES + ["week", "home_field", "spread_line"]
TOTAL_FEATURES = _EFF_FEATURES + ["week", "home_field", "total_line"]


def _clip(p: float) -> float:
    return min(max(p, _EPS), 1.0 - _EPS)


def _logit(p: float) -> float:
    p = _clip(p)
    return log(p / (1.0 - p))


# --------------------------------------------------------------------------
# Pure-ish helpers (unit tested)
# --------------------------------------------------------------------------

def fit_gbm(X, y, random_state: int = 0) -> HistGradientBoostingRegressor:
    """Fit and return a HistGradientBoostingRegressor with the brief's fixed
    hyperparameters, predicting a point margin or total (NOT a probability).
    """
    model = HistGradientBoostingRegressor(random_state=random_state, **_GBM_KWARGS)
    model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float))
    return model


def _oof_predict(model, X, y, n_splits: int = 5, random_state: int = 0) -> np.ndarray:
    """Out-of-fold predictions from K-fold CV, re-fitting a clone of `model`
    (same hyperparameters) per fold so no fold's prediction ever sees its
    own row during fitting. Falls back to `model`'s own (in-sample)
    predictions when there are too few rows for K-fold (e.g. small unit-test
    frames) -- never raises on a tiny frame.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    splits = min(n_splits, n)
    if splits < 2:
        return np.asarray(model.predict(X), dtype=float)
    kf = KFold(n_splits=splits, shuffle=True, random_state=random_state)
    oof = np.zeros(n)
    for train_idx, test_idx in kf.split(X):
        m = clone(model)
        m.fit(X[train_idx], y[train_idx])
        oof[test_idx] = m.predict(X[test_idx])
    return oof


def residual_sigma(model, X, y) -> float:
    """Std of OUT-OF-FOLD residuals (K-fold CV re-fit of `model`'s
    hyperparameters), not the in-sample residual std of the already-fitted
    `model` -- in-sample residuals understate a GBM's true dispersion, which
    would make `gbm_prob` overconfident. Feeds `gbm_prob`'s `sigma`.
    """
    y = np.asarray(y, dtype=float)
    oof = _oof_predict(model, X, y)
    resid = y - oof
    sigma = float(np.std(resid))
    return sigma if sigma > 0.0 else 1e-3  # guard against a degenerate zero on tiny/constant frames


def fit_meta(base_probs_matrix, labels) -> tuple[list[float], float]:
    """LogisticRegression(C=1.0) on the logit of each base probability.
    Returns (coef, intercept) directly consumable by
    `sportsmodel.serving.ensemble.ensemble_prob`.
    """
    X = np.array([[_logit(p) for p in row] for row in base_probs_matrix])
    y = np.asarray(labels, dtype=float)
    clf = LogisticRegression(C=1.0)
    clf.fit(X, y)
    coef = [float(c) for c in clf.coef_[0]]
    intercept = float(clf.intercept_[0])
    return coef, intercept


def _brier_logloss(probs, ys) -> dict:
    probs = np.clip(np.asarray(probs, dtype=float), _EPS, 1.0 - _EPS)
    ys = np.asarray(ys, dtype=float)
    brier = float(np.mean((probs - ys) ** 2))
    logloss = float(-np.mean(ys * np.log(probs) + (1.0 - ys) * np.log(1.0 - probs)))
    return {"brier": brier, "logloss": logloss}


def calibration_curve(probs, ys, n_bins: int = 10) -> dict:
    """Reliability diagnostic (fix for the brief's "Brier/logloss/calibration"
    interface -- `evaluate()` previously only reported Brier/logloss).

    Bins predicted probabilities into `n_bins` equal-width bins over [0, 1]
    and reports, per non-empty bin, the mean predicted probability, the
    empirical event rate, and the bin count -- plus the overall expected
    calibration error (ECE): the count-weighted mean absolute gap between
    predicted and empirical rate across bins. A near-diagonal reliability
    curve (mean_pred ~= empirical_rate in every bin) and a small ECE mean
    the ensemble's stated probabilities are trustworthy even when its
    Brier/log-loss skill is modest. Empty bins are omitted from `bins` and
    contribute nothing to `ece`. Pure -- no IO.
    """
    probs = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    ys = np.asarray(ys, dtype=float)
    n = len(probs)
    if n == 0:
        return {"bins": [], "ece": 0.0}
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, n_bins - 1)
    bins = []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(probs[mask].mean())
        empirical = float(ys[mask].mean())
        bins.append({"bin": b, "mean_pred": mean_pred, "empirical_rate": empirical, "count": count})
        ece += (count / n) * abs(mean_pred - empirical)
    return {"bins": bins, "ece": float(ece)}


def check_gate(metrics: dict) -> tuple[bool, list[str]]:
    """SHIP GATE: ensemble must beat p50, ratings, and gbm on both Brier and
    log-loss, for both markets. Returns (passed, [reasons for each failure]).
    """
    reasons = []
    for market in ("cover", "total"):
        m = metrics[market]
        for baseline in ("p50", "ratings", "gbm"):
            for stat in ("brier", "logloss"):
                ens_v, base_v = m["ensemble"][stat], m[baseline][stat]
                if not (ens_v < base_v):
                    reasons.append(
                        f"{market}: ensemble {stat}={ens_v:.5f} does not beat {baseline} {stat}={base_v:.5f}"
                    )
    return len(reasons) == 0, reasons


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def _load_training_frame() -> pd.DataFrame:
    """cover_dataset.parquet (labels + features + ratings/sim base probs)
    joined with schedules.parquet's actual margin/total (GBM regression
    targets -- the cover dataset only carries the binary cover/over labels,
    not the raw score margin the GBM predicts).
    """
    cover = pd.read_parquet(_ASSETS / "cover_dataset.parquet")
    sched = pd.read_parquet(_ASSETS / "schedules.parquet")
    if "game_type" in sched.columns:
        sched = sched[sched["game_type"] == "REG"]
    actuals = sched[["season", "week", "home_team", "away_team", "result", "total"]].rename(
        columns={"result": "margin_actual", "total": "total_actual"}
    )
    merged = cover.merge(
        actuals, on=["season", "week", "home_team", "away_team"], how="left", validate="one_to_one"
    )
    missing = int(merged["margin_actual"].isna().sum())
    if missing:
        raise ValueError(
            f"{missing}/{len(merged)} cover_dataset rows failed to join actual margin/total "
            "from schedules.parquet"
        )
    return merged.reset_index(drop=True)


# --------------------------------------------------------------------------
# Walk-forward evaluation
# --------------------------------------------------------------------------

def _walk_forward_market(df: pd.DataFrame, features: list[str], line_col: str,
                          target_col: str, label_col: str, ratings_col: str,
                          dist_kind: str, flip_line: bool) -> pd.DataFrame:
    """Season walk-forward for one market. Returns a per-row DataFrame of
    OOF (season, idx, ratings_p, gbm_p, ensemble_p, y) covering every test
    season (every season except the first, which has no prior seasons to
    train on).
    """
    seasons = sorted(df["season"].unique())
    records = []
    for test_season in seasons[1:]:
        train_df = df[df["season"] < test_season]
        test_df = df[df["season"] == test_season]
        if train_df.empty or test_df.empty:
            continue

        X_train = train_df[features].to_numpy(dtype=float)
        y_train = train_df[target_col].to_numpy(dtype=float)
        X_test = test_df[features].to_numpy(dtype=float)

        gbm = fit_gbm(X_train, y_train)
        sigma = residual_sigma(gbm, X_train, y_train)
        oof_train_pred = _oof_predict(gbm, X_train, y_train)

        def _to_prob(pred, line):
            line = -line if flip_line else line
            return gbm_prob(float(pred), sigma, dist_kind, float(line))

        train_gbm_p = [_to_prob(p, ln) for p, ln in zip(oof_train_pred, train_df[line_col])]
        test_pred = gbm.predict(X_test)
        test_gbm_p = [_to_prob(p, ln) for p, ln in zip(test_pred, test_df[line_col])]

        train_base = list(zip(train_df[ratings_col].tolist(), train_gbm_p))
        coef, intercept = fit_meta(train_base, train_df[label_col].tolist())

        test_ensemble_p = [
            ensemble_prob([r, g], coef, intercept)
            for r, g in zip(test_df[ratings_col], test_gbm_p)
        ]

        records.append(pd.DataFrame({
            "season": test_season,
            "idx": test_df.index,
            "ratings_p": test_df[ratings_col].to_numpy(dtype=float),
            "gbm_p": test_gbm_p,
            "ensemble_p": test_ensemble_p,
            "y": test_df[label_col].to_numpy(dtype=float),
        }))

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(
        columns=["season", "idx", "ratings_p", "gbm_p", "ensemble_p", "y"]
    )


def evaluate(df: pd.DataFrame) -> dict:
    """Season walk-forward for both markets. Returns
    {"metrics": {market: {p50, ratings, gbm, ensemble: {brier, logloss}, n}},
     "oof": {market: DataFrame}}.
    """
    oof_cover = _walk_forward_market(
        df, MARGIN_FEATURES, "spread_line", "margin_actual", "home_cover",
        "ratings_cover_p", "margin", flip_line=True,
    )
    oof_total = _walk_forward_market(
        df, TOTAL_FEATURES, "total_line", "total_actual", "over",
        "ratings_over_p", "total", flip_line=False,
    )

    metrics = {}
    for market, oof in (("cover", oof_cover), ("total", oof_total)):
        y = oof["y"].to_numpy()
        metrics[market] = {
            "p50": _brier_logloss(np.full(len(y), 0.5), y),
            "ratings": _brier_logloss(oof["ratings_p"], y),
            "gbm": _brier_logloss(oof["gbm_p"], y),
            "ensemble": _brier_logloss(oof["ensemble_p"], y),
            "ensemble_calibration": calibration_curve(oof["ensemble_p"], y),
            "n": int(len(y)),
        }

    return {"metrics": metrics, "oof": {"cover": oof_cover, "total": oof_total}}


def three_way_reference_2024(df: pd.DataFrame, oof: dict) -> dict:
    """Reference-only 3-way [ratings, sim, gbm] meta on the 2024 subset (the
    only season with sim coverage -- Task 5's bounded backfill). NOT shipped
    (see module docstring, Ruling 1). Reuses the walk-forward's own
    leakage-free OOF gbm probs for 2024 (that GBM was fit only on
    pre-2024 seasons) and reports a 5-fold in-2024 CV read on top of a
    reference in-sample fit, since a true walk-forward isn't possible with
    sim history covering only this one season.
    """
    out = {}
    for market, ratings_col, sim_col, label_col in (
        ("cover", "ratings_cover_p", "sim_cover_p", "home_cover"),
        ("total", "ratings_over_p", "sim_over_p", "over"),
    ):
        oof_m = oof[market]
        sub = oof_m[oof_m["season"] == 2024].copy()
        sub["sim_p"] = df.loc[sub["idx"], sim_col].to_numpy()
        sub = sub.dropna(subset=["sim_p"])
        if sub.empty:
            out[market] = {"n": 0}
            continue

        X3 = sub[["ratings_p", "sim_p", "gbm_p"]].to_numpy(dtype=float)
        y = sub["y"].to_numpy(dtype=float)

        n = len(y)
        splits = min(5, n)
        cv_pred = np.zeros(n)
        if splits >= 2:
            kf = KFold(n_splits=splits, shuffle=True, random_state=0)
            for train_idx, test_idx in kf.split(X3):
                coef, intercept = fit_meta(X3[train_idx], y[train_idx])
                cv_pred[test_idx] = [ensemble_prob(row, coef, intercept) for row in X3[test_idx]]
        else:
            cv_pred[:] = 0.5

        coef_all, intercept_all = fit_meta(X3, y)
        in_sample_pred = [ensemble_prob(row, coef_all, intercept_all) for row in X3]

        out[market] = {
            "n": int(n),
            "cv_metrics": _brier_logloss(cv_pred, y) if splits >= 2 else None,
            "in_sample_metrics": _brier_logloss(in_sample_pred, y),
            "coef": coef_all,
            "intercept": intercept_all,
        }
    return out


def _fit_final(df: pd.DataFrame, features: list[str], line_col: str, target_col: str,
               label_col: str, ratings_col: str, dist_kind: str, flip_line: bool):
    """Fit the FINAL (all-seasons) GBM + 2-way meta for one market, used for
    the shipped artifact. The meta's GBM-side training feature uses
    out-of-fold GBM predictions over the full dataset (same rationale as the
    walk-forward loop: don't let the meta learn to over-trust an in-sample
    fit).
    """
    X = df[features].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    gbm = fit_gbm(X, y)
    sigma = residual_sigma(gbm, X, y)
    oof_pred = _oof_predict(gbm, X, y)

    def _to_prob(pred, line):
        line = -line if flip_line else line
        return gbm_prob(float(pred), sigma, dist_kind, float(line))

    gbm_p = [_to_prob(p, ln) for p, ln in zip(oof_pred, df[line_col])]
    base = list(zip(df[ratings_col].tolist(), gbm_p))
    coef, intercept = fit_meta(base, df[label_col].tolist())
    return gbm, sigma, coef, intercept


def _print_metrics_table(metrics: dict) -> None:
    print(f"{'market':<8} {'baseline':<10} {'brier':>10} {'logloss':>10} {'n':>6}")
    for market in ("cover", "total"):
        for baseline in ("p50", "ratings", "gbm", "ensemble"):
            m = metrics[market][baseline]
            print(f"{market:<8} {baseline:<10} {m['brier']:>10.5f} {m['logloss']:>10.5f} {metrics[market]['n']:>6}")


def _print_calibration(metrics: dict) -> None:
    for market in ("cover", "total"):
        cal = metrics[market]["ensemble_calibration"]
        print(f"\n{market} ensemble reliability (ECE={cal['ece']:.5f}):")
        print(f"  {'bin':>3} {'mean_pred':>10} {'empirical':>10} {'count':>6}")
        for b in cal["bins"]:
            print(f"  {b['bin']:>3} {b['mean_pred']:>10.4f} {b['empirical_rate']:>10.4f} {b['count']:>6}")


def main() -> int:
    df = _load_training_frame()
    print(f"loaded {len(df)} rows, seasons {sorted(df['season'].unique().tolist())}")

    result = evaluate(df)
    metrics = result["metrics"]
    print("\n=== walk-forward out-of-sample metrics ===")
    _print_metrics_table(metrics)
    print("\n=== ensemble calibration (reliability curve, ~10 bins) ===")
    _print_calibration(metrics)

    ref = three_way_reference_2024(df, result["oof"])
    print("\n=== 3-way [ratings, sim, gbm] reference on 2024 subset (NOT shipped) ===")
    for market, r in ref.items():
        print(f"{market}: n={r.get('n')} cv={r.get('cv_metrics')} in_sample={r.get('in_sample_metrics')} "
              f"coef={r.get('coef')} intercept={r.get('intercept')}")

    passed, reasons = check_gate(metrics)
    print(f"\n=== SHIP GATE: {'PASS' if passed else 'FAIL'} ===")
    for r in reasons:
        print(f"  - {r}")

    if not passed:
        print("\nGate failed -- NOT writing cover_ensemble.json / GBM artifacts / calibration update.")
        return 1

    # Final artifacts: fit on ALL data.
    gbm_margin, sigma_margin, coef_cover, intercept_cover = _fit_final(
        df, MARGIN_FEATURES, "spread_line", "margin_actual", "home_cover",
        "ratings_cover_p", "margin", flip_line=True,
    )
    gbm_total, sigma_total, coef_total, intercept_total = _fit_final(
        df, TOTAL_FEATURES, "total_line", "total_actual", "over",
        "ratings_over_p", "total", flip_line=False,
    )

    margin_path = _ASSETS / "cover_gbm_margin.joblib"
    total_path = _ASSETS / "cover_gbm_total.joblib"
    joblib.dump(gbm_margin, margin_path)
    joblib.dump(gbm_total, total_path)

    ensemble_json = {
        "cover": {
            "coef": coef_cover,
            "intercept": intercept_cover,
            "sigma": sigma_margin,
            "features": MARGIN_FEATURES,
            "gbm_model_path": "assets/nfl/cover_gbm_margin.joblib",
            "dist_kind": "margin",
        },
        "total": {
            "coef": coef_total,
            "intercept": intercept_total,
            "sigma": sigma_total,
            "features": TOTAL_FEATURES,
            "gbm_model_path": "assets/nfl/cover_gbm_total.joblib",
            "dist_kind": "total",
        },
    }
    (_ASSETS / "cover_ensemble.json").write_text(json.dumps(ensemble_json, indent=2))
    print(f"\nwrote {_ASSETS / 'cover_ensemble.json'}")

    # Extend calibration.json with cover/total targets, fit on the walk-forward
    # OOF ensemble predictions (genuinely out-of-sample, not the final in-sample fit).
    calib = json.loads(_CALIBRATION_PATH.read_text()) if _CALIBRATION_PATH.exists() else {}
    for market in ("cover", "total"):
        oof = result["oof"][market]
        a, b = calibration.fit(oof["ensemble_p"].tolist(), oof["y"].tolist())
        calib[market] = [a, b]
    _CALIBRATION_PATH.write_text(json.dumps(calib, indent=2))
    print(f"wrote {_CALIBRATION_PATH} (added/updated cover, total)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
