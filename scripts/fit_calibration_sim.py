"""Fit probability calibration for the HYBRID prop model and write calibration.json.

The hybrid uses the Monte Carlo SIM for some markets and the closed-form
ANALYTIC model for others, so each market's Platt calibration must be fit
from whichever model actually produces it in production:

  - win_prob, hits, total_bases, home_run, outs_recorded: SIM-served in the
    hybrid -> fit here from scripts/backtest_sim.py + backtest_sim_props.py.
  - pitcher_ks, hits_allowed: ANALYTIC-served in the hybrid -> left untouched,
    copied through from the existing assets/calibration.json (which was fit
    by scripts/fit_calibration.py against the analytic backtests). Re-fitting
    them from sim data here would calibrate them against a model that never
    actually scores those two markets in production.
  - hrr: not fit anywhere -- no per-batter runs/RBI actual is recoverable
    from Statcast at the PA level (see backtest_sim_props.py's module
    docstring), so the hybrid applies identity calibration for it. Any
    existing hrr entry passes through unchanged.

Models fit_calibration.py's shape (same Platt fit via calibration.fit,
same before/after report) but swaps the analytic backtests for the sim
backtests on the sim-served targets, and merges in the kept analytic params
for the rest.

Usage:
    uv run python scripts/fit_calibration_sim.py --season 2025 --n-sims 1000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_sim
import backtest_sim_props
from sportsmodel.model import calibration

# Markets the hybrid serves from the Monte Carlo sim -- fit fresh from the sim backtests.
SIM_FIT_TARGETS = ("win_prob", "hits", "total_bases", "home_run", "outs_recorded")
# Markets the hybrid serves from the analytic model -- keep whatever's already in
# calibration.json (fit by scripts/fit_calibration.py against the analytic backtests).
ANALYTIC_KEEP_TARGETS = ("pitcher_ks", "hits_allowed")

CALIB_PATH = ROOT / "assets" / "calibration.json"


def merge_calibration(sim_params: dict, existing: dict) -> dict:
    """Merge freshly sim-fit params over an existing calibration dict.

    `sim_params` (typically keyed by SIM_FIT_TARGETS) overrides the matching
    entries in `existing`; everything else already in `existing` -- notably
    the ANALYTIC_KEEP_TARGETS (pitcher_ks/hits_allowed) and any hrr entry --
    passes through completely unchanged. This is the whole hybrid split: it's
    a plain "new values win, old values otherwise survive" merge, but which
    keys `sim_params` is allowed to carry is what encodes "sim-fit these,
    analytic-keep those."
    """
    merged = dict(existing)
    merged.update(sim_params)
    return merged


def fit_dist_affine(sim_pooled, emp_mean: float, emp_sd: float):
    """Method-of-moments location+scale that maps the sim's pooled marginal
    distribution onto the empirical moments: loc re-centers the mean, scale matches
    the SD. Returns (loc, scale). Used to calibrate the total and margin dists."""
    import numpy as np
    sim_pooled = np.asarray(sim_pooled, dtype=float)
    sim_mean = float(sim_pooled.mean())
    sim_sd = float(sim_pooled.std())
    loc = emp_mean - sim_mean
    scale = (emp_sd / sim_sd) if sim_sd > 0 else 1.0
    return loc, scale


def _fmt(params) -> str:
    if params is None:
        return "—"
    a, b = params
    return f"a={a:.3f}, b={b:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--n-sims", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import numpy as np
    import validate_sim_dist as vsd

    existing = json.loads(CALIB_PATH.read_text()) if CALIB_PATH.exists() else {}

    # Capture the pooled per-sim scores during the game backtest so we can fit the
    # total/margin distribution calibration (mean + width) from the same run.
    _homes, _aways = [], []
    _orig_ps = backtest_sim.pred_scores

    def _cap(sims):
        _homes.append(np.asarray(sims.home_score))
        _aways.append(np.asarray(sims.away_score))
        return _orig_ps(sims)

    backtest_sim.pred_scores = _cap
    try:
        game = backtest_sim.run_sim_backtest(args.season, args.n_sims, args.seed)
    finally:
        backtest_sim.pred_scores = _orig_ps
    props = backtest_sim_props.run_sim_props_backtest(args.season, args.n_sims, args.seed)

    # Totals/margin distribution calibration: moment-match the pooled sim marginal
    # distribution to the empirical moments. loc re-centers the mean the scoring
    # channels didn't fully close; scale finishes the width dispersion didn't reach.
    pooled_home = np.concatenate(_homes)
    pooled_away = np.concatenate(_aways)
    loc_t, scale_t = fit_dist_affine(pooled_home + pooled_away,
                                     vsd.EMPIRICAL["mean_total"], vsd.EMPIRICAL["sd_total"])
    loc_m, scale_m = fit_dist_affine(pooled_home - pooled_away,
                                     vsd.EMPIRICAL["mean_margin"], vsd.EMPIRICAL["sd_margin"])

    targets = {"win_prob": ([s[0] for s in game], [s[1] for s in game])}
    for market in SIM_FIT_TARGETS:
        if market == "win_prob":
            continue
        sc = props[market]
        targets[market] = (sc.p, sc.y)

    sim_params = {t: list(calibration.fit(p, y)) for t, (p, y) in targets.items()}
    merged = merge_calibration(sim_params, existing)
    merged["total_dist"] = {"loc": loc_t, "scale": scale_t}
    merged["margin_dist"] = {"loc": loc_m, "scale": scale_m}
    CALIB_PATH.write_text(json.dumps(merged, indent=2))
    print(f"total_dist calibration: loc={loc_t:+.3f} scale={scale_t:.3f}  "
          f"margin_dist: loc={loc_m:+.3f} scale={scale_m:.3f}")

    print(f"\nHYBRID calibration fit on {args.season} (sim n_sims={args.n_sims}). "
          f"Wrote {CALIB_PATH.relative_to(ROOT)}")
    print(f"\n{'target':14} {'source':10} {'raw pred':>9} {'actual':>8} {'calibrated':>11}   before -> after")
    for t in SIM_FIT_TARGETS:
        p, y = targets[t]
        n = len(p)
        raw = sum(p) / n
        act = sum(y) / n
        cal = sum(calibration.apply(x, merged[t]) for x in p) / n
        print(f"{t:14} {'sim-fit':10} {raw:>9.3f} {act:>8.3f} {cal:>11.3f}   "
              f"{_fmt(existing.get(t))} -> {_fmt(merged.get(t))}")
    for t in ANALYTIC_KEEP_TARGETS:
        print(f"{t:14} {'kept':10} {'':>9} {'':>8} {'':>11}   "
              f"{_fmt(existing.get(t))} -> {_fmt(merged.get(t))}")
    if "hrr" in existing or "hrr" in merged:
        print(f"{'hrr':14} {'passthru':10} {'':>9} {'':>8} {'':>11}   "
              f"{_fmt(existing.get('hrr'))} -> {_fmt(merged.get('hrr'))}")
    else:
        print(f"{'hrr':14} {'omitted':10}  (no per-batter actuals -- hybrid applies identity)")


if __name__ == "__main__":
    main()
