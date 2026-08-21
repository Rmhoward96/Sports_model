"""Compare the sim's marginal total/margin distribution to real MLB.

Runs the sim backtest for a month/season, pools all per-sim scores, and prints
the six acceptance-bar tail metrics side by side with the 2025 empirical targets.
Replaces the throwaway scratchpad measurements with a committed harness.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Empirical 2025 (2367 games), from the spec's Problem table.
EMPIRICAL = {"mean_total": 8.89, "sd_total": 4.59, "p_ge11": 0.329, "p_le5": 0.253,
             "p_shutout": 0.138, "p_blowout": 0.287, "sd_margin": 4.58, "mean_margin": 0.07}


def _metrics(total, margin, shutout) -> dict:
    return {
        "mean_total": float(np.mean(total)),
        "sd_total": float(np.std(total)),
        "p_ge11": float(np.mean(total >= 11)),
        "p_le5": float(np.mean(total <= 5)),
        "p_shutout": float(np.mean(shutout)),
        "p_blowout": float(np.mean(np.abs(margin) >= 5)),
        "sd_margin": float(np.std(margin)),
        "mean_margin": float(np.mean(margin)),
    }


def tail_metrics(home, away) -> dict:
    home = np.asarray(home)
    away = np.asarray(away)
    return _metrics(home + away, home - away, (home == 0) | (away == 0))


def run_pooled(season: int, month: int, n_sims: int, seed: int = 42, dispersion=...):
    """Run the sim backtest for one month; return pooled (home, away) sim arrays.
    `dispersion` sentinel `...` uses the production config; pass a Dispersion to
    override (the tuning search does this)."""
    import backtest_sim as bs
    homes, aways = [], []
    orig = bs.pred_scores

    def wrap(sims):
        homes.append(np.asarray(sims.home_score))
        aways.append(np.asarray(sims.away_score))
        return orig(sims)

    bs.pred_scores = wrap
    bs._MONTHS = [month]
    try:
        bs.run_sim_backtest(season, n_sims=n_sims, seed=seed, dispersion=dispersion)
    finally:
        bs.pred_scores = orig
    return np.concatenate(homes), np.concatenate(aways)


def _load_dist_cal():
    import json
    from sportsmodel import config
    p = config.PROJECT_ROOT / "assets" / "calibration.json"
    c = json.loads(p.read_text()) if p.exists() else {}
    t, m = c.get("total_dist", {}), c.get("margin_dist", {})
    return (t.get("loc", 0.0), t.get("scale", 1.0)), (m.get("loc", 0.0), m.get("scale", 1.0))


def calibrated_metrics(home, away) -> dict:
    """Apply the stored total/margin loc+scale to pooled sim scores and measure. Total
    and margin metrics come from the calibrated arrays directly; shutout is approximated
    from reconstructed (rounded) home/away, which the affine doesn't preserve exactly."""
    (lt, st), (lm, sm) = _load_dist_cal()
    total, margin = home + away, home - away
    ct = st * (total - total.mean()) + total.mean() + lt
    cm = sm * (margin - margin.mean()) + margin.mean() + lm
    ch = np.rint((ct + cm) / 2).clip(0)
    ca = np.rint((ct - cm) / 2).clip(0)
    return _metrics(ct, cm, (ch == 0) | (ca == 0))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--month", type=int, default=6)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--calibrated", action="store_true",
                    help="apply the fitted total/margin calibration before measuring")
    a = ap.parse_args()
    home, away = run_pooled(a.season, a.month, a.n_sims)
    m = calibrated_metrics(home, away) if a.calibrated else tail_metrics(home, away)
    print(f"\n{'metric':12} {'sim':>8} {'empirical':>10}  {'delta':>8}")
    for k in m:
        emp = EMPIRICAL.get(k)
        if emp is None:
            continue
        print(f"{k:12} {m[k]:>8.3f} {emp:>10.3f}  {m[k]-emp:>+8.3f}")


if __name__ == "__main__":
    main()
