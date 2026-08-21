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
             "p_shutout": 0.138, "p_blowout": 0.287, "sd_margin": 4.58}


def tail_metrics(home, away) -> dict:
    home = np.asarray(home)
    away = np.asarray(away)
    total = home + away
    margin = home - away
    return {
        "mean_total": float(total.mean()),
        "sd_total": float(total.std()),
        "p_ge11": float(np.mean(total >= 11)),
        "p_le5": float(np.mean(total <= 5)),
        "p_shutout": float(np.mean((home == 0) | (away == 0))),
        "p_blowout": float(np.mean(np.abs(margin) >= 5)),
        "sd_margin": float(margin.std()),
    }


def run_pooled(season: int, month: int, n_sims: int, seed: int = 42):
    """Run the sim backtest for one month; return pooled (home, away) sim arrays."""
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
        bs.run_sim_backtest(season, n_sims=n_sims, seed=seed)
    finally:
        bs.pred_scores = orig
    return np.concatenate(homes), np.concatenate(aways)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--month", type=int, default=6)
    ap.add_argument("--n-sims", type=int, default=2000)
    a = ap.parse_args()
    home, away = run_pooled(a.season, a.month, a.n_sims)
    m = tail_metrics(home, away)
    print(f"\n{'metric':12} {'sim':>8} {'empirical':>10}  {'delta':>8}")
    for k in EMPIRICAL:
        print(f"{k:12} {m[k]:>8.3f} {EMPIRICAL[k]:>10.3f}  {m[k]-EMPIRICAL[k]:>+8.3f}")


if __name__ == "__main__":
    main()
