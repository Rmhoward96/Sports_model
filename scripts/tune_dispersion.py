"""Tune the dispersion sigmas to match real MLB's outlier rates.

Coordinate search over (sigma_shared, sigma_team, sigma_pitcher), scoring each
candidate by how closely the sim's pooled tail metrics match empirical. p_roe/p_wp
are fixed (measured/literature); only the dispersion spread is tuned here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# metrics that describe distribution WIDTH/tails (not the mean -- calibration owns the mean)
_KEYS = ("sd_total", "p_ge11", "p_le5", "p_blowout", "sd_margin")


def objective(metrics: dict, empirical: dict) -> float:
    """Sum of squared relative errors on the width/tail metrics (lower is better)."""
    return sum(((metrics[k] - empirical[k]) / empirical[k]) ** 2 for k in _KEYS)


def coord_search(eval_fn, grid: dict) -> dict:
    """Coordinate descent over a {param: [values]} grid. Two sweeps; returns the best
    params dict. eval_fn(params) -> score to minimize."""
    best = {p: vals[0] for p, vals in grid.items()}
    best_score = eval_fn(best)
    for _sweep in range(2):
        improved = False
        for p, vals in grid.items():
            for v in vals:
                if v == best[p]:
                    continue
                cand = dict(best); cand[p] = v
                s = eval_fn(cand)
                if s < best_score:
                    best, best_score = cand, s
                    improved = True
        if not improved:
            break
    return best


def main():
    import argparse
    import validate_sim_dist as vsd
    from sportsmodel.sim.mlb.kernel import Dispersion
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--month", type=int, default=6)
    ap.add_argument("--n-sims", type=int, default=1000)
    a = ap.parse_args()

    grid = {
        "sigma_shared": [0.0, 0.08, 0.12, 0.16],
        "sigma_team": [0.0, 0.08, 0.12, 0.16],
        "sigma_pitcher": [0.0, 0.10, 0.18, 0.26],
    }

    def eval_fn(p):
        d = Dispersion(p["sigma_shared"], p["sigma_team"], p["sigma_pitcher"])
        home, away = vsd.run_pooled(a.season, a.month, a.n_sims, dispersion=d)
        m = vsd.tail_metrics(home, away)
        score = objective(m, vsd.EMPIRICAL)
        print(f"  sigmas={p}  sd_total={m['sd_total']:.2f} p_ge11={m['p_ge11']:.3f} "
              f"p_blowout={m['p_blowout']:.3f} score={score:.4f}")
        return score

    best = coord_search(eval_fn, grid)
    print(f"\nBEST: {best}")
    print("Write these into sim/mlb/config_dispersion.DISPERSION, then re-run "
          "validate_sim_dist.py at higher n to confirm.")


if __name__ == "__main__":
    main()
