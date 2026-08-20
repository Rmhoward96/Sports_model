"""Out-of-sample validation of the calibration fit on a different season.

Runs the backtests on --season (default 2025), applies the STORED calibration
(fit on 2024, assets/calibration.json) to those raw predictions, and reports raw
vs. calibrated mean P(over), Brier, and log-loss. If calibration helps out-of-sample,
calibrated mean should sit closer to the actual over-rate and Brier/log-loss shouldn't
worsen.

Usage:
    uv run python scripts/validate_calibration.py --season 2025
"""
from __future__ import annotations

import argparse
import sys
from math import log
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_game
import backtest_props
from sportsmodel.model import calibration


def _metrics(probs, ys):
    n = len(probs)
    eps = 1e-9
    pred = sum(probs) / n
    actual = sum(ys) / n
    brier = sum((p - y) ** 2 for p, y in zip(probs, ys)) / n
    ll = -sum(y * log(max(p, eps)) + (1 - y) * log(max(1 - p, eps))
              for p, y in zip(probs, ys)) / n
    return pred, actual, brier, ll


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    args = ap.parse_args()

    game = backtest_game.run_backtest(args.season)
    props = backtest_props.run_backtest(args.season)
    targets = {"win_prob": ([s[0] for s in game], [s[1] for s in game])}
    for m, sc in props.items():
        targets[m] = (sc.p, sc.y)

    params = calibration.load()
    print(f"\nOUT-OF-SAMPLE calibration check — {args.season} "
          f"(calibration fit on 2024)")
    print(f"{'target':14} {'actual':>7} | {'raw':>6} {'cal':>6}  pred | "
          f"{'raw':>6} {'cal':>6}  Brier | {'raw':>6} {'cal':>6}  logloss")
    for t, (p, y) in targets.items():
        r_pred, actual, r_brier, r_ll = _metrics(p, y)
        cp = [calibration.apply(x, params.get(t)) for x in p]
        c_pred, _, c_brier, c_ll = _metrics(cp, y)
        flag = "  <-- worse" if c_ll > r_ll + 1e-4 else ""
        print(f"{t:14} {actual:>7.3f} | {r_pred:>6.3f} {c_pred:>6.3f}       | "
              f"{r_brier:>6.4f} {c_brier:>6.4f}       | {r_ll:>6.4f} {c_ll:>6.4f}{flag}")


if __name__ == "__main__":
    main()
