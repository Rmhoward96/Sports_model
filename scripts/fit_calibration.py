"""Fit probability calibration from a walk-forward backtest and write calibration.json.

Runs the game + props backtests on the fit season, fits a Platt map per target
(win_prob + each prop market), writes assets/calibration.json, and prints a
before/after so you can see the systematic biases getting corrected.

Usage:
    uv run python scripts/fit_calibration.py --season 2024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backtest_game
import backtest_props
from sportsmodel.model import calibration


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2024)
    args = ap.parse_args()

    game = backtest_game.run_backtest(args.season)
    props = backtest_props.run_backtest(args.season)

    targets = {"win_prob": ([s[0] for s in game], [s[1] for s in game])}
    for market, sc in props.items():
        targets[market] = (sc.p, sc.y)

    params = {t: list(calibration.fit(p, y)) for t, (p, y) in targets.items()}
    out = ROOT / "assets" / "calibration.json"
    out.write_text(json.dumps(params, indent=2))

    print(f"\nFit on {args.season}. Wrote {out.relative_to(ROOT)}")
    print(f"{'target':14} {'raw pred':>9} {'actual':>8} {'calibrated':>11}")
    for t, (p, y) in targets.items():
        n = len(p)
        raw = sum(p) / n
        act = sum(y) / n
        cal = sum(calibration.apply(x, params[t]) for x in p) / n
        print(f"{t:14} {raw:>9.3f} {act:>8.3f} {cal:>11.3f}  (a={params[t][0]:.2f}, b={params[t][1]:.2f})")


if __name__ == "__main__":
    main()
