"""Fit the efficiency-rating hyperparameters (ridge, prior half-life/floor, retention) on
2016-2022 and write assets/cfb/eff_config.json. Report-only on 2023-2025 (nothing is selected
there). Pure local compute from the committed assets -- no network, no key.

Usage:
    uv run python scripts/fit_cfb_eff.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import eff_fit, v3_data  # noqa: E402
from sportsmodel.cfb.efficiency import EffConfig  # noqa: E402

OUT = ROOT / "assets" / "cfb" / "eff_config.json"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="fit and print; do not write eff_config.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    eff_games = v3_data.load_eff_games()
    talent = v3_data.read_asset("talent.parquet")
    priors = v3_data.load_priors_rows()
    default = EffConfig()
    cfg, train_loss = eff_fit.fit_eff_config(eff_games, priors, talent)
    print(f"fitted {cfg}\n  train {list(eff_fit.TRAIN_SEASONS)[0]}-{list(eff_fit.TRAIN_SEASONS)[-1]} "
          f"one-week-ahead PPA MSE: default={eff_fit.ppa_holdout_loss(eff_games, eff_fit.TRAIN_SEASONS, default, priors, talent):.5f}"
          f" fitted={train_loss:.5f}")
    print(f"  holdout {eff_fit.HOLDOUT_SEASONS} (report only): "
          f"default={eff_fit.ppa_holdout_loss(eff_games, eff_fit.HOLDOUT_SEASONS, default, priors, talent):.5f} "
          f"fitted={eff_fit.ppa_holdout_loss(eff_games, eff_fit.HOLDOUT_SEASONS, cfg, priors, talent):.5f}")
    edge = [p for p in eff_fit.ORDER if getattr(cfg, p) in (eff_fit.GRID[p][0], eff_fit.GRID[p][-1])]
    if edge:
        print(f"  NOTE: fitted value sits on the grid edge for {edge}")
    if args.dry_run:
        print("--dry-run: nothing written")
    else:
        OUT.write_text(json.dumps({k: float(v) for k, v in asdict(cfg).items()}, indent=2) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
