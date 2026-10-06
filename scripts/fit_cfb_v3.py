"""Fit cfb-ratings-v3 on 2016-2022 and write assets/cfb/v3_weights.json + gameline_v3.json.

Builds the leak-free v3 feature table over 2015-2025 (one walk-forward), fits the efficiency
points map, the margin/total blends (context terms by lasso, any of them can land on exactly 0)
and the moneyline sigmas on the TRAINING seasons only, and prints the fit. Nothing is judged
here -- the held-out comparison against v2 is scripts/gate_cfb_v3.py. Local compute only.

Usage:
    uv run python scripts/fit_cfb_v3.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import v3, v3_data, v3_fit, v3_table  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
WALK_SEASONS = tuple(range(2015, 2026))


def write_outputs(w: v3.V3Weights, assets_dir: Path) -> None:
    (assets_dir / "v3_weights.json").write_text(w.to_json())
    gl = v3.gameline_v3_dict(w.meta["sigma_margin"], w.meta["sigma_total"])
    (assets_dir / "gameline_v3.json").write_text(json.dumps(gl, indent=2) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="fit and print; write nothing")
    args = ap.parse_args(argv)
    t0 = time.time()
    inp = v3_data.load_v3_inputs(ASSETS)
    table = v3_table.build_table(v3_data.load_merged_schedule(ASSETS), inp, seasons=WALK_SEASONS)
    print(f"feature table: {len(table)} FBS-vs-FBS games, {time.time() - t0:.0f}s", flush=True)
    w = v3_fit.fit_v3(table)
    print(f"points map: {json.dumps(w.points_map.to_dict())}")
    print(f"margin blend: {json.dumps(w.margin.to_dict())}")
    print(f"total blend:  {json.dumps(w.total.to_dict())}")
    zeros = sorted(k for b in (w.margin, w.total) for k, c in b.coefs.items() if c == 0.0)
    print(f"terms fitted to exactly 0: {zeros}")
    print(f"train 2016-2022: n={w.meta['n_train']} margin MAE={w.meta['train_mae']['margin']:.3f} "
          f"total MAE={w.meta['train_mae']['total']:.3f} sigma_margin={w.meta['sigma_margin']:.3f} "
          f"sigma_total={w.meta['sigma_total']:.3f}")
    if args.dry_run:
        print("--dry-run: nothing written")
        return
    write_outputs(w, ASSETS)
    print(f"wrote assets/cfb/v3_weights.json and gameline_v3.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
