"""Choose the QB-profile weighting (H, k) by next-game prediction (plan R3).

  uv run python scripts/tune_qb_profile.py --mode gate      # fit seasons 2021-2023
  uv run python scripts/tune_qb_profile.py --mode serving   # fit 2021 -> last completed week

Loads nflverse weekly stats 1999..current (load_release("weekly", ...)), runs
qb_profile.tune, and writes assets/nfl/props_ml/qb_profile_params.json
{"gate": {...}, "serving": {...}} (the other mode's block is kept).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.nfl import qb_profile  # noqa: E402
from sportsmodel.nfl.injuries_nflverse import nfl_season  # noqa: E402
from sportsmodel.nfl.nflverse import load_release  # noqa: E402

PARAMS_PATH = ROOT / "assets" / "nfl" / "props_ml" / "qb_profile_params.json"
FIRST_SEASON = 1999
GATE_SEASONS = [2021, 2022, 2023]
SERVING_FIRST = 2021


def fit_seasons(mode: str, current: int) -> list[int]:
    """Scored seasons: 2021-2023 (gate) or 2021..current (serving; only played
    weeks exist in the weekly stats, so only labelled weeks are scored)."""
    if mode == "gate":
        return list(GATE_SEASONS)
    if mode == "serving":
        return list(range(SERVING_FIRST, current + 1))
    raise ValueError(f"unknown mode {mode!r}")


def git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                             text=True, check=True, timeout=30)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001 -- identity is best-effort
        return "unknown"


def write_params(path: Path, mode: str, block: dict) -> dict:
    """Merge `block` under `mode` into the params json, keeping the other mode's block."""
    path = Path(path)
    data = json.loads(path.read_text()) if path.exists() else {}
    data[mode] = block
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("gate", "serving"), required=True)
    ap.add_argument("--out", type=Path, default=PARAMS_PATH)
    args = ap.parse_args(argv)

    t0 = time.time()
    current = nfl_season(datetime.now(timezone.utc))
    weekly = load_release("weekly", list(range(FIRST_SEASON, current + 1)))
    qga = qb_profile.opponent_adjust(qb_profile.qb_games(weekly))
    t_load = time.time() - t0
    seasons = fit_seasons(args.mode, current)
    last = qga[qga["season"] == qga["season"].max()]
    res = qb_profile.tune(qga, seasons)
    t_all = time.time() - t0
    block = {**res, "fit_seasons": seasons, "qb_games": int(len(qga)),
             "data_through": {"season": int(last["season"].max()), "week": int(last["week"].max())},
             "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git": git_head()}
    write_params(args.out, args.mode, block)
    b = res["best"]
    print(f"{args.mode}: chosen (1-SE rule) H={res['H']} k={res['k']} mse={res['mse']:.5f}; "
          f"best H={b['H']} k={b['k']} mse={b['mse']:.5f}, SE={res['se']:.5f}; "
          f"{res['n_games']} QB-games (load {t_load:.1f}s, total {t_all:.1f}s)")
    for g in res["grid"]:
        print(f"  H={g['H']:g} k={g['k']:g} mse={g['mse']:.5f}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
