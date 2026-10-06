"""Sanity-check the committed CFBD game-data assets (units, havoc sides, drive points, coverage).

Exit 0 and print "OK" when everything looks right; otherwise print each problem and exit 1.
Run it after every backfill (plan Task 4) and before fitting. Local only (no network, no key).

Usage:
    uv run python scripts/check_cfb_game_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import data_checks, v3_data  # noqa: E402


def main() -> int:
    names = {"advanced": "advanced_games.parquet", "havoc": "havoc_games.parquet",
             "drives": "drive_games.parquet", "weather": "weather_games.parquet"}
    assets = {k: v3_data.read_asset(f) for k, f in names.items()}
    problems = data_checks.check_all(assets, v3_data.require_asset("schedules.parquet"))
    for k, df in assets.items():
        print(f"{k}: " + ("missing" if df is None else f"{len(df)} rows, seasons {df.season.min()}-{df.season.max()}"))
    if problems:
        print("\nPROBLEMS:\n- " + "\n- ".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
