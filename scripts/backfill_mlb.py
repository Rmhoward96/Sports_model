"""One-time historical Statcast backfill -> partitioned Parquet.

Usage:
    uv run python scripts/backfill_mlb.py                 # 2015 -> current year
    uv run python scripts/backfill_mlb.py --start 2018 --end 2020
    uv run python scripts/backfill_mlb.py --overwrite     # re-pull existing seasons
"""
from __future__ import annotations

import argparse
from datetime import date

from sportsmodel import config
from sportsmodel.ingest import mlb_statcast


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill MLB Statcast to Parquet.")
    parser.add_argument("--start", type=int, default=config.STATCAST_START_SEASON)
    parser.add_argument("--end", type=int, default=date.today().year)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    print(f"Backfilling Statcast {args.start}-{args.end} -> {config.RAW_DIR/'statcast'}")
    for season, path in mlb_statcast.backfill_range(
        args.start, args.end, overwrite=args.overwrite
    ):
        print(f"  season {season}: {path}")
    print("Done.")


if __name__ == "__main__":
    main()
