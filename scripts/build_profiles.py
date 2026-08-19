"""Build rate-profile tables from the Statcast Parquet backfill, and export small
committed snapshots the hosted prediction job reads.

Reads data/raw/statcast/**, classifies PAs, writes feat_batter_profile,
feat_pitcher_profile, ref_league_rates into the local DuckDB warehouse, then exports
each to assets/profiles/*.parquet (committed to the repo). Run after backfill_mlb.py;
re-run periodically as new games accrue, then commit the updated snapshots.

Usage:
    uv run python scripts/build_profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config, transforms
from sportsmodel.db import get_duckdb

PROFILE_DIR = config.PROJECT_ROOT / "assets" / "profiles"


def main() -> None:
    con = get_duckdb()
    counts = transforms.build_all(con)
    for table, n in counts.items():
        print(f"  {table}: {n:,} rows")

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    for table in counts:
        out = PROFILE_DIR / f"{table}.parquet"
        con.execute(f"COPY {table} TO '{out}' (FORMAT parquet)")
        print(f"  exported -> {out.relative_to(config.PROJECT_ROOT)}")
    con.close()
    print("Profiles built and exported. Commit assets/profiles/ to deploy them.")


if __name__ == "__main__":
    main()
