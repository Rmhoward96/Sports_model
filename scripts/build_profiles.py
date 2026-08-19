"""Build rate-profile tables from the Statcast Parquet backfill.

Reads data/raw/statcast/**, classifies PAs, and writes feat_batter_profile,
feat_pitcher_profile, and ref_league_rates into the local DuckDB warehouse.
Run after scripts/backfill_mlb.py.

Usage:
    uv run python scripts/build_profiles.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import transforms
from sportsmodel.db import get_duckdb


def main() -> None:
    con = get_duckdb()
    counts = transforms.build_all(con)
    for table, n in counts.items():
        print(f"  {table}: {n:,} rows")
    con.close()
    print("Profiles built.")


if __name__ == "__main__":
    main()
