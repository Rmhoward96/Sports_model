"""Build and commit the advancement transition table asset.

Derives empirical base-out advancement tables from all available Statcast play-by-play
(respecting transforms.set_cutoff(None) to use ALL available data), writes to
assets/advancement/mlb_advancement.parquet, and prints the row count.

Usage:
    uv run python scripts/build_advancement.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import duckdb
import polars as pl

from sportsmodel import config, transforms
from sportsmodel.sim.mlb.build_advancement import build_advancement_table


def main() -> None:
    # Use all available Statcast data (no cutoff)
    transforms.set_cutoff(None)

    # Connect to DuckDB and build the advancement table
    con = duckdb.connect(":memory:")
    rows = build_advancement_table(con)
    con.close()

    # Write to parquet asset
    out_dir = config.PROJECT_ROOT / "assets" / "advancement"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mlb_advancement.parquet"

    df = pl.DataFrame(rows)
    df.write_parquet(out)

    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
