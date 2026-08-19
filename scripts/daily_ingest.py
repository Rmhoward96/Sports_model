"""Daily current-data pull: today's + tomorrow's MLB slate and probable pitchers.

Lands raw Parquet under data/raw/schedule/ and loads it into the DuckDB warehouse as
`stg_schedule_raw`. This is the Phase-1 vertical-slice smoke test — it exercises the
ingest -> warehouse path end to end. Runs on GitHub Actions cron; also runnable locally.

Usage:
    uv run python scripts/daily_ingest.py
"""
from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from sportsmodel import config
from sportsmodel.db import get_duckdb, upsert_daily_schedule
from sportsmodel.ingest import mlb_statsapi


def main() -> None:
    days = [date.today(), date.today() + timedelta(days=1)]
    records: list[dict] = []
    for d in days:
        iso = d.isoformat()
        games = mlb_statsapi.fetch_schedule(iso)
        print(f"  {iso}: {len(games)} games")
        records.extend(games)

    if not records:
        print("No games scheduled in the window; nothing to write.")
        return

    df = pl.DataFrame(records)
    out_dir = config.RAW_DIR / "schedule"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"schedule_{days[0].isoformat()}.parquet"
    df.write_parquet(out)
    print(f"Wrote {len(df)} rows -> {out}")

    if config.DATABASE_URL:
        # Hosted mode (e.g. GitHub Actions): persist to Supabase, the durable store.
        n = upsert_daily_schedule(records)
        print(f"Upserted {n} rows into Supabase table daily_schedule.")
    else:
        # Local mode: load into DuckDB for local browsing.
        con = get_duckdb()
        con.execute(
            "CREATE OR REPLACE TABLE stg_schedule_raw AS SELECT * FROM read_parquet(?)",
            [str(out)],
        )
        n = con.execute("SELECT count(*) FROM stg_schedule_raw").fetchone()[0]
        con.close()
        print(f"Loaded {n} rows into local DuckDB table stg_schedule_raw.")


if __name__ == "__main__":
    main()
