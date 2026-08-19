# Sports Model

Multi-sport prediction engine (MLB-first). Ingests historical + current MLB data and
produces matchup-based projections for scores, totals, winners, and player props —
independent of sportsbook odds, for manual lookup against the books.

- **Design & research:** [`PLAN.md`](PLAN.md)
- **Data schema:** [`db/schema.sql`](db/schema.sql)
- **Model math (build-ready formulas):** [`docs/methodology.md`](docs/methodology.md)

## Stack

Python 3.11 · Parquet + DuckDB (local crunch) · Supabase Postgres (serving) ·
GitHub Actions cron (daily ingest) · Streamlit (lookup, later phase).

## Setup

```bash
uv sync                       # create the 3.11 env + install deps
cp .env.example .env          # then edit (DATABASE_URL optional for now)
uv run pytest                 # verify the model math
```

## Run

```bash
# Daily current-data pull (today + tomorrow slate, probable pitchers) -> DuckDB
uv run python scripts/daily_ingest.py

# One-time historical Statcast backfill (2015 -> current year) -> partitioned Parquet
uv run python scripts/backfill_mlb.py

# Apply the schema to Supabase Postgres (needs DATABASE_URL)
uv run python scripts/init_db.py
```

## Layout

```
src/sportsmodel/
  config.py            # env-driven settings (local-first defaults)
  db.py                # DuckDB + Postgres connections
  ingest/
    mlb_statsapi.py    # schedule + probable pitchers (free official API)
    mlb_statcast.py    # pitch-level backfill via pybaseball -> Parquet
  model/
    rates.py           # shrinkage + odds-ratio matchup blend  (methodology §A)
    distributions.py   # prop distributions from per-PA vectors (methodology §C)
    odds.py            # no-vig / CLV helpers                   (methodology §B.4)
scripts/               # runnable entry points
tests/                 # model-math tests
data/                  # raw Parquet + warehouse.duckdb (gitignored)
```

## Roadmap

Phase 1 (current): MLB vertical slice — ingest → warehouse → baseline model → lookup.
Then harden ingest (Phase 2), add NHL → NBA → NFL (Phase 3), deepen props with
Monte-Carlo game sim (Phase 4). See [`PLAN.md`](PLAN.md) §4.
