"""Central configuration. Reads .env; sensible local-first defaults."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("SM_DATA_DIR", str(PROJECT_ROOT / "data")))
RAW_DIR = DATA_DIR / "raw"
DUCKDB_PATH = Path(os.getenv("SM_DUCKDB_PATH", str(DATA_DIR / "warehouse.duckdb")))

# Backfill depth: Statcast full pitch tracking is reliable from 2015.
STATCAST_START_SEASON = int(os.getenv("SM_STATCAST_START_SEASON", "2015"))

# Optional Supabase Postgres serving layer. Unset => run fully local (DuckDB only).
DATABASE_URL = os.getenv("DATABASE_URL") or None

# Rough MLB season window used for chunked backfill pulls.
SEASON_START_MMDD = "03-15"
SEASON_END_MMDD = "11-15"
