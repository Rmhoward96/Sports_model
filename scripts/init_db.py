"""Apply db/schema.sql to the Supabase Postgres serving layer.

The schema is Postgres-flavored (JSONB, BIGSERIAL, GIN). The local DuckDB warehouse
is built by transforms over Parquet and does not use this DDL directly.

Usage:
    uv run python scripts/init_db.py
Requires DATABASE_URL in .env.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import get_postgres


def main() -> None:
    if not config.DATABASE_URL:
        raise SystemExit(
            "DATABASE_URL is not set. Add your Supabase connection string to .env "
            "(see .env.example). Skipping schema apply — you can still run local-only."
        )
    schema_sql = (config.PROJECT_ROOT / "db" / "schema.sql").read_text()
    with get_postgres() as conn, conn.cursor() as cur:
        cur.execute(schema_sql)
        conn.commit()
    print("Applied db/schema.sql to Supabase Postgres.")


if __name__ == "__main__":
    main()
