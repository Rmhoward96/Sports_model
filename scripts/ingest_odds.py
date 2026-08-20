"""Snapshot current MLB odds (game lines + player props) from The Odds API.

Joins to today's slate by team names, tags each row with captured_at, and stores to
odds_snapshot (Supabase if DATABASE_URL set, else local DuckDB). Run repeatedly through
the day (see capture-odds.yml); the last snapshot before a game's commence_time is its
closing line. Props are per-event and credit-heavy — disable with INGEST_PROPS=false.

Usage:
    uv run python scripts/ingest_odds.py
Requires ODDS_API_KEY.
"""
from __future__ import annotations

import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import get_duckdb, upsert_odds_snapshot
from sportsmodel.ingest import odds

INGEST_PROPS = os.getenv("INGEST_PROPS", "true").lower() != "false"


def load_game_lookup() -> dict[tuple[str, str, str], int]:
    """{(home_team_name, away_team_name, game_date): game_pk} over a multi-day window.

    Odds events span several upcoming days; matching by team + the event's resolved US
    game date (not just "today UTC") avoids the date-boundary mismatch with predictions.
    """
    start = (date.today() - timedelta(days=2)).isoformat()
    end = (date.today() + timedelta(days=3)).isoformat()
    cols = "game_pk, home_team_name, away_team_name, game_date"
    if config.DATABASE_URL:
        from sportsmodel.db import get_postgres
        with get_postgres() as pg, pg.cursor() as cur:
            cur.execute(f"SELECT {cols} FROM daily_schedule WHERE game_date BETWEEN %s AND %s",
                        [start, end])
            rows = cur.fetchall()
    else:
        con = get_duckdb(read_only=True)
        rows = con.execute(f"SELECT {cols} FROM stg_schedule_raw WHERE game_date BETWEEN ? AND ?",
                           [start, end]).fetchall()
        con.close()
    return {(home, away, str(gd)): pk for pk, home, away, gd in rows}


def main() -> None:
    captured_at = datetime.now(timezone.utc).isoformat()
    lookup = load_game_lookup()
    print(f"{len(lookup)} games on today's slate")

    rows = odds.parse_game_odds(odds.fetch_game_odds(), lookup, captured_at)
    print(f"game-line rows: {len(rows)}")

    if INGEST_PROPS:
        markets = list(odds.PROP_MARKET_MAP.values())
        events = odds.fetch_events()
        matched = [(e["id"], lookup.get((e.get("home_team"), e.get("away_team"),
                                         odds.resolved_game_date(e.get("commence_time")))))
                   for e in events]
        matched = [(eid, gp) for eid, gp in matched if gp is not None]
        before = len(rows)
        for eid, gp in matched:
            try:
                ep = odds.fetch_event_props(eid, markets)
                rows += odds.parse_prop_odds(ep, gp, captured_at)
            except Exception as e:  # one bad event shouldn't kill the run
                print(f"  prop fetch failed for event {eid}: {e}")
        print(f"prop rows: {len(rows) - before} (from {len(matched)} events)")

    print(f"credits remaining: {odds.last_requests_remaining}")
    if not rows:
        print("No odds rows; nothing to store.")
        return
    write_rows(rows)


def write_rows(rows: list[dict]) -> None:
    if config.DATABASE_URL:
        n = upsert_odds_snapshot(rows)
        print(f"Stored {n} odds rows in Supabase odds_snapshot.")
        return
    con = get_duckdb()
    con.execute("""
        CREATE TABLE IF NOT EXISTS odds_snapshot (
            game_pk BIGINT, market TEXT, side TEXT, player_name TEXT, book TEXT,
            line REAL, price INTEGER, commence_time TEXT, captured_at TEXT,
            PRIMARY KEY (game_pk, market, side, player_name, book, captured_at))
    """)
    cols = ["game_pk", "market", "side", "player_name", "book", "line",
            "price", "commence_time", "captured_at"]
    for r in rows:
        con.execute(
            f"INSERT OR IGNORE INTO odds_snapshot ({','.join(cols)}) "
            f"VALUES ({','.join(['?']*len(cols))})", [r.get(c) for c in cols])
    con.close()
    print(f"Stored {len(rows)} odds rows in local DuckDB odds_snapshot.")


if __name__ == "__main__":
    main()
