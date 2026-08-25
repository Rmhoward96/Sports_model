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

import argparse
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config, sports
from sportsmodel.db import get_duckdb, upsert_odds_snapshot
from sportsmodel.ingest import odds

INGEST_PROPS = os.getenv("INGEST_PROPS", "true").lower() != "false"

# If set (minutes), fetch props only for games commencing within this window from now,
# and never for games already underway. This captures near-close, post-lineup prop lines
# for whatever games are about to start — day or night — instead of pulling every event
# every run (which wastes credits and grabs inflated pre-lineup longshots). Unset/<=0
# keeps the old behavior of fetching props for every matched event.
def _prop_window() -> timedelta | None:
    raw = os.getenv("PROP_WINDOW_MIN")
    try:
        mins = int(raw) if raw else 0
    except ValueError:
        mins = 0
    return timedelta(minutes=mins) if mins > 0 else None


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


def _nfl_current_season_week(now: datetime) -> tuple[int, int]:
    """Rough (season, week) for `now`, used to pick the ESPN scoreboard window.

    NFL seasons are labeled by their starting year (games from Sept 2024 through
    Feb 2025 are season=2024); week 1 kicks off roughly the first Thursday after
    Labor Day. This only needs to land within a week or two of the truth -- it
    picks which ESPN scoreboard weeks to pull, and we pull the week before/after
    too, so exact boundaries don't matter for matching odds events to games.
    """
    season = now.year if now.month >= 3 else now.year - 1
    season_start = datetime(season, 9, 1, tzinfo=timezone.utc)
    week = max(1, min(18, (now - season_start).days // 7 + 1))
    return season, week


def _fetch_espn_games_nfl() -> list[dict]:
    """ESPN NFL games (game_pk/home_name/away_name/commence_time) for the current window."""
    from sportsmodel.nfl import espn

    season, week = _nfl_current_season_week(datetime.now(timezone.utc))
    games: list[dict] = []
    for wk in sorted({max(1, week - 1), week, min(18, week + 1)}):
        games.extend(espn.fetch_schedule(season, wk))
    return games


def _run_nfl(captured_at: str) -> list[dict]:
    """NFL odds -> game_pk via the ESPN schedule + matcher.

    The Odds API has no game_pk of its own (unlike MLB, where our schedule table
    already carries the Odds-API team names), so each NFL event is resolved
    against the ESPN scoreboard by home/away display name + date.
    """
    from sportsmodel.nfl import matcher

    cfg = sports.get("nfl")
    espn_games = _fetch_espn_games_nfl()
    print(f"{len(espn_games)} ESPN games in window")

    events = odds.fetch_game_odds(cfg)
    game_lookup: dict[tuple[str, str, str], int] = {}
    matched = skipped = 0
    for ev in events:
        gp = matcher.match_odds_event(ev, espn_games)
        if gp is None:
            skipped += 1
            print(f"  no ESPN match for NFL event {ev.get('id')}: "
                  f"{ev.get('home_team')} vs {ev.get('away_team')}")
            continue
        matched += 1
        key = (ev.get("home_team"), ev.get("away_team"), odds.resolved_game_date(ev.get("commence_time")))
        game_lookup[key] = gp
    print(f"NFL events: {matched} matched to game_pk, {skipped} unmatched")

    rows = odds.parse_game_odds(events, game_lookup, captured_at)
    print(f"game-line rows: {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sport", default="mlb", help="Sport key (default: mlb)")
    args = parser.parse_args()

    captured_at = datetime.now(timezone.utc).isoformat()

    if args.sport == "nfl":
        rows = _run_nfl(captured_at)
        print(f"credits remaining: {odds.last_requests_remaining}")
        if not rows:
            print("No odds rows; nothing to store.")
            return
        write_rows(rows)
        return

    lookup = load_game_lookup()
    print(f"{len(lookup)} games on today's slate")

    rows = odds.parse_game_odds(odds.fetch_game_odds(), lookup, captured_at)
    print(f"game-line rows: {len(rows)}")

    if INGEST_PROPS:
        markets = list(odds.PROP_MARKET_MAP.values())
        events = odds.fetch_events()
        now = datetime.now(timezone.utc)
        window = _prop_window()
        matched: list[tuple[str, int]] = []
        skipped_early = skipped_started = 0
        for e in events:
            gp = lookup.get((e.get("home_team"), e.get("away_team"),
                             odds.resolved_game_date(e.get("commence_time"))))
            if gp is None:
                continue
            if window is not None:
                ct = odds.parse_commence(e.get("commence_time"))
                if ct is not None and ct < now:
                    skipped_started += 1
                    continue
                if ct is not None and ct > now + window:
                    skipped_early += 1
                    continue
            matched.append((e["id"], gp))
        if window is not None:
            mins = int(window.total_seconds() // 60)
            print(f"prop window {mins}m: {len(matched)} games in-window, "
                  f"{skipped_early} too early, {skipped_started} already started")
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
