"""Snapshot current NFL + CFB game-line odds (moneyline/spread/total) from The Odds API.

Game lines only -- no player props. For each sport, events are joined to our
ESPN game_pk by (home_team, away_team, US_game_date) via that sport's odds-event
matcher (the Odds API carries no game_pk of its own), then stored to
odds_snapshot tagged with a shared captured_at. Run repeatedly through the day;
the last snapshot before a game's commence_time is its closing line.

Usage:
    uv run python scripts/ingest_odds.py
Requires ODDS_API_KEY.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import sports
from sportsmodel.db import upsert_odds_snapshot
from sportsmodel.ingest import odds

SPORTS = ("nfl", "cfb")


def build_game_lookup(
    events: list[dict],
    espn_games: list[dict],
    matcher_fn: Callable[[dict, list[dict]], int | None],
    resolved_game_date_fn: Callable[[str | None], str | None] = odds.resolved_game_date,
) -> dict[tuple[Any, Any, Any], int]:
    """{(home_team, away_team, US_game_date): game_pk} for Odds events matched to
    an ESPN game via `matcher_fn`. Pure and network-free: `events`/`espn_games` are
    plain dicts and `matcher_fn` is the sport's `match_odds_event`, so this is
    unit-testable without hitting either API. Unmatched events are dropped.
    """
    lookup: dict[tuple[Any, Any, Any], int] = {}
    for ev in events:
        gp = matcher_fn(ev, espn_games)
        if gp is None:
            continue
        key = (ev.get("home_team"), ev.get("away_team"),
               resolved_game_date_fn(ev.get("commence_time")))
        lookup[key] = gp
    return lookup


def _fetch_espn_games_nfl() -> list[dict]:
    """ESPN NFL games (game_pk/home_name/away_name/commence_time) for the current
    window: the live current week (or the Week-1 look-ahead pre-season) plus the
    adjacent weeks, so late/early-window odds events still find a match.
    """
    from sportsmodel.nfl import espn

    cur = espn.resolve_target_week()
    season, week, season_type = cur["season"], cur["week"], cur["season_type"]
    games: list[dict] = []
    for wk in sorted({w for w in (week - 1, week, week + 1) if w >= 1}):
        games.extend(espn.fetch_schedule(season, wk, season_type=season_type))
    return games


def _fetch_espn_games_cfb() -> list[dict]:
    """ESPN CFB games for the current window, mirroring the NFL fetch above.

    CFB has no `resolve_target_week`/look-ahead helper (see cfb/espn.py) -- just
    `fetch_current_week`, which returns ESPN's live (season, week, season_type)
    directly. Adjacent weeks are still pulled to cover odds events that land just
    outside the current week's window.
    """
    from sportsmodel.cfb import espn

    cur = espn.fetch_current_week()
    season, week, season_type = cur["season"], cur["week"], cur["season_type"]
    games: list[dict] = []
    for wk in sorted({w for w in (week - 1, week, week + 1) if w >= 1}):
        games.extend(espn.fetch_schedule(season, wk, season_type=season_type))
    return games


_ESPN_FETCHERS: dict[str, Callable[[], list[dict]]] = {
    "nfl": _fetch_espn_games_nfl,
    "cfb": _fetch_espn_games_cfb,
}


def _matcher_for(sport: str) -> Callable[[dict, list[dict]], int | None]:
    if sport == "nfl":
        from sportsmodel.nfl import matcher
    else:
        from sportsmodel.cfb import matcher
    return matcher.match_odds_event


def run_sport(sport: str, captured_at: str) -> list[dict]:
    cfg = sports.get(sport)
    espn_games = _ESPN_FETCHERS[sport]()
    print(f"[{sport}] {len(espn_games)} ESPN games in window")

    events = odds.fetch_game_odds(cfg)
    matcher_fn = _matcher_for(sport)
    game_lookup = build_game_lookup(events, espn_games, matcher_fn)
    matched = len(game_lookup)
    unmatched = len(events) - matched
    print(f"[{sport}] events: {len(events)} fetched, {matched} matched, {unmatched} unmatched")

    rows = odds.parse_game_odds(events, game_lookup, captured_at)
    print(f"[{sport}] rows: {len(rows)}")
    return rows


def main() -> None:
    if not os.getenv("ODDS_API_KEY"):
        raise RuntimeError("ODDS_API_KEY is not set (add it to .env / repo secrets).")

    captured_at = datetime.now(timezone.utc).isoformat()

    for sport in SPORTS:
        # The whole per-sport pipeline -- fetch, match, parse AND store -- is
        # inside the boundary: a transient failure anywhere (ESPN, the Odds API,
        # or the odds_snapshot write) logs and moves on to the next sport rather
        # than aborting the run, so one sport can never take the other down.
        try:
            rows = run_sport(sport, captured_at)
            if rows:
                n = upsert_odds_snapshot(rows)
                print(f"[{sport}] stored {n} odds rows in odds_snapshot")
            else:
                print(f"[{sport}] no odds rows; nothing to store")
            print(f"[{sport}] credits remaining: {odds.last_requests_remaining}")
        except Exception as e:  # one sport's failure shouldn't kill the other
            print(f"[{sport}] failed: {e}")
            continue


if __name__ == "__main__":
    main()
