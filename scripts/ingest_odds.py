"""Snapshot current NFL + CFB game-line (and NFL player-prop) odds from The Odds API.

For each sport, events are joined to our ESPN game_pk by (home_team, away_team,
US_game_date) via that sport's odds-event matcher (the Odds API carries no
game_pk of its own), then stored to odds_snapshot tagged with a shared
captured_at. Run repeatedly through the day; the last snapshot before a game's
commence_time is its closing line.

Player props are additionally captured, per event, for any sport whose
`SportConfig.prop_market_map` is non-empty (currently NFL only -- CFB's map is
empty and so no-ops) and only for events inside a post-lineup pre-kickoff
window: `now < commence_time <= now + prop_window_minutes() minutes`. The window
is controlled by PROP_SCOPE (slate → a full week so every upcoming game's props
are captured) or PROP_WINDOW_MIN (default 150 minutes; blank → 0 = disabled).
Props are one Odds-API call per in-window event, so credits scale with how many
games are in that window at run time. Set INGEST_PROPS=false or PROP_WINDOW_MIN=0
to disable prop capture entirely and fall back to game-lines-only behavior.

Usage:
    uv run python scripts/ingest_odds.py
Requires ODDS_API_KEY.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import sports
from sportsmodel.db import upsert_odds_snapshot
from sportsmodel.ingest import odds

SPORTS = ("nfl", "cfb")
SLATE_WINDOW_MIN = 7 * 24 * 60


def prop_window_minutes(env: dict[str, str]) -> int:
    """Prop-capture window in minutes. PROP_SCOPE=slate (the daily full-slate
    pull) widens it to a week so every upcoming game's props are captured;
    otherwise PROP_WINDOW_MIN (default 150; blank -> 0 = disabled)."""
    if env.get("PROP_SCOPE", "").lower() == "slate":
        return SLATE_WINDOW_MIN
    return int(env.get("PROP_WINDOW_MIN", "150") or 0)


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


def events_in_prop_window(
    events: list[dict],
    matcher_fn: Callable[[dict, list[dict]], int | None],
    espn_games: list[dict],
    now: datetime,
    window_min: int,
) -> list[tuple[dict, int]]:
    """Odds events matched to an ESPN game and commencing inside the prop-capture
    window: upcoming (not yet started) and within `window_min` minutes from `now`.

    Pure and network-free: `now` is passed in (tz-aware UTC) rather than read from
    the clock, so this is unit-testable. Unmatched events and events with no
    parseable commence_time are dropped, same as `build_game_lookup`.
    """
    result: list[tuple[dict, int]] = []
    for ev in events:
        gp = matcher_fn(ev, espn_games)
        if gp is None:
            continue
        c = odds.parse_commence(ev.get("commence_time"))
        if c is None:
            continue
        if now < c <= now + timedelta(minutes=window_min):
            result.append((ev, gp))
    return result


def props_enabled(env: dict[str, str], window_min: int) -> bool:
    """Whether prop capture is turned on: INGEST_PROPS isn't "false" and the
    window is positive. Factored out (env passed in, not read from os.environ)
    so it's testable without monkeypatching.
    """
    return env.get("INGEST_PROPS", "true").lower() != "false" and window_min > 0


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

    window_min = prop_window_minutes(os.environ)
    if cfg.prop_market_map and props_enabled(os.environ, window_min):
        prop_markets = list(cfg.prop_market_map.values())
        in_window = events_in_prop_window(
            events, matcher_fn, espn_games, datetime.now(timezone.utc), window_min
        )
        n_prop_rows = 0
        for ev, gp in in_window:
            try:
                props = odds.fetch_event_props(ev["id"], prop_markets, cfg)
                prop_rows = odds.parse_prop_odds(props, gp, captured_at, cfg)
                rows += prop_rows
                n_prop_rows += len(prop_rows)
            except Exception as e:  # one event's failure shouldn't abort the sport
                print(f"[{sport}] props: event {ev.get('id')} failed: {e}")
                continue
        print(f"[{sport}] props: {len(in_window)} events in window, {n_prop_rows} rows")

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
