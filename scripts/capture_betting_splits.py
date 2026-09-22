"""Capture public betting splits (cash%/ticket%) near close into Supabase
`nfl_betting_splits` / `cfb_betting_splits` -- serving-side input for the game
pages' Public Betting section (and NFL training-side market features).

Source: the Action Network public consensus via the Apify actor
`zen-studio/action-network-odds` (see `sportsmodel.nfl.action_network`). Action
Network has no first-party API, so this is a scraper run through Apify's
run-sync-get-dataset-items endpoint (needs `APIFY_TOKEN`).

DORMANT by default: a no-op unless `INGEST_SPLITS` is truthy AND `APIFY_TOKEN`
+ `DATABASE_URL` are set. Runs NFL + CFB by default (`SPORTS` env overrides,
e.g. `SPORTS=cfb`).

Window: like capture-odds, only games commencing within `SPLIT_WINDOW_MIN`
minutes (default 180) and not yet started. Idempotent per
(game_pk, market, side, captured_at).

game_pk resolution: the actor returns team abbreviations + names + a UTC start
time; we match against the same ESPN game source capture-odds uses. NFL matches
on normalized abbreviation; CFB's ESPN slate keys on team ID (not abbrev), so it
matches on team display NAME instead. Rows whose game isn't in the near-close
ESPN window are dropped.

Usage:
    INGEST_SPLITS=true APIFY_TOKEN=... uv run python scripts/capture_betting_splits.py
Smoke test one sport/week (see AN_* overrides):
    SPORTS=cfb AN_SEASON=2026 AN_WEEK=5 AN_DEBUG=1 SPLIT_WINDOW_MIN=6000 ...
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT))  # so `scripts.*` imports work when run as a file

from sportsmodel import config
from sportsmodel.db import upsert_cfb_betting_splits, upsert_nfl_betting_splits
from sportsmodel.nfl import action_network

# ESPN season_type int -> Action Network actor seasonType enum.
_ESPN_SEASONTYPE_TO_AN = {1: "pre", 2: "reg", 3: "post"}


def splits_enabled(env: dict[str, str], window_min: int) -> bool:
    """On only when explicitly enabled, a token is present, AND the window is
    positive. Default OFF (unlike odds' INGEST_PROPS which defaults on) -- the
    actor path must not run until deliberately switched on with a live token."""
    return (env.get("INGEST_SPLITS", "false").lower() == "true"
            and bool(env.get("APIFY_TOKEN"))
            and window_min > 0)


def slate_index(games: list[dict]) -> dict:
    """PURE. Map ``(home_team, away_team, utc_date) -> game_pk`` from ESPN games
    (NFL). Team codes are the ESPN parser's normalized abbreviations, matching
    `parse_action_network_splits`' normalized abbrevs; date is the UTC day of
    `commence_time`."""
    idx: dict = {}
    for g in games:
        c, pk = g.get("commence_time"), g.get("game_pk")
        if not c or pk is None:
            continue
        idx[(g.get("home_team"), g.get("away_team"), str(c)[:10])] = pk
    return idx


def slate_index_by_name(games: list[dict]) -> dict:
    """PURE. Map ``(home_name_norm, away_name_norm, utc_date) -> game_pk`` from
    ESPN games (CFB, whose team codes are IDs, not abbrevs). Names are normalized
    the same way as `action_network._norm_name`."""
    from sportsmodel.nfl.action_network import _norm_name
    idx: dict = {}
    for g in games:
        c, pk = g.get("commence_time"), g.get("game_pk")
        if not c or pk is None:
            continue
        idx[(_norm_name(g.get("home_name")), _norm_name(g.get("away_name")), str(c)[:10])] = pk
    return idx


def _in_window_games(fetch_games, now: datetime, window_min: int) -> list[dict]:
    """ESPN games within the near-close window (not yet started, commencing
    within window_min). `fetch_games` is the sport's ESPN fetcher."""
    out = []
    for g in fetch_games():
        c = g.get("commence_time")
        if not c:
            continue
        ct = datetime.fromisoformat(str(c).replace("Z", "+00:00"))
        if now < ct <= now + timedelta(minutes=window_min):
            out.append(g)
    return out


def _sport_cfg(sport: str) -> dict:
    """Per-sport wiring: leagues, ESPN fetcher, slate index + attach strategy,
    upsert target, and the actor season/week/seasonType scope."""
    if sport == "nfl":
        from scripts.ingest_odds import _fetch_espn_games_nfl
        from sportsmodel.nfl import espn as nfl_espn

        def scope():
            cur = nfl_espn.resolve_target_week()
            return {"season": cur["season"], "week": cur["week"],
                    "seasonType": _ESPN_SEASONTYPE_TO_AN.get(cur["season_type"], "reg")}

        return {"leagues": ("nfl",), "fetch_games": _fetch_espn_games_nfl,
                "index": slate_index, "attach": action_network.attach_game_pks,
                "upsert": upsert_nfl_betting_splits, "table": "nfl_betting_splits",
                "scope": scope}
    if sport == "cfb":
        from scripts.ingest_odds import _fetch_espn_games_cfb
        from sportsmodel.cfb import espn as cfb_espn

        def scope():
            cur = cfb_espn.fetch_current_week()
            return {"season": cur["season"], "week": cur["week"],
                    "seasonType": _ESPN_SEASONTYPE_TO_AN.get(cur["season_type"], "reg")}

        return {"leagues": ("ncaaf",), "fetch_games": _fetch_espn_games_cfb,
                "index": slate_index_by_name, "attach": action_network.attach_game_pks_by_name,
                "upsert": upsert_cfb_betting_splits, "table": "cfb_betting_splits",
                "scope": scope}
    raise ValueError(f"unsupported sport {sport!r}")


def _capture_sport(sport: str, token: str, window_min: int,
                   now: datetime, captured_at: str) -> int:
    """Capture one sport's splits: window -> slate index -> actor -> parse ->
    match -> upsert. Returns the number of rows upserted."""
    cfg = _sport_cfg(sport)
    games = _in_window_games(cfg["fetch_games"], now, window_min)
    index = cfg["index"](games)
    print(f"{sport} betting splits: {len(games)} games in the {window_min}m window")
    if not index:
        print(f"  {sport}: no games in window; nothing to capture")
        return 0

    # Scope the actor to the current season/week (its default date logic returns
    # 0 games off-day). AN_SEASON/AN_WEEK/AN_STATUS override for smoke tests.
    if os.getenv("AN_SEASON") and os.getenv("AN_WEEK"):
        extra = {"season": int(os.environ["AN_SEASON"]), "week": int(os.environ["AN_WEEK"])}
    else:
        extra = cfg["scope"]()
    status = tuple(s.strip() for s in (os.getenv("AN_STATUS") or "scheduled").split(",") if s.strip())

    try:
        items = action_network.fetch_splits(token, leagues=cfg["leagues"],
                                            game_status=status, extra_input=extra or None)
    except Exception as exc:  # noqa: BLE001 -- a failed actor run must not abort other sports
        print(f"  {sport}: Action Network actor run failed: {exc}")
        return 0

    if os.getenv("AN_DEBUG") and items:
        import json
        print(f"  AN_DEBUG {sport} first raw item (truncated):")
        print(json.dumps(items[0], indent=1, default=str)[:3500])

    parsed = action_network.parse_action_network_splits(items)
    rows = cfg["attach"](parsed, index)
    print(f"  {sport}: parsed {len(parsed)} rows from {len(items)} actor games; "
          f"{len(rows)} matched a window game")
    if parsed and not rows:
        s = parsed[0]
        print(f"  {sport} sample parsed row: {s.get('away_name')}@{s.get('home_name')} "
              f"({s.get('away_abbr')}@{s.get('home_abbr')}) {s['market']}/{s['side']} "
              f"ticket={s['ticket_pct']} start={s['start_time']}")

    for row in rows:
        row["commence_time"] = row.pop("start_time", None)
        row["captured_at"] = captured_at
    if not rows:
        return 0
    n = cfg["upsert"](rows)
    print(f"  {sport}: upserted {n} {cfg['table']} rows.")
    return n


def main() -> None:
    # An unset repo variable arrives as "" (present-but-empty), so a default
    # arg to getenv wouldn't apply -- `or 180` covers both unset and empty.
    window_min = int(os.getenv("SPLIT_WINDOW_MIN") or 180)
    if not splits_enabled(os.environ, window_min):
        print("betting-splits capture disabled "
              "(set INGEST_SPLITS=true and APIFY_TOKEN to enable)")
        return
    if not config.DATABASE_URL:
        sys.exit("DATABASE_URL required.")
    token = os.environ["APIFY_TOKEN"]

    sports = [s.strip() for s in (os.getenv("SPORTS") or "nfl,cfb").split(",") if s.strip()]
    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()
    total = 0
    for sport in sports:
        try:
            total += _capture_sport(sport, token, window_min, now, captured_at)
        except Exception as exc:  # noqa: BLE001 -- one sport failing must not abort the rest
            print(f"{sport}: capture failed: {exc}")
    print(f"done: {total} split rows upserted across {sports}")


if __name__ == "__main__":
    main()
