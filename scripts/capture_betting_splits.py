"""Capture NFL public betting splits (cash%/ticket%) near close into Supabase
`nfl_betting_splits` -- training-side input for the cover/total ensemble's
market-microstructure features (Phase 2).

Source: the Action Network public consensus via the Apify actor
`zen-studio/action-network-odds` (see `sportsmodel.nfl.action_network`). Action
Network has no first-party API, so this is a scraper run through Apify's
run-sync-get-dataset-items endpoint (needs `APIFY_TOKEN`).

DORMANT by default: a no-op unless `INGEST_SPLITS` is truthy AND `APIFY_TOKEN`
+ `DATABASE_URL` are set. Turn it on only when the actor + token are ready and
after a one-time smoke test against a live payload.

Window: like capture-odds, only games commencing within `SPLIT_WINDOW_MIN`
minutes (default 180) and not yet started -- splits are meaningful post-lineup,
near close. Idempotent per (game_pk, market, side, captured_at).

game_pk resolution: the actor returns team abbreviations + a UTC start time; we
match those (normalized) against the same ESPN game source capture-odds uses, so
game_pk is the ESPN event id our other tables key on. Any split row whose game
isn't in the near-close ESPN window is dropped.

Usage:
    INGEST_SPLITS=true APIFY_TOKEN=... uv run python scripts/capture_betting_splits.py
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
from sportsmodel.db import upsert_nfl_betting_splits
from sportsmodel.nfl import action_network


def splits_enabled(env: dict[str, str], window_min: int) -> bool:
    """On only when explicitly enabled, a token is present, AND the window is
    positive. Default OFF (unlike odds' INGEST_PROPS which defaults on) -- the
    actor path must not run until deliberately switched on with a live token."""
    return (env.get("INGEST_SPLITS", "false").lower() == "true"
            and bool(env.get("APIFY_TOKEN"))
            and window_min > 0)


def _in_window_games(now: datetime, window_min: int) -> list[dict]:
    """Upcoming NFL games within the near-close window, as returned by the ESPN
    schedule (game_pk, home_team, away_team, commence_time, ...). Reuses the same
    ESPN game source as scripts/ingest_odds.py so game_pk is the ESPN event id
    our other tables key on."""
    from scripts.ingest_odds import _fetch_espn_games_nfl  # local import: IO
    out = []
    for g in _fetch_espn_games_nfl():
        c = g.get("commence_time")
        if not c:
            continue
        ct = datetime.fromisoformat(str(c).replace("Z", "+00:00"))
        if now < ct <= now + timedelta(minutes=window_min):
            out.append(g)
    return out


def slate_index(games: list[dict]) -> dict:
    """PURE. Map ``(home_team, away_team, utc_date) -> game_pk`` from ESPN games.
    Team codes are already normalized by the ESPN parser, so they line up with
    `action_network.parse_action_network_splits`' normalized abbreviations; the
    date is the UTC day of `commence_time` (ESPN emits Zulu times, as does the
    actor's `startTime`, so the two dates agree)."""
    idx: dict = {}
    for g in games:
        c, pk = g.get("commence_time"), g.get("game_pk")
        if not c or pk is None:
            continue
        idx[(g.get("home_team"), g.get("away_team"), str(c)[:10])] = pk
    return idx


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

    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()
    games = _in_window_games(now, window_min)
    index = slate_index(games)
    print(f"nfl betting splits: {len(games)} games in the {window_min}m window")
    if not index:
        print("no games in window; nothing to capture")
        return

    try:
        items = action_network.fetch_splits(token, leagues=("nfl",),
                                             game_status=("upcoming",))
    except Exception as exc:  # noqa: BLE001 -- a failed actor run must exit cleanly
        sys.exit(f"Action Network actor run failed: {exc}")

    parsed = action_network.parse_action_network_splits(items)
    rows = action_network.attach_game_pks(parsed, index)
    print(f"parsed {len(parsed)} split rows from {len(items)} actor games; "
          f"{len(rows)} matched a game in the window")

    for row in rows:
        row["commence_time"] = row.pop("start_time", None)
        row["captured_at"] = captured_at

    if rows:
        print(f"Upserted {upsert_nfl_betting_splits(rows)} nfl_betting_splits rows.")
    else:
        print("no splits captured (no actor game matched a window game)")


if __name__ == "__main__":
    main()
