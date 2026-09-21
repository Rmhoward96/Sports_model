"""Capture NFL public betting splits (cash%/ticket%) near close into Supabase
`nfl_betting_splits` -- training-side input for the cover/total ensemble's
market-microstructure features (Phase 2).

DORMANT by default: this is a no-op unless `INGEST_SPLITS` is truthy (and
`SPORTSDATA_API_KEY` + `DATABASE_URL` are set). It stays off until the
SportsDataIO **Betting tier** is active, because the exact splits endpoint,
field names, and the SportsDataIO->ESPN game-id resolution must be confirmed
against a live payload first (see the marked spots below and
`sportsdata.parse_betting_splits`'s caveat).

Window: like capture-odds, only games commencing within `SPLIT_WINDOW_MIN`
minutes (default 180) and not yet started -- splits are meaningful post-lineup,
near close. Idempotent per (game_pk, market, side, captured_at).

Usage:
    INGEST_SPLITS=true uv run python scripts/capture_betting_splits.py
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import upsert_nfl_betting_splits
from sportsmodel.nfl import sportsdata


def splits_enabled(env: dict[str, str], window_min: int) -> bool:
    """On only when explicitly enabled AND the window is positive. Default OFF
    (unlike odds' INGEST_PROPS which defaults on) -- the Betting tier is not
    yet active, so this must not run until deliberately switched on."""
    return env.get("INGEST_SPLITS", "false").lower() == "true" and window_min > 0


def _in_window_games(now: datetime, window_min: int) -> list[dict]:
    """Upcoming NFL games within the near-close window, as
    {game_pk, home_team, away_team, commence_time}. Reuses the same ESPN game
    source as scripts/ingest_odds.py so game_pk is the ESPN event id our other
    tables key on."""
    from scripts.ingest_odds import _fetch_espn_games_nfl  # local import: IO
    from sportsmodel.nfl import espn as nfl_espn  # noqa: F401 (kept for parity)
    out = []
    for g in _fetch_espn_games_nfl():
        c = g.get("commence_time")
        if not c:
            continue
        ct = datetime.fromisoformat(str(c).replace("Z", "+00:00"))
        if now < ct <= now + timedelta(minutes=window_min):
            out.append(g)
    return out


def _fetch_and_parse_splits(game: dict, api_key: str) -> list[dict]:
    """Fetch + parse one game's splits. CONFIRM-AT-TIER: the SportsDataIO
    score-id resolution and BETTING_SPLITS_PATH templating below are written to
    the documented Betting swagger and must be verified against a live payload.
    Returns rows shaped for `upsert_nfl_betting_splits` (game_pk/commence/
    captured_at attached by the caller)."""
    score_id = game.get("sportsdata_score_id")  # CONFIRM: resolve via a SportsDataIO
    if score_id is None:                         # scores-by-date lookup once the tier is live.
        return []
    payload = sportsdata._get(sportsdata.BETTING_SPLITS_PATH.format(score_id=score_id), api_key)
    return sportsdata.parse_betting_splits(payload)


def main() -> None:
    window_min = int(os.getenv("SPLIT_WINDOW_MIN", "180") or 0)
    if not splits_enabled(os.environ, window_min):
        print("betting-splits capture disabled (set INGEST_SPLITS=true to enable)")
        return
    api_key = os.environ.get("SPORTSDATA_API_KEY")
    if not api_key:
        sys.exit("SPORTSDATA_API_KEY required when INGEST_SPLITS=true.")
    if not config.DATABASE_URL:
        sys.exit("DATABASE_URL required.")

    now = datetime.now(timezone.utc)
    captured_at = now.isoformat()
    games = _in_window_games(now, window_min)
    print(f"nfl betting splits: {len(games)} games in the {window_min}m window")

    records: list[dict] = []
    for g in games:
        try:
            for row in _fetch_and_parse_splits(g, api_key):
                row.update({"game_pk": g["game_pk"], "commence_time": g.get("commence_time"),
                            "captured_at": captured_at})
                records.append(row)
        except Exception as exc:  # noqa: BLE001 -- one bad game must not abort the batch
            print(f"  splits fetch failed for {g.get('game_pk')}: {exc}; skipping")

    if records:
        print(f"Upserted {upsert_nfl_betting_splits(records)} nfl_betting_splits rows.")
    else:
        print("no splits captured (window empty, or resolution not yet confirmed for the live tier)")


if __name__ == "__main__":
    main()
