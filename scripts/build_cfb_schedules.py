"""Build historical CFB game-line schedule asset from ESPN.

Pulls one scoreboard call per (season, week) via cfb.espn.fetch_schedule,
keeps only completed (STATUS_FINAL, both scores present) regular-season
games, and writes a flat schedule to assets/cfb/schedules.parquet with the
columns nfl.elo.run_elo / nfl.srs.compute_srs expect: season, week,
home_team, away_team, home_score, away_score (plus game_type="REG").

Usage:
    PYTHONPATH=src uv run --no-sync python scripts/build_cfb_schedules.py
    PYTHONPATH=src uv run --no-sync python scripts/build_cfb_schedules.py \
        --seasons 2023 2024 2025 --weeks 1 2 3
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from sportsmodel.cfb import espn


def _current_cfb_season(now: datetime | None = None) -> int:
    """CFB season year in progress (Aug-onward = that year; Jan-Feb bowls belong
    to the prior year's season). Keeps DEFAULT_SEASONS current without edits."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1


# Through the current season, so a rebuild/refresh always includes this year.
DEFAULT_SEASONS = list(range(2015, _current_cfb_season() + 1))
DEFAULT_WEEKS = list(range(1, 17))

OUT_PATH = Path(__file__).resolve().parents[1] / "assets" / "cfb" / "schedules.parquet"


def _fetch_week(season: int, week: int, retries: int = 3) -> list[dict]:
    last_exc = None
    for attempt in range(retries):
        try:
            return espn.fetch_schedule(season, week)
        except Exception as exc:  # noqa: BLE001 - retry any transient failure
            last_exc = exc
    print(f"  WARN: season {season} week {week} failed after {retries} attempts: {last_exc}")
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CFB schedules parquet from ESPN.")
    parser.add_argument("--seasons", type=int, nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--weeks", type=int, nargs="+", default=DEFAULT_WEEKS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--merge", action="store_true",
                        help="Merge into the existing parquet: replace only the "
                             "fetched seasons' rows, keep all other seasons. Use "
                             "for an in-season refresh so history isn't lost.")
    args = parser.parse_args()

    pairs = [(s, w) for s in args.seasons for w in args.weeks]
    print(f"Fetching {len(pairs)} (season, week) scoreboards...")

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_week, s, w): (s, w) for s, w in pairs}
        for fut in as_completed(futures):
            rows.extend(fut.result())

    df = pd.DataFrame(rows)
    print(f"Fetched {len(df)} raw rows across all weeks.")

    final = df[
        (df["status"] == "STATUS_FINAL")
        & df["home_score"].notna()
        & df["away_score"].notna()
    ].copy()

    final["game_type"] = "REG"
    final = final[
        ["season", "week", "home_team", "away_team", "home_score", "away_score", "game_type"]
    ]
    final = final.drop_duplicates(subset=["season", "week", "home_team", "away_team"])

    if args.merge and OUT_PATH.exists():
        # Preserve seasons we did NOT re-fetch; replace the fetched ones wholesale.
        existing = pd.read_parquet(OUT_PATH)
        fetched_seasons = set(args.seasons)
        kept = existing[~existing["season"].isin(fetched_seasons)]
        before = len(existing)
        final = pd.concat([kept, final], ignore_index=True)
        print(f"Merge: kept {len(kept)}/{before} existing rows outside seasons "
              f"{sorted(fetched_seasons)}, added {len(final) - len(kept)} refreshed rows.")

    final = final.drop_duplicates(subset=["season", "week", "home_team", "away_team"])
    final = final.sort_values(["season", "week", "home_team", "away_team"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(OUT_PATH, index=False, engine="pyarrow")

    n_games = len(final)
    n_teams = len(set(final["home_team"]) | set(final["away_team"]))
    fcs_share = (
        (final["home_team"].eq("FCS") | final["away_team"].eq("FCS")).mean()
        if n_games
        else 0.0
    )

    print(f"\nWrote {OUT_PATH} ({n_games} games)")
    print("Games per season:")
    print(final.groupby("season").size().to_string())
    print(f"Distinct teams (incl. FCS pseudo-team): {n_teams}")
    print(f"Share of games involving FCS: {fcs_share:.1%}")


if __name__ == "__main__":
    main()
