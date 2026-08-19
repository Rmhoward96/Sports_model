"""Statcast pitch-level backfill via pybaseball -> partitioned Parquet.

Writes one file per season under data/raw/statcast/season=YYYY/. Resumable: a season
whose file already exists is skipped, so a crash mid-backfill restarts cleanly.
"""
from __future__ import annotations

from pathlib import Path

from .. import config


def _season_path(season: int) -> Path:
    return config.RAW_DIR / "statcast" / f"season={season}" / f"statcast_{season}.parquet"


def backfill_season(season: int, *, overwrite: bool = False) -> Path:
    """Pull one season of Statcast pitch data and write it to Parquet.

    pybaseball auto-chunks the request internally; we still pull a single season at a
    time to keep memory bounded and make the backfill resumable per season.
    """
    out = _season_path(season)
    if out.exists() and not overwrite:
        return out

    from pybaseball import statcast  # heavy import; keep it lazy

    start = f"{season}-{config.SEASON_START_MMDD}"
    end = f"{season}-{config.SEASON_END_MMDD}"
    df = statcast(start_dt=start, end_dt=end)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def backfill_range(start_season: int, end_season: int, *, overwrite: bool = False):
    """Backfill an inclusive range of seasons; yields (season, path) as each lands."""
    for season in range(start_season, end_season + 1):
        yield season, backfill_season(season, overwrite=overwrite)
