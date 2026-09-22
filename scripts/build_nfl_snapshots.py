"""Rebuild the committed NFL asset snapshots (schedules/weekly/rosters/injuries)
from nflverse. Ranges are dynamic through the CURRENT season so a re-run pulls
in the live season's completed games (schedules is what the cover/total training
frame reads for labels + spread_line/total_line).

Weekly player_stats for the in-progress (and sometimes the just-finished) season
lag behind the other feeds on nflverse, so weekly is loaded season-by-season and
any season whose file isn't published yet is skipped -- a later re-run picks it
up automatically instead of 404-ing the whole build.
"""
import pathlib
from datetime import datetime, timezone

import pandas as pd

from sportsmodel.nfl import data
from sportsmodel.nfl.injuries_nflverse import nfl_season

OUT = pathlib.Path("assets/nfl")
OUT.mkdir(parents=True, exist_ok=True)

CURRENT = nfl_season(datetime.now(timezone.utc))
SCHED = list(range(2002, CURRENT + 1))
OTHER = list(range(2015, CURRENT + 1))


def _load_weekly_available(seasons: list[int]) -> pd.DataFrame:
    """Load weekly data season-by-season, skipping seasons nflverse hasn't
    published yet (they 404). Self-heals on a later run."""
    frames, loaded = [], []
    for s in seasons:
        try:
            frames.append(data.load_weekly([s]))
            loaded.append(s)
        except Exception as exc:  # noqa: BLE001 -- an unpublished season 404s; skip it
            print(f"  weekly {s}: skipped ({type(exc).__name__})")
    print("  weekly seasons loaded:", loaded)
    return pd.concat(frames, ignore_index=True)


data.load_schedules(SCHED).to_parquet(OUT / "schedules.parquet", index=False)
_load_weekly_available(OTHER).to_parquet(OUT / "weekly.parquet", index=False)
data.load_rosters(OTHER).to_parquet(OUT / "rosters.parquet", index=False)
data.load_injuries(OTHER).to_parquet(OUT / "injuries.parquet", index=False)
print(f"wrote (through {CURRENT}):", sorted(p.name for p in OUT.glob("*.parquet")))
