import pathlib
from sportsmodel.nfl import data

OUT = pathlib.Path("assets/nfl")
OUT.mkdir(parents=True, exist_ok=True)
SCHED = list(range(2002, 2026))
OTHER = list(range(2015, 2026))
# nflverse has not yet published player_stats_2025.parquet as of this run
# (verified 404 on the release asset; player_stats_2024.parquet resolves fine).
# Rosters/injuries for 2025 ARE published, so only weekly's range is trimmed.
WEEKLY_SEASONS = [y for y in OTHER if y != 2025]

data.load_schedules(SCHED).to_parquet(OUT / "schedules.parquet", index=False)
data.load_weekly(WEEKLY_SEASONS).to_parquet(OUT / "weekly.parquet", index=False)
data.load_rosters(OTHER).to_parquet(OUT / "rosters.parquet", index=False)
data.load_injuries(OTHER).to_parquet(OUT / "injuries.parquet", index=False)
print("wrote", sorted(p.name for p in OUT.glob("*.parquet")))
