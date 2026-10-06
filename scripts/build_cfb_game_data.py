"""Pull the per-game CFBD datasets behind cfb-ratings-v3 -> committed parquet under assets/cfb/.

Datasets (CFBD endpoint -> asset):
  games    /games              -> cfbd_games.parquet     game meta: week, season_type, venue, neutral, pre-game Elo
  havoc    /stats/game/havoc   -> havoc_games.parquet    per team-game havoc rates
  drives   /drives             -> drive_games.parquet    per team-game drive efficiency (needs `games`)
  weather  /games/weather      -> weather_games.parquet  per game weather (historical + forecast)
  talent   /talent             -> talent.parquet         yearly 247 talent composite
  ratings  /ratings/fpi + srs  -> prior_ratings.parquet  yearly FPI/SRS (residual check; next-season use only)
  venues   /venues             -> venues.parquet         static lat/lon/timezone/dome

Reads CFBD_API_KEY from the environment through CfbdClient (never logged). Every dataset is
always merged season-by-season: a season whose pull returns no rows keeps its committed rows. The
call counter is printed at the end.

Usage:
  CFBD_API_KEY=... uv run python scripts/build_cfb_game_data.py --probe          # print real response shapes
  CFBD_API_KEY=... uv run python scripts/build_cfb_game_data.py --seasons 2015 2016 ... 2026
  CFBD_API_KEY=... uv run python scripts/build_cfb_game_data.py --seasons 2026   # weekly refresh (other seasons kept)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sportsmodel.cfb import cfbd_games as cg  # noqa: E402
from sportsmodel.cfb.cfbd import CfbdClient  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
FILES = {"games": "cfbd_games.parquet", "havoc": "havoc_games.parquet", "drives": "drive_games.parquet",
         "weather": "weather_games.parquet", "talent": "talent.parquet",
         "ratings": "prior_ratings.parquet", "venues": "venues.parquet"}
KEYS = {"games": ["season", "game_id"], "havoc": ["season", "game_id", "team"],
        "drives": ["season", "game_id", "team"], "weather": ["season", "game_id"],
        "talent": ["season", "team"], "ratings": ["season", "team"], "venues": ["venue_id"]}
ORDER = ["games", "havoc", "drives", "weather", "talent", "ratings", "venues"]
SEASON_TYPES = ("regular", "postseason")


def merge_asset(existing: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Replace the seasons present in `new`, keep every other committed season (a season whose
    pull came back empty is therefore kept). Season-less assets (venues) are replaced whole
    when `new` has rows."""
    if new.empty:
        return existing.copy() if existing is not None else new.copy()
    if existing is None or existing.empty:
        out = new.copy()
    elif "season" in new.columns:
        out = pd.concat([existing[~existing["season"].isin(new["season"].unique())], new],
                        ignore_index=True)
    else:
        out = new.copy()
    return out.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)


def describe_shape(obj, depth: int = 3):
    """Key/type skeleton of a decoded JSON value (values are never shown) -- the --probe output."""
    if isinstance(obj, dict) and depth > 0:
        return {k: describe_shape(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [describe_shape(obj[0], depth)] if obj else []
    return type(obj).__name__


def _pull(client, dataset: str, year: int, meta: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    if dataset == "games":
        parts = [cg.parse_games_meta(client.get("/games", {"year": year, "seasonType": st}))
                 for st in SEASON_TYPES]
    elif dataset == "havoc":
        parts = [cg.parse_havoc_games(client.get("/stats/game/havoc", {"year": year, "seasonType": st}))
                 for st in SEASON_TYPES]
    elif dataset == "drives":
        if meta is None or meta.empty:
            raise SystemExit("drives needs cfbd_games.parquet (run the `games` dataset first)")
        parts = [cg.parse_drive_games(client.get("/drives", {"year": year, "seasonType": st}), meta)
                 for st in SEASON_TYPES]
    elif dataset == "weather":
        parts = [cg.parse_weather_games(client.get("/games/weather", {"year": year, "seasonType": st}))
                 for st in SEASON_TYPES]
    elif dataset == "talent":
        parts = [cg.parse_talent(client.get("/talent", {"year": year}))]
    elif dataset == "ratings":
        parts = [cg.parse_prior_ratings(client.get("/ratings/fpi", {"year": year}),
                                        client.get("/ratings/srs", {"year": year}), year)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["dropped"] = sum(p.attrs.get("dropped", 0) for p in parts)
    return df


def run(client, datasets: list[str], seasons: list[int], out_dir: Path) -> dict:
    """Pull `datasets` for `seasons`, merge into the committed parquet under out_dir
    (seasons not pulled are kept), and return {dataset: rows in the written file}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = {}
    games_path = out_dir / FILES["games"]
    meta = pd.read_parquet(games_path) if games_path.exists() else None
    for ds in [d for d in ORDER if d in datasets]:
        path = out_dir / FILES[ds]
        existing = pd.read_parquet(path) if path.exists() else None
        if ds == "venues":
            new = cg.parse_venues(client.get("/venues"))
        else:
            frames = [_pull(client, ds, y, meta) for y in seasons]
            new = pd.concat(frames, ignore_index=True)
            print(f"{ds}: {len(new)} rows pulled for {seasons[0]}-{seasons[-1]} "
                  f"({sum(f.attrs['dropped'] for f in frames)} dropped: unmapped/FCS/unknown game)",
                  flush=True)
            for y in seasons:
                if not (new["season"] == y).any():
                    kept = existing is not None and (existing["season"] == y).any()
                    print(f"::warning::build-cfb-game-data: {ds} {y} returned no rows"
                          + ("; KEEPING the committed rows" if kept else ""), flush=True)
        out = merge_asset(existing, new, KEYS[ds])
        out.to_parquet(path)
        wrote[ds] = len(out)
        if ds == "games":
            meta = out
        print(f"wrote {len(out)} rows -> {path.name}", flush=True)
    print(client.summary(), flush=True)
    return wrote


def probe(client, year: int = 2023) -> None:
    """One real call per endpoint: print the key/type skeleton of the first record."""
    import json
    calls = {"/games": {"year": year, "seasonType": "regular", "week": 3},
             "/stats/game/havoc": {"year": year, "week": 3},
             "/drives": {"year": year, "week": 3},
             "/games/weather": {"year": year, "week": 3},
             "/talent": {"year": year}, "/venues": None,
             "/ratings/fpi": {"year": year}, "/ratings/srs": {"year": year},
             "/stats/game/advanced": {"year": year, "week": 3}}
    for path, params in calls.items():
        data = client.get(path, params)
        print(f"== {path} {params or ''}: {len(data)} records")
        print(json.dumps(describe_shape(data[:1]), indent=1)[:2500])
    print(client.summary())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[dt.date.today().year])
    ap.add_argument("--datasets", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--probe", action="store_true", help="print one record's key/type skeleton per endpoint")
    args = ap.parse_args(argv)
    client = CfbdClient.from_env()
    if args.probe:
        probe(client)
        return
    run(client, args.datasets, sorted(args.seasons), ASSETS)


if __name__ == "__main__":
    main()
