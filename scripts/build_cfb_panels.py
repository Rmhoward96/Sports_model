"""Weekly CFB panel tables from the committed CFBD assets -> Supabase (db/migration_site_panels.sql):

* `cfb_team_insights`  -- season-to-date havoc / turnover margin / explosiveness with national ranks
  (cfb.panels.team_insights) from havoc_games, advanced_games and team_game_stats.
* `cfb_quarter_shares` -- per-team Q1-Q4 scoring shares, last two completed seasons + this one, shrunk to the
  league (cfb.panels.quarter_shares) from cfbd_games' line scores.

Runs after build-cfb-advanced.yml has refreshed those assets (Mondays). Pure assets in, upserts out: no CFBD
call here. `--season` defaults to the latest season with regular-season havoc rows. `--dry-run` prints counts
and a sample and writes nothing.

Usage:
    uv run python scripts/build_cfb_panels.py [--season 2026] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sportsmodel import config, db  # noqa: E402
from sportsmodel.cfb import panels  # noqa: E402
from sportsmodel.cfb.teams import load_fbs_ids  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"


def warn(msg: str) -> None:
    print(f"::warning::cfb-panels: {msg}", flush=True)


def _read(assets: Path, name: str, required: bool) -> pd.DataFrame | None:
    p = assets / name
    if p.exists():
        return pd.read_parquet(p)
    if required:
        sys.exit(f"{p} is missing (run scripts/build_cfb_game_data.py / build_cfb_advanced.py first)")
    warn(f"{name} missing; its columns stay NULL")
    return None


def build(assets: Path | None = None, season: int | None = None, fbs: set[str] | None = None
          ) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """(season, insights frame, quarter-share frame) from the assets under `assets` (default: ASSETS)."""
    assets = assets or ASSETS
    havoc = _read(assets, "havoc_games.parquet", True)
    advanced = _read(assets, "advanced_games.parquet", True)
    games = _read(assets, "cfbd_games.parquet", True)
    team_stats = _read(assets, "team_game_stats.parquet", False)
    if season is None:
        reg = havoc[havoc["season_type"].fillna("regular") == "regular"]
        if reg.empty:
            sys.exit("havoc_games.parquet has no regular-season rows")
        season = int(reg["season"].max())
    ins = panels.team_insights(havoc, advanced, team_stats, season, fbs=fbs)
    shares = panels.quarter_shares(games, season, fbs=fbs)
    if "home_q1" not in games.columns or games["home_q1"].notna().sum() == 0:
        warn("cfbd_games.parquet has no line scores (run the games backfill); no quarter shares")
    return season, ins, shares


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="compute and print; no DB writes")
    args = ap.parse_args(argv)
    if not args.dry_run and not config.DATABASE_URL:
        sys.exit("DATABASE_URL is not set (use --dry-run to compute without writing)")
    if not args.dry_run and (gone := db.missing_site_panel_tables(["cfb_team_insights", "cfb_quarter_shares"])):
        # until the user runs db/migration_site_panels.sql: skip cleanly (green job)
        for t in gone:
            print(f"::warning::cfb-panels: table {t} missing \u2014 run db/migration_site_panels.sql", flush=True)
        return
    season, ins, shares = build(season=args.season, fbs=set(load_fbs_ids()))
    print(f"cfb panels season {season}: team_insights={len(ins)} (ranked {int(ins['n_ranked'].max()) if len(ins) else 0}, "
          f"through week {int(ins['through_week'].max()) if len(ins) else '-'}), quarter_shares={len(shares)}", flush=True)
    if args.dry_run:
        print(ins.head(3).to_string(index=False))
        print(shares.head(3).to_string(index=False))
        return
    print(f"upserted: cfb_team_insights={db.upsert_cfb_team_insights(panels.to_rows(ins))} "
          f"cfb_quarter_shares={db.upsert_cfb_quarter_shares(panels.to_rows(shares))}", flush=True)


if __name__ == "__main__":
    main()
