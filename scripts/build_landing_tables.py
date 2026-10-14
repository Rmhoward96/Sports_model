"""Build the landing tables the early-week monitor uses to move a Pinnacle line to a book's line (profit step 3).

Reads only committed history -- no network, no database:
  * CFB: assets/cfb/lines.parquet closing `market_spread` / `market_total` (2015-2025), joined on
    (season, week, home_team, away_team) to the final scores in assets/cfb/schedules.parquet.
  * NFL: assets/nfl/schedules.parquet `spread_line` / `total_line` and scores (2010-2025).
Writes assets/early/landing_{sport}_{market}.json for cfb spread, cfb total and nfl spread (see
sportsmodel.early.landing for the format). Re-running with unchanged inputs rewrites identical files.

    uv run python scripts/build_landing_tables.py [--out assets/early]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.early import landing  # noqa: E402

CFB_SEASONS = (2015, 2025)
NFL_SEASONS = (2010, 2025)
CFB_KEYS = ["season", "week", "home_team", "away_team"]


def cfb_pairs(lines: pd.DataFrame, schedules: pd.DataFrame) -> dict[str, list[tuple[float, int]]]:
    """{'spread': [(expected home margin, home margin)], 'total': [(closing total, total points)]} for CFB games
    with a closing line and a final score. `market_spread` is positive when the home team is favoured."""
    lo, hi = CFB_SEASONS
    merged = lines[(lines.season >= lo) & (lines.season <= hi)].merge(
        schedules[CFB_KEYS + ["home_score", "away_score"]], on=CFB_KEYS, how="inner")
    merged = merged.dropna(subset=["home_score", "away_score"])
    margin = (merged.home_score - merged.away_score).astype(int)
    total = (merged.home_score + merged.away_score).astype(int)
    spread_ok = merged.market_spread.notna()
    total_ok = merged.market_total.notna()
    return {
        "spread": list(zip(merged.market_spread[spread_ok].astype(float), margin[spread_ok])),
        "total": list(zip(merged.market_total[total_ok].astype(float), total[total_ok])),
    }


def nfl_pairs(schedules: pd.DataFrame) -> list[tuple[float, int]]:
    """[(expected home margin, home margin)] for NFL games 2010-2025 (`spread_line` is positive when home is favoured)."""
    lo, hi = NFL_SEASONS
    games = schedules[(schedules.season >= lo) & (schedules.season <= hi)]
    games = games.dropna(subset=["spread_line", "home_score", "away_score"])
    margin = (games.home_score - games.away_score).astype(int)
    return list(zip(games.spread_line.astype(float), margin))


def build_all(assets: Path) -> dict[tuple[str, str], dict]:
    cfb = cfb_pairs(pd.read_parquet(assets / "cfb" / "lines.parquet"), pd.read_parquet(assets / "cfb" / "schedules.parquet"))
    nfl = nfl_pairs(pd.read_parquet(assets / "nfl" / "schedules.parquet"))
    cfb_src = "assets/cfb/lines.parquet joined to assets/cfb/schedules.parquet"
    return {
        ("cfb", "spread"): landing.build_table(cfb["spread"], sport="cfb", market="spread", seasons=CFB_SEASONS, source=cfb_src),
        ("cfb", "total"): landing.build_table(cfb["total"], sport="cfb", market="total", seasons=CFB_SEASONS, source=cfb_src),
        ("nfl", "spread"): landing.build_table(nfl, sport="nfl", market="spread", seasons=NFL_SEASONS,
                                               source="assets/nfl/schedules.parquet"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assets", default=str(ROOT / "assets"), help="folder holding cfb/ and nfl/ parquet inputs")
    ap.add_argument("--out", default=str(ROOT / "assets" / "early"), help="folder to write the JSON tables to")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for (sport, market), table in build_all(Path(args.assets)).items():
        path = landing.table_path(sport, market, out)
        path.write_text(json.dumps(table, sort_keys=True, separators=(",", ":")) + "\n")
        kept = len(table["entries"])
        print(f"[landing] {sport} {market}: {table['games']} games, {kept} grid values -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
