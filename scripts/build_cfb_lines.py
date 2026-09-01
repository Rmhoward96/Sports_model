"""Pull historical CFB betting lines from collegefootballdata.com -> assets/cfb/lines.parquet.

Reads CFBD_API_KEY from the environment (never hardcoded, never logged). This is a
ONE-TIME historical pull for the P2 game-line backtest; live P3 uses the Odds API
(americanfootball_ncaaf), not CFBD.

Per game we take the MEDIAN spread + over/under across the providers CFBD returns.
CFBD's `spread` is the home spread in book convention (home favored => negative); we
store `market_spread` in home-margin convention (home favored => positive) = -spread,
matching the nfl gameline path. Games where either team can't be matched to an FBS
ESPN id (FCS opponents, unrecognized names) are dropped.

Usage: CFBD_API_KEY=... uv run python scripts/build_cfb_lines.py --seasons 2015 ... 2024
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sportsmodel.cfb.teams import cfbd_to_espn  # noqa: E402

_API = "https://api.collegefootballdata.com/lines"
_OUT = Path(__file__).resolve().parents[1] / "assets" / "cfb" / "lines.parquet"


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def fetch_year(year: int, key: str) -> list:
    """CFBD lines for a season, retrying transient 5xx / network errors."""
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = httpx.get(_API, params={"year": year, "seasonType": "regular"},
                          headers={"Authorization": f"Bearer {key}"}, timeout=60)
            r.raise_for_status()
            return r.json()
        except httpx.HTTPStatusError as e:
            last = e
            if e.response.status_code >= 500:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except httpx.RequestError as e:
            last = e
            time.sleep(3 * (attempt + 1))
    raise last  # type: ignore[misc]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", default=list(range(2015, 2025)))
    args = ap.parse_args()

    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit("CFBD_API_KEY not set in environment (add it as a secret / export it).")

    rows, dropped = [], 0
    for y in args.seasons:
        games = fetch_year(y, key)
        for g in games:
            home = cfbd_to_espn(g.get("homeTeam", "") or "")
            away = cfbd_to_espn(g.get("awayTeam", "") or "")
            if not home or not away:
                dropped += 1
                continue
            lines = g.get("lines") or []
            spread = _median([ln.get("spread") for ln in lines])
            total = _median([ln.get("overUnder") for ln in lines])
            if spread is None and total is None:
                dropped += 1
                continue
            rows.append({
                "season": int(g["season"]), "week": int(g["week"]),
                "home_team": home, "away_team": away,
                "market_spread": (-spread) if spread is not None else None,  # -> home-margin
                "market_total": total,
            })
        print(f"{y}: {len(games)} games", flush=True)

    df = pd.DataFrame(rows)
    if len(df):  # CFBD can return a game more than once; keep one row per matchup
        df = df.drop_duplicates(subset=["season", "week", "home_team", "away_team"], keep="first")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT)
    print(f"wrote {len(df)} line rows ({dropped} dropped: unmatched/FCS/no-line) -> "
          f"{_OUT.relative_to(Path(__file__).resolve().parents[1])}")
    if len(df):
        print(f"seasons: {sorted(df.season.unique())} | "
              f"spread coverage {df.market_spread.notna().mean():.0%}, "
              f"total coverage {df.market_total.notna().mean():.0%}")


if __name__ == "__main__":
    main()
