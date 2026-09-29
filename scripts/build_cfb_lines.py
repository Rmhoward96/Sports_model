"""Pull historical CFB betting lines from collegefootballdata.com -> assets/cfb/lines.parquet.

Reads CFBD_API_KEY from the environment (never hardcoded, never logged). This is a
ONE-TIME historical pull for the P2 game-line backtest; live P3 uses the Odds API
(americanfootball_ncaaf), not CFBD.

Per game we take the MEDIAN across the providers CFBD returns of: closing spread and
over/under (`market_spread`, `market_total`), opening spread and over/under
(`spread_open`, `total_open`) and the closing moneylines (`ml_home`, `ml_away`, American,
rounded to int). CFBD's `spread`/`spreadOpen` are the home spread in book convention
(home favored => negative); we store home-margin convention (home favored => positive)
= -spread, matching the nfl gameline path. Games where either team can't be matched to
an FBS ESPN id (FCS opponents, unrecognized names) or that carry no price at all are
dropped. `n_providers` is the number of provider line objects on the game.

Usage: CFBD_API_KEY=... uv run python scripts/build_cfb_lines.py --seasons 2015 ... 2025
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
_PRICE_FIELDS = ("market_spread", "market_total", "spread_open", "total_open",
                 "ml_home", "ml_away")
_OUT = Path(__file__).resolve().parents[1] / "assets" / "cfb" / "lines.parquet"


def _median(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None


def _neg(v):
    return None if v is None else 0.0 - v


def _median_int(vals):
    m = _median(vals)
    return None if m is None else int(round(m))


def parse_game_lines(game: dict) -> dict | None:
    """One CFBD /lines game payload -> flat row of cross-provider medians (PURE).

    Spreads are converted to home-margin convention (= -CFBD value). Returns None when
    either team doesn't map to an FBS ESPN id or no price field is present at all.
    """
    home = cfbd_to_espn(game.get("homeTeam", "") or "")
    away = cfbd_to_espn(game.get("awayTeam", "") or "")
    if not home or not away:
        return None
    lines = game.get("lines") or []
    row = {
        "season": int(game["season"]), "week": int(game["week"]),
        "home_team": home, "away_team": away,
        "market_spread": _neg(_median([ln.get("spread") for ln in lines])),
        "market_total": _median([ln.get("overUnder") for ln in lines]),
        "spread_open": _neg(_median([ln.get("spreadOpen") for ln in lines])),
        "total_open": _median([ln.get("overUnderOpen") for ln in lines]),
        "ml_home": _median_int([ln.get("homeMoneyline") for ln in lines]),
        "ml_away": _median_int([ln.get("awayMoneyline") for ln in lines]),
        "n_providers": len(lines),
    }
    if all(row[k] is None for k in _PRICE_FIELDS):
        return None
    return row


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Per season, share of rows carrying each price field (moneyline needs both sides)."""
    g = pd.DataFrame({
        "season": df["season"],
        "market_spread": df["market_spread"].notna(),
        "spread_open": df["spread_open"].notna(),
        "market_total": df["market_total"].notna(),
        "total_open": df["total_open"].notna(),
        "moneyline": df["ml_home"].notna() & df["ml_away"].notna(),
    })
    return g.groupby("season").mean()


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
    ap.add_argument("--seasons", type=int, nargs="+", default=list(range(2015, 2026)))
    args = ap.parse_args()

    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit("CFBD_API_KEY not set in environment (add it as a secret / export it).")

    rows, dropped = [], 0
    for y in args.seasons:
        games = fetch_year(y, key)
        for g in games:
            row = parse_game_lines(g)
            if row is None:
                dropped += 1
                continue
            rows.append(row)
        print(f"{y}: {len(games)} games", flush=True)

    df = pd.DataFrame(rows)
    if len(df):  # stable dtypes even when a whole column is null
        for c in ("market_spread", "market_total", "spread_open", "total_open"):
            df[c] = pd.to_numeric(df[c]).astype("float64")
        for c in ("ml_home", "ml_away"):
            df[c] = pd.to_numeric(df[c]).astype("Int64")
    if len(df):  # CFBD can return a game more than once; keep one row per matchup
        df = df.drop_duplicates(subset=["season", "week", "home_team", "away_team"], keep="first")
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(_OUT)
    print(f"wrote {len(df)} line rows ({dropped} dropped: unmatched/FCS/no-line) -> "
          f"{_OUT.relative_to(Path(__file__).resolve().parents[1])}")
    if len(df):
        print("coverage by season (share of games with each field):")
        print(coverage(df).round(3).to_string())


if __name__ == "__main__":
    main()
