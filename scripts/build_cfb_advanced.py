"""Pull CFBD advanced per-game team stats -> assets/cfb/advanced_games.parquet.

Reads CFBD_API_KEY from the environment (never hardcoded, never logged). Source is
`/stats/game/advanced?year=Y&seasonType=regular|postseason` (both pulled by default; each
row is stamped with `season_type`, so consumers that want the regular season only filter on
it), which returns one object per
team-game: {gameId, season, week, team, opponent, offense{...}, defense{...}} where each
unit carries plays / ppa / successRate / explosiveness plus `passingPlays` and
`rushingPlays` sub-objects ({ppa, totalPPA, successRate, explosiveness}). Missing
sub-objects become NaN (never 0). CFBD's splits carry no play count, so pass/rush plays =
`plays` when present, else totalPPA / ppa when that ratio is a whole non-negative count
(ppa is full precision); otherwise NaN.

`team`/`opponent` are mapped to ESPN ids via `cfbd_to_espn`; rows where either side does
not map to an FBS id (FCS opponents, unrecognized names) are dropped and counted
(`df.attrs["dropped"]`).

Usage:
  CFBD_API_KEY=... uv run python scripts/build_cfb_advanced.py --seasons 2015 ... 2025
  CFBD_API_KEY=... uv run python scripts/build_cfb_advanced.py --seasons 2026 --merge
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sportsmodel.cfb.teams import cfbd_to_espn, load_fbs_ids  # noqa: E402

_API = "https://api.collegefootballdata.com/stats/game/advanced"
_OUT = ROOT / "assets" / "cfb" / "advanced_games.parquet"
_SCHEDULES = ROOT / "assets" / "cfb" / "schedules.parquet"

# (output suffix, CFBD path inside a unit object)
_METRICS = (
    ("plays", ("plays",)),
    ("ppa", ("ppa",)),
    ("success", ("successRate",)),
    ("explosiveness", ("explosiveness",)),
    ("pass_ppa", ("passingPlays", "ppa")),
    ("pass_success", ("passingPlays", "successRate")),
    ("pass_explosiveness", ("passingPlays", "explosiveness")),
    ("rush_ppa", ("rushingPlays", "ppa")),
    ("rush_success", ("rushingPlays", "successRate")),
    ("rush_explosiveness", ("rushingPlays", "explosiveness")),
    # down/distance splits + run-game shape (residual-check features for cfb-ratings-v3)
    ("std_down_success", ("standardDowns", "successRate")),
    ("pass_down_success", ("passingDowns", "successRate")),
    ("std_down_ppa", ("standardDowns", "ppa")),
    ("pass_down_ppa", ("passingDowns", "ppa")),
    ("line_yards", ("lineYards",)),
    ("stuff_rate", ("stuffRate",)),
    ("power_success", ("powerSuccess",)),
)
_SPLIT_PLAYS = (("pass_plays", ("passingPlays", "plays")),
                ("rush_plays", ("rushingPlays", "plays")))
_UNIT_FIELDS = tuple(s for s, _ in _METRICS)
_ID_COLS = ("season", "week", "season_type", "game_id", "team", "opponent")


def _unit_cols(prefix: str) -> list[str]:
    cols = [f"{prefix}_{s}" for s in _UNIT_FIELDS]
    cols += [f"{prefix}_{s}" for s, _ in _SPLIT_PLAYS] if prefix == "off" else []
    return cols


COLUMNS = list(_ID_COLS) + _unit_cols("off") + _unit_cols("def")


def _dig(obj, path):
    """Nested numeric lookup; missing / non-dict / None / non-numeric -> NaN."""
    for k in path:
        if not isinstance(obj, dict):
            return float("nan")
        obj = obj.get(k)
    if isinstance(obj, bool) or not isinstance(obj, (int, float)):
        return float("nan")
    return float(obj)


def _split_plays(unit, path) -> float:
    """Play count on a split: its `plays`, else totalPPA / ppa if that is a whole count."""
    n = _dig(unit, path)
    if not math.isnan(n):
        return n
    split = path[:-1]
    total, ppa = _dig(unit, (*split, "totalPPA")), _dig(unit, (*split, "ppa"))
    if math.isnan(total) or math.isnan(ppa) or abs(ppa) < 1e-9:
        return float("nan")
    n = total / ppa
    return float(round(n)) if n >= 0 and abs(n - round(n)) < 0.01 else float("nan")


def _unit_row(prefix: str, unit) -> dict:
    row = {f"{prefix}_{s}": _dig(unit, p) for s, p in _METRICS}
    if prefix == "off":
        row.update({f"{prefix}_{s}": _split_plays(unit, p) for s, p in _SPLIT_PLAYS})
    return row


def parse_advanced(payload: list[dict]) -> pd.DataFrame:
    """CFBD /stats/game/advanced payload -> one row per team-game (PURE).

    Rows whose team or opponent does not map to an FBS ESPN id are dropped; the count is
    on `df.attrs["dropped"]`.
    """
    rows, dropped = [], 0
    for g in payload:
        team = cfbd_to_espn(g.get("team", "") or "")
        opp = cfbd_to_espn(g.get("opponent", "") or "")
        if not team or not opp:
            dropped += 1
            continue
        row = {"season": int(g["season"]), "week": int(g["week"]),
               "season_type": str(g.get("seasonType") or "regular").lower(),
               "game_id": int(g["gameId"]), "team": team, "opponent": opp}
        row.update(_unit_row("off", g.get("offense")))
        row.update(_unit_row("def", g.get("defense")))
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS:
        if c in ("team", "opponent", "season_type"):
            df[c] = df[c].astype(str)
        elif c in ("season", "week", "game_id"):
            df[c] = df[c].astype("int64")
        else:
            df[c] = df[c].astype("float64")
    df.attrs["dropped"] = dropped
    return df


def merge_frames(existing: pd.DataFrame | None, new: pd.DataFrame, seasons) -> pd.DataFrame:
    """Replace only `seasons` in `existing` with `new`; keep one row per team-game."""
    if existing is None or not len(existing):
        out = new.copy()
    else:
        kept = existing[~existing["season"].isin(list(seasons))]
        out = pd.concat([kept, new], ignore_index=True) if len(kept) else new.copy()
    if "season_type" in out.columns:           # pre-v3 parquets carry no season_type: all regular
        out["season_type"] = out["season_type"].fillna("regular")
    out = out.drop_duplicates(subset=["season", "game_id", "team"], keep="last")
    out = out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
    return out.reindex(columns=COLUMNS)


def keep_postseason(existing: pd.DataFrame | None, new: pd.DataFrame, year: int) -> pd.DataFrame:
    """A season refresh whose postseason pull came back empty (mid-season, or an API
    blip after bowls) must not drop the postseason rows already committed."""
    if existing is None or "season_type" not in existing.columns:
        return new
    if (new["season_type"] == "postseason").any():
        return new
    old = existing[(existing["season"] == year) & (existing["season_type"] == "postseason")]
    return pd.concat([new, old], ignore_index=True) if len(old) else new


def coverage(adv: pd.DataFrame, sched: pd.DataFrame, fbs: set[str] | None = None) -> pd.DataFrame:
    """Per season: FBS-vs-FBS schedule games and the share with both teams' rows present."""
    fbs = load_fbs_ids() if fbs is None else fbs
    s = sched[sched["home_team"].astype(str).isin(fbs) & sched["away_team"].astype(str).isin(fbs)]
    sides = adv.groupby(["season", "game_id"])["team"].nunique()
    both = {k for k, v in sides.items() if v >= 2}
    s = s.assign(both=[(int(a), int(b)) in both for a, b in zip(s["season"], s["game_pk"])])
    g = s.groupby("season").agg(fbs_games=("both", "size"), both_sides=("both", "mean"))
    return g[g.index.isin(adv["season"].unique())]


def fetch_year(year: int, key: str, season_type: str = "regular") -> list:
    """CFBD advanced game stats for a season + season type, retrying transient 5xx / network errors."""
    last: Exception | None = None
    for attempt in range(4):
        try:
            r = httpx.get(_API, params={"year": year, "seasonType": season_type},
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
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=list(range(2015, dt.date.today().year + 1)))
    ap.add_argument("--season-types", nargs="+", default=["regular", "postseason"],
                    choices=["regular", "postseason"], help="CFBD seasonType values to pull")
    ap.add_argument("--merge", action="store_true",
                    help="refresh only the given seasons, keeping the rest of the parquet")
    args = ap.parse_args()

    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit("CFBD_API_KEY not set in environment (add it as a secret / export it).")

    existing = pd.read_parquet(_OUT) if args.merge and _OUT.exists() else None
    frames, dropped, replace = [], 0, []
    for y in args.seasons:
        parts = [parse_advanced(fetch_year(y, key, st)) for st in args.season_types]
        dropped += sum(p.attrs["dropped"] for p in parts)
        df = pd.concat(parts, ignore_index=True)
        if (len(df) and "regular" in args.season_types and existing is not None
                and (existing["season"] == y).any() and not (df["season_type"] == "regular").any()):
            df = df.iloc[0:0]      # postseason-only pull: never replace a season's committed regular rows
        if len(df) and "postseason" in args.season_types:
            df = keep_postseason(existing, df, y)
        print(f"{y}: {len(df)} team-games kept ({', '.join(args.season_types)}), "
              f"{df['off_pass_plays'].notna().mean() if len(df) else 0:.0%} with pass/rush counts",
              flush=True)
        if df.empty and existing is not None and (existing["season"] == y).any():
            # an empty pull must not wipe a season that is already committed
            print(f"::warning::build-cfb-advanced: {y} returned no usable rows; "
                  f"KEEPING the {int((existing['season'] == y).sum())} existing {y} rows",
                  flush=True)
            continue
        frames.append(df)
        replace.append(y)
    new = pd.concat(frames, ignore_index=True) if frames else parse_advanced([])

    if existing is not None:
        out = merge_frames(existing, new, replace)
    else:
        out = merge_frames(None, new, args.seasons)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(_OUT)
    print(f"wrote {len(out)} team-game rows ({dropped} dropped: unmapped/FCS) -> "
          f"{_OUT.relative_to(ROOT)}")
    if len(out) and _SCHEDULES.exists():
        print("coverage by season (FBS-vs-FBS schedule games with both sides present):")
        print(coverage(out, pd.read_parquet(_SCHEDULES)).round(3).to_string())


if __name__ == "__main__":
    main()
