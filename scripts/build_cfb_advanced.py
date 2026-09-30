"""Pull CFBD advanced per-game team stats -> assets/cfb/advanced_games.parquet.

Reads CFBD_API_KEY from the environment (never hardcoded, never logged). Source is
`/stats/game/advanced?year=Y&seasonType=regular`, which returns one object per
team-game: {gameId, season, week, team, opponent, offense{...}, defense{...}} where each
unit carries plays / ppa / successRate / explosiveness plus `passingPlays` and
`rushingPlays` sub-objects ({ppa, totalPPA, successRate, explosiveness}). Missing
sub-objects become NaN (never 0). Pass/rush play counts are only filled when CFBD supplies
a `plays` field on the split (it does not always); downstream derives shares elsewhere.

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
)
_SPLIT_PLAYS = (("pass_plays", ("passingPlays", "plays")),
                ("rush_plays", ("rushingPlays", "plays")))
_UNIT_FIELDS = tuple(s for s, _ in _METRICS)
_ID_COLS = ("season", "week", "game_id", "team", "opponent")


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


def _unit_row(prefix: str, unit) -> dict:
    row = {f"{prefix}_{s}": _dig(unit, p) for s, p in _METRICS}
    if prefix == "off":
        row.update({f"{prefix}_{s}": _dig(unit, p) for s, p in _SPLIT_PLAYS})
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
               "game_id": int(g["gameId"]), "team": team, "opponent": opp}
        row.update(_unit_row("off", g.get("offense")))
        row.update(_unit_row("def", g.get("defense")))
        rows.append(row)
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in COLUMNS:
        if c in ("team", "opponent"):
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
    out = out.drop_duplicates(subset=["season", "game_id", "team"], keep="last")
    return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


def coverage(adv: pd.DataFrame, sched: pd.DataFrame, fbs: set[str] | None = None) -> pd.DataFrame:
    """Per season: FBS-vs-FBS schedule games and the share with both teams' rows present."""
    fbs = load_fbs_ids() if fbs is None else fbs
    s = sched[sched["home_team"].astype(str).isin(fbs) & sched["away_team"].astype(str).isin(fbs)]
    sides = adv.groupby(["season", "game_id"])["team"].nunique()
    both = {k for k, v in sides.items() if v >= 2}
    s = s.assign(both=[(int(a), int(b)) in both for a, b in zip(s["season"], s["game_pk"])])
    g = s.groupby("season").agg(fbs_games=("both", "size"), both_sides=("both", "mean"))
    return g[g.index.isin(adv["season"].unique())]


def fetch_year(year: int, key: str) -> list:
    """CFBD advanced game stats for a season, retrying transient 5xx / network errors."""
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
    ap.add_argument("--seasons", type=int, nargs="+",
                    default=list(range(2015, dt.date.today().year + 1)))
    ap.add_argument("--merge", action="store_true",
                    help="refresh only the given seasons, keeping the rest of the parquet")
    args = ap.parse_args()

    key = os.environ.get("CFBD_API_KEY")
    if not key:
        sys.exit("CFBD_API_KEY not set in environment (add it as a secret / export it).")

    existing = pd.read_parquet(_OUT) if args.merge and _OUT.exists() else None
    frames, dropped, replace = [], 0, []
    for y in args.seasons:
        payload = fetch_year(y, key)
        df = parse_advanced(payload)
        dropped += df.attrs["dropped"]
        print(f"{y}: {len(payload)} team-games, {len(df)} kept", flush=True)
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
