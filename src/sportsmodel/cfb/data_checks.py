"""Sanity checks of the freshly backfilled CFBD game-data assets (PURE; returns problem strings).

The parsers were written from the CFBD OpenAPI document, not from live responses, so the backfill
step runs these checks to catch what a schema cannot say: units (temperature / wind / precipitation),
which side of the havoc object is "suffered" vs "created", drive points, and coverage. An empty list
means the assets look right; any problem string must be resolved (fix the parser / mapping and
re-pull) before fitting anything on them.
"""
from __future__ import annotations

import pandas as pd

TEMP_F = (40.0, 80.0)            # season mean of outdoor game temperatures, degrees F
WIND_MPH = (2.0, 16.0)           # season mean wind
PRECIP_MAX = 30.0                # a single reading above this is not inches (or mm of a real game)
HAVOC_RATE = (0.04, 0.30)
HAVOC_PAIR_MAX_GAP = 0.02        # mean |def_havoc[A] - off_havoc[B]|: the two sides of one game agree
POINTS_PER_DRIVE = (1.0, 3.5)
START_YD = (20.0, 40.0)
MIN_COVERAGE = 0.80              # share of a season's FBS-vs-FBS games a per-game asset must cover
COVERAGE_FROM_SEASON = 2018      # CFBD weather / havoc history is patchier before this (spec risk)


def _seasons(df):
    return sorted(df["season"].unique())


def check_weather(w: pd.DataFrame) -> list[str]:
    out = []
    if w.empty:
        return ["weather: no rows"]
    for s in _seasons(w):
        d = w[(w["season"] == s) & ~w["game_indoors"]]
        temp, wind, pr = d["temperature"].dropna(), d["wind_speed"].dropna(), d["precipitation"].dropna()
        if len(temp) and not TEMP_F[0] <= temp.mean() <= TEMP_F[1]:
            out.append(f"weather {s}: mean outdoor temperature {temp.mean():.1f} is not plausible degrees F "
                       f"({TEMP_F}); Celsius? convert in parse_weather_games")
        if len(wind) and not WIND_MPH[0] <= wind.mean() <= WIND_MPH[1]:
            out.append(f"weather {s}: mean wind {wind.mean():.1f} is not plausible mph ({WIND_MPH}); "
                       "m/s or km/h? convert in parse_weather_games")
        if len(pr) and (pr.max() > PRECIP_MAX or pr.min() < 0):
            out.append(f"weather {s}: precipitation range {pr.min():.2f}..{pr.max():.2f} looks wrong")
    return out


def check_havoc(h: pd.DataFrame) -> list[str]:
    out = []
    if h.empty:
        return ["havoc: no rows"]
    for s in _seasons(h):
        d = h[h["season"] == s]
        for col in ("off_havoc", "def_havoc"):
            m = d[col].dropna().mean()
            if not HAVOC_RATE[0] <= m <= HAVOC_RATE[1]:
                out.append(f"havoc {s}: mean {col} {m:.3f} outside {HAVOC_RATE}")
    j = h.merge(h, left_on=["game_id", "opponent"], right_on=["game_id", "team"], suffixes=("", "_opp"))
    gap = (j["def_havoc"] - j["off_havoc_opp"]).abs().dropna()
    if len(gap) and gap.mean() > HAVOC_PAIR_MAX_GAP:
        out.append(f"havoc: team's DEFENSE havoc differs from its opponent's OFFENSE havoc by {gap.mean():.3f} "
                   "on average -- the offense/defense sides may be the other way round")
    return out


def check_drives(d: pd.DataFrame) -> list[str]:
    out = []
    if d.empty:
        return ["drives: no rows"]
    for s in _seasons(d):
        x = d[d["season"] == s]
        ppd, sy = x["off_ppd"].dropna().mean(), x["off_start_yd"].dropna().mean()
        if not POINTS_PER_DRIVE[0] <= ppd <= POINTS_PER_DRIVE[1]:
            out.append(f"drives {s}: mean points/drive {ppd:.2f} outside {POINTS_PER_DRIVE}")
        if not START_YD[0] <= sy <= START_YD[1]:
            out.append(f"drives {s}: mean start yard line {sy:.1f} outside {START_YD}")
    j = d.merge(d, left_on=["game_id", "opponent"], right_on=["game_id", "team"], suffixes=("", "_opp"))
    gap = (j["def_points"] - j["off_points_opp"]).abs().dropna()
    if len(gap) and gap.max() > 0:
        out.append("drives: a team's def_points differs from its opponent's off_points")
    return out


def coverage(asset: pd.DataFrame, sched: pd.DataFrame, per: str = "team") -> pd.DataFrame:
    """Per season: share of REG FBS-vs-FBS schedule games `asset` covers (per='team': both team rows;
    per='game': one row per game)."""
    s = sched[(sched["game_type"] == "REG") & (sched["home_team"] != "FCS") & (sched["away_team"] != "FCS")]
    n = asset.groupby(["season", "game_id"]).size()
    need = 2 if per == "team" else 1
    ok = {k for k, v in n.items() if v >= need}
    s = s.assign(covered=[(int(a), int(b)) in ok for a, b in zip(s["season"], s["game_pk"])])
    g = s.groupby("season")["covered"].mean().rename("share")
    return g[g.index.isin(asset["season"].unique())].to_frame()


def check_coverage(assets: dict, sched: pd.DataFrame) -> list[str]:
    out = []
    for name, df, per in (("advanced", assets.get("advanced"), "team"), ("havoc", assets.get("havoc"), "team"),
                          ("drives", assets.get("drives"), "team"), ("weather", assets.get("weather"), "game")):
        if df is None or df.empty:
            out.append(f"{name}: asset missing or empty")
            continue
        reg = df[df["season_type"] == "regular"] if "season_type" in df else df
        cov = coverage(reg, sched, per)
        low = [int(s) for s, v in cov["share"].items() if s >= COVERAGE_FROM_SEASON and v < MIN_COVERAGE]
        if low:
            out.append(f"{name}: coverage below {MIN_COVERAGE:.0%} of FBS-vs-FBS games in seasons {low}")
    return out


def check_all(assets: dict, sched: pd.DataFrame) -> list[str]:
    """assets: {advanced, havoc, drives, weather, ...} frames (None = missing)."""
    out = check_coverage(assets, sched)
    for key, fn in (("weather", check_weather), ("havoc", check_havoc), ("drives", check_drives)):
        if assets.get(key) is not None:
            out += fn(assets[key])
    return out
