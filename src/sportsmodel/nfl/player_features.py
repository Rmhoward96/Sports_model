"""Player-week feature table for the props-ML models. PURE (DataFrames in,
DataFrames out). Leakage contract: every feature for (season S, week w) is
built from rows strictly before (S, w) -- enforced by shift(1) before any
rolling statistic and tested by perturbing week-w-and-later box scores.

Column prefixes select feature groups per ladder rung (sim.nfl.learned):
p_ usage/efficiency, ngs_ Next Gen Stats, tm_ own team, op_ opponent,
st_ status, cx_ context, mk_ market; y_ are labels (never features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

SKILL = ("QB", "RB", "WR", "TE")
KEYS = ["player_id", "season", "week"]
_STAT_COLS = {
    "targets": "y_targets", "carries": "y_carries", "attempts": "y_pass_att", "receptions": "y_receptions",
    "receiving_yards": "y_rec_yds", "rushing_yards": "y_rush_yds", "passing_yards": "y_pass_yds",
    "passing_tds": "y_pass_tds", "receiving_air_yards": "rec_air_yards", "receiving_yards_after_catch": "yac",
    "target_share": "target_share", "air_yards_share": "air_yards_share",
    "receiving_tds": "_rec_tds", "rushing_tds": "_rush_tds",
}


def _norm(code) -> str | None:
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def player_games(weekly: pd.DataFrame, snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> pd.DataFrame:
    """One row per REG-season player-game for QB/RB/WR/TE with offense_snaps > 0.
    Snap counts define who played (weekly omits players with no stats); weekly
    stats are left-joined and zero-filled (played + no stat = 0, not NaN)."""
    s = snaps[(snaps["game_type"] == "REG") & (snaps["offense_snaps"] > 0)
              & snaps["position"].isin(SKILL)].copy()
    s["player_id"] = s["pfr_player_id"].map(pfr2gsis)
    s["team"], s["opponent"] = s["team"].map(_norm), s["opponent"].map(_norm)
    s = s.dropna(subset=["player_id", "team", "opponent"])
    s = s.rename(columns={"offense_pct": "snap_pct"})[
        ["player_id", "season", "week", "team", "opponent", "position", "snap_pct"]]
    w = weekly[weekly["season_type"] == "REG"] if "season_type" in weekly.columns else weekly
    stats = w[KEYS + list(_STAT_COLS)].rename(columns=_STAT_COLS)
    out = s.merge(stats, on=KEYS, how="left").drop_duplicates(KEYS)
    stat_cols = list(_STAT_COLS.values())
    out[stat_cols] = out[stat_cols].fillna(0.0)
    out["y_anytime_td"] = ((out["_rec_tds"] + out["_rush_tds"]) > 0).astype(float)
    return out.drop(columns=["_rec_tds", "_rush_tds"]).reset_index(drop=True)


def team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, offense team) REG-season volume: pass attempts exclude
    sacks (matches sim rates B.3), neutral pass rate over wp 0.2-0.8, downs 1-2,
    quarters 1-3."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()].copy()
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    sack = p["sack"].fillna(0) == 1
    p["_pass"] = (p["play_type"] == "pass") & ~sack
    p["_rush"] = p["play_type"] == "run"
    p["_db"] = p["play_type"] == "pass"
    p["_press"] = p["_db"] & (sack | (p["qb_hit"].fillna(0) == 1))
    neutral = p["wp"].between(0.2, 0.8) & p["down"].isin([1, 2]) & (p["qtr"] <= 3)
    p["_n"], p["_npass"] = neutral, neutral & p["_db"]
    g = p.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        pass_att=("_pass", "sum"), rush_att=("_rush", "sum"), plays=("play_type", "size"),
        dropbacks=("_db", "sum"), pressures_allowed=("_press", "sum"), _n=("_n", "sum"), _npass=("_npass", "sum"))
    g["neutral_pass_rate"] = np.where(g["_n"] > 0, g["_npass"] / g["_n"].where(g["_n"] > 0, 1), np.nan)
    g["team"], g["opponent"] = g["posteam"].map(_norm), g["defteam"].map(_norm)
    return g.drop(columns=["posteam", "defteam", "_n", "_npass"]).dropna(subset=["team", "opponent"])


def player_redzone(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per player-game red-zone targets/carries (yardline_100 <= 20) and
    goal-line carries (<= 5)."""
    p = pbp[pbp["yardline_100"].notna()]
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    rz = p[p["yardline_100"] <= 20]
    t = rz.dropna(subset=["receiver_player_id"]).groupby(["receiver_player_id", "season", "week"]).size().rename("rz_targets")
    c = rz.dropna(subset=["rusher_player_id"]).groupby(["rusher_player_id", "season", "week"]).size().rename("rz_carries")
    gl = p[p["yardline_100"] <= 5].dropna(subset=["rusher_player_id"]).groupby(
        ["rusher_player_id", "season", "week"]).size().rename("gl_carries")
    for s in (t, c, gl):
        s.index = s.index.set_names(KEYS)
    return pd.concat([t, c, gl], axis=1).fillna(0.0).reset_index()
