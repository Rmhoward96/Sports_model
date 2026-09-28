"""Pass / run unit efficiency per team-game, opponent-adjusted and relative to
league (the `mx_` matchup features). PURE.

Offense metric per team-game from REG play-by-play (pass = dropbacks incl.
sacks, run = designed runs): pass_epa, nypd (net yards / dropback), sack_rate,
pass_success, rush_epa, ypc, rush_success, stuff_rate (carries <= 0 yds).
A defense's "allowed" value for a game is its opponent offense's value.

Adjustment reuses `efficiency.adjusted_efficiency` per metric (single pass,
same-season weeks < w only): `_adj` = as-of this season, `_prev` = last season
in full (week 99). Both are made relative to league (minus the league mean of
the same window), so + always means "above average" (more EPA / yards / sacks
/ success / stuffs). For defenses + means ALLOWS more than average (weaker),
except `sack_rate` / `stuff_rate` allowed, where + means the offense suffers
more (the defense is stronger at that). Plan rulings R1/R2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.efficiency import adjusted_efficiency
from sportsmodel.nfl.teams import normalize_team

UNIT_METRICS: tuple[str, ...] = ("pass_epa", "nypd", "sack_rate", "pass_success",
                                 "rush_epa", "ypc", "rush_success", "stuff_rate")
_KEYS = ["season", "week", "team"]


def _norm(code):
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def unit_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, offense team) with the UNIT_METRICS."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna() & pbp["defteam"].notna()]
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    p = p.assign(team=p["posteam"].map(_norm), opponent=p["defteam"].map(_norm),
                 _db=p["play_type"] == "pass", _run=p["play_type"] == "run",
                 _sack=p["sack"].fillna(0) == 1, _yds=p["yards_gained"].fillna(0.0),
                 _succ=p["success"].fillna(0.0), _epa=p["epa"])
    p = p.dropna(subset=["team", "opponent"])

    def agg(g: pd.DataFrame) -> pd.Series:
        db, run = g[g["_db"]], g[g["_run"]]
        nd, nr = len(db), len(run)
        return pd.Series({
            "opponent": g["opponent"].iloc[0],
            "pass_epa": db["_epa"].mean() if nd else np.nan,
            "nypd": db["_yds"].sum() / nd if nd else np.nan,
            "sack_rate": db["_sack"].mean() if nd else np.nan,
            "pass_success": db["_succ"].mean() if nd else np.nan,
            "rush_epa": run["_epa"].mean() if nr else np.nan,
            "ypc": run["_yds"].sum() / nr if nr else np.nan,
            "rush_success": run["_succ"].mean() if nr else np.nan,
            "stuff_rate": (run["_yds"] <= 0).mean() if nr else np.nan,
        })

    out = p.groupby(_KEYS).apply(agg, include_groups=False).reset_index()
    out[["season", "week"]] = out[["season", "week"]].astype("int64")
    return out


def _game_dict(ug: pd.DataFrame, metric: str) -> dict:
    """{(s, w, team): {"off": team's value, "def": value its opponent posted vs it, "opp"}}."""
    off = {(int(s), int(w), t): (v, o) for s, w, t, o, v in
           ug[["season", "week", "team", "opponent", metric]].itertuples(index=False)}
    out: dict = {}
    for (s, w, t), (v, o) in off.items():
        allowed = off.get((s, w, o), (None, None))[0]
        out[(s, w, t)] = {"off": None if pd.isna(v) else float(v),
                          "def": None if allowed is None or pd.isna(allowed) else float(allowed),
                          "opp": o}
    return out


def _relative(adj: dict, key: str) -> dict[str, float]:
    if not adj:
        return {}
    league = float(np.mean([v[key] for v in adj.values()]))
    return {t: v[key] - league for t, v in adj.items()}


def unit_features(unit_games_df: pd.DataFrame, team_weeks: pd.DataFrame) -> pd.DataFrame:
    """`team_weeks` (season, week, team, opponent) + mx_ columns (see module doc)."""
    games = {m: _game_dict(unit_games_df, m) for m in UNIT_METRICS}
    cache: dict = {}

    def rel(metric: str, s: int, w: int) -> tuple[dict, dict]:
        k = (metric, s, w)
        if k not in cache:
            adj = adjusted_efficiency(games[metric], s, w)
            cache[k] = (_relative(adj, "off_adj"), _relative(adj, "def_adj"))
        return cache[k]

    rows = []
    for s, w, team, opp in team_weeks[["season", "week", "team", "opponent"]].itertuples(index=False):
        s, w = int(s), int(w)
        r: dict = {}
        for m in UNIT_METRICS:
            off_now, def_now = rel(m, s, w)
            off_prev, def_prev = rel(m, s - 1, 99)
            r[f"mx_tm_{m}_adj"] = off_now.get(team, np.nan)
            r[f"mx_tm_{m}_prev"] = off_prev.get(team, np.nan)
            r[f"mx_op_{m}_allowed_adj"] = def_now.get(opp, np.nan)
            r[f"mx_op_{m}_allowed_prev"] = def_prev.get(opp, np.nan)
        r["mx_pass_edge"] = r["mx_tm_pass_epa_adj"] + r["mx_op_pass_epa_allowed_adj"]
        r["mx_rush_edge"] = r["mx_tm_rush_epa_adj"] + r["mx_op_rush_epa_allowed_adj"]
        r["mx_pass_minus_rush"] = r["mx_op_pass_epa_allowed_adj"] - r["mx_op_rush_epa_allowed_adj"]
        rows.append(r)
    feats = pd.DataFrame(rows, index=team_weeks.index)
    return pd.concat([team_weeks[["season", "week", "team", "opponent"]], feats], axis=1)
