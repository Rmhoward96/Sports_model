"""Opponent-adjusted pass / run unit ratings (NFL + CFB). PURE.

Per team-game offense metrics (``UNIT_GAME_COLUMNS``), one row per team per game:

* ``pass_epa`` / ``run_epa`` -- EPA per play (CFB: CFBD PPA per play, ~EPA).
* ``pass_success`` / ``run_success`` -- success rate.
* ``pass_explosive`` / ``run_explosive`` --
    NFL: share of plays that are explosive (dropback >= 20 yds, designed run >= 10 yds).
    CFB: CFBD ``explosiveness`` is a *magnitude* (avg PPA of successful plays), not a
    rate, so it is z-scored within season before use: each game's value is standardized
    against the mean / std (ddof=1) of every team-game of that season up to and including
    its week. Using only weeks <= the game's week keeps it leak-free; a new season
    restarts it (so CFBD definition drift across seasons washes out).
* ``pass_plays`` / ``run_plays`` -- plays by unit that game; ``pass_rate`` = dropbacks /
  (dropbacks + designed runs). CFB split counts are not always supplied by CFBD; when
  missing, ``pass_rate`` falls back to ``CFB_DEFAULT_PASS_RATE`` (0.5, a round league
  average) and the play columns stay NaN.

NFL conventions (nflverse pbp, REG + POST, 2-pt tries and plays without EPA dropped):
dropbacks = ``play_type == "pass"`` (incl. sacks) or ``qb_scramble == 1``; designed runs
= ``play_type == "run"`` and not a scramble. Scramble yards count as dropback yards.

A defense's value for a game is its opponent's offensive value in that game (paired by
``game_id``). CFB: when the opponent's own row is missing, its offense row is rebuilt
from this row's ``def_*`` columns (CFBD defense = what the opponent's offense did).

Ratings (``unit_ratings_asof``) -- for every metric, a single opponent-adjustment pass
identical to ``sportsmodel.nfl.efficiency.adjusted_efficiency`` (same-season games of
weeks < target only), then made relative to the league mean of that window (``_relative``
in ``sportsmodel.nfl.unit_efficiency``). ``off_*`` + = better offense; ``def_*`` + =
ALLOWS more than average (weaker defense), so ``off_X[team] + def_X[opp]`` is the
offense's edge. Early season: blended with the previous season's final value
(``window_ratings(s-1, 99)``) by ``w = games / (games + 3)`` on the current value, games =
team games this season before the target week (CFB: FBS-vs-FBS games only -- FCS games
have no advanced stats). No previous-season value -> current only; no games yet ->
previous only; neither -> the team is absent (callers grade it None).

This module does not touch ``sportsmodel.nfl.unit_efficiency`` (it feeds the live v2
model); the extra metrics are computed here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

METRICS: tuple[str, ...] = ("pass_epa", "pass_success", "pass_explosive",
                            "run_epa", "run_success", "run_explosive")
PLAY_COLS: tuple[str, ...] = ("pass_plays", "run_plays")
UNIT_GAME_COLUMNS = ["season", "week", "game_id", "team", "opponent",
                     *PLAY_COLS, "pass_rate", *METRICS]
BLEND_K = 3.0
CFB_DEFAULT_PASS_RATE = 0.5
NFL_PASS_EXPLOSIVE_YDS = 20
NFL_RUN_EXPLOSIVE_YDS = 10

_RATING_COLS = ([f"{s}_{m}" for s in ("off", "def") for m in METRICS] + ["pass_rate"]
                + [f"{s}_{p}" for s in ("off", "def") for p in PLAY_COLS])


def blend_weight(games) -> float:
    """Weight on the current-season value: games / (games + 3)."""
    return float(games) / (float(games) + BLEND_K)


def _norm(code):
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[UNIT_GAME_COLUMNS].copy()
    df[["season", "week"]] = df[["season", "week"]].astype("int64")
    for c in (*PLAY_COLS, "pass_rate", *METRICS):
        df[c] = df[c].astype("float64")
    return df.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)


# ------------------------------------------------------------------ team-games
def nfl_unit_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """nflverse play-by-play -> one offense row per (season, week, game, team)."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()
            & pbp["defteam"].notna() & pbp["epa"].notna()]
    if "season_type" in p.columns:
        p = p[p["season_type"].isin(["REG", "POST"])]
    if "two_point_attempt" in p.columns:
        p = p[p["two_point_attempt"].fillna(0) != 1]
    scramble = (p["qb_scramble"].fillna(0) == 1 if "qb_scramble" in p.columns
                else pd.Series(False, index=p.index))
    db = (p["play_type"] == "pass") | scramble
    run = (p["play_type"] == "run") & ~scramble
    yds = p["yards_gained"].fillna(0.0)
    succ = p["success"].fillna(0.0).astype(float)
    xpl = ((db & (yds >= NFL_PASS_EXPLOSIVE_YDS)) | (run & (yds >= NFL_RUN_EXPLOSIVE_YDS))
           ).astype(float)
    team, opp = p["posteam"].map(_norm), p["defteam"].map(_norm)
    keep = team.notna() & opp.notna()
    p, db, run, yds, succ, xpl, team, opp = (x[keep] for x in (p, db, run, yds, succ, xpl, team, opp))
    if "game_id" in p.columns:
        gid = p["game_id"].astype(str)
    else:
        lo = np.where(team < opp, team, opp)
        hi = np.where(team < opp, opp, team)
        gid = (p["season"].astype(int).astype(str) + "_" + p["week"].astype(int).map("{:02d}".format)
               + "_" + pd.Series(lo, index=p.index) + "_" + pd.Series(hi, index=p.index))
    f = pd.DataFrame({
        "season": p["season"], "week": p["week"], "game_id": gid, "team": team, "opponent": opp,
        "pass_plays": db.astype(float), "run_plays": run.astype(float),
        "pass_epa": p["epa"].where(db), "pass_success": succ.where(db),
        "pass_explosive": xpl.where(db), "run_epa": p["epa"].where(run),
        "run_success": succ.where(run), "run_explosive": xpl.where(run),
    })
    g = f.groupby(["season", "week", "game_id", "team", "opponent"], sort=False)
    out = g[list(PLAY_COLS)].sum().join(g[list(METRICS)].mean()).reset_index()
    tot = out["pass_plays"] + out["run_plays"]
    out["pass_rate"] = np.where(tot > 0, out["pass_plays"] / tot.where(tot > 0), np.nan)
    return _finish(out)


_CFB_MAP = {"pass_epa": "pass_ppa", "pass_success": "pass_success",
            "pass_explosive": "pass_explosiveness", "run_epa": "rush_ppa",
            "run_success": "rush_success", "run_explosive": "rush_explosiveness"}


def _season_to_date_z(df: pd.DataFrame, col: str) -> pd.Series:
    """z-score vs all team-games of the same season with week <= the row's week."""
    out = pd.Series(np.nan, index=df.index, dtype="float64")
    for _, s in df.groupby("season"):
        for wk in sorted(s["week"].unique()):
            ref = s.loc[s["week"] <= wk, col].dropna()
            mu, sd = (ref.mean(), ref.std(ddof=1)) if len(ref) else (np.nan, np.nan)
            idx = s.index[s["week"] == wk]
            if not np.isfinite(sd) or sd == 0:
                out.loc[idx] = np.where(df.loc[idx, col].notna(), 0.0, np.nan)
            else:
                out.loc[idx] = (df.loc[idx, col] - mu) / sd
    return out


def cfb_unit_games(advanced: pd.DataFrame) -> pd.DataFrame:
    """CFBD advanced game stats (``scripts/build_cfb_advanced.py`` schema) -> unit games."""
    a = advanced
    ids = {"season": a["season"], "week": a["week"], "game_id": a["game_id"]}
    off = pd.DataFrame({**ids, "team": a["team"].astype(str), "opponent": a["opponent"].astype(str),
                        **{m: a[f"off_{c}"] for m, c in _CFB_MAP.items()},
                        "pass_plays": a.get("off_pass_plays", np.nan),
                        "run_plays": a.get("off_rush_plays", np.nan)})
    mirror = pd.DataFrame({**ids, "team": a["opponent"].astype(str), "opponent": a["team"].astype(str),
                           **{m: a[f"def_{c}"] for m, c in _CFB_MAP.items()},
                           "pass_plays": np.nan, "run_plays": np.nan})
    have = set(zip(off["season"], off["game_id"], off["team"]))
    mirror = mirror[[k not in have for k in zip(mirror["season"], mirror["game_id"], mirror["team"])]]
    ug = pd.concat([off, mirror], ignore_index=True) if len(mirror) else off.reset_index(drop=True)
    ug = ug.drop_duplicates(subset=["season", "game_id", "team"], keep="first").reset_index(drop=True)
    pp, rp = ug["pass_plays"].astype(float), ug["run_plays"].astype(float)
    tot = pp + rp
    ug["pass_rate"] = np.where(np.isfinite(tot) & (tot > 0), pp / tot.where(tot > 0),
                               CFB_DEFAULT_PASS_RATE)
    for m in ("pass_explosive", "run_explosive"):
        ug[m] = _season_to_date_z(ug, m)
    return _finish(ug)


# --------------------------------------------------------------------- ratings
def _paired(rows: pd.DataFrame) -> pd.DataFrame:
    """Each offense row + what its opponent's offense did vs it (``def_*``)."""
    cols = list(METRICS) + list(PLAY_COLS)
    d = rows[["season", "game_id", "team", *cols]].rename(
        columns={"team": "opponent", **{c: f"def_{c}" for c in cols}})
    return rows.merge(d, on=["season", "game_id", "opponent"], how="left")


def window_ratings(unit_games: pd.DataFrame, season: int, upto_week: int) -> pd.DataFrame:
    """Unblended ratings from same-season weeks < ``upto_week`` (index = team).

    Per metric: the single-pass opponent adjustment of ``efficiency.adjusted_efficiency``
    then minus the league mean of the window. ``pass_rate`` and plays/game are plain means
    (``def_*_plays`` = opponents' plays per game vs the team). ``games`` = team-games used.
    """
    ug = unit_games
    rows = ug[(ug["season"] == season) & (ug["week"] < upto_week)]
    rows = rows.drop_duplicates(subset=["season", "game_id", "team"])
    if rows.empty:
        return pd.DataFrame(columns=_RATING_COLS + ["games"], index=pd.Index([], name="team"))
    m = _paired(rows)
    teams = pd.Index(sorted(m["team"].unique()), name="team")
    out = pd.DataFrame(index=teams)
    for metric in METRICS:
        ov, dv = m[metric], m[f"def_{metric}"]
        raw_off = ov.groupby(m["team"]).mean().dropna()
        raw_def = dv.groupby(m["team"]).mean().dropna()
        lo = float(raw_off.mean()) if len(raw_off) else 0.0
        ld = float(raw_def.mean()) if len(raw_def) else 0.0
        off_adj = (ov - (m["opponent"].map(raw_def) - ld).fillna(0.0)).groupby(m["team"]).mean()
        def_adj = (dv - (m["opponent"].map(raw_off) - lo).fillna(0.0)).groupby(m["team"]).mean()
        off_adj = off_adj.reindex(teams).fillna(lo)
        def_adj = def_adj.reindex(teams).fillna(ld)
        out[f"off_{metric}"] = off_adj - off_adj.mean()
        out[f"def_{metric}"] = def_adj - def_adj.mean()
    by = m.groupby("team")
    out["pass_rate"] = by["pass_rate"].mean().reindex(teams)
    for p in PLAY_COLS:
        out[f"off_{p}"] = by[p].mean().reindex(teams)
        out[f"def_{p}"] = by[f"def_{p}"].mean().reindex(teams)
    out["games"] = by.size().reindex(teams).astype("int64")
    return out[_RATING_COLS + ["games"]]


def _blend(cur: pd.DataFrame, prev: pd.DataFrame) -> pd.DataFrame:
    teams = cur.index.union(prev.index)
    c, p = cur.reindex(teams), prev.reindex(teams)
    games = c["games"].fillna(0).astype("int64")
    w = games / (games + BLEND_K)
    out = pd.DataFrame(index=teams)
    for col in _RATING_COLS:
        cv, pv = c[col].astype(float), p[col].astype(float)
        out[col] = np.where(cv.notna() & pv.notna(), w * cv + (1 - w) * pv,
                            np.where(cv.notna(), cv, pv))
    out["games"] = games
    out["weight"] = np.where(p[_RATING_COLS[0]].notna(), w, np.where(games > 0, 1.0, 0.0))
    return out


def unit_ratings_asof(unit_games: pd.DataFrame, season: int, week: int,
                      prev: pd.DataFrame | None = None) -> pd.DataFrame:
    """Blended ratings per team as of (season, week) -- see module doc.

    ``prev`` optionally supplies ``window_ratings(unit_games, season - 1, 99)`` (cache).
    ``weight`` = weight actually on the current season (1.0 when there is no previous
    value, 0.0 before the first game).
    """
    cur = window_ratings(unit_games, season, week)
    if prev is None:
        prev = window_ratings(unit_games, season - 1, 99)
    out = _blend(cur, prev)
    out.index.name = "team"
    out = out.reset_index()
    out.insert(0, "week", int(week))
    out.insert(0, "season", int(season))
    return out[["season", "week", "team", "games", "weight", *_RATING_COLS]]
