"""Power ratings and rankings (NFL + CFB). PURE.

Power rating = expected margin (points) vs an average team on a neutral field.

CFB (``cfb_power``) -- the ratings model itself (Elo + SRS blended by
``sportsmodel.nfl.ratings.expected_margin``), with the leak-free state of
``sportsmodel.cfb.walkforward.raw_model_predictions`` as of (season, week):

* schedule: REG games only, CFBD self-matches (home == away) dropped (same as
  ``backtest_cfb_gameline.load_merged_schedule`` / ``generate_cfb``);
* Elo: ``run_elo`` over every completed game before the target week, continuous
  across seasons; when the target season has no completed game yet the season
  carryover is applied (as ``run_elo`` would on the team's first game);
* SRS: ``compute_srs`` over the target season's completed games of weeks < the
  target week; ``games`` = those games per team (the SRS blend needs
  ``blend_cfg.srs_min_games``);
* the synthetic average FBS team: Elo = mean Elo of the FBS teams, SRS = mean SRS
  of the FBS teams (0 when every team is FBS; the pooled "FCS" pseudo-team makes
  the all-team SRS mean 0, so the FBS mean sits a hair above it), and enough
  games for the blend. The FBS-average rating is therefore ~0 -- exactly 0 only
  when every FBS team is on the same blend branch (all have >=
  ``srs_min_games``, or none has an SRS yet);
* neutral field: ``expected_margin`` is called with ``hfa_elo = 0``, which removes
  the home edge from both the Elo side and the SRS side (``hfa_points =
  hfa_elo / 25``) -- identical to ``expected_margin(...) - hfa_elo / 25``.

Every team seen in a completed game of the target or previous season is returned
with ``is_fbs`` (``sportsmodel.cfb.teams.load_fbs_ids``); the "FCS"
pseudo-team is kept (``is_fbs`` False) so it can count in SOS, and ``rankings``
never ranks it.

NFL (``nfl_power``) -- from the opponent-adjusted unit ratings of
``sportsmodel.context.units.unit_ratings_asof`` (EPA/play relative to league
average, early-season blended)::

    off_rating = off_pass_epa * off_pass_plays + off_run_epa * off_run_plays
    def_rating = -(def_pass_epa * lg_pass_plays + def_run_epa * lg_run_plays)
    rating     = off_rating + def_rating

``off_*_plays`` = the team's own plays/game in that unit (blended like the
ratings); ``lg_*_plays`` = the league-average plays/game in that unit, i.e. what
an average opponent's offense runs (``def_*_epa`` + = allows more, so it is
subtracted). The team's own ``def_*_plays`` is deliberately not used: vs an
average team the opponent runs an average offense. No home field (neutral).

Market scale (controller ruling P3). The raw EPA-points rating is roughly twice
as spread out as the market, so it is put on the closing-spread point scale:
for season S, ``nfl_market_scale`` fits, over REG games of seasons S-3..S-1 with
a closing ``spread_line`` (never season S), each game's raw rating difference
(home raw - away raw, both as of that game's week -- walk-forward):

    hfa   = mean closing home spread over the non-neutral games
    slope = no-intercept least squares of (spread_line - hfa * non_neutral) on raw_diff

and ``nfl_power(..., scale=...)`` reports ``rating = slope * raw_rating`` (likewise
``off_rating`` / ``def_rating``), keeping ``raw_rating`` and storing ``scale`` /
``hfa`` as columns and in ``df.attrs["market_scale"]``. The fit is frozen per
season (a pure function of prior seasons). No scale -> slope 1 (raw points).

Season weighting (user, 2026-09-30): the RANKINGS weight this season by
``w = games / (games + 1)`` (``POWER_BLEND_K``) -- 50% after 1 game, 75% after 3, ~90%
by week 9. CFB: ``cfb_power_current`` (this-season SRS vs the preseason prior). NFL:
``nfl_power`` over ``unit_ratings_asof(..., blend_k=POWER_BLEND_K)`` with the market
scale fit on the same blend. Matchup grades and the betting models keep their blends.

Rankings (``rankings``): rank 1 = highest rating (ties share the better rank);
``move`` = prev_rank - rank (+ = moved up); unit ranks by EPA/play adj (offense:
highest is 1; defense: lowest allowed is 1) among the ranked teams; SOS = mean
current rating of the opponents already played this season (games of weeks <
the ranking week; unrated opponents skipped); SU / ATS season records from the
game log, same format as ``context.history`` (SU "W-L[-T]", ATS "W-L-P", ATS
only over lined games). SOV = mean current rating of the teams beaten; home / road W-L[-T] (neutral in
neither). SOS / SOV / SU / ATS / home / road use REGULAR-SEASON games only (log rows
with ``is_post`` True are dropped) to match the ratings' basis (CFB state is
REG-only; the NFL market scale is fit on REG games).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from sportsmodel.cfb.teams import load_fbs_ids
from sportsmodel.context.history import _ats, _su
from sportsmodel.context.results_power import strength_of_victory, venue_records
from sportsmodel.context.units import (BLEND_K, POWER_BLEND_K, unit_ratings_asof,
                                      window_ratings)
from sportsmodel.nfl.elo import EloConfig, _carryover, elo_expected_margin, run_elo
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.srs import compute_srs
from sportsmodel.nfl.teams import normalize_team

CFB_POWER_COLUMNS = ["season", "week", "team", "rating", "elo", "srs", "games", "is_fbs"]
NFL_POWER_COLUMNS = ["season", "week", "team", "rating", "off_rating", "def_rating",
                     "raw_rating", "scale", "hfa", "games"]
MARKET_SCALE_SEASONS = 3
UNITS = (("pass_off", "off_pass_epa", False), ("run_off", "off_run_epa", False),
         ("pass_def", "def_pass_epa", True), ("run_def", "def_run_epa", True))


def _sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(["rating", "team"], ascending=[False, True],
                          kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------ CFB
def cfb_power(schedule_df: pd.DataFrame, elo_cfg: EloConfig, blend_cfg: BlendConfig,
              asof: tuple[int, int], *, fbs=None) -> pd.DataFrame:
    """Neutral-field rating vs the average FBS team, from games before ``asof``.

    ``asof`` = (season, week): state uses completed games of earlier seasons and
    of that season's weeks < week. ``fbs`` overrides the FBS membership set.
    """
    season, week = int(asof[0]), int(asof[1])
    fbs = set(load_fbs_ids()) if fbs is None else {str(t) for t in fbs}
    s = schedule_df
    if "game_type" in s.columns:
        s = s[s["game_type"] == "REG"]
    s = s[s["home_team"] != s["away_team"]]
    before = (s["season"] < season) | ((s["season"] == season) & (s["week"] < week))
    played = s[before].dropna(subset=["home_score", "away_score"])
    if played.empty:
        return pd.DataFrame(columns=CFB_POWER_COLUMNS)

    elo = run_elo(played, elo_cfg).final
    if int(played["season"].max()) < season:
        elo = {t: _carryover(r, elo_cfg) for t, r in elo.items()}
    cur = played[played["season"] == season]
    srs = compute_srs(cur) if len(cur) else {}
    games = pd.concat([cur["home_team"], cur["away_team"]]).value_counts().to_dict()

    recent = played[played["season"] >= season - 1]
    teams = sorted(set(recent["home_team"]) | set(recent["away_team"]))
    is_fbs = {t: str(t) in fbs for t in teams}
    fbs_teams = [t for t in teams if is_fbs[t]]
    avg_elo = float(np.mean([elo[t] for t in fbs_teams])) if fbs_teams else elo_cfg.base
    fbs_srs = [srs[t] for t in fbs_teams if t in srs]
    avg_srs = float(np.mean(fbs_srs)) if fbs_srs else 0.0

    neutral = dataclasses.replace(elo_cfg, hfa_elo=0.0)
    rows = []
    for t in teams:
        g = int(games.get(t, 0))
        rating = expected_margin(elo[t], avg_elo, srs.get(t), avg_srs, g,
                                 blend_cfg.srs_min_games, neutral, blend_cfg)
        rows.append({"season": season, "week": week, "team": t, "rating": float(rating),
                     "elo": float(elo[t]), "srs": float(srs[t]) if t in srs else np.nan,
                     "games": g, "is_fbs": is_fbs[t]})
    return _sorted(pd.DataFrame(rows, columns=CFB_POWER_COLUMNS))


CFB_CURRENT_COLUMNS = CFB_POWER_COLUMNS + ["prior", "weight"]


def cfb_power_current(schedule_df: pd.DataFrame, elo_cfg: EloConfig, blend_cfg: BlendConfig,
                      asof: tuple[int, int], *, priors: dict | None = None,
                      blend_k: float = POWER_BLEND_K, fbs=None) -> pd.DataFrame:
    """Power rating weighted to THIS season (the rankings' rating; module doc).

    ``rating = w * current + (1 - w) * prior``, ``w = games / (games + blend_k)``:

    * current = this season's SRS (opponent-adjusted margin, games of weeks < week)
      minus the FBS-average SRS -- points vs an average FBS team;
    * prior = the preseason rating ``priors[team]`` (Elo scale, the leak-free
      ``cfb.priors.season_priors``) minus the FBS-average prior, / 25 -> points;
      a team without a prior falls back to its carried-over Elo vs the FBS-average
      Elo, / 25;
    * games = this season's games before ``week``; no SRS yet -> the prior only.

    Same state (and columns) as ``cfb_power`` plus ``prior`` and ``weight``.
    """
    base = cfb_power(schedule_df, elo_cfg, blend_cfg, asof, fbs=fbs)
    if base.empty:
        return pd.DataFrame(columns=CFB_CURRENT_COLUMNS)
    priors = {str(k): float(v) for k, v in (priors or {}).items()}
    fb = base[base["is_fbs"].astype(bool)]
    avg_elo = float(fb["elo"].mean()) if len(fb) else elo_cfg.base
    fbs_srs = fb["srs"].dropna()
    avg_srs = float(fbs_srs.mean()) if len(fbs_srs) else 0.0
    fbs_pre = [priors[str(t)] for t in fb["team"] if str(t) in priors]
    avg_pre = float(np.mean(fbs_pre)) if fbs_pre else 0.0
    neutral = dataclasses.replace(elo_cfg, hfa_elo=0.0)
    pre, cur, w = [], [], []
    for t, e, srs, g in base[["team", "elo", "srs", "games"]].itertuples(index=False):
        pre.append((priors[str(t)] - avg_pre) / 25.0 if str(t) in priors
                   else elo_expected_margin(e, avg_elo, neutral))
        has = g > 0 and np.isfinite(srs)
        cur.append(srs - avg_srs if has else 0.0)
        w.append(g / (g + blend_k) if has else 0.0)
    pre, cur, w = np.array(pre), np.array(cur), np.array(w)
    out = base.copy()
    out["rating"] = w * cur + (1 - w) * pre
    out["prior"], out["weight"] = pre, w
    return _sorted(out[CFB_CURRENT_COLUMNS])


# ------------------------------------------------------------------------ NFL
def _league_plays(unit_ratings: pd.DataFrame, unit_games: pd.DataFrame | None,
                  season: int, week: int) -> dict[str, float]:
    """League-average plays/game per unit (what an average opponent runs).

    From ``unit_games``: every team-game of the ratings window (the season's weeks
    < week; the previous season when that is empty). Else / fallback: the mean of
    the ratings' own ``off_*_plays``.
    """
    lg = {u: float(unit_ratings[f"off_{u}_plays"].mean()) for u in ("pass", "run")}
    if unit_games is None or unit_games.empty:
        return lg
    win = unit_games[(unit_games["season"] == season) & (unit_games["week"] < week)]
    if win.empty:
        win = unit_games[unit_games["season"] == season - 1]
    for u in ("pass", "run"):
        v = win[f"{u}_plays"].mean() if len(win) else np.nan
        if np.isfinite(v):
            lg[u] = float(v)
    return lg


def nfl_power(unit_ratings: pd.DataFrame, unit_games: pd.DataFrame | None = None,
              scale: dict | None = None) -> pd.DataFrame:
    """Unit-rating points vs an average team on a neutral field (module doc).

    ``scale`` = ``nfl_market_scale(...)`` for the ratings' season (``slope``,
    ``hfa``); None -> raw points (slope 1, hfa NaN).
    """
    ur = unit_ratings
    slope = 1.0 if scale is None or scale.get("slope") is None else float(scale["slope"])
    hfa = np.nan if scale is None or scale.get("hfa") is None else float(scale["hfa"])
    if ur.empty:
        out = pd.DataFrame(columns=NFL_POWER_COLUMNS)
        out.attrs["market_scale"] = dict(scale or {}, slope=slope, hfa=hfa)
        return out
    keys = ur[["season", "week"]].drop_duplicates()
    if len(keys) != 1:
        raise ValueError("nfl_power expects unit ratings for a single (season, week)")
    season, week = int(keys["season"].iloc[0]), int(keys["week"].iloc[0])
    lg = _league_plays(ur, unit_games, season, week)
    off = sum(ur[f"off_{u}_epa"] * ur[f"off_{u}_plays"] for u in ("pass", "run"))
    dfn = -sum(ur[f"def_{u}_epa"] * lg[u] for u in ("pass", "run"))
    raw = (off + dfn).astype(float).to_numpy()
    out = pd.DataFrame({"season": season, "week": week, "team": ur["team"].to_numpy(),
                        "rating": slope * raw,
                        "off_rating": slope * off.astype(float).to_numpy(),
                        "def_rating": slope * dfn.astype(float).to_numpy(),
                        "raw_rating": raw, "scale": slope, "hfa": hfa,
                        "games": ur["games"].to_numpy() if "games" in ur else np.nan})
    out = _sorted(out[NFL_POWER_COLUMNS])
    out.attrs["market_scale"] = dict(scale or {}, slope=slope, hfa=hfa)
    return out


def _norm(code) -> str:
    """nflverse code -> canonical (as ``nfl_unit_games``); unknown codes kept as-is."""
    try:
        return normalize_team(str(code))
    except ValueError:
        return str(code)


def nfl_market_scale(unit_games: pd.DataFrame, schedules: pd.DataFrame, season: int,
                     n_seasons: int = MARKET_SCALE_SEASONS, blend_k: float = BLEND_K) -> dict:
    """Fit (slope, hfa) for ``season`` from REG games of the ``n_seasons`` before it.

    ``schedules``: nflverse schedule rows (season, week, home_team, away_team,
    spread_line; ``game_type`` and ``location`` used when present -- without
    ``location`` every game counts as a home game). Raw ratings are
    ``nfl_power(unit_ratings_asof(unit_games, s, week), unit_games)`` per game
    week, blended with ``blend_k`` (fit with the same blend the ratings are served
    with). Only rows of seasons < ``season`` are read. Returns ``{"season",
    "slope", "hfa", "n", "seasons"}``; slope/hfa None when no usable game.
    """
    seasons = list(range(season - n_seasons, season))
    ug = unit_games[unit_games["season"] < season]
    sc = schedules[schedules["season"].isin(seasons)]
    if "game_type" in sc.columns:
        sc = sc[sc["game_type"] == "REG"]
    sc = sc[sc["spread_line"].notna()]
    xs, ys, homes = [], [], []
    for s in sorted(sc["season"].unique()):
        prev = window_ratings(ug, int(s) - 1, 99)
        for wk, g in sc[sc["season"] == s].groupby("week"):
            raw = nfl_power(unit_ratings_asof(ug, int(s), int(wk), prev=prev,
                                               blend_k=blend_k), ug)
            r = raw.set_index("team")["raw_rating"] if len(raw) else pd.Series(dtype=float)
            for x in g.itertuples(index=False):
                h, a = _norm(x.home_team), _norm(x.away_team)
                if h not in r.index or a not in r.index:
                    continue
                xs.append(float(r[h] - r[a]))
                ys.append(float(x.spread_line))
                loc = getattr(x, "location", "Home")
                homes.append(0.0 if str(loc).lower() == "neutral" else 1.0)
    out = {"season": int(season), "slope": None, "hfa": None, "n": len(xs), "seasons": seasons}
    if not xs:
        return out
    x, y, home = np.array(xs), np.array(ys), np.array(homes)
    hfa = float(y[home == 1].mean()) if (home == 1).any() else 0.0
    denom = float(x @ x)
    if denom <= 0:
        return out
    out["hfa"] = hfa
    out["slope"] = float(x @ (y - hfa * home) / denom)
    return out


# ------------------------------------------------------------------- rankings
def _ranked(power: pd.DataFrame) -> pd.DataFrame:
    p = power
    if "is_fbs" in p.columns:
        p = p[p["is_fbs"].astype(bool)]
    p = p.dropna(subset=["rating"]).copy()
    p["rank"] = p["rating"].rank(ascending=False, method="min").astype("int64")
    return p


def _int_or_none(v):
    return None if pd.isna(v) else int(v)


def rankings(power_df: pd.DataFrame, prev_power_df: pd.DataFrame | None,
             unit_ratings: pd.DataFrame | None, log: pd.DataFrame | None) -> pd.DataFrame:
    """One row per ranked team, best first (see module doc).

    ``power_df`` carries ``season``/``week`` (the ranking cutoff); ``log`` is the
    team game log of the same sport (``context.game_log``); games of that season
    with a result and week < the ranking week count for SOS and records.
    ``prev_power_df`` = the power table as of the previous week's cutoff.
    """
    out = _ranked(power_df)
    out = out.sort_values(["rank", "team"], kind="stable").reset_index(drop=True)

    if prev_power_df is not None and not prev_power_df.empty:
        prev = _ranked(prev_power_df).set_index("team")["rank"]
        out["prev_rank"] = out["team"].map(prev).astype("Int64")
    else:
        out["prev_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    out["move"] = (out["prev_rank"] - out["rank"]).astype("Int64")

    have_units = (unit_ratings is not None and not unit_ratings.empty
                  and all(c in unit_ratings.columns for _, c, _ in UNITS))
    unit_vals = {}
    if have_units:
        u = unit_ratings.drop_duplicates("team").set_index("team").reindex(out["team"])
        for key, col, asc in UNITS:
            vals = u[col].astype(float)
            out[f"{key}_rank"] = vals.rank(ascending=asc, method="min").astype("Int64").to_numpy()
            unit_vals[key] = vals.to_numpy()
        out["units"] = [
            {key: {"rank": _int_or_none(out.at[i, f"{key}_rank"]),
                   "epa": None if np.isnan(unit_vals[key][i]) else float(unit_vals[key][i])}
             for key, _, _ in UNITS}
            for i in range(len(out))]
    else:
        for key, _, _ in UNITS:
            out[f"{key}_rank"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
        out["units"] = None

    out["sos"], out["su"], out["ats"] = np.nan, None, None
    out["sov"], out["home_record"], out["road_record"] = np.nan, None, None
    if log is not None and not log.empty and len(out):
        season, week = int(out["season"].iloc[0]), int(out["week"].iloc[0])
        d = log[(log["season"] == season) & (log["week"] < week) & log["su"].notna()]
        if "is_post" in d.columns:
            d = d[~d["is_post"].fillna(False).astype(bool)]
        opp_rating = power_df.drop_duplicates("team").set_index("team")["rating"]
        by_team = dict(tuple(d.groupby("team")))
        sos, su, ats = [], [], []
        for t in out["team"]:
            g = by_team.get(t, d.iloc[0:0])
            r = g["opponent"].map(opp_rating).dropna()
            sos.append(float(r.mean()) if len(r) else np.nan)
            su.append(_su(g))
            ats.append(_ats(g))
        out["sos"], out["su"], out["ats"] = sos, su, ats
        teams = out["team"].astype(str)
        out["sov"] = teams.map(strength_of_victory(d, opp_rating.rename(index=str))).to_numpy()
        if "venue" in d.columns:
            rec = venue_records(d)
            for c in ("home_record", "road_record"):
                col = rec[c] if c in rec else pd.Series(dtype=object)
                out[c] = teams.map(col).where(teams.isin(col.index), "0-0").to_numpy()
    out["su"] = out["su"].astype(object)
    out["ats"] = out["ats"].astype(object)
    return out
