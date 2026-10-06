"""CFB site-panel tables (pure): per-team quarter scoring shares and season-to-date insight ranks.

* `quarter_shares` -> `cfb_quarter_shares` (db/migration_site_panels.sql): the share of a team's points
  SCORED and ALLOWED in Q1-Q4 over the last two completed seasons plus the current one to date, shrunk toward
  the league average with weight n / (n + k) (n = games used, k = SHARE_K). Overtime is excluded; each
  vector of four shares sums to 1. The site's Projected Game Flow multiplies a team's projected score by
  the average of its own scored shares and its opponent's allowed shares.
* `team_insights` -> `cfb_team_insights`: current-season, regular-season, FBS-vs-FBS havoc / turnover /
  explosiveness rates with national ranks (1 = best) among teams with at least MIN_GAMES games (added with it).

Inputs are the committed assets under assets/cfb/ (team ids are ESPN ids as strings).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SHARE_K = 12.0                 # games of shrinkage: a team with n games keeps n / (n + 12) of its own pattern
SHARE_SEASONS_BACK = 2         # completed seasons before the current one that feed the shares
QUARTERS = (1, 2, 3, 4)
SHARE_COLUMNS = (["team"] + [f"scored_q{q}" for q in QUARTERS] + [f"allowed_q{q}" for q in QUARTERS]
                 + ["games_used"])


def quarter_shares(games: pd.DataFrame, season: int, *, k: float = SHARE_K,
                   fbs: set[str] | None = None) -> pd.DataFrame:
    """One row per team (columns SHARE_COLUMNS) from the cfbd_games frame.

    Games of seasons season-2..season (regular + postseason) with all of home/away q1..q4 present are used;
    the rest are skipped. Raw share = a team's points in a quarter / its points in all four quarters
    (scored) and the same for the points its opponents scored on it (allowed); the shrunk share is
    w * raw + (1 - w) * league with w = n / (n + k). The league share is the all-team-games share, the same
    for scored and allowed. A team with no usable game is absent; with `fbs`, only those teams are kept."""
    cols = [f"{s}_q{q}" for s in ("home", "away") for q in QUARTERS]
    g = games[games["season"].between(season - SHARE_SEASONS_BACK, season)]
    g = g.dropna(subset=cols)
    if g.empty:
        return pd.DataFrame(columns=SHARE_COLUMNS)
    home = pd.DataFrame({"team": g["home_team"].astype(str)})
    away = pd.DataFrame({"team": g["away_team"].astype(str)})
    for q in QUARTERS:
        home[f"scored_q{q}"], home[f"allowed_q{q}"] = g[f"home_q{q}"].to_numpy(), g[f"away_q{q}"].to_numpy()
        away[f"scored_q{q}"], away[f"allowed_q{q}"] = g[f"away_q{q}"].to_numpy(), g[f"home_q{q}"].to_numpy()
    tg = pd.concat([home, away], ignore_index=True)
    sc, al = [f"scored_q{q}" for q in QUARTERS], [f"allowed_q{q}" for q in QUARTERS]
    league = tg[sc].sum().to_numpy(dtype=float)
    league = league / league.sum()                       # all-team-games share of each quarter
    agg = tg.groupby("team")[sc + al].sum()
    agg["games_used"] = tg.groupby("team").size()
    if fbs is not None:
        agg = agg[agg.index.isin(fbs)]
    rows = []
    for team, r in agg.iterrows():
        n = float(r["games_used"])
        w = n / (n + k)
        out = {"team": str(team), "games_used": int(n)}
        for names in (sc, al):
            tot = float(r[names].sum())
            raw = r[names].to_numpy(dtype=float) / tot if tot > 0 else league
            shrunk = w * raw + (1 - w) * league
            out.update(dict(zip(names, (float(x) for x in shrunk / shrunk.sum()))))
        rows.append(out)
    return pd.DataFrame(rows, columns=SHARE_COLUMNS).sort_values("team").reset_index(drop=True)


def to_rows(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> DB-ready python rows: numpy scalars to int / float / str, NaN / NA to None. Rank and
    count columns (whole numbers) come back as int."""
    out = []
    for rec in df.to_dict("records"):
        row = {}
        for c, v in rec.items():
            if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
                row[c] = None
            elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                row[c] = int(v)
            elif isinstance(v, (np.floating, float)):
                row[c] = int(v) if (c.endswith("_rank") or c in ("games", "games_used", "n_ranked", "through_week", "season")) else float(v)
            else:
                row[c] = v
        out.append(row)
    return out


MIN_GAMES = 3                  # a team needs this many games to be ranked
INSIGHT_COLUMNS = ["season", "team", "games", "def_havoc_rate", "def_havoc_rank", "off_havoc_allowed_rate",
                   "off_havoc_allowed_rank", "turnover_margin", "turnover_margin_per_game",
                   "turnover_margin_rank", "off_explosiveness", "off_explosiveness_rank",
                   "def_explosiveness_allowed", "def_explosiveness_allowed_rank", "n_ranked", "through_week"]


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean ignoring NaN values; plain mean when the weights are unusable; NaN when no value."""
    v = values.astype(float)
    ok = v.notna()
    if not ok.any():
        return float("nan")
    w = weights.astype(float).where(ok)
    if w.notna().any() and w[ok].sum() > 0 and not w[ok].isna().any():
        return float((v[ok] * w[ok]).sum() / w[ok].sum())
    return float(v[ok].mean())


def _rank(s: pd.Series, eligible: pd.Series, *, best_high: bool) -> pd.Series:
    """1 = best among eligible teams (ties share the lowest rank); NaN for the rest."""
    r = s.where(eligible).rank(method="min", ascending=not best_high)
    return r.astype("Float64").where(r.notna(), pd.NA)


def team_insights(havoc: pd.DataFrame, advanced: pd.DataFrame, team_stats: pd.DataFrame | None,
                  season: int, *, min_games: int = MIN_GAMES, fbs: set[str] | None = None) -> pd.DataFrame:
    """One row per team (columns INSIGHT_COLUMNS) for `season`, regular season, through the last week that
    has havoc data. `havoc` = havoc_games (off_* = havoc the offense suffered, def_* = havoc the defense
    created), `advanced` = advanced_games (def_explosiveness = explosiveness ALLOWED), `team_stats` =
    team_game_stats (giveaways / takeaways; may be None -> turnover columns NULL). Rates are play-weighted
    season totals. Ranks (1 = best): defensive havoc created high, havoc allowed low, turnover margin per
    game high, explosiveness high (offense) / low (allowed). Teams under `min_games` games keep their
    values but get no rank; `n_ranked` is the size of the ranked field."""
    def reg(df):
        d = df[(df["season"] == season) & (df["season_type"].fillna("regular") == "regular")]
        return d[d["team"].astype(str).isin(fbs)] if fbs is not None else d
    h = reg(havoc)
    if h.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    through = int(h["week"].max())
    h = h[h["week"] <= through]
    a = reg(advanced)
    a = a[a["week"] <= through]
    ts = reg(team_stats) if team_stats is not None and len(team_stats) else None
    if ts is not None:
        ts = ts[ts["week"] <= through].dropna(subset=["giveaways", "takeaways"])

    rows = []
    for team, d in h.groupby(h["team"].astype(str)):
        def rate(ev, pl):
            plays = float(d[pl].sum())
            return float(d[ev].sum()) / plays if plays > 0 else float("nan")
        da = a[a["team"].astype(str) == team]
        row = {"season": int(season), "team": team, "games": int(len(d)),
               "def_havoc_rate": rate("def_havoc_events", "def_plays"),
               "off_havoc_allowed_rate": rate("off_havoc_events", "off_plays"),
               "off_explosiveness": _wmean(da["off_explosiveness"], da["off_plays"]) if len(da) else float("nan"),
               "def_explosiveness_allowed": _wmean(da["def_explosiveness"], da["def_plays"]) if len(da) else float("nan"),
               "turnover_margin": float("nan"), "turnover_margin_per_game": float("nan"),
               "through_week": through}
        if ts is not None:
            dt = ts[ts["team"].astype(str) == team]
            if len(dt):
                row["turnover_margin"] = float((dt["takeaways"] - dt["giveaways"]).sum())
                row["turnover_margin_per_game"] = row["turnover_margin"] / len(dt)
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df["games"] >= min_games
    for col, src, high in (("def_havoc_rank", "def_havoc_rate", True),
                           ("off_havoc_allowed_rank", "off_havoc_allowed_rate", False),
                           ("turnover_margin_rank", "turnover_margin_per_game", True),
                           ("off_explosiveness_rank", "off_explosiveness", True),
                           ("def_explosiveness_allowed_rank", "def_explosiveness_allowed", False)):
        df[col] = _rank(df[src], ok, best_high=high)
    df["n_ranked"] = int(ok.sum())
    return df[INSIGHT_COLUMNS].sort_values("team").reset_index(drop=True)
