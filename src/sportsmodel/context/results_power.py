"""Results-based power rating (NFL + CFB). PURE.

Rating = points vs an average team on a neutral field, earned from this season's
results (user decision 2026-09-30: point differential, home/road wins, strength of
schedule and strength of victory), shrunk toward a preseason prior early on.

Per game, from each team's side (``venue`` v = +1 home, -1 away, 0 neutral):

* adjusted margin = clip(margin, -cap, cap) - v * hfa   (a road win counts more)
* win credit      = result - P(win)                     (result 1 / 0.5 / 0)
  with P(win) = Phi((r_team - r_opp + v * hfa) / sigma) from the ratings as they
  stood BEFORE that game's week -- an upset road win earns a lot, a home loss to a
  weak team costs a lot, beating a bad team at home earns ~nothing
* performance     = adjusted margin + beta * win credit

Opponent adjustment (strength of schedule) with the preseason prior as ``k``
pseudo-games::

    r_t = (sum_g [perf_g + r_opp(g)] + k * prior_t) / (n_t + k)

i.e. ``w * (mean perf + mean opponent rating) + (1 - w) * prior`` with
``w = n / (n + k)`` (k = 1: 75% this season after 3 games). The system is linear
once the win credits are fixed (they use pregame ratings), so it is solved exactly:
``(diag(n + k) - A) r = S + k * prior`` (A = games between each pair); the matrix is
strictly diagonally dominant, so the solution is unique.

``season_walk`` steps through a season week by week: the ratings entering week j
use games of weeks < j only (leak-free); P(win) for week j's games is taken from
them. ``chain`` runs consecutive seasons, the prior for season S being the
caller's preseason prior where it has one, else ``rho`` x the team's final
rating of S-1 (centered), else 0.

Params (``ResultsParams``) are fit by ``scripts/fit_results_power.py`` to predict
next-week margins walk-forward and stored in ``assets/context/results_power.json``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from sportsmodel import config

PARAMS_PATH = config.PROJECT_ROOT / "assets" / "context" / "results_power.json"
GAME_COLUMNS = ["season", "week", "home_team", "away_team", "home_score", "away_score",
                "neutral"]


@dataclass(frozen=True)
class ResultsParams:
    cap: float | None      # blowout cap on |margin| (None = uncapped)
    hfa: float             # home-field edge, points
    beta: float            # points per unit of win credit
    sigma: float           # margin sd for P(win)
    rho: float             # shrink of last season's final rating when it is the prior
    k: float = 1.0         # prior weight in pseudo-games (user: games / (games + 1))


def load_params(sport: str, path: Path | None = None) -> ResultsParams:
    j = json.loads(Path(path or PARAMS_PATH).read_text())[sport]
    return ResultsParams(**{f: j[f] for f in ResultsParams.__dataclass_fields__ if f in j})


def params_dict(p: ResultsParams) -> dict:
    return asdict(p)


def played_games(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to ``GAME_COLUMNS`` (played games only, self-matches dropped)."""
    g = df.dropna(subset=["home_score", "away_score"])
    g = g[g["home_team"].astype(str) != g["away_team"].astype(str)]
    out = pd.DataFrame({
        "season": g["season"].astype("int64"), "week": g["week"].astype("int64"),
        "home_team": g["home_team"].astype(str), "away_team": g["away_team"].astype(str),
        "home_score": g["home_score"].astype(float), "away_score": g["away_score"].astype(float),
        "neutral": g["neutral"].fillna(False).astype(bool)})
    return out.sort_values(["season", "week"], kind="stable").reset_index(drop=True)


@dataclass
class SeasonWalk:
    teams: list[str]
    weeks: list[int]                       # the season's weeks, ascending
    asof: dict[int, np.ndarray]            # week -> ratings entering that week
    games_asof: dict[int, np.ndarray]      # week -> games played entering that week
    final: np.ndarray                      # after every game of the season
    final_games: np.ndarray
    pwin: np.ndarray                       # pregame P(home win) per game (row order)

    def ratings(self, week: int | None = None) -> pd.Series:
        """Ratings entering ``week`` (None / past the last week = final)."""
        r = self.final if week is None or week not in self.asof else self.asof[week]
        return pd.Series(r, index=self.teams, name="rating")

    def games(self, week: int | None = None) -> pd.Series:
        n = self.final_games if week is None or week not in self.games_asof else self.games_asof[week]
        return pd.Series(n, index=self.teams, name="games")


def _solve(n: np.ndarray, A: np.ndarray, S: np.ndarray, prior: np.ndarray, k: float) -> np.ndarray:
    M = np.diag(n + k) - A
    return np.linalg.solve(M, S + k * prior)


def season_walk(games: pd.DataFrame, prior: dict[str, float], p: ResultsParams,
                upto_week: int | None = None) -> SeasonWalk:
    """Walk one season's played games (``played_games`` rows of ONE season).

    ``prior``: {team: prior rating, points}; teams absent get 0. ``upto_week``:
    ignore games of weeks >= it (the live cutoff).
    """
    g = games if upto_week is None else games[games["week"] < upto_week]
    teams = sorted(set(g["home_team"]) | set(g["away_team"]) | set(prior))
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    pri = np.array([float(prior.get(t, 0.0)) for t in teams])
    h = g["home_team"].map(idx).to_numpy(dtype=np.int64)
    a = g["away_team"].map(idx).to_numpy(dtype=np.int64)
    m = (g["home_score"] - g["away_score"]).to_numpy(dtype=float)
    v = np.where(g["neutral"].to_numpy(dtype=bool), 0.0, 1.0)
    wk = g["week"].to_numpy(dtype=np.int64)
    res_h = np.where(m > 0, 1.0, np.where(m < 0, 0.0, 0.5))
    mc = m if p.cap is None else np.clip(m, -p.cap, p.cap)

    n = np.zeros(T)
    A = np.zeros((T, T))
    S = np.zeros(T)
    r = pri.copy()
    pwin = np.full(len(g), np.nan)
    asof, games_asof = {}, {}
    weeks = sorted({int(x) for x in wk})
    if upto_week is not None:
        weeks_all = weeks + ([int(upto_week)] if int(upto_week) not in weeks else [])
    else:
        weeks_all = weeks
    for w in weeks_all:
        r = _solve(n, A, S, pri, p.k)
        asof[w], games_asof[w] = r.copy(), n.copy()
        sel = np.flatnonzero(wk == w)
        if not len(sel):
            continue
        hi, ai, vi = h[sel], a[sel], v[sel]
        ph = norm.cdf((r[hi] - r[ai] + vi * p.hfa) / p.sigma)
        pwin[sel] = ph
        perf_h = mc[sel] - vi * p.hfa + p.beta * (res_h[sel] - ph)
        perf_a = -mc[sel] + vi * p.hfa + p.beta * ((1.0 - res_h[sel]) - (1.0 - ph))
        np.add.at(n, hi, 1.0)
        np.add.at(n, ai, 1.0)
        np.add.at(A, (hi, ai), 1.0)
        np.add.at(A, (ai, hi), 1.0)
        np.add.at(S, hi, perf_h)
        np.add.at(S, ai, perf_a)
    final = _solve(n, A, S, pri, p.k)
    return SeasonWalk(teams=teams, weeks=sorted(asof), asof=asof, games_asof=games_asof,
                      final=final, final_games=n.copy(), pwin=pwin)


def center(r: pd.Series, members=None) -> pd.Series:
    """Ratings minus the mean over ``members`` (default: every team)."""
    base = r if members is None else r[r.index.isin([str(x) for x in members])]
    return r - (float(base.mean()) if len(base) else 0.0)


def chain(games: pd.DataFrame, p: ResultsParams, seasons, preseason: dict | None = None,
          members=None) -> dict[int, SeasonWalk]:
    """Walk consecutive ``seasons`` (ascending). ``preseason``: {season: {team:
    prior points}} (e.g. CFB v2 priors); a team without one gets ``rho`` x its
    previous final rating (centered on ``members``), else 0."""
    out: dict[int, SeasonWalk] = {}
    prev: pd.Series | None = None
    for s in seasons:
        given = dict((preseason or {}).get(int(s), {}))
        prior = {} if prev is None else {t: p.rho * float(x) for t, x in prev.items()}
        prior.update({str(t): float(x) for t, x in given.items()})
        walk = season_walk(games[games["season"] == s], prior, p)
        out[int(s)] = walk
        prev = center(walk.ratings(None), members)
    return out


def predict_errors(games: pd.DataFrame, walks: dict[int, SeasonWalk], p: ResultsParams,
                   seasons) -> pd.DataFrame:
    """Walk-forward margin predictions for every game of ``seasons``: rating diff
    entering the game's week + venue hfa. Columns: season, week, home_team,
    away_team, pred, margin."""
    rows = []
    for s in seasons:
        g = games[games["season"] == s]
        w = walks[int(s)]
        idx = {t: i for i, t in enumerate(w.teams)}
        for wk, gw in g.groupby("week"):
            r = w.asof[int(wk)]
            hi = gw["home_team"].map(idx).to_numpy(dtype=np.int64)
            ai = gw["away_team"].map(idx).to_numpy(dtype=np.int64)
            v = np.where(gw["neutral"].to_numpy(dtype=bool), 0.0, 1.0)
            rows.append(pd.DataFrame({
                "season": int(s), "week": int(wk),
                "home_team": gw["home_team"].to_numpy(), "away_team": gw["away_team"].to_numpy(),
                "pred": r[hi] - r[ai] + v * p.hfa,
                "margin": (gw["home_score"] - gw["away_score"]).to_numpy(dtype=float)}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["season", "week", "home_team", "away_team", "pred", "margin"])


def strength_of_victory(log: pd.DataFrame, ratings: pd.Series) -> pd.Series:
    """Mean current rating of the teams each team has beaten (team game log rows:
    team, opponent, su). NaN for a team without a win."""
    wins = log[log["su"] == "W"]
    r = wins["opponent"].astype(str).map(ratings)
    return r.groupby(wins["team"].astype(str)).mean()


def venue_records(log: pd.DataFrame) -> pd.DataFrame:
    """Home / road W-L[-T] strings per team (neutral games in neither)."""
    def rec(d):
        w, l, t = (d["su"] == "W").sum(), (d["su"] == "L").sum(), (d["su"] == "T").sum()
        return f"{w}-{l}" + (f"-{t}" if t else "")
    out = {}
    for team, d in log.groupby(log["team"].astype(str)):
        out[team] = {"home_record": rec(d[d["venue"] == "home"]),
                     "road_record": rec(d[d["venue"] == "away"])}
    return pd.DataFrame.from_dict(out, orient="index")


POWER_FRAME_COLUMNS = ["season", "week", "team", "rating", "games", "is_fbs"]


def power_asof(games: pd.DataFrame, p: ResultsParams, season: int, week: int, *,
               preseason: dict | None = None, members=None, n_prior: int = 3,
               ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """(ratings entering ``week`` of ``season``, ratings entering the previous
    played week or None) as power frames (``POWER_FRAME_COLUMNS``), centered on
    ``members`` (CFB: FBS ids; NFL: None = every team). The prior chain starts
    ``n_prior`` seasons back (``rho ** n_prior`` of anything older survives).
    ``is_fbs`` = team in ``members`` (all True when ``members`` is None). Only teams
    with a game in ``season`` or ``season - 1`` are returned (a prior alone does not
    put a team in the rankings)."""
    season, week = int(season), int(week)
    before = [s for s in range(season - n_prior, season) if (games["season"] == s).any()]
    prior: dict[str, float] = {}
    if before:
        walks = chain(games, p, before, preseason=preseason, members=members)
        prior = {t: p.rho * float(x) for t, x in center(walks[before[-1]].ratings(), members).items()}
    prior.update({str(t): float(x) for t, x in (preseason or {}).get(season, {}).items()})
    walk = season_walk(games[games["season"] == season], prior, p, upto_week=week)
    mem = None if members is None else {str(m) for m in members}

    recent = games[games["season"].isin([season - 1, season])]
    seen = set(recent["home_team"]) | set(recent["away_team"])

    def frame(wk: int) -> pd.DataFrame:
        r = center(walk.ratings(wk), mem)
        r = r[r.index.isin(seen)]
        n = walk.games(wk)
        return pd.DataFrame({"season": season, "week": week if wk == week else wk,
                             "team": r.index, "rating": r.to_numpy(),
                             "games": n.reindex(r.index).fillna(0).astype("int64").to_numpy(),
                             "is_fbs": [True if mem is None else t in mem for t in r.index]},
                            columns=POWER_FRAME_COLUMNS)

    earlier = [w for w in walk.asof if w < week]
    return frame(week), (frame(max(earlier)) if earlier else None)
