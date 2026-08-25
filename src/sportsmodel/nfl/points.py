from __future__ import annotations
import math
import pandas as pd

# Sane-magnitude guard for the opponent-adjusted solve. Legitimate converged
# off/def deltas from real NFL scores are single/low-double digits; a value
# this large can only mean the Gauss-Seidel iteration is diverging.
_SANE_BOUND = 500.0

def _is_sane(values: dict) -> bool:
    return all(math.isfinite(v) and abs(v) < _SANE_BOUND for v in values.values())

def compute_points_ratings(games: pd.DataFrame, k_points: float = 4.0,
                           max_iter: int = 1000, tol: float = 1e-8):
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    scored = {t: [] for t in teams}
    allowed = {t: [] for t in teams}
    opps = {t: [] for t in teams}
    total_pts = 0.0
    for _, g in games.iterrows():
        h, a, hs, as_ = g["home_team"], g["away_team"], float(g["home_score"]), float(g["away_score"])
        scored[h].append(hs); allowed[h].append(as_); opps[h].append(a)
        scored[a].append(as_); allowed[a].append(hs); opps[a].append(h)
        total_pts += hs + as_
    n_team_games = sum(len(scored[t]) for t in teams)
    lg_avg = total_pts / n_team_games if n_team_games else 0.0
    off = {t: 0.0 for t in teams}
    deff = {t: 0.0 for t in teams}
    converged = False
    for _ in range(max_iter):
        # Snapshot the post-zero-mean-shift state from the END of the last
        # pass, so convergence is judged on the values this function
        # actually returns -- not on the raw pre-shift step size. The two
        # differ: a schedule can have a persistent, perfectly uniform
        # additive drift in the raw pre-shift update (same constant added
        # to every team's off, and to every team's def, each pass) that the
        # zero-mean pin exactly cancels every time, leaving the returned
        # dict truly stable. Judging convergence on the pre-shift step would
        # misclassify that stable case as never converging.
        prev_off = dict(off); prev_deff = dict(deff)
        for t in teams:
            games_t = scored[t]
            if not games_t:
                continue
            new_off = sum(scored[t][k] - lg_avg - deff[opps[t][k]]
                          for k in range(len(games_t))) / len(games_t)
            new_def = sum(allowed[t][k] - lg_avg - off[opps[t][k]]
                          for k in range(len(games_t))) / len(games_t)
            off[t] = new_off; deff[t] = new_def
        # pin each to zero-mean
        mo = sum(off.values()) / len(teams); md = sum(deff.values()) / len(teams)
        off = {t: off[t] - mo for t in teams}
        deff = {t: deff[t] - md for t in teams}
        post_delta = max(
            max(abs(off[t] - prev_off[t]) for t in teams),
            max(abs(deff[t] - prev_deff[t]) for t in teams),
        )
        if post_delta < tol:
            converged = True
            break

    if not (converged and _is_sane(off) and _is_sane(deff)):
        # Convergence guard: sparse/tree schedules (e.g. Week 1 of an NFL
        # season, made entirely of disjoint single-game components with no
        # cycles) leave the coupled off/def Gauss-Seidel solve with an extra,
        # unpinned degree of freedom. Unlike scalar SRS, a single zero-mean
        # constraint per off/def is not enough to resolve it there, and the
        # iteration drifts without bound instead of converging (verified:
        # max_delta plateaus at a nonzero constant and off/def grow linearly
        # every pass, never settling). Rather than return that arbitrary,
        # unbounded drift, fall back to unadjusted (but still zero-meaned)
        # PF/PA -- a finite, sane estimate with no opponent adjustment.
        off = {t: (sum(scored[t]) / len(scored[t]) - lg_avg) if scored[t] else 0.0
               for t in teams}
        deff = {t: (sum(allowed[t]) / len(allowed[t]) - lg_avg) if allowed[t] else 0.0
                for t in teams}
        if teams:
            mo = sum(off.values()) / len(teams)
            md = sum(deff.values()) / len(teams)
            off = {t: off[t] - mo for t in teams}
            deff = {t: deff[t] - md for t in teams}

    ratings = {}
    for t in teams:
        n = len(scored[t])
        factor = n / (n + k_points) if (n + k_points) > 0 else 0.0
        ratings[t] = {"off": off[t] * factor, "def": deff[t] * factor}
    return ratings, lg_avg

def expected_total(ratings: dict, lg_avg: float, home: str, away: str) -> float:
    ho = ratings.get(home, {"off": 0.0, "def": 0.0})
    ao = ratings.get(away, {"off": 0.0, "def": 0.0})
    home_pts = lg_avg + ho["off"] + ao["def"]
    away_pts = lg_avg + ao["off"] + ho["def"]
    return home_pts + away_pts
