from __future__ import annotations
import pandas as pd

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
    for _ in range(max_iter):
        max_delta = 0.0
        for t in teams:
            games_t = scored[t]
            if not games_t:
                continue
            new_off = sum(scored[t][k] - lg_avg - deff[opps[t][k]]
                          for k in range(len(games_t))) / len(games_t)
            new_def = sum(allowed[t][k] - lg_avg - off[opps[t][k]]
                          for k in range(len(games_t))) / len(games_t)
            max_delta = max(max_delta, abs(new_off - off[t]), abs(new_def - deff[t]))
            off[t] = new_off; deff[t] = new_def
        # pin each to zero-mean
        mo = sum(off.values()) / len(teams); md = sum(deff.values()) / len(teams)
        off = {t: off[t] - mo for t in teams}
        deff = {t: deff[t] - md for t in teams}
        if max_delta < tol:
            break
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
