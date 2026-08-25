from __future__ import annotations
import pandas as pd

def compute_srs(games: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-8) -> dict:
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    margins: dict[str, list[float]] = {t: [] for t in teams}
    opponents: dict[str, list[str]] = {t: [] for t in teams}
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        m = float(g["home_score"] - g["away_score"])
        margins[h].append(m); opponents[h].append(a)
        margins[a].append(-m); opponents[a].append(h)
    avg_margin = {t: (sum(margins[t]) / len(margins[t]) if margins[t] else 0.0)
                  for t in teams}
    rating = dict(avg_margin)
    for _ in range(max_iter):
        prev = dict(rating)
        # Gauss-Seidel: update each team's rating in place, using the
        # latest available values for its opponents (rather than the
        # previous pass's values for everyone). Plain Jacobi (compute
        # all-new-from-all-old) can oscillate indefinitely on small/
        # cyclic schedules instead of converging; Gauss-Seidel does not.
        for t in teams:
            opp = opponents[t]
            sos = sum(rating[o] for o in opp) / len(opp) if opp else 0.0
            rating[t] = avg_margin[t] + sos
        # zero-mean the ratings each pass to pin the free constant
        mean = sum(rating.values()) / len(rating)
        rating = {t: v - mean for t, v in rating.items()}
        if max(abs(rating[t] - prev[t]) for t in teams) < tol:
            break
    return rating
