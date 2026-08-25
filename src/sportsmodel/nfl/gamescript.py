from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

def team_game_volume(weekly: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    g = (weekly.groupby(["season", "week", "recent_team"], as_index=False)
         .agg(pass_att=("attempts", "sum"), rush_att=("carries", "sum")))
    g["plays"] = g["pass_att"] + g["rush_att"]
    g = g[g["plays"] > 0].copy()
    sch = schedules.dropna(subset=["spread_line", "total_line"])
    home = sch[["season", "week", "home_team", "spread_line", "total_line"]].rename(
        columns={"home_team": "recent_team"}).copy()
    home["team_margin"] = home["spread_line"]
    away = sch[["season", "week", "away_team", "spread_line", "total_line"]].rename(
        columns={"away_team": "recent_team"}).copy()
    away["team_margin"] = -away["spread_line"]
    lines = pd.concat([home, away], ignore_index=True)
    lines["implied_total"] = (lines["total_line"] + lines["team_margin"]) / 2.0
    return g.merge(lines[["season", "week", "recent_team", "team_margin", "implied_total"]],
                   on=["season", "week", "recent_team"])

@dataclass(frozen=True)
class GameScriptModel:
    pr_coef: tuple      # pass_rate = a0 + a1*team_margin + a2*implied_total
    plays_coef: tuple   # plays     = b0 + b1*team_margin + b2*implied_total

def _design(team_margin, implied_total, n):
    return np.column_stack([np.ones(n), np.asarray(team_margin), np.asarray(implied_total)])

def fit_gamescript(team_games: pd.DataFrame) -> GameScriptModel:
    tg = team_games[team_games["plays"] > 0]
    X = _design(tg["team_margin"], tg["implied_total"], len(tg))
    pass_rate = (tg["pass_att"] / tg["plays"]).to_numpy()
    plays = tg["plays"].to_numpy(dtype=float)
    pr_coef, *_ = np.linalg.lstsq(X, pass_rate, rcond=None)
    plays_coef, *_ = np.linalg.lstsq(X, plays, rcond=None)
    return GameScriptModel(tuple(pr_coef), tuple(plays_coef))

def project_team_volume(model: GameScriptModel, team_margin: float, implied_total: float) -> dict:
    a0, a1, a2 = model.pr_coef
    b0, b1, b2 = model.plays_coef
    pass_rate = min(max(a0 + a1 * team_margin + a2 * implied_total, 0.30), 0.75)
    plays = min(max(b0 + b1 * team_margin + b2 * implied_total, 45.0), 85.0)
    return {"pass_att": plays * pass_rate, "rush_att": plays * (1 - pass_rate), "plays": plays}
