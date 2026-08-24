from __future__ import annotations
import math
from dataclasses import dataclass, field
import pandas as pd

@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    hfa_elo: float = 65.0
    carryover: float = 0.75
    base: float = 1500.0

@dataclass
class EloResult:
    games: pd.DataFrame
    final: dict

def expected_home(elo_home: float, elo_away: float, cfg: EloConfig) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_home + cfg.hfa_elo) - elo_away) / 400.0))

def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    mov_input = abs(margin) if margin != 0 else 1.0
    return math.log(mov_input + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))

def elo_expected_margin(elo_home: float, elo_away: float, cfg: EloConfig) -> float:
    return ((elo_home + cfg.hfa_elo) - elo_away) / 25.0

def _carryover(rating: float, cfg: EloConfig) -> float:
    return cfg.base + cfg.carryover * (rating - cfg.base)

def run_elo(schedule_df: pd.DataFrame, cfg: EloConfig) -> EloResult:
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    prev_season = None
    rows = []
    for _, g in df.iterrows():
        season = g["season"]
        if prev_season is not None and season != prev_season:
            ratings = {t: _carryover(r, cfg) for t, r in ratings.items()}
        prev_season = season
        h, a = g["home_team"], g["away_team"]
        eh = ratings.get(h, cfg.base)
        ea = ratings.get(a, cfg.base)
        e_home = expected_home(eh, ea, cfg)
        hs, as_ = g["home_score"], g["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            rows.append({**g, "elo_home": eh, "elo_away": ea, "e_home": e_home})
            continue
        margin = hs - as_
        result_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        # winner-perspective pre-game diff (home carries HFA)
        if margin >= 0:
            elo_diff_winner = (eh + cfg.hfa_elo) - ea
        else:
            elo_diff_winner = ea - (eh + cfg.hfa_elo)
        mult = mov_multiplier(margin, elo_diff_winner)
        delta = cfg.k * mult * (result_home - e_home)
        ratings[h] = eh + delta
        ratings[a] = ea - delta
        rows.append({**g, "elo_home": eh, "elo_away": ea, "e_home": e_home})
    return EloResult(games=pd.DataFrame(rows), final=dict(ratings))
