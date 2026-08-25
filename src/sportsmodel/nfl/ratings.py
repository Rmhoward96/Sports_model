from __future__ import annotations
from dataclasses import dataclass
from .elo import EloConfig, elo_expected_margin

@dataclass(frozen=True)
class BlendConfig:
    w_sos: float = 0.0
    srs_min_games: int = 4

def expected_margin(elo_home, elo_away, srs_home, srs_away,
                    games_home, games_away, elo_cfg: EloConfig,
                    blend_cfg: BlendConfig) -> float:
    elo_m = elo_expected_margin(elo_home, elo_away, elo_cfg)
    enough = (games_home >= blend_cfg.srs_min_games
              and games_away >= blend_cfg.srs_min_games)
    if blend_cfg.w_sos <= 0 or not enough or srs_home is None or srs_away is None:
        return elo_m
    hfa_points = elo_cfg.hfa_elo / 25.0
    srs_m = (srs_home - srs_away) + hfa_points
    w = blend_cfg.w_sos
    return (1 - w) * elo_m + w * srs_m
