"""MLB play-by-play Monte Carlo kernel (scalar, correctness-first).

Simulates plate appearances through a base-out state machine using empirical
advancement tables, a sampled pitcher hook, and a times-through-order penalty.
A vectorized numpy version (equivalence-tested) replaces the hot loop later.
"""
from __future__ import annotations

from dataclasses import dataclass

from .advancement import TTO_MULT

# per-PA vector order -> outcome codes
_VEC_ORDER = ("p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out")
BB, K, S, D, T, HR, OUT_INPLAY = 0, 1, 2, 3, 4, 5, 6
_CODE = {"p_bb": BB, "p_k": K, "p_1b": S, "p_2b": D, "p_3b": T, "p_hr": HR, "p_out": OUT_INPLAY}


@dataclass
class Batter:
    player_id: int
    vec_vs_sp: dict
    vec_vs_bp: dict


@dataclass
class Pitcher:
    player_id: int
    avg_bf: float
    sd_bf: float


@dataclass
class GameSpec:
    home_order: list
    away_order: list
    home_starter: Pitcher
    away_starter: Pitcher


def sample_outcome(vec: dict, u: float) -> int:
    acc = 0.0
    for name in _VEC_ORDER:
        acc += vec[name]
        if u < acc:
            return _CODE[name]
    return OUT_INPLAY


def apply_tto(vec: dict, times_through: int) -> dict:
    """Worsen a starter's vector for the 2nd/3rd+ time through the order; renormalize."""
    idx = min(times_through, 3) - 1  # 1->0, 2->1, 3+->2
    if idx <= 0:
        return vec
    scaled = {k: vec[k] * TTO_MULT[k][idx] for k in _VEC_ORDER}
    tot = sum(scaled.values())
    return {k: v / tot for k, v in scaled.items()}
