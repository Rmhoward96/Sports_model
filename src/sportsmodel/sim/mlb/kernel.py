"""MLB play-by-play Monte Carlo kernel (scalar, correctness-first).

Simulates plate appearances through a base-out state machine using empirical
advancement tables, a sampled pitcher hook, and a times-through-order penalty.
A vectorized numpy version (equivalence-tested) replaces the hot loop later.
"""
from __future__ import annotations

from dataclasses import dataclass

from .advancement import AdvancementTable, TTO_MULT

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


class BaseState:
    __slots__ = ("first", "second", "third")

    def __init__(self, first=-1, second=-1, third=-1):
        self.first, self.second, self.third = first, second, third

    def occ(self) -> int:
        return (1 if self.first >= 0 else 0) | (2 if self.second >= 0 else 0) | (4 if self.third >= 0 else 0)

    def runners(self) -> int:
        return (self.first >= 0) + (self.second >= 0) + (self.third >= 0)


def _fill_from_mask(state: "BaseState", end_occ: int, batter_idx: int, lead_first: bool):
    """Place surviving runners + batter onto the bases named by end_occ.

    Survivors advance in order (3rd, 2nd, 1st are the closest-to-home); the batter
    occupies the lowest set base that isn't taken by a survivor. This gives a
    deterministic, consistent identity mapping for any abstract end state.
    """
    # collect existing runners nearest-to-home first, then the batter last
    survivors = [b for b in (state.third, state.second, state.first) if b >= 0]
    order_slots = [4, 2, 1]  # 3rd, 2nd, 1st bit values, nearest-home first
    state.first = state.second = state.third = -1
    occupied = []
    for bit in order_slots:
        if end_occ & bit:
            occupied.append(bit)
    # assign survivors to the highest occupied bases, batter to the lowest
    to_place = survivors + [batter_idx]
    # nearest-home bases get the runners who were already furthest along
    for bit, who in zip(occupied, to_place):
        if bit == 4:
            state.third = who
        elif bit == 2:
            state.second = who
        else:
            state.first = who


def resolve_pa(state: "BaseState", batter_idx: int, outcome: int, adv: AdvancementTable, u: float) -> tuple[int, int]:
    if outcome == K:
        return 0, 1
    if outcome == HR:
        runs = state.runners() + 1
        state.first = state.second = state.third = -1
        return runs, 0
    if outcome == BB:
        # force only: batter to first; bump a forced chain
        if state.first < 0:
            state.first = batter_idx
        elif state.second < 0:
            state.second, state.first = state.first, batter_idx
        elif state.third < 0:
            state.third, state.second, state.first = state.second, state.first, batter_idx
        else:
            # bases loaded walk forces in a run
            state.third, state.second, state.first = state.second, state.first, batter_idx
            return 1, 0
        return 0, 0
    # in-play hit or out -> table
    occ = state.occ()
    runners_before = state.runners()
    end_occ, runs = adv.sample(outcome, occ, u)
    runners_after = bin(end_occ).count("1")
    # outs_added from bookkeeping: batter + runners_before must all be accounted for as
    # (scored) or (still on base) or (out). runners_after excludes the batter's own slot.
    accounted_on_base = runners_after
    outs_added = (runners_before + 1) - accounted_on_base - runs
    outs_added = max(0, min(3, outs_added))
    _fill_from_mask(state, end_occ, batter_idx, lead_first=(outcome == S))
    return runs, outs_added
