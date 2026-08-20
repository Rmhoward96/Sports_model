"""MLB play-by-play Monte Carlo kernel (scalar, correctness-first).

Simulates plate appearances through a base-out state machine using empirical
advancement tables, a sampled pitcher hook, and a times-through-order penalty.
A vectorized numpy version (equivalence-tested) replaces the hot loop later.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..engine import GameSims
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


_BATTER_MARKETS = ("hits", "total_bases", "hr", "runs", "rbi", "hrr")


def _new_box(order):
    return {b.player_id: {m: 0 for m in _BATTER_MARKETS} for b in order}


def _tb_for(outcome: int) -> int:
    return {S: 1, D: 2, T: 3, HR: 4}.get(outcome, 0)


def _sim_one(spec: GameSpec, adv: AdvancementTable, rng, max_extra: int) -> tuple[int, int, dict, dict, dict, dict]:
    """One full game. Returns (home_runs, away_runs, home_box, away_box, hp_line, ap_line)."""
    home_box, away_box = _new_box(spec.home_order), _new_box(spec.away_order)
    # pitcher stat lines (starter only): K, hits allowed, outs
    hp = {"k": 0, "hits": 0, "outs": 0}
    ap = {"k": 0, "hits": 0, "outs": 0}
    hook_home = max(12, rng.normal(spec.home_starter.avg_bf, spec.home_starter.sd_bf))
    hook_away = max(12, rng.normal(spec.away_starter.avg_bf, spec.away_starter.sd_bf))
    scores = [0, 0]           # [away, home]
    idx = [0, 0]              # batting-order pointer [away, home]
    bf = [0, 0]               # batters faced by the [away pitcher, home pitcher]

    def half(bat_team, inning):
        # bat_team: 0 away, 1 home. Defense is the other team; its starter faces batters.
        order = spec.home_order if bat_team == 1 else spec.away_order
        box = home_box if bat_team == 1 else away_box
        defense = 0 if bat_team == 1 else 1
        hook = hook_away if defense == 0 else hook_home
        pline = ap if defense == 0 else hp
        state = BaseState()
        if inning > 9:
            # extra-innings runner-on-2nd (Manfred) rule: start the half-inning with a
            # runner on 2nd. We deliberately do NOT attribute this to a specific earlier
            # batter's "runs" stat (see resolve_pa identity caveat) -- any nonnegative
            # placeholder index works since only occupancy/outs/runs feed the box score.
            state.second = (idx[bat_team] - 1) % 9
        outs = 0
        while outs < 3:
            b = order[idx[bat_team] % 9]
            faced = bf[defense]
            starter_in = faced < hook
            times_through = faced // 9 + 1
            vec = apply_tto(b.vec_vs_sp, times_through) if starter_in else b.vec_vs_bp
            u = rng.random()
            outcome = sample_outcome(vec, u)
            u2 = rng.random()
            runs, outs_added = resolve_pa(state, idx[bat_team] % 9, outcome, adv, u2)
            # box score
            if outcome in (S, D, T, HR):
                box[b.player_id]["hits"] += 1
                box[b.player_id]["total_bases"] += _tb_for(outcome)
                if outcome == HR:
                    box[b.player_id]["hr"] += 1
            box[b.player_id]["rbi"] += runs
            scores[bat_team] += runs
            # crude runs-scored credit: distribute `runs` to the batter's team tally only;
            # per-batter "runs" credited to the batter on his own HR, else left aggregate
            # (BaseState runner identity is approximate -- see Task 6 carried decision).
            if outcome == HR:
                box[b.player_id]["runs"] += 1
            # pitcher line (starter only)
            if starter_in:
                if outcome == K:
                    pline["k"] += 1
                if outcome in (S, D, T, HR):
                    pline["hits"] += 1
                pline["outs"] += outs_added
            outs += outs_added
            bf[defense] += 1
            idx[bat_team] += 1

    inning = 1
    while True:
        half(0, inning)  # away bats (top)
        half(1, inning)  # home bats (bottom)
        if inning >= 9 and scores[1] != scores[0]:
            break
        if inning >= 9 + max_extra:  # hard cap
            if scores[0] == scores[1]:
                scores[1] += 1  # break ties at the cap deterministically
            break
        inning += 1

    # finalize hrr
    for box in (home_box, away_box):
        for pid, s in box.items():
            s["hrr"] = s["hits"] + s["runs"] + s["rbi"]
    return scores[1], scores[0], home_box, away_box, hp, ap


def simulate_scalar(spec: GameSpec, n_sims: int, rng, max_extra: int = 11) -> GameSims:
    # spec may carry an optional `adv` table (Task 10 wires it); default to an empty one.
    adv = getattr(spec, "adv", None) or AdvancementTable.from_rows([])
    hs = np.zeros(n_sims, dtype=np.int32)
    as_ = np.zeros(n_sims, dtype=np.int32)
    bstats = {b.player_id: {m: np.zeros(n_sims, np.int32) for m in _BATTER_MARKETS}
              for b in (*spec.home_order, *spec.away_order)}
    pstats = {spec.home_starter.player_id: {m: np.zeros(n_sims, np.int32) for m in ("k", "hits", "outs")},
              spec.away_starter.player_id: {m: np.zeros(n_sims, np.int32) for m in ("k", "hits", "outs")}}
    for i in range(n_sims):
        hr_, ar_, hbox, abox, hp, ap = _sim_one(spec, adv, rng, max_extra)
        hs[i], as_[i] = hr_, ar_
        for box in (hbox, abox):
            for pid, s in box.items():
                for m in _BATTER_MARKETS:
                    bstats[pid][m][i] = s[m]
        for pid, s in ((spec.home_starter.player_id, hp), (spec.away_starter.player_id, ap)):
            for m in ("k", "hits", "outs"):
                pstats[pid][m][i] = s[m]
    return GameSims(hs, as_, bstats, pstats)
