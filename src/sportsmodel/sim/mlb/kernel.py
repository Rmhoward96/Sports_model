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
    avg_outs: float
    sd_outs: float


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
    # hook is an OUTS-RECORDED threshold: the starter stays in while his own
    # accumulated outs (pline["outs"], the same quantity that becomes his
    # outs_recorded stat) are below this sampled value. Floor of 3 outs (1
    # inning min); top-clamped at 27 (a full 9-inning outing) for stability.
    hook_home = min(27.0, max(3.0, rng.normal(spec.home_starter.avg_outs, spec.home_starter.sd_outs)))
    hook_away = min(27.0, max(3.0, rng.normal(spec.away_starter.avg_outs, spec.away_starter.sd_outs)))
    scores = [0, 0]           # [away, home]
    idx = [0, 0]              # batting-order pointer [away, home]
    bf = [0, 0]               # batters faced by the [away pitcher, home pitcher] (drives TTO only)

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
            # starter_in is governed by his own accumulated outs (pline["outs"]),
            # not batters faced -- this IS the outs-recorded hook.
            starter_in = pline["outs"] < hook
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


# ---------------------------------------------------------------------------
# Vectorized kernel: masked lock-step per PA across all n_sims games at once.
#
# Key simplification vs. the scalar BaseState: box-score stats never read
# runner *identity* (see resolve_pa's docstring caveat -- identity placement
# is approximate by design), only base *occupancy* and outs/runs. So instead
# of tracking first/second/third player indices per sim, we carry a 3-bit
# occupancy mask (bit0=first, bit1=second, bit2=third) per sim, matching the
# `occ()` encoding BaseState/AdvancementTable already use. This lets every
# per-PA operation lift straight to a masked numpy array op with no per-sim
# Python loop.
#
# RNG draw order intentionally does NOT match simulate_scalar (this processes
# all sims lock-step per PA rather than one full game at a time) -- semantics
# match exactly, so aggregate distributions agree within Monte Carlo noise.
# ---------------------------------------------------------------------------

_POPCOUNT = np.array([bin(i).count("1") for i in range(8)], dtype=np.int64)


def _team_tensors(order: list) -> tuple[np.ndarray, np.ndarray]:
    """Precompute (3,9,7) TTO-variant starter vectors and (9,7) bullpen vectors.

    Axis 0 of the starter tensor is times_through - 1 (clamped to 0,1,2), axis 1
    is batting-order slot (0-8), axis 2 is the outcome code (matches _VEC_ORDER).
    """
    n = len(order)
    sp = np.zeros((3, n, 7))
    bp = np.zeros((n, 7))
    for slot, b in enumerate(order):
        for tt in (1, 2, 3):
            v = apply_tto(b.vec_vs_sp, tt)
            sp[tt - 1, slot] = [v[name] for name in _VEC_ORDER]
        bp[slot] = [b.vec_vs_bp[name] for name in _VEC_ORDER]
    return sp, bp


def _build_adv_vec(adv: AdvancementTable) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precompute a padded (56, maxK) cum/end/runs structure keyed by
    outcome_code*8+occ, mirroring AdvancementTable.sample (incl. its missing-key
    fallback) so the vectorized lookup matches it exactly.
    """
    table_entries = adv._table
    max_k = 1
    for cum, _ends, _runs in table_entries.values():
        max_k = max(max_k, len(cum))
    n_rows = 7 * 8  # outcome codes 0-6, occ 0-7 -> key = outcome*8+occ
    cum_mat = np.ones((n_rows, max_k))
    end_mat = np.zeros((n_rows, max_k), dtype=np.int64)
    runs_mat = np.zeros((n_rows, max_k), dtype=np.int64)
    code_to_name = {S: "p_1b", D: "p_2b", T: "p_3b", OUT_INPLAY: "p_out"}
    for code, name in code_to_name.items():
        for occ in range(8):
            row = code * 8 + occ
            entry = table_entries.get((name, occ))
            if entry is None:
                # matches AdvancementTable.sample's missing-key fallback
                end_mat[row, :] = (occ | 1) if code == S else occ
                runs_mat[row, :] = 0
                # cum_mat row stays all-ones (any u<1 selects col 0)
            else:
                cum, ends, runs = entry
                k = len(cum)
                cum_mat[row, :k] = cum
                end_mat[row, :k] = ends
                runs_mat[row, :k] = runs
                if k < max_k:
                    end_mat[row, k:] = ends[-1]
                    runs_mat[row, k:] = runs[-1]
                    # cum_mat tail stays 1.0 (default init) -> never selected
                    # since cum[k-1] is already forced to 1.0 by from_rows
    return cum_mat, end_mat, runs_mat


def _resolve_pa_vec(occ, outcome, u, cum_mat, end_mat, runs_mat):
    """Vectorized resolve_pa: same branches (K/HR/BB/table), occupancy-only."""
    n = len(occ)
    runs = np.zeros(n, dtype=np.int64)
    outs_added = np.zeros(n, dtype=np.int64)
    new_occ = occ.copy()
    runners_before = _POPCOUNT[occ]

    m_k = outcome == K
    outs_added[m_k] = 1  # pure out, occupancy unchanged

    m_hr = outcome == HR
    runs[m_hr] = runners_before[m_hr] + 1
    new_occ[m_hr] = 0

    m_bb = outcome == BB
    loaded = occ == 7
    # force-only chain: if not loaded, always fills the lowest empty base
    # (equivalent to the scalar first/second/third chain check); if loaded,
    # occupancy stays full and a run is forced in.
    bb_new_occ = np.where(loaded, 7, occ | (occ + 1))
    runs[m_bb & loaded] = 1
    new_occ[m_bb] = bb_new_occ[m_bb]

    m_tab = (outcome == S) | (outcome == D) | (outcome == T) | (outcome == OUT_INPLAY)
    key_idx = np.clip(outcome, 0, 6) * 8 + occ
    row_cum = cum_mat[key_idx]
    col = np.argmax(u[:, None] < row_cum, axis=1)
    table_end = end_mat[key_idx, col]
    table_runs = runs_mat[key_idx, col]
    runners_after = _POPCOUNT[table_end]
    table_outs = np.clip((runners_before + 1) - runners_after - table_runs, 0, 3)
    runs[m_tab] = table_runs[m_tab]
    outs_added[m_tab] = table_outs[m_tab]
    new_occ[m_tab] = table_end[m_tab]

    return runs, outs_added, new_occ


def simulate(spec: GameSpec, n_sims: int, rng, max_extra: int = 11) -> GameSims:
    """Vectorized kernel: same signature/semantics as simulate_scalar, but
    processes all n_sims games lock-step per PA using masked numpy ops.
    """
    adv = getattr(spec, "adv", None) or AdvancementTable.from_rows([])
    n = n_sims
    home_order, away_order = spec.home_order, spec.away_order
    home_sp, home_bp = _team_tensors(home_order)
    away_sp, away_bp = _team_tensors(away_order)
    cum_mat, end_mat, runs_mat = _build_adv_vec(adv)

    # hook is an OUTS-RECORDED threshold per sim (see simulate_scalar docstring
    # note above _sim_one's hook_home/hook_away for the rationale); same floor
    # of 3 outs / top clamp of 27, drawn from the same rng as the scalar path.
    hook_home = np.clip(rng.normal(spec.home_starter.avg_outs, spec.home_starter.sd_outs, size=n), 3.0, 27.0)
    hook_away = np.clip(rng.normal(spec.away_starter.avg_outs, spec.away_starter.sd_outs, size=n), 3.0, 27.0)

    scores_home = np.zeros(n, dtype=np.int64)
    scores_away = np.zeros(n, dtype=np.int64)
    idx_home = np.zeros(n, dtype=np.int64)
    idx_away = np.zeros(n, dtype=np.int64)
    bf_away_pitcher = np.zeros(n, dtype=np.int64)  # batters faced by the AWAY pitcher
    bf_home_pitcher = np.zeros(n, dtype=np.int64)  # batters faced by the HOME pitcher
    bf = [bf_away_pitcher, bf_home_pitcher]  # indexed by defense: 0=away, 1=home

    home_box = {m: np.zeros((9, n), dtype=np.int64) for m in _BATTER_MARKETS if m != "hrr"}
    away_box = {m: np.zeros((9, n), dtype=np.int64) for m in _BATTER_MARKETS if m != "hrr"}
    hp = {m: np.zeros(n, dtype=np.int64) for m in ("k", "hits", "outs")}
    ap = {m: np.zeros(n, dtype=np.int64) for m in ("k", "hits", "outs")}

    game_over = np.zeros(n, dtype=bool)
    sim_ids = np.arange(n)

    def play_half(bat_team: int, inning: int, active_game: np.ndarray) -> None:
        order = home_order if bat_team == 1 else away_order
        box = home_box if bat_team == 1 else away_box
        idx = idx_home if bat_team == 1 else idx_away
        defense = 1 - bat_team
        hook = hook_away if defense == 0 else hook_home
        bf_def = bf[defense]
        pline = ap if defense == 0 else hp
        sp_tensor = home_sp if bat_team == 1 else away_sp
        bp_tensor = home_bp if bat_team == 1 else away_bp

        # fresh BaseState per half; extra-inning Manfred-rule runner on 2nd
        occ = np.full(n, 2, dtype=np.int64) if inning > 9 else np.zeros(n, dtype=np.int64)
        outs = np.zeros(n, dtype=np.int64)

        while True:
            active = active_game & (outs < 3)
            if not active.any():
                break
            slot = idx % 9
            times_through = bf_def // 9 + 1
            tto_idx = np.clip(times_through, 1, 3) - 1
            # starter_in governed by his own accumulated outs (pline["outs"]),
            # matching the scalar kernel's outs-recorded hook exactly.
            starter_in = pline["outs"] < hook

            vec_if_sp = sp_tensor[tto_idx, slot]
            vec_if_bp = bp_tensor[slot]
            vec = np.where(starter_in[:, None], vec_if_sp, vec_if_bp)

            u1 = rng.random(n)
            cum = np.cumsum(vec, axis=1)
            cum[:, -1] = 1.0  # guard fp drift, same intent as scalar's fallback
            outcome = np.argmax(u1[:, None] < cum, axis=1)

            u2 = rng.random(n)
            runs, outs_added, new_occ = _resolve_pa_vec(occ, outcome, u2, cum_mat, end_mat, runs_mat)

            is_hit = (outcome == S) | (outcome == D) | (outcome == T) | (outcome == HR)
            tb = np.select([outcome == S, outcome == D, outcome == T, outcome == HR], [1, 2, 3, 4], default=0)
            is_hr = outcome == HR
            is_k = outcome == K

            m = active
            rows, cols = slot[m], sim_ids[m]
            box["hits"][rows, cols] += is_hit[m]
            box["total_bases"][rows, cols] += tb[m]
            box["hr"][rows, cols] += is_hr[m]
            box["rbi"][rows, cols] += runs[m]
            # per-batter "runs" credited only on the batter's own HR (see resolve_pa)
            box["runs"][rows, cols] += is_hr[m]

            if bat_team == 1:
                scores_home[m] += runs[m]
            else:
                scores_away[m] += runs[m]

            sm = m & starter_in  # pitcher line: starter only
            pline["k"][sm] += is_k[sm]
            pline["hits"][sm] += is_hit[sm]
            pline["outs"][sm] += outs_added[sm]

            outs[m] += outs_added[m]
            bf_def[m] += 1
            idx[m] += 1
            occ[m] = new_occ[m]

    inning = 1
    while True:
        active_game = ~game_over
        play_half(0, inning, active_game)  # away bats (top)
        play_half(1, inning, active_game)  # home bats (bottom)

        decided_now = active_game & (inning >= 9) & (scores_home != scores_away)
        game_over |= decided_now
        remaining = active_game & ~decided_now
        if inning >= 9 + max_extra:  # hard cap
            tie = remaining & (scores_home == scores_away)
            scores_home[tie] += 1  # break ties at the cap deterministically
            game_over |= remaining

        if game_over.all():
            break
        inning += 1

    bstats = {}
    for order, box in ((home_order, home_box), (away_order, away_box)):
        for slot, b in enumerate(order):
            line = {m: box[m][slot].astype(np.int32) for m in box}
            line["hrr"] = (box["hits"][slot] + box["runs"][slot] + box["rbi"][slot]).astype(np.int32)
            bstats[b.player_id] = line

    pstats = {
        spec.home_starter.player_id: {m: hp[m].astype(np.int32) for m in hp},
        spec.away_starter.player_id: {m: ap[m].astype(np.int32) for m in ap},
    }

    return GameSims(scores_home.astype(np.int32), scores_away.astype(np.int32), bstats, pstats)
