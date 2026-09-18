# NFL Sim Volume + Efficiency Recalibration (B.3) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Recalibrate the sim's offensive volume so the yardage-market means become honest — anchor per-game pass ATTEMPTS (excluding sacks) and rush ATTEMPTS to the team's real per-game counts, replacing the `n_plays × pass_rate` derivation that over-counts attempts — then, if the gate needs it, re-fit efficiency and per-position skew, all without regressing the B.2-calibrated receptions market.

**Architecture:** `rates.team_rates_from_pbp` computes new leakage-free per-team per-game rates (pass_att_pg, rush_att_pg, sack_rate, completion_pct) from nflverse pbp; `TeamRates` carries them; `kernel._simulate_team_drives` uses attempts/rushes from those rates (× game_env) instead of `n_plays × pass_rate`. Phased with a gate after volume before touching efficiency/skew. Builds on the dormant `sim-nfl-engine` branch.

**Tech Stack:** Python, numpy, pandas, nfl_data_py, pytest, uv.

**Spec:** docs/superpowers/specs/2026-09-18-nfl-sim-volume-recal-design.md

## Global Constraints

- **NFL only.** Branch `sim-nfl-engine` (B engine, dormant). Do NOT activate B (no migration, no workflow enable). Do not touch A/desk/+EV/CFB/MLB.
- **Leakage-free:** every rate uses only pbp rows strictly before the game's (season, week) — `team_rates_from_pbp` already applies `_before_cutoff`; keep it.
- **nflverse passing_yards is GROSS** (sacks tracked separately) — the sim must NOT subtract sack yardage from pass_yds. Modeling sacks means fewer *targets*, nothing else.
- **Receptions is a hard no-regress guardrail:** no change may push receptions out of its B.2-calibrated band (walk-forward propable p50 .45–.55, p90 .85–.93, MAE ≤ ~1.8).
- `import_pbp_data` returns all columns, so `sack`, `pass_attempt`, `rush_attempt`, `complete_pass`, `posteam`, `play_type`, `game_id`, `season`, `week` are available — no fetch change.
- Pure functions (rates, kernel helpers) have no IO. Test + commit per task; match style.
- Kernel back-compat: existing `TeamRates(...)` constructions in tests omit the new fields, so the new fields MUST have defaults and the kernel MUST fall back to the legacy `n_plays × pass_rate` path when `pass_att_pg <= 0`.

## File Structure

- `src/sportsmodel/sim/nfl/rates.py` — extend `team_rates_from_pbp` to compute the new per-game rates.
- `src/sportsmodel/sim/nfl/spec.py` — `TeamRates` gains the new fields (with defaults).
- `src/sportsmodel/sim/nfl/kernel.py` — `_simulate_team_drives` uses attempts/rush from the new rates.
- Tests under `tests/sim/nfl/`.

---

### Task 1: compute per-game volume rates in team_rates_from_pbp

**Files:** `src/sportsmodel/sim/nfl/rates.py`; Test `tests/sim/nfl/test_rates.py` (extend).

**Interfaces — Produces (added to each `TeamRates` returned by `team_rates_from_pbp`):**
- `pass_att_pg: float` — real pass ATTEMPTS per game = count of rows where `play_type=="pass"` AND `sack != 1`, divided by `n_games`. (Attempts exclude sacks; incompletions have `play_type=="pass"`, `sack==0` and ARE counted.)
- `rush_att_pg: float` — rush attempts per game = count of `play_type=="run"` rows / `n_games` (includes QB scrambles, which are rushes).
- `sack_rate: float` — sacks / pass PLAYS = count(`sack==1`) / count(`play_type=="pass"`), guarded /0. (Retained for later QB modeling; not consumed by the kernel in this plan.)
- `completion_pct: float` — completions / attempts = count(`complete_pass==1`) / count(`play_type=="pass" & sack!=1`), guarded /0. (For Task 5 efficiency re-fit.)

All computed on the SAME `_before_cutoff` frame and per-`posteam` group already in `team_rates_from_pbp`. `n_games = team_df["game_id"].nunique()`.

- [ ] **Step 1: failing test** — build a small pbp fixture (a couple game_ids for one posteam) with known counts of pass/run/sack/complete_pass rows, assert `pass_att_pg` excludes sacks, `rush_att_pg` counts runs, `sack_rate` and `completion_pct` are the expected ratios, and all are computed only from strictly-before-cutoff rows (include an at/after-cutoff row that must be ignored).
- [ ] Steps 2–5 (fail, implement, pass, full suite, commit `feat(sim/nfl): per-game attempt/sack/completion rates`).

---

### Task 2: add volume fields to TeamRates

**Files:** `src/sportsmodel/sim/nfl/spec.py`; Test `tests/sim/nfl/test_spec.py` (extend).

**Interfaces:** `TeamRates` gains, AFTER the existing 4 fields, with defaults:
```python
    pass_att_pg: float = 0.0
    rush_att_pg: float = 0.0
    sack_rate: float = 0.0
    completion_pct: float = 0.0
```
Defaults keep existing `TeamRates(drive_outcomes=..., pass_rate=..., drives_per_game=..., rz_td_rate=...)` constructions valid.

- [ ] **Step 1: failing test** — construct a `TeamRates` with only the original 4 fields and assert the new fields default to 0.0; construct one with all 8 and assert they round-trip.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): TeamRates carries per-game volume fields`).

---

### Task 3: kernel uses real attempts/rush volume (the fix)

**Files:** `src/sportsmodel/sim/nfl/kernel.py`; Test `tests/sim/nfl/test_kernel_simulate.py` and/or `test_kernel_attribution.py` (extend).

**Interfaces:** In `_simulate_team_drives` (kernel.py ~295-309), replace the play-count derivation:
```python
    # OLD:
    # n_plays = round(n_drives * _PLAYS_PER_DRIVE)
    # n_pass = round(n_plays * off.pass_rate)
    # n_rush = n_plays - n_pass
    # NEW: anchor to real per-game attempts (excludes sacks), scaled by the
    # same game_env that scales drives, so volume and scoring move together.
    if off.pass_att_pg > 0.0 or off.rush_att_pg > 0.0:
        n_pass = max(0, round(off.pass_att_pg * game_env))
        n_rush = max(0, round(off.rush_att_pg * game_env))
    else:
        # legacy fallback (keeps pre-B.3 tests/specs without volume fields working)
        n_plays = round(n_drives * _PLAYS_PER_DRIVE)
        n_pass = round(n_plays * off.pass_rate)
        n_rush = n_plays - n_pass
```
`n_pass` here is the TARGET count passed to `attribute_offense` (unchanged call). Drive/points logic (n_drives, sample_drive, n_off_tds) is UNCHANGED. Do NOT subtract sack yardage anywhere (passing_yards is gross).

- [ ] **Step 1: failing test** — with a `TeamRates` carrying `pass_att_pg=34`, `rush_att_pg=27` and a large `n_sims`, assert the mean total targets attributed (sum of receptions across players / catch_rate, or instrument via a spec with catch_rate=1.0 so receptions==targets) ≈ 34 and mean rush attributions ≈ 27 (within Monte-Carlo tolerance), and that a `TeamRates` WITHOUT volume fields (pass_att_pg=0) still runs via the legacy path (existing attribution test still green). Use catch_rate=1.0 / single-player specs to make counts observable.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): drive box volume from real per-game attempts`).

---

### Task 4: GATE A — re-gate volume-only, decide P3/P4

Not a code task — the controller runs the full propable walk-forward (real `scripts/backtest_sim_nfl.py`, DESK_SIM_N≈2000, VALIDATION_SEASONS) plus the signed-bias readout (reuse the diagnostic harness), and records per market: MAE, signed bias, p50, p90. Compare to the pre-B.3 B.2 baseline (pass +63/.677/.981, rush −6.5/.342/.825, rec_yds −1.7/.396/.861, receptions calibrated .456/.893).

Decision rules (record in ledger):
- **Receptions regressed out of band?** → Task 5 (efficiency re-fit) is REQUIRED.
- **pass_yds / rush_yds mean now honest but p50 still <.45?** → Task 6 (per-position skew) candidate.
- **A market reaches the band** → eligible for C; **plateaus** → surface with evidence.

---

### Task 5 (CONDITIONAL on GATE A): re-fit efficiency

**Files:** `src/sportsmodel/sim/nfl/kernel.py` and/or `usage.py`; Tests alongside.

**Trigger:** GATE A shows receptions/rec_yds regressed because fewer attempts (now real ~34 vs old ~38) reduced completions across the board.

**Behavior:** restore featured completion volume for the RIGHT reason — anchor the team's completion level to real `completion_pct` (from Task 1) rather than the player-level catch_rates that were compensating for inflated attempts. Concretely (implementer picks the cleaner seam): either scale player catch_rate so team completions ≈ `completion_pct × pass_att_pg`, or apply `completion_pct` as a team-level completion gate before per-player catch allocation. Keep it mean-preserving where possible; re-gate. Do NOT break receptions further.

- [ ] **Step 1: failing test** — a spec whose players' raw catch_rates would yield the wrong team completion total is corrected so simulated team completions ≈ `completion_pct × pass_att_pg` (within tolerance).
- [ ] Steps 2–5 (implement; commit `fit(sim/nfl): anchor completions to real completion pct`).

---

### Task 6 (CONDITIONAL on GATE A/B): per-position yard skew

**Files:** `src/sportsmodel/sim/nfl/kernel.py`; Tests alongside.

**Trigger:** means are honest but rec_yds/rush_yds **p50** still <.45 (single-game skew, as in B.2).

**Behavior:** make the per-play yardage shape position-aware (e.g., distinct shapes for WR/TE receptions vs RB receptions, and RB carries), tuned against the propable gate to lift p50 toward .50 without dropping p90 below .85. If a global/per-position shape change alters the `_skewed_play_yards` tail, update `test_kernel_attribution.py`'s tail assertion (max > k·mean) to the new shape — legitimately, the tail changed.

- [ ] **Step 1: failing test** — per-position shape lookup returns the configured shape for a given position; `_skewed_play_yards` still preserves the mean (scale = (mean−floor)/shape) at the new shapes.
- [ ] Steps 2–5 (implement; commit `fit(sim/nfl): per-position yardage skew`).

---

### Task 7: final GATE B + ledger

Not a code task — full 4-season walk-forward (N≈2000) via the real production script. Confirm per-market coverage + MAE + bias, **receptions no-regress**. Record final numbers + which markets reached the band (eligible for C) vs plateaued (surfaced with evidence). Then whole-branch review per SDD.

## Self-Review

- **Spec coverage:** P1 sacks + P2 volume are unified into real-attempt anchoring (T1–T3: attempts exclude sacks, rush from real counts) — the spec's keystone; GATE A (T4); efficiency re-fit (T5, conditional); per-position skew (T6, conditional); final gate (T7). Receptions no-regress guardrail stated in Global Constraints and checked at T4/T7. ✔
- **Type consistency:** `TeamRates` new fields are floats with defaults (T2), produced by `team_rates_from_pbp` (T1), consumed in `_simulate_team_drives` (T3) with a legacy fallback. `attribute_offense` signature unchanged. ✔
- **Leakage:** all rates on the existing `_before_cutoff` frame (T1). ✔
- **No placeholders:** T1–T3 have concrete columns, formulas, and code; T5/T6 are conditional with concrete triggers + interfaces; gate tasks specify the exact measurement.
- **Ordering:** 1 (rates) → 2 (TeamRates) → 3 (kernel) → 4 (GATE A) → [5 efficiency, 6 skew — conditional] → 7 (GATE B + review).
- **Correctness guard:** passing_yards is gross → no sack-yardage subtraction (Global Constraints); this is the most likely implementer mistake, called out explicitly.
