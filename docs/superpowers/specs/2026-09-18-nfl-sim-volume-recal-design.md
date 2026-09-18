# Sub-project B.3 — NFL sim volume + efficiency recalibration (design spec)

**Status:** draft for review · **Date:** 2026-09-18
**Program:** A(done) → B(built, dormant) → B.2(done: usage model; receptions CALIBRATED) → **B.3 (this: structural volume recal for the yardage markets)** → C(props productization).

## Why B.3 exists (the B.2 gate + the B.3 diagnosis)

B.2 calibrated **receptions** (walk-forward propable p50 .456, p90 .893) and hugely
improved every market vs B.1, but the **yardage markets** would not calibrate by
constant tuning. 13 walk-forward sweeps established *why* — a coupled structural
problem, not a tuning gap:

| market | sim mean | actual mean | bias | note |
|---|---|---|---|---|
| pass_yds | 293 | 230 | **+63 (over)** | team pass volume too high |
| rush_yds | 44.7 | 51.2 | **−6.5 (under)** | split too pass-heavy |
| rec_yds | 42.2 | 44.0 | −1.7 (≈right) | mean right, dist too skewed (median 28 vs 36) |
| receptions | 3.70 | 3.95 | −0.25 | calibrated (p50 .456) |

**Root cause:** the kernel converts *every* pass play into a target. Real NFL pass
plays ≈ 34 attempts + ~2.5 sacks; the sim treats all ~38 as targets. Those ~4
phantom targets/game spill to the roster tail → pass_yds over; and because the
pass/run split is correspondingly too pass-heavy, rush_yds is under. **Receptions
only looks calibrated because a compensating too-low effective catch/share offsets
the excess attempts** — a compensating-error tangle. Proof of coupling: every
lever that fixed one market (rec-yds shape, pass-volume scale, target/carry
sharpen, active-set trim) either overshot that market's mean (MAE up), broke its
p90, or degraded a coupled market (receptions). No independent constant config
calibrates all four.

## Goal

Recalibrate the sim's offensive **volume** (attempts, rushes, sacks, drives) to
reality so the yardage-market means become honest, then re-fit the **efficiency**
layer so receptions stays calibrated for the *right* reason — iterating to
shippable calibration on rec_yds/rush_yds/pass_yds **without regressing
receptions**. A market that reaches the bar earns C; one that plateaus gets
surfaced with evidence (same iterate-to-ship discipline as B.2).

## Non-goals

- Not C (no prop-odds ingestion / board / grading yet).
- Not a claim of edge on shopped yardage markets — B.3 is calibration; edge is
  forward CLV (same discipline as the game +EV pilot).
- No new data dependency — nflverse pbp only (already used), leakage-free.
- Does not touch A / desk / +EV / CFB / MLB, and does not activate B.

## Approach — PHASED, gate between phases (dependency-ordered)

Volume must be honest before efficiency can be re-fit, so the phases are ordered
and gated. Anchor every target to **per-team nflverse season aggregates**
computed strictly before the game's (season, week) — reproducible in CI, no new
dependency, captures team pace/pass-identity (vs flat league constants).

### Phase 1 — model sacks (the keystone)
Split pass plays into **sacks** (QB yardage loss, ~−6 yds, NO target) vs
**attempts** (targets). Derive a per-team sack rate (sacks / dropbacks) from pbp,
leakage-free. In the kernel, `n_dropbacks` → `n_sacks` + `n_attempts`; only
`n_attempts` become targets; sacks subtract QB yardage (pass_yds) and consume a
play without a rush. Expected effect: team targets → ~34, tail deflates, pass_yds
bias down.

### Phase 2 — calibrate team volume
Recalibrate drives/game and plays/drive (currently `_PLAYS_PER_DRIVE=6`,
`_MIN_DRIVES_PER_GAME=6`) so per-team **attempts ≈ real** and **rush ≈ real**
against nflverse season aggregates. Fixes the pass-heavy split → lifts rush_yds.
Prefer deriving team play/pass/rush volume from the team's own pbp rates over a
global constant where it matters.

**GATE A (after P1+P2):** re-run the propable walk-forward. Read pass_yds /
rush_yds bias + coverage and confirm **receptions did not regress**. Decide
whether P3/P4 are still needed (they may not be if volume alone calibrates).

### Phase 3 — re-fit efficiency (only if GATE A shows it's needed)
With attempts honest, the compensating errors surface: re-fit `catch_rate` and
target/carry shares so receptions & rec_yds means stay calibrated for the right
reason (not via offsetting the old excess attempts). Anchor catch_rate to real
per-team completion% ; keep shares from the B.2 active-usage model.

### Phase 4 — per-position yard skew (only if GATE A/B shows it's needed)
Tune the per-play yardage shapes **per position** (`_REC_YDS_SHAPE`,
`_RUSH_YDS_SHAPE`, possibly position-specific) against the now-correct means to
pull rec_yds/rush_yds **p50** toward .50 without breaking p90. (B.2 showed a
single global shape trades p50 for p90; per-position may separate them.)

**GATE B (final):** full 4-season walk-forward (N≈2000) via the real production
script. Per-market coverage bands + MAE + signed bias, receptions no-regress.

## Calibration targets (iterate toward shippable, not one-shot)

Propable players (pass att≥10, rush carries≥5, rec/receptions targets≥3),
walk-forward + leakage-free:

| market | MAE (near noise floor) | P50 | P90 |
|---|---|---|---|
| pass_yds | ≤ 65 | .45–.55 | .85–.93 |
| rush_yds | ≤ 24 | .45–.55 | .85–.93 |
| rec_yds | ≤ 20 | .45–.55 | .85–.93 |
| receptions | ≤ 1.5 (**no regress from B.2**) | **.45–.55 (guardrail)** | .85–.93 |

MAE targets sit near the irreducible single-game noise floor, so **coverage is
the shippability signal**, MAE a sanity check. Missing = refine; plateau =
surface with evidence, per market. **Receptions is a hard no-regress guardrail** —
no phase may push it out of band.

## Components / files (anticipated)

- `src/sportsmodel/sim/nfl/rates.py` — extend `team_rates_from_pbp` (or a new
  helper) to compute per-team **sack rate**, **attempts/game**, **rush/game**,
  **completion%**, leakage-free, from pbp season aggregates.
- `src/sportsmodel/sim/nfl/spec.py` — `TeamRates` gains sack_rate / volume fields.
- `src/sportsmodel/sim/nfl/kernel.py` — `_simulate_team_drives` splits dropbacks
  into sacks vs attempts; sacks subtract QB pass_yds + consume a play; volume
  from the calibrated rates. Kernel structure otherwise unchanged.
- `scripts/backtest_sim_nfl.py` — gate already computes propable coverage; add
  signed-bias + receptions-no-regress reporting (or reuse the sweep harness).
- Tests under `tests/sim/nfl/` per change.

## Open questions

- Sack yardage model: fixed ~−6 or sampled? (start fixed, refine if needed.)
- Does calibrating volume from team pbp rates suffice, or is a per-team
  attempts/rush *scale-to-target* needed? (measure at GATE A.)
- Whether pass_yds ever reaches the band (shopped market + QB=Σreceivers
  structure) — revisit after GATE A with evidence; may be the market that
  plateaus.
