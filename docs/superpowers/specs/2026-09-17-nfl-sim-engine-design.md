# Sub-project B — NFL drive-based Monte Carlo sim engine (design spec)

**Status:** draft for review · **Date:** 2026-09-17
**Program:** A (done) → **B (this)** → C (NFL player props). B is the foundation
C builds on; it also delivers the correlation + disagreement signals.

## Goal

A player-level Monte Carlo that, per NFL game, produces **N simulated games**,
each with a final score **and** a per-player stat line. Aggregated, that gives:
1. A **correlated** margin+total+win distribution (same sims → real
   margin↔total correlation, for SGP pricing in C and richer totals).
2. **Player stat distributions** (pass/rush/rec yards, receptions, TDs) — the
   raw material C turns into prop picks.
3. An **analytic-vs-sim disagreement signal** per game, as a desk conviction
   modifier.

## Honest framing (carried from A)

The sim will **not** beat the market on game lines — same rating-grade inputs,
same ceiling A just re-confirmed. Its value is **props** (a softer market where
a calibrated player sim can find edges), plus correlation and disagreement. So B
runs **alongside** the analytic game-line model; it does **not** replace the
predictions the desk/board already use for ML/spread/total.

## Reuse (this is why B is tractable)

`src/sportsmodel/sim/engine.py` is already sport-agnostic and provides the
entire output layer: `GameSims` (raw per-sim arrays), `home_win_prob`,
`total_pmf`, `margin_pmf`, `pred_scores`, and **`player_prop_dists`**. The MLB
path (`sim/mlb/kernel.py::simulate_scalar`) shows the pattern. **B = a new NFL
kernel that fills a `GameSims`**, consumed by the existing helpers. No new output
plumbing.

## Architecture — drive-based sim

A simulated game is a sequence of **drives** alternating possession:

1. **Drive count / pace** — plays and drives per team from each team's pace
   (prior-data plays/drive-per-game), with light variance.
2. **Drive outcome** — each drive draws from a multinomial over
   {TD, FG, punt, turnover, downs, end-of-half} whose probabilities come from
   the **offense's** drive-outcome rates adjusted by the **defense's** rates
   (a simple offense-vs-defense strength combine; start field position fixed at
   ~own 25 for v1, no full field-position chain). Score: TD = 7 (2-pt ignored
   v1), FG = 3.
3. **Player attribution** — within a game, the offense's plays split
   pass/run (team pass rate); targets/carries allocated by **usage shares**
   (target share, carry share, snap share); yards drawn from **per-player
   efficiency distributions** (yards/target, yards/carry, catch rate); TDs
   allocated by red-zone/TD share. Player stat arrays accumulate per sim.

v1 is deliberately drive-level, **not** play-by-play down-and-distance — enough
structure for correlated scores and realistic player lines, far less
calibration surface than a full PBP kernel.

### Inputs (nflverse, leakage-free / walk-forward)

All from nflverse, aggregated using only games **prior** to the one being
simulated (same discipline as the ratings model):
- **Team rates** — drive-outcome distribution, pass/run rate, plays & drives per
  game (pace), red-zone TD rate: from `import_pbp_data` aggregated by team
  (offense and defense).
- **Player usage** — target/carry/snap share, routes: `import_weekly_data` +
  `import_snap_counts` + `import_depth_charts`.
- **Player efficiency** — yards/target, yards/carry, catch rate, TD rate:
  `import_weekly_data` / `import_pbp_data`.
- **Availability** — the desk's nflverse injury report zeroes out OUT players and
  reallocates their usage share to teammates (ties the sim to real inactives).

## Deliverables

- `src/sportsmodel/sim/nfl/kernel.py` — `simulate_game(spec, n_sims, rng) ->
  GameSims` (fills scores + player stat arrays).
- `src/sportsmodel/sim/nfl/inputs.py` — build the per-game spec (team rate
  tables + player usage/efficiency) from the leakage-free nflverse aggregates.
- `src/sportsmodel/sim/nfl/rates.py` (or `build_*`) — the nflverse → rate-table
  aggregation.
- `scripts/backtest_sim_nfl.py` — walk-forward validation (below).
- `scripts/generate_sim_nfl.py` — run the sim for the upcoming slate; write
  game-level outputs (sim win/margin/total + disagreement) and (for C) player
  stat distributions.
- `db/migration_nfl_sim.sql` — a `nfl_sim_current` table/view for game-level sim
  outputs + disagreement; player-dist storage may defer to C.
- Reuse `sim/engine.py` unchanged where possible (extend only if a needed
  aggregation is missing).

## The disagreement signal

Per game: `disagreement = |analytic_home_win_prob − sim_home_win_prob|` (and/or
margin gap). Surfaced to the desk bundle so a large analytic-vs-sim divergence
**caps conviction** (a game the two methods disagree on is lower-confidence).
Exact combine rule TBD (open question); v1 can start as a displayed flag +
a conviction cap above a threshold.

## Validation (ship gate)

Walk-forward, NFL only:
- **Game level** — sim win-prob Brier and margin/total MAE vs actuals, compared
  to the analytic model. Gate: **comparable**, not necessarily better (if the
  sim is much worse it's miscalibrated and not fit to ship). Sanity: sim
  margin↔total correlation ≈ empirical.
- **Player level (the real gate for C)** — for pass/rush/rec yards and
  receptions: calibration of the sim's player distributions (quantile coverage /
  PIT) and mean-projection MAE vs actual, walk-forward. Props are only worth
  productizing (C) if these distributions are calibrated.

## Non-goals

- Not play-by-play fidelity (v1 is drive-level).
- Not replacing the analytic game-line prediction or the Pinnacle-anchored +EV
  base prob.
- Not prop odds ingestion / prop board / prop grading — that's C.

## Open questions

- **n_sims & perf** — target (e.g. 10k/game) and whether the scalar per-sim loop
  (like MLB) is fast enough for ~16 games/week, or needs vectorizing.
- **Player markets in scope for v1** — pass/rush/rec yds + receptions + anytime
  TD? (Align with what C will price.)
- **Disagreement combine** — display-only, conviction cap, or a bounded nudge?
- **Sim output storage** — new `nfl_sim_current` table vs extending
  `game_predictions`; defer player-dist storage to C or land it now.
- **Drive-model fidelity** — fixed own-25 start (v1) vs a light field-position
  chain, if v1 calibration is weak.
