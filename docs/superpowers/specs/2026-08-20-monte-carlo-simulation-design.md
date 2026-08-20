# Monte Carlo Simulation Engine — Design

**Date:** 2026-08-20
**Status:** Approved design, pre-implementation
**Author:** Ryan + Claude

## Goal

Replace the closed-form game/prop math with a **play-by-play Monte Carlo
simulator** that produces one internally-consistent, correlated outcome for the
whole game — so game totals get proper dispersion, win probability respects
variance, and every player prop (including HRR and outs_recorded, currently
approximated) falls out of the same simulation. Built as a **sport-agnostic
framework with a pluggable, sport-specific kernel** so NBA/NFL can reuse
everything except the kernel later.

## Why simulation, and why it generalizes

The single biggest accuracy lever is a *joint, correlated* game outcome: a
batter's hits correlate with team runs, RBIs depend on who is on base, totals
have real variance. A simulator captures all of this natively; a closed-form
model must re-derive bespoke math per market per sport. Simulation is the one
approach that travels to basketball (possessions) and football (drives) — same
framework, different kernel.

The current analytic model (`model/game.py`, `model/props.py`) is under-dispersed
(predictions cluster 0.4–0.6, totals hug league average, tails overconfident) and
approximates HRR/outs. The simulator directly targets these.

## Non-goals (YAGNI)

- No NBA/NFL kernels now. We design the *seams* to generalize; we build only MLB.
- No per-runner sprint-speed / batter-specific advancement (tier-3). League-aggregate
  Statcast advancement only for v1.
- No new bet types surfaced yet (run-line, team totals, NRFI come free from the sim
  but are not wired into the board in this project).
- The analytic model is **not deleted** — it stays as the backtest baseline the sim
  must beat, and a fast fallback.

## Architecture — sport-agnostic framework + MLB kernel

New package `src/sportsmodel/sim/`:

- **`sim/engine.py`** *(sport-agnostic)* — the neutral contract:
  `simulate_game(spec, n_sims, rng) → GameSims`, where `GameSims` holds raw result
  arrays (`home_score[N]`, `away_score[N]`, per-player stat arrays). Plus aggregation
  helpers that turn those arrays into stored outputs (win prob, total pmf, per-player
  pmfs). **Never mentions baseball.** Unchanged when NBA arrives.
- **`sim/mlb/kernel.py`** *(the only sport-specific simulation file)* — the PA-by-PA
  engine: base-out state machine, PA outcome sampling, pitcher hook + TTO. Pure
  functions over numpy arrays.
- **`sim/mlb/inputs.py`** — thin adapter assembling a `GameSpec` from existing loaders
  (`profiles`, `mlb_lineups`, `rates.matchup_vector`, `game.apply_*` context helpers).
  Reuse, not reinvent.
- **`sim/mlb/advancement.py`** — loads the Statcast-derived advancement table behind a
  clean interface `advance(outcome, base_out_state) → (end_state, runs, outs)` plus the
  league-average TTO deltas.

**Generalization contract:** any sport's kernel implements
`simulate(spec, n_sims, rng) → {game arrays, player arrays}`; `engine.py` aggregates
identically; everything downstream (distributions → EV → grading/CLV → backtest →
calibration) is shared. NBA later = write `sim/nba/kernel.py`, inherit the rest.

**Reused unchanged:** `model/rates.py`, `model/distributions.py` (the `dist`
representation), `model/calibration.py`, all `profiles.*` loaders. **The `dist` schema
stored and graded does not change** — so no DB, grading, or dashboard changes.

## The MLB kernel

### Inputs (`GameSpec`, from `inputs.py`)

- Both 9-batter orders (confirmed-or-projected via `mlb_lineups.lineups_for_game`), each
  batter carrying **two** context-adjusted per-PA vectors: vs the opposing **starter**
  and vs the opposing **bullpen** (park/weather/defense folded in via existing
  `game.apply_park_to_vector` / `apply_hr_multiplier` / `apply_bip_defense`, as the props
  path does today).
- Per-starter workload `(avg_bf, sd_bf, avg_outs, sd_outs)` from `profiles.load_pitcher_workload`
  → the hook distribution.
- The Statcast advancement table + league-average TTO deltas.

### The PA step (vectorized across N sims, masked lock-step)

State as parallel numpy arrays of length N: inning, half, outs, base occupancy **by
runner identity** (three int arrays of lineup index on 1B/2B/3B, −1 = empty), current
batter index per team, starter batters-faced + still-in flag per team, times-through-order.

Each step advances every sim in the current half-inning by one PA:

1. Gather each sim's current batter's vector — vs-starter (with the TTO penalty for that
   sim's times-through-order) if the starter is still in, else vs-bullpen.
2. Sample outcome ∈ {BB, K, 1B, 2B, 3B, HR, OUT} per sim (cumulative-sum + uniform draw).
3. Resolve runners:
   - **BB** = forced advance; **HR** = clear bases, all score; **K** = pure out (no advance).
   - **1B / 2B / 3B / OUT** = draw from the empirical advancement table for that
     (outcome, base-out state). Productive outs, sac flies, and double plays live inside
     this table — captured from data, not hand-coded.
   - Runner **identity** is reconciled to the table's abstract end-state deterministically
     (lead runners score/advance to fill the resulting occupied bases; batter takes the
     appropriate base), so we know exactly who scored → credit that runner a **run** and
     the batter an **RBI**.
4. Outs update; at 3 outs the half-inning ends (runners stranded), sides flip.

### Pitcher hook + TTO

Each sim samples the starter's exit as a batters-faced threshold `~ Normal(avg_bf, sd_bf)`
(clamped to a sane range); once crossed, that sim switches to bullpen vectors. Outs
accumulated **while the starter is in** = the `outs_recorded` prop. The TTO penalty is a
league-average multiplier worsening the starter's vector on the 2nd/3rd time through the
lineup (isolated constants in `advancement.py`).

### Extra innings

If tied after 9, continue with the current MLB regular-season "runner on 2nd to start each
half" (Manfred) rule, hard-capped (~20 innings) to guarantee termination.

### Accumulated per sim

- **Game:** home/away final runs.
- **Player (batter):** hits (1B+2B+3B+HR), total bases, HR, runs, RBI → **HRR = H+R+RBI**
  (computed, not approximated).
- **Player (starter):** Ks, hits allowed, outs recorded (only while the starter is in).

Every graded market falls out of one simulation.

### Performance

numpy vectorized across sims (state = parallel arrays, fancy-indexed per step), seeded RNG
for reproducible tests. Target ~20k sims/game live, ~5k in the backtest — fast enough for
the full 2025 walk-forward. `numpy` becomes an explicit dependency.

## Advancement-table builder (Statcast, tier-2 league-aggregate)

A `transforms`-style profiling job reads the Statcast parquet and emits the
`(outcome × base-out state) → distribution over (end state, runs, outs)` transition table
as a committed asset (`assets/…`), loaded by `advancement.py`.

Mechanism: within each half-inning, order PAs by `at_bat_number`; a runner's advancement on
a batted ball = pre-PA base state (`on_1b/2b/3b`) vs the **next** PA's base state in that
half-inning, plus runs scored (`post_bat_score − bat_score`); tally by (outcome × base-out
state) → normalize. Required columns confirmed present in the backfill: `on_1b`, `on_2b`,
`on_3b`, `events`, `at_bat_number`, `inning`, `inning_topbot`, `bat_score`, `post_bat_score`.

Edge cases handled: inning-ending PAs (no "next PA" — infer from runs + inning end);
steals/wild-pitches/errors moving runners between PAs (noise that mostly washes at league
aggregate; a cleaner pass can filter on `events`/`des` later).

**Point-in-time safety:** for the walk-forward backtest, the advancement table is built from
**prior-season data only**, so it cannot leak future information. (Advancement rates are
near-stationary, so this is both rigorous and stable.)

## Outputs & aggregation (`engine.py`)

- **Game:** `home_win_prob = mean(home_score > away_score)` (no ties — extra innings resolve),
  `pred_total = mean(total)`, total pmf → P(over any line), scores/margin from means.
- **Player:** empirical pmf per market → `dist = {kind:"pmf", pmf:[…]}`, the exact stored/graded
  representation. Upgrades `outs_recorded` from fitted Normal to empirical pmf. `projected_mean`
  and display `prob_over` computed as today.
- **Calibration:** existing `calibration.calibrate` still applies at scoring time. New
  model_versions → **refit calibration** on the sim's backtest output via `fit_calibration.py`.

## Integration (no DB/schema/dashboard changes)

- One new script `scripts/generate_sim.py` simulates each game **once** and writes **both**
  `game_predictions` and `prop_predictions` rows under new model_versions (e.g. `mlb-sim-v1`).
  Reuses existing schedule/profiles/lineups/weather loaders.
- New model_versions mean sim and analytic **coexist**; the dashboard already filters by
  `model_version`, so both are visible side-by-side before any switch.
- `scripts/backtest_sim.py` reuses the exact walk-forward point-in-time harness from
  `backtest_game.py` (cutoffs, no leakage) and reports the same metrics against the analytic
  baseline; prop calibration extends `backtest_props.py`.
- Workflows: once validated, add `generate-sim` to the daily cron, running **parallel** to the
  analytic scripts. Promote only after the backtest and early CLV agree.

## Validation gate (acceptance criteria)

Adopt the sim only if, on the 2025 walk-forward, it:

- (a) matches/beats analytic win-prob **Brier** and **log-loss**, and improves **tail
  calibration**;
- (b) **reduces total-runs MAE/RMSE** (attacks the under-dispersion); and
- (c) improves **prop calibration**, especially **HRR** and **outs_recorded**.

Tie on win prob but fix totals + props = still a win. Regress = keep analytic, iterate the
kernel. Nothing goes live before the backtest says so.

## Testing

- **Advancement builder:** transition rows normalize to 1 per (outcome, base-out state);
  spot-check known aggregates (e.g. runner-on-2nd scores from a single at a sane rate).
- **Kernel invariants (property tests):** outs always 0–3; base occupancy consistent with
  runner identities; a bases-loaded HR scores exactly 4; an all-outs lineup scores 0.
- **Reproducibility:** a fixed RNG seed reproduces identical sims.
- **Cross-check vs analytic:** sim mean runs ≈ `game.expected_runs` on a fixed vector within
  tolerance; TTO raises late-game hit rates; the hook produces an outs distribution centered
  near `avg_outs`.

## Build staging (for the implementation plan)

1. Advancement-table builder (Statcast, prior-season, point-in-time) + `sim/` package
   (engine, mlb kernel, inputs, advancement) + unit/property tests.
2. Game backtest (`backtest_sim.py`) → clear gates (a) + (b); tune kernel if needed.
3. Player-prop aggregation + prop backtest → gate (c); refit calibration.
4. Live `generate_sim.py` + workflow wiring, parallel to analytic; promote once backtest +
   early CLV agree.

## New dependency

- `numpy` (already transitively present via scipy/pandas; make it explicit in `pyproject.toml`).
