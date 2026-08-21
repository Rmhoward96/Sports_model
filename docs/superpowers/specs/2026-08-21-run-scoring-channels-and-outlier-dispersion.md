# Run-scoring channels, outlier dispersion, and totals calibration

**Date:** 2026-08-21
**Status:** Approved design (pre-plan)
**Component:** `sim/mlb/kernel.py`, `sim/mlb/build_advancement.py` (+ new rate
source), `model/calibration.py`, `scripts/generate_sim.py`, `scripts/fit_calibration_sim.py`,
`streamlit_app.py`, `scripts/grade_results.py`.

## Problem

The Monte Carlo sim under-produces runs and under-disperses game outcomes. Measured
on 2025 (2367 real games vs the sim pooled over 351 games in 2025-06, `n_sims=2000`):

| Metric | Real MLB | Sim now | Issue |
|---|---|---|---|
| Mean total | 8.89 | 7.90 | −0.99 (mean too low) |
| SD of total | 4.59 | 3.96 | −14% (too narrow) |
| P(total ≥ 11) | 32.9% | 24.0% | thin right tail |
| P(total ≤ 5) | 25.3% | 31.4% | too many low games |
| P(a team shut out) | 13.8% | 16.5% | slightly high |
| P(\|margin\| ≥ 5) | 28.7% | 24.3% | thin blowout tail |
| SD of margin | 4.58 | 4.01 | too narrow |

Two independent defects:

1. **Mean ~1 run low.** The base-out state machine models a *clean* defensive game.
   The advancement builder (`build_advancement.py`) already includes `sac_fly`/`sac_bunt`
   (productive outs score runners), but **deliberately excludes** `field_error` (reached
   on error) and `fielders_choice`. So there are no unearned runs, no reached-on-error
   baserunners, and no wild-pitch/passed-ball advancement — worth ~0.8 runs/game in real
   MLB, ≈ the observed gap.
2. **Tails genuinely thin even after a mean fix.** SD 3.96 vs 4.59; the high-scoring and
   blowout tails are underweight. Real baseball throws off crooked-number blowups and
   laughers that an i.i.d.-per-PA model with fixed team/pitcher means cannot reproduce.

These bias every distribution-derived market: over/under direction (driven by the mean),
run-line cover probability and alt lines (driven by margin dispersion), and totals/spread
EV. (The run-line *pick rule* was already fixed separately — commit 4d2caea — to select by
EV instead of mean-margin-vs-line; this spec fixes the underlying distribution.)

## Goals

- De-bias the sim's mean total to ~8.85 by adding the missing scoring channels
  structurally (root-cause fix), not just a scalar.
- Match real MLB's outlier frequency: the sim's marginal total and margin distributions
  hit the six empirical targets above within Monte Carlo noise.
- Add a residual totals/margin calibration layer as the safety net and validation gate.
- Do all of the above **without regressing** moneyline win-prob or prop calibration.

## Non-goals

- No change to the run-line pick rule (already EV-based).
- No new markets. No change to N_SIMS (20k live / 2k backtest).
- Not modeling every micro-channel (balks, catcher interference, etc.) — ROE + WP/PB
  cover the structural mass; the calibration layer absorbs the residual.

## Design

### 1. Missing scoring channels (kernel)

Both channels are added to **both** the scalar (`_sim_one`/`resolve_pa`) and vectorized
(`simulate`/`_resolve_pa_vec`) kernels; the existing scalar↔vectorized distributional
equivalence test must stay green.

**1a. Reached-on-error (ROE).** After an `OUT_INPLAY` outcome is sampled, draw an extra
uniform; with probability `p_roe` convert it to a *reached-on-error*: the batter is safe
on first with **no out recorded**, and runners advance using the existing single (`S`)
advancement table. Effect: adds a baserunner and removes an out, so innings occasionally
extend into crooked numbers — this fixes part of the mean and fattens the right tail.
- `p_roe` is a **measured constant** = count(`events = 'field_error'`) / count(PAs) from
  Statcast (respecting the walk-forward cutoff), sourced the same way advancement rates
  are. Expected ~0.015–0.02. Stored as a committed constant/asset, not tuned.
- Implementation note: ROE is a new terminal outcome path, not a new sampled outcome code
  — it is a post-hoc reinterpretation of an `OUT_INPLAY` draw, so it does not disturb the
  per-PA outcome vector or its renormalization.

**1b. Wild-pitch / passed-ball advance.** On `K` and `BB` PAs with at least one runner on
(the deterministic branches where the empirical table cannot capture mid-PA advancement),
with probability `p_wp` advance every runner one base (a runner on 3rd scores). Rate
`p_wp` measured from Statcast `wild_pitch` + `passed_ball` per PA-with-runners. Small but
real; applied only on K/BB to avoid double-counting the in-play PAs whose `runs` delta
already folds in mid-PA WP/PB runs.

Any residual mean gap after 1a/1b is left to the calibration layer (§3).

### 2. Outlier dispersion (both mechanisms)

All effects are sampled **once per simulated game** (per-sim arrays of length `n_sims`)
and applied by scaling the per-PA offensive probabilities, then renormalizing (the out
probability absorbs the change). Means are ~1 so the distribution widens without shifting.

**2a. Game environment effect.** Per sim, sample
`E_shared ~ LogNormal(-σ_s²/2, σ_s²)` and per team `E_home, E_away ~ LogNormal(-σ_t²/2, σ_t²)`
(the `-σ²/2` centering keeps E[·]=1). The batting team's non-out probabilities
(`p_bb, p_1b, p_2b, p_3b, p_hr`) are multiplied by `E_shared × E_team` and renormalized.
Shared → both offenses move together (high/low-scoring games); per-team → asymmetric
(blowouts).

**2b. Pitcher-quality effect.** Per sim, sample each starter's quality offset
`q ~ Normal(0, σ_p²)`; while that starter is in, multiply the *opposing* batters'
non-out probabilities by `exp(q)` (some sims he is shelled, some he is dominant). This
correlates the runs he allows within a game and drives the blow-up right tail and blowouts.

**Composition.** For a given PA the effective offensive multiplier is
`E_shared × E_bat_team × exp(q_current_pitcher)`. In the vectorized kernel these are
per-sim vectors broadcast against the `(n, 7)` outcome-probability array.

**Tuning.** `σ_s, σ_t, σ_p` are fit (coordinate/grid search in a small committed tuning
harness that runs the backtest and computes the six tail metrics) to match the empirical
targets. `p_roe, p_wp` are **not** tuned (measured from data). Target within Monte Carlo
noise at `n_sims=2000`; verify stability at higher n.

### 3. Calibration layer (totals + margin)

- After the kernel changes, fit a residual **location + scale** affine for the total and
  margin distributions on walk-forward 2025. Parameters `(loc, scale)` per target stored
  in `assets/calibration.json` alongside the existing Platt entries.
- Apply at **scoring time** (board `game_board` + grader `_grade_game`): remap the stored
  pmf's support by `x' = scale·(x − μ) + μ + loc` and re-bin, before `prob_over_dist` /
  `prob_cover`. A shared helper (e.g. `model/distributions.apply_affine(dist, loc, scale)`)
  keeps board and grader in sync.
- If §1–§2 land cleanly this is near-identity; it is retained as the safety net and is how
  residual bias is measured.
- **Refit all markets** (`fit_calibration_sim.py`) because widening the distributions
  shifts every sim-derived output, props included.

### 4. Validation & acceptance bar

This is the gate. It is **not** win-prob Brier alone (that would penalize the legitimate
within-game dispersion; the earlier input-sweep lesson was about *between-game* mean
dispersion, a different thing).

- **Distribution match:** sim's marginal total and margin distributions hit all six
  empirical targets (§Problem table) within Monte Carlo noise.
- **Reliability:** over/under and run-line predicted P(over)/P(cover) ≈ actual rate on
  walk-forward (reliability buckets), across the line range.
- **No regression:** win-prob Brier ≤ today's 0.246 (± MC noise); prop calibration
  (`backtest_sim_props.py`) no worse per market. If the dispersion work degrades props,
  dial `σ` down rather than ship.

### 5. Build sequence

1. **Channels.** Measure `p_roe`, `p_wp` from Statcast. Add ROE + WP/PB to scalar and
   vectorized kernels; keep equivalence test green. Re-measure mean & tails.
2. **Dispersion.** Add environment + pitcher-quality effects to both kernels. Build the
   tuning harness; fit `σ_s, σ_t, σ_p`. Re-measure tails.
3. **Calibration.** Add `apply_affine`; refit `calibration.json` (all markets); add
   totals/margin residual calibration; wire into board + grader. Run the full acceptance
   bar (distribution match + reliability + no-regression).
4. **Deploy.** `generate_sim` picks up the kernel automatically; commit `calibration.json`
   and tuned constants. Live N_SIMS=20k. No Supabase migration needed (schema unchanged).

### 6. Risks & mitigations

- **Whole-model blast radius.** Moneyline and props re-validate, not just totals — covered
  by the no-regression gate.
- **Scalar/vectorized drift.** Both kernels change; the equivalence test is the guard.
  New per-sim random draws must use the same RNG discipline (draw sizes/order documented)
  so the two paths agree within MC noise.
- **Double-counting mean.** ROE/WP add mean; the env/pitcher effects are mean-preserving
  (log-normal centered at 1); the calibration location term absorbs residual — so the mean
  is pinned in exactly one place (channels + calibration), not fought over by the σ's.
- **Prop side-effects.** The env/pitcher effects widen prop distributions (realistic: a
  hitter vs a shelled starter). The no-regression prop gate catches over-widening.

## Files touched

- `sim/mlb/kernel.py` — ROE + WP/PB branches; per-sim env/pitcher multipliers in both
  kernels; multiplier application + renormalization.
- `sim/mlb/build_advancement.py` (or a sibling) — measure `p_roe`, `p_wp` under the cutoff.
- `assets/` — committed `p_roe`/`p_wp` constants (or fold into the advancement asset) and
  refit `calibration.json`.
- `model/distributions.py` — `apply_affine(dist, loc, scale)` shared remap helper.
- `model/calibration.py` / `scripts/fit_calibration_sim.py` — totals/margin location+scale
  fit; refit all markets.
- `scripts/generate_sim.py` — no logic change (picks up kernel); confirm calibration is
  applied consistently.
- `streamlit_app.py`, `scripts/grade_results.py` — apply totals/margin calibration before
  `prob_over_dist`/`prob_cover`.
- New: a committed tuning/validation harness (backtest → six tail metrics + reliability +
  no-regression), replacing the throwaway scratchpad measurements.

## Acceptance

Ship only when: the six distribution targets match, over/under + run-line reliability hold,
and moneyline/prop calibration are no worse than today. Otherwise reduce the dispersion σ's
and/or lean more on the calibration layer, and re-measure.
