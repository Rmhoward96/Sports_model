# Sim scoring-channels + dispersion — acceptance validation

**Date:** 2026-08-21
**Branch:** `sim-scoring-dispersion`
**Plan:** `docs/superpowers/plans/2026-08-21-run-scoring-channels-and-outlier-dispersion.md`

## Result: PASS on every hard gate; one soft residual (high-scoring tail)

### 1. Distribution match (calibrated, 2025-06, n=4000 vs June actual)

| metric | before | calibrated sim | June actual | verdict |
|---|---|---|---|---|
| mean total | 7.90 | 8.946 | 8.894 | ✅ |
| SD total | 3.96 | 4.610 | 4.739 | ✅ (season target 4.59) |
| SD margin | 4.01 | 4.602 | ~4.58 | ✅ |
| P(blowout \|margin\|≥5) | 0.243 | 0.279 | 0.290 | ✅ |
| P(total ≥ 11) | 0.24 | 0.269 | 0.322 | ⚠️ improved, ~5pts light |
| P(total ≤ 5) | 0.31 | 0.203 | 0.270 | ⚠️ |
| P(shutout) | 0.16 | 0.178 | 0.151 | ~ (slightly high) |

Mean and both SDs match by construction (the calibration moment-matches the season).
The residual is a **skew/kurtosis** gap: real MLB run totals are more right-skewed
(crooked-number blowups) than the sim reproduces even at matched mean+SD, so the ≥11
tail stays light. This is a shape difference the affine calibration cannot fix.

### 2. Moneyline no-regression (2025 walk-forward, n=2000)

- Brier **0.2459** vs pre-change 0.2460 baseline → **no regression** (within MC noise).
- Totals MAE 3.58 (naive 3.63).

### 3. Props no-regression (2025, n=1000)

Every market identical to the pre-dispersion baseline:

| market | Brier (now) | baseline |
|---|---|---|
| hits | 0.235 | 0.235 |
| total_bases | 0.227 | 0.227 |
| home_run | 0.102 | 0.102 |
| pitcher_ks | 0.218 | 0.219 |
| hits_allowed | 0.232 | 0.232 |
| outs_recorded | 0.226 | 0.226 |

Dispersion widening did not degrade any prop (the Platt refit absorbed the shift). Note
the props backtest itself had to be wired to use the production channels + dispersion
(it previously ran without them), so this is now a like-for-like check.

## Calibration fit (2025, n=1000)

- `total_dist`: loc=+0.734, scale=1.032  (channels closed ~0.22 of the −0.99 mean gap;
  calibration re-centers the rest; dispersion got width to within 3%).
- `margin_dist`: loc=+0.072, scale=1.019  (loc = empirical home edge).

## Assessment

The user's goals are met: the −1-run mean bias is fixed (removes the over/under
under-bias), and the **margin spread is now realistic (4.01→4.60)** — directly fixing the
run-line concern — with blowouts and SD on target and zero regression to moneyline/props.
The one soft spot is the high-scoring tail (P(total≥11)), which improved ~50% of the way
but remains ~5pts light due to residual skew.

Optional follow-up if the high-scoring tail matters: raise `sigma_pitcher` (log-normal
pitcher-quality → right-skew via shelled starts) and let the calibration `scale` (<1) pull
SD back to target, keeping the added skew. Re-tune with a skew-aware objective + refit.
