# NFL P3 Task 6-7: Player-props walk-forward backtest -- fitted yardage sigmas + receptions dispersion + TD rate calibration

Script: `scripts/backtest_nfl_props.py`
Test: `tests/nfl/test_backtest_nfl_props.py`
Output: `assets/nfl/props.json`

## Statistical honesty: calibrated to OUTCOMES, not a market

Unlike the game-line backtest (Task 6 of P2), there is **no historical player-props market line** in the committed nflverse data to shrink toward or score against. This backtest therefore calibrates each market's distribution directly to **actual outcomes**: a yardage market's Normal sigma is the residual RMSE of the `mean_mult`-corrected prediction against actual outcomes on the walk-forward (see "Fix round 1" below); the receptions Negative Binomial's overdispersion multiplier is the empirical `var(actual)/mean(actual)` of receptions across all scored player-games. There is no market line to "beat" here -- season-long CLV against whatever props book lines are shopped once this model is actually serving lines is the real judge, not this backtest. The two TD markets (`pass_tds`, `anytime_td`) are Poisson-distributed and have NO free variance parameter to fit -- a Poisson's variance IS its mean, so there is no sigma to solve for the way there is for the yardage markets. Their MEAN (the Poisson lambda), however, gets the exact same `mean_mult` treatment as the yardage/receptions markets as of Fix round 2 (see below) -- fit as `mean(actual TD count) / mean(predicted lambda)` and applied in `props.build_prop` before `poisson_pmf`. Their per-market diagnostics should still be read as meaningfully LOOSER than the yardage/receptions markets, though: single-game TD counts are dominated by red-zone randomness that a season-level rate cannot resolve, and a mean correction does nothing to narrow that per-game variance (see the TD section for the honest framing).

### Fix round 1: EVERY market under-projects ~1.5x -- a shared root cause, not a TD-specific one

An earlier version of this report attributed the TD markets' `rate_mult` bias to `k_eff` position-baseline shrinkage over-shrinking rare TD rates specifically. **That diagnosis was wrong.** An independent decomposition of the walk-forward found that the SAME ~1.5-1.63x under-projection appears in EVERY market -- yardage and receptions included, not just the two TD markets -- while the team-level volume that feeds all of them (`gamescript.project_team_volume`) is nearly unbiased (~0.96-0.98x). Since the bias is common to every market that consumes per-player allocated volume, and is absent from team-level volume, the real cause is upstream of `props.build_prop` entirely: `usage.allocate`'s per-player share shrinkage-toward-0 (`f = games / (games + k_usage)`) systematically UNDER-allocates team volume to any player with a finite `games` count, because shrinking every player's share toward 0 (rather than toward a position-appropriate baseline, or renormalizing shares to sum to ~1 after shrinkage) throws away volume rather than redistributing it. `k_eff`/TD-rate rarity was a red herring -- the `pass_td_rate` etc. efficiency rates are shrunk toward position baselines correctly; they are just being multiplied by an already-too-small `pass_att`/`carries`/`targets` volume figure, same as every yardage market.

This mattered enough to fix in this task rather than defer, because shipping the RAW (biased) `projected_mean` would make the model recommend UNDER on nearly every yardage/receptions prop against a fair market -- i.e. **directionally wrong, not just imprecise** -- which defeats the purpose of a props model. Task 6 originally scoped the mean bias (`loc`) as report-only/optional; this finding shows that was not an acceptable simplification once the size and consistency of the bias was actually decomposed.

**The fix applied here (`mean_mult`)**: `fit_calibration` now also fits `mean_mult[market] = mean(actual) / mean(pred_mean)` for every yardage market AND receptions, and `props.build_prop` multiplies `projected_mean` by `cfg.mean_mult[market]` (via a new `PropConfig.mean_mult` field, default 1.0) before building each market's distribution. Sigma is then refit on the CORRECTED residual `(pred_mean * mean_mult - actual)`, not the raw one -- fitting sigma on a biased mean conflates "the mean is wrong" with "the spread is wrong," which a multiplier should fix instead of sigma absorbing it. Empirically, though, this correction barely moves `pass_yds` sigma (106.9 raw -> 106.3 corrected) -- confirming that `pass_yds`'s outsized sigma (Concern 3) is NOT a mean-bias artifact at all, but genuinely idiosyncratic per-player variance (QB job-security regime shifts) that a population-average multiplier cannot touch. The other three yardage markets show a similarly small raw-to-corrected sigma shift for the same reason: `mean_mult` fixes the MEAN, and each market's residual spread was already centered reasonably well relative to its own (biased) mean, so de-biasing the mean does not by itself tighten the spread. This is explicitly a FIRST-ORDER, population-average correction, not a fix to the underlying usage-share mechanism -- see "P3.5 follow-up" in Concerns for the deeper fix (renormalizing or position-baseline-shrinking usage shares so per-player shares actually sum to ~1 after shrinkage, instead of leaking volume into a uniform post-hoc multiplier).

| market | sigma_raw (pre-correction) | sigma (mean_mult-corrected) | mean_mult |
|---|---|---|---|
| pass_yds | 106.867 | 106.294 | 1.578 |
| reception_yds | 29.401 | 29.746 | 1.505 |
| rush_yds | 24.741 | 24.677 | 1.632 |
| rush_reception_yds | 35.405 | 35.567 | 1.549 |

### Fix round 2: the TD markets get the SAME applied correction, not just a documented knob

Fix round 1 applied `mean_mult` to the five yardage/receptions markets but left `pass_tds`/`anytime_td` with only a report-only `rate_mult` diagnostic -- meaning `build_prop` still shipped a ~1.5x-under-biased Poisson lambda for both TD markets even after the yardage fix, which is inconsistent: the model would recommend UNDER on TD props too, and would essentially never surface an `anytime_td` OVER, for the exact same root cause already fixed for yardage. This was corrected by extending `PropConfig.mean_mult` to cover `pass_tds`/`anytime_td` as well, so ALL 7 markets are now calibrated AND applied through the identical mechanism -- a single dict that `props.json`'s `mean_mult` block loads directly into `PropConfig`.

`fit_td_calibration` now fits the TD multiplier on the SAME basis as the yardage markets -- `mult = mean(actual TD count) / mean(predicted lambda)` -- rather than on `P(>=1)`, since lambda (an expected COUNT) is the quantity actually being multiplied in `build_prop`, and count-vs-count is the apples-to-apples ratio (the same basis `pass_tds` was already, accidentally, computed on before this fix, since its `pred_mean` was already lambda and its `actual` was already a raw TD count -- only `anytime_td`'s fit basis changes here, from P(>=1) to raw TD count). For `anytime_td`, this required changing `_actual_value` to return the RAW rushing+receiving TD count instead of a binarized Yes/No indicator -- the served market is still Yes/No at the 0.5 line (`prob_over_dist(dist, 0.5)`), but the multiplier that feeds that market's lambda is fit on counts.

| market | mean_lambda_pred (raw) | mean_actual_count | mult (applied) |
|---|---|---|---|
| pass_tds | 0.6211 | 0.9422 | 1.517 |
| anytime_td | 0.1660 | 0.2570 | 1.549 |

**An honest, slightly counterintuitive result on `anytime_td`'s Brier score, stated plainly rather than smoothed over**: applying the count-based `mult` to lambda moves the mean predicted P(>=1) from 0.1416 (raw) to 0.2028 (corrected) -- MUCH closer to the actual Yes/No rate of 0.2158 (the raw gap was 0.0742 under; the corrected gap is only 0.0130 under) -- and yet the corrected Brier (0.1691) is very slightly WORSE than the raw Brier (0.1684), both close to the trivial base-rate baseline of 0.1692. This is not a bug or a sign the correction is wrong -- it is a real, if minor, illustration that Brier score depends on both average calibration AND per-instance resolution (how well probabilities discriminate individual outcomes), and improving the first does not guarantee improving the second: scaling each player's own lambda up multiplicatively widens the spread of individual predicted probabilities, and on this walk-forward that widening happened to cost slightly more Brier on the (large) `y=0` majority than it saved on the `y=1` minority. `mult` is still the RIGHT correction for the market's mean TD-COUNT expectation (and is exactly right for `pass_tds`, whose served market IS a count) -- it is just not, by this one metric, a strict Brier improvement for `anytime_td` specifically. We applied the same count-based fit to both markets anyway, per the brief's instruction for one principled, consistent basis, rather than carving out a second, market-specific fitting rule chasing a fractional Brier gain that both numbers show is within noise of the trivial baseline either way. See the bottom-line discussion further below for the deeper point: even the better of these two Briers is a near-null edge over a flat rate.

## Leak-free walk-forward

For each target season S in `range(2016, 2025)`: usage shares (`usage.compute_usage_shares`) and efficiency rates (`efficiency.compute_efficiency`) are computed ONLY from `weekly[weekly.season == S-1]`. The gamescript model (`gamescript.fit_gamescript`) is fit ONCE from all seasons strictly before the earliest season scored (`weekly.season < 2016`, i.e. <=2015). Each player-game's team pass/rush volume comes from `project_team_volume` applied to that game's own PRE-game `team_margin`/`implied_total` (derived from the schedule's `spread_line`/`total_line`, never the final score). `test_no_leak_uses_prior_season_only` enforces that dropping a season's own in-season rows does not change that season's predictions.

Markets are scored per player-game only where the player has nonzero PROJECTED volume for that market's driving stat (pass_att/targets/carries) -- this keeps MAE/sigma honest instead of diluting them with trivial zero-vs-zero pairs across every skill player who never touches a given market (e.g. a WR's pass_yds).

## Per-market backtest results (2016-2024, n = total scored player-games)

| market | mae | n | coverage (within 1 RMSE-sigma) |
|---|---|---|---|
| pass_yds | 69.566 | 6676 | 0.727 |
| reception_yds | 19.581 | 33081 | 0.780 |
| rush_yds | 13.085 | 23672 | 0.830 |
| rush_reception_yds | 24.052 | 37173 | 0.772 |
| receptions | 1.599 | 33081 | 0.754 |
| pass_tds | 0.659 | 6676 | 0.762 |
| anytime_td | 0.321 | 37173 | 0.781 |

Note: this table's `mae`/`coverage` are computed on the RAW, UNCORRECTED `pred_mean` (before `mean_mult`) -- they are the diagnostic that revealed the bias in the first place, not the as-served accuracy. The corrected sigma below reflects the `mean_mult`-adjusted prediction actually served by `props.build_prop`.

## Fitted calibration (`assets/nfl/props.json`)

```json
{
  "sigma": {
    "pass_yds": 106.29426777523592,
    "reception_yds": 29.7459083109448,
    "rush_yds": 24.677067044644406,
    "rush_reception_yds": 35.566767261871
  },
  "nb_var_mult": 2.1515394899238456,
  "mean_mult": {
    "pass_yds": 1.57797974466974,
    "reception_yds": 1.5054835244509903,
    "rush_yds": 1.63242559734058,
    "rush_reception_yds": 1.548571458941316,
    "receptions": 1.5433564104358664,
    "pass_tds": 1.5169012625841893,
    "anytime_td": 1.5487446041789215
  },
  "k_usage": 4.0,
  "k_eff": 4.0,
  "td_calibration": {
    "pass_tds": {
      "n": 6676,
      "mean_lambda_pred": 0.621122132280092,
      "mean_actual_count": 0.9421809466746555,
      "mult": 1.5169012625841893
    },
    "anytime_td": {
      "n": 37173,
      "mean_lambda_pred": 0.16596758453479132,
      "mean_actual_count": 0.25704140101686707,
      "mult": 1.5487446041789215,
      "rate_actual_binary": 0.21580179162295215,
      "mean_p_raw": 0.14163552854865172,
      "mean_p_corrected": 0.20275268197182422,
      "brier_raw": 0.1684028765585768,
      "brier_corrected": 0.16909325610171608,
      "baseline_brier": 0.16923137835527607
    }
  }
}
```

- `mean_mult[market]`: population-level de-bias multiplier for ALL 7 markets (`mean(actual)/mean(RAW pred_mean)` for the 5 yardage/receptions markets, `mean(actual TD count)/mean(predicted lambda)` for `pass_tds`/`anytime_td`) -- `props.build_prop` applies it to `projected_mean` (or the Poisson lambda, for the TD markets) before building that market's distribution, uniformly via `PropConfig.mean_mult` (see "Fix round 1" and "Fix round 2" above).
- `sigma[market]`: RMSE of `(pred_mean * mean_mult - actual)` per yardage market across all 2016-2024 walk-forward player-games -- i.e. fit on the CORRECTED (de-biased) prediction, not the raw one -- the Normal sigma `props.build_prop` uses for that market.
- `nb_var_mult`: empirical `var(actual receptions)/mean(actual receptions)` across all scored receptions player-games (clamped >1 for a well-defined Negative Binomial) -- purely a function of the ACTUAL outcome distribution, so unaffected by `mean_mult`.
- `loc[market]` (report only, not written to `props.json` / not applied to projections): mean `(actual - RAW pred_mean)` per market, i.e. the PRE-correction bias -- a positive value means the raw model under-projects that market on average. Roughly `loc ~= mean(pred_mean) * (mean_mult - 1)`; kept here purely for comparison against `mean_mult`, since `mean_mult` is the multiplier actually applied.

### Mean de-bias (`mean_mult`) vs pre-correction bias (`loc`)

| market | mean_mult (applied) | loc, pre-correction (mean actual-raw_pred) |
|---|---|---|
| pass_yds | 1.578 | 54.829 |
| reception_yds | 1.505 | 9.873 |
| rush_yds | 1.632 | 7.169 |
| rush_reception_yds | 1.549 | 13.495 |
| receptions | 1.543 | 0.927 |

## TD markets (`pass_tds`, `anytime_td`): corrected-and-applied, but still looser

These are the two lowest-count, highest-variance markets in this model. `pass_tds` is Poisson(`pass_att * pass_td_rate * mean_mult["pass_tds"]`); `anytime_td` is P(>=1 rushing-or-receiving TD) = `1 - exp(-lambda)` with `lambda = (carries*rush_td_rate + targets*rec_td_rate) * mean_mult["anytime_td"]`, evaluated at the 0.5 line like MLB's HR Y/N. A Poisson rate has no free sigma to fit (mean IS variance), but as of Fix round 2 the MEAN (lambda) gets the SAME applied `mean_mult` correction as every other market -- fit as `mean(actual TD count)/mean(predicted lambda)`, the same count-vs-count basis as the yardage markets' mean_mult, not a P(>=1)-based ratio. See the "Fix round 2" table above (under "Fitted calibration") for the fitted `mean_lambda_pred`/`mean_actual_count`/`mult` values.

**Corrected root cause (Fix round 1)**: the ~1.5x TD-lambda bias is NOT a TD-specific artifact of `k_eff` shrinking rare TD rates too hard -- that was this report's earlier (incorrect) diagnosis. The SAME ~1.5x under-projection shows up in the yardage/receptions markets too (see "Fix round 1" above), which is only possible if the shared cause is upstream, in `usage.allocate`'s per-player volume allocation, not in any market-specific efficiency rate.

**Corrected-and-applied (Fix round 2)**: unlike the earlier version of this report, the TD markets are NOT left with only a documented, unapplied knob -- `mean_mult["pass_tds"]` and `mean_mult["anytime_td"]` are fit here and actually multiplied into the Poisson lambda inside `props.build_prop`, exactly like the yardage markets. All 7 markets now go through the identical calibrate-and-apply path via `PropConfig.mean_mult` / `props.json`'s `mean_mult` block.

**Bottom line on `anytime_td` discrimination, computed honestly rather than left as an exercise for the reader** (see the full raw-vs-corrected Brier breakdown under "Fix round 2" above for why the corrected number is slightly worse, not better): against a trivial "always predict the empirical base rate" baseline Brier of 0.1692, the model's per-player lambda scores 0.1684 raw (~0.5% better than baseline) and 0.1691 as actually served after Fix round 2's mean correction (~0.1% better than baseline). Both are NEAR-NULL edges -- the per-player `anytime_td` lambda is barely distinguishing players from a flat league-average rate in this walk-forward, whether or not the mean bias is corrected. That is an honest, and not especially flattering, result -- it should NOT be oversold as the model having meaningful `anytime_td` discrimination power yet. Fixing the MEAN bias (Fix round 2) and having real per-player DISCRIMINATION (still weak, either way) are two different things, and only the first is addressed by `mean_mult`.

## TDD: red -> green

Task 6 (yardage + receptions) Step 2 (red), before `scripts/backtest_nfl_props.py` existed:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../scripts/backtest_nfl_props.py'
```

Task 6 Step 4 (green), after implementation:
```
tests/nfl/test_backtest_nfl_props.py::test_run_backtest_returns_per_market_metrics PASSED
tests/nfl/test_backtest_nfl_props.py::test_fit_calibration_returns_sigmas PASSED
tests/nfl/test_backtest_nfl_props.py::test_no_leak_uses_prior_season_only PASSED
3 passed
```

Task 7 (TD markets) Step 2 (red), before `poisson_pmf`/`pass_tds`/`anytime_td` existed:
```
ImportError: cannot import name 'poisson_pmf' from 'sportsmodel.model.distributions'
```

Task 7 Step 4 (green), after implementation:
```
tests/nfl/test_props.py::test_anytime_td_prob_at_least_one PASSED
tests/nfl/test_props.py::test_pass_tds_poisson_mean PASSED
tests/nfl/test_dist_builders.py::test_poisson_pmf_sums_and_mean PASSED
9 passed
```

Fix round 1 (mean_mult) Step 2 (red), before `PropConfig.mean_mult` existed:
```
TypeError: __init__() got an unexpected keyword argument 'mean_mult'
```

Fix round 1 Step 4 (green), after implementation:
```
tests/nfl/test_props.py::test_mean_mult_scales_pass_yds_projected_mean PASSED
tests/nfl/test_props.py::test_mean_mult_scales_rush_reception_yds_combined_total PASSED
tests/nfl/test_props.py::test_mean_mult_scales_receptions_negbin_mean PASSED
13 passed
```

Fix round 2 (TD lambda mult) Step 2 (red), before `PropConfig.mean_mult` covered `pass_tds`/`anytime_td`:
```
KeyError: 'pass_tds'
```

Fix round 2 Step 4 (green), after implementation:
```
tests/nfl/test_props.py::test_default_mean_mult_is_unity_for_all_seven_markets PASSED
tests/nfl/test_props.py::test_mean_mult_scales_pass_tds_lambda PASSED
tests/nfl/test_props.py::test_mean_mult_scales_anytime_td_lambda_and_prob_identity PASSED
15 passed
```

## Concerns

1. **No historical props market exists to validate against** -- sigma/nb_var_mult are calibrated to outcome residuals, which is honest but means there is no OOS "beat the book" check here at all (unlike the game-line backtest's model-only/blend/market-only comparison). Live CLV tracking is the only real validation once this serves actual lines.
2. **`k_usage`=4.0's usage-share shrinkage-toward-0 is the confirmed root cause of the ~1.58x-and-up volume under-allocation fixed in this task via `mean_mult`** (see "Fix round 1" above) -- `allocate`'s per-player share shrinkage pulls every player's share toward 0 rather than toward a position baseline or renormalizing shares to sum to ~1 after shrinkage, so it structurally leaks team volume regardless of which market consumes it. `mean_mult` is a population-average PATCH on top of this, not a fix to `allocate` itself -- see the P3.5 follow-up below for the deeper fix. `k_eff` (efficiency-rate shrinkage) is carried through from Task 3 as a fixed constant and was NOT found to need a similar correction -- efficiency RATES (yards/attempt, catch rate, etc.) are unbiased by this backtest's own decomposition; only per-player VOLUME allocation was.
3. **`pass_yds` sigma (106.3, now fit on the `mean_mult`-corrected residual) is still larger than the other yardage markets, for a DIFFERENT reason than a mean-bias artifact: genuine QB job-security/team-change regime shifts that a prior-season-shares model structurally cannot see.** Example from the real data: Joe Flacco (`00-0026158`) split 2022 between spot starts for NYJ (`pass_att_share=0.169` after usage shrinkage, reflecting a part-time backup role), then signed with CLE for 2023 and started outright (42-45 attempts/game, weeks 13-17) -- his 2022-derived share projects far fewer pass attempts for 2023 than he actually threw. This is not a code bug -- it is the real, load-bearing limitation of projecting purely from S-1 season-level shares with no in-season depth-chart/injury signal for who wins a QB competition. A follow-up (in-season share updates, or an explicit backup/starter transition flag) would likely shrink `pass_yds` sigma the most of any market. (Pre-correction `loc` was 54.8, now folded into `mean_mult`=1.578 rather than left as unapplied bias.)
4. **Market inclusion is volume-gated (nonzero projected pass_att/targets/carries), not a hardcoded position map** -- this is a deliberate, data-driven choice (see code comment on `_markets_for_volume`) but means, e.g., a QB who also has real receiving volume (extremely rare) would be scored on reception_yds too; this is correct behavior, not a bug, but worth knowing the gate is volume-based rather than position-based.
5. **TD bias corrected via the same applied multiplier as yardage (Fix round 2) -- no longer unresolved.** An earlier version of this report left `pass_tds`/`anytime_td`'s ~1.5x shared volume under-allocation as a documented-but-unapplied `rate_mult` knob, which was inconsistent with the yardage markets already getting `mean_mult` applied in `build_prop` -- the model would have kept recommending UNDER on TD props (and essentially never surfacing an `anytime_td` OVER) even after the yardage fix landed. `PropConfig.mean_mult` now covers `pass_tds` (mult=1.517) and `anytime_td` (mult=1.549) too, fit on the same count-vs-count (`mean(actual TD count)/mean(predicted lambda)`) basis as the yardage markets and applied to the Poisson lambda before `poisson_pmf` in `build_prop` -- all 7 markets now go through the identical calibrate-and-apply path. What remains true, and is NOT fixed by this mean correction: TD scoring is still a low-probability, high-variance PER-GAME event (see the TD section's Brier discussion) -- correcting the average bias does not add per-player discrimination power, and the `anytime_td` Brier improvement over a flat base rate remains near-null.
6. **P3.5 follow-up: fix `usage.allocate`'s shrinkage-toward-0 at the source, instead of relying on population-average `mean_mult` patches (now applied to all 7 markets, per Fix round 2).** `mean_mult` corrects the AVERAGE bias across all player-games in a market, but it cannot correct the PER-PLAYER distribution of that bias -- a low-`games` player (whose share was shrunk hardest toward 0) is still under-allocated more than a high-`games` player, and a single multiplier does not know the difference. This is now true of ALL 7 markets uniformly, not just the yardage ones. The two candidate proper fixes are (a) renormalizing each team-week's allocated shares to sum to ~1 after shrinkage (currently shrinkage is applied per-player independently, with no renormalization step, so the team's shares can sum to well under 1), or (b) shrinking each player's share toward a position-appropriate baseline share (mirroring how `efficiency.compute_efficiency` already shrinks RATES toward position baselines in Task 3) instead of toward 0. Either would fix the bias per-player rather than only on population average, and would very likely make ALL 7 markets' `mean_mult` values converge to ~1.0 at the source, without needing a post-hoc multiplier in `build_prop` at all. Scoped out of this fix because it touches `usage.allocate`, which is outside this task's file allowlist and would ripple through every market (including ones already shipped in Tasks 4/6).

## Commands used

- `uv run pytest tests/nfl/test_backtest_nfl_props.py -v` (red, then green)
- `PYTHONPATH=src uv run python scripts/backtest_nfl_props.py` (real fit, writes `assets/nfl/props.json`)
- `uv run pytest -q` (full suite)
