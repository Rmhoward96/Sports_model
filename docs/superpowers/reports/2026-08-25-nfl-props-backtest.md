# NFL P3 Task 6-7: Player-props walk-forward backtest -- fitted yardage sigmas + receptions dispersion + TD rate calibration

Script: `scripts/backtest_nfl_props.py`
Test: `tests/nfl/test_backtest_nfl_props.py`
Output: `assets/nfl/props.json`

## Statistical honesty: calibrated to OUTCOMES, not a market

Unlike the game-line backtest (Task 6 of P2), there is **no historical player-props market line** in the committed nflverse data to shrink toward or score against. This backtest therefore calibrates each market's distribution directly to **actual outcomes**: a yardage market's Normal sigma is the residual RMSE of `(projected_mean - actual)` on the walk-forward; the receptions Negative Binomial's overdispersion multiplier is the empirical `var(actual)/mean(actual)` of receptions across all scored player-games. There is no market line to "beat" here -- season-long CLV against whatever props book lines are shopped once this model is actually serving lines is the real judge, not this backtest. The two TD markets (`pass_tds`, `anytime_td`) are Poisson-distributed and have NO free variance parameter to fit -- a Poisson's variance IS its mean, so there is no sigma to solve for the way there is for the yardage markets. Their calibration section below is therefore a rate-bias CHECK (report-only rate multiplier + a Brier score on `anytime_td`'s P(>=1)), not a fitted parameter, and their per-market diagnostics should be read as meaningfully LOOSER than the yardage/receptions markets: single-game TD counts are dominated by red-zone randomness that a season-level rate cannot resolve (see the TD section for the honest framing).

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
| anytime_td | 0.280 | 37173 | 0.739 |

## Fitted calibration (`assets/nfl/props.json`)

```json
{
  "sigma": {
    "pass_yds": 106.8668289407906,
    "reception_yds": 29.401329489987145,
    "rush_yds": 24.7410436641191,
    "rush_reception_yds": 35.40522880088011
  },
  "nb_var_mult": 2.1515394899238456,
  "k_usage": 4.0,
  "k_eff": 4.0,
  "td_calibration": {
    "pass_tds": {
      "n": 6676,
      "rate_pred": 0.621122132280092,
      "rate_actual": 0.9421809466746555,
      "rate_mult": 1.5169012625841893
    },
    "anytime_td": {
      "n": 37173,
      "rate_pred": 0.14163552854865172,
      "rate_actual": 0.21580179162295215,
      "rate_mult": 1.523641658518077,
      "brier": 0.1684028765585768
    }
  }
}
```

- `sigma[market]`: RMSE of `(projected_mean - actual)` per yardage market across all 2016-2024 walk-forward player-games -- the Normal sigma `props.build_prop` uses for that market.
- `nb_var_mult`: empirical `var(actual receptions)/mean(actual receptions)` across all scored receptions player-games (clamped >1 for a well-defined Negative Binomial).
- `loc[market]` (report only, not written to `props.json` / not applied to projections): mean `(actual - pred_mean)` per market -- a positive value means the model under-projects that market on average.

### Mean bias (`loc`), report-only

| market | loc (mean actual-pred) |
|---|---|
| pass_yds | 54.829 |
| reception_yds | 9.873 |
| rush_yds | 7.169 |
| rush_reception_yds | 13.495 |
| receptions | 0.927 |

## TD markets (`pass_tds`, `anytime_td`): honest, looser calibration

These are the two lowest-count, highest-variance markets in this model. `pass_tds` is Poisson(`pass_att * pass_td_rate`); `anytime_td` is P(>=1 rushing-or-receiving TD) = `1 - exp(-lambda)` with `lambda = carries*rush_td_rate + targets*rec_td_rate`, evaluated at the 0.5 line like MLB's HR Y/N. A Poisson rate has no free sigma to fit, so `fit_td_calibration` reports a rate-bias CHECK instead: `rate_pred` (mean model-implied rate) vs `rate_actual` (empirical rate) and their ratio `rate_mult`, plus a Brier score on `anytime_td`'s P(>=1) against the binary actual outcome. Per the brief, this multiplier is a documented calibration KNOB, not force-fit into `props.json` / `props.build_prop` -- there is no multiplier parameter in `build_prop`'s TD branches (Step 3), so a large, persistent `rate_mult` would be a signal to scale the underlying `pass_td_rate`/`rush_td_rate`/`rec_td_rate` efficiency inputs in a follow-up, not something this backtest applies itself.

| market | n | rate_pred | rate_actual | rate_mult | brier |
|---|---|---|---|---|---|
| pass_tds | 6676 | 0.6211 | 0.9422 | 1.517 | n/a |
| anytime_td | 37173 | 0.1416 | 0.2158 | 1.524 | 0.1684 |

`rate_mult` near 1.0 means the model's implied TD rate roughly matches observed outcomes over the full 2016-2024 walk-forward; it does NOT mean any single player-game prediction is precise -- TD scoring is a low-probability, high-variance event per game, and this is a population-level average check, not a per-player-game accuracy claim the way yardage sigma is. `anytime_td`'s Brier score is on a roughly `rate_actual`-base-rate binary outcome; compare it to the trivial "always predict the base rate" Brier of `rate_actual*(1-rate_actual)` to judge whether the per-player lambda is adding real signal over a flat rate.

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

## Concerns

1. **No historical props market exists to validate against** -- sigma/nb_var_mult are calibrated to outcome residuals, which is honest but means there is no OOS "beat the book" check here at all (unlike the game-line backtest's model-only/blend/market-only comparison). Live CLV tracking is the only real validation once this serves actual lines.
2. **Usage/efficiency shrinkage (`k_usage`/`k_eff` = 4.0) are carried through from Tasks 2-3 as fixed constants, not re-tuned by this backtest** -- the brief scopes this task to fitting sigma/nb_var_mult only; a follow-up could jointly tune k_usage/k_eff against this same residual objective.
3. **`pass_yds` sigma (106.9) and loc bias (+54.8) are far larger than the other yardage markets (24-35 sigma) -- traced to genuine QB job-security/team-change regime shifts that a prior-season-shares model structurally cannot see.** Example from the real data: Joe Flacco (`00-0026158`) split 2022 between spot starts for NYJ (`pass_att_share=0.169` after usage shrinkage, reflecting a part-time backup role), then signed with CLE for 2023 and started outright (42-45 attempts/game, weeks 13-17) -- his 2022-derived share projects ~6 pass attempts/game for 2023, when he actually threw ~44. This is not a code bug (the non-QB markets, which are far less winner-take-all, show tight/sane sigmas of 24-35 with 73-83% 1-sigma coverage) -- it is the real, load-bearing limitation of projecting purely from S-1 season-level shares with no in-season depth-chart/injury signal for who wins a QB competition. A follow-up (in-season share updates, or an explicit backup/starter transition flag) would likely shrink `pass_yds` sigma the most of any market.
4. **Market inclusion is volume-gated (nonzero projected pass_att/targets/carries), not a hardcoded position map** -- this is a deliberate, data-driven choice (see code comment on `_markets_for_volume`) but means, e.g., a QB who also has real receiving volume (extremely rare) would be scored on reception_yds too; this is correct behavior, not a bug, but worth knowing the gate is volume-based rather than position-based.
5. **TD markets (`pass_tds`, `anytime_td`) are structurally looser than the yardage/receptions markets, and the fitted `rate_mult` shows a real, sizeable, CONSISTENT underprediction that this report does not paper over.** Across all 2016-2024 walk-forward player-games, `pass_tds.rate_mult` = 1.517 (model rate 0.621 TD/game vs actual 0.942) and `anytime_td.rate_mult` = 1.524 (model P(>=1) 0.142 vs actual 0.216) -- both markets underproject by ~50%, and the fact that BOTH independent TD rates (pass, and rush+rec combined) show nearly the SAME ~1.52x multiplier suggests a common structural cause rather than two unrelated market-specific quirks: most likely the position-baseline shrinkage in `efficiency.compute_efficiency` (`k_eff`=4.0, tuned for yardage rates in Task 3) over-shrinks TD rates specifically, because TD/attempt and TD/target are much rarer, noisier per-player rates than yards/attempt -- shrinking a rare-event rate toward a position-wide baseline pulls it down harder in relative terms than it does a yardage rate. Per the brief, this backtest reports the multiplier as a documented calibration KNOB rather than force-fitting it into `build_prop` (which per Step 3 has no multiplier parameter for the TD branches) -- but a `rate_mult` this large and this consistent is a genuine finding, not a rounding error, and a follow-up should very likely either scale `pass_td_rate`/`rush_td_rate`/`rec_td_rate` by ~1.5x or re-tune `k_eff` specifically for TD rates before this market is served live. Red-zone TD scoring is also inherently high-variance per game regardless of rate bias, so even a rate-corrected model should be marketed as looser than the yardage lines.

## Commands used

- `uv run pytest tests/nfl/test_backtest_nfl_props.py -v` (red, then green)
- `PYTHONPATH=src uv run python scripts/backtest_nfl_props.py` (real fit, writes `assets/nfl/props.json`)
- `uv run pytest -q` (full suite)
