# NFL P3 Task 6: Player-props walk-forward backtest -- fitted yardage sigmas + receptions dispersion

Script: `scripts/backtest_nfl_props.py`
Test: `tests/nfl/test_backtest_nfl_props.py`
Output: `assets/nfl/props.json`

## Statistical honesty: calibrated to OUTCOMES, not a market

Unlike the game-line backtest (Task 6 of P2), there is **no historical player-props market line** in the committed nflverse data to shrink toward or score against. This backtest therefore calibrates each market's distribution directly to **actual outcomes**: a yardage market's Normal sigma is the residual RMSE of `(projected_mean - actual)` on the walk-forward; the receptions Negative Binomial's overdispersion multiplier is the empirical `var(actual)/mean(actual)` of receptions across all scored player-games. There is no market line to "beat" here -- season-long CLV against whatever props book lines are shopped once this model is actually serving lines is the real judge, not this backtest. TD markets (`pass_tds`, `anytime_td`) are deferred to Task 7; this backtest covers yardage + receptions only.

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
  "k_eff": 4.0
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

## TDD: red -> green

Step 2 (red), before `scripts/backtest_nfl_props.py` existed:
```
FileNotFoundError: [Errno 2] No such file or directory: '.../scripts/backtest_nfl_props.py'
```

Step 4 (green), after implementation:
```
tests/nfl/test_backtest_nfl_props.py::test_run_backtest_returns_per_market_metrics PASSED
tests/nfl/test_backtest_nfl_props.py::test_fit_calibration_returns_sigmas PASSED
tests/nfl/test_backtest_nfl_props.py::test_no_leak_uses_prior_season_only PASSED
3 passed
```

## Concerns

1. **No historical props market exists to validate against** -- sigma/nb_var_mult are calibrated to outcome residuals, which is honest but means there is no OOS "beat the book" check here at all (unlike the game-line backtest's model-only/blend/market-only comparison). Live CLV tracking is the only real validation once this serves actual lines.
2. **Usage/efficiency shrinkage (`k_usage`/`k_eff` = 4.0) are carried through from Tasks 2-3 as fixed constants, not re-tuned by this backtest** -- the brief scopes this task to fitting sigma/nb_var_mult only; a follow-up could jointly tune k_usage/k_eff against this same residual objective.
3. **`pass_yds` sigma (106.9) and loc bias (+54.8) are far larger than the other yardage markets (24-35 sigma) -- traced to genuine QB job-security/team-change regime shifts that a prior-season-shares model structurally cannot see.** Example from the real data: Joe Flacco (`00-0026158`) split 2022 between spot starts for NYJ (`pass_att_share=0.169` after usage shrinkage, reflecting a part-time backup role), then signed with CLE for 2023 and started outright (42-45 attempts/game, weeks 13-17) -- his 2022-derived share projects ~6 pass attempts/game for 2023, when he actually threw ~44. This is not a code bug (the non-QB markets, which are far less winner-take-all, show tight/sane sigmas of 24-35 with 73-83% 1-sigma coverage) -- it is the real, load-bearing limitation of projecting purely from S-1 season-level shares with no in-season depth-chart/injury signal for who wins a QB competition. A follow-up (in-season share updates, or an explicit backup/starter transition flag) would likely shrink `pass_yds` sigma the most of any market.
4. **Market inclusion is volume-gated (nonzero projected pass_att/targets/carries), not a hardcoded position map** -- this is a deliberate, data-driven choice (see code comment on `_markets_for_volume`) but means, e.g., a QB who also has real receiving volume (extremely rare) would be scored on reception_yds too; this is correct behavior, not a bug, but worth knowing the gate is volume-based rather than position-based.

## Commands used

- `uv run pytest tests/nfl/test_backtest_nfl_props.py -v` (red, then green)
- `PYTHONPATH=src uv run python scripts/backtest_nfl_props.py` (real fit, writes `assets/nfl/props.json`)
- `uv run pytest -q` (full suite)
