# +EV true-prob model -- walk-forward backtest / GO-NO-GO

**Sport:** nfl  **alpha:** 1.0
**Features:** elo_diff, last10_diff, sos_diff, sov_diff, rest_diff, off_epa_diff, def_epa_diff
**Games scored:** 2761  **Seasons scored:** [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

Leak-free walk-forward: each season S is scored by a model fit only on seasons < S, with sigma from TRAIN residuals and Platt calibration fit only on the prior seasons' own (win_prob, home_win) pairs. See scripts/backtest_trueprob.py's module docstring for the closing-line sign convention (verified empirically for NFL's spread_line).

## Margin / total accuracy vs the closing line

| metric | model MAE | line MAE | delta (model - line) |
|---|---|---|---|
| margin | 10.207 | 9.777 | +0.430 |
| total | 11.023 | 10.438 | +0.585 |

(delta < 0 means the model beats the closing line on this metric.)

## Win-probability calibration

Brier (raw): 0.2224
Brier (calibrated): 0.2226

| bin | n | mean_pred | mean_actual |
|---|---|---|---|
| [0.1, 0.2) | 39 | 0.166 | 0.282 |
| [0.2, 0.3) | 138 | 0.259 | 0.304 |
| [0.3, 0.4) | 325 | 0.356 | 0.348 |
| [0.4, 0.5) | 488 | 0.454 | 0.414 |
| [0.5, 0.6) | 597 | 0.551 | 0.534 |
| [0.6, 0.7) | 616 | 0.647 | 0.643 |
| [0.7, 0.8) | 397 | 0.748 | 0.733 |
| [0.8, 0.9) | 147 | 0.841 | 0.871 |
| [0.9, 1.0) | 14 | 0.911 | 0.857 |

## Edge proxies

- mean CLV proxy (signed model-minus-line margin edge, positive = model favors home more than the market): +0.201
- disagreement rate (model's side != the line's side): 0.167

## Verdict: NO-GO

- margin MAE: model=10.207 line=9.777 delta=+0.430 (FAIL vs tolerance 0.25)
- total MAE: model=11.023 line=10.438 delta=+0.585 (FAIL vs tolerance 0.25)
- calibration: mean|mean_pred-mean_actual| across bins=0.037 (OK vs tolerance 0.1)

NO-GO / partial: at least one of the margin/total MAE deltas or the calibration check failed the threshold above. Per the plan, STOP before building the +EV UI -- iterate on features/regularization, or conclude the +EV view would surface only noise and pause.
