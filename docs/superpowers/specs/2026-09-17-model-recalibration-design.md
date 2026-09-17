# Sub-project A — Game-line model recalibration (design spec)

**Status:** draft for review · **Date:** 2026-09-17
**Program:** A (this) → B (NFL drive-based sim) → C (NFL player props). A is
independent and ships first.

## Goal

Make the analytic game-line model's margin/total predictions **honest**:
- **Refresh the spread calibration** (σ_margin, σ_total) so cover/over/win
  probabilities match the empirical residual SD.
- **Add a systematic-bias correction** to the point estimate (margin, total) —
  the genuinely new capability — since generation is model-only and any model
  bias flows straight into the predictions. Bias is the only lever that can move
  ATS/total **hit rate** (it shifts the point estimate, not just the spread).
- Prove both with the existing **walk-forward backtest**, and ship only the
  changes that improve held-out metrics.

## What already exists (important — A extends, not reinvents)

`scripts/backtest_nfl_gameline.py` and `scripts/backtest_cfb_gameline.py` are
walk-forward fitters that both sports share (CFB imports the NFL
`GameLineConfig`/`build_gameline` engine). They already:

- **Fit σ** (`tune_sigmas` = RMSE of walk-forward `pred − actual`), written to
  `assets/{nfl,cfb}/gameline.json`. Live values today:
  CFB `σ_margin 16.67 / σ_total 16.89`; NFL `σ_margin 13.42 / σ_total 13.58`
  (the 13.2/10 in the dataclass are just defaults the JSON overrides).
- **Fit the shrink curves** (`tune_shrink`).
- **Report calibration** in `run_backtest`: `brier` (win prob) and `ou_acc =
  P(actual_total > pred_total)` — a totals directional-bias metric (~0.5 =
  unbiased).

So σ-fitting is done and roughly on-target (CFB margin residual SD ≈ 16.6 ≈ the
fitted 16.67). **The gap: the harness *measures* total bias (`ou_acc`) but
nothing *corrects* it, and there is no bias term in `build_gameline`.** Note
generation passes an empty market to `build_gameline` (predictions are
model-only, no market blend), so a systematic model bias is uncorrected.

## Motivating evidence (live graded games — small sample, hence robustness rule)

Signed residuals (pred − actual):

| Sport | n | margin bias | total bias | fitted σ_margin / σ_total |
|---|---|---|---|---|
| CFB | 96 | −4.0 | **+4.5** (totals run high) | 16.67 / 16.89 |
| NFL | 16 | +1.8 | −4.9 | 13.42 / 13.58 |

CFB's +4.5 total bias on ~96 games is the clearest actionable signal. NFL's 16
games are one week — not trustworthy alone (drives the robustness rule below).

## Non-goals (explicit)

- **Not** chasing point accuracy beyond the market (the NO-GO stands; A fixes
  bias + calibration, not the ceiling). A correction that doesn't beat the ship
  gate is not applied (bias stays 0 / σ unchanged).
- **Not** the simulation (B) or props (C).
- **Does not change the +EV board's base probability** (anchored to Pinnacle
  no-vig, not the model). A improves the predictions page (ATS/total + bias),
  the desk's `model.win_prob` context, and the calibrated analytic baseline B's
  disagreement signal compares against.

## Architecture

### 1. Bias term (the new capability)

- Add `bias_margin: float = 0.0`, `bias_total: float = 0.0` to `GameLineConfig`
  (`src/sportsmodel/nfl/gameline.py`); both sports' loaders read them (default
  0.0 when absent → old configs unaffected).
- `build_gameline` subtracts them from the model inputs **before** `shrink()`:
  `corrected = model_value − bias_*`. Residual is `pred − actual`, so a positive
  bias (model runs high, e.g. CFB totals) is subtracted.
- Both sports flow through this one choke point, so it's a single change.

### 2. Fit bias in the existing harness (robustly)

- Extend `backtest_*_gameline.py` with a bias fit: `bias_raw = mean(pred −
  actual)` over the walk-forward TRAIN span (same residuals `tune_sigmas`
  already computes), then **shrunk**:
  `bias = clamp(shrink_factor × bias_raw, −CAP, +CAP)`.
  - `shrink_factor` ∈ (0,1] (default 0.5) — take only the portion that
    generalizes out-of-sample.
  - `CAP` (default ±3.0 margin / ±3.0 total) bounds a bad fit.
- Estimate from a **robust sample — full prior season(s) via the walk-forward
  harness — not the 96/16 live games.** Report per sport.
- Re-fit σ in the same run (refresh), keeping the current values if the new
  estimate is within noise.

### 3. Validation (ship-gate)

`run_backtest` already emits `margin_mae`, `total_mae`, `brier`, `ou_acc`. Add a
held-out **before vs after** report (walk-forward): margin/total MAE, Brier
(win + a cover/over Brier), ATS %, total %, and `ou_acc` (should move toward
0.5 after a total-bias correction).

**Ship gate:** apply a change only if held-out calibration improves (Brier /
`ou_acc`→0.5) and hit rate does not degrade. σ and bias are judged
independently (σ won't move hit rate; bias can move both). If NFL's prior-season
sample is thin, hold NFL bias at 0 and refresh σ only.

### 4. Config update path (auditable, human-approved)

The fit **prints** proposed JSON values; a human pastes approved values into
`assets/{nfl,cfb}/gameline.json` (same pattern the harness already uses). No
auto-write of configs from CI.

## Files touched

- `src/sportsmodel/nfl/gameline.py` — `GameLineConfig` + `build_gameline` bias.
- `src/sportsmodel/nfl/config.py`, `scripts/generate_cfb.py::load_gameline` —
  read `bias_margin`/`bias_total` (default 0.0).
- `scripts/backtest_nfl_gameline.py`, `scripts/backtest_cfb_gameline.py` — bias
  fit + before/after calibration report.
- `assets/nfl/gameline.json`, `assets/cfb/gameline.json` — refreshed σ + new
  bias fields (only after the ship gate passes).
- Tests: `build_gameline` subtracts bias with the correct sign; config defaults
  bias to 0.0 when absent; bias-fit shrink/clamp helper is correct; a no-op
  (bias 0, σ unchanged) leaves predictions identical.

## Rollout

1. Land code with bias defaulted to 0.0 and σ unchanged — a no-op refactor that
   passes CI.
2. Run the recalibration backtest; review proposed σ/bias per sport vs the ship
   gate.
3. Paste approved values into the two gameline.json files; re-generate
   predictions (`generate-nfl`, `generate-cfb`).

## Open questions

- **Shrink factor / cap** — 0.5 and ±3.0 as defaults, or tune per sport on the
  folds? (Recommend: tune, decided on held-out Brier/MAE.)
- **Prior-season depth** — confirm `schedules.parquet` has enough completed
  prior-season games for a stable per-sport estimate (esp. NFL).
- **NFL bias** — hold at 0 until a prior-season estimate is available, σ-only
  refresh for NFL now? (Recommend yes.)
