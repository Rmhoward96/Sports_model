# +EV Sub-project 3: True-Probability Model + Backtest (GO/NO-GO) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** Fit a feature-based **margin** model and **total** model (from sub-project 2's `features.parquet`), turn them into calibrated win/cover/over probabilities via the existing distribution + calibration machinery, and run a **leak-free walk-forward backtest** whose output is the **go/no-go**: does the feature model's margin/total MAE beat the market's closing line, and are its probabilities well-calibrated? If not, that's a valid, honest NO-GO (like the base Elo model, MAE 13.36 vs line 12.37).

**Architecture:** A pure `model/trueprob.py` (fit + predict, regularized linear). A walk-forward backtest `scripts/backtest_trueprob.py --sport` mirroring `backtest_nfl_gameline.py`: at each season, train on prior seasons only, predict the current season, then compare to actuals AND to the closing line; fit residual σ; derive win prob via `distributions` + Platt calibration (`model/calibration.fit/apply`); report MAE-vs-actual, **MAE-vs-line (the edge test)**, Brier, reliability, and a CLV proxy.

**Tech Stack:** Python 3.12, numpy, pandas, pytest. Reuse `model/distributions.py` (`normal_to_margin_pmf`, `normal_to_pmf`, `prob_cover`, `prob_over_dist`), `model/calibration.py` (`fit`, `apply`).

**Spec:** `docs/superpowers/specs/2026-09-09-plus-ev-engine-design.md` (sub-project 3).

## Global Constraints

- **Leak-free walk-forward is non-negotiable.** For season S, the model is fit ONLY on seasons < S. Features are already leakage-free (sub-project 2: prior-season EPA/PPA, strictly-prior L10/SOS/SOV). Never fit on the season you score.
- **Honest go/no-go.** The headline metric is **margin MAE vs the closing line** (and total MAE vs the line). If the model does not beat (or at least match) the line, say so plainly — do not tune until it looks good. Report calibration (reliability/Brier) with equal prominence; a model that beats MAE but is miscalibrated is not usable.
- **Reuse the math.** Win prob = `prob_cover(normal_to_margin_pmf(margin, σ, offset), 0.0)`; cover @ line and over @ line via the same. Calibrate with `calibration.fit(probs, ys)` → `apply`. Do NOT hand-roll logistic/normal code that already exists.
- **Start simple.** Regularized linear (ridge via numpy `lstsq` on standardized features, or closed-form ridge). No gradient boosting / neural nets in v1 — interpretability + not overfitting a thin sample matters more than R². Impute missing features (e.g. debut-game NaNs, missing EPA) to the training-set mean, computed on the training split only (no leakage).
- **Desk fold is a scoring-time hook, not part of the backtest.** The desk didn't exist historically, so the backtest evaluates the base feature model. Provide the clamped desk-adjustment function (spec: ≤±3.5 pts margin, ≤~10pp prob) as a pure, unit-tested hook for sub-project 4 to apply at live scoring; the backtest does not use it.
- Run tests with `PYTHONPATH=src uv run pytest`. Nothing here touches the prediction pages, prediction model, or desk data.

## Feature data note

The backtest reads `assets/<sport>/features.parquet` (sub-project 2). If absent, generate it first: `CFBD_API_KEY=... uv run python scripts/build_features.py --sport nfl` (NFL pulls nflverse pbp — large; CFB pulls CFBD PPA + is slow). **NFL is the primary go/no-go** (cleaner, faster); CFB follows once NFL's pattern is validated.

---

### Task 1: True-probability model (pure fit + predict)

**Files:**
- Create: `src/sportsmodel/model/trueprob.py`
- Create: `tests/model/test_trueprob.py`

**Interfaces:**
- Produces (PURE, numpy/pandas in, arrays/dicts out):
  - `standardize(X, mu, sd) -> X_std` and helpers to compute `mu`/`sd` on a training matrix.
  - `fit_ridge(X, y, alpha) -> coef` (closed-form ridge: `coef = (XᵀX + αI)⁻¹ Xᵀy`, with an intercept column; do not penalize the intercept).
  - `FEATURES` — the ordered list of feature columns used (the `*_diff` columns from `features.parquet`: `elo_diff, last10_diff, sos_diff, sov_diff, rest_diff` (NFL) / rest proxies (CFB), `off_epa_diff, def_epa_diff` (NFL) / `_ppa_diff` (CFB); plus home-field is implicit in home−away diffs, so include a constant/intercept). Provide `feature_matrix(df, features, mu=None, sd=None) -> (X, mu, sd)` that selects columns, imputes NaN→train mean, standardizes.
  - `fit_model(train_df, target, features, alpha) -> Model` (Model = coef + mu + sd + features + target) and `predict(model, df) -> np.ndarray`.
  - `residual_sigma(y_true, y_pred) -> float` (std of residuals).

- [ ] **Step 1: Test-first** — `tests/model/test_trueprob.py`: build a small synthetic DataFrame where `margin` is a known linear function of two `*_diff` features + noise; assert `fit_model`/`predict` recover the signal (predicted correlates strongly with the true function; MAE < a loose bound), that NaN features are imputed to the train mean (a row with NaN doesn't crash and gets the mean), and `residual_sigma` returns the residual std. Assert `fit_ridge` with large `alpha` shrinks coefficients toward 0. Run (FAIL).

- [ ] **Step 2: Implement `model/trueprob.py`** to pass. Keep everything pure and numpy-based. Run (PASS).

- [ ] **Step 3: Commit** — `git commit -m "feat(+ev): true-prob feature model (ridge margin/total fit + predict)"`

---

### Task 2: Probability + calibration + desk hook (pure)

**Files:**
- Create: `src/sportsmodel/model/trueprob_prob.py` (or extend `trueprob.py`)
- Create: `tests/model/test_trueprob_prob.py`

**Interfaces:**
- `win_prob(margin, sigma) -> float` = `prob_cover(normal_to_margin_pmf(margin, sigma, offset), 0.0)` (pick a sane integer `offset`/range consistent with `normal_to_margin_pmf`'s contract — read its docstring).
- `cover_prob(margin, sigma, home_line) -> float`, `over_prob(total, sigma_total, line) -> float` (via `prob_cover`/`prob_over_dist`).
- `desk_adjust(margin, desk_pick) -> float` — the clamped scoring-time hook: shift `margin` by up to ±3.5 pts scaled by the desk's conviction tier (high/med/low) and side agreement, then clamp so the resulting `win_prob` moves ≤ ~0.10 vs the un-adjusted `win_prob`. Named constants `DESK_MAX_PTS = 3.5`, `DESK_MAX_PROB_DELTA = 0.10`. Pure; unit-test the clamp (a huge nominal shift is capped to ≤0.10 prob move; no desk pick → unchanged).

- [ ] **Step 1: Test-first** — assert `win_prob(0, σ) ≈ 0.5`; `win_prob(+7, σ) > 0.5`; monotonic in margin. `cover_prob`/`over_prob` sane and in [0,1]. `desk_adjust`: a high-conviction agree pick nudges the prob up but the clamp caps the move at ≤0.10; a low-conviction pick moves it less; no pick → identity. Run (FAIL).
- [ ] **Step 2: Implement** using `distributions`. Run (PASS).
- [ ] **Step 3: Commit** — `git commit -m "feat(+ev): win/cover/over probabilities + clamped desk hook"`

---

### Task 3: Walk-forward backtest → GO/NO-GO report

**Files:**
- Create: `scripts/backtest_trueprob.py`
- Create: `tests/test_backtest_trueprob.py` (pure metric helpers)

**Interfaces:**
- Pure helpers (tested): `margin_mae(preds)`, `total_mae(preds)`, `mae_vs_line(preds, line_key)` (the edge test), `brier(probs, outcomes)`, `reliability(probs, outcomes, bins)` (calibration table), `clv_proxy(preds)` (signed model-vs-closing-line, the disciplined proxy — mirror `backtest_cfb_priors`'s clv_proxy if present).
- `run_walk_forward(features_df, target_features, alpha) -> dict` — for each season S (with ≥1 prior season): fit margin+total on seasons < S (Task 1), predict S, fit σ on the TRAINING residuals, compute win prob (Task 2) + Platt-calibrate using prior-seasons' (prob, outcome) via `calibration.fit`, collect per-game rows (pred/actual margin+total, closing line, win prob, outcome). Return aggregated metrics.
- `main(--sport, --alpha)`: load `assets/<sport>/features.parquet` (the schedule-features + targets + closing line; NFL closing line = `spread_line`/`total_line` carried from the nflverse schedule — ensure sub-project 2's assembler kept them, or join them here from the schedule), run the walk-forward, and PRINT the go/no-go table: model margin MAE, **line margin MAE**, Δ; model total MAE, line total MAE, Δ; win-prob Brier + reliability; mean CLV proxy; % of games where model disagrees with the line. Write the report to `docs/superpowers/reports/2026-09-09-plus-ev-trueprob-backtest.md`.

- [ ] **Step 1: Test the pure helpers** on tiny hand-built prediction lists (known MAE, Brier, a reliability bucket). Run (FAIL→implement→PASS).
- [ ] **Step 2: Implement `run_walk_forward` + `main`.** Leak-free (train seasons < S). Impute/standardize using TRAIN stats only. Run the helper tests.
- [ ] **Step 3 (DATA RUN — controller may run):** ensure `assets/nfl/features.parquet` exists (generate via `build_features.py --sport nfl` if needed), then `PYTHONPATH=src uv run python scripts/backtest_trueprob.py --sport nfl`. Record the go/no-go numbers in the report. (CFB run follows.)
- [ ] **Step 4: Commit** — `git commit -m "feat(+ev): walk-forward true-prob backtest + go/no-go report"`

---

## The go/no-go decision (after Task 3's data run)

Read the report honestly:
- **GO** if the model's margin/total MAE is **≤ the closing line's** (or within noise) AND win-prob is well-calibrated (reliability roughly on the diagonal, Brier ≤ a market baseline). Then proceed to sub-project 4 (edge/EV engine) — there is plausibly real, surfaceable edge.
- **NO-GO / partial** if MAE is worse than the line (like the base model) or probabilities are miscalibrated. Then STOP before building the UI: report that the feature model does not beat the market, and either (a) iterate features/regularization, or (b) conclude the +EV view would surface only noise and pause. Either way, the honest measurement is the deliverable.

## Notes

- Closing line: NFL `spread_line`/`total_line` are in the nflverse schedule (home-line convention — confirm sign vs `prob_cover`'s `home_line`); CFB closing line from `assets/cfb/lines.parquet` (home-margin convention — convert). Make the convention explicit in `mae_vs_line`/`clv_proxy`.
- Keep the model simple; resist overfitting. The point is an honest number, not a leaderboard.
- Scoring the live slate (upcoming margin+σ/total+σ for sub-project 4) is a thin follow-on once GO — not in this plan's critical path.
