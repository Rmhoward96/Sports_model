# Shrink model prop probabilities toward the market (profit project, step 2 Part A) — design spec

Date: 2026-10-13 · Status: design approved in chat (three sections); awaiting user review of this file

**Evidence.** `docs/superpowers/reports/2026-10-13-edge-audit.md`:
- §3: NFL prop leans claim 64.8% and hit 51.8%; recalibration slope 0.15, SE 0.07, n = 1,587.
- §7 item 2: the roadmap step and its pass test.

**Where this fits.** Step 2 Part B, game-level model probabilities on the site, is a separate spec that follows this one.

## Goal

Replace every model probability that decides a prop pick with a probability shrunk toward the market, `market + k × (model − market)`. k is fitted walk-forward from graded legs and refit weekly. The model then moves a pick only by as much as the evidence says it has earned. A pre-registered test judges whether the shrunk model adds anything beyond the market price.

## User decisions (2026-10-13)

- **Switch now.** The shrunk probability drives picks, EV and parlays from day one, with no shadow period.
- **If the pass test fails,** the user and Claude decide together what happens to model prop picks. Nothing is removed automatically.
- **Two parts.**
  - Part A covers the live bet signals: the NFL and MLB prop boards and the prop legs in parlays.
  - Part B covers game-level model numbers on the site: Alpha, edges, the game-page read and model moneyline/cover/total probabilities.

## Current state (from code exploration)

- **Props board.**
  - `scripts/build_ev_props_board.py` calls `serving/props_ev.py::assemble_prop_rows`.
  - The model P(over) at the board's main line is `prob_over_dist(dist, line)`.
  - MLB applies `calibrate_fn=calibration.calibrate` (Platt, `assets/calibration.json`). NFL calibration is identity.
  - `market_prob` is the no-vig of the best soft over and best soft under at the same line (Pinnacle excluded from the soft pool).
  - EV: `ev_over = ev(p_over, best_over)`, `ev_under = ev(1 − p_over, best_under)`, and the better side is taken.
  - Pick rule: `is_pick = 0 < ev_best ≤ 0.25`.
  - Each row stores `model_prob`, `market_prob`, `edge` and `ev_best` in `ev_prop_picks`.
- **Parlays.** In `serving/best_parlays.py`, prop legs take `prob = model_prob` from `ev_prop_picks_current` (NFL only). Legs need `prob ≥ 0.55`.
- **Game-line +EV board and early tickets** are pure market (Pinnacle no-vig). No change.
- **Fit data.**
  - `nfl_prop_grades`, a view over `nfl_prop_lines` and `nfl_player_actuals`, holds every NFL sim lean that had a book line, not only board picks.
  - Its columns are `p_over`, the best `over_price` and `under_price` across books at the line, and the hit/miss/push result.
  - MLB has only graded board picks (`ev_prop_results`, about 90 legs), which are selection-biased.
- **Fit-and-commit template.** `train-cover-ensemble.yml` is the weekly DB read → fit → commit pattern. `report-ledger-gate.yml` has the 3-attempt `pull --rebase` push retry.

## 1. The shrink and how k is fitted

**The formula, per prop at the book's line:**

> `shrunk = clip(m + k × (p − m), 0.01, 0.99)`

- p is the model's probability of the over (after MLB's existing calibration).
- m is the market's no-vig probability of the over at the same line, computed exactly as the board computes `market_prob` today: best soft over and best soft under, vig removed.
- The fit uses the identical m construction.
- The under side uses `1 − shrunk`.

**k for one group of legs.** The Brier-minimising closed form, clamped:

> `k = clamp( Σ(y − m)(p − m) / Σ(p − m)², 0, 1 )`

- y is 1 when the over hit.
- Pushes are excluded.
- If `Σ(p − m)² = 0`, k = 0.

**The fit sample.**
- Graded NFL leans from `nfl_prop_grades`, with p = `p_over` and m = the no-vig of its `over_price` and `under_price`.
- Rows missing either price, or with an invalid American price (|price| < 100), are excluded.

**Pooling and minimums.**
- A pooled NFL k is fitted on all eligible legs.
- Each market gets its own k, partially pooled: `k_market = (n_m × k_m_raw + 300 × k_pool) / (n_m + 300)`.
- If the pooled sample has fewer than **300** legs, the NFL k is 0 for every market.

**Model version.**
- The fit uses legs from the current props model version, props-ML v2, whose version string is read from the data.
- If that gives fewer than 300 legs, it uses all versions pooled.

**MLB.** k = 0 for every MLB market in Part A, because there is no unbiased fit sample. The effect is that MLB prop picks depend only on price until a later spec adds an MLB fit.

**Walk-forward.** The k used during a week is fitted only on legs graded before that week. The weekly refit (§3) is what provides this.

**Committed parameters.** `assets/shrink_k.json` records, per `(sport, market)`:
- `k`, `n` and `k_raw`;
- the pooled values;
- `model_version_used` and `fitted_at`;
- the data window.

Market keys are the board's odds market keys. The sim-to-odds market mapping already used by the board is applied when fitting.

## 2. Serving changes

**Props board** (`serving/props_ev.py::assemble_prop_rows`, NFL and MLB).
- Right after p is computed at the book line (and after `calibrate_fn` for MLB), it is replaced by `shrunk` using `k(sport, market)` from `assets/shrink_k.json`.
- Everything downstream uses the shrunk value: EV at the best price, the pick decision, the stored `model_prob` and `edge`, and therefore the site's Alpha Score, edge and confidence. There are no site code changes.
- **Fail-safe.**
  - If the file is missing or unreadable, or the `(sport, market)` entry is missing, k = 0.
  - The script warns once per run.
  - It never falls back to the raw model.
- **Pick rule unchanged:** `0 < ev_best ≤ 0.25`. Far fewer picks are expected.

**Audit columns.**
- `ev_prop_picks` gains two nullable columns, `raw_model_prob` (p before shrinking) and `shrink_k` (k applied).
- The writer includes them only when the columns exist, so the board works before the migration.

**Parlays.**
- Prop legs read `model_prob` from `ev_prop_picks_current`, so they get the shrunk value with no code change.
- `MIN_LEG_PROB` stays at 0.55. Most prop legs will no longer qualify.

**Pick ledger.**
- `ev_prop` and `parlay_leg` rows record the shrunk `model_prob`, the number that made the pick.
- No ledger change.

**Unchanged:**
- the sims and props-ML models;
- `nfl_prop_lines` and `nfl_prop_grades`, and their site pages (the raw model's record, which is also the fit source);
- game-line boards, early tickets, and all game-level numbers (Part B).

## 3. Weekly refit and the pass test

**Refit job.** New `scripts/fit_shrink_k.py` and workflow `fit-shrink-k.yml`.
- **Schedule:** Tuesdays at 15:30 UTC, plus `workflow_dispatch`.
- **What it does:**
  - reads `nfl_prop_grades` and the model version of each leg (read-only);
  - computes k per §1;
  - writes `assets/shrink_k.json`;
  - commits and pushes with the 3-attempt `pull --rebase` retry.
- **Failure handling:** on any fetch or fit failure it exits non-zero and leaves the existing file untouched. It never writes a k it did not compute.
- **Workflow mechanics:**
  - Job `timeout-minutes` is set.
  - Secrets are passed through `env`, with no `${{ }}` inside `run:`.
  - `permissions: contents: write`.
- **First run:** dispatched manually right after merge.

**Pass test** (pre-registered from audit §7 item 2; frozen as of this spec).

*Evaluation set:*
- Each graded NFL lean in week W, keyed by `(season, week)` from `nfl_prop_lines`, is scored with the shrunk probability.
- That probability uses the k fitted by the §1 rules on legs graded in weeks before W only.
- Weeks whose earlier-leg count is below 300 are not scored.
- Already-graded weeks count, as the audit specified.
- Pushes are excluded.

*Judgement point:* judged **once**, at the first weekly report with ≥ **2,000** scored legs. Before that the status is "collecting".

*Pass requires both of these:*
1. **Accuracy vs the market.** `Brier(m) − Brier(shrunk) ≥ 0.001`. The paired bootstrap 95% CI of that difference must have its lower bound > 0. The bootstrap resamples games with replacement: 10,000 resamples, seed 20261013, 2.5th percentile.
2. **Calibration.** The logistic regression `y ~ a + b · logit(shrunk)` has slope b in [**0.8, 1.2**].

*Weekly report.* Committed by the same workflow to `docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md`. It shows:
- current k per market with n;
- scored legs so far;
- the running Brier difference and slope;
- the status.

Once judged, the verdict is carried forward unchanged.

*What the verdict means:*
- **Pass:** shrunk model prop picks stay. A pass means the model adds information beyond the board's market price. It does not prove profit; CLV in the pick ledger judges that.
- **Fail:** the user and Claude decide together. Until then, picks keep using the shrunk probabilities.

## Migration (user runs)

`db/migration_shrink_props.sql`:
- `ALTER TABLE ev_prop_picks ADD COLUMN IF NOT EXISTS raw_model_prob double precision, ADD COLUMN IF NOT EXISTS shrink_k double precision;`
- Idempotent and additive.
- Views that select explicit columns are unaffected.

## Testing

All tests are network-free and DB-free. The existing suite stays green.

- **Fitting:**
  - the closed-form k on synthetic data (recovers a known k), the clamp to [0, 1], and zero variance giving k = 0;
  - the 300-leg partial pooling;
  - the k = 0 rules (pooled n < 300, MLB, missing file or entry);
  - the model-version fallback;
  - invalid-price and push exclusion.
- **Serving:**
  - the shrink formula and its clipping;
  - the board applying the right k per sport and market;
  - MLB order: calibrate, then shrink;
  - `raw_model_prob` and `shrink_k` written only when the columns exist;
  - the pick decision using the shrunk EV;
  - parlay legs receiving shrunk probabilities.
- **Pass test:**
  - walk-forward scoring (no week uses its own or later legs);
  - the 300 earlier-leg threshold;
  - the 2,000 judgement point;
  - both pass conditions at their boundaries;
  - the bootstrap seed and game clustering;
  - the frozen verdict.
- **Refit job:** writes the file only on success, plus the workflow schedule, env-passed secrets, timeout and push retry.

## Out of scope

- Part B (game-level model probabilities, site Alpha/edge for game lines, the ledger's `model_game`).
- Any change to the sims or models.
- An MLB fit sample.
- Changing the pick thresholds.
- Removing any product.
- Site code.

## Risks

- **Few picks.** With k ≈ 0.15 the board shows far fewer prop picks and parlays are often empty. This is intended.
- **Biased market reference.** m comes from the best prices across books, not a sharp source, so it carries some bias. Using the same m in fit and serve keeps k internally consistent.
- **Model changes.** A new props model version resets the current-version sample. The fallback pools versions until 300 legs exist.
- **Information leak.** The audit has already seen the existing graded weeks, so the test is not fully blind on that data. The walk-forward scoring and the fixed thresholds limit the harm.
