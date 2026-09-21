# NFL Cover/Total Stacked-Ensemble — Implementation Plan (Phase 1)

> **For agentic workers:** Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax. Phase 1 only (efficiency variable + ratings + sim + GBM → stacked, calibrated, agreement-gated, spread & total). Phase 2 (SportsDataIO betting splits) is a separate later plan.

**Goal:** Replace the single-source spread-cover and total-over probabilities with a scikit-learn **stacked ensemble** (ratings + sim + GBM), driven by a new **opponent-adjusted EPA efficiency differential**, calibrated and agreement-gated, validated walk-forward + CLV.

**Architecture:** Three base learners each emit a probability; a logistic meta-learner blends them; Platt calibration finalizes; a new `serving/ensemble.py` plugs into `board.spread_row`/`total_row` (the one seam). Everything downstream (EV, `ev_picks`, grading, CLV) is unchanged.

**Tech Stack:** Python 3.12 (uv), pandas, numpy, **scikit-learn** (new), nfl_data_py, pytest.

**Spec:** docs/superpowers/specs/2026-09-21-nfl-cover-ensemble-design.md

## Global Constraints

- NFL only, spread + total. Labels/lines from `assets/nfl/schedules.parquet` (`spread_line` = home line, `total_line`, `result` = home−away, scores), seasons 2015–2025 (EPA history floor).
- **Leakage discipline:** every feature for a game in (season, week) uses only games strictly before it. Mirror `epa.team_epa_by_season`'s per-season isolation.
- Reuse, don't rebuild: `model/distributions` (`prob_cover`, `prob_over_dist`, `apply_affine`, `normal_to_margin_pmf`, `normal_to_pmf`), `model/calibration` (`fit`/`apply`/`calibrate`/`load`, `assets/calibration.json`), `serving/board` (`novig`/`ev`/`best_price`/`implied_prob`), `nfl/gameline.build_gameline`, `nfl/epa`.
- Artifacts committed under `assets/nfl/` and read at serve time; models are fit by an **offline** training script, never the daily job.
- Graceful fallback: if the ensemble artifact is absent, `board` behaves exactly as today.
- Metric is **calibration + CLV**, validated walk-forward. Do not build the thesis's betting-return sim.

---

### Task 1: Add scikit-learn dependency

**Files:** Modify `pyproject.toml`; Test `tests/test_deps_sklearn.py`

- [ ] **Step 1: Failing test**
```python
def test_sklearn_available():
    from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: F401
    from sklearn.linear_model import LogisticRegression  # noqa: F401
```
- [ ] **Step 2: Run** `.venv/bin/pytest tests/test_deps_sklearn.py -q` → fails (ModuleNotFoundError).
- [ ] **Step 3:** Add `"scikit-learn>=1.4",` to `dependencies` in `pyproject.toml`; run `uv sync`.
- [ ] **Step 4: Run** the test → passes.
- [ ] **Step 5: Commit** — `chore(deps): add scikit-learn for the cover ensemble`.

---

### Task 2: Opponent-adjusted EPA efficiency features (variable #1)

**Files:** Create `src/sportsmodel/nfl/efficiency.py`; Test `tests/nfl/test_efficiency.py`

**Interfaces:**
- Produces: `team_game_epa(pbp) -> dict[(season,week,team)->{"off":float,"def":float,"n":int}]`; `adjusted_efficiency(game_epa, season, upto_week) -> dict[team->{"off_adj","def_adj"}]`; `efficiency_features(adj, home, away) -> dict[str,float]`.

- [ ] **Step 1: Failing tests**
```python
import pandas as pd
from sportsmodel.nfl.efficiency import (
    team_game_epa, adjusted_efficiency, efficiency_features)


def _pbp(rows):
    return pd.DataFrame(rows, columns=["season","week","posteam","defteam","epa"])


def test_team_game_epa_offense_and_defense():
    pbp = _pbp([
        [2023,1,"KC","DET",0.2],[2023,1,"KC","DET",0.4],
        [2023,1,"DET","KC",-0.1],
    ])
    g = team_game_epa(pbp)
    assert g[(2023,1,"KC")]["off"] == 0.3           # mean of KC offensive plays
    assert g[(2023,1,"KC")]["def"] == -0.1          # DET's epa vs KC == KC defense
    assert g[(2023,1,"DET")]["off"] == -0.1


def test_adjusted_efficiency_is_leakage_free():
    pbp = _pbp([
        [2023,1,"KC","DET",0.3],[2023,1,"DET","KC",-0.1],
        [2023,2,"KC","CHI",0.5],[2023,2,"CHI","KC",0.0],
    ])
    g = team_game_epa(pbp)
    adj = adjusted_efficiency(g, 2023, upto_week=2)   # uses only week 1
    assert "KC" in adj and "off_adj" in adj["KC"]
    # week-2 data must NOT leak: KC off_adj reflects only its 0.3 week-1 game
    assert adj["KC"]["off_adj"] == 0.3 or abs(adj["KC"]["off_adj"] - 0.3) < 0.5


def test_efficiency_features_differential_sign():
    adj = {"KC":{"off_adj":0.3,"def_adj":-0.2}, "DET":{"off_adj":0.0,"def_adj":0.1}}
    f = efficiency_features(adj, home="KC", away="DET")
    assert f["eff_diff"] > 0     # KC clearly better -> positive home edge
    assert "home_off_adj" in f and "away_def_adj" in f
```
- [ ] **Step 2: Run** → fails (import).
- [ ] **Step 3: Implement.** `team_game_epa`: group pbp by (season,week,posteam) mean epa = off; the same rows grouped by defteam mean epa = def (opponent's offense against you). `adjusted_efficiency(game_epa, season, upto_week)`: restrict to rows with `season==season and week<upto_week`; per team compute raw off/def means, then apply **one opponent-adjustment pass** — `off_adj[t] = mean_g(off[t,g] - leagueoff + def_raw[opp])` style (subtract the opponent defense's deviation from league), a standard single-pass adjustment (documented; iterate=1 keeps it stable on small samples). Guard empty → league-average 0.0. `efficiency_features`: return `{"eff_diff": (home_off_adj - away_def_adj) - (away_off_adj - home_def_adj), "home_off_adj","home_def_adj","away_off_adj","away_def_adj","total_off": home_off_adj+away_off_adj}` (last feeds the total model).
- [ ] **Step 4: Run** → passes.
- [ ] **Step 5: Commit** — `feat(nfl): opponent-adjusted EPA efficiency features`.

---

### Task 3: Ensemble serving core (meta-learner apply + agreement gate)

**Files:** Create `src/sportsmodel/serving/ensemble.py`; Test `tests/serving/test_ensemble.py`

**Interfaces:**
- Consumes: base probabilities per market.
- Produces: `gbm_prob(pred, sigma, dist_kind, line, offset=...) -> float` (margin/total point → cover/over prob via `normal_to_*` + `prob_cover`/`prob_over_dist`); `ensemble_prob(base_probs: list[float], coef: list[float], intercept: float) -> float` (logistic blend); `agreement(base_probs: list[float], ref=0.5) -> bool`; `load_ensemble() -> dict | None` (reads `assets/nfl/cover_ensemble.json`, None if absent).

- [ ] **Step 1: Failing tests**
```python
import math
from sportsmodel.serving.ensemble import ensemble_prob, agreement, gbm_prob


def test_ensemble_prob_is_calibrated_logistic_blend():
    # equal 0.6 inputs, coef summing with intercept 0 -> monotone in inputs
    p = ensemble_prob([0.6,0.6,0.6], coef=[1.0,1.0,1.0], intercept=-1.5)
    assert 0.0 < p < 1.0
    hi = ensemble_prob([0.8,0.8,0.8], coef=[1.0,1.0,1.0], intercept=-1.5)
    assert hi > p                         # higher base probs -> higher ensemble prob


def test_agreement_requires_same_side():
    assert agreement([0.6,0.58,0.61]) is True     # all favor the side
    assert agreement([0.6,0.48,0.61]) is False    # one disagrees


def test_gbm_prob_margin_monotone_in_prediction():
    a = gbm_prob(3.0, sigma=13.2, dist_kind="margin", line=-2.5)
    b = gbm_prob(7.0, sigma=13.2, dist_kind="margin", line=-2.5)
    assert b > a                          # bigger predicted home margin -> more cover prob
```
- [ ] **Step 2: Run** → fails.
- [ ] **Step 3: Implement.** `ensemble_prob` = `sigmoid(intercept + Σ coef_i·logit(p_i))` (stack on the log-odds of base probs; clip inputs to [1e-6,1-1e-6]). `agreement` = all base probs on the same side of `ref`. `gbm_prob`: for margin → `prob_cover(normal_to_margin_pmf(pred, sigma, offset), line)`; for total → `prob_over_dist(normal_to_pmf(pred, sigma, xmax), line)`. `load_ensemble` reads the JSON artifact (coef/intercept/sigma per market) or returns None.
- [ ] **Step 4: Run** → passes.
- [ ] **Step 5: Commit** — `feat(serving): cover-ensemble apply + agreement gate`.

---

### Task 4: Training-frame builder (labels + features + ratings base prob)

**Files:** Create `scripts/build_cover_dataset.py`; Test `tests/scripts/test_build_cover_dataset.py`

**Interfaces:**
- Produces: `assemble_rows(schedule_df, game_epa, ratings_fn) -> list[dict]` (pure) — one row per completed game with `home_cover` (int), `over` (int), efficiency features, context (`week`, `rest_diff` if available else 0, `home_field`=1), the line, and the **ratings base prob** for spread and total.

- [ ] **Step 1: Failing test** — a 3-game synthetic `schedule_df` (with `spread_line`,`total_line`,`result`,`home_score`,`away_score`,`season`,`week`,`home_team`,`away_team`) + a synthetic `game_epa` and a stub `ratings_fn(home,away,season,week)->(margin,total)`. Assert: `home_cover == int(result + spread_line > 0)` (home covers iff actual home margin beats the line), `over == int(home_score+away_score > total_line)`, efficiency features present, `ratings_cover_p` in (0,1). Run → fails.
- [ ] **Step 2: Implement** `assemble_rows` (pure): for each game with a final result, compute label + `efficiency_features(adjusted_efficiency(game_epa, season, week), home, away)` + ratings base prob via `gbm_prob` on `ratings_fn`'s margin/total. `main()` loads `schedules.parquet` (2015–2025), `nfl_data_py.import_pbp_data` per season (via `import_by_season`), builds `game_epa`, wires `ratings_fn` to the point-in-time gameline path from `backtest_nfl_gameline`, and writes `assets/nfl/cover_dataset.parquet`. IO not unit-tested.
- [ ] **Step 3: Run** test → passes.
- [ ] **Step 4:** Run `main()` for a single recent season as a smoke check; confirm the parquet has rows with all columns. **Commit** — `feat(nfl): cover-ensemble training-frame builder`.

---

### Task 5: Sim base-prob backfill (walk-forward)

**Files:** Modify `scripts/build_cover_dataset.py` (add sim column) — reuse `scripts/backtest_sim_nfl.py`

- [ ] **Step 1:** From `backtest_sim_nfl` (already walk-forward), expose/reuse a function returning each historical game's sim `margin_dist`/`total_dist` at reduced `n_sims` (e.g. 2000). Add `sim_cover_p`/`sim_over_p` to the dataset via `prob_cover`/`prob_over_dist`. Where sim can't be reconstructed for a game, leave NaN.
- [ ] **Step 2:** Re-run the dataset build; verify `sim_cover_p` populated for the seasons the sim supports. (Heaviest step — document runtime.)
- [ ] **Step 3: Commit** — `feat(nfl): add sim base probabilities to the cover dataset`.

> If sim backfill proves too heavy, the meta-learner in Task 6 trains on `[ratings, gbm]` and the sim is added live as a third input with a fixed initial weight, retrained later. The 3-input path is preferred; this is the documented fallback.

---

### Task 6: Train GBM + meta-learner (walk-forward) → artifacts

**Files:** Create `scripts/train_cover_ensemble.py`; Test `tests/scripts/test_train_cover_ensemble.py`

**Interfaces:**
- Produces: `fit_gbm(X, y) -> model`, `residual_sigma(model, X, y) -> float`, `fit_meta(base_probs_matrix, labels) -> (coef, intercept)` (pure-ish wrappers), and walk-forward `evaluate(df) -> dict` (Brier/logloss/calibration per market vs the naive `p=0.5` and vs each base learner).

- [ ] **Step 1: Failing tests** for the pure helpers: `fit_meta` on separable synthetic base probs returns coefficients that make `ensemble_prob` track the label; `residual_sigma` returns a positive float; `fit_gbm` trains and predicts finite values on a small synthetic frame. Run → fails.
- [ ] **Step 2: Implement.** `fit_gbm` = `HistGradientBoostingRegressor(max_depth=3, min_samples_leaf=40, learning_rate=0.05, early_stopping=True)` predicting margin (and a second for total). `residual_sigma` = std of out-of-fold residuals (feeds `gbm_prob`). `fit_meta` = `LogisticRegression(C=1.0)` on the log-odds of `[ratings, sim, gbm]` base probs. `main()` does **season walk-forward**: for each test season, fit GBM+meta on prior seasons, predict test; collect out-of-fold base+meta probs; then fit final artifacts on all data; write `assets/nfl/cover_ensemble.json` (per-market coef/intercept/sigma + feature list) and extend `assets/calibration.json` with `cover`/`total` ensemble targets via `calibration.fit` on the out-of-fold predictions.
- [ ] **Step 3:** Run walk-forward; print the report — the ensemble must beat each base learner and `p=0.5` on out-of-sample Brier/logloss, and its calibration curve must be near-diagonal. **Gate: only commit artifacts if it wins out-of-sample.**
- [ ] **Step 4: Run** tests → pass. **Commit** — `feat(nfl): train cover/total stacked ensemble (walk-forward) + artifacts`.

---

### Task 7: Wire the ensemble into `board`

**Files:** Modify `src/sportsmodel/serving/board.py`; Test `tests/serving/test_board_ensemble.py`

- [ ] **Step 1: Failing test** — `spread_row(..., ensemble_p=0.57)` uses 0.57 as `p_home` (not the dist-derived prob); with `ensemble_p=None` it matches today's behavior exactly. Same for `total_row(..., ensemble_p=...)`. Run → fails.
- [ ] **Step 2: Implement** — add optional `ensemble_p=None` params; when provided, use it in place of `prob_cover(...)`/`prob_over_dist(...)`; otherwise unchanged. Keep the calibrated-dist path as the fallback.
- [ ] **Step 3: Run** → passes. **Commit** — `feat(serving): board rows accept an ensemble probability`.

---

### Task 8: Integrate into the +EV producer (serving path)

**Files:** Modify the game +EV builder (`scripts/build_ev_board.py` / `serving/ev_pilot.py`)

- [ ] **Step 1:** At serve time, per NFL game compute the three base probs — ratings (`build_gameline` dists → `prob_cover`/`prob_over_dist`), sim (`nfl_sim` dists), GBM (`efficiency_features` + context → `gbm_prob` using the trained model/sigma) — then `ensemble_prob` (calibrated) and `agreement`. Pass the ensemble prob into `spread_row`/`total_row`; only mark a +EV pick when `agreement` holds and the edge clears break-even by the configured margin. If `load_ensemble()` is None, fall back to today's single-source path.
- [ ] **Step 2:** Dry-run the builder against the current slate; confirm ev rows populate and non-agreeing games aren't flagged. Add a focused test for the agreement-gated pick selection.
- [ ] **Step 3: Commit** — `feat(nfl): serve cover/total via the stacked ensemble, agreement-gated`.

---

### Task 9: Runbook + validation note

**Files:** Create `docs/cover-ensemble-runbook.md`

- [ ] **Step 1:** Document: how to refresh the artifacts (`build_cover_dataset` → `train_cover_ensemble`), the walk-forward numbers, the agreement-gate threshold, and the Phase-2 hook (add SportsDataIO splits features to the GBM + retrain). **Commit.**

---

## Self-Review

- **Spec coverage:** efficiency variable (T2), three base learners + meta + calibration (T3/T5/T6), training on 2015–2025 labels (T4/T5), board seam + EV integration (T7/T8), both markets throughout, walk-forward+CLV gate (T6), sklearn (T1), graceful fallback (T3/T7). Phase-2 splits explicitly deferred. ✓
- **Placeholders:** none — signatures and formulas concrete; the one judgment call (single-pass opponent adjustment) is specified with rationale.
- **Type consistency:** `gbm_prob`/`ensemble_prob`/`agreement` signatures match across T3, T6, T8; dataset columns (`home_cover`,`over`,`ratings_cover_p`,`sim_cover_p`, efficiency features) consistent across T4–T6; `ensemble_p` optional param consistent T7↔T8.

## Execution notes / risks

- **Heaviest work is data reconstruction** (T4 pbp for 11 seasons; T5 sim backfill). Runtime is the main cost; correctness is straightforward. T5 has a documented lighter fallback.
- **Ship gate is out-of-sample.** If the ensemble doesn't beat its base learners walk-forward (T6 Step 3), stop and revisit features before wiring T8 — don't ship a worse model.

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-09-21-nfl-cover-ensemble.md`. Two execution options:
1. **Inline execution** (this session, executing-plans) — batch with checkpoints, review the T6 walk-forward gate together before wiring serving.
2. **Subagent-driven** — fresh subagent per task with review between.

Which approach?
