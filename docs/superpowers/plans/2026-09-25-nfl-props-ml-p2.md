# NFL Props ML — Props-2 (B models, blend, calibration, shadow serving) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-market distribution models (B), the A/B blend and PIT calibration, gate them walk-forward on all seven prop markets, then serve the result as `nfl-sim-ml-v1` — shadow first, switched live by one SQL row — with weekly retraining that publishes model files to a GitHub Release.

**Architecture:** Pure modules under `src/sportsmodel/model/props_ml/` (B models, blend, calibration, quantile mapping) are evaluated offline on stored backtest pmfs by a new `scripts/train_props_ml_b.py` (the Props-1 harness stays as is). `scripts/fit_props_ml_final.py` fits the gated pipeline on all data and writes model files + a config JSON; `src/sportsmodel/sim/nfl/ml_serving.py` applies them in `generate_sim_nfl.py` under `SIM_ML_MODE`. A one-row `nfl_sim_serving` table decides which model version the site/board views read.

**Tech Stack:** Python 3.12 (uv), pandas 3, numpy, scikit-learn 1.9 (HistGradientBoosting incl. quantile loss, IsotonicRegression, LogisticRegression), joblib, pytest, GitHub Actions + `gh release`.

**Spec:** `docs/superpowers/specs/2026-09-24-nfl-props-ml-design.md` (§3 B/blend/calibration/quantile mapping/output, §4 gate + rollout, §5 automation). Props-1 verdict: `docs/superpowers/reports/2026-09-24-props-ml-a-gate.md` (kept A toggles: volume, efficiency, context, market).

## Global Constraints

- Leakage: every model/weight/map used to predict season S week w is fit only on data strictly before (S, w); blend weights and calibration maps for test season S use only OOF results from test seasons < S (2021 → w = 1.0 i.e. pure A, identity calibration).
- Walk-forward only; no random splits; tests never touch the network; no new dependencies (scipy is NOT to be imported — negative binomial via `math.lgamma`).
- Markets (7): `pass_yds, rush_yds, rec_yds, receptions, rush_att, pass_tds, anytime_td`. pmf support = `backtest_sim_nfl.MARKET_MAX` (+ anytime_td = 2 bins [P(no TD), P(TD)]).
- Gate rules unchanged from Props-1: rung passes iff pooled relative-RPS skill 95% CI lower bound > 0 AND no market RPS worse than 1.01× AND no market ECE above baseline + 0.005; final gate kept vs baseline on all seasons AND 2025 alone. A market whose final pipeline does not beat the baseline on its own (RPS_c ≤ RPS_b) is served from the current sim (`source = "baseline"` in the config) — never silently.
- Nothing in serving changes behavior while `SIM_ML_MODE` is unset/`off` (default). The served version switches only via the `nfl_sim_serving` row.
- No DDL applied by agents: migrations are written as files; the user runs them. No secrets printed.
- Installed pandas is 3.x: missing strings are float NaN — use `pd.isna`, never `x or ""`.
- TDD; pristine test output; full `uv run pytest -q` green before each commit; commits end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Rulings made while planning

1. **Offline B ladder.** B, blend and calibration change only per-player marginals, so they're gated offline on stored pmfs from ONE baseline run and ONE kept-A run (no extra sims per rung). Quantile mapping (serving-only) preserves marginals by construction.
2. **New-market populations** (props_ml local, serving gate untouched): rush_att = baseline mean ≥ 5.0; anytime_td = the player-week is in the rec_yds or rush_yds population; pass_tds uses the existing `PROJECTED_USAGE_GATE["pass_tds"]`.
3. **A re-check on seven markets.** The kept A config is re-scored against the baseline on all seven markets; a market it worsens by >1% starts the B ladder with `source = "baseline"` for that market (A never replaces baseline there unless a later rung fixes it and the market's own RPS beats baseline).
4. **`_USAGE_CONCENTRATION` re-tune (spec §3) skipped** — the calibration rung addresses dispersion and a re-tune would invalidate the Props-1 A gate. Revisit only if calibration fails.
5. **Serving builds features with the same full 2016→now build** (parity > speed; ~3–4 min). `generate-sim-nfl.yml` timeout 15 → 35 min.
6. **Live injury matching** keeps names (the live report has no gsis) but compares normalized keys (`injury_report.name_key`) against depth `full_name`/`football_name` keys — fixes suffix/punctuation misses; nicknames remain.
7. **Served-version switch** = one-row table `nfl_sim_serving`; `nfl_sim_current` and `nfl_player_sim_current` filter on it. Shadow = ML rows written but not served. Live = user runs one `UPDATE`.
8. **Monthly job = fixed-config re-check** of the kept pipeline on the last 2 completed seasons + current (the full ladder is ~6 h and exceeds an Actions job); the full ladder stays a manual `workflow_dispatch` input.
9. **Quick gate (weekly retrain):** model trained without the last 2 completed weeks, scored on them vs baseline: pooled skill point estimate ≥ 0, no market RPS worse than 1.05×, coverage + fallback checks pass; else no publish and the run fails red.

## File Structure

| File | Responsibility |
|---|---|
| `scripts/backtest_sim_nfl.py` (modify) | Records for all 7 markets; optional `record_pmf` (float32 pmf per record) |
| `src/sportsmodel/model/props_ml/__init__.py` (create) | package |
| `src/sportsmodel/model/props_ml/dist_models.py` (create) | B: quantile→pmf, NB pmf, per-market fit/predict |
| `src/sportsmodel/model/props_ml/blend.py` (create) | pmf linear pool; logit blend for anytime_td; weight search |
| `src/sportsmodel/model/props_ml/pit_calibration.py` (create) | PIT isotonic map (fit/apply on pmfs); Platt for anytime_td |
| `src/sportsmodel/model/props_ml/quantile_map.py` (create) | remap sim draws onto a target pmf preserving ranks |
| `src/sportsmodel/model/props_eval.py` (modify) | market-specific populations for the 3 new markets |
| `scripts/train_props_ml_b.py` (create) | offline B/blend/calibration ladder + final gate → `b_gate.json` + report |
| `scripts/fit_props_ml_final.py` (create) | fit gated pipeline on all data → `data/props_ml/models/` + `props_ml_config.json`; `--holdout-weeks 2` quick gate |
| `src/sportsmodel/nfl/context.py` (modify) | Open-Meteo kickoff forecast fill for upcoming outdoor games |
| `src/sportsmodel/nfl/player_features.py` (modify) | `st_*` NaN (not 0) for weeks with no injury rows |
| `src/sportsmodel/sim/nfl/ml_serving.py` (create) | load artifacts, upcoming feature rows, A→sim→B→blend→calibrate→map → player dists |
| `scripts/generate_sim_nfl.py` (modify) | `SIM_ML_MODE`; `depth_charts_asof`; normalized-name injury match; ML version write |
| `db/migration_nfl_sim_serving.sql` (create) | `nfl_sim_serving` + version-filtered current views (user runs) |
| `scripts/compare_props_shadow.py` (create) | shadow report: v1 vs ML on finished games (RPS/PIT, Brier of P(over) vs graded lines) |
| `.github/workflows/train-props-ml.yml` (create) | weekly retrain + quick gate + Release publish; monthly re-check |
| `.github/workflows/{generate-sim-nfl,injury-watch,desk-auto-nfl}.yml` (modify) | download Release, pass `SIM_ML_MODE` |

---

### Task 1: Backtest records for seven markets + optional pmfs

**Files:** Modify `scripts/backtest_sim_nfl.py`, `src/sportsmodel/model/props_eval.py`; Tests `tests/sim/nfl/test_backtest_sim_nfl.py`, `tests/model/test_props_eval.py`.

**Interfaces — Produces:** `RECORD_MARKETS = ("pass_yds","rush_yds","rec_yds","receptions","rush_att","pass_tds","anytime_td")`; `_actual_player_stats` also returns `rush_att` (= `carries`), `pass_tds` (= `passing_tds`), `anytime_td` (= 1.0 if `receiving_tds + rushing_tds > 0` else 0.0); `run_backtest(..., record_pmf: bool = False)` — when True each record also carries `"pmf": np.float32 array`. `props_eval.population_from_baseline` gains the rulings' gates for rush_att (mean ≥ 5.0), anytime_td (key in rec_yds or rush_yds population for the same season/week/player), pass_tds (existing gate); `PLAYER_MARKETS` (report printing) unchanged.

- [ ] **Step 1: Failing tests**

```python
def test_actual_stats_include_new_markets():
    w = pd.DataFrame({"player_id": ["a"], "season": [2024], "week": [3], "passing_yards": [0], "rushing_yards": [40],
                      "receiving_yards": [12], "receptions": [2], "attempts": [0], "carries": [9], "targets": [3],
                      "passing_tds": [0], "receiving_tds": [0], "rushing_tds": [1]})
    s = bsn._actual_player_stats(w, 2024, 3)["a"]
    assert s["rush_att"] == 9 and s["pass_tds"] == 0 and s["anytime_td"] == 1.0

def test_population_new_markets():
    recs = [{"season": 2024, "week": 1, "player_id": "r", "market": "rush_yds", "mean": 60.0},
            {"season": 2024, "week": 1, "player_id": "r", "market": "rush_att", "mean": 14.0},
            {"season": 2024, "week": 1, "player_id": "r", "market": "anytime_td", "mean": 0.4},
            {"season": 2024, "week": 1, "player_id": "x", "market": "rush_att", "mean": 2.0},
            {"season": 2024, "week": 1, "player_id": "x", "market": "anytime_td", "mean": 0.05}]
    pop = population_from_baseline(recs)
    assert (2024, 1, "r", "rush_att") in pop and (2024, 1, "r", "anytime_td") in pop
    assert (2024, 1, "x", "rush_att") not in pop and (2024, 1, "x", "anytime_td") not in pop
```

Plus a test that `record_pmf=True` attaches a float32 pmf whose length is `MARKET_MAX[market] + 1` (anytime_td: 2) — use the module's existing pattern for exercising the record path with stubbed sources (see Props-1 tests for `record`); if no such pattern exists, extract the per-player record-building into a pure helper `_player_records(season, week, home, dists, actual_stats, record_pmf)` and test that.

- [ ] **Step 2: RED.** **Step 3: implement.** anytime_td record: `dist["pmf"]` is `[P(0), P(≥1)]`; actual clipped to {0,1}; `mean` = P(≥1). **Step 4: GREEN + full suite.** **Step 5: commit** `feat(props-ml): seven-market backtest records with optional pmfs`.

---

### Task 2: B distribution models

**Files:** Create `src/sportsmodel/model/props_ml/__init__.py`, `dist_models.py`; Test `tests/model/test_dist_models.py`.

**Interfaces — Produces:**
- `TAUS = np.round(np.arange(0.05, 0.951, 0.05), 2)` (19 quantiles).
- `quantiles_to_pmf(qvals: np.ndarray, kmax: int) -> np.ndarray` — `qvals` shape (19,) for `TAUS`; sort (monotone rearrangement), clip to [0, kmax]; CDF knots `(0 − 0.5 → 0)`, `(q_i → tau_i)`, `(kmax + 0.5 → 1)`, linear between knots (knots with equal x keep the larger tau); `pmf[k] = F(k + 0.5) − F(k − 0.5)`; renormalize to sum 1.
- `nb_pmf(mu: float, r: float | None, kmax: int) -> np.ndarray` — negative binomial with mean `mu`, size `r` (`r is None` or `r > 1e6` → Poisson), via `math.lgamma`; tail mass beyond kmax folded into `pmf[kmax]`.
- `bernoulli_pmf(p) -> np.array([1 − p, p])`.
- `@dataclass MarketModel(market, kind: "quantile"|"count"|"binary", cols, models, dispersion: dict | None)`; `fit_market(df, market, cols, *, upto, test_season, decay, max_iter) -> MarketModel` (rows before `upto` with the label present, played rows only; sample weight `decay ** (test_season − season)`; label columns: rec_yds→`y_rec_yds`, rush_yds→`y_rush_yds`, pass_yds→`y_pass_yds`, receptions→`y_receptions`, rush_att→`y_carries`, pass_tds→`y_pass_tds`, anytime_td→`y_anytime_td`); `predict_pmfs(model, rows, kmax) -> list[np.ndarray]`.
- Model settings: quantile markets `HistGradientBoostingRegressor(loss="quantile", quantile=tau, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, categorical_features="from_dtype", random_state=0, early_stopping=False, max_iter=max_iter)` per tau (yards labels clipped at 0 — the sim's support starts at 0); counts: same with `loss="poisson"` for the mean + NB size per usage tier (3 tiers by predicted-mean terciles on the training rows; method of moments `r = mean(mu)² / (var(y) − mean(mu))`, `None` when var ≤ mean); binary: `HistGradientBoostingClassifier` same settings.
- All-NaN training columns dropped per fit and recorded (same rule as `learned._fit_one`).

- [ ] **Step 1: Failing tests**

```python
def test_quantiles_to_pmf_point_mass_and_normalization():
    q = np.zeros(19); q[10:] = [5, 8, 12, 15, 20, 25, 30, 40, 55]
    p = quantiles_to_pmf(q, 200)
    assert abs(p.sum() - 1) < 1e-9 and p[0] >= 0.5 - 1e-9 and p.argmax() == 0

def test_quantiles_to_pmf_unsorted_input_is_rearranged():
    q = np.linspace(10, 100, 19)
    assert np.allclose(quantiles_to_pmf(q[::-1], 200), quantiles_to_pmf(q, 200))

def test_nb_pmf_mean_and_poisson_limit():
    p = nb_pmf(4.0, 3.0, 60); k = np.arange(61)
    assert abs((k * p).sum() - 4.0) < 1e-3 and abs(p.sum() - 1) < 1e-9
    assert np.allclose(nb_pmf(4.0, None, 60), nb_pmf(4.0, 1e9, 60), atol=1e-9)

def test_fit_market_respects_upto(small_tbl): ...   # rows at/after upto never train (spy on fit sizes)
def test_predict_pmfs_shapes(small_tbl): ...          # each pmf len kmax+1 (binary: 2), sums to 1
```

(`small_tbl`: synthetic player table fixture with `p_` features and all seven labels; 3 seasons.)

- [ ] **Steps 2–5:** RED → implement → GREEN + full suite → commit `feat(props-ml): B per-market distribution models`.

---

### Task 3: Blend and PIT calibration

**Files:** Create `src/sportsmodel/model/props_ml/blend.py`, `pit_calibration.py`; Test `tests/model/test_blend_calibration.py`.

**Interfaces — Produces:**
- `blend_pmf(pmf_a, pmf_b, w) -> np.ndarray` = `w·a + (1 − w)·b` (renormalized); `blend_binary(p_a, p_b, w)` = `sigmoid(w·logit(p_a) + (1 − w)·logit(p_b))` with p clipped to [1e-4, 1 − 1e-4].
- `choose_weight(rows, market, grid=np.round(np.arange(0, 1.01, 0.1), 1)) -> float` — rows carry `pmf_a, pmf_b, actual`; minimize mean RPS (binary: Brier); ties → larger w (prefer the gated A).
- `fit_pit_map(pits) -> np.ndarray` — knots `(u, G(u))` of the empirical CDF of finite PITs clipped to [0,1], anchored (0,0),(1,1), made monotone, thinned to ≤ 201 evenly spaced u knots; `apply_pit_map(pmf, knots) -> np.ndarray` — `F = cumsum(pmf)`, `F' = interp(F, knots)`, `pmf' = diff([0, F'])`, renormalized; identity knots leave pmf unchanged.
- `fit_platt(p, y) -> (a, b)`; `apply_platt(p, ab)` = `sigmoid(a·logit(p) + b)`; identity = (1, 0).

- [ ] **Step 1: Failing tests** — blend endpoints (w=1 → a, w=0 → b), binary blend monotone in w; `choose_weight` picks 0.0 when B is strictly better and 1.0 on ties; PIT map from uniform PITs ≈ identity (max |G(u) − u| < 0.02 on 20k draws); PIT map from overconfident forecasts (PITs piled near 0 and 1) widens a pmf (its variance increases after `apply_pit_map`); `apply_pit_map` preserves sum 1 and non-negativity; Platt on already-calibrated synthetic data ≈ (1, 0) within 0.1.
- [ ] **Steps 2–5:** RED → implement → GREEN + full suite → commit `feat(props-ml): pmf blend + PIT isotonic and Platt calibration`.

---

### Task 4: Quantile mapping onto sim draws

**Files:** Create `src/sportsmodel/model/props_ml/quantile_map.py`; Test `tests/model/test_quantile_map.py`.

**Interfaces — Produces:** `map_draws(draws: np.ndarray, target_pmf: np.ndarray, rng: np.random.Generator) -> np.ndarray` — ranks with random tie-breaking (`np.lexsort((rng.random(n), draws))`), `u = (rank + 0.5) / n`, value = smallest k with `cumsum(target)[k] ≥ u`; returns ints of the same length. `map_game_sims(sims, targets: dict[player_id, dict[market, pmf]], rng) -> sims` — replaces `sims.player_stats[pid][market]` for each target (anytime_td maps the 0/1 TD indicator array the aggregator derives; if the sims store TD counts, map the indicator and set counts ≥1 where the indicator is 1 — keep whatever representation `aggregate.nfl_player_prop_dists` reads for anytime_td, checked in the code).

- [ ] **Step 1: Failing tests** — mapped draws' empirical pmf matches the target (total variation < 0.02 for n=20000); Spearman rank correlation between two correlated draw vectors is preserved within 0.02 after mapping both; ties broken deterministically for a fixed rng seed; binary target p=0.3 → mean 0.3 ± 0.01.
- [ ] **Steps 2–5:** RED → implement → GREEN + full suite → commit `feat(props-ml): rank-preserving quantile mapping of sim draws`.

---

### Task 5: Offline B / blend / calibration ladder

**Files:** Create `scripts/train_props_ml_b.py`; Test `tests/scripts/test_train_props_ml_b.py`.

**Interfaces — Consumes:** Task 1 records with pmfs; Tasks 2–3; Props-1 `learned`, `props_eval`, `train_props_ml` helpers (import: `checked_paired_frame`, `baseline_ece_check`, `rung_passes`, `refit_block`, `file_fingerprint`, `sources_fingerprint`, `git_head`, `to_jsonable`, `save_records`, `load_records`, checkpoint/tag helpers — reuse, do not copy). **Produces:** `assets/nfl/props_ml/b_gate.json`, `docs/superpowers/reports/<date>-props-ml-b-gate.md`, and `assets/nfl/props_ml/pipeline.json` = `{kept_a_toggles, tuned: {season: [decay, max_iter]}, markets: {m: {"source": "ml"|"baseline", "w_final": float, "calibrate": bool}}}` consumed by Task 6.

Flow (`run_b_ladder(env, *, bsn, player_tbl, team_tbl, ...)`, pure-ish with injected backtest like Props-1's `run_harness`):
1. Tuned values per season from `assets/nfl/props_ml/a_gate.json`; kept A toggles from it.
2. Fetch backtest sources once; baseline run and kept-A run with `record_pmf=True` (checkpointed, tagged `…__b7`), coverage + fallback checks as Props-1.
3. Population = `population_from_baseline(baseline)` (7 markets).
4. **A re-check (7 markets):** `checked_paired_frame(baseline, A)`; per market, `source = "baseline"` if `rps_A > 1.01·rps_base` (ruling 3) — recorded.
5. **B OOF:** for each test season S and refit block r: `fit_market` for all 7 markets on rows before (S, r) with the kept A feature columns (`learned.feature_columns(player_tbl, kept)`) and season S's tuned decay/max_iter; predict pmfs for that block's population records.
6. **+B blend rung:** per market and season S: `w_S = choose_weight(records of test seasons < S)` (2021 → 1.0); candidate pmf = blend; recompute `rps`/`pit` per record (`rps_pmf`, `pit_pmf` with the same `pit_uniform`); decision = `rung_decision(kept vs cand)` + `baseline_ece_check`; kept on pass.
7. **+calibration rung:** per market and S: PIT knots (Platt for anytime_td) fit on the kept pipeline's OOF PITs from seasons < S (2021 → identity); apply; decide as above.
8. **Final gate:** kept pipeline vs baseline on all seasons and 2025 alone (existing final-gate logic). Per-market `source`: `"ml"` iff that market's final RPS ≤ baseline RPS on all seasons (else `"baseline"`).
9. `w_final` per market = `choose_weight` on all five seasons; `calibrate` = calibration rung kept.
10. Report: same style as Props-1 (verdict first, per-rung tables with 7 markets incl. Brier for anytime_td in the RPS column, CAVEATS, identities).

- [ ] **Step 1: Failing tests** (stubbed backtest + tiny tables, like `tests/scripts/test_train_props_ml.py`): 2021 uses w=1.0 and identity calibration; weights for S never see S's records (spy); a market worse under A is marked `source="baseline"`; `pipeline.json` shape; a failing blend rung leaves the kept pipeline at A.
- [ ] **Steps 2–4:** RED → implement → GREEN + full suite. **Step 5:** smoke `PROPS_ML_SEASONS=2025 PROPS_ML_N_SIMS=200 uv run python scripts/train_props_ml_b.py` (network; run in background) must finish and write outputs — do not commit smoke outputs. **Step 6: commit** `feat(props-ml): offline B/blend/calibration ladder and pipeline config`.

---

### Task 6: Final fit + model artifacts + quick gate

**Files:** Create `scripts/fit_props_ml_final.py`; Test `tests/scripts/test_fit_props_ml_final.py`.

**Interfaces — Produces:** `data/props_ml/models/` containing `learned.joblib` (LearnedModels), `b_<market>.joblib` (MarketModel), `calibration.json` (per-market knots / Platt from all OOF records), `props_ml_config.json` = `pipeline.json` + `{trained_through: [season, week], git, features: {player/team fingerprints}, created_at}`. CLI: `--holdout-weeks N` (quick gate; default 0 = final fit). Quick gate (ruling 9) runs the baseline and the pipeline on the held-out weeks via `run_backtest(spec_hook=…, record_pmf=True)` + offline B/blend/calibration, then `rung_decision`-style metrics; exit 1 with the reasons on failure; writes `data/props_ml/models/quick_gate.json` either way.

- [ ] **Step 1: Failing tests** — `save_artifacts`/`load_artifacts` round-trip (predictions identical after reload); `trained_through` = last completed (season, week) in the table; quick-gate decision function: passes on skill ≥ 0 & no market > 1.05×, fails otherwise with market-named reasons.
- [ ] **Steps 2–4:** RED → implement → GREEN + full suite. **Step 5: commit** `feat(props-ml): final fit, model artifacts and weekly quick gate`.

---

### Task 7: Served-version switch (migration)

**Files:** Create `db/migration_nfl_sim_serving.sql`; Test `tests/test_migration_nfl_sim_serving.py` (text assertions).

```sql
-- =============================================================================
-- nfl_sim_serving: which NFL sim model_version the site/board read.
-- =============================================================================
-- Idempotent -- safe to re-run. Run in the Supabase SQL Editor.
-- Shadow mode writes a second model_version (nfl-sim-ml-v1) into nfl_sim /
-- nfl_player_sim; the *_current views serve ONLY the version named here.
-- Go live:   UPDATE nfl_sim_serving SET model_version = 'nfl-sim-ml-v1';
-- Roll back: UPDATE nfl_sim_serving SET model_version = 'sim-nfl-v1';
CREATE TABLE IF NOT EXISTS nfl_sim_serving (
    id            INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    model_version TEXT NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO nfl_sim_serving (id, model_version) VALUES (1, 'sim-nfl-v1')
ON CONFLICT (id) DO NOTHING;
ALTER TABLE nfl_sim_serving ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS nfl_sim_serving_read ON nfl_sim_serving;
CREATE POLICY nfl_sim_serving_read ON nfl_sim_serving FOR SELECT USING (true);
GRANT SELECT ON nfl_sim_serving TO anon, authenticated;

CREATE OR REPLACE VIEW nfl_sim_current AS <existing definition from db/migration_nfl_sim.sql,
  with `AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)` added to its WHERE>;
CREATE OR REPLACE VIEW nfl_player_sim_current AS <existing definition, same filter added>;
GRANT SELECT ON nfl_sim_current, nfl_player_sim_current TO anon, authenticated;
```

Copy the two existing view definitions verbatim from `db/migration_nfl_sim.sql` (column lists must be identical — `CREATE OR REPLACE VIEW` cannot drop/reorder columns) and add only the filter.

- [ ] **Step 1: Failing test** — the file contains the table, the seed row, both views with the `nfl_sim_serving` filter, and each view's SELECT column list equals the one in `migration_nfl_sim.sql` (parse between `SELECT` and `FROM`). **Steps 2–4.** **Step 5: commit** `feat(db): served-version switch for NFL sim views (migration, user-run)`.

---

### Task 8: Live-input fixes (forecast weather, injury-NaN, name keys)

**Files:** Modify `src/sportsmodel/nfl/context.py`, `src/sportsmodel/nfl/player_features.py`, `src/sportsmodel/sim/nfl/usage.py`; Tests in the matching test files.

**Interfaces — Produces:**
- `context.parse_kickoff_forecast(payload: dict, kickoff_utc: pd.Timestamp) -> tuple[float, float]` (temp °F, wind mph at the nearest hour; NaN when missing) and `context.fill_forecast_weather(ctx, stadiums, games, fetch)` — for rows with NaN `cx_temp`/`cx_wind`, `cx_indoor != 1`, and kickoff within the next 7 days, call `fetch(lat, lon)` (Open-Meteo hourly `temperature_2m,wind_speed_10m`, `temperature_unit=fahrenheit`, `wind_speed_unit=mph`, `timezone=UTC`) and fill; network only through `fetch` (tests pass a fake).
- `player_features`: `st_questionable`, `st_vacated_tgt`, `st_vacated_car` are NaN for (season, week) pairs with no injury rows at all (0 only when the week has a report).
- `usage.active_usage(..., match_name_keys: bool = False)` — when True, injury/questionable names and depth names compare via `sportsmodel.nfl.injury_report.name_key` (default False keeps today's behavior).

- [ ] **Step 1: Failing tests** — forecast parse picks the nearest hour and converts nothing (units requested in mph/°F); `fill_forecast_weather` leaves indoor rows and far-future rows untouched and never calls `fetch` for them; `st_*` NaN for a week without injury rows; `active_usage(match_name_keys=True)` drops "D.J. Moore Jr." when the report says "DJ Moore". **Steps 2–5:** RED → implement → GREEN + full suite → commit `fix(props-ml): live inputs — kickoff forecast, injury NaN, normalized name keys`.

---

### Task 9: ML serving module

**Files:** Create `src/sportsmodel/sim/nfl/ml_serving.py`; Test `tests/sim/nfl/test_ml_serving.py`.

**Interfaces — Produces:**
- `load_artifacts(model_dir: Path) -> Artifacts | None` (None + printed reason when missing/unloadable/version-incompatible).
- `ml_player_dists(spec, sims_ml, feature_rows, team_rows, artifacts, rng) -> dict[player_id, dict[market, dist]]` — for each active player & market: A pmf from `sims_ml` (via `aggregate.nfl_player_prop_dists`), B pmf from `predict_pmfs`, blend with `w_final`, calibrate if `calibrate`, markets with `source == "baseline"` omitted (caller falls back to the current sim for those); then `quantile_map.map_game_sims` so `sims_ml` carries the final marginals; returns the dists in the existing `{"kind","pmf","mean"}` format.
- `build_ml_spec(spec, artifacts, feature_rows, team_rows)` = `learned.apply_to_spec(spec, artifacts.learned, feature_rows, team_rows, questionable=…, q_weight=1.0)`.

- [ ] **Step 1: Failing tests** with tiny fake artifacts built in the test (fit on a synthetic table with small max_iter): output markets exclude `source="baseline"` markets; each dist sums to 1 and has the market's support; after mapping, the sims' empirical marginal for a player/market matches the returned pmf (TV < 0.03, n=4000); `load_artifacts` returns None for an empty dir and for a config whose `feature columns` don't match the table. **Steps 2–5** → commit `feat(props-ml): ML serving pipeline (A→sim→B→blend→calibrate→map)`.

---

### Task 10: `generate_sim_nfl.py` integration

**Files:** Modify `scripts/generate_sim_nfl.py`; Test `tests/sim/nfl/test_generate_sim_nfl.py`.

**Behavior:**
- `SIM_ML_MODE` env: `off` (default) | `shadow` | `live`. `off` = byte-for-byte today's behavior except rulings 6/depth below (which apply in all modes and are covered by tests).
- Depth: `depth_charts_asof(load_release("depth", seasons), schedules)` replaces `normalize_depth_charts` (same code path as the backtest). Injury matching: `active_usage(..., match_name_keys=True)`.
- When mode ≠ off: `artifacts = load_artifacts(data/props_ml/models)`; build the feature tables with `scripts/build_player_features.py`'s functions for 2016→upto_season plus stubs for the target week (`active_stubs`) and `fill_forecast_weather`; for each game, `build_ml_spec` → `simulate_game(..., game_rng)` (per-game seed as the backtest) → `ml_player_dists`; write `nfl_sim` + `nfl_player_sim` rows under `model_version="nfl-sim-ml-v1"` (players: ML markets + current-sim dists for `source="baseline"` markets, so the ML version is a complete slate).
- Failure semantics: artifacts missing/incompatible or any ML exception → print `ML: FAILED <reason>`; in `shadow` exit 0 after writing the current sim; in `live` write the current sim, then exit 1 (red run, loud).
- Print one summary line: `ml_mode=… ml_games=… ml_players=… ml_status=ok|failed`.

- [ ] **Step 1: Failing tests** (monkeypatch IO): mode off never imports/loads artifacts; shadow writes both versions; live + missing artifacts → both current-sim upserts happen and `SystemExit(1)`; baseline-sourced markets in the ML slate come from the current sim. **Steps 2–5** → commit `feat(props-ml): SIM_ML_MODE shadow/live serving in generate_sim_nfl`.

---

### Task 11: Workflows

**Files:** Create `.github/workflows/train-props-ml.yml`; Modify `.github/workflows/generate-sim-nfl.yml`, `injury-watch.yml`, `desk-auto-nfl.yml`; Test `tests/test_workflows_props_ml.py` (YAML parse + key assertions).

- `train-props-ml.yml`: `schedule: cron "0 14 * * 2"` + `workflow_dispatch` (inputs: `full_ladder: boolean`); `permissions: contents: write`; `timeout-minutes: 120`; concurrency `props-ml-train`; steps: checkout → setup-uv → `uv sync` → `uv run python scripts/build_player_features.py` → `uv run python scripts/fit_props_ml_final.py --holdout-weeks 2` (quick gate; failure stops the job) → `uv run python scripts/fit_props_ml_final.py` → publish: `gh release view props-ml-latest` exists? → `gh release delete props-ml-prev -y --cleanup-tag || true`, re-create `props-ml-prev` from the downloaded current assets, then `gh release upload props-ml-latest data/props_ml/models/* --clobber` (create `props-ml-latest` with `--latest=false` if missing); `GH_TOKEN: ${{ github.token }}`. Monthly re-check step runs only on the first Tuesday (`if: ${{ github.event_name == 'schedule' && ... }}` computed in a shell step via `date +%d` ≤ 7) with `PROPS_ML_TOGGLES` = kept toggles and seasons = last 2 completed + current, uploading the report with `actions/upload-artifact`.
- `generate-sim-nfl.yml` / `injury-watch.yml` (nfl job) / `desk-auto-nfl.yml`: env `SIM_ML_MODE: ${{ vars.SIM_ML_MODE || 'off' }}`; before the sim step, `if: env.SIM_ML_MODE != 'off'` → `gh release download props-ml-latest -D data/props_ml/models --clobber || echo "no props-ml release"` with `GH_TOKEN: ${{ github.token }}`; `generate-sim-nfl.yml` `timeout-minutes: 35`.

- [ ] **Step 1: Failing test** parsing each YAML and asserting the cron, permissions, the quick-gate step preceding publish, `SIM_ML_MODE` env wiring and the download step condition. **Steps 2–5** → commit `ci(props-ml): weekly retrain + Release publish; SIM_ML_MODE wiring`.

---

### Task 12: Shadow comparison report

**Files:** Create `scripts/compare_props_shadow.py`; Test `tests/scripts/test_compare_props_shadow.py`.

**Behavior:** For finished games since a start date (`--since`, default: the first `nfl-sim-ml-v1` row's `created_at`): load both versions' `nfl_player_sim` dists, `nfl_player_actuals`, and graded lines from `nfl_prop_grades`; per market and version: n, RPS, decile ECE, and Brier of P(over line) vs the line outcome (pushes excluded); print a table and write `docs/superpowers/reports/<date>-props-ml-shadow.md` with a plain verdict line ("ML better / worse / inconclusive (n < 300 per market)"). Pure scoring function unit-tested; DB reads isolated in one function (not unit-tested).

- [ ] **Steps 1–5:** failing tests for the pure scorer (known pmfs/actuals/lines → expected metrics, inconclusive rule) → implement → GREEN → commit `feat(props-ml): shadow comparison report (v1 vs ML)`.

---

### Task 13: Run the B gate, fit, publish, hand off (controller)

- [ ] **Step 1:** `uv run python scripts/build_player_features.py`; full B ladder in the background: `PROPS_ML_N_SIMS=1000 PROPS_ML_RESUME=1 uv run python scripts/train_props_ml_b.py` (~2 h). Commit `b_gate.json`, `pipeline.json`, report.
- [ ] **Step 2:** If the final gate passes: `uv run python scripts/fit_props_ml_final.py --holdout-weeks 2` then `uv run python scripts/fit_props_ml_final.py`; first Release publish by hand with `gh release create props-ml-latest data/props_ml/models/* --latest=false --title "props-ml models" --notes "<config summary>"` (ask the user before publishing — it's an outward action).
- [ ] **Step 3:** User actions (ask): run `db/migration_nfl_sim_serving.sql`; set repo variable `SIM_ML_MODE=shadow`. Verify with a `workflow_dispatch` of generate-sim-nfl that both versions are written and the site still serves `sim-nfl-v1`.
- [ ] **Step 4:** After ≥1 week of finished games: `uv run python scripts/compare_props_shadow.py`; bring the report to the user with the go-live SQL (`UPDATE nfl_sim_serving SET model_version = 'nfl-sim-ml-v1';`) — the user decides.

## Self-review notes

- Spec coverage: B models, blend, calibration (isotonic PIT + Platt), quantile mapping, output version (§3) → Tasks 2–4, 9; gate/rollout (§4) → Tasks 1, 5, 6, 12, 13; automation/Release/`SIM_ML_MODE` (§5) → Tasks 6, 10, 11; live-input carry-overs from Props-1 → Tasks 8, 10.
- Deviations are listed under Rulings (offline ladder, new-market populations, A re-check, concentration re-tune skipped, full-build serving, name-key matching, served-version table, monthly re-check scope, quick-gate thresholds).
