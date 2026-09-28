# ML-only NFL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retire the Decision Desk and the legacy NFL models, gate the ML sim's game-level predictions against the Elo game model, and (on a pass) make the ML sim the only NFL source for winner/moneyline, spread, total, score and props — with one prediction per NFL game page.

**Architecture:** A walk-forward game-level gate (`scripts/gate_ml_game_lines.py`) pairs the ML sim and the Elo model game by game. Backend changes land behind a repo variable `ML_GAME_LINES` (off by default): when on, `generate_sim_nfl.py` also writes NFL `game_predictions` rows and ML-sim game rows. Desk code paths are removed from the +EV board and `injury-watch`; desk workflows are disabled (files kept). CappingAlpha drops desk UI, reads the served version, and shows one labelled prediction.

**Tech Stack:** Python 3.12 (uv), pandas 3, numpy, scikit-learn, pytest; GitHub Actions; vanilla JS site (CappingAlpha, non-git, user redeploys).

**Spec:** `docs/superpowers/specs/2026-09-28-ml-only-nfl-design.md`

## Global Constraints

- Walk-forward only; leakage-free (everything for (S, w) fit strictly before (S, w)); tests never touch the network or DB.
- No DDL applied by agents; no DB writes by agents; no secrets printed; no new dependencies; no scipy.
- Desk history stays: never drop `desk_picks` / `desk_pick_results` or delete desk code files (workflows are disabled, not deleted).
- CFB is unchanged except desk removal.
- `ML_GAME_LINES` unset/off ⇒ NFL game predictions keep coming from `generate-nfl` exactly as today.
- nflverse `spread_line` is POSITIVE when the HOME team is favored (home covers iff actual margin > spread_line; push when equal).
- pandas 3.x (missing strings are float NaN); TDD; pristine output; full `uv run pytest -q` green before each commit; commits end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Rulings made while planning

1. **Flag-gated switch.** `ML_GAME_LINES` (repo variable, default off) lets the code merge before the gate verdict; the switch is operational (variable + `gh workflow disable generate-nfl`).
2. **Gate uses the kept Props-1 A config**, the production sim settings (season_decay 0.4, questionable_weight 0.75, home_field 0.07, ratings_weight 0.5), refit every 4 weeks, 1000 sims/game — the same machinery `scripts/train_props_ml.py` already has (`make_hook`, `fit_models`, `refit_block`, tuned values from `assets/nfl/props_ml/a_gate.json`).
3. **Elo comparator** = `backtest_nfl_gameline.per_game_predictions` with the committed configs (`assets/nfl/rating.json` / `gameline.json` as `generate_nfl.py` loads them). Its cover/over probabilities use the gameline normal sigma (`gameline.GameLineConfig`); the ML sim's use its own margin/total pmfs. The report states that Elo's sigmas were tuned on these seasons (an in-sample edge for Elo — biases the gate toward NOT switching).
4. **Game-level quantity definitions:** home win prob = P(margin > 0) + 0.5·P(margin = 0); cover prob vs closing spread L = P(margin > L) (push mass excluded and renormalized); over prob vs closing total T = P(total > T) (push excluded and renormalized); games without a closing line are excluded from that market's Brier only.
5. **Desk removal from the +EV board** keeps `ev_pilot` pure: callers pass no desk data; the desk-delta parameter defaults to none/0 (no signature churn beyond removal of the desk join in `build_ev_board.load_games`).

## File Structure

| File | Responsibility |
|---|---|
| `src/sportsmodel/model/game_gate.py` (create) | pure game-level metrics from pmfs/normal: win/cover/over probs, Briers, MAEs, paired clustered bootstrap, decision |
| `scripts/gate_ml_game_lines.py` (create) | walk-forward runner: ML-sim game records (on_game hook) + Elo per-game preds → pairing → `assets/nfl/props_ml/game_gate.json` + report |
| `scripts/build_ev_board.py`, `src/sportsmodel/serving/ev_pilot.py` (modify) | drop the desk join/overlay |
| `.github/workflows/injury-watch.yml` (modify) | remove desk steps; `generate_nfl.py` step conditional on `ML_GAME_LINES != 'on'` |
| `scripts/generate_sim_nfl.py` (modify) | `ML_GAME_LINES=on` ⇒ write NFL `game_predictions` from the ML sim + ML-sim game rows in `nfl_sim` |
| `.github/workflows/generate-sim-nfl.yml` (modify) | pass `ML_GAME_LINES: ${{ vars.ML_GAME_LINES || 'off' }}` (also injury-watch nfl job) |
| CappingAlpha `app.js` + `*.html` (modify) | remove desk UI; served-version filter; "Model: ML v1" label; single prediction block |

---

### Task 1: Pure game-level gate metrics

**Files:** Create `src/sportsmodel/model/game_gate.py`; Test `tests/model/test_game_gate.py`.

**Interfaces — Produces:**
- `win_prob_pmf(margin_pmf, offset) -> float`, `cover_prob_pmf(margin_pmf, offset, line) -> float | None`, `over_prob_pmf(total_pmf, line) -> float | None` (pmf over integer support; `offset` = margin value of index 0; None when the line is None/NaN or all mass is a push).
- `cover_prob_normal(mean, sigma, line)`, `over_prob_normal(mean, sigma, line)` (continuous: P(X > line); None for missing line).
- `game_metrics(records) -> dict` — records carry `season, week, home, win_prob, cover_prob, over_prob, pred_margin, pred_total, actual_margin, actual_total, spread_line, total_line`; returns per-metric values: win Brier (ties 0.5), cover Brier (push rows excluded), over Brier (push excluded), margin MAE, total MAE, n per metric.
- `paired_diffs(ml_records, elo_records) -> pd.DataFrame` — inner join on (season, week, home), per-game squared/absolute errors for both.
- `gate_decision(df, n_boot=1000, seed=0) -> dict` — point estimates + 95% cluster (season-week) bootstrap CIs of ML − Elo for each metric; pass iff win/cover/over Brier ML ≤ Elo (point) and margin/total MAE ML ≤ 1.01 × Elo (point); `reasons` name each failing metric.

- [ ] **Step 1: Failing tests** — hand-computed pmf cases (a symmetric margin pmf gives win 0.5; a line on an integer with push mass excluded and renormalized; half-point lines never push), normal cases vs closed-form, tie handling in win Brier, push exclusion in cover/over Brier, pass/fail decisions with named reasons, bootstrap resamples clusters with replacement and is seed-deterministic (reuse the concat-per-cluster pattern of `props_eval.cluster_bootstrap`; do not copy it — generalize or import).
- [ ] **Steps 2–5:** RED → implement → GREEN + full suite → commit `feat(game-gate): pure game-level metrics and decision`.

### Task 2: Walk-forward game-level gate runner

**Files:** Create `scripts/gate_ml_game_lines.py`; Test `tests/scripts/test_gate_ml_game_lines.py`.

**Interfaces — Consumes:** `scripts/train_props_ml.py` (`make_hook`, `guard_hook`, `refit_block`, `REFIT_WEEKS`, `PROD`, `Q_WEIGHT`, fingerprint helpers), `learned.fit_models`, `backtest_sim_nfl.run_backtest(..., on_game=, spec_hook=, sources=)` + `fetch_backtest_sources`, `sim.engine` margin/total pmf helpers (check `sim/engine.py` for `margin_pmf`/`total_pmf` and their offset convention), `backtest_nfl_gameline.per_game_predictions` + committed configs (check `generate_nfl.py` for how it loads them), schedules closing `spread_line`/`total_line`, Task 1. **Produces:** `assets/nfl/props_ml/game_gate.json`, `docs/superpowers/reports/<date>-ml-game-gate.md`.

Flow: fetch sources once; fit A models per (season, refit block) with the kept toggles + per-season tuned values from `a_gate.json` (ruling 2); run `run_backtest` with the hook and an `on_game` callback that records each game's win/cover/over probs (from the sim pmfs, ruling 4), pred margin/total (means) and actuals; compute the Elo per-game predictions on the same schedule; pair; `gate_decision`; write json + report (verdict first; per-metric table ML vs Elo with CIs; the in-sample caveat for Elo; coverage counts; identity: git, feature fingerprints). Checkpoint the ML game records (`data/props_ml/game_records__<tag>.parquet`) with identity so a crash can resume (`PROPS_ML_RESUME=1`). Env: `PROPS_ML_SEASONS`, `PROPS_ML_N_SIMS` (default 1000).

- [ ] **Step 1: Failing tests** with a stubbed backtest + stubbed Elo predictions: records built per game via on_game; pairing; games missing on either side reported (coverage) and excluded; json/report written; resume uses the checkpoint.
- [ ] **Steps 2–4:** RED → implement → GREEN + full suite. **Step 5:** smoke `PROPS_ML_SEASONS=2025 PROPS_ML_N_SIMS=200` (network; background) must complete; don't commit smoke outputs. **Step 6: commit** `feat(game-gate): walk-forward ML-sim vs Elo game-level gate`.

### Task 3: Retire the desk from the +EV board and injury-watch

**Files:** Modify `scripts/build_ev_board.py`, `src/sportsmodel/serving/ev_pilot.py` (only if needed), `.github/workflows/injury-watch.yml`; Tests `tests/test_build_ev_board.py`, `tests/serving/...` (existing), `tests/test_workflows_props_ml.py` or a new `tests/test_workflows_ml_only.py`.

- `load_games` no longer joins `desk_current` (and `GAME_COLS` drops desk fields); the board's true prob = Pinnacle no-vig (desk delta 0 everywhere). Existing tests updated to the no-desk behavior; a test proves the SQL no longer references `desk_current`.
- `injury-watch.yml`: remove `desk_inputs.py`, `synthesize_desk_picks.py`, `write_desk_picks.py` steps (and any `--bundle desk_bundle.json` usage — `injury_watch.py --record` must still work without a bundle; check its CLI and adjust the record step to the non-bundle form if needed). Both nfl and cfb jobs.
- Workflow files for desk jobs are NOT deleted (disabled by the controller after merge).
- Commit `refactor: retire the Decision Desk from the +EV board and injury-watch`.

### Task 4: `ML_GAME_LINES` switch in the sim producer

**Files:** Modify `scripts/generate_sim_nfl.py`, `.github/workflows/generate-sim-nfl.yml`, `.github/workflows/injury-watch.yml`; Tests `tests/sim/nfl/test_generate_sim_nfl.py`, workflow tests.

- `ML_GAME_LINES` env (`on`|`off`, default off; anything else = off + warning). When `on` AND the served version is `nfl-sim-ml-v1` AND the ML path succeeded for a game: (a) that game's `nfl_sim` ML-version row comes from `sims_ml` (reverses Props-2 I3 for the switched state); (b) a `game_predictions` row (same columns `generate_nfl.build_game_row` produces: home_win_prob, pred scores, pred_margin, pred_total, margin_dist, total_dist, market_spread/total if available, names, game_date, commence_time) with `model_version = "nfl-sim-ml-v1"`, built from the ML sim's margin/total pmfs (reuse the gameline dist format — check `gameline.build_gameline`'s `margin_dist` shape/offset and emit the same shape), upserted via `db.upsert_game_predictions`. Games that fell back write `game_predictions` from the current sim the same way (the slate stays complete).
- `ML_GAME_LINES=off` ⇒ no `game_predictions` writes and nfl_sim ML rows stay on the current sim (today's behavior, byte-identical).
- Workflows: pass `ML_GAME_LINES: ${{ vars.ML_GAME_LINES || 'off' }}` in generate-sim-nfl.yml and injury-watch nfl job; in injury-watch, the `generate_nfl.py` step runs only when `ML_GAME_LINES != 'on'`.
- Tests: off = unchanged writes; on + served ML = game_predictions rows from ML sim with the expected columns/shape; on + served v1 = no game_predictions writes (warning); fallback game rows; workflow conditions.
- Commit `feat: ML_GAME_LINES switch — ML sim writes NFL game predictions`.

### Task 5: Site (CappingAlpha)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js` and `*.html` (non-git; verify with `node --check` and a small node harness in the session scratchpad like earlier site tasks).

- Remove all Decision Desk UI (sections, record, badges, links, nav entries, and `desk_*` fetches).
- Game page: read `nfl_sim_serving` (anon-readable) once; filter `nfl_player_sim` and `nfl_sim` fetches by `model_version=eq.<served>` (fallback: current newest-row behavior if the read fails).
- Replace the Elo-vs-sim comparison table + disagreement badge with a single prediction block: winner, win %, projected score, spread and total leans; plus predicted vs actual once graded. Add a small "Model: ML v1" label to the NFL prediction block and the props section when the served version is `nfl-sim-ml-v1` (and the prediction row's model_version matches).
- CFB pages: only desk removal.
- Bump every `app.js?v=` cache key.
- Report what a harness verified (desk gone, served filter applied, label rendering, `node --check` clean).

### Task 6: Run the gate, switch, hand off (controller)

- [ ] **Step 1:** `uv run python scripts/build_player_features.py`; full gate in the background: `PROPS_ML_N_SIMS=1000 PROPS_ML_RESUME=1 uv run python scripts/gate_ml_game_lines.py`. Commit `game_gate.json` + report.
- [ ] **Step 2:** Merge the branch (ask first). Disable desk workflows + `train-cover-ensemble` (`gh workflow disable …`).
- [ ] **Step 3 (gate PASS only):** set repo variable `ML_GAME_LINES=on`; `gh workflow disable generate-nfl`; dispatch `generate-sim-nfl`; verify `predictions_current` NFL rows are `nfl-sim-ml-v1` for every upcoming game. **Gate FAIL:** stop and bring the numbers to the user.
- [ ] **Step 4:** Tell the user to redeploy CappingAlpha; verify the game page shows one labelled prediction.

## Self-review notes

- Spec coverage: §1 desk → Tasks 3, 5, 6; §2 other models → Task 6 (disable) + Task 4 (generate-nfl conditional); §3 gate → Tasks 1–2, 6; §4 switch → Task 4, 6; §5 site → Task 5.
- Rulings recorded above (flag-gated switch, gate config, Elo comparator + in-sample caveat, probability definitions, pure ev_pilot).
