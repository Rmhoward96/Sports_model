# Prediction Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Reframe to a confidence-ranked prediction tool judged on accuracy. Add a CFB predictions producer + accuracy grading; reframe the site.

**Spec:** docs/superpowers/specs/2026-09-01-prediction-tool.md

## Global Constraints

- No betting-edge claims anywhere: accuracy only (winner-correct, margin/total error). No ROI/CLV/ATS in the prediction path.
- CFB producer reuses `nfl.ratings/points/gameline` read-only; additive. NFL prediction math unchanged. MLB stays scrapped.
- `game_pk` = ESPN event id; sport tags on every row.
- `PYTHONPATH=src uv run --no-sync` for all runs. Full suite green after each task.
- Prereq: `assets/cfb/gameline.json` (fitted σ) must be on the branch — merge branch `cfb-p2-gate` to main first, then rebase.

---

### Task 1: DB — prediction_accuracy table + views + upsert helper

**Files:** Create `db/migration_prediction_tool.sql`; Modify `src/sportsmodel/db.py`; Test `tests/test_db_prediction_accuracy.py`

- `prediction_accuracy` table: `sport, game_pk, game_date, home_team_name, away_team_name, win_prob, predicted_winner, actual_winner, winner_correct bool, pred_margin, actual_margin, margin_error, pred_total, actual_total, total_error, graded_at` (PK sport+game_pk).
- Views: `accuracy_by_confidence` (sport, conf_tier via width_bucket on win_prob → games, winner-correct %, avg margin_error), `predictions_current` (upcoming game_predictions with predicted winner/scores/win_prob, sport-tagged). GRANT SELECT to anon.
- `db.upsert_prediction_accuracy(rows)` mirroring the existing upsert helpers (column list, ON CONFLICT sport+game_pk).

- [ ] Steps: write migration; write upsert helper + a test that builds the column tuple correctly (mirror `test` for existing upserts, no live DB). Run suite. Commit.

### Task 2: CFB predictions producer

**Files:** Create `scripts/generate_cfb.py`; add `cfb.espn.fetch_current_week`/`parse_current_week` if missing (mirror nfl); Test `tests/cfb/test_generate_cfb.py`

- Resolve current CFB week (ESPN scoreboard, mirror `nfl.espn.fetch_current_week`; CFB `seasontype`/`week`), fetch that week's FBS schedule.
- Load `assets/cfb/rating.json` (EloConfig/BlendConfig) + `assets/cfb/gameline.json` (GameLineConfig σ). Historical ratings from `assets/cfb/schedules.parquet` + season-to-date. `expected_margin` + `expected_total` → `build_gameline(model_margin, model_total, market=None, week, gl_cfg)` (**market=None ⇒ model-only, no shrink**) → margin_dist/total_dist/home_win_prob/pred scores.
- Skip games where either side is `FCS`.
- Write `game_predictions` (sport=`cfb`, model_version `cfb-ratings-v1`) via the existing `upsert_game_predictions`.

- [ ] Steps: write producer (mirror `generate_nfl.py` game-line half; drop props/odds). Unit-test the per-game row build with a stubbed schedule + ratings (no network). Run suite. Commit. (Live run happens via the workflow in Task 5.)

### Task 3: Accuracy grader

**Files:** Create `scripts/grade_predictions.py`; Test `tests/test_grade_predictions.py`

- For each finished game with a `game_predictions` row (sport in {nfl, cfb}) not yet in `prediction_accuracy`: fetch the final via `RESULTS_PROVIDERS[sport]` (ESPN), compute `predicted_winner` (higher win_prob side), `winner_correct`, `margin_error`, `total_error`; upsert to `prediction_accuracy`. **No odds, no closing line.**
- Rolling window like `grade_results` (recent game_dates), idempotent.

- [ ] Steps: write grader; test `winner_correct`/`margin_error` logic on a stubbed prediction + final (no network). Run suite. Commit.

### Task 4: Site reframe (external app.js)

**Files:** Modify `/Users/ryan/Desktop/CappingAlpha/app.js` (+ add `cfb.html`, nav)

- League page (`buildLeague`): **confidence-ranked predictions** — query `predictions_current?sport=eq.<s>`; each game shows projected score, predicted winner, confidence % (win_prob), sorted by confidence desc. Drop the +EV/pick/odds framing. CFB + NFL.
- Track Record (`buildTrack`): **accuracy** — query `accuracy_by_confidence` + recent `prediction_accuracy`; show overall winner %, winner % by confidence tier, avg margin error. Drop ROI/units/CLV.
- Nav: add CFB. Dashboard: reframe to predictions + accuracy.

- [ ] Steps: edit app.js; smoke-test locally in the browser (predictions + accuracy render, no console errors); no automated tests (external). Note: user redeploys.

### Task 5: Workflows

**Files:** Create `.github/workflows/generate-cfb.yml`; Modify `.github/workflows/grade-results.yml` (or new `grade-predictions.yml`)

- `generate-cfb.yml`: weekly CFB prediction generation (cron for the CFB game week + workflow_dispatch), `uv run python scripts/generate_cfb.py`, no exact-hour gate (idempotent).
- Accuracy grading leg: run `grade_predictions.py` for nfl + cfb (replaces the betting grade for the reframed sports).

- [ ] Steps: write workflows (mirror generate-nfl.yml). Commit. (Dispatch after merge + migration.)

## Coordination (human steps)

Merge `cfb-p2-gate` → main first (for `gameline.json`). After the branch merges: user runs `db/migration_prediction_tool.sql` in Supabase, then dispatch `generate-cfb.yml`, then redeploy CappingAlpha.
