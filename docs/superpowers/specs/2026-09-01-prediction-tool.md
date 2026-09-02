# Prediction Tool — Reframe Spec

**Date:** 2026-09-01
**Status:** approved (pivot from betting board to prediction tool)
**Goal:** Reframe the product from a "+EV betting board" to a **game-prediction tool** judged on *accuracy*, not ROI. The model is a strong predictor (CFB 73.8% straight-up, NFL similar) but not a market-beater (proven three ways: favorites priced to win rate, ATS cover 49.7%, biggest disagreements 38.9%). So we present forecasts + a published accuracy track record, and add CFB as a first-class predictor.

## What changes (and what doesn't)

**Doesn't change:** the model already produces predictions — `game_predictions` stores `home_win_prob`, `pred_home_score`, `pred_away_score`, `margin_dist`, `total_dist`. NFL's `generate_nfl` already writes these.

**Changes:**
1. **Add a CFB predictions producer** so CFB predictions exist.
2. **Grade predictions on ACCURACY** (did the predicted winner win? how close was the margin/total?), not bets (no CLV, no ROI, no ATS).
3. **Reframe the site** to confidence-ranked forecasts + an accuracy record.
4. **Drop the betting framing** from the primary view (no +EV picks, no "bet this"). Market lines, if shown, are labeled **analysis** ("model sees this closer than the number"), explicitly not a recommendation — because we proved that's not an edge.

## Components

1. **`scripts/generate_cfb.py`** — CFB predictions producer. ESPN current week (reuse `cfb.espn` current-week resolver, add if missing) → `nfl.ratings.expected_margin` (from `assets/cfb/rating.json`) + `nfl.points.expected_total` → `nfl.gameline.build_gameline` **model-only (no market shrink)** using `assets/cfb/gameline.json`'s fitted `sigma_margin`/`sigma_total` → `margin_dist`, `total_dist`, `home_win_prob`, projected scores. Skip FBS-vs-FCS games (no meaningful prediction). Write to `game_predictions` (sport=`cfb`, model_version `cfb-ratings-v1`). No odds, no props. (Merge the P2 `gameline.json`/backtest from branch `cfb-p2-gate` — the fitted sigmas are the honest prediction distribution even though the market gate failed.)

2. **`scripts/grade_predictions.py`** — accuracy grader (sport-agnostic). For each finished game with a prediction: `winner_correct` (higher-win-prob team == actual winner), `margin_error = |pred_margin − actual_margin|`, `total_error = |pred_total − actual_total|`, plus `win_prob` (for confidence bucketing). Uses the existing `RESULTS_PROVIDERS` (ESPN finals for NFL/CFB). Writes a `prediction_accuracy` table (game_pk, sport, game_date, win_prob, winner_correct, margin_error, total_error, predicted/actual). **No odds, no closing line, no profit.**

3. **DB:** `prediction_accuracy` table + two views: `accuracy_by_confidence` (sport × win-prob tier → games, winner-correct %, avg margin_error) and `predictions_current` (upcoming games with predicted winner/score/confidence). Migration `db/migration_prediction_tool.sql`.

4. **Site (`app.js`, external):** each league page = **confidence-ranked predictions** (projected score, winner, confidence %, sorted by confidence), and the Track Record page = **prediction accuracy** (overall winner %, winner % by confidence tier, avg margin error) instead of ROI/units/CLV. Dashboard reframed the same way. CFB added to nav.

5. **Workflows:** `generate-cfb.yml` (weekly CFB predictions) + a `grade-predictions` leg (accuracy) alongside/replacing the betting grade for the reframed sports.

## Confidence

Confidence = `home_win_prob` distance from 0.5 (a pick-em is low confidence, a 90% game is high). Tiers: 50–60 / 60–70 / 70–80 / 80%+. The headline story is "high-confidence picks are very accurate" — which the data supports (CFB ~90% on big favorites).

## Global constraints

- Reframe is honest: **no betting-edge claims.** Market comparison is analysis-only, explicitly labeled not a bet.
- Accuracy metrics only (winner-correct, margin/total error) — no ROI, CLV, or ATS in the prediction track record.
- CFB producer reuses `nfl.ratings/points/gameline` read-only; additive.
- NFL predictions already exist; reframe reuses them (no NFL model change).
- MLB stays scrapped.

## Scope / phasing

- **v1:** CFB producer + accuracy grader + DB + site reframe (CFB + NFL as predictors).
- **Deferred:** props-as-predictions, richer per-team accuracy splits, model-vs-market analysis view.

## Out of scope

- Any betting-edge / +EV / CLV framing (deliberately removed).
- CFB props. MLB.
