# Sub-project C — NFL player-props productization (scope / design draft)

**Status:** scope for review · **Date:** 2026-09-18
**Program:** A→B→B.2→B.3(honest volume)→B.4(projected-usage gate; sim CALIBRATES rush_yds+rec_yds, receptions nearly) → **C (this: turn the calibrated sim into a live prop +EV board with CLV grading)**.

## Why C now

B.4 established (walk-forward, projected/expected-role gate, 2021-24) that the NFL
sim produces **calibrated** player-prop distributions for **rush_yds** (bias −2.9,
p50 .461, p90 .886) and **rec_yds** (+1.8, .487, .912), with **receptions** very
close (+0.1, .574, .933). Calibration is necessary, not sufficient — the real
test is beating the market (CLV), exactly as with the game +EV pilot. C makes the
sim's distributions live and measurable against real prop lines so that forward
CLV can judge edge.

## Goal

For each upcoming NFL game, project player-prop distributions from the sim
(nfl_player_sim), compare to book prop lines at the book's actual line, surface
**+EV props on projected-featured players only** (the projected-usage gate = the
deployment population B.4 validated), and grade them forward on **CLV + result**
— the same discipline as the game +EV board.

## Deployment principle (from B.4 — the key constraint)

Only offer/track a prop where the sim PROJECTS the player featured enough to have
a line (`is_propable_projected`: sim dist mean ≥ market threshold). Grading on
projected (expected) role, not realized usage, is what makes the calibration
hold; deploying the same way avoids the selection bias that made the model look
broken. EV is evaluated at the BOOK's actual line via the sim's full distribution
(P(over) from the pmf), calibrated — mirroring the MLB/NFL prop-EV reframe
already shipped for other markets.

## Phase 0 — survey existing prop infrastructure (do FIRST, read-only)

Much prop machinery already exists from the MLB era + game +EV pilot; C should
REUSE, not rebuild. Survey and document what's sport-agnostic vs MLB-specific:
- Odds ingestion: `scripts/ingest_odds.py` / `ingest/odds.py` / `capture-odds.yml`
  — already map our market codes ↔ Odds API player-prop keys; confirm the NFL
  player-prop keys (player_rush_yds / player_reception_yds / player_receptions),
  event-level fetch, and the `PROP_WINDOW_MIN` post-lineup capture gating.
- Serving: `odds_snapshot` (line + median-price + main-line selection),
  `board`/`board_picks`, `serving/board.py`; the prop-EV math
  (`distributions.prob_over_dist`, no-vig, EV = model_P × decimal − 1).
- Grading + CLV: `scripts/grade_results.py` prop branch, `prediction_results`/
  `picks`, closing-line logic, `fit_calibration_live.py`.
- The sim producer: `scripts/generate_sim_nfl.py` + `nfl_player_sim` table +
  `db/migration_nfl_sim.sql` (currently DORMANT).
Output: a short reuse map → which components take an NFL-props config vs need code.

## Approach (phased; refined after Phase 0)

1. **Activate the sim producer** (prerequisite; user's call): run
   `db/migration_nfl_sim.sql`, enable `generate-sim-nfl.yml`, merge
   `sim-nfl-engine` — so `nfl_player_sim` distributions are produced live.
2. **Prop-odds capture** for NFL player markets (rush_yds/rec_yds/receptions),
   post-lineup window, into `odds_snapshot` — reuse ingest_odds with the NFL
   prop keys; verify the player-name ↔ sim gsis/name join.
3. **Prop +EV board**: join sim distributions (projected-featured only) to the
   captured prop lines; compute P(over) at the book line from the pmf, no-vig
   market P, EV; surface +EV props; write to the board/picks tables with
   market/side/line/EV like the game board.
4. **Grade + CLV**: extend the grader to fetch prop actuals (nflverse weekly /
   ESPN box) and closing prop lines; record result + CLV per prop; feed the
   live-calibration refit loop.
5. **Front-end** (external CappingAlpha app.js, user redeploys): a props view
   listing +EV props + a props track-record (CLV) — mirror the game board.

## Gate (forward, not backtest)

Calibration is already shown (B.4). C's gate is **forward CLV** accumulated over
real weeks — same bar as the game +EV pilot: enough graded props before ROI/CLV
is signal not noise; a market with positive CLV earns continued offering, one
with flat/negative CLV gets cut (the exact discipline applied to MLB props).

## Non-goals / notes

- Not re-opening the model (B.2/B.3 sim is the producer; B.4 fixed the lens).
- pass_yds excluded initially (p90 tail fat + mean bias; low priority, shopped).
- receptions included but flagged (p50 .574 slightly high) — watch its CLV.
- NFL only. Activation (migration/workflow/merge) is the user's explicit call and
  is a prerequisite, not something C does autonomously.

## Risk

Calibrated ≠ profitable. The yardage/reception prop markets are shopped; edge (if
any) is likely in softer lines + line-shopping + post-lineup timing, proven only
by forward CLV. C's value is making that measurable; it does not assume edge.
