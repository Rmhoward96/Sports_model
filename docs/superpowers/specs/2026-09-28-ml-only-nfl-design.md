# ML-only NFL: retire the Desk and the legacy NFL game model (design spec)

Date: 2026-09-28 · Status: approved design, pending spec review

## Goal

Make the props-ML model (`nfl-sim-ml-v1`, live since 2026-09-28) the only model
producing NFL predictions — moneyline/winner, spread, total, projected score and
player props — and retire the Decision Desk everywhere. Trends stay. College
football keeps its own model (it has no sim), minus the desk.

## Decisions (user, 2026-09-28)

1. **CFB:** keep the CFB game model (`generate-cfb`, `cfb-ratings-v1`); only NFL switches.
2. **Game lines:** gate first — the ML sim replaces the NFL Elo game model only if a
   walk-forward comparison says it's at least as good; if not, the numbers go to the
   user, who decides.
3. **Desk:** retire, keep history — stop every desk workflow and remove it from the
   site; `desk_picks` / `desk_pick_results` stay in the database untouched.
4. **One prediction per NFL game page:** the Elo-vs-sim comparison and disagreement
   badge go away; the page shows the ML prediction (with the actual result once
   graded). Games played before the switch keep the prediction that was served then.

## What exists today

| Model / job | Produces | Consumers |
|---|---|---|
| NFL Elo game model (`generate-nfl`, `nfl-elo-v1`) | `game_predictions` NFL rows → `predictions_current` | site tiles/predictions, accuracy grading, trends pages |
| CFB game model (`generate-cfb`) | CFB `game_predictions` | CFB pages, CFB +EV |
| Decision Desk (`desk-auto-nfl/cfb`, `desk-inputs`, `write-desk-picks`, `grade-desk-picks`, injury-watch rerun steps) | `desk_picks` | site Desk section + record; **+EV game-line board overlay** (`build_ev_board.load_games` joins `desk_current`; `serving/ev_pilot` adds a clamped desk delta to Pinnacle's no-vig prob) |
| Cover ensemble (`train-cover-ensemble`) | nothing (gate never passed) | — |
| ML props sim (`generate-sim-nfl`, `nfl-sim-ml-v1`) | `nfl_sim` / `nfl_player_sim` (served via `nfl_sim_serving`) | game-page props, +EV props, prop parlays |

The +EV game-line board is **market-anchored**: its true probability is Pinnacle's
no-vig price (+ the desk delta). No game model feeds it.

## Design

### 1. Retire the Desk
- Disable workflows `desk-auto-nfl`, `desk-auto-cfb`, `desk-inputs`, `write-desk-picks`,
  `grade-desk-picks` (`gh workflow disable`; files kept so history/re-enable is possible).
- `injury-watch.yml`: remove the desk steps (desk_inputs, synthesize, write) — the
  rerun chain becomes model → sim → +EV board → parlays → record.
- `build_ev_board.py`: drop the `desk_current` join; `ev_pilot` true prob = Pinnacle
  no-vig (desk delta always 0). The board becomes pure line shopping vs Pinnacle.
- Site: remove the Desk section, desk record, and any desk badges/links.
- Database: nothing dropped.

### 2. Retire the other non-ML models
- Disable `train-cover-ensemble`.
- After the switch (§4), disable `generate-nfl` and remove its step from `injury-watch`.
- Keep: `generate-cfb`, trends (`build-trends`, trend blocks), odds capture, actuals, all grading.

### 3. Game-level gate (walk-forward 2021–2025, same games)
Compare the ML sim (kept Props-1 A config + production sim settings) against the NFL
Elo game model's walk-forward predictions (`scripts/backtest_nfl_gameline.py` path):

| Metric | ML sim must be |
|---|---|
| Win-prob Brier | ≤ Elo (point estimate) |
| Cover Brier vs closing spread | ≤ Elo |
| Over Brier vs closing total | ≤ Elo |
| Margin MAE, total MAE | ≤ 1.01 × Elo |

Paired clustered-bootstrap CIs reported for each difference. Pass → §4. Fail → report
to the user with the numbers; no switch without their decision.

### 4. The switch (only after the gate passes)
- `generate_sim_nfl.py` also writes the NFL `game_predictions` rows (same columns
  `generate_nfl.py` writes: winner, home win prob, projected scores, margin/total
  distributions, model spread/total leans) from the ML sim, `model_version =
  "nfl-sim-ml-v1"`. `predictions_current`, grading and the site read them unchanged.
- The ML version's `nfl_sim` game rows come from the ML sim (reverses Props-2 ruling
  I3, which kept them on the current sim because they were ungated — §3 gates them).
- Disable `generate-nfl`; remove it from `injury-watch`.
- Rollback: re-enable `generate-nfl` and point `nfl_sim_serving` back to `sim-nfl-v1`.

### 5. Site (CappingAlpha, user redeploys)
- Game page reads the served version from `nfl_sim_serving` and filters `nfl_player_sim`
  / `nfl_sim` to it (today it shows the newest rows, so a rollback wouldn't reach it).
- "Model: ML v1" label on props and NFL game predictions.
- Remove the Elo-vs-sim comparison table + disagreement badge; the prediction block
  shows predicted vs actual for finished games.
- Remove all desk UI.

## Out of scope / explicitly not changing
- The +EV game-line base stays market-anchored (Pinnacle no-vig). Using the ML sim's
  game probabilities as the +EV base would need its own EV backtest (every prior
  model-based game EV attempt failed its gate).
- CFB modelling.
- Deleting desk data or code.

## Risks
- Gate fails → NFL game predictions stay on Elo until the user decides.
- Game-page history: pre-switch games keep Elo predictions (by design); the accuracy
  record mixes models across the switch date — the site notes the switch date.
- `injury-watch` edits touch a production workflow; tests assert its step list.
