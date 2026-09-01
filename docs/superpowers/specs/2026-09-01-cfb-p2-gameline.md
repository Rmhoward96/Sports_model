# CFB P2 — Game-Line Distributions + Market Gate (Design Spec)

**Date:** 2026-09-01
**Status:** approved (game-lines CFB v1); depends on a one-time CFBD historical-lines pull.
**Goal:** Turn P1's CFB ratings into bettable game-line distributions (spread + total) with market shrinkage, and run the honest gate: **does the model beat the CFB closing line out-of-sample?** (This is the whole thesis — CFB lines are softer than NFL/MLB.)

## Reuse (read-only, like P1)

The NFL P2 modules are league-agnostic: `nfl.points.compute_points_ratings`/`expected_total`, `nfl.shrink.shrink`/`w_curve`/`ShrinkParams`, `nfl.gameline.build_gameline`/`GameLineConfig`. CFB **reuses them** with CFB data + retuned config. P2 adds only: a CFB **lines** dataset, a CFB points/total model fit, and the market backtest.

## New data: CFB historical lines (CFBD)

ESPN carries no historical odds (verified 0/49). Source = **collegefootballdata.com (CFBD)**, free API.
- **`scripts/build_cfb_lines.py`** — reads `CFBD_API_KEY` from the environment (never hardcoded); `GET https://api.collegefootballdata.com/lines?year=<Y>&seasonType=regular` (Bearer auth) for 2015–2024; per game take a consensus (median) `spread` + `overUnder`. **Match each CFBD game to our ESPN-keyed schedule** by `(season, week, normalized home, normalized away)` — CFBD uses "school" names (e.g. "Ohio State") vs our ESPN ids, so add `cfb/teams.py: cfbd_to_espn(name) -> team_id|None` built by normalizing CFBD names against `fbs_teams.json` displayNames (strip mascot, lowercase, alias table for the ~10–15 that differ). Write `assets/cfb/lines.parquet` (`season, week, home_team, away_team, market_spread, market_total`), spread in **home-margin convention** (home favored ⇒ positive, matching `nfl` gameline).
- **Key handling:** `CFBD_API_KEY` is a GitHub secret / local env var the user supplies; the assistant never sees it. The pull runs once (locally by the user, or via a manual `build-cfb-lines.yml`) and the committed parquet is the deliverable. **Live P3 uses the existing Odds API (`americanfootball_ncaaf`), not CFBD** — CFBD is only this one-time historical pull.

## Model + gate

- **`scripts/backtest_cfb_gameline.py`** — leak-free walk-forward reusing `nfl.points` (opponent-adjusted PF/PA → `expected_total`) and P1 ratings (`nfl.ratings.expected_margin` from `assets/cfb/rating.json`). Fit `sigma_margin`, `sigma_total`, and the `ShrinkParams` `w(week)` curve against `assets/cfb/lines.parquet`; discretize to `margin_dist`/`total_dist` via `build_gameline`. Write `assets/cfb/gameline.json` (same shape as `assets/nfl/gameline.json`: `sigma_margin, sigma_total, offset, total_max, w_margin{start,floor,decay}, w_total{...}`).
- **THE GATE:** OOS (held-out seasons) **margin MAE and total MAE** for model-only vs blend vs market-only, plus **cover/over accuracy vs the closing line** (the real edge signal — does the model pick the right side of the CFB number > 52.4%?). Ship P3 only if the blend beats market-only on MAE **or** cover accuracy clears breakeven OOS. Unlike NFL (where lines are sharp and the model just matched the close), CFB is where we expect the model to actually beat the number — this gate says whether that's real.

## Global constraints

- CFB-only; reuse `nfl.points/shrink/gameline` **read-only**; additive. No NFL/MLB changes.
- `CFBD_API_KEY` never hardcoded or seen by the assistant; CFBD used only for the one-time historical pull.
- Spread stored in home-margin convention (positive = home favored), matching the NFL gameline path.
- Committed assets: `assets/cfb/lines.parquet`, `assets/cfb/gameline.json`.
- Honest gate: P3 proceeds only if the model beats the CFB close OOS (MAE or cover-accuracy). A negative result is a real answer.

## Out of scope (P2)

- Live producer / odds capture / grading / board / frontend / workflow (P3).
- Props (out for all of CFB v1).
- Odds-API↔ESPN live name matching (P3) — P2's matching is CFBD↔ESPN on historical data only.
