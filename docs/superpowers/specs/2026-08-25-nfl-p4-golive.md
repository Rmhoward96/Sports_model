# NFL P4 — Go-live wire-up (producer + board + odds/matcher + grading + automation)

**Date:** 2026-08-25
**Status:** Approved design (pre-plan)
**Parent spec:** `docs/superpowers/specs/2026-08-23-nfl-model.md` (P4 phase)
**Depends on:** P0–P3 (all merged to main): `SportConfig` + sport-parameterized producers (P0);
`nfl/{teams,data,elo,srs,ratings,espn,matcher}.py` + `rating.json` (P1); `nfl/{points,shrink,
gameline}.py` + `gameline.json` (P2); `nfl/{gamescript,usage,efficiency,props,universe}.py` +
`props.json` (P3).

## Problem & goal

P1–P3 built and validated the NFL models; the serving/board/grade infra is sport-aware (P0). P4
is the **integration/wire-up** that makes NFL live on the board and track record for **Week 1
(~Sept 10, 2026)**: a producer that writes NFL `game_predictions`/`prop_predictions`, the DB
`sport` column + `generate_board` NFL pass, NFL odds capture + the `game_pk` matcher, an NFL
results provider for grading, the automation workflows, and front-end logos. It also threads the
integration seams parked across P1–P3.

Built **board-first** (single phase): schema → producer → board → odds/matcher → grading →
automation → front-end → validation.

## Scope / non-goals

**In (P4):**
- `sport` column migration on `game_predictions`/`prop_predictions`.
- `scripts/generate_nfl.py` — the NFL producer (Elo/SoS → gameline → dists; universe → props).
- `generate_board.py` NFL-boardable (BOARDABLE_SPORTS + PROP_MARKETS + sport-filtered reads).
- NFL odds capture (`ingest_odds --sport nfl`) + the Odds-API→ESPN `game_pk` matcher wiring.
- NFL results provider (`nfl_results`) registered in `grade_results.RESULTS_PROVIDERS`.
- Workflows: `generate-nfl.yml` + `--sport nfl` passes on capture/grade.
- Front-end NFL logos/abbr in `app.js`.
- Threading the parked seams (see below).
- End-to-end validation on a past NFL week.

**Out:** changing the MLB pipeline (strictly additive/parameterized); NBA; live/in-game;
re-tuning the P1–P3 models (P4 consumes the committed `rating.json`/`gameline.json`/`props.json`).

**Parked seams threaded in P4 (from P1–P3 ledgers):**
- P1: surface ESPN `displayName` so the matcher can join Odds-API events (`espn.parse_schedule`
  currently emits only abbreviations); thread `commence_shift_hours` (NFL=0); the `nfl_results`
  `home_score`/`away_score` key contract.
- P2: load `gameline.json` → `GameLineConfig` (do NOT serve on bare defaults, `sigma_total=10` vs
  fitted 13.58).
- P3: load `props.json` → `PropConfig` (round-trip test exists); **real backup-share
  redistribution** when a starter is OUT (the producer bumps the backup's projected volume —
  `universe.bump_backup` is a placeholder).

## A. DB schema

`db/migration_nfl_sport.sql` (idempotent; you run it in the Supabase SQL Editor):
- `ALTER TABLE game_predictions ADD COLUMN IF NOT EXISTS sport TEXT DEFAULT 'mlb';`
- `ALTER TABLE prop_predictions  ADD COLUMN IF NOT EXISTS sport TEXT DEFAULT 'mlb';`
- Backfill existing rows to `'mlb'` (the DEFAULT covers new rows; an explicit `UPDATE ... WHERE
  sport IS NULL` for pre-existing).
- `board_picks`/`picks` already carry `sport`. NFL `game_pk` = ESPN event id (~9 digits) does not
  overlap MLB StatsAPI game_pk (~6 digits); the `sport` column disambiguates reads regardless.

## B. Producer — `scripts/generate_nfl.py`

Analog of `generate_sim.py`. For the upcoming NFL slate:
1. **Schedule:** `espn.fetch_schedule(season, week)` → games with `game_pk`, kickoff, teams,
   status. Determine the current NFL season+week (from the date / the committed schedule).
2. **Ratings:** load `assets/nfl/rating.json` → `EloConfig`/`BlendConfig`; run Elo (P1
   `run_elo`) over committed `schedules.parquet` through the prior week (carryover) → current team
   ratings; season-to-date SRS + points ratings from the current season's played games.
3. **Game line:** for each game, `ratings.expected_margin` (P1) + `points.expected_total` (P2's
   points model) → `gameline.build_gameline(model_margin, model_total, market, week, cfg)` with
   `cfg` loaded from `gameline.json`; `market` = the latest captured NFL line for the game (from
   `odds_snapshot`) or `{None,None}` (model-only) if none yet. → `margin_dist`/`total_dist`/
   `home_win_prob`/`pred_home_score`/`pred_away_score`/`pred_total`/`pred_margin`.
4. **Player universe + props:** `universe.active_universe(rosters, injuries, espn_inactives, season,
   week)` (ESPN inactives via `espn.fetch_inactives`); `usage.compute_usage_shares` +
   `efficiency.compute_efficiency` from the latest committed weekly (prefer newest season);
   `gamescript.project_team_volume` (from the game's implied team totals + margin) →
   `usage.allocate` per player; **when a starter is OUT, redistribute their share to the next
   depth-chart player** (real backup bump); `props.build_prop` per market with `PropConfig` loaded
   from `props.json`. Never emit a prop for an inactive player.
5. **Write:** `game_predictions` (`sport='nfl'`, `model_version='nfl-elo-v1'`) +
   `prop_predictions` (`sport='nfl'`, `model_version='nfl-props-v1'`), keyed by ESPN `game_pk`,
   same column shapes MLB uses (margin_dist/total_dist/win_prob/pred_*; player_id/market/dist/
   line/projected_mean/player_name/team_name).
- Prints a summary (`predicted N games, M prop rows`); idempotent upsert.

## C. `generate_board.py` — NFL pass

- Add `'nfl'` to `BOARDABLE_SPORTS` and `PROP_MARKETS_BY_SPORT['nfl'] = (the 7 markets)`.
- Sport-filter the prediction reads: `game_predictions ... WHERE game_date >= %s AND sport = %s`;
  `prop_predictions ... WHERE game_pk = %s AND sport = %s AND model_version = (latest for that
  game+sport)`.
- The board math (best-book price, EV, EV-or-pass, CLV) is UNCHANGED — NFL dists are the same
  serving format (P2/P3 verified consumable by `prob_over_dist`/`prob_cover`). A `--sport nfl` run
  writes `sport='nfl'` `board_picks`/`picks` rows.

## D. NFL odds capture + matcher

- `ingest_odds.py` already sport-parameterized (P0). `capture-odds --sport nfl` fetches game lines
  (`h2h`/`spreads`/`totals`) + the 7 prop market keys for `americanfootball_nfl` → `odds_snapshot`.
- **Matcher wiring:** Odds-API returns events with full team names + commence_time; ESPN provides
  the schedule (`game_pk` = event id + team `displayName`). `espn.parse_schedule` gains
  `home_name`/`away_name` (the `displayName`s) so `matcher.match_odds_event(odds_event,
  espn_games)` resolves each Odds-API event → the ESPN `game_pk`. Store odds under that `game_pk`.
- This VERIFIES the Odds-API NFL team-name strings (parked P0/P1 seam) — add normalization if they
  differ from ESPN's `displayName`. NFL `commence_shift_hours=0` used for date resolution.

## E. `grade_results.py` — NFL results provider

- New `src/sportsmodel/ingest/nfl_results.py`: `final_game_pks(start, end)` (finished ESPN games in
  the window) + `fetch_results(game_pk)` returning a dict with `home_score`/`away_score` (the
  shared game-line contract, from `espn.fetch_final`) AND per-player actuals for the 7 prop markets
  (passing_yards, passing_tds, receiving_yards, receptions, rushing_yards, rush+rec, anytime-TD
  Y/N) from the ESPN box score, keyed so `_actual_for(market, side, res, player_id)` resolves them.
- Register `RESULTS_PROVIDERS['nfl'] = nfl_results`. `grade_pick`/`_actual_for` are generic; a
  `--sport nfl` grade run only touches `sport='nfl'` pending picks (P0). Game-line + prop grading +
  CLV work per-sport.

## F. Workflows

- `generate-nfl.yml`: run `generate_nfl.py` weekly (after Monday-night games, so the next week's
  ratings/carryover are current) + pre-kickoff refresh runs (Sunday morning, as inactives/lines
  firm up). Cron design uses the **P4-cron lesson**: idempotent, run on every firing (no brittle
  exact-hour gate).
- `capture-odds.yml` + `grade-results.yml`: add an NFL leg (`--sport nfl`) alongside the MLB one
  (either extra steps in the same workflow or parallel `nfl-*` workflows), so NFL odds capture,
  board regen, and grading run on their own cadence.
- Secrets/config reuse the existing `ODDS_API_KEY`/`DATABASE_URL` (Session pooler).

## G. Front-end (`app.js`)

- Add an NFL `TEAM_ABBR` map + the ESPN NFL logo path (`https://a.espncdn.com/i/teamlogos/nfl/500/
  {abbr}.png`), mirroring the MLB `logoImg`/`pickLogo` helpers. The NFL page, nav, and market
  filters already exist; board + track-record render NFL rows via the existing `sport` column.

## H. Validation → go-live

- **Past-week end-to-end validation:** run `generate_nfl` for a completed 2025 (or preseason) week,
  regenerate the board, and grade it — confirm predictions are sane (small deviation from the
  market on game lines; plausible player projections; no props for inactives), odds match by
  `game_pk`, and grading produces results + CLV. A validation note under `docs/superpowers/reports/`.
- **Go-live:** enable the NFL workflows for Week 1. The board shows NFL game lines + the 7 prop
  markets with best-book prices, EV, and the EV-or-pass gate; grading + CLV run off ESPN finals.

## Acceptance / launch bar (Week 1)

- The migration runs; NFL `game_predictions`/`prop_predictions` written with `sport='nfl'`.
- `generate_board --sport nfl` produces `sport='nfl'` `board_picks`/`picks`; the site shows NFL
  game lines + the 7 prop markets with best-book prices + EV + EV-or-pass; no props for inactives.
- NFL odds match predictions by `game_pk` (matcher verified on live Odds-API names).
- `grade_results --sport nfl` grades NFL finals + player box scores → `picks` → track record,
  filterable by league; CLV computed.
- **No MLB regression** — MLB predictions/board/grade unchanged; full suite green; NFL additive.
- Past-week validation passed before go-live.

## Testing approach (TDD)

- `generate_nfl`: unit-test the per-game assembly (ratings→gameline→dists; universe→props) with
  injected fixtures (no live calls); a game with a starter OUT bumps the backup; an inactive gets
  no prop; the written rows carry `sport='nfl'` + the right model_version + serving-shaped dists.
- `generate_board`: a `--sport nfl` `build_rows` tags `sport='nfl'` and reads NFL predictions
  (extend the existing sport test); the NFL prop markets are the 7.
- matcher/espn: `parse_schedule` now emits `home_name`/`away_name`; `match_odds_event` resolves a
  fixture Odds-API event → the ESPN game_pk (live-name verification is a runtime step).
- `nfl_results`: `fetch_results` parses a committed ESPN box-score fixture → home/away score + the
  per-player prop actuals under the contract `_actual_for` expects; `final_game_pks` gates on
  STATUS_FINAL.
- Config loaders: `GameLineConfig`/`PropConfig` load from `gameline.json`/`props.json` and build a
  valid game line / all 7 props (props round-trip test exists).
- Full MLB suite stays green (the sport parameterization is additive).

## Risks

- **Odds-API NFL team names unverified until live** — the matcher's live path is finalized at the
  first NFL capture; add normalization then. Historical/board joins key on `game_pk`.
- **Week-1 cold start** — 2024 rates (2025 weekly unpublished); game lines lean on market
  shrinkage; props are directionally calibrated (the `mean_mult` fix). CLV is the season judge.
- **ESPN box-score shape for prop actuals** — `nfl_results.fetch_results` is finalized against a
  real box score during validation; fixture-tested for the parse.
- **game_pk scheme across ESPN/Odds-API/nflverse** — all key on the ESPN event id (P1-verified via
  `schedules.espn`); the matcher is the one live-fragile join.
- **Cron reliability** — use the idempotent-every-firing pattern (the daily-ingest gate lesson), no
  brittle exact-hour gate.
