# CFB efficiency ratings (cfb-ratings-v3) — design spec

Date: 2026-10-06 · Status: design approved in chat; awaiting user review of this file
Project 1 of 3 using the user's new CFBD Patreon Tier 3 key (75,000 calls/month + GraphQL realtime). Project 2 = CFB site panels (venue, weather, havoc insights, game flow). Project 3 = live in-game model. Each gets its own spec.

## Goal

Make CFB game projections (margin, total, moneyline) more accurate by adding opponent-adjusted per-play efficiency ratings and game-day context from CFBD to the existing margin/Elo + priors model, and ship it as `cfb-ratings-v3` only if it beats v2 on held-out seasons.

## Current state (v2, live since 2026-09-30, PR #59)

- `scripts/generate_cfb.py` (`GAME_MODEL_VERSION = "cfb-ratings-v2"`): pre-game Elo (`assets/cfb/rating.json`: k 40, hfa_elo 70, carryover 0.9, w_sos 0.45, srs_min_games 3) + season-to-date SRS / opponent-adjusted points, blended with a leak-free preseason prior (`priors.parquet`, `priors_weights.json`, `priors_decay.json`); margin → total/ML via `assets/cfb/gameline.json` (sigma_margin 16.67, sigma_total 16.89, biases). Predictions are model-only (no market input).
- Shared leak-free walk-forward: `src/sportsmodel/cfb/walkforward.py` (`walk`, `raw_model_predictions`, `live_week_state`; ratings refreshed once per completed week). Backtests: `scripts/backtest_cfb_gameline.py`, `backtest_cfb_ratings.py`, `backtest_cfb_priors.py`.
- Held-out 2023–25 (v2): margin MAE 12.61, total MAE 13.10, ATS vs close 49.0%; the closing line's own margin MAE is 12.02.
- Data on hand: `assets/cfb/advanced_games.parquet` (CFBD `/stats/game/advanced`, 2015–2026, REGULAR season only, per team-game off/def plays, PPA, success, explosiveness, pass/rush splits), `lines.parquet` (closes 2015+, openers + moneylines 2021+), `schedules.parquet`, `priors.parquet`. Advanced stats feed only the matchup grades (`src/sportsmodel/context/units.py`), not the ratings.

## User decisions (2026-10-05/06)

- Key: paid tier, Tier 3 or higher; stays in repo secret `CFBD_API_KEY` (user updates the secret value).
- Goal set: sharper game projections + site panels + live model, built in that order (this spec = projections).
- Approach A (efficiency ratings layer on the current model) **plus** a residual check (regularized regression on v3's misses using features A does not use) — report only.
- Ship gate: lower margin AND total MAE than v2 on 2023–25, ATS and O/U vs close no worse than v2, ML calibration no worse. Track record continues (not restarted) when v3 ships.

## 1. New CFBD data

All pulls read `CFBD_API_KEY` from the environment (never logged), cache to committed parquet under `assets/cfb/`, map CFBD team names to ESPN ids via the existing `cfbd_to_espn` (drop + count unmapped/FCS rows), and stamp every row with `season`, `week`, `season_type`, `game_id`.

| Data | CFBD source | Coverage | Asset |
|---|---|---|---|
| Advanced game stats incl. postseason | `/stats/game/advanced` with `seasonType=postseason` added | 2015→ | `advanced_games.parquet` (existing; add `season_type`) |
| Havoc per game (front-seven/DB havoc, total havoc rate) | `/stats/game/havoc` | 2015→ (as available) | `havoc_games.parquet` |
| Drives (points per drive, points per scoring opportunity = trips inside the 40, avg starting field position) | `/drives` aggregated per team-game | 2015→ | `drive_games.parquet` |
| Weather (temp, wind speed, precipitation, dome flag) | `/games/weather` (historical + forecasts for upcoming games) | 2015→ (as available) | `weather_games.parquet` |
| Team talent composite | `/talent` | 2015→ yearly | `talent.parquet` |
| Venues (lat/lon, elevation, dome) | `/venues` | static | `venues.parquet` |

Rest days come from `schedules.parquet`; travel distance and time-zone change from venue coordinates (home venue vs game venue, neutral sites included).
Budget: full backfill ≈ a few hundred calls; in-season weekly refresh ≈ 20–40 calls/week + daily weather for upcoming games (≤ ~10 calls/day) — well under 1% of 75,000/month. A shared CFBD client helper (`src/sportsmodel/cfb/cfbd.py`) adds retry/backoff and a per-run call counter that is logged.

**Leak rule:** a game's features for week W use only games completed before week W (same invariant as `walkforward.py`: appending a later game never changes an earlier game's features). Season-level CFBD aggregates computed at season end (e.g. `/ratings/sp`, season advanced stats) are NEVER used for in-season games of that season.

## 2. Efficiency ratings

Per week, per team, opponent-adjusted offense and defense ratings for these metrics (per play unless noted): PPA (overall, rush, pass), success rate, explosiveness, havoc rate, points per scoring opportunity.

- **Adjustment:** one ridge regression per metric per week over all season-to-date team-games: `metric(off team, def team) = league_mean + off_adj[team] − def_adj[opp] + hfa·home`, weighted by plays; ridge penalty fitted in the walk-forward. FBS-only games; FCS games excluded (as today).
- **Early-season prior:** each team's adjusted rating starts at its previous-season final adjusted rating shrunk toward the league mean by returning production and talent (reuse `priors.py` inputs: returning production, recruiting/talent, portal), and moves to the season-to-date estimate as games accumulate (weight by plays; decay curve fitted, same shape family as `priors_decay.json`).
- **To points:** for a matchup, expected PPA per play for each offense = league + off_adj(team) − def_adj(opp) (rush/pass mixed by each offense's season-to-date pass rate); expected plays per team from both teams' season-to-date pace (plays per game, prior-shrunk); expected points per side = linear map of (expected PPA/play × expected plays, success, explosiveness, finishing) fitted on 2016–22. Efficiency margin = home pts − away pts; efficiency total = sum.

## 3. Blend and context adjustments (v3)

- `margin_v3 = a1·margin_v2_ratings + a2·margin_eff + a3·prior_margin(decaying) + hfa + Σ context_m`
- `total_v3  = b1·total_eff + b2·total_v2 + Σ context_t`
- Context (additive, each fitted, each can be fitted to 0): weather on totals (wind ≥ ~15 mph, temperature < ~40°F, precipitation; dome = 0); travel (distance bucket, ≥ 2 time zones east/west); rest (short week, bye week before the game); talent gap (early-season only, decays with games played).
- Weights and context sizes fitted on 2016–22 in the walk-forward. Moneyline: margin → win probability through the existing calibrated mapping, refit for v3 (`gameline_v3.json`).
- Market data is never an input (model-only, as v2).

## 4. Gate and residual check

- **Baseline reproduction first:** run v2 through the same harness and reproduce its 2023–25 numbers (margin MAE 12.61, total MAE 13.10, ATS 49.0%) within rounding; if not, stop and reconcile before comparing.
- **v3 ships only if on 2023–25 (combined):** margin MAE < v2; total MAE < v2; ATS vs closing spread ≥ v2; O/U vs closing total ≥ v2; ML log-loss ≤ v2 and ECE ≤ v2 + 0.005. Report per season (2023, 2024, 2025) too; the verdict uses the combined numbers, and the report flags any season where v3 is worse.
- **Residual check (report only):** ridge (and a shallow gradient-boosted model for comparison) trained on v3's 2016–22 residuals (margin and total) using features v3 does not use — CFBD FPI/SRS/Elo, pregame WP extras (no market-derived fields), player usage, down/distance splits, any section-1 feature v3 left at weight 0. Report out-of-sample R² and MAE change on 2023–25 and the top features; a feature that reduces held-out MAE in all three seasons is a v4 candidate. Nothing from the residual model ships in this project.
- Output: `assets/cfb/v3_gate.json` (all metrics, per season, pass/fail per criterion, residual-check summary) + a markdown summary in the PR.

## 5. Serving

- `generate_cfb.py` gains `cfb-ratings-v3`, selected by repo variable `CFB_MODEL_VERSION` (default stays `v2` until the user approves the gate result; switching back = one variable change). Committed weights in `assets/cfb/v3_weights.json`.
- Weekly Monday job (`build-cfb-advanced.yml`, 12:00 UTC) also refreshes postseason advanced stats, havoc and drives and recomputes efficiency ratings; a game-day weather pull runs before the daily CFB generate. Weather missing for a game → that game's weather adjustment = 0 (never fabricated).
- Track record continues across the version change (no restart). Site unchanged in this project.

## Testing

Unit tests (network-free, fixtures): each new CFBD parser (incl. unmapped/FCS drops, NaN not 0), the ridge adjustment on a tiny synthetic league (recovers known offsets), the leak invariant (appending a later game leaves earlier features unchanged), context feature construction (dome → no weather effect, travel/time-zone math), the gate evaluator (pass/fail logic), and `CFB_MODEL_VERSION` switching. Existing CFB tests stay green.

## Out of scope

Site panels (project 2), live in-game model / GraphQL (project 3), CFB player props, changing the +EV board logic, restarting the CFB track record, market-informed predictions.

## Risks

- CFBD weather/havoc historical coverage may be patchy before ~2018 → features missing for early seasons are NaN and fitted with missing-indicator handling; the gate still judges 2023–25.
- Efficiency ratings may mostly duplicate the margin signal → the gate will show it; the per-component report says which pieces helped.
- v2 already shows no edge vs the close; v3 may improve MAE without ATS gains — the gate requires "no worse", not "beats the close".
