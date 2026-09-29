# CFB Profit Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CFB ratings model's side-of-the-line as the source of CFB bets with per-market models trained to beat the price being bet, sized by fractional Kelly, gated on held-out bankroll growth, and served live with stakes on the site.

**Architecture:** Richer history (CFBD openers/closes/moneylines, ESPN schedule context, priors back to 2015) → a shared leak-free CFB walk-forward (`sportsmodel.cfb.walkforward`) → a feature table with one row per (game, market, price point) → per-market HGB classifiers with a monotone line constraint + walk-forward isotonic calibration (`sportsmodel.cfb.profit_model`) → a pure Kelly policy + bankroll simulator (`sportsmodel.cfb.kelly`) → `scripts/gate_cfb_profit.py` (tune 2019–22, verdict 2023–25) → final fit/Release → live `generate_cfb_bets.py` writing `cfb_model_bets` → grading → site.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn (HistGradientBoostingClassifier, IsotonicRegression), CFBD + ESPN APIs, Supabase Postgres, GitHub Actions, CappingAlpha static JS.

**Spec:** `docs/superpowers/specs/2026-09-29-cfb-profit-model-design.md`

## Global Constraints

- CFB game lines only (moneyline, spread, total). NFL untouched.
- Every feature for a game uses only information available before that game (ratings from strictly earlier weeks; priors are preseason; the priced line is the line being bet).
- Spread convention everywhere: home-margin line `L` (the expected home margin; home favored ⇒ positive, exactly `market_spread` today — e.g. `L = 7` is the home team −7 at the book). A home spread bet wins iff `actual_margin > L`, pushes iff `actual_margin == L`, else loses; the away bet is the reverse.
- Kelly: `edge = p·d − 1`, `stake = kelly_frac · edge / (d − 1)`, cap **3 %** per bet, **15 %** per slate day (proportional scale-down), at most one side per (game, market), push returns the stake.
- Tuning seasons **2019–2022**; verdict seasons **2023, 2024, 2025** — never used for any choice.
- Ship rule per market (spec §4): log growth > 0 combined AND in ≥ 2 of 3 verdict seasons at the opener; ROI 95 % CI lower bound > 0 (opener, combined); ECE < 0.03; beats the ratings-model baseline on combined log growth.
- The ratings model (`cfb-ratings-v1`) and its game-page projections are unchanged; the Pinnacle line-shopping board is unchanged.
- `CFB_PROFIT_MODEL` repo variable defaults `off` (today's behavior).
- The user runs all DB migrations/writes; commit/push/merge/publish only when the user asks.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- `.venv/bin/python -m pytest -q` green after every task.

## Rulings made while planning

- P1 Weather is left out of v1 (spec: optional; venue data is patchy).
- P2 Preseason priors are backfilled to 2015 (portal data starts 2021 → NaN before; HGB routes NaN natively).
- P3 The line feature carries a monotone constraint (harder line ⇒ lower win probability) in the spread and total models; moneyline has no line feature (its price enters only the Kelly step).
- P4 The ratings walk-forward currently lives in `scripts/backtest_cfb_gameline.py`; it moves to `src/sportsmodel/cfb/walkforward.py` (behavior-identical, the backtest imports it) so the gate and the live step share one implementation.
- P5 Historical moneyline tests use the CFBD median closing moneylines; spread/total tests use −110 both sides (CFBD gives no juice history).

---

### Task 1: CFBD lines v2 (openers, closes, moneylines)

**Files:** Modify `scripts/build_cfb_lines.py`, `.github/workflows/build-cfb-lines.yml`; Test `tests/scripts/test_build_cfb_lines.py` (create).

**Interfaces:**
- `parse_game_lines(game: dict) -> dict | None` (PURE): from one CFBD `/lines` game payload → `{season, week, home_team, away_team, market_spread, market_total, spread_open, total_open, ml_home, ml_away, n_providers}`. `market_spread`/`spread_open` in home-margin convention (= −CFBD `spread`/`spreadOpen`), each the median across providers (None when no provider has it); `ml_home`/`ml_away` = median of `homeMoneyline`/`awayMoneyline` (American, rounded to int); returns None when every price field is None or teams don't map.
- `coverage(df) -> pd.DataFrame`: per season, share of rows with each of `market_spread, spread_open, market_total, total_open, ml_home&ml_away`.
- CLI default `--seasons 2015..2025`; prints the coverage table. Existing columns keep their meaning (downstream readers unchanged).
- Workflow `build-cfb-lines.yml`: `workflow_dispatch` input `seasons` (default `2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025`); commits the parquet to the dispatched ref.

- [ ] Step 1: failing tests — a fixture game with 3 providers (one missing `spreadOpen`, one missing moneylines) → medians correct, sign flip correct, `n_providers == 3`; a game with no price fields → None; coverage on a 2-season frame.
- [ ] Step 2–4: implement, run, full suite green. (The network pull is run later by the controller via the workflow — the API key is a GitHub secret.)
- [ ] Step 5: commit `feat(cfb): CFBD openers, closes and moneylines`.

---

### Task 2: Schedule context + priors backfill

**Files:** Modify `src/sportsmodel/cfb/espn.py` (`parse_schedule`), `scripts/build_cfb_schedules.py`, `.github/workflows/build-cfb-priors.yml`, `scripts/build_cfb_priors.py` (only if a pre-2021 endpoint errors); Test `tests/cfb/test_espn.py` (or the existing espn test file), `tests/scripts/test_build_cfb_priors.py` (existing).

**Interfaces:**
- `parse_schedule` rows gain `game_pk` (ESPN event id, int), `start_date` (ISO UTC), `neutral_site` (bool), `conference_game` (bool), `home_conf` / `away_conf` (ESPN conference id or None). Existing keys unchanged.
- `schedules.parquet` gains those columns; `build_cfb_schedules.py --seasons 2015 … 2025` rebuilds them (ESPN is public; run it locally in this task and commit the parquet).
- `build-cfb-priors.yml`: `workflow_dispatch` input `seasons` (default `2015 2026`); the builder tolerates an endpoint returning nothing for early seasons (the field stays NaN for that season) instead of failing.
- FBS membership stays `assets/cfb/fbs_teams.json`.

- [ ] Step 1: failing tests — `parse_schedule` on a fixture event with `neutralSite`, `conferenceCompetition`, conference ids; priors builder with an empty portal payload for 2016 → rows written with `portal_net` NaN.
- [ ] Steps 2–4: implement; rebuild schedules 2015–2025 locally (report rows per season and the share of neutral-site / conference games); full suite green.
- [ ] Step 5: commit `feat(cfb): schedule context columns; priors backfill to 2015`.

---

### Task 3: Shared walk-forward + feature table

**Files:** Create `src/sportsmodel/cfb/walkforward.py`, `src/sportsmodel/cfb/profit_features.py`; Modify `scripts/backtest_cfb_gameline.py` (import from walkforward, behavior identical); Tests `tests/cfb/test_walkforward.py`, `tests/cfb/test_profit_features.py`.

**Interfaces:**
- `walkforward.raw_model_predictions(schedule_df, elo_cfg, blend_cfg) -> list[dict]` — moved verbatim from `backtest_cfb_gameline._raw_model_predictions` plus pass-through of every extra schedule/line column present on the row (`game_pk, start_date, neutral_site, conference_game, home_conf, away_conf, spread_open, total_open, ml_home, ml_away`) and pre-game `elo_home`, `elo_away`, `srs_home`, `srs_away` (None when unknown). `backtest_cfb_gameline` keeps its public behavior (its tests pass unchanged).
- `profit_features.build_rows(raw: list[dict], priors: pd.DataFrame, fbs: set[str]) -> pd.DataFrame` — one row per (game, market ∈ {spread, total, moneyline}, price_point ∈ {open, close}) where that price exists:
  - keys: `season, week, game_pk, home_team, away_team, market, price_point`;
  - `line` (spread: home-margin line; total: the total; moneyline: NaN), `ml_home`, `ml_away` (moneyline rows);
  - features `f_*`: `f_model_margin`, `f_model_total`, `f_elo_diff`, `f_srs_diff`, `f_edge_pts` (spread: model_margin − line; total: model_total − line; ML: model_margin), `f_move` (priced line − opener; 0 at the opener; NaN if no opener), `f_neutral`, `f_conf_game`, `f_week`, `f_class` (0 = both FBS P4, 1 = P4 vs G5, 2 = both G5, 3 = FBS vs FCS — conference-id lists in the module), `f_rest_home`, `f_rest_away` (days since each team's previous game, NaN if none this season), and per-side + difference priors `f_sp_*`, `f_ret_*`, `f_rec_*`, `f_portal_*`, `f_coach_new_*`, `f_qb_ret_*`;
  - label `y`: spread: 1 if home covers `actual_margin > line`, 0 if not, row dropped on push; total: over; moneyline: home win (ties dropped).
- `FEATURE_COLS: dict[str, list[str]]` per market (the moneyline list excludes `line`/`f_move`).

- [ ] Step 1: failing tests — (a) `raw_model_predictions` equals the old function's output on a small fixture (parity); (b) appending a later game never changes earlier rows (leak test); (c) `build_rows` label/push logic for spread and total at open and close, `f_move` sign, rest days, class mapping; (d) priors join by season+team with NaN when missing.
- [ ] Steps 2–4: implement, run, full suite green.
- [ ] Step 5: commit `feat(cfb): shared walk-forward and profit feature table`.

---

### Task 4: Per-market probability models

**Files:** Create `src/sportsmodel/cfb/profit_model.py`; Test `tests/cfb/test_profit_model.py`.

**Interfaces:**
- `fit_market(df: pd.DataFrame, market: str, *, max_iter=300, seed=0) -> MarketModel` — `HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=100, l2_regularization=1.0, max_iter=max_iter, random_state=seed, monotonic_cst=…)` on `FEATURE_COLS[market]`; monotone constraint `-1` on `line` for spread (higher home line ⇒ lower P(home covers)) and total (higher total ⇒ lower P(over)); `+1` on `f_edge_pts`.
- `MarketModel.predict(df) -> np.ndarray` raw probabilities; `.cols`.
- `walk_forward_oof(df, market, seasons: list[int], min_train_season=2015) -> pd.DataFrame` — for each season S in `seasons`: fit on rows with `season < S`, predict season S → columns `p_raw`; then isotonic calibration for season S fitted on the OOF `p_raw`/`y` of seasons < S (needs ≥ 2 prior OOF seasons; else identity) → `p`.
- `ece(p, y, bins=10) -> float`.
- `Calibrator` (isotonic, `out_of_bounds="clip"`), `fit_calibrator(p_raw, y)`.

- [ ] Step 1: failing tests — synthetic data where `y ~ Bernoulli(sigmoid(0.3·(edge − 0)))`: OOF AUC > 0.6; monotone check: predictions fall as `line` rises with everything else fixed; calibrated ECE < 0.03 on a large synthetic set; walk-forward never trains on the predicted season (assert via a planted season-S-only feature).
- [ ] Steps 2–4: implement, run, full suite green. Step 5: commit `feat(cfb): per-market calibrated probability models`.

---

### Task 5: Kelly policy + bankroll simulator

**Files:** Create `src/sportsmodel/cfb/kelly.py`; Test `tests/cfb/test_kelly.py`.

**Interfaces (PURE):**
- `american_to_decimal(a: float) -> float`.
- `choose_side(p_home: float, d_home: float, d_away: float, min_edge: float) -> tuple[str, float] | None` — for spread/total `p_home` = P(home covers)/P(over) and the other side's p = 1 − p_home; returns `("home"|"away" or "over"|"under", edge)` for the larger positive edge ≥ min_edge, else None.
- `stake_fraction(p: float, d: float, kelly_frac: float, cap: float = 0.03) -> float` = `min(cap, kelly_frac · max(0, p·d − 1) / (d − 1))`.
- `apply_day_cap(stakes: list[float], cap: float = 0.15) -> list[float]` (proportional scale-down).
- `simulate(bets: pd.DataFrame, start: float = 100.0) -> dict` — `bets` has `day, stake_frac, dec, result ∈ {win, loss, push}` in chronological order; stakes within a day are placed on the bankroll at the start of that day; returns `{end_bankroll, log_growth_total, log_growth_per_bet, n_bets, staked, profit, roi, max_drawdown}`.
- `flat_pnl(bets) -> dict` ($10 per bet).
- `roi_ci(bets, n_boot=1000, seed=0) -> tuple[float, float]` — season-week cluster bootstrap of ROI on money staked.

- [ ] Step 1: failing tests — hand-computed cases: `stake_fraction(0.55, 1.909, 0.25)`; day cap scaling to 15 %; a push leaves the bankroll unchanged; `simulate` on a 3-day script matches a hand computation; drawdown on a known path; `choose_side` picks the larger edge and respects `min_edge`.
- [ ] Steps 2–4: implement, run, full suite green. Step 5: commit `feat(cfb): Kelly policy and bankroll simulator`.

---

### Task 6: The gate

**Files:** Create `scripts/gate_cfb_profit.py`; Test `tests/scripts/test_gate_cfb_profit.py`.

**Interfaces:**
- Loads schedules + lines + priors, runs `walkforward.raw_model_predictions` once (2015–2025), builds the feature table, and for each market runs `walk_forward_oof` over seasons 2017–2025 (training from 2015).
- Baseline probabilities: the ratings model's own `P(home covers | line)` / `P(over)` / `P(home win)` from a Normal with `sigma_margin` / `sigma_total` of `assets/cfb/gameline.json` around `model_margin` / `model_total` (no market shrink), push mass excluded.
- Policy grid per market: `kelly_frac ∈ {1/4, 1/3, 1/2}`, `min_edge ∈ {0, .01, .02, .03, .04, .05, .06}` — chosen on 2019–22 at the opener by `log_growth_total` (ties → smaller kelly_frac, then larger min_edge); the same tuning for the baseline.
- Verdict 2023–25 at the opener AND at the close: per season and combined `simulate`, `flat_pnl`, `roi_ci`, ECE, bets placed; coverage rules from spec §1 (a season counts for a market at a price point only with ≥ 70 % coverage; moneyline also ≥ 70 % both-moneyline coverage).
- `ship_decision(market_results) -> dict` implements the Global-Constraints ship rule exactly.
- Outputs `assets/cfb/profit_gate.json` and `docs/superpowers/reports/<date>-cfb-profit-gate.md` (per market: chosen policy, verdict table by season × price point, calibration deciles, baseline comparison, pass/fail with reasons).

- [ ] Step 1: failing tests — with the model/walk-forward stubbed: tuning never reads verdict-season rows (planted verdict-only signal ignored); ship_decision truth table (each clause failing alone fails the market); coverage rule excludes a thin season; report renders.
- [ ] Steps 2–4: implement, run (tests only — the real run is the controller's), full suite green.
- [ ] Step 5: commit `feat(cfb): profit gate (tune 2019-22, verdict 2023-25)`.

---

### Task 7: Final fit, artifacts, weekly retrain

**Files:** Create `scripts/fit_cfb_profit.py`, `.github/workflows/train-cfb-profit.yml`; Test `tests/scripts/test_fit_cfb_profit.py`.

**Interfaces:**
- `fit_cfb_profit.py`: refuses unless `assets/cfb/profit_gate.json` exists and at least one market passed; fits each PASSING market on all completed games, fits its calibrator on all OOF rows, writes `data/cfb_profit/models/{market}.joblib`, `calib_{market}.joblib`, `cfb_profit_config.json` (`model_version: "cfb-profit-v1"`, passing markets with their `kelly_frac`/`min_edge`, feature cols, trained_through, git) atomically; `--quick-check N`: holdout the last N completed weeks after the gate's data (like props-ML), fail if pooled log growth < 0.
- `train-cfb-profit.yml`: weekly (Tue) + dispatch; refresh lines/schedules/priors inputs it needs, quick check, fit, publish Release `cfb-profit-latest` with `MANIFEST.sha256` last (props-ML pattern); skipped with a notice when no market passed.

- [ ] Step 1: failing tests — refusal with no passing market; only passing markets fitted; config contents; atomic write; quick-check fail path.
- [ ] Steps 2–4: implement, run, full suite green. Step 5: commit `feat(cfb): final fit, artifacts and weekly retrain`.

---

### Task 8: Live bets + storage

**Files:** Create `scripts/generate_cfb_bets.py`, `db/migration_cfb_model_bets.sql`, `.github/workflows/generate-cfb-bets.yml`; Modify `src/sportsmodel/db.py` (upsert + reads); Test `tests/scripts/test_generate_cfb_bets.py`, `tests/test_workflows_cfb_profit.py`.

**Interfaces:**
- Migration: table `cfb_model_bets (sport, game_pk, market, side, line, price, book, model_prob, edge, stake_pct, kelly_frac, model_version, commence_time, created_at)` PK `(game_pk, market, created_at)`; view `cfb_model_bets_current` = latest row per (game_pk, market) with `commence_time > now()`; table `cfb_model_bet_results (game_pk, market, side, line, price, book, stake_pct, model_prob, result, flat_pnl, kelly_units, clv, graded_at)`; RLS public read + anon grants; `in_track_record` respected in any record view.
- `generate_cfb_bets.py`: loads artifacts (manifest-verified download dir `data/cfb_profit/models`), builds live feature rows for upcoming CFB games with the SAME `walkforward` + `profit_features` code (current season to date + upcoming week; the priced line = each book's current line), prices both sides at the **best available price for the best line** per side across tracked books in the latest capture (`odds_snapshot`, `commence_time > now()`), applies `choose_side` / `stake_fraction` / `apply_day_cap` per slate day, writes rows; no-op with a notice when `CFB_PROFIT_MODEL` is not `on` or artifacts are missing.
- `generate-cfb-bets.yml`: runs after `capture-odds` for CFB (same cron as `build-ev-board` CFB), `if: vars.CFB_PROFIT_MODEL == 'on'`, downloads + verifies the Release.

- [ ] Step 1: failing tests — best-line/best-price selection per side; day cap applied per slate day; off switch writes nothing; missing artifacts → notice, exit 0; feature parity: a live row for a historical game equals the table row the gate used.
- [ ] Steps 2–4: implement, run, full suite green. Step 5: commit `feat(cfb): live model bets`.

---

### Task 9: Grading + track record views

**Files:** Create `scripts/grade_cfb_bets.py`; Modify the grading workflow that runs `grade_ev.py` (add a step); `db/migration_cfb_model_bets.sql` (append views); Test `tests/scripts/test_grade_cfb_bets.py`.

**Interfaces:**
- Grade each game's LAST flagged bet per market before kickoff (`cfb_model_bets` latest `created_at < commence_time`): result from the final score (ESPN finals as `grade_ev` does), `flat_pnl` ($10), `kelly_units` (= stake_pct × 100 × (d − 1) on a win, −stake_pct × 100 on a loss, 0 push — units of a 100-unit bankroll, non-compounding per bet for the record), `clv` vs the Pinnacle closing no-vig price for that side/line when captured.
- Views: `cfb_model_bet_record` (by market: n, W-L-P, flat P&L, ROI, avg CLV) and `cfb_model_bankroll_daily` (compounded bankroll path from 100 units by game day), both filtered by `in_track_record('cfb', commence_time)`.

- [ ] Step 1: failing tests — grading truth table incl. push; last-flag-before-kickoff selection; CLV sign.
- [ ] Steps 2–4: implement, run, full suite green. Step 5: commit `feat(cfb): grade model bets + record views`.

---

### Task 10: Site (CappingAlpha, external)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`, `*.html` (cache key bump), `settings.html` if the settings form lives there.

- Settings: `bankroll` (default 1000, persisted with the existing settings helper).
- +EV page: read `cfb_model_bets_current`; show CFB model bets with a **Model** tag and `Stake X.X% (≈ $Y)`; line-shop picks unchanged; filters/sorts keep working.
- CFB game page: a "Model bets" block per flagged market (side, line @ price at book, model prob, edge, stake).
- Track record: "CFB model bets" section from `cfb_model_bet_record` + `cfb_model_bankroll_daily` (bankroll chart from 100 units, flat $10 P&L, ROI, win rate, CLV, by market).
- Everything hidden when the views are missing or empty (site deploys before the migration safely).
- Verify with `node --check app.js` and a node harness over the new render functions (fixtures for empty/one/many bets, stake dollars from bankroll); snapshot the originals before editing.

- [ ] Steps: snapshot, implement, harness, cache key bump, diff file for review.

---

### Task 11: Controller — data pulls, gate run, report

- [ ] Dispatch `build-cfb-lines` (seasons 2015–2025) and `build-cfb-priors` (2015–2026) on the branch; pull; check coverage tables.
- [ ] Run `scripts/gate_cfb_profit.py`; commit `profit_gate.json` + report.
- [ ] Report the verdict per market to the user (numbers either way). Only on pass and the user's go-ahead: merge, run the migration (user), publish via `train-cfb-profit`, set `CFB_PROFIT_MODEL=on`, dispatch `generate-cfb-bets`, verify rows, user redeploys the site.
