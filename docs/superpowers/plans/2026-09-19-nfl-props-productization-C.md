# NFL Player-Props Productization (C) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the now-live, calibrated NFL sim (`nfl_player_sim`) into a prop **+EV board with CLV grading** for rush_yds / rec_yds / receptions, offered only on **projected-featured** players, mirroring the game +EV pilot (`build_ev_board.py` → `ev_picks` → `grade_ev.py`).

**Architecture:** New `build_ev_props_board.py` joins `nfl_player_sim_current` (sim prop distributions) to latest prop odds in `odds_snapshot`, computes P(over) at the book line via `serving.board.prob_over_dist` + no-vig market prob + EV, and lands +EV prop rows in a new `ev_prop_picks` table; `grade_ev_props.py` grades them vs nflverse weekly actuals with CLV into `ev_prop_results`. Reuses the game pilot's EV/no-vig math and odds ingestion (NFL prop market map already wired). Builds on main (sim merged + live).

**Tech Stack:** Python, numpy, pandas, psycopg (Postgres/Supabase), nfl_data_py, pytest, uv.

**Spec:** docs/superpowers/specs/2026-09-18-nfl-props-productization-C-scope.md

## Global Constraints

- **NFL only.** Sim is merged + live on main. Do not touch MLB/CFB/desk/game-EV.
- **Deployment gate = projected usage:** only offer/track a prop where the sim projects the player featured (`is_propable_projected(market, dist_mean)` in scripts/backtest_sim_nfl.py — reuse its thresholds). This is the population B.4 validated; grading realized-usage-selected props is what made the model look broken.
- **EV at the book's actual line** via the sim's full pmf (`serving.board.prob_over_dist`), no-vig market prob, `EV = model_P × decimal − 1` — identical discipline to the game pilot and the MLB prop-EV reframe.
- **Leakage-free / pre-game only:** sim dists + prop lines are pre-game; grading uses post-game actuals + the closing line.
- **User runs DB migrations** (SQL files); commit/push only what's asked; branch off main.
- **Calibration ≠ edge.** The gate is forward CLV, not a backtest (calibration already shown in B.4). C makes edge measurable; it does not assume it.
- Market-name map: sim markets `rush_yds`/`rec_yds`/`receptions`/`pass_yds`; odds markets (`SportConfig["nfl"].prop_market_map`) `rush_yds`/`reception_yds`/`receptions`/`pass_yds`. **`rec_yds`↔`reception_yds` must be reconciled.** Sim `player_id` = gsis; odds carry `player_name` → name↔gsis join.
- Ship rush_yds + rec_yds first (calibrated); receptions flagged (p50 .574) — include but watch its CLV. pass_yds excluded (fat tail + bias).

## File Structure

- `src/sportsmodel/sim/nfl/usage.py` — depth-chart fallback in `active_usage` (Task 1).
- `src/sportsmodel/serving/props_ev.py` (new) — pure prop-EV assembly + market/name reconciliation (Tasks 2, 4).
- `db/migration_ev_prop_picks.sql` (new) — `ev_prop_picks` + `ev_prop_results` tables (Task 3; user runs).
- `src/sportsmodel/db.py` — `upsert_ev_prop_picks`, `upsert_ev_prop_results` (Task 3).
- `scripts/build_ev_props_board.py` (new) — board producer (Task 4).
- `scripts/grade_ev_props.py` (new) — grading + CLV (Task 5).
- `.github/workflows/` — build-ev-props + grade-ev-props (Task 6).
- CappingAlpha `app.js` (EXTERNAL; user redeploys) — props view (Task 7).
- Tests under `tests/`.

---

### Task 1: depth-chart fallback in active_usage (enabler)

**Files:** `src/sportsmodel/sim/nfl/usage.py`; Test `tests/sim/nfl/test_usage_active.py` (extend).

**Why:** `active_usage` selects the active set from the depth chart for the TARGET (season, week). When that week's chart isn't published yet (early season, or the current season lagging), the active set is empty → 0 player rows (observed live: n_empty_active=15). Fall back to the most recent AVAILABLE depth-chart (season, week) at or before the target, per team.

**Interface (unchanged signature):** `active_usage(...)` — internal change only. Add a helper `_latest_depth_week(depth_df, team, upto_season, upto_week) -> tuple[int,int] | None`: the max (season, week) in `depth_df` for `club_code==team` with `(season, week) <= (upto_season, upto_week)`; None if none. Use its rows for the active set when the exact (upto_season, upto_week) has none. This is pre-game info (a prior published depth chart), NOT leakage. Document it.

- [ ] **Step 1: failing test** — depth_df has team rows for (2025, wk18) but none for the target (2026, wk1); active_usage returns a non-empty active set built from the (2025, wk18) chart (with shares/efficiency still from the leakage-free recent weekly window). A team with NO depth rows at/before target → still empty (graceful).
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): depth-chart fallback to latest available week`).

---

### Task 2: prop market/name reconciliation (pure)

**Files:** `src/sportsmodel/serving/props_ev.py` (new); Test `tests/serving/test_props_ev.py` (new).

**Interfaces — Produces:**
- `SIM_TO_ODDS_MARKET = {"rush_yds":"rush_yds", "rec_yds":"reception_yds", "receptions":"receptions"}` (pass_yds excluded from C v1).
- `odds_market_for(sim_market) -> str | None`.
- `normalize_player_name(name) -> str` — lowercased, punctuation/suffix-stripped, for joining sim player names (from nfl_player_sim, which carries the player's display name) to odds `player_name`. (Mirror any existing name-normalizer in the codebase if present — search `normalize` in ingest/ and serving/.)

- [ ] **Step 1: failing test** — market map round-trips (rec_yds→reception_yds); unknown→None; name normalization collapses "A.J. Brown"/"AJ Brown"/"A.J. Brown Jr." to a common key. Concrete cases in the test.
- [ ] Steps 2–5 (commit `feat(props): market + player-name reconciliation`).

---

### Task 3: ev_prop_picks / ev_prop_results tables + upserts

**Files:** `db/migration_ev_prop_picks.sql` (new; user runs), `src/sportsmodel/db.py`; Test `tests/test_db_ev_props.py` (new, mirror tests/test_db_nfl_sim.py's row-shape tests).

**Schema (mirror ev_picks, player-oriented):** `ev_prop_picks(sport, game_pk, player_id, player_name, market, side, line, model_version, matchup, commence_time, model_prob, market_prob, edge, ev_best, best_book, best_price, pinnacle_price, open_pinnacle_price, is_pick, created_at)` — PK `(game_pk, player_id, market, line, model_version)`; `open_pinnacle_price` frozen on first insert (CLV), RLS public-read, `ev_prop_picks_current` view (upcoming only). `ev_prop_results(... , actual, result, clv, profit, novig_close, ...)` mirroring ev_results.

**Interfaces — Produces:** `db.upsert_ev_prop_picks(rows)`, `db.upsert_ev_prop_results(rows)` (mirror upsert_ev_picks/upsert_ev_results exactly, incl. the frozen-open-price DO-UPDATE exclusion + json where needed).

- [ ] **Step 1: failing test** — upsert builds the expected INSERT/ON CONFLICT SQL + column list (mirror the existing db upsert tests; no live DB).
- [ ] Steps 2–5 (commit `feat(db): ev_prop_picks + ev_prop_results upserts + migration`).

---

### Task 4: prop +EV board producer

**Files:** `scripts/build_ev_props_board.py` (new), `src/sportsmodel/serving/props_ev.py` (extend); Test `tests/serving/test_props_ev.py` (extend) + `tests/test_build_ev_props_board.py` (pure assemble seam).

**Interfaces — Produces (pure):**
`assemble_prop_rows(sim_rows, odds_rows, model_version) -> list[dict]` where `sim_rows` = nfl_player_sim_current ({game_pk, player_id, player_name, market, dist, commence_time, ...}) and `odds_rows` = latest prop odds_snapshot ({game_pk, market, player_name, side, line, book, price}). For each sim (player, market) that (a) clears `is_propable_projected(sim_market, dist["mean"])` and (b) has a matched odds line (market reconciled + name-joined): compute `p_over = prob_over_dist(dist, line)`, no-vig market prob from over/under prices, `ev` per side (reuse `serving.board.prop_row` or its helpers), pick the +EV side, emit an ev_prop_picks row (is_pick = ev_best > 0). Exclude lines priced worse than the game pilot's cutoff if one applies.
IO `main(--sport nfl)`: load nfl_player_sim_current + latest prop odds (mirror build_ev_board.load_latest_odds but for player-prop markets), call assemble_prop_rows, upsert_ev_prop_picks, clear opposite-side non-picks.

- [ ] **Step 1: failing test** — pure `assemble_prop_rows`: a projected-featured player with a matched line gets a row with correct p_over/ev and the +EV side; a NON-projected-featured player (dist mean below threshold) is excluded; a player with no matched odds line is excluded; rec_yds sim market matches a reception_yds odds line.
- [ ] Steps 2–5 (commit `feat(props): +EV prop board producer`).

---

### Task 5: prop grading + CLV

**Files:** `scripts/grade_ev_props.py` (new), `src/sportsmodel/serving/props_ev.py` (extend); Test `tests/test_grade_ev_props.py` (new).

**Behavior (mirror grade_ev.py):** for each graded prop pick, fetch the player's ACTUAL market value that week (nflverse `import_weekly_data` → rushing_yards/receiving_yards/receptions, joined by gsis) + the CLOSING prop line (last odds_snapshot before commence for that player/market/line), compute result (over/under hit vs the pick line), profit at pick-time price, and CLV vs the closing no-vig prob. Upsert ev_prop_results. Pure helpers unit-tested (result + CLV math); heavy IO not unit-tested.

- [ ] **Step 1: failing test** — pure: given a pick (side, line, pick-price) + actual + closing line, `grade_prop(...)` returns correct result/profit/clv (over hit, under hit, push on exact line).
- [ ] Steps 2–5 (commit `feat(props): prop grading + CLV`).

---

### Task 6: workflows

**Files:** `.github/workflows/build-ev-props.yml`, `.github/workflows/grade-ev-props.yml`.

**Behavior:** build-ev-props runs after prop-odds capture (post-lineup window), reads nfl_player_sim_current + odds_snapshot, writes ev_prop_picks — mirror the game board's schedule. grade-ev-props runs periodically (reads nflverse + DB, no Odds API credits) like grade-ev. Confirm NFL prop-odds CAPTURE is enabled (ingest_odds fetches NFL player-prop markets within PROP_WINDOW_MIN; set the repo var / INGEST_PROPS if needed — document the ops step, don't hardcode secrets).

- [ ] Steps: add workflows mirroring the game-EV ones; commit `ci(props): build + grade prop EV board`. (No unit test; validated by a manual dispatch during rollout.)

---

### Task 7: front-end props view (EXTERNAL — user redeploys)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js` (external; user redeploys to Cloudflare).

**Behavior:** add an NFL props view reading `ev_prop_picks_current` (list +EV props: player, market, line, model P, EV, best book/price) and a props track-record reading `ev_prop_results` (CLV, hit rate, ROI) — mirror the existing game board/track views. Provide the diff; user redeploys + hard-refreshes.

- [ ] Steps: implement in app.js; hand the user the redeploy instruction. (No automated test; verify in the browser pane against live data.)

---

### Task 8: rollout gate (controller-driven)

Not a code task — after Tasks 1–6 land + the user runs the migration and enables capture: dispatch build-ev-props once (post-lineup), confirm ev_prop_picks populates for projected-featured players, then let CLV accrue over real weeks (the forward gate). A market with positive CLV over enough graded props earns continued offering; flat/negative gets cut — same discipline as the game/MLB pilots. Surface findings.

## Self-Review

- **Spec coverage:** projected-usage deployment gate (T4 via is_propable_projected), reuse of odds ingestion + EV math (T2/T4), sim dists (source), CLV grading (T5), forward-CLV gate (T8), front-end (T7). Depth-chart fallback (T1) unblocks player rows. ✔
- **Type consistency:** sim `dist` (pmf dict) → `prob_over_dist`; ev_prop_picks cols mirror ev_picks; `assemble_prop_rows(sim_rows, odds_rows, model_version)` pure. ✔
- **Leakage:** sim + lines pre-game; grading post-game; depth fallback uses a PRIOR published chart (pre-game). ✔
- **No placeholders:** each task has interfaces + concrete test intent; market/name reconciliation pinned; ev_picks template identified.
- **Ordering:** 1 (fallback, unblocks rows) → 2 (reconcile) → 3 (tables) → 4 (board) → 5 (grade) → 6 (workflows) → 7 (front-end) → 8 (forward gate).
- **Reuse check:** SportConfig NFL prop map, ingest_odds prop path, serving/board.py prop_row + no-vig, build_ev_board/grade_ev templates — all confirmed present in Phase 0.
