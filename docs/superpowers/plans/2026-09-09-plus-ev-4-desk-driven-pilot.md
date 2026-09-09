# +EV Sub-project 4: Desk-Driven Forward Pilot (edge/EV + grading + page) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`).

**Goal:** A disciplined, forward-only **+EV board** for NFL/CFB that surfaces bets ONLY where the **decision desk** (injuries/news) moves the number enough vs the **Pinnacle no-vig line** to be +EV. Everything else is a "pass." EV shown at both the Pinnacle price and the best soft book; forward CLV-graded vs the Pinnacle close.

**Why this shape (post-NO-GO):** Sub-project 3's backtest showed the feature model is well-calibrated but does NOT beat the closing line (NFL margin MAE 10.21 vs 9.78). So the model can't be the edge. The honest design: **the base true probability IS the Pinnacle no-vig line** (sharpest estimate available), and the **desk overlays a clamped adjustment** (≤±3.5 pts / ≤~10pp, using #3's calibrated σ + `apply_desk`/prob functions). The gap = the desk's contribution. No desk pick on a market → true = market → edge 0 → pass.

**Spec:** `docs/superpowers/specs/2026-09-09-plus-ev-engine-design.md` (sub-projects 4-6, scoped to desk-driven).

## Global Constraints

- **Market-anchored base.** `base_prob` for each market/side = Pinnacle no-vig implied. `true_prob = clamp(base_prob + desk_delta, 0..1)`, where `desk_delta` is the desk's clamped prob move. The model (#3) supplies only σ + the prob-conversion functions; it is NOT the base.
- **Reuse, don't rebuild:** `serving/board.py` (`implied_prob`, `novig`, `ev`, `best_price`, `MAJOR_BOOKS` incl. `pinnacle`, `EV_CEILING`); `model/trueprob_prob.py` (`win_prob`, `cover_prob`, `over_prob`, `desk_margin_shift`, `DESK_MAX_PROB_DELTA`). σ constants: `SIGMA_MARGIN=13.2`, `SIGMA_TOTAL=10.0` (gameline defaults, validated on-diagonal by #3's calibration).
- **Discipline gates:** a row `is_pick` only when `desk_delta` is non-zero (a desk actually spoke), `|edge| ≥ MIN_EDGE` (default 0.02), and `0 < ev_pinnacle ≤ EV_CEILING`. Default is **pass**. Never present a bet the desk didn't touch.
- **Forward-only / honest:** the board is labeled "assistive · not a proven edge"; picks are graded forward vs the Pinnacle CLOSE (CLV) into `ev_results`. No historical claim.
- **Data sources:** desk picks from `desk_current` (already populated for NFL); odds from `odds_snapshot` (Pinnacle + soft books, from sub-project 1 — needs a live `capture-odds` run before a live board). Upcoming games only (`commence_time > now`).
- Secrets/DB via env in `main()`; the user runs SQL; `app.js` is external. Run tests `PYTHONPATH=src uv run pytest`.

---

### Task 1: Pilot edge/EV engine (pure) + board builder

**Files:**
- Create: `src/sportsmodel/serving/ev_pilot.py` (pure)
- Create: `scripts/build_ev_board.py` (thin IO)
- Create: `tests/serving/__init__.py`, `tests/serving/test_ev_pilot.py`

**Interfaces (PURE in `ev_pilot.py`):**
- `market_margin(pinnacle_home_spread) -> float` = `-pinnacle_home_spread` (book convention home spread → implied home margin).
- `desk_prob_delta(kind, base_number, line, sigma, desk_shift) -> float` — the clamped prob move the desk induces: for `kind="ml"` = `win_prob(base_number+desk_shift, sigma) - win_prob(base_number, sigma)`; `kind="spread"` = `cover_prob(base_number+desk_shift, sigma, line) - cover_prob(base_number, sigma, line)`; `kind="total"` = `over_prob(base_number+desk_shift, sigma, line) - over_prob(base_number, sigma, line)`. Clamp the result to ±`DESK_MAX_PROB_DELTA`. `desk_shift=0` → 0.
- `ev_rows_for_game(game) -> list[dict]` where `game` carries: matchup/game_pk/commence_time/sport; the desk pick (ml_pick/spread_side/total_side/conviction_tier or None); and the Pinnacle two-way prices + soft-book price entries per market. For each of moneyline/spread/total where a two-way Pinnacle price exists:
  - `base_prob` = `novig(price_side, price_other)` (Pinnacle);
  - `desk_shift` = `desk_margin_shift(desk_pick)` signed toward the desk's side for that market (0 if the desk has no pick on that market);
  - `desk_delta` = `desk_prob_delta(...)` around the market number (ml/spread use market margin `-pinnacle_home_spread`; total uses the Pinnacle total line);
  - `true_prob` = clamp(base_prob + desk_delta, 0..1);
  - `edge = true_prob - base_prob`; `ev_pinnacle = ev(true_prob, pinnacle_price)`; `ev_best = ev(true_prob, best_soft_price)` (via `best_price`), with `best_book`;
  - `is_pick` = desk_delta != 0 and |edge| ≥ MIN_EDGE and 0 < ev_pinnacle ≤ EV_CEILING.
  Emit one row per market/side actually considered (the picked side; mirror board.py's side-selection so we emit the side the desk/edge favors).
- Constants: `SIGMA_MARGIN`, `SIGMA_TOTAL`, `MIN_EDGE=0.02`.

- [ ] **Step 1: Test-first** `tests/serving/test_ev_pilot.py`:
  - No desk pick → `desk_delta==0`, `true_prob==base_prob`, `edge==0`, `is_pick False` (pass) for every market.
  - A high-conviction desk home ML pick on a pick'em (Pinnacle -110/-110 → base 0.5) raises true_prob but by ≤`DESK_MAX_PROB_DELTA`; `edge>0`; `is_pick` True when EV clears.
  - `ev_best ≥ ev_pinnacle` when a soft book prices better than Pinnacle (assert the soft-book benefit shows).
  - `market_margin(-3.0)==3.0`.
  - EV ceiling: a contrived huge edge is NOT a pick (capped by EV_CEILING).
  Write first (FAIL), implement, run (PASS).
- [ ] **Step 2: `build_ev_board.py` `main(--sport)`** — read `desk_current` (sport, upcoming) and `odds_snapshot` (latest snapshot per game/market/side/book, Pinnacle + MAJOR_BOOKS), join by game_pk, build one `game` dict per upcoming game, call `ev_rows_for_game`, and upsert to `ev_picks` (Task 2 table). Print counts (games, rows, picks, passes). Pure assembly (`assemble_games(desk_rows, odds_rows)`) factored out and unit-tested.
- [ ] **Step 3: Run tests + ast.parse the script. Commit** `feat(+ev): desk-driven pilot edge/EV engine + board builder`.

---

### Task 2: DB migration — ev_picks + ev_current + ev_results

**Files:** Create `db/migration_ev_pilot.sql`

- [ ] **Step 1:** Idempotent SQL (the user runs it):
  - `ev_picks` (PK `sport, game_pk, market, side, model_version`): matchup, commence_time, base_prob, true_prob, edge, desk_delta, conviction_tier, pinnacle_price, ev_pinnacle, best_book, best_price, ev_best, is_pick, created_at default now().
  - `ev_current` view: upcoming (`commence_time > now()`), DISTINCT ON latest model_version per (sport,game_pk,market,side), ordered by commence_time — for the page. Include a flag so the page can show picks vs. passes.
  - `ev_results` (PK `sport, game_pk, market, side`): won/cover/over result, clv vs Pinnacle close, graded_at.
  - RLS public-read + GRANT SELECT to anon/authenticated, mirroring `migration_decision_desk.sql`.
- [ ] **Step 2:** `db.upsert_ev_picks` / `db.upsert_ev_results` in `src/sportsmodel/db.py` (mirror `upsert_desk_picks`). Commit `feat(+ev): ev_pilot schema + upserts`.

---

### Task 3: Forward CLV grader

**Files:** Create `scripts/grade_ev.py`, `tests/test_grade_ev.py`

- [ ] Pure `grade_ev_pick(pick, final, pinnacle_close) -> dict` — did the picked side win/cover/hit; `clv` = pinnacle close vs pick-time price/number (mirror `grade_desk_picks`'s CLV math + `nfl/cfb espn.fetch_final`). `main(--days)` grades finished games idempotently → `ev_results`. Test the pure grader (win/cover/over correct; push→None; CLV sign). Commit `feat(+ev): forward CLV grader for the +EV board`.

---

### Task 4: Front-end +EV toggle/page (external, controller-applied)

- [ ] In `/Users/ryan/Desktop/CappingAlpha/app.js`: add a **Predictions ↔ +EV** toggle on the NFL/CFB pages; the +EV view reads `ev_current` and renders per row: matchup/market/side, **Pinnacle implied %**, **true %**, **edge**, **EV @ Pinnacle**, **EV @ best book (name)**, the soft-book delta, and the desk's contribution (reuse the desk reasoning panel where a desk pick drove it). Honest empty/pass state. Persist the toggle in `localStorage`. Controller-applied (outside worktree); user redeploys. No prediction-page change.

## Live-run prerequisites (after merge)
- Trigger `capture-odds` (NFL+CFB) so `odds_snapshot` has live Pinnacle + soft-book lines.
- `desk_current` populated (desk run) for the slate.
- Then `build_ev_board.py --sport nfl` → `ev_current` → page. Grade forward with `grade_ev.py` after games; the board earns trust only once the forward CLV record is positive.
