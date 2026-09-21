# NFL Game-Page Simulation Visual Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add an Apple/MVPEAV-style simulation panel to each NFL game page: median team totals, an interactive Spread/Total/team-total distribution histogram, a median boxscore by player, and a click-to-open per-player stat distribution — all from the sim's stored pmfs.

**Architecture:** Persist the sim's game-level distributions (margin, total, each team's score) to `nfl_sim` alongside the scalars already there; player pmfs already live in `nfl_player_sim.dist`. The external `app.js` renders lightweight inline-SVG histograms with an interactive line input (→ Under/At/Over %).

**Tech Stack:** Python 3.12 (numpy, psycopg), pytest; vanilla JS/SVG in CappingAlpha `app.js`.

**Spec:** this plan (design approved in-chat: full scope incl. spread/total histogram, interactive).

## Global Constraints

- NFL only. Reuse `sim/engine.py` pmf helpers (`margin_pmf`, `total_pmf`, `stat_pmf`); `NflGameSims` duck-types `GameSims` (home_score/away_score arrays).
- Dist JSON formats (consumed by the front-end): margin = `{"kind":"margin","offset":O,"pmf":[...]}` (value = i-O); score/total = `{"kind":"pmf","pmf":[...]}` (value = i). pmf is per-point.
- User runs migrations and redeploys `app.js`; commit/push only backend. Regenerate sim after backend merge.
- Idempotent upserts unchanged; started games keep their frozen pre-game sim.

---

### Task 1: Persist game-level distributions in `nfl_sim`

**Files:**
- Create: `db/migration_nfl_sim_dist.sql`
- Modify: `src/sportsmodel/db.py` (`_NFL_SIM_COLS`, `upsert_nfl_sim` JSON-encode dist cols)
- Modify: `scripts/generate_sim_nfl.py` (compute dists into sim_rows)
- Test: `tests/test_db_nfl_sim.py` (dist cols json-encoded), `tests/sim/nfl/test_generate_sim_nfl.py` (assemble emits dists)

**Interfaces:**
- Produces: `nfl_sim.margin_dist/total_dist/away_score_dist/home_score_dist` (JSONB); each sim row dict carries those four keys.

- [ ] **Step 1: Migration** — `ALTER TABLE nfl_sim ADD COLUMN IF NOT EXISTS margin_dist JSONB` (×4: margin_dist, total_dist, away_score_dist, home_score_dist); `CREATE OR REPLACE VIEW nfl_sim_current` re-selecting the existing columns **plus** the four dists; `GRANT SELECT ON nfl_sim_current TO anon, authenticated`. (Read the current view body from `db/migration_nfl_sim.sql` and add the four columns to the SELECT.)

- [ ] **Step 2: Failing db test** — extend `tests/test_db_nfl_sim.py`: an `upsert_nfl_sim` row carrying `margin_dist={"kind":"margin","offset":2,"pmf":[0.5,0.5]}` produces a tuple whose `margin_dist` position is a JSON **string** (`json.loads` round-trips), and the four dist cols are in `_NFL_SIM_COLS`. Run → fails.

- [ ] **Step 3: Implement db** — add the four names to `_NFL_SIM_COLS`; in `upsert_nfl_sim`'s row tuple, `json.dumps(r.get(c)) if c in _NFL_SIM_DIST_COLS else ...` (define `_NFL_SIM_DIST_COLS = {"margin_dist","total_dist","away_score_dist","home_score_dist"}`), keeping the existing model_version default. Run → passes.

- [ ] **Step 4: Failing generate test** — in `tests/sim/nfl/test_generate_sim_nfl.py`, assert `assemble_sim_rows(...)[0][0]` (a sim row) has `margin_dist["kind"]=="margin"`, `total_dist["kind"]=="pmf"`, and `len(away_score_dist["pmf"])>0`. Run → fails.

- [ ] **Step 5: Implement generate** — in `assemble_sim_rows`, import `margin_pmf, total_pmf, stat_pmf` from `sportsmodel.sim.engine`; add to the sim_rows dict:
```python
"margin_dist": margin_pmf(sims, half_range=45),
"total_dist": {"kind": "pmf", "pmf": total_pmf(sims, max_total=90)},
"away_score_dist": {"kind": "pmf", "pmf": stat_pmf(sims.away_score, 70)},
"home_score_dist": {"kind": "pmf", "pmf": stat_pmf(sims.home_score, 70)},
```
Run → passes.

- [ ] **Step 6:** Full suite green. **Commit** — `feat(sim/nfl): persist game margin/total/team-score distributions`.

---

### Task 2: Front-end — SVG histogram renderer with interactive line (external `app.js`)

**Files:** Modify `/Users/ryan/Desktop/CappingAlpha/app.js`

- [ ] **Step 1:** Add pure helpers:
  - `distValueAt(dist, i)` → `dist.kind==="margin" ? i - dist.offset : i`.
  - `distProbs(dist, line)` → iterate pmf, summing `under` (value<line), `at` (value===line), `over` (value>line); returns `{under,at,over}` as fractions.
  - `distBins(dist, targetBars=26)` → group the per-point pmf into ~26 display buckets, returning `[{lo,hi,label,freq}]` (freq summed), for readable bars (per-point yardage pmfs are too fine to draw raw).
- [ ] **Step 2:** `histogramSVG(dist, {line, accent, height})` → an inline `<svg>` bar chart from `distBins`, bars colored `accent`, bars at/over the line tinted differently, plus a vertical line marker at `line` when set. Viewport-width responsive (`viewBox` + `width:100%`).
- [ ] **Step 3:** `probVsLineRow(dist, line)` → the "Under X% · At Y% · Over Z%" strip from `distProbs`.
- [ ] **Step 4:** A small delegated-event handler: a line `<input>` with `data-dist-target` re-renders its chart + prob row on input (store the current dists on a module object keyed by an id; on input, recompute and replace `innerHTML` of the chart container). Debounce not needed (cheap).
- [ ] **Step 5:** `node --check app.js`; unit-check the pure helpers in the browser console against a known pmf (e.g. a symmetric margin dist → under≈over at line 0).

---

### Task 3: Front-end — Game Simulation panel (team totals + tabbed histogram)

**Files:** Modify `app.js` (`buildGame`, new `gameSimVisual(r, sim)`), `injectStylesOnce`

- [ ] **Step 1:** `buildGame` already fetches `nfl_sim` (base) → `simRows[0]` has the four dists. Pass `simRows[0]` to a new `gameSimVisual(r, sim)` and insert its output after `simSec`.
- [ ] **Step 2:** `gameSimVisual`:
  - **Median Team Totals:** away median vs home median score from `Math.round((sim.sim_total - sim.sim_margin)/2)` (away) and `+ ...+sim_margin` (home) — or from `away_score_dist`/`home_score_dist` medians; show `AWAY  21  vs  24  HOME` with team names/colors (away-left, home-right, matching the hero theme).
  - **Tabs:** Spread (`margin_dist`), Total (`total_dist`), `${away} total` (`away_score_dist`), `${home} total` (`home_score_dist`). Clicking a tab swaps the active dist into the chart; a line `<input>` drives `histogramSVG` + `probVsLineRow`. Default line = the dist's median (rounded).
  - Guard: if `sim` lacks the dist fields (old row pre-migration), render the existing scalar `simSection` only and skip the histogram (no crash).
- [ ] **Step 3:** CSS in `injectStylesOnce`: `.sim-visual`, `.sim-tabs`/`.sim-tab.active`, `.sim-hist svg`, `.prob-line` (the Under/At/Over strip), `.team-totals` (big score, away-left/home-right). Match the dark theme + team accents.
- [ ] **Step 4:** Bump `app.js?v=`; `node --check`; browser-verify tabs switch, line input updates bars + Under/At/Over, and a game whose sim row predates the migration still renders.

---

### Task 4: Front-end — Boxscore (Median by Player)

**Files:** Modify `app.js` (new `boxscoreSection(sims, r)`)

- [ ] **Step 1:** Group `sims` (nfl_player_sim rows, one per player/market with `mean`) by team (home/away via `s.team` vs `r.home_team_name`) and by role: QBs (pos QB), Rushers (rush_yds mean > 0), Receivers (rec_yds mean > 0 or receptions > 0). Dedup markets per player (already deduped upstream by `dedupLatest`).
- [ ] **Step 2:** Render two columns (away | home), each with QBs (Pass Yds/Pass TD/Rush Yds — round means), Rushers (Rush Yds/Rush TD/… from `pass_tds`? no — Rush TD not stored; show Rush Yds + a TD proxy from anytime_td%?). Show the stats we HAVE: QB = Pass Yds, Pass TD (mean), Rush Yds; Rushers = Rush Yds, Att n/a (not stored) → show Rush Yds + Any TD%; Receivers = Rec Yds, Receptions, Any TD%. Keep columns to stored stats only; label honestly.
- [ ] **Step 3:** CSS `.boxscore` (two-column, compact tables mirroring `prop-proj`). Insert after the game-sim panel.
- [ ] **Step 4:** Browser-verify against a game with player sims.

---

### Task 5: Front-end — per-player interactive distribution (click a player)

**Files:** Modify `app.js`

- [ ] **Step 1:** Make each boxscore/props player row clickable (`data-player-id`, `data-market` default the row's headline market). On click, render a panel below (mirroring the reference's "Katin Houser — Pass Yds" block): a stat `<select>` (the player's available markets from their `sims` dists), a line `<input>`, the `histogramSVG` of that market's `dist` (from `nfl_player_sim.dist`), and `probVsLineRow`.
- [ ] **Step 2:** Reuse Task 2's renderer + handler (player dists are `{kind:"pmf",pmf}`; `distValueAt` already handles them). Default line = dist median.
- [ ] **Step 3:** A Close button hides the panel. CSS `.player-dist`.
- [ ] **Step 4:** Bump `app.js?v=`; `node --check`; browser-verify: click a player → distribution shows, changing stat/line updates bars + Under/At/Over. Verify a player with only `anytime_td` (binary) renders sensibly (2-bar).

---

## Self-Review

- **Coverage:** team totals + spread/total/team histogram (T1 backend + T3), boxscore medians (T4), per-player distribution (T2 renderer + T5) — the four reference components. ✓
- **Placeholders:** dist formats and pmf helpers concrete; the only honest gap is Rush TD / attempts not being stored — T4 shows only stored stats and labels honestly (not a placeholder, a scope note).
- **Type consistency:** dist JSON shape identical across backend (T1) and front-end accessors (T2); `distValueAt`/`distProbs`/`histogramSVG` used by both T3 and T5.

## Deployment (user, in order)

1. Run `db/migration_nfl_sim_dist.sql`.
2. Merge to main; regenerate sim (`gh workflow run generate-sim-nfl.yml`) so upcoming games get the dists.
3. Redeploy `app.js` + `*.html` (bumped cache-buster).
