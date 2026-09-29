# Matchup Grades, Team History and Power Rankings (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show matchup grades (unit vs unit, A–F), team history (L5/L10/L20/season, streaks, splits) and power rankings for NFL and CFB, built from one leak-free team data layer and a daily job.

**Architecture:** A new package `sportsmodel.context` with pure modules — `game_log` (one row per team per game with closing lines), `history` (rolling records/streaks/splits as of a game), `units` (opponent-adjusted pass/run unit ratings incl. success and explosiveness), `matchup` (grades), `power` (ratings + rankings) — fed by nflverse (NFL) and CFB assets + a new CFBD advanced-stats pull. `scripts/build_team_context.py` writes four tables daily; the CappingAlpha site renders them.

**Tech Stack:** Python 3.12, pandas, numpy, nflverse, CFBD API, Supabase Postgres, GitHub Actions, CappingAlpha static JS.

**Spec:** `docs/superpowers/specs/2026-09-29-matchup-history-rankings-design.md`

## Global Constraints

- Both sports (NFL, CFB). Every value for a game uses only games that kicked off before it.
- Team line convention: the team's closing spread from its own perspective, + = favored by that many points (NFL `spread_line` is home-favored-positive; CFB `market_spread` is home-margin, home-favored-positive). A team covers iff `margin > team_line`, pushes iff equal. O/U over iff `total_points > closing_total`.
- Windows L5, L10, L20 span seasons; "season" = season-to-date. Games without a closing line count for SU only.
- Unit ratings are opponent-adjusted relative to league average with early-season blend `w_current = games / (games + 3)` toward the previous season's final value.
- Matchup weights: 0.6 EPA, 0.25 success rate, 0.15 explosiveness; overall = pass/run weighted by the offense's pass rate. Grades: percentiles vs all team-games of the previous three complete seasons (frozen per season): A ≥ 90, B 70–90, C 30–70, D 10–30, F < 10; "early" if < 3 games played.
- Power rating = expected margin vs an average team on a neutral field. CFB: ratings model (Elo + SRS blend). NFL: unit-rating points (Σ EPA/play adj × plays/game, offense minus defense).
- Do NOT change `sportsmodel.nfl.unit_efficiency` (it feeds the live v2 model); compute extra metrics in `sportsmodel.context.units`.
- Site labels matchup grades and history "descriptive — not a pick".
- The user runs all DB migrations; commit/push/merge only when asked. End every commit with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. `.venv/bin/python -m pytest -q` green after every task.

---

### Task 1: CFBD advanced game stats pull

**Files:** Create `scripts/build_cfb_advanced.py`, `.github/workflows/build-cfb-advanced.yml`; Test `tests/scripts/test_build_cfb_advanced.py`.

**Interfaces:**
- `parse_advanced(payload: list[dict]) -> pd.DataFrame` (PURE): CFBD `/stats/game/advanced?year=Y&seasonType=regular` → one row per team-game: `season, week, game_id (CFBD), team (ESPN id via cfbd_to_espn), opponent, off_plays, off_ppa, off_success, off_explosiveness, off_pass_ppa, off_pass_success, off_pass_explosiveness, off_rush_ppa, off_rush_success, off_rush_explosiveness, def_*` (same fields for defense), plus `off_pass_plays`, `off_rush_plays` where given. Unmapped teams dropped and counted.
- CLI `--seasons` (default 2015..current) writes `assets/cfb/advanced_games.parquet` and prints per-season coverage (share of FBS-vs-FBS schedule games with both sides present). `--merge` refreshes only the given seasons.
- Workflow: `workflow_dispatch` input `seasons` + a weekly cron (Mon 12:00 UTC) for the current season with `--merge`; commits the parquet (push with `git pull --rebase` retry to avoid racing other data commits).
- Verify the real payload's field names before coding (CFBD docs / a saved sample in tests — no key locally: write the parser to the documented schema: `offense.{plays, ppa, successRate, explosiveness, passingPlays{ppa,successRate,explosiveness}, rushingPlays{…}}`, `defense.{…}`).

- [ ] Steps: failing tests (fixture with 2 games incl. an unmapped team, missing sub-objects → NaN) → implement → full suite → commit `feat(cfb): CFBD advanced game stats`.

---

### Task 2: Team game log

**Files:** Create `src/sportsmodel/context/__init__.py`, `src/sportsmodel/context/game_log.py`; Test `tests/context/test_game_log.py`.

**Interfaces:**
- `nfl_game_log(schedules: pd.DataFrame) -> pd.DataFrame` — from nflverse schedules (REG + POST; `location` Neutral → neutral), two rows per game.
- `cfb_game_log(schedules, lines, live_close: pd.DataFrame | None) -> pd.DataFrame` — CFB assets; `live_close` = current-season closing consensus from `odds_snapshot` (median across books of the last capture before kickoff, spread in home-margin convention, total) used where `lines` lacks the game.
- `live_closing_consensus(odds_rows: pd.DataFrame) -> pd.DataFrame` (PURE): from `odds_snapshot` rows (game_pk, market, side, line, price, book, captured_at, commence_time) → per game_pk `close_spread_home` (home-margin), `close_total`.
- Columns (both): `sport, season, week, game_key, kickoff (UTC), date_et, team, opponent, venue ∈ {home, away, neutral}, pf, pa, margin, team_line, total_line, role ∈ {fav, dog, pick}, su ∈ {W, L, T}, ats ∈ {W, L, P, None}, ou ∈ {O, U, P, None}` (+ `game_pk` where known). Unplayed games are included with outcomes None (for upcoming lookups).

- [ ] Steps: failing tests (convention: home favored −7 book line → home team_line +7, away −7; cover/push/over logic; neutral; missing line → ats None; live consensus median of last captures) → implement → full suite → commit `feat(context): team game log`.

---

### Task 3: History

**Files:** Create `src/sportsmodel/context/history.py`; Test `tests/context/test_history.py`.

**Interfaces:**
- `team_history_asof(log: pd.DataFrame, team: str, asof: pd.Timestamp) -> dict` — using only that team's games with `kickoff < asof` and a result: for `L5, L10, L20, season` → `{su: "W-L[-T]", ats: "W-L-P", ou: "O-U-P", avg_margin, avg_cover, avg_total_vs_line, n}`; `streaks` → `{su: "W3", ats: "L2", ou: "O4"}` (current run; plus "U6 of 7"-style when ≥ 6 of the last 7 share an outcome); `splits` (season) → home, away, fav, dog each `{su, ats, n}`.
- `history_for_games(log, games: pd.DataFrame) -> pd.DataFrame` — one row per (game, side) for upcoming games (JSON-able dicts in columns `windows`, `streaks`, `splits`).
- Leak test: adding a game at/after `asof` never changes the result.

- [ ] Steps: failing tests (hand-computed records on a 12-game fixture spanning 2 seasons; pushes; missing lines; streak rules; splits; leak test) → implement → full suite → commit `feat(context): team history`.

---

### Task 4: Unit ratings and matchup grades

**Files:** Create `src/sportsmodel/context/units.py`, `src/sportsmodel/context/matchup.py`; Test `tests/context/test_units.py`, `tests/context/test_matchup.py`.

**Interfaces:**
- `nfl_unit_games(pbp) -> pd.DataFrame` — per team-game offense: pass/run EPA per play, success rate, explosive share (pass ≥ 20 yds, run ≥ 10 yds; scrambles are dropbacks), plays per game by unit, pass rate (dropbacks / plays).
- `cfb_unit_games(advanced) -> pd.DataFrame` — the same columns from Task 1's parquet (PPA ≈ EPA; CFBD explosiveness is a magnitude, so for CFB the explosive component uses CFBD `explosiveness` standardized within season — documented).
- `unit_ratings_asof(unit_games, season, week) -> pd.DataFrame` — per team: opponent-adjusted (single-pass, like `efficiency.adjusted_efficiency`) offense and defense values for every metric, relative to league mean of the window, blended with last season's final values by `w = g / (g + 3)`; `games` column.
- `matchup.side_scores(off_team_row, def_team_row) -> dict` — pass/run/overall raw scores per the Global Constraints weights.
- `matchup.grade_table(history_scores: pd.DataFrame) -> dict` — percentile cutoffs from the previous three complete seasons' per-game scores (computed with ratings as of each game).
- `matchup.grades_for_games(games, ratings, cutoffs) -> pd.DataFrame` — per (game, side): overall/pass/run scores + letters + `early`.
- CFB: FCS opponents (no advanced stats) → grade None.

- [ ] Steps: failing tests (a strong pass O vs weak pass D grades high both ways; symmetric run case; blend weight at g = 0, 3, 9; cutoffs from prior seasons only; leak test; FCS → None) → implement → full suite → commit `feat(context): unit ratings and matchup grades`.

---

### Task 5: Power ratings and rankings

**Files:** Create `src/sportsmodel/context/power.py`; Test `tests/context/test_power.py`.

**Interfaces:**
- `cfb_power(schedule_df, elo_cfg, blend_cfg, asof) -> pd.DataFrame` — per FBS team: rating = `ratings.expected_margin` vs a synthetic average FBS team (league-mean Elo and SRS 0) on a neutral field (hfa removed), using pre-`asof` state from `sportsmodel.cfb.walkforward` / `run_elo` + `compute_srs`.
- `nfl_power(unit_ratings, unit_games) -> pd.DataFrame` — rating = Σ over pass/run of (offense EPA adj × team plays/game in that unit) − (defense EPA allowed adj × opponent-average plays in that unit).
- `rankings(power_df, prev_power_df, unit_ratings, log) -> pd.DataFrame` — rank, rating, prev_rank, move, off/def unit ranks (pass off, run off, pass def, run def), SOS = mean rating of opponents played, SU and ATS record.

- [ ] Steps: failing tests (a synthetic league where one team is +10 better ranks first with rating ≈ 10; neutral field removes hfa; move computed vs previous week; SOS; leak test) → implement → full suite → commit `feat(context): power ratings and rankings`.

---

### Task 6: Daily job + storage

**Files:** Create `scripts/build_team_context.py`, `db/migration_team_context.sql`, `.github/workflows/build-team-context.yml`; Modify `src/sportsmodel/db.py`; Test `tests/scripts/test_build_team_context.py`, `tests/test_workflows_team_context.py`.

**Interfaces:**
- Migration: `team_game_log(sport, season, week, game_key, team, …)` PK `(sport, game_key, team)`; `team_history(sport, game_pk, side, team, windows jsonb, streaks jsonb, splits jsonb, computed_at)` PK `(sport, game_pk, side)`; `matchup_grades(sport, game_pk, side, team, overall, pass, run, overall_score, pass_score, run_score, early, units jsonb, computed_at)` PK `(sport, game_pk, side)`; `power_rankings(sport, season, week, team, rank, rating, prev_rank, move, units jsonb, sos, su, ats, computed_at)` PK `(sport, season, week, team)`; view `power_rankings_current` (latest week per sport); RLS public read + anon grants.
- `build_team_context.py --sport {nfl,cfb,all}`: loads sources (NFL: nflverse schedules + pbp current and prior 3 seasons; CFB: assets + `odds_snapshot` closing consensus + advanced parquet), computes game log, history + grades for upcoming games (next 8 days), rankings for the upcoming week, upserts; `--dry-run` prints counts and a sample without DB writes.
- Workflow: daily 13:00 UTC + dispatch; runs after data refreshes; secrets DATABASE_URL (+ CFBD key only if refreshing advanced stats — the separate Task 1 workflow owns that).

- [ ] Steps: failing tests (dry-run on fixtures produces all four frames; upserts called with the right keys; missing advanced parquet → CFB grades skipped with a warning, rest continues) → implement → full suite → commit `feat(context): daily team-context job and tables`.

---

### Task 7: Site (CappingAlpha, external)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`, `*.html` (cache key → `20260930a`), new `rankings.html` (or `?page=rankings&sport=`, following the existing page pattern).

- Game page (NFL + CFB): **Matchup** section (per side: overall grade chip A–F + pass/run grades with unit bars, "early" tag) and **History** section (L5/L10/L20/season table for both teams, streak chips, home/away & fav/dog splits, power rank + rating); both labelled "descriptive — not a pick"; hidden when data missing.
- **Power Rankings** page per sport + nav link: rank, team (logo), rating, ▲/▼ move, unit ranks, SOS, record, ATS; sortable columns; CFB conference filter.
- Verify: snapshot originals, `node --check app.js`, a node harness over the new renderers (empty/one/many; early tag; missing grades for FCS), cache bump, diff file for review.

- [ ] Steps: snapshot → implement → harness → cache bump → diff.

---

### Task 8: Controller — data pull, dry run, rollout

- [ ] Dispatch `build-cfb-advanced` (2015→current) on the branch; pull; check coverage.
- [ ] `build_team_context.py --sport all --dry-run` on real data; spot-check a few NFL and CFB teams' history against known records, and the top-10 rankings for sanity.
- [ ] Report to the user; on their go-ahead: merge (feat/cfb-profit + this branch), user runs `db/migration_team_context.sql`, dispatch `build-team-context`, verify tables, user redeploys the site.
