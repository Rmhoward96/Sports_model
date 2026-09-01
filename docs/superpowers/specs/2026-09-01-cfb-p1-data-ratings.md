# CFB P1 — Data + Team Registry + Ratings (Design Spec)

**Date:** 2026-09-01
**Status:** approved to build (game-lines-only CFB v1; ESPN-only data; phased P1→P3)
**Goal:** Stand up College Football (FBS) game-line ratings by feeding CFB data into the existing, league-agnostic NFL Elo/SRS engine — producing a backtested, OOS-validated `assets/cfb/rating.json` that P2 turns into game-line distributions.

## Why this is small

The NFL rating math is already league-agnostic: `nfl.elo.run_elo(schedule_df, cfg)`, `nfl.srs.compute_srs(games)`, `nfl.ratings.expected_margin(...)` operate on generic game data + config with no NFL-specific assumptions. **CFB reuses them read-only.** P1 therefore adds only a CFB *data layer* + retuned config — not a new rating engine.

## Architecture

ESPN's `college-football` API (same shape as `nfl/espn.py`, different path) supplies schedules + scores with consistent team ids/names, so P1 is entirely ESPN-internal — **no cross-source name matching yet** (that's a P3 concern when odds are joined). `game_pk = ESPN event id` (int), exactly like NFL.

## Components

1. **`src/sportsmodel/cfb/teams.py`** — canonical CFB team registry.
   - Canonical key = **ESPN team id** (int, stable). `normalize(espn_team) -> team_id`.
   - **FCS handling:** any non-FBS opponent maps to a single `FCS` pseudo-team id. ESPN marks FBS vs FCS via the team's group/classification; teams not in the FBS set collapse to `FCS`. This anchors FBS ratings (all the FBS-beats-FCS results pull against one aggregate weak team) without pretending to rate 100+ individual FCS programs.
   - SSOT for the FBS team set (built from ESPN's teams endpoint; committed as a small json/table so the set is deterministic and reviewable).

2. **`src/sportsmodel/cfb/espn.py`** — ESPN college-football adapter (mirror of `nfl/espn.py`).
   - `fetch_schedule(season, week, season_type=2)`, `parse_schedule(payload)` → rows `{game_pk, home_team_id, away_team_id, home_name, away_name, home_score, away_score, commence_time, status, week, season}`.
   - `fetch_final(event_id)` / `parse_final` (STATUS_FINAL gate), `fetch_current_week`/`parse_current_week` — same contracts as NFL.
   - FBS/FCS tagging via `teams.normalize` so both sides carry a team id (FCS → the pseudo id).

3. **`src/sportsmodel/cfb/data.py` + committed `assets/cfb/schedules.parquet`** — historical CFB schedules/results for building Elo. Columns match what `nfl.elo.run_elo`/`nfl.srs.compute_srs` consume (`season, week, home_team, away_team, home_score, away_score, game_type`). Built by a refresh script from ESPN. **History window 2015–2025** (CFB roster turnover is high — deep history adds little; recent seasons dominate).

4. **Reuse (read-only):** `nfl.elo` (margin-adjusted Elo + carryover), `nfl.srs` (retrodictive SRS), `nfl.ratings` (Elo×SRS margin blend). CFB feeds them its schedule_df + a **CFB-tuned** `EloConfig`/`BlendConfig`. (If importing `nfl.*` from `cfb.*` ever feels wrong, a later cleanup extracts a shared `football/` module — out of scope for v1.)

5. **`scripts/build_cfb_schedules.py`** — offline builder: pull ESPN `college-football` regular-season finals for `--seasons 2015..2025`, normalize teams, write `assets/cfb/schedules.parquet`. Threaded, cached (analog of `build_umpire_factors.py`).

6. **`scripts/backtest_cfb_ratings.py`** — leak-free walk-forward (pre-game Elo + season-to-date SRS), tunes `EloConfig` (k, hfa, carryover) + `BlendConfig` (w_sos, srs_min_games) on a TRAIN span by margin MAE, validates OOS, and writes `assets/cfb/rating.json`. Beats baselines (home-always, prior-season, naive-margin) on margin MAE before it's accepted.

## Key CFB-specific expectations (vs NFL)

- **Home-field advantage is larger** (~2.5–3.5 pts vs NFL ~2) → `hfa` retunes upward.
- **Carryover is lower** (more roster turnover / transfer portal) → `carryover` retunes downward.
- **Bigger blowouts / talent gaps** → the MOV multiplier already damps runaway margins; verify it holds on CFB's wider score distribution.
- **FCS games** stay in the Elo fit (they inform FBS ratings via the FCS anchor) but will be **flagged for board exclusion in P3** (blowouts, no edge, often no line).

## Global constraints

- CFB-only; **reuse `nfl.elo/srs/ratings` read-only** — do not modify NFL or MLB code. Additive.
- Canonical team key = ESPN team id; `game_pk` = ESPN event id (int).
- Non-FBS opponents collapse to exactly one `FCS` pseudo-team — never rate individual FCS programs.
- Committed, reviewable assets: `assets/cfb/schedules.parquet`, `assets/cfb/rating.json`.
- P1 is ESPN-internal only — no odds, no odds↔ESPN name matching (deferred to P3).
- The ratings must beat naive baselines OOS on margin MAE before P2 builds on them.

## Out of scope (P1)

- Game-line distributions / market shrinkage (P2), odds capture + grading + board + frontend + workflow (P3).
- Player props (out of scope for the entire CFB v1).
- collegefootballdata.com (CFBD) enrichment — ESPN-only for v1.
- Odds↔ESPN team-name matching for 130+ teams (P3, the main new engineering).
