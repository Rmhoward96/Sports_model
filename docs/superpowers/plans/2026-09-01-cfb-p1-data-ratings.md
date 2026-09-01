# CFB P1 — Data + Team Registry + Ratings — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Feed CFB (FBS) data into the existing league-agnostic NFL Elo/SRS engine to produce a backtested, OOS-validated `assets/cfb/rating.json`.

**Architecture:** ESPN `college-football` API → `cfb/teams.py` (ESPN-id registry, non-FBS→one `FCS` anchor) + `cfb/espn.py` (schedule/score adapter) → committed `assets/cfb/schedules.parquet` → reuse `nfl.elo`/`nfl.srs`/`nfl.ratings` with CFB-tuned config → `assets/cfb/rating.json`.

**Tech Stack:** Python 3.12, ESPN public API (no key), pandas/pyarrow, reuse `sportsmodel.nfl.{elo,srs,ratings}`.

**Spec:** docs/superpowers/specs/2026-09-01-cfb-p1-data-ratings.md

## Global Constraints

- CFB-only; **reuse `nfl.elo/srs/ratings` read-only** — never modify NFL or MLB code. Additive.
- Canonical team key = `str(ESPN team id)`; non-FBS opponents collapse to exactly one `"FCS"` id. `game_pk` = ESPN event id (int).
- Committed, reviewable assets under `assets/cfb/`.
- P1 is ESPN-internal — no odds, no odds↔ESPN name matching (P3).
- Ratings must beat naive baselines (home-always, prior-season, naive-margin) OOS on margin MAE before acceptance (Task 4 is the gate).
- Use `PYTHONPATH=src uv run --no-sync` for all test/script runs (space in repo path).

---

### Task 1: CFB team registry + FBS set

**Files:**
- Create: `src/sportsmodel/cfb/__init__.py` (empty), `src/sportsmodel/cfb/teams.py`, `assets/cfb/fbs_teams.json`
- Test: `tests/cfb/__init__.py` (empty), `tests/cfb/test_teams.py`

**Interfaces:**
- Produces: `FCS = "FCS"`; `load_fbs_ids() -> set[str]` (from `assets/cfb/fbs_teams.json`, keys are ESPN ids as strings); `normalize(espn_team_id) -> str` (`str(id)` if in the FBS set else `FCS`).

- [ ] **Step 1:** build `assets/cfb/fbs_teams.json` — fetch ESPN FBS teams once: `GET https://site.api.espn.com/apis/site/v2/sports/football/college-football/teams?groups=80&limit=400` → for each `sports[0].leagues[0].teams[].team` record `{id: displayName}`. Write sorted `{espn_id_str: display_name}` (~130 teams). (Write a throwaway snippet to produce it; the committed json is the deliverable, not the fetch code.)
- [ ] **Step 2: failing test**
```python
from sportsmodel.cfb.teams import normalize, load_fbs_ids, FCS

def test_fbs_passthrough_and_fcs_collapse():
    fbs = load_fbs_ids()
    assert len(fbs) > 120                    # ~130 FBS programs
    some_fbs = next(iter(fbs))
    assert normalize(int(some_fbs)) == some_fbs
    assert normalize("99999999") == FCS      # unknown id -> FCS anchor
    assert normalize(some_fbs) == some_fbs    # str or int in
```
- [ ] **Step 3:** run → fail. **Step 4:** implement `teams.py` (load json via `config.PROJECT_ROOT`, `{}`-safe; `normalize` coerces to `str` then membership-checks). **Step 5:** run → pass. **Step 6:** commit (code + fbs_teams.json).

### Task 2: ESPN college-football adapter

**Files:**
- Create: `src/sportsmodel/cfb/espn.py`
- Test: `tests/cfb/test_espn.py`, fixture `tests/fixtures/cfb/espn_scoreboard.json` (a trimmed real payload: 1 FBS-FBS final + 1 FBS-FCS game)

**Interfaces:**
- Consumes: `cfb.teams.normalize`.
- Produces: `parse_schedule(payload) -> list[dict]` rows `{game_pk:int, home_team:str, away_team:str, home_name:str, away_name:str, home_score:int|None, away_score:int|None, commence_time:str, status:str, week:int, season:int}` (teams normalized; FCS side → "FCS"). `parse_final(event) -> dict|None` (`{home_score,away_score,final:True}`, gated on `status.type.name=="STATUS_FINAL"`). `fetch_schedule(season, week, season_type=2) -> list[dict]` (calls `.../scoreboard?dates=<season>&seasontype=<n>&week=<n>&groups=80`, then `parse_schedule`). Mirror `nfl/espn.py` structure.

- [ ] **Step 1:** capture a small real payload into the fixture (one FBS-FBS final + one FBS-FCS). **Step 2: failing tests** — `parse_schedule` normalizes teams (FCS side becomes "FCS"), types game_pk/scores, sets week/season; `parse_final` gates on STATUS_FINAL. **Step 3:** run → fail. **Step 4:** implement mirroring `nfl/espn.py` (httpx, `groups=80` for FBS). **Step 5:** run → pass. **Step 6:** commit.

### Task 3: Historical schedules builder + asset

**Files:**
- Create: `scripts/build_cfb_schedules.py`, `assets/cfb/schedules.parquet` (generated, committed)

**Interfaces:**
- Consumes: `cfb.espn.fetch_schedule`. Pulls `--seasons 2015..2025`, regular season weeks 1–16, threaded with retry; writes a DataFrame with columns `season, week, home_team, away_team, home_score, away_score, game_type` (all finals, `game_type="REG"`, teams normalized) to parquet. Prints game count + a summary (n games, n distinct FBS teams, % games involving FCS).

- [ ] **Step 1:** write the script (argparse `--seasons`, ThreadPoolExecutor per (season,week), skip non-final; analog of `build_umpire_factors.py`). **Step 2:** run `PYTHONPATH=src uv run --no-sync python scripts/build_cfb_schedules.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025`; sanity-check (~800–900 FBS games/season, ~130 FBS teams, FCS share ~5–8%). **Step 3:** commit script + parquet.

### Task 4: Ratings backtest + tuning (THE GATE)

**Files:**
- Create: `scripts/backtest_cfb_ratings.py`, `assets/cfb/rating.json`
- (reads) `assets/cfb/schedules.parquet`; reuses `nfl.elo`, `nfl.srs`, `nfl.ratings`

**Interfaces:**
- Leak-free walk-forward: for each test season, ratings use only prior seasons + season-to-date games (pre-game Elo, season-to-date SRS). Tune `nfl.elo.EloConfig` (k, hfa, carryover) + `nfl.ratings.BlendConfig` (w_sos, srs_min_games) on a TRAIN span by margin MAE; validate OOS; write tuned params to `assets/cfb/rating.json` (same shape `nfl/rating.json` uses). Compare vs baselines: home-always (margin = hfa), prior-season rating, naive-margin (0).

- [ ] **Step 1:** write the backtest (mirror `scripts/backtest_nfl_elo.py`; CFB param grid with **wider hfa search ~2–4** and **lower carryover search ~0.2–0.5**). **Step 2:** run it; **record OOS margin MAE vs each baseline in the ledger — this is the ship-gate.** Accept only if it beats naive-margin and prior-season OOS. **Step 3:** commit script + `rating.json` + a short findings note under `docs/superpowers/reports/`.
