# +EV Sub-project 1: Odds Ingestion (Pinnacle + soft books, NFL + CFB) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Pull live NFL + CFB game odds (moneyline/spread/total) from The Odds API — including **Pinnacle** (sharp baseline) and the major soft books — into the existing `odds_snapshot` table, so downstream sub-projects can compute implied probability and EV.

**Architecture:** This is a **restore-and-adapt** of the Odds API stack that commit `45ed03d` ("cut the Odds API") deleted. Recover the deleted modules from git (`45ed03d^`), then adapt: (a) default `regions="us,eu"` so Pinnacle (an EU bookmaker) is captured, (b) add a **CFB** sport config + matcher (mirror NFL), (c) drive game-lines ingestion for NFL **and** CFB into `odds_snapshot`. Props are out of scope for this sub-project (game lines only).

**Tech Stack:** Python 3.12, httpx, tenacity, pytest; The Odds API v4; Supabase Postgres (`odds_snapshot`); GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-09-plus-ev-engine-design.md` (sub-project 1).

## Global Constraints

- **Restore, don't reinvent.** The deleted files are correct and tested; recover them verbatim from `git show 45ed03d^:<path>` and make only the adaptations this plan names. Do not rewrite the Odds API client or matcher from scratch.
- **Pinnacle is an EU bookmaker on The Odds API** — the pull MUST use `regions=us,eu` or Pinnacle is silently absent. The book slug is `pinnacle` (already in `serving/board.py::MAJOR_BOOKS`).
- **Game lines only** (h2h/totals/spreads) for NFL + CFB. No player props in this sub-project — do not restore the prop-fetch paths into the live run (the prop *parsers* may come back with the module, but the ingester must not call per-event prop endpoints).
- **Reuse `odds_snapshot`** (already defined in `db/serving_bootstrap.sql`): columns `game_pk, market, side, player_name, book, line, price, commence_time, captured_at`; game lines use `player_name=''`. Do not create a new odds table.
- Secrets (`ODDS_API_KEY`, `DATABASE_URL`) read only from env in `main()`; never logged.
- Run tests with `PYTHONPATH=src uv run pytest`.
- The prediction model/pages and the decision desk are untouched by this sub-project.

## Reference: what `45ed03d` deleted (recover these)

- `src/sportsmodel/ingest/odds.py` (182 lines) — the Odds API client: `_get`, `fetch_events`, `fetch_game_odds(cfg, regions="us")`, `parse_game_odds(events, game_lookup, captured_at)`, `parse_commence`, `resolved_game_date`, `_row`, `GAME_MARKETS=["h2h","totals","spreads"]`, `PROP_MARKET_MAP`, lazy `_mlb()`.
- `src/sportsmodel/sports.py` (52 lines) — `SportConfig(key, odds_sport, game_markets, prop_market_map, commence_shift_hours)` + `SPORTS` (has `mlb`, `nfl`; **CFB missing**) + `get(sport)`.
- `src/sportsmodel/nfl/matcher.py` (17 lines) — `match_odds_event(ev, espn_games)` → game_pk by home/away display name + date.
- `scripts/ingest_odds.py` (240 lines) — the runner (NFL path uses the matcher; MLB path uses the date-key join).
- `tests/test_odds.py`, `tests/test_sports.py`, `tests/nfl/test_matcher.py`.
- `.github/workflows/capture-odds.yml`, `.github/workflows/debug-odds.yml`, `scripts/debug_odds.py`.
- `src/sportsmodel/db.py` — `upsert_odds_snapshot(...)` (removed in the 116-line db.py cut).

Recover file content with: `git show 45ed03d^:<path>`.

---

### Task 1: Restore the Odds API stack (as-is) + verify

**Files (restore verbatim from `45ed03d^`):**
- Create: `src/sportsmodel/ingest/odds.py`
- Create: `src/sportsmodel/sports.py`
- Create: `src/sportsmodel/nfl/matcher.py`
- Create: `tests/test_odds.py`, `tests/test_sports.py`, `tests/nfl/test_matcher.py`
- Modify: `src/sportsmodel/db.py` — restore ONLY `upsert_odds_snapshot` (and any tiny helper it needs) from `45ed03d^:src/sportsmodel/db.py`; do not restore the board/graded-picks helpers.

- [ ] **Step 1: Recover the modules**

For each of `src/sportsmodel/ingest/odds.py`, `src/sportsmodel/sports.py`, `src/sportsmodel/nfl/matcher.py`, `tests/test_odds.py`, `tests/test_sports.py`, `tests/nfl/test_matcher.py`, run `git show 45ed03d^:<path>` and write the exact content to that path. (These are self-contained; recover as-is.)

- [ ] **Step 2: Restore `upsert_odds_snapshot` into db.py**

Inspect `git show 45ed03d^:src/sportsmodel/db.py` and copy back ONLY the `upsert_odds_snapshot` function (plus any private helper it references that no longer exists) into the current `src/sportsmodel/db.py`. Match the current file's import style and connection helpers (`get_postgres` / `get_duckdb`). Do NOT restore `upsert_board_picks` / `insert_new_picks` / `update_graded_picks`.

- [ ] **Step 3: Ensure `odds_snapshot` DDL is available for tests**

`odds_snapshot` is defined in `db/serving_bootstrap.sql`. If any restored test needs a local DuckDB table, ensure it creates it the same way the original test did (recover the test's own fixture). Do not add a new migration file — the table already exists in Supabase.

- [ ] **Step 4: Run the restored tests**

Run: `PYTHONPATH=src uv run pytest tests/test_odds.py tests/test_sports.py tests/nfl/test_matcher.py -v`
Expected: PASS. If a test fails only because it references an MLB prop path or a removed helper, note it in the report; do NOT weaken an assertion — fix the restore.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ingest/odds.py src/sportsmodel/sports.py src/sportsmodel/nfl/matcher.py src/sportsmodel/db.py tests/test_odds.py tests/test_sports.py tests/nfl/test_matcher.py
git commit -m "feat(+ev): restore Odds API stack (client, sports config, matcher, db upsert)"
```

---

### Task 2: Pinnacle regions + CFB sport config + CFB matcher

**Files:**
- Modify: `src/sportsmodel/ingest/odds.py` (regions default)
- Modify: `src/sportsmodel/sports.py` (add CFB)
- Create: `src/sportsmodel/cfb/matcher.py`
- Create: `tests/cfb/test_matcher.py`
- Modify: `tests/test_sports.py` (assert CFB config)

**Interfaces:**
- Produces: `sports.get("cfb")` → `SportConfig(key="cfb", odds_sport="americanfootball_ncaaf", game_markets=GAME_MARKETS, prop_market_map={}, commence_shift_hours=8)`; `cfb.matcher.match_odds_event(ev, espn_games)`; `odds.fetch_game_odds(cfg, regions="us,eu")` by default.

- [ ] **Step 1: Default `regions="us,eu"` for Pinnacle**

In `src/sportsmodel/ingest/odds.py`, change `fetch_game_odds`'s signature default from `regions: str = "us"` to `regions: str = "us,eu"` and update its docstring to note Pinnacle is an EU bookmaker so both regions are pulled. (Leave `fetch_event_props`/`fetch_events` as-is; they're unused this sub-project.)

- [ ] **Step 2: Add CFB to the sport registry**

In `src/sportsmodel/sports.py`, add to `SPORTS`:

```python
    "cfb": SportConfig(
        key="cfb",
        odds_sport="americanfootball_ncaaf",
        game_markets=GAME_MARKETS,
        prop_market_map={},
        commence_shift_hours=8,
    ),
```

- [ ] **Step 2b: Extend `tests/test_sports.py`**

Add a test asserting `get("cfb").odds_sport == "americanfootball_ncaaf"`, `get("cfb").game_markets == GAME_MARKETS`, and `get("cfb").commence_shift_hours == 8`. Run it.

- [ ] **Step 3: CFB matcher (mirror NFL)**

Read the restored `src/sportsmodel/nfl/matcher.py` to see the exact `match_odds_event(ev, espn_games)` contract (it matches an Odds API event to a game_pk by home/away display name + resolved date against a list of ESPN games with keys like `game_pk`, `home_name`/`home_team`, `away_name`/`away_team`, `commence_time`). Create `src/sportsmodel/cfb/matcher.py` with the same function and matching logic. If the NFL matcher is fully sport-agnostic (only reads event/ESPN-game fields, no NFL-specific normalization), the CFB matcher may simply re-export it — but only if there is genuinely nothing NFL-specific; otherwise copy and adapt name-normalization to CFB (`cfb.teams.normalize`). State which you did and why in the report.

- [ ] **Step 4: `tests/cfb/test_matcher.py`**

Mirror `tests/nfl/test_matcher.py` with CFB team names (e.g. an Odds API event "Georgia Bulldogs" @ "Alabama Crimson Tide" resolving to a game_pk given a matching ESPN game, and a no-match case returning None). Run it.

- [ ] **Step 5: Run the affected tests**

Run: `PYTHONPATH=src uv run pytest tests/test_sports.py tests/nfl/test_matcher.py tests/cfb/test_matcher.py tests/test_odds.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/ingest/odds.py src/sportsmodel/sports.py src/sportsmodel/cfb/matcher.py tests/cfb/test_matcher.py tests/test_sports.py
git commit -m "feat(+ev): Pinnacle regions (us,eu) + CFB odds config & matcher"
```

---

### Task 3: NFL + CFB game-lines ingester (game lines only)

**Files:**
- Create: `scripts/ingest_odds.py`
- Create/Modify: `tests/test_ingest_odds.py` (a focused test of the sport-loop assembly, pure where possible)

**Interfaces:**
- Consumes: `sports.get("nfl"|"cfb")`, `odds.fetch_game_odds(cfg, regions="us,eu")`, `odds.parse_game_odds`, `nfl.matcher`/`cfb.matcher`, `nfl.espn`/`cfb.espn` schedule fetchers, `db.upsert_odds_snapshot`.

**Background:** Recover `git show 45ed03d^:scripts/ingest_odds.py` as the base. It already had an NFL game-lines + props flow (`_run_nfl`) using an ESPN-schedule matcher and an MLB date-key flow in `main()`. For this sub-project:
- Keep the **game-lines** path; **remove the props path** (no `fetch_event_props` / `INGEST_PROPS` / prop windows) — game lines only.
- Run for **NFL and CFB** (loop over both sports), not MLB. Each sport: fetch ESPN games for the current window (NFL via `nfl.espn.resolve_target_week` + `fetch_schedule`; CFB via the CFB ESPN schedule fetcher — inspect `src/sportsmodel/cfb/espn.py` for the equivalent of `fetch_schedule`/current-week, and use it; if CFB lacks a week resolver, fetch the current scoreboard/slate the same way `generate_cfb.py` does — read that script to reuse its slate-fetch), match Odds events → game_pk via the sport's matcher, `parse_game_odds` → `odds_snapshot` via `upsert_odds_snapshot`.
- `fetch_game_odds` is called with `regions="us,eu"` (the new default; passing nothing is fine).
- Print per-sport: events fetched, matched, unmatched, rows written, and `odds.last_requests_remaining` (credit budget).

- [ ] **Step 1: Recover and inspect the old ingester**

`git show 45ed03d^:scripts/ingest_odds.py` — read it fully. Also read the current `src/sportsmodel/cfb/espn.py` and `scripts/generate_cfb.py` to find how the CFB upcoming slate (game_pk = ESPN event id, home/away display names, commence_time) is fetched, so the CFB branch matches predictions' game identity.

- [ ] **Step 2: Write the NFL+CFB game-lines ingester**

Create `scripts/ingest_odds.py` that, in `main()`:
1. reads `ODDS_API_KEY` (fail fast if unset) and `captured_at = datetime.now(timezone.utc).isoformat()`,
2. for `sport in ("nfl", "cfb")`: `cfg = sports.get(sport)`; fetch that sport's ESPN games for the window; `events = odds.fetch_game_odds(cfg)`; build `game_lookup[(home, away, resolved_game_date(commence))] = game_pk` using the sport's matcher over the ESPN games; `rows = odds.parse_game_odds(events, game_lookup, captured_at)`; `upsert_odds_snapshot(rows)`; print counts + `odds.last_requests_remaining`,
3. no props anywhere.

Keep a pure helper (e.g. `build_game_lookup(events, espn_games, matcher_fn)`) so it's unit-testable without network.

- [ ] **Step 3: Test the pure assembly**

Create `tests/test_ingest_odds.py` that imports the pure helper(s) via importlib and asserts: given fake Odds events + fake ESPN games + a matcher, `build_game_lookup` maps matched events and drops unmatched; and that `parse_game_odds` over a fixture with a `pinnacle` bookmaker produces moneyline/spread/total rows with `book="pinnacle"`. (Reuse the `parse_game_odds` fixture style from the restored `tests/test_odds.py`.)

- [ ] **Step 4: Run**

Run: `PYTHONPATH=src uv run pytest tests/test_ingest_odds.py tests/test_odds.py -v`
Expected: PASS.

- [ ] **Step 5: Byte-compile the script**

Run: `PYTHONPATH=src uv run python -c "import ast; ast.parse(open('scripts/ingest_odds.py').read()); print('ok')"`

- [ ] **Step 6: Commit**

```bash
git add scripts/ingest_odds.py tests/test_ingest_odds.py
git commit -m "feat(+ev): NFL+CFB game-lines odds ingester (game lines only)"
```

---

### Task 4: Capture workflow (CI)

**Files:**
- Create: `.github/workflows/capture-odds.yml`

- [ ] **Step 1: Recover and adapt the workflow**

Base it on `git show 45ed03d^:.github/workflows/capture-odds.yml`. Adapt: run `scripts/ingest_odds.py` for NFL+CFB game lines; env `ODDS_API_KEY` + `DATABASE_URL` from secrets; a `workflow_dispatch` trigger (a schedule/cron is optional — include the original cron if present, but `workflow_dispatch` must work for on-demand capture and near-close snapshots). Remove any prop-specific env (`INGEST_PROPS`, `PROP_WINDOW_MIN`). Update the top comment to say NFL+CFB game lines with Pinnacle (regions us,eu).

- [ ] **Step 2: Validate YAML**

Run: `uv run --with pyyaml python -c "import yaml; yaml.safe_load(open('.github/workflows/capture-odds.yml'))" && echo ok`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/capture-odds.yml
git commit -m "ci(+ev): capture-odds workflow for NFL+CFB game lines"
```

---

## Post-plan (controller / user)

- **User:** add `ODDS_API_KEY` (The Odds API) as a GitHub Actions secret before the first live run.
- After merge, a live `capture-odds` run should show Pinnacle rows in `odds_snapshot` (`SELECT DISTINCT book FROM odds_snapshot` includes `pinnacle`) — the acceptance check for this sub-project.
- No Supabase migration needed (`odds_snapshot` already exists).
