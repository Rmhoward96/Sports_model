# +EV Sub-project 2: Feature Pipeline (Elo, L10, rest, EPA/PPA, SOS, SOV) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Produce a per-game **feature table** for NFL and CFB — pre-game Elo, last-10 win %, rest, offensive/defensive EPA (NFL) / PPA (CFB), strength of schedule, strength of victory — for **historical games** (targets attached, to fit the true-prob model in sub-project 3) and for **upcoming games** (to score them). No model or odds here; just features.

**Architecture:** Reuse the existing Elo engine (`nfl.elo.run_elo` → per-game pre-game `elo_home`/`elo_away`; CFB reuses it) and schedules. Add the two feature sources the repo lacks: **NFL EPA** via `nfl_data_py.import_pbp_data` (aggregate `epa` by `posteam`/`defteam`), and **CFB PPA** via a new CFBD `/ppa/teams` fetch. A pure schedule-feature module computes L10/SOS/SOV/rest. A `build_features.py --sport` assembler joins them into `assets/<sport>/features.parquet` and exposes an upcoming-game feature builder.

**Tech Stack:** Python 3.12, pandas, `nfl_data_py` (already a dep), httpx/CFBD, pytest.

**Spec:** `docs/superpowers/specs/2026-09-09-plus-ev-engine-design.md` (sub-project 2).

## Global Constraints

- **Reuse, don't rebuild:** Elo comes from `nfl.elo.run_elo(schedule).games` (columns include `elo_home`, `elo_away`, `e_home` — pre-game) and `.final` (current ratings for upcoming games). CFB reuses the same engine (see `scripts/generate_cfb.py`). Do not write a new Elo.
- **Pure core, thin IO:** every feature computation is a pure function over already-loaded DataFrames/dicts (unit-tested with tiny fixtures); network/file IO lives only in `main()`/loaders.
- **Sport asymmetry — rest:** NFL `schedules.parquet` has `gameday`/`gametime` → exact rest **days**. CFB `schedules.parquet` has only `season, week, home_team, away_team, home_score, away_score, game_type` (**no dates**) → rest is a **week-gap / bye proxy** (weeks since the team's previous game; bye flag when the gap > 1). Document this per sport; never fabricate CFB dates.
- **No leakage:** every feature for a game must use only information available **before** that game (prior games only). Elo is already pre-game; L10/SOS/SOV must exclude the current game and all later ones.
- **Team identity:** teams are keyed as they are in each sport's `schedules.parquet` (NFL abbreviations, CFB display names) throughout the feature table; the assembler does not need ESPN displayName mapping (that join happens later, in sub-project 4, against `predictions_current`).
- Secrets (`CFBD_API_KEY`) only from env in `main()`. Run tests with `PYTHONPATH=src uv run pytest`.
- Nothing here touches the prediction pages, the prediction model, the desk, or odds ingestion.

---

### Task 1: CFB team off/def PPA (CFBD `/ppa/teams`)

**Files:**
- Modify: `src/sportsmodel/cfb/cfbd.py` (add `parse_team_ppa`; the module already has `_get`)
- Create: `tests/cfb/test_cfbd_ppa.py`
- Create: `tests/fixtures/cfb/cfbd_ppa.json`

**Interfaces:**
- Produces: `cfbd.parse_team_ppa(payload) -> dict[str, dict]` → `{team: {"off_ppa": float|None, "def_ppa": float|None}}` from CFBD `/ppa/teams`. Live fetch via existing `_get("/ppa/teams", api_key, {"year": season, "excludeGarbageTime": "true"})`.

- [ ] **Step 1: Fixture** — Create `tests/fixtures/cfb/cfbd_ppa.json` mirroring CFBD `/ppa/teams` shape: a list of `{"team": "...", "conference": "...", "offense": {"overall": 0.23, "passing": ..., "rushing": ...}, "defense": {"overall": 0.05, ...}}`, with ~3 teams and one row missing `offense`/`defense` (null-guard case).

- [ ] **Step 2: Parser** — Add `parse_team_ppa(payload)` to `cfbd.py`: for each row, `off = row.get("offense") or {}`, `def_ = row.get("defense") or {}`, emit `{team: {"off_ppa": off.get("overall"), "def_ppa": def_.get("overall")}}`; skip rows with null `team`. Mirror the null-hardening style of the other `parse_*` funcs (see `parse_returning`).

- [ ] **Step 3: Test** — `tests/cfb/test_cfbd_ppa.py`: assert the 3 teams parse with correct off/def values, the missing-blocks row yields `{"off_ppa": None, "def_ppa": None}`, and null-team rows are skipped. Run: `PYTHONPATH=src uv run pytest tests/cfb/test_cfbd_ppa.py -v`.

- [ ] **Step 4: Commit** — `git commit -m "feat(+ev): CFBD team off/def PPA parser"`

---

### Task 2: NFL team off/def EPA (nflverse pbp)

**Files:**
- Create: `src/sportsmodel/nfl/epa.py`
- Create: `tests/nfl/test_epa.py`

**Interfaces:**
- Produces: `epa.team_epa_from_pbp(pbp: pd.DataFrame) -> dict[str, dict]` (PURE) → `{team: {"off_epa": mean epa/play on offense, "def_epa": mean epa/play allowed on defense}}`. And a thin loader `epa.load_team_epa(seasons: list[int]) -> dict[str, dict]` that calls `nfl_data_py.import_pbp_data(seasons)` and returns `team_epa_from_pbp` of it (season-to-date over the given seasons). Team keys normalized via `nfl.teams.normalize_team`.

- [ ] **Step 1: Pure aggregator + test-first**

Write `tests/nfl/test_epa.py` with a tiny fixture DataFrame of pbp rows — columns `posteam`, `defteam`, `epa` (a handful of plays across 2 teams, including a NaN `epa` row and a null-team row to drop) — and assert `team_epa_from_pbp` returns each team's `off_epa` = mean `epa` over rows where it is `posteam` (NaN/na dropped), `def_epa` = mean `epa` over rows where it is `defteam`. Run it; expect FAIL (module missing).

- [ ] **Step 2: Implement `team_epa_from_pbp`**

In `src/sportsmodel/nfl/epa.py`: drop rows with null `posteam`/`defteam` or NaN `epa` appropriately (offense aggregation drops null `posteam`/NaN `epa`; defense drops null `defteam`/NaN `epa`); `off = pbp.groupby("posteam")["epa"].mean()`, `def_ = pbp.groupby("defteam")["epa"].mean()`; assemble `{team: {"off_epa": off.get(team), "def_epa": def_.get(team)}}` over the union of teams; normalize team keys with `normalize_team`. Keep it pure (DataFrame in, dict out). Run the test → PASS.

- [ ] **Step 3: Thin loader**

Add `load_team_epa(seasons)`: `import nfl_data_py as nfl; pbp = nfl.import_pbp_data(seasons); return team_epa_from_pbp(pbp[["posteam","defteam","epa"]])`. (Do not unit-test the network loader; the pure aggregator is the tested unit. A module-level docstring notes `import_pbp_data` is large — callers pass the minimal season span.)

- [ ] **Step 4: Commit** — `git commit -m "feat(+ev): NFL team off/def EPA from nflverse pbp"`

---

### Task 3: Schedule-derived features (L10, SOS, SOV, rest) — pure

**Files:**
- Create: `src/sportsmodel/features/__init__.py`, `src/sportsmodel/features/schedule.py`
- Create: `tests/features/__init__.py`, `tests/features/test_schedule.py`

**Interfaces:**
- Produces (all PURE, no IO): given a schedule DataFrame (with pre-game Elo already attached per game — see below) sorted by (season, week), for a target team up to a given game, compute:
  - `last10_win_pct(prior_games, team) -> float|None` — win % over that team's last ≤10 completed prior games.
  - `sos(prior_games, team) -> float|None` — mean **pre-game opponent Elo** the team has faced to date.
  - `sov(prior_games, team) -> float|None` — mean pre-game Elo of opponents the team has **beaten** to date.
  - `rest_days_nfl(prior_games, team, this_gameday) -> int|None` — days since the team's previous game (from `gameday`); None if no prior game.
  - `rest_weeks_cfb(prior_games, team, this_week) -> dict` — `{"weeks_since_prev": int|None, "off_bye": bool}` from the `week` gap (gap>1 → off_bye True).

**Background for the implementer:** The opponent-Elo needed by SOS/SOV is the opponent's **pre-game** Elo *in the game they played this team* — take it from the Elo-augmented schedule (`run_elo(...).games` gives `elo_home`/`elo_away` per game; when team is home its opponent's pre-game Elo is `elo_away`, and vice-versa). So these functions operate on the Elo-augmented per-game rows, not raw schedules. Define the exact row schema you consume in a docstring (`season, week, home_team, away_team, home_score, away_score, elo_home, elo_away`, plus `gameday` for NFL).

- [ ] **Step 1: Write failing tests** — `tests/features/test_schedule.py` with a small hand-built Elo-augmented schedule (one team across ~4 games: some wins/losses, known opponent pre-game Elos, known dates/weeks). Assert each function's output on that fixture (L10 over <10 games; SOS = mean opp elo; SOV = mean elo of beaten opponents only; NFL rest = day diff; CFB rest weeks + bye when a week is skipped). Run → FAIL.

- [ ] **Step 2: Implement `features/schedule.py`** to pass. Enforce **no leakage** (only rows strictly before the target game). Run → PASS.

- [ ] **Step 3: Commit** — `git commit -m "feat(+ev): schedule-derived features (L10, SOS, SOV, rest)"`

---

### Task 4: Feature assembler → `assets/<sport>/features.parquet`

**Files:**
- Create: `scripts/build_features.py`
- Create: `tests/test_build_features.py`

**Interfaces:**
- Produces: `build_features.assemble(sport, elo_games_df, epa_or_ppa_by_team, upcoming=False) -> pd.DataFrame` (PURE over already-loaded inputs) — one row per game with, for each feature, a `home_*` and `away_*` column **and** a `*_diff` (home − away): `elo`, `last10`, `sos`, `sov`, `rest` (NFL days / CFB weeks+bye), `off_epa`/`def_epa` (NFL) or `off_ppa`/`def_ppa` (CFB). Historical rows also carry targets `margin` (home_score−away_score) and `total` (home_score+away_score) and `game_pk`/identity. `main(--sport)` loads the sport's schedule, runs `run_elo` to attach pre-game Elos, loads EPA (`nfl.epa.load_team_epa`) or PPA (CFBD `/ppa/teams` via the new parser), calls `assemble`, and writes `assets/<sport>/features.parquet`.

- [ ] **Step 1: Failing test** — `tests/test_build_features.py`: build a tiny Elo-augmented schedule + a fake `epa_by_team` dict, call `assemble("nfl", ...)`, assert the row for a game has correct `home_elo`/`away_elo`/`elo_diff`, the L10/SOS/SOV/rest columns wired from Task 3, the EPA columns joined by team, and `margin`/`total` targets. Include one game whose team is missing from the EPA dict → EPA columns are NaN/None (graceful), row still present. Run → FAIL.

- [ ] **Step 2: Implement `assemble` + `main`.** `assemble` joins per-game: pull `home_elo`/`away_elo` from the Elo-augmented row; compute L10/SOS/SOV/rest for each side via Task 3 over the prior-games slice; join EPA/PPA by team; compute diffs; attach targets for completed games. `main(--sport)`: for `nfl` load schedule + `epa.load_team_epa`; for `cfb` load schedule + CFBD PPA (needs `CFBD_API_KEY`); write parquet. Print row count + feature columns. Run test → PASS.

- [ ] **Step 3: Byte-compile** `scripts/build_features.py` (`ast.parse`). Run full suite `PYTHONPATH=src uv run pytest -q`.

- [ ] **Step 4: Commit** — `git commit -m "feat(+ev): per-game feature assembler -> assets/<sport>/features.parquet"`

---

## Output contract (consumed by sub-project 3)

`assets/<sport>/features.parquet`: one row per historical game with `home_*`/`away_*`/`*_diff` for {elo, last10, sos, sov, rest, off/def EPA-or-PPA} + `margin`/`total` targets + game identity. An `assemble(..., upcoming=True)` path produces the same feature columns (no targets) for the current slate, using `.final` Elo and latest EPA/PPA. Sub-project 3 fits margin/total models on this.

## Post-plan / notes

- CFB rest is a bye/week-gap proxy (no dates in CFB schedules) — a future enrichment (CFBD/ESPN game dates) could upgrade it; parked, non-blocking.
- `import_pbp_data` pulls large parquet; the assembler passes only the needed season span. Historical fitting depth (how many seasons) is settled in sub-project 3.
- No new DB table in this sub-project — features are a parquet asset (fitting input). Serving upcoming features to the page is wired in sub-project 4/5.
