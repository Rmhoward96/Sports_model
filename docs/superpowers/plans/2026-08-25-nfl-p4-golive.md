# NFL P4 — Go-live wire-up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the P1–P3 NFL models into the live serving path — a producer writing NFL `game_predictions`/`prop_predictions`, the DB `sport` column + `generate_board` NFL pass, NFL odds capture + `game_pk` matcher, an NFL results provider for grading, automation workflows, and front-end logos — so NFL game lines + the 7 prop markets go live for Week 1 (~Sept 10, 2026).

**Architecture:** New `scripts/generate_nfl.py` (analog of `generate_sim.py`) loads the committed `rating.json`/`gameline.json`/`props.json` into their configs and assembles per-game predictions from P1 (Elo/SoS) → P2 (gameline dists) and the P3 prop engine, writing `sport='nfl'` rows. `generate_board.py`/`grade_results.py` (P0-parameterized) get NFL boardable + an NFL results provider. `ingest_odds.py` gains an NFL event→`game_pk` matching path via `espn` + `matcher`. Logic is unit-tested offline (fixtures/injected); DB writes + live odds/results run in GitHub Actions. MLB is untouched (additive/parameterized).

**Tech Stack:** Python 3.12 (uv), pandas/numpy, httpx (ESPN/Odds API), pytest. Consumes P1–P3 `src/sportsmodel/nfl/*` + committed `assets/nfl/*.json`.

**Spec:** `docs/superpowers/specs/2026-08-25-nfl-p4-golive.md` (parent `docs/superpowers/specs/2026-08-23-nfl-model.md`).

## Global Constraints

- **MLB behavior is unchanged.** All existing tests pass; NFL is additive/parameterized. The default `--sport mlb` path is byte-for-byte unchanged.
- **Producer writes** `sport='nfl'`, `model_version='nfl-elo-v1'` (game) / `'nfl-props-v1'` (props), keyed by ESPN `game_pk` (event id).
- **Configs loaded, not defaulted:** `GameLineConfig` from `assets/nfl/gameline.json` (never bare defaults — `sigma_total` fitted 13.58 vs default 10.0); `PropConfig` from `assets/nfl/props.json` (round-trip test exists); `EloConfig`/`BlendConfig` from `assets/nfl/rating.json`.
- **`game_pk` = ESPN event id** (~9 digits; no overlap with MLB's ~6-digit StatsAPI). The `sport` column disambiguates reads.
- **Injury rule:** never emit a prop for an OUT/inactive player; a starter-Out bumps the backup's share.
- **DB writes / live odds+results need `DATABASE_URL`/`ODDS_API_KEY` (Actions only).** Unit tests use fixtures/injected data — no live calls in the suite. Live verification (Odds-API NFL names, ESPN box-score shape) is an explicit runtime step (Task 9 validation).
- **Cron:** idempotent, run on every firing (the daily-ingest gate lesson — no brittle exact-hour gate).
- **Branch:** `nfl-p4-golive`. One commit per task (TDD red→green→commit). Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- `db/migration_nfl_sport.sql` — NEW (Task 1). `db/serving_bootstrap.sql` — MODIFY (ALTERs).
- `src/sportsmodel/nfl/espn.py` — MODIFY (parse_schedule adds names). `src/sportsmodel/nfl/matcher.py` — used as-is.
- `scripts/ingest_odds.py` — MODIFY (NFL event→game_pk matching path).
- `src/sportsmodel/nfl/config.py` — NEW (Task 3, config loaders). `src/sportsmodel/db.py` — MODIFY (sport-aware upserts).
- `scripts/generate_nfl.py` — NEW (Task 4). `scripts/generate_board.py` — MODIFY (Task 5).
- `src/sportsmodel/ingest/nfl_results.py` — NEW (Task 6). `scripts/grade_results.py` — MODIFY (Task 7).
- `.github/workflows/generate-nfl.yml` — NEW; `capture-odds.yml`/`grade-results.yml` — MODIFY (Task 8).
- `app.js` (external CappingAlpha site) — MODIFY (Task 9). `docs/superpowers/reports/2026-08-25-nfl-golive-validation.md` — NEW (Task 9).
- Tests: `tests/nfl/test_espn.py`, `test_matcher.py`, `test_config.py`, `test_generate_nfl.py`, `test_generate_board.py`, `test_nfl_results.py`, `test_grade_picks.py`.

---

### Task 1: DB schema — `sport` column

**Files:** Create `db/migration_nfl_sport.sql`; Modify `db/serving_bootstrap.sql`.

**Goal:** Add `sport` to the prediction tables so the board can filter NFL rows. No pytest (SQL deliverable you run in Supabase).

- [ ] **Step 1: Write `db/migration_nfl_sport.sql`** (idempotent)

```sql
-- NFL P4: sport column on the prediction tables (board_picks/picks already have it).
ALTER TABLE game_predictions ADD COLUMN IF NOT EXISTS sport TEXT DEFAULT 'mlb';
ALTER TABLE prop_predictions  ADD COLUMN IF NOT EXISTS sport TEXT DEFAULT 'mlb';
UPDATE game_predictions SET sport = 'mlb' WHERE sport IS NULL;
UPDATE prop_predictions  SET sport = 'mlb' WHERE sport IS NULL;
CREATE INDEX IF NOT EXISTS idx_game_predictions_sport ON game_predictions (sport, game_date);
CREATE INDEX IF NOT EXISTS idx_prop_predictions_sport ON prop_predictions (sport, game_date);
```

- [ ] **Step 2: Add the same `ALTER ... ADD COLUMN IF NOT EXISTS sport` lines to `db/serving_bootstrap.sql`** (so a fresh bootstrap includes the column), next to the `game_predictions`/`prop_predictions` definitions.

- [ ] **Step 3: Verify the SQL parses** (sqlite/`sqlparse` is not required; just confirm the file is well-formed and committed). Note in the commit body that the USER runs `db/migration_nfl_sport.sql` in the Supabase SQL Editor before the first NFL producer run.

- [ ] **Step 4: Commit**

```bash
git add db/migration_nfl_sport.sql db/serving_bootstrap.sql
git commit -m "feat(nfl): sport column migration for game_predictions/prop_predictions"
```

---

### Task 2: ESPN schedule names + Odds→`game_pk` matcher wiring

**Files:** Modify `src/sportsmodel/nfl/espn.py`, `scripts/ingest_odds.py`; Test `tests/nfl/test_espn.py`, `tests/nfl/test_matcher.py`.

**Interfaces:** `espn.parse_schedule` output gains `home_name`/`away_name` (the `displayName`s) so `matcher.match_odds_event(odds_event, espn_games)` (P1) resolves an Odds-API event → ESPN `game_pk`. `ingest_odds` gains an NFL path that maps Odds-API NFL events to `game_pk` via the ESPN schedule + matcher.

- [ ] **Step 1: Write the failing test** (espn names + matcher over parse_schedule output)

```python
# tests/nfl/test_espn.py (add)
def test_parse_schedule_emits_display_names():
    import json, pathlib
    from sportsmodel.nfl.espn import parse_schedule
    fix = json.loads((pathlib.Path(__file__).parent.parent / "fixtures/nfl/espn_scoreboard.json").read_text())
    g = parse_schedule(fix)[0]
    assert g["home_name"] and g["away_name"]        # full display names present
    assert g["game_pk"] == 401671789
```
Update the committed `tests/fixtures/nfl/espn_scoreboard.json` competitors to include `"team": {"abbreviation": "KC", "displayName": "Kansas City Chiefs"}` (and the away team) so the parser can read `displayName`.

```python
# tests/nfl/test_matcher.py (add)
def test_match_over_parse_schedule_output():
    from sportsmodel.nfl.matcher import match_odds_event
    espn_games = [{"game_pk": 401671789, "home_name": "Kansas City Chiefs",
                   "away_name": "Baltimore Ravens", "commence_time": "2024-09-06T00:20Z"}]
    ev = {"home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
          "commence_time": "2024-09-06T00:20:00Z"}
    assert match_odds_event(ev, espn_games) == 401671789
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_espn.py tests/nfl/test_matcher.py -k "display_names or parse_schedule_output" -v` → FAIL.

- [ ] **Step 3: Implement** — in `espn.parse_schedule`, add to each row:
```python
    "home_name": c["home"]["team"].get("displayName"),
    "away_name": c["away"]["team"].get("displayName"),
```
Then in `scripts/ingest_odds.py`, add an NFL matching path: when `cfg.key == "nfl"`, fetch the ESPN schedule (`espn.fetch_schedule`) for the window → `espn_games` (with `home_name`/`away_name`/`game_pk`), and for each Odds-API event resolve `game_pk = matcher.match_odds_event(ev, espn_games)` (skip unmatched, log). MLB keeps its existing team-name+date path (unchanged). Keep the live ESPN/Odds fetch behind the sport branch; the unit tests cover `parse_schedule` + `match_odds_event` only.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_espn.py tests/nfl/test_matcher.py -v` → PASS (existing tests included).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/espn.py scripts/ingest_odds.py tests/nfl/test_espn.py tests/nfl/test_matcher.py tests/fixtures/nfl/espn_scoreboard.json
git commit -m "feat(nfl): ESPN display names + Odds-API->game_pk matcher wiring for NFL odds"
```

---

### Task 3: Config loaders + sport-aware upserts

**Files:** Create `src/sportsmodel/nfl/config.py`; Modify `src/sportsmodel/db.py`; Test `tests/nfl/test_config.py`.

**Interfaces:** `config.load_rating() -> (EloConfig, BlendConfig)`, `config.load_gameline() -> GameLineConfig`, `config.load_props() -> PropConfig` — each reads the committed `assets/nfl/*.json`. `db.upsert_game_predictions`/`upsert_prop_predictions` accept a `sport` key per record (default `'mlb'`), writing the new column.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_config.py
from sportsmodel.nfl.config import load_rating, load_gameline, load_props
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.props import PropConfig

def test_load_rating():
    elo, blend = load_rating()
    assert isinstance(elo, EloConfig) and isinstance(blend, BlendConfig)
    assert elo.k > 0 and 0 <= blend.w_sos <= 1

def test_load_gameline_uses_fitted_sigma_not_default():
    gl = load_gameline()
    assert isinstance(gl, GameLineConfig)
    assert gl.sigma_total != 10.0        # fitted (~13.58), not the illustrative default
    assert gl.offset == 75

def test_load_props_builds_all_seven():
    from sportsmodel.nfl.props import build_prop
    cfg = load_props()
    assert isinstance(cfg, PropConfig)
    vol = {"pass_att": 34.0, "carries": 15.0, "targets": 8.0}
    eff = {"ypa": 7.5, "catch_rate": 0.65, "ypr": 11.0, "ypc": 4.3,
           "pass_td_rate": 0.05, "rec_td_rate": 0.06, "rush_td_rate": 0.03}
    for m in ("pass_yds","reception_yds","rush_yds","rush_reception_yds","receptions","pass_tds","anytime_td"):
        assert "dist" in build_prop(m, vol, eff, cfg)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_config.py -v` → ImportError.

- [ ] **Step 3: Implement `config.py`**

```python
# src/sportsmodel/nfl/config.py
from __future__ import annotations
import json, pathlib
from .elo import EloConfig
from .ratings import BlendConfig
from .gameline import GameLineConfig
from .shrink import ShrinkParams
from .props import PropConfig

_ASSETS = pathlib.Path(__file__).resolve().parents[3] / "assets" / "nfl"

def _load(name: str) -> dict:
    return json.loads((_ASSETS / name).read_text())

def load_rating() -> tuple[EloConfig, BlendConfig]:
    j = _load("rating.json")
    return (EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"], base=j["base"]),
            BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"]))

def load_gameline() -> GameLineConfig:
    j = _load("gameline.json")
    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"],
                          offset=j["offset"], total_max=j["total_max"],
                          w_margin=ShrinkParams(**j["w_margin"]), w_total=ShrinkParams(**j["w_total"]))

def load_props() -> PropConfig:
    j = _load("props.json")
    return PropConfig(sigma=j["sigma"], nb_var_mult=j["nb_var_mult"], mean_mult=j["mean_mult"])
```
(Confirm the actual field names in each JSON before finalizing — `assets/nfl/{rating,gameline,props}.json` are committed; read them.) Then in `db.py`, extend `upsert_game_predictions` and `upsert_prop_predictions` to include a `sport` column: add `sport` to the INSERT column list and read `r.get("sport", "mlb")` from each record, and to the ON CONFLICT/REPLACE as appropriate. **MLB records that omit `sport` default to `'mlb'` — byte-for-byte unchanged.**

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_config.py -v` + full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/config.py src/sportsmodel/db.py tests/nfl/test_config.py
git commit -m "feat(nfl): config loaders (rating/gameline/props json) + sport-aware prediction upserts"
```

---

### Task 4: Producer — `scripts/generate_nfl.py`

**Files:** Create `scripts/generate_nfl.py`; Test `tests/nfl/test_generate_nfl.py`.

**Interfaces:** `build_game_row(game, ratings_ctx, gl_cfg) -> dict` (a `game_predictions`-shaped row with `sport='nfl'`); `build_prop_rows(game, universe, usage_shares, eff, team_volume, props_cfg) -> list[dict]` (prop rows, sport='nfl', never for an inactive, backup-bumped); `main()` assembles the slate and upserts. The assembly funcs are pure (injected inputs) and unit-tested; `main()`'s ESPN/DB I/O is the thin live wrapper.

- [ ] **Step 1: Write the failing test** (injected — no live calls)

```python
# tests/nfl/test_generate_nfl.py
import importlib.util, pathlib
_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "generate_nfl.py"
_s = importlib.util.spec_from_file_location("generate_nfl", _p)
gn = importlib.util.module_from_spec(_s); _s.loader.exec_module(gn)
from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.props import PropConfig

def test_build_game_row_is_serving_shaped():
    game = {"game_pk": 401, "game_date": "2026-09-10", "commence_time": "2026-09-11T00:20Z",
            "home_team": "KC", "away_team": "BAL", "home_name": "Kansas City Chiefs",
            "away_name": "Baltimore Ravens"}
    ctx = {"model_margin": 3.0, "model_total": 45.0, "market": {"spread_line": None, "total_line": None},
           "week": 1}
    row = gn.build_game_row(game, ctx, GameLineConfig())
    assert row["sport"] == "nfl" and row["model_version"] == "nfl-elo-v1"
    assert row["game_pk"] == 401 and row["margin_dist"]["kind"] == "margin"
    assert row["total_dist"]["kind"] == "pmf" and 0 <= row["home_win_prob"] <= 1

def test_build_prop_rows_excludes_inactive_and_tags_sport():
    game = {"game_pk": 401, "game_date": "2026-09-10", "home_team": "KC", "away_team": "BAL"}
    universe = [{"player_id": "wr1", "player_name": "WR One", "team": "KC", "position": "WR"}]
    shares = {"wr1": {"target_share": 0.25, "carry_share": 0.0, "pass_att_share": 0.0,
                      "position": "WR", "team": "KC", "player_name": "WR One"}}
    eff = {"wr1": {"ypa":0,"pass_td_rate":0,"catch_rate":0.65,"ypr":11.0,"rec_td_rate":0.06,"ypc":0,"rush_td_rate":0}}
    tv = {"KC": {"pass_att": 34.0, "rush_att": 24.0, "plays": 58.0}}
    rows = gn.build_prop_rows(game, universe, shares, eff, tv, PropConfig())
    assert rows and all(r["sport"] == "nfl" for r in rows)
    assert any(r["market"] == "reception_yds" for r in rows)
    assert all(r["player_id"] == "wr1" for r in rows)   # only the active WR
```

- [ ] **Step 2: Run to verify fail** — module/functions missing.

- [ ] **Step 3: Implement `scripts/generate_nfl.py`.** Real code:
- `build_game_row(game, ctx, gl_cfg)`: `row = gameline.build_gameline(ctx["model_margin"], ctx["model_total"], ctx["market"], ctx["week"], gl_cfg)`; return `{**row, "sport": "nfl", "model_version": "nfl-elo-v1", "game_pk": game["game_pk"], "game_date": game["game_date"], "home_team_name": game["home_name"], "away_team_name": game["away_name"]}` (map to the `game_predictions` columns MLB uses: `pred_home_score`/`pred_away_score`/`pred_total`/`pred_margin`/`home_win_prob`/`total_dist`/`margin_dist` — JSON-encode the dists like generate_sim does).
- `build_prop_rows(game, universe, shares, eff, team_volume, props_cfg)`: for each active player in `universe` with a share, `vol = usage.allocate(shares[pid], team_volume[shares[pid]["team"]])`; for each market in the player's applicable set (QB → pass_yds/pass_tds; skill → reception_yds/receptions/rush_yds/rush_reception_yds/anytime_td as position warrants), `p = props.build_prop(market, vol, eff[pid], props_cfg)`; append a `prop_predictions`-shaped row (`sport='nfl'`, `model_version='nfl-props-v1'`, `game_pk`, `player_id`, `player_name`, `team_name`, `market`, `projected_mean`, `dist` JSON, `line=None`). Skip players not in `universe` (inactive already filtered).
- `main()`: resolve the current NFL season/week; `elo/blend = config.load_rating()`, `gl = config.load_gameline()`, `pc = config.load_props()`; run Elo over committed `schedules.parquet` (P1) for current ratings; season-to-date SRS + points ratings; fit `gamescript` from committed weekly+schedules; `usage`/`efficiency` from the latest committed weekly; ESPN schedule (`espn.fetch_schedule`) + inactives (`espn.fetch_inactives`); `universe.active_universe(...)`; per game compute `model_margin` (P1 `ratings.expected_margin`) + `model_total` (P2 `points.expected_total`), read the latest NFL market line from `odds_snapshot` (or None), `project_team_volume`, `build_game_row` + `build_prop_rows`; **when a starter is OUT, bump the backup** (redistribute the share). Upsert via `db.upsert_game_predictions`/`upsert_prop_predictions` (sport-aware, Task 3). Print `predicted N games, M prop rows`. DB write only if `config.DATABASE_URL` set (mirror generate_sim).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_generate_nfl.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_nfl.py tests/nfl/test_generate_nfl.py
git commit -m "feat(nfl): generate_nfl producer -> game_predictions + prop_predictions (sport=nfl)"
```

---

### Task 5: `generate_board.py` — NFL pass

**Files:** Modify `scripts/generate_board.py`; Test `tests/nfl/test_generate_board.py` (or the existing `tests/test_generate_board.py`).

**Interfaces:** `'nfl'` added to `BOARDABLE_SPORTS` and `PROP_MARKETS_BY_SPORT`; the `game_predictions`/`prop_predictions` reads gain `AND sport = %s`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_board.py (add)
def test_nfl_is_boardable_with_seven_markets():
    assert "nfl" in gb.BOARDABLE_SPORTS
    assert set(gb.PROP_MARKETS_BY_SPORT["nfl"]) == {
        "pass_yds","pass_tds","reception_yds","receptions","rush_yds","rush_reception_yds","anytime_td"}
```

- [ ] **Step 2: Run to verify fail** — FAIL (`nfl` not boardable).

- [ ] **Step 3: Implement** — `BOARDABLE_SPORTS = {"mlb", "nfl"}`; `PROP_MARKETS_BY_SPORT["nfl"] = ("pass_yds","pass_tds","reception_yds","receptions","rush_yds","rush_reception_yds","anytime_td")`. Add `AND sport = %s` to the `game_predictions` read (params `[start, sport]`) and to the `prop_predictions` read (params include `sport`), so a `--sport nfl` run reads only NFL predictions. The board math (best-book price, EV, EV-or-pass, CLV) is unchanged. Do NOT alter the MLB `--sport mlb` behavior (its query now also filters `sport='mlb'`, which the migration guarantees exists on all rows).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_generate_board.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_board.py tests/test_generate_board.py
git commit -m "feat(nfl): make generate_board NFL-boardable (sport-filtered prediction reads)"
```

---

### Task 6: NFL results provider — `nfl_results.py`

**Files:** Create `src/sportsmodel/ingest/nfl_results.py`, `tests/fixtures/nfl/espn_summary.json`; Test `tests/nfl/test_nfl_results.py`.

**Interfaces:** `final_game_pks(start, end) -> set[int]` (ESPN games STATUS_FINAL in the window); `fetch_results(game_pk) -> dict | None` with `home_runs`/`away_runs`-analog keys the grader expects for game lines (`home_score`/`away_score`) PLUS per-player prop actuals resolvable by `_actual_for(market, side, res, player_id)`. Parsers fixture-tested; live fetch is a thin wrapper.

- [ ] **Step 1: Create the fixture** `tests/fixtures/nfl/espn_summary.json` — a trimmed ESPN summary/boxscore for one final game with home/away scores and a couple of players' passing/receiving/rushing/TD stats (enough to test the parse). Keep it minimal but shaped like the real endpoint.

- [ ] **Step 2: Write the failing test**

```python
# tests/nfl/test_nfl_results.py
import json, pathlib
from sportsmodel.ingest.nfl_results import parse_results

def test_parse_results_scores_and_player_actuals():
    fix = json.loads((pathlib.Path(__file__).parent.parent / "fixtures/nfl/espn_summary.json").read_text())
    res = parse_results(fix)
    assert isinstance(res["home_score"], int) and isinstance(res["away_score"], int)
    assert res["final"] is True
    # per-player actuals keyed by player_id -> {market: value}
    pid = next(iter(res["players"]))
    assert set(res["players"][pid]).issuperset(
        {"pass_yds","reception_yds","rush_yds","receptions","pass_tds","anytime_td"})
```

- [ ] **Step 3: Run to verify fail** — ImportError.

- [ ] **Step 4: Implement `nfl_results.py`** — `parse_results(summary)` reads the ESPN summary: final scores (cast int, gate STATUS_FINAL) and, from the boxscore `players` blocks, each athlete's passing_yards / receiving_yards / rushing_yards / receptions / passing_tds and an `anytime_td = 1 if (rush_td+rec_td) >= 1 else 0`, plus `rush_reception_yds = rush+rec`. Return `{"home_score","away_score","final","players": {player_id: {market: value}}}`. `fetch_results(game_pk)` = `httpx.get(summary endpoint) → parse_results`. `final_game_pks(start,end)` = ESPN scoreboard over the window filtered to STATUS_FINAL. Provide a `_actual_for`-compatible accessor OR document the key mapping so `grade_results` (Task 7) can resolve `res["players"][player_id][market]` for props and `res["home_score"]/res["away_score"]` for game lines.

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/nfl/test_nfl_results.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/ingest/nfl_results.py tests/nfl/test_nfl_results.py tests/fixtures/nfl/espn_summary.json
git commit -m "feat(nfl): NFL results provider (ESPN finals + box-score prop actuals)"
```

---

### Task 7: Register NFL provider in `grade_results.py`

**Files:** Modify `scripts/grade_results.py`; Test `tests/test_grade_picks.py`.

**Interfaces:** `RESULTS_PROVIDERS['nfl'] = nfl_results`; the prop-actual resolution (`_actual_for` / the grading loop) handles the NFL provider's per-player `res["players"][pid][market]` shape for the 7 markets (and `res["home_score"]/["away_score"]` for game lines — the shared contract).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grade_picks.py (add)
def test_nfl_provider_registered():
    assert "nfl" in gr.RESULTS_PROVIDERS
    prov = gr.RESULTS_PROVIDERS["nfl"]
    assert hasattr(prov, "final_game_pks") and hasattr(prov, "fetch_results")
```

- [ ] **Step 2: Run to verify fail** — FAIL.

- [ ] **Step 3: Implement** — `from sportsmodel.ingest import nfl_results`; `RESULTS_PROVIDERS = {"mlb": mlb_results, "nfl": nfl_results}`. Ensure the grading loop resolves prop actuals for NFL: for a prop pick with `player_id`, read the actual from `res["players"].get(player_id, {}).get(<market-key>)`; for game-line picks read `res["home_score"]/["away_score"]` (map to the same `_actual_for` path MLB uses — MLB's `res["home_runs"]/["away_runs"]` and NFL's `res["home_score"]/["away_score"]` may need a small per-sport key normalization; do the minimal adapter so `_actual_for`/the margin+total computation is sport-agnostic). The `--sport nfl` run already filters pending picks to `sport='nfl'` (P0). MLB path unchanged.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/ -q` → all PASS (MLB grading unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/grade_results.py tests/test_grade_picks.py
git commit -m "feat(nfl): register NFL results provider + prop-actual resolution in grade_results"
```

---

### Task 8: Workflows

**Files:** Create `.github/workflows/generate-nfl.yml`; Modify `.github/workflows/capture-odds.yml`, `.github/workflows/grade-results.yml`.

**Goal:** Automate the NFL producer + NFL odds capture/board + NFL grading. No pytest (YAML); validate structurally.

- [ ] **Step 1: Write `generate-nfl.yml`** — schedule: a weekly run (e.g. Tuesday ~14:00 UTC, after Monday-night games) + Sunday-morning refresh runs (as inactives/lines firm up), plus `workflow_dispatch`. **Idempotent, run on every firing — NO exact-hour gate** (the daily-ingest lesson). Steps: checkout, uv sync, `uv run python scripts/generate_nfl.py` with `DATABASE_URL`/`ODDS_API_KEY`/`SM_DATA_DIR` env.

- [ ] **Step 2: Add an NFL leg to `capture-odds.yml`** — after the MLB steps, run `uv run python scripts/ingest_odds.py --sport nfl` then `uv run python scripts/generate_board.py --sport nfl` (same `ODDS_API_KEY`/`DATABASE_URL`). (During NFL season these fire alongside MLB; harmless off-season — the NFL slate is empty.)

- [ ] **Step 3: Add an NFL leg to `grade-results.yml`** — run `uv run python scripts/grade_results.py --sport nfl` alongside the MLB grade step.

- [ ] **Step 4: Structural check** — confirm each YAML is well-formed (indentation, `on:`/`jobs:` present, the new steps invoke the right scripts with the right `--sport` args). Note: these run live in Actions; verified end-to-end in Task 9.

- [ ] **Step 5: Commit**

```bash
git add .github/workflows/generate-nfl.yml .github/workflows/capture-odds.yml .github/workflows/grade-results.yml
git commit -m "feat(nfl): workflows — generate-nfl + NFL legs on capture-odds/grade-results"
```

---

### Task 9: Front-end logos + end-to-end validation + go-live

**Files:** Modify `app.js` (external CappingAlpha site, path per the user); Create `docs/superpowers/reports/2026-08-25-nfl-golive-validation.md`.

**Goal:** NFL logos on the site + a live end-to-end validation on a past/current week before flipping Week 1 on. This task has manual/runtime steps (external file + live workflow runs + DB); no pytest.

- [ ] **Step 1: Front-end** — in `app.js`, add an NFL `TEAM_ABBR` map (32 teams) + the ESPN NFL logo path `https://a.espncdn.com/i/teamlogos/nfl/500/{abbr}.png`, mirroring the MLB `logoImg`/`pickLogo` helpers, so NFL `board_picks`/`picks` rows (which carry `sport='nfl'` + team) render logos. The NFL page/nav/market-filters already exist. (This file lives in the separate CappingAlpha folder the user deploys to Cloudflare — the user redeploys after this edit.)

- [ ] **Step 2: Run the migration** (user) — `db/migration_nfl_sport.sql` in the Supabase SQL Editor (Task 1).

- [ ] **Step 3: End-to-end validation** (runtime, via `workflow_dispatch`) — trigger `generate-nfl.yml`, then `capture-odds.yml` (NFL leg), then `grade-results.yml` (NFL leg) for a current/past NFL week; confirm from the logs + Supabase: `game_predictions`/`prop_predictions` written with `sport='nfl'`; `board_picks` NFL rows with best-book prices + EV + EV-or-pass; odds matched by `game_pk` (matcher — verify the live Odds-API NFL team names resolve; add normalization if any don't); grading produces results + CLV on finals. **This is where the Odds-API NFL name strings and the ESPN box-score shape are verified live** — fix any parser/normalization mismatch found.

- [ ] **Step 4: Write the validation report** `docs/superpowers/reports/2026-08-25-nfl-golive-validation.md` — what ran, row counts, any name/box-score normalization applied, sane-check of predictions vs market, and the go-live checklist (workflows enabled for Week 1).

- [ ] **Step 5: Commit**

```bash
git add app.js docs/superpowers/reports/2026-08-25-nfl-golive-validation.md
git commit -m "feat(nfl): front-end logos + go-live validation report"
```

---

## Self-Review

**Spec coverage:** schema (T1), ESPN names + matcher + odds wiring (T2), config loaders + sport-aware upserts (T3), producer (T4), board NFL pass (T5), results provider (T6), grade registration (T7), workflows (T8), front-end + validation + go-live (T9). All spec sections A–H covered.

**Parked seams threaded:** load gameline.json/props.json → T3; ESPN displayName → T2; commence_shift_hours (NFL=0, via SportConfig — already `0` from P0's config) → used in ingest_odds NFL path (T2); nfl_results home_score/away_score contract → T6/T7; real backup-bump → T4. ✓

**Placeholder scan:** T9's live steps (Odds-API names, ESPN box-score shape) are explicit runtime verifications with a fixture-tested parser fallback — a documented deferral to validation, not a blank. T4's `main()` is described with exact module calls; the pure assembly funcs (`build_game_row`/`build_prop_rows`) are given for TDD. No TBD.

**Type consistency:** producer consumes P1–P3 exactly (`EloConfig`/`BlendConfig`/`GameLineConfig`/`PropConfig` from the loaders; `build_gameline`/`build_prop`/`active_universe`/`allocate` signatures); the written rows match the `game_predictions`/`prop_predictions` columns MLB uses + `sport`; `RESULTS_PROVIDERS`/`--sport`/board `sport` filters consistent across T5/T7. The DB upsert `sport` default keeps MLB unchanged.
