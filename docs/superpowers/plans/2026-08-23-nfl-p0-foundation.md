# NFL P0 — Foundation (API spike + multi-sport pipeline) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the NFL data sources (nflverse / ESPN / The Odds API) and make the existing MLB pipeline (`ingest_odds`, `generate_board`, `grade_results`) sport-parameterized — with MLB behavior byte-for-byte unchanged — so NFL (and later NBA) can plug in.

**Architecture:** Introduce a `SportConfig` registry that captures everything currently hardcoded to MLB (Odds-API sport key, game/prop market maps, the commence→date shift, prop-market membership, calibration key prefix, results provider). Thread it through the three producers. MLB is the default config; the existing 140-test suite is the regression guard. A leading spike commits a findings doc that the P1 (NFL data) plan depends on.

**Tech Stack:** Python 3.12 (uv), httpx (Odds API), `nfl_data_py` (nflverse), pytest.

**Spec:** `docs/superpowers/specs/2026-08-23-nfl-model.md`

## Global Constraints

- **MLB behavior is unchanged.** Every existing test in `tests/` must still pass after each task; the parameterization defaults to the MLB config.
- **Sport keys:** Odds API — MLB `baseball_mlb`, NFL `americanfootball_nfl`.
- **NFL prop market keys (Odds API):** `player_pass_yds`, `player_pass_tds`, `player_reception_yds`, `player_receptions`, `player_rush_yds`, `player_rush_reception_yds`, `player_anytime_td`. NFL game markets: `h2h`, `spreads`, `totals` (same as MLB).
- **This plan does NOT build the NFL model or NFL data ingest** — only the API spike + the sport-parameterization seam. NFL schedule/nflverse ingest, the game_pk matcher, and the model are P1+ (separate plans, informed by Task 1's findings).
- **Branch:** `nfl-p0-foundation`. One commit per task (TDD red→green→commit).

---

## File Structure

- `src/sportsmodel/sports.py` — NEW. `SportConfig` dataclass + `SPORTS` registry (`mlb`, `nfl`), holding the per-sport values now hardcoded in the producers.
- `src/sportsmodel/ingest/odds.py` — MODIFY. Replace module-level `SPORT`/`GAME_MARKETS`/`PROP_MARKET_MAP` usage with a passed-in `SportConfig`; keep MLB as default.
- `scripts/generate_board.py` — MODIFY. Take `--sport` (default `mlb`); pull prop markets, calibration keys, and the `sport` tag from `SportConfig`.
- `scripts/grade_results.py` — MODIFY. Results fetch via a per-sport provider (MLB = existing `mlb_results`); `grade_pick`/`_actual_for` already generic.
- `docs/superpowers/reports/2026-08-23-nfl-data-spike.md` — NEW (Task 1 output).
- Tests: `tests/test_sports.py`, plus additions to `tests/test_odds.py`.

---

### Task 1: External-data verification spike

**Files:**
- Create: `docs/superpowers/reports/2026-08-23-nfl-data-spike.md`

**Goal:** Confirm the exact shapes P1 depends on, so later tasks aren't guesses. Investigation only — no production code kept.

- [ ] **Step 1: Verify `nfl_data_py` is installable and inspect columns**

```bash
cd "/Users/ryan/Desktop/Sports Model"
uv add nfl_data_py
uv run python - <<'PY'
import nfl_data_py as nfl
sched = nfl.import_schedules([2024])
wk = nfl.import_weekly_data([2024], columns=None)
inj = nfl.import_injuries([2024])
rost = nfl.import_seasonal_rosters([2024]) if hasattr(nfl,'import_seasonal_rosters') else nfl.import_rosters([2024])
for name, df in [("schedules",sched),("weekly",wk),("injuries",inj),("rosters",rost)]:
    print(f"\n== {name} ({len(df)} rows) ==")
    print(list(df.columns))
    print(df.head(2).to_dict("records"))
PY
```
Record: the exact column names for game results (home/away team, scores, game_id, week, gameday, kickoff), for weekly player stats (player_id, player_name, team, passing_yards, passing_tds, attempts, targets, receptions, receiving_yards, receiving_tds, carries, rushing_yards, rushing_tds), for injuries (report status), and rosters (player_id ↔ name ↔ team ↔ position ↔ depth).

- [ ] **Step 2: Verify the ESPN scoreboard/schedule endpoint**

```bash
uv run python - <<'PY'
import httpx
r = httpx.get("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard", timeout=20).json()
ev = r["events"][0]
print("keys:", list(ev.keys()))
print("id:", ev["id"], "date:", ev["date"], "name:", ev["name"])
c = ev["competitions"][0]["competitors"]
print("teams:", [(t["homeAway"], t["team"]["displayName"], t.get("score")) for t in c])
print("status:", ev["status"]["type"]["name"])
PY
```
Record: the ESPN event `id` (candidate NFL `game_pk`), date, team display names (must match what The Odds API returns for the matcher), final-score fields, and status/"STATUS_FINAL".

- [ ] **Step 3: Verify The Odds API NFL market keys**

```bash
uv run python - <<'PY'
import os, httpx
k=os.environ["ODDS_API_KEY"]
base="https://api.the-odds-api.com/v4"
evs=httpx.get(f"{base}/sports/americanfootball_nfl/events",params={"apiKey":k}).json()
print("events:",len(evs), evs[0] if evs else None)
if evs:
  eid=evs[0]["id"]
  for mkt in ["player_pass_yds","player_pass_tds","player_reception_yds","player_receptions","player_rush_yds","player_rush_reception_yds","player_anytime_td"]:
    try:
      d=httpx.get(f"{base}/sports/americanfootball_nfl/events/{eid}/odds",params={"apiKey":k,"regions":"us","markets":mkt,"oddsFormat":"american"}).json()
      books=d.get("bookmakers",[])
      print(mkt,"->",len(books),"books", books[0]["markets"][0]["outcomes"][:2] if books else "none")
    except Exception as e: print(mkt,"ERR",e)
PY
```
Record: which of the seven prop keys actually return outcomes (in preseason some may be empty — note that), the team-name strings The Odds API uses for NFL (compare to ESPN's), and the player-name format in prop outcomes.

- [ ] **Step 4: Write the findings doc** capturing all three sources' exact fields, the chosen **NFL `game_pk` scheme** (recommend: the ESPN event id, an integer), whether ESPN and Odds-API team names match (and any normalization needed for the matcher), and any preseason data gaps. This doc is the input to the P1 plan.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-08-23-nfl-data-spike.md pyproject.toml uv.lock
git commit -m "spike(nfl): verify nflverse/ESPN/Odds-API shapes; add nfl_data_py"
```

---

### Task 2: `SportConfig` registry

**Files:**
- Create: `src/sportsmodel/sports.py`
- Test: `tests/test_sports.py`

**Interfaces:**
- Produces: `SportConfig` dataclass with fields `key: str` (our sport tag, `"mlb"`/`"nfl"`), `odds_sport: str` (Odds-API key), `game_markets: list[str]`, `prop_market_map: dict[str, str]` (our-name → Odds-API-key), `commence_shift_hours: int`, and `get(sport: str) -> SportConfig` via a `SPORTS` dict. Values for MLB copied verbatim from `ingest/odds.py` (`baseball_mlb`, `["h2h","totals","spreads"]`, the existing `PROP_MARKET_MAP`, shift 10). NFL uses `americanfootball_nfl`, the same game markets, the seven prop keys from Global Constraints, and shift 0 (NFL kicks off Sun/Mon/Thu; no cross-midnight-UTC remap like MLB — confirmed acceptable for date resolution; the matcher keys on the schedule, not a shifted date).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sports.py
from sportsmodel.sports import get, SPORTS

def test_mlb_config_matches_legacy_constants():
    from sportsmodel.ingest import odds
    m = get("mlb")
    assert m.odds_sport == "baseball_mlb"
    assert m.game_markets == ["h2h", "totals", "spreads"]
    assert m.prop_market_map == odds.PROP_MARKET_MAP
    assert m.commence_shift_hours == 10

def test_nfl_config_present_with_seven_prop_markets():
    n = get("nfl")
    assert n.odds_sport == "americanfootball_nfl"
    assert set(n.prop_market_map.values()) == {
        "player_pass_yds", "player_pass_tds", "player_reception_yds",
        "player_receptions", "player_rush_yds", "player_rush_reception_yds",
        "player_anytime_td"}

def test_unknown_sport_raises():
    import pytest
    with pytest.raises(KeyError):
        get("cricket")
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_sports.py -v` → ImportError.

- [ ] **Step 3: Implement `sports.py`** — the dataclass + `SPORTS = {"mlb": SportConfig(...), "nfl": SportConfig(...)}` + `get()`. Import `PROP_MARKET_MAP` from `ingest.odds` for the MLB config so there is one source of truth. Map NFL our-names (`pass_yds`, `pass_tds`, `reception_yds`, `receptions`, `rush_yds`, `rush_reception_yds`, `anytime_td`) → the seven Odds-API keys.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_sports.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sports.py tests/test_sports.py
git commit -m "feat(sports): SportConfig registry (mlb + nfl) for multi-sport pipeline"
```

---

### Task 3: Parameterize `ingest/odds.py` by `SportConfig`

**Files:**
- Modify: `src/sportsmodel/ingest/odds.py`
- Test: `tests/test_odds.py`

**Interfaces:**
- Consumes: `sports.get`.
- Produces: `fetch_game_odds(cfg, regions="us")`, `fetch_events(cfg)`, `fetch_event_props(cfg, event_id, markets, regions="us")` now take a `SportConfig` and build the URL from `cfg.odds_sport` / `cfg.game_markets`. A module-level `_MLB = get("mlb")` keeps a zero-arg default path so existing callers/tests work: give each function a `cfg=None` default that falls back to `_MLB`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_odds.py (add)
def test_fetch_functions_use_sport_config(monkeypatch):
    from sportsmodel.ingest import odds
    from sportsmodel.sports import get
    calls = {}
    monkeypatch.setattr(odds, "_get", lambda path, params: calls.setdefault("last", (path, params)) or [])
    odds.fetch_game_odds(get("nfl"))
    assert calls["last"][0] == "/sports/americanfootball_nfl/odds"
    odds.fetch_game_odds()  # default MLB
    assert calls["last"][0] == "/sports/baseball_mlb/odds"
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_odds.py -k sport_config -v` → FAIL (signature).

- [ ] **Step 3: Implement** — add `cfg=None` params, resolve `cfg = cfg or _MLB`, build paths/markets from `cfg`. Leave the MLB constants (`SPORT`, `GAME_MARKETS`, `PROP_MARKET_MAP`) in place (the config references them) so nothing else breaks.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_odds.py -v` → all PASS (existing MLB tests included).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ingest/odds.py tests/test_odds.py
git commit -m "feat(odds): sport-parameterize fetch_* via SportConfig (MLB default)"
```

---

### Task 4: Parameterize `generate_board.py` by sport

**Files:**
- Modify: `scripts/generate_board.py`
- Test: `tests/test_generate_board.py`

**Interfaces:**
- Consumes: `sports.get`.
- Produces: `main()` accepts `--sport` (default `mlb`); `_enrich` tags rows with the passed sport instead of the hardcoded `"mlb"`; `PROP_MARKETS` and the calibration keys are read from the sport config / a per-sport constant. `build_rows(game, prop_preds, odds, cals, sport="mlb")` gains a `sport` arg that flows to `_enrich`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generate_board.py (add)
def test_build_rows_tags_sport():
    game = _game()
    rows = gb.build_rows(game, [], {
        ("moneyline","home",""): {None: [("draftkings", -120)]},
        ("moneyline","away",""): {None: [("fanduel", 110)]},
    }, ((0.0,1.0),(0.0,1.0)), sport="nfl")
    assert rows and all(r["sport"] == "nfl" for r in rows)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_generate_board.py -k tags_sport -v` → FAIL (`build_rows` has no `sport` arg).

- [ ] **Step 3: Implement** — add `sport="mlb"` to `build_rows` and `_enrich`; `main()` gets `--sport` and passes it through; the PROP_MARKETS list and the `game_predictions`/`prop_predictions` model_version filter come from a small per-sport mapping (mlb = current values; nfl values are placeholders wired in P4 when NFL predictions exist — leave `PROP_MARKETS` selection keyed on the sport arg with the MLB set as today). No behavior change for the default MLB path.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_generate_board.py -v` → all PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_board.py tests/test_generate_board.py
git commit -m "feat(board): sport-parameterize generate_board (--sport, sport-tagged rows)"
```

---

### Task 5: Parameterize `grade_results.py` results source

**Files:**
- Modify: `scripts/grade_results.py`
- Test: `tests/test_grade_picks.py`

**Interfaces:**
- Consumes: `sports.get`.
- Produces: a `RESULTS_PROVIDERS: dict[str, module-like]` seam — `mlb` → the existing `mlb_results` (with `.final_game_pks(start,end)` and `.fetch_results(game_pk)`); `main()` selects the provider by a `--sport` arg (default `mlb`). `grade_pick`/`_actual_for` are unchanged (already generic over margin/total/stat). NFL provider is registered in P1; here only the seam + MLB wiring, so MLB grading is byte-for-byte the same.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grade_picks.py (add)
def test_results_provider_registry_has_mlb():
    assert "mlb" in gr.RESULTS_PROVIDERS
    prov = gr.RESULTS_PROVIDERS["mlb"]
    assert hasattr(prov, "final_game_pks") and hasattr(prov, "fetch_results")
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_grade_picks.py -k provider_registry -v` → FAIL.

- [ ] **Step 3: Implement** — `RESULTS_PROVIDERS = {"mlb": mlb_results}`; add `--sport` (default `mlb`) to `main()`; replace the direct `mlb_results.final_game_pks(...)`/`mlb_results.fetch_results(...)` calls with `prov = RESULTS_PROVIDERS[args.sport]; prov.final_game_pks(...)` etc. Filter pending `picks` by `sport = args.sport` in the query so an NFL grade run only touches NFL picks.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/ -q` → all PASS (full suite; MLB grading unchanged).

- [ ] **Step 5: Commit**

```bash
git add scripts/grade_results.py tests/test_grade_picks.py
git commit -m "feat(grade): results-provider seam + --sport (MLB provider registered)"
```

---

## Self-Review

**Spec coverage (P0 slice):**
- Multi-sport parameterization of `ingest_odds`/`generate_board`/`grade_results` → Tasks 3,4,5. ✓
- Verify NFL data sources + prop-market availability → Task 1. ✓
- NFL Odds-API keys captured in config → Task 2. ✓
- NFL schedule ingest, game_pk matcher, nflverse snapshots, the model → **explicitly deferred to P1+** (Global Constraints), because their tasks depend on Task 1's verified field shapes. Noted, not a gap.

**Placeholder scan:** Task 4 Step 3 notes NFL PROP_MARKETS values are "wired in P4" — that is a real deferral (no NFL predictions exist yet to board), not an unfilled blank; the MLB path is fully specified. No other placeholders.

**Type consistency:** `SportConfig` fields (`key`, `odds_sport`, `game_markets`, `prop_market_map`, `commence_shift_hours`) are used consistently across Tasks 2–3. `get(sport)`, `RESULTS_PROVIDERS`, and the `sport`/`--sport` arg names are consistent across Tasks 2–5.
