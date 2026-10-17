# "Take these" list, Pikkit scorecard and CLV test — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build three things:
- a site page listing ML, spread and total bets where a list book beats Pinnacle's same-pull no-vig price by ≥ 1.5%;
- a browser-only Pikkit scorecard;
- a pre-registered CLV test that grades every list pick against Pinnacle's close.

**Architecture:** A new package, `src/sportsmodel/take/`, holds the pure engine: n-way power de-vig, Odds API parsing, pick finding, good-until, grading and the verdict. A separate store module, `take/store.py`, holds the SQL.

Four scripts do the IO, each with a `Deps` dataclass as in `scripts/early_monitor.py`:
- pull, every 2 hours;
- close and grade, every 10 minutes;
- scores, daily;
- the weekly report.

Each script has its own GitHub Actions workflow. The site adds two static pages, `take.html` and `scorecard.html`, that follow the `ev.html` pattern. The scorecard parses the CSV in `site/js/pikkit.js` and never sends it anywhere.

**Tech Stack:** Python 3.12, httpx through `sportsmodel.ingest.odds`, psycopg through `sportsmodel.db.get_postgres`, numpy, and pytest. On the site: vanilla JS, Supabase REST through `sb`/`sbAll` in `site/app.js`, and node:test using `site/tests/load.mjs`.

**Spec:** `docs/superpowers/specs/2026-10-17-take-these-clv-design.md`, approved 2026-10-17.

## Global Constraints

**Secrets and migrations**
- Never print, commit or source `.env`. `ODDS_API_KEY` exists only in GitHub secrets, and workflows use the existing `DATABASE_URL` secret.
- The user runs migrations themselves. Ship one idempotent `db/migration_take_these.sql`. Nothing reads the new tables before the migration runs: every script warns and exits 0 when tables are missing, which is the `_missing_public_tables` guard.

**Leagues, markets and sport keys**
- Leagues: `nfl, cfb, mlb, nhl, nba, wnba, epl, mls`. Markets: `ml, spread, total`.
  - No props; never college props.
  - Soccer leagues are `epl` and `mls`. Their moneyline is 3-way (`home`, `draw`, `away`) and draw is pickable.
- Odds API sport keys, verbatim:
  - nfl: `americanfootball_nfl`
  - cfb: `americanfootball_ncaaf`
  - mlb: `baseball_mlb`
  - nhl: `icehockey_nhl`
  - nba: `basketball_nba`
  - wnba: `basketball_wnba`
  - epl: `soccer_epl`
  - mls: `soccer_usa_mls`

**Books**
- List books (Odds API keys, in this order): `draftkings, fanduel, betmgm, williamhill_us, fanatics, hardrockbet, ballybet`.
  - Plus `pinnacle` as the reference, for 8 keys, which bill as one region: a pull costs 3 credits.
  - **Ruling (2026-10-17):** The Odds API's bookmaker list has no Bet365 key for any US league; there is only `bet365_au`, for AFL/NRL. Per the spec's fallback, every league runs on the other 7 books. `bet365` is NOT sent, because an unknown key risks a 422. The go-live doc and the pull's dry run report this.
- `board.USER_BOOKS` is unchanged.

**Pick rule**
- EV = `fair_prob × decimal(price) − 1`. A pick needs EV ≥ `0.015`; use `EPS = 1e-12` for float noise, so exactly 1.5% passes.
- Eligible prices: −250 ≤ American ≤ +500.
- Pinnacle's market `last_update` must be no more than 30 minutes before the pull time. No pick once `commence_time ≤ pulled_at`.
- Good-until: `decimal = 1.015 / fair_prob`, converted to American and rounded toward the bettor-safe side.

**Credit floors**
- Pulls skip below 1,500 credits remaining. Close and scores skip below 300.
- An unknown remaining balance (before the first response) runs.
- Existing floors are untouched.

**Test parameters**
- `TEST_START = 2026-11-01T00:00:00Z`, `N_REQUIRED = 300`, beat-close ≥ `0.55`.
- Mean CLV > 0. Cluster-bootstrap 95% lower bound > 0, with clusters = `event_id`, `10_000` resamples, seed `20261017`, percentile 2.5.
- Judged once. The verdict row is append-only.

**Close and retention**
- A close is valid only if taken 0–30 minutes before start. The close job targets ungraded picks starting in 5–20 minutes.
- Raw prices are kept 7 days; picks, closes, results and the verdict are kept forever.

**Pull schedule and on/off switch**
- Pulls run every 2 hours, 12:00–04:00 UTC.
- The cadence is repo variable `TAKE_PULL_EVERY_HOURS` (default `2`).
- On/off is repo variable `TAKE_THESE` (`on` runs the scheduled jobs). `workflow_dispatch` always runs.

**Site**
- Don't relabel anything existing. New nav entries are appended.
- Cache version: bump every `?v=20261016a` to `?v=20261017a`.
- Desktop-first, but it must work at 375px width.
- Python tests run with `uv run pytest`, site tests with `cd site && npm test`.

**Git**
- Work on branch `feat/take-these`, never main.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Pinnacle missing a side.** Pinnacle's market can lack a side: a soccer ML with no draw price, or a spread with only the home line. That market makes no picks and nothing raises (Task 3 test `test_partial_pinnacle_market_makes_no_picks`).
2. **Unmatched outcome names.** An Odds API outcome whose `name` matches neither team (and is not "Draw", "Over" or "Under") is dropped. It is never assigned to a side (Task 2 test `test_unknown_outcome_name_dropped`).
3. **Lines that are equal but formatted differently.** −3.0 vs −3 must count as the same line, and quarter lines like 2.25 must compare exactly. All line comparisons go through one `same_line(a, b)` with tolerance 1e-6 (Task 3 test `test_line_float_tolerance`).
4. **A close taken from a stale start time.** If the game was moved after the last pull, the close job must compute minutes-to-start from the `commence_time` in its own response and update `take_events` from it (Task 6 test `test_close_uses_response_commence_time`).
5. **Real Pikkit exports.** They can start with a UTF-8 BOM, have `bet_info` with quoted commas, blank `closing_line`, `SETTLED_VOID` / `SETTLED_CASH_OUT` / `PLACED` statuses and multi-league `leagues` like `NFL | NCAAFB`. Parsing must not shift columns (Task 9 test `parseCsv handles BOM, quotes, blank close`).

---

## File map

| File | Responsibility |
|---|---|
| `src/sportsmodel/take/__init__.py` | empty |
| `src/sportsmodel/take/constants.py` | leagues, books, thresholds, floors, test constants, `League` |
| `src/sportsmodel/take/devig.py` | `novig_power_n`, `american_to_decimal`, `decimal_to_american_safe` |
| `src/sportsmodel/take/parse.py` | Odds API events → price rows and event rows; `/scores` → finals |
| `src/sportsmodel/take/picks.py` | `find_picks`, `good_until`, `same_line` |
| `src/sportsmodel/take/grade.py` | `close_fair`, `grade_pick`, `settle` |
| `src/sportsmodel/take/verdict.py` | `cluster_lower`, `test_rows`, `decide` |
| `src/sportsmodel/take/store.py` | all SQL for the take tables |
| `src/sportsmodel/ingest/odds.py` | add `fetch_sports()` and `fetch_scores(cfg, days_from=3)` |
| `db/migration_take_these.sql` | tables, triggers, views, RLS |
| `scripts/take_pull.py`, `scripts/take_close.py`, `scripts/take_scores.py`, `scripts/report_take_these.py` | IO |
| `.github/workflows/take-these-pull.yml`, `take-these-close.yml`, `take-these-scores.yml`, `report-take-these.yml` | schedules |
| `site/take.html`, `site/js/pages/take.js` | the list and the test panel |
| `site/scorecard.html`, `site/js/pikkit.js`, `site/js/pages/scorecard.js` | the scorecard |
| `site/js/shell.js`, `site/app.js`, `site/*.html` | nav, render dispatch, cache bump |
| `tests/take/…`, `site/tests/take.test.mjs`, `site/tests/pikkit.test.mjs`, `site/tests/scorecard.test.mjs` | tests |
| `docs/superpowers/take-these-go-live.md` | go-live checklist |

Shared row shapes, used by Tasks 2–7. They are plain dicts; the keys are binding.

- **price row:** `{event_id, league, market, side, book, line (float|None; the side's own line), price (int American), last_update (datetime|None), pulled_at (datetime)}`
- **event row:** `{event_id, league, home_team, away_team, commence_time (datetime)}`
- **pick:**
  - `{event_id, league, market, side, book, line, price, ev, fair_prob, good_until, better_line (bool), ref_line (Pinnacle's own line for this side; None for ml), pin_prices (dict side→{"line","price"}), others (list of {"book","line","price","ev"} sorted by ev desc), pulled_at, commence_time}`
- **close row:** `{event_id, market, prices (dict side→{"line","price"}), captured_at, minutes_to_start}`
- **result row:**
  - `{pick_id, graded_by ('clv'|'direction'|'no_close'), close_fair (float|None), clv (float|None), beat (bool|None), close_line (float|None), close_minutes (float|None), graded_at}`
  - plus, after `/scores`: `{outcome ('win'|'loss'|'push'|None), units (float|None)}`

---

### Task 1: Constants and n-way power de-vig

**Files:**
- Create: `src/sportsmodel/take/__init__.py`, `src/sportsmodel/take/constants.py`, `src/sportsmodel/take/devig.py`
- Test: `tests/take/__init__.py`, `tests/take/test_devig.py`

**Interfaces:**
- Produces, in `constants.py`:
  - `LEAGUES: dict[str, str]` (league → odds sport key, as in Global Constraints), `SOCCER = ("epl", "mls")`, `MODEL_LEAGUES = ("nfl", "cfb", "mlb")`.
  - `TAKE_BOOKS: tuple[str, ...]` (7 keys, in order), `REFERENCE = "pinnacle"`, `BOOKMAKERS = TAKE_BOOKS + (REFERENCE,)`.
  - `MARKETS = ("ml", "spread", "total")`, `API_MARKET = {"ml": "h2h", "spread": "spreads", "total": "totals"}`.
  - `EV_MIN = 0.015`, `EPS = 1e-12`, `PRICE_MIN = -250`, `PRICE_MAX = 500`, `PIN_MAX_AGE_MIN = 30`.
  - `PULL_FLOOR = 1500`, `CLOSE_FLOOR = 300`, `SCORES_FLOOR = 300`.
  - `CLOSE_LO_MIN = 5`, `CLOSE_HI_MIN = 20`, `CLOSE_VALID_MAX_MIN = 30`, `EVENTS_HORIZON_DAYS = 7`, `PRICE_RETENTION_DAYS = 7`.
  - `TEST_START = datetime(2026, 11, 1, tzinfo=UTC)`, `N_REQUIRED = 300`, `BEAT_MIN = 0.55`, `RESAMPLES = 10_000`, `SEED = 20261017`, `LOWER_PERCENTILE = 2.5`, `DIRECTION_STEP = 0.5`.
  - `@dataclass(frozen=True) class League: key: str; odds_sport: str`, duck-typed for the `odds.fetch_*` functions, plus `league(key) -> League`.
  - `budget_ok(remaining, floor: int) -> tuple[bool, str | None]`: use `early.polling.parse_credits`. Unknown remaining → `(True, None)`. Below the floor → `(False, "<n> Odds API credits left (< <floor>): skipping")`.
- Produces, in `devig.py`:
  - `american_to_decimal(price: int) -> float`.
  - `novig_power_n(prices: list[int]) -> list[float]`.
  - `decimal_to_american_safe(dec: float) -> int`: the smallest-for-the-book American price whose decimal is ≥ `dec`.

- [ ] **Step 1: Write the failing tests** in `tests/take/test_devig.py`:

```python
def test_two_way_matches_board():
    from sportsmodel.serving.board import novig_power
    for a, b in [(-110, -110), (-150, 130), (-300, 240), (105, -125)]:
        p = novig_power_n([a, b])
        assert abs(p[0] - novig_power(a, b)) < 1e-9 and abs(sum(p) - 1) < 1e-9

def test_three_way_sums_to_one_and_keeps_order():
    p = novig_power_n([150, 230, 190])          # home, draw, away
    assert abs(sum(p) - 1) < 1e-9 and p[0] > p[2] > p[1]

def test_no_vig_market_is_proportional():
    p = novig_power_n([100, 100])
    assert p == pytest.approx([0.5, 0.5])

def test_decimal_to_american_safe():
    assert decimal_to_american_safe(1.015 / 0.5) == 103
    assert decimal_to_american_safe(1.015 / 0.6) == -144
    assert decimal_to_american_safe(1.015 / 0.25) == 306
    assert decimal_to_american_safe(2.0) == 100

def test_budget_ok():
    assert budget_ok(None, 1500) == (True, None)
    assert budget_ok("1499", 1500)[0] is False
    assert budget_ok("1500", 1500) == (True, None)

def test_league_keys():
    assert LEAGUES["mls"] == "soccer_usa_mls" and len(LEAGUES) == 8
    assert "bet365" not in BOOKMAKERS and len(BOOKMAKERS) == 8
```

- [ ] **Step 2: Run** `uv run pytest tests/take/test_devig.py -v`. Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement.** `novig_power_n` finds k such that Σ (1/decᵢ)ᵏ = 1, using bisection on k ∈ [1, 10] as `board.novig_power` does, and falls back to proportional when the bracket fails. Don't import or modify `board.novig_power` in the implementation.

- [ ] **Step 4: Run** the same command. Expected: PASS.

- [ ] **Step 5: Commit** `feat(take): constants and n-way power de-vig`.

---

### Task 2: Parse Odds API responses

**Files:**
- Create: `src/sportsmodel/take/parse.py`, `tests/take/fixtures/odds_nfl.json`, `tests/take/fixtures/odds_epl.json`, `tests/take/fixtures/scores_nfl.json`
- Modify: `src/sportsmodel/ingest/odds.py` (append two fetchers)
- Test: `tests/take/test_parse.py`

**Interfaces:**
- Consumes: `constants.API_MARKET`, `SOCCER`.
- Produces:
  - `parse_events(events: list[dict], league: str) -> list[dict]`: event rows.
  - `parse_prices(events: list[dict], league: str, pulled_at: datetime) -> list[dict]`: price rows for every bookmaker present.
    - Market key `h2h` → `ml`, `spreads` → `spread`, `totals` → `total`.
    - Side: outcome name equal to `home_team` → `home`, `away_team` → `away`, `"Draw"` → `draw` (soccer `ml` only), `"Over"`/`"Under"` → `over`/`under`.
    - `line` = outcome `point` (None for ml).
    - `last_update` = the bookmaker's market `last_update`, parsed.
  - `parse_scores(scores: list[dict]) -> dict[str, tuple[int, int]]`: `event_id → (home_score, away_score)`, for `completed` events only.
  - `odds.fetch_sports() -> list[dict]` (GET `/sports`, free).
  - `odds.fetch_scores(cfg, days_from: int = 3) -> list[dict]` (GET `/sports/{cfg.odds_sport}/scores` with `daysFrom`, using `_get_once`).

- [ ] **Step 1: Write the fixtures.** They are hand-made, in Odds API v4 shape.
  - **NFL fixture:** one event, home "Kansas City Chiefs", away "Buffalo Bills", commence 2026-11-02T18:00:00Z.
    - Bookmakers pinnacle, draftkings and fanduel, each with h2h, spreads (home −3 / away +3) and totals (47.5).
    - draftkings has away +3.5.
    - One draftkings h2h outcome is named "KC Chiefs", so it matches neither team.
  - **EPL fixture:** one event, h2h with home/Draw/away at pinnacle and fanduel.
  - **Scores fixture:** one completed and one not completed.

- [ ] **Step 2: Write the failing tests:**

```python
def test_parse_prices_sides_and_lines():
    rows = parse_prices(load("odds_nfl.json"), "nfl", PULLED)
    dk = {(r["market"], r["side"]): r for r in rows if r["book"] == "draftkings"}
    assert dk[("spread", "away")]["line"] == 3.5 and dk[("total", "over")]["line"] == 47.5
    assert all(r["league"] == "nfl" and r["pulled_at"] == PULLED for r in rows)

def test_unknown_outcome_name_dropped():
    rows = parse_prices(load("odds_nfl.json"), "nfl", PULLED)
    assert not [r for r in rows if r["book"] == "draftkings" and r["market"] == "ml" and r["side"] not in ("home", "away")]
    assert len([r for r in rows if r["book"] == "draftkings" and r["market"] == "ml"]) == 1

def test_soccer_draw_side():
    rows = parse_prices(load("odds_epl.json"), "epl", PULLED)
    assert {r["side"] for r in rows if r["book"] == "pinnacle"} == {"home", "draw", "away"}

def test_parse_events_and_scores():
    ev = parse_events(load("odds_nfl.json"), "nfl")[0]
    assert ev["commence_time"] == datetime(2026, 11, 2, 18, tzinfo=UTC)
    assert list(parse_scores(load("scores_nfl.json")).values()) == [(27, 24)]
```

- [ ] **Step 3: Run** `uv run pytest tests/take/test_parse.py -v`. Expected: FAIL.

- [ ] **Step 4: Implement** `parse.py` and the two fetchers. `fetch_scores` uses params `{"apiKey": ..., "daysFrom": days_from, "dateFormat": "iso"}`, in the same style as `fetch_bookmaker_odds`.

- [ ] **Step 5: Run.** Expected: PASS.

- [ ] **Step 6: Commit** `feat(take): parse odds, events and scores`.

---

### Task 3: Pick engine

**Files:**
- Create: `src/sportsmodel/take/picks.py`
- Test: `tests/take/test_picks.py`

**Interfaces:**
- Consumes: price rows, event rows, `devig.novig_power_n`, `devig.american_to_decimal`, `devig.decimal_to_american_safe`, and the constants.
- Produces:
  - `same_line(a: float | None, b: float | None) -> bool`: tolerance 1e-6; `None == None` is True.
  - `good_until(fair_prob: float) -> int`: `decimal_to_american_safe(1.015 / fair_prob)`.
  - `pinnacle_fair(rows_for_market: list[dict], league: str, market: str, pulled_at: datetime) -> dict[str, dict] | None`
    - Returns side → `{"line", "price", "fair"}`, or None when the market is incomplete or stale.
    - **Complete** means:
      - `ml`: home and away, plus draw for soccer;
      - `spread`: home and away with `same_line(home.line, -away.line)`;
      - `total`: over and under on the same line.
    - **Stale** means `pulled_at − last_update > 30 min`, or `last_update` is None.
  - `find_picks(prices: list[dict], events: list[dict], pulled_at: datetime) -> list[dict]`: picks, one per (event_id, market, side) at most.

**Rules `find_picks` implements** (spec §1):
- Skip events with `commence_time ≤ pulled_at`.
- A book row qualifies when all of these hold:
  - its book is in `TAKE_BOOKS`;
  - `PRICE_MIN ≤ price ≤ PRICE_MAX`;
  - its line is the same as Pinnacle's or better. "Better" means:
    - spread: book line > Pinnacle's line for that side;
    - over: book line < Pinnacle's;
    - under: book line > Pinnacle's;
  - EV ≥ `EV_MIN − EPS`, using `fair` at Pinnacle's line.
- The best book is the highest EV. Ties go first to same-line over better-line, then to `TAKE_BOOKS` order.
- `others` holds every other qualifying book.
- `better_line` is True when the chosen book's line differs from Pinnacle's.
- `ref_line` is Pinnacle's line for the side.
- `pin_prices` holds Pinnacle's sides for that market.

- [ ] **Step 1: Write the failing tests.** Build rows inline with a helper `row(book, market, side, line, price, age_min=1)`.

```python
def test_same_line_pick():          # pin -110/-110 at -3; DK home -3 at +105 -> ev = 0.5*2.05-1 = 0.025
    p = find_picks([*pin_spread(-3, -110, -110), row("draftkings", "spread", "home", -3, 105)], [EV], PULLED)
    assert len(p) == 1 and p[0]["book"] == "draftkings" and not p[0]["better_line"]
    assert p[0]["ev"] == pytest.approx(0.025) and p[0]["ref_line"] == -3 and p[0]["good_until"] == 103

def test_better_line_pick_marked_and_uses_pinnacle_fair():
    p = find_picks([*pin_spread(-3, -110, -110), row("fanduel", "spread", "away", 3.5, 103)], [EV], PULLED)
    assert p[0]["better_line"] and p[0]["line"] == 3.5 and p[0]["ref_line"] == 3

def test_worse_line_no_pick():
    assert find_picks([*pin_spread(-3, -110, -110), row("fanduel", "spread", "away", 2.5, 120)], [EV], PULLED) == []

def test_threshold_and_price_range():
    # +102 at fair .5 -> ev .01 (no); -260 never; +550 never
    ...assert each yields []

def test_stale_pinnacle_no_picks():
    assert find_picks([*pin_spread(-3, -110, -110, age_min=31), row("draftkings", "spread", "home", -3, 120)], [EV], PULLED) == []

def test_started_game_no_picks(): ...   # commence == pulled_at -> []

def test_partial_pinnacle_market_makes_no_picks():
    # soccer ml with pinnacle home/away only, and a spread with only pinnacle home -> []

def test_soccer_draw_pickable():
    # pin 150/230/190; FD draw +260 -> one pick, side "draw"

def test_line_float_tolerance():
    # pinnacle home -3.0 / away 3; DK home -3.0000001 at +105 -> same-line pick; 2.25 vs 2.25 total same line

def test_best_book_and_others():
    # DK +105 and FD +110 on the same side -> book fanduel, others == [draftkings]

def test_non_list_book_ignored():   # "betrivers" row never makes a pick
```

- [ ] **Step 2: Run** `uv run pytest tests/take/test_picks.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** `picks.py`.

- [ ] **Step 4: Run.** Expected: PASS.

- [ ] **Step 5: Commit** `feat(take): pick engine (same line, better line, freshness, good-until)`.

---

### Task 4: Grading and settlement

**Files:**
- Create: `src/sportsmodel/take/grade.py`
- Test: `tests/take/test_grade.py`

**Interfaces:**
- Consumes: the pick (as stored: `market, side, line, price, ref_line, league`), the close row, and `novig_power_n`.
- Produces:
  - `close_valid(close: dict | None) -> bool`: `0 ≤ minutes_to_start ≤ 30`.
  - `close_fair(close_prices: dict, league: str, market: str, side: str) -> float | None`: the n-way de-vig over the close's sides. Requires the market to be complete, as in Task 3.
  - `grade_pick(pick: dict, close: dict | None, graded_at: datetime) -> dict`: a result row without outcome or units.
  - `settle(pick: dict, home: int, away: int) -> tuple[str, float]`: `(outcome, units)` at the pick's own line and price.

**Rules (spec §1 "Grading each pick"):**
- No valid close, or the close has no complete market: `graded_by="no_close"`, with clv/beat None.
- `ml`: `clv = close_fair × dec(price) − 1`, `beat = clv > 0`, `graded_by="clv"`.
- `spread` / `total`:
  - Let L = the close's line for the pick's side.
  - If `same_line(L, ref_line)`: CLV as above, using `close_fair` at L.
  - Otherwise, `move` toward the pick is:
    - spread: `ref_line − L`;
    - over: `L − ref_line`;
    - under: `ref_line − L`.
  - Then `beat = move ≥ DIRECTION_STEP − 1e-9`, `clv=None`, `graded_by="direction"`, and `close_line=L`.
  - **Ruling:** a move toward the pick smaller than 0.5 (e.g. a 0.25 soccer quarter line) is a miss. That is conservative: it can only lower the beat rate.
- Settlement:
  - **ml:** the side with the higher score wins. A soccer tie means `draw` wins and home/away lose. A non-soccer tie is a push.
  - **spread:** the side's margin plus its line is > 0 for a win, 0 for a push.
  - **total:** compare the sum with the line.
  - **Units:** win = `dec − 1`, loss = `−1`, push = `0`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_ml_clv():               # close pin -120/+100 power fair for home; pick home +105
    r = grade_pick(pick("ml", "home", None, 105), close(ml(-120, 100), 8), T)
    assert r["graded_by"] == "clv" and r["clv"] == pytest.approx(close_fair(...) * 2.05 - 1) and r["beat"] is True

def test_spread_close_at_R_uses_clv():           # better-line pick away +3.5, ref 3; close at 3 -> clv at fair(3)
def test_spread_moved_toward_pick_is_beat():     # pick home -3 (ref -3), close home -3.5 -> direction, beat
def test_spread_moved_away_is_miss():            # close home -2.5 -> beat False, clv None
def test_total_under_direction():                # pick under ref 47.5, close 46.5 -> beat
def test_quarter_move_is_miss():                 # ref 2.5 over, close 2.75 -> move .25 -> beat False
def test_no_close_when_late_or_early():
    assert grade_pick(p, close(..., minutes=-1), T)["graded_by"] == "no_close"
    assert grade_pick(p, close(..., minutes=31), T)["graded_by"] == "no_close"
    assert grade_pick(p, None, T)["graded_by"] == "no_close"
def test_settle():
    assert settle(pick("spread", "away", 3.5, 110), 27, 24) == ("win", pytest.approx(1.1))
    assert settle(pick("ml", "draw", None, 250, league="epl"), 1, 1) == ("win", pytest.approx(2.5))
    assert settle(pick("ml", "home", None, -150, league="epl"), 1, 1) == ("loss", -1.0)
    assert settle(pick("total", "over", 47.0, -110), 27, 20)[0] == "push"
```

- [ ] **Step 2: Run** `uv run pytest tests/take/test_grade.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement.**

- [ ] **Step 4: Run.** Expected: PASS.

- [ ] **Step 5: Commit** `feat(take): CLV grading with reference-line rule, settlement`.

---

### Task 5: Migration and store

**Files:**
- Create: `db/migration_take_these.sql`, `src/sportsmodel/take/store.py`
- Test: `tests/take/test_migration_and_store.py`

**Tables** (all `IF NOT EXISTS`; re-runnable):

| Table | Key | Columns / notes |
|---|---|---|
| `take_events` | `event_id text PK` | `league` (CHECK in the 8), `home_team`, `away_team`, `commence_time timestamptz`, `game_pk bigint NULL`, `home_score int NULL`, `away_score int NULL`, `completed bool default false`, `updated_at` |
| `take_prices` | PK `(pulled_at, event_id, market, side, book)` | `league`, `line double NULL`, `price int`, `last_update timestamptz NULL`; index on `pulled_at`. Pruned after 7 days. |
| `take_picks` | `id bigserial`, UNIQUE `(event_id, market, side)` | `league, book, line, price, ev, fair_prob, good_until, better_line, ref_line, pin_prices jsonb, first_seen timestamptz, run_id`. **Append-only**: a trigger refuses UPDATE/DELETE/TRUNCATE, and write grants are revoked from anon/authenticated, copying `early_tickets_append_only`. |
| `take_pick_state` | PK `(event_id, market, side)` | `league, book, line, price, ev, fair_prob, good_until, better_line, others jsonb, first_seen, last_seen, active bool, dropped_at timestamptz NULL` |
| `take_closes` | PK `(event_id, market)` | `prices jsonb, captured_at, minutes_to_start double`. An upsert overwrites only when `EXCLUDED.captured_at > take_closes.captured_at`. |
| `take_results` | `pick_id bigint PK REFERENCES take_picks(id)` | `graded_by` (CHECK in clv/direction/no_close), `close_fair, clv, beat, close_line, close_minutes, graded_at, outcome` (CHECK win/loss/push or NULL), `units, settled_at` |
| `take_api_calls` | `id bigserial` | `called_at, job` (pull/close/scores), `league, endpoint, credits_used int NULL, credits_remaining int NULL, books_missing text[] NULL, run_id` |
| `take_verdict` | `id smallint PK CHECK (id = 1)` | `judged_at, n, beat_rate, mean_clv, lower, passed bool, criteria jsonb, pick_ids bigint[]`. Append-only trigger like `take_picks`. |

**Views** (security invoker, public read):
- `take_list_v`: `take_pick_state ⨝ take_events`. It includes rows with `active` or `coalesce(dropped_at, commence_time) > now() − interval '24 hours'`, and exposes `game_pk`.
- `take_test_v`: `take_picks ⨝ take_events ⟕ take_results`, every column the report and site need: `id, event_id, league, market, side, book, line, price, ev, better_line, first_seen, commence_time, graded_by, clv, beat, graded_at, outcome, units`.

**RLS:** enable it on every table, with `FOR SELECT USING (true)` policies and `GRANT SELECT ... TO anon, authenticated`, as at the end of `db/migration_early_tickets.sql`.

**Interfaces.** All functions take `connect_timeout` / `statement_timeout_ms` keywords and default to `db.LEDGER_CONNECT_TIMEOUT` / `db.LEDGER_STATEMENT_TIMEOUT_MS`. Connections come from `db.get_postgres`, and timeouts from `db._limit_statements`.
- `TABLES: tuple[str, ...]` (the 8 tables); `missing_tables(**t) -> list[str]` uses `db._missing_public_tables`.
- `upsert_events(rows: list[dict], **t) -> int`: updates `commence_time`, teams and `game_pk` (keeping an existing game_pk when the new one is None).
- `insert_prices(rows, **t) -> int`; `prune_prices(before: datetime, **t) -> int`.
- `insert_picks(picks, run_id: str, **t) -> int`: `ON CONFLICT (event_id, market, side) DO NOTHING`.
- `sync_state(league: str, event_ids: list[str], picks: list[dict], pulled_at: datetime, **t) -> None`:
  - upserts each pick: `active=true`, `last_seen=pulled_at`, keeping `first_seen`;
  - then sets `active=false, dropped_at=pulled_at` for active rows of `league` whose event is in `event_ids` and whose key was not in `picks`.
- `log_call(job, league, endpoint, credits_used, credits_remaining, books_missing, run_id, **t) -> None`.
- `close_targets(now: datetime, lo_min: float, hi_min: float, **t) -> dict[str, set[str]]`: league → markets that hold ungraded picks (no `take_results` row) whose event starts in [lo, hi] minutes.
- `upsert_closes(rows, **t) -> int`.
- `ungraded_started(now, **t) -> list[dict]`: picks with `commence_time ≤ now` and no result. Each comes with its close row (or None) under key `"close"`.
- `insert_results(rows, **t) -> int`: `ON CONFLICT (pick_id) DO NOTHING`.
- `unsettled_leagues(**t) -> list[str]`: leagues with a result whose outcome is NULL and an event that started more than 3 hours ago.
- `unsettled(league, **t) -> list[dict]`: those picks, with `event_id, market, side, line, price, league`.
- `store_scores(finals: dict[str, tuple[int, int]], **t) -> int`; `set_outcomes(rows: list[dict], **t) -> int`.
- `test_rows(**t) -> list[dict]`: everything from `take_test_v`.
- `get_verdict(**t) -> dict | None`; `insert_verdict(row, **t) -> bool`: `ON CONFLICT DO NOTHING`, returns whether it inserted.

- [ ] **Step 1: Write the failing tests.** They are static SQL checks plus tuple builders, in the style of `tests/early/test_migration_and_db.py`, with no live DB.

```python
def test_migration_idempotent_shape():
    for t in store.TABLES:
        assert f"CREATE TABLE IF NOT EXISTS {t}" in MIGRATION
    assert "DROP POLICY IF EXISTS" in MIGRATION and "CREATE OR REPLACE VIEW take_list_v" in MIGRATION
    assert "'soccer'" not in MIGRATION and "'epl'" in MIGRATION and "'mls'" in MIGRATION

def test_take_picks_and_verdict_append_only():
    assert "BEFORE UPDATE OR DELETE ON take_picks" in MIGRATION and "BEFORE UPDATE OR DELETE ON take_verdict" in MIGRATION
    assert "REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON take_picks" in MIGRATION

def test_closes_upsert_keeps_latest():
    assert "WHERE EXCLUDED.captured_at > take_closes.captured_at" in store.UPSERT_CLOSES_SQL

def test_pick_tuple_order_matches_columns():
    t = store.pick_tuple(SAMPLE_PICK, "run1")
    assert len(t) == len(store.PICK_COLUMNS) and t[store.PICK_COLUMNS.index("pin_prices")] == json.dumps(SAMPLE_PICK["pin_prices"])
```

- [ ] **Step 2: Run** `uv run pytest tests/take/test_migration_and_store.py -v`. Expected: FAIL.

- [ ] **Step 3: Write** the migration and `store.py`. The migration header comment states:
  - what each table is for;
  - "run before setting TAKE_THESE=on";
  - the rollback: `DROP VIEW take_list_v, take_test_v; DROP TABLE ...` in dependency order, as a comment only.

- [ ] **Step 4: Run.** Expected: PASS.

- [ ] **Step 5: Commit** `feat(take): migration and store`.

---

### Task 6: Pull script and close/grade script, with workflows

**Files:**
- Create: `scripts/take_pull.py`, `scripts/take_close.py`, `.github/workflows/take-these-pull.yml`, `.github/workflows/take-these-close.yml`
- Test: `tests/take/test_take_pull.py`, `tests/take/test_take_close.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- `take_pull.py`:
  - `@dataclass Deps` holds:
    - `missing_tables, fetch_events, fetch_odds, remaining, last_cost`;
    - `match_game_pks: Callable[[str, list[dict]], dict[str, int]]`, for event_id → game_pk on MODEL_LEAGUES only, with a failure → `{}`;
    - `upsert_events, insert_prices, insert_picks, sync_state, prune_prices, log_call`.
  - `due(now: datetime, every_hours: int) -> bool`: `now.hour % every_hours == 0`. A bad or empty value counts as 2.
  - `run_pull(now, deps, *, run_id="local", dry_run=False, leagues=tuple(LEAGUES)) -> dict`, which returns `{"leagues": {league: {...}}, "credits": int, "picks": int, "skipped": [..]}`.
  - `main(argv)`, with flags `--dry-run`, `--scheduled` (applies `due()` with `TAKE_PULL_EVERY_HOURS`) and `--league` (repeatable).
- `take_close.py`:
  - `@dataclass Deps` holds `missing_tables, prime (fetch_sports), fetch_odds, remaining, last_cost, ungraded_started, insert_results, close_targets, upsert_closes, upsert_events, log_call`.
  - `run_close(now, deps, *, run_id="local", dry_run=False) -> dict`.
  - `main(argv)`, with flag `--dry-run`.

**`run_pull` flow, per league, in the order of `LEAGUES`:**
1. Free `fetch_events`. If no event starts within `EVENTS_HORIZON_DAYS`, skip and record it as skipped; this costs 0 credits.
2. Check `budget_ok(remaining(), PULL_FLOOR)`. If it fails, warn and stop all remaining leagues.
3. Call `fetch_odds(league_cfg, list(BOOKMAKERS), ["h2h","spreads","totals"])`.
4. Parse it, then `find_picks`.
5. Print `books seen` / `missing` for every league, in dry runs too.
6. Write in this order: `upsert_events` (with game_pks), `insert_prices`, `insert_picks`, `sync_state`, then `log_call(job="pull", ...)` with the credits used and remaining.
   - A failure in one league warns and moves on (the `early_monitor` pattern).
   - After all leagues: `prune_prices(now − 7 days)`.
   - Exit 1 only when every attempted league failed.
   - The missing-tables guard warns and exits 0.
   - `writes_disabled()` (SM_LEDGER_OFF) means a dry run.

**`run_close` flow:**
1. **Grade first.** For each pick from `ungraded_started(now)`, use `grade_pick(pick, pick["close"], now)`, then `insert_results`. This is free, so it runs even when credits are low.
2. `targets = close_targets(now, 5, 20)`. If there are none, return; that costs nothing.
3. Free `prime()`, then check `budget_ok(remaining(), CLOSE_FLOOR)`.
4. Per league, call `fetch_odds(cfg, ["pinnacle"], [API_MARKET[m] for m in sorted(markets)])`.
5. For each returned event that has a target pick:
   - take `minutes_to_start` from **the response's** `commence_time`;
   - `upsert_events` that commence time;
   - store a close row only when 0 ≤ minutes_to_start ≤ 30.
6. `log_call(job="close")`.

- [ ] **Step 1: Write the failing tests.** They use fake Deps (in-memory lists) and the Task 2 fixtures.

```python
def test_pull_skips_league_with_no_events_and_spends_nothing():
    out = run_pull(NOW, fake(events={"nfl": []}), leagues=("nfl",))
    assert out["skipped"] == ["nfl"] and out["credits"] == 0 and fake.calls["fetch_odds"] == 0

def test_pull_requests_eight_keys_three_markets():
    run_pull(NOW, f := fake(), leagues=("nfl",))
    cfg, books, markets = f.calls_args["fetch_odds"][0]
    assert cfg.odds_sport == "americanfootball_nfl" and len(books) == 8 and markets == ["h2h", "spreads", "totals"]

def test_pull_stops_below_floor():
    out = run_pull(NOW, fake(remaining="1499"), leagues=("nfl", "nba"))
    assert fake.calls["fetch_odds"] == 0

def test_pull_writes_picks_state_and_logs_credits(): ...
def test_pull_dry_run_writes_nothing(): ...
def test_due():
    assert due(datetime(2026, 11, 1, 14, tzinfo=UTC), 2) and not due(datetime(2026, 11, 1, 13, tzinfo=UTC), 2)
    assert due(datetime(2026, 11, 1, 14, tzinfo=UTC), "") is True

def test_close_grades_started_picks_even_when_broke():
    out = run_close(NOW, fake_close(remaining="10", ungraded=[pick_with_close]))
    assert len(fake_close.results) == 1 and fake_close.calls["fetch_odds"] == 0

def test_close_requests_only_needed_markets(): # targets {"nhl": {"total"}} -> markets ["totals"], books ["pinnacle"]

def test_close_uses_response_commence_time():
    # stored start 12:15 (target), response says 13:15 -> minutes_to_start 70 -> no close stored, upsert_events gets 13:15
```

- [ ] **Step 2: Run** `uv run pytest tests/take/test_take_pull.py tests/take/test_take_close.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement both scripts.** Copy the header, `_warn` prefix (`::warning::take-pull:` / `take-close:`), `_db_deps()` and `main` from `scripts/early_monitor.py`.
  - `match_game_pks` wraps `ingest_odds.build_game_lookup(events, ingest_odds._ESPN_FETCHERS[league](), ingest_odds._matcher_for(league))` and maps each event id to its `game_pk`. Wrap it in try/except → `{}`.

- [ ] **Step 4: Write the workflows**, copying `early-monitor.yml`. Both take a `dry_run` dispatch input and secrets `DATABASE_URL` and `ODDS_API_KEY`.

| Workflow | Cron | Job condition | Concurrency | Timeout | Runs |
|---|---|---|---|---|---|
| `take-these-pull.yml` | `"0 12-23,0-4 * * *"` | `if: vars.TAKE_THESE == 'on' \|\| github.event_name == 'workflow_dispatch'` | `take-these-pull` | 8 min | `scripts/take_pull.py --scheduled` on schedule events; plain or `--dry-run` on dispatch; env `TAKE_PULL_EVERY_HOURS: ${{ vars.TAKE_PULL_EVERY_HOURS }}` |
| `take-these-close.yml` | `"3-59/10 * * * *"` | same `if` | `take-these-close` | 5 min | `scripts/take_close.py` |

  The header comments state the credit math from spec §4.

- [ ] **Step 5: Run** `uv run pytest tests/take -v`. Expected: PASS.

- [ ] **Step 6: Commit** `feat(take): pull and close/grade jobs with workflows`.

---

### Task 7: Scores, verdict and weekly report

**Files:**
- Create: `src/sportsmodel/take/verdict.py`, `scripts/take_scores.py`, `scripts/report_take_these.py`, `.github/workflows/take-these-scores.yml`, `.github/workflows/report-take-these.yml`
- Test: `tests/take/test_verdict.py`, `tests/take/test_take_scores.py`, `tests/take/test_report_take_these.py`

**Interfaces:**
- `verdict.py`:
  - `cluster_lower(rows: list[dict], *, key="event_id", value="clv", resamples=RESAMPLES, seed=SEED, percentile=LOWER_PERCENTILE) -> float | None`: the same algorithm as `early.passtest.cluster_bootstrap_lower`, keyed by `key`. Don't modify passtest.
  - `test_rows(rows: list[dict]) -> list[dict]`: rows with `first_seen ≥ TEST_START` and `graded_by in ("clv", "direction")`, sorted by `(graded_at, id)`.
  - `stats(rows) -> dict`, with keys `n, beat_rate, mean_clv, n_clv, lower, n_direction, n_better_line`.
  - `decide(all_rows: list[dict], existing: dict | None, as_of: datetime) -> dict`:
    - if `existing` is set, return it unchanged with `"carried": True`;
    - if there are fewer than 300 test rows, return `{"judged": False, "n": ..., **stats}`;
    - otherwise judge **the first 300** by `(graded_at, id)` and return `{"judged": True, "passed": bool, "pick_ids": [...300], "criteria": {"beat_rate>=0.55": bool, "mean_clv>0": bool, "lower>0": bool}, **stats}`.
    - **Ruling:** the first 300 to be graded, because that set is fixed the moment it exists. Ordering by first_seen could change as later games grade.
- `take_scores.py`:
  - `Deps` holds `missing_tables, prime, fetch_scores, remaining, last_cost, unsettled_leagues, unsettled, store_scores, set_outcomes, log_call`.
  - `run_scores(now, deps, *, run_id, dry_run) -> dict`. Per league in `unsettled_leagues()`: check `budget_ok(SCORES_FLOOR)`, then `fetch_scores(cfg, 3)`, then `parse_scores`, then `store_scores`, then `settle` for each unsettled pick whose event is final, then `set_outcomes`, then `log_call(job="scores")`.
- `report_take_these.py`:
  - `build_report(as_of, rows, verdict_row, calls) -> tuple[str, dict | None]`. Returns the markdown and the verdict row to insert, or None.
  - `main()` writes `docs/superpowers/reports/YYYY-MM-DD-take-these.md` and calls `store.insert_verdict` when the verdict was newly judged.

**Report sections:**
- **Progress:** "N of 300 graded picks since Nov 1", with the three criteria shown live and the verdict once judged (with the date).
- **Per league and per market:** n, beat rate, mean CLV.
- Better-line picks reported separately.
- Number graded by direction; no-close count.
- Results: W-L-P and units.
- October shakedown summary, labelled "not judged".
- Credits used in the last 7 days per job, from `take_api_calls`.
- Books missing per league in the last 7 days.
- The line: "Your own Pikkit beat-close rate over the same dates is on the Scorecard page (Pikkit's close is not necessarily Pinnacle's)."

- [ ] **Step 1: Write the failing tests:**

```python
def test_not_judged_below_300():
    assert decide(rows(299), None, AS_OF)["judged"] is False

def test_judges_first_300_by_graded_at():
    d = decide(rows(320), None, AS_OF)
    assert d["judged"] and len(d["pick_ids"]) == 300 and d["pick_ids"] == [r["id"] for r in sorted_first_300]

def test_october_picks_excluded():
    assert decide(rows(300, first_seen=datetime(2026, 10, 31, 23, tzinfo=UTC)), None, AS_OF)["judged"] is False

def test_pass_and_fail_criteria():
    # 300 rows, beat 56%, clv +0.01 each -> passed; beat 54% -> not passed; criteria keys exact

def test_existing_verdict_never_recomputed():
    old = {"judged": True, "passed": False}
    assert decide(rows(400, all_beat=True), old, AS_OF) == {**old, "carried": True}

def test_cluster_lower_deterministic_and_clustered():
    a = cluster_lower(R); b = cluster_lower(R)
    assert a == b and cluster_lower([]) is None

def test_scores_settles_and_skips_below_floor(): ...
def test_report_contains_progress_and_caveat():
    md, new = build_report(AS_OF, rows(10), None, [])
    assert "10 of 300 graded picks" in md and "Pikkit's close is not necessarily Pinnacle's" in md and new is None
```

- [ ] **Step 2: Run** `uv run pytest tests/take/test_verdict.py tests/take/test_take_scores.py tests/take/test_report_take_these.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement** verdict.py, take_scores.py and report_take_these.py. The report reuses `scripts/report_early_tickets.py`'s file layout and `main` shape.

- [ ] **Step 4: Write the workflows:**
  - `take-these-scores.yml`: cron `"30 9 * * *"` with the same `TAKE_THESE` `if`.
  - `report-take-these.yml`: copy `report-early-tickets.yml` exactly (the commit with `[skip ci]` and the push-retry loop), with cron `"0 13 * * 1"` and no `TAKE_THESE` gate. The report exits 0 without writing when there are zero picks.

- [ ] **Step 5: Run** `uv run pytest tests/take -v`. Expected: PASS.

- [ ] **Step 6: Commit** `feat(take): scores, pre-registered verdict, weekly report`.

---

### Task 8: "Take these" page

**Files:**
- Create: `site/take.html`, `site/js/pages/take.js`
- Modify:
  - `site/js/shell.js` (append to `NAV_ITEMS`: `["take","Take These","take.html"]`, `["scorecard","Scorecard","scorecard.html"]`)
  - `site/app.js` (dispatch in `render()`: `page === "take"` → `buildTakePage()` / `wireTakePage()`; `page === "scorecard"` → `buildScorecardPage()` / `wireScorecardPage()`)
- Test: `site/tests/take.test.mjs`; update `site/tests/shell.test.mjs` if it asserts the nav length.

**Interfaces:**
- Consumes:
  - `take_list_v` (`sbAll("take_list_v?select=*&order=commence_time.asc")`);
  - `take_test_v` (`sbAll("take_test_v?select=id,event_id,league,market,side,better_line,first_seen,graded_by,clv,beat,graded_at,outcome,units")`);
  - `take_verdict` (`sb("take_verdict?select=*")`);
  - `game_predictions` for the `game_pk`s on the list (`game_pk=in.(…)`, latest `generated_at` per game_pk, fields `pred_margin,pred_total`);
  - `shrinkLean(sport, market, model, line)` from `js/shrink.js`.
- Produces pure helpers, prefixed `take` because the scripts share the global scope:
  - `takeAmerican(n) -> string`: e.g. "+105" or "−110", using a real minus sign as the site does elsewhere.
  - `takeBetLabel(p) -> string`:
    - spread: "Chiefs −3.5" (short team name via data.js `shortTeam`);
    - total: "Over 8.5";
    - ml: "Arsenal ML";
    - draw: "Draw".
  - `takeEdge(p) -> string`: "≥ 2.1%" when `better_line`, otherwise "2.1%".
  - `takeCountdown(commenceIso, now: Date) -> string`: "2h 15m", or "1d 3h" past 24h.
  - `takeSplit(rows, now) -> {live, dropped}`:
    - live = `active && commence > now`, sorted by edge desc;
    - dropped = the rest, sorted by `dropped_at || commence` desc.
  - `takeModelNote(p, pred) -> "model agrees" | "model disagrees" | ""`:
    - only for MODEL_LEAGUES with spread or total and a pred;
    - spread: `shrinkLean(league, "spread", pred.pred_margin, homeLine)`, where homeLine = `p.side === "home" ? p.ref_line : -p.ref_line`;
    - total: `shrinkLean(league, "total", pred.pred_total, p.ref_line)`;
    - null lean → "".
  - `takeProgress(testRows, verdict) -> {graded, needed: 300, beatRate, meanClv, judged, passed}`. It counts only `first_seen ≥ "2026-11-01T00:00:00Z"` with `graded_by` of clv or direction.

**Layout:**
- The progress panel at the top. Before Nov 1 it says "Test starts Nov 1; October picks are a shakedown and are not judged".
- Then the live table with columns League · Matchup · Start (countdown + local time) · Bet · Book & price (other qualifying books smaller beneath) · Fair · Edge · Good until · Found / Checked · Model.
- Then "Dropped in the last 24 hours", greyed (`opacity:.5`).
- Empty state: "No bets beat Pinnacle's fair price right now."
- At < 640px the table becomes stacked cards, using the existing board card classes.
- The page refreshes on the site's existing 5-minute auto-render.

- [ ] **Step 1: Write the failing tests**, with FILES = the ev.test.mjs list plus `js/pages/take.js`:

```js
test("takeBetLabel", () => {
  assert.equal(g.takeBetLabel({ league: "epl", market: "ml", side: "draw" }), "Draw");
  assert.equal(g.takeBetLabel({ league: "mlb", market: "total", side: "over", line: 8.5 }), "Over 8.5");
});
test("takeEdge marks better line", () => {
  assert.equal(g.takeEdge({ ev: 0.021, better_line: true }), "≥ 2.1%");
  assert.equal(g.takeEdge({ ev: 0.021, better_line: false }), "2.1%");
});
test("takeSplit drops started and inactive picks", () => { /* active future -> live; active past -> dropped; inactive -> dropped */ });
test("takeModelNote agrees/disagrees/blank", () => { /* nhl -> ""; ml -> ""; nfl spread with pred -> agrees when lean side === p.side */ });
test("takeProgress ignores October and no_close", () => { /* 2 Nov clv rows + 1 Oct + 1 no_close -> graded 2 */ });
test("takeCountdown", () => { assert.equal(g.takeCountdown("2026-11-02T18:00:00Z", new Date("2026-11-02T15:45:00Z")), "2h 15m"); });
```

- [ ] **Step 2: Run** `cd site && npm test`. Expected: the new tests FAIL and the existing ones pass.

- [ ] **Step 3: Implement.** `take.html` copies `ev.html` with `data-page="take"`, adds `js/pages/take.js` before `js/boot.js`, keeps `shrink-k.js`/`shrink.js`, and sets the title "Take These · CappingAlpha".

- [ ] **Step 4: Run** `cd site && npm test`. Expected: PASS.

- [ ] **Step 5: Commit** `feat(site): take-these page and nav entries`.

---

### Task 9: Scorecard page

**Files:**
- Create: `site/scorecard.html`, `site/js/pikkit.js`, `site/js/pages/scorecard.js`
- Test: `site/tests/pikkit.test.mjs`, `site/tests/scorecard.test.mjs`

**Interfaces:**
- `pikkit.js` holds pure functions only, with no DOM:
  - `parseCsv(text) -> object[]`: RFC 4180 with a header row. It strips a leading BOM and handles quoted fields containing commas, `""` escapes and newlines.
  - `normBook(name) -> string`:
    - Draftkings Sportsbook → DraftKings
    - Fanduel Sportsbook → FanDuel
    - Caesars Sportsbook → Caesars
    - Hard Rock Sportsbook → Hard Rock
    - Fanatics, BetMGM, Bet365 and Novig stay as they are
    - otherwise strip a trailing " Sportsbook".
  - `legMarket(leg) -> "moneyline"|"spread"|"total"|"prop"|"other"`, checked in this order:
    1. a period marker (`/\b(1st|2nd|3rd|\d+th)\b|Inning|Half|Quarter|Period/i`) → other;
    2. `/Moneyline/i` → moneyline;
    3. `/\bSpread\b|Run Line|Puck Line/i` → spread;
    4. `/^(Over|Under) [\d.]+ Total\b/i` → total;
    5. `/^(Over|Under) [\d.]+ |\d+\+|Yds|Yards|Touchdown|Scorer|Hits|Strikeouts|Doubles|Rebounds|Assists|Points|Shots|Saves/i` → prop;
    6. otherwise other.
  - `toBets(rows, unitSize) -> bet[]`. Each bet is `{id, book, type, status, odds, close, placed (Date), legs, market, league, tag, units, profitUnits}`.
    - VOID rows are dropped.
    - `legs` = the number of `bet_info` parts split on " | ".
    - `market` = the shared leg market, or "other" when legs differ.
    - `league` = `leagues`, with "Multiple" when it contains " | ".
    - `tag` = `tags || "untagged"`.
    - `close` = number or null.
  - `beat(b) -> bool|null` (`odds > close`); `priceVsClose(b) -> number|null` (`odds / close − 1`).
  - `summary(bets) -> {bets, withClose, beatRate, avgVsClose, units, profitUnits}`.
  - `weekKey(date) -> "YYYY-MM-DD"`: the Monday of its week in America/New_York.
  - `weekly(bets) -> [{week, ...summary}]`, newest first.
  - `rolling(bets, n = 100) -> [{placed, rate}]`: over bets with a close in placed order, one point per bet, with a window of up to n.
  - `legBucket(n) -> "1"|"2"|"3"|"4"|"5+"`; `unitBucket(u) -> "≤0.5u"|"≤1u"|"≤2u"|">2u"`.
  - `DIMENSIONS = {league, book, market, type, legs, stake, tag}`, as accessors.
  - `breakdowns(bets) -> {dim: [{value, ...summary}]}`.
  - `alert(bets) -> null | {rate, slices: [{dim, value, n, rate, priorRate}]}`.
    - current = the last 100 bets with a close; prior = the 100 before that.
    - It fires when `current rate < 0.40`.
    - Slices have ≥ 15 bets in the current window and a rate < 0.40, sorted ascending by rate. `priorRate` is null when the slice has no bets in prior.
- `pages/scorecard.js`:
  - `buildScorecardPage()` / `wireScorecardPage()`: a drop zone plus a file input, read with FileReader and never sent.
  - The CSV text is stored under localStorage key `scorecard.csv` (try/catch), and the unit size under `scorecard.unit` (default 1; the input saves on change).
  - The view has, in order: the alert banner, headline summary, weekly table, rolling-100 line (the existing site SVG line helper from `js/ui.js`, or a simple inline SVG), breakdown tables (one per dimension), and the side-by-side.
  - **Side-by-side:** the list's beat rate from `take_test_v` (graded clv/direction picks with first_seen within the user's first and last placed dates) next to the user's rate over the same dates, with the caption "Pikkit's close is not necessarily Pinnacle's."
  - Pure helper: `scSideBySide(bets, testRows) -> {mine, list, from, to}`.

- [ ] **Step 1: Write the failing tests** (synthetic rows only; never the user's real CSV):

```js
test("parseCsv handles BOM, quotes, blank close", () => {
  const rows = g.parseCsv('﻿bet_id,odds,closing_line,bet_info\n1,1.91,,"Over 8.5 Total Runs, live | Yankees Moneyline"\n');
  assert.equal(rows[0].bet_id, "1"); assert.equal(rows[0].closing_line, ""); assert.match(rows[0].bet_info, /, live \| /);
});
test("normBook", () => { assert.equal(g.normBook("Draftkings Sportsbook"), "DraftKings"); assert.equal(g.normBook("Novig"), "Novig"); });
test("legMarket", () => {
  assert.equal(g.legMarket("Duke Blue Devils -6.0 Spread Duke Blue Devils at Georgia Tech Yellow Jackets"), "spread");
  assert.equal(g.legMarket("New York Yankees -1.5 Run Line Baltimore Orioles @ New York Yankees"), "spread");
  assert.equal(g.legMarket("PHI Flyers Moneyline PHI Flyers @ DET Red Wings"), "moneyline");
  assert.equal(g.legMarket("Under 4.5 Total Runs - 1st 3 Innings ARI Diamondbacks @ COL Rockies"), "other");
  assert.equal(g.legMarket("Under 0.5 Nolan Arenado - Hits Miami Marlins at Arizona Diamondbacks"), "prop");
  assert.equal(g.legMarket("Over 47.5 Total Points BUF Bills @ KC Chiefs"), "total");
});
test("toBets drops VOID, counts legs, tags untagged", () => { /* SETTLED_VOID dropped; 3-leg parlay legs 3; blank tags -> "untagged" */ });
test("beat and priceVsClose", () => { /* odds 2.0 close 1.9 -> true, ≈0.0526; close null -> null, null */ });
test("weekKey is Monday in US Eastern", () => {
  assert.equal(g.weekKey(new Date("2026-10-19T03:00:00Z")), "2026-10-12");   // Sun 23:00 ET belongs to the prior week
  assert.equal(g.weekKey(new Date("2026-10-19T05:00:00Z")), "2026-10-19");
});
test("unitBucket and legBucket edges", () => { assert.equal(g.unitBucket(0.5), "≤0.5u"); assert.equal(g.unitBucket(2.01), ">2u"); assert.equal(g.legBucket(7), "5+"); });
test("alert fires below 40% and lists slices worst first with prior rate", () => { /* 200 bets: prior 50%, current 30%; DraftKings 20 bets at 20% listed before FanDuel 16 at 35%; slice with 10 bets omitted */ });
test("alert null at or above 40%", () => {});
test("scSideBySide uses the user's date range", () => {});
```

- [ ] **Step 2: Run** `cd site && npm test`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.** `scorecard.html` copies `ev.html` with `data-page="scorecard"` and adds `js/pikkit.js` and `js/pages/scorecard.js` before `js/boot.js`.

- [ ] **Step 4: Run** `cd site && npm test`. Expected: PASS.

- [ ] **Step 5: Commit** `feat(site): Pikkit scorecard (browser-only)`.

---

### Task 10: Cache bump, go-live doc and verification

**Files:**
- Modify: every `site/*.html` (`?v=20261016a` → `?v=20261017a`)
- Create: `docs/superpowers/take-these-go-live.md`
- Modify: the memory file `take-these-clv.md`, outside the repo

- [ ] **Step 1:** Run `grep -rl "v=20261016a" site | xargs sed -i '' 's/v=20261016a/v=20261017a/g'`. Then `grep -rn "v=20261016a" site` should print nothing.

- [ ] **Step 2: Write the go-live doc**, in order:
  1. Paste `db/migration_take_these.sql` into the Supabase SQL editor.
  2. Run "take-these-pull" with dry_run, which costs about 3 credits per in-season league. Check the "books seen / missing" lines; Bet365 is not offered by the Odds API for these leagues (2026-10-17).
  3. Verify that the 8 sport keys appear in the `/sports` output (the dry run prints any missing one).
  4. Set repo variable `TAKE_THESE=on`, and optionally `TAKE_PULL_EVERY_HOURS`. For October, leave it off or run dispatches by hand, since credits are short.
  5. Deploy the site.
  6. Rollback: set `TAKE_THESE` to off, and pause the three workflows.

  Also add to `take_pull.py --dry-run`: call the free `odds.fetch_sports()` and print any of the 8 keys it does not list. Add a test for this, `test_dry_run_reports_unknown_sport_keys`, in `tests/take/test_take_pull.py`.

- [ ] **Step 3: Run the full suites:**
  - `uv run pytest -q`. Expected: all pass, with no new failures against main.
  - `cd site && npm test`. Expected: all pass.

- [ ] **Step 4: Check the site in a browser.** Start the site with the existing `.claude/launch.json` entry (or a static server on `site/`) and open `take.html` and `scorecard.html`.
  - Before the migration runs, `take.html` should show the site's "Couldn't load data" message. That is acceptable until go-live.
  - `scorecard.html` should render the drop zone. Drop a synthetic CSV built from the test fixture, check the tables and the alert, and check the layout at 375px.

- [ ] **Step 5: Commit** `chore(take): cache bump and go-live checklist`.
