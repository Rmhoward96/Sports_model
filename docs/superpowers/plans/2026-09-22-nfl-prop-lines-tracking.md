# NFL Prop Lines + Sim Prop Accuracy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show the book line + over/under prices next to every sim player-prop projection, and grade the sim's projections (pass/rush/rec yds, receptions, rush attempts) against the line and the actual.

**Architecture:** Fix the silently-broken actuals loader, add a `rush_att` market end-to-end (sim → odds capture → actuals), add a pure prop-line builder whose rows land in a new `nfl_prop_lines` table from the existing `build_ev_props_board` job, and grade via a SQL view joining lines to actuals. The CappingAlpha front-end reads both.

**Tech Stack:** Python 3 / uv / pytest, Supabase Postgres (psycopg), GitHub Actions, vanilla JS front-end (`/Users/ryan/Desktop/CappingAlpha/app.js`, external repo).

**Spec:** `docs/superpowers/specs/2026-09-22-nfl-prop-lines-tracking-design.md`

## Global Constraints

- The user runs every Supabase migration themselves; never apply DDL.
- Secrets live in GitHub Actions; never print or paste them.
- Commit on branch `feat/nfl-prop-lines-tracking`; merge/push only when the user asks.
- `+EV` prop pick selection (`assemble_prop_rows`, `ev_prop_picks`) must not change behavior; `SIM_TO_ODDS_MARKET` stays as-is.
- Tracked markets: `pass_yds, rush_yds, rec_yds, receptions, rush_att`.
- Odds budget: one extra daily full-slate prop pull (15:00 UTC); props use `regions=us`.
- Run tests with `uv run pytest`.

**Ruling (spec deviation):** the spec says `SIM_TO_ODDS_MARKET` gains pass_yds/rush_att. Instead a separate `LINE_MARKETS` map is used for the line builder so the +EV path is untouched by construction (tests pin `SIM_TO_ODDS_MARKET`). Cost if wrong: none functionally.

---

### Task 1: Fix actuals loader + `rush_att` actual

**Files:**
- Modify: `src/sportsmodel/nfl/weekly_actuals.py` (WEEKLY_STAT_COLS)
- Modify: `scripts/capture_nfl_player_actuals.py` (`_weekly_for`, imports)
- Modify: `scripts/grade_ev_props.py` (`_weekly_for`, imports)
- Test: `tests/nfl/test_weekly_actuals.py`

**Produces:** `nfl_player_actuals.market` may now be `rush_att` (value = nflverse `carries`).

- [ ] **Step 1: Failing test** — in `tests/nfl/test_weekly_actuals.py` replace `test_all_six_markets_mapped` with:

```python
def test_all_markets_mapped():
    assert set(WEEKLY_STAT_COLS) == {
        "pass_yds", "pass_tds", "rush_yds", "rec_yds", "receptions", "anytime_td", "rush_att"}


def test_rush_att_reads_carries():
    assert player_market_actual(pd.Series({"carries": 17}), "rush_att") == 17.0
```

- [ ] **Step 2:** `uv run pytest tests/nfl/test_weekly_actuals.py -q` → FAIL.
- [ ] **Step 3: Implement** — add `"rush_att": ("carries",),` to `WEEKLY_STAT_COLS`. In both scripts replace the weekly loader:

```python
def _weekly_for(season: int) -> pd.DataFrame | None:
    if season not in _weekly_cache:
        # nfl_data_py's import_weekly_data points at a retired nflverse URL
        # (silently empty); read the canonical stats_player_week release.
        _weekly_cache[season] = load_release("weekly", [season], required=False)
    return _weekly_cache[season]
```

and change the imports: drop `import nfl_data_py as nfl_data_py_import` and `import_by_season` if now unused; add `from sportsmodel.nfl.nflverse import load_release`. Update the capture script's module docstring "six projected markets" → list incl. rush_att.
- [ ] **Step 4:** `uv run pytest tests/nfl/test_weekly_actuals.py tests/test_grade_ev_props.py -q` → PASS.
- [ ] **Step 5: Commit** `fix(nfl): read player actuals from the live nflverse release + rush_att`.

### Task 2: Sim emits `rush_att`

**Files:**
- Modify: `src/sportsmodel/sim/nfl/kernel.py` (`_PLAYER_STAT_NAMES`, box-stat init, carry loop)
- Modify: `src/sportsmodel/sim/nfl/aggregate.py` (pmf market list)
- Modify: `scripts/generate_sim_nfl.py`, `scripts/backtest_sim_nfl.py` (`MARKET_MAX`)
- Test: `tests/sim/nfl/test_kernel_simulate.py`

**Produces:** `nfl_player_sim.market = 'rush_att'` rows (pmf over 0..40 carries).

- [ ] **Step 1: Failing tests** — in `test_player_stats_keys_equal_full_roster` change the key set to include `"rush_att"`; add:

```python
def test_rush_att_counts_carries_and_bounds_rush_yds_volume():
    sims = simulate_game(_spec(), 400, np.random.default_rng(7))
    total_att = sum(s["rush_att"].sum() for s in sims.player_stats.values())
    assert total_att > 0
    for s in sims.player_stats.values():
        assert (s["rush_att"] >= 0).all()
        # a player with zero carries in a sim has zero rush yards that sim
        assert (s["rush_yds"][s["rush_att"] == 0] == 0).all()
```

- [ ] **Step 2:** `uv run pytest tests/sim/nfl/test_kernel_simulate.py -q` → FAIL (KeyError rush_att).
- [ ] **Step 3: Implement** — kernel: `_PLAYER_STAT_NAMES = ("pass_yds", "rush_yds", "rec_yds", "receptions", "td", "pass_tds", "rush_att")`; box init dict adds `"rush_att": 0`; in the carry loop add `pdata["rush_att"] += int(n_carries)` before the per-carry loop. Aggregate list becomes `["pass_yds", "rush_yds", "rec_yds", "receptions", "pass_tds", "rush_att"]`. Both `MARKET_MAX` dicts add `"rush_att": 40`. Update docstrings listing box keys.
- [ ] **Step 4:** `uv run pytest tests/sim -q` → PASS.
- [ ] **Step 5: Commit** `feat(sim/nfl): emit per-player rush attempts (rush_att)`.

### Task 3: Capture rush-attempt lines + daily full-slate prop pull

**Files:**
- Modify: `src/sportsmodel/sports.py` (NFL `prop_market_map`)
- Modify: `scripts/ingest_odds.py` (add `prop_window_minutes`, use it)
- Modify: `.github/workflows/capture-odds.yml` (15:00 UTC daily cron + `PROP_SCOPE`)
- Test: `tests/test_ingest_odds.py`, `tests/test_sports.py`, `tests/test_build_ev_props_board.py`

- [ ] **Step 1: Failing tests** — append `"player_rush_attempts"` to the pinned lists in `test_nfl_prop_markets_match_prop_market_map_values` and `test_nfl_config_present_with_seven_prop_markets` (rename → `..._prop_markets`), and `"rush_att"` to the code set in `test_nfl_prop_market_codes_match_sport_config_prop_market_map_keys`. Add to `tests/test_ingest_odds.py`:

```python
def test_prop_window_minutes_default_and_env():
    assert ingest_odds.prop_window_minutes({}) == 150
    assert ingest_odds.prop_window_minutes({"PROP_WINDOW_MIN": "90"}) == 90
    assert ingest_odds.prop_window_minutes({"PROP_WINDOW_MIN": ""}) == 0


def test_prop_window_minutes_slate_scope_covers_a_week():
    assert ingest_odds.prop_window_minutes({"PROP_SCOPE": "slate"}) == 7 * 24 * 60
    assert ingest_odds.prop_window_minutes({"PROP_SCOPE": "SLATE", "PROP_WINDOW_MIN": "90"}) == 7 * 24 * 60
```

- [ ] **Step 2:** run those files → FAIL.
- [ ] **Step 3: Implement** — `sports.py` NFL map: add `"rush_att": "player_rush_attempts",` after `rush_reception_yds`. `ingest_odds.py`:

```python
SLATE_WINDOW_MIN = 7 * 24 * 60


def prop_window_minutes(env: dict[str, str]) -> int:
    """Prop-capture window in minutes. PROP_SCOPE=slate (the daily full-slate
    pull) widens it to a week so every upcoming game's props are captured;
    otherwise PROP_WINDOW_MIN (default 150; blank -> 0 = disabled)."""
    if env.get("PROP_SCOPE", "").lower() == "slate":
        return SLATE_WINDOW_MIN
    return int(env.get("PROP_WINDOW_MIN", "150") or 0)
```

and in `run_sport` replace `window_min = int(os.getenv("PROP_WINDOW_MIN", "150") or 0)` with `window_min = prop_window_minutes(os.environ)`. Update the module docstring. Workflow: add `- cron: "0 15 * * *"   # daily full-slate prop pull (PROP_SCOPE=slate)`, a `workflow_dispatch` input `prop_scope` (string, default ""), and env `PROP_SCOPE: ${{ github.event.inputs.prop_scope || (github.event.schedule == '0 15 * * *' && 'slate' || '') }}`.
- [ ] **Step 4:** `uv run pytest tests/test_ingest_odds.py tests/test_sports.py tests/test_build_ev_props_board.py -q` → PASS.
- [ ] **Step 5: Commit** `feat(odds): capture rush-attempt props + daily full-slate prop pull`.

### Task 4: Pure prop-line builder

**Files:**
- Modify: `src/sportsmodel/serving/props_ev.py` (add `LINE_MARKETS`, `assemble_prop_line_rows`)
- Test: `tests/serving/test_props_ev.py`

**Produces:** `assemble_prop_line_rows(sim_rows: list[dict], odds_rows: list[dict]) -> list[dict]` with keys exactly `game_pk, player_id, player_name, team, market, line, over_price, over_book, under_price, under_book, projection, p_over, lean, n_books, commence_time`.

- [ ] **Step 1: Failing tests**:

```python
from sportsmodel.serving.props_ev import LINE_MARKETS, assemble_prop_line_rows

def _sim(market="rec_yds", mean=40.0, pmf=None, name="A.J. Brown"):
    pmf = pmf or [0.0] * 30 + [0.02] * 50   # mass on 30..79
    return {"game_pk": 1, "player_id": "00-1", "name": name, "team": "PHI",
            "market": market, "mean": mean, "dist": {"kind": "pmf", "pmf": pmf},
            "commence_time": "2026-09-27T17:00:00Z"}

def _o(market, side, line, book, price, name="AJ Brown"):
    return {"game_pk": 1, "market": market, "side": side, "player_name": name,
            "book": book, "line": line, "price": price}

def test_line_markets_cover_tracked_markets():
    assert LINE_MARKETS == {"pass_yds": "pass_yds", "rush_yds": "rush_yds",
                            "rec_yds": "reception_yds", "receptions": "receptions",
                            "rush_att": "rush_att"}

def test_line_row_main_line_best_prices_and_lean():
    odds = [_o("reception_yds", "over", 54.5, "draftkings", -110),
            _o("reception_yds", "under", 54.5, "draftkings", -110),
            _o("reception_yds", "over", 54.5, "fanduel", -105),
            _o("reception_yds", "under", 54.5, "fanduel", -120),
            _o("reception_yds", "over", 60.5, "fanatics", +100)]
    [r] = assemble_prop_line_rows([_sim()], odds)
    assert r["line"] == 54.5 and r["n_books"] == 2
    assert (r["over_book"], r["over_price"]) == ("fanduel", -105)
    assert (r["under_book"], r["under_price"]) == ("draftkings", -110)
    assert r["projection"] == 40.0 and r["market"] == "rec_yds"
    assert abs(r["p_over"] - 0.5) < 1e-9 and r["lean"] == "under"   # 25 bins of 0.02 above 54.5

def test_line_row_no_usage_gate_and_pass_yds_included():
    sim = _sim(market="pass_yds", mean=12.0, name="Backup QB")
    odds = [_o("pass_yds", "over", 10.5, "draftkings", -110, "Backup QB"),
            _o("pass_yds", "under", 10.5, "draftkings", -110, "Backup QB")]
    assert len(assemble_prop_line_rows([sim], odds)) == 1

def test_line_row_skips_unmatched_untracked_and_one_sided_ok():
    assert assemble_prop_line_rows([_sim(market="anytime_td")], []) == []
    assert assemble_prop_line_rows([_sim()], []) == []
    [r] = assemble_prop_line_rows([_sim()], [_o("reception_yds", "over", 54.5, "draftkings", -110)])
    assert r["under_price"] is None and r["over_price"] == -110
```

(Adjust the p_over assertion to the computed value if the fixture's arithmetic differs: bins 55..79 = 25 × 0.02 = 0.5 → lean must be "under" because `p_over > 0.5` is required for "over".)
- [ ] **Step 2:** `uv run pytest tests/serving/test_props_ev.py -q` → FAIL (ImportError).
- [ ] **Step 3: Implement** in `props_ev.py` after `assemble_prop_rows`:

```python
# Sim market -> odds-side code for the prop-LINES table (every projection with a
# book line, tracked for accuracy). Separate from SIM_TO_ODDS_MARKET so the +EV
# board's market scope is untouched: this one includes pass_yds and rush_att.
LINE_MARKETS: dict[str, str] = {
    "pass_yds": "pass_yds",
    "rush_yds": "rush_yds",
    "rec_yds": "reception_yds",
    "receptions": "receptions",
    "rush_att": "rush_att",
}


def _best_any(entries: list[tuple]) -> tuple | None:
    """(book, price) at the highest decimal odds across ALL books (the line
    display shows the best available price, not only MAJOR_BOOKS)."""
    entries = [(bk, p) for bk, p in entries if p]
    return max(entries, key=lambda e: decimal_odds(e[1])) if entries else None


def assemble_prop_line_rows(sim_rows: list[dict], odds_rows: list[dict]) -> list[dict]:
    """PURE. One nfl_prop_lines row per sim (game, player, market) in
    LINE_MARKETS that has a book line: the main line (most books, ties ->
    lowest), best over/under price across books (either may be None), the
    sim mean as `projection`, P(over) from the sim distribution at the line,
    and the sim's `lean` ("over" iff p_over > 0.5). No usage gate, no EV
    filter. Rows with no odds match or a NaN P(over) are skipped."""
    by_key: dict[tuple, list[dict]] = {}
    for o in odds_rows:
        by_key.setdefault(
            (o.get("game_pk"), o.get("market"), normalize_player_name(o.get("player_name"))), []
        ).append(o)
    rows: list[dict] = []
    for s in sim_rows:
        odds_market = LINE_MARKETS.get(s.get("market"))
        if odds_market is None or s.get("mean") is None:
            continue
        cands = [o for o in by_key.get((s.get("game_pk"), odds_market,
                                        normalize_player_name(s.get("name"))), [])
                 if o.get("line") is not None]
        if not cands:
            continue
        books_by_line: dict[float, set] = {}
        for o in cands:
            books_by_line.setdefault(o["line"], set()).add(o.get("book"))
        line = min(books_by_line, key=lambda l: (-len(books_by_line[l]), l))
        at = [o for o in cands if o["line"] == line]
        over = _best_any([(o.get("book"), o.get("price")) for o in at if o.get("side") == "over"])
        under = _best_any([(o.get("book"), o.get("price")) for o in at if o.get("side") == "under"])
        p_over = prob_over_dist(s.get("dist"), line)
        if p_over != p_over:  # NaN
            continue
        rows.append({
            "game_pk": s.get("game_pk"), "player_id": s.get("player_id"),
            "player_name": s.get("name"), "team": s.get("team"), "market": s.get("market"),
            "line": line,
            "over_price": over[1] if over else None, "over_book": over[0] if over else None,
            "under_price": under[1] if under else None, "under_book": under[0] if under else None,
            "projection": float(s["mean"]), "p_over": float(p_over),
            "lean": "over" if p_over > 0.5 else "under",
            "n_books": len(books_by_line[line]), "commence_time": s.get("commence_time"),
        })
    return rows
```

(`dist` may arrive as a JSON string from the DB — if `isinstance(dist, str)`, `json.loads` it first; add `import json` and handle in the function. Add a test for the string case.)
- [ ] **Step 4:** `uv run pytest tests/serving -q` → PASS.
- [ ] **Step 5: Commit** `feat(props): pure prop-line builder for every sim projection`.

### Task 5: `nfl_prop_lines` table, grading view, upsert, board wiring

**Files:**
- Create: `db/migration_nfl_prop_lines.sql`
- Modify: `src/sportsmodel/db.py` (add `_NFL_PROP_LINES_COLS`, `upsert_nfl_prop_lines`)
- Modify: `scripts/build_ev_props_board.py` (add `team` already in SIM_COLS; write lines; `--lines-only`)
- Modify: `.github/workflows/build-ev-props.yml` (daily `35 15 * * *` cron running `--lines-only`)
- Test: `tests/test_db_nfl_prop_lines.py` (new, FakeConn pattern from `tests/test_db_nfl_player_actuals.py`)

- [ ] **Step 1: Migration** `db/migration_nfl_prop_lines.sql`:

```sql
-- nfl_prop_lines: the book line + best over/under price for EVERY sim player
-- projection that has a line (no usage gate, no +EV filter), and
-- nfl_prop_grades: those rows graded vs nfl_player_actuals. Idempotent.
CREATE TABLE IF NOT EXISTS nfl_prop_lines (
    game_pk        BIGINT NOT NULL,
    player_id      TEXT   NOT NULL,
    player_name    TEXT,
    team           TEXT,
    market         TEXT   NOT NULL,   -- pass_yds|rush_yds|rec_yds|receptions|rush_att
    line           DOUBLE PRECISION NOT NULL,
    over_price     INTEGER,
    over_book      TEXT,
    under_price    INTEGER,
    under_book     TEXT,
    projection     DOUBLE PRECISION,
    p_over         DOUBLE PRECISION,
    lean           TEXT,
    n_books        INTEGER,
    commence_time  TIMESTAMPTZ,
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market)
);
CREATE INDEX IF NOT EXISTS idx_nfl_prop_lines_commence ON nfl_prop_lines (commence_time);

CREATE OR REPLACE VIEW nfl_prop_grades AS
  SELECT l.game_pk, l.player_id, l.player_name, l.team, l.market, l.line,
         l.over_price, l.under_price, l.projection, l.p_over, l.lean,
         l.commence_time, a.actual, a.season, a.week,
         CASE WHEN a.actual = l.line THEN 'push'
              WHEN (l.lean = 'over'  AND a.actual > l.line)
                OR (l.lean = 'under' AND a.actual < l.line) THEN 'hit'
              ELSE 'miss' END            AS result,
         abs(l.projection - a.actual)    AS sim_err,
         abs(l.line - a.actual)          AS line_err,
         l.projection - a.actual         AS sim_bias
  FROM nfl_prop_lines l
  JOIN nfl_player_actuals a
    ON a.game_pk = l.game_pk AND a.player_id = l.player_id AND a.market = l.market
  WHERE l.commence_time <= now();

ALTER TABLE nfl_prop_lines ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_prop_lines" ON nfl_prop_lines;
CREATE POLICY "public read nfl_prop_lines" ON nfl_prop_lines FOR SELECT USING (true);
GRANT SELECT ON nfl_prop_lines, nfl_prop_grades TO anon, authenticated;
```

- [ ] **Step 2: Failing test** `tests/test_db_nfl_prop_lines.py` (copy FakeCursor/FakeConn from `tests/test_db_nfl_player_actuals.py`):

```python
def test_upsert_nfl_prop_lines_shape(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    rec = {c: None for c in db._NFL_PROP_LINES_COLS}
    rec.update(game_pk=1, player_id="00-1", market="rec_yds", line=54.5)
    assert db.upsert_nfl_prop_lines([rec]) == 1
    assert "ON CONFLICT (game_pk, player_id, market) DO UPDATE" in sink["sql"]
    assert "updated_at = now()" in sink["sql"]
    assert sink["rows"][0][db._NFL_PROP_LINES_COLS.index("line")] == 54.5
    assert db.upsert_nfl_prop_lines([]) == 0
```

- [ ] **Step 3:** run → FAIL.
- [ ] **Step 4: Implement** `db.py` (mirror `upsert_nfl_player_actuals`):

```python
_NFL_PROP_LINES_COLS = [
    "game_pk", "player_id", "player_name", "team", "market", "line",
    "over_price", "over_book", "under_price", "under_book",
    "projection", "p_over", "lean", "n_books", "commence_time",
]


def upsert_nfl_prop_lines(records: list[dict]) -> int:
    """Upsert the book line + best prices for each sim prop projection into
    `nfl_prop_lines`. Idempotent on (game_pk, player_id, market): each board
    build overwrites in place (the main line can move), so the last write
    before kickoff is the graded line. Requires db/migration_nfl_prop_lines.sql."""
    if not records:
        return 0
    key = ("game_pk", "player_id", "market")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _NFL_PROP_LINES_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_NFL_PROP_LINES_COLS))
    sql = (
        f"INSERT INTO nfl_prop_lines ({', '.join(_NFL_PROP_LINES_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market) DO UPDATE SET {updates}, updated_at = now()"
    )
    rows = [tuple(r.get(c) for c in _NFL_PROP_LINES_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
```

`build_ev_props_board.py` `main()`: add `parser.add_argument("--lines-only", action="store_true", help="Write nfl_prop_lines only (skip the +EV board)")`; after loading odds:

```python
    line_rows = assemble_prop_line_rows(sim_rows, odds_rows)
    n_lines = upsert_nfl_prop_lines(line_rows)
    print(f"[build_ev_props_board] prop_lines={n_lines}")
    if args.lines_only:
        return
```

(import `assemble_prop_line_rows` and `upsert_nfl_prop_lines`). Workflow `build-ev-props.yml`: add cron `"35 15 * * *"`, a `workflow_dispatch` boolean input `lines_only` (default false), and make the run step pass `--lines-only` for that cron or input: `run: uv run python scripts/build_ev_props_board.py --sport nfl ${{ (github.event.schedule == '35 15 * * *' || github.event.inputs.lines_only == 'true') && '--lines-only' || '' }}`. Also give `capture-player-actuals.yml` a `workflow_dispatch` input `days` (default "8") used as `--days "${{ github.event.inputs.days || '8' }}"` so week 1–2 can be backfilled.
- [ ] **Step 5:** `uv run pytest -q` (full suite) → PASS.
- [ ] **Step 6: Commit** `feat(props): nfl_prop_lines table + grading view, written by the prop board`.

### Task 6: Front-end (CappingAlpha `app.js`)

**Files:** Modify `/Users/ryan/Desktop/CappingAlpha/app.js`, bump `app.js?v=` in `*.html` to `20260922g`.

- [ ] **Step 1: Game page fetch** — add `lines = []` to the `let` and fetch `sb(\`nfl_prop_lines?game_pk=eq.${game}\`).catch(() => [])` in the NFL `Promise.all`; pass `lines` as a 4th arg to `propsProjectionSection(sims, props, playerActuals, lines)`.
- [ ] **Step 2: Cells** — in `propsProjectionSection`: build `lineBy` keyed `${player_id}|${market}`. `fmtAm = (p) => p == null ? "—" : (p > 0 ? `+${p}` : `${p}`)`. `cell` appends, when a line exists, `<span class="prop-line">o/u ${line} · ${fmtAm(over)}/${fmtAm(under)}</span>` and marks the sim's lean side (bold the leaned price). `actualLine`: when a line exists, hit = actual on the lean side of the line (push shows "P"); else keep the current met-projection fallback. Add a `RUSH ATT` column (`cell(p, "rush_att", 1)`) after RUSH YDS; include `rush_att` in `rel()` (× 5). Update the note text: "o/u = main book line · best over/under price; ✓ = actual landed on the sim's side of the line".
- [ ] **Step 3: Track Record** — `propLineGradesRows()` = `sb("nfl_prop_grades?select=market,result,sim_err,line_err,sim_bias&limit=5000").catch(() => [])` added to `buildTrack`'s `Promise.all`; new `propAccuracySection(rows)` rendered in the `data-evview="ev"` block before `propTrackSection`: table per market (Pass Yds / Rush Yds / Rec Yds / Receptions / Rush Att + All) with columns N, LEAN HIT % (hits/(hits+misses)), SIM MAE, LINE MAE, BIAS (mean sim_bias, signed). Empty state: "No graded prop lines yet — fills in after games with captured lines settle." Add `rush_att: "Rush Att"` to `PROP_MKT`.
- [ ] **Step 4: CSS** — `.prop-proj .prop-line{display:block;font-size:10.5px;opacity:.7;margin-top:2px}` and `.prop-proj .prop-line b{opacity:1;color:#8fc1ff}`.
- [ ] **Step 5:** `node --check app.js`; serve locally (`python3 -m http.server 8799` in CappingAlpha) and browser-check an NFL game page + Track Record.
- [ ] **Step 6:** No git in CappingAlpha unless it is a repo; report the files changed.

### Task 7: Rollout + verification

- [ ] Push branch; user runs `db/migration_nfl_prop_lines.sql`.
- [ ] Merge to main (user approval); dispatch `capture-player-actuals` with `days=16` → verify `nfl_player_actuals` rows for weeks 1–2.
- [ ] Dispatch `generate-sim-nfl` (rush_att in `nfl_player_sim`), then `capture-odds` with `prop_scope=slate` and `build-ev-props` with `lines_only=true` → verify `nfl_prop_lines` count by market. (Weeks 1–2 have no stored lines, so grades begin with Week 3.)
- [ ] Browser-check the site after the user redeploys.
