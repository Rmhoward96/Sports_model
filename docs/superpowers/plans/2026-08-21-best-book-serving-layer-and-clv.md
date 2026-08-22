# Best-Book Serving Layer + CLV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Precompute the betting board (best-book prices) and a CLV-tracking bet log into Supabase tables/views so the Lovable "Blue Edge" front-end can render the board and track record from pure reads, replacing the Streamlit app.

**Architecture:** A shared pure-Python pick-math module (`sportsmodel/serving/board.py`) is reused by a new producer (`scripts/generate_board.py`, which writes the live `board_picks` table and appends locked bets to `picks` on first-+EV) and by a refactored `grade_results.py` (which grades the `picks` log at the bet price and computes CLV vs the consensus no-vig close). Two SQL views summarize the track record. All four objects are public-read (RLS).

**Tech Stack:** Python 3.12 (uv), psycopg3 (Supabase Session pooler), numpy, pytest; Supabase Postgres + PostgREST for the front-end.

**Spec:** `docs/superpowers/specs/2026-08-21-best-book-serving-layer-and-clv.md`

## Global Constraints

- **Best-book pricing**: for a side, the best book = highest **decimal** odds (most favorable to bettor); store the book name. Consensus (median per side across books) is used only for the no-vig `implied_prob` and the CLV close reference.
- **CLV**: `clv = novig_close − novig_bet` (fraction; display as %). `novig_bet` at first-+EV capture, `novig_close` at the last capture before `commence_time`.
- **Bet lock**: `picks` rows are insert-once via `ON CONFLICT (game_pk, market, player_id) DO NOTHING` — first +EV price/side/book is permanent.
- **is_pick** = `ev > 0` for every market. Only +EV picks are inserted into `picks`.
- **Fresh-start floor**: reuse `grade_results.FRESH_START = "2026-08-21"`; never grade earlier games.
- **MLB only**: `sport = 'mlb'` on every row. NFL/NBA out of scope.
- **Model versions**: game `mlb-hybrid-v1`, props `mlb-hybrid-props-v1`; read the latest-generated version per game (reuse the dedup helpers).
- **odds_snapshot columns**: `game_pk, market, side, player_name, line, book, price, captured_at, commence_time`.
- **Markets**: `moneyline, spread, total, hits, total_bases, home_run, hrr, pitcher_ks, hits_allowed, outs_recorded`.
- **Commit cadence**: one commit per task (TDD red→green→commit). Branch `serving-layer`.

---

## File Structure

- `src/sportsmodel/serving/__init__.py` — new package.
- `src/sportsmodel/serving/board.py` — pure pick math: `decimal_odds`, `implied_prob`, `novig`, `best_price`, `ev`, and per-market builders (`moneyline_row`, `total_row`, `spread_row`, `prop_row`). Reuses `model.distributions` (`prob_over_dist`, `prob_cover`, `apply_affine`).
- `scripts/generate_board.py` — producer: reads predictions + odds + calibration, builds `board_picks`, appends `picks` on first-+EV.
- `src/sportsmodel/db.py` — add `upsert_board_picks`, `insert_new_picks`, `update_graded_picks`.
- `db/serving_bootstrap.sql` — `board_picks`, `picks` tables, the two views, RLS policies (idempotent; user runs in Supabase).
- `scripts/grade_results.py` — refactor: grade the `picks` log; retire the `prediction_results` writes.
- `docs/serving_contract.md` — the front-end data contract (schema + example `supabase.from()` queries).
- Tests: `tests/test_serving_board.py`, `tests/test_grade_picks.py`.

---

## Phase 1 — Shared pick-math module

### Task 1: Odds primitives (`best_price`, `ev`, `novig`)

**Files:**
- Create: `src/sportsmodel/serving/__init__.py` (empty), `src/sportsmodel/serving/board.py`
- Test: `tests/test_serving_board.py`

**Interfaces:**
- Produces:
  - `decimal_odds(american: int) -> float`
  - `implied_prob(american: int) -> float`
  - `novig(price_side: int, price_other: int) -> float` — no-vig prob of `price_side`.
  - `best_price(entries: list[tuple[str, int]]) -> tuple[str, int] | None` — (book, american) with the highest decimal odds; None if empty.
  - `ev(prob: float, american: int) -> float` — `prob*decimal_odds(american) - 1`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_serving_board.py
import pytest
from sportsmodel.serving import board as b

def test_decimal_and_implied():
    assert b.decimal_odds(100) == pytest.approx(2.0)
    assert b.decimal_odds(-110) == pytest.approx(1.9090909, rel=1e-4)
    assert b.implied_prob(100) == pytest.approx(0.5)

def test_best_price_picks_highest_decimal():
    # +120 (dec 2.2) beats +110 (2.1) and -105 (1.952) for the bettor
    assert b.best_price([("DK", 110), ("FD", 120), ("MGM", -105)]) == ("FD", 120)
    # among negatives, -105 (1.952) beats -120 (1.833)
    assert b.best_price([("DK", -120), ("FD", -105)]) == ("FD", -105)
    assert b.best_price([]) is None

def test_novig_removes_hold():
    # symmetric -110/-110 -> 0.5
    assert b.novig(-110, -110) == pytest.approx(0.5)
    # favorite side gets >0.5
    assert b.novig(-200, 170) > 0.5

def test_ev_sign():
    assert b.ev(0.6, 100) == pytest.approx(0.2)   # 0.6*2 - 1
    assert b.ev(0.4, 100) == pytest.approx(-0.2)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_serving_board.py -v` → ImportError.

- [ ] **Step 3: Implement**

```python
# src/sportsmodel/serving/board.py
"""Pure pick-math for the serving board: best-book selection, EV, no-vig, and
per-market row builders. Shared by scripts/generate_board.py and grade_results.py."""
from __future__ import annotations

from ..model.distributions import apply_affine, prob_cover, prob_over_dist


def decimal_odds(american: int) -> float:
    a = float(american)
    return 1 + (a / 100 if a > 0 else 100 / -a)


def implied_prob(american: int) -> float:
    return 1.0 / decimal_odds(american)


def novig(price_side: int, price_other: int) -> float:
    io, iu = implied_prob(price_side), implied_prob(price_other)
    return io / (io + iu)


def best_price(entries):
    """(book, american) with the highest decimal odds (best for the bettor)."""
    entries = [(bk, p) for bk, p in (entries or []) if p]
    if not entries:
        return None
    return max(entries, key=lambda e: decimal_odds(e[1]))


def ev(prob: float, american: int) -> float:
    return prob * decimal_odds(american) - 1
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_serving_board.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/serving/__init__.py src/sportsmodel/serving/board.py tests/test_serving_board.py
git commit -m "feat(serving): odds primitives (best_price, ev, novig)"
```

---

### Task 2: Moneyline + total + spread row builders

**Files:**
- Modify: `src/sportsmodel/serving/board.py`
- Test: `tests/test_serving_board.py`

**Interfaces:**
- Consumes: Task 1 primitives; `prob_over_dist`, `prob_cover`, `apply_affine`.
- Produces (each returns a dict or None; keys: `market, side, line, pick_label, model_prob, implied_prob, ev, odds, book, is_pick`):
  - `moneyline_row(home_wp, home_entries, away_entries, home_name, away_name) -> dict | None` — favored team by `home_wp` vs no-vig; always returns a row when both prices exist (ML has no pass); `is_pick = ev>0`.
  - `total_row(total_dist, total_cal, main_line, over_entries, under_entries) -> dict | None` — `p_over = prob_over_dist(apply_affine(total_dist,*total_cal), main_line)`; EV each side at best price; return higher-EV side; `is_pick = ev>0`.
  - `spread_row(margin_dist, margin_cal, home_line, home_entries, away_entries, home_name, away_name) -> dict | None` — `p_home = prob_cover(apply_affine(margin_dist,*margin_cal), home_line)`; EV each side; higher-EV side; `is_pick = ev>0`.
- `home_entries`/`over_entries` etc. are `list[tuple[str,int]]` of (book, american) at the relevant line.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_serving_board.py  (add)
def test_moneyline_row_favors_and_prices_best_book():
    row = b.moneyline_row(0.60, [("DK", -130), ("FD", -120)], [("MGM", 110)],
                          "Home", "Away")
    assert row["side"] == "home" and row["pick_label"] == "Home ML"
    assert row["book"] == "FD" and row["odds"] == -120   # best (highest decimal) home price
    assert row["model_prob"] == pytest.approx(0.60)
    assert row["is_pick"] is True   # 0.60 * dec(-120) - 1 > 0

def test_total_row_picks_ev_side_not_mean():
    # right-skewed total: mass at 8 and 12 -> P(over 8.5) < 0.5 -> under side
    pmf = [0.0]*21; pmf[8], pmf[12] = 0.6, 0.4
    dist = {"kind": "pmf", "pmf": pmf}
    row = b.total_row(dist, (0.0, 1.0), 8.5, [("DK", -110)], [("FD", -105)])
    assert row["side"] == "under" and row["pick_label"] == "Under 8.5"
    assert row["model_prob"] == pytest.approx(0.6)   # P(under) = 1 - 0.4

def test_spread_row_uses_cover_prob():
    # margin dist mass at +3 -> home -1.5 covers w.p. 1
    md = {"kind": "margin", "offset": 10, "pmf": [0.0]*21}
    md["pmf"][3+10] = 1.0
    row = b.spread_row(md, (0.0, 1.0), -1.5, [("DK", -140)], [("FD", 120)], "Home", "Away")
    assert row["side"] == "home" and row["pick_label"] == "Home -1.5"
    assert row["model_prob"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_serving_board.py -k "row" -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# add to board.py
def _mkrow(market, side, line, label, model_p, market_p, price, bk):
    e = ev(model_p, price)
    return {"market": market, "side": side, "line": line, "pick_label": label,
            "model_prob": model_p, "implied_prob": market_p, "ev": e,
            "odds": price, "book": bk, "is_pick": e > 0}

def moneyline_row(home_wp, home_entries, away_entries, home_name, away_name):
    hb, ab = best_price(home_entries), best_price(away_entries)
    if hb is None or ab is None:
        return None
    novig_home = novig(hb[1], ab[1])
    if home_wp >= novig_home if False else home_wp >= 0.5:  # placeholder replaced below
        pass
    # favored side = higher model prob
    if home_wp >= 1 - home_wp:
        return _mkrow("moneyline", "home", None, f"{home_name} ML", home_wp, novig_home, hb[1], hb[0])
    return _mkrow("moneyline", "away", None, f"{away_name} ML", 1 - home_wp, 1 - novig_home, ab[1], ab[0])

def total_row(total_dist, total_cal, main_line, over_entries, under_entries):
    ob, ub = best_price(over_entries), best_price(under_entries)
    if ob is None or ub is None:
        return None
    p_over = prob_over_dist(apply_affine(total_dist, *total_cal), main_line)
    if p_over != p_over:  # NaN
        return None
    novig_over = novig(ob[1], ub[1])
    ev_o, ev_u = ev(p_over, ob[1]), ev(1 - p_over, ub[1])
    if ev_o >= ev_u:
        return _mkrow("total", "over", main_line, f"Over {main_line:g}", p_over, novig_over, ob[1], ob[0])
    return _mkrow("total", "under", main_line, f"Under {main_line:g}", 1 - p_over, 1 - novig_over, ub[1], ub[0])

def spread_row(margin_dist, margin_cal, home_line, home_entries, away_entries, home_name, away_name):
    hb, ab = best_price(home_entries), best_price(away_entries)
    if hb is None or ab is None:
        return None
    p_home = prob_cover(apply_affine(margin_dist, *margin_cal), home_line)
    if p_home != p_home:
        return None
    novig_home = novig(hb[1], ab[1])
    ev_h, ev_a = ev(p_home, hb[1]), ev(1 - p_home, ab[1])
    if ev_h >= ev_a:
        return _mkrow("spread", "home", home_line, f"{home_name} {home_line:+g}", p_home, novig_home, hb[1], hb[0])
    return _mkrow("spread", "away", -home_line, f"{away_name} {-home_line:+g}", 1 - p_home, 1 - novig_home, ab[1], ab[0])
```
Note: delete the dead placeholder branch in `moneyline_row` (kept minimal — favored side is the higher model prob).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_serving_board.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/serving/board.py tests/test_serving_board.py
git commit -m "feat(serving): moneyline/total/spread best-book row builders"
```

---

### Task 3: Prop row builder

**Files:** Modify `src/sportsmodel/serving/board.py`; Test `tests/test_serving_board.py`

**Interfaces:**
- Produces: `prop_row(market, dist, cal_target, main_line, over_entries, under_entries) -> dict | None`. `p_over = calibrate(cal_target, prob_over_dist(dist, main_line))` — reuse `model.calibration.calibrate`. Over-only markets (HR) with no under: `ev_under = -inf`. Returns higher-EV side; `is_pick = ev>0`.

- [ ] **Step 1: Write the failing test**

```python
def test_prop_row_over_only_home_run_passes_when_negative_ev():
    # HR longshot: model 12% at +650 (dec 7.5) -> EV 0.12*7.5-1 = -0.10 -> is_pick False
    pmf = [0.88, 0.12]
    row = b.prop_row("home_run", {"kind": "pmf", "pmf": pmf}, "home_run", 0.5,
                     [("DK", 650)], [])
    assert row["side"] == "over" and row["is_pick"] is False

def test_prop_row_picks_over_when_plus_ev():
    pmf = [0.0]*7; pmf[2] = 1.0  # P(over 1.5) = 1.0
    row = b.prop_row("hits", {"kind": "pmf", "pmf": pmf}, "hits", 1.5,
                     [("FD", 120)], [("DK", -140)])
    assert row["side"] == "over" and row["is_pick"] is True
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_serving_board.py -k prop -v` → FAIL.

- [ ] **Step 3: Implement**

```python
# add import at top: from ..model.calibration import calibrate
def prop_row(market, dist, cal_target, main_line, over_entries, under_entries):
    ob = best_price(over_entries)
    if ob is None:
        return None
    p_over = calibrate(cal_target, prob_over_dist(dist, main_line))
    if p_over != p_over:
        return None
    ub = best_price(under_entries)
    ev_o = ev(p_over, ob[1])
    ev_u = ev(1 - p_over, ub[1]) if ub else float("-inf")
    if ev_o >= ev_u:
        return _mkrow(market, "over", main_line, f"Over {main_line:g}", p_over,
                      novig(ob[1], ub[1]) if ub else implied_prob(ob[1]), ob[1], ob[0])
    return _mkrow(market, "under", main_line, f"Under {main_line:g}", 1 - p_over,
                  1 - novig(ob[1], ub[1]), ub[1], ub[0])
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_serving_board.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/serving/board.py tests/test_serving_board.py
git commit -m "feat(serving): prop best-book row builder with over-only handling"
```

---

## Phase 2 — Tables, upserts, views

### Task 4: SQL bootstrap (tables + views + RLS)

**Files:** Create `db/serving_bootstrap.sql`

**Interfaces:** Produces the `board_picks` and `picks` tables, `track_record_segments` and `cumulative_units_weekly` views, and public-read RLS. Idempotent (`IF NOT EXISTS` / `CREATE OR REPLACE`). The user runs it in the Supabase SQL Editor.

- [ ] **Step 1: Write the SQL file** (exact DDL from the spec §Data model)

```sql
-- db/serving_bootstrap.sql
CREATE TABLE IF NOT EXISTS board_picks (
  sport TEXT, game_pk BIGINT, game_date DATE, commence_time TIMESTAMPTZ, matchup TEXT,
  market TEXT, market_label TEXT, player_id BIGINT, player_name TEXT, team TEXT,
  pick_label TEXT, side TEXT, line REAL,
  odds INT, book TEXT, model_prob REAL, implied_prob REAL, ev REAL, is_pick BOOLEAN,
  generated_at TIMESTAMPTZ DEFAULT now(),
  PRIMARY KEY (game_pk, market, player_id)
);
CREATE TABLE IF NOT EXISTS picks (
  game_pk BIGINT, market TEXT, player_id BIGINT,
  sport TEXT, game_date DATE, commence_time TIMESTAMPTZ, matchup TEXT,
  market_label TEXT, player_name TEXT, team TEXT, pick_label TEXT, side TEXT, line REAL,
  bet_odds INT, bet_book TEXT, model_prob REAL, novig_bet REAL, ev_bet REAL,
  bet_at TIMESTAMPTZ DEFAULT now(),
  status TEXT DEFAULT 'pending',
  actual REAL, result TEXT, profit REAL,
  novig_close REAL, clv REAL, graded_at TIMESTAMPTZ,
  PRIMARY KEY (game_pk, market, player_id)
);
CREATE OR REPLACE VIEW track_record_segments AS
  SELECT sport, market,
    count(*) FILTER (WHERE result='win')  AS wins,
    count(*) FILTER (WHERE result='loss') AS losses,
    count(*) FILTER (WHERE result='push') AS pushes,
    round(avg((result='win')::int) FILTER (WHERE result IN ('win','loss')) * 100, 1) AS win_pct,
    round(sum(profit)::numeric, 2) AS units,
    round((sum(profit) / nullif(count(*),0))::numeric * 100, 1) AS roi,
    round(avg(ev_bet)::numeric * 100, 1) AS avg_ev,
    round(avg(clv)::numeric * 100, 1) AS avg_clv
  FROM picks WHERE status='graded' GROUP BY sport, market;
CREATE OR REPLACE VIEW cumulative_units_weekly AS
  SELECT week, units,
         sum(units) OVER (ORDER BY week) AS cumulative_units
  FROM (SELECT date_trunc('week', game_date)::date AS week, sum(profit) AS units
        FROM picks WHERE status='graded' GROUP BY 1) w ORDER BY week;
ALTER TABLE board_picks ENABLE ROW LEVEL SECURITY;
ALTER TABLE picks ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read board_picks" ON board_picks;
CREATE POLICY "public read board_picks" ON board_picks FOR SELECT USING (true);
DROP POLICY IF EXISTS "public read picks" ON picks;
CREATE POLICY "public read picks" ON picks FOR SELECT USING (true);
GRANT SELECT ON track_record_segments, cumulative_units_weekly TO anon, authenticated;
```

- [ ] **Step 2: Hand to user** — instruct the user to run it in Supabase and confirm a subsequent Actions write still succeeds (owner bypasses RLS). No automated test (DDL).

- [ ] **Step 3: Commit**

```bash
git add db/serving_bootstrap.sql
git commit -m "feat(db): board_picks + picks tables, track-record views, public-read RLS"
```

---

### Task 5: DB upsert/insert helpers

**Files:** Modify `src/sportsmodel/db.py`; Test `tests/test_grade_picks.py` (helper-shape test only — no live DB)

**Interfaces:**
- Produces: `upsert_board_picks(rows: list[dict]) -> int` (ON CONFLICT (game_pk,market,player_id) DO UPDATE all cols + generated_at), `insert_new_picks(rows: list[dict]) -> int` (INSERT ... ON CONFLICT (game_pk,market,player_id) DO NOTHING), `update_graded_picks(rows: list[dict]) -> int` (UPDATE status/actual/result/profit/novig_close/clv/graded_at WHERE game_pk,market,player_id).
- Follow the existing `upsert_prediction_results` pattern (cols list, executemany, get_postgres()).

- [ ] **Step 1: Write a column-contract test**

```python
# tests/test_grade_picks.py
from sportsmodel import db
def test_board_and_picks_helpers_exist():
    for fn in ("upsert_board_picks", "insert_new_picks", "update_graded_picks"):
        assert hasattr(db, fn)
```

- [ ] **Step 2: Run to verify fail** — FAIL (AttributeError).

- [ ] **Step 3: Implement** the three helpers in `db.py` mirroring `upsert_prediction_results` (build `INSERT ... VALUES ... ON CONFLICT ...`, `executemany`, commit). `board_picks` cols = the table columns minus `generated_at`; `picks` insert cols = the pre-grade columns; `update_graded_picks` sets the graded columns by PK.

- [ ] **Step 4: Run to verify pass** — PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/db.py tests/test_grade_picks.py
git commit -m "feat(db): board_picks/picks upsert + graded-update helpers"
```

---

## Phase 3 — Producer

### Task 6: `generate_board.py`

**Files:** Create `scripts/generate_board.py`; Test `tests/test_generate_board.py` (pure helpers)

**Interfaces:**
- Consumes: serving `board` module; `db.upsert_board_picks`, `db.insert_new_picks`; `grade_results._latest_per_game`/`_latest_version_props` (import) or re-derive; `calibration.load`.
- Produces: a pure helper `build_rows(game, preds, odds_by_key, cals) -> list[dict]` that assembles board rows for one game across markets (testable), plus a `main()` that reads Supabase (game_predictions/prop_predictions latest version + latest odds per (market,side,line,book,player)), builds rows, upserts `board_picks`, and inserts +EV rows into `picks` (mapping board-row dict → picks row with `bet_odds=odds, bet_book=book, novig_bet=implied_prob, ev_bet=ev, bet_at=now`).

- [ ] **Step 1: Write the failing test** for `build_rows` (feed one synthetic game's predictions + odds; assert it yields a moneyline row, a total row, and the props with best-book prices and correct `is_pick`). Use the board module builders so the test is about wiring, not math.

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** `build_rows` + `main`. `main` SQL:
  - Latest game preds: reuse `_latest_per_game` over `SELECT game_pk, game_date, model_version, home_team_name, away_team_name, generated_at, commence_time, pred_total, total_dist, pred_margin, margin_dist, home_win_prob FROM game_predictions WHERE game_date >= today`.
  - Latest odds: `SELECT market, side, player_name, line, book, price FROM (SELECT DISTINCT ON (market,side,player_name,line,book) ... FROM odds_snapshot WHERE game_pk=%s AND captured_at<=commence_time ORDER BY ..., captured_at DESC) t` — group into `{(market,side,player,line): [(book,price)...]}` and pick main line = most-booked (reuse the consensus main-line rule).
  - Build rows via the board module; set `sport='mlb'`, `matchup`, `market_label`, `commence_time`.
  - `upsert_board_picks(all_rows)`; `insert_new_picks([r for r in all_rows if r['is_pick']] mapped to picks shape)`.

- [ ] **Step 4: Run to verify pass** + a live smoke run:
```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/generate_board.py   # requires DATABASE_URL
```
Expected: prints N board rows upserted, M picks inserted.

- [ ] **Step 5: Commit**

```bash
git add scripts/generate_board.py tests/test_generate_board.py
git commit -m "feat(serving): generate_board producer (board_picks + first-+EV picks log)"
```

---

## Phase 4 — Grading refactor + CLV

### Task 7: Grade the `picks` log at bet price + CLV

**Files:** Modify `scripts/grade_results.py`; Test `tests/test_grade_picks.py`

**Interfaces:**
- Produces: `grade_pick(pick_row, res, close_consensus) -> dict | None` — pure: given a pending pick, the game result, and the consensus no-vig close for the picked side, return the graded fields (`actual, result, profit, novig_close, clv`). `profit = decimal(bet_odds)-1` on win, `-1` on loss, `0` push. `clv = novig_close - novig_bet`. Reuses `_grade_ou` semantics for side/line outcome per market.
- `main()` changes: instead of the game/prop lean loop writing `prediction_results`, load pending `picks` for final games, compute `close_consensus` per pick (reuse `closing_lines` → consensus no-vig for the picked side), call `grade_pick`, and `db.update_graded_picks(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grade_picks.py  (add)
import importlib.util, pathlib
_p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "grade_results.py"
_s = importlib.util.spec_from_file_location("grade_results", _p)
gr = importlib.util.module_from_spec(_s); _s.loader.exec_module(gr)

def test_grade_pick_win_profit_and_clv():
    pick = {"game_pk":1,"market":"total","player_id":0,"side":"under","line":8.5,
            "bet_odds":-105,"novig_bet":0.50}
    # actual total 7 -> under wins; consensus no-vig close for under = 0.54
    out = gr.grade_pick(pick, actual=7.0, novig_close=0.54)
    assert out["result"] == "win"
    assert out["profit"] > 0
    assert out["clv"] == pytest.approx(0.04)   # 0.54 - 0.50

def test_grade_pick_loss_is_minus_one_unit():
    pick = {"game_pk":1,"market":"moneyline","player_id":0,"side":"home","line":None,
            "bet_odds":120,"novig_bet":0.45}
    out = gr.grade_pick(pick, actual=-1.0, novig_close=0.47)  # actual: home lost (margin<0)
    assert out["result"] == "loss" and out["profit"] == -1.0
    assert out["clv"] == pytest.approx(0.02)
```
(The test passes the already-decided `actual` outcome value per market — `grade_pick` maps side/line/actual → win/loss/push; define `actual` semantics per market in the impl: moneyline `actual` = margin sign, total `actual` = total runs, spread `actual` = margin, props `actual` = the stat.)

- [ ] **Step 2: Run to verify fail.**

- [ ] **Step 3: Implement** `grade_pick` (a small dispatch on `market` reusing `_grade_ou`-style comparisons and `_decimal`), then rewrite `main()`'s write path to grade `picks` and call `update_graded_picks`. Remove the `prediction_results` append/`upsert_prediction_results` call. Keep `FRESH_START`, the finals filter, and `closing_lines`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_grade_picks.py tests/test_grade_results.py -q`. (Delete now-obsolete `_grade_game`/`prediction_results` tests, or keep `_grade_game` pure tests if the function is retained for reuse; prefer moving its per-market win/loss logic into `grade_pick`.)

- [ ] **Step 5: Commit**

```bash
git add scripts/grade_results.py tests/test_grade_picks.py
git commit -m "feat(grade): grade the picks log at bet price + CLV vs consensus close"
```

---

## Phase 5 — Wiring, contract, retirement

### Task 8: Workflow wiring + front-end contract + retire Streamlit

**Files:** Modify `.github/workflows/capture-odds.yml`; Create `docs/serving_contract.md`; (later) remove `streamlit_app.py`.

- [ ] **Step 1** — Add a `generate_board` step to `capture-odds.yml` after ingest (same `DATABASE_URL` env), so the board refreshes with each capture:
```yaml
      - name: Regenerate serving board (best-book + picks log)
        env: { DATABASE_URL: ${{ secrets.DATABASE_URL }} }
        run: uv run python scripts/generate_board.py
```

- [ ] **Step 2** — Write `docs/serving_contract.md`: the `board_picks`/`picks` schemas, the two views, and example queries (board = `board_picks` where `sport='mlb' and game_date=today`; top edges = order by `ev desc`; track record = `track_record_segments` + `cumulative_units_weekly`), with the Supabase anon-key REST usage. This is what the user hands to Lovable.

- [ ] **Step 3** — Commit workflow + contract:
```bash
git add .github/workflows/capture-odds.yml docs/serving_contract.md
git commit -m "chore(serving): run generate_board after capture; add front-end contract"
```

- [ ] **Step 4** — After the user confirms parity in Lovable for one slate, remove the Streamlit app and drop `prediction_results` (separate commit; user runs the `DROP TABLE prediction_results;` when ready):
```bash
git rm streamlit_app.py requirements.txt
git commit -m "chore: retire Streamlit dashboard (replaced by Lovable serving layer)"
```

---

## Self-Review

**Spec coverage:**
- board_picks (live board, best-book) → Tasks 2,3,4,6. ✓
- picks (first-+EV lock) → Tasks 4,5,6. ✓
- best-book pricing + book name → Task 1 (`best_price`) + builders. ✓
- CLV (novig_close − novig_bet) → Task 7. ✓
- Grading at bet price, retire prediction_results → Task 7, Task 8 step 4. ✓
- track-record views → Task 4. ✓
- RLS public read → Task 4. ✓
- producer after capture → Task 8. ✓
- shared pick math → Tasks 1-3 (`serving/board.py`). ✓
- front-end contract → Task 8. ✓
- MLB only → Global Constraints (`sport='mlb'`). ✓

**Placeholder scan:** Task 2's `moneyline_row` intentionally shows and then removes a dead placeholder branch — the implementer note says delete it; the correct logic (favored = higher model prob) is present. No other placeholders.

**Type consistency:** row-dict keys (`market, side, line, pick_label, model_prob, implied_prob, ev, odds, book, is_pick`) are consistent across Tasks 2-3-6; `picks` mapping in Task 6 renames `odds→bet_odds, book→bet_book, implied_prob→novig_bet, ev→ev_bet`. `grade_pick` (Task 7) consumes `bet_odds, novig_bet, side, line, market`. Consistent.
