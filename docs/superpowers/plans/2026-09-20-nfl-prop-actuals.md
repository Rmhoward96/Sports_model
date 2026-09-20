# NFL Prop Actuals Pipeline Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture each finished NFL game's actual player box stats (nflverse weekly) into a new `nfl_player_actuals` table and show them beside the sim's projected props on the game page.

**Architecture:** A pure resolver module (`weekly_actuals`) maps the six projected markets to nflverse weekly columns and resolves a game's NFL week via the schedules `espn` join. A capture script reads finished games' projected player_ids from `nfl_player_sim`, resolves each player's realized stats, and upserts them. A scheduled workflow runs it. The external `app.js` renders an actual + ✓/✗ under each projection.

**Tech Stack:** Python 3.12 (uv), nfl_data_py, psycopg (Supabase), pytest; external CappingAlpha `app.js` (vanilla JS).

**Spec:** docs/superpowers/specs/2026-09-20-nfl-prop-actuals-design.md

## Global Constraints

- NFL only. Player id namespace is nflverse `gsis_id` (same as `nfl_player_sim.player_id`).
- Six markets exactly: `pass_yds`, `pass_tds`, `rush_yds`, `rec_yds`, `receptions`, `anytime_td`. `anytime_td` actual = `rushing_tds + receiving_tds` (>=1 means scored).
- Reuse the resolution recipe from `scripts/grade_ev_props.py`; leave that script's behavior unchanged.
- Actuals are NOT model-versioned. Idempotent upsert on `(game_pk, player_id, market)`.
- User runs all Supabase migrations and redeploys `app.js`; never commit odds/credentials.
- One-bad-row-must-not-abort-the-batch: unresolved games/players are skipped, not raised.

---

### Task 1: `weekly_actuals` pure resolver module

**Files:**
- Create: `src/sportsmodel/nfl/weekly_actuals.py`
- Test: `tests/nfl/test_weekly_actuals.py`

**Interfaces:**
- Produces: `WEEKLY_STAT_COLS: dict[str, tuple[str, ...]]`; `week_for_game(sched_df, game_pk) -> int | None`; `player_market_actual(weekly_row, market) -> float | None`.

- [ ] **Step 1: Write failing tests**

```python
import pandas as pd
from sportsmodel.nfl.weekly_actuals import (
    WEEKLY_STAT_COLS, week_for_game, player_market_actual)


def _sched(rows):
    return pd.DataFrame(rows, columns=["espn", "week"])


def test_week_for_game_matches_espn_column():
    sched = _sched([["401872934", 3], ["401872947", 3]])
    assert week_for_game(sched, 401872934) == 3          # int game_pk vs str espn
    assert week_for_game(sched, "401872947") == 3


def test_week_for_game_no_match_or_missing_col_returns_none():
    assert week_for_game(_sched([["1", 1]]), 999) is None
    assert week_for_game(pd.DataFrame({"week": [1]}), 1) is None
    assert week_for_game(pd.DataFrame(), 1) is None


def test_all_six_markets_mapped():
    assert set(WEEKLY_STAT_COLS) == {
        "pass_yds", "pass_tds", "rush_yds", "rec_yds", "receptions", "anytime_td"}


def test_player_market_actual_reads_and_sums():
    row = pd.Series({"passing_yards": 251.0, "rushing_tds": 1, "receiving_tds": 2,
                     "receptions": 5})
    assert player_market_actual(row, "pass_yds") == 251.0
    assert player_market_actual(row, "anytime_td") == 3.0     # rush+rec TDs
    assert player_market_actual(row, "receptions") == 5.0


def test_player_market_actual_nan_or_missing_is_none():
    row = pd.Series({"passing_yards": float("nan")})
    assert player_market_actual(row, "pass_yds") is None
    assert player_market_actual(row, "rush_yds") is None      # column absent
    assert player_market_actual(pd.Series({"rushing_tds": float("nan"),
                                           "receiving_tds": 1}), "anytime_td") == 1.0
```

- [ ] **Step 2: Run tests, verify they fail** — `.venv/bin/pytest tests/nfl/test_weekly_actuals.py -q` → ImportError.

- [ ] **Step 3: Implement**

```python
"""Pure resolvers mapping the sim's projected player markets to their realized
nflverse `import_weekly_data` values, and a game's NFL week via the schedules
`espn` join. Shared by scripts/capture_nfl_player_actuals.py (and available to
grade_ev_props.py). No IO."""
from __future__ import annotations

import pandas as pd

# Projected market -> the nflverse weekly column(s) carrying its realized value.
# anytime_td has no single column: it's "did the player score", = rush + rec TDs.
WEEKLY_STAT_COLS: dict[str, tuple[str, ...]] = {
    "pass_yds": ("passing_yards",),
    "pass_tds": ("passing_tds",),
    "rush_yds": ("rushing_yards",),
    "rec_yds": ("receiving_yards",),
    "receptions": ("receptions",),
    "anytime_td": ("rushing_tds", "receiving_tds"),
}


def week_for_game(sched_df: pd.DataFrame, game_pk) -> int | None:
    """NFL week for `game_pk` (an ESPN event id) via an exact match against
    nflverse `import_schedules`' `espn` column. None if unresolvable."""
    if sched_df is None or sched_df.empty or "espn" not in sched_df.columns:
        return None
    matches = sched_df[sched_df["espn"].astype(str) == str(game_pk)]
    if matches.empty:
        return None
    week = matches.iloc[0]["week"]
    return int(week) if pd.notna(week) else None


def player_market_actual(weekly_row: pd.Series, market: str) -> float | None:
    """Realized value for `market` from one weekly-data row. Sums the columns
    for `anytime_td`. None if any needed column is absent or all-NaN."""
    cols = WEEKLY_STAT_COLS.get(market)
    if not cols:
        return None
    total = 0.0
    seen = False
    for col in cols:
        if col not in weekly_row:
            continue
        val = weekly_row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            continue
        total += float(val)
        seen = True
    return total if seen else None
```

- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** — `feat(nfl): weekly_actuals resolver for projected-market realized stats`.

---

### Task 2: `nfl_player_actuals` table + `db.upsert_nfl_player_actuals`

**Files:**
- Create: `db/migration_nfl_player_actuals.sql`
- Modify: `src/sportsmodel/db.py`
- Test: `tests/test_db_nfl_player_actuals.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `db.upsert_nfl_player_actuals(records: list[dict]) -> int`; table `nfl_player_actuals`.

- [ ] **Step 1: Write the migration** (exact SQL from the spec's Storage section — table, index, RLS policy, GRANT).

- [ ] **Step 2: Write failing tests** (mirror `tests/test_db_nfl_sim.py`'s FakeConn/FakeCursor):

```python
def test_upsert_nfl_player_actuals_empty_returns_zero(monkeypatch):
    monkeypatch.setattr(db_module, "get_postgres",
                        lambda: (_ for _ in ()).throw(AssertionError("no DB for empty")))
    assert db.upsert_nfl_player_actuals([]) == 0


def test_upsert_nfl_player_actuals_tuple_and_sql(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    row = {"game_pk": 401872934, "player_id": "00-0030431", "player_name": "Robert Woods",
           "market": "rec_yds", "actual": 37.0, "season": 2026, "week": 3}
    assert db.upsert_nfl_player_actuals([row]) == 1
    (tup,) = sink["rows"]
    cols = db._NFL_PLAYER_ACTUALS_COLS
    assert tup[cols.index("actual")] == 37.0
    assert tup[cols.index("market")] == "rec_yds"
    assert "INSERT INTO nfl_player_actuals" in sink["sql"]
    assert "ON CONFLICT (game_pk, player_id, market) DO UPDATE" in sink["sql"]
    assert "captured_at" not in sink["sql"]   # DEFAULT/updated separately, not upserted
```

(FakeCursor needs `executemany`; no `execute` needed here — plain upsert.)

- [ ] **Step 3: Run tests, verify fail** (no attribute).

- [ ] **Step 4: Implement in `db.py`** (place beside `upsert_nfl_player_sim`):

```python
_NFL_PLAYER_ACTUALS_COLS = [
    "game_pk", "player_id", "player_name", "market", "actual", "season", "week",
]


def upsert_nfl_player_actuals(records: list[dict]) -> int:
    """Upsert realized player box stats into `nfl_player_actuals`. Idempotent on
    (game_pk, player_id, market) -- re-capturing a finished game overwrites in
    place. `captured_at` is set once by the table DEFAULT and left out of the
    update. Requires DATABASE_URL + nfl_player_actuals (db/migration_nfl_player_actuals.sql)."""
    if not records:
        return 0
    key = ("game_pk", "player_id", "market")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _NFL_PLAYER_ACTUALS_COLS if c not in key)
    placeholders = ", ".join(["%s"] * len(_NFL_PLAYER_ACTUALS_COLS))
    sql = (
        f"INSERT INTO nfl_player_actuals ({', '.join(_NFL_PLAYER_ACTUALS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market) DO UPDATE SET {updates}, captured_at = now()"
    )
    rows = [tuple(r.get(c) for c in _NFL_PLAYER_ACTUALS_COLS) for r in records]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
```

- [ ] **Step 5: Run tests, verify pass. Commit** — `feat(db): nfl_player_actuals table + upsert helper`.

---

### Task 3: `capture_nfl_player_actuals.py` (pure assembly + IO main)

**Files:**
- Create: `scripts/capture_nfl_player_actuals.py`
- Test: `tests/scripts/test_capture_nfl_player_actuals.py`

**Interfaces:**
- Consumes: `weekly_actuals.WEEKLY_STAT_COLS/week_for_game/player_market_actual`; `db.upsert_nfl_player_actuals`; `nfl.injuries_nflverse.nfl_season`; `nfl.data.load_schedules`; `nfl.nflverse.import_by_season`.
- Produces: `assemble_actual_rows(projected, weekly_frame, game_pk, season, week) -> list[dict]`.

- [ ] **Step 1: Write failing tests for the pure assembler:**

```python
import pandas as pd
from scripts.capture_nfl_player_actuals import assemble_actual_rows


def _weekly(rows):
    return pd.DataFrame(rows)


def test_assemble_one_row_per_resolved_player_market():
    projected = [("00-A", "Star WR"), ("00-B", "Backup")]
    weekly = _weekly([
        {"player_id": "00-A", "week": 3, "receiving_yards": 88.0, "receptions": 6,
         "rushing_yards": 0.0, "receiving_tds": 1, "rushing_tds": 0,
         "passing_yards": 0.0, "passing_tds": 0},
    ])  # 00-B absent from weekly -> skipped entirely
    rows = assemble_actual_rows(projected, weekly, 401, 2026, 3)
    by = {(r["player_id"], r["market"]): r["actual"] for r in rows}
    assert by[("00-A", "rec_yds")] == 88.0
    assert by[("00-A", "anytime_td")] == 1.0
    assert not any(r["player_id"] == "00-B" for r in rows)   # unresolved player skipped
    assert all(r["game_pk"] == 401 and r["season"] == 2026 and r["week"] == 3 for r in rows)


def test_assemble_skips_all_nan_market_but_keeps_others():
    projected = [("00-A", "QB")]
    weekly = _weekly([{"player_id": "00-A", "week": 3, "passing_yards": 260.0,
                       "passing_tds": 2, "rushing_yards": float("nan"),
                       "receiving_yards": float("nan"), "receptions": float("nan"),
                       "receiving_tds": float("nan"), "rushing_tds": float("nan")}])
    markets = {r["market"] for r in assemble_actual_rows(projected, weekly, 401, 2026, 3)}
    assert "pass_yds" in markets and "pass_tds" in markets
    assert "rush_yds" not in markets            # all-NaN -> None -> skipped
```

- [ ] **Step 2: Run tests, verify fail.**

- [ ] **Step 3: Implement** the script — pure `assemble_actual_rows` + a `main(--days 8)` that: opens `get_postgres`, selects finished under-captured games (query below), and per game resolves season/week/weekly frame and upserts. Query for pending games:

```sql
SELECT DISTINCT s.game_pk, s.commence_time
FROM nfl_player_sim s
WHERE s.commence_time <= now() AND s.commence_time >= %(start)s
  AND NOT EXISTS (SELECT 1 FROM nfl_player_actuals a WHERE a.game_pk = s.game_pk)
ORDER BY s.commence_time
```

Pure assembler:

```python
def assemble_actual_rows(projected, weekly_frame, game_pk, season, week):
    """projected = [(player_id, player_name), ...] the sim projected for this
    game; weekly_frame = that season's nflverse weekly data. One row per
    (player, market) that resolves to a non-None actual for the game's week."""
    wk = weekly_frame[weekly_frame["week"] == week] if (
        weekly_frame is not None and not weekly_frame.empty and "week" in weekly_frame.columns
    ) else None
    out = []
    for pid, pname in projected:
        if wk is None:
            break
        prow = wk[wk["player_id"] == pid]
        if prow.empty:
            continue
        row = prow.iloc[0]
        for market in WEEKLY_STAT_COLS:
            actual = player_market_actual(row, market)
            if actual is None:
                continue
            out.append({"game_pk": game_pk, "player_id": pid, "player_name": pname,
                        "market": market, "actual": actual, "season": season, "week": week})
    return out
```

`main()` mirrors `grade_ev_props.py`'s caching (`_schedule_for`, `_weekly_for`), gets each game's projected `(player_id, name)` from `nfl_player_sim`, prints `games=/players=/actuals=/skipped=`. IO not unit-tested.

- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Sanity-run** `PYTHONPATH=src .venv/bin/python -c "import scripts.capture_nfl_player_actuals"` (import clean). **Commit** — `feat(nfl): capture player actuals for finished games`.

---

### Task 4: `capture-player-actuals.yml` workflow

**Files:**
- Create: `.github/workflows/capture-player-actuals.yml`

- [ ] **Step 1: Write the workflow** — copy `generate-sim-nfl.yml`'s job scaffold (checkout, uv setup, `DATABASE_URL` env), swap the run step to `uv run python scripts/capture_nfl_player_actuals.py --days 8`. Triggers: `workflow_dispatch: {}` + `schedule: - cron: "0 4,15 * * 0,1,2,5,6"`. No Odds API secret.
- [ ] **Step 2: Validate YAML** — `.venv/bin/python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in ['.github/workflows/capture-player-actuals.yml']]"`.
- [ ] **Step 3: Commit** — `ci(nfl): scheduled capture of player actuals`.

---

### Task 5: Front-end — actuals in the projected-props table (external `app.js`)

**Files:**
- Modify: `/Users/ryan/Desktop/CappingAlpha/app.js` (`buildGame`, `propsProjectionSection`, `injectStylesOnce`)
- Modify: `/Users/ryan/Desktop/CappingAlpha/*.html` (cache-buster bump)

- [ ] **Step 1:** In `buildGame`, add to the NFL `Promise.all`: `sb(\`nfl_player_actuals?game_pk=eq.${game}\`).catch(() => [])` and pass it as a third arg: `propsProjectionSection(sims, props, actuals)`.

- [ ] **Step 2:** In `propsProjectionSection(sims, props, actuals)` build `const actualBy = new Map(); (actuals||[]).forEach(a => actualBy.set(\`${a.player_id}|${a.market}\`, a.actual)); const hasActuals = actualBy.size > 0;`. Player rows are keyed by `player_id` (available on each sim row `s.player_id`) — carry it into `byPlayer` values as `id: s.player_id`.

- [ ] **Step 3:** Extend `cell(p, mk, dec)` so that when `hasActuals` and an actual exists for `${p.id}|${mk}`, it appends `<span class="prop-actual ${act >= proj ? "hit" : "miss"}">→ ${fmt(act)} ${act >= proj ? "✓" : "✗"}</span>` (compare against the projected mean `p.m[mk]`; treat missing projection as no marker). Extend `anyTd(p)` so with an actual it appends `TD ✓` when `actual >= 1` else `— ✗`.

- [ ] **Step 4:** Add CSS to `injectStylesOnce()`: `.prop-actual{display:block;font-size:11px;font-weight:600}.prop-actual.hit{color:#48d69a}.prop-actual.miss{color:#ff8a8a}`. Update the section `.sim-note` to mention "→ actual, ✓/✗ vs the projected number" when actuals show.

- [ ] **Step 5:** Bump `app.js?v=` in all `*.html` to a new value; `node --check app.js`.

- [ ] **Step 6:** Verify in the browser against a captured game (local server): projected values show an actual + ✓/✗ beneath; a not-yet-captured game renders unchanged. (No commit — external repo; user redeploys.)

---

## Self-Review

- **Spec coverage:** resolver (T1), table+upsert (T2), capture+assembly (T3), workflow (T4), UI (T5) — all spec sections have a task. ✓
- **Placeholders:** none — all code and SQL are concrete.
- **Type consistency:** `assemble_actual_rows` emits dict keys exactly matching `_NFL_PLAYER_ACTUALS_COLS` (game_pk, player_id, player_name, market, actual, season, week); `WEEKLY_STAT_COLS` keys are the six sim markets used identically in T1/T3/T5; `week_for_game`/`player_market_actual` signatures match across T1 and T3. ✓

## Deployment (user, in order)

1. Run `db/migration_nfl_player_actuals.sql`.
2. Merge to main; enable `capture-player-actuals.yml`; dispatch once to backfill.
3. Redeploy `app.js` + `*.html` (bumped cache-buster) to Cloudflare.
