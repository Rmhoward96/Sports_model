# Pick Ledger + Pre-Kick Close Capture Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record every pick the system surfaces exactly as it was at pick time in an append-only ledger, capture a true closing price 5 to 25 minutes before every kickoff (game lines, and the props we pick; US books plus Pinnacle), and grade every pick against that close nightly, so CLV and ROI are measured honestly before any edge claim is tested (profit roadmap, step 1).

**Architecture:** A new `sportsmodel.ledger` package holds pure logic (pick keys, the append rule, window and budget planning, close-reference and CLV maths, grading, the weekly gate) plus a best-effort recorder; the existing pick writers (+EV game board, +EV props board, best parlays, the three model-game generators) each gain one `safe_record(...)` call after their own database write. Two new scheduled jobs write only new tables and `odds_snapshot`: `capture_close.py` (every 10 minutes inside game-day hours, idle runs exit after one database read) and `grade_ledger.py` (nightly), plus a weekly gate report that commits a markdown file. All schema is one idempotent migration the USER runs.

**Tech Stack:** Python 3.11+, psycopg 3 (Supabase Postgres), httpx + tenacity (The Odds API client already in `ingest/odds.py`), GitHub Actions, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-13-pick-ledger-close-capture-design.md` (the authority). Evidence: `docs/superpowers/reports/2026-10-13-edge-audit.md` on branch `docs/edge-audit` (read it with `git show docs/edge-audit:docs/superpowers/reports/2026-10-13-edge-audit.md`).

**Baseline before this plan:** `uv run pytest -q` on `feat/pick-ledger` at `fd3d653` = 2602 passed (about 4 minutes). The new tests add 260 under `tests/ledger/` plus 4 `test_every_workflow_parses` cases (one per new workflow), so the final full run must show 2866 passed.

## Global Constraints

Every task's requirements include this section. Values are copied from the spec.

- **Scope:** "Any change to picks, models, probabilities, boards or the site" is out of scope; so are "steps 2 and 3 of the roadmap" and "bet sizing". "Existing tables and pages are unchanged." Single user; no site changes in this step.
- **Ledger rows (append-only):** table `pick_ledger`, one row per observation; a row is appended "when a `pick_key` is first seen, and again only when `best_price` changes, `line` changes, or `model_prob` / `ref_prob` moves by >= 1.0 percentage point since that key's last row". "Rows are never updated or deleted: enforced by a DB trigger that raises on UPDATE/DELETE, plus `REVOKE UPDATE, DELETE` from every non-owner role."
- **Best-effort:** "A ledger failure logs a warning and never fails the job that produced the pick."
- **Migration:** tables `pick_ledger`, `close_capture`, `pick_ledger_results`; indexes `(sport, game_pk)` and `(pick_key, observed_at)`; append-only trigger on `pick_ledger`; "RLS enabled with anon SELECT only and writes revoked from anon/authenticated (same pattern as `db/migration_site_panels.sql`). Idempotent, additive only. Jobs warn and exit 0 until the tables exist (same missing-table guard as the site-panel jobs)."
- **Close capture:** "every 10 minutes during game windows - Thu/Fri/Sat/Sun/Mon for football, daily Mar-Nov for MLB - bounded so idle hours cost nothing (a run with no kickoff in range exits before any API call)." Game lines: one Odds API call per sport (markets h2h/spreads/totals; regions us,eu). Props: one per-event call (regions us,eu; the sport's live prop markets) for each target event with at least one prop pick in the ledger. Captured prices go to `odds_snapshot`; one `close_capture` row per event, unique on (`sport`, `game_pk`, `kind`); "a failed attempt is retried by the next run while still in the window".
- **Pinnacle props probe first:** "(about 10 credits, one NFL event, regions=eu): record whether Pinnacle prop prices exist; if none, prop captures use regions=us only (saves credits) and the report states it."
- **Credits:** Odds API plan is 30,000 credits/month; current usage is about 2,300/week (about 10,000/month); this step adds about 360/week (about 1,550/month) during NFL + CFB + MLB postseason. "Budget guard: the script reads `x-requests-remaining`; below 3,000 remaining it skips prop captures and warns; below 1,000 it skips everything and warns. Credits per run are logged."
- **Grading:** close reference is "Pinnacle no-vig from the `close_capture` snapshot; else >= 3-book US-consensus no-vig at the exact line; else NULL with `close_kind = 'missing'`". `clv_prob` = "close fair prob - pick-time implied prob of `best_price`, same line only"; `clv_pts` for spreads/totals when the close line differs.
- **Weekly gate:** ">= 95% of graded games have a game-line close captured within 30 min of kickoff", ">= 90% of graded prop picks have a Pinnacle or >= 3-book close at their exact line", "0 ledger rows updated or deleted", "for 2 consecutive football weekends". "A miss blocks steps 2-3."
- **No database writes from the executing agent.** The migration (`db/migration_pick_ledger.sql`) is a file the USER runs in the Supabase SQL editor. Only the merged, scheduled jobs write to the database. The local `.env` holds a real `DATABASE_URL`: never run `capture_close.py`, `grade_ledger.py` or `report_ledger_gate.py` locally without `--dry-run` (and `--dry-run` only after the migration exists), and never unset `SM_LEDGER_OFF` in a shell that has `DATABASE_URL`. Every test uses injected fakes.
- **Network:** the only network step is Task 4's one-off Pinnacle probe (about 8 credits), run through GitHub Actions because `ODDS_API_KEY` exists only in GitHub secrets, not in the local `.env`.
- **Conventions:** follow the repo's patterns (script `main()` plus importlib-loaded script tests, FakeConn-style database tests, the block-YAML workflow reader in `tests/test_workflows_props_ml.py`). Project convention: implementers and reviewers run on Sonnet (pass `model: "sonnet"` explicitly). End every commit message with the Co-Authored-By trailer your session is configured with.

## Review Focus

The input classes and failure modes the spec implies but does not spell out, most likely to bite first. Each has a pinning test in the task that owns the code.

1. **Writers re-running after kickoff.** The audit found `game_predictions.generated_at` after kickoff for most graded games, and boards rebuild on stale data. A post-kickoff (or kickoff-less) observation is not bettable and must never enter the ledger. Task 2: `test_picks_for_started_games_or_without_a_kickoff_are_never_recorded`; grading also ignores any row at or after kickoff (Task 7 `test_rows_at_or_after_kickoff_are_never_bettable`).
2. **The ledger breaking a pick writer.** Missing migration, database hiccup, or a bug in an adapter must not fail the board build or the generator. Task 2: `test_every_failure_mode_is_swallowed_with_a_warning`, `test_safe_record_swallows_a_broken_adapter`; Task 3: `test_build_ev_board_survives_a_ledger_failure`, `test_a_ledger_failure_never_fails_a_model_game_writer`.
3. **Tests or local runs reaching production through `.env`.** `DATABASE_URL` is in the local `.env`, so any test that runs a script `main()` could write to the real ledger. Task 2 adds `SM_LEDGER_OFF=1` to the root `conftest.py`; the recorder and every new script `main()` refuse to write when it is set. Pinned by `test_the_test_suite_cannot_reach_the_production_ledger` (Task 2) and each script's `test_main_refuses_under_the_test_ledger_guard` (Tasks 5, 7, 8).
4. **The close is on a different number than the pick.** Lines move, Pinnacle may quote another number or no props at all, `odds_snapshot.line` is a float32 `REAL`, Caesars appears under two book keys, offshore books must not count. Behaviour: compare probabilities only at the same line, measure moved lines in points, count US books by display name, `missing` when nothing comparable. Task 6: the `close_reference` and `clv_pts` tests.
5. **Cron jitter and repeats.** GitHub delays, skips and occasionally duplicates scheduled runs; most runs are idle by design. Behaviour: an idle run makes zero network calls; a 20-minute capture window holds two runs per event; an `ok` or `empty` capture is final and never bought twice; only `failed` is retried; the budget guard stops spending below the floors. Task 5: `test_idle_run_makes_no_network_call_at_all`, `test_already_captured_events_cost_nothing`, `test_a_failed_call_is_recorded_failed_and_retried_next_run`, the budget tests, and the workflow coverage tests.

## Decisions on spec ambiguities (resolved here; the plan follows these)

| # | Spec text | Resolution | Why |
|---|---|---|---|
| 1 | `pick_key` is "a deterministic text key of the above, excluding price", and "above" lists `line` | The key excludes the **line** as well as the price (identity = source, sport, game_pk, market, side, player_id, parlay_id) | Otherwise the "line changes" append trigger is dead code and a number move splits one pick into two keys; "first row / last pre-kick row per key" needs one key per bet slot |
| 2 | Model picks come from "generate/board/props/parlay scripts and `grade_predictions` for model game picks" | Model game picks are recorded in the three **generators** (`generate_cfb`, `generate_sim_nfl`, `generate_sim`) right after their `game_predictions` upsert, using exactly `grade_predictions._accuracy_row`'s side logic (parity-tested). `grade_predictions` itself is not hooked | It runs after the game is final, so a row it wrote would be post-kickoff and not bettable. `generate_nfl.py` (Elo) and `generate_predictions.py` are not in the spec's list and are left alone. `model_version` joins the key for `model_game` only (two models can lean the same side) |
| 3 | Window "5-35 minutes" before kickoff | **5 to 25 minutes** | The gate needs a close within 30 minutes of kickoff; the first run to see an event at 31-35 minutes would fail it for about half of all games. A 20-minute window still holds two runs per event at a 10-minute cadence, so one skipped firing loses nothing |
| 4 | `pick_ledger_results` is "one row per `pick_key`, graded from the FIRST row ... and also from its LAST pre-kick row" | Primary key `(pick_key, basis)`: two rows per key, `basis` = `first` or `last` | The two gradings can differ (different line or price); one wide row would double every column |
| 5 | Model game picks have `best_price`, `ref_prob`... "NULL when the source has none" | `model_game` rows carry no price: units are graded at flat -110 on spreads and totals (the audit's convention), moneyline units and `clv_prob` are NULL, `clv_pts` still measures the line move. `ev_game` rows: `ref_prob` = the board's Pinnacle no-vig `base_prob`, `ref_kind = pinnacle_novig`, `model_prob` NULL. `ev_prop` rows: `ref_prob` = the board's `market_prob`, `ref_kind` NULL (it is not a graded reference) | Copy what each writer already computes; never invent a price |
| 6 | Gate item "0 ledger rows updated or deleted (trigger log + row-count audit)" | A raising trigger leaves no log, so the audit is a **row count plus a content fingerprint** (md5 over every row up to the previous report's cutoff), recorded in each report and recomputed by the next one; any difference fails the check. The trigger and revokes are verified by the migration tests | A superuser who disables the trigger would still be caught |
| 7 | "A failed attempt is retried by the next run" | Statuses `ok`, `empty` (call succeeded, nothing returned) and `failed`. Only `failed` is retried; `ok` and `empty` are final | Re-buying an empty response burns credits for nothing |
| 8 | Props: "the sport's live prop markets" | Request only the markets in which the ledger holds a pick for that event (NFL typically 4 of 8) | Matches the spec's own 360 credits/week figure; 8 markets x 2 regions x 15 games would not |
| 9 | "Schedules already ingested" as the kickoff source | Two-stage idle check: (a) a database read of `predictions_current` for any kickoff in the next 60 minutes, zero network; (b) only for sports that pass, the live schedule (ESPN / StatsAPI, free) decides the exact 5-25 minute targets, so flexed kickoffs are honoured | Idle runs stay at zero network calls; targets use the freshest times |
| 10 | Ledger index `(pick_key, observed_at)` | Created UNIQUE so a retried insert is a no-op; `close_capture` gains a `regions` column | Idempotency; lets the report state whether Pinnacle was requested |
| 11 | Close "from the `close_capture` snapshot" | A game with no capture falls back to its last pre-kick snapshot within 180 minutes, flagged by `close_minutes_to_kick` | Existing games (before this job runs) stay measurable; the gate still reads real captures only |
| 12 | Gate "graded games" | Games with a first-basis ledger result among NFL/CFB in the window Wednesday 12:00 to Tuesday 12:00 UTC; report runs Tuesday 15:47 UTC | A concrete denominator and calendar |

## Credit budget and cron

**Credits (The Odds API bills markets x regions per call).** Free: the events list, ESPN/StatsAPI schedules, database reads, idle runs (zero network calls).

| Call | Cost | When |
|---|---|---|
| Probe (Task 4, one-off) | 8 credits (8 NFL prop markets x `eu` x 1 event) | once |
| Game lines | 3 markets x 2 regions = **6 per sport-run** that has a target | one call per sport per run, however many games kick off together |
| Props | (markets with ledger picks) x regions per event = **8** with `us,eu` (NFL, typically 4 markets), **4** with `us` | one call per target event with at least one prop pick |

Weekly estimate in season (NFL + CFB + MLB postseason): lines about 186 (NFL about 7 kickoff slots, CFB about 10, MLB postseason about 14 slots, x 6) + NFL props about 120 (15 events x 8) + MLB props about 45 = **about 350 to 360 a week, about 1,550 a month**, on top of the current 2,300 a week: about 2,660 a week, about 11,500 of 30,000 a month. If the probe finds no Pinnacle props, set `CLOSE_PROP_REGIONS=us` and props halve (about 270 a week total). A full MLB regular season adds roughly 300-400 a week more for MLB props (spec). Guard: below 3,000 credits remaining props stop, below 1,000 everything stops, with a warning.

**Cron (UTC; GitHub jitters and skips).** `capture-close.yml` has four entries: football `3-59/10` over `13-23` on Sun/Mon/Thu/Fri/Sat plus a late spill `3-59/10` over `0-4` on Sun/Mon/Tue/Fri/Sat (months Aug-Feb), and baseball `8-59/10` over `15-23` and `0-2` (months Mar-Nov). Minutes avoid :00/:30 (GitHub's busiest), the two schedules never fire in the same minute, and a test proves every real kickoff pattern (Thursday night, Friday-night CFB, Saturday noon to the 11:59pm Hawaii kick, Sunday London 9:30am through the 4:25pm window, Sunday and Monday night, Thanksgiving, a December Saturday, January bowls, the Super Bowl, MLB day and west-coast night games) has a run inside its 25-to-5-minute window. About 980 runs a week in October, about 500 in summer; each idle run is one checkout, `uv sync` and one database read (the repository is public, so Actions minutes are free).

## File Structure

New package `src/sportsmodel/ledger/` keeps the pure logic apart from the scripts (which only do IO), mirroring how `serving/` separates pure engines from `scripts/`.

| File | Action | Responsibility |
|---|---|---|
| `db/migration_pick_ledger.sql` | Create | The three tables, indexes, append-only trigger, RLS, grants/revokes (USER runs it) |
| `src/sportsmodel/db.py` | Modify | Shared missing-table helper; `LEDGER_COLUMNS`/`LEDGER_KEYS`; `missing_ledger_tables`, `insert_pick_ledger`, `latest_pick_ledger_rows`, `upsert_close_capture`, `upsert_pick_ledger_results` |
| `conftest.py` | Modify | `SM_LEDGER_OFF=1` for the whole test run (production-database guard) |
| `src/sportsmodel/ledger/__init__.py` | Create | Package marker |
| `src/sportsmodel/ledger/core.py` | Create | `pick_key`, `observation`, the append rule (`change_reason`, `plan_appends`) |
| `src/sportsmodel/ledger/record.py` | Create | Best-effort `record_observations` / `safe_record`, `DbStore`, `writes_disabled` |
| `src/sportsmodel/ledger/sources.py` | Create | Adapters from each writer's rows to observations (`ev_game`, `ev_prop`, `parlay_leg`, `model_game`) |
| `scripts/build_ev_board.py`, `build_ev_props_board.py`, `build_best_parlays.py`, `generate_cfb.py`, `generate_sim_nfl.py`, `generate_sim.py` | Modify | One `safe_record(...)` call each, after the writer's own database write |
| `src/sportsmodel/ingest/odds.py` | Modify | Also record `x-requests-last` (credits the last call cost) |
| `src/sportsmodel/ledger/probe.py` | Create | Pinnacle-props verdict over one event payload |
| `scripts/probe_pinnacle_props.py`, `.github/workflows/probe-pinnacle-props.yml` | Create | The one-off, manually dispatched probe |
| `src/sportsmodel/ledger/close.py` | Create | Capture window, idle prefilter, budget guard, credit split, prop-market planning |
| `scripts/capture_close.py`, `.github/workflows/capture-close.yml` | Create | The close-capture job and its cron |
| `src/sportsmodel/ledger/reference.py` | Create | Close reference selection, CLV maths, settlement, units |
| `src/sportsmodel/ledger/grading.py` | Create | First/last basis rows, close merge, `grade_pick_row` / `grade_rows` |
| `scripts/grade_ledger.py`, `.github/workflows/grade-ledger.yml` | Create | Nightly grading job |
| `src/sportsmodel/ledger/gate.py` | Create | Coverage maths, audit comparison, two-weekend rule, report render/parse |
| `scripts/report_ledger_gate.py`, `.github/workflows/report-ledger-gate.yml` | Create | Weekly gate report job |
| `tests/ledger/` (`__init__.py` plus 15 test files) | Create | One test file per module/script/workflow |

Not touched: every model, board computation, `ev_pilot.py`, `props_ev.py`, `best_parlays.py`, the grading scripts, `grade_predictions.py`, the site, `capture-odds.yml`.

---

### Task 1: Migration and database helpers

**Files:**
- Create: `db/migration_pick_ledger.sql`
- Modify: `src/sportsmodel/db.py` (replace the end-of-file `missing_site_panel_tables` / `site_panel_tables_ready` with a shared helper, append the ledger block)
- Create: `tests/ledger/__init__.py` (empty), `tests/ledger/test_migration_and_db.py`

**Interfaces:**
- Consumes: `db.get_postgres()`, `db.SITE_PANEL_COLUMNS` (existing).
- Produces (later tasks rely on these exact names):
  - `db.LEDGER_COLUMNS: dict[str, list[str]]` keyed `pick_ledger` / `close_capture` / `pick_ledger_results`; `db.LEDGER_KEYS: dict[str, tuple[str, ...]]`; `db.LEDGER_LAST_COLS: list[str]`
  - `db.missing_ledger_tables(tables: list[str] | None = None) -> list[str]` (default: all three; `KeyError` for an unknown name; connection errors propagate)
  - `db.insert_pick_ledger(rows: list[dict]) -> int` (INSERT ... ON CONFLICT (pick_key, observed_at) DO NOTHING)
  - `db.latest_pick_ledger_rows(pick_keys: list[str]) -> dict[str, dict]` (most recent row per key, `LEDGER_LAST_COLS`)
  - `db.upsert_close_capture(rows: list[dict]) -> int` (only a `failed` row is ever superseded)
  - `db.upsert_pick_ledger_results(rows: list[dict]) -> int` (idempotent on `(pick_key, basis)`)
- Shape of the tables (the migration is the source of truth; `LEDGER_COLUMNS` lists the columns the code writes).

- [ ] **Step 1: Write the failing tests**

Create `tests/ledger/__init__.py` as an empty file, then `tests/ledger/test_migration_and_db.py`:

````python
"""db/migration_pick_ledger.sql and the ledger helpers in db.py: SQL shape, tuple building, column agreement,
the append-only guarantees and the missing-table guard. No live DB (fake connections)."""
import re
from pathlib import Path

import pytest

from sportsmodel import db

MIGRATION = (Path(__file__).resolve().parents[2] / "db" / "migration_pick_ledger.sql").read_text()
TABLES = ["pick_ledger", "close_capture", "pick_ledger_results"]


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


# ---- migration --------------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_every_written_column_exists_in_the_migration_and_keys_are_declared(table):
    body = _create_table(table)
    cols = re.findall(r"^\s{2}(\w+)\s", body, re.MULTILINE)
    assert set(db.LEDGER_COLUMNS[table]) <= set(cols), set(db.LEDGER_COLUMNS[table]) - set(cols)
    key = ", ".join(db.LEDGER_KEYS[table])
    if table == "pick_ledger":
        assert "id               bigserial PRIMARY KEY" in body
        assert f"CREATE UNIQUE INDEX IF NOT EXISTS pick_ledger_key_observed_idx ON pick_ledger ({key});" in MIGRATION
    else:
        assert f"PRIMARY KEY ({key})" in body


def test_pick_ledger_is_append_only_by_trigger_for_update_delete_and_truncate():
    assert "CREATE OR REPLACE FUNCTION pick_ledger_append_only() RETURNS trigger" in MIGRATION
    assert "RAISE EXCEPTION" in MIGRATION
    assert re.search(r"CREATE TRIGGER pick_ledger_no_update_delete\s+BEFORE UPDATE OR DELETE ON pick_ledger\s+"
                     r"FOR EACH ROW EXECUTE FUNCTION pick_ledger_append_only\(\);", MIGRATION)
    assert re.search(r"CREATE TRIGGER pick_ledger_no_truncate\s+BEFORE TRUNCATE ON pick_ledger\s+"
                     r"FOR EACH STATEMENT EXECUTE FUNCTION pick_ledger_append_only\(\);", MIGRATION)
    assert MIGRATION.count("DROP TRIGGER IF EXISTS") == 2          # re-runnable


def test_rls_select_policy_and_write_revokes_for_every_table():
    for t in TABLES:
        assert f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;" in MIGRATION
        assert f'CREATE POLICY "public read {t}" ON {t} FOR SELECT USING (true);' in MIGRATION
    m = re.search(r"REVOKE ([A-Z, ]+) ON ([^;]+) FROM anon, authenticated;", MIGRATION)
    assert m and set(re.split(r",\s*", m.group(1))) == {"INSERT", "UPDATE", "DELETE", "TRUNCATE"}
    for t in TABLES:
        assert f"public.{t}" in m.group(2)
    assert re.search(r"GRANT SELECT ON pick_ledger, close_capture, pick_ledger_results TO anon, authenticated;", MIGRATION)
    assert MIGRATION.index("REVOKE INSERT") < MIGRATION.index("GRANT SELECT ON")


def test_ledger_revokes_update_delete_truncate_from_public_and_service_role():
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM PUBLIC;" in MIGRATION
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM service_role" in MIGRATION
    assert "rolname = 'service_role'" in MIGRATION                  # guarded: the role may not exist locally


def test_migration_is_idempotent_additive_and_writes_no_data():
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == 3
    assert MIGRATION.count("DROP POLICY IF EXISTS") == 3
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE|ALTER TABLE \w+ DROP)\b", MIGRATION,
                         re.MULTILINE | re.IGNORECASE)
    assert "CREATE INDEX IF NOT EXISTS" in MIGRATION and "CREATE INDEX pick" not in MIGRATION


def test_check_constraints_pin_the_vocabularies():
    assert "CHECK (source IN ('model_game', 'ev_game', 'ev_prop', 'parlay_leg'))" in MIGRATION
    assert "ref_kind IN ('pinnacle_novig', 'us_consensus_novig')" in MIGRATION
    assert "CHECK (kind IN ('lines', 'props'))" in MIGRATION
    assert "CHECK (status IN ('ok', 'empty', 'failed'))" in MIGRATION
    assert "CHECK (basis IN ('first', 'last'))" in MIGRATION
    assert "CHECK (result IN ('W', 'L', 'P'))" in MIGRATION
    assert "close_kind IN ('pinnacle_novig', 'us_consensus_novig', 'missing')" in MIGRATION


# ---- db helpers ---------------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, sink, fetch=None):
        self.sink, self.fetch = sink, fetch or []

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def execute(self, sql, params=None):
        self.sink["sql"], self.sink["params"] = sql, params

    def fetchall(self):
        return self.fetch

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink, fetch=None):
        self.sink, self.fetch = sink, fetch

    def cursor(self):
        return FakeCursor(self.sink, self.fetch)

    def commit(self):
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _boom():
    raise AssertionError("get_postgres should not be called")


@pytest.mark.parametrize("fn", ["insert_pick_ledger", "upsert_close_capture", "upsert_pick_ledger_results"])
def test_empty_rows_never_connect(monkeypatch, fn):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(db, fn)([]) == 0


def test_latest_rows_with_no_keys_never_connects(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert db.latest_pick_ledger_rows([]) == {}


def test_insert_pick_ledger_is_insert_only_idempotent_and_ordered_by_column(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.LEDGER_COLUMNS["pick_ledger"]} | {"pick_key": "k", "sport": "nfl", "game_pk": 7}
    assert db.insert_pick_ledger([row]) == 1
    sql, (tup,) = sink["sql"], sink["rows"]
    assert sql.startswith("INSERT INTO pick_ledger (") and "ON CONFLICT (pick_key, observed_at) DO NOTHING" in sql
    assert "UPDATE" not in sql.replace("DO NOTHING", "") and "DELETE" not in sql
    cols = db.LEDGER_COLUMNS["pick_ledger"]
    assert len(tup) == len(cols) and tup[cols.index("pick_key")] == "k" and tup[cols.index("game_pk")] == 7
    assert sink["committed"] is True


def test_latest_pick_ledger_rows_maps_by_key(monkeypatch):
    sink = {}
    fetched = [("k1", -110, 3.5, 0.55, 0.52, "2026-10-13T00:00:00Z")]
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink, fetched))
    out = db.latest_pick_ledger_rows(["k1", "k2"])
    assert out == {"k1": dict(zip(db.LEDGER_LAST_COLS, fetched[0]))}
    assert "DISTINCT ON (pick_key)" in sink["sql"] and "ORDER BY pick_key, observed_at DESC, id DESC" in sink["sql"]
    assert sink["params"] == (["k1", "k2"],)


def test_close_capture_upsert_only_supersedes_a_failed_row(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.LEDGER_COLUMNS["close_capture"]} | {"sport": "nfl", "game_pk": 1, "kind": "lines"}
    assert db.upsert_close_capture([row]) == 1
    sql = sink["sql"]
    assert "ON CONFLICT (sport, game_pk, kind) DO UPDATE SET" in sql
    assert sql.rstrip().endswith("WHERE close_capture.status = 'failed'")
    assert "status = EXCLUDED.status" in sql and "sport = EXCLUDED.sport" not in sql


def test_results_upsert_keys_on_pick_key_and_basis(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.LEDGER_COLUMNS["pick_ledger_results"]} | {"pick_key": "k", "basis": "first"}
    assert db.upsert_pick_ledger_results([row]) == 1
    sql = sink["sql"]
    assert "ON CONFLICT (pick_key, basis) DO UPDATE SET" in sql and "graded_at = now()" in sql
    assert "result = EXCLUDED.result" in sql and "pick_key = EXCLUDED.pick_key" not in sql


class _RegCursor(FakeCursor):
    def __init__(self, existing, log):
        self.existing, self.log, self.last = existing, log, None

    def execute(self, sql, params):
        self.log.append((sql, params))
        name = params[0].split(".", 1)[1]
        self.last = (params[0],) if name in self.existing else (None,)

    def fetchone(self):
        return self.last


class _RegConn(FakeConn):
    def __init__(self, existing, log):
        self.existing, self.log = existing, log

    def cursor(self):
        return _RegCursor(self.existing, self.log)


def test_missing_ledger_tables_checks_to_regclass_in_public_in_order(monkeypatch):
    log = []
    monkeypatch.setattr(db, "get_postgres", lambda: _RegConn({"pick_ledger"}, log))
    assert db.missing_ledger_tables() == ["close_capture", "pick_ledger_results"]
    assert [p[0] for _, p in log] == [f"public.{t}" for t in TABLES]
    assert db.missing_ledger_tables(["pick_ledger"]) == []


def test_missing_ledger_tables_rejects_unknown_names_and_propagates_connection_errors(monkeypatch):
    with pytest.raises(KeyError):
        db.missing_ledger_tables(["users; drop table x"])

    def down():
        raise ConnectionError("db down")

    monkeypatch.setattr(db, "get_postgres", down)
    with pytest.raises(ConnectionError):
        db.missing_ledger_tables(["pick_ledger"])


def test_site_panel_guard_still_works_after_the_shared_helper_refactor(monkeypatch):
    log = []
    monkeypatch.setattr(db, "get_postgres", lambda: _RegConn({"game_info"}, log))
    assert db.missing_site_panel_tables(["game_info", "cfb_team_insights"]) == ["cfb_team_insights"]
    with pytest.raises(KeyError):
        db.missing_site_panel_tables(["pick_ledger"])
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_migration_and_db.py`
Expected: collection ERROR, `FileNotFoundError: ... db/migration_pick_ledger.sql` (the test module reads the migration at import).

- [ ] **Step 3: Write the migration**

````sql
-- =============================================================================
-- Pick ledger + pre-kick close capture (profit project, step 1).
-- =============================================================================
-- Idempotent and additive only -- safe to re-run. Run this in the Supabase SQL Editor BEFORE the new jobs
-- first run (until then every ledger job warns and exits 0; nothing existing changes).
--
--   pick_ledger          APPEND-ONLY log: one row per OBSERVATION of a pick (first sighting, then again only
--                        when the price / line moves or a probability moves >= 1 percentage point). Written
--                        best-effort by the pick writers. A trigger refuses UPDATE / DELETE / TRUNCATE and
--                        the write privileges are revoked from every role except the table owner.
--   close_capture        one row per (sport, game_pk, kind) recording a pre-kick closing-price capture
--                        (kind = lines | props). Written by scripts/capture_close.py. A 'failed' attempt may be
--                        superseded by the next one; an 'ok' or 'empty' row is final and never replaced.
--   pick_ledger_results  one row per (pick_key, basis): the pick graded from its FIRST ledger row (the
--                        earliest bettable observation) and from its LAST pre-kick row. Written nightly by
--                        scripts/grade_ledger.py.

CREATE TABLE IF NOT EXISTS pick_ledger (
  id               bigserial PRIMARY KEY,
  pick_key         text NOT NULL,                 -- deterministic key of the bet slot, never the price or line
  source           text NOT NULL CHECK (source IN ('model_game', 'ev_game', 'ev_prop', 'parlay_leg')),
  sport            text NOT NULL,
  game_pk          bigint NOT NULL,
  market           text NOT NULL,
  side             text NOT NULL,
  line             double precision,              -- the SIDE's own line (home spread, total, prop line); NULL for moneyline
  player_id        text,
  player_name      text,
  parlay_id        text,
  model_version    text,
  model_prob       double precision,              -- model P(this side wins at `line`); NULL when the source has none
  best_price       integer,                       -- American price at the best book, as seen at observed_at
  best_book        text,
  ref_prob         double precision,              -- fair probability of this side at `line` per the writer
  ref_kind         text CHECK (ref_kind IS NULL OR ref_kind IN ('pinnacle_novig', 'us_consensus_novig')),
  ref_books        integer,
  ev_pct           double precision,              -- EV in percent, as the source computed it
  observed_at      timestamptz NOT NULL,
  commence_time    timestamptz NOT NULL,
  minutes_to_kick  double precision NOT NULL,
  writer           text NOT NULL,                 -- script name
  writer_run       text NOT NULL,                 -- GitHub run id, or 'local'
  recorded_at      timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pick_ledger_game_idx ON pick_ledger (sport, game_pk);
-- UNIQUE so a retried insert of the same observation is a no-op (ON CONFLICT DO NOTHING).
CREATE UNIQUE INDEX IF NOT EXISTS pick_ledger_key_observed_idx ON pick_ledger (pick_key, observed_at);

CREATE OR REPLACE FUNCTION pick_ledger_append_only() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'pick_ledger is append-only: % is not allowed', TG_OP
    USING ERRCODE = 'restrict_violation';
END;
$$;

DROP TRIGGER IF EXISTS pick_ledger_no_update_delete ON pick_ledger;
CREATE TRIGGER pick_ledger_no_update_delete
  BEFORE UPDATE OR DELETE ON pick_ledger
  FOR EACH ROW EXECUTE FUNCTION pick_ledger_append_only();

DROP TRIGGER IF EXISTS pick_ledger_no_truncate ON pick_ledger;
CREATE TRIGGER pick_ledger_no_truncate
  BEFORE TRUNCATE ON pick_ledger
  FOR EACH STATEMENT EXECUTE FUNCTION pick_ledger_append_only();

CREATE TABLE IF NOT EXISTS close_capture (
  sport            text NOT NULL,
  game_pk          bigint NOT NULL,
  event_id         text,                          -- The Odds API event id
  kind             text NOT NULL CHECK (kind IN ('lines', 'props')),
  captured_at      timestamptz NOT NULL,          -- equals captured_at of the odds_snapshot rows it stored
  commence_time    timestamptz NOT NULL,
  minutes_to_kick  double precision NOT NULL,
  credits_used     integer NOT NULL DEFAULT 0,
  status           text NOT NULL CHECK (status IN ('ok', 'empty', 'failed')),
  regions          text,                          -- e.g. 'us,eu'
  markets          text,
  updated_at       timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_pk, kind)
);

CREATE TABLE IF NOT EXISTS pick_ledger_results (
  pick_key               text NOT NULL,
  basis                  text NOT NULL CHECK (basis IN ('first', 'last')),
  ledger_id              bigint NOT NULL,
  source                 text NOT NULL,
  sport                  text NOT NULL,
  game_pk                bigint NOT NULL,
  market                 text NOT NULL,
  side                   text NOT NULL,
  line                   double precision,
  player_id              text,
  player_name            text,
  best_price             integer,
  observed_at            timestamptz NOT NULL,
  commence_time          timestamptz NOT NULL,
  minutes_to_kick        double precision NOT NULL,
  result                 text NOT NULL CHECK (result IN ('W', 'L', 'P')),
  units_at_pick          double precision,        -- 1 unit staked at best_price (-110 for model spread/total picks)
  close_kind             text NOT NULL CHECK (close_kind IN ('pinnacle_novig', 'us_consensus_novig', 'missing')),
  close_fair_prob        double precision,
  close_line             double precision,
  close_same_line        boolean,
  close_books            integer,
  close_minutes_to_kick  double precision,
  clv_prob               double precision,        -- close fair prob - implied prob of best_price, same line only
  clv_pts                double precision,        -- line points gained vs the close (spreads / totals / props)
  beat_close             boolean,
  graded_at              timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (pick_key, basis)
);

CREATE INDEX IF NOT EXISTS pick_ledger_results_game_idx ON pick_ledger_results (sport, game_pk);
CREATE INDEX IF NOT EXISTS pick_ledger_results_commence_idx ON pick_ledger_results (commence_time);

ALTER TABLE pick_ledger ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read pick_ledger" ON pick_ledger;
CREATE POLICY "public read pick_ledger" ON pick_ledger FOR SELECT USING (true);
ALTER TABLE close_capture ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read close_capture" ON close_capture;
CREATE POLICY "public read close_capture" ON close_capture FOR SELECT USING (true);
ALTER TABLE pick_ledger_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read pick_ledger_results" ON pick_ledger_results;
CREATE POLICY "public read pick_ledger_results" ON pick_ledger_results FOR SELECT USING (true);

-- Defence in depth: the API roles can read these tables and nothing else (the jobs write as the DB owner).
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.pick_ledger, public.close_capture, public.pick_ledger_results FROM anon, authenticated;
GRANT SELECT ON pick_ledger, close_capture, pick_ledger_results TO anon, authenticated;

-- The ledger is append-only for EVERY non-owner role: also strip UPDATE / DELETE / TRUNCATE from PUBLIC and
-- from the other Supabase API role (service_role) when it exists. The owner is held back by the trigger.
REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON public.pick_ledger FROM service_role';
  END IF;
END;
$$;
````

- [ ] **Step 4: Write the database helpers**

Apply this diff to `src/sportsmodel/db.py`. It (a) pulls the `to_regclass` loop of `missing_site_panel_tables` (the last function of the "site panels" section) into a shared `_missing_public_tables` helper, leaving the site-panel guard's behaviour and tests unchanged, and (b) appends the ledger block at the end of the file:

````diff
--- a/src/sportsmodel/db.py
+++ b/src/sportsmodel/db.py
@@ -1160,11 +1160,7 @@
 
 
-def missing_site_panel_tables(tables: list[str]) -> list[str]:
-    """The subset of `tables` (keys of SITE_PANEL_COLUMNS) that do not exist in the `public` schema yet, in
-    the order given. Lets the site-panel jobs warn and exit cleanly until db/migration_site_panels.sql has
-    been run. One read-only connection; any connection error propagates (a real outage still fails the job)."""
-    unknown = [t for t in tables if t not in SITE_PANEL_COLUMNS]
-    if unknown:
-        raise KeyError(f"not a site-panel table: {unknown}")
+def _missing_public_tables(tables: list[str]) -> list[str]:
+    """The subset of `tables` that do not exist in the `public` schema, in the order given (one read-only
+    connection, `to_regclass` per table). Any connection error propagates (a real outage still fails the job)."""
     with get_postgres() as conn, conn.cursor() as cur:
         missing = []
@@ -1174,6 +1170,108 @@
                 missing.append(t)
     return missing
+
+
+def missing_site_panel_tables(tables: list[str]) -> list[str]:
+    """The subset of `tables` (keys of SITE_PANEL_COLUMNS) that do not exist in the `public` schema yet, in
+    the order given. Lets the site-panel jobs warn and exit cleanly until db/migration_site_panels.sql has
+    been run. One read-only connection; any connection error propagates (a real outage still fails the job)."""
+    unknown = [t for t in tables if t not in SITE_PANEL_COLUMNS]
+    if unknown:
+        raise KeyError(f"not a site-panel table: {unknown}")
+    return _missing_public_tables(tables)
 
 
 def site_panel_tables_ready(tables: list[str]) -> bool:
     return not missing_site_panel_tables(tables)
+
+
+# ------------------------------------------------------------------ pick ledger
+# db/migration_pick_ledger.sql. pick_ledger is APPEND-ONLY (trigger + revoked privileges): the only write
+# helper for it is insert_pick_ledger. close_capture / pick_ledger_results are ordinary upsert tables.
+LEDGER_COLUMNS: dict[str, list[str]] = {
+    "pick_ledger": ["pick_key", "source", "sport", "game_pk", "market", "side", "line", "player_id",
+                    "player_name", "parlay_id", "model_version", "model_prob", "best_price", "best_book",
+                    "ref_prob", "ref_kind", "ref_books", "ev_pct", "observed_at", "commence_time",
+                    "minutes_to_kick", "writer", "writer_run"],
+    "close_capture": ["sport", "game_pk", "event_id", "kind", "captured_at", "commence_time",
+                      "minutes_to_kick", "credits_used", "status", "regions", "markets"],
+    "pick_ledger_results": ["pick_key", "basis", "ledger_id", "source", "sport", "game_pk", "market", "side",
+                            "line", "player_id", "player_name", "best_price", "observed_at", "commence_time",
+                            "minutes_to_kick", "result", "units_at_pick", "close_kind", "close_fair_prob",
+                            "close_line", "close_same_line", "close_books", "close_minutes_to_kick",
+                            "clv_prob", "clv_pts", "beat_close"],
+}
+LEDGER_KEYS: dict[str, tuple[str, ...]] = {
+    "pick_ledger": ("pick_key", "observed_at"), "close_capture": ("sport", "game_pk", "kind"),
+    "pick_ledger_results": ("pick_key", "basis"),
+}
+# What the append rule needs from a key's most recent row.
+LEDGER_LAST_COLS = ["pick_key", "best_price", "line", "model_prob", "ref_prob", "observed_at"]
+
+
+def missing_ledger_tables(tables: list[str] | None = None) -> list[str]:
+    """The ledger tables (default: all three) that do not exist in `public` yet, in the order given. Lets the
+    ledger jobs warn and exit 0 until db/migration_pick_ledger.sql has been run (same guard as the site panels)."""
+    tables = list(LEDGER_COLUMNS) if tables is None else tables
+    unknown = [t for t in tables if t not in LEDGER_COLUMNS]
+    if unknown:
+        raise KeyError(f"not a ledger table: {unknown}")
+    return _missing_public_tables(tables)
+
+
+def insert_pick_ledger(rows: list[dict]) -> int:
+    """Append observations to `pick_ledger`. INSERT only (the table refuses anything else); a retried
+    observation (same pick_key + observed_at) is a no-op via ON CONFLICT DO NOTHING."""
+    if not rows:
+        return 0
+    cols = LEDGER_COLUMNS["pick_ledger"]
+    sql = (f"INSERT INTO pick_ledger ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
+           f"ON CONFLICT (pick_key, observed_at) DO NOTHING")
+    vals = [tuple(r.get(c) for c in cols) for r in rows]
+    with get_postgres() as conn, conn.cursor() as cur:
+        cur.executemany(sql, vals)
+        conn.commit()
+    return len(vals)
+
+
+def latest_pick_ledger_rows(pick_keys: list[str]) -> dict[str, dict]:
+    """{pick_key: its most recent row (LEDGER_LAST_COLS)} for the keys that have one. Read-only."""
+    if not pick_keys:
+        return {}
+    with get_postgres() as conn, conn.cursor() as cur:
+        cur.execute(
+            f"SELECT DISTINCT ON (pick_key) {', '.join(LEDGER_LAST_COLS)} FROM pick_ledger "
+            "WHERE pick_key = ANY(%s) ORDER BY pick_key, observed_at DESC, id DESC", (list(pick_keys),))
+        return {r[0]: dict(zip(LEDGER_LAST_COLS, r)) for r in cur.fetchall()}
+
+
+def upsert_close_capture(rows: list[dict]) -> int:
+    """Record close captures. Idempotent on (sport, game_pk, kind); an existing 'ok' or 'empty' row is final and
+    NEVER replaced (an event is captured once), while a 'failed' row is superseded by the next attempt."""
+    if not rows:
+        return 0
+    cols, key = LEDGER_COLUMNS["close_capture"], LEDGER_KEYS["close_capture"]
+    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
+    sql = (f"INSERT INTO close_capture ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
+           f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets}, updated_at = now() "
+           f"WHERE close_capture.status = 'failed'")
+    vals = [tuple(r.get(c) for c in cols) for r in rows]
+    with get_postgres() as conn, conn.cursor() as cur:
+        cur.executemany(sql, vals)
+        conn.commit()
+    return len(vals)
+
+
+def upsert_pick_ledger_results(rows: list[dict]) -> int:
+    """Upsert graded ledger results; idempotent on (pick_key, basis) (a re-grade overwrites in place)."""
+    if not rows:
+        return 0
+    cols, key = LEDGER_COLUMNS["pick_ledger_results"], LEDGER_KEYS["pick_ledger_results"]
+    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
+    sql = (f"INSERT INTO pick_ledger_results ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
+           f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets}, graded_at = now()")
+    vals = [tuple(r.get(c) for c in cols) for r in rows]
+    with get_postgres() as conn, conn.cursor() as cur:
+        cur.executemany(sql, vals)
+        conn.commit()
+    return len(vals)
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger tests/test_db_site_panels.py tests/test_db_ev.py tests/test_db_ev_props.py tests/test_db_best_parlays.py tests/test_db_prediction_accuracy.py`
Expected: all PASS (`tests/ledger` alone: 19 passed).

- [ ] **Step 6: Commit**

```bash
git add db/migration_pick_ledger.sql src/sportsmodel/db.py tests/ledger/__init__.py tests/ledger/test_migration_and_db.py
git commit -m "feat(ledger): pick_ledger / close_capture / pick_ledger_results migration and db helpers"
```

---

### Task 2: `pick_key`, the append rule and the best-effort recorder

**Files:**
- Create: `src/sportsmodel/ledger/__init__.py`, `src/sportsmodel/ledger/core.py`, `src/sportsmodel/ledger/record.py`
- Modify: `conftest.py` (production-database guard)
- Create: `tests/ledger/test_core.py`, `tests/ledger/test_record.py`

**Interfaces:**
- Consumes: Task 1's `db.missing_ledger_tables`, `db.latest_pick_ledger_rows`, `db.insert_pick_ledger` (through `DbStore`, imported lazily).
- Produces:
  - `core.SOURCES = ("model_game", "ev_game", "ev_prop", "parlay_leg")`, `core.REF_KINDS`, `core.PROB_MOVE = 0.01`
  - `core.as_utc(value) -> datetime | None` (datetime / ISO string -> aware UTC)
  - `core.pick_key(source, sport, game_pk, market, side, *, player_id=None, parlay_id=None, model_version=None) -> str` (never includes price or line; `model_version` only for `model_game`)
  - `core.observation(*, source, sport, game_pk, market, side, commence_time, line=None, player_id=None, player_name=None, parlay_id=None, model_version=None, model_prob=None, best_price=None, best_book=None, ref_prob=None, ref_kind=None, ref_books=None, ev_pct=None) -> dict` (the candidate; includes `pick_key`)
  - `core.change_reason(cand, last) -> "first" | "price" | "line" | "prob" | None`
  - `core.plan_appends(candidates, last_by_key, *, now, writer, writer_run) -> (rows, counts)` (rows gain `observed_at`, `minutes_to_kick`, `writer`, `writer_run`; started games and games without a kickoff are never returned)
  - `record.writes_disabled() -> bool` (env `SM_LEDGER_OFF` in 1/true/yes/on)
  - `record.record_observations(candidates, *, writer, now=None, store=None, run_id=None) -> dict` (never raises; `store` has `missing_tables()`, `latest(keys)`, `insert(rows)`)
  - `record.safe_record(build, *, writer, **kw) -> dict` (`build` is a zero-argument callable returning candidates; adapter errors are swallowed too)

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_core.py`:

````python
"""Pick key, observation record and the append rule. Pure -- no DB, no network."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel.ledger import core

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
KICK = NOW + timedelta(hours=26)


def _obs(**kw):
    base = dict(source="ev_prop", sport="nfl", game_pk=401, market="rec_yds", side="over", commence_time=KICK,
                line=64.5, player_id="00-0036", player_name="A. Brown", model_prob=0.58, best_price=-110,
                best_book="draftkings", ref_prob=0.52, ref_kind=None, ev_pct=6.0)
    return core.observation(**{**base, **kw})


def _last(**kw):
    base = dict(best_price=-110, line=64.5, model_prob=0.58, ref_prob=0.52)
    return {**base, **kw}


def plan(cands, last=None, now=NOW):
    return core.plan_appends(cands, last or {}, now=now, writer="w", writer_run="r1")


# ---- pick_key ------------------------------------------------------------------------------------

def test_pick_key_is_deterministic_and_ignores_price_and_line():
    a, b = _obs(), _obs(best_price=+120, line=66.5, model_prob=0.7, best_book="fanduel")
    assert a["pick_key"] == b["pick_key"] == "ev_prop|nfl|401|rec_yds|over|00-0036||"


def test_pick_key_separates_every_identity_component():
    base = _obs()["pick_key"]
    for change in (dict(side="under"), dict(market="receptions"), dict(game_pk=402), dict(sport="mlb"),
                   dict(player_id="00-0099"), dict(source="parlay_leg", parlay_id="dk|g:1")):
        assert _obs(**change)["pick_key"] != base, change


def test_pick_key_includes_model_version_only_for_model_game():
    mg = lambda v: core.observation(source="model_game", sport="nfl", game_pk=1, market="spread", side="home",
                                    commence_time=KICK, model_version=v)["pick_key"]
    assert mg("sim-nfl-ml-v2") != mg("elo-v1") and mg("elo-v1").endswith("|elo-v1|")
    assert _obs(model_version="props-sim-v1")["pick_key"] == _obs(model_version="props-sim-v2")["pick_key"]


def test_pick_key_rejects_an_unknown_source_and_observation_an_unknown_ref_kind():
    with pytest.raises(ValueError):
        core.pick_key("desk", "nfl", 1, "spread", "home")
    with pytest.raises(ValueError):
        _obs(ref_kind="vibes")


def test_observation_coerces_numeric_types():
    o = _obs(game_pk="401", best_price="-115", line=64, model_prob=0)
    assert o["game_pk"] == 401 and o["best_price"] == -115 and o["line"] == 64.0 and o["model_prob"] == 0.0


# ---- append rule ---------------------------------------------------------------------------------

def test_first_sighting_is_appended_with_timing_and_provenance():
    rows, counts = plan([_obs()])
    (r,) = rows
    assert counts == {"first": 1}
    assert r["observed_at"] == NOW and r["commence_time"] == KICK and r["minutes_to_kick"] == 26 * 60
    assert r["writer"] == "w" and r["writer_run"] == "r1"


def test_unchanged_pick_is_not_appended():
    o = _obs()
    rows, counts = plan([o], {o["pick_key"]: _last()})
    assert rows == [] and counts == {"unchanged": 1}


@pytest.mark.parametrize("cand, why", [
    (dict(best_price=-105), "price"),
    (dict(best_price=None), "price"),
    (dict(line=65.5), "line"),
    (dict(line=None), "line"),
    (dict(model_prob=0.59), "prob"),          # exactly 1.0pp (float noise must not hide it)
    (dict(model_prob=0.57), "prob"),
    (dict(ref_prob=0.53), "prob"),
    (dict(ref_prob=None), "prob"),
])
def test_price_line_or_one_point_probability_move_appends(cand, why):
    o = _obs(**cand)
    rows, counts = plan([o], {o["pick_key"]: _last()})
    assert counts == {why: 1} and len(rows) == 1


@pytest.mark.parametrize("cand", [dict(model_prob=0.5899), dict(model_prob=0.5701), dict(ref_prob=0.5249),
                                  dict(line=64.5 + 1e-12)])
def test_sub_threshold_moves_are_ignored(cand):
    o = _obs(**cand)
    rows, counts = plan([o], {o["pick_key"]: _last()})
    assert rows == [] and counts == {"unchanged": 1}


def test_the_comparison_is_against_the_keys_last_row_not_the_first():
    o = _obs(model_prob=0.585)
    assert plan([o], {o["pick_key"]: _last(model_prob=0.58)})[0] == []                 # 0.5pp from last
    assert len(plan([o], {o["pick_key"]: _last(model_prob=0.572)})[0]) == 1           # 1.3pp from last


def test_picks_for_started_games_or_without_a_kickoff_are_never_recorded():
    started = _obs(commence_time=NOW - timedelta(minutes=1))
    at_kick = _obs(player_id="2", commence_time=NOW)
    nokick = _obs(player_id="3", commence_time=None)
    garbled = _obs(player_id="4", commence_time="soon")
    rows, counts = plan([started, at_kick, nokick, garbled])
    assert rows == [] and counts == {"started": 2, "invalid": 2}


def test_duplicate_keys_in_one_batch_keep_the_last_candidate():
    rows, _ = plan([_obs(best_price=-110), _obs(best_price=-120)])
    assert [r["best_price"] for r in rows] == [-120]


def test_string_kickoffs_and_naive_datetimes_are_read_as_utc():
    rows, _ = plan([_obs(commence_time="2026-10-15T17:00:00Z"),
                    _obs(player_id="9", commence_time=datetime(2026, 10, 15, 17))])
    assert {r["commence_time"] for r in rows} == {datetime(2026, 10, 15, 17, tzinfo=timezone.utc)}
````

`tests/ledger/test_record.py`:

````python
"""Best-effort recording: a ledger failure logs a warning and never fails the writer."""
from datetime import datetime, timedelta, timezone

from sportsmodel import db
from sportsmodel.ledger import core, record

NOW = datetime(2026, 10, 14, 15, 0, tzinfo=timezone.utc)
KICK = NOW + timedelta(hours=5)


def _obs(**kw):
    return core.observation(**{**dict(source="ev_game", sport="cfb", game_pk=9, market="spread", side="home",
                                      commence_time=KICK, line=-3.5, best_price=-110), **kw})


class FakeStore:
    def __init__(self, missing=(), last=None, fail=None):
        self.missing, self.last, self.fail, self.inserted, self.asked = list(missing), last or {}, fail, [], []

    def missing_tables(self):
        if self.fail == "missing":
            raise ConnectionError("db down")
        return self.missing

    def latest(self, keys):
        self.asked.append(keys)
        if self.fail == "latest":
            raise RuntimeError("boom")
        return self.last

    def insert(self, rows):
        if self.fail == "insert":
            raise RuntimeError("insert refused")
        self.inserted += rows
        return len(rows)


def test_new_picks_are_inserted_with_the_run_id(monkeypatch):
    monkeypatch.setenv("GITHUB_RUN_ID", "12345")
    store = FakeStore()
    out = record.record_observations([_obs()], writer="build_ev_board", now=NOW, store=store)
    assert out["status"] == "ok" and out["appended"] == 1
    (row,) = store.inserted
    assert row["writer"] == "build_ev_board" and row["writer_run"] == "12345" and row["observed_at"] == NOW


def test_local_runs_are_labelled_local(monkeypatch):
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    store = FakeStore()
    record.record_observations([_obs()], writer="w", now=NOW, store=store)
    assert store.inserted[0]["writer_run"] == "local"


def test_unchanged_picks_insert_nothing():
    o = _obs()
    store = FakeStore(last={o["pick_key"]: {"best_price": -110, "line": -3.5, "model_prob": None, "ref_prob": None}})
    out = record.record_observations([o], writer="w", now=NOW, store=store)
    assert out["appended"] == 0 and store.inserted == []


def test_missing_table_warns_and_skips(capsys):
    store = FakeStore(missing=["pick_ledger"])
    out = record.record_observations([_obs()], writer="w", now=NOW, store=store)
    assert out == {"status": "no_table"} and store.inserted == []
    assert "::warning::pick-ledger: table pick_ledger missing" in capsys.readouterr().out


def test_every_failure_mode_is_swallowed_with_a_warning(capsys):
    for mode in ("missing", "latest", "insert"):
        out = record.record_observations([_obs()], writer="w", now=NOW, store=FakeStore(fail=mode))
        assert out["status"] == "error", mode
    assert capsys.readouterr().out.count("::warning::pick-ledger") == 3


def test_safe_record_swallows_a_broken_adapter(capsys):
    def build():
        raise KeyError("matchup")

    out = record.safe_record(build, writer="w", store=FakeStore())
    assert out["status"] == "error" and "building observations failed" in capsys.readouterr().out


def test_safe_record_passes_adapter_output_through():
    store = FakeStore()
    out = record.safe_record(lambda: [_obs(), None], writer="w", now=NOW, store=store)
    assert out["appended"] == 1


def test_empty_batches_never_touch_the_store():
    store = FakeStore(fail="missing")
    assert record.record_observations([], writer="w", store=store) == {"status": "empty"}


def test_the_test_suite_cannot_reach_the_production_ledger(monkeypatch):
    """conftest.py sets SM_LEDGER_OFF: with no injected store the default DB store is never built."""
    assert record.writes_disabled()
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("DB touched")))
    assert record.record_observations([_obs()], writer="w", now=NOW) == {"status": "disabled"}
    monkeypatch.setenv("SM_LEDGER_OFF", "0")
    assert not record.writes_disabled()
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_core.py tests/ledger/test_record.py`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'sportsmodel.ledger'`.

- [ ] **Step 3: Write the implementation**

`src/sportsmodel/ledger/__init__.py`:

````python
"""Pick ledger + pre-kick close capture (profit project, step 1). See
docs/superpowers/specs/2026-10-13-pick-ledger-close-capture-design.md."""
````

`src/sportsmodel/ledger/core.py`:

````python
"""Pick-ledger core: the pick key, the observation record and the append rule. PURE -- no IO.

A ledger row is one OBSERVATION of a pick. The key names the bet slot (which game, market, side, player,
parlay); it deliberately excludes the price AND the line, so a price or number moving between rebuilds is one
pick observed twice, not two picks. `plan_appends` decides which candidate observations become new rows:
a key's first sighting, then again only when the best price changes, the line changes, or the model / reference
probability moves by 1.0 percentage point or more since that key's last row.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

SOURCES = ("model_game", "ev_game", "ev_prop", "parlay_leg")
REF_KINDS = ("pinnacle_novig", "us_consensus_novig")
PROB_MOVE = 0.01          # 1.0 percentage point
LINE_EPS = 1e-9


def as_utc(value) -> datetime | None:
    """A tz-aware UTC datetime from a datetime or ISO string (naive = UTC); None when missing / unparseable."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    if not isinstance(value, datetime):
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def pick_key(source: str, sport: str, game_pk, market: str, side: str, *, player_id=None,
             parlay_id: str | None = None, model_version: str | None = None) -> str:
    """Deterministic key of a bet slot. Fixed positions, never the price or the line.

    `model_version` is part of the key ONLY for model_game picks (two models can lean the same side of the same
    game; each is its own track record). parlay_id comes last because it may itself contain '|'."""
    if source not in SOURCES:
        raise ValueError(f"unknown ledger source: {source!r}")
    mv = (model_version or "") if source == "model_game" else ""
    return "|".join([source, str(sport), str(game_pk), str(market), str(side),
                     "" if player_id is None else str(player_id), mv, parlay_id or ""])


def _f(x):
    return None if x is None else float(x)


def observation(*, source: str, sport: str, game_pk, market: str, side: str, commence_time, line=None,
                player_id=None, player_name=None, parlay_id=None, model_version=None, model_prob=None,
                best_price=None, best_book=None, ref_prob=None, ref_kind=None, ref_books=None,
                ev_pct=None) -> dict:
    """A candidate observation (everything a writer knows). `observed_at`, `minutes_to_kick`, `writer` and
    `writer_run` are stamped by `plan_appends`."""
    if ref_kind is not None and ref_kind not in REF_KINDS:
        raise ValueError(f"unknown ref_kind: {ref_kind!r}")
    return {
        "pick_key": pick_key(source, sport, game_pk, market, side, player_id=player_id,
                             parlay_id=parlay_id, model_version=model_version),
        "source": source, "sport": sport, "game_pk": int(game_pk), "market": market, "side": side,
        "line": _f(line), "player_id": None if player_id is None else str(player_id),
        "player_name": player_name, "parlay_id": parlay_id, "model_version": model_version,
        "model_prob": _f(model_prob), "best_price": None if best_price is None else int(best_price),
        "best_book": best_book, "ref_prob": _f(ref_prob), "ref_kind": ref_kind,
        "ref_books": None if ref_books is None else int(ref_books), "ev_pct": _f(ev_pct),
        "commence_time": commence_time,
    }


def _line_changed(a, b) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return abs(float(a) - float(b)) > LINE_EPS


def _prob_moved(a, b) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return round(abs(float(a) - float(b)), 9) >= PROB_MOVE      # round: 0.60 - 0.59 is 1.0pp, not 0.9999999pp


def change_reason(cand: dict, last: dict | None) -> str | None:
    """Why `cand` should be appended after `last` (the key's most recent row), or None to skip it."""
    if last is None:
        return "first"
    if cand.get("best_price") != last.get("best_price"):
        return "price"
    if _line_changed(cand.get("line"), last.get("line")):
        return "line"
    if _prob_moved(cand.get("model_prob"), last.get("model_prob")) or \
            _prob_moved(cand.get("ref_prob"), last.get("ref_prob")):
        return "prob"
    return None


def plan_appends(candidates: list[dict], last_by_key: dict[str, dict], *, now, writer: str,
                 writer_run: str) -> tuple[list[dict], dict]:
    """(rows to insert, counts). Duplicate keys within a batch keep the LAST candidate. A pick whose game has
    already started (commence_time <= now) is never recorded -- a post-kickoff observation is not bettable --
    and neither is one with no usable commence_time."""
    now = as_utc(now)
    counts: Counter = Counter()
    by_key = {c["pick_key"]: c for c in candidates}
    rows: list[dict] = []
    for key, cand in by_key.items():
        kick = as_utc(cand.get("commence_time"))
        if kick is None:
            counts["invalid"] += 1
            continue
        if kick <= now:
            counts["started"] += 1
            continue
        reason = change_reason(cand, last_by_key.get(key))
        if reason is None:
            counts["unchanged"] += 1
            continue
        counts[reason] += 1
        rows.append({**cand, "commence_time": kick, "observed_at": now,
                     "minutes_to_kick": (kick - now).total_seconds() / 60.0,
                     "writer": writer, "writer_run": writer_run})
    return rows, dict(counts)
````

`src/sportsmodel/ledger/record.py`:

````python
"""Best-effort ledger recording. The pick writers call `safe_record`; whatever goes wrong here (missing
table, DB down, a bug in an adapter) logs a warning and NEVER fails the job that produced the picks."""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Callable, Iterable

from .core import plan_appends


def writes_disabled() -> bool:
    """SM_LEDGER_OFF=1 turns every ledger DB write off. The test suite sets it (root conftest.py) because the
    local .env carries a real DATABASE_URL; the workflows never set it."""
    return os.getenv("SM_LEDGER_OFF", "").strip().lower() in ("1", "true", "yes", "on")


class DbStore:
    """The production store: the db.py ledger helpers, imported lazily."""

    def missing_tables(self) -> list[str]:
        from sportsmodel import db
        return db.missing_ledger_tables(["pick_ledger"])

    def latest(self, keys: list[str]) -> dict[str, dict]:
        from sportsmodel import db
        return db.latest_pick_ledger_rows(keys)

    def insert(self, rows: list[dict]) -> int:
        from sportsmodel import db
        return db.insert_pick_ledger(rows)


def record_observations(candidates: Iterable[dict], *, writer: str, now=None, store=None,
                        run_id: str | None = None) -> dict:
    """Append the candidates that the append rule accepts. Returns a summary dict; never raises."""
    try:
        candidates = [c for c in candidates if c]
        if not candidates:
            return {"status": "empty"}
        if store is None:
            if writes_disabled():
                return {"status": "disabled"}
            store = DbStore()
        gone = store.missing_tables()
        if gone:
            print(f"::warning::pick-ledger: table {gone[0]} missing — run db/migration_pick_ledger.sql "
                  f"({writer}: {len(candidates)} observations not recorded)", flush=True)
            return {"status": "no_table"}
        keys = sorted({c["pick_key"] for c in candidates})
        rows, counts = plan_appends(
            candidates, store.latest(keys), now=now or datetime.now(timezone.utc), writer=writer,
            writer_run=run_id or os.getenv("GITHUB_RUN_ID") or "local")
        if rows:
            store.insert(rows)
        print(f"[pick-ledger] {writer}: {len(candidates)} picks seen, {len(rows)} appended {counts}", flush=True)
        return {"status": "ok", "appended": len(rows), **counts}
    except Exception as exc:  # noqa: BLE001 -- the ledger must never fail the writer
        print(f"::warning::pick-ledger: {writer}: {type(exc).__name__}: {exc}", flush=True)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def safe_record(build: Callable[[], Iterable[dict]], *, writer: str, **kw) -> dict:
    """`record_observations(build(), ...)` with the candidate adapter inside the same safety net."""
    try:
        candidates = list(build())
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::pick-ledger: {writer}: building observations failed: {type(exc).__name__}: {exc}",
              flush=True)
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    return record_observations(candidates, writer=writer, **kw)
````

Then add the production-database guard to the root `conftest.py` (the local `.env` holds a real `DATABASE_URL`, so no test may ever reach the ledger through it):

````diff
--- a/conftest.py
+++ b/conftest.py
@@ -3,4 +3,5 @@
 hatchling editable .pth fragile). Belt-and-suspenders alongside `uv sync`.
 """
+import os
 import sys
 from pathlib import Path
@@ -9,2 +10,6 @@
 if str(SRC) not in sys.path:
     sys.path.insert(0, str(SRC))
+
+# The local .env carries a real DATABASE_URL: no test may reach the pick ledger (or the capture / grading jobs
+# built on it) through the production database. Tests that exercise the ledger inject a fake store instead.
+os.environ["SM_LEDGER_OFF"] = "1"
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger`
Expected: PASS (51 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ledger/__init__.py src/sportsmodel/ledger/core.py src/sportsmodel/ledger/record.py conftest.py tests/ledger/test_core.py tests/ledger/test_record.py
git commit -m "feat(ledger): pick_key, append rule and best-effort recorder with SM_LEDGER_OFF guard"
```

---

### Task 3: Ledger hooks in every pick writer

**Files:**
- Create: `src/sportsmodel/ledger/sources.py`
- Modify: `scripts/build_ev_board.py`, `scripts/build_ev_props_board.py`, `scripts/build_best_parlays.py`, `scripts/generate_cfb.py`, `scripts/generate_sim_nfl.py`, `scripts/generate_sim.py`
- Create: `tests/ledger/test_sources.py`, `tests/ledger/test_hooks.py`

**Interfaces:**
- Consumes: Task 2's `core.observation`, `record.safe_record`; existing `model.distributions.pick_dist/spread_pick_prob/total_pick_prob`, `serving.board.decimal_odds`.
- Produces (pure adapters; each returns a `list[dict]` of `core.observation` dicts):
  - `sources.ev_game_observations(picks, games)`: `picks` = `ev_rows_for_game` rows with `is_pick` (they carry `model_version`, `base_prob`, `best_price`, `best_book`, `ev_best`, `commence_time`); `games` = `assemble_games` output (for the Pinnacle main line)
  - `sources.ev_game_line(game, market, side) -> float | None`
  - `sources.ev_prop_observations(rows)`: `assemble_prop_rows` rows with `is_pick`
  - `sources.parlay_leg_observations(tickets)`: `build_best_parlays` tickets (legs carry `price`, `prob`)
  - `sources.model_game_observations(rows)`: `game_predictions` rows as the generators write them; side logic identical to `grade_predictions._accuracy_row`
- Hook points (all after the writer's own database write; a ledger problem is invisible to the writer):
  `build_ev_board.main` (after `clear_other_side_picks`), `build_ev_props_board.main` (after `upsert_ev_prop_picks`), `build_best_parlays.main` (after `replace_unlocked_best_parlays`), `generate_cfb.write_game_rows` (new helper called from `main`), `generate_sim_nfl._write_game_lines`, `generate_sim.write_game_predictions` (database branch only).

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_sources.py` (includes a parity test that imports `scripts/grade_predictions.py` and checks the recorded side wins exactly when `_accuracy_row` says the pick was correct):

````python
"""Adapters from each writer's rows to ledger observations. Pure."""
import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ledger import sources

KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)
_gp = importlib.util.spec_from_file_location(
    "grade_predictions", Path(__file__).resolve().parents[2] / "scripts" / "grade_predictions.py")
grade_predictions = importlib.util.module_from_spec(_gp)
_gp.loader.exec_module(grade_predictions)


# ---- ev_game -------------------------------------------------------------------------------------

def _game():
    return {"game_pk": 7, "pinnacle": {"moneyline": {"home": -150, "away": 130},
                                       "spread": {"home": -110, "away": -110, "home_line": -3.5},
                                       "total": {"over": -108, "under": -112, "line": 47.5}}}


def _ev(market, side, **kw):
    return {"sport": "nfl", "game_pk": 7, "market": market, "side": side, "commence_time": KICK,
            "model_version": "ev-pilot-v1", "base_prob": 0.55, "best_price": 105, "best_book": "fanduel",
            "ev_best": 0.032, "is_pick": True, **kw}


def test_ev_game_lines_follow_the_side():
    obs = {(o["market"], o["side"]): o for o in sources.ev_game_observations(
        [_ev("spread", "home"), _ev("spread", "away"), _ev("total", "under"), _ev("moneyline", "home")], [_game()])}
    assert obs[("spread", "home")]["line"] == -3.5 and obs[("spread", "away")]["line"] == 3.5
    assert obs[("total", "under")]["line"] == 47.5 and obs[("moneyline", "home")]["line"] is None


def test_ev_game_observation_fields():
    (o,) = sources.ev_game_observations([_ev("spread", "away")], [_game()])
    assert o["source"] == "ev_game" and o["model_prob"] is None
    assert (o["ref_prob"], o["ref_kind"], o["ref_books"]) == (0.55, "pinnacle_novig", 1)
    assert (o["best_price"], o["best_book"]) == (105, "fanduel") and o["ev_pct"] == pytest.approx(3.2)


def test_ev_game_skips_a_spread_whose_line_cannot_be_read():
    assert sources.ev_game_observations([_ev("spread", "home")], [{"game_pk": 7, "pinnacle": {}}]) == []
    assert len(sources.ev_game_observations([_ev("moneyline", "home")], [{"game_pk": 7, "pinnacle": {}}])) == 1


# ---- ev_prop -------------------------------------------------------------------------------------

def test_ev_prop_observation_fields_and_skips():
    row = {"sport": "nfl", "game_pk": 7, "player_id": "00-1", "player_name": "A. Brown", "market": "rec_yds",
           "side": "over", "line": 64.5, "model_version": "props-sim-v1", "commence_time": KICK,
           "model_prob": 0.58, "market_prob": 0.51, "ev_best": 0.07, "best_book": "dk", "best_price": -105}
    (o,) = sources.ev_prop_observations([row, {**row, "line": None}, {**row, "player_id": None}])
    assert o["source"] == "ev_prop" and o["player_id"] == "00-1" and o["line"] == 64.5
    assert (o["model_prob"], o["ref_prob"], o["ref_kind"]) == (0.58, 0.51, None) and o["ev_pct"] == pytest.approx(7.0)
    assert o["pick_key"] == "ev_prop|nfl|7|rec_yds|over|00-1||"


# ---- parlay legs -----------------------------------------------------------------------------------

def test_parlay_legs_become_one_observation_each():
    leg_g = {"key": "g:7:spread:home", "kind": "game", "sport": "nfl", "game_pk": 7, "market": "spread",
             "side": "home", "line": -3.5, "prob": 0.6, "price": -110, "commence_time": KICK,
             "player_id": None, "player_name": None}
    leg_p = {**leg_g, "key": "p:8:00-1:rec_yds:over", "kind": "prop", "game_pk": 8, "market": "rec_yds",
             "side": "over", "line": 64.5, "price": 100, "player_id": "00-1", "player_name": "A. Brown"}
    ticket = {"parlay_id": "draftkings|g:7:spread:home|p:8:00-1:rec_yds:over", "book": "draftkings",
              "model_version": "ev-parlays-v1", "legs": [leg_g, leg_p]}
    a, b = sources.parlay_leg_observations([ticket])
    assert a["source"] == "parlay_leg" and a["parlay_id"] == ticket["parlay_id"] and a["best_book"] == "draftkings"
    assert a["model_prob"] == 0.6 and a["best_price"] == -110
    assert a["ev_pct"] == pytest.approx((0.6 * (1 + 100 / 110) - 1) * 100)
    assert b["player_id"] == "00-1" and b["pick_key"] != a["pick_key"]
    assert a["pick_key"].endswith("|" + ticket["parlay_id"])


# ---- model_game --------------------------------------------------------------------------------------

def _pmf_margin(mean_home_by, sd=13.0, half=60):
    import math
    pmf = [math.exp(-0.5 * ((i - half - mean_home_by) / sd) ** 2) for i in range(2 * half + 1)]
    s = sum(pmf)
    return json.dumps({"kind": "margin", "offset": half, "pmf": [p / s for p in pmf]})


def _pred(**kw):
    base = {"sport": "cfb", "game_pk": 5, "model_version": "cfb-v2", "commence_time": KICK,
            "home_win_prob": 0.7, "pred_home_score": 30.0, "pred_away_score": 20.0,
            "market_spread": -7.0, "market_total": 47.0,
            "margin_dist": _pmf_margin(10.0),
            "total_dist": json.dumps({"kind": "pmf", "pmf": [1.0 / 100] * 100})}
    return {**base, **kw}


def test_model_game_picks_use_prediction_accuracys_side_logic():
    obs = {o["market"]: o for o in sources.model_game_observations([_pred()])}
    assert obs["moneyline"]["side"] == "home" and obs["moneyline"]["model_prob"] == 0.7
    assert obs["spread"]["side"] == "home" and obs["spread"]["line"] == -7.0          # 10 + (-7) > 0
    assert obs["total"]["side"] == "over" and obs["total"]["line"] == 47.0             # 50 > 47
    assert 0.5 < obs["spread"]["model_prob"] < 1.0
    assert all(o["best_price"] is None and o["source"] == "model_game" for o in obs.values())


def test_model_game_away_lean_and_under_lean():
    obs = {o["market"]: o for o in sources.model_game_observations(
        [_pred(home_win_prob=0.4, pred_home_score=24.0, pred_away_score=23.0, market_spread=-7.0, market_total=52.0)])}
    assert obs["moneyline"]["side"] == "away" and obs["moneyline"]["model_prob"] == pytest.approx(0.6)
    assert obs["spread"]["side"] == "away" and obs["spread"]["line"] == 7.0            # 1 + (-7) < 0 -> away +7
    assert obs["total"]["side"] == "under"


def test_model_game_no_pick_on_exact_ties_or_missing_market():
    obs = sources.model_game_observations([_pred(pred_home_score=27.0, pred_away_score=20.0, market_spread=-7.0,
                                                 market_total=None)])
    assert {o["market"] for o in obs} == {"moneyline"}                                  # margin 7 + -7 == 0 -> none
    tie_total = sources.model_game_observations([_pred(pred_home_score=25.0, pred_away_score=22.0, market_total=47.0)])
    assert {o["market"] for o in tie_total} == {"moneyline", "spread"}                  # total 47 == market 47 -> none
    assert {o["market"] for o in sources.model_game_observations([_pred(market_spread=None, market_total=None)])} == {"moneyline"}


def test_model_game_prob_is_null_without_a_distribution():
    obs = {o["market"]: o for o in sources.model_game_observations([_pred(margin_dist=None, total_dist=None)])}
    assert obs["spread"]["model_prob"] is None and obs["total"]["model_prob"] is None


def test_model_game_falls_back_to_pred_margin_and_total_columns():
    row = _pred(pred_home_score=None, pred_away_score=None, pred_margin=10.0, pred_total=50.0)
    assert {o["market"] for o in sources.model_game_observations([row])} == {"moneyline", "spread", "total"}


@pytest.mark.parametrize("pred_margin, pred_total, ms, mt, final", [
    (10.0, 50.0, -7.0, 47.0, (31, 17)),          # home covers, over
    (10.0, 50.0, -7.0, 47.0, (20, 17)),          # away covers, under
    (-3.0, 41.0, 2.5, 44.5, (14, 24)),           # home +2.5, model margin -3 -> away lean; under lean
    (2.0, 55.0, -9.5, 49.5, (30, 24)),
])
def test_side_logic_agrees_with_grade_predictions(pred_margin, pred_total, ms, mt, final):
    """The ledger's lean and prediction_accuracy's lean are the same bet: the side we record wins exactly when
    `_accuracy_row` says spread_pick_correct / total_pick_correct (pushes excluded)."""
    ph, pa = (pred_total + pred_margin) / 2, (pred_total - pred_margin) / 2
    pred = _pred(pred_home_score=ph, pred_away_score=pa, market_spread=ms, market_total=mt, home_win_prob=0.6)
    row = grade_predictions._accuracy_row(
        {**pred, "home_team_name": "H", "away_team_name": "A"},
        {"home_score": final[0], "away_score": final[1], "market_spread": ms, "market_total": mt})
    obs = {o["market"]: o for o in sources.model_game_observations([pred])}
    margin = final[0] - final[1]
    side_covers = (margin + ms > 0) if obs["spread"]["side"] == "home" else (margin + ms < 0)
    if row["spread_pick_correct"] is not None:
        assert row["spread_pick_correct"] == side_covers
    tot = final[0] + final[1]
    side_hits = (tot > mt) if obs["total"]["side"] == "over" else (tot < mt)
    if row["total_pick_correct"] is not None:
        assert row["total_pick_correct"] == side_hits
````

`tests/ledger/test_hooks.py` (stubs every IO call in each script; checks the hook runs after the writer's own write, sees only the surfaced picks, and cannot fail the writer):

````python
"""The ledger hooks in each pick writer: they see exactly the picks the writer just wrote, run AFTER the writer's
own database write, and can never fail the writer. Every IO call in the scripts is stubbed."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ledger import record

ROOT = Path(__file__).resolve().parents[2]
KICK = datetime.now(timezone.utc) + timedelta(hours=30)


def load_script(name):
    spec = importlib.util.spec_from_file_location(f"ledger_hook_{name}", ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Recorder:
    """Stands in for `safe_record`: runs the adapter and keeps what it built, in call order with other events."""

    def __init__(self, events=None):
        self.events, self.calls = events if events is not None else [], []

    def __call__(self, build, *, writer, **kw):
        self.events.append(("ledger", writer))
        self.calls.append((writer, build()))
        return {"status": "ok"}


# ---- build_ev_board ---------------------------------------------------------------------------------

def _stub_board(mod, monkeypatch, events):
    game = {"sport": "nfl", "game_pk": 101, "matchup": "A @ B", "commence_time": KICK}
    odds = []
    for market, side, line, price in [
            ("moneyline", "home", None, -150), ("moneyline", "away", None, 130),
            ("spread", "home", -3.5, -110), ("spread", "away", 3.5, -110),
            ("total", "over", 47.5, -110), ("total", "under", 47.5, -110)]:
        odds.append({"game_pk": 101, "market": market, "side": side, "book": "pinnacle", "line": line, "price": price})
    odds.append({"game_pk": 101, "market": "spread", "side": "home", "book": "draftkings", "line": -3.5, "price": 105})
    monkeypatch.setattr(mod, "load_upcoming_games", lambda sport: [game])
    monkeypatch.setattr(mod, "load_latest_odds", lambda pks: odds)
    monkeypatch.setattr(mod, "upsert_ev_picks", lambda rows: events.append(("upsert", len(rows))) or len(rows))
    monkeypatch.setattr(mod, "clear_other_side_picks", lambda rows: events.append(("clear", 0)) or 0)
    monkeypatch.setattr(mod, "upsert_ev_parlays", lambda rows: 0)
    monkeypatch.setattr(mod, "demote_stale_parlays", lambda *a: 0)
    monkeypatch.setattr(sys, "argv", ["build_ev_board.py", "--sport", "nfl"])


def test_build_ev_board_records_its_surfaced_picks_after_its_own_writes(monkeypatch):
    mod, events = load_script("build_ev_board"), []
    _stub_board(mod, monkeypatch, events)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main()
    assert [e[0] for e in events] == ["upsert", "clear", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "build_ev_board"
    (o,) = obs
    assert (o["source"], o["market"], o["side"], o["line"], o["best_price"], o["best_book"]) == \
        ("ev_game", "spread", "home", -3.5, 105, "draftkings")
    assert o["ref_kind"] == "pinnacle_novig" and o["ref_prob"] == pytest.approx(0.5) and o["model_version"] == "ev-pilot-v1"


def test_build_ev_board_survives_a_ledger_failure(monkeypatch, capsys):
    mod, events = load_script("build_ev_board"), []
    _stub_board(mod, monkeypatch, events)
    monkeypatch.setattr(mod, "ev_game_observations", lambda *a: (_ for _ in ()).throw(KeyError("boom")))
    mod.main()                                                    # real safe_record: must not raise
    assert "::warning::pick-ledger: build_ev_board: building observations failed" in capsys.readouterr().out
    assert ("upsert", 3) in events and ("clear", 0) in events   # the board itself was written


# ---- build_ev_props_board ---------------------------------------------------------------------------

def test_build_ev_props_board_records_only_is_pick_rows_after_the_upsert(monkeypatch):
    mod, events = load_script("build_ev_props_board"), []
    base = {"sport": "nfl", "game_pk": 1, "player_id": "00-1", "player_name": "A. Brown", "market": "rec_yds",
            "side": "over", "line": 64.5, "model_version": "props-sim-v1", "commence_time": KICK,
            "model_prob": 0.58, "market_prob": 0.5, "ev_best": 0.05, "best_book": "dk", "best_price": -105}
    rows = [{**base, "is_pick": True}, {**base, "player_id": "00-2", "is_pick": False}]
    monkeypatch.setattr(mod, "load_sim_rows", lambda sport: [{"game_pk": 1}])
    monkeypatch.setattr(mod, "load_latest_prop_odds", lambda sport, pks: [])
    monkeypatch.setattr(mod, "assemble_prop_line_rows", lambda a, b: [])
    monkeypatch.setattr(mod, "upsert_nfl_prop_lines", lambda rows: 0)
    monkeypatch.setattr(mod, "assemble_prop_rows", lambda *a, **k: rows)
    monkeypatch.setattr(mod, "upsert_ev_prop_picks", lambda r: events.append(("upsert", len(r))))
    monkeypatch.setattr(mod, "clear_stale_prop_line_picks", lambda picks: events.append(("clear", len(picks))) or 0)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main(["--sport", "nfl"])
    assert [e[0] for e in events][:2] == ["upsert", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "build_ev_props_board" and [o["player_id"] for o in obs] == ["00-1"]
    assert obs[0]["line"] == 64.5 and obs[0]["ref_prob"] == 0.5 and obs[0]["model_prob"] == 0.58


# ---- build_best_parlays -----------------------------------------------------------------------------

def test_build_best_parlays_records_every_leg_after_the_replace(monkeypatch):
    mod, events = load_script("build_best_parlays"), []
    leg = lambda pk, market: {"key": f"g:{pk}", "kind": "game", "sport": "nfl", "game_pk": pk, "market": market,
                              "side": "home", "line": -3.5, "prob": 0.6, "price": -110, "commence_time": KICK,
                              "player_id": None, "player_name": None, "label": f"Team {pk}"}
    ticket = {"parlay_id": "dk|g:1|g:2", "book": "dk", "legs": [leg(1, "spread"), leg(2, "spread")],
              "first_commence": None, "ev": 0.05, "parlay_price": 260}
    monkeypatch.setattr(mod, "load_game_picks", lambda: [])
    monkeypatch.setattr(mod, "load_prop_picks", lambda: [])
    monkeypatch.setattr(mod, "assemble_game_legs", lambda a, b: [])
    monkeypatch.setattr(mod, "assemble_prop_legs", lambda a, b: [])
    monkeypatch.setattr(mod, "build_best_parlays", lambda legs, excluded_keys: [ticket])
    monkeypatch.setattr(mod, "locked_leg_keys", lambda: frozenset())
    monkeypatch.setattr(mod, "replace_unlocked_best_parlays", lambda t: events.append(("replace", len(t))) or len(t))
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main()
    assert [e[0] for e in events] == ["replace", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "build_best_parlays" and len(obs) == 2
    assert {o["parlay_id"] for o in obs} == {"dk|g:1|g:2"} and obs[0]["model_version"] == "ev-parlays-v1"


# ---- the three model-game writers --------------------------------------------------------------------

def _pred_row(**kw):
    return {"sport": "nfl", "game_pk": 5, "model_version": "sim-nfl-ml-v2", "commence_time": KICK,
            "home_win_prob": 0.62, "pred_home_score": 27.0, "pred_away_score": 20.0, "market_spread": -3.0,
            "market_total": 44.5, "margin_dist": None, "total_dist": None, **kw}


def test_generate_cfb_write_game_rows_upserts_then_records(monkeypatch):
    mod, events = load_script("generate_cfb"), []
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: events.append(("upsert", len(rows))))
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.write_game_rows([_pred_row(sport="cfb")])
    assert [e[0] for e in events] == ["upsert", "ledger"]
    ((writer, obs),) = rec.calls
    assert writer == "generate_cfb" and {o["market"] for o in obs} == {"moneyline", "spread", "total"}


def test_generate_sim_nfl_game_lines_record_the_rows_they_wrote(monkeypatch):
    mod, events = load_script("generate_sim_nfl"), []
    wrote = []
    monkeypatch.setattr(mod, "game_prediction_row", lambda g, sims, cfg, mv: {**_pred_row(), "margin_dist": {}, "total_dist": {}})
    monkeypatch.setattr(mod.nfl_config, "load_gameline", lambda: None)
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: wrote.extend(rows) or events.append(("upsert", len(rows))))
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod._write_game_lines([{"game_pk": 5}], {5: object()}, {5})
    assert [e[0] for e in events] == ["upsert", "ledger"]
    assert rec.calls[0][0] == "generate_sim_nfl" and len(rec.calls[0][1]) >= 1


def test_generate_sim_records_mlb_picks_only_on_the_database_branch(monkeypatch):
    mod, events = load_script("generate_sim"), []
    pred = {k: v for k, v in _pred_row().items() if k != "sport"} | {"model_version": "sim-mlb-v1"}
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: events.append(("upsert", len(rows))) or len(rows))
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    monkeypatch.setattr(mod.config, "DATABASE_URL", "postgresql://x")
    mod.write_game_predictions([pred])
    assert [e[0] for e in events] == ["upsert", "ledger"]
    (writer, obs) = rec.calls[0]
    assert writer == "generate_sim" and {o["sport"] for o in obs} == {"mlb"}


def test_a_ledger_failure_never_fails_a_model_game_writer(monkeypatch, capsys):
    mod = load_script("generate_cfb")
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: None)
    monkeypatch.setattr(mod, "model_game_observations", lambda rows: (_ for _ in ()).throw(RuntimeError("x")))
    mod.write_game_rows([_pred_row()])                            # real safe_record swallows the adapter error
    assert "::warning::pick-ledger: generate_cfb" in capsys.readouterr().out


def test_the_hooks_never_reach_the_database_under_test(monkeypatch):
    """SM_LEDGER_OFF (conftest.py) means even the real safe_record cannot touch the DB from a script main()."""
    assert record.writes_disabled()
    mod = load_script("generate_cfb")
    monkeypatch.setattr(mod, "upsert_game_predictions", lambda rows: None)
    mod.write_game_rows([_pred_row()])
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_sources.py tests/ledger/test_hooks.py`
Expected: collection ERROR, `ImportError: cannot import name 'sources' from 'sportsmodel.ledger'`.

- [ ] **Step 3: Write the adapters**

`src/sportsmodel/ledger/sources.py`:

````python
"""Adapters: what each pick writer already holds -> candidate ledger observations. PURE -- no IO.

Every adapter returns `core.observation` dicts; `record.safe_record` appends the ones the append rule accepts.
Nothing here changes a pick, a probability or a price: it only copies what the writer computed.
"""
from __future__ import annotations

from ..model.distributions import pick_dist, spread_pick_prob, total_pick_prob
from ..serving.board import decimal_odds
from . import core


def _pct(x):
    return None if x is None else float(x) * 100.0


def ev_game_line(game: dict, market: str, side: str) -> float | None:
    """The SIDE's own line on the +EV game board: home spread (negated for away), the total, None for moneyline.
    Read from the Pinnacle main line the board prices against (`assemble_games` output)."""
    pin = (game or {}).get("pinnacle") or {}
    if market == "spread":
        hl = (pin.get("spread") or {}).get("home_line")
        return None if hl is None else (hl if side == "home" else -hl)
    if market == "total":
        return (pin.get("total") or {}).get("line")
    return None


def ev_game_observations(picks: list[dict], games: list[dict]) -> list[dict]:
    """+EV game-line picks (`ev_rows_for_game` rows with is_pick) -> ev_game observations. The reference is the
    board's own Pinnacle no-vig `base_prob`; the board carries no model probability. A spread / total pick whose
    line cannot be read is skipped (it could not be graded)."""
    by_pk = {g.get("game_pk"): g for g in games}
    out = []
    for r in picks:
        line = ev_game_line(by_pk.get(r["game_pk"]), r["market"], r["side"])
        if r["market"] in ("spread", "total") and line is None:
            continue
        base = r.get("base_prob")
        out.append(core.observation(
            source="ev_game", sport=r["sport"], game_pk=r["game_pk"], market=r["market"], side=r["side"],
            commence_time=r.get("commence_time"), line=line, model_version=r.get("model_version"),
            best_price=r.get("best_price"), best_book=r.get("best_book"), ref_prob=base,
            ref_kind="pinnacle_novig" if base is not None else None, ref_books=1 if base is not None else None,
            ev_pct=_pct(r.get("ev_best"))))
    return out


def ev_prop_observations(rows: list[dict]) -> list[dict]:
    """+EV prop picks (`assemble_prop_rows` rows with is_pick) -> ev_prop observations. `ref_prob` is the board's
    own `market_prob` (no-vig of the best over / best under prices); it is not a graded reference, so ref_kind
    stays NULL. Graded closes come from the close capture, never from this field."""
    out = []
    for r in rows:
        if r.get("line") is None or r.get("player_id") is None:
            continue
        out.append(core.observation(
            source="ev_prop", sport=r["sport"], game_pk=r["game_pk"], market=r["market"], side=r["side"],
            commence_time=r.get("commence_time"), line=r["line"], player_id=r["player_id"],
            player_name=r.get("player_name"), model_version=r.get("model_version"),
            model_prob=r.get("model_prob"), best_price=r.get("best_price"), best_book=r.get("best_book"),
            ref_prob=r.get("market_prob"), ev_pct=_pct(r.get("ev_best"))))
    return out


def parlay_leg_observations(tickets: list[dict]) -> list[dict]:
    """Each leg of each +EV parlay ticket -> a parlay_leg observation at the ticket's book and the leg's price."""
    out = []
    for t in tickets:
        for leg in t.get("legs") or []:
            price, prob = leg.get("price"), leg.get("prob")
            ev = None if price is None or prob is None else (float(prob) * decimal_odds(price) - 1) * 100.0
            out.append(core.observation(
                source="parlay_leg", sport=leg["sport"], game_pk=leg["game_pk"], market=leg["market"],
                side=leg["side"], commence_time=leg.get("commence_time"), line=leg.get("line"),
                player_id=leg.get("player_id"), player_name=leg.get("player_name"), parlay_id=t["parlay_id"],
                model_version=t.get("model_version"), model_prob=prob, best_price=price,
                best_book=t.get("book"), ev_pct=ev))
    return out


def _margin_total(row: dict) -> tuple[float | None, float | None]:
    """(pred_margin, pred_total) the way scripts/grade_predictions.py `_accuracy_row` derives them."""
    ph, pa = row.get("pred_home_score"), row.get("pred_away_score")
    if ph is not None and pa is not None:
        return ph - pa, ph + pa
    return row.get("pred_margin"), row.get("pred_total")


def model_game_observations(rows: list[dict]) -> list[dict]:
    """`game_predictions` rows as the generators write them -> the model's game picks, using the SAME side logic
    as `prediction_accuracy` (scripts/grade_predictions.py):

      moneyline  home when home_win_prob >= 0.5, else away;
      spread     edge = pred_margin + market_spread: > 0 home (line = market_spread), < 0 away
                 (line = -market_spread), == 0 no pick;
      total      Over when pred_total > market_total, Under when below, equal = no pick.

    model_prob is the model's probability for that side at that line (spread / total from the stored
    distribution, pushes excluded; NULL without one). No market price is attached: model picks are graded at
    flat -110 on spreads / totals and have no price on moneylines."""
    out = []
    for r in rows:
        base = dict(source="model_game", sport=r.get("sport"), game_pk=r["game_pk"],
                    commence_time=r.get("commence_time"), model_version=r.get("model_version"))
        hwp = r.get("home_win_prob")
        if hwp is not None:
            home = hwp >= 0.5
            out.append(core.observation(**base, market="moneyline", side="home" if home else "away",
                                        model_prob=hwp if home else 1.0 - hwp))
        margin, total = _margin_total(r)
        ms, mt = r.get("market_spread"), r.get("market_total")
        if margin is not None and ms is not None and margin + ms != 0:
            home = margin + ms > 0
            out.append(core.observation(
                **base, market="spread", side="home" if home else "away", line=ms if home else -ms,
                model_prob=spread_pick_prob(pick_dist("margin_dist", r), margin, ms)))
        if total is not None and mt is not None and total != mt:
            out.append(core.observation(
                **base, market="total", side="over" if total > mt else "under", line=mt,
                model_prob=total_pick_prob(pick_dist("total_dist", r), total, mt)))
    return out
````

- [ ] **Step 4: Hook the writers**

Apply these diffs (each adds an import and one `safe_record(...)` call; `generate_cfb.py` also moves its two lines into a small `write_game_rows` helper so the hook is testable without running the whole model):

`scripts/build_ev_board.py`:

````diff
--- a/scripts/build_ev_board.py
+++ b/scripts/build_ev_board.py
@@ -33,4 +33,6 @@
     assemble_games, ev_rows_for_game, MIN_EV, EV_CEILING, MAX_STRAIGHT_JUICE,
 )
+from sportsmodel.ledger.record import safe_record
+from sportsmodel.ledger.sources import ev_game_observations
 from sportsmodel.serving.parlay import build_parlay
 
@@ -134,4 +136,7 @@
     cleared = clear_other_side_picks(all_rows)
 
+    # Pick ledger: the surfaced picks exactly as they stand now. Best-effort -- never fails the build.
+    safe_record(lambda: ev_game_observations(picks, games), writer="build_ev_board")
+
     # Auto-parlay the juiced-favorite legs into a single +EV ticket. Non-fatal:
     # if the ev_parlays table isn't migrated yet, the straight board above is
````

`scripts/build_ev_props_board.py`:

````diff
--- a/scripts/build_ev_props_board.py
+++ b/scripts/build_ev_props_board.py
@@ -42,4 +42,6 @@
     upsert_nfl_prop_lines,
 )
+from sportsmodel.ledger.record import safe_record
+from sportsmodel.ledger.sources import ev_prop_observations
 from sportsmodel.serving.props_ev import (
     MLB_SIM_TO_ODDS_MARKET,
@@ -207,4 +209,6 @@
 
     picks = [r for r in rows if r["is_pick"]]
+    # Pick ledger: the surfaced picks exactly as they stand now. Best-effort -- never fails the build.
+    safe_record(lambda: ev_prop_observations(picks), writer="build_ev_props_board")
     # A player's main line can shift between builds (books move the number);
     # since `line` is part of ev_prop_picks' primary key, the OLD line's row
````

`scripts/build_best_parlays.py`:

````diff
--- a/scripts/build_best_parlays.py
+++ b/scripts/build_best_parlays.py
@@ -15,4 +15,6 @@
 
 from sportsmodel.db import get_postgres, locked_leg_keys, replace_unlocked_best_parlays
+from sportsmodel.ledger.record import safe_record
+from sportsmodel.ledger.sources import parlay_leg_observations
 from sportsmodel.serving.best_parlays import (
     assemble_game_legs, assemble_prop_legs, build_best_parlays)
@@ -89,4 +91,6 @@
         t["model_version"] = MODEL_VERSION
     n = replace_unlocked_best_parlays(tickets)
+    # Pick ledger: every leg of every ticket as built. Best-effort -- never fails the build.
+    safe_record(lambda: parlay_leg_observations(tickets), writer="build_best_parlays")
     priced = sum(1 for l in legs if l["book_prices"])
     print(f"[build_best_parlays] legs={len(legs)} priced={priced} tickets={n}")
````

`scripts/generate_cfb.py`:

````diff
--- a/scripts/generate_cfb.py
+++ b/scripts/generate_cfb.py
@@ -59,4 +59,6 @@
 from sportsmodel.cfb.walkforward import SeasonState, live_week_state, model_margin_total
 from sportsmodel.db import upsert_game_predictions
+from sportsmodel.ledger.record import safe_record
+from sportsmodel.ledger.sources import model_game_observations
 from sportsmodel.nfl.elo import EloConfig
 from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
@@ -298,4 +300,11 @@
 
 
+def write_game_rows(game_rows: list[dict]) -> None:
+    """Persist the week's game_predictions, then record the model's game picks in the pick ledger
+    (best-effort -- a ledger problem never fails the job)."""
+    upsert_game_predictions(game_rows)
+    safe_record(lambda: model_game_observations(game_rows), writer="generate_cfb")
+
+
 def main() -> None:
     cur = espn.fetch_current_week()
@@ -322,5 +331,5 @@
 
     if config.DATABASE_URL:
-        upsert_game_predictions(game_rows)
+        write_game_rows(game_rows)
 
     print(f"predicted {len(game_rows)} games")
````

`scripts/generate_sim_nfl.py`:

````diff
--- a/scripts/generate_sim_nfl.py
+++ b/scripts/generate_sim_nfl.py
@@ -174,4 +174,6 @@
 from sportsmodel.db import (get_postgres, served_nfl_sim_version, upsert_game_predictions,
                             upsert_nfl_player_sim, upsert_nfl_sim, upsert_nfl_sim_slate)
+from sportsmodel.ledger.record import safe_record
+from sportsmodel.ledger.sources import model_game_observations
 from sportsmodel.nfl import config as nfl_config
 from sportsmodel.nfl.injuries_nflverse import nfl_season
@@ -694,4 +696,6 @@
                      "total_dist": json.dumps(row["total_dist"])})
     upsert_game_predictions(rows)
+    # Pick ledger: the model's game picks at generation time. Best-effort -- never fails the job.
+    safe_record(lambda: model_game_observations(rows), writer="generate_sim_nfl")
     n_ml = sum(1 for r in rows if r["game_pk"] in ml_served)
     print(f"game_lines: wrote {len(rows)} game_predictions rows under {model_version} "
````

`scripts/generate_sim.py`:

````diff
--- a/scripts/generate_sim.py
+++ b/scripts/generate_sim.py
@@ -38,4 +38,6 @@
 from sportsmodel.ingest import mlb_lineups, mlb_statsapi
 from sportsmodel.ingest.odds import parse_commence
+from sportsmodel.ledger.record import safe_record
+from sportsmodel.ledger.sources import model_game_observations
 from sportsmodel.model import calibration, distributions, game, props, rates
 from sportsmodel.sim import engine
@@ -244,4 +246,7 @@
         n = upsert_game_predictions(preds)
         print(f"Upserted {n} rows into Supabase game_predictions.")
+        # Pick ledger: the model's game picks at generation time. Best-effort -- never fails the job.
+        safe_record(lambda: model_game_observations([{**p, "sport": "mlb"} for p in preds]),
+                    writer="generate_sim")
         return
     con = get_duckdb()
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger tests/test_build_ev_board.py tests/test_build_ev_props_board.py tests/test_build_best_parlays.py tests/test_generate_sim_mlb.py tests/test_grade_predictions.py`
Expected: all PASS (`tests/ledger` alone: 74 passed).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/ledger/sources.py scripts/build_ev_board.py scripts/build_ev_props_board.py scripts/build_best_parlays.py scripts/generate_cfb.py scripts/generate_sim_nfl.py scripts/generate_sim.py tests/ledger/test_sources.py tests/ledger/test_hooks.py
git commit -m "feat(ledger): record picks from the board, props, parlay and model-game writers"
```

---

### Task 4: Pinnacle-props probe (one `[NETWORK]` step)

**Files:**
- Modify: `src/sportsmodel/ingest/odds.py` (record `x-requests-last`)
- Create: `src/sportsmodel/ledger/probe.py`, `scripts/probe_pinnacle_props.py`, `.github/workflows/probe-pinnacle-props.yml`
- Create: `tests/ledger/test_probe.py`

**Interfaces:**
- Consumes: existing `odds.fetch_events(cfg)` (free) and `odds.fetch_event_props(event_id, markets, cfg, regions=...)`, `odds.parse_commence`, `sports.get("nfl")`.
- Produces:
  - `odds.last_requests_last: str | None` (the `x-requests-last` header: credits the last call cost; used by Task 5's credit logging)
  - `probe.pinnacle_props_verdict(payload: dict | None) -> {"verdict": "pinnacle_has_props" | "pinnacle_absent" | "inconclusive", "pinnacle_outcomes": int, "pinnacle_markets": list[str], "bookmaker_outcomes": dict[str, int]}`
  - `probe_pinnacle_props.next_event(events, now)`, `.run(now, fetch_events, fetch_props, remaining, last_cost) -> dict`, `.main() -> int`; prints `PROBE_VERDICT=...`, `PROBE_CREDITS_LAST=...`, `PROBE_CREDITS_REMAINING=...`
  - Repository variable **`CLOSE_PROP_REGIONS`** (read by Task 5's workflow): unset means `us,eu`; the controller sets it to `us` when the verdict is `pinnacle_absent`.

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_probe.py`:

````python
"""The Pinnacle-props probe: verdict logic, event choice, the odds client's credit headers and the workflow."""
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ingest import odds
from sportsmodel.ledger.probe import pinnacle_props_verdict
from tests.test_workflows_props_ml import WF, load, runs, steps_of

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("probe_pinnacle_props", ROOT / "scripts" / "probe_pinnacle_props.py")
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

NOW = datetime(2026, 10, 13, 15, 0, tzinfo=timezone.utc)


def _book(key, *market_outcomes):
    return {"key": key, "markets": [{"key": f"player_m{i}", "outcomes": [{}] * n}
                                    for i, n in enumerate(market_outcomes)]}


def test_verdict_pinnacle_has_props():
    v = pinnacle_props_verdict({"bookmakers": [_book("pinnacle", 4, 0, 2), _book("betfair_ex_eu", 6)]})
    assert v["verdict"] == "pinnacle_has_props" and v["pinnacle_outcomes"] == 6
    assert v["pinnacle_markets"] == ["player_m0", "player_m2"] and v["bookmaker_outcomes"]["betfair_ex_eu"] == 6


def test_verdict_pinnacle_absent_when_other_books_have_props():
    v = pinnacle_props_verdict({"bookmakers": [_book("betfair_ex_eu", 3), _book("pinnacle", 0, 0)]})
    assert v["verdict"] == "pinnacle_absent" and v["pinnacle_outcomes"] == 0


@pytest.mark.parametrize("payload", [None, {}, {"bookmakers": []}, {"bookmakers": [_book("pinnacle", 0)]}])
def test_verdict_inconclusive_when_no_book_returned_any_prop(payload):
    assert pinnacle_props_verdict(payload)["verdict"] == "inconclusive"


def test_next_event_is_the_soonest_unstarted_one():
    ev = lambda i, h: {"id": i, "commence_time": (NOW + timedelta(hours=h)).isoformat()}
    assert probe.next_event([ev("late", 80), ev("past", -2), ev("soon", 30), {"id": "x"}], NOW)["id"] == "soon"
    assert probe.next_event([ev("past", -2)], NOW) is None


def test_run_reports_the_verdict_and_credits():
    asked = []
    events = [{"id": "e1", "home_team": "B", "away_team": "A", "commence_time": (NOW + timedelta(hours=30)).isoformat()}]
    rep = probe.run(NOW, lambda: events, lambda eid: asked.append(eid) or {"bookmakers": [_book("pinnacle", 5)]},
                    lambda: "19000", lambda: "8")
    assert asked == ["e1"] and rep["verdict"] == "pinnacle_has_props" and rep["event"] == "A @ B"
    assert (rep["credits_last"], rep["credits_remaining"]) == ("8", "19000")


def test_run_with_no_upcoming_event_spends_nothing():
    def boom(eid):
        raise AssertionError("paid call")
    rep = probe.run(NOW, lambda: [], boom, lambda: "19000", lambda: None)
    assert rep["verdict"] == "inconclusive" and rep["reason"] == "no upcoming NFL event"


def test_the_probe_costs_about_eight_credits_one_event_eu_only():
    from sportsmodel import sports
    assert len(sports.get("nfl").prop_market_map) == 8          # 8 markets x 1 region x 1 event = 8 credits


def test_main_refuses_without_a_key(monkeypatch, capsys):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    assert probe.main() == 1 and "ODDS_API_KEY is not set" in capsys.readouterr().out


# ---- odds client: credit headers --------------------------------------------------------------------

class _Resp:
    headers = {"x-requests-remaining": "18990", "x-requests-last": "8"}

    def raise_for_status(self):
        pass

    def json(self):
        return {"ok": True}


class _Client:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, params=None):
        return _Resp()


def test_get_records_remaining_and_last_call_cost(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    monkeypatch.setattr(odds.httpx, "Client", _Client)
    assert odds._get("/sports/x/events", {}) == {"ok": True}
    assert odds.last_requests_remaining == "18990" and odds.last_requests_last == "8"


# ---- workflow --------------------------------------------------------------------------------------

def test_probe_workflow_is_manual_read_only_and_runs_the_script_with_the_key():
    wf = load("probe-pinnacle-props.yml")
    assert list(wf["on"]) == ["workflow_dispatch"] and wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    (step,) = [s for s in steps_of(job) if runs("scripts/probe_pinnacle_props.py")(s)]
    assert step["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}" and "DATABASE_URL" not in step["env"]
    text = (WF / "probe-pinnacle-props.yml").read_text()
    assert "git push" not in text and "DATABASE_URL" not in text
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_probe.py`
Expected: collection ERROR, `ModuleNotFoundError: No module named 'sportsmodel.ledger.probe'`.

- [ ] **Step 3: Write the implementation**

Record the credit-cost header in the Odds API client (additive; `last_requests_remaining` behaves as before):

````diff
--- a/src/sportsmodel/ingest/odds.py
+++ b/src/sportsmodel/ingest/odds.py
@@ -46,6 +46,8 @@
                         if k in ("total_bases", "pitcher_ks", "hits_allowed", "outs_recorded")}
 
-# credits remaining, updated from the last response header (for logging)
+# credits remaining and credits the LAST call cost, updated from the last response headers
+# (x-requests-remaining / x-requests-last) -- for logging and the close-capture budget guard
 last_requests_remaining: str | None = None
+last_requests_last: str | None = None
 
 
@@ -59,9 +61,10 @@
 @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
 def _get(path: str, params: dict[str, Any]) -> Any:
-    global last_requests_remaining
+    global last_requests_remaining, last_requests_last
     with httpx.Client(timeout=25) as client:
         resp = client.get(f"{BASE}{path}", params={**params, "apiKey": _key()})
         resp.raise_for_status()
         last_requests_remaining = resp.headers.get("x-requests-remaining")
+        last_requests_last = resp.headers.get("x-requests-last")
         return resp.json()
````

`src/sportsmodel/ledger/probe.py`:

````python
"""Does Pinnacle publish player props through The Odds API? Pure verdict over ONE event-odds payload
(`GET /v4/sports/{sport}/events/{id}/odds?regions=eu&markets=<prop markets>`)."""
from __future__ import annotations

PINNACLE = "pinnacle"


def pinnacle_props_verdict(payload: dict | None) -> dict:
    """Classify one event-odds payload.

      pinnacle_has_props  Pinnacle returned at least one prop outcome -- keep regions=us,eu for prop captures.
      pinnacle_absent     other books returned props but Pinnacle returned none -- conclusive: prop captures
                          use regions=us only (half the credits).
      inconclusive        NO book returned any prop (props not posted yet, or the event is too far out) --
                          re-run closer to kickoff before deciding.
    """
    books: dict[str, int] = {}
    pin_markets: list[str] = []
    for bk in (payload or {}).get("bookmakers") or []:
        n = 0
        for m in bk.get("markets") or []:
            k = len(m.get("outcomes") or [])
            n += k
            if bk.get("key") == PINNACLE and k:
                pin_markets.append(m.get("key"))
        books[bk.get("key")] = books.get(bk.get("key"), 0) + n
    pin_outcomes = books.get(PINNACLE, 0)
    if pin_outcomes > 0:
        verdict = "pinnacle_has_props"
    elif any(n > 0 for n in books.values()):
        verdict = "pinnacle_absent"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "pinnacle_outcomes": pin_outcomes, "pinnacle_markets": sorted(set(pin_markets)),
            "bookmaker_outcomes": dict(sorted(books.items()))}
````

`scripts/probe_pinnacle_props.py`:

````python
"""One-off probe: does Pinnacle publish NFL player props through The Odds API?

Costs about 8 credits (the 8 NFL prop markets x regions=eu x ONE event; listing events is free). It picks the
next upcoming NFL event, asks for its props from the EU region only, and prints a verdict:

    PROBE_VERDICT=pinnacle_has_props | pinnacle_absent | inconclusive

Run it through .github/workflows/probe-pinnacle-props.yml (ODDS_API_KEY lives in GitHub secrets, not in .env).
It never touches the database.

    ODDS_API_KEY=... uv run python scripts/probe_pinnacle_props.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import sports
from sportsmodel.ingest import odds
from sportsmodel.ledger.probe import pinnacle_props_verdict


def next_event(events: list[dict], now: datetime) -> dict | None:
    """The soonest event that has not started yet."""
    upcoming = [(c, e) for e in events if (c := odds.parse_commence(e.get("commence_time"))) and c > now]
    return min(upcoming, key=lambda t: t[0])[1] if upcoming else None


def run(now: datetime, fetch_events, fetch_props, remaining, last_cost) -> dict:
    """The probe on injected IO (`fetch_events() -> list`, `fetch_props(event_id) -> dict`,
    `remaining()` / `last_cost()` -> header strings or None). Returns the report dict."""
    event = next_event(fetch_events(), now)
    if event is None:
        return {"verdict": "inconclusive", "reason": "no upcoming NFL event", "credits_last": None,
                "credits_remaining": remaining()}
    payload = fetch_props(event["id"])
    return {**pinnacle_props_verdict(payload), "event": f"{event.get('away_team')} @ {event.get('home_team')}",
            "commence_time": event.get("commence_time"), "credits_last": last_cost(),
            "credits_remaining": remaining()}


def main() -> int:
    if not os.getenv("ODDS_API_KEY"):
        print("::error::ODDS_API_KEY is not set")
        return 1
    cfg = sports.get("nfl")
    markets = list(cfg.prop_market_map.values())
    report = run(
        datetime.now(timezone.utc),
        lambda: odds.fetch_events(cfg),
        lambda eid: odds.fetch_event_props(eid, markets, cfg, regions="eu"),
        lambda: odds.last_requests_remaining,
        lambda: odds.last_requests_last)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    print(f"PROBE_VERDICT={report['verdict']}")
    print(f"PROBE_CREDITS_LAST={report['credits_last']}")
    print(f"PROBE_CREDITS_REMAINING={report['credits_remaining']}")
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"### Pinnacle props probe\n\n**{report['verdict']}**\n\n```json\n{text}\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

`.github/workflows/probe-pinnacle-props.yml`:

````yaml
name: probe-pinnacle-props

# ONE-OFF, manual only: does Pinnacle publish NFL player props through The Odds API? Costs about 8 credits
# (8 prop markets x regions=eu x one upcoming event; listing events is free). Prints
# PROBE_VERDICT=pinnacle_has_props | pinnacle_absent | inconclusive in the log and the job summary.
# Never touches the database. Result decides CLOSE_PROP_REGIONS for capture-close.yml (see the pick-ledger plan).
on:
  workflow_dispatch: {}

permissions:
  contents: read

jobs:
  probe:
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
      - name: Sync deps
        run: uv sync
      - name: Probe Pinnacle NFL props (about 8 credits)
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
        run: uv run python scripts/probe_pinnacle_props.py
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger tests/test_odds.py tests/test_ingest_odds.py tests/test_workflows_props_ml.py`
Expected: all PASS (`tests/ledger` alone: 87 passed; `test_every_workflow_parses` now also parses the probe workflow).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ingest/odds.py src/sportsmodel/ledger/probe.py scripts/probe_pinnacle_props.py .github/workflows/probe-pinnacle-props.yml tests/ledger/test_probe.py
git commit -m "feat(ledger): Pinnacle props probe workflow and Odds API credit header"
```

- [ ] **Step 6: `[NETWORK]` Run the probe once (controller; about 8 credits; needs the user's go-ahead to push)**

`ODDS_API_KEY` is in GitHub secrets only (it is not in the local `.env`), so the probe runs as a manually dispatched workflow. The controller does this once, after Step 5, and records the verdict in the final PR description. This step spends 8 of the 30,000 monthly credits (the spec budgets about 10) and reads/writes no database.

1. Ask the user for permission to push the branch (pushing is only needed for the probe): `git push -u origin feat/pick-ledger`.
2. Dispatch it from the branch: `gh workflow run probe-pinnacle-props.yml --ref feat/pick-ledger`.
3. Find and follow the run (wait a few seconds after dispatching):
   ```bash
   RUN=$(gh run list --workflow=probe-pinnacle-props.yml --branch feat/pick-ledger --limit 1 --json databaseId --jq '.[0].databaseId')
   gh run watch "$RUN" --exit-status
   gh run view "$RUN" --log | grep -E "PROBE_|pinnacle"
   ```
4. Read `PROBE_VERDICT`:
   - `pinnacle_has_props`: keep the default (`CLOSE_PROP_REGIONS` unset = `us,eu`). Nothing to do.
   - `pinnacle_absent` (conclusive: other EU books returned props, Pinnacle none): ask the user, then `gh variable set CLOSE_PROP_REGIONS --body us`. This is a GitHub repository variable (not the database); it halves the prop credits. The weekly gate report (Task 8) states the regions the prop closes actually used; also note the verdict in the PR.
   - `inconclusive` (no book had props posted yet, normally because the event is too far out): re-dispatch on a day when the next game's props are up (Wednesday or Thursday for a Thursday-night game). Costs another 8 credits; do not decide on an inconclusive result.
5. If `gh workflow run` answers that it cannot find the workflow (GitHub only lists `workflow_dispatch` workflows that exist on the default branch), do NOT work around it: either (a) run the probe immediately after the user merges the PR, before the first football Thursday, or (b) ask the user to run it once locally with their own key: `ODDS_API_KEY=... uv run python scripts/probe_pinnacle_props.py` (prints the same verdict; no database access). Until a verdict exists, the safe default `us,eu` costs at most about 90 extra credits a week.

---

### Task 5: Close-capture job with the budget guard

**Files:**
- Create: `src/sportsmodel/ledger/close.py`, `scripts/capture_close.py`, `.github/workflows/capture-close.yml`
- Create: `tests/ledger/test_close.py`, `tests/ledger/test_capture_close_script.py`, `tests/ledger/test_workflows_capture.py`

**Interfaces:**
- Consumes: Task 1's `db.missing_ledger_tables`, `db.upsert_close_capture`, `db.upsert_odds_snapshot` (existing); Task 4's `odds.last_requests_last`; existing `odds.fetch_events`, `odds.fetch_game_odds`, `odds.fetch_event_props`, `odds.parse_game_odds`, `odds.parse_prop_odds`, `ingest_odds.build_game_lookup`, `ingest_odds._ESPN_FETCHERS`, `ingest_odds._matcher_for` (imported from `scripts/ingest_odds.py` by adding `scripts/` to `sys.path`); Task 2's `record.writes_disabled`; `props_ev.SIM_TO_ODDS_MARKET` / `MLB_SIM_TO_ODDS_MARKET`.
- Produces:
  - `close.WINDOW_MIN = 5`, `WINDOW_MAX = 25`, `PREFILTER_MIN = 60`, `BUDGET_PROPS_FLOOR = 3000`, `BUDGET_ALL_FLOOR = 1000`
  - `close.minutes_to_kick(commence, now)`, `close.prefilter_sports(candidates, now, horizon_min=60) -> list[str]`, `close.select_targets(games, captured, sport, kind, now, lo=5, hi=25) -> list[dict]`, `close.prop_regions(env) -> str`, `close.estimated_cost(n_markets, regions) -> int`, `close.parse_credits(value) -> int | None`, `close.split_credits(total, n) -> list[int]`, `close.budget_decision(remaining) -> BudgetDecision(lines, props, warning)`, `close.api_markets_for(sport, ledger_markets, prop_market_map) -> list[str]`, `close.capture_row(...) -> dict` (columns of `db.LEDGER_COLUMNS["close_capture"]`)
  - `capture_close.Deps` (dataclass of injectable IO), `capture_close.run_capture(now, deps, *, prop_regions, dry_run) -> dict` (`status` is `idle`, `no_table` or `ok`), `capture_close.main(argv) -> int`
  - Rows written: `odds_snapshot` rows for the target games only (all with `captured_at = run time`), and one `close_capture` row per event and kind (`lines` / `props`) with status `ok` / `empty` / `failed`.

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_close.py` (pure planning):

````python
"""Close-capture planning: window selection, budget guard, credit maths, prop-market planning. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel import sports
from sportsmodel.ledger import close

NOW = datetime(2026, 10, 18, 16, 40, tzinfo=timezone.utc)


def _g(pk, minutes, **kw):
    return {"game_pk": pk, "commence_time": (NOW + timedelta(minutes=minutes)).isoformat(), **kw}


def test_window_is_five_to_twenty_five_minutes_inclusive():
    games = [_g(1, 4.9), _g(2, 5), _g(3, 15), _g(4, 25), _g(5, 25.1), _g(6, -3), {"game_pk": 7, "commence_time": None}]
    assert [g["game_pk"] for g in close.select_targets(games, set(), "nfl", "lines", NOW)] == [2, 3, 4]


def test_the_window_closes_inside_the_gates_thirty_minute_bound():
    assert close.WINDOW_MAX <= 30 and close.WINDOW_MIN == 5


def test_already_captured_events_are_skipped_per_kind_and_sport():
    games = [_g(1, 12), _g(2, 12)]
    captured = {("nfl", 1, "lines"), ("cfb", 2, "lines"), ("nfl", 2, "props")}
    assert [g["game_pk"] for g in close.select_targets(games, captured, "nfl", "lines", NOW)] == [2]
    assert [g["game_pk"] for g in close.select_targets(games, captured, "nfl", "props", NOW)] == [1]


def test_targets_carry_their_minutes_to_kick():
    (t,) = close.select_targets([_g(1, 12)], set(), "nfl", "lines", NOW)
    assert t["minutes_to_kick"] == pytest.approx(12)


def test_prefilter_lists_only_sports_with_a_kickoff_in_the_next_hour_in_fixed_order():
    cands = [{"sport": "mlb", "game_pk": 1, "commence_time": NOW + timedelta(minutes=50)},
             {"sport": "nfl", "game_pk": 2, "commence_time": (NOW + timedelta(minutes=59)).isoformat()},
             {"sport": "cfb", "game_pk": 3, "commence_time": NOW + timedelta(minutes=61)},
             {"sport": "cfb", "game_pk": 4, "commence_time": NOW - timedelta(minutes=1)},
             {"sport": "cfb", "game_pk": 5, "commence_time": None}]
    assert close.prefilter_sports(cands, NOW) == ["nfl", "mlb"]
    assert close.prefilter_sports([], NOW) == []


@pytest.mark.parametrize("remaining, lines, props, warns", [
    (30000, True, True, False), (3000, True, True, False), (2999, True, False, True), (1000, True, False, True),
    (999, False, False, True), (0, False, False, True), (None, True, True, True)])
def test_budget_thresholds(remaining, lines, props, warns):
    d = close.budget_decision(remaining)
    assert (d.lines, d.props, d.warning is not None) == (lines, props, warns)


def test_parse_credits_and_estimated_cost():
    assert close.parse_credits("19000") == 19000 and close.parse_credits("8.0") == 8
    assert close.parse_credits(None) is None and close.parse_credits("n/a") is None
    assert close.estimated_cost(3, "us,eu") == 6 and close.estimated_cost(4, "us") == 4


@pytest.mark.parametrize("total, n", [(6, 1), (6, 4), (7, 3), (0, 2), (5, 5)])
def test_split_credits_sums_exactly(total, n):
    parts = close.split_credits(total, n)
    assert len(parts) == n and sum(parts) == total and max(parts) - min(parts) <= 1
    assert close.split_credits(5, 0) == []


@pytest.mark.parametrize("env, expected", [({}, "us,eu"), ({"CLOSE_PROP_REGIONS": ""}, "us,eu"),
                                           ({"CLOSE_PROP_REGIONS": "us"}, "us"), ({"CLOSE_PROP_REGIONS": " US , EU "}, "us,eu"),
                                           ({"CLOSE_PROP_REGIONS": "uk"}, "us,eu")])
def test_prop_regions_default_and_override(env, expected):
    assert close.prop_regions(env) == expected


def test_api_markets_cover_only_the_markets_with_picks():
    nfl = sports.get("nfl").prop_market_map
    assert close.api_markets_for("nfl", {"rec_yds", "pass_tds"}, nfl) == ["player_pass_tds", "player_reception_yds"]
    assert close.api_markets_for("nfl", {"nonsense"}, nfl) == [] and close.api_markets_for("nfl", set(), nfl) == []
    mlb = sports.get("mlb").prop_market_map
    assert close.api_markets_for("mlb", {"total_bases", "pitcher_ks"}, mlb) == ["batter_total_bases", "pitcher_strikeouts"]
    assert close.api_markets_for("cfb", {"rec_yds"}, {}) == []


def test_capture_row_shape_matches_the_table_columns():
    from sportsmodel import db
    row = close.capture_row(sport="nfl", game=_g(1, 12), event_id="e1", kind="lines", now=NOW, credits=6,
                            status="ok", regions="us,eu", markets=["h2h", "spreads"])
    assert set(row) == set(db.LEDGER_COLUMNS["close_capture"])
    assert row["minutes_to_kick"] == pytest.approx(12) and row["markets"] == "h2h,spreads"
    assert row["captured_at"] == NOW.isoformat()
````

`tests/ledger/test_capture_close_script.py` (the whole run on injected IO: idle runs, windowing, retry, props, regions, budget guard, dry run; plus the default SQL):

````python
"""scripts/capture_close.py end to end on injected IO: idle runs make no call, windowing, retry, budget guard,
Pinnacle/regions handling, what is stored. No network, no database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("capture_close", ROOT / "scripts" / "capture_close.py")
cc = importlib.util.module_from_spec(_spec)
sys.modules["capture_close"] = cc            # @dataclass needs its module registered
_spec.loader.exec_module(cc)

NOW = datetime(2026, 10, 18, 16, 40, tzinfo=timezone.utc)


def iso(minutes):
    return (NOW + timedelta(minutes=minutes)).isoformat().replace("+00:00", "Z")


def game(pk, minutes, home="Home", away="Away"):
    return {"game_pk": pk, "home_name": home, "away_name": away, "commence_time": iso(minutes)}


def event(eid, minutes, home="Home", away="Away"):
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": iso(minutes)}


def odds_event(eid, minutes, home="Home", away="Away"):
    price = lambda p: {"price": p}
    return {**event(eid, minutes, home, away), "bookmakers": [
        {"key": "pinnacle", "markets": [
            {"key": "h2h", "outcomes": [{"name": home, **price(-150)}, {"name": away, **price(130)}]},
            {"key": "spreads", "outcomes": [{"name": home, "point": -3.5, **price(-110)}, {"name": away, "point": 3.5, **price(-110)}]}]}]}


class Fake:
    """Injected IO that records every call; any call not expected in a scenario can be made to explode."""

    def __init__(self, **over):
        self.calls, self.stored_snaps, self.stored_caps = [], [], []
        self.remaining_value = over.pop("remaining", "20000")
        self.cost = over.pop("cost", "6")
        self.candidates = over.pop("candidates", [])
        self.schedule = over.pop("schedule", {})
        self.captured = over.pop("captured", set())
        self.prop_markets = over.pop("prop_markets", {})
        self.events = over.pop("events", [])
        self.line_events = over.pop("line_events", [])
        self.props_payload = over.pop("props_payload", {"bookmakers": []})
        self.fail = over.pop("fail", set())
        self.missing = over.pop("missing", [])
        assert not over, over

    def _log(self, name, *args):
        self.calls.append((name, *args))

    def deps(self):
        def fetch_game_odds(cfg, regions):
            self._log("lines", cfg.key, regions)
            if "lines" in self.fail:
                raise RuntimeError("odds api 500")
            return self.line_events

        def fetch_event_props(eid, markets, cfg, regions):
            self._log("props", eid, tuple(markets), regions)
            if "props" in self.fail:
                raise RuntimeError("odds api 500")
            return self.props_payload

        return cc.Deps(
            missing_tables=lambda: self.missing,
            load_candidates=lambda now, end: self.candidates,
            load_captured=lambda pks: self.captured,
            load_prop_markets=lambda pks: self.prop_markets,
            fetch_schedule=lambda sport: self._log("schedule", sport) or self.schedule.get(sport, []),
            match=cc.ingest_odds._matcher_for,
            fetch_events=lambda cfg: self._log("events", cfg.key) or self.events,
            fetch_game_odds=fetch_game_odds, fetch_event_props=fetch_event_props,
            remaining=lambda: self.remaining_value, last_cost=lambda: self.cost,
            store_snapshots=lambda rows: self.stored_snaps.extend(rows) or len(rows),
            store_captures=lambda rows: self.stored_caps.extend(rows) or len(rows))

    def network_calls(self):
        return [c for c in self.calls if c[0] in ("events", "lines", "props", "schedule")]


def cand(sport, pk, minutes):
    return {"sport": sport, "game_pk": pk, "commence_time": iso(minutes)}


def run(fake, **kw):
    return cc.run_capture(NOW, fake.deps(), **kw)


# ---- idle runs --------------------------------------------------------------------------------------

def test_idle_run_makes_no_network_call_at_all():
    fake = Fake()                                                  # the DB knows of no kickoff
    assert run(fake) == {"status": "idle"} and fake.calls == []


def test_a_kickoff_far_away_is_idle_too():
    fake = Fake(candidates=[cand("nfl", 1, 180), cand("cfb", 2, -30)])
    assert run(fake) == {"status": "idle"} and fake.calls == []


def test_nearby_kickoff_but_nothing_in_the_window_makes_no_odds_api_call():
    fake = Fake(candidates=[cand("nfl", 1, 45)], schedule={"nfl": [game(1, 45)]})        # 45 min out: too early
    out = run(fake)
    assert out["status"] == "ok" and out["credits"] == 0
    assert fake.calls == [("schedule", "nfl")]                      # schedule only: no events, lines or props call


def test_missing_tables_warn_and_stop_before_anything_else(capsys):
    fake = Fake(missing=["close_capture"], candidates=[cand("nfl", 1, 12)])
    assert run(fake) == {"status": "no_table"} and fake.calls == []
    assert "::warning::capture-close: table close_capture missing" in capsys.readouterr().out


# ---- a normal capture ----------------------------------------------------------------------------

def _nfl_fake(**kw):
    base = dict(candidates=[cand("nfl", 1, 12), cand("nfl", 2, 40)],
                schedule={"nfl": [game(1, 12), game(2, 40, "H2", "A2")]},
                events=[event("ev1", 12), event("ev2", 40, "H2", "A2")],
                line_events=[odds_event("ev1", 12), odds_event("ev2", 40, "H2", "A2")])
    return Fake(**{**base, **kw})


def test_lines_capture_stores_only_the_target_games_and_one_row_per_event():
    fake = _nfl_fake()
    out = run(fake)
    assert out["credits"] == 6 and out["snapshots"] == len(fake.stored_snaps) > 0
    assert {r["game_pk"] for r in fake.stored_snaps} == {1}                           # game 2 is 40 min out
    assert {r["captured_at"] for r in fake.stored_snaps} == {NOW.isoformat()}
    ((sport, pk, kind, status),) = out["rows"]
    assert (sport, pk, kind, status) == ("nfl", 1, "lines", "ok")
    (cap,) = fake.stored_caps
    assert cap["event_id"] == "ev1" and cap["credits_used"] == 6 and cap["regions"] == "us,eu"
    assert cap["minutes_to_kick"] == pytest.approx(12) and cap["captured_at"] == NOW.isoformat()
    assert ("lines", "nfl", "us,eu") in fake.calls and not [c for c in fake.calls if c[0] == "props"]


def test_two_target_games_split_one_call_cost_exactly():
    fake = _nfl_fake(candidates=[cand("nfl", 1, 12), cand("nfl", 2, 14)],
                     schedule={"nfl": [game(1, 12), game(2, 14, "H2", "A2")]},
                     line_events=[odds_event("ev1", 12), odds_event("ev2", 14, "H2", "A2")], cost="6")
    run(fake)
    assert sorted(c["credits_used"] for c in fake.stored_caps) == [3, 3]
    assert len([c for c in fake.calls if c[0] == "lines"]) == 1                      # ONE call per sport


def test_an_event_without_prices_is_recorded_empty():
    fake = _nfl_fake(line_events=[])
    out = run(fake)
    assert out["rows"] == [("nfl", 1, "lines", "empty")]


def test_already_captured_events_cost_nothing():
    fake = _nfl_fake(captured={("nfl", 1, "lines")})
    out = run(fake)
    assert out["credits"] == 0 and fake.stored_caps == [] and "events" not in [c[0] for c in fake.calls]


def test_a_failed_call_is_recorded_failed_and_retried_next_run():
    fake = _nfl_fake(fail={"lines"})
    out = run(fake)
    assert out["rows"] == [("nfl", 1, "lines", "failed")] and out["credits"] == 0
    assert fake.stored_caps[0]["status"] == "failed" and fake.stored_snaps == []
    # 'failed' is not in the loader's final set, so the next run selects the event again and succeeds
    fake2 = _nfl_fake(captured=set())
    assert run(fake2)["rows"] == [("nfl", 1, "lines", "ok")]


def test_the_credit_cost_falls_back_to_markets_times_regions_without_the_header():
    fake = _nfl_fake(cost=None)
    assert run(fake)["credits"] == 6


# ---- props -------------------------------------------------------------------------------------------

PROP_PAYLOAD = {"id": "ev1", "commence_time": iso(12), "bookmakers": [
    {"key": "draftkings", "markets": [{"key": "player_reception_yds", "outcomes": [
        {"name": "Over", "description": "A. Brown", "point": 64.5, "price": -110},
        {"name": "Under", "description": "A. Brown", "point": 64.5, "price": -110}]}]}]}


def test_props_are_captured_only_for_events_with_ledger_prop_picks_and_only_their_markets():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, props_payload=PROP_PAYLOAD, cost="2")
    out = run(fake)
    assert ("props", "ev1", ("player_reception_yds",), "us,eu") in fake.calls
    kinds = {(c["kind"], c["status"], c["credits_used"]) for c in fake.stored_caps}
    assert kinds == {("lines", "ok", 2), ("props", "ok", 2)}
    props_rows = [r for r in fake.stored_snaps if r["player_name"]]
    assert {(r["market"], r["side"], r["line"]) for r in props_rows} == {("reception_yds", "over", 64.5), ("reception_yds", "under", 64.5)}
    assert {r["captured_at"] for r in fake.stored_snaps} == {NOW.isoformat()}
    assert out["credits"] == 4


def test_no_prop_picks_means_no_props_call():
    fake = _nfl_fake(prop_markets={})
    run(fake)
    assert not [c for c in fake.calls if c[0] == "props"]


def test_prop_regions_override_is_used_and_pinnacle_free_runs_pay_half():
    fake = _nfl_fake(prop_markets={1: {"rec_yds", "rush_yds", "receptions", "pass_tds"}}, props_payload=PROP_PAYLOAD, cost=None)
    run(fake, prop_regions="us")
    (call,) = [c for c in fake.calls if c[0] == "props"]
    assert call[3] == "us" and len(call[2]) == 4
    caps = {c["kind"]: c["credits_used"] for c in fake.stored_caps}
    assert caps["props"] == 4                                          # 4 markets x 1 region (8 with us,eu)
    fake2 = _nfl_fake(prop_markets={1: {"rec_yds", "rush_yds", "receptions", "pass_tds"}}, props_payload=PROP_PAYLOAD, cost=None)
    run(fake2, prop_regions="us,eu")
    assert {c["kind"]: c["credits_used"] for c in fake2.stored_caps}["props"] == 8


def test_props_already_captured_are_not_bought_again_and_a_props_retry_needs_no_lines_call():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, props_payload=PROP_PAYLOAD, captured={("nfl", 1, "lines")})
    run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == ["props"]
    fake2 = _nfl_fake(prop_markets={1: {"rec_yds"}}, captured={("nfl", 1, "lines"), ("nfl", 1, "props")})
    run(fake2)
    assert not [c for c in fake2.calls if c[0] in ("lines", "props")]


def test_a_failed_props_call_is_recorded_failed_without_losing_the_lines_capture():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, fail={"props"})
    out = run(fake)
    assert sorted((k, s) for _, _, k, s in out["rows"]) == [("lines", "ok"), ("props", "failed")]


def test_cfb_has_no_props_and_never_asks_for_the_ledger():
    fake = Fake(candidates=[cand("cfb", 9, 12)], schedule={"cfb": [game(9, 12)]}, events=[event("c1", 12)],
                line_events=[odds_event("c1", 12)], prop_markets={9: {"rec_yds"}})
    run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == ["lines"]


# ---- budget guard ------------------------------------------------------------------------------------

def test_below_three_thousand_credits_skips_props_but_keeps_lines(capsys):
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, remaining="2500")
    run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == ["lines"]
    assert "skipping prop close captures" in capsys.readouterr().out


def test_below_one_thousand_credits_skips_everything_and_stores_nothing(capsys):
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}}, remaining="999")
    out = run(fake)
    assert [c[0] for c in fake.calls if c[0] in ("lines", "props")] == [] and fake.stored_snaps == [] and fake.stored_caps == []
    assert out["credits"] == 0 and "skipping ALL close captures" in capsys.readouterr().out


def test_an_unreadable_credit_header_fails_open_with_a_warning(capsys):
    fake = _nfl_fake(remaining=None)
    run(fake)
    assert "lines" in [c[0] for c in fake.calls] and "unavailable" in capsys.readouterr().out


# ---- dry run, sports, main -------------------------------------------------------------------------

def test_dry_run_plans_but_spends_and_writes_nothing():
    fake = _nfl_fake(prop_markets={1: {"rec_yds"}})
    out = run(fake, dry_run=True)
    assert out["planned"] == [{"sport": "nfl", "lines": [1], "props": [1]}]
    assert not [c for c in fake.calls if c[0] in ("events", "lines", "props")] and fake.stored_caps == []


def test_one_sport_failing_does_not_stop_the_next():
    fake = Fake(candidates=[cand("nfl", 1, 12), cand("cfb", 2, 12)],
                schedule={"cfb": [game(2, 12)]}, events=[event("c1", 12)], line_events=[odds_event("c1", 12)])
    deps = fake.deps()

    def schedule(sport):
        if sport == "nfl":
            raise RuntimeError("espn down")
        return fake.schedule[sport]

    deps.fetch_schedule = schedule
    out = cc.run_capture(NOW, deps)
    assert out["rows"] == [("cfb", 2, "lines", "ok")]


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert cc.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_default_db_queries(monkeypatch):
    from sportsmodel import db
    seen = []

    class Cur:
        def execute(self, sql, params):
            seen.append((sql, params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn(Cur):
        def cursor(self):
            return Cur()

    monkeypatch.setattr(db, "get_postgres", lambda: Conn())
    deps = cc._db_deps()
    deps.load_candidates(NOW, NOW + timedelta(hours=1))
    deps.load_captured([1, 2])
    deps.load_prop_markets([1])
    sqls = [s for s, _ in seen]
    assert "FROM predictions_current" in sqls[0] and "commence_time > %s AND commence_time <= %s" in sqls[0]
    assert "status IN ('ok', 'empty')" in sqls[1] and seen[1][1] == ([1, 2],)
    assert "FROM pick_ledger" in sqls[2] and "source IN ('ev_prop', 'parlay_leg')" in sqls[2]
````

`tests/ledger/test_workflows_capture.py` (cron coverage for every real kickoff pattern, jitter-safe minutes, season months, bounded run count, secrets, serialisation). It also hosts the tiny cron evaluator Task 8 reuses:

````python
"""Structural checks of .github/workflows/capture-close.yml: cadence, jitter-safe minutes, game-day-only hours,
serialised runs, secrets, never commits. Also hosts the tiny cron evaluator the other workflow tests reuse."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

CAPTURE = "capture-close.yml"


def _expand(field: str, lo: int, hi: int) -> set[int]:
    out: set[int] = set()
    for part in field.split(","):
        step = 1
        if "/" in part:
            part, s = part.split("/")
            step = int(s)
        if part == "*":
            a, b = lo, hi
        elif "-" in part:
            a, b = map(int, part.split("-"))
        else:
            a = b = int(part)
            if step != 1:
                b = hi
        out |= set(range(a, b + 1, step))
    return out


def cron_sets(expr: str):
    m, h, dom, mon, dow = expr.split()
    assert dom == "*", expr
    return _expand(m, 0, 59), _expand(h, 0, 23), _expand(mon, 1, 12), _expand(dow, 0, 6)


def fires(expr: str, dt: datetime) -> bool:
    m, h, mon, dow = cron_sets(expr)
    return dt.minute in m and dt.hour in h and dt.month in mon and (dt.isoweekday() % 7) in dow


def any_fire_in(crons: list[str], start: datetime, end: datetime) -> bool:
    t = start.replace(second=0, microsecond=0)
    while t <= end:
        if any(fires(c, t) for c in crons):
            return True
        t += timedelta(minutes=1)
    return False


def utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def crons():
    return [c["cron"] for c in load(CAPTURE)["on"]["schedule"]]


def test_four_schedules_with_ten_minute_cadence_off_the_peak_minutes(crons):
    assert len(crons) == 4
    football, baseball = crons[:2], crons[2:]
    for c in football:
        assert cron_sets(c)[0] == {3, 13, 23, 33, 43, 53}
    for c in baseball:
        assert cron_sets(c)[0] == {8, 18, 28, 38, 48, 58}
    assert not (cron_sets(football[0])[0] & cron_sets(baseball[0])[0])          # the schedules never fire together
    assert all(0 not in cron_sets(c)[0] and 30 not in cron_sets(c)[0] for c in crons)


def test_seasons_football_aug_to_feb_baseball_mar_to_nov(crons):
    assert cron_sets(crons[0])[2] == {1, 2, 8, 9, 10, 11, 12} == cron_sets(crons[1])[2]
    assert cron_sets(crons[2])[2] == set(range(3, 12)) == cron_sets(crons[3])[2]


@pytest.mark.parametrize("label, kick, months_ok", [
    ("Thu-night NFL 8:15pm ET", utc(2026, 10, 16, 0, 15), True),           # Fri 00:15 UTC
    ("Fri-night CFB 7pm ET", utc(2026, 10, 16, 23, 0), True),
    ("Sat noon CFB", utc(2026, 10, 17, 16, 0), True),
    ("Sat late CFB 10:30pm ET", utc(2026, 10, 18, 2, 30), True),            # Sun 02:30 UTC
    ("Sat Hawaii CFB 11:59pm ET", utc(2026, 10, 18, 4, 0), True),
    ("Sun London NFL 9:30am ET", utc(2026, 10, 18, 13, 30), True),
    ("Sun 1pm ET NFL", utc(2026, 10, 18, 17, 0), True),
    ("Sun 4:25pm ET NFL", utc(2026, 10, 18, 20, 25), True),
    ("SNF", utc(2026, 10, 19, 0, 20), True),                                # Mon 00:20 UTC
    ("MNF", utc(2026, 10, 20, 0, 15), True),                                # Tue 00:15 UTC
    ("Thanksgiving 4:30pm ET", utc(2026, 11, 26, 21, 30), True),
    ("Saturday NFL in December", utc(2026, 12, 19, 21, 30), True),
    ("Bowl game in January", utc(2027, 1, 2, 1, 0), True),
    ("Super Bowl Sunday", utc(2027, 2, 14, 23, 30), True),
    ("MLB day game", utc(2026, 10, 21, 17, 8), True),
    ("MLB west-coast night game", utc(2026, 10, 22, 2, 10), True),
    ("MLB 1:05pm ET Sunday in April", utc(2027, 4, 11, 17, 5), True),
])
def test_every_kickoff_has_a_run_inside_its_capture_window(crons, label, kick, months_ok):
    lo, hi = kick - timedelta(minutes=25), kick - timedelta(minutes=5)
    assert any_fire_in(crons, lo, hi), label


def test_midweek_and_the_offseason_never_fire(crons):
    football, baseball = crons[:2], crons[2:]
    assert not any_fire_in(football, utc(2026, 10, 21, 0, 0), utc(2026, 10, 22, 12, 59))   # Wed 00:00 - Thu 12:59 UTC
    assert not any_fire_in(football, utc(2026, 7, 5, 0, 0), utc(2026, 7, 12, 0, 0))        # football in July
    assert not any_fire_in(baseball, utc(2027, 1, 5, 0, 0), utc(2027, 1, 12, 0, 0))        # baseball in January
    assert not any_fire_in(baseball, utc(2026, 10, 21, 3, 0), utc(2026, 10, 21, 14, 59))   # baseball overnight / morning


def test_runs_per_week_are_bounded(crons):
    t, n = utc(2026, 10, 12, 0, 0), 0
    while t < utc(2026, 10, 19, 0, 0):
        n += any(fires(c, t) for c in crons)
        t += timedelta(minutes=1)
    assert 700 <= n <= 1100          # idle-by-design runs; each exits after one DB read


def test_dispatch_has_a_dry_run_input():
    inp = load(CAPTURE)["on"]["workflow_dispatch"]["inputs"]["dry_run"]
    assert inp["type"] == "boolean" and str(inp["default"]).lower() == "false"


def test_job_runs_the_script_with_secrets_regions_and_serialises_runs():
    wf = load(CAPTURE)
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["group"] == "capture-close" and str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
    (job,) = wf["jobs"].values()
    assert str(job["timeout-minutes"]) == "8"
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/capture_close.py"))]
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert step["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}"
    assert step["env"]["CLOSE_PROP_REGIONS"] == "${{ vars.CLOSE_PROP_REGIONS }}"
    assert "'--dry-run'" in step["run"]
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("capture_close.py"))


def test_capture_workflow_never_commits_and_never_sets_the_test_guard():
    text = (WF / CAPTURE).read_text()
    assert "git push" not in text and "git commit" not in text and "SM_LEDGER_OFF" not in text
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_close.py tests/ledger/test_capture_close_script.py tests/ledger/test_workflows_capture.py`
Expected: collection ERRORs: `ImportError: cannot import name 'close' from 'sportsmodel.ledger'` and `FileNotFoundError: ... scripts/capture_close.py`.

- [ ] **Step 3: Write the implementation**

`src/sportsmodel/ledger/close.py`:

````python
"""Pre-kick close capture: window selection, credit budget guard and prop-market planning. PURE -- no IO.

Window. A run captures the events that kick off in WINDOW_MIN..WINDOW_MAX minutes and have no final capture yet.
The spec's outer bound was 35 minutes; it is 25 here because the weekly gate asks for a close within 30 minutes of
kickoff, and the first run to see an event at 31-35 minutes would fail it. With a 10-minute cadence a 20-minute
window still holds two runs, so one skipped or late GitHub cron firing loses nothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from ..ingest.odds import parse_commence
from ..serving.props_ev import MLB_SIM_TO_ODDS_MARKET, SIM_TO_ODDS_MARKET

WINDOW_MIN = 5
WINDOW_MAX = 25
PREFILTER_MIN = 60               # DB look-ahead that decides whether a run is idle (no network at all)
BUDGET_PROPS_FLOOR = 3000        # remaining credits below this: skip prop captures
BUDGET_ALL_FLOOR = 1000          # remaining credits below this: skip everything
LINES_REGIONS = "us,eu"
DEFAULT_PROP_REGIONS = "us,eu"
SIM_TO_ODDS_BY_SPORT = {"nfl": SIM_TO_ODDS_MARKET, "mlb": MLB_SIM_TO_ODDS_MARKET}


def minutes_to_kick(commence, now: datetime) -> float | None:
    """Minutes from `now` to `commence` (ISO string or datetime); None when unparseable."""
    c = parse_commence(commence) if isinstance(commence, str) else commence
    if c is None:
        return None
    return (c - now).total_seconds() / 60.0


def prefilter_sports(candidates: list[dict], now: datetime, horizon_min: int = PREFILTER_MIN) -> list[str]:
    """Sports (nfl, cfb, mlb order) with at least one DB-known kickoff in (now, now + horizon]. None -> the run
    is idle and must exit before any network call."""
    live = set()
    for c in candidates:
        m = minutes_to_kick(c.get("commence_time"), now)
        if m is not None and 0 < m <= horizon_min:
            live.add(c.get("sport"))
    return [s for s in ("nfl", "cfb", "mlb") if s in live]


def select_targets(games: list[dict], captured: set[tuple], sport: str, kind: str, now: datetime,
                   lo: float = WINDOW_MIN, hi: float = WINDOW_MAX) -> list[dict]:
    """Schedule rows (game_pk, commence_time, ...) kicking off in [lo, hi] minutes with no final capture of
    `kind` yet. `captured` is a set of (sport, game_pk, kind) for the rows already stored as 'ok' or 'empty'
    (a 'failed' attempt is not final: the next run retries it)."""
    out = []
    for g in games:
        m = minutes_to_kick(g.get("commence_time"), now)
        if m is None or not (lo <= m <= hi):
            continue
        if (sport, g["game_pk"], kind) in captured:
            continue
        out.append({**g, "minutes_to_kick": m})
    return out


def prop_regions(env: dict) -> str:
    """Regions for prop captures: CLOSE_PROP_REGIONS (repo variable), default 'us,eu'. Set it to 'us' once the
    Pinnacle probe says Pinnacle publishes no props -- that halves the prop credits."""
    raw = (env.get("CLOSE_PROP_REGIONS") or "").strip().lower().replace(" ", "")
    parts = [p for p in raw.split(",") if p in ("us", "eu")]
    return ",".join(parts) if parts else DEFAULT_PROP_REGIONS


def n_regions(regions: str) -> int:
    return len([r for r in regions.split(",") if r])


def estimated_cost(n_markets: int, regions: str) -> int:
    """The Odds API bills markets x regions per call."""
    return n_markets * n_regions(regions)


def parse_credits(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def split_credits(total: int, n: int) -> list[int]:
    """Spread a call's cost over `n` event rows so the rows sum to the cost exactly."""
    if n <= 0:
        return []
    base, extra = divmod(max(int(total), 0), n)
    return [base + (1 if i < extra else 0) for i in range(n)]


@dataclass(frozen=True)
class BudgetDecision:
    lines: bool
    props: bool
    warning: str | None = None


def budget_decision(remaining: int | None) -> BudgetDecision:
    """< 1,000 credits remaining: skip everything; < 3,000: skip props, keep game lines; otherwise capture all.
    An unreadable header fails open (the capture is the point) but says so."""
    if remaining is None:
        return BudgetDecision(True, True, "x-requests-remaining unavailable; capturing without a budget check")
    if remaining < BUDGET_ALL_FLOOR:
        return BudgetDecision(False, False, f"{remaining} Odds API credits left (< {BUDGET_ALL_FLOOR}): "
                                            "skipping ALL close captures")
    if remaining < BUDGET_PROPS_FLOOR:
        return BudgetDecision(True, False, f"{remaining} Odds API credits left (< {BUDGET_PROPS_FLOOR}): "
                                           "skipping prop close captures")
    return BudgetDecision(True, True, None)


def api_markets_for(sport: str, ledger_markets: set[str], prop_market_map: dict[str, str]) -> list[str]:
    """Odds API market keys to request for one event: only the markets the ledger actually holds picks in
    (ledger markets are the writers' sim codes -> odds-side code -> API key), sorted, unknown ones dropped."""
    to_odds = SIM_TO_ODDS_BY_SPORT.get(sport, {})
    keys = {prop_market_map[to_odds[m]] for m in ledger_markets if m in to_odds and to_odds[m] in prop_market_map}
    return sorted(keys)


def capture_row(*, sport: str, game: dict, event_id: str | None, kind: str, now: datetime, credits: int,
                status: str, regions: str, markets: list[str]) -> dict:
    """One close_capture row (columns of db.LEDGER_COLUMNS['close_capture'])."""
    m = minutes_to_kick(game["commence_time"], now)
    return {"sport": sport, "game_pk": game["game_pk"], "event_id": event_id, "kind": kind,
            "captured_at": now.isoformat(), "commence_time": game["commence_time"],
            "minutes_to_kick": m, "credits_used": int(credits), "status": status, "regions": regions,
            "markets": ",".join(markets)}
````

`scripts/capture_close.py`:

````python
"""Capture a true closing price shortly before kickoff (pick-ledger step 1).

Run every ~10 minutes during game windows (.github/workflows/capture-close.yml). Each run:

  1. Idle check, DB only: no kickoff known within the next hour -> exit with ZERO network calls.
  2. For each sport with a nearby kickoff: fetch the live schedule (ESPN / StatsAPI, free), keep the games kicking
     off in WINDOW_MIN..WINDOW_MAX minutes that have no final (ok / empty) close_capture yet. None -> no Odds API call.
  3. Free Odds API events call: reads x-requests-remaining for the budget guard (< 3,000 skips props, < 1,000
     skips everything) and gives the event ids.
  4. Game lines: ONE call per sport (h2h / spreads / totals, regions us,eu) stored to odds_snapshot for the
     target games only. Props: one call per target event that holds >= 1 prop pick in the ledger, only for the
     markets those picks are in (regions CLOSE_PROP_REGIONS, default us,eu).
  5. One close_capture row per event and kind (ok / empty / failed). A failed attempt is retried by the next run
     while the event is still in the window; an 'ok' or 'empty' row is final (never replaced, never re-bought).

Credits: lines = 3 markets x 2 regions = 6 per sport-run; props = (markets with picks) x regions per event.

    uv run python scripts/capture_close.py [--dry-run] [--now 2026-10-18T16:40:00Z]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import ingest_odds  # noqa: E402  (scripts/ingest_odds.py: schedule fetchers, event matchers, game lookup)
from sportsmodel import sports  # noqa: E402
from sportsmodel.ingest import odds  # noqa: E402
from sportsmodel.ledger import close  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

SPORT_ORDER = ("nfl", "cfb", "mlb")
LINES_MARKETS = "h2h,spreads,totals"


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    load_candidates: Callable[[datetime, datetime], list[dict]]        # (now, horizon_end) -> {sport, game_pk, commence_time}
    load_captured: Callable[[list[int]], set]                           # -> {(sport, game_pk, kind)} of 'ok' rows
    load_prop_markets: Callable[[list[int]], dict]                      # -> {game_pk: {ledger prop markets}}
    fetch_schedule: Callable[[str], list[dict]]
    match: Callable[[str], Callable]                                    # sport -> matcher(event, games) -> game_pk | None
    fetch_events: Callable[[Any], list[dict]]
    fetch_game_odds: Callable[[Any, str], list[dict]]
    fetch_event_props: Callable[[str, list, Any, str], dict]
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    store_snapshots: Callable[[list[dict]], int]
    store_captures: Callable[[list[dict]], int]


def _warn(msg: str) -> None:
    print(f"::warning::capture-close: {msg}", flush=True)


def _call_cost(deps: Deps, n_markets: int, regions: str) -> int:
    got = close.parse_credits(deps.last_cost())
    return got if got is not None else close.estimated_cost(n_markets, regions)


def _capture_sport(sport: str, now: datetime, deps: Deps, prop_regions: str, dry_run: bool, out: dict) -> None:
    cfg = sports.get(sport)
    schedule = deps.fetch_schedule(sport)
    in_window = [g for g in schedule if (m := close.minutes_to_kick(g.get("commence_time"), now)) is not None
                 and close.WINDOW_MIN <= m <= close.WINDOW_MAX]
    if not in_window:
        return
    pks = [g["game_pk"] for g in in_window]
    captured = deps.load_captured(pks)
    line_targets = close.select_targets(in_window, captured, sport, "lines", now)
    prop_markets = deps.load_prop_markets(pks) if cfg.prop_market_map else {}
    prop_targets = [g for g in close.select_targets(in_window, captured, sport, "props", now)
                    if prop_markets.get(g["game_pk"])]
    if not line_targets and not prop_targets:
        return

    if dry_run:
        out["planned"].append({"sport": sport, "lines": [g["game_pk"] for g in line_targets],
                               "props": [g["game_pk"] for g in prop_targets]})
        return

    events = deps.fetch_events(cfg)                                    # free; also refreshes the credit header
    decision = close.budget_decision(close.parse_credits(deps.remaining()))
    if decision.warning:
        _warn(decision.warning)
    matcher = deps.match(sport)
    event_by_pk = {}
    for ev in events:
        gp = matcher(ev, schedule)
        if gp is not None:
            event_by_pk[gp] = ev

    snapshot_rows: list[dict] = []
    capture_rows: list[dict] = []
    captured_at = now.isoformat()

    if line_targets and decision.lines:
        regions = close.LINES_REGIONS
        target_pks = {g["game_pk"] for g in line_targets}
        try:
            line_events = deps.fetch_game_odds(cfg, regions)
            lookup = ingest_odds.build_game_lookup(line_events, schedule, matcher)
            rows = [r for r in odds.parse_game_odds(line_events, lookup, captured_at) if r["game_pk"] in target_pks]
            cost = _call_cost(deps, len(cfg.game_markets), regions)
            per_game = close.split_credits(cost, len(line_targets))
            snapshot_rows += rows
            have = {r["game_pk"] for r in rows}
            for g, credits in zip(line_targets, per_game):
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=(event_by_pk.get(g["game_pk"]) or {}).get("id"), kind="lines",
                    now=now, credits=credits, status="ok" if g["game_pk"] in have else "empty", regions=regions,
                    markets=LINES_MARKETS.split(",")))
            out["credits"] += cost
        except Exception as exc:  # noqa: BLE001 -- one sport's failed call is retried by the next run
            _warn(f"{sport} lines capture failed: {type(exc).__name__}: {exc}")
            for g in line_targets:
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=(event_by_pk.get(g["game_pk"]) or {}).get("id"), kind="lines",
                    now=now, credits=0, status="failed", regions=regions, markets=LINES_MARKETS.split(",")))
        decision = close.budget_decision(close.parse_credits(deps.remaining()))   # re-check after the spend

    if prop_targets and decision.props:
        for g in prop_targets:
            ev = event_by_pk.get(g["game_pk"])
            markets = close.api_markets_for(sport, prop_markets[g["game_pk"]], cfg.prop_market_map)
            if ev is None or not markets:
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=(ev or {}).get("id"), kind="props", now=now, credits=0,
                    status="failed", regions=prop_regions, markets=markets))
                _warn(f"{sport} {g['game_pk']}: no Odds API event / prop markets for the props capture")
                continue
            try:
                payload = deps.fetch_event_props(ev["id"], markets, cfg, prop_regions)
                rows = odds.parse_prop_odds(payload, g["game_pk"], captured_at, cfg)
                cost = _call_cost(deps, len(markets), prop_regions)
                snapshot_rows += rows
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=ev["id"], kind="props", now=now, credits=cost,
                    status="ok" if rows else "empty", regions=prop_regions, markets=markets))
                out["credits"] += cost
            except Exception as exc:  # noqa: BLE001
                _warn(f"{sport} {g['game_pk']} props capture failed: {type(exc).__name__}: {exc}")
                capture_rows.append(close.capture_row(
                    sport=sport, game=g, event_id=ev.get("id"), kind="props", now=now, credits=0,
                    status="failed", regions=prop_regions, markets=markets))
            if not close.budget_decision(close.parse_credits(deps.remaining())).props:
                _warn("credit floor reached mid-run; remaining prop captures skipped")
                break

    # Snapshots first, then the capture rows: a capture row implies its prices are stored.
    out["snapshots"] += deps.store_snapshots(snapshot_rows) if snapshot_rows else 0
    out["captures"] += deps.store_captures(capture_rows) if capture_rows else 0
    out["rows"] += [(r["sport"], r["game_pk"], r["kind"], r["status"]) for r in capture_rows]


def run_capture(now: datetime, deps: Deps, *, prop_regions: str = close.DEFAULT_PROP_REGIONS,
                dry_run: bool = False) -> dict:
    """One run. Returns a summary dict; status 'idle' means nothing was fetched from any network."""
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_pick_ledger.sql")
        return {"status": "no_table"}
    candidates = deps.load_candidates(now, now + timedelta(minutes=close.PREFILTER_MIN))
    live = close.prefilter_sports(candidates, now)
    if not live:
        return {"status": "idle"}
    out: dict = {"status": "ok", "credits": 0, "snapshots": 0, "captures": 0, "rows": [], "planned": []}
    for sport in live:
        try:
            _capture_sport(sport, now, deps, prop_regions, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- schedule / matching trouble in one sport must not stop the others
            _warn(f"{sport} failed: {type(exc).__name__}: {exc}")
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db

    def q(sql, params):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def candidates(now, end):
        rows = q("SELECT sport, game_pk, commence_time FROM predictions_current "
                 "WHERE commence_time > %s AND commence_time <= %s", (now, end))
        return [{"sport": s, "game_pk": pk, "commence_time": c} for s, pk, c in rows]

    def captured(pks):
        rows = q("SELECT sport, game_pk, kind FROM close_capture WHERE status IN ('ok', 'empty') "
                 "AND game_pk = ANY(%s)", (pks,))
        return {(s, pk, k) for s, pk, k in rows}

    def prop_markets(pks):
        rows = q("SELECT DISTINCT game_pk, market FROM pick_ledger WHERE game_pk = ANY(%s) "
                 "AND player_id IS NOT NULL AND source IN ('ev_prop', 'parlay_leg')", (pks,))
        out: dict = {}
        for pk, market in rows:
            out.setdefault(pk, set()).add(market)
        return out

    return Deps(
        missing_tables=lambda: db.missing_ledger_tables(["pick_ledger", "close_capture"]),
        load_candidates=candidates, load_captured=captured, load_prop_markets=prop_markets,
        fetch_schedule=lambda sport: ingest_odds._ESPN_FETCHERS[sport](),
        match=ingest_odds._matcher_for,
        fetch_events=odds.fetch_events, fetch_game_odds=odds.fetch_game_odds,
        fetch_event_props=lambda eid, markets, cfg, regions: odds.fetch_event_props(eid, markets, cfg, regions=regions),
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        store_snapshots=db.upsert_odds_snapshot, store_captures=db.upsert_close_capture)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="select targets and print the plan; no Odds API call, no writes")
    ap.add_argument("--now", default=None, help="as-of time (ISO, default: now UTC)")
    args = ap.parse_args(argv)
    if writes_disabled() and not args.dry_run:
        print("capture-close: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::capture-close: DATABASE_URL is not set")
        return 1
    now = odds.parse_commence(args.now) if args.now else datetime.now(timezone.utc)
    result = run_capture(now, _db_deps(), prop_regions=close.prop_regions(os.environ), dry_run=args.dry_run)
    print(f"[capture-close] {result}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

`.github/workflows/capture-close.yml`:

````yaml
name: capture-close

# Pre-kick CLOSING prices for the pick ledger (scripts/capture_close.py; db/migration_pick_ledger.sql). Every ~10
# minutes inside game-day hours the job looks for games kicking off in 5-25 minutes that have no close yet, pulls
# ONE game-lines call per sport (h2h / spreads / totals, regions us,eu) and one props call per event that holds
# a ledger prop pick (regions CLOSE_PROP_REGIONS, default us,eu), and stores the prices in odds_snapshot plus one
# close_capture row per event. A run with no kickoff in range exits after a database read: zero API calls.
#
# Cron is UTC and GitHub delays or skips firings under load, so: minutes are off the :00/:30 peaks (:03 football,
# :08 baseball, so the two schedules never fire together), the capture window is 20 minutes wide (two runs per
# event), and a failed or skipped attempt is simply retried by the next run. Football days are Thu-Mon (ET): the
# first block is 13:00-23:59 UTC on Sun/Mon/Thu/Fri/Sat, the second the late spill 00:00-04:59 UTC on
# Sun/Mon/Tue/Fri/Sat (Thu-night, Fri-night, Sat-night, SNF, MNF kickoffs). Months: football Aug-Feb, baseball Mar-Nov.
# Credits: about 6 per sport-run with a target plus (markets with picks x regions) per prop event, roughly
# 360 a week in season; the script's budget guard skips props below 3,000 credits left and everything below 1,000.
on:
  schedule:
    - cron: "3-59/10 13-23 * 1,2,8-12 0,1,4,5,6"
    - cron: "3-59/10 0-4 * 1,2,8-12 0,1,2,5,6"
    - cron: "8-59/10 15-23 * 3-11 *"
    - cron: "8-59/10 0-2 * 3-11 *"
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Plan only: print the targets, make no Odds API call and no writes"
        type: boolean
        default: false

permissions:
  contents: read

# Overlapping runs would both buy the same close; queue instead (one run may wait, none is cancelled mid-call).
concurrency:
  group: capture-close
  cancel-in-progress: false

jobs:
  capture:
    runs-on: ubuntu-latest
    timeout-minutes: 8
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
      - name: Sync deps
        run: uv sync
      - name: Capture pre-kick closing prices
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          CLOSE_PROP_REGIONS: ${{ vars.CLOSE_PROP_REGIONS }}
        run: uv run python scripts/capture_close.py ${{ github.event.inputs.dry_run == 'true' && '--dry-run' || '' }}
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger tests/test_ingest_odds.py tests/test_workflows_props_ml.py`
Expected: all PASS (`tests/ledger` alone: 159 passed; `test_every_workflow_parses` parses the new workflow).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ledger/close.py scripts/capture_close.py .github/workflows/capture-close.yml tests/ledger/test_close.py tests/ledger/test_capture_close_script.py tests/ledger/test_workflows_capture.py
git commit -m "feat(ledger): pre-kick close capture job with budget guard"
```

---

### Task 6: Close reference and CLV maths

**Files:**
- Create: `src/sportsmodel/ledger/reference.py`
- Create: `tests/ledger/test_reference.py`

**Interfaces:**
- Consumes: existing `serving.board.MAJOR_BOOKS`, `decimal_odds`, `implied_prob`, `novig`; `serving.props_ev.SIM_TO_ODDS_MARKET`, `MLB_SIM_TO_ODDS_MARKET`, `normalize_player_name`.
- Produces (pure; used by Task 7):
  - `reference.close_reference(pick, rows) -> {"close_kind": "pinnacle_novig" | "us_consensus_novig" | "missing", "fair_prob": float | None, "close_line": float | None, "same_line": bool | None, "books": int}`. `pick` needs `sport, market, side, line, player_id, player_name` (a pick with a `player_id` is a prop); `rows` are close-snapshot rows (`market, side, book, price, line, player_name`). Order: Pinnacle at the exact line; else >= 3 US books (deduplicated by display name, Pinnacle and offshore books excluded) at the exact line, median no-vig; else Pinnacle on another number (line move only, `fair_prob` None); else missing. `fair_prob` is always at the PICK'S line.
  - `reference.clv_prob(fair_prob, best_price) -> float | None`; `reference.clv_pts(market, side, pick_line, close_line) -> float | None` (side's own line; positive = better than the close); `reference.beat_close(prob_clv, pts_clv) -> bool | None`
  - `reference.settle_game(market, side, line, margin, total) -> "W" | "L" | "P" | None` (`margin` = home - away; `line` is the side's own line); `reference.settle_prop(side, line, actual)`; `reference.units_at_pick(result, price)`; `reference.STANDARD_PRICE = -110`

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_reference.py`:

````python
"""Close reference selection and CLV maths. Pure."""
import pytest

from sportsmodel.ledger import reference as ref


def row(market, side, book, price, line=None, player_name=""):
    return {"market": market, "side": side, "book": book, "price": price, "line": line, "player_name": player_name}


def two_way(market, book, p_a, p_b, line_a=None, line_b=None, sides=("home", "away"), player=""):
    return [row(market, sides[0], book, p_a, line_a, player), row(market, sides[1], book, p_b, line_b, player)]


def spread_pick(side="home", line=-3.5):
    return {"sport": "nfl", "market": "spread", "side": side, "line": line, "player_id": None, "player_name": None}


def prop_pick(side="over", line=64.5):
    return {"sport": "nfl", "market": "rec_yds", "side": side, "line": line, "player_id": "00-1", "player_name": "A.J. Brown Jr."}


# ---- reference selection --------------------------------------------------------------------------------

def test_pinnacle_at_the_exact_line_wins_and_is_no_vigged():
    rows = (two_way("spread", "pinnacle", -110, -110, -3.5, 3.5)
            + two_way("spread", "draftkings", -120, 100, -3.5, 3.5))
    out = ref.close_reference(spread_pick(), rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(0.5)
    assert out["same_line"] is True and out["close_line"] == -3.5 and out["books"] == 1


def test_away_side_reads_the_mirrored_line():
    rows = two_way("spread", "pinnacle", -115, -105, -3.5, 3.5)
    out = ref.close_reference(spread_pick("away", 3.5), rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(1 - ref.novig(-115, -105))


def test_without_pinnacle_three_us_books_at_the_exact_line_give_the_median():
    rows = (two_way("total", "draftkings", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "fanduel", -115, -105, 47.5, 47.5, ("over", "under"))
            + two_way("total", "betmgm", -120, 100, 47.5, 47.5, ("over", "under")))
    pick = {"sport": "nfl", "market": "total", "side": "over", "line": 47.5, "player_id": None, "player_name": None}
    out = ref.close_reference(pick, rows)
    fairs = sorted([0.5, ref.novig(-115, -105), ref.novig(-120, 100)])
    assert out["close_kind"] == "us_consensus_novig" and out["fair_prob"] == pytest.approx(fairs[1])
    assert out["books"] == 3 and out["same_line"] is True


def test_two_us_books_are_not_a_consensus():
    rows = (two_way("total", "draftkings", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "fanduel", -115, -105, 47.5, 47.5, ("over", "under")))
    pick = {"sport": "nfl", "market": "total", "side": "over", "line": 47.5, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "missing"


def test_a_book_at_another_line_does_not_count_toward_the_consensus():
    rows = (two_way("total", "draftkings", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "fanduel", -110, -110, 47.5, 47.5, ("over", "under"))
            + two_way("total", "betmgm", -110, -110, 48.5, 48.5, ("over", "under")))
    pick = {"sport": "nfl", "market": "total", "side": "over", "line": 47.5, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "missing"


def test_caesars_and_williamhill_are_one_book_and_offshore_books_do_not_count():
    rows = []
    for book in ("williamhill_us", "caesars", "bovada", "draftkings"):
        rows += two_way("moneyline", book, -150, 130)
    pick = {"sport": "nfl", "market": "moneyline", "side": "home", "line": None, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "missing"            # draftkings + one Caesars = 2 books
    rows += two_way("moneyline", "fanduel", -150, 130)
    out = ref.close_reference(pick, rows)
    assert out["close_kind"] == "us_consensus_novig" and out["books"] == 3


def test_moneyline_has_no_line_to_match():
    rows = two_way("moneyline", "pinnacle", -200, 170)
    pick = {"sport": "cfb", "market": "moneyline", "side": "away", "line": None, "player_id": None, "player_name": None}
    out = ref.close_reference(pick, rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(1 - ref.novig(-200, 170))
    assert out["same_line"] is True and out["close_line"] is None


def test_pinnacle_on_another_line_gives_the_line_move_but_no_probability():
    rows = two_way("spread", "pinnacle", -110, -110, -4.5, 4.5)
    out = ref.close_reference(spread_pick("home", -3.5), rows)
    assert out == {"close_kind": "pinnacle_novig", "fair_prob": None, "close_line": -4.5, "same_line": False, "books": 1}


def test_consensus_at_the_exact_line_beats_a_moved_pinnacle_but_reports_the_moved_line():
    rows = two_way("spread", "pinnacle", -110, -110, -4.5, 4.5)
    for b in ("draftkings", "fanduel", "betmgm"):
        rows += two_way("spread", b, -110, -110, -3.5, 3.5)
    out = ref.close_reference(spread_pick("home", -3.5), rows)
    assert out["close_kind"] == "us_consensus_novig" and out["fair_prob"] == pytest.approx(0.5)
    assert out["close_line"] == -4.5 and out["same_line"] is False


def test_nothing_quoted_is_missing():
    assert ref.close_reference(spread_pick(), []) == {
        "close_kind": "missing", "fair_prob": None, "close_line": None, "same_line": None, "books": 0}
    assert ref.close_reference(spread_pick(), two_way("total", "pinnacle", -110, -110, 47.5, 47.5, ("over", "under")))["close_kind"] == "missing"


def test_a_one_sided_quote_is_not_usable():
    rows = [row("spread", "pinnacle", -110, -110, -3.5)]
    assert ref.close_reference(spread_pick(), rows)["close_kind"] == "missing"


def test_props_match_the_player_by_normalised_name_and_the_exact_line():
    rows = (two_way("reception_yds", "pinnacle", -120, 100, 64.5, 64.5, ("over", "under"), player="AJ Brown")
            + two_way("reception_yds", "pinnacle", -110, -110, 64.5, 64.5, ("over", "under"), player="Someone Else"))
    out = ref.close_reference(prop_pick(), rows)
    assert out["close_kind"] == "pinnacle_novig" and out["fair_prob"] == pytest.approx(ref.novig(-120, 100))


def test_prop_consensus_ignores_other_players_and_other_lines():
    rows = []
    for b in ("draftkings", "fanduel"):
        rows += two_way("reception_yds", b, -110, -110, 64.5, 64.5, ("over", "under"), player="A.J. Brown")
    rows += two_way("reception_yds", "betmgm", -110, -110, 66.5, 66.5, ("over", "under"), player="A.J. Brown")
    rows += two_way("reception_yds", "caesars", -110, -110, 64.5, 64.5, ("over", "under"), player="Other Guy")
    assert ref.close_reference(prop_pick(), rows)["close_kind"] == "missing"
    rows += two_way("reception_yds", "betmgm", -110, -110, 64.5, 64.5, ("over", "under"), player="A.J. Brown")
    out = ref.close_reference(prop_pick(), rows)
    assert out["close_kind"] == "us_consensus_novig" and out["books"] == 3


def test_game_market_rows_are_not_confused_with_prop_rows_and_unknown_markets_are_missing():
    rows = two_way("spread", "pinnacle", -110, -110, -3.5, 3.5, player="Some Player")
    assert ref.close_reference(spread_pick(), rows)["close_kind"] == "missing"        # player rows are props
    odd = {"sport": "nfl", "market": "anytime_td", "side": "over", "line": 0.5, "player_id": "1", "player_name": "x"}
    assert ref.close_reference(odd, [])["close_kind"] == "missing"


def test_float32_lines_still_match():
    rows = two_way("total", "pinnacle", -110, -110, 4.099999904632568, 4.099999904632568, ("over", "under"))
    pick = {"sport": "mlb", "market": "total", "side": "over", "line": 4.1, "player_id": None, "player_name": None}
    assert ref.close_reference(pick, rows)["close_kind"] == "pinnacle_novig"


# ---- CLV maths -----------------------------------------------------------------------------------------

def test_clv_prob_is_close_fair_minus_the_implied_price_taken():
    assert ref.clv_prob(0.55, -110) == pytest.approx(0.55 - 110 / 210)
    assert ref.clv_prob(0.50, 105) == pytest.approx(0.5 - 100 / 205)
    assert ref.clv_prob(None, -110) is None and ref.clv_prob(0.5, None) is None


@pytest.mark.parametrize("market, side, pick, close, expected", [
    ("spread", "home", -3.5, -4.5, 1.0),      # took -3.5, market now -4.5: a point better
    ("spread", "home", -3.5, -2.5, -1.0),
    ("spread", "away", 3.5, 2.5, 1.0),
    ("spread", "away", 3.5, 4.5, -1.0),
    ("total", "over", 45.5, 47.0, 1.5),
    ("total", "over", 45.5, 44.5, -1.0),
    ("total", "under", 45.5, 44.5, 1.0),
    ("total", "under", 45.5, 47.0, -1.5),
    ("spread", "home", -3.5, -3.5, 0.0),
    ("moneyline", "home", None, None, None),
    ("spread", "home", -3.5, None, None),
])
def test_clv_pts_signs(market, side, pick, close, expected):
    got = ref.clv_pts(market, side, pick, close)
    assert got == expected if expected is None else got == pytest.approx(expected)


def test_prop_clv_pts_uses_the_total_convention():
    assert ref.clv_pts("rec_yds", "over", 64.5, 66.5) == pytest.approx(2.0)
    assert ref.clv_pts("rec_yds", "under", 64.5, 66.5) == pytest.approx(-2.0)


@pytest.mark.parametrize("prob, pts, expected", [
    (0.02, 0.0, True), (-0.01, 0.0, False), (None, 0.0, None), (0.03, None, True), (None, None, None),
    (-0.05, 1.0, True), (0.05, -1.0, False),            # a moved line is judged in points, not by price
])
def test_beat_close(prob, pts, expected):
    assert ref.beat_close(prob, pts) is expected


# ---- settlement and units ------------------------------------------------------------------------------

@pytest.mark.parametrize("market, side, line, margin, total, expected", [
    ("moneyline", "home", None, 7, 40, "W"), ("moneyline", "away", None, 7, 40, "L"), ("moneyline", "home", None, 0, 40, "P"),
    ("spread", "home", -3.5, 7, 40, "W"), ("spread", "home", -7.0, 7, 40, "P"), ("spread", "home", -10.5, 7, 40, "L"),
    ("spread", "away", 3.5, 7, 40, "L"), ("spread", "away", 7.5, 7, 40, "W"), ("spread", "away", 7.0, 7, 40, "P"),
    ("total", "over", 38.5, 7, 40, "W"), ("total", "under", 40.0, 7, 40, "P"), ("total", "under", 38.5, 7, 40, "L"),
    ("spread", "home", None, 7, 40, None), ("total", "over", 40.5, 7, None, None), ("moneyline", "home", None, None, 40, None),
    ("prop", "home", 1, 1, 1, None),
])
def test_settle_game(market, side, line, margin, total, expected):
    assert ref.settle_game(market, side, line, margin, total) == expected


@pytest.mark.parametrize("side, line, actual, expected", [
    ("over", 64.5, 70, "W"), ("over", 64.5, 60, "L"), ("over", 64.0, 64, "P"),
    ("under", 64.5, 60, "W"), ("under", 64.5, 70, "L"), ("under", 64.0, 64, "P"),
    ("over", 64.5, None, None), ("over", None, 70, None), ("home", 64.5, 70, None)])
def test_settle_prop(side, line, actual, expected):
    assert ref.settle_prop(side, line, actual) == expected


def test_units_at_pick():
    assert ref.units_at_pick("W", -110) == pytest.approx(100 / 110)
    assert ref.units_at_pick("W", 150) == pytest.approx(1.5)
    assert ref.units_at_pick("L", 150) == -1.0 and ref.units_at_pick("P", -110) == 0.0
    assert ref.units_at_pick("W", None) is None and ref.units_at_pick(None, -110) is None
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_reference.py`
Expected: collection ERROR, `ImportError: cannot import name 'reference' from 'sportsmodel.ledger'`.

- [ ] **Step 3: Write the implementation**

`src/sportsmodel/ledger/reference.py`:

````python
"""Closing reference prices and CLV maths for ledger picks. PURE -- no IO.

The close reference for a pick's side is, in order:
  1. Pinnacle's no-vig probability at the pick's EXACT line          (close_kind = pinnacle_novig)
  2. the median no-vig probability of >= 3 US books at the exact line (close_kind = us_consensus_novig)
  3. Pinnacle quoting a DIFFERENT line (no comparable probability; only the line move is measured)
  4. nothing                                                           (close_kind = missing)

CLV is computed against a probability only when the close is at the pick's own line; a line that moved is
measured in points instead (`clv_pts`), never by comparing probabilities across different numbers.
"""
from __future__ import annotations

from statistics import median

from ..serving.board import MAJOR_BOOKS, decimal_odds, implied_prob, novig
from ..serving.props_ev import MLB_SIM_TO_ODDS_MARKET, SIM_TO_ODDS_MARKET, normalize_player_name

PINNACLE = "pinnacle"
MIN_CONSENSUS_BOOKS = 3
LINE_EPS = 1e-4                    # odds_snapshot.line is REAL (float32): compare with a tolerance
STANDARD_PRICE = -110              # model spread / total picks carry no market price; graded at flat -110
OTHER_SIDE = {"home": "away", "away": "home", "over": "under", "under": "over"}
PROP_TO_ODDS = {"nfl": SIM_TO_ODDS_MARKET, "mlb": MLB_SIM_TO_ODDS_MARKET}


def _same(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) <= LINE_EPS


def odds_market(sport: str, market: str, is_prop: bool) -> str | None:
    """The odds_snapshot market code for a ledger market (game markets are identical; props map sim -> odds)."""
    return PROP_TO_ODDS.get(sport, {}).get(market) if is_prop else market


def _mirror(market: str, line):
    """The line the OTHER side of the same two-way market carries at the same number."""
    if line is None:
        return None
    return -float(line) if market == "spread" else float(line)


def _book_entries(rows: list[dict], omarket: str, player_key: str | None) -> dict[str, dict[str, list[tuple]]]:
    """{book: {side: [(line, price), ...]}} for one market (and one player, for props)."""
    out: dict[str, dict[str, list[tuple]]] = {}
    for r in rows:
        if r.get("market") != omarket or r.get("price") is None:
            continue
        if player_key is not None and normalize_player_name(r.get("player_name")) != player_key:
            continue
        if player_key is None and (r.get("player_name") or ""):
            continue
        out.setdefault(r["book"], {}).setdefault(r["side"], []).append((r.get("line"), int(r["price"])))
    return out


def _price_at(entries: list[tuple], line) -> int | None:
    for l, p in entries or []:
        if line is None or _same(l, line):
            return p
    return None


def _book_fair(sides: dict[str, list[tuple]], market: str, side: str, line) -> float | None:
    """One book's no-vig probability for `side` at the exact `line`, or None if either leg is missing there."""
    mine = _price_at(sides.get(side), line)
    theirs = _price_at(sides.get(OTHER_SIDE[side]), _mirror(market, line))
    return None if mine is None or theirs is None else novig(mine, theirs)


def _us_books(book_entries: dict) -> dict[str, dict]:
    """US books deduplicated by display name (williamhill_us and caesars are one book); Pinnacle excluded."""
    seen, out = set(), {}
    for book, sides in sorted(book_entries.items()):
        name = MAJOR_BOOKS.get(book)
        if book == PINNACLE or name is None or name in seen:
            continue
        seen.add(name)
        out[book] = sides
    return out


def close_reference(pick: dict, rows: list[dict]) -> dict:
    """Close reference for one ledger pick from the close snapshot `rows` of its game.

    `pick`: sport, market, side, line, player_name, player_id (props are the picks carrying a player_id).
    Returns {close_kind, fair_prob, close_line, same_line, books}: `fair_prob` is the side's no-vig probability
    AT THE PICK'S LINE (None when only a moved-line Pinnacle quote exists); `close_line` is the side's own line
    at the close (Pinnacle's when it quotes one, else the pick's line)."""
    sport, market, side, line = pick["sport"], pick["market"], pick["side"], pick.get("line")
    is_prop = pick.get("player_id") is not None
    omarket = odds_market(sport, market, is_prop)
    missing = {"close_kind": "missing", "fair_prob": None, "close_line": None, "same_line": None, "books": 0}
    if omarket is None or side not in OTHER_SIDE:
        return missing
    player_key = normalize_player_name(pick.get("player_name")) if is_prop else None
    entries = _book_entries(rows, omarket, player_key)

    pin = entries.get(PINNACLE)
    pin_fair = _book_fair(pin, market, side, line) if pin else None
    if pin_fair is not None:
        return {"close_kind": "pinnacle_novig", "fair_prob": pin_fair, "close_line": line,
                "same_line": True, "books": 1}

    pin_line = None
    if pin and line is not None:                 # Pinnacle quotes this market, but at another number
        cands = [(l, p) for l, p in pin.get(side, []) if l is not None]
        theirs = {round(float(l), 4) for l, _ in pin.get(OTHER_SIDE[side], []) if l is not None}
        cands = [(l, p) for l, p in cands if round(_mirror(market, l), 4) in theirs]
        if cands:
            pin_line = min(cands, key=lambda lp: abs(float(lp[0]) - float(line)))[0]

    us = _us_books(entries)
    fairs = [f for f in (_book_fair(s, market, side, line) for s in us.values()) if f is not None]
    if len(fairs) >= MIN_CONSENSUS_BOOKS:
        close_line = pin_line if pin_line is not None else line
        return {"close_kind": "us_consensus_novig", "fair_prob": median(fairs), "close_line": close_line,
                "same_line": _same(close_line, line) if line is not None else True, "books": len(fairs)}

    if pin_line is not None:
        return {"close_kind": "pinnacle_novig", "fair_prob": None, "close_line": float(pin_line),
                "same_line": False, "books": 1}
    return missing


def clv_prob(fair_prob: float | None, best_price: int | None) -> float | None:
    """Close fair probability minus the implied probability of the price taken (same line only). Positive =
    the price taken was better than the close. The taker's vig is included, so a zero-skill bet scores ~ -2 to -4pp."""
    if fair_prob is None or best_price is None:
        return None
    return float(fair_prob) - implied_prob(int(best_price))


def clv_pts(market: str, side: str, pick_line, close_line) -> float | None:
    """Line points gained versus the close, from the bettor's side: spreads use the side's own line
    (pick - close), Over gains when the close is higher (close - pick), Under the reverse. None for moneylines
    or when either line is missing."""
    if market == "moneyline" or pick_line is None or close_line is None:
        return None
    d = float(pick_line) - float(close_line)
    if market == "spread" or side == "under":
        return d
    return -d if side == "over" else None


def beat_close(prob_clv: float | None, pts_clv: float | None) -> bool | None:
    """True when the pick beat the close: on points if the line moved, else on probability. None when neither
    can be measured."""
    if pts_clv is not None and abs(pts_clv) > LINE_EPS:
        return pts_clv > 0
    if prob_clv is not None:
        return prob_clv > 0
    return None


def settle_game(market: str, side: str, line, margin: float | None, total: float | None) -> str | None:
    """W / L / P for a game-line pick. `margin` = home - away; `line` is the SIDE's own line (spread) or the
    total. None when the inputs needed are missing."""
    if market == "moneyline":
        if margin is None:
            return None
        sm = margin if side == "home" else -margin
    elif market == "spread":
        if margin is None or line is None:
            return None
        sm = (margin if side == "home" else -margin) + float(line)
    elif market == "total":
        if total is None or line is None:
            return None
        sm = (total - float(line)) if side == "over" else (float(line) - total)
    else:
        return None
    return "W" if sm > 0 else "L" if sm < 0 else "P"


def settle_prop(side: str, line, actual: float | None) -> str | None:
    if actual is None or line is None or side not in ("over", "under"):
        return None
    d = float(actual) - float(line)
    sm = d if side == "over" else -d
    return "W" if sm > 0 else "L" if sm < 0 else "P"


def units_at_pick(result: str | None, price: int | None) -> float | None:
    """Profit in units for 1 unit staked at `price`."""
    if result is None or price is None:
        return None
    return {"W": decimal_odds(int(price)) - 1.0, "L": -1.0, "P": 0.0}[result]
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger`
Expected: PASS (220 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ledger/reference.py tests/ledger/test_reference.py
git commit -m "feat(ledger): close reference, CLV and settlement maths"
```

---

### Task 7: Nightly ledger grading

**Files:**
- Create: `src/sportsmodel/ledger/grading.py`, `scripts/grade_ledger.py`, `.github/workflows/grade-ledger.yml`
- Create: `tests/ledger/test_grading.py`, `tests/ledger/test_grade_ledger_script.py`, `tests/ledger/test_workflows_grade.py`

**Interfaces:**
- Consumes: Task 6's `reference.*`; Task 2's `core.as_utc`, `record.writes_disabled`; Task 1's `db.missing_ledger_tables`, `db.upsert_pick_ledger_results`, `db.LEDGER_COLUMNS`. Existing tables read: `pick_ledger`, `pick_ledger_results`, `close_capture`, `odds_snapshot`, `prediction_accuracy` (`actual_margin`, `actual_total`, written by `grade_predictions.py` for NFL/CFB/MLB), `nfl_player_actuals`, `ev_prop_results` (`actual`).
- Produces:
  - `grading.basis_rows(rows) -> {"first": row, "last": row}` (pre-kick rows only)
  - `grading.merge_close(primary, fallback) -> {(game_pk, is_prop): {"rows": [...], "captured_at": datetime}}` (capture rows beat the fallback per game and kind)
  - `grading.grade_pick_row(row, basis, finals, actuals, close) -> dict | None` (keys exactly `db.LEDGER_COLUMNS["pick_ledger_results"]`; None while the result is unknown)
  - `grading.grade_rows(ledger_rows, finals, actuals, close) -> list[dict]` (both bases per key)
  - `grade_ledger.Loaders` (injectable IO dataclass), `grade_ledger.grade_all(loaders, days) -> dict`, `grade_ledger.main(argv) -> int`
- Known limitation, by design: a prop whose player never reaches `ev_prop_results`/`nfl_player_actuals` (for example an MLB player who is not on the final board) stays ungraded; the gate then simply sees fewer graded props. No network is used for grading.

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_grading.py`:

````python
"""Ledger grading: first / last basis rows, results, units, close reference, CLV. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel import db
from sportsmodel.ledger import grading

KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)


def lrow(pick_key="k1", id=1, hours_before=48, **kw):
    base = dict(id=id, pick_key=pick_key, source="ev_game", sport="nfl", game_pk=7, market="spread", side="home",
                line=-3.5, player_id=None, player_name=None, best_price=-105, observed_at=KICK - timedelta(hours=hours_before),
                commence_time=KICK, minutes_to_kick=hours_before * 60.0)
    return {**base, **kw}


def snap(market, side, book, price, line=None, player="", at=KICK - timedelta(minutes=18)):
    return {"game_pk": 7, "market": market, "side": side, "book": book, "price": price, "line": line,
            "player_name": player, "captured_at": at}


PIN_SPREAD = [snap("spread", "home", "pinnacle", -110, -3.5), snap("spread", "away", "pinnacle", -110, 3.5)]
FINAL_HOME_BY_7 = {7: {"actual_margin": 7.0, "actual_total": 44.0}}


def close_of(rows):
    return grading.merge_close(rows, [])


# ---- first / last basis -----------------------------------------------------------------------------------

def test_first_and_last_are_the_earliest_and_latest_pre_kick_rows():
    rows = [lrow(id=2, hours_before=20), lrow(id=1, hours_before=48), lrow(id=3, hours_before=2)]
    b = grading.basis_rows(rows)
    assert b["first"]["id"] == 1 and b["last"]["id"] == 3


def test_a_single_row_is_both_first_and_last():
    b = grading.basis_rows([lrow()])
    assert b["first"] is b["last"]


def test_rows_at_or_after_kickoff_are_never_bettable():
    late = lrow(id=9, hours_before=-1)
    assert grading.basis_rows([late]) == {}
    b = grading.basis_rows([lrow(id=1, hours_before=5), late])
    assert b["last"]["id"] == 1


# ---- grading ---------------------------------------------------------------------------------------------

def test_game_pick_result_units_and_clv_against_pinnacle():
    (r1,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD)) if g["basis"] == "first"]
    assert r1["result"] == "W" and r1["units_at_pick"] == pytest.approx(100 / 105)
    assert r1["close_kind"] == "pinnacle_novig" and r1["close_fair_prob"] == pytest.approx(0.5)
    assert r1["clv_prob"] == pytest.approx(0.5 - 105 / 205) and r1["clv_pts"] == 0.0
    assert r1["beat_close"] is False and r1["close_same_line"] is True and r1["close_books"] == 1
    assert r1["close_minutes_to_kick"] == pytest.approx(18) and r1["ledger_id"] == 1
    assert set(r1) == set(db.LEDGER_COLUMNS["pick_ledger_results"])


def test_both_bases_are_written_and_can_differ_by_line():
    rows = [lrow(id=1, hours_before=60, line=-3.5, best_price=-110), lrow(id=2, hours_before=3, line=-7.5, best_price=-110)]
    out = {g["basis"]: g for g in grading.grade_rows(rows, FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD))}
    assert out["first"]["result"] == "W" and out["last"]["result"] == "L"          # won by 7: -3.5 covers, -7.5 does not
    assert out["first"]["clv_pts"] == 0.0 and out["last"]["clv_pts"] == pytest.approx(-4.0)
    assert out["last"]["close_same_line"] is False and out["last"]["clv_prob"] is None
    assert out["last"]["beat_close"] is False


def test_line_move_in_the_bettors_favor_beats_the_close_on_points():
    close = close_of([snap("spread", "home", "pinnacle", -110, -4.5), snap("spread", "away", "pinnacle", -110, 4.5)])
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, close) if g["basis"] == "first"]
    assert g["close_kind"] == "pinnacle_novig" and g["close_fair_prob"] is None
    assert g["clv_pts"] == pytest.approx(1.0) and g["beat_close"] is True and g["clv_prob"] is None


def test_missing_close_is_recorded_not_dropped():
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, {}) if g["basis"] == "first"]
    assert g["close_kind"] == "missing" and g["clv_prob"] is None and g["clv_pts"] is None
    assert g["beat_close"] is None and g["close_minutes_to_kick"] is None and g["result"] == "W"


def test_ungradable_picks_are_left_pending():
    assert grading.grade_rows([lrow()], {}, {}, close_of(PIN_SPREAD)) == []                   # no final yet
    prop = lrow(source="ev_prop", market="rec_yds", side="over", line=64.5, player_id="00-1", player_name="A. Brown")
    assert grading.grade_rows([prop], FINAL_HOME_BY_7, {}, {}) == []                          # no player actual yet


def test_model_game_picks_are_graded_at_flat_minus_110_without_a_price():
    row = lrow(source="model_game", best_price=None, market="spread")
    (g,) = [g for g in grading.grade_rows([row], FINAL_HOME_BY_7, {}, close_of(PIN_SPREAD)) if g["basis"] == "first"]
    assert g["units_at_pick"] == pytest.approx(100 / 110) and g["best_price"] is None and g["clv_prob"] is None
    ml = lrow(source="model_game", best_price=None, market="moneyline", line=None)
    (g,) = [g for g in grading.grade_rows([ml], FINAL_HOME_BY_7, {}, {}) if g["basis"] == "first"]
    assert g["result"] == "W" and g["units_at_pick"] is None                                  # no price, no units


def test_prop_pick_uses_player_actuals_and_the_props_close():
    prop = lrow(source="ev_prop", market="rec_yds", side="over", line=64.5, player_id="00-1", player_name="A. Brown",
                best_price=-105)
    rows = [snap("reception_yds", "over", "pinnacle", -120, 64.5, "A. Brown"),
            snap("reception_yds", "under", "pinnacle", 100, 64.5, "A. Brown")]
    close = grading.merge_close(rows, [])
    out = grading.grade_rows([prop], {}, {(7, "00-1", "rec_yds"): 80.0}, close)
    g = [g for g in out if g["basis"] == "first"][0]
    assert g["result"] == "W" and g["close_kind"] == "pinnacle_novig" and g["close_same_line"] is True
    assert g["clv_prob"] == pytest.approx(grading.ref.novig(-120, 100) - 105 / 205)
    push = grading.grade_rows([prop], {}, {(7, "00-1", "rec_yds"): 64.5}, close)
    assert push[0]["result"] == "P" and push[0]["units_at_pick"] == 0.0


def test_props_and_lines_closes_are_kept_apart_per_game():
    both = PIN_SPREAD + [snap("reception_yds", "over", "pinnacle", -110, 64.5, "A. Brown")]
    close = grading.merge_close(both, [])
    assert set(close) == {(7, False), (7, True)}


def test_capture_rows_beat_the_stale_fallback_per_game_and_kind():
    old = KICK - timedelta(minutes=170)
    fb = [snap("spread", "home", "pinnacle", -130, -3.5, at=old), snap("spread", "away", "pinnacle", 110, 3.5, at=old)]
    merged = grading.merge_close(PIN_SPREAD, fb)
    assert merged[(7, False)]["captured_at"] == KICK - timedelta(minutes=18) and len(merged[(7, False)]["rows"]) == 2
    only_fb = grading.merge_close([], fb)
    (g,) = [g for g in grading.grade_rows([lrow()], FINAL_HOME_BY_7, {}, only_fb) if g["basis"] == "first"]
    assert g["close_minutes_to_kick"] == pytest.approx(170)          # the gate reads this: an old close is visible as one
````

`tests/ledger/test_grade_ledger_script.py`:

````python
"""scripts/grade_ledger.py on injected loaders and recorded SQL. No database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("grade_ledger", ROOT / "scripts" / "grade_ledger.py")
gl = importlib.util.module_from_spec(_spec)
sys.modules["grade_ledger"] = gl
_spec.loader.exec_module(gl)

KICK = datetime(2026, 10, 18, 17, 0, tzinfo=timezone.utc)


def loaders(**over):
    calls = {}
    base = dict(
        missing_tables=lambda: [],
        pending_keys=lambda days: ["k1"],
        ledger_rows=lambda keys: [{"id": 1, "pick_key": "k1", "source": "ev_game", "sport": "nfl", "game_pk": 7,
                                   "market": "moneyline", "side": "home", "line": None, "player_id": None,
                                   "player_name": None, "best_price": 120, "observed_at": KICK - timedelta(hours=30),
                                   "commence_time": KICK, "minutes_to_kick": 1800.0}],
        finals=lambda pks: {7: {"actual_margin": 3.0, "actual_total": 40.0}},
        prop_actuals=lambda pks: {},
        close_rows=lambda pks: ([{"game_pk": 7, "market": "moneyline", "side": "home", "book": "pinnacle", "price": 110,
                                  "line": None, "player_name": "", "captured_at": KICK - timedelta(minutes=15)},
                                 {"game_pk": 7, "market": "moneyline", "side": "away", "book": "pinnacle", "price": -125,
                                  "line": None, "player_name": "", "captured_at": KICK - timedelta(minutes=15)}], []),
        store=lambda rows: calls.setdefault("stored", rows) and len(rows))   # remember the rows, return their count
    return gl.Loaders(**{**base, **over}), calls


def test_grades_pending_keys_on_both_bases_and_stores_them():
    ld, calls = loaders()
    out = gl.grade_all(ld, 14)
    assert out["status"] == "ok" and out["keys"] == 1 and out["graded"] == 2
    assert {r["basis"] for r in calls["stored"]} == {"first", "last"} and out["first_basis_close_kinds"] == {"pinnacle_novig": 1}


def test_nothing_pending_stores_nothing():
    ld, calls = loaders(pending_keys=lambda days: [])
    assert gl.grade_all(ld, 14) == {"status": "ok", "keys": 0, "graded": 0} and "stored" not in calls


def test_missing_tables_warn_and_stop(capsys):
    ld, calls = loaders(missing_tables=lambda: ["pick_ledger_results"], pending_keys=lambda d: pytest.fail("queried"))
    assert gl.grade_all(ld, 14) == {"status": "no_table"}
    assert "::warning::grade-ledger: table pick_ledger_results missing" in capsys.readouterr().out


def test_unfinished_games_stay_pending():
    ld, calls = loaders(finals=lambda pks: {})
    assert gl.grade_all(ld, 14)["graded"] == 0 and "stored" not in calls


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert gl.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_default_queries(monkeypatch):
    from sportsmodel import db
    seen = []

    class Cur:
        description = [("a",)]

        def execute(self, sql, params):
            seen.append((" ".join(sql.split()), params))

        def fetchall(self):
            return []

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn(Cur):
        def cursor(self):
            return Cur()

    monkeypatch.setattr(db, "get_postgres", lambda: Conn())
    ld = gl._db_loaders()
    ld.pending_keys(14)
    ld.ledger_rows(["k"])
    ld.finals([7])
    ld.prop_actuals([7])
    ld.close_rows([7])
    sql = [s for s, _ in seen]
    assert "interval '4 hours'" in sql[0] and "r.basis = 'last'" in sql[0] and seen[0][1] == (14,)
    assert "FROM pick_ledger WHERE pick_key = ANY(%s) ORDER BY pick_key, observed_at, id" in sql[1]
    assert "FROM prediction_accuracy" in sql[2] and "FROM nfl_player_actuals" in sql[3] and "FROM ev_prop_results" in sql[4]
    assert "JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at" in sql[5]
    assert "c.status = 'ok'" in sql[5] and "interval '180 minutes'" in sql[6]
    assert not any(w in s.upper().split() for s in sql for w in ("INSERT", "UPDATE", "DELETE"))
````

`tests/ledger/test_workflows_grade.py`:

````python
"""Structural checks of .github/workflows/grade-ledger.yml: nightly, after the other graders, free of Odds API credits."""
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of


def test_grade_ledger_runs_nightly_after_the_other_graders_and_never_spends_credits():
    wf = load("grade-ledger.yml")
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["10 14 * * *"]
    assert wf["on"]["workflow_dispatch"]["inputs"]["days"]["default"] == "14"
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/grade_ledger.py"))]
    assert step["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}          # no ODDS_API_KEY: grading is free
    assert step["run"] == 'uv run python scripts/grade_ledger.py --days "${{ github.event.inputs.days || \'14\' }}"'
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("grade_ledger.py"))
    text = (WF / "grade-ledger.yml").read_text()
    assert "git push" not in text and "ODDS_API_KEY" not in text.split("jobs:")[1]
    # later than grade-ev (13:20) and grade-ev-props (13:30) so the night's finals and actuals are in
    assert "20 13 * * *" in (WF / "grade-ev.yml").read_text() and "30 13 * * *" in (WF / "grade-ev-props.yml").read_text()
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_grading.py tests/ledger/test_grade_ledger_script.py tests/ledger/test_workflows_grade.py`
Expected: collection ERRORs: `ImportError: cannot import name 'grading' from 'sportsmodel.ledger'` and `FileNotFoundError: ... scripts/grade_ledger.py`.

- [ ] **Step 3: Write the implementation**

`src/sportsmodel/ledger/grading.py`:

````python
"""Grade ledger picks against results and the captured close. PURE -- no IO.

Every pick_key is graded twice into pick_ledger_results: basis 'first' (its FIRST ledger row = the earliest
bettable observation) and basis 'last' (its LAST row before kickoff). A pick is graded only once its result is
known; the close comes from `reference.close_reference` over the close snapshot of its game.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from . import reference as ref
from .core import as_utc


def basis_rows(rows: list[dict]) -> dict[str, dict]:
    """{'first': earliest, 'last': latest} among a key's rows observed BEFORE kickoff (a row at or after kickoff
    was never bettable). Empty when no row qualifies."""
    ok = []
    for r in rows:
        obs, kick = as_utc(r.get("observed_at")), as_utc(r.get("commence_time"))
        if obs is not None and kick is not None and obs < kick:
            ok.append((obs, r.get("id") or 0, r))
    if not ok:
        return {}
    ok.sort(key=lambda t: (t[0], t[1]))
    return {"first": ok[0][2], "last": ok[-1][2]}


def merge_close(primary: list[dict], fallback: list[dict]) -> dict[tuple, dict]:
    """{(game_pk, is_prop): {'rows': [...], 'captured_at': datetime}}. `primary` = snapshot rows stored by the
    close capture; `fallback` = the latest pre-kick snapshot within the stale cutoff, used only for a
    (game, kind) with no capture. Rows carry game_pk, player_name and captured_at."""
    out: dict[tuple, dict] = {}
    for rows in (primary, fallback):
        grouped: dict[tuple, list[dict]] = defaultdict(list)
        for r in rows:
            grouped[(r["game_pk"], bool(r.get("player_name")))].append(r)
        for key, rs in grouped.items():
            if key not in out:
                out[key] = {"rows": rs, "captured_at": max(as_utc(r["captured_at"]) for r in rs)}
    return out


def grade_pick_row(row: dict, basis: str, finals: dict, actuals: dict, close: dict) -> dict | None:
    """One pick_ledger_results row for ledger `row`, or None while its result is unknown.

    finals: {game_pk: {'actual_margin', 'actual_total'}}; actuals: {(game_pk, player_id, market): float};
    close: merge_close output."""
    is_prop = row.get("player_id") is not None
    if is_prop:
        result = ref.settle_prop(row["side"], row.get("line"),
                                 actuals.get((row["game_pk"], str(row["player_id"]), row["market"])))
    else:
        f = finals.get(row["game_pk"]) or {}
        result = ref.settle_game(row["market"], row["side"], row.get("line"), f.get("actual_margin"),
                                 f.get("actual_total"))
    if result is None:
        return None

    price = row.get("best_price")
    if price is None and row.get("source") == "model_game" and row["market"] in ("spread", "total"):
        price = ref.STANDARD_PRICE                 # model picks carry no market price: flat -110, as the audit did
    snap = close.get((row["game_pk"], is_prop))
    reference = ref.close_reference(row, snap["rows"]) if snap else \
        {"close_kind": "missing", "fair_prob": None, "close_line": None, "same_line": None, "books": 0}
    prob_clv = ref.clv_prob(reference["fair_prob"], row.get("best_price"))
    pts_clv = ref.clv_pts(row["market"], row["side"], row.get("line"), reference["close_line"])
    kick = as_utc(row["commence_time"])
    close_minutes = None
    if snap and reference["close_kind"] != "missing":
        close_minutes = (kick - snap["captured_at"]).total_seconds() / 60.0
    return {
        "pick_key": row["pick_key"], "basis": basis, "ledger_id": row["id"], "source": row["source"],
        "sport": row["sport"], "game_pk": row["game_pk"], "market": row["market"], "side": row["side"],
        "line": row.get("line"), "player_id": row.get("player_id"), "player_name": row.get("player_name"),
        "best_price": row.get("best_price"), "observed_at": row["observed_at"], "commence_time": row["commence_time"],
        "minutes_to_kick": row["minutes_to_kick"], "result": result,
        "units_at_pick": ref.units_at_pick(result, price),
        "close_kind": reference["close_kind"], "close_fair_prob": reference["fair_prob"],
        "close_line": reference["close_line"], "close_same_line": reference["same_line"],
        "close_books": reference["books"] or None, "close_minutes_to_kick": close_minutes,
        "clv_prob": prob_clv, "clv_pts": pts_clv, "beat_close": ref.beat_close(prob_clv, pts_clv),
    }


def grade_rows(ledger_rows: list[dict], finals: dict, actuals: dict, close: dict) -> list[dict]:
    """Grade every pick_key found in `ledger_rows` (all rows of those keys) on both bases."""
    by_key: dict[str, list[dict]] = defaultdict(list)
    for r in ledger_rows:
        by_key[r["pick_key"]].append(r)
    out = []
    for rows in by_key.values():
        for basis, r in basis_rows(rows).items():
            g = grade_pick_row(r, basis, finals, actuals, close)
            if g is not None:
                out.append(g)
    return out
````

`scripts/grade_ledger.py`:

````python
"""Grade the pick ledger against results and the pre-kick close -> pick_ledger_results (nightly).

For every pick_key whose game started more than 4 hours ago and that has no result yet, grade the key's FIRST
ledger row (earliest bettable observation) and its LAST pre-kick row: result (W / L / P), units at the price taken,
and CLV against the close captured by scripts/capture_close.py (Pinnacle no-vig; else a >= 3 US-book consensus at
the exact line; else 'missing'). Results come from rows we already store (prediction_accuracy, ev_prop_results,
nfl_player_actuals): no network. Idempotent.

    uv run python scripts/grade_ledger.py [--days 14]
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger.grading import grade_rows, merge_close  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

STALE_CLOSE_MIN = 180      # no capture for a game: use its last pre-kick snapshot if it is at most this old
SETTLE_HOURS = 4           # grade a key only this long after kickoff (results and actuals have landed)


@dataclass
class Loaders:
    missing_tables: Callable[[], list[str]]
    pending_keys: Callable[[int], list[str]]
    ledger_rows: Callable[[list[str]], list[dict]]
    finals: Callable[[list[int]], dict]
    prop_actuals: Callable[[list[int]], dict]
    close_rows: Callable[[list[int]], tuple[list[dict], list[dict]]]       # (captured, fallback)
    store: Callable[[list[dict]], int]


def grade_all(ld: Loaders, days: int) -> dict:
    gone = ld.missing_tables()
    if gone:
        print(f"::warning::grade-ledger: table {gone[0]} missing — run db/migration_pick_ledger.sql", flush=True)
        return {"status": "no_table"}
    keys = ld.pending_keys(days)
    if not keys:
        return {"status": "ok", "keys": 0, "graded": 0}
    rows = ld.ledger_rows(keys)
    pks = sorted({r["game_pk"] for r in rows})
    captured, fallback = ld.close_rows(pks)
    results = grade_rows(rows, ld.finals(pks), ld.prop_actuals(pks), merge_close(captured, fallback))
    n = ld.store(results) if results else 0
    kinds: dict[str, int] = {}
    for r in results:
        if r["basis"] == "first":
            kinds[r["close_kind"]] = kinds.get(r["close_kind"], 0) + 1
    return {"status": "ok", "keys": len(keys), "graded": n, "first_basis_close_kinds": kinds}


def _db_loaders() -> Loaders:
    from sportsmodel import db

    def q(sql, params):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def pending(days):
        return [r["pick_key"] for r in q(
            f"SELECT DISTINCT l.pick_key FROM pick_ledger l "
            f"WHERE l.commence_time <= now() - interval '{SETTLE_HOURS} hours' "
            f"AND l.commence_time >= now() - make_interval(days => %s) "
            f"AND NOT EXISTS (SELECT 1 FROM pick_ledger_results r WHERE r.pick_key = l.pick_key AND r.basis = 'last')",
            (days,))]

    def ledger_rows(keys):
        cols = ["id"] + db.LEDGER_COLUMNS["pick_ledger"]
        return q(f"SELECT {', '.join(cols)} FROM pick_ledger WHERE pick_key = ANY(%s) "
                 f"ORDER BY pick_key, observed_at, id", (keys,))

    def finals(pks):
        rows = q("SELECT game_pk, actual_margin, actual_total FROM prediction_accuracy WHERE game_pk = ANY(%s)", (pks,))
        return {r["game_pk"]: r for r in rows}

    def prop_actuals(pks):
        out = {}
        for sql in ("SELECT game_pk, player_id, market, actual FROM nfl_player_actuals "
                    "WHERE game_pk = ANY(%s) AND actual IS NOT NULL",
                    "SELECT game_pk, player_id, market, actual FROM ev_prop_results "
                    "WHERE game_pk = ANY(%s) AND actual IS NOT NULL"):
            for r in q(sql, (pks,)):
                out[(r["game_pk"], str(r["player_id"]), r["market"])] = float(r["actual"])
        return out

    def close_rows(pks):
        captured = q(
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM close_capture c JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at "
            "WHERE c.game_pk = ANY(%s) AND c.status = 'ok' "
            "AND ((c.kind = 'lines' AND s.player_name = '') OR (c.kind = 'props' AND s.player_name <> ''))", (pks,))
        fallback = q(
            "WITH latest AS (SELECT game_pk, (player_name <> '') AS is_prop, max(captured_at) AS ts "
            "  FROM odds_snapshot WHERE game_pk = ANY(%s) AND captured_at <= commence_time "
            f"  AND captured_at >= commence_time - interval '{STALE_CLOSE_MIN} minutes' "
            "  GROUP BY game_pk, (player_name <> '')) "
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM odds_snapshot s JOIN latest l ON l.game_pk = s.game_pk AND l.is_prop = (s.player_name <> '') "
            "AND l.ts = s.captured_at", (pks,))
        return captured, fallback

    return Loaders(
        missing_tables=lambda: db.missing_ledger_tables(), pending_keys=pending, ledger_rows=ledger_rows,
        finals=finals, prop_actuals=prop_actuals, close_rows=close_rows, store=db.upsert_pick_ledger_results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=14, help="grade games that started within this many days")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("grade-ledger: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::grade-ledger: DATABASE_URL is not set")
        return 1
    print(f"[grade-ledger] {grade_all(_db_loaders(), args.days)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

`.github/workflows/grade-ledger.yml`:

````yaml
name: grade-ledger

# Nightly: grade the pick ledger against results and the pre-kick close -> pick_ledger_results
# (scripts/grade_ledger.py; db/migration_pick_ledger.sql). Every pick_key whose game started more than 4 hours ago and has
# no result yet is graded from its FIRST ledger row (earliest bettable observation) and its LAST pre-kick row:
# result, units at the price taken, and CLV against the captured close. Reads rows we already store
# (prediction_accuracy, ev_prop_results, nfl_player_actuals, odds_snapshot, close_capture): no Odds API credits,
# no network. Daily 14:10 UTC, after grade-predictions (every 6h), grade-ev (13:20) and grade-ev-props (13:30) so
# the prior night's finals and actuals are in. Idempotent.
on:
  workflow_dispatch:
    inputs:
      days:
        description: "Grade games that started within this many days"
        type: string
        default: "14"
  schedule:
    - cron: "10 14 * * *"

permissions:
  contents: read

concurrency:
  group: grade-ledger
  cancel-in-progress: false

jobs:
  grade:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
      - name: Sync deps
        run: uv sync
      - name: Grade ledger picks (result, units, CLV vs the close)
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/grade_ledger.py --days "${{ github.event.inputs.days || '14' }}"
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger tests/test_workflows_props_ml.py`
Expected: all PASS (`tests/ledger` alone: 239 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ledger/grading.py scripts/grade_ledger.py .github/workflows/grade-ledger.yml tests/ledger/test_grading.py tests/ledger/test_grade_ledger_script.py tests/ledger/test_workflows_grade.py
git commit -m "feat(ledger): nightly ledger grading against results and the close"
```

---

### Task 8: Weekly gate report

**Files:**
- Create: `src/sportsmodel/ledger/gate.py`, `scripts/report_ledger_gate.py`, `.github/workflows/report-ledger-gate.yml`
- Create: `tests/ledger/test_gate.py`, `tests/ledger/test_workflows_gate.py`

**Interfaces:**
- Consumes: Task 7's `pick_ledger_results` rows; Task 5's `close_capture` rows; Task 2's `core.as_utc`, `record.writes_disabled`; Task 1's `db.missing_ledger_tables`; `tests.ledger.test_workflows_capture.cron_sets` (test helper).
- Produces:
  - `gate.window_end_for(now) -> datetime` (latest Tuesday 12:00 UTC at or before `now`)
  - `gate.lines_coverage(graded_games, lines_minutes) -> {"covered", "total", "pct"}`, `gate.props_coverage(prop_results)`, `gate.audit_check(previous, recomputed) -> {"passed", "detail"}`, `gate.evaluate_week(lines, props, audit) -> {"checks": [...], "week_pass": bool}`, `gate.step_status(week_pass, window_start, previous) -> {"streak", "step1_pass"}`
  - `gate.render_report(...) -> str` ending in a machine-readable `<!-- ledger-gate: {...} -->` block; `gate.parse_report(text) -> dict | None`; `gate.latest_previous(reports, window_end) -> dict | None`
  - `gate.regions_line(prop_regions) -> str` (the report states which Odds API regions the prop closes used, so "Pinnacle was NOT requested for props" is on the record when the probe came back negative)
  - `report_ledger_gate.collect(q, start, end, previous)`, `.build_report(end, data, previous) -> (markdown, result)`, `.main(argv) -> int` (writes `docs/superpowers/reports/YYYY-MM-DD-ledger-gate.md`; read-only against the database)

- [ ] **Step 1: Write the failing tests**

`tests/ledger/test_gate.py`:

````python
"""The weekly gate: coverage maths, the row-audit, two-consecutive-weekend logic, the report round trip, and
scripts/report_ledger_gate.py on a fake query function. Pure / no database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel.ledger import gate

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("report_ledger_gate", ROOT / "scripts" / "report_ledger_gate.py")
rg = importlib.util.module_from_spec(_spec)
sys.modules["report_ledger_gate"] = rg
_spec.loader.exec_module(rg)

END = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)          # a Tuesday
START = END - timedelta(days=7)


# ---- window ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("now, expected", [
    (datetime(2026, 10, 20, 15, 45, tzinfo=timezone.utc), END),                        # Tuesday afternoon
    (datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc), END),
    (datetime(2026, 10, 20, 11, 59, tzinfo=timezone.utc), END - timedelta(days=7)),    # before noon: last week's
    (datetime(2026, 10, 23, 3, 0, tzinfo=timezone.utc), END),                          # Friday
    (datetime(2026, 10, 26, 9, 0, tzinfo=timezone.utc), END)])                         # Monday
def test_window_end_is_the_latest_tuesday_noon(now, expected):
    assert gate.window_end_for(now) == expected


# ---- coverage ----------------------------------------------------------------------------------------

def test_lines_coverage_counts_only_football_closes_within_thirty_minutes():
    games = [("nfl", 1), ("nfl", 2), ("cfb", 3), ("cfb", 4), ("mlb", 5), ("nfl", 1)]
    mins = {("nfl", 1): 12.0, ("nfl", 2): 31.0, ("cfb", 3): 0.0, ("mlb", 5): 5.0}          # cfb 4: no capture at all
    cov = gate.lines_coverage(games, mins)
    assert cov == {"covered": 2, "total": 4, "pct": 50.0}


def test_lines_coverage_boundaries_and_an_empty_window():
    assert gate.lines_coverage([("nfl", 1)], {("nfl", 1): 30.0})["covered"] == 1
    assert gate.lines_coverage([("nfl", 1)], {("nfl", 1): -0.5})["covered"] == 0            # captured after kickoff
    assert gate.lines_coverage([], {}) == {"covered": 0, "total": 0, "pct": None}


def test_props_coverage_needs_a_kind_and_the_exact_line():
    rows = [{"close_kind": "pinnacle_novig", "close_same_line": True},
            {"close_kind": "us_consensus_novig", "close_same_line": True},
            {"close_kind": "pinnacle_novig", "close_same_line": False},          # Pinnacle on another number
            {"close_kind": "missing", "close_same_line": None}]
    assert gate.props_coverage(rows) == {"covered": 2, "total": 4, "pct": 50.0}
    assert gate.props_coverage([])["pct"] is None


# ---- audit -------------------------------------------------------------------------------------------

PREV = {"n": 100, "md5": "abc", "as_of": "2026-10-13T12:00:00+00:00"}


def test_audit_baseline_unchanged_and_tampered():
    assert gate.audit_check(None, {"n": 5, "md5": "x"})["passed"] is True
    assert gate.audit_check(PREV, {"n": 100, "md5": "abc"})["passed"] is True
    assert gate.audit_check(PREV, {"n": 99, "md5": "abc"})["passed"] is False              # a row vanished
    bad = gate.audit_check(PREV, {"n": 100, "md5": "zzz"})                                 # a row was edited
    assert bad["passed"] is False and "was 100 / abc, now 100 / zzz" in bad["detail"]
    assert gate.audit_check(PREV, None)["passed"] is False


# ---- evaluation and the two-weekend rule ---------------------------------------------------------------

GOOD_LINES = {"covered": 19, "total": 20, "pct": 95.0}
GOOD_PROPS = {"covered": 45, "total": 50, "pct": 90.0}
GOOD_AUDIT = {"passed": True, "detail": "ok"}


def test_every_check_must_pass_and_the_thresholds_are_inclusive():
    assert gate.evaluate_week(GOOD_LINES, GOOD_PROPS, GOOD_AUDIT)["week_pass"] is True
    for lines, props, audit in [({"covered": 18, "total": 20, "pct": 90.0}, GOOD_PROPS, GOOD_AUDIT),
                                (GOOD_LINES, {"covered": 44, "total": 50, "pct": 88.0}, GOOD_AUDIT),
                                (GOOD_LINES, GOOD_PROPS, {"passed": False, "detail": "tampered"})]:
        assert gate.evaluate_week(lines, props, audit)["week_pass"] is False


def test_nothing_measured_is_a_fail_not_a_vacuous_pass():
    ev = gate.evaluate_week({"covered": 0, "total": 0, "pct": None}, GOOD_PROPS, GOOD_AUDIT)
    assert ev["week_pass"] is False and "nothing measured" in ev["checks"][0]["detail"]


def test_step_one_needs_two_consecutive_passing_weekends():
    assert gate.step_status(True, START, None) == {"streak": 1, "step1_pass": False}
    prev = {"week_pass": True, "window_end": START.isoformat(), "streak": 1}
    assert gate.step_status(True, START, prev) == {"streak": 2, "step1_pass": True}
    assert gate.step_status(False, START, prev) == {"streak": 0, "step1_pass": False}
    assert gate.step_status(True, START, {**prev, "week_pass": False}) == {"streak": 1, "step1_pass": False}
    gap = {**prev, "window_end": (START - timedelta(days=7)).isoformat()}                  # a skipped week breaks the streak
    assert gate.step_status(True, START, gap) == {"streak": 1, "step1_pass": False}


# ---- report round trip -----------------------------------------------------------------------------------

def _report(pass_week=True, streak=1):
    ev = gate.evaluate_week(GOOD_LINES if pass_week else {"covered": 1, "total": 20, "pct": 5.0}, GOOD_PROPS, GOOD_AUDIT)
    st = {"streak": streak, "step1_pass": streak >= 2}
    audit = {**GOOD_AUDIT, "state": {"n": 7, "md5": "m", "as_of": END.isoformat()}}
    return gate.render_report(window_start=START, window_end=END, evaluation=ev, status=st, audit=audit,
                              lines=GOOD_LINES, props=GOOD_PROPS,
                              summary=[{"source": "ev_prop", "n": 50, "mean_clv_prob": -0.021, "beat_close_pct": 0.31, "units": -2.5}])


def test_report_round_trips_through_its_machine_block():
    text = _report()
    assert text.startswith("# Pick ledger gate: weekend ending 2026-10-20")
    assert "| ev_prop | 50 | -2.10 pp | 31.0% | -2.50 |" in text
    p = gate.parse_report(text)
    assert p["week_pass"] is True and p["streak"] == 1 and p["window_end"] == END.isoformat()
    assert p["audit"] == {"n": 7, "md5": "m", "as_of": END.isoformat()}


def test_report_states_whether_pinnacle_was_requested_for_props():
    base = dict(window_start=START, window_end=END, evaluation=gate.evaluate_week(GOOD_LINES, GOOD_PROPS, GOOD_AUDIT),
                status={"streak": 1, "step1_pass": False}, audit={**GOOD_AUDIT, "state": {}}, lines=GOOD_LINES,
                props=GOOD_PROPS, summary=[])
    both = gate.render_report(**base, prop_regions={"us,eu": 12, "us": 3})
    assert "`us` x 3, `us,eu` x 12" in both and "Pinnacle (EU) prop prices were requested." in both
    us_only = gate.render_report(**base, prop_regions={"us": 15})
    assert "Pinnacle was NOT requested for props" in us_only
    assert "Prop closes captured this weekend: none." in gate.render_report(**base)


def test_report_verdict_text():
    assert "STEP 1 PASSED" in _report(streak=2)
    assert "one more consecutive passing weekend" in _report(streak=1)
    assert "steps 2 and 3 stay blocked" in _report(pass_week=False, streak=0)


def test_latest_previous_picks_the_nearest_earlier_report_and_ignores_junk():
    older = _report().replace(END.isoformat(), (END - timedelta(days=14)).isoformat())
    prior = _report()
    assert gate.latest_previous([older, "no block here", prior], END + timedelta(days=7))["window_end"] == END.isoformat()
    assert gate.latest_previous([prior], END) is None                                   # not strictly earlier
    assert gate.parse_report("<!-- ledger-gate: {not json} -->") is None


# ---- the script on a fake query function --------------------------------------------------------------

def fake_q(games=(), lines=(), props=(), summary=(), audit=(7, "m"), regions=()):
    seen = []

    def q(sql, params=()):
        seen.append(sql)
        if "FROM pick_ledger t" in sql:
            return [audit]
        if "kind = 'props'" in sql:
            return list(regions)
        if "DISTINCT sport, game_pk" in sql:
            return list(games)
        if "FROM close_capture" in sql:
            return list(lines)
        if "close_same_line FROM" in sql:
            return list(props)
        return list(summary)

    q.seen = seen
    return q


def test_collect_and_build_report_pass_then_fail_on_a_tampered_ledger():
    games = [("nfl", 1), ("cfb", 2)]
    q = fake_q(games=games, lines=[("nfl", 1, 14.0), ("cfb", 2, 9.0)],
               props=[("pinnacle_novig", True)] * 9 + [("missing", None)], audit=(7, "m"))
    data = rg.collect(q, START, END, None)
    text, res = rg.build_report(END, data, None)
    assert res["evaluation"]["week_pass"] is True and res["status"] == {"streak": 1, "step1_pass": False}
    assert "Prop closes captured this weekend: none." in text
    prev = gate.parse_report(text)
    q2 = fake_q(games=games, lines=[("nfl", 1, 14.0), ("cfb", 2, 9.0)], props=[("pinnacle_novig", True)] * 10, audit=(7, "CHANGED"))
    data2 = rg.collect(q2, END, END + timedelta(days=7), prev)
    _, res2 = rg.build_report(END + timedelta(days=7), data2, prev)
    assert res2["evaluation"]["week_pass"] is False
    assert [c["passed"] for c in res2["evaluation"]["checks"]] == [True, True, False]
    assert all("INSERT" not in s and "UPDATE" not in s and "DELETE" not in s for s in q.seen + q2.seen)


def test_second_clean_weekend_passes_step_one():
    q = fake_q(games=[("nfl", 1)], lines=[("nfl", 1, 14.0)], props=[("us_consensus_novig", True)] * 10)
    data = rg.collect(q, START, END, None)
    text, _ = rg.build_report(END, data, None)
    prev = gate.parse_report(text)
    nxt = END + timedelta(days=7)
    data2 = rg.collect(q, END, nxt, prev)
    _, res = rg.build_report(nxt, data2, prev)
    assert res["status"] == {"streak": 2, "step1_pass": True}


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert rg.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out
````

`tests/ledger/test_workflows_gate.py`:

````python
"""Structural checks of .github/workflows/report-ledger-gate.yml (weekly, commits only the report) and the rule that
no ledger workflow carries the test-suite guard."""
from tests.ledger.test_workflows_capture import cron_sets
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of


def test_gate_workflow_runs_weekly_on_tuesday_after_grading_and_commits_only_the_report():
    wf = load("report-ledger-gate.yml")
    (cron,) = [c["cron"] for c in wf["on"]["schedule"]]
    m, h, mon, dow = cron_sets(cron)
    assert dow == {2} and h == {15} and m == {47}                                       # Tuesday, after grade-ledger 14:10
    assert wf["permissions"] == {"contents": "write"} and "workflow_dispatch" in wf["on"]
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    gen = steps[step_index(steps, runs("scripts/report_ledger_gate.py"))]
    assert gen["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}
    commit = steps[step_index(steps, runs("git commit"))]
    assert "git add docs/superpowers/reports/\n" in commit["run"] and "[skip ci]" in commit["run"]
    assert step_index(steps, runs("scripts/report_ledger_gate.py")) < step_index(steps, runs("git commit"))
    assert "git add ." not in commit["run"] and "git add -A" not in commit["run"]
    assert wf["concurrency"]["group"] == "report-ledger-gate"


def test_no_ledger_workflow_sets_the_test_guard():
    for name in ("capture-close.yml", "grade-ledger.yml", "report-ledger-gate.yml", "probe-pinnacle-props.yml"):
        assert "SM_LEDGER_OFF" not in (WF / name).read_text(), name
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/ledger/test_gate.py tests/ledger/test_workflows_gate.py`
Expected: collection ERROR, `ImportError: cannot import name 'gate' from 'sportsmodel.ledger'`.

- [ ] **Step 3: Write the implementation**

`src/sportsmodel/ledger/gate.py`:

````python
"""The weekly step-1 gate for the pick ledger. PURE -- no IO.

A football weekend passes when ALL hold:
  * >= 95% of graded NFL / CFB games have a game-lines close captured within 30 minutes of kickoff,
  * >= 90% of graded prop picks have a Pinnacle or >= 3-book US close at their exact line,
  * no ledger row was updated or deleted (the row count and a content fingerprint of every row older than the
    previous report are unchanged since that report),
and step 1 passes only after TWO CONSECUTIVE passing weekends. A miss blocks steps 2-3 of the profit roadmap.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from .core import as_utc

LINES_MIN_PCT = 95.0
LINES_WITHIN_MIN = 30.0
PROPS_MIN_PCT = 90.0
WEEKENDS_REQUIRED = 2
MARKER = "ledger-gate"
FOOTBALL = ("nfl", "cfb")


def window_end_for(now: datetime) -> datetime:
    """The most recent Tuesday 12:00 UTC at or before `now` (the weekend's window closes Tuesday noon, after
    Monday night football has finished and been graded)."""
    now = as_utc(now)
    days_back = (now.isoweekday() - 2) % 7              # Tuesday = 2
    end = (now - timedelta(days=days_back)).replace(hour=12, minute=0, second=0, microsecond=0)
    return end if end <= now else end - timedelta(days=7)


def lines_coverage(graded_games: list[tuple], lines_minutes: dict[tuple, float | None]) -> dict:
    """Share of graded football games whose game-lines close was captured 0..30 minutes before kickoff."""
    games = {(s, pk) for s, pk in graded_games if s in FOOTBALL}
    covered = sum(1 for g in games if (m := lines_minutes.get(g)) is not None and 0 <= m <= LINES_WITHIN_MIN)
    return {"covered": covered, "total": len(games), "pct": 100.0 * covered / len(games) if games else None}


def props_coverage(prop_results: list[dict]) -> dict:
    """Share of graded prop picks with a Pinnacle or >= 3-book close at their exact line."""
    covered = sum(1 for r in prop_results if r.get("close_kind") in ("pinnacle_novig", "us_consensus_novig")
                  and r.get("close_same_line") is True)
    n = len(prop_results)
    return {"covered": covered, "total": n, "pct": 100.0 * covered / n if n else None}


def audit_check(previous: dict | None, recomputed: dict | None) -> dict:
    """Compare the previous report's audit block ({'n', 'md5', 'as_of'}) with the same measurement recomputed
    NOW over rows observed up to that same instant. Any difference means a row changed or vanished."""
    if previous is None:
        return {"passed": True, "detail": "baseline established (no previous report to compare with)"}
    rec = recomputed or {}
    if rec.get("n") == previous.get("n") and rec.get("md5") == previous.get("md5"):
        return {"passed": True, "detail": f"{previous['n']} rows up to {previous['as_of']} unchanged"}
    return {"passed": False, "detail": f"rows up to {previous.get('as_of')}: was {previous.get('n')} / "
                                       f"{previous.get('md5')}, now {rec.get('n')} / {rec.get('md5')}"}


def evaluate_week(lines: dict, props: dict, audit: dict) -> dict:
    checks = []
    for name, cov, floor, what in (
            ("game-line closes within 30 min of kickoff", lines, LINES_MIN_PCT, "graded NFL/CFB games"),
            ("prop closes at the exact line (Pinnacle or >= 3 books)", props, PROPS_MIN_PCT, "graded prop picks")):
        if cov["total"] == 0:
            checks.append({"name": name, "passed": False, "detail": f"no {what} in the window (nothing measured)"})
        else:
            checks.append({"name": name, "passed": cov["pct"] >= floor,
                           "detail": f"{cov['covered']}/{cov['total']} = {cov['pct']:.1f}% (need {floor:.0f}%)"})
    checks.append({"name": "no ledger row updated or deleted", "passed": audit["passed"], "detail": audit["detail"]})
    return {"checks": checks, "week_pass": all(c["passed"] for c in checks)}


def step_status(week_pass: bool, window_start: datetime, previous: dict | None) -> dict:
    """Step 1 passes when this weekend AND the immediately preceding weekend both passed."""
    prev_ok = bool(previous and previous.get("week_pass")
                   and as_utc(previous.get("window_end")) == as_utc(window_start))
    streak = (1 + (int(previous.get("streak", 1)) if prev_ok else 0)) if week_pass else 0
    return {"streak": streak, "step1_pass": streak >= WEEKENDS_REQUIRED}


def regions_line(prop_regions: dict[str, int] | None) -> str:
    """One sentence on which Odds API regions the prop closes used (the report states whether Pinnacle was asked)."""
    if not prop_regions:
        return "Prop closes captured this weekend: none."
    parts = ", ".join(f"`{r}` x {n}" for r, n in sorted(prop_regions.items()))
    pin = any("eu" in (r or "").split(",") for r in prop_regions)
    return (f"Prop closes captured this weekend by regions: {parts}. "
            + ("Pinnacle (EU) prop prices were requested." if pin else
               "Pinnacle was NOT requested for props (US books only; the probe found no Pinnacle props)."))


def render_report(*, window_start: datetime, window_end: datetime, evaluation: dict, status: dict,
                  audit: dict, lines: dict, props: dict, summary: list[dict],
                  prop_regions: dict[str, int] | None = None) -> str:
    ws, we = as_utc(window_start), as_utc(window_end)
    lines_md = "\n".join(f"| {c['name']} | {'PASS' if c['passed'] else 'FAIL'} | {c['detail']} |"
                         for c in evaluation["checks"])
    sums = "\n".join(
        f"| {r['source']} | {r['n']} | {_fmt(r.get('mean_clv_prob'), 100, '{:+.2f} pp')} | "
        f"{_fmt(r.get('beat_close_pct'), 100, '{:.1f}%')} | {_fmt(r.get('units'), 1, '{:+.2f}')} |" for r in summary
    ) or "| (none) | 0 | | | |"
    verdict = ("STEP 1 PASSED: two consecutive football weekends met every check. Steps 2 and 3 are unblocked."
               if status["step1_pass"] else
               "Step 1 is NOT yet passed. " + ("This weekend met every check; one more consecutive passing weekend is needed."
                                                if evaluation["week_pass"] else "A check failed; steps 2 and 3 stay blocked."))
    block = {"window_start": ws.isoformat(), "window_end": we.isoformat(), "week_pass": evaluation["week_pass"],
             "streak": status["streak"], "step1_pass": status["step1_pass"], "audit": audit["state"],
             "lines": lines, "props": props}
    return f"""# Pick ledger gate: weekend ending {we:%Y-%m-%d}

Window: {ws:%Y-%m-%d %H:%M} to {we:%Y-%m-%d %H:%M} UTC. Generated by `scripts/report_ledger_gate.py`.

**{verdict}**

| Check | Result | Detail |
|---|---|---|
{lines_md}

Consecutive passing weekends: {status['streak']} of {WEEKENDS_REQUIRED} required.

{regions_line(prop_regions)}

## First-observation results in the window (informational, not part of the gate)

| Source | Picks graded | Mean CLV vs close (same line) | Beat the close | Units at the price taken |
|---|---|---|---|---|
{sums}

<!-- {MARKER}: {json.dumps(block, sort_keys=True)} -->
"""


def _fmt(v, scale, fmt) -> str:
    return "n/a" if v is None else fmt.format(float(v) * scale)


_BLOCK = re.compile(rf"<!-- {MARKER}: (\{{.*\}}) -->")


def parse_report(text: str) -> dict | None:
    """The machine-readable block of a previous report, or None."""
    m = _BLOCK.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def latest_previous(reports: list[str], window_end: datetime) -> dict | None:
    """Of the parsed reports, the one with the greatest window_end strictly before `window_end`."""
    best = None
    for text in reports:
        p = parse_report(text)
        if not p or not p.get("window_end"):
            continue
        end = as_utc(p["window_end"])
        if end is not None and end < as_utc(window_end) and (best is None or end > as_utc(best["window_end"])):
            best = p
    return best
````

`scripts/report_ledger_gate.py`:

````python
"""Weekly step-1 gate report for the pick ledger -> docs/superpowers/reports/YYYY-MM-DD-ledger-gate.md.

Checks the football weekend that ended last Tuesday 12:00 UTC (see sportsmodel.ledger.gate): >= 95% of graded
games with a game-lines close captured within 30 minutes of kickoff, >= 90% of graded prop picks with a Pinnacle
or >= 3-book close at their exact line, and no ledger row updated or deleted (row count + content fingerprint of
everything older than the previous report); two consecutive passing weekends pass step 1. Read-only against the
database; writes only the markdown file (the workflow commits it).

    uv run python scripts/report_ledger_gate.py [--end 2026-10-20T12:00:00Z] [--reports-dir docs/superpowers/reports]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger import gate  # noqa: E402
from sportsmodel.ledger.core import as_utc  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"
AUDIT_SQL = ("SELECT count(*), md5(string_agg(md5(t::text), '' ORDER BY t.id)) FROM pick_ledger t "
             "WHERE t.observed_at <= %s")


def _db_query():
    from sportsmodel import db

    def q(sql, params=()):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    return q


def audit_state(q, as_of: datetime) -> dict:
    n, md5 = q(AUDIT_SQL, (as_of,))[0]
    return {"n": int(n), "md5": md5, "as_of": as_of.isoformat()}


def collect(q, start: datetime, end: datetime, previous: dict | None) -> dict:
    """All the numbers the report needs, via read-only queries `q(sql, params) -> rows`."""
    games = [(s, pk) for s, pk in q(
        "SELECT DISTINCT sport, game_pk FROM pick_ledger_results WHERE basis = 'first' "
        "AND sport IN ('nfl', 'cfb') AND commence_time >= %s AND commence_time < %s", (start, end))]
    lines_minutes: dict = {}
    if games:
        for s, pk, m in q("SELECT sport, game_pk, min(minutes_to_kick) FROM close_capture WHERE kind = 'lines' "
                          "AND status = 'ok' AND game_pk = ANY(%s) GROUP BY sport, game_pk", ([pk for _, pk in games],)):
            lines_minutes[(s, pk)] = m
    props = [{"close_kind": k, "close_same_line": same} for k, same in q(
        "SELECT close_kind, close_same_line FROM pick_ledger_results WHERE basis = 'first' AND source = 'ev_prop' "
        "AND commence_time >= %s AND commence_time < %s", (start, end))]
    summary = [{"source": s, "n": n, "mean_clv_prob": c, "beat_close_pct": b, "units": u} for s, n, c, b, u in q(
        "SELECT source, count(*), avg(clv_prob), avg(beat_close::int), sum(units_at_pick) FROM pick_ledger_results "
        "WHERE basis = 'first' AND commence_time >= %s AND commence_time < %s GROUP BY source ORDER BY source",
        (start, end))]
    prop_regions = {r: n for r, n in q(
        "SELECT regions, count(*) FROM close_capture WHERE kind = 'props' AND status IN ('ok', 'empty') "
        "AND captured_at >= %s AND captured_at < %s GROUP BY regions", (start, end))}
    recomputed = audit_state(q, as_utc(previous["audit"]["as_of"])) if previous and previous.get("audit") else None
    return {"games": games, "lines_minutes": lines_minutes, "props": props, "summary": summary,
            "prop_regions": prop_regions, "recomputed_audit": recomputed, "current_audit": audit_state(q, end)}


def build_report(end: datetime, data: dict, previous: dict | None) -> tuple[str, dict]:
    start = end - timedelta(days=7)
    lines = gate.lines_coverage(data["games"], data["lines_minutes"])
    props = gate.props_coverage(data["props"])
    prev_audit = previous["audit"] if previous and previous.get("audit") else None
    audit = gate.audit_check(prev_audit, data["recomputed_audit"])
    audit["state"] = data["current_audit"]
    evaluation = gate.evaluate_week(lines, props, audit)
    status = gate.step_status(evaluation["week_pass"], start, previous)
    text = gate.render_report(window_start=start, window_end=end, evaluation=evaluation, status=status,
                              audit=audit, lines=lines, props=props, summary=data["summary"],
                              prop_regions=data["prop_regions"])
    return text, {"evaluation": evaluation, "status": status}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--end", default=None, help="window end (ISO, default: the most recent Tuesday 12:00 UTC)")
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the report instead of writing the file")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("report-ledger-gate: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::report-ledger-gate: DATABASE_URL is not set")
        return 1
    from sportsmodel import db
    if db.missing_ledger_tables():
        print("::warning::report-ledger-gate: ledger tables missing — run db/migration_pick_ledger.sql", flush=True)
        return 0
    end = as_utc(args.end) if args.end else gate.window_end_for(datetime.now(timezone.utc))
    reports_dir = Path(args.reports_dir)
    previous = gate.latest_previous([p.read_text() for p in sorted(reports_dir.glob("*-ledger-gate.md"))], end)
    text, result = build_report(end, collect(_db_query(), end - timedelta(days=7), end, previous), previous)
    if args.dry_run:
        print(text)
        return 0
    path = reports_dir / f"{end:%Y-%m-%d}-ledger-gate.md"
    path.write_text(text)
    print(f"[ledger-gate] wrote {path}: week_pass={result['evaluation']['week_pass']} "
          f"step1_pass={result['status']['step1_pass']}", flush=True)
    for c in result["evaluation"]["checks"]:
        if not c["passed"]:
            print(f"::warning::ledger-gate FAIL: {c['name']}: {c['detail']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

`.github/workflows/report-ledger-gate.yml`:

````yaml
name: report-ledger-gate

# Weekly step-1 gate for the pick ledger: every Tuesday after Monday-night football has finished and been graded,
# check the weekend that just ended (scripts/report_ledger_gate.py) and commit docs/superpowers/reports/
# YYYY-MM-DD-ledger-gate.md. Passing needs, for two consecutive football weekends: >= 95% of graded games with a
# game-lines close captured within 30 minutes of kickoff, >= 90% of graded prop picks with a Pinnacle or >= 3-book
# close at their exact line, and no ledger row updated or deleted. A miss blocks steps 2-3 of the profit roadmap.
# Read-only against the database; writes only the markdown report. Tuesday 15:47 UTC, after grade-ledger (14:10).
on:
  workflow_dispatch: {}
  schedule:
    - cron: "47 15 * * 2"

permissions:
  contents: write

concurrency:
  group: report-ledger-gate
  cancel-in-progress: false

jobs:
  report:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
      - name: Sync deps
        run: uv sync
      - name: Write the gate report
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/report_ledger_gate.py
      - name: Commit the report
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/superpowers/reports/
          if git diff --staged --quiet; then
            echo "no new report"
            exit 0
          fi
          git commit -m "docs(report): pick ledger gate [skip ci]"
          # other data workflows push to the same branch: rebase onto theirs and retry
          for attempt in 1 2 3; do
            if git push; then
              exit 0
            fi
            echo "push rejected (attempt $attempt)"
            [ "$attempt" -lt 3 ] && git pull --rebase
          done
          echo "push failed after 3 attempts" >&2
          exit 1
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/ledger tests/test_workflows_props_ml.py`
Expected: all PASS (`tests/ledger` alone: 260 passed).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/ledger/gate.py scripts/report_ledger_gate.py .github/workflows/report-ledger-gate.yml tests/ledger/test_gate.py tests/ledger/test_workflows_gate.py
git commit -m "feat(ledger): weekly step-1 gate report"
```

---

### Task 9: Final verification

No new code. Evidence before claiming done (superpowers:verification-before-completion).

**Files:** none modified (a failing check sends you back to the owning task).

- [ ] **Step 1: Full suite**

Run: `uv run pytest -q` (about 4 minutes)
Expected: `2866 passed` (the 2602 baseline, the 260 in `tests/ledger/`, and 4 new `test_every_workflow_parses[...]` cases). No test may skip because of the ledger.

- [ ] **Step 2: Only the intended files changed**

Run: `git diff --stat main...HEAD`
Expected: exactly the files in the File Structure table (new: the migration, `src/sportsmodel/ledger/*`, `scripts/{capture_close,grade_ledger,report_ledger_gate,probe_pinnacle_props}.py`, four workflows, `tests/ledger/*`, this plan and the spec; modified: `db.py`, `ingest/odds.py`, `conftest.py` and the six writer scripts). Nothing under `site/`, `data/`, `assets/`, no model or board module, no existing workflow.

- [ ] **Step 3: Safety greps (no stray database writes, no leaked guard)**

```bash
# the only INSERT/UPDATE/DELETE statements added to Python live in db.py's helpers
grep -rnE "INSERT INTO|UPDATE |DELETE FROM" src/sportsmodel/ledger scripts/capture_close.py scripts/grade_ledger.py scripts/report_ledger_gate.py scripts/probe_pinnacle_props.py || echo "no writes outside db.py: OK"
# the test guard must never appear in a workflow
grep -rn "SM_LEDGER_OFF" .github || echo "guard not in workflows: OK"
# no new Python creates, alters or drops anything: schema comes only from the migration file
grep -nE "CREATE TABLE|ALTER TABLE|DROP |TRUNCATE" src/sportsmodel/ledger/*.py scripts/capture_close.py scripts/grade_ledger.py scripts/report_ledger_gate.py scripts/probe_pinnacle_props.py || echo "no DDL in code: OK"
git status --short
```
Expected: the three `OK` lines print (`no writes outside db.py`, `guard not in workflows`, `no DDL in code`), and `git status` is clean.

- [ ] **Step 4: Dry plan of the user's go-live (read it back, do not execute any of it)**

Confirm the "Handoff to the user" section below matches what was built (names of the repository variable, workflows and scripts), and that the PR description will carry: the probe verdict from Task 4, the credit table, and the list of resolved ambiguities.

- [ ] **Step 5: Hand back**

Do NOT run the migration, merge, or dispatch any workflow other than the Task 4 probe. Report the branch, the commits, the verdict of the probe, and the test counts, then follow superpowers:finishing-a-development-branch.

---

## Handoff to the user (after review; the executing agent does none of this)

1. **Run the migration.** Paste `db/migration_pick_ledger.sql` into the Supabase SQL editor and run it once (safe to re-run). Until you do, every ledger job prints a `::warning::` and exits 0, and the writers carry on exactly as today.
2. **Check the append-only guarantee** (optional, inside a transaction that rolls back; this is for you to run, not the agent):
   ```sql
   BEGIN;
   INSERT INTO pick_ledger (pick_key, source, sport, game_pk, market, side, observed_at, commence_time,
                            minutes_to_kick, writer, writer_run)
   VALUES ('t', 'ev_game', 'nfl', 1, 'spread', 'home', now(), now() + interval '1 day', 1440, 'manual', 'local');
   UPDATE pick_ledger SET best_price = 100;   -- must fail: pick_ledger is append-only: UPDATE is not allowed
   ROLLBACK;
   ```
3. **Merge the PR.** The schedules begin on merge: `capture-close` (every 10 minutes in game-day hours), `grade-ledger` (daily 14:10 UTC), `report-ledger-gate` (Tuesday 15:47 UTC). Existing workflows are unchanged.
4. **Props regions.** If the probe said `pinnacle_absent`, confirm `CLOSE_PROP_REGIONS` is `us` (Settings, Secrets and variables, Actions, Variables). Otherwise leave it unset.
5. **Watch the first football weekend.** Expect, per game, a `close_capture` row `kind = lines` with `minutes_to_kick` between 5 and 25; props rows for games with prop picks; `pick_ledger` rows from the next board builds. The first gate report appears the Tuesday after (the first run only establishes the audit baseline); step 1 passes after two consecutive weekends that meet every check.
6. **Credits.** Read `x-requests-remaining` in the run logs (`[capture-close] {...}` lines carry the credits per run). The guard stops props below 3,000 and everything below 1,000.

## Self-Review

Run against the spec with fresh eyes after writing the plan.

**1. Spec coverage**

| Spec requirement | Task / evidence |
|---|---|
| `pick_ledger` columns (identity, model, market, timing, provenance) | Task 1 migration; `LEDGER_COLUMNS` agreement test |
| Append rule (first, price, line, >= 1.0 pp) | Task 2 `plan_appends`, boundary tests (`0.60 - 0.59`) |
| Append-only: trigger + revokes + non-owner roles | Task 1 migration tests (trigger for UPDATE/DELETE/TRUNCATE, `REVOKE ... FROM PUBLIC`, `service_role`, anon/authenticated) |
| Best-effort writes, never fail the producer | Task 2 `safe_record`; Task 3 hook tests |
| Writers: generate/board/props/parlay scripts; model game picks use `prediction_accuracy` side logic | Task 3 (six hooks; parity test with `_accuracy_row`); decision 2 explains why `grade_predictions` is not hooked |
| Indexes `(sport, game_pk)`, `(pick_key, observed_at)` | Task 1 migration (second is UNIQUE, decision 10) |
| `capture_close.py` + `capture-close.yml`, 10-minute cadence in game windows, idle exit before any API call | Task 5 (cron coverage tests, `test_idle_run_makes_no_network_call_at_all`) |
| Select events kicking off soon, uncaptured; group by sport; one lines call per sport; one props call per event with prop picks | Task 5 `_capture_sport`; decision 3 changes 35 to 25 minutes, decision 9 the two-stage idle check |
| Store in `odds_snapshot`; `close_capture` unique on (`sport`, `game_pk`, `kind`); failed retried | Task 5; Task 1 `upsert_close_capture` (only `failed` superseded) |
| Pinnacle props probe first, then regions=us if none | Task 4 (probe) + `CLOSE_PROP_REGIONS` read in Task 5 (`close.prop_regions`) |
| Budget guard 3,000 / 1,000, credits logged | Task 5 `budget_decision`, summary line, tests at 3000/2999/1000/999 |
| `grade_ledger.py` nightly -> `pick_ledger_results`, first and last row, close reference order, metrics | Tasks 6-7 (decision 4 for the two rows per key) |
| Weekly gate: 95% within 30 min, 90% props, 0 updated/deleted, 2 consecutive weekends, markdown committed by the workflow | Task 8 (decisions 6 and 12) |
| Migration idempotent/additive, RLS, jobs warn and exit 0 until tables exist | Task 1 tests; guards in Tasks 2, 5, 7, 8 |
| Tests listed in the spec (pick_key determinism, append rule, failure isolation, window selection, already-captured skip, retry, idle, budget thresholds, probe handling, reference selection, CLV maths, gate logic, workflow schedules) | Tasks 2, 3, 4, 5, 6, 8 respectively |
| Credits: about 360/week added | Credit section above and the `capture-close.yml` header |

No gaps found. Spec items deliberately reinterpreted are listed in the decisions table, each with its reason.

**2. Placeholder scan.** No `TBD`/`TODO`/"implement later"/"similar to Task N" in the plan; every code step carries the code. The only parameters left to the operator are the repository variable `CLOSE_PROP_REGIONS` (decided from the probe) and the Co-Authored-By trailer.

**3. Type consistency.** Names checked across tasks: `db.LEDGER_COLUMNS` (Task 1) is what `close.capture_row` (Task 5) and `grading.grade_pick_row` (Task 7) are tested against (`set(row) == set(columns)`); `core.observation` keys match `LEDGER_COLUMNS["pick_ledger"]` plus the stamped `observed_at`/`minutes_to_kick`/`writer`/`writer_run`; `ledger` market codes are the writers' sim codes (`rec_yds`) and `reference.odds_market` maps them to `odds_snapshot` codes (`reception_yds`) with the same maps `close.api_markets_for` uses; `odds.last_requests_last` (Task 4) is what `capture_close._call_cost` reads; `record.writes_disabled` (Task 2) is what the three script `main()`s call; `tests.ledger.test_workflows_capture.cron_sets` (Task 5) is imported by Task 8's tests.

**4. Review Focus.** Each of the five lines names its pinning tests; all exist in the tasks listed.

**Known limits, accepted:** postponed or cancelled games never grade; a Thursday-night kickoff moved earlier than the database knows (a flexed game announced after the generators ran) can miss the 60-minute prefilter until the next generator run; MLB regular-season props are not budgeted here (spec: about 300-400 credits a week more).
