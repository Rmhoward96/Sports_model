# Early-Week Stale-Number Monitor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Log paper tickets, never edited, whenever one of the user's nine US sportsbooks hangs a CFB total, CFB spread or NFL spread whose value against Pinnacle's same-poll fair probability is at least +1.5% EV with kickoff at least 24 hours away; grade every ticket against Pinnacle's pre-kick close; and judge the idea once, against a pass test frozen before any result exists (profit roadmap, step 3). No money is at risk and no alert is sent.

**Architecture:** A new pure package `src/sportsmodel/early/` holds everything that has a rule in it: the committed landing tables and their loader, the line adjustment and EV, the flagging engine (one poll's prices to new tickets), polling cost and the credit floor, ticket grading, the frozen pass test and the weekly report. Three small IO scripts wrap it (`early_monitor.py` hourly, `grade_early_tickets.py` daily, `report_early_tickets.py` weekly) plus a build-once script for the landing tables. One idempotent migration, which the USER runs, adds two tables. `capture_close.py` gains one candidate source so every ticketed game gets a pre-kick line capture. The monitor writes only `odds_snapshot`, `early_tickets` and `early_ticket_results`.

**Tech Stack:** Python 3.11+, psycopg 3 (Supabase Postgres), httpx + tenacity (The Odds API client already in `ingest/odds.py`), pandas (the build-once script only), numpy (the bootstrap), GitHub Actions, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-13-early-week-monitor-design.md` (the authority; read it with this plan). Evidence: `docs/superpowers/reports/2026-10-13-edge-audit.md` sections 5 and 6 item 3. Depends on step 1 (`docs/superpowers/specs/2026-10-13-pick-ledger-close-capture-design.md`, merged 2d16461) for the pre-kick close capture, the append-only pattern, the `SM_LEDGER_OFF` guard and the settlement and CLV helpers in `sportsmodel.ledger`.

**Baseline before this plan:** `uv run pytest -q` on `feat/early-monitor` at `25f938e` = 3002 passed (about 4 minutes). The new tests add 215 under `tests/early/` plus 3 `test_every_workflow_parses` cases (one per new workflow), so the final full run must show 3220 passed.

## Global Constraints

Every task's requirements include this section. Values are copied from the spec.

- **Scope:** out of scope are "Alerts and notifications.", "Any change to models, boards, picks or the site.", "NFL totals, moneylines and props." and "Steps 2 (probability shrink) and any bet sizing." The system "logs paper tickets, never edits them, grades them against the close, and judges them once against a pass test that was fixed before any results came in."
- **Markets in the test:** "CFB totals, CFB spreads and NFL spreads". "NFL totals and all moneylines are excluded."
- **Reference price:** "Pinnacle's two-way no-vig fair probability at Pinnacle's current line, from the **same poll**." "No Pinnacle two-way quote for a game and market in that poll means no tickets for that game and market from that poll." "Pinnacle fair outside [0.15, 0.85] means no ticket."
- **Landing tables:** committed JSON under `assets/early/`, built once by `scripts/build_landing_tables.py`. CFB: `assets/cfb/lines.parquet` closing `market_spread` / `market_total`, 2015-2025, joined to final scores from `assets/cfb/schedules.parquet`. NFL: `assets/nfl/schedules.parquet` `spread_line` / `total_line` and scores, 2010-2025. Spreads use the home margin M, totals the total points T. For each line value L on a half-point grid the empirical probability of each integer result among games whose closing line is within +/-1.0 point of L, "widening to +/-2.0 if that gives fewer than 300 games".
- **Line adjustment (Pinnacle's Lp to the book's Lb, one side):** take the table at Lp; P_win(Lp) = fair x (1 - P_push(Lp)); move to Lb by adding or removing the table mass of every result value between the two lines (the mass at Lp's push value becomes win or loss by the direction of the move, the mass at Lb's push value becomes P_push(Lb)); that gives P_win(Lb), P_push(Lb), P_loss(Lb). "If |Lb - Lp| > 3.0 points, there is no ticket".
- **EV and flag:** EV = P_win x (decimal - 1) - P_loss, a push contributes 0. Flag: EV >= +0.015 (+1.5%) and kickoff >= 24 h after the poll. Several qualifying books for one side: the highest EV; ties broken by book name.
- **Tickets:** "One ticket per (sport, game_pk, market, side)", logged the first time it is flagged; later sightings never create a ticket. Stored: sport, game_pk, market, side; book, book line, American price; Pinnacle line, Pinnacle fair, P_win, P_push, P_loss and EV; observed_at, commence_time, hours_to_kick; poll run id. Game matching uses the existing odds-event to game_pk matchers of `scripts/ingest_odds.py`.
- **Polling:** workflow `early-monitor.yml`, hourly at minute 0 from Sunday 16:00 UTC to Wednesday 04:00 UTC, 60 polls a week, crons `0 16-23 * 1,2,8-12 0`, `0 * * 1,2,8-12 1,2` and `0 0-3 * 1,2,8-12 3`, plus `workflow_dispatch` with a `dry_run` input passed through `env`. One Odds API odds call per sport using the `bookmakers` parameter, not `regions`: the 9 US books plus `pinnacle` are 10 books and bill as one region. CFB `spreads,totals` = 2 credits, NFL `spreads` = 1 credit, about 3 credits a poll, about 180 a week. The 9 books: DraftKings, FanDuel, BetMGM, Caesars, ESPN BET, Fanatics, Hard Rock, Bally, BetRivers.
- **Credit floor:** the monitor "skips its remaining calls when `x-requests-remaining` < 1,500, and warns. The value is unknown before the first response, so the first call always runs." Ordering as credits fall: < 2,500 stops regular prop pulls; < 1,500 this monitor; < 800 regular game-line pulls; < 600 close-capture props; < 200 all close capture.
- **Storage and writes:** each poll's prices go to the existing `odds_snapshot` table "in the same row shape `ingest_odds` writes"; tickets are written incrementally, after each sport.
- **Failure handling:** "A failed poll is not retried." Missing tables: warn and exit 0 (the missing-table guard of the ledger jobs). Under `SM_LEDGER_OFF` the job warns and exits 0, dry-run included. It writes only `odds_snapshot`, `early_tickets` and `early_ticket_results`; "It never touches boards, picks, models or the site."
- **Close coverage:** the close-capture candidate query in `scripts/capture_close.py` adds `early_tickets` to its sources.
- **Grading:** `scripts/grade_early_tickets.py`, workflow `grade-early-tickets.yml` daily at 14:20 UTC, upserts `early_ticket_results` (one row per ticket; not append-only). Close = Pinnacle's two-way no-vig fair from the `close_capture` lines snapshot (5-25 min pre-kick); absent means `close_kind = 'missing'`, excluded from the test, share reported. Metrics: `close_ev` (primary: the ticket's book line and price at Pinnacle's closing fair through the same landing-table adjustment and EV formula), `beat_close` = `close_ev` > 0, `clv_pts`, `result` (W/L/P) from finals keyed by (sport, game_pk) "using the same final providers as `grade_ledger`", `units`, `available_next_poll` (same book, same line, same or better price in the next poll, "30 min to 3 h after the ticket; otherwise NULL").
- **Pass test (frozen):** `scripts/report_early_tickets.py`, workflow `report-early-tickets.yml` Tuesday 15:52 UTC, commits a markdown report to `docs/superpowers/reports/`. Judged once, at the first weekly report with >= 150 graded tickets with a Pinnacle close across the three markets and >= 60 of them CFB totals; before that "collecting". CFB spreads and NFL spreads need >= 40 graded tickets, else "insufficient". A market passes only if: (1) mean `close_ev` >= +0.015; (2) game-clustered 95% lower bound of mean `close_ev` > 0, cluster bootstrap over games, 10,000 resamples, fixed seed 20261013, 2.5th percentile; (3) `beat_close` rate >= 55%; (4) mean `close_ev` over tickets with `available_next_poll = true` > 0. "Realized ROI is reported, never gated". A passing market goes to a separate short design for live alerts at small stakes (a later profit check, >= 300 tickets and ROI 95% lower bound > -3%, decides any stake increase); a failing market is dropped.
- **Migration:** `db/migration_early_tickets.sql`, idempotent and additive only. `early_tickets`: bigserial id, unique (sport, game_pk, market, side), append-only trigger on UPDATE, DELETE and TRUNCATE with `SET search_path = ''`. `early_ticket_results`: PK ticket_id. RLS with anon SELECT only; writes revoked from anon, authenticated, PUBLIC and service_role on `early_tickets`, as in `db/migration_pick_ledger.sql`.
- **Layout:** pure logic in the new package `src/sportsmodel/early/`; tests in `tests/early/`.
- **No database or network access from the executing agent.** The migration is a file the USER runs in the Supabase SQL editor. Only the merged, scheduled jobs write to the database or call the Odds API. The local `.env` holds a real `DATABASE_URL`: never run `early_monitor.py`, `grade_early_tickets.py` or `report_early_tickets.py` locally (they refuse under `SM_LEDGER_OFF`, which the root `conftest.py` sets for every test; never unset it in a shell that has `DATABASE_URL`). Every test uses injected fakes. The only commands this plan runs are `pytest`, `git`, and `scripts/build_landing_tables.py` (Task 2), which reads committed parquet files and touches neither the network nor the database.
- **Conventions:** follow the repo's patterns (script `main()` plus importlib-loaded script tests, FakeConn-style database tests, the block-YAML workflow reader in `tests/test_workflows_props_ml.py`). Project convention: implementers and reviewers run on Sonnet (pass `model: "sonnet"` explicitly).
- **Commits:** every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Every commit command below passes it as a second `-m`.

## Review Focus

The input classes and failure modes the spec implies but does not spell out, most likely to bite first. Each has a pinning test in the task that owns the code.

1. **Quarter-point and other off-grid lines.** Pinnacle and the consensus closes in the history quote lines such as 54.75; a half-point landing table cannot value them. A line off the half-point grid is never rounded: no ticket at flag time, `close_kind = 'missing'` at grading. Task 3: `test_off_grid_lines_are_refused_never_rounded`; Task 4: `test_quarter_point_lines_are_never_rounded_into_a_ticket`; Task 7: `test_a_quarter_point_closing_line_is_missing_not_rounded`.
2. **Lines the history cannot speak for.** A 38-point CFB favourite or a 90-point total has too few historical games (a grid value still under 300 games at +/-2.0 is left out of the table), and a book more than 3 points from Pinnacle is a data error. Behaviour: no ticket. Task 2: `test_grid_values_still_short_at_two_points_are_left_out`; Task 3: `test_three_point_cap_is_inclusive`; Task 4: `test_a_line_outside_the_committed_landing_tables_makes_no_ticket`, `test_a_book_more_than_three_points_from_pinnacle_is_a_data_error_not_a_ticket`.
3. **More than one quote for a market.** Pinnacle can show a main line and alternates; a book can post two lines for one side. Behaviour: Pinnacle's reference is the pair closest to even; every book offer is valued at its own line and the best one wins, deterministically. Task 4: `test_pinnacle_with_several_quoted_pairs_uses_the_one_closest_to_even`, `test_a_book_posting_alternate_lines_is_valued_at_each_line_and_the_best_one_wins`, `test_output_is_deterministic_and_ordered_by_game`.
4. **A close that moved a long way, or never came.** A total that falls five points after the ticket must count against the ticket, not drop it from the sample (that would flatter the test); a game with no Pinnacle two-way at the close is excluded and its share reported; a game with no final stays ungraded and is counted. Task 3: `test_a_move_bigger_than_the_available_probability_is_refused_strictly_and_clamped_when_grading`; Task 7: `test_a_close_that_moved_more_than_three_points_still_counts_and_counts_against_the_ticket`, `test_no_pinnacle_quote_at_the_close_is_missing_and_never_beats_it`, `test_a_game_without_a_result_stays_ungraded`; Task 9: `test_summary_counts_tickets_by_market_and_week_and_ungraded`.
5. **Duplicated, repeated and half-failed runs.** GitHub delays, skips and occasionally duplicates cron firings, and a database hiccup can land after the credits are spent. Behaviour: a second run in the same hour logs nothing new (one ticket per side; the insert is `ON CONFLICT DO NOTHING`); a failed poll is not retried and does not stop the other sport; a snapshot-write failure never stops the tickets; free reads happen before credits are spent. Task 1: `test_insert_early_tickets_is_insert_only_idempotent_and_ordered_by_column`; Task 5: `test_a_second_run_in_the_same_hour_logs_no_new_ticket`, `test_a_failed_poll_is_not_retried_and_the_other_sport_goes_on`, `test_a_snapshot_write_failure_does_not_stop_the_tickets`, `test_the_schedule_and_existing_ticket_reads_happen_before_credits_are_spent`.

Also pinned, as in step 1: tests and local runs cannot reach the production database through `.env` (each script's `test_main_refuses_under_the_test_ledger_guard`, Tasks 5, 7, 9), and the monitor can never starve the close capture (`test_what_each_job_may_still_spend_at_each_balance`, Task 10).

## Decisions on spec ambiguities (resolved here; the plan follows these)

| # | Spec text | Resolution | Why |
|---|---|---|---|
| 1 | "The 9 US books plus `pinnacle` make 10 books" (keys not listed) | Keys: `draftkings`, `fanduel`, `betmgm`, `williamhill_us`, `espnbet`, `fanatics`, `hardrockbet`, `ballybet`, `betrivers`, `pinnacle`. Caesars is `williamhill_us`, the key The Odds API uses; the existing `MAJOR_BOOKS` map lists both `williamhill_us` and `caesars` and has no `betrivers`, so the monitor keeps its own list in `early/constants.py` | The `bookmakers` parameter needs exact keys. A bookmaker key the API does not know is at best ignored, so `caesars` is not requested |
| 2 | Landing tables key on "each line value L"; CFB `market_spread` and NFL `spread_line` signs are not stated | Both sources are positive when the HOME team is favoured (verified: correlation with the home margin is +0.67 for CFB and +0.43 for NFL). Tables are keyed by the **expected home margin**, which is minus the home team's betting line, so a table line and a "threshold" are the same thing for spreads and totals (`early/adjust.py` converts: home spread -2.5 = threshold 2.5, away spread +2.5 = threshold 2.5, total 54.5 = threshold 54.5) | One operation moves any side of any market from Pinnacle's number to a book's; no per-market sign branches in the maths |
| 3 | "`lines.parquet` joined to final scores from `schedules.parquet`" | Join on `(season, week, home_team, away_team)`: unique on both sides, 7,882 of 7,902 line rows match (the 20 left have no score), 15 rows have no closing total. Seasons 2015-2025 for CFB, 2010-2025 for NFL (all game types) | These are the only columns the two files share; checked with `uv run python` on the committed parquet |
| 4 | "within +/-1.0 point of L ... widening to +/-2.0 if that gives fewer than 300 games"; silent when ±2.0 is still short | A grid value still under 300 games at +/-2.0 is **omitted**; a lookup there answers `None` and the pair gets no ticket. Result on today's data (expected home margin / total): CFB spreads -16 to +26, CFB totals 41 to 70, NFL spreads -8.5 to +12 | A table built from a few dozen games would invent key-number mass. Conservative: fewer tickets, never a made-up one |
| 5 | "Pinnacle's current line" can be a quarter point | Lines off the half-point grid are never rounded: no ticket at flag time, and `close_kind = 'missing'` at grading | The model has win / push / loss on whole and half lines only; quarter lines are split bets |
| 6 | "mass at Lb's push value becomes P_push(Lb)" (which table?) | One table throughout: the one at Lp. Mass is moved between win, push and loss by re-classifying each integer result whose outcome changes between the two thresholds, so P_win + P_push + P_loss stays exactly 1. If a component would go negative the pair gets no ticket (the table says the move is bigger than the probability available) | A single distribution is internally consistent and testable |
| 7 | "`close_ev` ... through the same landing-table adjustment and EV formula as section 1", and the 3.0-point cap is part of that adjustment | At grading the cap is **off** and a negative component is clamped and renormalised: the cap is a flag-time data-error guard, and applying it at the close would drop exactly the tickets whose number moved a long way against us, flattering the test | An honest test counts adverse moves as losses. Everything else (table, fair, formula) is the same function |
| 8 | Pinnacle shows more than one pair for a game and market | Use the pair whose fair is closest to 50% (the main line); a remaining tie goes to the lower threshold | The Odds API returns the main line for `spreads` / `totals`, but alternates must not be able to change the answer |
| 9 | "The value is unknown before the first response, so the first call always runs" | Kept as the fallback. In addition a **free** `/events` call (which returns `x-requests-remaining` and costs no credits, as `capture_close.py` already relies on) runs first, so even the first paid call of an hourly run is guarded; if that call fails or has no header the first paid call runs | Without it every hourly run would spend 2 credits below the floor before learning the balance (about 120 credits a week) |
| 10 | `workflow_dispatch` with a `dry_run` input (behaviour not stated) | A dry run makes the paid odds calls (about 3 credits) and prints the tickets it would log, but writes nothing; `SM_LEDGER_OFF` refuses it, as the spec says | A dry run that polls nothing could not show whether the books and Pinnacle line up |
| 11 | `available_next_poll`: "the next monitor poll ... computed from `odds_snapshot`" | `odds_snapshot` has no source column, so monitor polls cannot be told from the regular pulls. The next poll is the **first snapshot at least 30 minutes after the ticket that carries a Pinnacle quote for that market**; if it is more than 3 hours after the ticket the value is NULL (not false). Same book, same side, same line (float32 tolerance) and a decimal price at least as good = true | Computable from the table the spec names; the regular pulls are legitimate evidence of availability too |
| 12 | `close_kind = 'missing'` "if Pinnacle is absent at the close" | Also used when Pinnacle's closing line is off the grid or the table has no answer there (the ticket has no usable `close_ev`). Vocabulary: `pinnacle_novig` or `missing`. Only `pinnacle_novig` rows with a `close_ev` count toward the 150 / 60 / 40 minimums | One rule for "excluded from the test"; the report states the missing share |
| 13 | Judged "once" | The decision block of each report is machine-readable (an HTML comment, as in the step-1 gate report). Once a report says `judged`, every later report copies that verdict unchanged and says later weeks are informational | Re-reading data each week could flip a frozen verdict |
| 14 | Weekly report "on Tue 15:52 UTC" all year | The report script writes nothing in a week with no new ticket and no new grade (the monitor is off March to July) | No empty weekly commits in the off-season |
| 15 | "Hourly ... every week of the football season (Aug-Jan)" but the crons say months `1,2,8-12` | Crons copied verbatim, so February is included (the Super Bowl falls there). No change to the spec's cron strings | The cron strings are the spec's exact values |
| 16 | `result` from "the same final providers as `grade_ledger`" | `prediction_accuracy`, as `grade_ledger` reads it. Games the model does not predict (for example an FBS team against an FCS team) have no row there and so never grade; the report counts them as "not graded yet" | Keeps the grader free of network calls; the shortfall is visible, not silent (Risks in the go-live checklist) |
| 17 | Each poll's prices are written to `odds_snapshot` | Written as specified (about 3,000 to 4,000 rows a poll, about 200,000 a week in season). The go-live checklist tells the user to watch database size after the first week | Spec decision; flagged because it is the largest ongoing cost of the test |

## Credit budget and cron

**Credits (The Odds API bills markets x regions per call).** Free: the `/events` list used to read the balance, ESPN schedules, database reads.

| Call | Cost | When |
|---|---|---|
| CFB poll (`spreads,totals`, `bookmakers` = 10 keys = one region) | 2 | every poll |
| NFL poll (`spreads`) | 1 | every poll |
| `/events` balance read | 0 | once per run |
| Dry run | about 3 | manual dispatch only |

About 3 credits a poll, 60 polls a week: **about 180 a week, about 780 a month** in season (2.6% of the 30,000-credit plan). Alongside the existing spend (regular ingest about 2,300 a week, close capture about 360) the monitor runs while the balance is at least 1,500 and skips the rest of a run with a warning below that.

**Cron (UTC; GitHub jitters and skips, and :00 is its busiest minute, accepted because the spec fixes it).**

| Workflow | Cron | Meaning |
|---|---|---|
| `early-monitor.yml` | `0 16-23 * 1,2,8-12 0`, `0 * * 1,2,8-12 1,2`, `0 0-3 * 1,2,8-12 3` | Sun 16:00 to Wed 03:00, hourly: 8 + 24 + 24 + 4 = 60 polls a week, months Aug-Feb |
| `grade-early-tickets.yml` | `20 14 * * *` | daily, after grade-predictions (every 6 hours) and grade-ledger (14:10) |
| `report-early-tickets.yml` | `52 15 * * 2` | Tuesday, after the Monday-night games are graded; grade-ledger-gate runs at 15:47 |

## File Structure

New package `src/sportsmodel/early/` keeps the pure logic apart from the scripts (which only do IO), mirroring `sportsmodel.ledger`.

| File | Action | Responsibility |
|---|---|---|
| `db/migration_early_tickets.sql` | Create | The two tables, append-only trigger, RLS, grants / revokes (USER runs it) |
| `src/sportsmodel/db.py` | Modify | `EARLY_COLUMNS` / `EARLY_KEYS`, `missing_early_tables`, `insert_early_tickets`, `upsert_early_ticket_results`, `early_ticket_keys` |
| `src/sportsmodel/early/__init__.py`, `constants.py` | Create | Package marker; every number the spec fixes, in one place |
| `src/sportsmodel/early/landing.py` | Create | Table builder (pure), `Pmf`, `LandingTable`, `load_tables` |
| `scripts/build_landing_tables.py`, `assets/early/landing_{cfb_spread,cfb_total,nfl_spread}.json` | Create | Build-once script and its committed output |
| `src/sportsmodel/early/adjust.py` | Create | Thresholds, `adjust` (line move), `value_side`, EV, flag and fair-range predicates |
| `src/sportsmodel/early/flags.py` | Create | `pinnacle_reference`, `best_offer`, `flag_poll` (one poll's rows to new tickets) |
| `src/sportsmodel/early/polling.py` | Create | `bookmakers` parameter, markets and cost per poll, credit-floor decision |
| `src/sportsmodel/ingest/odds.py` | Modify | `fetch_bookmaker_odds` |
| `scripts/early_monitor.py`, `.github/workflows/early-monitor.yml` | Create | The hourly job and its cron |
| `scripts/capture_close.py` | Modify | `early_tickets` as a close-capture candidate source |
| `src/sportsmodel/early/grading.py` | Create | `closing_view`, `available_next_poll`, `grade_ticket`, `grade_tickets` |
| `scripts/grade_early_tickets.py`, `.github/workflows/grade-early-tickets.yml` | Create | The daily grading job |
| `src/sportsmodel/early/passtest.py` | Create | The frozen pass test and the cluster bootstrap |
| `src/sportsmodel/early/report.py` | Create | Summary numbers, markdown, decision block |
| `scripts/report_early_tickets.py`, `.github/workflows/report-early-tickets.yml` | Create | The weekly report job |
| `docs/superpowers/reports/2026-10-13-early-monitor-go-live.md` | Create | The user's go-live checklist |
| `tests/early/` (`__init__.py`, `builders.py` plus 15 test files) | Create | One test file per module, script or workflow group |

Not touched: every model, board, pick writer, `ingest_odds.py`, `ev_pilot.py`, the pick-ledger code and tables, the site, and every existing workflow.

---

### Task 1: Migration and database helpers

**Files:**
- Create: `db/migration_early_tickets.sql`
- Modify: `src/sportsmodel/db.py` (append the early-monitor block at the end of the file)
- Create: `tests/early/__init__.py` (empty), `tests/early/test_migration_and_db.py`

**Interfaces:**
- Consumes (existing in `db.py`): `get_postgres`, `_pg(connect_timeout)`, `_limit_statements(cur, statement_timeout_ms)`, `_missing_public_tables(tables, connect_timeout, statement_timeout_ms)`.
- Produces (later tasks rely on these exact names):
  - `db.EARLY_COLUMNS: dict[str, list[str]]` keyed `early_tickets` / `early_ticket_results` (the columns the code writes, in insert order); `db.EARLY_KEYS: dict[str, tuple[str, ...]]`
  - `db.missing_early_tables(tables: list[str] | None = None, *, connect_timeout: int | None = None, statement_timeout_ms: int | None = None) -> list[str]` (default: both; `KeyError` for an unknown name; connection errors propagate)
  - `db.insert_early_tickets(rows: list[dict], *, connect_timeout=None, statement_timeout_ms=None) -> int` (INSERT ... ON CONFLICT (sport, game_pk, market, side) DO NOTHING; returns rows attempted)
  - `db.upsert_early_ticket_results(rows: list[dict]) -> int` (idempotent on `ticket_id`)
  - `db.early_ticket_keys(sport: str, after, *, connect_timeout=None, statement_timeout_ms=None) -> set[tuple[str, int, str, str]]` ((sport, game_pk, market, side) of tickets whose game kicks off after `after`)

- [ ] **Step 1: Write the failing tests**

Create `tests/early/__init__.py` as an empty file, then `tests/early/test_migration_and_db.py`:

````python
"""db/migration_early_tickets.sql and the early-monitor helpers in db.py: SQL shape, tuple building, column
agreement, the append-only guarantees and the missing-table guard. No live DB (fake connections)."""
import re
from pathlib import Path

import pytest

from sportsmodel import db

MIGRATION = (Path(__file__).resolve().parents[2] / "db" / "migration_early_tickets.sql").read_text()
TABLES = ["early_tickets", "early_ticket_results"]


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


# ---- migration --------------------------------------------------------------------------------

@pytest.mark.parametrize("table", TABLES)
def test_every_written_column_exists_in_the_migration(table):
    cols = re.findall(r"^\s{2}(\w+)\s", _create_table(table), re.MULTILINE)
    assert set(db.EARLY_COLUMNS[table]) <= set(cols), set(db.EARLY_COLUMNS[table]) - set(cols)


def test_tickets_have_a_bigserial_id_and_the_unique_ticket_key():
    body = _create_table("early_tickets")
    assert "id              bigserial PRIMARY KEY" in body
    assert "UNIQUE (sport, game_pk, market, side)" in body
    assert db.EARLY_KEYS["early_tickets"] == ("sport", "game_pk", "market", "side")


def test_results_are_keyed_by_ticket_id():
    body = _create_table("early_ticket_results")
    assert "ticket_id              bigint PRIMARY KEY REFERENCES early_tickets (id)" in body
    assert db.EARLY_KEYS["early_ticket_results"] == ("ticket_id",)


def test_tickets_are_append_only_by_trigger_for_update_delete_and_truncate():
    assert "CREATE OR REPLACE FUNCTION early_tickets_append_only() RETURNS trigger" in MIGRATION
    assert "RAISE EXCEPTION" in MIGRATION
    assert re.search(r"CREATE TRIGGER early_tickets_no_update_delete\s+BEFORE UPDATE OR DELETE ON early_tickets\s+"
                     r"FOR EACH ROW EXECUTE FUNCTION early_tickets_append_only\(\);", MIGRATION)
    assert re.search(r"CREATE TRIGGER early_tickets_no_truncate\s+BEFORE TRUNCATE ON early_tickets\s+"
                     r"FOR EACH STATEMENT EXECUTE FUNCTION early_tickets_append_only\(\);", MIGRATION)
    assert MIGRATION.count("DROP TRIGGER IF EXISTS") == 2          # re-runnable
    assert re.search(r"CREATE OR REPLACE FUNCTION early_tickets_append_only\(\) RETURNS trigger\s+"
                     r"LANGUAGE plpgsql SET search_path = ''", MIGRATION)


def test_rls_select_policy_and_write_revokes_for_both_tables():
    for t in TABLES:
        assert f"ALTER TABLE {t} ENABLE ROW LEVEL SECURITY;" in MIGRATION
        assert f'CREATE POLICY "public read {t}" ON {t} FOR SELECT USING (true);' in MIGRATION
    m = re.search(r"REVOKE ([A-Z, ]+) ON ([^;]+) FROM anon, authenticated;", MIGRATION)
    assert m and set(re.split(r",\s*", m.group(1))) == {"INSERT", "UPDATE", "DELETE", "TRUNCATE"}
    for t in TABLES:
        assert f"public.{t}" in m.group(2)
    assert "GRANT SELECT ON early_tickets, early_ticket_results TO anon, authenticated;" in MIGRATION
    assert MIGRATION.index("REVOKE INSERT") < MIGRATION.index("GRANT SELECT ON")


def test_tickets_revoke_update_delete_truncate_from_public_and_service_role():
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM PUBLIC;" in MIGRATION
    assert "REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM service_role" in MIGRATION
    assert "rolname = 'service_role'" in MIGRATION                  # guarded: the role may not exist locally


def test_migration_is_idempotent_additive_and_writes_no_data():
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == 2
    assert MIGRATION.count("DROP POLICY IF EXISTS") == 2
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE|ALTER TABLE \w+ DROP)\b", MIGRATION,
                         re.MULTILINE | re.IGNORECASE)
    assert "CREATE INDEX IF NOT EXISTS" in MIGRATION and "CREATE INDEX early" not in MIGRATION


def test_check_constraints_pin_the_vocabularies():
    assert "CHECK (sport IN ('cfb', 'nfl'))" in MIGRATION
    assert "CHECK (market IN ('spread', 'total'))" in MIGRATION
    assert "CHECK (side IN ('home', 'away', 'over', 'under'))" in MIGRATION
    assert "CHECK (result IN ('W', 'L', 'P'))" in MIGRATION
    assert "CHECK (close_kind IN ('pinnacle_novig', 'missing'))" in MIGRATION


# ---- db helpers ---------------------------------------------------------------------------------

class FakeCursor:
    def __init__(self, sink, fetch=None):
        self.sink, self.fetch = sink, fetch or []

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def execute(self, sql, params=None):
        self.sink.setdefault("executed", []).append(sql)
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


def _boom(*a, **k):
    raise AssertionError("get_postgres should not be called")


@pytest.mark.parametrize("fn", ["insert_early_tickets", "upsert_early_ticket_results"])
def test_empty_rows_never_connect(monkeypatch, fn):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(db, fn)([]) == 0


def test_insert_early_tickets_is_insert_only_idempotent_and_ordered_by_column(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.EARLY_COLUMNS["early_tickets"]} | {"sport": "cfb", "game_pk": 7, "side": "over"}
    assert db.insert_early_tickets([row]) == 1
    sql, (tup,) = sink["sql"], sink["rows"]
    assert sql.startswith("INSERT INTO early_tickets (")
    assert "ON CONFLICT (sport, game_pk, market, side) DO NOTHING" in sql
    assert "UPDATE" not in sql.replace("DO NOTHING", "") and "DELETE" not in sql
    cols = db.EARLY_COLUMNS["early_tickets"]
    assert len(tup) == len(cols) and tup[cols.index("sport")] == "cfb" and tup[cols.index("game_pk")] == 7
    assert sink["committed"] is True


def test_insert_early_tickets_passes_the_timeouts_through(monkeypatch):
    sink, seen = {}, {}
    monkeypatch.setattr(db, "get_postgres", lambda connect_timeout=None: seen.update(t=connect_timeout) or FakeConn(sink))
    row = {c: None for c in db.EARLY_COLUMNS["early_tickets"]}
    db.insert_early_tickets([row], connect_timeout=10, statement_timeout_ms=15_000)
    assert seen["t"] == 10 and sink["executed"][0] == "SET LOCAL statement_timeout = 15000"


def test_results_upsert_keys_on_ticket_id_and_stamps_graded_at(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {c: None for c in db.EARLY_COLUMNS["early_ticket_results"]} | {"ticket_id": 5, "result": "W"}
    assert db.upsert_early_ticket_results([row]) == 1
    sql = sink["sql"]
    assert "ON CONFLICT (ticket_id) DO UPDATE SET" in sql and "graded_at = now()" in sql
    assert "result = EXCLUDED.result" in sql and "ticket_id = EXCLUDED.ticket_id" not in sql


class _RegCursor(FakeCursor):
    def __init__(self, existing, log):
        self.existing, self.log, self.last = existing, log, None

    def execute(self, sql, params=None):
        if params is None:                                  # SET LOCAL statement_timeout
            return
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


def test_missing_early_tables_checks_to_regclass_in_public_in_order(monkeypatch):
    log = []
    monkeypatch.setattr(db, "get_postgres", lambda: _RegConn({"early_ticket_results"}, log))
    assert db.missing_early_tables() == ["early_tickets"]
    assert [p[0] for _, p in log] == [f"public.{t}" for t in TABLES]
    assert db.missing_early_tables(["early_ticket_results"]) == []


def test_missing_early_tables_rejects_unknown_names_and_propagates_connection_errors(monkeypatch):
    with pytest.raises(KeyError):
        db.missing_early_tables(["pick_ledger"])

    def down(*a, **k):
        raise ConnectionError("db down")

    monkeypatch.setattr(db, "get_postgres", down)
    with pytest.raises(ConnectionError):
        db.missing_early_tables(["early_tickets"])


def test_the_ledger_guard_does_not_know_the_early_tables():
    with pytest.raises(KeyError):
        db.missing_ledger_tables(["early_tickets"])


def test_early_ticket_keys_returns_the_logged_sides_of_upcoming_games(monkeypatch):
    sink = {}
    fetched = [("cfb", 7, "total", "over"), ("cfb", 8, "spread", "home")]
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink, fetched))
    assert db.early_ticket_keys("cfb", "2026-10-13T00:00:00Z") == set(fetched)
    assert "FROM early_tickets WHERE sport = %s AND commence_time > %s" in sink["sql"]
    assert sink["params"] == ("cfb", "2026-10-13T00:00:00Z")
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_migration_and_db.py`
Expected: collection ERROR, `FileNotFoundError: ... db/migration_early_tickets.sql` (the test module reads the migration at import).

- [ ] **Step 3: Write the migration**

Create `db/migration_early_tickets.sql` (the user runs it; the executing agent never does):

````sql
-- =============================================================================
-- Early-week stale-number monitor (profit project, step 3).
-- =============================================================================
-- Idempotent and additive only -- safe to re-run. Run this in the Supabase SQL Editor BEFORE the new jobs first run
-- (until then every early-monitor job warns and exits 0; nothing existing changes).
--
--   early_tickets         APPEND-ONLY paper tickets: one row per (sport, game_pk, market, side), logged the FIRST time
--                         a US book's number beats Pinnacle's same-poll fair by +1.5% EV with kickoff >= 24 h away.
--                         Written hourly by scripts/early_monitor.py. A trigger refuses UPDATE / DELETE / TRUNCATE and
--                         the write privileges are revoked from every role except the table owner.
--   early_ticket_results  one row per ticket, graded daily by scripts/grade_early_tickets.py against Pinnacle's
--                         pre-kick close (close EV, beat-close, line CLV, W/L/P, units, availability one poll later).
--                         An ordinary upsert table (a re-grade overwrites in place).

CREATE TABLE IF NOT EXISTS early_tickets (
  id              bigserial PRIMARY KEY,
  sport           text NOT NULL CHECK (sport IN ('cfb', 'nfl')),
  game_pk         bigint NOT NULL,
  market          text NOT NULL CHECK (market IN ('spread', 'total')),
  side            text NOT NULL CHECK (side IN ('home', 'away', 'over', 'under')),
  book            text NOT NULL,                  -- The Odds API bookmaker key of the best US book at the first sighting
  book_line       double precision NOT NULL,      -- the SIDE's own line at that book (home spread, away spread, total)
  price           integer NOT NULL,               -- American price at that book
  pinnacle_line   double precision NOT NULL,      -- Pinnacle's line for the same side, same poll
  pinnacle_fair   double precision NOT NULL,      -- Pinnacle's two-way no-vig probability for this side at its own line
  p_win           double precision NOT NULL,      -- at the BOOK's line, from the landing-table line adjustment
  p_push          double precision NOT NULL,
  p_loss          double precision NOT NULL,
  ev              double precision NOT NULL,      -- p_win * (decimal - 1) - p_loss, per unit staked
  observed_at     timestamptz NOT NULL,           -- the poll's capture time
  commence_time   timestamptz NOT NULL,
  hours_to_kick   double precision NOT NULL,
  poll_run_id     text NOT NULL,                  -- GitHub run id, or 'local'
  recorded_at     timestamptz NOT NULL DEFAULT now(),
  UNIQUE (sport, game_pk, market, side)
);

CREATE INDEX IF NOT EXISTS early_tickets_commence_idx ON early_tickets (commence_time);

CREATE OR REPLACE FUNCTION early_tickets_append_only() RETURNS trigger
LANGUAGE plpgsql SET search_path = '' AS $$
BEGIN
  RAISE EXCEPTION 'early_tickets is append-only: % is not allowed', TG_OP
    USING ERRCODE = 'restrict_violation';
END;
$$;

DROP TRIGGER IF EXISTS early_tickets_no_update_delete ON early_tickets;
CREATE TRIGGER early_tickets_no_update_delete
  BEFORE UPDATE OR DELETE ON early_tickets
  FOR EACH ROW EXECUTE FUNCTION early_tickets_append_only();

DROP TRIGGER IF EXISTS early_tickets_no_truncate ON early_tickets;
CREATE TRIGGER early_tickets_no_truncate
  BEFORE TRUNCATE ON early_tickets
  FOR EACH STATEMENT EXECUTE FUNCTION early_tickets_append_only();

CREATE TABLE IF NOT EXISTS early_ticket_results (
  ticket_id              bigint PRIMARY KEY REFERENCES early_tickets (id),
  sport                  text NOT NULL,
  game_pk                bigint NOT NULL,
  market                 text NOT NULL,
  side                   text NOT NULL,
  book                   text NOT NULL,
  book_line              double precision NOT NULL,
  price                  integer NOT NULL,
  observed_at            timestamptz NOT NULL,
  commence_time          timestamptz NOT NULL,
  result                 text NOT NULL CHECK (result IN ('W', 'L', 'P')),
  units                  double precision,         -- 1 unit staked at the ticket's price
  close_kind             text NOT NULL CHECK (close_kind IN ('pinnacle_novig', 'missing')),
  close_fair             double precision,         -- Pinnacle's closing no-vig probability for the ticket's side
  close_line             double precision,         -- Pinnacle's closing line for the ticket's side
  close_minutes_to_kick  double precision,
  close_ev               double precision,         -- the ticket's line and price valued at Pinnacle's closing fair (primary)
  beat_close             boolean,                  -- close_ev > 0
  clv_pts                double precision,         -- line points gained vs Pinnacle's closing line, in the ticket side's favour
  available_next_poll    boolean,                  -- same book, same line, same or better price 30 min to 3 h later; NULL outside
  graded_at              timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS early_ticket_results_game_idx ON early_ticket_results (sport, game_pk);

ALTER TABLE early_tickets ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read early_tickets" ON early_tickets;
CREATE POLICY "public read early_tickets" ON early_tickets FOR SELECT USING (true);
ALTER TABLE early_ticket_results ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read early_ticket_results" ON early_ticket_results;
CREATE POLICY "public read early_ticket_results" ON early_ticket_results FOR SELECT USING (true);

-- Defence in depth: the API roles can read these tables and nothing else (the jobs write as the DB owner).
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON public.early_tickets, public.early_ticket_results FROM anon, authenticated;
GRANT SELECT ON early_tickets, early_ticket_results TO anon, authenticated;

-- The tickets are append-only for EVERY non-owner role: also strip UPDATE / DELETE / TRUNCATE from PUBLIC and from the
-- other Supabase API role (service_role) when it exists. The owner is held back by the trigger.
REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM PUBLIC;
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role') THEN
    EXECUTE 'REVOKE UPDATE, DELETE, TRUNCATE ON public.early_tickets FROM service_role';
  END IF;
END;
$$;
````

- [ ] **Step 4: Append the database helpers**

Append to the end of `src/sportsmodel/db.py` (nothing else in the file changes):

````python
# ------------------------------------------------------------------ early-week monitor
# db/migration_early_tickets.sql. early_tickets is APPEND-ONLY (trigger + revoked privileges): the only write helper for
# it is insert_early_tickets. early_ticket_results is an ordinary upsert table written by the daily grader.
EARLY_COLUMNS: dict[str, list[str]] = {
    "early_tickets": ["sport", "game_pk", "market", "side", "book", "book_line", "price", "pinnacle_line",
                      "pinnacle_fair", "p_win", "p_push", "p_loss", "ev", "observed_at", "commence_time",
                      "hours_to_kick", "poll_run_id"],
    "early_ticket_results": ["ticket_id", "sport", "game_pk", "market", "side", "book", "book_line", "price",
                             "observed_at", "commence_time", "result", "units", "close_kind", "close_fair",
                             "close_line", "close_minutes_to_kick", "close_ev", "beat_close", "clv_pts",
                             "available_next_poll"],
}
EARLY_KEYS: dict[str, tuple[str, ...]] = {
    "early_tickets": ("sport", "game_pk", "market", "side"), "early_ticket_results": ("ticket_id",),
}


def missing_early_tables(tables: list[str] | None = None, *, connect_timeout: int | None = None,
                         statement_timeout_ms: int | None = None) -> list[str]:
    """The early-monitor tables (default: both) that do not exist in `public` yet, in the order given. Lets the
    monitor jobs warn and exit 0 until db/migration_early_tickets.sql has been run (same guard as the ledger)."""
    tables = list(EARLY_COLUMNS) if tables is None else tables
    unknown = [t for t in tables if t not in EARLY_COLUMNS]
    if unknown:
        raise KeyError(f"not an early-monitor table: {unknown}")
    return _missing_public_tables(tables, connect_timeout, statement_timeout_ms)


def insert_early_tickets(rows: list[dict], *, connect_timeout: int | None = None,
                         statement_timeout_ms: int | None = None) -> int:
    """Append paper tickets. INSERT only (the table refuses anything else); a ticket for a (sport, game_pk, market,
    side) that already has one is a no-op via ON CONFLICT DO NOTHING. Returns the number of rows ATTEMPTED."""
    if not rows:
        return 0
    cols, key = EARLY_COLUMNS["early_tickets"], EARLY_KEYS["early_tickets"]
    sql = (f"INSERT INTO early_tickets ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT ({', '.join(key)}) DO NOTHING")
    vals = [tuple(r.get(c) for c in cols) for r in rows]
    with _pg(connect_timeout) as conn, conn.cursor() as cur:
        _limit_statements(cur, statement_timeout_ms)
        cur.executemany(sql, vals)
        conn.commit()
    return len(vals)


def upsert_early_ticket_results(rows: list[dict]) -> int:
    """Upsert graded tickets; idempotent on ticket_id (a re-grade overwrites in place). Returns rows ATTEMPTED."""
    if not rows:
        return 0
    cols, key = EARLY_COLUMNS["early_ticket_results"], EARLY_KEYS["early_ticket_results"]
    sets = ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c not in key)
    sql = (f"INSERT INTO early_ticket_results ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
           f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {sets}, graded_at = now()")
    vals = [tuple(r.get(c) for c in cols) for r in rows]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, vals)
        conn.commit()
    return len(vals)


def early_ticket_keys(sport: str, after, *, connect_timeout: int | None = None,
                      statement_timeout_ms: int | None = None) -> set[tuple]:
    """{(sport, game_pk, market, side)} of the tickets already logged for `sport` games kicking off after `after`
    (the monitor's one-ticket-per-side rule; a game already started can never flag again). Read-only."""
    with _pg(connect_timeout) as conn, conn.cursor() as cur:
        _limit_statements(cur, statement_timeout_ms)
        cur.execute("SELECT sport, game_pk, market, side FROM early_tickets WHERE sport = %s AND commence_time > %s",
                    (sport, after))
        return {(r[0], int(r[1]), r[2], r[3]) for r in cur.fetchall()}
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early tests/ledger/test_migration_and_db.py tests/test_db_site_panels.py`
Expected: all PASS (`tests/early` alone: 18 passed).

- [ ] **Step 6: Commit**

```bash
git add db/migration_early_tickets.sql src/sportsmodel/db.py tests/early/__init__.py tests/early/test_migration_and_db.py
git commit -m "feat(early): early_tickets / early_ticket_results migration and db helpers" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Landing tables (builder, committed JSON, loader)

**Files:**
- Create: `src/sportsmodel/early/__init__.py`, `src/sportsmodel/early/constants.py`, `src/sportsmodel/early/landing.py`
- Create: `scripts/build_landing_tables.py`
- Create (generated by the script, then committed): `assets/early/landing_cfb_spread.json`, `assets/early/landing_cfb_total.json`, `assets/early/landing_nfl_spread.json`
- Create: `tests/early/test_landing.py`, `tests/early/test_build_landing_script.py`

**Interfaces:**
- Consumes: `sportsmodel.config.PROJECT_ROOT` (existing); the committed `assets/cfb/lines.parquet`, `assets/cfb/schedules.parquet`, `assets/nfl/schedules.parquet`.
- Produces:
  - `constants`: every spec number (`FLAG_EV_MIN = 0.015`, `FLAG_EV_EPS`, `MIN_HOURS_TO_KICK = 24.0`, `FAIR_MIN`, `FAIR_MAX`, `MAX_LINE_GAP = 3.0`, `WINDOW_NARROW = 1.0`, `WINDOW_WIDE = 2.0`, `MIN_GAMES = 300`, `GRID_STEP = 0.5`, `LINE_EPS`, `GRID_EPS`, `PINNACLE`, `US_BOOKS` (9 keys), `BOOKMAKERS` (10 keys), `SPORT_ORDER = ("cfb", "nfl")`, `MARKETS`, `API_MARKETS`, `SIDES`, `LANDING_TABLES`, `MONITOR_FLOOR = 1500`, `NEXT_POLL_MIN_MINUTES = 30.0`, `NEXT_POLL_MAX_MINUTES = 180.0`, `SETTLE_HOURS = 4`)
  - `landing.Pmf(n, window, probs)` with `.mass(value) -> float`; `landing.on_grid(center) -> bool`; `landing.grid_key(center) -> str`; `landing.table_path(sport, market, directory=None) -> Path`
  - `landing.build_table(pairs, *, sport, market, seasons, source) -> dict` (pairs of `(closing line, integer result)`)
  - `landing.LandingTable(raw).pmf(center) -> Pmf | None`; `landing.load_tables(directory=None) -> {(sport, market): LandingTable}` for cfb spread, cfb total, nfl spread
  - script helpers `cfb_pairs(lines_df, schedules_df)`, `nfl_pairs(schedules_df)`, `build_all(assets_dir)`, `main(argv)`

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_landing.py`:

````python
"""landing.py: table building (windows, widening, thin tails), lookups, and the committed JSON assets. No network."""
import json

import pytest

from sportsmodel.early import landing
from sportsmodel.early.constants import LANDING_TABLES, MIN_GAMES


def pairs_at(line, results):
    return [(line, r) for r in results]


# 400 games closing at 3.0: 40 land on exactly 3 (the key number), the rest spread over -20..20 skipping 3
RESULTS = [3] * 40 + [r for r in range(-20, 21) if r != 3] * 9 + [0] * 0
BASE = pairs_at(3.0, RESULTS)


def table(pairs, market="spread"):
    return landing.build_table(pairs, sport="nfl", market=market, seasons=(2010, 2025), source="test")


def test_the_sample_is_big_enough_for_these_tests():
    assert len(BASE) >= MIN_GAMES


def test_pmf_sums_to_one_and_keeps_the_key_number_mass():
    t = landing.LandingTable(table(BASE))
    pmf = t.pmf(3.0)
    assert sum(pmf.probs.values()) == pytest.approx(1.0)
    assert pmf.mass(3) == pytest.approx(40 / len(BASE)) and pmf.mass(3) > pmf.mass(4) * 3
    assert pmf.n == len(BASE) and pmf.window == 1.0


def test_whole_number_lines_carry_push_mass_and_half_lines_are_on_the_grid_too():
    t = landing.LandingTable(table(BASE + pairs_at(4.0, RESULTS)))
    assert t.pmf(3.0).mass(3) == pytest.approx(40 / len(BASE))     # the push mass of a whole-number threshold
    assert t.pmf(3.5) is not None and t.pmf(3.5).n == 2 * len(BASE)


def test_window_is_one_point_unless_that_gives_fewer_than_300_games_then_two():
    narrow = BASE
    assert landing.LandingTable(table(narrow)).pmf(3.0).window == 1.0
    sparse = pairs_at(3.0, RESULTS[:100]) + pairs_at(4.5, RESULTS[:150]) + pairs_at(1.5, RESULTS[:150])
    t = landing.LandingTable(table(sparse))
    assert t.pmf(3.0).window == 2.0 and t.pmf(3.0).n == 400           # 100 within +/-1, 400 within +/-2
    assert t.pmf(3.0).mass(3) > 0


def test_grid_values_still_short_at_two_points_are_left_out():
    sparse = pairs_at(3.0, RESULTS[:100]) + pairs_at(20.0, RESULTS[:100])
    t = landing.LandingTable(table(sparse))
    assert t.pmf(3.0) is None and t.pmf(20.0) is None and t.pmf(10.0) is None


def test_the_window_edges_are_inclusive():
    edge = pairs_at(2.0, RESULTS[:150]) + pairs_at(4.0, RESULTS[:150])            # exactly 1.0 point either side of 3.0
    assert landing.LandingTable(table(edge)).pmf(3.0).n == 300


def test_off_grid_and_out_of_range_lines_answer_none():
    t = landing.LandingTable(table(BASE))
    assert t.pmf(3.25) is None and t.pmf(3.75) is None and t.pmf(99.5) is None and t.pmf(None) is None


def test_a_pick_em_never_looks_up_negative_zero():
    pairs = pairs_at(0.0, RESULTS)
    t = landing.LandingTable(table(pairs))
    assert landing.grid_key(-0.0) == "0.0" and t.pmf(-0.0) is not None and t.pmf(0.0) is not None


def test_counts_not_probabilities_are_stored_so_the_loader_normalises_exactly():
    raw = table(BASE)
    entry = raw["entries"]["3.0"]
    assert entry["lo"] == -20 and all(isinstance(c, int) for c in entry["counts"]) and sum(entry["counts"]) == entry["n"]
    assert raw["result"] == "home_margin" and table(BASE, "total")["result"] == "total_points"
    assert raw["games"] == len(BASE) and raw["seasons"] == [2010, 2025]


def test_building_is_deterministic_and_order_independent():
    assert json.dumps(table(BASE), sort_keys=True) == json.dumps(table(list(reversed(BASE))), sort_keys=True)


def test_build_rejects_empty_input_and_unknown_markets():
    with pytest.raises(ValueError):
        table([])
    with pytest.raises(ValueError):
        table(BASE, market="moneyline")


# ---- the committed assets ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def committed():
    return landing.load_tables()


def test_all_three_committed_tables_load(committed):
    assert set(committed) == set(LANDING_TABLES)


def test_committed_tables_normalise_and_respect_the_sample_floor(committed):
    for (sport, market), t in committed.items():
        assert t.meta["entries"], (sport, market)
        for key, entry in t.meta["entries"].items():
            pmf = t.pmf(float(key))
            assert sum(pmf.probs.values()) == pytest.approx(1.0)
            assert entry["n"] >= MIN_GAMES and entry["window"] in (1.0, 2.0)


def test_committed_nfl_spread_table_keeps_its_key_numbers(committed):
    pmf = committed[("nfl", "spread")].pmf(3.0)                       # closing line 3: home favoured by 3
    assert pmf.mass(3) > 0.08 and pmf.mass(7) > 0.04 and pmf.mass(3) > 2 * pmf.mass(4)
    assert committed[("nfl", "spread")].pmf(-3.0).mass(-3) > 0.07


def test_committed_cfb_tables_cover_the_lines_the_monitor_will_see(committed):
    spread, total = committed[("cfb", "spread")], committed[("cfb", "total")]
    for centre in (-7.0, -3.5, 3.0, 10.5, 21.0):
        assert spread.pmf(centre) is not None, centre
    for centre in (45.5, 52.5, 55.5, 60.0, 65.5):
        assert total.pmf(centre) is not None, centre
    assert abs(sum(k * v for k, v in total.pmf(55.5).probs.items()) - 55.5) < 3.0       # centred on its line


def test_a_missing_table_file_is_a_loud_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        landing.load_tables(tmp_path)
````

Create `tests/early/test_build_landing_script.py`:

````python
"""scripts/build_landing_tables.py on tiny synthetic frames: joins, sign convention, filters, output files."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("build_landing_tables", ROOT / "scripts" / "build_landing_tables.py")
bl = importlib.util.module_from_spec(_spec)
sys.modules["build_landing_tables"] = bl
_spec.loader.exec_module(bl)


def lines(rows):
    return pd.DataFrame(rows, columns=["season", "week", "home_team", "away_team", "market_spread", "market_total"])


def schedules(rows):
    return pd.DataFrame(rows, columns=["season", "week", "home_team", "away_team", "home_score", "away_score"])


def test_cfb_pairs_join_on_season_week_and_both_team_ids():
    ln = lines([(2024, 3, "12", "99", 7.0, 55.5), (2024, 3, "13", "98", -3.0, 48.5)])
    sc = schedules([(2024, 3, "12", "99", 31, 20), (2024, 3, "13", "98", 10, 17), (2024, 4, "12", "99", 0, 0)])
    out = bl.cfb_pairs(ln, sc)
    assert out["spread"] == [(7.0, 11), (-3.0, -7)]                 # (expected home margin, home margin)
    assert out["total"] == [(55.5, 51), (48.5, 27)]


def test_cfb_spread_sign_is_positive_for_a_home_favourite_as_in_the_source():
    ln = lines([(2024, 1, "1", "2", 14.0, 50.0)])
    sc = schedules([(2024, 1, "1", "2", 35, 14)])
    assert bl.cfb_pairs(ln, sc)["spread"] == [(14.0, 21)]


def test_cfb_rows_without_a_line_or_a_score_or_outside_the_seasons_are_dropped():
    ln = lines([(2024, 1, "1", "2", 3.0, None), (2024, 1, "3", "4", 3.0, 50.0), (2014, 1, "5", "6", 3.0, 50.0),
                (2026, 1, "7", "8", 3.0, 50.0), (2024, 1, "9", "10", 3.0, 50.0)])
    sc = schedules([(2024, 1, "1", "2", 20, 10), (2024, 1, "3", "4", 20, None), (2014, 1, "5", "6", 20, 10),
                    (2026, 1, "7", "8", 20, 10), (2024, 1, "9", "10", 20, 10)])
    out = bl.cfb_pairs(ln, sc)
    assert out["spread"] == [(3.0, 10), (3.0, 10)] and out["total"] == [(50.0, 30)]


def test_nfl_pairs_use_spread_line_and_scores_for_2010_to_2025():
    sc = pd.DataFrame({"season": [2009, 2010, 2025, 2026, 2024], "spread_line": [3.0, -4.0, 7.0, 1.0, None],
                       "home_score": [20, 13, 27, 10, 20.0], "away_score": [10, 16, 20, 9, 17.0]})
    assert bl.nfl_pairs(sc) == [(-4.0, -3), (7.0, 7)]


def test_main_writes_the_three_tables_into_the_output_folder(tmp_path, monkeypatch):
    fake = {("cfb", "spread"): {"sport": "cfb", "market": "spread", "entries": {}},
            ("cfb", "total"): {"sport": "cfb", "market": "total", "entries": {}, "games": 0},
            ("nfl", "spread"): {"sport": "nfl", "market": "spread", "entries": {}, "games": 0}}
    for v in fake.values():
        v.setdefault("games", 0)
    monkeypatch.setattr(bl, "build_all", lambda assets: fake)
    assert bl.main(["--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["landing_cfb_spread.json", "landing_cfb_total.json",
                                                         "landing_nfl_spread.json"]
    assert json.loads((tmp_path / "landing_nfl_spread.json").read_text())["sport"] == "nfl"
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_landing.py tests/early/test_build_landing_script.py`
Expected: collection ERRORs, `ModuleNotFoundError: No module named 'sportsmodel.early'` and `FileNotFoundError` for `scripts/build_landing_tables.py`.

- [ ] **Step 3: Write the package marker and the constants**

Create `src/sportsmodel/early/__init__.py`:

````python
"""Early-week stale-number monitor (profit project, step 3). See
docs/superpowers/specs/2026-10-13-early-week-monitor-design.md."""
````

Create `src/sportsmodel/early/constants.py`:

````python
"""Every number the early-week monitor's spec fixes, in one place. PURE -- no IO.

Copied from docs/superpowers/specs/2026-10-13-early-week-monitor-design.md. The pass-test numbers live in
`passtest.py` (frozen there); everything the polling, flagging and grading code shares lives here.
"""
from __future__ import annotations

# -- flag rule (spec section 1)
FLAG_EV_MIN = 0.015                  # +1.5% EV
FLAG_EV_EPS = 1e-12                  # float noise only: an EV of 0.015 - 1e-13 still flags
MIN_HOURS_TO_KICK = 24.0             # kickoff must be at least 24 h after the poll
FAIR_MIN, FAIR_MAX = 0.15, 0.85      # Pinnacle fair outside this range: no ticket
MAX_LINE_GAP = 3.0                   # |book line - Pinnacle line| above this many points: no ticket

# -- landing tables (spec section 1)
WINDOW_NARROW = 1.0                  # games whose closing line is within +/-1.0 point of L
WINDOW_WIDE = 2.0                    # ...widened to +/-2.0 when that gives fewer than MIN_GAMES
MIN_GAMES = 300
GRID_STEP = 0.5                      # half-point grid of line values
LINE_EPS = 1e-4                      # odds_snapshot.line is a float32 REAL: compare lines with a tolerance
GRID_EPS = 1e-6

# -- books and markets (spec sections 1 and 2)
PINNACLE = "pinnacle"
US_BOOKS = ("draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "hardrockbet",
            "ballybet", "betrivers")        # williamhill_us is Caesars' Odds API key
BOOKMAKERS = US_BOOKS + (PINNACLE,)         # the 10 bookmaker keys: one billed region
SPORT_ORDER = ("cfb", "nfl")
MARKETS = {"cfb": ("spread", "total"), "nfl": ("spread",)}          # NFL totals and all moneylines are excluded
API_MARKETS = {"cfb": ("spreads", "totals"), "nfl": ("spreads",)}   # The Odds API market keys per poll
SIDES = {"spread": ("home", "away"), "total": ("over", "under")}
LANDING_TABLES = (("cfb", "spread"), ("cfb", "total"), ("nfl", "spread"))

# -- credits (spec section 2)
MONITOR_FLOOR = 1500                 # remaining credits below this: the monitor skips its remaining calls

# -- grading (spec section 3)
NEXT_POLL_MIN_MINUTES = 30.0         # available_next_poll: the next poll must be 30 min ...
NEXT_POLL_MAX_MINUTES = 180.0        # ... to 3 h after the ticket, else NULL
SETTLE_HOURS = 4                     # grade a ticket only this long after kickoff (finals have landed)
````

- [ ] **Step 4: Write the landing-table builder and loader**

Create `src/sportsmodel/early/landing.py`:

````python
"""Landing tables: how a game's result lands around its closing line. PURE except `load_tables` (reads JSON).

For one sport and market, and for each line value L on a half-point grid, a table holds the empirical distribution
of the integer RESULT of games whose closing line was within +/-1.0 point of L (widened to +/-2.0 when that gives
fewer than 300 games; a grid value still short at +/-2.0 is left out and answers None). The result is the home margin
for spreads and the total points for totals, so key-number mass (NFL margins 3, 7, 10, 14; common CFB totals) and the
push mass of whole-number lines are kept exactly as history delivered them.

The line value L is the "table-space centre": for spreads it is the expected home margin (positive = home favoured,
the nflverse `spread_line` and the CFB `market_spread` convention, i.e. minus the home team's betting line); for
totals it is the total.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .. import config
from .constants import GRID_EPS, GRID_STEP, LANDING_TABLES, MIN_GAMES, WINDOW_NARROW, WINDOW_WIDE

ASSETS = config.PROJECT_ROOT / "assets" / "early"
RESULT_VARIABLE = {"spread": "home_margin", "total": "total_points"}


@dataclass(frozen=True)
class Pmf:
    """Distribution of the integer result around one grid line. `probs` sums to 1."""
    n: int                       # games in the sample
    window: float                # +/- points the sample was drawn from (1.0 or 2.0)
    probs: dict[int, float]

    def mass(self, value) -> float:
        return self.probs.get(int(round(value)), 0.0)


def table_path(sport: str, market: str, directory: Path | None = None) -> Path:
    return Path(directory or ASSETS) / f"landing_{sport}_{market}.json"


def grid_key(center: float) -> str:
    """JSON key of a grid value. `+ 0.0` turns -0.0 into 0.0 so a pick-em never looks up '-0.0'."""
    return f"{float(center) + 0.0:.1f}"


def on_grid(center) -> bool:
    return center is not None and abs(2.0 * float(center) - round(2.0 * float(center))) < GRID_EPS


def build_table(pairs: list[tuple[float, int]], *, sport: str, market: str, seasons: tuple[int, int],
                source: str) -> dict:
    """The JSON-ready table from (closing line, integer result) pairs. Grid values run from the lowest to the highest
    observed line in half-point steps; each keeps the narrow window if it holds at least MIN_GAMES games, else the wide
    one, else is omitted. Counts (not probabilities) are stored, so the loader normalises exactly."""
    if market not in RESULT_VARIABLE:
        raise ValueError(f"unknown market: {market!r}")
    pairs = sorted((float(line), int(result)) for line, result in pairs)
    if not pairs:
        raise ValueError("no (line, result) pairs")
    lines = [p[0] for p in pairs]
    results = [p[1] for p in pairs]
    first = math.floor(lines[0] / GRID_STEP) * GRID_STEP
    last = math.ceil(lines[-1] / GRID_STEP) * GRID_STEP
    entries: dict[str, dict] = {}
    for i in range(int(round((last - first) / GRID_STEP)) + 1):
        center = first + i * GRID_STEP
        for window in (WINDOW_NARROW, WINDOW_WIDE):
            lo = bisect_left(lines, center - window - 1e-9)
            hi = bisect_right(lines, center + window + 1e-9)
            if hi - lo >= MIN_GAMES:
                counts = Counter(results[lo:hi])
                low, high = min(counts), max(counts)
                entries[grid_key(center)] = {"n": hi - lo, "window": window, "lo": low,
                                             "counts": [counts.get(v, 0) for v in range(low, high + 1)]}
                break
    return {"sport": sport, "market": market, "result": RESULT_VARIABLE[market], "step": GRID_STEP,
            "window_narrow": WINDOW_NARROW, "window_wide": WINDOW_WIDE, "min_games": MIN_GAMES,
            "games": len(pairs), "seasons": list(seasons), "source": source, "entries": entries}


class LandingTable:
    """One loaded table; `pmf(center)` answers None for an off-grid, out-of-range or too-thin line."""

    def __init__(self, raw: dict):
        self.sport, self.market, self.meta = raw["sport"], raw["market"], raw
        self._entries = raw["entries"]
        self._cache: dict[str, Pmf] = {}

    def pmf(self, center) -> Pmf | None:
        if not on_grid(center):
            return None
        key = grid_key(center)
        if key in self._cache:
            return self._cache[key]
        entry = self._entries.get(key)
        if entry is None:
            return None
        total = float(sum(entry["counts"]))
        pmf = Pmf(n=int(entry["n"]), window=float(entry["window"]),
                  probs={entry["lo"] + i: c / total for i, c in enumerate(entry["counts"]) if c})
        self._cache[key] = pmf
        return pmf


def load_tables(directory: Path | None = None) -> dict[tuple[str, str], LandingTable]:
    """{(sport, market): LandingTable} for every market the monitor tests. A missing asset raises
    FileNotFoundError (the JSON is committed; its absence is a broken checkout, not a quiet day)."""
    return {(s, m): LandingTable(json.loads(table_path(s, m, directory).read_text())) for s, m in LANDING_TABLES}
````

- [ ] **Step 5: Write the build-once script**

Create `scripts/build_landing_tables.py`:

````python
"""Build the landing tables the early-week monitor uses to move a Pinnacle line to a book's line (profit step 3).

Reads only committed history -- no network, no database:
  * CFB: assets/cfb/lines.parquet closing `market_spread` / `market_total` (2015-2025), joined on
    (season, week, home_team, away_team) to the final scores in assets/cfb/schedules.parquet.
  * NFL: assets/nfl/schedules.parquet `spread_line` / `total_line` and scores (2010-2025).
Writes assets/early/landing_{sport}_{market}.json for cfb spread, cfb total and nfl spread (see
sportsmodel.early.landing for the format). Re-running with unchanged inputs rewrites identical files.

    uv run python scripts/build_landing_tables.py [--out assets/early]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.early import landing  # noqa: E402

CFB_SEASONS = (2015, 2025)
NFL_SEASONS = (2010, 2025)
CFB_KEYS = ["season", "week", "home_team", "away_team"]


def cfb_pairs(lines: pd.DataFrame, schedules: pd.DataFrame) -> dict[str, list[tuple[float, int]]]:
    """{'spread': [(expected home margin, home margin)], 'total': [(closing total, total points)]} for CFB games
    with a closing line and a final score. `market_spread` is positive when the home team is favoured."""
    lo, hi = CFB_SEASONS
    merged = lines[(lines.season >= lo) & (lines.season <= hi)].merge(
        schedules[CFB_KEYS + ["home_score", "away_score"]], on=CFB_KEYS, how="inner")
    merged = merged.dropna(subset=["home_score", "away_score"])
    margin = (merged.home_score - merged.away_score).astype(int)
    total = (merged.home_score + merged.away_score).astype(int)
    spread_ok = merged.market_spread.notna()
    total_ok = merged.market_total.notna()
    return {
        "spread": list(zip(merged.market_spread[spread_ok].astype(float), margin[spread_ok])),
        "total": list(zip(merged.market_total[total_ok].astype(float), total[total_ok])),
    }


def nfl_pairs(schedules: pd.DataFrame) -> list[tuple[float, int]]:
    """[(expected home margin, home margin)] for NFL games 2010-2025 (`spread_line` is positive when home is favoured)."""
    lo, hi = NFL_SEASONS
    games = schedules[(schedules.season >= lo) & (schedules.season <= hi)]
    games = games.dropna(subset=["spread_line", "home_score", "away_score"])
    margin = (games.home_score - games.away_score).astype(int)
    return list(zip(games.spread_line.astype(float), margin))


def build_all(assets: Path) -> dict[tuple[str, str], dict]:
    cfb = cfb_pairs(pd.read_parquet(assets / "cfb" / "lines.parquet"), pd.read_parquet(assets / "cfb" / "schedules.parquet"))
    nfl = nfl_pairs(pd.read_parquet(assets / "nfl" / "schedules.parquet"))
    cfb_src = "assets/cfb/lines.parquet joined to assets/cfb/schedules.parquet"
    return {
        ("cfb", "spread"): landing.build_table(cfb["spread"], sport="cfb", market="spread", seasons=CFB_SEASONS, source=cfb_src),
        ("cfb", "total"): landing.build_table(cfb["total"], sport="cfb", market="total", seasons=CFB_SEASONS, source=cfb_src),
        ("nfl", "spread"): landing.build_table(nfl, sport="nfl", market="spread", seasons=NFL_SEASONS,
                                               source="assets/nfl/schedules.parquet"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--assets", default=str(ROOT / "assets"), help="folder holding cfb/ and nfl/ parquet inputs")
    ap.add_argument("--out", default=str(ROOT / "assets" / "early"), help="folder to write the JSON tables to")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for (sport, market), table in build_all(Path(args.assets)).items():
        path = landing.table_path(sport, market, out)
        path.write_text(json.dumps(table, sort_keys=True, separators=(",", ":")) + "\n")
        kept = len(table["entries"])
        print(f"[landing] {sport} {market}: {table['games']} games, {kept} grid values -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

- [ ] **Step 6: Build the three committed tables (local, no network, no database)**

Run: `uv run python scripts/build_landing_tables.py`
Expected output (numbers as of 2026-10-13; a few more or fewer games are fine, the committed-asset tests assert structure and ranges only):

```
[landing] cfb spread: 7882 games, 85 grid values -> .../assets/early/landing_cfb_spread.json
[landing] cfb total: 7867 games, 59 grid values -> .../assets/early/landing_cfb_total.json
[landing] nfl spread: 4363 games, 42 grid values -> .../assets/early/landing_nfl_spread.json
```

Run it a second time and `git status`: the files must be unchanged (the build is deterministic).

- [ ] **Step 7: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early/test_landing.py tests/early/test_build_landing_script.py`
Expected: 21 passed.

- [ ] **Step 8: Commit**

```bash
git add src/sportsmodel/early/__init__.py src/sportsmodel/early/constants.py src/sportsmodel/early/landing.py scripts/build_landing_tables.py assets/early/landing_cfb_spread.json assets/early/landing_cfb_total.json assets/early/landing_nfl_spread.json tests/early/test_landing.py tests/early/test_build_landing_script.py
git commit -m "feat(early): landing tables built from CFB and NFL history, committed as JSON" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Line adjustment and EV (pure)

**Files:**
- Create: `src/sportsmodel/early/adjust.py`
- Create: `tests/early/test_adjust.py`

**Interfaces:**
- Consumes: `landing.Pmf`, `landing.on_grid`; `constants.FAIR_MIN`, `FAIR_MAX`, `FLAG_EV_EPS`, `FLAG_EV_MIN`, `GRID_EPS`, `MAX_LINE_GAP`; `serving.board.decimal_odds` (existing).
- Produces:
  - `adjust.Valuation(p_win, p_push, p_loss, ev)` (frozen dataclass)
  - `adjust.threshold(market, side, line) -> float | None` (table space: minus the line for the home spread, the line for the away spread and totals); `adjust.own_line(market, side, c) -> float` (its inverse); `adjust.direction(market, side) -> int | None` (+1 home / over, -1 away / under)
  - `adjust.adjust(fair, d, cp, cb, pmf, *, cap=MAX_LINE_GAP, strict=True) -> (p_win, p_push, p_loss) | None`
  - `adjust.value_side(market, side, fair, pinnacle_line, book_line, price, pmf, *, cap=MAX_LINE_GAP, strict=True) -> Valuation | None` (both lines are the side's own lines)
  - `adjust.ev_of(p_win, p_loss, price) -> float`; `adjust.is_flag(ev) -> bool`; `adjust.fair_in_range(fair) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_adjust.py`:

````python
import pytest

from sportsmodel.early import adjust
from sportsmodel.early.landing import Pmf


def pmf(masses):
    total = sum(masses.values())
    return Pmf(n=1000, window=1.0, probs={k: v / total for k, v in masses.items()})


# NFL-like home-margin table: 10% of the mass sits on the key number 3, 5% on 2 and 4, the rest elsewhere.
NFL_MASSES = {0: 0.04, 1: 0.03, 2: 0.05, 3: 0.10, 4: 0.05, 5: 0.03, 6: 0.04, 7: 0.07, 8: 0.03, -3: 0.10, -7: 0.07,
              10: 0.04, -1: 0.03, -2: 0.05, -4: 0.05, -5: 0.03, -6: 0.04, -8: 0.03, 14: 0.03, -10: 0.04, 20: 0.05}
assert sum(NFL_MASSES.values()) == pytest.approx(1.0)
NFL = pmf(NFL_MASSES)
FAIR = 0.55


def moved(side_fair, d, cp, cb, table=NFL, **kw):
    return adjust.adjust(side_fair, d, cp, cb, table, **kw)


def test_threshold_and_direction_by_market_and_side():
    assert adjust.threshold("spread", "home", -2.5) == 2.5        # home laying 2.5: home must win by 3+, result > 2.5
    assert adjust.threshold("spread", "away", 2.5) == 2.5        # the away side's own line is +2.5
    assert adjust.threshold("total", "over", 54.5) == 54.5 == adjust.threshold("total", "under", 54.5)
    assert adjust.threshold("spread", "home", None) is None and adjust.threshold("moneyline", "home", 1) is None
    assert [adjust.direction("spread", "home"), adjust.direction("spread", "away"),
            adjust.direction("total", "over"), adjust.direction("total", "under")] == [1, -1, 1, -1]
    assert adjust.own_line("spread", "home", 2.5) == -2.5 and adjust.own_line("spread", "away", 2.5) == 2.5
    assert adjust.own_line("total", "under", 54.5) == 54.5


def test_same_line_is_an_identity_on_a_half_point_line():
    assert moved(FAIR, 1, 2.5, 2.5) == pytest.approx((FAIR, 0.0, 1 - FAIR))


def test_same_whole_number_line_keeps_the_table_push_mass_and_the_conditional_fair():
    win, push, loss = moved(FAIR, 1, 3.0, 3.0)
    assert push == pytest.approx(0.10) and win == pytest.approx(FAIR * 0.90) and loss == pytest.approx(0.45 * 0.90)


def test_a_half_point_across_the_key_number_moves_win_probability_by_the_table_mass():
    # Pinnacle home -2.5 (c = 2.5: home wins on margin >= 3); the book hangs -3.5 (needs margin >= 4)
    win, push, loss = moved(FAIR, 1, 2.5, 3.5)
    assert win == pytest.approx(FAIR - 0.10) and loss == pytest.approx(1 - FAIR + 0.10) and push == 0.0


def test_moving_to_a_whole_number_turns_the_crossed_mass_into_a_push():
    win, push, loss = moved(FAIR, 1, 2.5, 3.0)          # -2.5 -> -3: margin 3 stops winning and pushes
    assert (win, push, loss) == pytest.approx((FAIR - 0.10, 0.10, 1 - FAIR))


def test_pinnacle_push_mass_becomes_a_win_when_the_book_line_is_better():
    # Pinnacle -3 (push on 3, fair is conditional on no push); the book offers -2.5: the push becomes a win
    win, push, loss = moved(FAIR, 1, 3.0, 2.5)
    assert win == pytest.approx(FAIR * 0.9 + 0.10) and push == 0.0 and loss == pytest.approx(0.45 * 0.9)


def test_pinnacle_push_mass_becomes_a_loss_when_the_book_line_is_worse():
    win, push, loss = moved(FAIR, 1, 3.0, 3.5)
    assert win == pytest.approx(FAIR * 0.9) and push == 0.0 and loss == pytest.approx(0.45 * 0.9 + 0.10)


def test_the_down_direction_mirrors_the_up_direction():
    # away side at the same thresholds: it wins on margin < c
    win, push, loss = moved(0.45, -1, 3.5, 2.5)         # away +3.5 -> +2.5: margin 3 now beats the away side
    assert win == pytest.approx(0.45 - 0.10) and loss == pytest.approx(0.55 + 0.10)


def test_probabilities_always_sum_to_one():
    for cp, cb in [(2.5, 3.5), (3.0, 2.5), (-3.0, -2.0), (7.0, 6.5), (0.0, 1.5), (3.0, 3.0)]:
        for d in (1, -1):
            assert sum(moved(0.5, d, cp, cb)) == pytest.approx(1.0)


def test_three_point_cap_is_inclusive():
    assert moved(FAIR, 1, 2.5, 5.5) is not None                    # exactly 3.0 points
    assert moved(FAIR, 1, 2.5, 5.5 + 0.5) is None                  # 3.5 points: no ticket
    assert moved(FAIR, 1, 5.5 + 0.5, 2.5) is None
    assert moved(FAIR, 1, 2.5, 6.5, cap=None) is not None          # grading removes the cap


def test_off_grid_lines_are_refused_never_rounded():
    assert moved(FAIR, 1, 2.75, 2.5) is None and moved(FAIR, 1, 2.5, 2.25) is None


def test_a_move_bigger_than_the_available_probability_is_refused_strictly_and_clamped_when_grading():
    wide = pmf({r: 1.0 for r in range(-10, 11)})
    assert moved(0.15, 1, 0.5, 6.5, table=wide) is None           # the table moves 6 / 21 of the mass out of a 0.15 win
    win, push, loss = moved(0.15, 1, 0.5, 6.5, table=wide, strict=False, cap=None)
    assert win == 0.0 and push == 0.0 and loss == pytest.approx(1.0)


def test_ev_formula_with_a_push_contributing_zero():
    assert adjust.ev_of(0.5, 0.5, 100) == pytest.approx(0.0)
    assert adjust.ev_of(0.55, 0.45, -110) == pytest.approx(0.55 * (100 / 110) - 0.45)
    assert adjust.ev_of(0.40, 0.40, 120) == pytest.approx(0.40 * 1.2 - 0.40)       # push mass 0.20 is simply absent


def test_flag_threshold_boundary_is_inclusive():
    assert adjust.is_flag(0.015) and adjust.is_flag(0.0150001) and adjust.is_flag(0.015 - 1e-13)
    assert not adjust.is_flag(0.0149) and not adjust.is_flag(None) and not adjust.is_flag(-0.2)


def test_fair_range_bounds_are_inclusive():
    assert adjust.fair_in_range(0.15) and adjust.fair_in_range(0.85) and adjust.fair_in_range(0.5)
    assert not adjust.fair_in_range(0.1499) and not adjust.fair_in_range(0.8501)


def test_value_side_end_to_end_for_a_home_spread():
    v = adjust.value_side("spread", "home", FAIR, -2.5, -2.5, -105, NFL)
    assert v.p_win == pytest.approx(FAIR) and v.p_push == 0.0
    assert v.ev == pytest.approx(FAIR * (100 / 105) - (1 - FAIR))
    worse = adjust.value_side("spread", "home", FAIR, -2.5, -3.5, -105, NFL)
    assert worse.ev < v.ev and worse.p_win == pytest.approx(FAIR - 0.10)


def test_value_side_for_an_away_spread_uses_the_side_own_lines():
    # away +2.5 at Pinnacle, away +3.5 at the book: the book gives the underdog an extra point (better)
    v = adjust.value_side("spread", "away", 1 - FAIR, 2.5, 3.5, -110, NFL)
    assert v.p_win == pytest.approx(1 - FAIR + 0.10)


def test_value_side_for_totals_under_and_over():
    totals = pmf({50: 0.1, 51: 0.1, 52: 0.1, 53: 0.1, 54: 0.1, 55: 0.1, 56: 0.1, 57: 0.1, 58: 0.1, 59: 0.1})
    over = adjust.value_side("total", "over", 0.5, 54.5, 53.5, -110, totals)        # book total is lower: better Over
    assert over.p_win == pytest.approx(0.5 + 0.1)
    under = adjust.value_side("total", "under", 0.5, 54.5, 55.5, -110, totals)      # book total is higher: better Under
    assert under.p_win == pytest.approx(0.5 + 0.1)


def test_value_side_returns_none_without_a_table_price_or_known_side():
    assert adjust.value_side("spread", "home", FAIR, -2.5, -2.5, -110, None) is None
    assert adjust.value_side("spread", "home", FAIR, -2.5, -2.5, None, NFL) is None
    assert adjust.value_side("spread", "sideways", FAIR, -2.5, -2.5, -110, NFL) is None
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_adjust.py`
Expected: collection ERROR, `ImportError: cannot import name 'adjust' from 'sportsmodel.early'`.

- [ ] **Step 3: Write the implementation**

Create `src/sportsmodel/early/adjust.py`:

````python
"""Line adjustment and EV: value a sportsbook's number using Pinnacle's fair probability at Pinnacle's number. PURE.

Everything is done in "table space" (see landing.py): a side wins when the integer result R beats a threshold c,
`R > c` for direction +1 (home spread, Over) and `R < c` for direction -1 (away spread, Under), pushes when R == c
(whole-number thresholds only). For a spread the threshold is minus the home team's betting line, for a total it is the
total, so both sides of one market at one book share one threshold and moving a bet from Pinnacle's number to a book's
is the same operation for every market and side.

Pinnacle's two-way no-vig probability is conditional on no push, so at Pinnacle's threshold
    P_win = fair * (1 - P_push),  P_loss = (1 - fair) * (1 - P_push),  P_push from the landing table.
Moving to the book's threshold re-classifies the table mass of every integer result whose outcome changes (the old push
value becomes a win or a loss, the new push value becomes the push); the three probabilities still sum to 1.
EV (1 unit staked) = P_win * (decimal - 1) - P_loss; a push returns the stake and contributes 0.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..serving.board import decimal_odds
from .constants import FAIR_MAX, FAIR_MIN, FLAG_EV_EPS, FLAG_EV_MIN, GRID_EPS, MAX_LINE_GAP
from .landing import Pmf, on_grid

DIRECTION = {("spread", "home"): 1, ("spread", "away"): -1, ("total", "over"): 1, ("total", "under"): -1}
NEGATIVE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class Valuation:
    p_win: float
    p_push: float
    p_loss: float
    ev: float


def direction(market: str, side: str) -> int | None:
    return DIRECTION.get((market, side))


def threshold(market: str, side: str, line) -> float | None:
    """Table-space threshold for a side's own line: minus the line for the home spread, the line itself for the away
    spread and for totals (the away team's line is minus the home team's). None for an unknown market / side."""
    if line is None or (market, side) not in DIRECTION:
        return None
    line = float(line)
    return (-line if side == "home" else line) if market == "spread" else line


def own_line(market: str, side: str, c: float) -> float:
    """Inverse of `threshold`: the side's own line for a table-space threshold."""
    return (-c if side == "home" else c) if market == "spread" else c


def _classify(r: int, d: int, c: float) -> int:
    x = (r - c) * d
    return 1 if x > 0 else (-1 if x < 0 else 0)


def ev_of(p_win: float, p_loss: float, price: int) -> float:
    """EV per unit staked at an American price; a push contributes 0."""
    return p_win * (decimal_odds(price) - 1.0) - p_loss


def is_flag(ev: float | None) -> bool:
    return ev is not None and ev + FLAG_EV_EPS >= FLAG_EV_MIN


def fair_in_range(fair: float) -> bool:
    return FAIR_MIN <= fair <= FAIR_MAX


def adjust(fair: float, d: int, cp: float, cb: float, pmf: Pmf, *, cap: float | None = MAX_LINE_GAP,
           strict: bool = True) -> tuple[float, float, float] | None:
    """(P_win, P_push, P_loss) at threshold `cb` for a side whose Pinnacle no-vig probability is `fair` at threshold `cp`.

    None when either threshold is off the half-point grid, when |cb - cp| exceeds `cap` (None = no cap), or, with
    strict=True, when the table says the move is bigger than the probability available (a negative component). With
    strict=False a negative component is clamped to 0 and the three are renormalised (used when grading a ticket against
    a close that moved a long way: that must count as a loss, not be dropped)."""
    if not on_grid(cp) or not on_grid(cb):
        return None
    if cap is not None and abs(cb - cp) > cap + GRID_EPS:
        return None
    push = pmf.mass(cp) if abs(cp - round(cp)) < GRID_EPS else 0.0
    bucket = {1: fair * (1.0 - push), 0: push, -1: (1.0 - fair) * (1.0 - push)}
    for r in range(math.floor(min(cp, cb)) - 1, math.ceil(max(cp, cb)) + 2):
        before, after = _classify(r, d, cp), _classify(r, d, cb)
        if before != after:
            mass = pmf.mass(r)
            bucket[before] -= mass
            bucket[after] += mass
    if min(bucket.values()) < -NEGATIVE_TOLERANCE:
        if strict:
            return None
    clamped = {k: max(v, 0.0) for k, v in bucket.items()}
    total = sum(clamped.values())
    if total <= 0:
        return None
    return clamped[1] / total, clamped[0] / total, clamped[-1] / total


def value_side(market: str, side: str, fair: float, pinnacle_line, book_line, price: int, pmf: Pmf | None, *,
               cap: float | None = MAX_LINE_GAP, strict: bool = True) -> Valuation | None:
    """Valuation of `side` at the book's line and price from Pinnacle's fair for that side at Pinnacle's line (both
    lines are the side's own). None when the table has no answer or the move is not allowed."""
    d = direction(market, side)
    cp, cb = threshold(market, side, pinnacle_line), threshold(market, side, book_line)
    if d is None or cp is None or cb is None or pmf is None or price is None:
        return None
    moved = adjust(fair, d, cp, cb, pmf, cap=cap, strict=strict)
    if moved is None:
        return None
    p_win, p_push, p_loss = moved
    return Valuation(p_win, p_push, p_loss, ev_of(p_win, p_loss, int(price)))
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early/test_adjust.py`
Expected: 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/early/adjust.py tests/early/test_adjust.py
git commit -m "feat(early): line adjustment from Pinnacle's number to a book's, and EV" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The flagging engine (pure)

**Files:**
- Create: `src/sportsmodel/early/flags.py`
- Create: `tests/early/builders.py` (shared test builders, used by Tasks 4, 5, 7, 8 and 9), `tests/early/test_flags.py`

**Interfaces:**
- Consumes: `adjust.value_side`, `adjust.threshold`, `adjust.is_flag`, `adjust.fair_in_range`, `adjust.MAX_LINE_GAP`; `landing.LandingTable` (`.pmf(center)`); `constants.LINE_EPS`, `MARKETS`, `MIN_HOURS_TO_KICK`, `PINNACLE`, `SIDES`, `US_BOOKS`; `ingest.odds.parse_commence`; `serving.board.novig`.
- Produces:
  - `flags.Reference(threshold, fair, line)` (frozen dataclass; `fair` and `line` are `{side: float}`)
  - `flags.pinnacle_reference(rows, market) -> Reference | None` (rows = one game's odds_snapshot-shaped dicts for one market, any books)
  - `flags.best_offer(rows, market, side, ref, table, *, cap=MAX_LINE_GAP) -> (book, line, price, Valuation) | None`
  - `flags.flag_poll(rows, *, sport, observed_at, tables, existing, poll_run_id) -> list[dict]`: new tickets, each exactly the keys of `db.EARLY_COLUMNS["early_tickets"]` (`sport, game_pk, market, side, book, book_line, price, pinnacle_line, pinnacle_fair, p_win, p_push, p_loss, ev, observed_at, commence_time, hours_to_kick, poll_run_id`), ordered by (game_pk, market, side order); `tables` = `landing.load_tables()` (or any `{(sport, market): object with .pmf(center)}`), `existing` = `{(sport, game_pk, market, side)}`
  - test helpers in `tests/early/builders.py`: `NOW`, `StubTable`, `mass_at`, `iso`, `row`, `two_way`, `total_rows`, and (used from Task 8) `res`, `many`, `totals`, `spreads`

- [ ] **Step 1: Write the shared test builders and the failing tests**

Create `tests/early/builders.py`:

````python
"""Shared builders for the early-monitor tests: stub landing tables, odds_snapshot-shaped rows, graded results."""
from datetime import datetime, timezone

from sportsmodel.early.landing import Pmf, on_grid

NOW = datetime(2026, 10, 13, 16, 0, tzinfo=timezone.utc)      # a Tuesday poll


class StubTable:
    """A landing table answering the same Pmf for every on-grid line (None off the grid), like LandingTable."""

    def __init__(self, pmf: Pmf):
        self._pmf = pmf

    def pmf(self, center):
        return self._pmf if on_grid(center) else None


def mass_at(values, m: float) -> Pmf:
    """m of the mass on each of the given integer results, the rest split between two far-away results (never
    between the lines the tests move across)."""
    values = (values,) if isinstance(values, int) else tuple(values)
    rest = (1 - m * len(values)) / 2
    return Pmf(n=1000, window=1.0, probs={**{v: m for v in values}, min(values) - 15: rest, max(values) + 15: rest})


def iso(dt: datetime) -> str:
    return dt.isoformat().replace("+00:00", "Z")


def row(game_pk, market, side, book, line, price, kick, *, captured_at=NOW, player_name=""):
    return {"game_pk": game_pk, "market": market, "side": side, "player_name": player_name, "book": book,
            "line": line, "price": price, "captured_at": captured_at.isoformat(), "commence_time": iso(kick)}


def two_way(game_pk, market, book, line_a, price_a, line_b, price_b, kick, sides):
    return [row(game_pk, market, sides[0], book, line_a, price_a, kick),
            row(game_pk, market, sides[1], book, line_b, price_b, kick)]


def total_rows(game_pk, kick, pin_line=55.5, pin=(-110, -110), books=None):
    """Pinnacle's Over/Under at pin_line plus {book: (line, over_price, under_price)} for the user's books."""
    out = two_way(game_pk, "total", "pinnacle", pin_line, pin[0], pin_line, pin[1], kick, ("over", "under"))
    for book, (line, over, under) in (books or {}).items():
        out += two_way(game_pk, "total", book, line, over, line, under, kick, ("over", "under"))
    return out


def res(sport="cfb", market="total", game_pk=1, ev=0.05, beat=None, avail=True, kind="pinnacle_novig", result="W", units=0.9):
    return {"sport": sport, "market": market, "game_pk": game_pk, "close_ev": ev, "close_kind": kind,
            "beat_close": (None if ev is None else ev > 0) if beat is None else beat, "available_next_poll": avail, "result": result, "units": units}


def many(n, **kw):
    """n tickets in n different games."""
    base = kw.pop("game_start", 1000)
    return [res(game_pk=base + i, **kw) for i in range(n)]


def totals(n=60, **kw):
    return many(n, sport="cfb", market="total", game_start=1000, **kw)


def spreads(n=45, **kw):
    return many(n, sport="cfb", market="spread", game_start=2000, **kw) + many(n, sport="nfl", market="spread", game_start=3000, **kw)
````

Create `tests/early/test_flags.py`:

````python
"""flags.py: Pinnacle's reference, the best book per side, and every filter of the flag rule. Pure."""
from datetime import timedelta

import pytest

from sportsmodel.early import flags
from sportsmodel.early.constants import FLAG_EV_MIN

from .builders import NOW, StubTable, mass_at, row, total_rows, two_way

# A game 72 hours away; Pinnacle totals 55.5 at -110/-110 (fair 50%); a book at 54.5 gains the table mass on 55.
KICK = NOW + timedelta(hours=72)
M_FLAGS = 0.0318          # EV of Over 54.5 -110 = -0.04545 + 1.909 * m = +1.525%: just above the 1.5% line
M_SHORT = 0.0314          # ... and +1.449%: just below


def tables(m=M_FLAGS):
    return {("cfb", "total"): StubTable(mass_at((55, 56), m)), ("cfb", "spread"): StubTable(mass_at(3, m)),
            ("nfl", "spread"): StubTable(mass_at(3, m))}


def poll(rows, *, sport="cfb", existing=None, m=M_FLAGS, observed_at=NOW):
    return flags.flag_poll(rows, sport=sport, observed_at=observed_at, tables=tables(m), existing=existing or set(),
                           poll_run_id="run-1")


def good_rows(**books):
    return total_rows(1, KICK, books=books or {"draftkings": (54.5, -110, -110)})


def test_the_main_case_makes_one_over_ticket_with_every_stored_field():
    (t,) = poll(good_rows())
    assert (t["sport"], t["game_pk"], t["market"], t["side"], t["book"]) == ("cfb", 1, "total", "over", "draftkings")
    assert (t["book_line"], t["price"], t["pinnacle_line"]) == (54.5, -110, 55.5)
    assert t["pinnacle_fair"] == pytest.approx(0.5)
    assert t["p_win"] == pytest.approx(0.5 + M_FLAGS) and t["p_push"] == 0.0 and t["p_loss"] == pytest.approx(0.5 - M_FLAGS)
    assert t["ev"] == pytest.approx(-0.04545454 + 1.9090909 * M_FLAGS, abs=1e-6) and t["ev"] >= FLAG_EV_MIN
    assert t["observed_at"] == NOW and t["commence_time"] == KICK and t["hours_to_kick"] == pytest.approx(72.0)
    assert t["poll_run_id"] == "run-1"


def test_ticket_fields_are_exactly_the_early_tickets_columns():
    from sportsmodel import db
    (t,) = poll(good_rows())
    assert set(t) == set(db.EARLY_COLUMNS["early_tickets"])


def test_the_flag_threshold_is_one_and_a_half_percent_ev():
    assert len(poll(good_rows(), m=M_FLAGS)) == 1
    assert poll(good_rows(), m=M_SHORT) == []


def test_the_under_side_of_a_higher_book_total_flags_too():
    (t,) = poll(total_rows(1, KICK, books={"fanduel": (56.5, -110, -110)}))
    assert (t["side"], t["book"], t["book_line"]) == ("under", "fanduel", 56.5) and t["pinnacle_line"] == 55.5


def test_kickoff_must_be_at_least_24_hours_after_the_poll():
    assert len(poll(total_rows(1, NOW + timedelta(hours=24), books={"draftkings": (54.5, -110, -110)}))) == 1
    assert poll(total_rows(1, NOW + timedelta(hours=24) - timedelta(seconds=1),
                           books={"draftkings": (54.5, -110, -110)})) == []


def test_no_pinnacle_two_way_in_this_poll_means_no_ticket():
    rows = [r for r in good_rows() if r["book"] != "pinnacle"]
    assert poll(rows) == []
    one_leg = [r for r in good_rows() if not (r["book"] == "pinnacle" and r["side"] == "under")]
    assert poll(one_leg) == []
    wrong_line = [dict(r, line=56.5) if (r["book"] == "pinnacle" and r["side"] == "under") else r for r in good_rows()]
    assert poll(wrong_line) == []


def test_pinnacle_fair_outside_15_to_85_percent_means_no_ticket_for_either_side():
    lopsided = total_rows(1, KICK, pin=(-800, 550), books={"draftkings": (54.5, -110, -110), "fanduel": (56.5, 300, -110)})
    assert poll(lopsided) == []
    inside = total_rows(1, KICK, pin=(-600, 450), books={"draftkings": (54.5, -110, -110)})
    assert flags.pinnacle_reference(inside, "total").fair["over"] == pytest.approx(0.8251, abs=1e-3)
    assert poll(inside)                                   # 0.825 is inside the range: tickets are allowed


def test_the_highest_ev_book_wins_and_ties_go_to_the_first_book_name():
    rows = good_rows(fanduel=(54.5, -105, -110), draftkings=(54.5, -110, -110), betmgm=(54.5, -105, -110))
    (t,) = poll(rows)
    assert t["book"] == "betmgm" and t["price"] == -105           # better price, and betmgm < fanduel alphabetically
    tie = good_rows(fanduel=(54.5, -110, -110), draftkings=(54.5, -110, -110))
    assert poll(tie)[0]["book"] == "draftkings"


def test_only_the_users_nine_books_can_hold_a_ticket():
    rows = good_rows(bovada=(53.5, 150, -110), betonlineag=(53.5, 150, -110))
    assert poll(rows) == []                                      # offshore books: never
    pinnacle_only = total_rows(1, KICK)
    assert poll(pinnacle_only) == []                              # Pinnacle itself is the reference, not a bet


def test_a_book_more_than_three_points_from_pinnacle_is_a_data_error_not_a_ticket():
    assert poll(good_rows(draftkings=(51.5, -110, -110))) == []
    assert len(poll(good_rows(draftkings=(52.5, -110, -110)))) == 1        # exactly 3.0 points is allowed


def test_one_ticket_per_side_ever_later_polls_never_add_a_second():
    first = poll(good_rows())
    assert len(first) == 1
    existing = {(t["sport"], t["game_pk"], t["market"], t["side"]) for t in first}
    assert poll(good_rows(), existing=existing) == []
    # a different side of the same market is still free to flag
    both = good_rows(draftkings=(54.5, -110, -110), fanduel=(56.5, -110, -110))
    assert {t["side"] for t in poll(both)} == {"over", "under"}
    assert {t["side"] for t in poll(both, existing=existing)} == {"under"}


def test_only_the_tested_markets_are_flagged():
    nfl_totals = [dict(r, game_pk=2) for r in good_rows()]
    assert poll(nfl_totals, sport="nfl") == []                     # NFL totals are excluded
    moneyline = [row(3, "moneyline", "home", "pinnacle", None, -150, KICK), row(3, "moneyline", "away", "pinnacle", None, 130, KICK),
                 row(3, "moneyline", "home", "draftkings", None, -120, KICK), row(3, "moneyline", "away", "draftkings", None, 130, KICK)]
    assert poll(moneyline) == [] and poll(moneyline, sport="nfl") == []
    props = [dict(r, player_name="A. Player") for r in good_rows()]
    assert poll(props) == []


def test_spread_sides_use_each_sides_own_lines():
    rows = (two_way(7, "spread", "pinnacle", -3.5, -110, 3.5, -110, KICK, ("home", "away"))
            + two_way(7, "spread", "betmgm", -2.5, -110, 2.5, -110, KICK, ("home", "away")))
    (t,) = poll(rows, sport="nfl")                                   # home -2.5 vs Pinnacle -3.5 gains the margin-3 mass
    assert (t["side"], t["book"], t["book_line"], t["pinnacle_line"]) == ("home", "betmgm", -2.5, -3.5)
    away = (two_way(7, "spread", "pinnacle", -2.5, -110, 2.5, -110, KICK, ("home", "away"))
            + two_way(7, "spread", "betmgm", -3.5, -110, 3.5, -110, KICK, ("home", "away")))
    (t,) = poll(away, sport="nfl")                                   # away +3.5 vs +2.5 gains the margin-3 mass
    assert (t["side"], t["book_line"], t["pinnacle_line"]) == ("away", 3.5, 2.5)


def test_pinnacle_with_several_quoted_pairs_uses_the_one_closest_to_even():
    rows = (two_way(1, "total", "pinnacle", 55.5, -110, 55.5, -110, KICK, ("over", "under"))
            + two_way(1, "total", "pinnacle", 52.5, -180, 52.5, 150, KICK, ("over", "under")))
    ref = flags.pinnacle_reference(rows, "total")
    assert ref.threshold == 55.5 and ref.fair["over"] == pytest.approx(0.5)


def test_quarter_point_lines_are_never_rounded_into_a_ticket():
    assert poll(good_rows(draftkings=(54.25, -110, -110))) == []
    quarter_pin = total_rows(1, KICK, pin_line=55.75, books={"draftkings": (54.5, -110, -110)})
    assert poll(quarter_pin) == []


def test_a_game_with_no_parseable_kickoff_or_no_table_is_skipped():
    rows = [dict(r, commence_time=None) for r in good_rows()]
    assert poll(rows) == []
    assert flags.flag_poll(good_rows(), sport="cfb", observed_at=NOW, tables={}, existing=set(), poll_run_id="r") == []


def test_output_is_deterministic_and_ordered_by_game():
    rows = [dict(r, game_pk=pk) for pk in (9, 4) for r in good_rows()]
    assert [t["game_pk"] for t in poll(rows)] == [4, 9]
    assert poll(rows) == poll(list(reversed(rows)))


def test_a_book_posting_alternate_lines_is_valued_at_each_line_and_the_best_one_wins():
    rows = good_rows(draftkings=(54.5, -110, -110)) + two_way(1, "total", "draftkings", 53.5, -150, 53.5, 125, KICK, ("over", "under"))
    (t,) = [t for t in poll(rows) if t["side"] == "over"]
    assert (t["book_line"], t["price"]) == (54.5, -110)               # 53.5 at -150 is worse than 54.5 at -110 here
    cheap_alt = good_rows(draftkings=(54.5, -110, -110)) + two_way(1, "total", "draftkings", 53.5, 120, 53.5, -140, KICK, ("over", "under"))
    (t,) = [t for t in poll(cheap_alt) if t["side"] == "over"]
    assert (t["book_line"], t["price"]) == (53.5, 120)


def test_a_line_outside_the_committed_landing_tables_makes_no_ticket():
    from sportsmodel.early import landing
    real = landing.load_tables()
    heavy = (two_way(5, "spread", "pinnacle", -38.5, -110, 38.5, -110, KICK, ("home", "away"))
             + two_way(5, "spread", "draftkings", -37.5, -110, 37.5, -110, KICK, ("home", "away")))
    assert flags.flag_poll(heavy, sport="cfb", observed_at=NOW, tables=real, existing=set(), poll_run_id="r") == []
    ordinary = (two_way(5, "spread", "pinnacle", -3.5, -110, 3.5, -110, KICK, ("home", "away"))
                + two_way(5, "spread", "draftkings", -2.5, -110, 2.5, -110, KICK, ("home", "away")))
    got = flags.flag_poll(ordinary, sport="cfb", observed_at=NOW, tables=real, existing=set(), poll_run_id="r")
    assert all(t["market"] == "spread" for t in got)               # an ordinary CFB line is inside the table (it may or may not flag)
    assert real[("cfb", "spread")].pmf(3.5) is not None and real[("cfb", "spread")].pmf(38.5) is None
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_flags.py`
Expected: collection ERROR, `ImportError: cannot import name 'flags' from 'sportsmodel.early'`.

- [ ] **Step 3: Write the implementation**

Create `src/sportsmodel/early/flags.py`:

````python
"""The flagging engine: one poll's prices -> new paper tickets. PURE -- no IO.

For every game and market in the test, Pinnacle's two-way quote in THIS poll gives the fair probability at Pinnacle's
line; every one of the user's nine US books is then valued at its own line and price (adjust.value_side); the highest-EV
book per side becomes a ticket when its EV is at least +1.5% and kickoff is at least 24 hours away. A side that already
has a ticket (any earlier poll) never gets a second one.

`rows` are odds_snapshot-shaped dicts (game_pk, market, side, book, line, price, commence_time, ...) as produced by
`ingest.odds.parse_game_odds`.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from ..ingest.odds import parse_commence
from ..serving.board import novig
from . import adjust
from .constants import LINE_EPS, MARKETS, MIN_HOURS_TO_KICK, PINNACLE, SIDES, US_BOOKS
from .landing import LandingTable


@dataclass(frozen=True)
class Reference:
    """Pinnacle's two-way quote for one game and market: the table-space threshold, and per side the no-vig fair
    probability and Pinnacle's own line for that side."""
    threshold: float
    fair: dict[str, float]
    line: dict[str, float]


def _offers(rows: list[dict], book: str, side: str) -> list[tuple[float, int]]:
    return [(float(r["line"]), int(r["price"])) for r in rows
            if r.get("book") == book and r.get("side") == side and r.get("line") is not None and r.get("price") is not None]


def pinnacle_reference(rows: list[dict], market: str) -> Reference | None:
    """Pinnacle's two-way quote from `rows` (one game, one market, any books), or None without a complete pair.

    A pair is a spread's home and away quotes at mirrored lines, or a total's Over and Under at the same line. If
    Pinnacle shows several pairs, the one whose fair is closest to 50% (the main line) wins; a remaining tie goes to the
    lower threshold, so the choice is deterministic."""
    first, second = SIDES[market]
    pairs = []
    for l1, p1 in _offers(rows, PINNACLE, first):
        for l2, p2 in _offers(rows, PINNACLE, second):
            mirrored = abs(l1 + l2) <= LINE_EPS if market == "spread" else abs(l1 - l2) <= LINE_EPS
            if mirrored:
                fair_first = novig(p1, p2)
                pairs.append((abs(fair_first - 0.5), adjust.threshold(market, first, l1), fair_first, l1, l2))
    if not pairs:
        return None
    _, c, fair_first, l1, l2 = min(pairs, key=lambda t: (round(t[0], 12), t[1]))
    return Reference(threshold=c, fair={first: fair_first, second: 1.0 - fair_first}, line={first: l1, second: l2})


def best_offer(rows: list[dict], market: str, side: str, ref: Reference, table: LandingTable, *, cap=adjust.MAX_LINE_GAP):
    """The highest-EV (book, line, price, valuation) among the user's US books for `side`, or None. Ties on EV go to the
    alphabetically first book, then the lower line, so results are deterministic."""
    pmf = table.pmf(ref.threshold)
    best = None
    for book in sorted(US_BOOKS):
        for line, price in sorted(_offers(rows, book, side)):
            v = adjust.value_side(market, side, ref.fair[side], ref.line[side], line, price, pmf, cap=cap)
            if v is None:
                continue
            key = (-round(v.ev, 12), book, line)
            if best is None or key < best[0]:
                best = (key, book, line, price, v)
    return None if best is None else best[1:]


def flag_poll(rows: list[dict], *, sport: str, observed_at: datetime, tables: dict, existing: set[tuple],
              poll_run_id: str) -> list[dict]:
    """New tickets for one sport's poll. `tables` is `landing.load_tables()`; `existing` the set of
    (sport, game_pk, market, side) that already have a ticket."""
    by_game_market: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("market") in MARKETS.get(sport, ()) and not r.get("player_name"):
            by_game_market[(r["game_pk"], r["market"])].append(r)
    tickets: list[dict] = []
    for (game_pk, market), group in sorted(by_game_market.items()):
        table = tables.get((sport, market))
        kick = next((k for k in (parse_commence(r.get("commence_time")) for r in group) if k is not None), None)
        if table is None or kick is None:
            continue
        hours = (kick - observed_at).total_seconds() / 3600.0
        if hours < MIN_HOURS_TO_KICK:
            continue
        ref = pinnacle_reference(group, market)
        if ref is None or not adjust.fair_in_range(ref.fair[SIDES[market][0]]):
            continue
        for side in SIDES[market]:
            if (sport, game_pk, market, side) in existing:
                continue
            found = best_offer(group, market, side, ref, table)
            if found is None or not adjust.is_flag(found[3].ev):
                continue
            book, line, price, v = found
            tickets.append({
                "sport": sport, "game_pk": game_pk, "market": market, "side": side, "book": book, "book_line": line,
                "price": price, "pinnacle_line": ref.line[side], "pinnacle_fair": ref.fair[side], "p_win": v.p_win,
                "p_push": v.p_push, "p_loss": v.p_loss, "ev": v.ev, "observed_at": observed_at, "commence_time": kick,
                "hours_to_kick": hours, "poll_run_id": poll_run_id})
    return tickets
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early/test_flags.py tests/early/test_adjust.py`
Expected: 19 + 19 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/early/flags.py tests/early/builders.py tests/early/test_flags.py
git commit -m "feat(early): flagging engine turns one poll's prices into paper tickets" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The hourly monitor script (`early_monitor.py`)

**Files:**
- Create: `src/sportsmodel/early/polling.py`
- Modify: `src/sportsmodel/ingest/odds.py` (add `fetch_bookmaker_odds` before `fetch_event_props`)
- Create: `scripts/early_monitor.py`
- Create: `tests/early/test_polling.py`, `tests/early/test_early_monitor_script.py`

**Interfaces:**
- Consumes: Task 1 `db.missing_early_tables`, `db.early_ticket_keys`, `db.insert_early_tickets`, `db.LEDGER_CONNECT_TIMEOUT` / `db.LEDGER_STATEMENT_TIMEOUT_MS` (existing), `db.upsert_odds_snapshot` (existing); Task 2 `landing.load_tables`, `constants.*`; Task 4 `flags.flag_poll`; existing `scripts/ingest_odds.py` (`build_game_lookup`, `_ESPN_FETCHERS`, `_matcher_for`), `ingest.odds` (`fetch_events`, `parse_game_odds`, `last_requests_remaining`, `last_requests_last`), `sports.get`, `ledger.record.writes_disabled`.
- Produces:
  - `polling.bookmakers_param() -> str`, `polling.api_markets(sport) -> list[str]`, `polling.regions_billed(n_books=10) -> int`, `polling.poll_cost(sport) -> int` (CFB 2, NFL 1), `polling.parse_credits(value) -> int | None`, `polling.BudgetDecision(ok, warning=None)`, `polling.budget_decision(remaining) -> BudgetDecision` (below 1,500 not ok; unknown ok)
  - `odds.fetch_bookmaker_odds(cfg, bookmakers: list[str], markets: list[str]) -> list[dict]`
  - `early_monitor.Deps` (dataclass of injectable IO), `early_monitor.run_poll(now, deps, *, run_id="local", dry_run=False) -> dict`, `early_monitor._db_deps()`, `early_monitor.main(argv=None) -> int`

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_polling.py`:

````python
"""polling.py and ingest.odds.fetch_bookmaker_odds: the bookmakers list, cost per poll and the credit floor. No network."""
import pytest

from sportsmodel import sports
from sportsmodel.early import polling
from sportsmodel.early.constants import API_MARKETS, BOOKMAKERS, MONITOR_FLOOR, US_BOOKS
from sportsmodel.ingest import odds


def test_the_ten_bookmaker_keys():
    assert BOOKMAKERS == ("draftkings", "fanduel", "betmgm", "williamhill_us", "espnbet", "fanatics", "hardrockbet",
                          "ballybet", "betrivers", "pinnacle")
    assert len(BOOKMAKERS) == 10 == len(set(BOOKMAKERS)) and set(BOOKMAKERS) - set(US_BOOKS) == {"pinnacle"}
    assert polling.bookmakers_param() == ",".join(BOOKMAKERS)


def test_cost_per_poll_is_one_credit_per_market_because_ten_books_bill_as_one_region():
    assert polling.regions_billed() == 1 and polling.regions_billed(11) == 2 and polling.regions_billed(1) == 1
    assert polling.api_markets("cfb") == ["spreads", "totals"] and polling.poll_cost("cfb") == 2
    assert polling.api_markets("nfl") == ["spreads"] and polling.poll_cost("nfl") == 1
    assert polling.poll_cost("cfb") + polling.poll_cost("nfl") == 3           # about 3 credits a poll
    assert 60 * 3 == 180                                                       # 60 polls a week: about 180 credits


def test_the_credit_floor_boundary_is_1500():
    assert MONITOR_FLOOR == 1500
    assert polling.budget_decision(1500).ok is True and polling.budget_decision("1500").ok is True
    low = polling.budget_decision(1499)
    assert low.ok is False and "1499" in low.warning and "1500" in low.warning
    assert polling.budget_decision("1499.0").ok is False and polling.budget_decision(0).ok is False


@pytest.mark.parametrize("unknown", [None, "", "n/a", float("nan"), []])
def test_an_unknown_remaining_value_runs_the_first_call(unknown):
    assert polling.budget_decision(unknown) == polling.BudgetDecision(True)


def test_fetch_bookmaker_odds_sends_bookmakers_and_never_regions(monkeypatch):
    seen = {}
    monkeypatch.setattr(odds, "_get", lambda path, params: seen.update(path=path, params=params) or [])
    assert odds.fetch_bookmaker_odds(sports.get("cfb"), list(BOOKMAKERS), ["spreads", "totals"]) == []
    assert seen["path"] == "/sports/americanfootball_ncaaf/odds"
    assert seen["params"] == {"bookmakers": polling.bookmakers_param(), "markets": "spreads,totals",
                              "oddsFormat": "american", "dateFormat": "iso"}
    assert "regions" not in seen["params"]
    odds.fetch_bookmaker_odds(sports.get("nfl"), list(BOOKMAKERS), polling.api_markets("nfl"))
    assert seen["path"] == "/sports/americanfootball_nfl/odds" and seen["params"]["markets"] == "spreads"


def test_every_polled_sport_has_a_market_list_and_a_sport_config():
    for sport in API_MARKETS:
        assert sports.get(sport).odds_sport
````

Create `tests/early/test_early_monitor_script.py`:

````python
"""scripts/early_monitor.py end to end on injected IO: bookmakers fetch, the credit floor, snapshot storage, incremental
ticket writes, failure isolation, dry run and the guards. No network, no database."""
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from .builders import NOW, StubTable, iso, mass_at

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("early_monitor", ROOT / "scripts" / "early_monitor.py")
em = importlib.util.module_from_spec(_spec)
sys.modules["early_monitor"] = em            # @dataclass needs its module registered
_spec.loader.exec_module(em)

KICK = NOW + timedelta(hours=72)
M = 0.0318


def schedule_game(pk, home="Home", away="Away", kick=KICK):
    return {"game_pk": pk, "home_name": home, "away_name": away, "commence_time": iso(kick)}


def outcomes_total(line, over, under):
    return [{"name": "Over", "point": line, "price": over}, {"name": "Under", "point": line, "price": under}]


def total_event(eid, home="Home", away="Away", kick=KICK, pin_line=55.5, dk_line=54.5):
    def book(key, line, over=-110, under=-110):
        return {"key": key, "markets": [{"key": "totals", "outcomes": outcomes_total(line, over, under)}]}
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": iso(kick),
            "bookmakers": [book("pinnacle", pin_line), book("draftkings", dk_line), book("bovada", 50.5)]}


def spread_event(eid, home="NFL Home", away="NFL Away", kick=KICK):
    def book(key, home_line):
        return {"key": key, "markets": [{"key": "spreads", "outcomes": [
            {"name": home, "point": home_line, "price": -110}, {"name": away, "point": -home_line, "price": -110}]}]}
    return {"id": eid, "home_team": home, "away_team": away, "commence_time": iso(kick),
            "bookmakers": [book("pinnacle", -3.5), book("betmgm", -2.5)]}


TABLES = {("cfb", "total"): StubTable(mass_at((55, 56), M)), ("cfb", "spread"): StubTable(mass_at(3, M)),
          ("nfl", "spread"): StubTable(mass_at(3, M))}


class Fake:
    """Injected IO that records every call and every store, in order."""

    def __init__(self, **over):
        self.log, self.snaps, self.tickets = [], [], []
        self.remaining_value = over.pop("remaining", "20000")
        self.cost = over.pop("cost", {"cfb": "2", "nfl": "1"})
        self.schedule = over.pop("schedule", {"cfb": [schedule_game(1)], "nfl": [schedule_game(2, "NFL Home", "NFL Away")]})
        self.events = over.pop("events", {"cfb": [total_event("c1")], "nfl": [spread_event("n1")]})
        self.existing = over.pop("existing", set())
        self.fail = over.pop("fail", {})                    # sport -> exception raised by the odds call
        self.fail_store = over.pop("fail_store", {})        # "snaps" / "tickets" -> exception
        self.fail_prime = over.pop("fail_prime", None)
        self.prime_remaining = over.pop("prime_remaining", None)
        self.missing = over.pop("missing", [])
        assert not over, over

    def deps(self):
        state = {"remaining": self.prime_remaining if self.prime_remaining is not None else None, "cost": None}

        def prime(cfg):
            self.log.append(("prime", cfg.key))
            if self.fail_prime:
                raise self.fail_prime
            if self.prime_remaining is not None:
                state["remaining"] = self.prime_remaining

        def fetch_odds(cfg, bookmakers, markets):
            self.log.append(("odds", cfg.key, tuple(bookmakers), tuple(markets)))
            if cfg.key in self.fail:
                raise self.fail[cfg.key]
            state["remaining"], state["cost"] = self.remaining_value, self.cost[cfg.key]
            return self.events[cfg.key]

        def store_snaps(rows):
            self.log.append(("snaps", len(rows)))
            if "snaps" in self.fail_store:
                raise self.fail_store["snaps"]
            self.snaps.extend(rows)
            return len(rows)

        def store_tickets(rows):
            self.log.append(("tickets", len(rows)))
            if "tickets" in self.fail_store:
                raise self.fail_store["tickets"]
            self.tickets.extend(rows)
            self.existing = self.existing | {(t["sport"], t["game_pk"], t["market"], t["side"]) for t in rows}
            return len(rows)

        return em.Deps(
            missing_tables=lambda: self.missing,
            load_existing=lambda sport, now: self.existing,
            load_tables=lambda: TABLES,
            fetch_schedule=lambda sport: self.log.append(("schedule", sport)) or self.schedule[sport],
            match=em.ingest_odds._matcher_for, prime=prime, fetch_odds=fetch_odds,
            remaining=lambda: state["remaining"], last_cost=lambda: state["cost"],
            store_snapshots=store_snaps, store_tickets=store_tickets)

    def calls(self, name):
        return [c for c in self.log if c[0] == name]


def run(fake, **kw):
    return em.run_poll(NOW, fake.deps(), run_id="run-7", **kw)


# ---- the poll itself --------------------------------------------------------------------------------------

def test_one_odds_call_per_sport_with_the_ten_bookmakers_and_the_right_markets():
    fake = Fake()
    out = run(fake)
    cfb, nfl = fake.calls("odds")
    assert cfb[1] == "cfb" and cfb[2] == tuple(em.polling.bookmakers_param().split(",")) and len(cfb[2]) == 10
    assert cfb[3] == ("spreads", "totals")
    assert nfl[1] == "nfl" and nfl[3] == ("spreads",)
    assert out["credits"] == 3 and out["status"] == "ok"


def test_the_free_events_call_primes_the_credit_header_once_before_any_paid_call():
    fake = Fake()
    run(fake)
    names = [c[0] for c in fake.log]
    assert names.index("prime") < names.index("odds") and names.count("prime") == 1


def test_snapshots_use_the_ingest_row_shape_for_the_tested_markets_and_listed_books_only():
    fake = Fake()
    run(fake)
    cfb = [r for r in fake.snaps if r["game_pk"] == 1]
    assert {r["book"] for r in cfb} == {"pinnacle", "draftkings"}                       # bovada was not asked for
    assert {r["market"] for r in cfb} == {"total"} and {r["captured_at"] for r in fake.snaps} == {NOW.isoformat()}
    row = next(r for r in cfb if r["book"] == "draftkings" and r["side"] == "over")
    assert set(row) == {"game_pk", "market", "side", "player_name", "book", "line", "price", "captured_at", "commence_time"}
    assert row["line"] == 54.5 and row["price"] == -110 and row["player_name"] == ""
    assert {r["market"] for r in fake.snaps if r["game_pk"] == 2} == {"spread"}


def test_a_ticket_is_logged_with_this_polls_run_id_and_the_matched_game_pk():
    fake = Fake()
    out = run(fake)
    (cfb,) = [t for t in fake.tickets if t["sport"] == "cfb"]
    assert (cfb["game_pk"], cfb["side"], cfb["book"], cfb["book_line"]) == (1, "over", "draftkings", 54.5)
    assert cfb["poll_run_id"] == "run-7" and cfb["observed_at"] == NOW
    (nfl,) = [t for t in fake.tickets if t["sport"] == "nfl"]
    assert (nfl["market"], nfl["side"], nfl["book"], nfl["book_line"]) == ("spread", "home", "betmgm", -2.5)
    assert out["tickets"] == 2 and out["snapshots"] == len(fake.snaps)


def test_events_that_match_no_game_are_dropped():
    fake = Fake(events={"cfb": [total_event("x", home="Nobody", away="Else")], "nfl": []})
    out = run(fake)
    assert fake.snaps == [] and fake.tickets == [] and out["sports"]["cfb"]["matched"] == 0


def test_a_second_run_in_the_same_hour_logs_no_new_ticket():
    fake = Fake()
    run(fake)
    first = list(fake.tickets)
    assert len(first) == 2
    run(fake)                                                         # a duplicated cron firing: same prices, same book
    assert fake.tickets == first


def test_tickets_that_already_exist_are_not_logged_again():
    fake = Fake(existing={("cfb", 1, "total", "over")})
    run(fake)
    assert [t for t in fake.tickets if t["sport"] == "cfb"] == []


# ---- credit floor ----------------------------------------------------------------------------------------

def test_at_1500_remaining_the_next_call_still_runs_at_1499_it_does_not(capsys):
    ok = Fake(remaining="1500")
    run(ok)
    assert [c[1] for c in ok.calls("odds")] == ["cfb", "nfl"]
    low = Fake(remaining="1499")
    out = run(low)
    assert [c[1] for c in low.calls("odds")] == ["cfb"] and out["skipped"] == ["nfl"]
    printed = capsys.readouterr().out
    assert "::warning::early-monitor: 1499 Odds API credits left (< 1500)" in printed


def test_an_unknown_first_value_runs_the_first_call_and_the_value_it_returns_guards_the_second():
    fake = Fake(remaining="900")                       # nothing known until the first paid response says 900
    out = run(fake)
    assert [c[1] for c in fake.calls("odds")] == ["cfb"] and out["credits"] == 2


def test_a_low_balance_read_by_the_free_call_stops_even_the_first_paid_call(capsys):
    fake = Fake(prime_remaining="1200")
    out = run(fake)
    assert fake.calls("odds") == [] and out["credits"] == 0 and out["skipped"] == ["cfb", "nfl"]
    assert "skipping its remaining calls" in capsys.readouterr().out


def test_if_the_free_call_fails_the_first_call_still_runs(capsys):
    fake = Fake(fail_prime=RuntimeError("events 500"))
    run(fake)
    assert [c[1] for c in fake.calls("odds")] == ["cfb", "nfl"]
    assert "credit check call failed" in capsys.readouterr().out


def test_credits_fall_back_to_markets_times_one_region_without_the_cost_header():
    fake = Fake(cost={"cfb": None, "nfl": None})
    assert run(fake)["credits"] == 3


# ---- incremental writes and failures -------------------------------------------------------------------------

def test_each_sports_snapshots_then_tickets_are_stored_before_the_next_sport_is_fetched():
    fake = Fake()
    run(fake)
    stores = [c[0] for c in fake.log if c[0] in ("odds", "snaps", "tickets")]
    assert stores == ["odds", "snaps", "tickets", "odds", "snaps", "tickets"]


def test_a_failure_in_the_second_sport_keeps_the_first_sports_tickets(capsys):
    fake = Fake(fail={"nfl": RuntimeError("odds api 500")})
    out = run(fake)
    assert [t["sport"] for t in fake.tickets] == ["cfb"] and out["sports"]["nfl"]["failed"].startswith("RuntimeError")
    assert "nfl poll failed" in capsys.readouterr().out


def test_a_failed_poll_is_not_retried_and_the_other_sport_goes_on():
    fake = Fake(fail={"cfb": RuntimeError("odds api 500")})
    out = run(fake)
    assert len(fake.calls("odds")) == 2 and [t["sport"] for t in fake.tickets] == ["nfl"]
    assert out["credits"] == 1


def test_a_snapshot_write_failure_does_not_stop_the_tickets():
    fake = Fake(fail_store={"snaps": RuntimeError("db hiccup")})
    out = run(fake)
    assert out["tickets"] == 2 and out["snapshots"] == 0


def test_an_interrupt_in_the_second_sport_still_leaves_the_first_sports_tickets_stored():
    fake = Fake(fail={"nfl": KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run(fake)
    assert [t["sport"] for t in fake.tickets] == ["cfb"]


def test_the_schedule_and_existing_ticket_reads_happen_before_credits_are_spent():
    fake = Fake()
    deps = fake.deps()
    deps.load_existing = lambda sport, now: (_ for _ in ()).throw(RuntimeError("db down"))
    out = em.run_poll(NOW, deps)
    assert fake.calls("odds") == [] and out["credits"] == 0


# ---- guards and dry run ------------------------------------------------------------------------------------

def test_missing_tables_warn_and_stop_before_any_network_call(capsys):
    fake = Fake(missing=["early_tickets"])
    assert run(fake) == {"status": "no_table"} and fake.log == []
    assert "::warning::early-monitor: table early_tickets missing" in capsys.readouterr().out


def test_dry_run_polls_but_writes_nothing_and_reports_what_it_would_log():
    fake = Fake()
    out = run(fake, dry_run=True)
    assert len(fake.calls("odds")) == 2 and fake.snaps == [] and fake.tickets == []
    assert fake.calls("snaps") == [] and fake.calls("tickets") == []
    assert [t["sport"] for t in out["would_write"]] == ["cfb", "nfl"] and out["tickets"] == 0


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert em.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_a_dry_run_is_also_refused_under_the_test_ledger_guard(monkeypatch, capsys):
    monkeypatch.setattr(em, "_db_deps", lambda: pytest.fail("must not build DB deps under the guard"))
    monkeypatch.setattr(em, "run_poll", lambda *a, **k: pytest.fail("must not poll under the guard"))
    assert em.main(["--dry-run"]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_main_passes_run_id_and_dry_run_through(monkeypatch, capsys):
    seen = {}
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)             # the root conftest sets the guard; this test exercises the run
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setenv("ODDS_API_KEY", "unused")
    monkeypatch.setenv("GITHUB_RUN_ID", "4242")
    monkeypatch.setattr(em, "_db_deps", lambda: object())
    monkeypatch.setattr(em, "run_poll", lambda now, deps, **kw: seen.update(now=now, **kw) or {"status": "ok", "would_write": [
        {"sport": "cfb", "game_pk": 1, "market": "total", "side": "over", "book": "draftkings", "book_line": 54.5,
         "price": -110, "ev": 0.0152, "hours_to_kick": 72.0}]})
    assert em.main(["--dry-run"]) == 0
    assert seen["run_id"] == "4242" and seen["dry_run"] is True and seen["now"].tzinfo is not None
    assert "DRY RUN would log: cfb 1 total over draftkings +54.5 -110 ev=+1.52%" in capsys.readouterr().out


def test_main_needs_both_secrets(monkeypatch, capsys):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    assert em.main([]) == 1 and "ODDS_API_KEY is not set" in capsys.readouterr().out
    monkeypatch.delenv("DATABASE_URL")
    monkeypatch.setenv("ODDS_API_KEY", "k")
    assert em.main([]) == 1


def test_production_wiring_uses_the_db_helpers_with_timeouts(monkeypatch):
    from sportsmodel import db
    seen = {}
    monkeypatch.setattr(db, "missing_early_tables", lambda **kw: seen.update(missing=kw) or [])
    monkeypatch.setattr(db, "early_ticket_keys", lambda sport, now, **kw: seen.update(keys=(sport, now, kw)) or set())
    monkeypatch.setattr(db, "insert_early_tickets", lambda rows, **kw: seen.update(insert=kw) or len(rows))
    deps = em._db_deps()
    deps.missing_tables()
    deps.load_existing("cfb", NOW)
    deps.store_tickets([{}])
    assert seen["missing"] == {"connect_timeout": 10, "statement_timeout_ms": 15_000}
    assert seen["keys"] == ("cfb", NOW, {"connect_timeout": 10, "statement_timeout_ms": 15_000})
    assert seen["insert"] == {"connect_timeout": 10, "statement_timeout_ms": 15_000}
    assert deps.store_snapshots is db.upsert_odds_snapshot
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_polling.py tests/early/test_early_monitor_script.py`
Expected: collection ERRORs: `ImportError: cannot import name 'polling' from 'sportsmodel.early'` and `FileNotFoundError` for `scripts/early_monitor.py`.

- [ ] **Step 3: Write the polling module**

Create `src/sportsmodel/early/polling.py`:

````python
"""What one monitor poll asks the Odds API for, what it costs, and when the credit floor stops it. PURE -- no IO.

The call uses the `bookmakers` parameter instead of `regions`: the 9 US books plus Pinnacle are 10 bookmakers, which the
Odds API bills as one region, so a poll costs one credit per market: 2 for CFB (spreads, totals), 1 for NFL (spreads).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .constants import API_MARKETS, BOOKMAKERS, MONITOR_FLOOR

BOOKMAKERS_PER_REGION = 10


def bookmakers_param() -> str:
    """The `bookmakers` query value: the 10 keys, comma separated."""
    return ",".join(BOOKMAKERS)


def api_markets(sport: str) -> list[str]:
    return list(API_MARKETS[sport])


def regions_billed(n_books: int = len(BOOKMAKERS)) -> int:
    return max(1, math.ceil(n_books / BOOKMAKERS_PER_REGION))


def poll_cost(sport: str) -> int:
    """Credits one poll of `sport` is billed: markets x regions."""
    return len(API_MARKETS[sport]) * regions_billed()


def parse_credits(value) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return None


@dataclass(frozen=True)
class BudgetDecision:
    ok: bool
    warning: str | None = None


def budget_decision(remaining) -> BudgetDecision:
    """May the monitor make its next paid call, given `x-requests-remaining` from the last response?

    Below MONITOR_FLOOR (1,500) it may not and says why. The value is unknown before the first response of a run (or
    when the header is missing or unreadable), and an unknown value runs: the first call always runs."""
    credits = parse_credits(remaining)
    if credits is None:
        return BudgetDecision(True)
    if credits < MONITOR_FLOOR:
        return BudgetDecision(False, f"{credits} Odds API credits left (< {MONITOR_FLOOR}): "
                                     "the early-week monitor is skipping its remaining calls")
    return BudgetDecision(True)
````

- [ ] **Step 4: Add the bookmakers fetch to the odds client**

Apply to `src/sportsmodel/ingest/odds.py` (adds one function; nothing existing changes):

````diff
--- a/src/sportsmodel/ingest/odds.py
+++ b/src/sportsmodel/ingest/odds.py
@@ -110,6 +110,16 @@
     })
 
 
+def fetch_bookmaker_odds(cfg: "SportConfig | None", bookmakers: list[str], markets: list[str]) -> list[dict]:
+    """Game-line odds for exactly the named bookmakers (the early-week monitor), using the `bookmakers` parameter in
+    place of `regions`. Ten or fewer bookmakers bill as ONE region, so the cost is one credit per market."""
+    cfg = cfg or _mlb()
+    return _get(f"/sports/{cfg.odds_sport}/odds", {
+        "bookmakers": ",".join(bookmakers), "markets": ",".join(markets),
+        "oddsFormat": "american", "dateFormat": "iso",
+    })
+
+
 def fetch_event_props(event_id: str, markets: list[str], cfg: "SportConfig | None" = None, regions: str = "us") -> dict:
     cfg = cfg or _mlb()
     return _get(f"/sports/{cfg.odds_sport}/events/{event_id}/odds", {
````

- [ ] **Step 5: Write the monitor script**

Create `scripts/early_monitor.py`:

````python
"""Hourly early-week poll: log paper tickets where a US book's number beats Pinnacle's fair (profit step 3).

Run hourly from Sunday 16:00 UTC to Wednesday 04:00 UTC in football season (.github/workflows/early-monitor.yml).
Each run, for CFB (spreads + totals, 2 credits) and then NFL (spreads, 1 credit):

  1. Credit floor: below 1,500 Odds API credits remaining the remaining calls are skipped with a warning. A free
     /events call first reads x-requests-remaining, so even the first paid call is guarded; if the value is unknown
     the call runs.
  2. ONE Odds API odds call per sport with `bookmakers` = the 9 US books + pinnacle (10 keys = one billed region).
  3. Events are matched to game_pk with the same matchers as scripts/ingest_odds.py; the rows go to odds_snapshot in
     the shape ingest_odds writes.
  4. The flagging engine (sportsmodel.early.flags) turns this poll's prices into new tickets: one per
     (sport, game_pk, market, side), best US book, EV >= +1.5% against Pinnacle's same-poll fair, kickoff >= 24 h away.
     Tickets are appended right after each sport, so a failure part-way keeps what was already found.

A failed poll is not retried (the next hourly run proceeds normally). Missing tables: warn and exit 0. Under the test
guard SM_LEDGER_OFF: warn and exit 0, --dry-run included. Writes only odds_snapshot and early_tickets.

    uv run python scripts/early_monitor.py [--dry-run]

--dry-run still makes the paid odds calls (about 3 credits) but writes nothing: it prints the tickets it would log.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

import ingest_odds  # noqa: E402  (scripts/ingest_odds.py: schedule fetchers, event matchers, game lookup)
from sportsmodel import sports  # noqa: E402
from sportsmodel.early import flags, landing, polling  # noqa: E402
from sportsmodel.early.constants import BOOKMAKERS, MARKETS, SPORT_ORDER  # noqa: E402
from sportsmodel.ingest import odds  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402


@dataclass
class Deps:
    """Every side effect of a run, injectable so the whole flow is testable without a network or a database."""
    missing_tables: Callable[[], list[str]]
    load_existing: Callable[[str, datetime], set]                      # (sport, now) -> {(sport, game_pk, market, side)}
    load_tables: Callable[[], dict]                                    # landing tables
    fetch_schedule: Callable[[str], list[dict]]
    match: Callable[[str], Callable]                                   # sport -> matcher(event, games) -> game_pk | None
    prime: Callable[[Any], Any]                                        # the free /events call: refreshes the credit header
    fetch_odds: Callable[[Any, list[str], list[str]], list[dict]]      # (cfg, bookmakers, markets)
    remaining: Callable[[], Any]
    last_cost: Callable[[], Any]
    store_snapshots: Callable[[list[dict]], int]
    store_tickets: Callable[[list[dict]], int]


def _warn(msg: str) -> None:
    print(f"::warning::early-monitor: {msg}", flush=True)


def _poll_sport(sport: str, now: datetime, deps: Deps, tables: dict, run_id: str, dry_run: bool, out: dict) -> None:
    cfg = sports.get(sport)
    schedule = deps.fetch_schedule(sport)                              # free reads first: a failure here costs no credits
    existing = deps.load_existing(sport, now)
    events = deps.fetch_odds(cfg, list(BOOKMAKERS), polling.api_markets(sport))
    cost = polling.parse_credits(deps.last_cost())
    out["credits"] += cost if cost is not None else polling.poll_cost(sport)     # count it the moment it is spent
    lookup = ingest_odds.build_game_lookup(events, schedule, deps.match(sport))
    captured_at = now.isoformat()
    rows = [r for r in odds.parse_game_odds(events, lookup, captured_at)
            if r["book"] in BOOKMAKERS and r["market"] in MARKETS[sport]]
    tickets = flags.flag_poll(rows, sport=sport, observed_at=now, tables=tables, existing=existing, poll_run_id=run_id)
    out["sports"][sport] = {"events": len(events), "matched": len(lookup), "rows": len(rows), "tickets": len(tickets)}
    if dry_run:
        out["would_write"] += tickets
        return
    if rows:
        try:
            out["snapshots"] += deps.store_snapshots(rows)
        except Exception as exc:  # noqa: BLE001 -- tickets are the point: still log what this poll found
            _warn(f"{sport}: storing the odds snapshot failed: {type(exc).__name__}: {exc}")
    if tickets:
        out["tickets"] += deps.store_tickets(tickets)                  # right now: a later sport's failure loses nothing


def run_poll(now: datetime, deps: Deps, *, run_id: str = "local", dry_run: bool = False) -> dict:
    """One poll of every sport. Returns a summary dict; status 'no_table' means nothing was fetched."""
    gone = deps.missing_tables()
    if gone:
        _warn(f"table {gone[0]} missing — run db/migration_early_tickets.sql")
        return {"status": "no_table"}
    tables = deps.load_tables()
    out: dict = {"status": "ok", "credits": 0, "snapshots": 0, "tickets": 0, "sports": {}, "would_write": [], "skipped": []}
    try:
        deps.prime(sports.get(SPORT_ORDER[0]))                         # free: reads x-requests-remaining before any paid call
    except Exception as exc:  # noqa: BLE001 -- without the header the first paid call simply runs (unknown = run)
        _warn(f"credit check call failed: {type(exc).__name__}: {exc}")
    for sport in SPORT_ORDER:
        decision = polling.budget_decision(deps.remaining())
        if not decision.ok:
            _warn(decision.warning)
            out["skipped"] = [s for s in SPORT_ORDER if s not in out["sports"]]
            break
        try:
            _poll_sport(sport, now, deps, tables, run_id, dry_run, out)
        except Exception as exc:  # noqa: BLE001 -- a failed poll is not retried; the next sport and the next hour go on
            _warn(f"{sport} poll failed: {type(exc).__name__}: {exc}")
            out["sports"].setdefault(sport, {"failed": f"{type(exc).__name__}: {exc}"[:200]})
    out["remaining"] = deps.remaining()
    return out


# ---- production wiring --------------------------------------------------------------------------------

def _db_deps() -> Deps:
    from sportsmodel import db

    timeouts = {"connect_timeout": db.LEDGER_CONNECT_TIMEOUT, "statement_timeout_ms": db.LEDGER_STATEMENT_TIMEOUT_MS}
    return Deps(
        missing_tables=lambda: db.missing_early_tables(**timeouts),
        load_existing=lambda sport, now: db.early_ticket_keys(sport, now, **timeouts),
        load_tables=landing.load_tables,
        fetch_schedule=lambda sport: ingest_odds._ESPN_FETCHERS[sport](),
        match=ingest_odds._matcher_for,
        prime=odds.fetch_events, fetch_odds=odds.fetch_bookmaker_odds,
        remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
        store_snapshots=db.upsert_odds_snapshot,
        store_tickets=lambda rows: db.insert_early_tickets(rows, **timeouts))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true", help="poll (about 3 credits) and print the tickets it would log; write nothing")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("early-monitor: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    for name in ("DATABASE_URL", "ODDS_API_KEY"):
        if not os.getenv(name):
            print(f"::error::early-monitor: {name} is not set")
            return 1
    result = run_poll(datetime.now(timezone.utc), _db_deps(), run_id=os.getenv("GITHUB_RUN_ID") or "local",
                      dry_run=args.dry_run)
    for t in result.get("would_write", []):
        print(f"[early-monitor] DRY RUN would log: {t['sport']} {t['game_pk']} {t['market']} {t['side']} "
              f"{t['book']} {t['book_line']:+g} {t['price']:+d} ev={t['ev']:+.2%} kick in {t['hours_to_kick']:.1f}h", flush=True)
    print(f"[early-monitor] { {k: v for k, v in result.items() if k != 'would_write'} }", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early/test_polling.py tests/early/test_early_monitor_script.py tests/test_odds.py tests/test_ingest_odds.py tests/test_sports.py`
Expected: all PASS (the two new files: 35 passed).

- [ ] **Step 7: Commit**

```bash
git add src/sportsmodel/early/polling.py src/sportsmodel/ingest/odds.py scripts/early_monitor.py tests/early/test_polling.py tests/early/test_early_monitor_script.py
git commit -m "feat(early): hourly monitor script with bookmakers fetch, credit floor and incremental ticket writes" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Monitor workflow and the close-capture candidate

**Files:**
- Create: `.github/workflows/early-monitor.yml`
- Modify: `scripts/capture_close.py` (new candidate source: `early_tickets`)
- Create: `tests/early/test_workflows.py` (the monitor group; Tasks 7 and 9 append their groups), `tests/early/test_capture_close_candidates.py`

**Interfaces:**
- Consumes: Task 5 `scripts/early_monitor.py` (the workflow's command); Task 1 `db.missing_early_tables`; existing `scripts/capture_close.py` (`Deps`, `run_capture`, `_db_deps`) and `tests/ledger/test_capture_close_script.py` (its `Fake`, `cand`, `game`, `event`, `odds_event`, `NOW`, `cc`); the cron evaluator `cron_sets` / `fires` / `utc` in `tests/ledger/test_workflows_capture.py`; the block-YAML reader in `tests/test_workflows_props_ml.py`.
- Produces:
  - workflow `early-monitor.yml` (name `early-monitor`, concurrency group `early-monitor`)
  - `capture_close.Deps.load_early_candidates: Callable[[datetime, datetime], list[dict]]` (defaults to returning `[]`, so every existing caller and test is unchanged); `run_capture` now concatenates `load_candidates(now, horizon) + load_early_candidates(now, horizon)`

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_workflows.py` (the monitor group; later tasks append to this file):

````python
"""Structural checks of the three early-monitor workflows: the hourly window, the env-passed dry run, secrets, serialised
runs, and that only the report workflow commits. Reuses the tiny cron evaluator of the capture-close tests."""
from datetime import timedelta

import pytest

from tests.ledger.test_workflows_capture import cron_sets, fires, utc
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

MONITOR, GRADE, REPORT = "early-monitor.yml", "grade-early-tickets.yml", "report-early-tickets.yml"
CRONS = ["0 16-23 * 1,2,8-12 0", "0 * * 1,2,8-12 1,2", "0 0-3 * 1,2,8-12 3"]


@pytest.fixture(scope="module")
def crons():
    return [c["cron"] for c in load(MONITOR)["on"]["schedule"]]


def fired(crons, start, end):
    out, t = [], start
    while t < end:
        if any(fires(c, t) for c in crons):
            out.append(t)
        t += timedelta(minutes=1)
    return out


# ---- early-monitor.yml ---------------------------------------------------------------------------------------

def test_the_three_crons_are_the_specs():
    assert [c["cron"] for c in load(MONITOR)["on"]["schedule"]] == CRONS


def test_hourly_at_minute_zero_60_polls_a_week(crons):
    week = fired(crons, utc(2026, 10, 12, 0, 0), utc(2026, 10, 19, 0, 0))
    assert len(week) == 60 and {t.minute for t in week} == {0}
    assert len(fired(crons, utc(2026, 11, 2, 0, 0), utc(2026, 11, 9, 0, 0))) == 60


def test_the_window_runs_from_sunday_1600_to_wednesday_0300_utc(crons):
    week = fired(crons, utc(2026, 10, 11, 0, 0), utc(2026, 10, 18, 0, 0))
    assert week[0] == utc(2026, 10, 11, 16, 0) and week[-1] == utc(2026, 10, 14, 3, 0)         # Sun 16:00 ... Wed 03:00
    assert not fired(crons, utc(2026, 10, 14, 4, 0), utc(2026, 10, 18, 16, 0))                  # Wed 04:00 until Sun 16:00
    assert not fired(crons, utc(2026, 10, 11, 0, 0), utc(2026, 10, 11, 16, 0))                  # Sunday before 16:00


def test_football_months_only_aug_to_feb(crons):
    assert cron_sets(crons[0])[2] == {1, 2, 8, 9, 10, 11, 12} == cron_sets(crons[1])[2] == cron_sets(crons[2])[2]
    for month in (3, 4, 5, 6, 7):
        assert not fired(crons, utc(2027, month, 1, 0, 0), utc(2027, month, 8, 0, 0))
    assert fired(crons, utc(2026, 8, 30, 0, 0), utc(2026, 9, 6, 0, 0))                          # the first week of the season


def test_dispatch_has_an_env_passed_dry_run_input():
    wf = load(MONITOR)
    inp = wf["on"]["workflow_dispatch"]["inputs"]["dry_run"]
    assert inp["type"] == "boolean" and str(inp["default"]).lower() == "false"
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/early_monitor.py"))]
    assert step["env"]["DRY_RUN"] == "${{ inputs.dry_run }}"
    assert '[ "$DRY_RUN" = "true" ]' in step["run"] and "--dry-run" in step["run"]
    assert "${{" not in step["run"]                                   # no expression interpolated into the shell script


def test_job_has_the_secrets_serialises_runs_and_caches_uv():
    wf = load(MONITOR)
    assert wf["permissions"] == {"contents": "read"}
    assert wf["concurrency"]["group"] == "early-monitor" and str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
    (job,) = wf["jobs"].values()
    assert str(job["timeout-minutes"]) == "6"
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/early_monitor.py"))]
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}" and step["env"]["ODDS_API_KEY"] == "${{ secrets.ODDS_API_KEY }}"
    uv = steps[step_index(steps, lambda st: "setup-uv" in str(st.get("uses", "")))]
    assert str(uv["with"]["enable-cache"]).lower() == "true"
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("early_monitor.py"))
````

Create `tests/early/test_capture_close_candidates.py`:

````python
"""scripts/capture_close.py gains early_tickets as a candidate source: a ticketed game with no prediction and no ledger
pick still gets its pre-kick line capture, and nothing changes before the migration has run. No network, no database."""
from datetime import timedelta

import pytest

from sportsmodel import db
from tests.ledger import test_capture_close_script as ccs

cc, NOW = ccs.cc, ccs.NOW


def cfb_fake(**kw):
    base = dict(schedule={"cfb": [ccs.game(9, 12)]}, events=[ccs.event("c1", 12)], line_events=[ccs.odds_event("c1", 12)])
    return ccs.Fake(**{**base, **kw})


def test_a_ticketed_game_alone_makes_the_run_non_idle_and_gets_its_lines_captured():
    fake = cfb_fake()                                              # no prediction, no ledger pick: load_candidates is empty
    deps = fake.deps()
    deps.load_early_candidates = lambda now, end: [ccs.cand("cfb", 9, 12)]
    out = cc.run_capture(NOW, deps)
    assert out["rows"] == [("cfb", 9, "lines", "ok")] and ("lines", "cfb", "us,eu") in fake.calls


def test_without_early_candidates_the_same_game_is_idle_as_before():
    fake = cfb_fake()
    assert cc.run_capture(NOW, fake.deps()) == {"status": "idle"} and fake.calls == []


def test_a_far_away_early_candidate_stays_idle():
    fake = cfb_fake()
    deps = fake.deps()
    deps.load_early_candidates = lambda now, end: [ccs.cand("cfb", 9, 600)]
    assert cc.run_capture(NOW, deps) == {"status": "idle"}


def test_both_sources_are_asked_with_the_same_one_hour_horizon():
    fake = cfb_fake()
    deps = fake.deps()
    seen = []
    deps.load_candidates = lambda now, end: seen.append(("base", now, end)) or []
    deps.load_early_candidates = lambda now, end: seen.append(("early", now, end)) or []
    cc.run_capture(NOW, deps)
    assert seen == [("base", NOW, NOW + timedelta(minutes=60)), ("early", NOW, NOW + timedelta(minutes=60))]


class Cur:
    def __init__(self, seen, rows):
        self.seen, self.rows = seen, rows

    def execute(self, sql, params=None):
        self.seen.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class Conn(Cur):
    def cursor(self):
        return self


def test_production_query_reads_early_tickets_only_once_the_table_exists(monkeypatch):
    seen = []
    kick = NOW + timedelta(minutes=12)
    monkeypatch.setattr(db, "get_postgres", lambda: Conn(seen, [("cfb", 9, kick)]))
    monkeypatch.setattr(db, "missing_early_tables", lambda tables=None, **kw: [])
    rows = cc._db_deps().load_early_candidates(NOW, NOW + timedelta(hours=1))
    assert rows == [{"sport": "cfb", "game_pk": 9, "commence_time": kick}]
    sql, params = seen[0]
    assert sql.startswith("SELECT DISTINCT sport, game_pk, commence_time FROM early_tickets WHERE commence_time > %s")
    assert params == (NOW, NOW + timedelta(hours=1))


def test_before_the_migration_the_early_source_is_empty_and_runs_no_query(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", lambda: pytest.fail("no query expected"))
    monkeypatch.setattr(db, "missing_early_tables", lambda tables=None, **kw: ["early_tickets"])
    assert cc._db_deps().load_early_candidates(NOW, NOW + timedelta(hours=1)) == []


def test_production_deps_wire_the_early_source():
    deps = cc._db_deps()
    assert deps.load_early_candidates is not None and deps.load_early_candidates.__name__ == "early_candidates"
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_workflows.py tests/early/test_capture_close_candidates.py`
Expected: FAIL: the workflow tests with `FileNotFoundError: ... early-monitor.yml`; the candidate tests with assertion failures (`run_capture` still ignores the new source and reports `idle`) and `AttributeError` for the production wiring.

- [ ] **Step 3: Write the workflow**

Create `.github/workflows/early-monitor.yml`:

````yaml
name: early-monitor

# Early-week stale-number monitor (profit step 3; scripts/early_monitor.py, db/migration_early_tickets.sql). Hourly at
# minute 0 from Sunday 16:00 UTC to Wednesday 04:00 UTC (60 polls a week) in football season: ONE Odds API odds call per
# sport using the `bookmakers` parameter (the 9 US books + pinnacle = 10 keys = one billed region): CFB spreads + totals
# (2 credits), NFL spreads (1 credit), about 180 credits a week. The prices go to odds_snapshot; a US book whose number beats
# Pinnacle's same-poll fair by +1.5% EV, with kickoff at least 24 h away, becomes ONE paper ticket per
# (sport, game_pk, market, side) in the append-only early_tickets table. No alerts, no bets, no change to boards, picks or
# the site. The job skips its remaining calls below 1,500 credits (the regular prop pulls stop at 2,500, regular game lines
# at 800, close-capture props at 600 and all close capture at 200). A failed poll is not retried: the next hour runs normally.
# Crons are UTC; months 1,2,8-12 are Aug-Feb, so March to July never fires.
on:
  schedule:
    - cron: "0 16-23 * 1,2,8-12 0"
    - cron: "0 * * 1,2,8-12 1,2"
    - cron: "0 0-3 * 1,2,8-12 3"
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Poll once (about 3 credits) and print the tickets it would log; write nothing"
        type: boolean
        default: false

permissions:
  contents: read

# Overlapping polls would double-spend credits and race on the same tickets; queue instead.
concurrency:
  group: early-monitor
  cancel-in-progress: false

jobs:
  poll:
    runs-on: ubuntu-latest
    timeout-minutes: 6
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
        with:
          enable-cache: true
      - name: Sync deps
        run: uv sync
      - name: Poll the books and log paper tickets
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          DRY_RUN: ${{ inputs.dry_run }}      # empty on schedule events, "true"/"false" on workflow_dispatch
        run: |
          if [ "$DRY_RUN" = "true" ]; then
            uv run python scripts/early_monitor.py --dry-run
          else
            uv run python scripts/early_monitor.py
          fi
````

- [ ] **Step 4: Add the candidate source to the close capture**

Apply to `scripts/capture_close.py`:

````diff
--- a/scripts/capture_close.py
+++ b/scripts/capture_close.py
@@ -2,7 +2,9 @@
 
 Run every ~10 minutes during game windows (.github/workflows/capture-close.yml). Each run:
 
-  1. Idle check, DB only: no kickoff known within the next hour -> exit with ZERO network calls.
+  1. Idle check, DB only: no kickoff known within the next hour -> exit with ZERO network calls. The kickoffs come from
+     predictions_current, pick_ledger and (once its migration has run) early_tickets, so every game the early-week
+     monitor ticketed gets its pre-kick line capture too.
   2. For each sport with a nearby kickoff: fetch the live schedule (ESPN / StatsAPI, free), keep the games kicking
      off in WINDOW_MIN..WINDOW_MAX minutes that have no final (ok / empty) close_capture yet. None -> no Odds API call.
   3. Free Odds API events call: reads x-requests-remaining for the budget guard (< 600 skips props, < 200
@@ -57,6 +59,8 @@
     last_cost: Callable[[], Any]
     store_snapshots: Callable[[list[dict]], int]
     store_captures: Callable[[list[dict]], int]
+    # games holding an early-week monitor ticket (same shape as load_candidates); none until its migration has run
+    load_early_candidates: Callable[[datetime, datetime], list[dict]] = lambda now, end: []
 
 
 def _warn(msg: str) -> None:
@@ -192,7 +196,8 @@
     if gone:
         _warn(f"table {gone[0]} missing — run db/migration_pick_ledger.sql")
         return {"status": "no_table"}
-    candidates = deps.load_candidates(now, now + timedelta(minutes=close.PREFILTER_MIN))
+    horizon = now + timedelta(minutes=close.PREFILTER_MIN)
+    candidates = deps.load_candidates(now, horizon) + deps.load_early_candidates(now, horizon)
     live = close.prefilter_sports(candidates, now)
     if not live:
         return {"status": "idle"}
@@ -237,6 +242,15 @@
             out.setdefault(pk, set()).add(market)
         return out
 
+    def early_candidates(now, end):
+        # Games with an early-week monitor ticket may have no model prediction and no ledger pick (a CFB total the model
+        # does not cover), so they are added here. Skipped until db/migration_early_tickets.sql has run.
+        if db.missing_early_tables(["early_tickets"]):
+            return []
+        rows = q("SELECT DISTINCT sport, game_pk, commence_time FROM early_tickets "
+                 "WHERE commence_time > %s AND commence_time <= %s", (now, end))
+        return [{"sport": s, "game_pk": pk, "commence_time": c} for s, pk, c in rows]
+
     return Deps(
         missing_tables=lambda: db.missing_ledger_tables(["pick_ledger", "close_capture"]),
         load_candidates=candidates, load_captured=captured, load_prop_markets=prop_markets,
@@ -245,7 +259,8 @@
         fetch_events=odds.fetch_events, fetch_game_odds=odds.fetch_game_odds,
         fetch_event_props=lambda eid, markets, cfg, regions: odds.fetch_event_props(eid, markets, cfg, regions=regions),
         remaining=lambda: odds.last_requests_remaining, last_cost=lambda: odds.last_requests_last,
-        store_snapshots=db.upsert_odds_snapshot, store_captures=db.upsert_close_capture)
+        store_snapshots=db.upsert_odds_snapshot, store_captures=db.upsert_close_capture,
+        load_early_candidates=early_candidates)
 
 
 def _parse_now(value: str) -> datetime:
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early/test_workflows.py tests/early/test_capture_close_candidates.py tests/ledger tests/test_workflows_props_ml.py`
Expected: all PASS (13 new tests at this point: six workflow tests and the seven in `test_capture_close_candidates.py`; the existing `tests/ledger/test_capture_close_script.py` is unchanged and still green, and `test_every_workflow_parses` now has a case for `early-monitor.yml`).

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/early-monitor.yml scripts/capture_close.py tests/early/test_workflows.py tests/early/test_capture_close_candidates.py
git commit -m "feat(early): early-monitor workflow, and ticketed games become close-capture candidates" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Grading (`grade_early_tickets.py`)

**Files:**
- Create: `src/sportsmodel/early/grading.py`
- Create: `scripts/grade_early_tickets.py`, `.github/workflows/grade-early-tickets.yml`
- Modify: `tests/early/test_workflows.py` (append the grading group)
- Create: `tests/early/test_grading.py`, `tests/early/test_grade_script.py`

**Interfaces:**
- Consumes: Task 3 `adjust.value_side`; Task 4 `flags.pinnacle_reference`, tests' `builders`; Task 1 `db.missing_early_tables`, `db.upsert_early_ticket_results`, `db.EARLY_COLUMNS`; Task 2 `landing.load_tables`, `constants.*`; existing `ledger.reference` (`settle_game(market, side, line, margin, total)`, `units_at_pick(result, price)`, `clv_pts(market, side, pick_line, close_line)`), `ledger.core.as_utc`, `serving.board.decimal_odds`.
- Produces:
  - `grading.closing_view(ticket, close_rows, tables) -> dict` (`close_kind` `pinnacle_novig` / `missing`, `close_fair`, `close_line`, `close_minutes_to_kick`, `close_ev`)
  - `grading.available_next_poll(ticket, later_rows) -> bool | None`
  - `grading.grade_ticket(ticket, final, close_rows, later_rows, tables) -> dict | None` (a row with exactly the keys of `db.EARLY_COLUMNS["early_ticket_results"]`; `None` while the game has no result)
  - `grading.grade_tickets(tickets, finals, close_by_game, later_by_game, tables) -> list[dict]`
  - `grade_early_tickets.Loaders`, `grade_all(ld, days) -> dict`, `_db_loaders()`, `main(argv)`; workflow `grade-early-tickets.yml`
  - input shapes: ticket = early_tickets row (`id, sport, game_pk, market, side, book, book_line, price, observed_at, commence_time, ...`); `finals` = `{(sport, game_pk): {"actual_margin", "actual_total"}}`; snapshot rows = odds_snapshot dicts (`game_pk, market, side, player_name, book, line, price, captured_at`)

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_grading.py`:

````python
"""early/grading.py: close EV, missing closes, line CLV signs, pushes and available_next_poll. Pure."""
from datetime import timedelta

import pytest

from sportsmodel.early import grading
from sportsmodel.early.constants import PINNACLE
from sportsmodel.ledger import reference

from .builders import NOW, StubTable, mass_at, row

KICK = NOW + timedelta(hours=60)
CLOSE_AT = KICK - timedelta(minutes=15)
M = 0.0318
TABLES = {("cfb", "total"): StubTable(mass_at((55, 56), M)), ("nfl", "spread"): StubTable(mass_at(3, M)),
          ("cfb", "spread"): StubTable(mass_at(3, M))}


def ticket(**over):
    base = {"id": 11, "sport": "cfb", "game_pk": 1, "market": "total", "side": "over", "book": "draftkings",
            "book_line": 54.5, "price": -110, "observed_at": NOW, "commence_time": KICK}
    return base | over


def close_total(line=55.5, over=-110, under=-110, at=CLOSE_AT):
    return [row(1, "total", "over", PINNACLE, line, over, KICK, captured_at=at),
            row(1, "total", "under", PINNACLE, line, under, KICK, captured_at=at)]


FINAL = {"actual_margin": 3.0, "actual_total": 58.0}


def grade(t=None, final=FINAL, close=None, later=(), tables=TABLES):
    return grading.grade_ticket(t or ticket(), final, close_total() if close is None else close, list(later), tables)


# ---- close EV and its companions -----------------------------------------------------------------------------

def test_close_ev_values_the_ticket_at_pinnacles_closing_fair():
    g = grade()
    assert g["close_kind"] == "pinnacle_novig" and g["close_fair"] == pytest.approx(0.5) and g["close_line"] == 55.5
    assert g["close_ev"] == pytest.approx(-0.04545454 + 1.9090909 * M, abs=1e-6)       # Over 54.5 vs a 55.5 close
    assert g["beat_close"] is True and g["close_minutes_to_kick"] == pytest.approx(15.0)


def test_close_ev_uses_the_closing_fair_not_the_flag_time_fair():
    sharp = grade(close=close_total(over=-130, under=110))          # the market moved toward the Over: fair above 50%
    assert sharp["close_fair"] > 0.5 and sharp["close_ev"] > grade()["close_ev"]
    soft = grade(close=close_total(over=110, under=-130))
    assert soft["close_ev"] < 0 and soft["beat_close"] is False


def test_no_pinnacle_quote_at_the_close_is_missing_and_never_beats_it():
    no_pin = [dict(r, book="draftkings") for r in close_total()]
    for rows in (no_pin, close_total()[:1], []):
        g = grade(close=rows)
        assert g["close_kind"] == "missing" and g["close_ev"] is None and g["beat_close"] is None
        assert g["close_fair"] is None and g["clv_pts"] is None and g["result"] == "W"     # still graded W/L/units


def test_a_quarter_point_closing_line_is_missing_not_rounded():
    assert grade(close=close_total(line=55.75))["close_kind"] == "missing"


def test_a_close_that_moved_more_than_three_points_still_counts_and_counts_against_the_ticket():
    g = grade(close=close_total(line=49.5))                           # the total fell 5 points; the Over 54.5 got worse
    assert g["close_kind"] == "pinnacle_novig" and g["close_ev"] is not None
    assert g["clv_pts"] == pytest.approx(-5.0)


def test_close_ev_exactly_zero_is_not_a_beat():
    flat = {("cfb", "total"): StubTable(mass_at((55, 56), 0.0))}
    g = grade(close=close_total(line=54.5), tables=flat)               # same line, fair 50%, price -110: EV = -4.5%
    assert g["close_ev"] < 0 and g["beat_close"] is False
    even = grade(t=ticket(price=100), close=close_total(line=54.5, over=100, under=100), tables=flat)
    assert even["close_ev"] == pytest.approx(0.0) and even["beat_close"] is False


# ---- clv_pts ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("market, side, book_line, close_line, expected", [
    ("total", "over", 54.5, 55.5, 1.0),         # we bet Over lower than the close: gained a point
    ("total", "over", 56.5, 55.5, -1.0),
    ("total", "under", 56.5, 55.5, 1.0),
    ("total", "under", 54.5, 55.5, -1.0),
    ("spread", "home", -2.5, -3.5, 1.0),        # home -2.5 vs a -3.5 close: gained a point
    ("spread", "away", 3.5, 2.5, 1.0),          # away +3.5 vs a +2.5 close
    ("spread", "away", 2.5, 3.5, -1.0),
])
def test_clv_pts_is_signed_in_the_ticket_sides_favour(market, side, book_line, close_line, expected):
    assert reference.clv_pts(market, side, book_line, close_line) == expected


def test_spread_ticket_clv_and_close_use_the_tickets_own_side_lines():
    t = ticket(sport="nfl", market="spread", side="away", book_line=3.5)
    close = [row(1, "spread", "home", PINNACLE, -2.5, -110, KICK, captured_at=CLOSE_AT),
             row(1, "spread", "away", PINNACLE, 2.5, -110, KICK, captured_at=CLOSE_AT)]
    g = grade(t=t, final={"actual_margin": 3.0, "actual_total": 40.0}, close=close)
    assert g["close_line"] == 2.5 and g["clv_pts"] == pytest.approx(1.0) and g["beat_close"] is True
    assert g["result"] == "W"                                           # home won by 3, so the away side +3.5 covers


# ---- result, units, pushes -------------------------------------------------------------------------------------

@pytest.mark.parametrize("total, result, units", [(60.0, "W", 100 / 110), (50.0, "L", -1.0)])
def test_result_and_units_at_the_tickets_price(total, result, units):
    g = grade(final={"actual_margin": 3.0, "actual_total": total})
    assert g["result"] == result and g["units"] == pytest.approx(units)


def test_a_push_pays_the_stake_back_and_is_still_graded():
    g = grade(t=ticket(book_line=58.0), final=FINAL)                   # total 58 on a 58 line
    assert g["result"] == "P" and g["units"] == 0.0 and g["close_kind"] == "pinnacle_novig"


def test_a_game_without_a_result_stays_ungraded():
    assert grade(final=None) is None and grade(final={}) is None
    assert grading.grade_tickets([ticket()], {}, {1: close_total()}, {}, TABLES) == []


def test_result_columns_match_the_table_columns():
    from sportsmodel import db
    assert set(grade()) == set(db.EARLY_COLUMNS["early_ticket_results"])


def test_grade_tickets_routes_finals_closes_and_later_polls_by_game():
    t2 = ticket(id=12, game_pk=2, sport="cfb")
    out = grading.grade_tickets([ticket(), t2], {("cfb", 1): FINAL}, {1: close_total()}, {}, TABLES)
    assert [g["ticket_id"] for g in out] == [11] and out[0]["close_kind"] == "pinnacle_novig"


# ---- available_next_poll ---------------------------------------------------------------------------------------

def later(minutes, *, price=-110, line=54.5, book="draftkings", side="over", pinnacle=True):
    at = NOW + timedelta(minutes=minutes)
    rows = [row(1, "total", side, book, line, price, KICK, captured_at=at)]
    if pinnacle:
        rows += [row(1, "total", "over", PINNACLE, 55.5, -110, KICK, captured_at=at),
                 row(1, "total", "under", PINNACLE, 55.5, -110, KICK, captured_at=at)]
    return rows


def avail(rows, **over):
    return grading.available_next_poll(ticket(**over), rows)


def test_same_line_same_or_better_price_one_poll_later_is_available():
    assert avail(later(60)) is True
    assert avail(later(60, price=-105)) is True
    assert avail(later(60, price=120)) is True                          # across the +/-100 boundary: better


def test_a_worse_price_a_moved_line_or_a_pulled_book_is_not_available():
    assert avail(later(60, price=-115)) is False
    assert avail(later(60, line=55.5)) is False
    assert avail(later(60, line=53.5)) is False
    assert avail(later(60, book="fanduel")) is False
    assert avail(later(60, side="under")) is False


def test_the_next_poll_must_be_30_minutes_to_3_hours_after_the_ticket():
    assert avail(later(30)) is True and avail(later(180)) is True
    assert avail(later(29)) is None                                      # too soon to be a new look, and nothing else
    assert avail(later(181)) is None
    assert avail([]) is None
    assert avail(later(20) + later(75, price=-115)) is False             # a 20-minute snapshot is skipped; the 75-minute poll decides


def test_the_first_qualifying_poll_decides_not_a_later_one():
    assert avail(later(60, price=-115) + later(120)) is False
    assert avail(later(60) + later(120, price=-150)) is True


def test_a_snapshot_without_a_pinnacle_quote_is_not_a_poll():
    assert avail(later(60, pinnacle=False)) is None
    assert avail(later(60, pinnacle=False) + later(90)) is True


def test_the_tickets_book_key_is_matched_exactly_and_line_with_a_float32_tolerance():
    assert avail(later(60, line=54.5000001)) is True
    assert avail(later(60), book="fanduel") is False


def test_available_next_poll_is_stored_on_the_result():
    assert grade(later=later(60))["available_next_poll"] is True
    assert grade(later=[])["available_next_poll"] is None
````

Create `tests/early/test_grade_script.py`:

````python
"""scripts/grade_early_tickets.py on injected loaders and recorded SQL. No database."""
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

import pytest

from .builders import NOW, StubTable, mass_at, row

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("grade_early_tickets", ROOT / "scripts" / "grade_early_tickets.py")
ge = importlib.util.module_from_spec(_spec)
sys.modules["grade_early_tickets"] = ge
_spec.loader.exec_module(ge)

KICK = NOW + timedelta(hours=60)
TABLES = {("cfb", "total"): StubTable(mass_at((55, 56), 0.0318))}


def ticket(i=11, pk=1):
    return {"id": i, "sport": "cfb", "game_pk": pk, "market": "total", "side": "over", "book": "draftkings",
            "book_line": 54.5, "price": -110, "observed_at": NOW, "commence_time": KICK}


def close(pk=1):
    at = KICK - timedelta(minutes=15)
    return {pk: [row(pk, "total", "over", "pinnacle", 55.5, -110, KICK, captured_at=at),
                 row(pk, "total", "under", "pinnacle", 55.5, -110, KICK, captured_at=at)]}


def loaders(**over):
    calls = {}
    base = dict(
        missing_tables=lambda: [],
        pending_tickets=lambda days: [ticket()],
        finals=lambda pks: {("cfb", 1): {"actual_margin": 3.0, "actual_total": 60.0}},
        close_rows=lambda pks: close(), later_rows=lambda pks: {},
        load_tables=lambda: TABLES,
        store=lambda rows: calls.setdefault("stored", rows) and len(rows))
    return ge.Loaders(**{**base, **over}), calls


def test_grades_pending_tickets_and_stores_one_row_per_ticket():
    ld, calls = loaders()
    out = ge.grade_all(ld, 14)
    assert out == {"status": "ok", "pending": 1, "graded": 1, "close_kinds": {"pinnacle_novig": 1}}
    (row_,) = calls["stored"]
    assert row_["ticket_id"] == 11 and row_["result"] == "W" and row_["close_ev"] > 0


def test_nothing_pending_stores_nothing():
    ld, calls = loaders(pending_tickets=lambda days: [])
    assert ge.grade_all(ld, 14) == {"status": "ok", "pending": 0, "graded": 0} and "stored" not in calls


def test_a_game_without_a_final_stays_ungraded_and_is_retried_the_next_day():
    ld, calls = loaders(finals=lambda pks: {})
    out = ge.grade_all(ld, 14)
    assert out["pending"] == 1 and out["graded"] == 0 and "stored" not in calls


def test_a_missing_close_is_graded_but_marked_missing():
    ld, calls = loaders(close_rows=lambda pks: {})
    out = ge.grade_all(ld, 14)
    assert out["close_kinds"] == {"missing": 1} and calls["stored"][0]["close_ev"] is None


def test_missing_tables_warn_and_stop(capsys):
    ld, calls = loaders(missing_tables=lambda: ["early_ticket_results"], pending_tickets=lambda d: pytest.fail("queried"))
    assert ge.grade_all(ld, 14) == {"status": "no_table"}
    assert "::warning::grade-early-tickets: table early_ticket_results missing" in capsys.readouterr().out


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert ge.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_days_is_clamped_to_1_60(monkeypatch):
    seen = []
    monkeypatch.setattr(ge, "grade_all", lambda ld, days: seen.append(days) or {})
    monkeypatch.setattr(ge, "_db_loaders", lambda: None)
    monkeypatch.setattr(ge, "writes_disabled", lambda: False)
    monkeypatch.setenv("DATABASE_URL", "postgres://x")
    for arg in ("0", "-5", "14", "9999"):
        assert ge.main(["--days", arg]) == 0
    assert seen == [1, 1, 14, 60]


def test_main_needs_the_database_url(monkeypatch, capsys):
    monkeypatch.setattr(ge, "writes_disabled", lambda: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert ge.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out


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
    ld = ge._db_loaders()
    ld.pending_tickets(14)
    ld.finals([7])
    ld.close_rows([7])
    ld.later_rows([7])
    sql = [s for s, _ in seen]
    assert "FROM early_tickets t" in sql[0] and "interval '4 hours'" in sql[0] and seen[0][1] == (14,)
    assert "NOT EXISTS (SELECT 1 FROM early_ticket_results r WHERE r.ticket_id = t.id)" in sql[0]
    assert "FROM prediction_accuracy" in sql[1] and seen[1][1] == ([7],)
    assert "JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at" in sql[2]
    assert "c.kind = 'lines' AND c.status = 'ok'" in sql[2] and "s.player_name = ''" in sql[2]
    assert "FROM odds_snapshot s" in sql[3] and "interval '3 hours'" in sql[3] and "t.observed_at" in sql[3]
    assert not any(w in s.upper().split() for s in sql for w in ("INSERT", "UPDATE", "DELETE"))
    for col in ("t.id", "t.book_line", "t.price", "t.observed_at", "t.commence_time"):
        assert col in sql[0]
````

Append to `tests/early/test_workflows.py`:

````python
# ---- grade-early-tickets.yml ---------------------------------------------------------------------------------

def test_grading_runs_daily_at_1420_after_the_other_graders_and_never_spends_credits():
    wf = load(GRADE)
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["20 14 * * *"]
    assert wf["on"]["workflow_dispatch"]["inputs"]["days"]["default"] == "14"
    assert wf["permissions"] == {"contents": "read"} and wf["concurrency"]["group"] == "grade-early-tickets"
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/grade_early_tickets.py"))]
    assert step["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}", "DAYS": "${{ github.event.inputs.days || '14' }}"}
    assert step["run"] == 'uv run python scripts/grade_early_tickets.py --days "$DAYS"'
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("grade_early_tickets.py"))
    text = (WF / GRADE).read_text()
    assert "git push" not in text and "ODDS_API_KEY" not in text.split("jobs:")[1]
    assert "10 14 * * *" in (WF / "grade-ledger.yml").read_text()                                # after grade-ledger
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_grading.py tests/early/test_grade_script.py tests/early/test_workflows.py`
Expected: collection ERRORs for the grading tests (`ImportError: cannot import name 'grading'`, `FileNotFoundError` for `scripts/grade_early_tickets.py`); the new workflow test fails with `FileNotFoundError: ... grade-early-tickets.yml`.

- [ ] **Step 3: Write the grading module**

Create `src/sportsmodel/early/grading.py`:

````python
"""Grade early tickets against the result and Pinnacle's pre-kick close. PURE -- no IO.

Primary metric `close_ev`: the ticket's book line and price valued at Pinnacle's CLOSING fair, through the same
landing-table adjustment and EV formula the flag used. Unlike the flag, the adjustment is uncapped and clamped here (a
close that moved a long way against the ticket must count as a loss, not drop the ticket from the test).
"""
from __future__ import annotations

from collections import defaultdict
from datetime import timedelta

from ..ledger import reference as ref
from ..ledger.core import as_utc
from ..serving.board import decimal_odds
from . import adjust
from .constants import LINE_EPS, NEXT_POLL_MAX_MINUTES, NEXT_POLL_MIN_MINUTES, PINNACLE
from .flags import pinnacle_reference

MISSING = {"close_kind": "missing", "close_fair": None, "close_line": None, "close_minutes_to_kick": None,
           "close_ev": None}


def closing_view(ticket: dict, close_rows: list[dict], tables: dict) -> dict:
    """close_kind / close_fair / close_line / close_minutes_to_kick / close_ev for one ticket from the close snapshot rows
    of its game. 'missing' when Pinnacle has no complete two-way quote at the close (or its line is off the table's
    half-point grid, or the table has no answer there)."""
    market, side = ticket["market"], ticket["side"]
    rows = [r for r in close_rows if r.get("market") == market and not r.get("player_name")]
    pin = pinnacle_reference(rows, market)
    table = tables.get((ticket["sport"], market))
    if pin is None or table is None:
        return dict(MISSING)
    v = adjust.value_side(market, side, pin.fair[side], pin.line[side], ticket["book_line"], ticket["price"],
                          table.pmf(pin.threshold), cap=None, strict=False)
    if v is None:
        return dict(MISSING)
    kick = as_utc(ticket["commence_time"])
    captured = max(as_utc(r["captured_at"]) for r in rows if r.get("book") == PINNACLE)
    return {"close_kind": "pinnacle_novig", "close_fair": pin.fair[side], "close_line": pin.line[side],
            "close_minutes_to_kick": (kick - captured).total_seconds() / 60.0, "close_ev": v.ev}


def available_next_poll(ticket: dict, later_rows: list[dict]) -> bool | None:
    """Was the same book still offering the same line at the same or a better price one poll later?

    `later_rows`: odds_snapshot rows of the ticket's game and market captured after the ticket. The next poll is the first
    snapshot, at least 30 minutes after the ticket, that carries a Pinnacle quote for the market (the monitor's polls
    always do, so do the regular pulls; a snapshot without Pinnacle is not a poll). It must be within 3 hours of the
    ticket, otherwise the answer is None (not False): the number was not observed again in time."""
    obs = as_utc(ticket["observed_at"])
    by_time = defaultdict(list)
    for r in later_rows:
        if r.get("market") == ticket["market"] and not r.get("player_name"):
            by_time[as_utc(r["captured_at"])].append(r)
    polls = sorted(t for t, rs in by_time.items()
                   if t - obs >= timedelta(minutes=NEXT_POLL_MIN_MINUTES) and any(r.get("book") == PINNACLE for r in rs))
    if not polls or polls[0] - obs > timedelta(minutes=NEXT_POLL_MAX_MINUTES):
        return None
    need = decimal_odds(ticket["price"])
    return any(r.get("book") == ticket["book"] and r.get("side") == ticket["side"] and r.get("line") is not None
               and r.get("price") is not None and abs(float(r["line"]) - float(ticket["book_line"])) <= LINE_EPS
               and decimal_odds(int(r["price"])) >= need - 1e-9 for r in by_time[polls[0]])


def grade_ticket(ticket: dict, final: dict | None, close_rows: list[dict], later_rows: list[dict], tables: dict) -> dict | None:
    """One early_ticket_results row, or None while the game has no result."""
    f = final or {}
    result = ref.settle_game(ticket["market"], ticket["side"], ticket["book_line"], f.get("actual_margin"),
                             f.get("actual_total"))
    if result is None:
        return None
    view = closing_view(ticket, close_rows, tables)
    close_ev = view["close_ev"]
    return {
        "ticket_id": ticket["id"], "sport": ticket["sport"], "game_pk": ticket["game_pk"], "market": ticket["market"],
        "side": ticket["side"], "book": ticket["book"], "book_line": ticket["book_line"], "price": ticket["price"],
        "observed_at": ticket["observed_at"], "commence_time": ticket["commence_time"], "result": result,
        "units": ref.units_at_pick(result, ticket["price"]), **view,
        "beat_close": None if close_ev is None else close_ev > 0,
        "clv_pts": None if view["close_line"] is None else ref.clv_pts(
            ticket["market"], ticket["side"], ticket["book_line"], view["close_line"]),
        "available_next_poll": available_next_poll(ticket, later_rows),
    }


def grade_tickets(tickets: list[dict], finals: dict, close_by_game: dict, later_by_game: dict, tables: dict) -> list[dict]:
    """Grade every ticket whose game is final. `finals` {(sport, game_pk): {actual_margin, actual_total}};
    `close_by_game` / `later_by_game` {game_pk: [snapshot rows]}."""
    out = []
    for t in tickets:
        g = grade_ticket(t, finals.get((t["sport"], t["game_pk"])), close_by_game.get(t["game_pk"], []),
                         later_by_game.get(t["game_pk"], []), tables)
        if g is not None:
            out.append(g)
    return out
````

- [ ] **Step 4: Write the grading script and workflow**

Create `scripts/grade_early_tickets.py`:

````python
"""Grade the early-week paper tickets against the result and Pinnacle's pre-kick close -> early_ticket_results (daily).

For every ticket whose game started more than 4 hours ago and that has no result row yet: result (W / L / P) and units at
the ticket's price, `close_ev` (the ticket's line and price valued at Pinnacle's closing no-vig fair through the same
landing-table adjustment as the flag), `beat_close`, `clv_pts`, and `available_next_poll` (same book, same line, same or
better price one poll later). The close is Pinnacle's two-way quote in the `close_capture` lines snapshot written by
scripts/capture_close.py; no Pinnacle quote there means close_kind 'missing' (the ticket is excluded from the test). Results
come from rows we already store (prediction_accuracy, odds_snapshot, close_capture): no network, no Odds API credits.
A game with no final result stays ungraded. Idempotent.

    uv run python scripts/grade_early_tickets.py [--days 14]
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.early import landing  # noqa: E402
from sportsmodel.early.constants import NEXT_POLL_MAX_MINUTES, SETTLE_HOURS  # noqa: E402
from sportsmodel.early.grading import grade_tickets  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

MIN_DAYS, MAX_DAYS = 1, 60     # --days is clamped to this range
LATER_HOURS = int(NEXT_POLL_MAX_MINUTES // 60)


@dataclass
class Loaders:
    missing_tables: Callable[[], list[str]]
    pending_tickets: Callable[[int], list[dict]]
    finals: Callable[[list[int]], dict]
    close_rows: Callable[[list[int]], dict]            # {game_pk: close-capture snapshot rows}
    later_rows: Callable[[list[int]], dict]            # {game_pk: odds_snapshot rows within 3 h after any of its tickets}
    load_tables: Callable[[], dict]
    store: Callable[[list[dict]], int]


def grade_all(ld: Loaders, days: int) -> dict:
    gone = ld.missing_tables()
    if gone:
        print(f"::warning::grade-early-tickets: table {gone[0]} missing — run db/migration_early_tickets.sql", flush=True)
        return {"status": "no_table"}
    tickets = ld.pending_tickets(days)
    if not tickets:
        return {"status": "ok", "pending": 0, "graded": 0}
    pks = sorted({t["game_pk"] for t in tickets})
    results = grade_tickets(tickets, ld.finals(pks), ld.close_rows(pks), ld.later_rows(pks), ld.load_tables())
    n = ld.store(results) if results else 0
    kinds: dict[str, int] = {}
    for r in results:
        kinds[r["close_kind"]] = kinds.get(r["close_kind"], 0) + 1
    return {"status": "ok", "pending": len(tickets), "graded": n, "close_kinds": kinds}


def _db_loaders() -> Loaders:
    from sportsmodel import db

    def q(sql, params):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]

    def by_game(rows):
        out = defaultdict(list)
        for r in rows:
            out[r["game_pk"]].append(r)
        return dict(out)

    def pending(days):
        cols = ", ".join(f"t.{c}" for c in ["id"] + db.EARLY_COLUMNS["early_tickets"])
        return q(f"SELECT {cols} FROM early_tickets t "
                 f"WHERE t.commence_time <= now() - interval '{SETTLE_HOURS} hours' "
                 f"AND t.commence_time >= now() - make_interval(days => %s) "
                 f"AND NOT EXISTS (SELECT 1 FROM early_ticket_results r WHERE r.ticket_id = t.id) ORDER BY t.id", (days,))

    def finals(pks):
        rows = q("SELECT sport, game_pk, actual_margin, actual_total FROM prediction_accuracy WHERE game_pk = ANY(%s)", (pks,))
        return {(r["sport"], r["game_pk"]): r for r in rows}

    def close_rows(pks):
        return by_game(q(
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM close_capture c JOIN odds_snapshot s ON s.game_pk = c.game_pk AND s.captured_at = c.captured_at "
            "WHERE c.game_pk = ANY(%s) AND c.kind = 'lines' AND c.status = 'ok' AND s.player_name = '' "
            "AND s.market IN ('spread', 'total')", (pks,)))

    def later_rows(pks):
        return by_game(q(
            "SELECT s.game_pk, s.market, s.side, s.player_name, s.book, s.line, s.price, s.captured_at "
            "FROM odds_snapshot s WHERE s.game_pk = ANY(%s) AND s.player_name = '' AND s.market IN ('spread', 'total') "
            "AND EXISTS (SELECT 1 FROM early_tickets t WHERE t.game_pk = s.game_pk AND s.captured_at > t.observed_at "
            f"AND s.captured_at <= t.observed_at + interval '{LATER_HOURS} hours')", (pks,)))

    return Loaders(
        missing_tables=lambda: db.missing_early_tables(), pending_tickets=pending, finals=finals,
        close_rows=close_rows, later_rows=later_rows, load_tables=landing.load_tables,
        store=db.upsert_early_ticket_results)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--days", type=int, default=14, help="grade games that started within this many days (1-60)")
    args = ap.parse_args(argv)
    args.days = max(MIN_DAYS, min(MAX_DAYS, args.days))
    if writes_disabled():
        print("grade-early-tickets: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::grade-early-tickets: DATABASE_URL is not set")
        return 1
    print(f"[grade-early-tickets] {grade_all(_db_loaders(), args.days)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

Create `.github/workflows/grade-early-tickets.yml`:

````yaml
name: grade-early-tickets

# Daily: grade the early-week paper tickets against the result and Pinnacle's pre-kick close -> early_ticket_results
# (scripts/grade_early_tickets.py; db/migration_early_tickets.sql). Every ticket whose game started more than 4 hours ago and
# has no result yet gets: W/L/P and units at the ticket's price, close_ev (the ticket's line and price valued at Pinnacle's
# closing fair), beat_close, clv_pts and available_next_poll. Reads rows we already store (prediction_accuracy, odds_snapshot,
# close_capture): no Odds API credits, no network. Daily 14:20 UTC, after grade-predictions (every 6h) and grade-ledger
# (14:10) so the prior night's finals are in. Idempotent.
on:
  workflow_dispatch:
    inputs:
      days:
        description: "Grade games that started within this many days"
        type: string
        default: "14"
  schedule:
    - cron: "20 14 * * *"

permissions:
  contents: read

concurrency:
  group: grade-early-tickets
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
      - name: Grade early tickets (result, units, close EV, availability)
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          DAYS: ${{ github.event.inputs.days || '14' }}
        run: uv run python scripts/grade_early_tickets.py --days "$DAYS"
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early`
Expected: all PASS (the grading files: 36 passed, plus the grading workflow test).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/early/grading.py scripts/grade_early_tickets.py .github/workflows/grade-early-tickets.yml tests/early/test_grading.py tests/early/test_grade_script.py tests/early/test_workflows.py
git commit -m "feat(early): grade tickets against Pinnacle's close (close EV, line CLV, availability)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The frozen pass test (pure)

**Files:**
- Create: `src/sportsmodel/early/passtest.py`
- Create: `tests/early/test_passtest.py`

**Interfaces:**
- Consumes: `numpy`; graded-result dicts shaped like `early_ticket_results` rows (`sport, market, game_pk, close_kind, close_ev, beat_close, available_next_poll, result, units`); `tests/early/builders.py` helpers `res`, `many`, `totals`, `spreads` (added in Task 4).
- Produces:
  - constants `MIN_GRADED_TOTAL = 150`, `MIN_CFB_TOTALS = 60`, `MIN_OTHER_MARKET = 40`, `MIN_MEAN_CLOSE_EV = 0.015`, `MIN_BEAT_RATE = 0.55`, `BOOTSTRAP_RESAMPLES = 10_000`, `BOOTSTRAP_SEED = 20261013`, `LOWER_PERCENTILE = 2.5`, `MARKETS = (("cfb","total"), ("cfb","spread"), ("nfl","spread"))`, `LABELS`
  - `passtest.usable(r) -> bool`, `passtest.usable_rows(results, sport=None, market=None) -> list[dict]`
  - `passtest.cluster_bootstrap_lower(rows, *, resamples=10_000, seed=20261013, percentile=2.5) -> float | None`
  - `passtest.market_stats(rows, *, bootstrap=True) -> dict` (`n, games, mean_close_ev, lower, beat_rate, avail_n, avail_observed, avail_share, avail_mean_close_ev, w, l, p, units, roi`)
  - `passtest.ready(results) -> bool`; `passtest.judge_market(sport, market, results) -> dict` (`verdict` `PASS` / `FAIL` / `INSUFFICIENT`, `stats`, `criteria`, `detail`)
  - `passtest.decide(results, previous, as_of) -> dict` (`status` `collecting` or `judged`; a `judged` previous is returned unchanged)

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_passtest.py`:

````python
"""passtest.py: collecting vs judged, per-market minimums, each of the four criteria, the cluster bootstrap. Pure."""
import inspect

import pytest

from sportsmodel.early import passtest

from .builders import many, res, spreads, totals


# ---- frozen numbers ----------------------------------------------------------------------------------------

def test_the_frozen_numbers_match_the_spec():
    assert (passtest.MIN_GRADED_TOTAL, passtest.MIN_CFB_TOTALS, passtest.MIN_OTHER_MARKET) == (150, 60, 40)
    assert (passtest.MIN_MEAN_CLOSE_EV, passtest.MIN_BEAT_RATE) == (0.015, 0.55)
    assert (passtest.BOOTSTRAP_RESAMPLES, passtest.BOOTSTRAP_SEED, passtest.LOWER_PERCENTILE) == (10_000, 20261013, 2.5)
    sig = inspect.signature(passtest.cluster_bootstrap_lower).parameters
    assert sig["resamples"].default == 10_000 and sig["seed"].default == 20261013 and sig["percentile"].default == 2.5


# ---- collecting versus judged ----------------------------------------------------------------------------------

def test_before_the_minimums_the_status_is_collecting_and_nothing_can_pass_or_fail():
    assert passtest.decide([], None, "2026-10-20")["status"] == "collecting"
    out = passtest.decide(totals(59) + spreads(45), None, "2026-10-20")          # 149 usable: one short of 150 and of 60 totals
    assert out["status"] == "collecting" and "markets" not in out
    assert out["graded"] == 59 + 90 and out["cfb_totals"] == 59


def test_both_conditions_are_needed_150_graded_and_60_cfb_totals():
    assert passtest.ready(totals(60) + spreads(45)) is True                      # 60 + 90 = 150
    assert passtest.ready(totals(60) + spreads(44)) is False                     # 148 graded
    assert passtest.ready(totals(59) + spreads(46)) is False                     # 151 graded but only 59 totals
    assert passtest.ready(totals(150)) is True


def test_only_tickets_with_a_pinnacle_close_count_toward_the_minimums():
    missing = many(40, kind="missing", ev=None, game_start=9000)
    assert passtest.ready(totals(60) + spreads(45) + missing) is True
    assert passtest.ready(totals(60) + spreads(44) + missing) is False
    nulls = [res(kind="pinnacle_novig", ev=None, game_pk=g) for g in range(8000, 8010)]
    assert passtest.ready(totals(60) + spreads(44) + nulls) is False


def test_judging_happens_once_and_is_frozen_afterwards():
    first = passtest.decide(totals(60) + spreads(45), None, "2026-12-01")
    assert first["status"] == "judged" and first["judged_on"] == "2026-12-01"
    later = passtest.decide(totals(60, ev=-0.5) + spreads(45, ev=-0.5), first, "2026-12-08")
    assert later is first and later["judged_on"] == "2026-12-01"                  # new, worse data cannot re-judge it
    still = passtest.decide([], {"status": "collecting"}, "2026-12-08")
    assert still["status"] == "collecting"                                         # a collecting previous is not frozen


# ---- per-market minimums -----------------------------------------------------------------------------------------

def test_cfb_spreads_and_nfl_spreads_need_40_graded_tickets_else_insufficient():
    rows = (totals(100) + many(50, sport="cfb", market="spread", game_start=2000)
            + many(39, sport="nfl", market="spread", game_start=3000))
    out = passtest.decide(rows, None, "d")
    assert out["status"] == "judged"
    verdicts = {k: v["verdict"] for k, v in out["markets"].items()}
    assert verdicts == {"cfb_total": "PASS", "cfb_spread": "PASS", "nfl_spread": "INSUFFICIENT"}
    assert "39 graded tickets" in out["markets"]["nfl_spread"]["detail"] and out["markets"]["nfl_spread"]["criteria"] == []


def test_forty_graded_spread_tickets_are_enough_to_be_judged():
    rows = totals(100) + many(40, sport="cfb", market="spread", game_start=2000) + many(40, sport="nfl", market="spread", game_start=3000)
    out = passtest.decide(rows, None, "d")
    assert {v["verdict"] for v in out["markets"].values()} == {"PASS"}


def test_each_market_is_judged_on_its_own_tickets():
    rows = totals(60) + many(45, sport="cfb", market="spread", game_start=2000, ev=-0.05) + many(45, sport="nfl", market="spread", game_start=3000)
    out = passtest.decide(rows, None, "d")
    assert {k: v["verdict"] for k, v in out["markets"].items()} == {"cfb_total": "PASS", "cfb_spread": "FAIL", "nfl_spread": "PASS"}


# ---- the four criteria, one failing at a time --------------------------------------------------------------------------

def verdict(rows, sport="cfb", market="total"):
    v = passtest.judge_market(sport, market, rows)
    return v["verdict"], [c["passed"] for c in v["criteria"]]


def test_all_four_criteria_met_passes():
    assert verdict(totals(60)) == ("PASS", [True, True, True, True])


def test_criterion_1_mean_close_ev_at_least_one_and_a_half_percent():
    assert verdict(totals(60, ev=0.015)) == ("PASS", [True, True, True, True])            # the boundary is inclusive
    assert verdict(totals(60, ev=0.0149)) == ("FAIL", [False, True, True, True])


def test_criterion_2_clustered_lower_bound_above_zero():
    noisy = many(34, ev=0.3, game_start=1000) + many(26, ev=-0.3, game_start=2000)          # mean +4%, 56.7% beat the close
    assert verdict(noisy) == ("FAIL", [True, False, True, True])


def test_criterion_3_beat_close_rate_at_least_55_percent():
    rows = many(29, ev=0.10, game_start=1000) + many(31, ev=-0.002, game_start=2000)         # mean +4.7%, only 48% beat
    assert verdict(rows) == ("FAIL", [True, True, False, True])
    exactly = many(33, ev=0.05, game_start=1000) + many(27, ev=0.02, beat=False, game_start=2000)   # 33 / 60 = 55.0%
    assert verdict(exactly)[1][2] is True


def test_criterion_4_tickets_still_available_a_poll_later_must_also_have_a_positive_mean():
    never = totals(60, avail=False)
    assert verdict(never) == ("FAIL", [True, True, True, False])
    unobserved = totals(60, avail=None)
    assert verdict(unobserved) == ("FAIL", [True, True, True, False])                         # NULL is not evidence of availability
    rows = many(10, ev=-0.2, avail=True, game_start=5000) + many(50, ev=0.2, avail=False, game_start=1000)
    v = passtest.judge_market("cfb", "total", rows)
    assert v["criteria"][3]["passed"] is False and v["criteria"][0]["passed"] is True
    assert "-0.2000 over 10 tickets" in v["criteria"][3]["detail"]


def test_realized_roi_is_reported_never_gated():
    losing = totals(60, units=-1.0, result="L")                      # every ticket lost money, but beat the close
    v = passtest.judge_market("cfb", "total", losing)
    assert v["verdict"] == "PASS" and v["stats"]["roi"] == -1.0 and v["stats"]["l"] == 60 and v["stats"]["units"] == -60.0


def test_market_stats_counts_results_units_and_availability_share():
    rows = [res(avail=True, result="W", units=0.9), res(avail=False, result="L", units=-1.0, game_pk=2),
            res(avail=None, result="P", units=0.0, game_pk=3)]
    s = passtest.market_stats(rows)
    assert (s["n"], s["games"], s["w"], s["l"], s["p"]) == (3, 3, 1, 1, 1)
    assert s["units"] == pytest.approx(-0.1) and s["roi"] == pytest.approx(-0.1 / 3)
    assert s["avail_share"] == 0.5 and s["avail_observed"] == 2 and s["avail_n"] == 1


# ---- the cluster bootstrap ---------------------------------------------------------------------------------------

def test_a_constant_close_ev_has_that_value_as_its_lower_bound():
    assert passtest.cluster_bootstrap_lower(many(20, ev=0.03)) == pytest.approx(0.03)


def test_resampling_whole_games_not_tickets_keeps_the_correlated_tickets_together():
    game_a = [res(game_pk=1, ev=0.5) for _ in range(30)]
    game_b = [res(game_pk=2, ev=-0.5) for _ in range(30)]
    # ticket-level resampling would give a tight interval around 0; two games can both be drawn as B: mean -0.5
    assert passtest.cluster_bootstrap_lower(game_a + game_b) == pytest.approx(-0.5)


def test_same_seed_same_answer_and_a_different_seed_a_different_one():
    rows = [res(game_pk=g, ev=((g * 37) % 11 - 4) / 100) for g in range(80)]
    first = passtest.cluster_bootstrap_lower(rows)
    assert passtest.cluster_bootstrap_lower(rows) == first
    assert passtest.cluster_bootstrap_lower(list(reversed(rows))) == first            # input order does not matter
    assert passtest.cluster_bootstrap_lower(rows, seed=1) != first


def test_the_bootstrap_uses_seed_20261013_and_10000_resamples(monkeypatch):
    seen = {"seeds": [], "rows": 0}
    real = passtest.np.random.default_rng

    class Spy:
        def __init__(self, seed):
            self.rng = real(seed)
            seen["seeds"].append(seed)

        def integers(self, low, high, size):
            seen["rows"] += size[0]
            return self.rng.integers(low, high, size=size)

    monkeypatch.setattr(passtest.np.random, "default_rng", Spy)
    passtest.cluster_bootstrap_lower(many(25, ev=0.01))
    assert seen["seeds"] == [20261013] and seen["rows"] == 10_000


def test_the_lower_bound_is_the_2_5th_percentile_below_the_mean():
    rows = [res(game_pk=g, ev=((g * 37) % 11 - 4) / 100) for g in range(80)]
    mean = passtest.market_stats(rows, bootstrap=False)["mean_close_ev"]
    assert passtest.cluster_bootstrap_lower(rows) < mean


def test_an_empty_sample_has_no_bound():
    assert passtest.cluster_bootstrap_lower([]) is None
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_passtest.py`
Expected: collection ERROR, `ImportError: cannot import name 'passtest' from 'sportsmodel.early'`.

- [ ] **Step 3: Write the implementation**

Create `src/sportsmodel/early/passtest.py`:

````python
"""The pre-registered pass test for the early-week monitor. PURE (numpy only) -- no IO.

FROZEN as of docs/superpowers/specs/2026-10-13-early-week-monitor-design.md section 4: nothing here may change once
results exist. The test is judged ONCE, at the first weekly report where there are at least 150 graded tickets with a
Pinnacle close across the three markets AND at least 60 of them are CFB totals; before that the status is "collecting"
and running numbers cannot pass or fail anything. Each market is then judged separately (CFB spreads and NFL spreads
need at least 40 graded tickets, else "insufficient") and passes only if ALL four hold:
  1. mean close_ev >= +0.015
  2. the game-clustered 95% lower bound of mean close_ev > 0 (cluster bootstrap over games: 10,000 resamples, seed
     20261013, 2.5th percentile)
  3. beat_close rate >= 55%
  4. mean close_ev over tickets with available_next_poll = true > 0
Realized ROI is reported, never gated.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MIN_GRADED_TOTAL = 150
MIN_CFB_TOTALS = 60
MIN_OTHER_MARKET = 40
MIN_MEAN_CLOSE_EV = 0.015
MIN_BEAT_RATE = 0.55
EPS = 1e-12                                  # float noise only: a mean of exactly 0.015 passes
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261013
LOWER_PERCENTILE = 2.5
CHUNK = 1000                                  # resamples drawn per block: bounds memory, fixed so results are reproducible
MARKETS = (("cfb", "total"), ("cfb", "spread"), ("nfl", "spread"))
LABELS = {("cfb", "total"): "CFB totals", ("cfb", "spread"): "CFB spreads", ("nfl", "spread"): "NFL spreads"}
CRITERIA = ("mean close_ev >= +1.5%", "clustered 95% lower bound > 0", "beat-close rate >= 55%",
            "mean close_ev where available_next_poll = true > 0")


def usable(r: dict) -> bool:
    """A graded ticket that counts toward the test: it has a Pinnacle close and a close EV."""
    return r.get("close_kind") == "pinnacle_novig" and r.get("close_ev") is not None


def usable_rows(results: list[dict], sport: str | None = None, market: str | None = None) -> list[dict]:
    return [r for r in results if usable(r) and (sport is None or r["sport"] == sport)
            and (market is None or r["market"] == market)]


def cluster_bootstrap_lower(rows: list[dict], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                            percentile: float = LOWER_PERCENTILE) -> float | None:
    """Lower bound of the mean close_ev: resample GAMES with replacement (all tickets of a drawn game come along), take
    each resample's ticket-weighted mean, return the `percentile`-th percentile. Deterministic for a given seed."""
    per_game: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        per_game[(r["sport"], r["game_pk"])].append(float(r["close_ev"]))
    if not per_game:
        return None
    games = sorted(per_game)
    sums = np.array([sum(per_game[g]) for g in games])
    counts = np.array([len(per_game[g]) for g in games], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, CHUNK):
        idx = rng.integers(0, len(games), size=(min(CHUNK, resamples - start), len(games)))
        means.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return float(np.percentile(np.concatenate(means), percentile))


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def market_stats(rows: list[dict], *, bootstrap: bool = True) -> dict:
    """Running numbers for one market's usable tickets (also the four test statistics)."""
    evs = [float(r["close_ev"]) for r in rows]
    avail = [r for r in rows if r.get("available_next_poll") is True]
    seen = [r for r in rows if r.get("available_next_poll") is not None]
    units = [float(r["units"]) for r in rows if r.get("units") is not None]
    return {
        "n": len(rows), "games": len({(r["sport"], r["game_pk"]) for r in rows}), "mean_close_ev": _mean(evs),
        "lower": cluster_bootstrap_lower(rows) if (bootstrap and rows) else None,
        "beat_rate": _mean([1.0 if r.get("beat_close") else 0.0 for r in rows]),
        "avail_n": len(avail), "avail_observed": len(seen), "avail_share": _mean([1.0 if r["available_next_poll"] else 0.0 for r in seen]),
        "avail_mean_close_ev": _mean([float(r["close_ev"]) for r in avail]),
        "w": sum(1 for r in rows if r.get("result") == "W"), "l": sum(1 for r in rows if r.get("result") == "L"),
        "p": sum(1 for r in rows if r.get("result") == "P"), "units": sum(units), "roi": _mean(units),
    }


def ready(results: list[dict]) -> bool:
    """The judgement point: >= 150 usable graded tickets across the three markets and >= 60 of them CFB totals."""
    rows = [r for r in usable_rows(results) if (r["sport"], r["market"]) in MARKETS]
    return len(rows) >= MIN_GRADED_TOTAL and len(usable_rows(results, "cfb", "total")) >= MIN_CFB_TOTALS


def judge_market(sport: str, market: str, results: list[dict]) -> dict:
    """PASS / FAIL / INSUFFICIENT for one market, with the four criteria and their numbers."""
    rows = usable_rows(results, sport, market)
    stats = market_stats(rows)
    floor = MIN_CFB_TOTALS if (sport, market) == ("cfb", "total") else MIN_OTHER_MARKET
    if stats["n"] < floor:
        return {"sport": sport, "market": market, "verdict": "INSUFFICIENT", "stats": stats, "criteria": [],
                "detail": f"{stats['n']} graded tickets with a Pinnacle close, {floor} needed"}
    checks = [
        (stats["mean_close_ev"] + EPS >= MIN_MEAN_CLOSE_EV, f"{stats['mean_close_ev']:+.4f} (need >= {MIN_MEAN_CLOSE_EV:+.3f})"),
        (stats["lower"] > 0, f"{stats['lower']:+.4f} (need > 0)"),
        (stats["beat_rate"] + EPS >= MIN_BEAT_RATE, f"{stats['beat_rate']:.1%} (need >= {MIN_BEAT_RATE:.0%})"),
        (stats["avail_mean_close_ev"] is not None and stats["avail_mean_close_ev"] > 0,
         "no ticket with available_next_poll = true" if stats["avail_mean_close_ev"] is None
         else f"{stats['avail_mean_close_ev']:+.4f} over {stats['avail_n']} tickets (need > 0)"),
    ]
    criteria = [{"name": n, "passed": bool(ok), "detail": d} for n, (ok, d) in zip(CRITERIA, checks)]
    verdict = "PASS" if all(c["passed"] for c in criteria) else "FAIL"
    return {"sport": sport, "market": market, "verdict": verdict, "stats": stats, "criteria": criteria, "detail": ""}


def decide(results: list[dict], previous: dict | None, as_of: str) -> dict:
    """The decision block of this week's report. Once a previous report judged the test the verdicts are frozen and
    carried forward untouched (the test is judged ONCE); otherwise judge now if the judgement point is reached, else
    'collecting'."""
    if previous and previous.get("status") == "judged":
        return previous
    graded = len([r for r in usable_rows(results) if (r["sport"], r["market"]) in MARKETS])
    progress = {"graded": graded, "need_graded": MIN_GRADED_TOTAL, "cfb_totals": len(usable_rows(results, "cfb", "total")),
                "need_cfb_totals": MIN_CFB_TOTALS}
    if not ready(results):
        return {"status": "collecting", **progress}
    verdicts = {f"{s}_{m}": judge_market(s, m, results) for s, m in MARKETS}
    return {"status": "judged", "judged_on": as_of, **progress, "markets": verdicts}
````

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early/test_passtest.py`
Expected: 21 passed.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/early/passtest.py tests/early/test_passtest.py
git commit -m "feat(early): the frozen pass test and its game-clustered bootstrap" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Weekly report (`report_early_tickets.py`)

**Files:**
- Create: `src/sportsmodel/early/report.py`
- Create: `scripts/report_early_tickets.py`, `.github/workflows/report-early-tickets.yml`
- Modify: `tests/early/test_workflows.py` (append the report group and the no-git-writes guard)
- Create: `tests/early/test_report.py`, `tests/early/test_report_script.py`

**Interfaces:**
- Consumes: Task 8 `passtest.decide`, `passtest.usable`, `passtest.usable_rows`, `passtest.market_stats`, `passtest.MARKETS`, `passtest.LABELS`; Task 1 `db.missing_early_tables`; `constants.SETTLE_HOURS`; existing `ledger.core.as_utc`, `ledger.record.writes_disabled`.
- Produces:
  - `report.KEYS` (`{(sport, market): "cfb_total" | "cfb_spread" | "nfl_spread"}`), `report.summarize(tickets, results, as_of) -> dict` (`tickets` per market with `all` / `week`, `tickets_total`, `graded`, `with_close`, `missing_close`, `missing_share`, `ungraded`, `markets` running stats), `report.render_report(*, as_of, summary, decision) -> str`, `report.parse_report(text) -> dict | None`, `report.latest_decision(reports, as_of) -> dict | None`
  - script: `collect(q) -> (tickets, results)`, `has_activity(tickets, results, as_of) -> bool`, `build_report(as_of, tickets, results, previous) -> (text, decision)`, `main(argv)`; workflow `report-early-tickets.yml`; report files `docs/superpowers/reports/YYYY-MM-DD-early-monitor.md`

- [ ] **Step 1: Write the failing tests**

Create `tests/early/test_report.py`:

````python
"""report.py: summary numbers, the markdown for both statuses and the decision block round trip. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel.early import passtest, report

from .builders import res, spreads, totals

AS_OF = datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc)          # a Tuesday


def tick(i, sport="cfb", market="total", days_ago=2, kick_hours_ago=60):
    return {"id": i, "sport": sport, "market": market, "observed_at": AS_OF - timedelta(days=days_ago),
            "commence_time": AS_OF - timedelta(hours=kick_hours_ago)}


def result(ticket_id, **kw):
    return res(**kw) | {"ticket_id": ticket_id}


def test_summary_counts_tickets_by_market_and_week_and_ungraded():
    tickets = [tick(1), tick(2, days_ago=10), tick(3, sport="nfl", market="spread"), tick(4, kick_hours_ago=3),
               tick(5, kick_hours_ago=100)]
    s = report.summarize(tickets, [result(1)], AS_OF)
    assert s["tickets"]["cfb_total"] == {"all": 4, "week": 3} and s["tickets"]["nfl_spread"] == {"all": 1, "week": 1}
    assert s["tickets"]["cfb_spread"] == {"all": 0, "week": 0} and s["tickets_total"] == 5
    assert s["ungraded"] == 3          # 2, 3 and 5 started long ago without a result; 4 kicked off 3 hours ago (not yet due); 1 graded


def test_summary_close_coverage_and_missing_share():
    results = [result(1), result(2, kind="missing", ev=None), result(3), result(4, kind="missing", ev=None)]
    s = report.summarize([], results, AS_OF)
    assert (s["graded"], s["with_close"], s["missing_close"], s["missing_share"]) == (4, 2, 2, 0.5)
    assert report.summarize([], [], AS_OF)["missing_share"] is None


def test_running_numbers_use_only_tickets_with_a_pinnacle_close():
    results = [result(1, ev=0.04, avail=True), result(2, ev=-0.02, avail=False, game_pk=2),
               result(3, kind="missing", ev=None, game_pk=3)]
    m = report.summarize([], results, AS_OF)["markets"]["cfb_total"]
    assert m["n"] == 2 and m["mean_close_ev"] == pytest.approx(0.01) and m["beat_rate"] == 0.5 and m["avail_share"] == 0.5


def test_a_collecting_report_states_progress_and_never_says_pass_or_fail():
    results = [result(i, game_pk=i) for i in range(10)]
    decision = passtest.decide(results, None, "2026-11-03")
    text = report.render_report(as_of=AS_OF, summary=report.summarize([], results, AS_OF), decision=decision)
    assert "COLLECTING" in text and "10 of 150" in text and "10 of 60 CFB totals" in text
    assert "PASS" not in text and "FAIL" not in text and "INSUFFICIENT" not in text
    assert "bowl season" in text and "Running numbers" in text


def test_a_judged_report_lists_each_markets_verdict_and_criteria():
    results = [result(i, game_pk=i) for i in range(100)] + [
        result(1000 + i, sport="cfb", market="spread", game_pk=1000 + i) for i in range(50)] + [
        result(2000 + i, sport="nfl", market="spread", game_pk=2000 + i) for i in range(39)]
    decision = passtest.decide(results, None, "2026-11-03")
    text = report.render_report(as_of=AS_OF, summary=report.summarize([], results, AS_OF), decision=decision)
    assert "JUDGED on 2026-11-03" in text and "| CFB totals | PASS |" in text and "| CFB spreads | PASS |" in text
    assert "| NFL spreads | INSUFFICIENT |" in text and "39 graded tickets" in text
    assert "separate short design for live alerts" in text and "never gated" in text
    assert "informational only" not in text                       # judged this very week


def test_a_later_report_carries_the_frozen_verdict_and_says_so():
    frozen = passtest.decide(totals(100) + spreads(45), None, "2026-11-03")
    later = datetime(2026, 11, 17, 15, 52, tzinfo=timezone.utc)
    text = report.render_report(as_of=later, summary=report.summarize([], [], later), decision=frozen)
    assert "JUDGED on 2026-11-03" in text and "Later weeks below are informational only" in text


def test_the_decision_block_round_trips_and_the_latest_earlier_one_is_chosen():
    d1 = passtest.decide([], None, "2026-10-20")
    d2 = passtest.decide(totals(100) + spreads(45), None, "2026-11-03")
    r1 = report.render_report(as_of=datetime(2026, 10, 20, 15, 52, tzinfo=timezone.utc), summary=report.summarize([], [], AS_OF), decision=d1)
    r2 = report.render_report(as_of=datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc), summary=report.summarize([], [], AS_OF), decision=d2)
    assert report.parse_report(r2)["decision"]["status"] == "judged"
    assert report.latest_decision([r1, r2], datetime(2026, 11, 10, 15, 52, tzinfo=timezone.utc))["status"] == "judged"
    assert report.latest_decision([r1, r2], datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc))["status"] == "collecting"
    assert report.latest_decision([r1, r2], datetime(2026, 10, 20, 15, 52, tzinfo=timezone.utc)) is None
    assert report.latest_decision(["not a report", ""], AS_OF) is None


def test_parse_ignores_a_corrupt_block():
    assert report.parse_report("<!-- early-monitor: {not json} -->") is None and report.parse_report(None) is None
````

Create `tests/early/test_report_script.py`:

````python
"""scripts/report_early_tickets.py: decision carry-forward, the quiet-week guard, file output and the guards. No database."""
import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel import db
from sportsmodel.early import report

from .builders import res, spreads, totals

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("report_early_tickets", ROOT / "scripts" / "report_early_tickets.py")
rs = importlib.util.module_from_spec(_spec)
sys.modules["report_early_tickets"] = rs
_spec.loader.exec_module(rs)

AS_OF = datetime(2026, 11, 3, 15, 52, tzinfo=timezone.utc)


def ticket(i, days_ago=1):
    return {"id": i, "sport": "cfb", "market": "total", "side": "over", "observed_at": AS_OF - timedelta(days=days_ago),
            "commence_time": AS_OF - timedelta(hours=60)}


def graded(r, i, days_ago=1):
    return r | {"ticket_id": i, "graded_at": AS_OF - timedelta(days=days_ago)}


def test_a_collecting_report_is_built_from_the_numbers():
    results = [graded(r, i) for i, r in enumerate(totals(10), 1)]
    text, decision = rs.build_report(AS_OF, [ticket(i) for i in range(1, 11)], results, None)
    assert decision["status"] == "collecting" and "COLLECTING" in text and "10 of 150" in text


def test_the_first_report_that_meets_both_minimums_judges_and_later_ones_carry_it_forward():
    rows = [graded(r, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    text, decision = rs.build_report(AS_OF, [], rows, None)
    assert decision["status"] == "judged" and decision["judged_on"] == "2026-11-03"
    later = AS_OF + timedelta(days=14)
    worse = [graded(r | {"close_ev": -0.5, "beat_close": False}, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    text2, decision2 = rs.build_report(later, [], worse, report.parse_report(text)["decision"])
    assert decision2 == decision and "Later weeks below are informational only" in text2


def test_activity_means_a_ticket_logged_or_graded_in_the_last_seven_days():
    assert rs.has_activity([ticket(1, days_ago=2)], [], AS_OF) is True
    assert rs.has_activity([ticket(1, days_ago=9)], [], AS_OF) is False
    assert rs.has_activity([ticket(1, days_ago=9)], [graded(res(), 1, days_ago=1)], AS_OF) is True
    assert rs.has_activity([], [], AS_OF) is False
    assert rs.has_activity([ticket(1, days_ago=9)], [res() | {"ticket_id": 1, "graded_at": None}], AS_OF) is False


def run_main(monkeypatch, tmp_path, tickets, results, *extra, missing=()):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setattr(db, "missing_early_tables", lambda *a, **k: list(missing))
    monkeypatch.setattr(rs, "_db_query", lambda: lambda sql, params=(): tickets if "early_tickets" in sql else results)
    return rs.main(["--as-of", AS_OF.isoformat(), "--reports-dir", str(tmp_path), *extra])


def test_main_writes_a_dated_file_and_the_next_run_reads_its_decision_block(monkeypatch, tmp_path):
    rows = [graded(r, i) for i, r in enumerate(totals(100) + spreads(45), 1)]
    assert run_main(monkeypatch, tmp_path, [], rows) == 0
    (path,) = tmp_path.glob("*-early-monitor.md")
    assert path.name == "2026-11-03-early-monitor.md" and "JUDGED on 2026-11-03" in path.read_text()
    assert report.latest_decision([path.read_text()], AS_OF + timedelta(days=7))["status"] == "judged"


def test_a_quiet_week_writes_no_file(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, [ticket(1, days_ago=30)], []) == 0
    assert list(tmp_path.iterdir()) == [] and "no report written" in capsys.readouterr().out


def test_dry_run_prints_and_writes_nothing(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, [ticket(1)], [], "--dry-run") == 0
    assert list(tmp_path.iterdir()) == [] and "# Early-week monitor: report of 2026-11-03" in capsys.readouterr().out


def test_missing_tables_warn_and_exit_zero(monkeypatch, tmp_path, capsys):
    assert run_main(monkeypatch, tmp_path, [], [], missing=["early_tickets"]) == 0
    assert "tables missing" in capsys.readouterr().out and list(tmp_path.iterdir()) == []


def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert rs.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_main_needs_the_database_url(monkeypatch, capsys):
    monkeypatch.setattr(rs, "writes_disabled", lambda: False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert rs.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out


def test_report_queries_are_read_only(monkeypatch):
    seen = []
    monkeypatch.setattr(db, "get_postgres", lambda: pytest.fail("no connection expected"))
    rs.collect(lambda sql, params=(): seen.append(sql) or [])
    assert "FROM early_tickets" in seen[0] and "FROM early_ticket_results" in seen[1]
    assert not any(w in s.upper().split() for s in seen for w in ("INSERT", "UPDATE", "DELETE"))
````

Append to `tests/early/test_workflows.py`:

````python
# ---- report-early-tickets.yml --------------------------------------------------------------------------------

def test_report_runs_tuesday_1552_and_commits_only_the_reports_folder():
    wf = load(REPORT)
    (cron,) = [c["cron"] for c in wf["on"]["schedule"]]
    m, h, mon, dow = cron_sets(cron)
    assert dow == {2} and h == {15} and m == {52}                                                # after grade-early-tickets 14:20
    assert wf["permissions"] == {"contents": "write"} and "workflow_dispatch" in wf["on"]
    assert wf["concurrency"]["group"] == "report-early-tickets"
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    gen = steps[step_index(steps, runs("scripts/report_early_tickets.py"))]
    assert gen["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}
    commit = steps[step_index(steps, runs("git commit"))]
    assert "git add docs/superpowers/reports/\n" in commit["run"] and "[skip ci]" in commit["run"]
    assert "git add ." not in commit["run"] and "git add -A" not in commit["run"]
    assert step_index(steps, runs("scripts/report_early_tickets.py")) < step_index(steps, runs("git commit"))


def test_only_the_report_workflow_writes_to_git_and_none_sets_the_test_guard():
    for name in (MONITOR, GRADE):
        text = (WF / name).read_text()
        assert "git push" not in text and "git commit" not in text, name
    for name in (MONITOR, GRADE, REPORT):
        assert "SM_LEDGER_OFF" not in (WF / name).read_text(), name
````

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/early/test_report.py tests/early/test_report_script.py tests/early/test_workflows.py`
Expected: collection ERRORs (`ImportError: cannot import name 'report'`, `FileNotFoundError` for `scripts/report_early_tickets.py`); the new workflow tests fail with `FileNotFoundError: ... report-early-tickets.yml`.

- [ ] **Step 3: Write the report module**

Create `src/sportsmodel/early/report.py`:

````python
"""The weekly early-monitor report: summary numbers, markdown, and the machine-readable decision block. PURE."""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta

from ..ledger.core import as_utc
from . import passtest
from .constants import SETTLE_HOURS

MARKER = "early-monitor"
KEYS = {(s, m): f"{s}_{m}" for s, m in passtest.MARKETS}


def summarize(tickets: list[dict], results: list[dict], as_of: datetime) -> dict:
    """Counts and running numbers. `tickets`: early_tickets rows (id, sport, market, observed_at, commence_time);
    `results`: early_ticket_results rows."""
    as_of = as_utc(as_of)
    week_ago = as_of - timedelta(days=7)
    counts = {}
    for key, (sport, market) in zip(KEYS.values(), passtest.MARKETS):
        mine = [t for t in tickets if (t["sport"], t["market"]) == (sport, market)]
        counts[key] = {"all": len(mine), "week": sum(1 for t in mine if as_utc(t["observed_at"]) > week_ago)}
    graded_ids = {r["ticket_id"] for r in results if "ticket_id" in r}
    settle = timedelta(hours=SETTLE_HOURS)
    ungraded = sum(1 for t in tickets if t["id"] not in graded_ids and as_utc(t["commence_time"]) <= as_of - settle)
    good = passtest.usable_rows(results)
    missing = sum(1 for r in results if not passtest.usable(r))
    return {
        "tickets": counts, "tickets_total": sum(c["all"] for c in counts.values()),
        "graded": len(results), "with_close": len(good), "missing_close": missing,
        "missing_share": missing / len(results) if results else None, "ungraded": ungraded,
        "markets": {KEYS[(s, m)]: passtest.market_stats(passtest.usable_rows(results, s, m), bootstrap=False)
                    for s, m in passtest.MARKETS},
    }


def _pct(v, spec="{:+.2%}") -> str:
    return "n/a" if v is None else spec.format(v)


def _decision_md(decision: dict, as_of: datetime) -> str:
    if decision["status"] == "collecting":
        return (f"**Status: COLLECTING.** {decision['graded']} of {decision['need_graded']} graded tickets with a Pinnacle "
                f"close across the three markets, {decision['cfb_totals']} of {decision['need_cfb_totals']} CFB totals. "
                "Running numbers cannot pass or fail the test before both minimums are met. If they are not reached by "
                "the end of the CFB bowl season, collection continues next season.")
    head = f"**Status: JUDGED on {decision['judged_on']}.** The test is judged once; this verdict is frozen."
    if decision["judged_on"] != f"{as_utc(as_of):%Y-%m-%d}":
        head += " Later weeks below are informational only."
    rows = []
    for key in (KEYS[mk] for mk in passtest.MARKETS):
        v = decision["markets"][key]
        label = passtest.LABELS[(v["sport"], v["market"])]
        if v["verdict"] == "INSUFFICIENT":
            rows.append(f"| {label} | INSUFFICIENT | {v['detail']} |")
            continue
        detail = "; ".join(f"{c['name']}: {'ok' if c['passed'] else 'MISSED'} ({c['detail']})" for c in v["criteria"])
        rows.append(f"| {label} | {v['verdict']} | {detail} |")
    meaning = ("A market that passes goes to a separate short design for live alerts at small stakes; a later profit check "
               "(at least 300 tickets, ROI 95% lower bound above -3%) decides any stake increase. A market that fails is "
               "dropped. Realized ROI is reported, never gated, at this stage.")
    return head + "\n\n| Market | Verdict | Criteria |\n|---|---|---|\n" + "\n".join(rows) + "\n\n" + meaning


def render_report(*, as_of: datetime, summary: dict, decision: dict) -> str:
    as_of = as_utc(as_of)
    mk = list(zip(KEYS.values(), passtest.MARKETS))
    by_market = "\n".join(
        f"| {passtest.LABELS[m]} | {summary['tickets'][k]['all']} | {summary['tickets'][k]['week']} |" for k, m in mk)
    running = "\n".join(
        f"| {passtest.LABELS[m]} | {s['n']} | {_pct(s['mean_close_ev'])} | {_pct(s['beat_rate'], '{:.1%}')} | "
        f"{_pct(s['avail_share'], '{:.1%}')} ({s['avail_observed']} observed) | {s['w']}/{s['l']}/{s['p']} | "
        f"{_pct(s['units'], '{:+.2f}')} | {_pct(s['roi'], '{:+.1%}')} |"
        for k, m in mk for s in [summary["markets"][k]])
    block = {"as_of": as_of.isoformat(), "decision": decision}
    return f"""# Early-week monitor: report of {as_of:%Y-%m-%d}

Generated by `scripts/report_early_tickets.py` (paper tickets only: no money is at risk).

{_decision_md(decision, as_of)}

## Tickets by market

| Market | All time | Last 7 days |
|---|---|---|
{by_market}

## Close coverage

{summary['graded']} tickets graded, {summary['with_close']} with a Pinnacle close, {summary['missing_close']} without
({_pct(summary['missing_share'], '{:.1%}')} excluded from every number below). {summary['ungraded']} tickets whose game started
more than {SETTLE_HOURS} hours ago are not graded yet (no final result or no grade run since).

## Running numbers (tickets with a Pinnacle close)

| Market | Tickets | Mean close EV | Beat the close | Still available a poll later | W/L/P | Units | ROI per ticket |
|---|---|---|---|---|---|---|---|
{running}

<!-- {MARKER}: {json.dumps(block, sort_keys=True)} -->
"""


_BLOCK = re.compile(rf"<!-- {MARKER}: (\{{.*\}}) -->")


def parse_report(text: str) -> dict | None:
    """The machine-readable block of a report ({'as_of', 'decision'}), or None."""
    m = _BLOCK.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except ValueError:
        return None


def latest_decision(reports: list[str], as_of: datetime) -> dict | None:
    """The decision block of the most recent report strictly before `as_of`, or None."""
    best = None
    for text in reports:
        p = parse_report(text)
        if not p or not p.get("as_of") or "decision" not in p:
            continue
        at = as_utc(p["as_of"])
        if at is not None and at < as_utc(as_of) and (best is None or at > as_utc(best["as_of"])):
            best = p
    return None if best is None else best["decision"]
````

- [ ] **Step 4: Write the report script and workflow**

Create `scripts/report_early_tickets.py`:

````python
"""Weekly early-week monitor report -> docs/superpowers/reports/YYYY-MM-DD-early-monitor.md (profit step 3).

Tickets by market, close coverage, running mean close EV / beat-close rate / availability share, W/L/P and units, and the
decision status of the pre-registered pass test (sportsmodel.early.passtest). The test is judged ONCE, at the first report
where there are >= 150 graded tickets with a Pinnacle close and >= 60 of them CFB totals; until then the status is
"collecting" and nothing can pass or fail. A judged verdict is carried forward unchanged by every later report (it is read
back from the previous report's machine-readable block). Read-only against the database; writes only the markdown file
(the workflow commits it). A week with no new tickets and no new grades writes nothing.

    uv run python scripts/report_early_tickets.py [--as-of 2026-10-20T15:52:00Z] [--reports-dir docs/superpowers/reports]
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.early import passtest, report  # noqa: E402
from sportsmodel.ledger.core import as_utc  # noqa: E402
from sportsmodel.ledger.record import writes_disabled  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"
TICKETS_SQL = "SELECT id, sport, market, side, observed_at, commence_time FROM early_tickets ORDER BY id"
RESULTS_SQL = ("SELECT ticket_id, sport, game_pk, market, side, result, units, close_kind, close_ev, beat_close, clv_pts, "
               "available_next_poll, graded_at FROM early_ticket_results ORDER BY ticket_id")


def _db_query():
    from sportsmodel import db

    def q(sql, params=()):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    return q


def collect(q) -> tuple[list[dict], list[dict]]:
    """(tickets, results) via read-only queries `q(sql, params) -> list of dict rows`."""
    return q(TICKETS_SQL, ()), q(RESULTS_SQL, ())


def has_activity(tickets: list[dict], results: list[dict], as_of: datetime) -> bool:
    """Anything new in the last 7 days: a ticket logged or a ticket graded."""
    since = as_utc(as_of) - timedelta(days=7)
    return (any(as_utc(t["observed_at"]) > since for t in tickets)
            or any(r.get("graded_at") is not None and as_utc(r["graded_at"]) > since for r in results))


def build_report(as_of: datetime, tickets: list[dict], results: list[dict], previous: dict | None) -> tuple[str, dict]:
    decision = passtest.decide(results, previous, f"{as_utc(as_of):%Y-%m-%d}")
    text = report.render_report(as_of=as_of, summary=report.summarize(tickets, results, as_of), decision=decision)
    return text, decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default=None, help="report time (ISO, default: now UTC)")
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the report instead of writing the file")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("report-early-tickets: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::report-early-tickets: DATABASE_URL is not set")
        return 1
    from sportsmodel import db
    if db.missing_early_tables():
        print("::warning::report-early-tickets: early-monitor tables missing — run db/migration_early_tickets.sql", flush=True)
        return 0
    as_of = as_utc(args.as_of) if args.as_of else datetime.now(timezone.utc)
    tickets, results = collect(_db_query())
    if not has_activity(tickets, results, as_of):
        print("[early-monitor-report] no new tickets or grades in the last 7 days; no report written", flush=True)
        return 0
    reports_dir = Path(args.reports_dir)
    previous = report.latest_decision([p.read_text() for p in sorted(reports_dir.glob("*-early-monitor.md"))], as_of)
    text, decision = build_report(as_of, tickets, results, previous)
    if args.dry_run:
        print(text)
        return 0
    path = reports_dir / f"{as_of:%Y-%m-%d}-early-monitor.md"
    path.write_text(text)
    print(f"[early-monitor-report] wrote {path}: status={decision['status']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

Create `.github/workflows/report-early-tickets.yml`:

````yaml
name: report-early-tickets

# Weekly early-week monitor report: every Tuesday after the Monday-night games have finished and been graded, write
# docs/superpowers/reports/YYYY-MM-DD-early-monitor.md (scripts/report_early_tickets.py): tickets by market, close coverage,
# running mean close EV, beat-close rate and availability share, W/L/P and units, and the decision status of the
# pre-registered pass test. The test is judged ONCE, at the first report with >= 150 graded tickets (a Pinnacle close) and
# >= 60 CFB totals; until then the status is "collecting". Read-only against the database; writes only the markdown report. A
# week with no new tickets and no new grades writes nothing. Tuesday 15:52 UTC, after grade-early-tickets (14:20).
on:
  workflow_dispatch: {}
  schedule:
    - cron: "52 15 * * 2"

permissions:
  contents: write

concurrency:
  group: report-early-tickets
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
      - name: Write the weekly report
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/report_early_tickets.py
      - name: Commit the report
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add docs/superpowers/reports/
          if git diff --staged --quiet; then
            echo "no new report"
            exit 0
          fi
          git commit -m "docs(report): early-week monitor [skip ci]"
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

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/early`
Expected: all PASS (the report files: 18 passed, plus the two report-workflow tests; `tests/early` in total: 203 passed before Task 10 adds the floor tests).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/early/report.py scripts/report_early_tickets.py .github/workflows/report-early-tickets.yml tests/early/test_report.py tests/early/test_report_script.py tests/early/test_workflows.py
git commit -m "feat(early): weekly report with the decision status of the pass test" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Credit-floor ordering, go-live checklist and final verification

**Files:**
- Create: `tests/early/test_floors.py`
- Create: `docs/superpowers/reports/2026-10-13-early-monitor-go-live.md`

**Interfaces:**
- Consumes: `polling.budget_decision`, `constants.MONITOR_FLOOR`; existing `ingest.odds` (`INGEST_PROPS_FLOOR`, `INGEST_LINES_FLOOR`, `ingest_budget_decision(remaining) -> (lines_ok, props_ok, reason)`), `ledger.close` (`BUDGET_PROPS_FLOOR`, `BUDGET_ALL_FLOOR`, `budget_decision(remaining) -> BudgetDecision(lines, props, warning)`).
- Produces: the floor-ordering test (pins that the monitor stops after the regular prop pulls and before the regular game lines and the whole close capture); the user's go-live checklist.

- [ ] **Step 1: Write the floor-ordering test**

Create `tests/early/test_floors.py`:

````python
"""The credit floors, as credits fall: regular prop pulls stop first, then the early-week monitor, then regular game-line
pulls, then close-capture props, then all close capture. The monitor must never starve the close capture. No network."""
import pytest

from sportsmodel.early import polling
from sportsmodel.early.constants import MONITOR_FLOOR
from sportsmodel.ingest import odds
from sportsmodel.ledger import close


def test_the_floors_are_strictly_ordered():
    assert odds.INGEST_PROPS_FLOOR == 2500 and MONITOR_FLOOR == 1500 and odds.INGEST_LINES_FLOOR == 800
    assert close.BUDGET_PROPS_FLOOR == 600 and close.BUDGET_ALL_FLOOR == 200
    assert (odds.INGEST_PROPS_FLOOR > MONITOR_FLOOR > odds.INGEST_LINES_FLOOR > close.BUDGET_PROPS_FLOOR
            > close.BUDGET_ALL_FLOOR)


@pytest.mark.parametrize("remaining, ingest_props, monitor, ingest_lines, close_props, close_lines", [
    (20000, True, True, True, True, True),
    (2500, True, True, True, True, True),
    (2499, False, True, True, True, True),             # regular prop pulls stop first
    (1500, False, True, True, True, True),
    (1499, False, False, True, True, True),            # then this monitor
    (800, False, False, True, True, True),
    (799, False, False, False, True, True),            # then regular game-line pulls
    (600, False, False, False, True, True),
    (599, False, False, False, False, True),           # then close-capture props
    (200, False, False, False, False, True),
    (199, False, False, False, False, False),          # then all close capture
])
def test_what_each_job_may_still_spend_at_each_balance(remaining, ingest_props, monitor, ingest_lines, close_props, close_lines):
    lines_ok, props_ok, _ = odds.ingest_budget_decision(remaining)
    capture = close.budget_decision(remaining)
    assert (props_ok, polling.budget_decision(remaining).ok, lines_ok, capture.props, capture.lines) == (
        ingest_props, monitor, ingest_lines, close_props, close_lines)
````

- [ ] **Step 2: Run it**

Run: `uv run pytest -q tests/early/test_floors.py`
Expected: 12 passed (it only reads constants and pure functions that already exist, so it passes at once; it is here to freeze the ordering against future changes to either side).

- [ ] **Step 3: Write the go-live checklist**

Create `docs/superpowers/reports/2026-10-13-early-monitor-go-live.md`:

````markdown
# Early-week monitor: go-live checklist (2026-10-13)

Branch `feat/early-monitor`. Everything below is for you to run; nothing here has been executed. No migration was run, nothing was pushed, no workflow was dispatched, and no code ever connected to the database or the Odds API during the build.

Until step 1 is done every early-monitor job prints a `::warning::` and exits 0, and nothing existing changes. Step 1 is safe to do before or after the merge.

## 1. Run the migration

Paste `db/migration_early_tickets.sql` into the Supabase SQL editor and run it once (idempotent, safe to re-run). It creates `early_tickets` (append-only: trigger plus revoked privileges) and `early_ticket_results`.

Optional append-only check (rolls back; run it yourself in the SQL editor):

```sql
BEGIN;
INSERT INTO early_tickets (sport, game_pk, market, side, book, book_line, price, pinnacle_line, pinnacle_fair,
                           p_win, p_push, p_loss, ev, observed_at, commence_time, hours_to_kick, poll_run_id)
VALUES ('cfb', 1, 'total', 'over', 'draftkings', 54.5, -110, 55.5, 0.5, 0.53, 0, 0.47, 0.016,
        now(), now() + interval '3 days', 72, 'manual');
UPDATE early_tickets SET price = 100;   -- must fail: early_tickets is append-only: UPDATE is not allowed
ROLLBACK;
```

## 2. Check the credit balance before merging

The plan has 30,000 credits a month, reset on the 1st. On 2026-10-13 about 5,700 were left for October. The monitor costs about 3 credits a poll, about 180 a week, and stops itself below 1,500 remaining (the regular prop pulls stop at 2,500, regular game lines at 800, close-capture props at 600, all close capture at 200). With the regular ingest and the close capture also spending, expect the monitor to be paused by its floor, with `::warning::early-monitor: ... skipping its remaining calls`, in the last days of a month when the balance is low. That is the floors working as designed, not a fault; polls resume after the reset. If you would rather start from a full balance, merge on or after Nov 1.

## 3. Merge and push

Push `feat/early-monitor`, open the PR, merge to `main`. Schedules begin on merge:

| Workflow | Schedule (UTC) |
|---|---|
| `early-monitor` | hourly at minute 0, Sunday 16:00 to Wednesday 03:00, months Aug to Feb |
| `grade-early-tickets` | daily 14:20 |
| `report-early-tickets` | Tuesday 15:52 |

Existing workflows are unchanged except that `capture-close` now also treats games that hold an early-week ticket as candidates, so every ticketed game gets a pre-kick line capture (it reads `early_tickets` only once the migration has run).

## 4. Optional dry run

Costs about 3 Odds API credits (one CFB call of 2, one NFL call of 1), writes nothing, and prints the paper tickets it would have logged. It must run from `main`, so only after the merge:

```
gh workflow run early-monitor.yml --ref main -f dry_run=true
```

In the run log look for `[early-monitor] DRY RUN would log: ...` lines and the summary line (`events`, `matched`, `rows`, `tickets` per sport). A `matched` count far below `events` means the ESPN matcher is missing games; those games produce no tickets (conservative).

## 5. Storage

Each poll stores every price it saw in `odds_snapshot`, in the same row shape the regular ingest writes: roughly 3,000 to 4,000 rows per poll (a CFB week of 60 to 100 events times spreads and totals times up to 10 books, plus NFL), so about 200,000 rows a week while the season runs. Check your Supabase database size after the first week; if it is heading toward your plan's limit, decide whether to prune old `odds_snapshot` rows or to limit what the monitor stores (a small change to `scripts/early_monitor.py`).

## 6. First week (Sunday after the merge)

- After each poll: `early_tickets` gains zero or a few rows. Look at them:

```sql
SELECT sport, market, side, book, book_line, price, round(ev::numeric, 4) AS ev, hours_to_kick, observed_at
FROM early_tickets ORDER BY id DESC LIMIT 20;
```

- Games with a ticket get a `close_capture` lines row 5 to 25 minutes before kickoff (`SELECT * FROM close_capture WHERE game_pk IN (SELECT game_pk FROM early_tickets)`).
- The day after the games, `grade-early-tickets` writes `early_ticket_results`:

```sql
SELECT close_kind, count(*) FROM early_ticket_results GROUP BY 1;
```

  A large `missing` share means Pinnacle often has no two-way quote in the close capture; the weekly report states the share.
- The first report appears the Tuesday after (`docs/superpowers/reports/YYYY-MM-DD-early-monitor.md`). It says COLLECTING until there are 150 graded tickets with a Pinnacle close, 60 of them CFB totals. Nothing can pass or fail before that.

## 7. What the result means

The test is judged once, at the first report with at least 150 graded tickets (Pinnacle close) of which at least 60 are CFB totals. Each market is judged on its own: mean close EV at least +1.5%, game-clustered 95% lower bound above 0, beat-close rate at least 55%, and a positive mean close EV where the number was still available a poll later. A market that passes goes to a separate short design for live alerts at small stakes (a later profit check, at least 300 tickets with the ROI 95% lower bound above -3%, decides any stake increase). A market that fails is dropped. If the minimum is not reached by the end of the CFB bowl season, collection continues next season. Realized ROI is reported, never gated.

Tickets on games with no row in `prediction_accuracy` (for example an FBS team against an FCS team, which the CFB model skips) never get a final score from the grader and show up in the report's "not graded yet" count; they do not count toward the test.

## 8. Turning it off

Disable the workflow in GitHub (Actions, early-monitor, the three-dot menu, Disable workflow), or `gh workflow disable early-monitor.yml`. The tables are additive and harmless to leave; `early_tickets` is append-only by design, so only the table owner can drop it.
````

- [ ] **Step 4: Commit**

```bash
git add tests/early/test_floors.py docs/superpowers/reports/2026-10-13-early-monitor-go-live.md
git commit -m "test(early): credit-floor ordering; docs: early-monitor go-live checklist" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Full suite**

Run: `uv run pytest -q` (about 4 minutes)
Expected: `3220 passed` (the 3002 baseline, the 215 tests in `tests/early/`, and 3 new `test_every_workflow_parses[...]` cases). No test may skip because of the new package, and `tests/ledger` must be unchanged and green.

- [ ] **Step 6: Only the intended files changed**

Run: `git diff --stat main...HEAD`
Expected: exactly the files in the File Structure table plus the spec and this plan (new: the migration, `src/sportsmodel/early/*`, `scripts/{early_monitor,grade_early_tickets,report_early_tickets,build_landing_tables}.py`, `assets/early/*.json`, three workflows, `tests/early/*`, the go-live checklist; modified: `src/sportsmodel/db.py`, `src/sportsmodel/ingest/odds.py`, `scripts/capture_close.py`). Nothing under `site/`, `data/`, no model or board module, no existing workflow, no other test file.

- [ ] **Step 7: Safety greps (no stray database writes, no leaked guard)**

```bash
# the only INSERT/UPDATE/DELETE statements added to Python live in db.py's helpers
grep -rnE "INSERT INTO|UPDATE |DELETE FROM" src/sportsmodel/early scripts/early_monitor.py scripts/grade_early_tickets.py scripts/report_early_tickets.py scripts/build_landing_tables.py || echo "no writes outside db.py: OK"
# the test guard must never appear in a workflow
grep -rn "SM_LEDGER_OFF" .github || echo "guard not in workflows: OK"
# no new Python creates, alters or drops anything: schema comes only from the migration file
grep -nE "CREATE TABLE|ALTER TABLE|DROP |TRUNCATE" src/sportsmodel/early/*.py scripts/early_monitor.py scripts/grade_early_tickets.py scripts/report_early_tickets.py scripts/build_landing_tables.py || echo "no DDL in code: OK"
git status --short
```
Expected: the three `OK` lines print and `git status` prints nothing.

- [ ] **Step 8: Hand back**

Do NOT run the migration, merge, push, or dispatch any workflow. Report the branch, the commits, the final test counts and the list of resolved ambiguities (the decisions table above, for the PR description), then follow superpowers:finishing-a-development-branch.

---

## Handoff to the user (after review; the executing agent does none of this)

The full checklist is `docs/superpowers/reports/2026-10-13-early-monitor-go-live.md`. In short:

1. **Run the migration** `db/migration_early_tickets.sql` in the Supabase SQL editor (safe to re-run). Until then every early-monitor job prints a `::warning::` and exits 0, and `capture_close.py` reads `early_tickets` only once it exists.
2. **Mind the credit balance.** About 180 credits a week; the monitor pauses itself below 1,500 remaining. About 5,700 were left for October on 2026-10-13, so late-October polls may be skipped until the 1st.
3. **Merge the PR.** Schedules begin on merge (`early-monitor` hourly Sun 16:00 to Wed 03:00 UTC, `grade-early-tickets` daily 14:20, `report-early-tickets` Tuesday 15:52). Optionally dispatch `early-monitor.yml` with `dry_run=true` once (about 3 credits, writes nothing).
4. **Watch database size** after the first week: about 200,000 `odds_snapshot` rows a week in season.
5. **Read the weekly report.** It says COLLECTING until 150 graded tickets with a Pinnacle close, 60 of them CFB totals; only then is the test judged, once.

## Self-Review

Run against the spec with fresh eyes after writing the plan.

**1. Spec coverage**

| Spec requirement | Task / evidence |
|---|---|
| Flag rule: markets (CFB totals, CFB spreads, NFL spreads; NFL totals and moneylines excluded) | Task 2 `constants.MARKETS`; Task 4 `test_only_the_tested_markets_are_flagged` |
| Reference: Pinnacle two-way no-vig fair, same poll; no pair, no ticket; fair outside [0.15, 0.85], no ticket | Task 4 `test_no_pinnacle_two_way_in_this_poll_means_no_ticket`, `test_pinnacle_fair_outside_15_to_85_percent_means_no_ticket_for_either_side` |
| Landing tables: sources, seasons, half-point grid, +/-1.0 widening to +/-2.0 below 300 games, key-number and push mass | Task 2 (`build_table`, builder script, committed JSON, tests); decisions 2-5 |
| Line adjustment: table at Lp, conditional fair, mass between lines, push conversion, 3.0-point cap | Task 3 (identity, key-number, push conversion both ways, cap, sum-to-one tests); decision 6 |
| EV formula (push contributes 0), 1.5% flag boundary, 24 h rule, best book, deterministic ties | Task 3 `test_ev_formula_with_a_push_contributing_zero`, `test_flag_threshold_boundary_is_inclusive`; Task 4 `test_the_flag_threshold_is_one_and_a_half_percent_ev`, `test_kickoff_must_be_at_least_24_hours_after_the_poll`, `test_the_highest_ev_book_wins_and_ties_go_to_the_first_book_name` |
| One ticket per (sport, game_pk, market, side); stored fields; matchers from `ingest_odds.py` | Task 4 `test_one_ticket_per_side_ever_later_polls_never_add_a_second`, `test_ticket_fields_are_exactly_the_early_tickets_columns`; Task 5 (`ingest_odds._matcher_for`, `build_game_lookup`, unmatched events dropped) |
| `early_tickets` append-only by trigger and revokes | Task 1 migration tests (trigger for UPDATE / DELETE / TRUNCATE, `search_path`, PUBLIC and `service_role` revokes) |
| Polling: hourly crons, 60 a week, `workflow_dispatch` `dry_run` through `env` | Task 6 workflow tests |
| `bookmakers` parameter with the 10 keys, one billed region, CFB 2 / NFL 1 credits | Task 5 `test_the_ten_bookmaker_keys`, `test_cost_per_poll_is_one_credit_per_market_because_ten_books_bill_as_one_region`, `test_fetch_bookmaker_odds_sends_bookmakers_and_never_regions` |
| Credit floor 1,500 (1,499 / 1,500 boundary, unknown first call), floor ordering | Task 5 `test_the_credit_floor_boundary_is_1500`, `test_an_unknown_remaining_value_runs_the_first_call`, `test_at_1500_remaining_the_next_call_still_runs_at_1499_it_does_not`; Task 10 ordering test |
| Storage in `odds_snapshot` in the ingest row shape; tickets written after each sport | Task 5 `test_snapshots_use_the_ingest_row_shape_...`, `test_each_sports_snapshots_then_tickets_are_stored_before_the_next_sport_is_fetched` |
| No retry; missing tables warn and exit 0; `SM_LEDGER_OFF` refusal incl. dry run; writes only three tables | Task 5 (`test_a_failed_poll_is_not_retried_...`, `test_missing_tables_warn_and_stop_before_any_network_call`, the two guard tests); Tasks 7 and 9 equivalents; Task 10 safety greps |
| Close coverage: `capture_close.py` adds `early_tickets` | Task 6 (`load_early_candidates`, seven tests); decision in the Task 6 diff |
| Grading: daily 14:20, upsert one row per ticket, Pinnacle close from the lines snapshot, `missing` excluded and reported | Task 7 (`grade_early_tickets.py`, workflow test); Task 9 reports the missing share |
| Metrics: `close_ev`, `beat_close`, `clv_pts` sign, `result`, `units` (pushes), `available_next_poll` (same / better price, 30 min to 3 h, NULL outside) | Task 7 `test_grading.py` (27 tests) |
| Pass test: judged once at 150 / 60; per-market 40 minimum; four criteria; bootstrap 10,000 / seed 20261013 / 2.5th percentile; ROI never gated | Task 8 (frozen-numbers test, collecting vs judged, each criterion failing alone, the bootstrap tests); Task 9 carry-forward |
| Weekly report Tue 15:52, committed to `docs/superpowers/reports/` | Task 9 (report, script, workflow tests) |
| Migration: idempotent, additive, RLS, revokes | Task 1 `test_migration_is_idempotent_additive_and_writes_no_data`, `test_rls_select_policy_and_write_revokes_for_both_tables` |
| Out of scope (alerts, models / boards / picks / site, NFL totals / moneylines / props, steps 2 and bet sizing) | Nothing in these is touched; Task 10 Step 6 `git diff --stat` and the safety greps |

No gaps found. Spec items deliberately reinterpreted are listed in the decisions table, each with its reason.

**2. Placeholder scan.** No `TBD` / `TODO` / "implement later" / "similar to Task N" in the plan; every code step carries the code. The only value an operator supplies is the Co-Authored-By trailer, which this plan states in full.

**3. Type consistency.** Names checked across tasks: `db.EARLY_COLUMNS["early_tickets"]` (Task 1) is exactly the key set of a ticket from `flags.flag_poll` (Task 4, pinned by a test) and of `db.insert_early_tickets` rows (Task 5); `db.EARLY_COLUMNS["early_ticket_results"]` is exactly the key set of `grading.grade_ticket` (Task 7, pinned) and what `passtest` / `report` read (Tasks 8, 9); `adjust.value_side(market, side, fair, pinnacle_line, book_line, price, pmf, *, cap, strict)` (Task 3) is what `flags.best_offer` (cap, strict default) and `grading.closing_view` (cap=None, strict=False) call; `flags.pinnacle_reference(rows, market)` / `Reference` (Task 4) is what `grading.closing_view` uses; `landing.LandingTable.pmf(center)` is called with the table-space threshold everywhere; `landing.load_tables()` keys `(sport, market)` match `constants.LANDING_TABLES`; `polling.budget_decision` and `odds.last_requests_remaining` / `last_requests_last` feed `early_monitor.Deps.remaining` / `last_cost`; `passtest.decide(results, previous, as_of)` (Task 8) is what `report_early_tickets.build_report` calls, and `report.latest_decision` returns the `previous` it expects; `tests/early/builders.py` (Task 4) is imported by the tests of Tasks 5, 7, 8 and 9 under the same names.

**4. Review Focus.** Each of the five lines names its pinning tests; all exist in the tasks listed.

**Known limits, accepted:** postponed or cancelled games never grade; tickets on games with no `prediction_accuracy` row never get a final score from the grader; quarter-point lines and lines outside the historical range produce no ticket (fewer tickets, never a made-up one); the cron minute :00 is GitHub's busiest, so a poll can be late or skipped and is not retried (the spec's rule); the `available_next_poll` definition follows decision 11, not a literal reading of "monitor poll"; the monitor stores about 200,000 `odds_snapshot` rows a week.
