# Shrink Model Prop Probabilities Toward the Market Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every model probability that decides a prop pick (the NFL and MLB prop boards, the prop legs of the +EV parlays, and through them the pick ledger's `ev_prop` and `parlay_leg` rows) with `market + k × (model − market)`, where k is fitted walk-forward from graded NFL legs and refit every Tuesday, and judge once, against a test frozen before any result exists, whether the shrunk model adds anything beyond the market price (profit roadmap, step 2 Part A).

**Architecture:** A new pure package `src/sportsmodel/shrink/` holds everything that has a rule in it: the formula, the closed-form k with pooling and minimums, the loader of the committed `assets/shrink_k.json` (which answers k = 0 whenever the file or an entry is unusable), the walk-forward pass test and the weekly report text. `serving/props_ev.py::assemble_prop_rows` gains one optional `shrink` argument; `scripts/build_ev_props_board.py` always passes the loaded table, so before the first fit the board is price-only. One IO script (`scripts/fit_shrink_k.py`, read-only against the database) and one weekly workflow write the k file and the gate report. One idempotent migration, which the USER runs, adds two nullable audit columns; the board writes them only when they exist.

**Tech Stack:** Python 3.11+, numpy (the bootstrap and the logistic fit), psycopg 3 (Supabase Postgres, read-only here), GitHub Actions, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-13-shrink-props-design.md` (the authority; read it with this plan). Evidence: `docs/superpowers/reports/2026-10-13-edge-audit.md` section 3 (NFL prop leans claim 64.8%, hit 51.8%, recalibration slope 0.15, SE 0.07, n = 1,587) and section 7 item 2. Reuses the merged pieces `sportsmodel.ledger.record.writes_disabled` (the `SM_LEDGER_OFF` guard), `db._existing_columns` (optional-column probe) and the cluster-bootstrap pattern of `sportsmodel.early.passtest` (mirrored, not imported).

**Baseline before this plan:** `uv run pytest -q` on `feat/shrink-props` at `606b799` = 3244 passed, 2 skipped (about 4 minutes). The new tests add 196 (188 in nine new files, 4 in `tests/test_build_ev_props_board.py`, 1 in `tests/ledger/test_sources.py`, 2 in `tests/serving/test_best_parlays.py`, and 1 new `test_every_workflow_parses` case for the new workflow), so the final full run must show 3440 passed, 2 skipped. Two existing tests change on purpose (see Task 3): the MLB board test now expects a price-only row, and the ledger hook test pins `k = 1` so it keeps testing which line is recorded.

**How this plan was checked:** every file, edit and test below was built task by task from a clean archive of `606b799` in a scratch copy outside the repository (no network, no database), each task's tests failing before its implementation and passing after, the final tree identical to the one the plan describes, and the full suite green.

## Global Constraints

Every task's requirements include this section. Values are copied from the spec.

- **Scope (Part A):** "the live bet signals: the NFL and MLB prop boards and the prop legs in parlays." Out of scope: "Part B (game-level model probabilities, site Alpha/edge for game lines, the ledger's `model_game`)", "Any change to the sims or models", "An MLB fit sample", "Changing the pick thresholds", "Removing any product", "Site code".
- **Formula:** `shrunk = clip(m + k × (p − m), 0.01, 0.99)`. p is the model's probability of the over (after MLB's existing calibration); m is the market's no-vig probability of the over at the same line, "computed exactly as the board computes `market_prob` today: best soft over and best soft under, vig removed"; "The fit uses the identical m construction"; "The under side uses `1 − shrunk`."
- **k for one group of legs:** `k = clamp( Σ(y − m)(p − m) / Σ(p − m)², 0, 1 )`; y is 1 when the over hit; pushes are excluded; "If `Σ(p − m)² = 0`, k = 0."
- **Fit sample:** graded NFL leans from `nfl_prop_grades`, p = `p_over`, m = the no-vig of its `over_price` and `under_price`; "Rows missing either price, or with an invalid American price (|price| < 100), are excluded."
- **Pooling and minimums:** a pooled NFL k on all eligible legs; each market partially pooled, `k_market = (n_m × k_m_raw + 300 × k_pool) / (n_m + 300)`; "If the pooled sample has fewer than **300** legs, the NFL k is 0 for every market."
- **Model version:** the fit uses legs from the current props model version (props-ML v2), "whose version string is read from the data. If that gives fewer than 300 legs, it uses all versions pooled."
- **MLB:** "k = 0 for every MLB market in Part A".
- **Walk-forward:** "The k used during a week is fitted only on legs graded before that week."
- **Committed parameters:** `assets/shrink_k.json` records, per `(sport, market)`, `k`, `n` and `k_raw`; the pooled values; `model_version_used` and `fitted_at`; the data window. Market keys are the board's odds market keys.
- **Serving:** the shrink replaces p "right after p is computed at the book line (and after `calibrate_fn` for MLB)"; everything downstream (EV at the best price, the pick decision, the stored `model_prob` and `edge`) uses the shrunk value. Fail-safe: "If the file is missing or unreadable, or the `(sport, market)` entry is missing, k = 0. The script warns once per run. It never falls back to the raw model." Pick rule unchanged: `0 < ev_best ≤ 0.25`. Parlays: `MIN_LEG_PROB` stays 0.55; prop legs read `model_prob` with no code change. Pick ledger: no change.
- **Audit columns:** `ev_prop_picks` gains nullable `raw_model_prob` (p before shrinking) and `shrink_k` (k applied); "The writer includes them only when the columns exist, so the board works before the migration."
- **Migration:** `db/migration_shrink_props.sql`: `ALTER TABLE ev_prop_picks ADD COLUMN IF NOT EXISTS raw_model_prob double precision, ADD COLUMN IF NOT EXISTS shrink_k double precision;` Idempotent and additive; views that select explicit columns are unaffected.
- **Refit job:** `scripts/fit_shrink_k.py` and `fit-shrink-k.yml`, "Tuesdays at 15:30 UTC, plus `workflow_dispatch`". It reads `nfl_prop_grades` and each leg's model version (read-only), writes `assets/shrink_k.json`, commits and pushes "with the 3-attempt `pull --rebase` retry". "on any fetch or fit failure it exits non-zero and leaves the existing file untouched. It never writes a k it did not compute." Job `timeout-minutes` is set; secrets are passed through `env`, "with no `${{ }}` inside `run:`"; `permissions: contents: write`. First run: dispatched manually right after merge.
- **Pass test (frozen):** each graded NFL lean in week W, keyed by `(season, week)`, is scored with the shrunk probability using the k fitted on legs graded in weeks before W only; "Weeks whose earlier-leg count is below 300 are not scored"; pushes excluded. Judged **once**, at the first weekly report with ≥ **2,000** scored legs; before that the status is "collecting". Pass needs both: (1) `Brier(m) − Brier(shrunk) ≥ 0.001` and the paired bootstrap 95% CI of that difference has its lower bound > 0 (resamples games with replacement: 10,000 resamples, seed 20261013, 2.5th percentile); (2) the logistic regression `y ~ a + b · logit(shrunk)` has slope b in [**0.8, 1.2**]. The weekly report is committed to `docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md` and shows current k per market with n, scored legs, the running Brier difference and slope, and the status; once judged the verdict is carried forward unchanged. A pass "does not prove profit; CLV in the pick ledger judges that"; on a fail "the user and Claude decide together. Until then, picks keep using the shrunk probabilities."
- **Tests:** "All tests are network-free and DB-free. The existing suite stays green."
- **No database or network access from the executing agent.** The migration is a file the USER runs in the Supabase SQL editor. Only the merged, scheduled workflow reads the database. The local `.env` holds a real `DATABASE_URL`: never source it, and never run `scripts/fit_shrink_k.py` locally (it refuses under `SM_LEDGER_OFF`, which the root `conftest.py` sets for every test; never unset it in a shell that has `DATABASE_URL`). Every test uses injected fakes. The only commands this plan runs are `pytest`, `git`, `grep` and `ls`.
- **Conventions:** follow the repo's patterns (importlib-loaded script tests, FakeConn-style database tests, the block-YAML workflow reader in `tests/test_workflows_props_ml.py`). Project convention: implementers and reviewers run on Sonnet (pass `model: "sonnet"` explicitly).
- **Commits:** every commit message ends with the Co-Authored-By line the executing agent's harness provides. In the session that wrote this plan that line is `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`; every commit command below passes it as a second `-m`. If your harness provides a different line, use yours in its place.

## Review Focus

The input classes and failure modes the spec implies but does not spell out, most likely to bite first. Each has a pinning test in the task that owns the code.

1. **A missing, corrupt or hand-edited `assets/shrink_k.json`** (not there yet, empty, a list, `NaN`, k above 1 or below 0, a string, an MLB entry set to 0.9). A reasonable person expects the board to keep running price-only, never the raw model. Task 2: `test_a_missing_file_gives_k_zero_everywhere_and_a_warning`, `test_an_unusable_file_gives_k_zero_and_never_raises`, `test_an_entry_whose_k_is_not_a_number_in_zero_one_is_ignored_not_trusted`, `test_mlb_is_always_zero_even_if_the_file_says_otherwise`, `test_a_market_missing_from_the_table_is_price_only_never_the_raw_model`; Task 3: `test_with_no_shrink_file_the_board_runs_price_only_and_warns_once`.
2. **Bad graded rows.** Prices of 0, `None`, `+50`, `NaN` or `-99`; pushes; a missing `p_over`, lean, season or week; markets the board never serves (`pass_yds`, `rush_att`); a leg whose model version is unknown; no variance at all. Behavior: excluded or k = 0, never a crash and never a made-up leg. Task 1: `test_rows_with_a_missing_or_invalid_american_price_are_excluded`, `test_pushes_are_excluded`, `test_rows_with_no_probability_lean_result_or_week_are_excluded`, `test_markets_the_board_does_not_serve_are_dropped`, `test_a_missing_model_version_is_kept_as_none`, `test_zero_variance_and_no_legs_give_zero`.
3. **A market with no graded history, and what a k = 0 board picks.** `pass_tds` has no row in `nfl_prop_lines`, so it can never be fitted; MLB has no fit sample. Behavior: an explicit k = 0 entry (price-only), never borrowing the pooled k from nothing. At k = 0 the shrunk probability is the market's no-vig probability, so a prop is a pick only when the two best prices across books imply under 100% (a real cross-book gap), and then both sides tie. Task 1: `test_a_market_with_no_legs_gets_k_zero_not_the_pooled_k`, `test_a_market_with_zero_variance_has_raw_k_zero_and_is_pulled_toward_the_pool`; Task 2: `test_at_k_zero_only_cross_book_pricing_gaps_are_picks_and_both_sides_tie`, `test_mlb_is_price_only_in_part_a_even_if_a_table_lists_mlb_markets`; Task 3: `test_mlb_main_builds_mlb_rows_without_touching_nfl_prop_lines`.
4. **Look-ahead and the frozen thresholds.** A week must never train on itself or later weeks (including across a season boundary, and with legs arriving in any order); the 300 earlier-leg line and the 2,000 judgement line are exact (299 vs 300, 1,999 vs 2,000); a verdict, once judged, never moves even when later data would flip it. Task 4: `test_a_week_is_scored_with_the_k_fitted_on_earlier_weeks_only`, `test_later_weeks_never_change_an_earlier_weeks_score`, `test_the_300_earlier_leg_threshold_is_exact`, `test_weeks_are_ordered_by_season_then_week_and_legs_may_arrive_out_of_order`, `test_1999_scored_legs_are_still_collecting_and_2000_are_judged`, `test_a_judged_verdict_is_carried_forward_unchanged_whatever_the_new_data_say`.
5. **Half-failed, repeated and concurrent runs.** The database drops mid-fit; a read comes back truncated (fewer graded legs than the committed file was fitted on); the job is dispatched twice in a day; two workflows push at once; a board build reads the file while it is being replaced; the migration has not been run yet. Behavior: the existing file stays, the board keeps running, a push race is retried. Task 5: `test_a_fetch_failure_exits_nonzero_and_leaves_the_existing_file_and_reports_untouched`, `test_a_failure_while_serving_lookup_also_keeps_the_file`, `test_a_read_with_fewer_legs_than_the_committed_file_is_refused`, `test_write_atomic_replaces_without_leaving_a_temp_file`, `test_the_push_retries_three_times_with_pull_rebase`; Task 3: `test_before_the_migration_the_board_is_written_without_them_and_warns`.

Also pinned: tests and local runs cannot reach the production database through `.env` (`test_main_refuses_under_the_test_ledger_guard`, `test_the_script_never_reaches_a_database_in_tests`, Task 5), and the fit queries are read-only (`test_the_queries_are_read_only_and_ask_for_the_legs_version_and_the_served_version`, Task 5).

## Decisions on spec ambiguities (resolved here; the plan follows these)

| # | Spec text | Resolution | Why |
|---|---|---|---|
| 1 | "Market keys are the board's odds market keys" while `ev_prop_picks.market` stores the sim market (`rec_yds`) | k is keyed `(sport, odds_market)`: `reception_yds`, `rush_yds`, `receptions`, `pass_tds` for NFL. The board looks k up by its `odds_market` variable; the row still stores the sim market | The spec's rule and the board's existing split between the two names; the fit converts through `SIM_TO_ODDS_MARKET` |
| 2 | "A pooled NFL k is fitted on all eligible legs" while `nfl_prop_grades` also holds `pass_yds` and `rush_att`, which the board never serves | Eligible means the sim market maps through `SIM_TO_ODDS_MARKET`. `pass_yds` and `rush_att` legs are excluded from the fit, the pool and the pass test | The spec says the board's mapping "is applied when fitting"; a k fitted partly on markets nobody bets would not describe the bets. Consequence: fewer legs, so the 2,000-leg judgement point comes later (flagged in the go-live checklist) |
| 3 | The market formula at `n_m = 0` | A board market with no graded legs (`pass_tds`, absent from `nfl_prop_lines`) gets an explicit entry `k = 0, n = 0, k_raw = null`, not the pooled k | The blend formula would hand a market with zero evidence the pooled k; the fail-safe principle (no evidence, no model) is stricter, and the explicit entry avoids a permanent "missing entry" warning |
| 4 | "the model version of each leg" and "the current props model version, props-ML v2, whose version string is read from the data" | `nfl_prop_lines` stores no version. A leg's version is the `nfl_player_sim` row for the same (game, player, market) whose `mean` is nearest the line row's `projection` (ties to the newest). The current version is `nfl_sim_serving.model_version` (not `ev_prop_picks.model_version`, which is the constant `props-sim-v1`). A leg whose sim row is gone has version NULL and counts only in the all-versions pool | The only data that carries versions; a failed lookup of the serving row fails the job rather than guessing |
| 5 | "The fit uses the identical m construction" | The fit's m is `board.novig(over_price, under_price)` on the best prices stored in `nfl_prop_lines`, which come from ALL books (Pinnacle included); the board's m uses the best NON-Pinnacle prices. The formula and function are identical; the price pool differs and cannot be changed (the stored columns are all there is) | Stated as a known limit in the go-live checklist; k stays internally consistent |
| 6 | y for an "under" lean | y = 1 when the OVER hit: `(lean == over) == (result == hit)`; a push is never a leg | The spec defines y as over-hit; grades are stored relative to the lean |
| 7 | The version rule inside the walk-forward pass test | The current-version string is the one fixed at fit time; for each scored week the rule is applied to that week's earlier legs (current version if it has at least 300 of them, else all versions) | One rule everywhere; the test and the live k use the same function (`fit.fit_table`) |
| 8 | "Weeks whose earlier-leg count is below 300 are not scored" | Earlier legs = every eligible leg (all versions) in strictly earlier `(season, week)`; the count is of legs, not weeks; 299 is not scored, 300 is. `(2025, 18) < (2026, 1)` | Exact, testable, and independent of input order |
| 9 | k for a group whose `Σ(p − m)² = 0` | The market's `k_raw` is 0, and its blended k still pools toward the pooled k as the formula says; only a group with no legs at all is forced to k = 0 (decision 3) | Literal reading of "If `Σ(p − m)² = 0`, k = 0" for the raw k |
| 10 | "k = 0 for every MLB market" and "writes `assets/shrink_k.json`" per `(sport, market)` | The file lists the four MLB markets with `k = 0, n = 0` for audit; the loader ignores the file's MLB values and always answers 0 for MLB, and an MLB lookup is never reported as a missing entry | A hand edit of the file cannot switch MLB on in Part A |
| 11 | Where in `assemble_prop_rows` the shrink goes (it needs m, which needs the best soft prices) | After the best soft over and under are found (so m exists) and after `calibrate_fn`; before EV and the side decision. The argument is `shrink=None` by default, meaning "do not shrink" so the existing join-logic tests stay valid; the only production caller always passes a table | Keeps the pure join function backward compatible without a silent raw fallback in production (a test pins that `main` passes the table for both sports) |
| 12 | `raw_model_prob` ("p before shrinking") for an under pick | The pick side's raw probability (`1 − p` for under), like `model_prob`; `shrink_k` is recorded even when 0. Rows always carry both keys (`None` when no table was given); the writer probes `information_schema` once per upsert, only when a record carries them | Mirrors `model_prob`; the same probe pattern as `upsert_prediction_accuracy` |
| 13 | "The writer includes them only when the columns exist" | Before the migration the board is written without them and a one-line `WARN` is printed per upsert call (the existing optional-column behaviour) | Existing pattern; the board never fails on a missing column |
| 14 | "judged once, at the first weekly report with ≥ 2,000 scored legs" | Every run of `fit_shrink_k.py` (scheduled or dispatched) writes a dated report; the first run that reaches 2,000 judges, and every later report copies that decision from the earliest judged report. Two runs on one day overwrite the same file | Same carry-forward design as the early-monitor report |
| 15 | Slope of "the logistic regression `y ~ a + b · logit(shrunk)`" | Unpenalised maximum likelihood by Newton-Raphson (cross-checked against an independent optimiser in the tests); an undefined slope (no variance, separation, no convergence) cannot pass | No new dependency; fail-safe |
| 16 | "paired bootstrap ... resamples games" | The per-leg gain is `(m − y)² − (shrunk − y)²`; games (`game_pk`) are drawn with replacement; each resample's statistic is the leg-weighted mean gain; `rng = numpy.random.default_rng(20261013)` drawing in blocks of 1,000 resamples, as `early/passtest.py` does | Frozen, reproducible, and consistent with the previous pass test |
| 17 | "leaves the existing file untouched" | Also refused: a read with fewer eligible legs than the committed file was fitted on (graded legs only accumulate, so fewer means truncated). The file is written atomically (temp file, then replace) | Cheap protection against writing a k that is "computed" but wrong |
| 18 | The fit window | All graded legs ever (no rolling window): the spec says "all eligible legs" | Literal |
| 19 | MLB board test and "far fewer picks" | The existing MLB board test now expects `is_pick` false with `shrink_k = 0.0` (the spec's k = 0 for MLB); the ledger hook test pins k = 1 so it still tests which line is recorded | Behaviour change mandated by the spec, made explicit in the tests |

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/sportsmodel/shrink/__init__.py` | Create | Package marker |
| `src/sportsmodel/shrink/formula.py` | Create | Clip bounds, `shrink_prob`, `valid_american`, `market_prob_over` |
| `src/sportsmodel/shrink/fit.py` | Create | Grade rows to legs, closed-form k, version selection, pooling, the committed document, the truncated-read guard |
| `src/sportsmodel/shrink/store.py` | Create | `ShrinkTable`, `load_table`, the k = 0 fail-safe, MLB = 0 |
| `src/sportsmodel/shrink/passtest.py` | Create | The frozen pass test: walk-forward scoring, bootstrap, slope, verdict, `decide` |
| `src/sportsmodel/shrink/report.py` | Create | Weekly report text, decision block, carry-forward |
| `src/sportsmodel/serving/props_ev.py` | Modify | `assemble_prop_rows(..., shrink=None)`, `raw_model_prob` and `shrink_k` on rows |
| `src/sportsmodel/db.py` | Modify | `upsert_ev_prop_picks` writes the two audit columns only when they exist |
| `db/migration_shrink_props.sql` | Create | The two nullable columns (USER runs it) |
| `scripts/build_ev_props_board.py` | Modify | Loads the table, passes it for both sports, warns once |
| `scripts/fit_shrink_k.py` | Create | The weekly job: read, fit, write the k file and the report |
| `.github/workflows/fit-shrink-k.yml` | Create | Tuesday 15:30 UTC fit, commit and push with retry |
| `docs/superpowers/reports/2026-10-13-shrink-props-go-live.md` | Create | The user's go-live checklist |
| `tests/shrink/` (`__init__.py`, `test_formula.py`, `test_fit.py`, `test_store.py`, `test_passtest.py`, `test_report.py`), `tests/serving/test_props_ev_shrink.py`, `tests/test_db_ev_prop_shrink_cols.py`, `tests/scripts/test_fit_shrink_k.py`, `tests/test_workflows_shrink.py` | Create | One test file per module, script or workflow |
| `tests/test_build_ev_props_board.py`, `tests/ledger/test_hooks.py`, `tests/ledger/test_sources.py`, `tests/serving/test_best_parlays.py` | Modify | Updated and added tests (Task 3) |

Not touched: the sims and props-ML models, `nfl_prop_lines` / `nfl_prop_grades` and their pages, `serving/best_parlays.py`, `ledger/sources.py` and every ledger module, the game-line boards, the early monitor, every other workflow, the site. `assets/shrink_k.json` is NOT created by this plan: only the scheduled workflow writes it.

---
### Task 1: Formula, grade-row legs and the fit (pure)

**Files:**
- Create: `src/sportsmodel/shrink/__init__.py` (empty), `src/sportsmodel/shrink/formula.py`, `src/sportsmodel/shrink/fit.py`
- Create: `tests/shrink/__init__.py` (empty), `tests/shrink/test_formula.py`, `tests/shrink/test_fit.py`

**Interfaces:**
- Consumes (existing): `serving.board.novig(price_side: int, price_other: int) -> float`; `serving.props_ev.SIM_TO_ODDS_MARKET: dict[str, str]` (`rush_yds`, `rec_yds` -> `reception_yds`, `receptions`, `pass_tds`) and `MLB_SIM_TO_ODDS_MARKET`.
- Produces (later tasks rely on these exact names):
  - `formula.CLIP_LO = 0.01`, `formula.CLIP_HI = 0.99`
  - `formula.valid_american(price) -> bool` (finite, `|price| >= 100`)
  - `formula.market_prob_over(over_price, under_price) -> float | None`
  - `formula.shrink_prob(p: float, m: float, k: float) -> float`
  - `fit.MIN_POOL_LEGS = 300`, `fit.PRIOR_LEGS = 300`, `fit.ALL_VERSIONS = "all"`, `fit.FIT_SPORT = "nfl"`
  - `fit.legs_from_grades(rows: list[dict], market_map: dict[str, str] | None = None) -> list[dict]` (leg: `{season, week, game_pk, market, p, m, y, version}`; `market` is the ODDS key)
  - `fit.closed_form_k(legs: list[dict]) -> float`
  - `fit.select_sample(legs, current_version: str | None) -> tuple[list[dict], str]`
  - `fit.fit_table(legs, current_version) -> {"model_version_used", "pooled": {"k", "n", "k_raw"}, "markets": {"nfl": {market: {"k", "n", "k_raw"}}, "mlb": {...}}}`
  - `fit.build_document(legs, current_version, fitted_at: str) -> dict` (the contents of `assets/shrink_k.json`)
  - `fit.has_fewer_legs(existing: dict | None, new: dict) -> bool`

- [ ] **Step 1: Write the failing tests**

Create `tests/shrink/__init__.py` as an empty file, then `tests/shrink/test_formula.py`:

````python
"""The shrink formula, the clip and the market probability: exact values from the spec. No IO."""
import math

import pytest

from sportsmodel.serving.board import novig
from sportsmodel.shrink import formula


def test_the_clip_bounds_are_the_specs():
    assert (formula.CLIP_LO, formula.CLIP_HI) == (0.01, 0.99)


def test_k_zero_is_the_market_and_k_one_is_the_model():
    assert formula.shrink_prob(0.70, 0.50, 0.0) == pytest.approx(0.50)
    assert formula.shrink_prob(0.70, 0.50, 1.0) == pytest.approx(0.70)
    assert formula.shrink_prob(0.70, 0.50, 0.25) == pytest.approx(0.55)          # m + k (p - m)
    assert formula.shrink_prob(0.30, 0.50, 0.25) == pytest.approx(0.45)          # works below the market too


def test_the_result_is_clipped_to_one_and_ninety_nine_percent():
    assert formula.shrink_prob(1.0, 0.5, 1.0) == 0.99
    assert formula.shrink_prob(0.0, 0.5, 1.0) == 0.01
    assert formula.shrink_prob(0.5, 0.995, 0.0) == 0.99          # a market the book prices at 99.5% is clipped too
    assert formula.shrink_prob(0.5, 0.004, 0.0) == 0.01


@pytest.mark.parametrize("price", [100, -100, 150, -110, 100.0, -250, 10000, "-110"])
def test_valid_american_prices(price):
    assert formula.valid_american(price) is True


@pytest.mark.parametrize("price", [0, 99, -99, 50, -50, None, "x", "", float("nan"), float("inf"), -float("inf"), True])
def test_invalid_american_prices(price):
    assert formula.valid_american(price) is False


def test_market_prob_over_is_the_boards_novig_of_best_over_and_best_under():
    assert formula.market_prob_over(-110, -110) == pytest.approx(0.5)
    assert formula.market_prob_over(150, -170) == pytest.approx(novig(150, -170))
    assert formula.market_prob_over(-130, 110) == pytest.approx(novig(-130, 110))
    assert 0 < formula.market_prob_over(-130, 110) < 1


def test_market_prob_over_is_none_when_either_price_is_unusable():
    assert formula.market_prob_over(None, -110) is None
    assert formula.market_prob_over(-110, 0) is None
    assert formula.market_prob_over(-99, -110) is None
    assert formula.market_prob_over(float("nan"), -110) is None
    assert not math.isnan(formula.market_prob_over(100, 100))
````

and `tests/shrink/test_fit.py`:

````python
"""Closed-form k, grade-row to leg conversion, version fallback, pooling and the 300-leg rules. No IO."""
import random

import pytest

from sportsmodel.serving.props_ev import MLB_SIM_TO_ODDS_MARKET, SIM_TO_ODDS_MARKET
from sportsmodel.shrink import fit
from sportsmodel.shrink.formula import market_prob_over


def leg(p=0.6, m=0.5, y=1, market="rush_yds", version="nfl-sim-ml-v2", season=2026, week=1, game_pk=1):
    return {"season": season, "week": week, "game_pk": game_pk, "market": market, "p": p, "m": m, "y": y,
            "version": version}


def legs_with_k(k, n, **kw):
    """n legs whose closed-form k is exactly k: p - m = 0.1 everywhere and y = m + k (p - m) (an expected outcome)."""
    return [leg(p=0.6, m=0.5, y=0.5 + k * 0.1, **kw) for _ in range(n)]


def grade(**kw):
    base = {"game_pk": 1, "player_id": "00-1", "market": "rush_yds", "p_over": 0.6, "over_price": -110,
            "under_price": -110, "lean": "over", "result": "hit", "season": 2026, "week": 3,
            "model_version": "nfl-sim-ml-v2"}
    return base | kw


# ---- the constants ---------------------------------------------------------------------------------------------

def test_the_300_leg_constants_are_the_specs():
    assert fit.MIN_POOL_LEGS == 300 and fit.PRIOR_LEGS == 300


# ---- closed-form k ---------------------------------------------------------------------------------------------

def test_closed_form_recovers_a_known_k_exactly_on_expected_outcomes():
    assert fit.closed_form_k(legs_with_k(0.3, 50)) == pytest.approx(0.3)
    assert fit.closed_form_k(legs_with_k(0.0, 50)) == pytest.approx(0.0, abs=1e-12)
    assert fit.closed_form_k(legs_with_k(1.0, 50)) == pytest.approx(1.0)


def test_closed_form_recovers_a_known_k_on_simulated_bernoulli_outcomes():
    rng = random.Random(7)
    legs = []
    for _ in range(40_000):
        m = rng.uniform(0.35, 0.65)
        p = min(max(m + rng.uniform(-0.15, 0.15), 0.02), 0.98)
        q = m + 0.30 * (p - m)                       # the market is right 70% of the way to the truth: true k = 0.30
        legs.append(leg(p=p, m=m, y=1 if rng.random() < q else 0))
    assert fit.closed_form_k(legs) == pytest.approx(0.30, abs=0.05)


def test_closed_form_clamps_to_zero_and_one():
    assert fit.closed_form_k([leg(p=0.7, m=0.5, y=0), leg(p=0.3, m=0.5, y=1)]) == 0.0      # the model points the wrong way
    assert fit.closed_form_k(legs_with_k(2.5, 10)) == 1.0                                   # the model is under-confident


def test_zero_variance_and_no_legs_give_zero():
    assert fit.closed_form_k([leg(p=0.5, m=0.5, y=1) for _ in range(10)]) == 0.0
    assert fit.closed_form_k([]) == 0.0


# ---- grade rows to legs ----------------------------------------------------------------------------------------

def test_a_row_becomes_a_leg_with_the_boards_market_key_and_the_over_outcome():
    [l] = fit.legs_from_grades([grade(market="rec_yds", over_price=-120, under_price=100)])
    assert l["market"] == "reception_yds" and l["season"] == 2026 and l["week"] == 3 and l["version"] == "nfl-sim-ml-v2"
    assert l["p"] == 0.6 and l["m"] == pytest.approx(market_prob_over(-120, 100)) and l["y"] == 1


@pytest.mark.parametrize("lean,result,y", [("over", "hit", 1), ("over", "miss", 0), ("under", "hit", 0), ("under", "miss", 1)])
def test_y_is_one_when_the_over_hit_whatever_the_lean(lean, result, y):
    [l] = fit.legs_from_grades([grade(lean=lean, result=result)])
    assert l["y"] == y


def test_pushes_are_excluded():
    assert fit.legs_from_grades([grade(result="push")]) == []


@pytest.mark.parametrize("over,under", [(None, -110), (-110, None), (0, -110), (-110, 0), (-99, -110), (99, -110),
                                        (50, -110), (float("nan"), -110), ("n/a", -110)])
def test_rows_with_a_missing_or_invalid_american_price_are_excluded(over, under):
    assert fit.legs_from_grades([grade(over_price=over, under_price=under)]) == []


@pytest.mark.parametrize("bad", [dict(p_over=None), dict(p_over=float("nan")), dict(lean=None), dict(lean="push"),
                                 dict(result=None), dict(season=None), dict(week=None)])
def test_rows_with_no_probability_lean_result_or_week_are_excluded(bad):
    assert fit.legs_from_grades([grade(**bad)]) == []


def test_markets_the_board_does_not_serve_are_dropped():
    rows = [grade(market="pass_yds"), grade(market="rush_att"), grade(market="made_up"), grade(market="receptions")]
    assert [l["market"] for l in fit.legs_from_grades(rows)] == ["receptions"]


def test_a_missing_model_version_is_kept_as_none():
    [l] = fit.legs_from_grades([grade(model_version=None)])
    assert l["version"] is None


# ---- version selection -----------------------------------------------------------------------------------------

def test_the_current_version_is_used_when_it_has_300_legs():
    legs = legs_with_k(0.2, 300, version="v2") + legs_with_k(0.9, 500, version="v1")
    sample, used = fit.select_sample(legs, "v2")
    assert used == "v2" and len(sample) == 300 and {l["version"] for l in sample} == {"v2"}


def test_299_legs_of_the_current_version_fall_back_to_every_version():
    legs = legs_with_k(0.2, 299, version="v2") + legs_with_k(0.9, 500, version="v1") + legs_with_k(0.5, 5, version=None)
    sample, used = fit.select_sample(legs, "v2")
    assert used == fit.ALL_VERSIONS == "all" and len(sample) == 804


def test_no_current_version_pools_everything():
    assert fit.select_sample(legs_with_k(0.2, 400), None)[1] == "all"


# ---- the table --------------------------------------------------------------------------------------------------

def test_fewer_than_300_pooled_legs_means_k_zero_for_every_market():
    t = fit.fit_table(legs_with_k(0.4, 299, market="rush_yds"), None)
    assert t["pooled"]["n"] == 299 and t["pooled"]["k"] == 0.0 and t["pooled"]["k_raw"] == pytest.approx(0.4)
    assert all(e["k"] == 0.0 for e in t["markets"]["nfl"].values())
    assert t["markets"]["nfl"]["rush_yds"]["n"] == 299 and t["markets"]["nfl"]["rush_yds"]["k_raw"] == pytest.approx(0.4)


def test_exactly_300_pooled_legs_fits():
    t = fit.fit_table(legs_with_k(0.4, 300, market="rush_yds"), None)
    assert t["pooled"]["k"] == pytest.approx(0.4)
    assert t["markets"]["nfl"]["rush_yds"]["k"] == pytest.approx(0.4)             # one market: k_raw == k_pool, so the blend is 0.4


def test_each_market_is_partially_pooled_toward_the_pooled_k():
    legs = legs_with_k(0.10, 100, market="rush_yds") + legs_with_k(0.50, 300, market="receptions") \
        + legs_with_k(0.30, 700, market="reception_yds")
    t = fit.fit_table(legs, None)
    k_pool = fit.closed_form_k(legs)
    assert t["pooled"]["n"] == 1100 and t["pooled"]["k"] == pytest.approx(k_pool)
    rush = t["markets"]["nfl"]["rush_yds"]
    assert rush["n"] == 100 and rush["k_raw"] == pytest.approx(0.10)
    assert rush["k"] == pytest.approx((100 * 0.10 + 300 * k_pool) / (100 + 300))
    rec = t["markets"]["nfl"]["receptions"]
    assert rec["k"] == pytest.approx((300 * 0.50 + 300 * k_pool) / (300 + 300))


def test_a_market_with_no_legs_gets_k_zero_not_the_pooled_k():
    t = fit.fit_table(legs_with_k(0.4, 400, market="rush_yds"), None)
    assert t["markets"]["nfl"]["pass_tds"] == {"k": 0.0, "n": 0, "k_raw": None}
    assert set(t["markets"]["nfl"]) == set(SIM_TO_ODDS_MARKET.values())


def test_a_market_with_zero_variance_has_raw_k_zero_and_is_pulled_toward_the_pool():
    legs = legs_with_k(0.4, 400, market="rush_yds") + [leg(p=0.5, m=0.5, y=1, market="receptions") for _ in range(50)]
    t = fit.fit_table(legs, None)
    rec = t["markets"]["nfl"]["receptions"]
    assert rec["k_raw"] == 0.0 and rec["k"] == pytest.approx((50 * 0.0 + 300 * t["pooled"]["k"]) / 350)


def test_mlb_is_zero_for_every_market_whatever_the_data():
    t = fit.fit_table(legs_with_k(0.9, 2000), None)
    assert set(t["markets"]["mlb"]) == set(MLB_SIM_TO_ODDS_MARKET.values())
    assert all(e["k"] == 0.0 for e in t["markets"]["mlb"].values())


def test_the_table_reports_the_version_it_used_and_falls_back_with_the_sample():
    legs = legs_with_k(0.2, 320, version="v2") + legs_with_k(0.8, 400, version="v1")
    cur = fit.fit_table(legs, "v2")
    assert cur["model_version_used"] == "v2" and cur["pooled"]["n"] == 320 and cur["pooled"]["k"] == pytest.approx(0.2)
    short = fit.fit_table(legs_with_k(0.2, 100, version="v2") + legs_with_k(0.8, 400, version="v1"), "v2")
    assert short["model_version_used"] == "all" and short["pooled"]["n"] == 500


def test_the_clamped_k_values_stay_in_zero_one():
    t = fit.fit_table(legs_with_k(5.0, 400, market="rush_yds") + legs_with_k(-3.0, 400, market="receptions"), None)
    for e in t["markets"]["nfl"].values():
        assert 0.0 <= e["k"] <= 1.0


# ---- the committed document ------------------------------------------------------------------------------------

def test_the_document_records_k_n_k_raw_pooled_version_fitted_at_and_the_data_window():
    legs = legs_with_k(0.3, 200, week=2) + legs_with_k(0.3, 150, week=5, market="receptions") + legs_with_k(0.3, 10, week=1, season=2025)
    doc = fit.build_document(legs, "nfl-sim-ml-v2", "2026-10-20T15:31:00+00:00")
    assert doc["schema"] == 1 and doc["fitted_at"] == "2026-10-20T15:31:00+00:00" and doc["current_version"] == "nfl-sim-ml-v2"
    assert doc["model_version_used"] == "nfl-sim-ml-v2"
    assert doc["data_window"] == {"first": [2025, 1], "last": [2026, 5], "n_legs": 360}
    assert doc["pooled"]["sport"] == "nfl" and doc["pooled"]["n"] == 360 and doc["pooled"]["k"] == pytest.approx(0.3)
    assert doc["markets"]["nfl"]["rush_yds"]["n"] == 210 and set(doc["markets"]) == {"nfl", "mlb"}
    import json
    assert json.loads(json.dumps(doc)) == doc                                   # plain JSON types only


def test_an_empty_fit_is_a_valid_all_zero_document():
    doc = fit.build_document([], None, "2026-10-20T15:31:00+00:00")
    assert doc["data_window"] == {"first": None, "last": None, "n_legs": 0}
    assert doc["pooled"]["k"] == 0.0 and all(e["k"] == 0.0 for s in doc["markets"].values() for e in s.values())


def test_a_fit_on_fewer_eligible_legs_than_the_committed_file_is_flagged():
    old = fit.build_document(legs_with_k(0.3, 500), None, "t")
    assert fit.has_fewer_legs(old, fit.build_document(legs_with_k(0.3, 499), None, "t")) is True
    assert fit.has_fewer_legs(old, fit.build_document(legs_with_k(0.3, 500), None, "t")) is False
    assert fit.has_fewer_legs(old, fit.build_document(legs_with_k(0.3, 900), None, "t")) is False
    for odd in (None, {}, {"data_window": None}, {"data_window": {"n_legs": "x"}}, []):
        assert fit.has_fewer_legs(odd, old) is False                               # no usable previous count: nothing to protect
````

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/shrink`
Expected: `2 errors` during collection, `ModuleNotFoundError: No module named 'sportsmodel.shrink'`.

- [ ] **Step 3: Create the package and the formula**

Create `src/sportsmodel/shrink/__init__.py` as an empty file, then `src/sportsmodel/shrink/formula.py`:

````python
"""The shrink formula and the market probability it shrinks toward. PURE -- no IO.

From docs/superpowers/specs/2026-10-13-shrink-props-design.md section 1:

    shrunk = clip(m + k * (p - m), 0.01, 0.99)

p is the model's probability of the OVER (after MLB's calibration), m the market's no-vig probability of the over at
the same line, built exactly as the board builds `market_prob`: `board.novig(best over price, best under price)`.
The under side uses 1 - shrunk.
"""
from __future__ import annotations

import math

from ..serving.board import novig

CLIP_LO = 0.01
CLIP_HI = 0.99


def valid_american(price) -> bool:
    """True for a usable American price: a finite number with |price| >= 100 (so 0, +50 and NaN are not)."""
    if isinstance(price, bool):
        return False
    try:
        value = float(price)
    except (TypeError, ValueError):
        return False
    return math.isfinite(value) and abs(value) >= 100.0


def market_prob_over(over_price, under_price) -> float | None:
    """No-vig probability of the over from the best over and best under American prices; None if either is invalid."""
    if not (valid_american(over_price) and valid_american(under_price)):
        return None
    return novig(over_price, under_price)


def shrink_prob(p: float, m: float, k: float) -> float:
    """m + k * (p - m), clipped to [0.01, 0.99]. k = 0 gives the market (clipped); k = 1 gives the model (clipped)."""
    return min(max(m + k * (p - m), CLIP_LO), CLIP_HI)
````

- [ ] **Step 4: Write the fit**

Create `src/sportsmodel/shrink/fit.py`:

````python
"""Fitting k for the prop shrink. PURE -- no IO.

Spec section 1. For one group of legs the Brier-minimising closed form, clamped:

    k = clamp( sum(y - m)(p - m) / sum(p - m)^2, 0, 1 )      (y = 1 when the OVER hit; pushes excluded; k = 0 if the denominator is 0)

A pooled NFL k is fitted on all eligible legs; each market gets its own k, partially pooled:

    k_market = (n_m * k_m_raw + 300 * k_pool) / (n_m + 300)

If the pooled sample has fewer than 300 legs every k is 0. The sample is the legs of the CURRENT props model version; if
that gives fewer than 300 legs, all versions are pooled.
"""
from __future__ import annotations

import math

from ..serving.props_ev import SIM_TO_ODDS_MARKET, MLB_SIM_TO_ODDS_MARKET
from .formula import market_prob_over

MIN_POOL_LEGS = 300          # pooled sample below this: k = 0 everywhere; version sample below this: pool all versions
PRIOR_LEGS = 300             # weight of the pooled k in each market's partially pooled k
ALL_VERSIONS = "all"         # model_version_used when the version fallback pooled every version
FIT_SPORT = "nfl"


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def legs_from_grades(rows: list[dict], market_map: dict[str, str] | None = None) -> list[dict]:
    """`nfl_prop_grades` rows (plus an optional model_version) -> fit legs {season, week, game_pk, market, p, m, y,
    version}. `market` is the board's ODDS market key (the sim-to-odds mapping is applied; unmapped sim markets such as
    pass_yds and rush_att are dropped). Excluded: a result other than hit / miss (pushes), a lean other than over /
    under, a missing or non-finite p_over, a missing season or week, and any row whose best over or under price is
    missing or not a valid American price (|price| < 100). y is 1 when the OVER hit."""
    market_map = SIM_TO_ODDS_MARKET if market_map is None else market_map
    legs = []
    for r in rows:
        market = market_map.get(r.get("market"))
        result, lean = r.get("result"), r.get("lean")
        if market is None or result not in ("hit", "miss") or lean not in ("over", "under"):
            continue
        p = _num(r.get("p_over"))
        m = market_prob_over(r.get("over_price"), r.get("under_price"))
        if p is None or m is None or r.get("season") is None or r.get("week") is None:
            continue
        legs.append({
            "season": int(r["season"]), "week": int(r["week"]), "game_pk": r.get("game_pk"), "market": market,
            "p": p, "m": m, "y": 1 if (lean == "over") == (result == "hit") else 0,
            "version": r.get("model_version"),
        })
    return legs


def closed_form_k(legs: list[dict]) -> float:
    """The clamped Brier-minimising k for these legs; 0.0 when there are no legs or no variance in p - m."""
    num = sum((l["y"] - l["m"]) * (l["p"] - l["m"]) for l in legs)
    den = sum((l["p"] - l["m"]) ** 2 for l in legs)
    if den <= 0.0:
        return 0.0
    return min(1.0, max(0.0, num / den))


def select_sample(legs: list[dict], current_version: str | None) -> tuple[list[dict], str]:
    """(sample, model_version_used): the legs of the current version if there are at least 300, else every leg."""
    if current_version:
        own = [l for l in legs if l.get("version") == current_version]
        if len(own) >= MIN_POOL_LEGS:
            return own, current_version
    return list(legs), ALL_VERSIONS


def fit_table(legs: list[dict], current_version: str | None) -> dict:
    """k per NFL board market, partially pooled, plus the pooled values. Every board market (SIM_TO_ODDS_MARKET value)
    gets an entry; a market with no legs gets k = 0, n = 0, k_raw = None. MLB gets k = 0 for every market (no unbiased
    fit sample exists in Part A)."""
    sample, used = select_sample(legs, current_version)
    n_pool = len(sample)
    k_pool_raw = closed_form_k(sample)
    enough = n_pool >= MIN_POOL_LEGS
    k_pool = k_pool_raw if enough else 0.0
    nfl = {}
    for market in sorted(set(SIM_TO_ODDS_MARKET.values())):
        own = [l for l in sample if l["market"] == market]
        n = len(own)
        if n == 0:
            nfl[market] = {"k": 0.0, "n": 0, "k_raw": None}
            continue
        raw = closed_form_k(own)
        k = (n * raw + PRIOR_LEGS * k_pool) / (n + PRIOR_LEGS) if enough else 0.0
        nfl[market] = {"k": k, "n": n, "k_raw": raw}
    mlb = {market: {"k": 0.0, "n": 0, "k_raw": None} for market in sorted(set(MLB_SIM_TO_ODDS_MARKET.values()))}
    return {"model_version_used": used, "pooled": {"k": k_pool, "n": n_pool, "k_raw": k_pool_raw},
            "markets": {"nfl": nfl, "mlb": mlb}}


def build_document(legs: list[dict], current_version: str | None, fitted_at: str) -> dict:
    """The contents of assets/shrink_k.json: the fit_table result plus when it was fitted, which version string was
    current, and the data window (first and last (season, week) of the eligible legs and how many there were in all)."""
    doc = fit_table(legs, current_version)
    weeks = sorted({(l["season"], l["week"]) for l in legs})
    return {
        "schema": 1, "fitted_at": fitted_at, "current_version": current_version,
        "model_version_used": doc["model_version_used"],
        "data_window": {"first": list(weeks[0]) if weeks else None, "last": list(weeks[-1]) if weeks else None,
                        "n_legs": len(legs)},
        "pooled": {"sport": FIT_SPORT, **doc["pooled"]}, "markets": doc["markets"],
    }


def has_fewer_legs(existing: dict | None, new: dict) -> bool:
    """True when the committed file was fitted on MORE eligible legs than this fit saw: graded legs only ever accumulate,
    so a smaller count means the read was truncated and the existing file must be kept."""
    try:
        before = existing["data_window"]["n_legs"]
    except (TypeError, KeyError):
        return False
    return isinstance(before, int) and new["data_window"]["n_legs"] < before
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/shrink`
Expected: `68 passed` (25 in `test_formula.py`, 43 in `test_fit.py`).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/shrink tests/shrink
git commit -m "feat(shrink): shrink formula, grade-row legs and the closed-form k fit" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The k loader (fail-safe) and the board integration in `props_ev`

**Files:**
- Create: `src/sportsmodel/shrink/store.py`
- Modify: `src/sportsmodel/serving/props_ev.py` (`assemble_prop_rows`)
- Create: `tests/shrink/test_store.py`, `tests/serving/test_props_ev_shrink.py`

**Interfaces:**
- Consumes (Task 1): `formula.shrink_prob(p, m, k) -> float`. Existing: `serving.props_ev.assemble_prop_rows(sim_rows, odds_rows, model_version, *, sport="nfl", market_map=None, gate=None, calibrate_fn=None)`, `config.PROJECT_ROOT`.
- Produces:
  - `store.SHRINK_K_PATH: Path` (`<project>/assets/shrink_k.json`), `store.MLB_K = 0.0`
  - `store.ShrinkTable(ks: Mapping[tuple[str, str], float] | None = None, *, problem: str | None = None)` with `.k(sport: str, market: str) -> float` (never raises, always in [0, 1], MLB always 0, records misses in `.missed`), `.warning() -> str | None` (one `::warning::shrink: ...` line), `.problem`, and `ShrinkTable.from_dict(doc) -> ShrinkTable`
  - `store.load_table(path: Path | str | None = None) -> ShrinkTable` (missing, unreadable or malformed file gives a table that answers 0)
  - `assemble_prop_rows(..., shrink=None)`: any object with `.k(sport, odds_market) -> float`; rows gain `raw_model_prob` (the pick side's probability before shrinking) and `shrink_k` (`None` when `shrink` is `None`)

- [ ] **Step 1: Write the failing tests**

Create `tests/shrink/test_store.py`:

````python
"""assets/shrink_k.json loading and the k = 0 fail-safe. No network, no database."""
import json
import pytest

from sportsmodel import config
from sportsmodel.shrink import store
from sportsmodel.shrink.store import ShrinkTable, load_table


def write(tmp_path, doc, name="shrink_k.json"):
    path = tmp_path / name
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return path


GOOD = {"markets": {"nfl": {"rush_yds": {"k": 0.2, "n": 400, "k_raw": 0.25}, "reception_yds": {"k": 0.1, "n": 50, "k_raw": 0.0}},
                    "mlb": {"total_bases": {"k": 0.0, "n": 0, "k_raw": None}}}}


def test_the_default_path_is_assets_shrink_k_json_in_the_project():
    assert store.SHRINK_K_PATH == config.PROJECT_ROOT / "assets" / "shrink_k.json"


def test_a_good_file_is_read_per_sport_and_market(tmp_path):
    assert load_table(write(tmp_path, GOOD)).k("nfl", "rush_yds") == 0.2
    t = ShrinkTable.from_dict(GOOD)
    assert t.k("nfl", "rush_yds") == 0.2 and t.k("nfl", "reception_yds") == 0.1
    assert t.warning() is None and t.problem is None


def test_a_missing_file_gives_k_zero_everywhere_and_a_warning(tmp_path):
    t = load_table(tmp_path / "nope.json")
    assert t.k("nfl", "rush_yds") == 0.0 and t.k("nfl", "anything") == 0.0
    assert "does not exist yet" in t.warning() and "k=0" in t.warning()


@pytest.mark.parametrize("text", ["", "{not json", "[]", "null", "42", '"x"', '{"markets": 5}', '{"markets": []}', "{}"])
def test_an_unusable_file_gives_k_zero_and_never_raises(tmp_path, text):
    t = load_table(write(tmp_path, text))
    assert t.k("nfl", "rush_yds") == 0.0 and t.warning().startswith("::warning::shrink:")


def test_an_unreadable_path_gives_k_zero(tmp_path):
    t = load_table(tmp_path)                       # a directory, not a file
    assert t.k("nfl", "rush_yds") == 0.0 and "unreadable" in t.warning()


def test_a_non_utf8_file_gives_k_zero(tmp_path):
    path = tmp_path / "shrink_k.json"
    path.write_bytes(b"\xff\xfe\x00bad")
    assert load_table(path).k("nfl", "rush_yds") == 0.0


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-0.1", "1.5", '"0.2"', "null", "true", "[0.2]", "{}"])
def test_an_entry_whose_k_is_not_a_number_in_zero_one_is_ignored_not_trusted(tmp_path, bad):
    doc = '{"markets": {"nfl": {"rush_yds": {"k": %s, "n": 5}, "receptions": {"k": 0.3}}}}' % bad
    t = load_table(write(tmp_path, doc))
    assert t.k("nfl", "rush_yds") == 0.0 and t.k("nfl", "receptions") == 0.3
    assert "nfl/rush_yds" in t.warning() and "malformed" in t.warning()


def test_k_of_exactly_zero_and_one_are_valid():
    t = ShrinkTable.from_dict({"markets": {"nfl": {"a": {"k": 0}, "b": {"k": 1}, "c": {"k": 1.0}}}})
    assert (t.k("nfl", "a"), t.k("nfl", "b"), t.k("nfl", "c")) == (0.0, 1.0, 1.0) and t.problem is None


def test_a_missing_market_entry_is_zero_and_warned_once_per_market():
    t = ShrinkTable.from_dict(GOOD)
    assert t.k("nfl", "pass_tds") == 0.0 and t.k("nfl", "pass_tds") == 0.0 and t.k("nfl", "receptions") == 0.0
    w = t.warning()
    assert w.count("nfl/pass_tds") == 1 and "nfl/receptions" in w and w.count("::warning::") == 1


def test_mlb_is_always_zero_even_if_the_file_says_otherwise():
    t = ShrinkTable.from_dict({"markets": {"mlb": {"total_bases": {"k": 0.9}}, "nfl": {}}})
    assert t.k("mlb", "total_bases") == 0.0 and t.k("mlb", "pitcher_ks") == 0.0
    assert t.warning() is None                          # an MLB lookup is never a "missing entry"


def test_the_committed_file_if_present_is_a_valid_table():
    path = store.SHRINK_K_PATH
    if path.exists():                                   # absent until the first fit_shrink_k run
        t = load_table(path)
        assert t.problem is None
````

and `tests/serving/test_props_ev_shrink.py`:

````python
"""The shrink inside serving.props_ev.assemble_prop_rows: formula, k per sport and market, MLB order, the pick decision,
the audit fields and the k = 0 board. Pure."""
import pytest

from sportsmodel.model import calibration
from sportsmodel.serving.board import EV_CEILING, decimal_odds, ev as _ev, novig as _novig
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    always_propable,
    assemble_prop_rows,
)
from sportsmodel.shrink.formula import shrink_prob
from sportsmodel.shrink.store import ShrinkTable

MV = "props-sim-v1-test"


def _pmf(mapping, length):
    pmf = [0.0] * length
    for k, p in mapping.items():
        pmf[k] = p
    return {"kind": "pmf", "pmf": pmf}


def _sim(market="rec_yds", p_over=0.69, name="A.J. Brown"):
    """A sim row whose P(over) at line 68.5 is exactly p_over (mass at 75 vs 50)."""
    return {"game_pk": 1001, "player_id": "p1", "name": name, "market": market, "mean": 70.0,
            "dist": _pmf({50: 1 - p_over, 75: p_over}, 76),
            "commence_time": "2026-09-21T17:00:00Z", "matchup": "PHI @ DAL"}


def _o(side, book, price, market="reception_yds", line=68.5, name="AJ Brown"):
    return {"game_pk": 1001, "market": market, "side": side, "player_name": name, "book": book, "line": line, "price": price}


ODDS = [_o("over", "draftkings", -125), _o("under", "fanduel", 105), _o("over", "pinnacle", -130)]
M = _novig(-125, 105)                         # the market's no-vig over probability for ODDS


class KStub:
    """Records which (sport, market) the board asks for and answers a fixed k."""

    def __init__(self, k):
        self.k_value, self.asked = k, []

    def k(self, sport, market):
        self.asked.append((sport, market))
        return self.k_value


def nfl(sim=None, odds=None, **kw):
    return assemble_prop_rows([sim or _sim()], odds or ODDS, MV, **kw)


def mlb(sim, odds, **kw):
    return assemble_prop_rows([sim], odds, MV, sport="mlb", market_map=MLB_SIM_TO_ODDS_MARKET, gate=always_propable, **kw)


# ---- no table: the legacy path used by the join tests ---------------------------------------------------------------

def test_without_a_table_the_model_probability_is_untouched_and_shrink_k_is_none():
    [r] = nfl()
    assert r["model_prob"] == pytest.approx(0.69) and r["raw_model_prob"] == pytest.approx(0.69) and r["shrink_k"] is None


# ---- the formula inside the board --------------------------------------------------------------------------------

def test_k_one_third_moves_the_model_probability_a_third_of_the_way_to_the_market():
    [r] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 1 / 3}))
    expected = shrink_prob(0.69, M, 1 / 3)
    assert r["side"] == "over" and r["model_prob"] == pytest.approx(expected) and expected == pytest.approx(M + (0.69 - M) / 3)
    assert r["raw_model_prob"] == pytest.approx(0.69) and r["shrink_k"] == pytest.approx(1 / 3)
    assert r["market_prob"] == pytest.approx(M) and r["edge"] == pytest.approx(expected - M)
    assert r["ev_best"] == pytest.approx(_ev(expected, -125)) and r["best_price"] == -125 and r["best_book"] == "draftkings"


def test_k_one_is_the_raw_model_and_k_zero_is_the_market():
    [full] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert full["model_prob"] == pytest.approx(0.69) and full["edge"] == pytest.approx(0.69 - M)
    [flat] = nfl(shrink=ShrinkTable())
    assert flat["model_prob"] == pytest.approx(M) and flat["edge"] == pytest.approx(0.0, abs=1e-12) and flat["shrink_k"] == 0.0
    assert flat["raw_model_prob"] == pytest.approx(0.69)


def test_the_shrunk_probability_is_clipped_to_the_specs_bounds():
    [r] = nfl(sim=_sim(p_over=1.0), shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert r["model_prob"] == 0.99 and r["raw_model_prob"] == 1.0
    [u] = nfl(sim=_sim(p_over=0.0), odds=[_o("over", "draftkings", 400), _o("under", "fanduel", -150)],
              shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert u["side"] == "under" and u["model_prob"] == pytest.approx(0.99) and u["raw_model_prob"] == 1.0


def test_the_under_side_uses_one_minus_the_shrunk_probability_and_records_the_raw_under_probability():
    odds = [_o("over", "draftkings", 140), _o("under", "fanduel", -165)]
    [r] = nfl(sim=_sim(p_over=0.20), odds=odds, shrink=ShrinkTable({("nfl", "reception_yds"): 0.5}))
    m = _novig(140, -165)
    shrunk_over = shrink_prob(0.20, m, 0.5)
    assert r["side"] == "under" and r["model_prob"] == pytest.approx(1 - shrunk_over)
    assert r["raw_model_prob"] == pytest.approx(0.80) and r["market_prob"] == pytest.approx(1 - m)
    assert r["edge"] == pytest.approx((1 - shrunk_over) - (1 - m)) and r["ev_best"] == pytest.approx(_ev(1 - shrunk_over, -165))


# ---- which k, for which sport and market ----------------------------------------------------------------------------

def test_the_board_asks_for_k_by_sport_and_the_odds_market_key_not_the_sim_market():
    stub = KStub(0.5)
    nfl(shrink=stub)
    assert stub.asked == [("nfl", "reception_yds")]                    # rec_yds (sim) -> reception_yds (odds)
    stub = KStub(0.5)
    sim = _sim(market="total_bases", p_over=0.55, name="Shohei Ohtani") | {"game_pk": 849819}
    odds = [{"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani", "book": "draftkings", "line": 68.5, "price": 110},
            {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani", "book": "fanduel", "line": 68.5, "price": -130}]
    mlb(sim, odds, shrink=stub)
    assert stub.asked == [("mlb", "total_bases")]


def test_each_market_gets_its_own_k():
    table = ShrinkTable({("nfl", "reception_yds"): 0.4, ("nfl", "rush_yds"): 0.1})
    [rec] = nfl(shrink=table)
    odds_rush = [_o("over", "draftkings", -125, market="rush_yds"), _o("under", "fanduel", 105, market="rush_yds")]
    [rush] = nfl(sim=_sim(market="rush_yds"), odds=odds_rush, shrink=table)
    assert rec["shrink_k"] == 0.4 and rush["shrink_k"] == 0.1
    assert rec["model_prob"] == pytest.approx(shrink_prob(0.69, M, 0.4)) and rush["model_prob"] == pytest.approx(shrink_prob(0.69, M, 0.1))


def test_a_market_missing_from_the_table_is_price_only_never_the_raw_model():
    odds = [_o("over", "draftkings", -125, market="pass_tds"), _o("under", "fanduel", 105, market="pass_tds")]
    sim = _sim(market="pass_tds") | {"mean": 1.5}
    table = ShrinkTable({("nfl", "rush_yds"): 0.3})
    [r] = nfl(sim=sim, odds=odds, shrink=table)
    assert r["shrink_k"] == 0.0 and r["model_prob"] == pytest.approx(M) and r["raw_model_prob"] == pytest.approx(0.69)
    assert ("nfl", "pass_tds") in table.missed


def test_mlb_is_price_only_in_part_a_even_if_a_table_lists_mlb_markets():
    sim = _sim(market="total_bases", p_over=0.55, name="Shohei Ohtani") | {"game_pk": 849819}
    odds = [{"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani", "book": "draftkings", "line": 68.5, "price": 110},
            {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani", "book": "fanduel", "line": 68.5, "price": -130}]
    [r] = mlb(sim, odds, shrink=ShrinkTable({("mlb", "total_bases"): 0.9}))
    assert r["shrink_k"] == 0.0 and r["model_prob"] == pytest.approx(_novig(110, -130)) and r["raw_model_prob"] == pytest.approx(0.55)


def test_mlb_calibration_runs_before_the_shrink():
    sim = _sim(market="total_bases", p_over=0.55, name="Shohei Ohtani") | {"game_pk": 849819}
    odds = [{"game_pk": 849819, "market": "total_bases", "side": "over", "player_name": "Shohei Ohtani", "book": "draftkings", "line": 68.5, "price": 110},
            {"game_pk": 849819, "market": "total_bases", "side": "under", "player_name": "Shohei Ohtani", "book": "fanduel", "line": 68.5, "price": -130}]
    [r] = mlb(sim, odds, calibrate_fn=lambda market, p: 0.40, shrink=KStub(0.5))
    m = _novig(110, -130)
    over = shrink_prob(0.40, m, 0.5)                                   # calibrated 0.40 (not the raw 0.55), then shrunk
    assert r["side"] == "under" and r["model_prob"] == pytest.approx(1 - over) and r["raw_model_prob"] == pytest.approx(0.60)


# ---- the pick decision uses the shrunk EV -----------------------------------------------------------------------------

def test_a_raw_pick_that_the_shrink_removes_is_not_a_pick():
    [raw] = nfl()
    assert raw["is_pick"] is True and 0 < raw["ev_best"] <= EV_CEILING
    [shrunk] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 0.05}))
    assert shrunk["ev_best"] == pytest.approx(_ev(shrink_prob(0.69, M, 0.05), -125)) and shrunk["ev_best"] < 0
    assert shrunk["is_pick"] is False


def test_a_shrunk_probability_can_pick_the_other_side():
    odds = [_o("over", "draftkings", -150), _o("under", "fanduel", 130)]
    [raw] = nfl(sim=_sim(p_over=0.62), odds=odds)
    [shrunk] = nfl(sim=_sim(p_over=0.62), odds=odds, shrink=ShrinkTable({("nfl", "reception_yds"): 0.0}))
    assert raw["side"] == "over"
    m = _novig(-150, 130)
    ev_over, ev_under = _ev(m, -150), _ev(1 - m, 130)
    assert shrunk["side"] == ("over" if ev_over >= ev_under else "under")          # decided by the shrunk probability


def test_the_ev_ceiling_still_applies_to_the_shrunk_ev():
    odds = [_o("over", "draftkings", 300), _o("under", "fanduel", -400)]
    [r] = nfl(sim=_sim(p_over=0.60), odds=odds, shrink=ShrinkTable({("nfl", "reception_yds"): 1.0}))
    assert r["ev_best"] > EV_CEILING and r["is_pick"] is False


def test_at_k_zero_only_cross_book_pricing_gaps_are_picks_and_both_sides_tie():
    arb = [_o("over", "draftkings", 105), _o("under", "fanduel", 105)]             # best prices from two books imply under 100%
    [r] = nfl(odds=arb, shrink=ShrinkTable())
    assert r["is_pick"] is True and r["ev_best"] == pytest.approx(1 / (2 / 2.05) - 1)       # the tie: either side pays the same
    normal = [_o("over", "draftkings", -110), _o("under", "fanduel", -110)]
    [n] = nfl(odds=normal, shrink=ShrinkTable())
    assert n["is_pick"] is False and n["ev_best"] < 0


def test_rows_always_carry_the_audit_keys():
    [r] = nfl(shrink=ShrinkTable({("nfl", "reception_yds"): 0.2}))
    assert {"raw_model_prob", "shrink_k"} <= set(r)
    assert r["model_prob"] != r["raw_model_prob"]
````

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/shrink/test_store.py tests/serving/test_props_ev_shrink.py`
Expected: `2 errors` during collection, `ModuleNotFoundError: No module named 'sportsmodel.shrink.store'`.

- [ ] **Step 3: Write the loader**

Create `src/sportsmodel/shrink/store.py`:

````python
"""The committed k parameters (assets/shrink_k.json) and how the board reads them. The fail-safe lives here.

If the file is missing or unreadable, or the (sport, market) entry is missing or not a number in [0, 1], k is 0: the board
is price-only. It NEVER falls back to the raw model. MLB is always 0 in Part A, whatever the file says.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping

from .. import config

SHRINK_K_PATH = config.PROJECT_ROOT / "assets" / "shrink_k.json"
MLB_K = 0.0                                  # Part A: no unbiased MLB fit sample, so MLB is price-only


def _valid_k(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0.0 <= x <= 1.0)


class ShrinkTable:
    """k by (sport, odds market). `k()` never raises and never returns anything outside [0, 1]."""

    def __init__(self, ks: Mapping[tuple[str, str], float] | None = None, *, problem: str | None = None):
        self._ks = dict(ks or {})
        self.problem = problem               # why the whole file was unusable, or None
        self.missed: set[tuple[str, str]] = set()

    @classmethod
    def from_dict(cls, doc) -> "ShrinkTable":
        """Build from the parsed JSON document; anything malformed becomes a table that answers 0."""
        if not isinstance(doc, dict) or not isinstance(doc.get("markets"), dict):
            return cls(problem="has no 'markets' object")
        ks, bad = {}, []
        for sport, markets in doc["markets"].items():
            if not isinstance(markets, dict):
                bad.append(str(sport))
                continue
            for market, entry in markets.items():
                k = entry.get("k") if isinstance(entry, dict) else None
                if _valid_k(k):
                    ks[(sport, market)] = float(k)
                else:
                    bad.append(f"{sport}/{market}")
        return cls(ks, problem=("has malformed entries (ignored): " + ", ".join(sorted(bad))) if bad else None)

    def k(self, sport: str, market: str) -> float:
        if sport == "mlb":
            return MLB_K
        key = (sport, market)
        if key in self._ks:
            return self._ks[key]
        self.missed.add(key)
        return 0.0

    def warning(self) -> str | None:
        """One line for the run log if anything fell back to k = 0, else None."""
        parts = []
        if self.problem:
            parts.append(self.problem)
        if self.missed:
            parts.append("no k for " + ", ".join(f"{s}/{m}" for s, m in sorted(self.missed)))
        return ("::warning::shrink: " + "; ".join(parts) + "; using k=0 (price-only) there") if parts else None


def load_table(path: Path | str | None = None) -> ShrinkTable:
    """Read assets/shrink_k.json. A missing, unreadable or malformed file gives a table that answers k = 0 everywhere."""
    path = Path(path) if path is not None else SHRINK_K_PATH
    try:
        text = path.read_text()
    except FileNotFoundError:
        return ShrinkTable(problem=f"{path.name} does not exist yet (first fit has not run)")
    except (OSError, UnicodeDecodeError) as exc:
        return ShrinkTable(problem=f"{path.name} is unreadable ({type(exc).__name__})")
    try:
        doc = json.loads(text)
    except ValueError:
        return ShrinkTable(problem=f"{path.name} is not valid JSON")
    table = ShrinkTable.from_dict(doc)
    if table.problem:
        table.problem = f"{path.name} {table.problem}"
    return table
````

- [ ] **Step 4: Apply the shrink inside `assemble_prop_rows`**

In `src/sportsmodel/serving/props_ev.py` (edit 1 of 6), replace:

````python
from datetime import timedelta

from ..model.distributions import prob_over_dist
from .board import EV_CEILING, best_price, decimal_odds, ev, novig

# Sim aggregate market -> Odds API / SportConfig["nfl"].prop_market_map key.
````

with:

````python
from datetime import timedelta

from ..model.distributions import prob_over_dist
from ..shrink.formula import shrink_prob
from .board import EV_CEILING, best_price, decimal_odds, ev, novig

# Sim aggregate market -> Odds API / SportConfig["nfl"].prop_market_map key.
````

In `src/sportsmodel/serving/props_ev.py` (edit 2 of 6), replace:

````python
    market_map: dict[str, str] | None = None,
    gate=None,
    calibrate_fn=None,
) -> list[dict]:
    """Join sim player-prop distributions to book prop odds and emit +EV pick
    rows shaped exactly like `sportsmodel.db._EV_PROP_PICKS_COLS`.
````

with:

````python
    market_map: dict[str, str] | None = None,
    gate=None,
    calibrate_fn=None,
    shrink=None,
) -> list[dict]:
    """Join sim player-prop distributions to book prop odds and emit +EV pick
    rows shaped exactly like `sportsmodel.db._EV_PROP_PICKS_COLS`.
````

In `src/sportsmodel/serving/props_ev.py` (edit 3 of 6), replace:

````python
    unchanged. MLB passes `sport="mlb"`, `market_map=MLB_SIM_TO_ODDS_MARKET`,
    `gate=always_propable` and `calibrate_fn=calibration.calibrate` (the model's P(over)
    at the book's line is Platt-corrected per market, as the pre-pause MLB board did).
    """
    market_map = SIM_TO_ODDS_MARKET if market_map is None else market_map
    gate = is_propable_projected if gate is None else gate
````

with:

````python
    unchanged. MLB passes `sport="mlb"`, `market_map=MLB_SIM_TO_ODDS_MARKET`,
    `gate=always_propable` and `calibrate_fn=calibration.calibrate` (the model's P(over)
    at the book's line is Platt-corrected per market, as the pre-pause MLB board did).

    `shrink` (profit step 2 Part A): a `sportsmodel.shrink.store.ShrinkTable`. When given, the over probability (after
    `calibrate_fn`) is replaced by `shrink_prob(p, m, k)` with m = the no-vig over probability of the best soft over and
    best soft under, and k = `shrink.k(sport, odds_market)`; EV, the side, the pick decision, `model_prob` and `edge` all
    use the shrunk value, and the row records `raw_model_prob` (the pick side's probability before shrinking) and
    `shrink_k`. None (the default) leaves the model untouched and writes `shrink_k = None`; every production caller
    (scripts/build_ev_props_board.py) passes a table, which answers k = 0 when the committed file is missing.
    """
    market_map = SIM_TO_ODDS_MARKET if market_map is None else market_map
    gate = is_propable_projected if gate is None else gate
````

In `src/sportsmodel/serving/props_ev.py` (edit 4 of 6), replace:

````python
            continue

        novig_over = novig(ob[1], ub[1])
        ev_over = ev(p_over, ob[1])
        ev_under = ev(1 - p_over, ub[1])

````

with:

````python
            continue

        novig_over = novig(ob[1], ub[1])
        raw_p_over, k_applied = p_over, None
        if shrink is not None:
            k_applied = shrink.k(sport, odds_market)
            p_over = shrink_prob(p_over, novig_over, k_applied)
        ev_over = ev(p_over, ob[1])
        ev_under = ev(1 - p_over, ub[1])

````

In `src/sportsmodel/serving/props_ev.py` (edit 5 of 6), replace:

````python
            model_prob, market_prob, ev_best, picked, side_entries = (
                p_over, novig_over, ev_over, ob, over_entries,
            )
        else:
            side = "under"
            model_prob, market_prob, ev_best, picked, side_entries = (
                1 - p_over, 1 - novig_over, ev_under, ub, under_entries,
            )

        pinnacle_price = None
        for bk, price in side_entries:
````

with:

````python
            model_prob, market_prob, ev_best, picked, side_entries = (
                p_over, novig_over, ev_over, ob, over_entries,
            )
            raw_model_prob = raw_p_over
        else:
            side = "under"
            model_prob, market_prob, ev_best, picked, side_entries = (
                1 - p_over, 1 - novig_over, ev_under, ub, under_entries,
            )
            raw_model_prob = 1 - raw_p_over

        pinnacle_price = None
        for bk, price in side_entries:
````

In `src/sportsmodel/serving/props_ev.py` (edit 6 of 6), replace:

````python
            "matchup": sim_row.get("matchup"),
            "commence_time": sim_row.get("commence_time"),
            "model_prob": model_prob,
            "market_prob": market_prob,
            "edge": model_prob - market_prob,
            "ev_best": ev_best,
````

with:

````python
            "matchup": sim_row.get("matchup"),
            "commence_time": sim_row.get("commence_time"),
            "model_prob": model_prob,
            "raw_model_prob": raw_model_prob,
            "shrink_k": k_applied,
            "market_prob": market_prob,
            "edge": model_prob - market_prob,
            "ev_best": ev_best,
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/shrink tests/serving`
Expected: `211 passed` (every existing `tests/serving` test, including the existing `assemble_prop_rows` tests, is unchanged and still passes because `shrink` defaults to `None`).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/shrink/store.py src/sportsmodel/serving/props_ev.py tests/shrink/test_store.py tests/serving/test_props_ev_shrink.py
git commit -m "feat(shrink): k loader with the k=0 fail-safe; shrink inside assemble_prop_rows" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Audit columns, migration and board wiring

**Files:**
- Create: `db/migration_shrink_props.sql`, `tests/test_db_ev_prop_shrink_cols.py`
- Modify: `src/sportsmodel/db.py` (`upsert_ev_prop_picks`), `scripts/build_ev_props_board.py` (`main`)
- Modify (tests): `tests/test_build_ev_props_board.py`, `tests/ledger/test_hooks.py`, `tests/ledger/test_sources.py`, `tests/serving/test_best_parlays.py`

**Interfaces:**
- Consumes (Task 2): `store.load_table() -> ShrinkTable`, `ShrinkTable.warning()`, `assemble_prop_rows(..., shrink=...)`. Existing: `db._existing_columns(cur, table, columns) -> set[str]`.
- Produces:
  - `db._EV_PROP_PICKS_OPTIONAL_COLS = ["raw_model_prob", "shrink_k"]`; `db.upsert_ev_prop_picks(records)` writes them only when a record carries them AND the table has them (one `information_schema` probe per call; a one-line `WARN` when absent)
  - `scripts/build_ev_props_board.py` module attribute `load_table` (tests monkeypatch it); `main` passes the table to `assemble_prop_rows` for both sports and prints `shrink.warning()` once

- [ ] **Step 1: Write the failing tests**

Create `tests/test_db_ev_prop_shrink_cols.py`:

````python
"""ev_prop_picks' optional audit columns (raw_model_prob, shrink_k): written only when the table has them, and
db/migration_shrink_props.sql. No live DB (fake connections), same pattern as tests/test_db_prediction_accuracy.py."""
import re
from pathlib import Path

from sportsmodel import db
from sportsmodel import db as db_module

MIGRATION = (Path(__file__).resolve().parents[1] / "db" / "migration_shrink_props.sql").read_text()


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def execute(self, sql, params=None):
        self.sink.setdefault("probes", []).append((sql, params))

    def fetchall(self):
        return [(c,) for c in self.sink.get("existing_cols", [])]

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return FakeCursor(self.sink)

    def commit(self):
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def row(**o):
    base = {"sport": "nfl", "game_pk": 1, "player_id": "p", "market": "rush_yds", "line": 54.5, "model_version": "v",
            "model_prob": 0.55, "raw_model_prob": 0.66, "shrink_k": 0.25, "is_pick": True}
    return base | o


def test_the_optional_columns_are_the_two_audit_columns():
    assert db._EV_PROP_PICKS_OPTIONAL_COLS == ["raw_model_prob", "shrink_k"]
    assert not set(db._EV_PROP_PICKS_OPTIONAL_COLS) & set(db._EV_PROP_PICKS_COLS)


def test_both_columns_are_written_after_the_migration_and_updated_on_conflict(monkeypatch):
    sink = {"existing_cols": ["raw_model_prob", "shrink_k"]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    assert db.upsert_ev_prop_picks([row()]) == 1
    cols = db._EV_PROP_PICKS_COLS + ["raw_model_prob", "shrink_k"]
    (tup,) = sink["rows"]
    assert len(tup) == len(cols) and tup[-2:] == (0.66, 0.25) and tup[cols.index("model_prob")] == 0.55
    assert "raw_model_prob = EXCLUDED.raw_model_prob" in sink["sql"] and "shrink_k = EXCLUDED.shrink_k" in sink["sql"]
    assert "open_pinnacle_price = EXCLUDED" not in sink["sql"]                       # the immutable column rule is unchanged
    assert len(sink["probes"]) == 1 and "information_schema.columns" in sink["probes"][0][0] and sink["committed"]


def test_before_the_migration_the_board_is_written_without_them_and_warns(monkeypatch, capsys):
    sink = {"existing_cols": []}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    assert db.upsert_ev_prop_picks([row(), row(player_id="q")]) == 2
    assert all(len(t) == len(db._EV_PROP_PICKS_COLS) for t in sink["rows"])
    assert "raw_model_prob" not in sink["sql"] and "shrink_k" not in sink["sql"]
    out = capsys.readouterr().out
    assert "WARN" in out and "migration_shrink_props.sql" in out and len(sink["probes"]) == 1


def test_only_the_existing_optional_column_is_written(monkeypatch, capsys):
    sink = {"existing_cols": ["shrink_k"]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    db.upsert_ev_prop_picks([row()])
    assert sink["rows"][0][-1] == 0.25 and "raw_model_prob" not in sink["sql"] and "shrink_k = EXCLUDED.shrink_k" in sink["sql"]
    assert "raw_model_prob" in capsys.readouterr().out


def test_a_none_shrink_k_is_still_a_carried_key_and_written_as_null(monkeypatch):
    sink = {"existing_cols": ["raw_model_prob", "shrink_k"]}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    db.upsert_ev_prop_picks([row(shrink_k=None)])
    assert sink["rows"][0][-1] is None


def test_no_probe_when_records_carry_neither_key(monkeypatch):
    sink = {}
    monkeypatch.setattr(db_module, "get_postgres", lambda: FakeConn(sink))
    legacy = {k: v for k, v in row().items() if k not in ("raw_model_prob", "shrink_k")}
    assert db.upsert_ev_prop_picks([legacy]) == 1
    assert "probes" not in sink and len(sink["rows"][0]) == len(db._EV_PROP_PICKS_COLS)


def test_an_empty_list_never_connects(monkeypatch):
    monkeypatch.setattr(db_module, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("no connection")))
    assert db.upsert_ev_prop_picks([]) == 0


def test_the_migration_is_the_specs_idempotent_additive_alter():
    stmts = [l for l in MIGRATION.splitlines() if l.strip() and not l.startswith("--")]
    assert stmts == ["ALTER TABLE ev_prop_picks ADD COLUMN IF NOT EXISTS raw_model_prob double precision, "
                     "ADD COLUMN IF NOT EXISTS shrink_k double precision;"]
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE|UPDATE|INSERT)\b", "\n".join(stmts), re.IGNORECASE)
````

Then edit four existing test files. The MLB board test changes on purpose (MLB is price-only in Part A), the ledger hook test pins k = 1 so it keeps testing which line is recorded, and three groups of tests pin that nothing else needs to change (parlay legs and ledger rows already carry whatever `model_prob` the board wrote).

In `tests/test_build_ev_props_board.py`, replace:

````python
    assert seen["sport"] == "mlb"
    [row] = written["rows"]
    assert row["sport"] == "mlb" and row["model_version"] == "props-mlb-v1"
    assert row["market"] == "total_bases" and row["side"] == "over" and row["is_pick"] is True
    assert row["player_id"] == "660271"
    assert cleared["picks"] == [row]
````

with:

````python
    assert seen["sport"] == "mlb"
    [row] = written["rows"]
    assert row["sport"] == "mlb" and row["model_version"] == "props-mlb-v1"
    # MLB is price-only in Part A (k = 0): the model's 0.55 no longer makes this a pick; the audit columns say why.
    assert row["market"] == "total_bases" and row["is_pick"] is False
    assert row["raw_model_prob"] in (pytest.approx(0.55), pytest.approx(0.45)) and row["shrink_k"] == 0.0
    assert row["model_prob"] == pytest.approx(row["market_prob"])
    assert row["player_id"] == "660271"
    assert cleared["picks"] == []


# ---- profit step 2 Part A: the shrink on the board ------------------------------------------------------------------

from sportsmodel.shrink.store import ShrinkTable, load_table  # noqa: E402


def _nfl_slate():
    sim = [{"game_pk": 1, "player_id": "00-1", "name": "A. Brown", "market": "rec_yds", "mean": 66.0,
            "dist": {"kind": "normal", "mean": 66.0, "sd": 20.0}, "commence_time": "2026-10-04T17:00:00Z", "matchup": "A @ B"}]
    odds = [{"game_pk": 1, "market": "reception_yds", "side": side, "player_name": "A. Brown", "book": book,
             "line": 64.5, "price": price, "captured_at": "2026-10-04T12:00:00+00:00"}
            for side, book, price in [("over", "draftkings", -105), ("under", "fanduel", -115),
                                      ("over", "fanduel", -115), ("under", "draftkings", -105)]]
    return sim, odds


def _run_nfl(monkeypatch, table):
    sim, odds = _nfl_slate()
    monkeypatch.setattr(build_ev_props_board, "load_sim_rows", lambda sport: sim)
    monkeypatch.setattr(build_ev_props_board, "load_latest_prop_odds", lambda sport, pks: odds)
    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines", lambda rows: 0)
    monkeypatch.setattr(build_ev_props_board, "load_table", lambda: table)
    written = {}
    monkeypatch.setattr(build_ev_props_board, "upsert_ev_prop_picks", lambda rows: written.setdefault("rows", rows))
    monkeypatch.setattr(build_ev_props_board, "clear_stale_prop_line_picks", lambda picks: 0)
    monkeypatch.setattr(build_ev_props_board, "safe_record", lambda *a, **k: None)
    build_ev_props_board.main(["--sport", "nfl"])
    return written["rows"]


def test_main_reads_the_committed_shrink_file_by_default():
    import inspect
    assert "load_table()" in inspect.getsource(build_ev_props_board.main)
    assert build_ev_props_board.load_table is load_table


def test_main_hands_the_table_to_assemble_for_both_sports(monkeypatch):
    table, seen = ShrinkTable({("nfl", "reception_yds"): 0.2}), []
    monkeypatch.setattr(build_ev_props_board, "load_table", lambda: table)
    monkeypatch.setattr(build_ev_props_board, "load_sim_rows", lambda sport: [])
    monkeypatch.setattr(build_ev_props_board, "load_latest_prop_odds", lambda sport, pks: [])
    monkeypatch.setattr(build_ev_props_board, "upsert_nfl_prop_lines", lambda rows: 0)
    monkeypatch.setattr(build_ev_props_board, "upsert_ev_prop_picks", lambda rows: 0)
    monkeypatch.setattr(build_ev_props_board, "clear_stale_prop_line_picks", lambda picks: 0)
    monkeypatch.setattr(build_ev_props_board, "safe_record", lambda *a, **k: None)
    monkeypatch.setattr(build_ev_props_board, "assemble_prop_rows", lambda *a, **k: seen.append(k.get("shrink")) or [])
    build_ev_props_board.main(["--sport", "nfl"])
    build_ev_props_board.main(["--sport", "mlb"])
    assert seen == [table, table]


def test_with_no_shrink_file_the_board_runs_price_only_and_warns_once(monkeypatch, tmp_path, capsys):
    rows = _run_nfl(monkeypatch, load_table(tmp_path / "shrink_k.json"))
    [row] = rows
    assert row["shrink_k"] == 0.0 and row["model_prob"] == pytest.approx(row["market_prob"]) and row["raw_model_prob"] != row["model_prob"]
    out = capsys.readouterr().out
    assert out.count("::warning::shrink:") == 1 and "does not exist yet" in out


def test_with_a_fitted_table_the_board_prices_the_shrunk_probability(monkeypatch, capsys):
    [row] = _run_nfl(monkeypatch, ShrinkTable({("nfl", "reception_yds"): 0.5}))
    raw, k = row["raw_model_prob"], row["shrink_k"]
    assert k == 0.5 and row["model_prob"] == pytest.approx(row["market_prob"] + 0.5 * (raw - row["market_prob"]))
    assert "::warning::shrink:" not in capsys.readouterr().out
````

In `tests/ledger/test_hooks.py` (edit 1 of 2), replace:

````python

from sportsmodel import db
from sportsmodel.ledger import record, sources

ROOT = Path(__file__).resolve().parents[2]
KICK = datetime.now(timezone.utc) + timedelta(hours=30)
````

with:

````python

from sportsmodel import db
from sportsmodel.ledger import record, sources
from sportsmodel.shrink.store import ShrinkTable

ROOT = Path(__file__).resolve().parents[2]
KICK = datetime.now(timezone.utc) + timedelta(hours=30)
````

In `tests/ledger/test_hooks.py` (edit 2 of 2), replace:

````python
    monkeypatch.setattr(mod, "upsert_nfl_prop_lines", lambda r: 0)
    monkeypatch.setattr(mod, "upsert_ev_prop_picks", lambda r: events.append(("upsert", len(r))))
    monkeypatch.setattr(mod, "clear_stale_prop_line_picks", lambda picks: 0)
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main(["--sport", "nfl"])
````

with:

````python
    monkeypatch.setattr(mod, "upsert_nfl_prop_lines", lambda r: 0)
    monkeypatch.setattr(mod, "upsert_ev_prop_picks", lambda r: events.append(("upsert", len(r))))
    monkeypatch.setattr(mod, "clear_stale_prop_line_picks", lambda picks: 0)
    # k = 1: the raw model decides, as before the shrink; the test is about WHICH LINE is recorded, not the probability
    monkeypatch.setattr(mod, "load_table", lambda: ShrinkTable({("nfl", "reception_yds"): 1.0}))
    rec = Recorder(events)
    monkeypatch.setattr(mod, "safe_record", rec)
    mod.main(["--sport", "nfl"])
````

In `tests/ledger/test_sources.py`, replace:

````python
    assert o["source"] == "ev_prop" and o["player_id"] == "00-1" and o["line"] == 64.5
    assert (o["model_prob"], o["ref_prob"], o["ref_kind"]) == (0.58, 0.51, None) and o["ev_pct"] == pytest.approx(7.0)
    assert o["pick_key"] == "ev_prop|nfl|7|rec_yds|over|00-1||"


# ---- parlay legs -----------------------------------------------------------------------------------
````

with:

````python
    assert o["source"] == "ev_prop" and o["player_id"] == "00-1" and o["line"] == 64.5
    assert (o["model_prob"], o["ref_prob"], o["ref_kind"]) == (0.58, 0.51, None) and o["ev_pct"] == pytest.approx(7.0)
    assert o["pick_key"] == "ev_prop|nfl|7|rec_yds|over|00-1||"


def test_ev_prop_observation_records_the_shrunk_probability_that_made_the_pick():
    row = {"sport": "nfl", "game_pk": 7, "player_id": "00-1", "player_name": "A. Brown", "market": "rec_yds",
           "side": "over", "line": 64.5, "model_version": "props-sim-v1", "commence_time": KICK,
           "model_prob": 0.52, "raw_model_prob": 0.66, "shrink_k": 0.25, "market_prob": 0.50, "ev_best": 0.04,
           "best_book": "dk", "best_price": 100}
    (o,) = sources.ev_prop_observations([row])
    assert o["model_prob"] == 0.52 and "raw_model_prob" not in o and "shrink_k" not in o       # no ledger change in Part A


# ---- parlay legs -----------------------------------------------------------------------------------
````

In `tests/serving/test_best_parlays.py`, replace:

````python
    one_push = settle_parlay(legs, ["win", "push", "win"])
    assert abs(one_push["payout_dec"] - (1 + 100 / 110) * (1 + 100 / 120)) < 1e-9
    assert settle_parlay(legs, ["push", "push", "push"]) == {"result": "push", "pnl": 0.0, "payout_dec": 1.0}
````

with:

````python
    one_push = settle_parlay(legs, ["win", "push", "win"])
    assert abs(one_push["payout_dec"] - (1 + 100 / 110) * (1 + 100 / 120)) < 1e-9
    assert settle_parlay(legs, ["push", "push", "push"]) == {"result": "push", "pnl": 0.0, "payout_dec": 1.0}


# ---- profit step 2 Part A: prop legs carry the SHRUNK probability ---------------------------------------------------

def _shrunk_pick(game_pk, model_prob):
    return {"game_pk": game_pk, "player_id": f"00-{game_pk}", "player_name": "A.J. Brown", "market": "rec_yds",
            "side": "over", "line": 66.5, "model_prob": model_prob, "matchup": "M", "commence_time": "2026-10-04T17:00:00Z"}


def _shrunk_odds(game_pk):
    return {"game_pk": game_pk, "market": "reception_yds", "side": "over", "player_name": "AJ Brown", "book": "draftkings",
            "line": 66.5, "price": -110}


def test_prop_legs_take_the_shrunk_model_prob_from_the_board_row_unchanged():
    legs = assemble_prop_legs([_shrunk_pick(1, 0.5234)], [_shrunk_odds(1)])
    assert [l["prob"] for l in legs] == [0.5234]


def test_shrunk_prop_legs_below_the_055_floor_no_longer_build_a_ticket_but_strong_ones_still_do():
    games = (1, 2, 3)
    odds = [_shrunk_odds(g) for g in games]
    weak = assemble_prop_legs([_shrunk_pick(g, 0.52) for g in games], odds)
    assert build_best_parlays(weak) == []                                    # MIN_LEG_PROB stays 0.55: shrunk legs rarely qualify
    strong = assemble_prop_legs([_shrunk_pick(g, 0.62) for g in games], odds)
    [t] = build_best_parlays(strong)
    assert t["n_legs"] == 3 and abs(t["true_prob"] - 0.62 ** 3) < 1e-12
````

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q --continue-on-collection-errors tests/test_db_ev_prop_shrink_cols.py tests/test_build_ev_props_board.py tests/ledger/test_hooks.py tests/ledger/test_sources.py tests/serving/test_best_parlays.py`
Expected: `6 failed, 94 passed, 1 error`. The error is a `FileNotFoundError` for `db/migration_shrink_props.sql` (`test_db_ev_prop_shrink_cols.py` reads it at import). The six failures are the MLB board test (it now expects a price-only row), the four new board-script tests (the script has no `load_table` yet) and the ledger hook test (`mod.load_table` does not exist yet).

(The new parlay and ledger tests already pass: they pin existing behaviour.)

- [ ] **Step 3: Create the migration**

Create `db/migration_shrink_props.sql`:

````sql
-- =============================================================================
-- Shrink prop probabilities toward the market (profit step 2 Part A): two audit columns on ev_prop_picks.
-- =============================================================================
-- Idempotent and additive -- safe to re-run. Run this in the Supabase SQL Editor.
--
--   raw_model_prob : the pick side's model probability BEFORE shrinking (after MLB's calibration)
--   shrink_k       : the k applied (0 = price-only; the board uses k = 0 until the first fit has run)
--
-- model_prob keeps meaning "the number that made the pick" (the shrunk probability). The board writes the two new
-- columns only once they exist, so running this before or after the merge is safe. Views that select explicit columns
-- (ev_prop_picks_current and the rest) are unaffected.
ALTER TABLE ev_prop_picks ADD COLUMN IF NOT EXISTS raw_model_prob double precision, ADD COLUMN IF NOT EXISTS shrink_k double precision;
````

- [ ] **Step 4: Write the audit columns only when they exist**

In `src/sportsmodel/db.py` (edit 1 of 2), replace:

````python

_EV_PROP_DEFAULT_MODEL_VERSION = "props-sim-v1"


def upsert_ev_prop_picks(records: list[dict]) -> int:
    """Upsert +EV player-prop board rows into Supabase `ev_prop_picks`.
````

with:

````python

_EV_PROP_DEFAULT_MODEL_VERSION = "props-sim-v1"

# Columns added later by db/migration_shrink_props.sql (raw model probability before the shrink, and the k applied).
# Written only when the records carry them AND the table has them (probed once per upsert call via information_schema),
# so the board keeps working in the window before the migration is run.
_EV_PROP_PICKS_OPTIONAL_COLS = ["raw_model_prob", "shrink_k"]


def upsert_ev_prop_picks(records: list[dict]) -> int:
    """Upsert +EV player-prop board rows into Supabase `ev_prop_picks`.
````

In `src/sportsmodel/db.py` (edit 2 of 2), replace:

````python
    if not records:
        return 0
    key = ("game_pk", "player_id", "market", "line", "model_version")
    updates = ", ".join(
        f"{c} = EXCLUDED.{c}"
        for c in _EV_PROP_PICKS_COLS
        if c not in key and c not in _EV_PROP_PICKS_IMMUTABLE
    )
    placeholders = ", ".join(["%s"] * len(_EV_PROP_PICKS_COLS))
    sql = (
        f"INSERT INTO ev_prop_picks ({', '.join(_EV_PROP_PICKS_COLS)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT (game_pk, player_id, market, line, model_version) DO UPDATE SET {updates}"
    )
    rows = [
        tuple(
            r.get(c, _EV_PROP_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c)
            for c in _EV_PROP_PICKS_COLS
        )
        for r in records
    ]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
````

with:

````python
    if not records:
        return 0
    key = ("game_pk", "player_id", "market", "line", "model_version")
    wanted = [c for c in _EV_PROP_PICKS_OPTIONAL_COLS if any(c in r for r in records)]
    with get_postgres() as conn, conn.cursor() as cur:
        present = _existing_columns(cur, "ev_prop_picks", wanted) if wanted else set()
        missing = [c for c in wanted if c not in present]
        if missing:
            print(f"WARN ev_prop_picks has no {', '.join(missing)} column(s) -- run db/migration_shrink_props.sql; "
                  "board written without them")
        cols = _EV_PROP_PICKS_COLS + [c for c in wanted if c in present]
        updates = ", ".join(
            f"{c} = EXCLUDED.{c}"
            for c in cols
            if c not in key and c not in _EV_PROP_PICKS_IMMUTABLE
        )
        placeholders = ", ".join(["%s"] * len(cols))
        sql = (
            f"INSERT INTO ev_prop_picks ({', '.join(cols)}) "
            f"VALUES ({placeholders}) "
            f"ON CONFLICT (game_pk, player_id, market, line, model_version) DO UPDATE SET {updates}"
        )
        rows = [
            tuple(
                r.get(c, _EV_PROP_DEFAULT_MODEL_VERSION) if c == "model_version" else r.get(c)
                for c in cols
            )
            for r in records
        ]
        cur.executemany(sql, rows)
        conn.commit()
    return len(rows)
````

- [ ] **Step 5: Wire the table into the board script**

In `scripts/build_ev_props_board.py` (edit 1 of 2), replace:

````python
    upsert_nfl_prop_lines,
)
from sportsmodel.ledger.record import safe_record
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    always_propable,
````

with:

````python
    upsert_nfl_prop_lines,
)
from sportsmodel.ledger.record import safe_record
from sportsmodel.shrink.store import load_table
from sportsmodel.serving.props_ev import (
    MLB_SIM_TO_ODDS_MARKET,
    always_propable,
````

In `scripts/build_ev_props_board.py` (edit 2 of 2), replace:

````python
    if args.lines_only:
        return

    if is_mlb:
        rows = assemble_prop_rows(
            sim_rows, odds_rows, model_version, sport="mlb",
            market_map=MLB_SIM_TO_ODDS_MARKET, gate=always_propable,
            calibrate_fn=calibration.calibrate)
    else:
        rows = assemble_prop_rows(sim_rows, odds_rows, model_version)
    upsert_ev_prop_picks(rows)

    picks = [r for r in rows if r["is_pick"]]
````

with:

````python
    if args.lines_only:
        return

    # Profit step 2 Part A: every model probability that decides a pick is shrunk toward the market with the committed
    # k (assets/shrink_k.json). A missing or unreadable file, or a missing market entry, means k = 0 (price-only); it never
    # falls back to the raw model. MLB is always k = 0 in Part A.
    shrink = load_table()
    if is_mlb:
        rows = assemble_prop_rows(
            sim_rows, odds_rows, model_version, sport="mlb",
            market_map=MLB_SIM_TO_ODDS_MARKET, gate=always_propable,
            calibrate_fn=calibration.calibrate, shrink=shrink)
    else:
        rows = assemble_prop_rows(sim_rows, odds_rows, model_version, shrink=shrink)
    warning = shrink.warning()
    if warning:
        print(warning, flush=True)
    upsert_ev_prop_picks(rows)

    picks = [r for r in rows if r["is_pick"]]
````

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_db_ev_prop_shrink_cols.py tests/test_build_ev_props_board.py tests/test_db_ev_props.py tests/ledger tests/serving`
Expected: `542 passed`.

- [ ] **Step 7: Commit**

```bash
git add db/migration_shrink_props.sql src/sportsmodel/db.py scripts/build_ev_props_board.py tests/test_db_ev_prop_shrink_cols.py tests/test_build_ev_props_board.py tests/ledger/test_hooks.py tests/ledger/test_sources.py tests/serving/test_best_parlays.py
git commit -m "feat(shrink): optional audit columns, migration and board wiring (k=0 before the first fit)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The frozen pass test and the weekly report text (pure)

**Files:**
- Create: `src/sportsmodel/shrink/passtest.py`, `src/sportsmodel/shrink/report.py`
- Create: `tests/shrink/test_passtest.py`, `tests/shrink/test_report.py`

**Interfaces:**
- Consumes (Task 1): `fit.MIN_POOL_LEGS`, `fit.fit_table(legs, current_version)`, `fit.build_document(...)` (in the tests), `formula.shrink_prob`. A leg is `{season, week, game_pk, market, p, m, y, version}`.
- Produces:
  - `passtest.JUDGEMENT_LEGS = 2000`, `MIN_BRIER_GAIN = 0.001`, `SLOPE_LO = 0.8`, `SLOPE_HI = 1.2`, `BOOTSTRAP_RESAMPLES = 10_000`, `BOOTSTRAP_SEED = 20261013`, `LOWER_PERCENTILE = 2.5`, `CRITERIA` (three names)
  - `passtest.score_walk_forward(legs, current_version) -> list[dict]` (scored leg: `{season, week, game_pk, market, p, m, y, k, shrunk}`)
  - `passtest.brier_gain(scored) -> float | None`, `cluster_bootstrap_lower(scored, *, resamples=..., seed=..., percentile=...) -> float | None`, `calibration_slope(scored) -> float | None`
  - `passtest.evaluate(scored, *, resamples=...) -> dict | None` (`n, games, brier_market, brier_shrunk, gain, lower, slope`)
  - `passtest.verdict(gain, lower, slope) -> {"verdict": "PASS"|"FAIL", "criteria": [{name, passed, detail}]}`
  - `passtest.decide(scored, previous: dict | None, as_of: str, *, resamples=...) -> dict` (`status` `collecting` or `judged`; a previous `judged` decision is returned unchanged)
  - `report.render_report(*, as_of: datetime, doc: dict, decision: dict) -> str`, `report.parse_report(text) -> dict | None`, `report.latest_decision(reports: list[str], as_of: datetime) -> dict | None`

- [ ] **Step 1: Write the failing tests**

Create `tests/shrink/test_passtest.py`:

````python
"""The frozen shrink pass test: walk-forward scoring, the 300 earlier-leg rule, the 2,000 judgement point, both pass
conditions at their boundaries, the seeded game-clustered bootstrap and the frozen verdict. Pure."""
import random

import numpy as np
import pytest
from scipy.optimize import minimize

from sportsmodel.shrink import fit, passtest
from sportsmodel.shrink.formula import shrink_prob


def leg(week, game_pk, p=0.6, m=0.5, y=1, market="rush_yds", season=2026, version="v2"):
    return {"season": season, "week": week, "game_pk": game_pk, "market": market, "p": p, "m": m, "y": y, "version": version}


def informative(week, n, game0=0, k=0.4, market="rush_yds", season=2026):
    """n legs whose true k is exactly `k` (expected outcomes): p - m = 0.1, y = m + k * 0.1."""
    return [leg(week, game0 + i // 2, p=0.6, m=0.5, y=0.5 + k * 0.1, market=market, season=season) for i in range(n)]


def scored_leg(gp, m, y, shrunk):
    return {"season": 2026, "week": 9, "game_pk": gp, "market": "rush_yds", "p": shrunk, "m": m, "y": y, "k": 1.0, "shrunk": shrunk}


def test_the_frozen_constants_are_the_specs():
    assert passtest.JUDGEMENT_LEGS == 2000 and passtest.MIN_BRIER_GAIN == 0.001
    assert (passtest.SLOPE_LO, passtest.SLOPE_HI) == (0.8, 1.2)
    assert (passtest.BOOTSTRAP_RESAMPLES, passtest.BOOTSTRAP_SEED, passtest.LOWER_PERCENTILE) == (10_000, 20261013, 2.5)


# ---- walk-forward scoring ---------------------------------------------------------------------------------------

def test_a_week_is_scored_with_the_k_fitted_on_earlier_weeks_only():
    legs = informative(1, 200, k=0.2) + informative(2, 150, game0=1000, k=0.2) + informative(3, 40, game0=2000, k=0.9)
    scored = passtest.score_walk_forward(legs, "v2")
    assert {s["week"] for s in scored} == {3}                       # weeks 1 and 2 have < 300 earlier legs
    expected_k = fit.fit_table(legs[:350], "v2")["markets"]["nfl"]["rush_yds"]["k"]
    assert expected_k == pytest.approx(0.2) and all(s["k"] == pytest.approx(0.2) for s in scored)    # not 0.9: week 3 never trains itself
    assert scored[0]["shrunk"] == pytest.approx(shrink_prob(0.6, 0.5, expected_k))


def test_later_weeks_never_change_an_earlier_weeks_score():
    base = informative(1, 300, k=0.2) + informative(2, 50, game0=1000, k=0.2)
    before = passtest.score_walk_forward(base, "v2")
    after = passtest.score_walk_forward(base + informative(3, 400, game0=2000, k=0.95), "v2")
    assert [s for s in after if s["week"] == 2] == before and len(before) == 50


def test_the_300_earlier_leg_threshold_is_exact():
    short = passtest.score_walk_forward(informative(1, 299) + informative(2, 10, game0=900), "v2")
    assert short == []
    enough = passtest.score_walk_forward(informative(1, 300) + informative(2, 10, game0=900), "v2")
    assert len(enough) == 10 and {s["week"] for s in enough} == {2}


def test_every_week_after_the_threshold_is_scored_and_earlier_weeks_accumulate():
    legs = informative(1, 300) + informative(2, 20, game0=1000) + informative(3, 20, game0=2000)
    scored = passtest.score_walk_forward(legs, "v2")
    assert sorted({s["week"] for s in scored}) == [2, 3] and len(scored) == 40


def test_weeks_are_ordered_by_season_then_week_and_legs_may_arrive_out_of_order():
    old = informative(18, 300, season=2025)
    new = informative(1, 10, game0=900, season=2026)
    scored = passtest.score_walk_forward(new + old, "v2")                  # 2026 week 1 comes after 2025 week 18
    assert len(scored) == 10 and {(s["season"], s["week"]) for s in scored} == {(2026, 1)}
    assert passtest.score_walk_forward(old + new, "v2") == scored


def test_the_version_rule_applies_inside_each_walk_forward_fit():
    v1 = informative(1, 400, k=0.8)
    for l in v1:
        l["version"] = "v1"
    v2 = informative(1, 300, game0=5000, k=0.1)
    scored = passtest.score_walk_forward(v1 + v2 + informative(2, 5, game0=9000), "v2")
    assert {round(s["k"], 6) for s in scored} == {0.1}                     # 300 current-version legs exist, so v1 is ignored
    only_v1 = passtest.score_walk_forward(v1 + informative(2, 5, game0=9000), "v2")
    assert {round(s["k"], 6) for s in only_v1} == {0.8}                    # fewer than 300 current-version legs: all versions pooled


def test_a_market_with_no_earlier_legs_is_scored_at_the_market_probability():
    legs = informative(1, 400, market="rush_yds") + [leg(2, 9000, p=0.7, m=0.5, y=1, market="receptions")]
    [s] = passtest.score_walk_forward(legs, "v2")
    assert s["market"] == "receptions" and s["k"] == 0.0 and s["shrunk"] == pytest.approx(0.5)


def test_no_legs_and_too_few_legs_score_nothing():
    assert passtest.score_walk_forward([], "v2") == [] and passtest.score_walk_forward(informative(1, 50), "v2") == []


# ---- the paired Brier gain ----------------------------------------------------------------------------------------

def test_the_gain_is_market_brier_minus_shrunk_brier():
    sc = [scored_leg(1, m=0.5, y=1, shrunk=0.6), scored_leg(2, m=0.5, y=0, shrunk=0.4)]
    assert passtest.brier_gain(sc) == pytest.approx(0.25 - 0.16)
    worse = [scored_leg(1, m=0.5, y=1, shrunk=0.4)]
    assert passtest.brier_gain(worse) == pytest.approx(0.25 - 0.36) and passtest.brier_gain([]) is None


def test_k_zero_scores_exactly_zero_gain_so_it_can_never_pass():
    sc = [scored_leg(i, m=0.55, y=i % 2, shrunk=0.55) for i in range(50)]
    assert passtest.brier_gain(sc) == 0.0
    assert passtest.verdict(0.0, 0.0, 1.0)["verdict"] == "FAIL"


# ---- the clustered bootstrap --------------------------------------------------------------------------------------

def _mixed(n_games, per_game, seed=3):
    rng = random.Random(seed)
    out = []
    for g in range(n_games):
        bias = rng.uniform(-0.05, 0.08)                                    # one shared shock per game: the legs inside are correlated
        for _ in range(per_game):
            out.append(scored_leg(g, m=0.5, y=1 if rng.random() < 0.5 + bias else 0, shrunk=0.5 + 0.3 * bias))
    return out


def test_the_bootstrap_is_deterministic_for_the_frozen_seed():
    sc = _mixed(60, 4)
    assert passtest.cluster_bootstrap_lower(sc) == passtest.cluster_bootstrap_lower(list(reversed(sc)))
    assert passtest.cluster_bootstrap_lower(sc, seed=1) != passtest.cluster_bootstrap_lower(sc)


def test_the_bootstrap_matches_a_hand_rolled_resample_of_games_with_the_frozen_seed():
    sc = _mixed(30, 3)
    gains = {}
    for s, g in zip(sc, passtest._gains(sc)):
        gains.setdefault(s["game_pk"], []).append(g)
    games = sorted(gains, key=str)
    rng = np.random.default_rng(20261013)
    means = []
    for _ in range(10):
        idx = rng.integers(0, len(games), size=(1000, len(games)))
        for row in idx:
            vals = [x for i in row for x in gains[games[i]]]
            means.append(sum(vals) / len(vals))
    assert passtest.cluster_bootstrap_lower(sc) == pytest.approx(float(np.percentile(means, 2.5)), abs=1e-12)


def test_resampling_whole_games_widens_the_interval_versus_resampling_legs():
    rng = random.Random(9)
    sc = []
    for g in range(40):
        y = 1 if rng.random() < 0.5 else 0                                  # every leg of a game shares one outcome
        sc += [scored_leg(g, m=0.5, y=y, shrunk=0.56) for _ in range(25)]
    clustered = passtest.cluster_bootstrap_lower(sc)
    one_leg_per_game = [dict(s, game_pk=i) for i, s in enumerate(sc)]       # pretend every leg is its own game
    assert clustered < passtest.cluster_bootstrap_lower(one_leg_per_game) - 0.01


def test_a_single_game_has_a_degenerate_interval_equal_to_its_mean_gain():
    sc = [scored_leg(7, m=0.5, y=1, shrunk=0.6), scored_leg(7, m=0.5, y=0, shrunk=0.45)]
    assert passtest.cluster_bootstrap_lower(sc) == pytest.approx(passtest.brier_gain(sc))
    assert passtest.cluster_bootstrap_lower([]) is None


# ---- the calibration slope ----------------------------------------------------------------------------------------

def _scipy_slope(sc):
    x = np.log(np.array([s["shrunk"] for s in sc]) / (1 - np.array([s["shrunk"] for s in sc])))
    y = np.array([s["y"] for s in sc], dtype=float)

    def nll(b):
        z = b[0] + b[1] * x
        return float(np.sum(np.logaddexp(0, z) - y * z))
    return float(minimize(nll, [0.0, 1.0], method="BFGS", options={"gtol": 1e-10}).x[1])


def test_the_slope_matches_an_independent_maximum_likelihood_fit():
    rng = random.Random(5)
    sc = []
    for i in range(3000):
        shrunk = rng.uniform(0.3, 0.7)
        true = 1 / (1 + np.exp(-(0.1 + 1.4 * np.log(shrunk / (1 - shrunk)))))
        sc.append(scored_leg(i, m=0.5, y=1 if rng.random() < true else 0, shrunk=shrunk))
    assert passtest.calibration_slope(sc) == pytest.approx(_scipy_slope(sc), abs=1e-4)
    assert 1.0 < passtest.calibration_slope(sc) < 1.9


def test_a_calibrated_forecast_has_slope_near_one():
    rng = random.Random(11)
    sc = [scored_leg(i, 0.5, 0, 0.5) | {"shrunk": s} for i, s in enumerate(rng.uniform(0.2, 0.8) for _ in range(6000))]
    for s in sc:
        s["y"] = 1 if rng.random() < s["shrunk"] else 0
    assert passtest.calibration_slope(sc) == pytest.approx(1.0, abs=0.2)


def test_the_slope_is_none_when_it_is_undefined():
    assert passtest.calibration_slope([scored_leg(i, 0.5, i % 2, 0.5) for i in range(20)]) is None          # no variance
    assert passtest.calibration_slope([scored_leg(1, 0.5, 1, 0.6)]) is None                                  # too few
    separated = [scored_leg(i, 0.5, 1 if i >= 10 else 0, 0.2 if i < 10 else 0.8) for i in range(20)]
    assert passtest.calibration_slope(separated) is None                                                     # perfect separation


# ---- the two pass conditions at their boundaries --------------------------------------------------------------------

@pytest.mark.parametrize("gain,lower,slope,expected", [
    (0.001, 1e-9, 1.0, "PASS"),              # a gain of exactly 0.001 passes
    (0.00099, 1e-9, 1.0, "FAIL"),
    (0.01, 0.0, 1.0, "FAIL"),                # the lower bound must be strictly above 0
    (0.01, -1e-9, 1.0, "FAIL"),
    (0.01, 1e-9, 0.8, "PASS"),               # slope bounds are inclusive
    (0.01, 1e-9, 1.2, "PASS"),
    (0.01, 1e-9, 0.7999, "FAIL"),
    (0.01, 1e-9, 1.2001, "FAIL"),
    (0.01, 1e-9, None, "FAIL"),              # an undefined slope cannot pass
    (None, None, None, "FAIL"),
])
def test_verdict_boundaries(gain, lower, slope, expected):
    assert passtest.verdict(gain, lower, slope)["verdict"] == expected


def test_a_gain_under_the_threshold_fails_even_with_perfect_calibration_and_a_positive_bound():
    v = passtest.verdict(0.0005, 0.0001, 1.0)
    assert v["verdict"] == "FAIL" and [c["passed"] for c in v["criteria"]] == [False, True, True]


# ---- the judgement point and the frozen verdict ---------------------------------------------------------------------

def _scored(n, per_game=2, edge=0.03):
    """n scored legs where the shrunk probability genuinely beats the market (a clear PASS if judged)."""
    rng = random.Random(21)
    out = []
    for i in range(n):
        true = rng.uniform(0.4, 0.6)
        shrunk = true
        out.append(scored_leg(i // per_game, m=true - edge * (1 if i % 2 else -1), y=1 if rng.random() < true else 0, shrunk=shrunk))
    return out


def test_1999_scored_legs_are_still_collecting_and_2000_are_judged():
    d = passtest.decide(_scored(1999), None, "2026-12-01", resamples=200)
    assert d["status"] == "collecting" and d["scored"] == 1999 and d["need_scored"] == 2000 and "verdict" not in d
    j = passtest.decide(_scored(2000), None, "2026-12-08", resamples=200)
    assert j["status"] == "judged" and j["judged_on"] == "2026-12-08" and j["verdict"] in ("PASS", "FAIL")
    assert [c["name"] for c in j["criteria"]] == list(passtest.CRITERIA)


def test_collecting_reports_running_numbers_but_cannot_pass_or_fail():
    d = passtest.decide(_scored(300), None, "2026-12-01", resamples=200)
    assert d["status"] == "collecting" and d["running"]["n"] == 300 and d["running"]["gain"] is not None
    assert passtest.decide([], None, "2026-12-01")["running"] is None


def test_a_judged_verdict_is_carried_forward_unchanged_whatever_the_new_data_say():
    frozen = {"status": "judged", "judged_on": "2026-11-17", "scored": 2100, "need_scored": 2000, "running": None,
              "verdict": "FAIL", "criteria": []}
    assert passtest.decide(_scored(4000), frozen, "2027-01-05", resamples=200) is frozen
    assert passtest.decide([], frozen, "2027-01-05") is frozen


def test_a_previous_collecting_decision_does_not_freeze_anything():
    prev = {"status": "collecting", "scored": 10, "need_scored": 2000, "running": None}
    assert passtest.decide(_scored(2000), prev, "2026-12-08", resamples=200)["status"] == "judged"


def test_evaluate_returns_both_briers_the_gain_the_bound_and_the_slope():
    ev = passtest.evaluate(_scored(2000), resamples=300)
    assert ev["n"] == 2000 and ev["games"] == 1000
    assert ev["gain"] == pytest.approx(ev["brier_market"] - ev["brier_shrunk"])
    assert ev["lower"] < ev["gain"] and ev["slope"] is not None
````

and `tests/shrink/test_report.py`:

````python
"""The weekly shrink report text and its machine-readable decision block. Pure."""
from datetime import datetime, timedelta, timezone

import pytest

from sportsmodel.shrink import fit, passtest, report

AS_OF = datetime(2026, 11, 3, 15, 30, tzinfo=timezone.utc)


def doc(n=400):
    legs = [{"season": 2026, "week": 4, "game_pk": i, "market": "rush_yds", "p": 0.6, "m": 0.5, "y": 0.54, "version": "v2"}
            for i in range(n)]
    return fit.build_document(legs, "v2", "2026-11-03T15:30:00+00:00")


COLLECTING = {"status": "collecting", "scored": 640, "need_scored": 2000,
              "running": {"n": 640, "games": 320, "brier_market": 0.2501, "brier_shrunk": 0.2499, "gain": 0.0002,
                          "lower": -0.0005, "slope": 0.91}}
JUDGED = {"status": "judged", "judged_on": "2026-12-08", "scored": 2010, "need_scored": 2000, "running": COLLECTING["running"],
          "verdict": "FAIL", "criteria": [{"name": n, "passed": ok, "detail": "x"} for n, ok in zip(passtest.CRITERIA, (True, False, True))]}


def test_a_collecting_report_shows_k_per_market_progress_and_the_running_numbers():
    text = report.render_report(as_of=AS_OF, doc=doc(), decision=COLLECTING)
    assert text.startswith("# Shrink prop probabilities: report of 2026-11-03")
    assert "COLLECTING" in text and "640 of 2000" in text
    assert "| pooled | 0.400 | 400 | 0.400 |" in text and "| rush_yds | 0.400 | 400 | 0.400 |" in text
    assert "| pass_tds | 0.000 | 0 | n/a |" in text
    assert "gain +0.00020" in text and "slope 0.910" in text and "lower bound -0.00050" in text
    assert "model version `v2`" in text and "season 2026 week 4 to season 2026 week 4" in text


def test_a_judged_report_names_the_verdict_and_says_it_is_frozen():
    text = report.render_report(as_of=AS_OF, doc=doc(), decision=JUDGED)
    assert "JUDGED on 2026-12-08: FAIL" in text and "informational only" in text
    assert "MISSED" in text and "the user and Claude decide together" in text


def test_an_empty_fit_still_renders():
    text = report.render_report(as_of=AS_OF, doc=fit.build_document([], None, "t"), decision=passtest.decide([], None, "2026-11-03"))
    assert "no legs yet" in text and "0 of 2000" in text


def test_the_decision_block_round_trips():
    block = report.parse_report(report.render_report(as_of=AS_OF, doc=doc(), decision=JUDGED))
    assert block["decision"] == JUDGED and block["as_of"] == AS_OF.isoformat()


@pytest.mark.parametrize("text", ["", None, "no block here", "<!-- shrink-gate: {not json} -->"])
def test_garbage_has_no_block(text):
    assert report.parse_report(text) is None


def _rep(days_from_now, decision):
    return report.render_report(as_of=AS_OF + timedelta(days=days_from_now), doc=doc(), decision=decision)


def test_latest_decision_prefers_the_earliest_judged_report_before_the_date():
    early, later = dict(JUDGED, judged_on="2026-12-08"), dict(JUDGED, judged_on="2026-12-15", verdict="PASS")
    reports = [_rep(35, early), _rep(42, later), _rep(7, COLLECTING)]
    assert report.latest_decision(reports, AS_OF + timedelta(days=60)) == early


def test_latest_decision_ignores_reports_on_or_after_the_date_and_falls_back_to_the_latest_collecting_one():
    reports = [_rep(7, COLLECTING), _rep(14, dict(COLLECTING, scored=900))]
    assert report.latest_decision(reports, AS_OF + timedelta(days=14))["scored"] == 640        # the day-14 report is not "before"
    assert report.latest_decision(reports, AS_OF + timedelta(days=30))["scored"] == 900
    assert report.latest_decision([], AS_OF) is None and report.latest_decision(["junk"], AS_OF) is None


def test_a_corrupt_newer_report_does_not_unfreeze_an_older_judged_verdict():
    reports = [_rep(35, JUDGED), "<!-- shrink-gate: {broken -->"]
    assert report.latest_decision(reports, AS_OF + timedelta(days=90)) == JUDGED
````

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q tests/shrink/test_passtest.py tests/shrink/test_report.py`
Expected: `2 errors` during collection, `ImportError: cannot import name 'passtest' from 'sportsmodel.shrink'`.

- [ ] **Step 3: Write the pass test**

Create `src/sportsmodel/shrink/passtest.py`:

````python
"""The pre-registered pass test for the prop shrink. PURE (numpy only) -- no IO.

FROZEN as of docs/superpowers/specs/2026-10-13-shrink-props-design.md section 3: nothing here may change once results
exist. Evaluation set: each graded NFL lean in week W (keyed by (season, week)) is scored with the shrunk probability,
using the k fitted by the section 1 rules on legs graded in weeks BEFORE W only; weeks with fewer than 300 earlier legs are
not scored; pushes are excluded. Judged ONCE, at the first weekly report with at least 2,000 scored legs (before that the
status is "collecting"). It passes only if BOTH hold:
  1. Brier(market) - Brier(shrunk) >= 0.001, and the paired game-clustered bootstrap 95% CI of that difference has its
     lower bound > 0 (resamples games with replacement: 10,000 resamples, seed 20261013, 2.5th percentile)
  2. the logistic regression y ~ a + b * logit(shrunk) has slope b in [0.8, 1.2]
A pass means the model adds information beyond the board's market price; it does not prove profit (CLV does).
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from .fit import MIN_POOL_LEGS, fit_table
from .formula import shrink_prob

JUDGEMENT_LEGS = 2000
MIN_BRIER_GAIN = 0.001
SLOPE_LO, SLOPE_HI = 0.8, 1.2
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261013
LOWER_PERCENTILE = 2.5
CHUNK = 1000                                  # resamples drawn per block: bounds memory, fixed so results are reproducible
EPS = 1e-12                                   # float noise only: a gain of exactly 0.001 or a slope of exactly 0.8 passes
CRITERIA = ("Brier gain over the market >= 0.001", "paired game-clustered 95% lower bound > 0", "calibration slope in [0.8, 1.2]")


def week_key(leg: dict) -> tuple[int, int]:
    return (leg["season"], leg["week"])


def score_walk_forward(legs: list[dict], current_version: str | None) -> list[dict]:
    """Every leg of every week that has at least 300 legs in EARLIER weeks, with the shrunk probability from the k fitted
    on those earlier legs alone (the section 1 rules, including the model-version fallback). A market with no earlier legs
    has k = 0, so its shrunk probability is the market's."""
    by_week: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for leg in legs:
        by_week[week_key(leg)].append(leg)
    scored = []
    for wk in sorted(by_week):
        earlier = [l for l in legs if week_key(l) < wk]
        if len(earlier) < MIN_POOL_LEGS:
            continue
        ks = {m: e["k"] for m, e in fit_table(earlier, current_version)["markets"]["nfl"].items()}
        for l in by_week[wk]:
            k = ks.get(l["market"], 0.0)
            scored.append({"season": l["season"], "week": l["week"], "game_pk": l["game_pk"], "market": l["market"],
                           "p": l["p"], "m": l["m"], "y": l["y"], "k": k, "shrunk": shrink_prob(l["p"], l["m"], k)})
    return scored


def _gains(scored: list[dict]) -> list[float]:
    """Per-leg Brier(market) - Brier(shrunk): positive where the shrunk probability beat the market's."""
    return [(s["m"] - s["y"]) ** 2 - (s["shrunk"] - s["y"]) ** 2 for s in scored]


def brier_gain(scored: list[dict]) -> float | None:
    g = _gains(scored)
    return sum(g) / len(g) if g else None


def cluster_bootstrap_lower(scored: list[dict], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
                            percentile: float = LOWER_PERCENTILE) -> float | None:
    """Lower bound of the paired Brier gain: resample GAMES with replacement (all legs of a drawn game come along), take
    each resample's leg-weighted mean gain, return the `percentile`-th percentile. Deterministic for a given seed."""
    per_game: dict = defaultdict(list)
    for s, g in zip(scored, _gains(scored)):
        per_game[s["game_pk"]].append(g)
    if not per_game:
        return None
    games = sorted(per_game, key=str)
    sums = np.array([sum(per_game[g]) for g in games])
    counts = np.array([len(per_game[g]) for g in games], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, resamples, CHUNK):
        idx = rng.integers(0, len(games), size=(min(CHUNK, resamples - start), len(games)))
        means.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    return float(np.percentile(np.concatenate(means), percentile))


def calibration_slope(scored: list[dict]) -> float | None:
    """b in the logistic regression y ~ a + b * logit(shrunk), by Newton-Raphson (IRLS). None when it is undefined: no
    variance in the predictor, a singular fit, or no convergence (for example perfect separation)."""
    if len(scored) < 3:
        return None
    s = np.array([r["shrunk"] for r in scored], dtype=float)
    y = np.array([r["y"] for r in scored], dtype=float)
    x = np.log(s / (1.0 - s))
    if float(np.ptp(x)) < 1e-9:
        return None
    X = np.column_stack([np.ones_like(x), x])
    beta = np.zeros(2)
    for _ in range(100):
        mu = 1.0 / (1.0 + np.exp(-np.clip(X @ beta, -35.0, 35.0)))
        w = mu * (1.0 - mu)
        try:
            step = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (y - mu))
        except np.linalg.LinAlgError:
            return None
        beta = beta + step
        if not np.all(np.isfinite(beta)) or float(np.max(np.abs(beta))) > 50.0:
            return None
        if float(np.max(np.abs(step))) < 1e-10:
            return float(beta[1])
    return None


def evaluate(scored: list[dict], *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict | None:
    """The running numbers: n, games, both Brier scores, the gain, its clustered lower bound and the slope."""
    if not scored:
        return None
    n = len(scored)
    return {
        "n": n, "games": len({s["game_pk"] for s in scored}),
        "brier_market": sum((s["m"] - s["y"]) ** 2 for s in scored) / n,
        "brier_shrunk": sum((s["shrunk"] - s["y"]) ** 2 for s in scored) / n,
        "gain": brier_gain(scored), "lower": cluster_bootstrap_lower(scored, resamples=resamples),
        "slope": calibration_slope(scored),
    }


def verdict(gain: float | None, lower: float | None, slope: float | None) -> dict:
    """PASS only if the gain is >= 0.001 AND its clustered lower bound is > 0 AND the slope is in [0.8, 1.2]."""
    checks = [
        (gain is not None and gain + EPS >= MIN_BRIER_GAIN,
         "n/a" if gain is None else f"{gain:+.5f} (need >= {MIN_BRIER_GAIN:+.3f})"),
        (lower is not None and lower > 0, "n/a" if lower is None else f"{lower:+.5f} (need > 0)"),
        (slope is not None and SLOPE_LO - EPS <= slope <= SLOPE_HI + EPS,
         "undefined" if slope is None else f"{slope:.3f} (need {SLOPE_LO} to {SLOPE_HI})"),
    ]
    criteria = [{"name": n, "passed": bool(ok), "detail": d} for n, (ok, d) in zip(CRITERIA, checks)]
    return {"verdict": "PASS" if all(c["passed"] for c in criteria) else "FAIL", "criteria": criteria}


def decide(scored: list[dict], previous: dict | None, as_of: str, *, resamples: int = BOOTSTRAP_RESAMPLES) -> dict:
    """The decision block of this week's report. Once a previous report judged the test the verdict is carried forward
    untouched (the test is judged ONCE); otherwise judge now if there are at least 2,000 scored legs, else 'collecting'."""
    if previous and previous.get("status") == "judged":
        return previous
    running = evaluate(scored, resamples=resamples)
    base = {"scored": len(scored), "need_scored": JUDGEMENT_LEGS, "running": running}
    if len(scored) < JUDGEMENT_LEGS:
        return {"status": "collecting", **base}
    v = verdict(running["gain"], running["lower"], running["slope"])
    return {"status": "judged", "judged_on": as_of, **base, **v}
````

- [ ] **Step 4: Write the report text**

Create `src/sportsmodel/shrink/report.py`:

````python
"""The weekly shrink report (docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md). PURE -- text only.

Shows the current k per market with n, the scored legs so far, the running Brier gain and calibration slope, and the
status of the pre-registered pass test (sportsmodel.shrink.passtest). The decision block is machine-readable (an HTML
comment): once a report says `judged`, every later report copies that verdict unchanged.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone

from . import passtest

MARKER = "shrink-gate"
_BLOCK = re.compile(rf"<!-- {MARKER}: (\{{.*\}}) -->")


def _utc(x) -> datetime | None:
    if x is None:
        return None
    if isinstance(x, str):
        try:
            x = datetime.fromisoformat(x.replace("Z", "+00:00"))
        except ValueError:
            return None
    return x.replace(tzinfo=timezone.utc) if x.tzinfo is None else x.astimezone(timezone.utc)


def _fmt(x, spec="{:+.5f}") -> str:
    return "n/a" if x is None else spec.format(x)


def _status_md(decision: dict) -> str:
    running = decision.get("running")
    scored, need = decision["scored"], decision["need_scored"]
    if decision["status"] == "collecting":
        head = (f"**Status: COLLECTING.** {scored} of {need} scored legs. The pass test is judged once, at the first weekly "
                "report with enough scored legs; until then nothing can pass or fail.")
    else:
        rows = "\n".join(f"- {c['name']}: {'ok' if c['passed'] else 'MISSED'} ({c['detail']})" for c in decision["criteria"])
        head = (f"**Status: JUDGED on {decision['judged_on']}: {decision['verdict']}.** Frozen: later weeks below are "
                f"informational only.\n\n{rows}")
    if running:
        head += (f"\n\nRunning numbers on {running['n']} scored legs in {running['games']} games: Brier(market) "
                 f"{running['brier_market']:.5f}, Brier(shrunk) {running['brier_shrunk']:.5f}, gain {_fmt(running['gain'])}, "
                 f"clustered 95% lower bound {_fmt(running['lower'])}, calibration slope {_fmt(running['slope'], '{:.3f}')}.")
    meaning = ("A pass means the shrunk model adds information beyond the board's market price; it does not prove profit "
               "(CLV in the pick ledger judges that). A fail means the user and Claude decide together what happens to "
               "model prop picks; until then picks keep using the shrunk probabilities.")
    return head + "\n\n" + meaning


def render_report(*, as_of: datetime, doc: dict, decision: dict) -> str:
    """`doc` is the committed assets/shrink_k.json contents (sportsmodel.shrink.fit.build_document)."""
    as_of = _utc(as_of)
    markets = doc["markets"]["nfl"]
    rows = "\n".join(f"| {m} | {e['k']:.3f} | {e['n']} | {_fmt(e['k_raw'], '{:.3f}')} |" for m, e in sorted(markets.items()))
    pooled, window = doc["pooled"], doc["data_window"]
    span = "no legs yet" if not window["first"] else f"season {window['first'][0]} week {window['first'][1]} to season {window['last'][0]} week {window['last'][1]}"
    block = {"as_of": as_of.isoformat(), "decision": decision}
    return f"""# Shrink prop probabilities: report of {as_of:%Y-%m-%d}

Generated by `scripts/fit_shrink_k.py` (weekly refit; the board reads `assets/shrink_k.json`).

{_status_md(decision)}

## Current k (NFL)

Fitted on {pooled['n']} legs of model version `{doc['model_version_used']}` (current version `{doc['current_version']}`; "all" means every version was pooled because the current one had fewer than 300 legs). Data window: {span}; {window['n_legs']} eligible legs in all.

| Market | k | n | k_raw |
|---|---|---|---|
| pooled | {pooled['k']:.3f} | {pooled['n']} | {_fmt(pooled['k_raw'], '{:.3f}')} |
{rows}

A market with k = 0 is priced at the market's probability only. A pooled sample under 300 legs gives k = 0 everywhere. MLB is k = 0 in this part.

<!-- {MARKER}: {json.dumps(block, sort_keys=True)} -->
"""


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
    """The decision to carry into the report of `as_of`: the EARLIEST judged decision among the reports strictly before
    `as_of` (the test is judged once, so a verdict never moves), else the most recent such report's decision, else None."""
    prior = []
    for text in reports:
        p = parse_report(text)
        if not p or not isinstance(p.get("decision"), dict):
            continue
        at = _utc(p.get("as_of"))
        if at is not None and at < _utc(as_of):
            prior.append((at, p["decision"]))
    prior.sort(key=lambda x: x[0])
    judged = [d for _, d in prior if d.get("status") == "judged"]
    return judged[0] if judged else (prior[-1][1] if prior else None)
````

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/shrink`
Expected: `140 passed` (34 in `test_passtest.py`, 11 in `test_report.py`, plus Tasks 1 and 2).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/shrink/passtest.py src/sportsmodel/shrink/report.py tests/shrink/test_passtest.py tests/shrink/test_report.py
git commit -m "feat(shrink): frozen walk-forward pass test and weekly report text" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The weekly fit script and workflow

**Files:**
- Create: `scripts/fit_shrink_k.py`, `.github/workflows/fit-shrink-k.yml`
- Create: `tests/scripts/test_fit_shrink_k.py`, `tests/test_workflows_shrink.py`

**Interfaces:**
- Consumes (Tasks 1, 2, 4): `fit.legs_from_grades`, `fit.build_document`, `fit.has_fewer_legs`, `passtest.score_walk_forward`, `passtest.decide`, `report.render_report`, `report.latest_decision`, `store.SHRINK_K_PATH`; existing `ledger.record.writes_disabled()`, `db.get_postgres()`.
- Produces:
  - `scripts/fit_shrink_k.py`: `main(argv=None) -> int` with `--as-of`, `--out` (default `assets/shrink_k.json`), `--reports-dir`, `--dry-run`; `collect(q) -> (grade rows, served version | None)`; `build(rows, current_version, as_of, previous) -> (document, report text, decision)`; `write_atomic(path, text)`; `GRADES_SQL`, `SERVING_SQL`; exit 0 under `SM_LEDGER_OFF` (refuses), 1 on a missing `DATABASE_URL`, a fetch or fit failure, or a truncated read
  - `fit-shrink-k.yml`: Tuesdays 15:30 UTC plus `workflow_dispatch`; commits `assets/shrink_k.json` and `docs/superpowers/reports/*-shrink-gate.md`

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_fit_shrink_k.py`:

````python
"""scripts/fit_shrink_k.py: the fit and report from injected rows, the write-only-on-success rule, the guards. No database."""
import importlib.util
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from sportsmodel import db
from sportsmodel.shrink import report
from sportsmodel.shrink.store import ShrinkTable, load_table

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("fit_shrink_k", ROOT / "scripts" / "fit_shrink_k.py")
fs = importlib.util.module_from_spec(_spec)
sys.modules["fit_shrink_k"] = fs
_spec.loader.exec_module(fs)

AS_OF = datetime(2026, 10, 20, 15, 30, tzinfo=timezone.utc)


def grade(i, week, market="rush_yds", lean="over", result="hit", version="nfl-sim-ml-v2", p=0.6, over=-110, under=-110):
    return {"game_pk": 100 + i // 3, "player_id": f"00-{i}", "market": market, "line": 54.5, "over_price": over,
            "under_price": under, "p_over": p, "lean": lean, "result": result, "season": 2026, "week": week,
            "model_version": version}


def slate(weeks=(1, 2, 3, 4), per_week=120):
    """Alternating hit / miss leans: the model carries no information, so the fitted k is small but the machinery runs."""
    return [grade(w * 1000 + i, w, result="hit" if i % 2 else "miss") for w in weeks for i in range(per_week)]


def fake_q(rows, served="nfl-sim-ml-v2"):
    def q(sql, params=()):
        return rows if "nfl_prop_grades" in sql else ([{"model_version": served}] if served else [])
    return q


def run_main(monkeypatch, tmp_path, rows, *extra, served="nfl-sim-ml-v2", q=None):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgres://unused")
    monkeypatch.setattr(fs, "_db_query", lambda: q or fake_q(rows, served))
    out, reports = tmp_path / "shrink_k.json", tmp_path / "reports"
    reports.mkdir(exist_ok=True)
    code = fs.main(["--as-of", AS_OF.isoformat(), "--out", str(out), "--reports-dir", str(reports), *extra])
    return code, out, reports


# ---- the happy path -----------------------------------------------------------------------------------------------

def test_a_successful_run_writes_the_k_file_and_the_dated_report(monkeypatch, tmp_path, capsys):
    code, out, reports = run_main(monkeypatch, tmp_path, slate())
    assert code == 0
    doc = json.loads(out.read_text())
    assert doc["fitted_at"] == AS_OF.isoformat() and doc["current_version"] == "nfl-sim-ml-v2"
    assert doc["pooled"]["n"] == 480 and doc["data_window"] == {"first": [2026, 1], "last": [2026, 4], "n_legs": 480}
    assert doc["markets"]["nfl"]["rush_yds"]["n"] == 480 and all(e["k"] == 0.0 for e in doc["markets"]["mlb"].values())
    (rep,) = reports.glob("*-shrink-gate.md")
    assert rep.name == "2026-10-20-shrink-gate.md" and report.parse_report(rep.read_text())["decision"]["status"] == "collecting"
    assert "wrote shrink_k.json" in capsys.readouterr().out
    assert not list(tmp_path.glob("*.tmp"))


def test_the_written_file_is_read_back_by_the_board_loader(monkeypatch, tmp_path):
    rows = [grade(i, 1 + i // 100, lean="over", result="hit" if i % 5 else "miss", p=0.62) for i in range(500)]
    code, out, _ = run_main(monkeypatch, tmp_path, rows)
    table = load_table(out)
    assert code == 0 and table.problem is None
    assert 0.0 <= table.k("nfl", "rush_yds") <= 1.0 and table.k("mlb", "total_bases") == 0.0
    assert table.k("nfl", "pass_tds") == 0.0 and table.warning() is None      # a market with no legs is an explicit k = 0 entry: no warning


def test_a_pooled_sample_under_300_legs_writes_an_all_zero_file(monkeypatch, tmp_path):
    code, out, _ = run_main(monkeypatch, tmp_path, slate(weeks=(1,), per_week=250))
    doc = json.loads(out.read_text())
    assert code == 0 and doc["pooled"]["n"] == 250 and doc["pooled"]["k"] == 0.0
    assert all(e["k"] == 0.0 for e in doc["markets"]["nfl"].values())


def test_the_current_version_comes_from_the_serving_table_and_falls_back_to_all_versions(monkeypatch, tmp_path):
    rows = [grade(i, 1 + i // 200, version="nfl-sim-ml-v1") for i in range(600)]
    code, out, _ = run_main(monkeypatch, tmp_path, rows, served="nfl-sim-ml-v2")
    assert json.loads(out.read_text())["model_version_used"] == "all"
    code, out2, _ = run_main(monkeypatch, tmp_path, rows, served="nfl-sim-ml-v1")
    assert json.loads(out2.read_text())["model_version_used"] == "nfl-sim-ml-v1"


def test_pushes_invalid_prices_and_unserved_markets_do_not_count(monkeypatch, tmp_path):
    rows = slate(weeks=(1, 2), per_week=100) + [grade(9001, 2, result="push"), grade(9002, 2, over=50),
                                                grade(9003, 2, under=None), grade(9004, 2, market="pass_yds"),
                                                grade(9005, 2, market="rush_att")]
    code, out, _ = run_main(monkeypatch, tmp_path, rows)
    assert json.loads(out.read_text())["data_window"]["n_legs"] == 200


def test_dry_run_prints_and_writes_nothing(monkeypatch, tmp_path, capsys):
    code, out, reports = run_main(monkeypatch, tmp_path, slate(), "--dry-run")
    text = capsys.readouterr().out
    assert code == 0 and not out.exists() and list(reports.iterdir()) == []
    assert '"pooled"' in text and "# Shrink prop probabilities: report of 2026-10-20" in text


# ---- never writes a k it did not compute ---------------------------------------------------------------------------

def test_a_fetch_failure_exits_nonzero_and_leaves_the_existing_file_and_reports_untouched(monkeypatch, tmp_path, capsys):
    out = tmp_path / "shrink_k.json"
    out.write_text('{"keep": "me"}\n')

    def boom(sql, params=()):
        raise ConnectionError("db down")
    code, out, reports = run_main(monkeypatch, tmp_path, [], q=boom)
    assert code == 1 and out.read_text() == '{"keep": "me"}\n' and list(reports.iterdir()) == []
    assert "::error::fit-shrink-k: ConnectionError" in capsys.readouterr().out


def test_a_failure_while_serving_lookup_also_keeps_the_file(monkeypatch, tmp_path):
    out = tmp_path / "shrink_k.json"
    out.write_text("old")

    def half(sql, params=()):
        if "nfl_sim_serving" in sql:
            raise RuntimeError('relation "nfl_sim_serving" does not exist')
        return slate()
    code, out, reports = run_main(monkeypatch, tmp_path, [], q=half)
    assert code == 1 and out.read_text() == "old" and list(reports.iterdir()) == []


def test_a_fit_failure_keeps_the_file(monkeypatch, tmp_path):
    out = tmp_path / "shrink_k.json"
    out.write_text("old")
    monkeypatch.setattr(fs.fit, "build_document", lambda *a, **k: (_ for _ in ()).throw(ValueError("bad fit")))
    code, out, reports = run_main(monkeypatch, tmp_path, slate())
    assert code == 1 and out.read_text() == "old" and list(reports.iterdir()) == []


def test_a_read_with_fewer_legs_than_the_committed_file_is_refused(monkeypatch, tmp_path, capsys):
    code, out, reports = run_main(monkeypatch, tmp_path, slate())
    first = out.read_text()
    (reports / "2026-10-20-shrink-gate.md").unlink()
    code2, out2, _ = run_main(monkeypatch, tmp_path, slate(weeks=(1, 2), per_week=120))
    assert code == 0 and code2 == 1 and out2.read_text() == first and list(reports.iterdir()) == []
    assert "truncated" in capsys.readouterr().out


def test_a_read_with_more_legs_replaces_the_file(monkeypatch, tmp_path):
    run_main(monkeypatch, tmp_path, slate(weeks=(1, 2), per_week=120))
    code, out, _ = run_main(monkeypatch, tmp_path, slate())
    assert code == 0 and json.loads(out.read_text())["data_window"]["n_legs"] == 480


def test_write_atomic_replaces_without_leaving_a_temp_file(tmp_path):
    path = tmp_path / "sub" / "shrink_k.json"
    fs.write_atomic(path, "one")
    fs.write_atomic(path, "two")
    assert path.read_text() == "two" and [p.name for p in path.parent.iterdir()] == ["shrink_k.json"]


# ---- the frozen verdict -------------------------------------------------------------------------------------------

def test_a_previously_judged_verdict_is_carried_into_the_new_report(monkeypatch, tmp_path):
    rows = slate()
    code, out, reports = run_main(monkeypatch, tmp_path, rows)
    frozen = {"status": "judged", "judged_on": "2026-10-13", "scored": 2100, "need_scored": 2000, "running": None,
              "verdict": "PASS", "criteria": []}
    prior = report.render_report(as_of=AS_OF - timedelta(days=7), doc=json.loads(out.read_text()), decision=frozen)
    (reports / "2026-10-13-shrink-gate.md").write_text(prior)
    (reports / "2026-10-20-shrink-gate.md").unlink()
    code, _, reports = run_main(monkeypatch, tmp_path, rows)
    new = report.parse_report((reports / "2026-10-20-shrink-gate.md").read_text())["decision"]
    assert code == 0 and new == frozen


# ---- guards and read-only queries ------------------------------------------------------------------------------------

def test_main_refuses_under_the_test_ledger_guard(capsys):
    assert fs.main([]) == 0 and "SM_LEDGER_OFF" in capsys.readouterr().out


def test_main_needs_the_database_url(monkeypatch, capsys):
    monkeypatch.delenv("SM_LEDGER_OFF", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert fs.main([]) == 1 and "DATABASE_URL is not set" in capsys.readouterr().out


def test_the_script_never_reaches_a_database_in_tests(monkeypatch):
    monkeypatch.setattr(db, "get_postgres", lambda *a, **k: pytest.fail("no connection expected"))
    assert fs.main([]) == 0                                               # SM_LEDGER_OFF is set by the root conftest


def test_the_queries_are_read_only_and_ask_for_the_legs_version_and_the_served_version():
    seen = []
    fs.collect(lambda sql, params=(): seen.append(sql) or [])
    assert "FROM nfl_prop_grades" in seen[0] and "nfl_player_sim" in seen[0] and "abs(s.mean - g.projection)" in seen[0]
    assert "result IN ('hit', 'miss')" in seen[0] and "FROM nfl_sim_serving" in seen[1]
    assert not any(w in s.upper().split() for s in seen for w in ("INSERT", "UPDATE", "DELETE", "TRUNCATE", "DROP", "ALTER"))


def test_collect_returns_none_when_the_serving_table_has_no_row():
    assert fs.collect(fake_q([], served=None))[1] is None
````

and `tests/test_workflows_shrink.py`:

````python
"""Structural checks of .github/workflows/fit-shrink-k.yml: Tuesday 15:30 UTC, env-passed secrets, timeout, serialised runs,
write permission, and the 3-attempt pull --rebase push retry. Reuses the block-YAML reader and the tiny cron evaluator."""
from datetime import timedelta

from tests.ledger.test_workflows_capture import cron_sets, fires, utc
from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

FIT = "fit-shrink-k.yml"


def test_the_schedule_is_tuesdays_at_1530_utc_plus_manual_dispatch():
    wf = load(FIT)
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["30 15 * * 2"] and "workflow_dispatch" in wf["on"]
    t, week = utc(2026, 10, 12, 0, 0), []
    while t < utc(2026, 10, 19, 0, 0):
        if fires("30 15 * * 2", t):
            week.append(t)
        t += timedelta(minutes=1)
    assert week == [utc(2026, 10, 13, 15, 30)]


def test_it_runs_after_the_player_actuals_capture_and_before_the_early_monitor_report():
    m, h, mon, dow = cron_sets(load("capture-player-actuals.yml")["on"]["schedule"][0]["cron"])
    assert 2 in dow and 15 in h and m == {0}                                   # Tuesday 15:00 grades Monday night
    assert [c["cron"] for c in load("report-early-tickets.yml")["on"]["schedule"]] == ["52 15 * * 2"]


def test_the_job_has_a_timeout_write_permission_and_serialised_runs():
    wf = load(FIT)
    assert wf["permissions"] == {"contents": "write"}
    assert wf["concurrency"]["group"] == "fit-shrink-k" and str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
    (job,) = wf["jobs"].values()
    assert str(job["timeout-minutes"]) == "10"


def test_secrets_go_through_env_and_no_expression_is_interpolated_into_a_shell_script():
    (job,) = load(FIT)["jobs"].values()
    steps = steps_of(job)
    fit = steps[step_index(steps, runs("scripts/fit_shrink_k.py"))]
    assert fit["env"] == {"DATABASE_URL": "${{ secrets.DATABASE_URL }}"}
    assert all("${{" not in st.get("run", "") for st in steps)
    text = (WF / FIT).read_text()
    assert "ODDS_API_KEY" not in text and "SM_LEDGER_OFF" not in text


def test_fit_runs_before_the_commit_and_the_commit_stages_only_the_two_outputs():
    (job,) = load(FIT)["jobs"].values()
    steps = steps_of(job)
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("fit_shrink_k.py")) < step_index(steps, runs("git commit"))
    commit = steps[step_index(steps, runs("git commit"))]["run"]
    assert "assets/shrink_k.json docs/superpowers/reports/*-shrink-gate.md" in commit and "[skip ci]" in commit
    assert "git add ." not in commit and "git add -A" not in commit


def test_the_push_retries_three_times_with_pull_rebase():
    (job,) = load(FIT)["jobs"].values()
    commit = steps_of(job)[step_index(steps_of(job), runs("git commit"))]["run"]
    assert "for attempt in 1 2 3" in commit and "git pull --rebase" in commit and "git push" in commit
    assert 'push failed after 3 attempts' in commit and commit.rstrip().endswith("exit 1")


def test_a_failed_fit_step_stops_the_job_before_any_commit():
    (job,) = load(FIT)["jobs"].values()
    steps = steps_of(job)
    for st in steps:
        assert "continue-on-error" not in st and "if" not in st, st               # default `success()`: no commit after a failed fit
````

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest -q --continue-on-collection-errors tests/scripts/test_fit_shrink_k.py tests/test_workflows_shrink.py`
Expected: `6 failed, 1 passed, 1 error`. The error is a `FileNotFoundError` for `scripts/fit_shrink_k.py` (loaded at import by `test_fit_shrink_k.py`); six workflow tests fail with `FileNotFoundError` for `fit-shrink-k.yml` (the seventh only reads existing workflows and passes).

- [ ] **Step 3: Write the script**

Create `scripts/fit_shrink_k.py`:

````python
"""Weekly refit of the prop shrink (profit step 2 Part A): read the graded NFL prop leans, fit k, write assets/shrink_k.json
and the dated gate report docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md.

Read-only against the database (`nfl_prop_grades`, the sim table for each leg's model version, `nfl_sim_serving` for the
current version). The k file is written ONLY after the fetch and the whole fit succeeded; on any failure this exits non-zero and
leaves the existing file untouched. It never writes a k it did not compute, and refuses to replace a file fitted on more legs
than it just read (a truncated read). The workflow (fit-shrink-k.yml) commits the two files.

    uv run python scripts/fit_shrink_k.py [--as-of 2026-10-20T15:30:00Z] [--out assets/shrink_k.json]
                                          [--reports-dir docs/superpowers/reports] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.ledger.record import writes_disabled  # noqa: E402
from sportsmodel.shrink import fit, passtest, report  # noqa: E402
from sportsmodel.shrink.store import SHRINK_K_PATH  # noqa: E402

REPORTS = Path(__file__).resolve().parents[1] / "docs" / "superpowers" / "reports"

# Every graded sim lean with a book line. The model version of a leg is the nfl_player_sim row for the same (game, player,
# market) whose mean equals the line row's projection (nfl_prop_lines does not store it); a leg whose sim row is gone gets NULL
# and counts only in the all-versions pool.
GRADES_SQL = """
SELECT g.game_pk, g.player_id, g.market, g.line, g.over_price, g.under_price, g.p_over, g.lean, g.result,
       g.season, g.week, v.model_version
FROM nfl_prop_grades g
LEFT JOIN LATERAL (
    SELECT s.model_version
    FROM nfl_player_sim s
    WHERE s.game_pk = g.game_pk AND s.player_id = g.player_id AND s.market = g.market
    ORDER BY abs(s.mean - g.projection) ASC, s.created_at DESC
    LIMIT 1
) v ON true
WHERE g.result IN ('hit', 'miss')
"""
SERVING_SQL = "SELECT model_version FROM nfl_sim_serving WHERE id = 1"


def _db_query():
    from sportsmodel import db

    def q(sql, params=()):
        with db.get_postgres() as conn, conn.cursor() as cur:
            cur.execute(sql, params)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
    return q


def collect(q) -> tuple[list[dict], str | None]:
    """(graded rows, the served props model version) via read-only queries `q(sql, params) -> list of dict rows`."""
    rows = q(GRADES_SQL, ())
    served = q(SERVING_SQL, ())
    return rows, (served[0]["model_version"] if served else None)


def read_existing(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def write_atomic(path: Path, text: str) -> None:
    """Write `text` to a temp file next to `path`, then replace: a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def build(rows: list[dict], current_version: str | None, as_of: datetime, previous: dict | None):
    """(document, report text, decision): everything computed before anything is written."""
    legs = fit.legs_from_grades(rows)
    doc = fit.build_document(legs, current_version, as_of.isoformat())
    scored = passtest.score_walk_forward(legs, current_version)
    decision = passtest.decide(scored, previous, f"{as_of:%Y-%m-%d}")
    return doc, report.render_report(as_of=as_of, doc=doc, decision=decision), decision


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--as-of", default=None, help="fit time (ISO, default: now UTC)")
    ap.add_argument("--out", default=str(SHRINK_K_PATH))
    ap.add_argument("--reports-dir", default=str(REPORTS))
    ap.add_argument("--dry-run", action="store_true", help="print the document and report instead of writing them")
    args = ap.parse_args(argv)
    if writes_disabled():
        print("fit-shrink-k: SM_LEDGER_OFF is set; refusing to run", flush=True)
        return 0
    if not os.getenv("DATABASE_URL"):
        print("::error::fit-shrink-k: DATABASE_URL is not set")
        return 1
    as_of = datetime.fromisoformat(args.as_of.replace("Z", "+00:00")) if args.as_of else datetime.now(timezone.utc)
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    out, reports_dir = Path(args.out), Path(args.reports_dir)
    try:
        rows, current_version = collect(_db_query())
        previous = report.latest_decision([p.read_text() for p in sorted(reports_dir.glob("*-shrink-gate.md"))], as_of)
        doc, text, decision = build(rows, current_version, as_of, previous)
    except Exception as exc:  # noqa: BLE001 -- any fetch or fit failure: keep the existing file, fail the job
        print(f"::error::fit-shrink-k: {type(exc).__name__}: {exc}; assets/shrink_k.json left untouched")
        return 1
    if fit.has_fewer_legs(read_existing(out), doc):
        print(f"::error::fit-shrink-k: this read has {doc['data_window']['n_legs']} eligible legs, fewer than the committed "
              "file was fitted on; the read looks truncated, file left untouched")
        return 1
    if args.dry_run:
        print(json.dumps(doc, indent=2, sort_keys=True))
        print(text)
        return 0
    write_atomic(out, json.dumps(doc, indent=2, sort_keys=True) + "\n")
    path = reports_dir / f"{as_of:%Y-%m-%d}-shrink-gate.md"
    path.write_text(text)
    print(f"[fit-shrink-k] wrote {out.name}: pooled k={doc['pooled']['k']:.3f} on n={doc['pooled']['n']} "
          f"(version {doc['model_version_used']}); {path.name}: status={decision['status']} scored={decision['scored']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
````

- [ ] **Step 4: Write the workflow**

Create `.github/workflows/fit-shrink-k.yml`:

````yaml
name: fit-shrink-k

# Weekly refit of the prop shrink (profit step 2 Part A, docs/superpowers/specs/2026-10-13-shrink-props-design.md).
# Every Tuesday, after capture-player-actuals (15:00) has graded Monday night: read the graded NFL prop leans
# (read-only), fit k per market (scripts/fit_shrink_k.py), write assets/shrink_k.json and the dated gate report
# docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md, and commit both. The prop board (build-ev-props) reads
# assets/shrink_k.json from the repo; until the first fit lands it is price-only (k = 0).
# On ANY fetch or fit failure the script exits non-zero and leaves the existing file untouched, and nothing is committed.
# FIRST RUN: dispatch this workflow by hand right after merge (gh workflow run fit-shrink-k.yml --ref main).
on:
  workflow_dispatch: {}
  schedule:
    - cron: "30 15 * * 2"

permissions:
  contents: write

concurrency:
  group: fit-shrink-k
  cancel-in-progress: false

jobs:
  fit:
    runs-on: ubuntu-latest
    timeout-minutes: 10
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
      - name: Sync deps
        run: uv sync
      - name: Fit k and write the gate report
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/fit_shrink_k.py
      - name: Commit k and the report
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          for f in assets/shrink_k.json docs/superpowers/reports/*-shrink-gate.md; do
            [ -f "$f" ] && git add "$f"
          done
          if git diff --staged --quiet; then
            echo "no change to commit"
            exit 0
          fi
          git commit -m "chore: weekly shrink-k refit [skip ci]"
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

Run: `uv run pytest -q tests/scripts/test_fit_shrink_k.py tests/test_workflows_shrink.py tests/test_workflows_props_ml.py`
Expected: `138 passed` (18 + 7 new, plus the existing workflow tests, which now include one more `test_every_workflow_parses` case).

- [ ] **Step 6: Commit**

```bash
git add scripts/fit_shrink_k.py .github/workflows/fit-shrink-k.yml tests/scripts/test_fit_shrink_k.py tests/test_workflows_shrink.py
git commit -m "feat(shrink): weekly fit script and workflow (read-only DB, atomic write, frozen gate report)" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Go-live checklist and final verification

**Files:**
- Create: `docs/superpowers/reports/2026-10-13-shrink-props-go-live.md`

**Interfaces:**
- Consumes: everything above.
- Produces: the user's go-live checklist (run the migration, merge, dispatch the first fit, what changes on the board).

- [ ] **Step 1: Write the go-live checklist**

Create `docs/superpowers/reports/2026-10-13-shrink-props-go-live.md`:

`````markdown
# Shrink prop probabilities: go-live checklist (2026-10-13)

Branch `feat/shrink-props`. Everything below is for you to run; nothing here has been executed. No migration was run, nothing was pushed, no workflow was dispatched, and no code ever connected to the database during the build.

## What this changes, in plain words

Until now the prop board trusted the model's probability. The audit showed that was too bold: the model's prop leans claimed 64.8% and hit 51.8%. From the first fit on, every prop pick is priced on a probability **pulled toward the market's price**: `market + k x (model - market)`, where k is how much the model has earned from its own graded history (the audit points to about 0.15). The same shrunk number drives the pick, the EV, the edge and Alpha Score on the site, the +EV parlays and the pick ledger. The sims and models are not touched, and the raw record on the NFL prop-lines pages stays as it was.

Expect **far fewer prop picks** and often **no prop legs in parlays** (a leg still needs a shrunk probability of at least 0.55). That is the intended result, not a fault.

## 1. Run the migration

Paste `db/migration_shrink_props.sql` into the Supabase SQL editor and run it once (idempotent, safe to re-run, adds two empty columns to `ev_prop_picks`). It is safe before or after the merge: until the columns exist the board simply leaves them out and prints a `WARN` line.

Check it worked (should return two rows):

```sql
SELECT column_name FROM information_schema.columns
WHERE table_name = 'ev_prop_picks' AND column_name IN ('raw_model_prob', 'shrink_k');
```

## 2. Merge, then fit at once

Merge the PR. There is no k file yet, so every prop board build that runs before the first fit is **price-only** (k = 0): the model is ignored and a prop is a pick only when two books' best prices disagree enough to be +EV on their own (rare). So do step 3 right after the merge, not later in the week.

## 3. Dispatch the first fit

From a terminal: `gh workflow run fit-shrink-k.yml --ref main`. Or on GitHub: Actions, `fit-shrink-k`, Run workflow, branch `main`.

It takes about a minute. When it is green:

- The log ends with a line like `[fit-shrink-k] wrote shrink_k.json: pooled k=0.15 on n=... (version nfl-sim-ml-v2); ...-shrink-gate.md: status=collecting scored=...`.
- A commit `chore: weekly shrink-k refit [skip ci]` appears on `main` with two files: `assets/shrink_k.json` and `docs/superpowers/reports/YYYY-MM-DD-shrink-gate.md`.
- The next prop board build (or `gh workflow run build-ev-props.yml --ref main`) picks the file up.

If it is red, nothing was written and nothing changed: the board stays price-only until a fit succeeds. Read the log; the error line names the cause (database unreachable, a missing table, or a read that returned fewer graded legs than the committed file, which is refused as a truncated read).

How to read `assets/shrink_k.json`: `pooled.k` and `pooled.n` are the overall number and the legs behind it. If `pooled.n` is under 300 every k is 0 (not enough evidence yet), by design. `markets.nfl.<market>.k` is what each market uses; a market with `n` = 0 (for example `pass_tds`, which has no graded history) stays price-only. MLB is 0 everywhere in this part.

## 4. What you will see after it is live

| Where | Before | After |
|---|---|---|
| +EV props board (NFL, MLB) | the model's probability decides the pick | the shrunk probability decides: far fewer picks; MLB props are price-only for now, so almost none |
| Alpha Score / edge / confidence on prop rows | from the model's raw edge | from the shrunk edge (smaller, honest) |
| +EV parlays | prop legs often qualify | prop legs rarely reach 0.55; parlays may be empty |
| Pick ledger (`ev_prop`, `parlay_leg`) | model probability | the shrunk probability that made the pick |
| `ev_prop_picks` rows | `model_prob` | `model_prob` (shrunk), plus `raw_model_prob` and `shrink_k` for audit |
| NFL prop-lines / grades pages | raw model | unchanged (they are the fit's source) |

## 5. Every Tuesday, and the pass test

`fit-shrink-k` runs Tuesdays at 15:30 UTC (after Monday night is graded), refits k on all graded legs, and commits the file and a dated report in `docs/superpowers/reports/`. The report shows the current k per market with n, how many legs have been scored so far, the running accuracy gain over the market and the calibration slope, and the status.

The pass test is judged **once**, at the first weekly report with at least 2,000 scored legs (each week is scored with a k fitted only on earlier weeks, so there is no peeking). Until then it says COLLECTING and nothing can pass or fail. That count may not be reached this season: only the three markets the board serves are scored, and the first 300 legs are used up before any week can be scored. If the season ends first, collection continues next season. After it is judged, the verdict never changes in later reports.

- **Pass** (the shrunk model beats the market price by at least 0.001 Brier with the lower confidence bound above 0, and its calibration slope is between 0.8 and 1.2): shrunk model prop picks stay. This does not prove profit; closing-line value in the pick ledger judges that.
- **Fail:** you and Claude decide together what happens to model prop picks. Nothing is removed automatically; picks keep using the shrunk probabilities until then.

## 6. If something looks wrong

- **Board has no prop picks at all:** expected right after the merge (price-only) and when k is small. Check `assets/shrink_k.json` exists on `main` and `pooled.n` is at least 300.
- **A `::warning::shrink: ... using k=0` line in a build log:** the k file is missing, unreadable, or lacks a market. The board is price-only for those markets and keeps running. It never falls back to the raw model.
- **To go back to the old behaviour:** revert the merge commit on GitHub (Revert button on the PR). The two new columns can stay; they are harmless.

## 7. What was not changed

The sims and props-ML models, `nfl_prop_lines` and `nfl_prop_grades`, the game-line boards and early-week tickets, all game-level numbers on the site (a separate Part B follows), the pick thresholds (a pick is still `0 < EV <= 25%`), and all site code.

## 8. Known limits

- The market probability used in the fit comes from the best prices across all books on the prop-lines table (Pinnacle included); the board uses the best non-Pinnacle prices. The formula is identical, so k is internally consistent, but the two are not the same number to the last decimal.
- Legs are matched to their model version by the sim row whose mean equals the line row's projection. Legs whose sim row no longer exists count only in the all-versions pool. Until 300 legs of the current model (props-ML v2) are graded, k is fitted on all versions pooled, which includes the older model's legs.
- The audit already saw the existing graded weeks, so the test is not fully blind on that data. The walk-forward scoring and fixed thresholds limit the harm.
`````

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/reports/2026-10-13-shrink-props-go-live.md
git commit -m "docs: shrink-props go-live checklist" -m "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Full suite**

Run: `uv run pytest -q` (about 4 minutes)
Expected: `3440 passed, 2 skipped` (the 3244 baseline plus the 196 new tests). No test may skip because of the new package.

- [ ] **Step 4: Only the intended files changed**

Run: `git diff --stat main...HEAD`
Expected: exactly the files in the File Structure table plus the spec and this plan (new: `src/sportsmodel/shrink/*`, `scripts/fit_shrink_k.py`, `db/migration_shrink_props.sql`, `.github/workflows/fit-shrink-k.yml`, the go-live checklist, nine new test files; modified: `src/sportsmodel/serving/props_ev.py`, `src/sportsmodel/db.py`, `scripts/build_ev_props_board.py` and four existing test files). Nothing under `site/`, `data/` or `assets/`, no model, no ledger module, no existing workflow.

- [ ] **Step 5: Safety checks**

```bash
# the pure package never touches a database or the network
grep -rnE "psycopg|get_postgres|httpx|requests|urllib" src/sportsmodel/shrink || echo "shrink package is pure: OK"
# the fit script only reads: no write statements in the new Python (db.py's writer is the only INSERT)
grep -nE "INSERT INTO|UPDATE |DELETE FROM|DROP |ALTER TABLE|TRUNCATE|CREATE TABLE" src/sportsmodel/shrink/*.py scripts/fit_shrink_k.py || echo "no writes or DDL in the new code: OK"
# the test guard must never appear in a workflow
grep -rn "SM_LEDGER_OFF" .github || echo "guard not in workflows: OK"
# only the scheduled workflow may create the k file
ls assets/shrink_k.json 2>/dev/null || echo "no k file committed: OK"
git status --short
```
Expected: the four `OK` lines print and `git status` prints nothing.

- [ ] **Step 6: Hand back**

Do NOT run the migration, merge, push, or dispatch any workflow. Report the branch, the commits, the final test counts and the decisions table (for the PR description), then follow superpowers:finishing-a-development-branch.

---

## Handoff to the user (after review; the executing agent does none of this)

The full checklist is `docs/superpowers/reports/2026-10-13-shrink-props-go-live.md`. In short:

1. **Run the migration** `db/migration_shrink_props.sql` in the Supabase SQL editor (safe to re-run, safe before or after the merge).
2. **Merge the PR.** From then until the first fit, the prop board is price-only (k = 0).
3. **Immediately dispatch the first fit:** `gh workflow run fit-shrink-k.yml --ref main`. It commits `assets/shrink_k.json` and the first gate report. If it is red nothing changed.
4. **Expect far fewer prop picks** and often no prop legs in parlays; MLB props are price-only in this part.
5. **Read the Tuesday report.** It says COLLECTING until 2,000 scored legs; only then is the test judged, once. A fail means you and Claude decide together; nothing is removed automatically.

## Self-Review

Run against the spec with fresh eyes after writing the plan.

**1. Spec coverage**

| Spec requirement | Task / evidence |
|---|---|
| Formula `clip(m + k × (p − m), 0.01, 0.99)`; under side `1 − shrunk`; m = board's no-vig of best soft over / under | Task 1 `test_the_result_is_clipped_to_one_and_ninety_nine_percent`, `test_market_prob_over_is_the_boards_novig_...`; Task 2 `test_the_under_side_uses_one_minus_the_shrunk_probability_...` |
| k closed form, clamp [0, 1], zero variance gives 0, pushes excluded | Task 1 `test_closed_form_recovers_a_known_k_exactly_on_expected_outcomes`, `..._on_simulated_bernoulli_outcomes`, `test_closed_form_clamps_to_zero_and_one`, `test_zero_variance_and_no_legs_give_zero`, `test_pushes_are_excluded` |
| Fit sample from `nfl_prop_grades`; invalid / missing price excluded | Task 1 `test_rows_with_a_missing_or_invalid_american_price_are_excluded`; Task 5 `test_pushes_invalid_prices_and_unserved_markets_do_not_count`, the grades query test |
| Pooled k; per-market partial pooling with 300; pooled < 300 gives k = 0; exactly 300 fits | Task 1 `test_each_market_is_partially_pooled_toward_the_pooled_k`, `test_fewer_than_300_pooled_legs_means_k_zero_for_every_market`, `test_exactly_300_pooled_legs_fits` |
| Model version: current if ≥ 300 legs else all; version read from data | Task 1 `test_the_current_version_is_used_when_it_has_300_legs`, `test_299_legs_of_the_current_version_fall_back_to_every_version`; Task 5 `test_the_current_version_comes_from_the_serving_table_and_falls_back_to_all_versions`; decision 4 |
| MLB k = 0 for every market | Task 1 `test_mlb_is_zero_for_every_market_whatever_the_data`; Task 2 `test_mlb_is_always_zero_even_if_the_file_says_otherwise`, `test_mlb_is_price_only_in_part_a_...`; Task 3 MLB board test |
| Walk-forward (k only from earlier-graded legs) | Task 4 `test_a_week_is_scored_with_the_k_fitted_on_earlier_weeks_only`, `test_later_weeks_never_change_an_earlier_weeks_score`; weekly refit Task 5 |
| Committed parameters: k, n, k_raw per (sport, market), pooled, model_version_used, fitted_at, window | Task 1 `test_the_document_records_k_n_k_raw_pooled_version_fitted_at_and_the_data_window`; Task 5 `test_a_successful_run_writes_the_k_file_and_the_dated_report` |
| Serving: shrink after calibrate, EV / pick / `model_prob` / `edge` use shrunk; pick rule unchanged | Task 2 `test_mlb_calibration_runs_before_the_shrink`, `test_a_raw_pick_that_the_shrink_removes_is_not_a_pick`, `test_the_ev_ceiling_still_applies_to_the_shrunk_ev` |
| Fail-safe: missing file / entry gives k = 0, warn once, never raw | Task 2 store tests and `test_a_market_missing_from_the_table_is_price_only_never_the_raw_model`; Task 3 `test_with_no_shrink_file_the_board_runs_price_only_and_warns_once`, `test_main_hands_the_table_to_assemble_for_both_sports` |
| Audit columns written only when they exist; migration SQL verbatim | Task 3 `tests/test_db_ev_prop_shrink_cols.py` (8 tests incl. `test_the_migration_is_the_specs_idempotent_additive_alter`) |
| Parlays read `model_prob` with no code change; `MIN_LEG_PROB` 0.55 | Task 3 `test_prop_legs_take_the_shrunk_model_prob_from_the_board_row_unchanged`, `test_shrunk_prop_legs_below_the_055_floor_no_longer_build_a_ticket_but_strong_ones_still_do` |
| Pick ledger unchanged, records the shrunk `model_prob` | Task 3 `test_ev_prop_observation_records_the_shrunk_probability_that_made_the_pick` |
| Refit job: Tuesday 15:30 UTC + dispatch, read-only, 3-attempt `pull --rebase` retry, failure leaves file untouched, timeout, env secrets, `contents: write` | Task 5 workflow tests (7) and script tests (18) |
| Pass test: walk-forward set, 300 earlier legs, judged once at 2,000, Brier gain ≥ 0.001 with CI lower > 0, 10,000 resamples / seed 20261013 / 2.5th percentile, slope in [0.8, 1.2], frozen verdict | Task 4 (constants test, threshold and boundary tests, bootstrap golden-style and clustering tests, slope cross-check) |
| Weekly report content and path | Task 4 report tests; Task 5 writes `YYYY-MM-DD-shrink-gate.md` |
| Out of scope (Part B, models, MLB fit, thresholds, site) | Nothing in these is touched; Task 6 Step 4 `git diff --stat` |

No gaps found. Spec items deliberately interpreted are in the decisions table, each with its reason.

**2. Placeholder scan.** No `TBD` / `TODO` / "implement later" / "similar to Task N"; every code step carries the code or the exact hunk.

**3. Type consistency.** Checked across tasks: the leg dict `{season, week, game_pk, market, p, m, y, version}` is produced by `fit.legs_from_grades` (Task 1) and consumed unchanged by `fit.fit_table`, `passtest.score_walk_forward` (Task 4) and `scripts/fit_shrink_k.build` (Task 5); `fit.fit_table(...)["markets"]["nfl"][market]["k"]` is what `passtest` reads and exactly the shape `store.ShrinkTable.from_dict` reads from the committed document (`doc["markets"][sport][market]["k"]`); `ShrinkTable.k(sport, odds_market)` is called by `assemble_prop_rows` with the odds key and by nothing else; `fit.build_document(...)` (Task 1) is what `report.render_report(doc=...)` (Task 4) and `fit_shrink_k.build` (Task 5) pass around; `passtest.decide(scored, previous, as_of)` returns the dict `report.render_report` renders and `report.latest_decision` returns as `previous`.

**4. Review Focus.** Each of the five lines names its pinning tests; every cited test name was checked to exist in the test files above.

**Known limits, accepted:** (a) the 2,000 scored-leg judgement point counts only the three served markets that have graded history and skips the first 300 legs, so it may not be reached this season; (b) `pass_tds` and MLB are price-only; (c) the fit's m uses the best prices across all books (Pinnacle included) while the board's m excludes Pinnacle (decision 5); (d) a leg's model version is inferred from the nearest sim mean (decision 4); legs before the props-ML models were served come from older models and are pooled until 300 current-version legs exist; (e) the audit already saw the existing graded weeks, so the test is not fully blind on that data; (f) at k = 0 a board pick needs a genuine cross-book price gap, so the board is nearly empty until the first fit lands.
