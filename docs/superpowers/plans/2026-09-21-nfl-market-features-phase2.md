# NFL Market-Microstructure Features (Phase 2) — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or executing-plans. Steps use checkbox (`- [ ]`) syntax. This plan builds the CAPTURE + FEATURE code now; validation (re-gate) is deferred until market data accrues (a season+). Nothing user-facing ships in this plan.

**Goal:** Add betting-splits + line-movement features to the cover/total GBM and capture them going forward, so the existing walk-forward gate can eventually test them. No serving/UI change.

**Architecture:** Pure feature builders (`nfl/market_features.py`) over `odds_snapshot` (line movement) and a new `nfl_betting_splits` table (splits); a splits capture pipeline (SportsDataIO Betting tier); integration into `build_cover_dataset.assemble_rows` → `train_cover_ensemble` feature lists. GBM (`HistGradientBoostingRegressor`) consumes NaN natively, so features are NaN for the historical tail and populated forward — no backfill.

**Tech Stack:** Python 3.12 (uv), pandas/numpy, scikit-learn, httpx (SportsDataIO), psycopg, pytest.

**Spec:** docs/superpowers/specs/2026-09-21-nfl-market-features-phase2-design.md

## Global Constraints

- NFL only, spread + total. **Capture + wait**: no serving/UI change, no artifacts shipped in this plan.
- Live-first: features are NaN where the underlying data is absent (the whole historical tail). Never impute or fabricate; NaN is correct. HistGradientBoosting handles NaN.
- Leakage-safe: every feature for a game uses only `odds_snapshot`/splits rows strictly at/before the decision timestamp (or `commence_time`).
- Reuse: `serving/board.novig`/`implied_prob` for no-vig; `model/distributions` as needed; existing `nfl/sportsdata.py` adapter pattern (subscription-key header, tenacity backoff); existing `capture-odds` workflow windowing pattern.
- Sign conventions carried from Phase 1: `odds_snapshot` sides are `home|away|over|under`; cover uses ESPN-signed home line at serving. Keep splits/movement keyed consistently.
- User runs migrations, sets repo vars/secrets, and controls when capture is enabled. Tests pristine.

---

### Task 1: `normalize_team` prerequisite fix (Phase-1 follow-up) — do first

**Files:** Modify `scripts/build_cover_dataset.py`; Test `tests/scripts/test_build_cover_dataset.py`

- [ ] **Step 1: Failing test** — an `assemble_rows` case where `schedule_df` uses a non-canonical team code (e.g. `"WSH"`) while `game_epa`/`sim_lookup` are keyed by `normalize_team` output (`"WAS"`); assert the row still resolves efficiency + sim features (not NaN-from-miss). Run → fails (KeyError or silent NaN).
- [ ] **Step 2: Implement** — in `assemble_rows`, run `home`/`away` through `normalize_team` before every lookup (`adj[...]`, `sim_lookup[...]`, and the new market joins in Task 5). Run → passes.
- [ ] **Step 3: Commit** — `fix(nfl): normalize team codes before feature lookups in assemble_rows`.

---

### Task 2: `nfl_betting_splits` table + upsert

**Files:** Create `db/migration_nfl_betting_splits.sql`; Modify `src/sportsmodel/db.py`; Test `tests/test_db_nfl_betting_splits.py`

**Interfaces:** Produces `db.upsert_nfl_betting_splits(records) -> int`; table `nfl_betting_splits`.

- [ ] **Step 1: Migration** — `CREATE TABLE IF NOT EXISTS nfl_betting_splits (game_pk BIGINT NOT NULL, market TEXT NOT NULL, side TEXT NOT NULL, cash_pct DOUBLE PRECISION, ticket_pct DOUBLE PRECISION, commence_time TIMESTAMPTZ, captured_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (game_pk, market, side, captured_at))` + `idx ... (game_pk, market)`. No anon RLS needed (training-side). 
- [ ] **Step 2: Failing test** (FakeConn/FakeCursor, mirror `tests/test_db_nfl_sim.py`): empty→0; a row builds the tuple in `_NFL_BETTING_SPLITS_COLS` order; SQL is `INSERT ... ON CONFLICT (game_pk, market, side, captured_at) DO UPDATE`. Run → fails.
- [ ] **Step 3: Implement** `_NFL_BETTING_SPLITS_COLS = ["game_pk","market","side","cash_pct","ticket_pct","commence_time","captured_at"]` + `upsert_nfl_betting_splits` (idempotent per PK). Run → passes.
- [ ] **Step 4: Commit** — `feat(db): nfl_betting_splits table + upsert`.

---

### Task 3: SportsDataIO splits adapter + parser

**Files:** Modify `src/sportsmodel/nfl/sportsdata.py`; Test `tests/sim/nfl/test_sportsdata_parsers.py` (or the existing sportsdata parser test)

**Interfaces:** Produces `sportsdata.BETTING_SPLITS_PATH`, `sportsdata.parse_betting_splits(payload) -> list[dict]` (rows shaped for `upsert_nfl_betting_splits` minus `captured_at`).

- [ ] **Step 1: Failing test** — feed `parse_betting_splits` a MOCK payload matching SportsDataIO's betting-splits/consensus shape (per-game, per-market home/away or over/under cash% + ticket%); assert it yields rows `{game_pk, market, side, cash_pct, ticket_pct, commence_time}` with markets mapped to `spread|total|moneyline` and sides to `home|away|over|under`. Run → fails.
- [ ] **Step 2: Implement** the pure `parse_betting_splits` + the `_get` path constant. **Confirm the exact endpoint/field names against the live Betting tier at implementation** — the parser is written against the documented shape and adjusted once a real payload is available (note this in the report). The fetch wrapper reuses `_get` (key header + backoff); not unit-tested. Run → passes.
- [ ] **Step 3: Commit** — `feat(nfl): SportsDataIO betting-splits parser`.

---

### Task 4: `market_features.py` pure builders (line movement + splits)

**Files:** Create `src/sportsmodel/nfl/market_features.py`; Test `tests/nfl/test_market_features.py`

**Interfaces:** `line_movement(snapshots, game_pk, market, decision_ts) -> dict`; `sharp_vs_soft(snapshots, game_pk, market, side, decision_ts) -> float | None`; `reverse_line_movement(line_move, splits_row) -> float | None`; `split_features(splits_row) -> dict`.

- [ ] **Step 1: Failing tests** (synthetic `odds_snapshot`-shaped frames + splits dicts):
  - `line_movement`: opening = earliest snapshot's line, close = latest ≤ decision_ts, `dline`/`abs_dline` correct; a snapshot AFTER `decision_ts` is excluded (leakage test).
  - `sharp_vs_soft`: Pinnacle no-vig prob minus soft consensus no-vig (reuse `board.novig`); None when Pinnacle or soft side absent.
  - `reverse_line_movement`: positive when the line moved toward the side the public ticket% is *against*; None when splits absent.
  - `split_features`: returns `cash_pct`, `ticket_pct`, `cash_minus_ticket`; NaN-safe on None.
  Run → fails.
- [ ] **Step 2: Implement** the four pure functions (no IO). Leakage-safe filtering to `captured_at <= decision_ts`. Reuse `serving.board.novig`/`implied_prob`. Run → passes.
- [ ] **Step 3: Commit** — `feat(nfl): market-microstructure feature builders (line movement + splits)`.

---

### Task 5: Splits capture pipeline

**Files:** Create `scripts/capture_betting_splits.py`; Create `.github/workflows/capture-betting-splits.yml`

- [ ] **Step 1:** `capture_betting_splits.py` (IO): for NFL games commencing within a near-close window (reuse `capture-odds`'s window logic / `PROP_WINDOW_MIN`-style var, splits are meaningful post-lineup), fetch via `sportsdata` splits adapter, stamp `captured_at`, `upsert_nfl_betting_splits`. Gated on `SPORTSDATA_API_KEY` + an `INGEST_SPLITS` var (no-op/skip when unset) so it's dormant until the Betting tier is live. Print a one-line summary.
- [ ] **Step 2:** Workflow `capture-betting-splits.yml` — schedule mirroring `capture-odds.yml`'s game-day cadence; `workflow_dispatch`; `DATABASE_URL` + `SPORTSDATA_API_KEY` secrets, `INGEST_SPLITS` var. Validate YAML.
- [ ] **Step 3:** Import-clean check (`PYTHONPATH=src ... -c "import scripts.capture_betting_splits"` via importlib). **Commit** — `feat(nfl): betting-splits capture script + workflow (dormant until INGEST_SPLITS)`.

---

### Task 6: Integrate market features into the dataset + GBM

**Files:** Modify `scripts/build_cover_dataset.py`, `scripts/train_cover_ensemble.py`; Test `tests/scripts/test_build_cover_dataset.py`

- [ ] **Step 1: Failing test** — extend `assemble_rows` (and its signature) to accept optional `odds_snapshots` + `splits` lookups; assert a row carries the new market feature columns (line-movement, sharp-vs-soft, cash_minus_ticket), and that they are **NaN** when the lookups are empty (the historical-tail case). Run → fails.
- [ ] **Step 2: Implement** — thread `odds_snapshots`/`splits` (optional, default empty) into `assemble_rows`, compute features via `market_features` (normalized team keys, per Task 1), emit NaN when absent. `main()` loads `odds_snapshot` + `nfl_betting_splits` from Supabase where available. Add the new columns to `MARGIN_FEATURES`/`TOTAL_FEATURES` in `train_cover_ensemble.py` (HistGradientBoosting handles NaN — no imputation). Run → passes.
- [ ] **Step 3:** Do NOT re-fit or write artifacts (the gate still can't pass without accrued data). Full suite green. **Commit** — `feat(nfl): add market features to cover dataset + GBM feature set (NaN-native)`.

---

### Task 7: Runbook + operational note

**Files:** Modify `docs/cover-ensemble-runbook.md` (or create if absent)

- [ ] **Step 1:** Document the accumulate-then-re-gate loop: enable capture (`INGEST_SPLITS`, verify `odds_snapshot` populating), let it accrue a season+, periodically re-run `build_cover_dataset` → `train_cover_ensemble`, and wire into serving (the deferred Phase-1 board/+EV tasks) **only** if the gate passes. Note the SportsDataIO Betting endpoint confirmation step. **Commit.**

---

## Self-Review

- **Spec coverage:** splits ingest (T2/T3/T5), line-movement + splits features (T4), integration (T6), the `normalize_team` prereq (T1), capture-and-wait discipline (no artifacts/serving in any task), runbook (T7). ✓
- **Placeholders:** the SportsDataIO endpoint/field names are the one confirm-at-implementation item (tier not yet active) — flagged explicitly in T3, not a silent gap. All else concrete.
- **Type consistency:** `parse_betting_splits` rows → `upsert_nfl_betting_splits` (`_NFL_BETTING_SPLITS_COLS`) match; `market_features` builder outputs consumed by `assemble_rows` (T6) match their signatures (T4); `assemble_rows` optional lookups (T6) are additive to Phase-1 signature.

## Execution notes / risks

- **No validation in this plan** — the gate can't pass until data accrues; success here = correct, tested capture + feature code, not a shipping model.
- **External prereqs:** SportsDataIO Betting tier active (T3/T5), and `odds_snapshot` actually populating (currently empty — verify `capture-odds`). Both are the user's to enable.

## Execution handoff

Plan saved to `docs/superpowers/plans/2026-09-21-nfl-market-features-phase2.md`. Execute via **subagent-driven** (fresh subagent per task + review) or **inline** (executing-plans). Which approach?
