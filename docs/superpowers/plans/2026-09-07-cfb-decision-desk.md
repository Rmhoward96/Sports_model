# CFB Decision Desk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the plumbing, storage, immutable pre-kickoff pick-writer, forward CLV grader, and site display for a CFB decision desk whose three agents (statistics / sports-analyst / news) are run on-demand in a Claude Code session and produce tiered ML / Spread / Total picks per game.

**Architecture:** A SportsDataIO adapter + a desk-input assembler produce a per-slate JSON bundle (our model output + recent form + injuries/news/weather). The three agents + synthesis run **in-session** (a documented runbook, not automated code) and emit a structured picks JSON. A pick-writer validates it, enforces pre-kickoff immutability, and upserts `desk_picks`. A pure-code grader (automated in CI) later compares each pick to the final **and the closing line** (ATS/total cover + CLV). The site shows the picks, tiers, rationale, and a running CLV/accuracy record. The LLM reasoning is never automated code — only the structured input/output plumbing is built here.

**Tech Stack:** Python 3.12, `httpx` + `tenacity` (SportsDataIO, mirroring the `cfbd`/`espn` adapters), `psycopg` (Supabase), `pytest`. Reuses `sportsmodel.cfb.espn` (closing line via pickcenter), the existing `predictions_current`/`lines.parquet` data, and the `grade_vs_market` convention logic. Front-end is the external CappingAlpha `app.js`.

**Spec:** `docs/superpowers/specs/2026-09-07-cfb-decision-desk-design.md`

## Global Constraints

- **`SPORTSDATA_API_KEY` is a secret** (env / GitHub Actions), never printed, logged, or committed. Live SportsDataIO calls happen only in the assembler's `main()`; parsers are tested against **committed sample-JSON fixtures**, never live calls (same handling as `CFBD_API_KEY`).
- **Picks are immutable once a game starts.** The pick-writer MUST refuse to write or overwrite a pick for any game whose `commence_time` is not strictly in the future. This protects the forward CLV test — a pick logged after kickoff is worthless. Pre-kickoff re-runs (idempotent per `game_pk, model_version`) may overwrite.
- **The desk is graded FORWARD only.** No backtest. The grader compares logged pre-kickoff picks to results + the **closing** line. `desk_picks` stores the **line the desk took at pick time**; CLV = closing line − pick-time line, signed by the side taken.
- **Market-line conventions stay consistent with existing code.** ESPN pickcenter = sportsbook convention (home favored → negative); CFBD `lines.parquet` = home-margin (home favored → positive). Reuse `grade_vs_market`/`parse_market`; do not conflate them.
- **Assistive framing.** The site labels the desk "assistive — not a proven edge" until the CLV record justifies otherwise. No language claiming an edge before the forward test earns it.
- **The three agents + synthesis are NOT automated code** — they are a documented runbook (Task 6) the controller follows in-session. This plan builds only the structured input bundle, the picks-JSON contract, storage, grading, and display around that.
- **Pure/IO separation, TDD, reuse existing patterns** (`cfbd`/`espn` adapters, `grade_predictions.py`, `predictions_current`, `db.upsert_*`). Market-independence does NOT apply here — the desk deliberately consumes the market line (as the thing to beat).

---

## File Structure

- Create `src/sportsmodel/cfb/sportsdata.py` — SportsDataIO adapter: pure parsers (injuries, news, weather) + a retried `_get(path, api_key, params)` client.
- Create `tests/fixtures/cfb/sportsdata_{injuries,news,weather}.json` — sample responses.
- Create `scripts/desk_inputs.py` — assemble the per-slate JSON bundle (model output + recent form + news) → `--out <path>`.
- Create `db/migration_decision_desk.sql` — `desk_picks`, `desk_pick_results`, `desk_current` + `desk_record` views, grants.
- Modify `src/sportsmodel/db.py` — `upsert_desk_picks`, `upsert_desk_pick_results`.
- Create `scripts/write_desk_picks.py` — validate a picks JSON (schema, tiers, sides, pre-kickoff), upsert `desk_picks`.
- Create `scripts/grade_desk_picks.py` — grade `desk_picks` vs finals + closing line → `desk_pick_results` (ATS/total cover + CLV).
- Create `.github/workflows/grade-desk-picks.yml` — scheduled grader (no LLM, no SportsData key; needs `DATABASE_URL`).
- Create `docs/decision-desk-runbook.md` — the in-session agent procedure + the picks-JSON contract.
- Tests: `tests/cfb/test_sportsdata_parsers.py`, `tests/cfb/test_desk_inputs.py`, `tests/test_write_desk_picks.py`, `tests/test_grade_desk_picks.py`, `tests/test_db_desk.py`.
- Modify external `/Users/ryan/Desktop/CappingAlpha/app.js` — Decision Desk section (picks + tiers + rationale + running record).

---

## Task 1: SportsDataIO adapter (pure parsers + fixtures)

**Files:**
- Create: `src/sportsmodel/cfb/sportsdata.py`
- Create: `tests/fixtures/cfb/sportsdata_{injuries,news,weather}.json`
- Test: `tests/cfb/test_sportsdata_parsers.py`

**Interfaces:**
- Produces: `parse_injuries(payload) -> dict[team, list[dict]]` (per team: player, position, status, note); `parse_news(payload) -> list[dict]` (headline, teams, published, summary); `parse_weather(payload) -> dict[game_key, dict]` (temp, wind, precip, dome) keyed however the provider keys games; `_get(path, api_key, params=None) -> Any` (retried, Bearer/`Ocp-Apim-Subscription-Key` header per SportsDataIO's scheme).

- [ ] **Step 1: Write failing tests** for each parser against a small fixture. Example:

```python
# tests/cfb/test_sportsdata_parsers.py
import json, pathlib
from sportsmodel.cfb import sportsdata
FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "cfb"

def test_parse_injuries_groups_by_team_with_status():
    payload = json.loads((FIX / "sportsdata_injuries.json").read_text())
    out = sportsdata.parse_injuries(payload)
    assert "Alabama" in out
    qb = next(p for p in out["Alabama"] if p["position"] == "QB")
    assert qb["status"] in {"Out", "Questionable", "Doubtful", "Probable"}
```

Write the three fixtures as minimal SportsDataIO-shaped JSON (base field names on SportsDataIO's CFB Injuries/News/Weather schemas — e.g. injuries carry `Team`/`Name`/`Position`/`Status`/`BodyPart`). Keep them tiny (2–3 rows).

- [ ] **Step 2: Run tests, verify they fail** (module/functions absent).
- [ ] **Step 3: Implement** the parsers as pure functions over decoded JSON, plus `_get` mirroring `cfb.espn._get`'s tenacity retry (3 attempts, exp backoff) with SportsDataIO's auth header. `_get` takes `api_key` as a param (no env access in the module).
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(cfb): SportsDataIO adapter + fixtures`).

## Task 2: Desk input assembler

**Files:**
- Create: `scripts/desk_inputs.py`
- Test: `tests/cfb/test_desk_inputs.py`

**Interfaces:**
- Consumes: Task 1 parsers; `predictions_current` (via Supabase read) for model margin/total/win-prob; `assets/cfb/schedules.parquet` for recent form; ESPN/CFBD closing-adjacent line for the pick-time market number.
- Produces: `build_bundle(games, model_rows, form_rows, injuries, news, weather) -> list[dict]` (PURE) — one entry per upcoming FBS game with: game_pk, matchup, commence_time, market_spread + market_total (pick-time line), model block (margin/total/win_prob/priors), recent-form block (last-N results, ATS/pace trend), and news block (injuries/weather/headlines for both teams). `main()` gathers the live inputs and writes the bundle JSON to `--out`.

- [ ] **Step 1: Write a failing test** for `build_bundle` — inject small in-memory model/form/news dicts for two games and assert each bundle entry carries the model block, the form block, the news block, and the pick-time `market_spread`/`market_total`. Assert games missing a line still appear (line fields None) and that only upcoming games (commence_time in the future relative to an injected `now`) are included.
- [ ] **Step 2: Run test, verify fail.**
- [ ] **Step 3: Implement** `build_bundle` (pure) + a thin `main()` that: reads `SPORTSDATA_API_KEY` (fail fast if unset), pulls injuries/news/weather via Task 1, reads `predictions_current` for CFB, computes recent form from `schedules.parquet`, attaches the pick-time line (reuse `cfb.espn` pickcenter for upcoming games), and writes the bundle to `--out` (default a scratch path). Network path is not unit-tested; `build_bundle` is.
- [ ] **Step 4: Run test, verify pass.**
- [ ] **Step 5: Commit** (`feat(cfb): desk input assembler`).

## Task 3: Decision-desk schema + db helpers

**Files:**
- Create: `db/migration_decision_desk.sql`
- Modify: `src/sportsmodel/db.py`
- Test: `tests/test_db_desk.py`

**Interfaces:**
- Produces: `upsert_desk_picks(records) -> int` (idempotent on `sport, game_pk, model_version`); `upsert_desk_pick_results(records) -> int` (idempotent on `sport, game_pk`).

- [ ] **Step 1: Write failing tests** (column-order tuple building, no live DB — mirror `tests/test_db_prediction_accuracy.py`'s FakeConn pattern) asserting the emitted SQL targets `desk_picks`/`desk_pick_results`, the ON CONFLICT keys are right, and the row tuple is built in column order incl. `conviction_tier`, `spread_side`, `spread_line`, `total_side`, `total_line`, `rationale`, `agent_notes`.
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Write the migration** — `desk_picks` (sport, game_pk, model_version, game_date, commence_time, matchup, ml_pick, spread_side, spread_line, total_side, total_line, confidence, conviction_tier, rationale, agent_notes jsonb, created_at; PK sport,game_pk,model_version); `desk_pick_results` (sport, game_pk, ml_correct, spread_cover, total_result, clv_spread, clv_total, graded_at; PK sport,game_pk); a `desk_current` view (upcoming picks for the site) and a `desk_record` view (accuracy + ATS-vs-line% + mean CLV, grouped by conviction_tier); RLS public-read grants mirroring `prediction_accuracy`. Then implement the two `db.py` helpers.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(desk): schema + db helpers`).

## Task 4: Pick writer (validate + immutable pre-kickoff upsert)

**Files:**
- Create: `scripts/write_desk_picks.py`
- Test: `tests/test_write_desk_picks.py`

**Interfaces:**
- Consumes: a picks JSON (the runbook's contract — Task 6), Task 3's `upsert_desk_picks`.
- Produces: `validate_picks(picks) -> list[str]` (PURE — returns a list of problems: unknown tier, bad side, missing line, out-of-range confidence, missing required field) and `writable_picks(picks, now) -> list[dict]` (PURE — drops any pick whose `commence_time <= now`, returning only pre-kickoff picks).

- [ ] **Step 1: Write failing tests**: a valid picks list passes `validate_picks` (empty problems); a pick with `conviction_tier="huge"` / `spread_side="left"` / `confidence=1.7` is flagged; `writable_picks` drops a pick whose commence_time is in the past and keeps a future one.
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `validate_picks` (tier ∈ {high,medium,low}; sides ∈ {home,away}/{over,under}; confidence ∈ [0,1]; required fields present), `writable_picks` (pre-kickoff filter), and a `main()` that loads the JSON, runs validation (aborts on problems, printing them), filters to pre-kickoff, and upserts via `upsert_desk_picks`. `main()` refuses to write anything if validation fails.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(desk): pick writer with pre-kickoff immutability`).

## Task 5: Forward grader (vs final + closing line + CLV)

**Files:**
- Create: `scripts/grade_desk_picks.py`
- Create: `.github/workflows/grade-desk-picks.yml`
- Test: `tests/test_grade_desk_picks.py`

**Interfaces:**
- Consumes: `desk_picks`, ESPN finals + closing line (`cfb.espn.fetch_final` already returns `market_spread`/`market_total`), Task 3's `upsert_desk_pick_results`.
- Produces: `grade_pick(pick, final) -> dict` (PURE) — `ml_correct` (picked winner won), `spread_cover` (did the picked side cover the **closing** line), `total_result` (did picked over/under hit vs the closing total), `clv_spread`/`clv_total` (closing line − pick-time line, signed toward the side taken; positive = the desk beat the close).

- [ ] **Step 1: Write failing tests** with hand-computed cases: a home-spread pick that covers the closing number → `spread_cover` win; an over pick vs a closing total it beats → `total_result` win; CLV signs (took home −3 at pick time, closed −5 → the desk's number was better → positive `clv_spread`). Push handling → None.
- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `grade_pick` (reuse the sportsbook-convention cover logic from `grade_predictions`/`backtest_cfb_priors.grade_vs_market`; ESPN closing line is sportsbook convention) and a `main()` that pulls ungraded `desk_picks` for finished games, grades them, and upserts `desk_pick_results`. Idempotent; skips games not yet final. Then author `grade-desk-picks.yml` (every 6h, `DATABASE_URL` only — no SportsData/LLM keys), mirroring `grade-predictions.yml`.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(desk): forward CLV grader + workflow`).

## Task 6: The desk runbook + picks-JSON contract

**Files:**
- Create: `docs/decision-desk-runbook.md`

- [ ] **Step 1:** Write the runbook the controller follows in-session: (1) run `desk_inputs.py` to produce the bundle; (2) dispatch three agents over the whole-slate bundle — **statistics** (model vs line + matchup), **sports-analyst** (recent form/momentum/ATS trends), **news** (injuries/weather/situational) — each returning a structured per-game brief; (3) a synthesis pass that, per game, emits ML/Spread/Total picks with a `conviction_tier`, a `confidence`, a rationale citing which agent drove each call, and the `agent_notes`; (4) write the picks JSON and run `write_desk_picks.py` (pre-kickoff only). Include the **exact picks-JSON schema** `validate_picks` enforces (fields, tiers, sides, confidence range) and the rule that every pick must cite at least one concrete fact (a named injury, a specific trend) — no bare narrative. Document the "best bets = high tier" convention.
- [ ] **Step 2:** Cross-check the schema in the runbook against `validate_picks` in Task 4 (they must match field-for-field).
- [ ] **Step 3: Commit** (`docs: decision-desk runbook + picks JSON contract`).

## Task 7: Front-end — Decision Desk section

**Files:**
- Modify: `/Users/ryan/Desktop/CappingAlpha/app.js`

- [ ] **Step 1:** Add a **Decision Desk** section to the CFB page reading `desk_current` (upcoming picks): per game show the matchup (+ logos, reusing `logoImg`), the three picks (ML / Spread `spread_side spread_line` / Total `total_side total_line`), the `conviction_tier` (badge), and the rationale (expandable). Highlight `high` tier as "best bets".
- [ ] **Step 2:** Add a running-record strip reading `desk_record`: ML accuracy, ATS-vs-line %, total-vs-line %, and **mean CLV**, segmented by tier — with an explicit "assistive · not a proven edge" label until the record earns otherwise.
- [ ] **Step 3:** `node -c app.js` to parse-check; verify against live data once the migration + a sample pick exist (or a stubbed `sb`).
- [ ] **Step 4: Commit** (external app.js is not git-tracked; note it in the report for the user to redeploy).

---

## Self-Review

**Spec coverage:** SportsDataIO provider (T1), the three agents' inputs (T2 bundle) + procedure (T6 runbook), storage (T3), pre-kickoff immutability (T4), forward CLV grading (T5), tiered every-game coverage + best-bets + assistive framing + running record (T3 views, T7 display). All spec sections map to a task.

**Placeholder scan:** SportsDataIO response shapes are pinned by committed fixtures (T1); the LLM agent reasoning is intentionally a runbook (T6), not code — not a placeholder. No "TBD"/vague-handling steps.

**Type consistency:** the picks-JSON contract is defined once (T6) and enforced once (`validate_picks`, T4); `desk_picks` columns (T3) match what the writer upserts (T4) and the grader/joins read (T5) and the views/display consume (T3/T7); `conviction_tier` ∈ {high,medium,low} and sides ∈ {home,away}/{over,under} are used consistently across T4/T6/T7.

**Ordering note:** T6 (runbook) and T4 (`validate_picks`) share the picks schema — T6 Step 2 cross-checks them. T7's live verification and any real desk run depend on the user provisioning `SPORTSDATA_API_KEY` and running the in-session procedure; grading (T5) is automated and needs only `DATABASE_URL`.
