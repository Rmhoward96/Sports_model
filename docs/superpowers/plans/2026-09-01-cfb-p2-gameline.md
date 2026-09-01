# CFB P2 — Game-Line Distributions + Market Gate — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** CFB game-line distributions (spread + total) with market shrinkage, gated on beating the CFB closing line OOS. Reuses `nfl.points/shrink/gameline` read-only.

**Spec:** docs/superpowers/specs/2026-09-01-cfb-p2-gameline.md

## Global Constraints

- CFB-only; reuse `nfl.points/shrink/gameline` read-only; additive. No NFL/MLB changes.
- `CFBD_API_KEY` read from env only — never hardcoded, never printed, never in git. CFBD is the one-time historical pull; live P3 uses the Odds API.
- Spread stored home-margin convention (positive = home favored).
- `PYTHONPATH=src uv run --no-sync` for all runs.
- Task 4 gate: ship P3 only if the blend beats market-only OOS on MAE, or cover/over accuracy clears ~52.4%.

---

### Task 1: CFBD→ESPN team-name matcher

**Files:** Modify `src/sportsmodel/cfb/teams.py`; Test `tests/cfb/test_teams.py`

**Interfaces:** `cfbd_to_espn(name: str) -> str | None` — normalize a CFBD "school" name to our ESPN team id (or `None` if unmatched). Build a `{normalized_displayName: espn_id}` index from `fbs_teams.json` (lowercase, strip the mascot — the ESPN displayName is "School Mascot", CFBD gives "School"); add a small alias table for the ~10–15 that differ (e.g. "UConn"↔"Connecticut", "Ole Miss"↔"Mississippi", "NC State"↔"North Carolina State", "App State"↔"Appalachian State", "Miami"↔"Miami" (FL) vs "Miami (OH)", "Southern Miss", "UMass", "UTSA", "FIU", "UL Monroe"/"Louisiana Monroe", "Sam Houston", "Hawai'i").

- [ ] **Step 1: failing test** — a set of known CFBD names map to the right ESPN id (verify each id is in `load_fbs_ids()`); an unknown/FCS name → `None`. Include the tricky aliases above.
- [ ] **Step 2:** run → fail. **Step 3:** implement (build the stripped-displayName index once; alias table; return id or None). **Step 4:** run → pass. **Step 5:** commit.

### Task 2: CFB lines fetcher + build workflow

**Files:** Create `scripts/build_cfb_lines.py`, `.github/workflows/build-cfb-lines.yml`

**Interfaces:** `build_cfb_lines.py` — reads `CFBD_API_KEY` from env; for `--seasons 2015..2024`, `GET https://api.collegefootballdata.com/lines?year=<Y>&seasonType=regular` with header `Authorization: Bearer <key>`. Each item: `{season, week, homeTeam, awayTeam, lines:[{provider, spread, overUnder}...]}` — take the **median** `spread` and `overUnder` across providers. CFBD `spread` is the **home** spread in book convention (home favored ⇒ negative); store `market_spread = -spread` (home-margin convention: home favored ⇒ positive) and `market_total = overUnder`. Match `homeTeam`/`awayTeam` via `cfb.teams.cfbd_to_espn`; **drop rows where either side is unmatched or FCS** (no FBS-FCS lines). Write `assets/cfb/lines.parquet` (`season, week, home_team, away_team, market_spread, market_total`). Print: rows written, rows dropped (unmatched), seasons covered.

- [ ] **Step 1:** write `build_cfb_lines.py` (never log the key; fail clearly if `CFBD_API_KEY` unset).
- [ ] **Step 2:** write `.github/workflows/build-cfb-lines.yml` — `workflow_dispatch` only; checkout, uv sync, `env: CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}`, run the script, then commit + push `assets/cfb/lines.parquet` back to the branch (`permissions: contents: write`; git config the actions bot; skip commit if no diff).
- [ ] **Step 3:** commit both. (The actual data pull happens when the user dispatches the workflow after adding the secret — see Coordination below.)

### Task 3: Game-line backtest + gate

**Files:** Create `scripts/backtest_cfb_gameline.py`, `assets/cfb/gameline.json`, `docs/superpowers/reports/2026-09-01-cfb-gameline-backtest.md`
**Depends on:** `assets/cfb/lines.parquet` existing (Task 2 workflow run).

**Interfaces:** Leak-free walk-forward. Reuse P1 ratings (`assets/cfb/rating.json` → `nfl.ratings.expected_margin`) for model_margin; `nfl.points.compute_points_ratings`/`expected_total` for model_total. Join to `assets/cfb/lines.parquet` on `(season, week, home_team, away_team)`. Fit `sigma_margin`, `sigma_total`, and `ShrinkParams` `w(week)` for margin + total against the market; write `assets/cfb/gameline.json` (shape of `assets/nfl/gameline.json`). Mirror `scripts/backtest_nfl_gameline.py`.

- [ ] **Step 1:** write the backtest (leak-free; fit σ + w; `build_gameline` to discretize). **Step 2:** run it; report OOS **margin MAE + total MAE** (model-only / blend / market-only) AND **cover accuracy + over/under accuracy vs the closing line**. **This is the gate — record in the ledger.** **Step 3:** commit script + `gameline.json` + the report note.

## Coordination (out-of-band, human step)

Between Task 2 and Task 3: the user adds the `CFBD_API_KEY` GitHub secret, then the `build-cfb-lines.yml` workflow is dispatched to produce + commit `assets/cfb/lines.parquet`. Task 3 runs only after that asset exists.
