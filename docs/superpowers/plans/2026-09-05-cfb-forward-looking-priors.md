# CFB Forward-Looking Priors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the CFB model's backward-looking cold start with a CFBD-sourced forward-looking preseason prior (SP+ base + fitted portal / coaching / QB-continuity / returning-starters / strength-of-schedule adjustments) that decays into the in-season ratings, kept independent of the betting market and shipped only after a backtest proves it.

**Architecture:** A new CFBD ingest (`build_cfb_priors.py`) writes one committed snapshot (`assets/cfb/priors.parquet`) of per-team-season forward-looking features. A pure module (`cfb/priors.py`) assembles those features into a preseason rating `R_pre` on the model's Elo scale, and a pure decaying-blend function folds `R_pre` into the existing league-agnostic ratings engine (seeds Week 0, fades to in-season ratings by mid-season). A backtest (`backtest_cfb_priors.py`) fits the weights + decay and runs the accuracy/edge/ablation gate, writing `assets/cfb/priors_weights.json`. Only then does `generate_cfb.py` consume the fitted config. No market data ever enters `R_pre` or the blend.

**Tech Stack:** Python 3.12, `httpx` + `tenacity` (CFBD calls, mirroring `ingest.odds`/`build_cfb_lines`), `pandas`/`pyarrow` (parquet), `duckdb` (existing), `pytest`. Reuses `sportsmodel.nfl.{elo,srs,ratings,points,gameline,shrink}` read-only (the league-agnostic engine CFB already uses).

**Spec:** `docs/superpowers/specs/2026-09-05-cfb-forward-looking-priors-design.md`

## Global Constraints

- **CFBD key is a GitHub Actions secret (`CFBD_API_KEY`); it is never printed, logged, or committed.** Live CFBD calls happen only in the ingest script's `main()` / the workflow. Parsers are tested against **committed sample-JSON fixtures**, never live calls. (Same constraint the `build_cfb_lines` work ran under.)
- **The model stays independent of the betting market.** No spread/total/moneyline/line-derived value may enter `R_pre`, the blend, or any priors feature. The market is used ONLY in the backtest's edge gate as the benchmark to beat — never as an input.
- **No hand-set magic numbers for the model.** Every prior weight (`w_portal`, `w_coach`, `w_qb`, `w_starters`, `w_sos_prior`, `w_sos_shift`) and the decay parameters are **fitted by the backtest** and stored in `assets/cfb/priors_weights.json`. Code reads them from that file (with a documented all-zeros fallback that reduces to today's behavior).
- **Pure vs I/O separation.** Parsing, feature assembly, `R_pre` computation, and the decay blend are pure functions (no network, no DB, no file reads) and are unit-tested directly. Live HTTP/parquet/DB I/O is confined to script `main()`s.
- **Reuse, don't fork, the engine.** `sportsmodel.nfl.{elo,srs,ratings,points,gameline,shrink}` are consumed read-only, exactly as `generate_cfb.py` / `backtest_cfb_*` already do.
- **TDD**: write the failing test first, minimal implementation, frequent commits.
- **Nothing labeled an "edge" ships without passing the spec's validation gate** (early-season accuracy improvement AND beating the closing line on the disagreement buckets).

---

## File Structure

- Create `tests/fixtures/cfb/cfbd_sp.json`, `cfbd_returning.json`, `cfbd_recruiting.json`, `cfbd_portal.json`, `cfbd_coaches.json`, `cfbd_games.json` — small hand-written sample CFBD responses (2–3 teams) for parser tests.
- Create `src/sportsmodel/cfb/cfbd.py` — pure CFBD response parsers (one per endpoint) + a retried `_get` client for `main()` use.
- Create `scripts/build_cfb_priors.py` — CFBD ingest `main()`; writes `assets/cfb/priors.parquet`.
- Create `src/sportsmodel/cfb/priors.py` — pure feature→`R_pre` assembly + the decaying-blend function + config dataclasses/loaders.
- Create `assets/cfb/priors.parquet` — generated snapshot (committed by the ingest workflow).
- Create `assets/cfb/priors_weights.json` — fitted weights + decay (written by the backtest).
- Create `scripts/backtest_cfb_priors.py` — fits weights/decay, runs accuracy + edge + ablation, writes `priors_weights.json`.
- Create `.github/workflows/build-cfb-priors.yml` — manual/seasonal ingest (needs `CFBD_API_KEY` + commits the parquet).
- Modify `scripts/generate_cfb.py` — load priors + fitted weights, apply the decaying blend to the margin rating.
- Modify `src/sportsmodel/cfb/__init__.py` if needed for exports.
- Tests: `tests/cfb/test_cfbd_parsers.py`, `tests/cfb/test_priors.py`, `tests/cfb/test_priors_blend.py`, and additions to `tests/cfb/test_generate_cfb.py`.

---

## Task 1: CFBD response parsers (pure) + fixtures

**Files:**
- Create: `src/sportsmodel/cfb/cfbd.py`
- Create: `tests/fixtures/cfb/cfbd_{sp,returning,recruiting,portal,coaches,games}.json`
- Test: `tests/cfb/test_cfbd_parsers.py`

**Interfaces:**
- Produces: `parse_sp(payload) -> dict[team,float]` (SP+ overall rating by team); `parse_returning(payload) -> dict[team,float]` (percent production returning); `parse_recruiting(payload) -> dict[team,float]` (class points/rank); `parse_portal(payload) -> dict[team,dict]` (incoming/outgoing rating sums → net); `parse_coaches(payload, season) -> dict[team,bool]` (first-year-HC flag for `season`); `parse_games(payload) -> list[dict]` (home_team, away_team, season for SoS). All keyed by CFBD school display name; a later step maps to ESPN ids via `cfb.teams`.

- [ ] **Step 1: Write failing tests** for each parser against a small fixture. Example:

```python
# tests/cfb/test_cfbd_parsers.py
import json, pathlib
from sportsmodel.cfb import cfbd
FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "cfb"

def test_parse_sp_maps_team_to_rating():
    payload = json.loads((FIX / "cfbd_sp.json").read_text())
    out = cfbd.parse_sp(payload)
    assert out["Alabama"] == 28.4          # 'rating' field
    assert "Kent State" in out

def test_parse_portal_nets_incoming_minus_outgoing():
    payload = json.loads((FIX / "cfbd_portal.json").read_text())
    out = cfbd.parse_portal(payload)
    # incoming ratings summed for destination, outgoing for origin
    assert out["Ole Miss"]["net"] == round(out["Ole Miss"]["in"] - out["Ole Miss"]["out"], 4)

def test_parse_coaches_flags_first_year_hc():
    payload = json.loads((FIX / "cfbd_coaches.json").read_text())
    out = cfbd.parse_coaches(payload, season=2024)
    assert out["Washington"] is True       # new HC in 2024
    assert out["Georgia"] is False         # returning HC
```

Write the fixtures as minimal but realistic CFBD JSON: `cfbd_sp.json` = `[{"team":"Alabama","rating":28.4,...},...]`; `cfbd_portal.json` = `[{"origin":"...","destination":"...","rating":0.9,...}]`; `cfbd_coaches.json` = `[{"school":"Washington","seasons":[{"year":2024,...}]},...]` (a coach whose earliest listed season at the school == the target season is first-year). Base shapes on the CFBD OpenAPI schema.

- [ ] **Step 2: Run tests, verify they fail** (`cfbd` module / functions not defined).
- [ ] **Step 3: Implement the parsers** in `cfbd.py` as pure functions over the decoded JSON. Add a `@retry`-decorated `_get(path, params)` (mirroring `ingest.odds._get` / `nfl.espn._get`: 3 attempts, exponential backoff) that reads the key via `os.environ["CFBD_API_KEY"]` with an `Authorization: Bearer` header — used only by callers in `main()`, not by the parsers.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(cfb): CFBD response parsers + fixtures`).

## Task 2: Priors ingest script → `priors.parquet`

**Files:**
- Create: `scripts/build_cfb_priors.py`
- Test: `tests/cfb/test_build_cfb_priors.py` (test the pure assembly of a per-team-season row from parser outputs; mock/skip network)

**Interfaces:**
- Consumes: Task 1 parsers; `cfb.teams` name→ESPN-id mapping (`cfbd_to_espn`).
- Produces: `assets/cfb/priors.parquet` with columns `season, team_espn_id, team_name, sp_rating, returning_pct, recruiting_points, portal_net, coach_first_year(bool), prior_sos, forward_sos_shift`. One row per FBS team-season, for every season in a `--seasons START END` range.

- [ ] **Step 1: Write a failing test** for `build_priors_rows(parsed, season) -> list[dict]` — a pure function that takes a dict of the six parser outputs for a season (plus prior/next-season SP+ for SoS) and returns the per-team rows, computing `portal_net`, `prior_sos` (avg of last season's opponents' SP+), and `forward_sos_shift` (this season's avg opponent SP+ minus last season's). Assert the row shape and one computed SoS value against a tiny hand-checked fixture.
- [ ] **Step 2: Run test, verify it fails.**
- [ ] **Step 3: Implement** `build_priors_rows` (pure) and a thin `main()` that: resolves `--seasons`, calls the retried CFBD `_get` per endpoint per season, runs the parsers, calls `build_priors_rows`, maps names→ESPN ids (dropping non-FBS/unmapped with a logged count), concatenates, and writes the parquet. `main()` requires `CFBD_API_KEY`; on missing key it exits with the same clear message pattern as `ingest.odds`.
- [ ] **Step 4: Run test, verify pass.** (Network path is exercised later by the workflow, not in unit tests.)
- [ ] **Step 5: Commit** (`feat(cfb): build_cfb_priors ingest -> priors.parquet`).

## Task 3: Preseason prior assembly (pure) → `R_pre`

**Files:**
- Create: `src/sportsmodel/cfb/priors.py`
- Test: `tests/cfb/test_priors.py`

**Interfaces:**
- Consumes: a priors row (dict from Task 2) + a `PriorWeights` config.
- Produces: `PriorWeights` dataclass (`sp_scale, sp_offset, w_portal, w_coach, w_qb, w_starters, w_sos_prior, w_sos_shift`); `load_weights(path) -> PriorWeights` (JSON loader, all-zeros-except-identity-SP+-map default if file missing); `zscore(values: dict) -> dict`; `preseason_rating(row, z, weights) -> float` where `z` carries the FBS-wide z-scored features for the season.

- [ ] **Step 1: Write failing tests**:

```python
# tests/cfb/test_priors.py
from sportsmodel.cfb.priors import PriorWeights, preseason_rating, zscore

def test_zscore_centers_and_scales():
    z = zscore({"A": 10.0, "B": 20.0, "C": 30.0})
    assert abs(z["B"]) < 1e-9                       # mean maps to 0
    assert z["C"] > 0 and z["A"] < 0

def test_preseason_rating_is_sp_base_plus_weighted_adjustments():
    w = PriorWeights(sp_scale=25.0, sp_offset=1500.0, w_portal=10.0, w_coach=-8.0,
                     w_qb=12.0, w_starters=6.0, w_sos_prior=0.0, w_sos_shift=0.0)
    row = {"sp_rating": 1.0, "coach_first_year": True, "qb_returning": True}
    z = {"portal_net": 2.0, "returning_starters": 1.0, "prior_sos": 0.0, "forward_sos_shift": 0.0}
    r = preseason_rating(row, z, w)
    # 1500 + 25*1.0 + 10*2.0 + (-8)*1 + 12*(+1) + 6*1.0
    assert r == 1500.0 + 25.0 + 20.0 - 8.0 + 12.0 + 6.0

def test_zero_weights_reduce_to_sp_only():
    w = PriorWeights(sp_scale=25.0, sp_offset=1500.0)   # rest default 0
    r = preseason_rating({"sp_rating": 2.0, "coach_first_year": False, "qb_returning": True},
                         {"portal_net": 5.0, "returning_starters": 5.0, "prior_sos": 5.0, "forward_sos_shift": 5.0}, w)
    assert r == 1500.0 + 50.0
```

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `PriorWeights` (frozen dataclass, all adjustment weights default `0.0`, `sp_scale` default `1.0`, `sp_offset` default `1500.0`), `zscore` (population std; if std==0 return all-zeros), `load_weights`, and `preseason_rating` = `sp_offset + sp_scale*sp_rating + w_portal*z.portal_net + w_coach*coach_flag + w_qb*qb_flag_signed + w_starters*z.returning_starters + w_sos_prior*z.prior_sos + w_sos_shift*z.forward_sos_shift`. `qb_flag_signed = +1 if qb_returning else -1`; `coach_flag = 1 if coach_first_year else 0`.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(cfb): preseason prior assembly (R_pre)`).

## Task 4: Decaying-prior blend (pure)

**Files:**
- Modify: `src/sportsmodel/cfb/priors.py`
- Test: `tests/cfb/test_priors_blend.py`

**Interfaces:**
- Produces: `DecayConfig` (`half_life_games: float`, `prior_floor: float = 0.0`); `prior_weight(games_played, cfg) -> float` in [0,1]; `blend_rating(r_pre, in_season_rating, games_played, cfg) -> float`.

- [ ] **Step 1: Write failing tests**:

```python
# tests/cfb/test_priors_blend.py
from sportsmodel.cfb.priors import DecayConfig, prior_weight, blend_rating

def test_prior_dominates_at_zero_games():
    assert prior_weight(0, DecayConfig(half_life_games=3)) == 1.0
    assert blend_rating(1800.0, 1500.0, 0, DecayConfig(half_life_games=3)) == 1800.0

def test_prior_halves_at_half_life():
    w = prior_weight(3, DecayConfig(half_life_games=3))
    assert abs(w - 0.5) < 1e-9

def test_in_season_dominates_late():
    r = blend_rating(1800.0, 1500.0, 20, DecayConfig(half_life_games=3))
    assert abs(r - 1500.0) < 1.0   # prior almost gone by 20 games
```

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `prior_weight = max(prior_floor, 0.5 ** (games_played / half_life_games))` and `blend_rating = w*r_pre + (1-w)*in_season_rating`.
- [ ] **Step 4: Run tests, verify pass.**
- [ ] **Step 5: Commit** (`feat(cfb): decaying-prior blend`).

## Task 5: Backtest — fit weights/decay + accuracy + edge + ablation

**Files:**
- Create: `scripts/backtest_cfb_priors.py`
- Create: `assets/cfb/priors_weights.json` (output of this task's run)
- Test: `tests/cfb/test_backtest_cfb_priors.py` (test the pure metric helpers: margin/total MAE by week, ATS-vs-closing, and the disagreement-bucket win rate)

**Interfaces:**
- Consumes: `assets/cfb/{schedules,lines,priors}.parquet`, the Task 3/4 pure functions, and the existing `backtest_cfb_gameline`/`backtest_cfb_ratings` walk-forward core.
- Produces: `assets/cfb/priors_weights.json` (fitted `PriorWeights` + `DecayConfig`); printed accuracy, edge, and ablation tables.

- [ ] **Step 1: Write failing tests** for the pure metric helpers:

```python
def test_ats_vs_closing_counts_cover_correctly():
    from backtest_cfb_priors import ats_result
    # model picks home -7; closing home -3; home wins by 5 -> home covers closing -> model (home) wins ATS
    assert ats_result(model_margin=7, closing_home_spread=-3, actual_margin=5) == "win"

def test_disagreement_bucket_winrate():
    from backtest_cfb_priors import bucket_winrate
    rows = [{"gap": 8, "ats": "win"}, {"gap": 9, "ats": "loss"}, {"gap": 1, "ats": "win"}]
    wr = bucket_winrate(rows, min_gap=5)
    assert wr == 0.5   # only the two gap>=5 games count
```

(Load the script via `importlib`, the pattern `tests/test_grade_predictions.py` uses.)

- [ ] **Step 2: Run tests, verify fail.**
- [ ] **Step 3: Implement** `backtest_cfb_priors.py`:
  - Walk-forward per season/week: compute in-season ratings from prior weeks (existing engine), compute `R_pre` from `priors.parquet` (Task 3), blend (Task 4), predict margin/total (existing `expected_margin`/`expected_total`/`build_gameline`).
  - **Fit** `PriorWeights` (adjustment weights + `sp_scale/offset`) and `DecayConfig.half_life_games` by minimizing early-season (Weeks 1–5) margin MAE via a coarse grid / `scipy`-free coordinate search over a documented range; hold out alternating seasons to avoid overfit.
  - **Accuracy table:** per-week margin/total MAE, current model vs prior-seeded.
  - **Edge table:** join `lines.parquet` closing spread; bucket by `|model_margin − closing|`; report ATS win% and mean CLV per bucket.
  - **Ablation:** refit with each of {portal, coach, qb, starters, sos_prior, sos_shift} zeroed; report the accuracy/edge delta each contributes.
  - Write the fitted config to `assets/cfb/priors_weights.json` **only for factors that improve out-of-sample early-season MAE**; zero out (drop) the rest and say so in the printout.
- [ ] **Step 4: Run tests, verify pass**; run the backtest itself (locally against committed parquets; `priors.parquet` for past seasons must exist first — produced by the Task 2 workflow run, see Task 7). Record the accuracy/edge/ablation results in the SDD ledger.
- [ ] **Step 5: Commit** (`feat(cfb): priors backtest + fitted weights`), including `priors_weights.json`.

## Task 6: Wire priors into the live producer

**Files:**
- Modify: `scripts/generate_cfb.py`
- Test: `tests/cfb/test_generate_cfb.py` (extend)

**Interfaces:**
- Consumes: `assets/cfb/{priors.parquet,priors_weights.json}` + Task 3/4 functions.

- [ ] **Step 1: Write a failing test** asserting that, with a non-zero prior for a team and `games_played=0`, `build_game_rows` produces a margin driven by `R_pre` (not the base Elo), and that with `games_played` large the prior's influence is negligible (reuse the existing pure `build_game_rows` test harness, injecting a small priors table + weights).
- [ ] **Step 2: Run test, verify fail.**
- [ ] **Step 3: Implement**: in `_season_to_date_ratings`/`build_game_rows`, after computing the in-season Elo/SRS margin rating per team, blend with `R_pre` (looked up by team-season from `priors.parquet`) via `blend_rating(..., games_played, decay)`. Load `priors_weights.json` once in `main()`; if the file or a team's prior is missing, fall back to `prior_weight=0` (today's behavior). Market-independence: nothing here reads lines/odds.
- [ ] **Step 4: Run test, verify pass**; run `generate_cfb.py` dry (no `DATABASE_URL`) and confirm it still prints a game count.
- [ ] **Step 5: Commit** (`feat(cfb): generate_cfb uses forward-looking priors`).

## Task 7: Ingest workflow

**Files:**
- Create: `.github/workflows/build-cfb-priors.yml`

- [ ] **Step 1:** Author a `workflow_dispatch` (+ optional yearly `schedule`) workflow mirroring `build-cfb-lines.yml`: checkout, uv sync, run `python scripts/build_cfb_priors.py --seasons <START> <END>` with `env: CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}`, then commit the updated `assets/cfb/priors.parquet` back (same commit-bot step `build-cfb-lines`/`refresh-profiles` use).
- [ ] **Step 2:** Validate YAML shape locally (structure only).
- [ ] **Step 3: Commit** (`ci: build-cfb-priors workflow`).
- [ ] **Step 4 (operational, not code):** The user (or the controller, if permitted) dispatches this once to populate `priors.parquet` for the backtest seasons + current season, then Task 5's backtest runs, then Task 6 goes live.

---

## Self-Review

**Spec coverage:** SP+ base (T1/T3), portal (T1/T2/T3), coaching (T1/T3), QB continuity + returning starters (T2/T3), SoS prior + shift (T2/T3), decaying blend (T4), independence from market (Global Constraints + T5/T6), validation gate incl. edge + ablation (T5). All spec sections map to a task.

**Placeholder scan:** empirical values (weights, decay half-life) are deliberately *outputs of Task 5*, not placeholders — code ships with an all-zeros/identity default that reduces to today's model, and the fitted `priors_weights.json` is produced and committed by the backtest. CFBD response shapes are pinned by committed fixtures in Task 1.

**Type consistency:** `PriorWeights`/`DecayConfig` defined in T3/T4 and consumed unchanged in T5/T6; `preseason_rating`/`blend_rating`/`prior_weight` signatures stable across tasks; priors rows use the same column names from T2 through T6.

**Ordering note:** Task 7 (workflow) is authored last but must be *run* to produce `priors.parquet` before Task 5's backtest can execute — called out in Task 5 Step 4 and Task 7 Step 4.
