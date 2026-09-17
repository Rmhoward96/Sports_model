# NFL Drive-Based Sim Engine (Sub-project B) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A player-level, drive-based Monte Carlo that fills a per-game result
container (scores + per-player stat arrays), producing correlated game
distributions, player stat distributions (raw material for props/C), and an
analytic-vs-sim disagreement signal.

**Architecture:** New `sportsmodel.sim.nfl` package (spec → rates → inputs →
kernel → aggregation), reusing `sim/engine.py`'s generic score helpers. Two
scripts (`generate_sim_nfl.py`, `backtest_sim_nfl.py`) and a migration. Fully
leakage-free (walk-forward) inputs from nflverse.

**Tech Stack:** Python, numpy, nfl_data_py (nflverse), psycopg (Supabase),
pytest, uv.

**Spec:** docs/superpowers/specs/2026-09-17-nfl-sim-engine-design.md

## Global Constraints

- **NFL only.** No CFB, no MLB changes.
- **Leakage-free:** every rate/usage/efficiency input for a game in season S,
  week W uses only data strictly before (S,W). Backtests are walk-forward.
- **Reuse `sim/engine.py`** score helpers (`home_win_prob`, `total_pmf`,
  `margin_pmf`, `pred_scores`) via duck-typing on `.home_score`/`.away_score`;
  do NOT fork them. Player aggregation is NFL-specific (engine's
  `player_prop_dists` is MLB-hardcoded — do not touch it).
- **Pure kernels/helpers** (no IO) so they're unit-testable; IO only in
  `rates.py` fetch wrappers, the two scripts, and db functions.
- **Honesty:** B runs ALONGSIDE the analytic game-line model; it does not
  replace ML/spread/total predictions nor the Pinnacle-anchored +EV base prob.
- **v1 player markets:** `pass_yds`, `rush_yds`, `rec_yds`, `receptions`,
  `anytime_td`. **n_sims** default 10_000. Secrets (DATABASE_URL) live only in
  CI; the user runs SQL migrations.
- Test/commit per task. Match surrounding code style.

## File Structure

- `src/sportsmodel/sim/nfl/__init__.py`
- `src/sportsmodel/sim/nfl/spec.py` — dataclasses: `TeamRates`, `PlayerInput`,
  `NflGameSpec`, and result container `NflGameSims`.
- `src/sportsmodel/sim/nfl/kernel.py` — `sample_drive`, `attribute_offense`,
  `simulate_game`.
- `src/sportsmodel/sim/nfl/aggregate.py` — `nfl_player_prop_dists`,
  `disagreement`.
- `src/sportsmodel/sim/nfl/rates.py` — nflverse → `TeamRates`/player inputs
  (leakage-free), pure aggregation + thin fetch.
- `src/sportsmodel/sim/nfl/inputs.py` — assemble `NflGameSpec` (rates + roster +
  injury zeroing).
- `scripts/generate_sim_nfl.py`, `scripts/backtest_sim_nfl.py`
- `db/migration_nfl_sim.sql`
- `src/sportsmodel/db.py` — `upsert_nfl_sim`, `upsert_nfl_player_sim`.
- Tests mirror each module under `tests/sim/nfl/`.

---

### Task 1: Spec + result container

**Files:** Create `src/sportsmodel/sim/nfl/__init__.py`,
`src/sportsmodel/sim/nfl/spec.py`; Test `tests/sim/nfl/test_spec.py`.

**Interfaces — Produces:**
- `TeamRates`: `drive_outcomes: dict[str,float]` (keys `td,fg,punt,turnover,downs,end`; sum≈1), `pass_rate: float`, `drives_per_game: float`, `rz_td_rate: float`.
- `PlayerInput`: `player_id:str`, `name:str`, `pos:str`, `target_share:float`, `carry_share:float`, `ypt:float` (yards/target), `ypc:float` (yards/carry), `catch_rate:float`, `td_share:float`.
- `NflGameSpec`: `home/away: TeamRates`, `home_players/away_players: list[PlayerInput]`, `home_team/away_team: str`.
- `NflGameSims` (dataclass): `home_score, away_score: np.ndarray`, `player_stats: dict[str, dict[str, np.ndarray]]` (player_id → market → array).

- [ ] **Step 1: failing test** — construct each dataclass; assert fields/defaults; assert `NflGameSims` score helpers from engine work on it:
```python
import numpy as np
from sportsmodel.sim.nfl.spec import TeamRates, PlayerInput, NflGameSpec, NflGameSims
from sportsmodel.sim.engine import home_win_prob, margin_pmf
def test_nflgamesims_duck_types_into_engine():
    s = NflGameSims(home_score=np.array([24,17]), away_score=np.array([20,21]),
                    player_stats={})
    assert home_win_prob(s) == 0.5
    assert margin_pmf(s)["kind"] == "margin"
def test_team_rates_and_spec_construct():
    tr = TeamRates(drive_outcomes={"td":.25,"fg":.15,"punt":.4,"turnover":.1,"downs":.05,"end":.05},
                   pass_rate=0.58, drives_per_game=11.0, rz_td_rate=0.6)
    assert abs(sum(tr.drive_outcomes.values())-1.0) < 1e-9
```
- [ ] **Step 2:** run → fail (module missing).
- [ ] **Step 3:** implement the dataclasses (frozen where sensible; `NflGameSims` mutable).
- [ ] **Step 4:** run → pass.
- [ ] **Step 5:** commit `feat(sim/nfl): game spec + result container`.

---

### Task 2: Drive outcome sampling (pure kernel helper)

**Files:** `src/sportsmodel/sim/nfl/kernel.py`; Test `tests/sim/nfl/test_kernel_drive.py`.

**Interfaces — Consumes:** `TeamRates`. **Produces:**
`sample_drive(off: TeamRates, deff: TeamRates, rng) -> tuple[str, int]` returns
(outcome, points). Combine offense/defense drive-outcome rates (elementwise
average then renormalize for v1), draw the outcome from the resulting
multinomial via `rng.random()`, map TD→7, FG→3, else 0.

- [ ] **Step 1: failing test** — deterministic with seeded rng; points map;
  distribution over many draws ≈ combined rates (±tol); always returns a valid
  outcome key.
```python
import numpy as np
from sportsmodel.sim.nfl.kernel import sample_drive
from sportsmodel.sim.nfl.spec import TeamRates
def _tr(**o): 
    base={"td":.2,"fg":.15,"punt":.4,"turnover":.15,"downs":.05,"end":.05}; base.update(o); return TeamRates(base,0.58,11.0,0.6)
def test_sample_drive_points_map_and_dist():
    rng=np.random.default_rng(0); off=_tr(); deff=_tr()
    outs=[sample_drive(off,deff,rng) for _ in range(20000)]
    td=sum(1 for o,_ in outs if o=="td")/20000
    assert 0.15 < td < 0.25
    assert all((p==7)==(o=="td") and (p==3)==(o=="fg") for o,p in outs)
```
- [ ] Steps 2–5 (fail, implement, pass, commit `feat(sim/nfl): drive outcome sampler`).

---

### Task 3: Offense player attribution (pure kernel helper)

**Files:** `kernel.py`; Test `tests/sim/nfl/test_kernel_attribution.py`.

**Interfaces — Produces:** `attribute_offense(players: list[PlayerInput],
n_pass: int, n_rush: int, n_off_tds: int, rng) -> dict[str, dict[str,int]]` —
allocate `n_pass` targets by `target_share` (multinomial), `n_rush` carries by
`carry_share`; yards per target ~ draw around `ypt` (e.g. Normal, floored at a
sensible min, receptions gated by `catch_rate`), yards per carry ~ around `ypc`;
allocate `n_off_tds` by `td_share`; QB pass_yds = sum of team rec_yds; return per
player `{pass_yds,rush_yds,rec_yds,receptions,td}`.

- [ ] **Step 1: failing test** — shares respected over many draws; receptions ≤
  targets; QB pass_yds ≈ Σ rec_yds; TD count conserved; deterministic with rng.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): offense player attribution`).

---

### Task 4: `simulate_game` (assemble drives → NflGameSims)

**Files:** `kernel.py`; Test `tests/sim/nfl/test_kernel_simulate.py`.

**Interfaces — Produces:** `simulate_game(spec: NflGameSpec, n_sims: int, rng)
-> NflGameSims`. Per sim: draw each team's drive count (Poisson/round around
`drives_per_game`), sample each drive, split offensive plays pass/run via
`pass_rate` to feed `attribute_offense`, accumulate scores + per-player arrays.

- [ ] **Step 1: failing test** — shapes = n_sims; scores non-negative and in a
  sane NFL range (mean total 30–60); `home_win_prob` between 0 and 1; a
  dominant-offense spec beats a weak one; player_stats keys = roster ids;
  reproducible with a fixed seed.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): drive-based simulate_game`).

---

### Task 5: NFL aggregation (player prop dists + disagreement)

**Files:** `src/sportsmodel/sim/nfl/aggregate.py`; Test `tests/sim/nfl/test_aggregate.py`.

**Interfaces — Consumes:** `NflGameSims`, engine `stat_pmf`/`_pmf_mean`.
**Produces:**
- `nfl_player_prop_dists(sims, market_max: dict) -> dict[pid, dict[market, {kind,pmf,mean}]]` for `pass_yds,rush_yds,rec_yds,receptions` (pmf via `stat_pmf`) and `anytime_td` (`{pmf:[1-p,p], mean}` with `p=P(td>=1)`).
- `disagreement(analytic_home_win_prob: float, sim_home_win_prob: float) -> float` = `abs(a-b)`.

- [ ] **Step 1: failing test** — pmf sums to 1 per market; `anytime_td` p in [0,1]; mean matches array mean; `disagreement(0.6,0.5)==0.1`.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): player prop aggregation + disagreement`).

---

### Task 6: Rate aggregation from nflverse (leakage-free) — pure core + thin fetch

**Files:** `src/sportsmodel/sim/nfl/rates.py`; Test `tests/sim/nfl/test_rates.py`.

**Interfaces — Produces:**
- `team_rates_from_pbp(pbp_df, upto_season, upto_week) -> dict[team, TeamRates]` — PURE over a passed DataFrame; filter to rows strictly before (S,W); aggregate drive outcomes, pass rate, drives/game, RZ TD rate per team (offense; defense symmetric). 
- `player_inputs_from_weekly(weekly_df, snaps_df, upto_season, upto_week) -> dict[team, list[PlayerInput]]` — PURE; target/carry/snap shares + ypt/ypc/catch_rate/td_share from prior games (recency-weighted ok).
- `fetch_nflverse(seasons) -> dict` — thin IO wrapper (`import_pbp_data`, `import_weekly_data`, `import_snap_counts`) used by scripts only.

- [ ] **Step 1: failing test** — small synthetic DataFrames → correct filtered aggregates; a row at/after (S,W) is excluded (leakage guard); shares sum to ≈1 per team.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): leakage-free nflverse rate aggregation`).

---

### Task 7: Assemble `NflGameSpec` for a game (+ injury zeroing)

**Files:** `src/sportsmodel/sim/nfl/inputs.py`; Test `tests/sim/nfl/test_inputs.py`.

**Interfaces — Consumes:** `team_rates_*`, `player_inputs_*`, an injuries dict
(from `sportsmodel.nfl.injuries_nflverse`, `{team:[{player,status,...}]}`).
**Produces:** `build_spec(home_team, away_team, rates, players, injuries) ->
NflGameSpec` — drop/zero OUT players and **renormalize** remaining usage shares
so the sim reflects real inactives.

- [ ] **Step 1: failing test** — an OUT WR's target share is removed and the
  remaining WRs' shares renormalize to 1; DOUBTFUL treated as OUT (config);
  spec has both teams' rates+players.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): assemble game spec + injury zeroing`).

---

### Task 8: DB tables + upserts

**Files:** `db/migration_nfl_sim.sql`; `src/sportsmodel/db.py`; Test `tests/test_db_nfl_sim.py` (FakeConn pattern from `tests/test_db_ev.py`).

**Interfaces — Produces:**
- Migration: `nfl_sim` (sport implicit; game_pk, model_version, matchup,
  commence_time, sim_home_win_prob, sim_margin, sim_total, disagreement,
  created_at; PK game_pk,model_version) + `nfl_sim_current` view
  (commence_time>now, DISTINCT ON latest) + RLS public read. `nfl_player_sim`
  (game_pk, player_id, name, pos, team, market, mean, prob_over payload as JSONB
  or pmf) + view. GRANT anon.
- `upsert_nfl_sim(records)`, `upsert_nfl_player_sim(records)` — mirror
  `upsert_ev_picks` (created_at frozen; JSON columns via `json.dumps`).

- [ ] **Step 1: failing test** — tuple built in column order; JSON columns
  serialized; `created_at` not in DO UPDATE; empty→0.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): nfl_sim tables + upserts`).

---

### Task 9: `generate_sim_nfl.py` (current slate → DB)

**Files:** `scripts/generate_sim_nfl.py`; Test `tests/sim/nfl/test_generate_sim_nfl.py` (unit-test the pure assembly seam; the IO main is thin).

**Behavior:** read upcoming games from `predictions_current` (has
analytic home_win_prob + commence_time), fetch nflverse, build specs (leakage =
current season/week), run `simulate_game`, aggregate game dists + player dists,
compute `disagreement` vs the analytic `home_win_prob`, upsert `nfl_sim` +
`nfl_player_sim`. Print `games=/players=/mean_disagreement=`.

- [ ] **Step 1: failing test** — a pure `assemble_sim_rows(games, sims_by_game,
  analytic_by_game)` returns rows with correct disagreement + fields.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): generate_sim_nfl slate runner`).

---

### Task 10: `backtest_sim_nfl.py` (walk-forward ship gate)

**Files:** `scripts/backtest_sim_nfl.py`; Test `tests/sim/nfl/test_backtest_sim_nfl.py` (pure metric helpers).

**Behavior:** walk-forward over completed seasons: for each game, build a
leakage-free spec, simulate, and record sim win-prob/margin/total vs actual, and
player projected mean vs actual for the v1 markets. Emit: **game** Brier + margin/
total MAE (vs the analytic model's numbers for reference), margin↔total
correlation vs empirical; **player** mean MAE + a calibration/coverage check
(share of actuals below the sim's p50/p90 ≈ 0.5/0.9). This is the ship gate.

- [ ] **Step 1: failing test** — pure helpers: Brier, coverage (`share_below(q,
  actuals, quantiles)`), correlation.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): walk-forward sim backtest`).

---

### Task 11: Desk disagreement integration

**Files:** `scripts/desk_inputs.py` (add sim disagreement to the NFL bundle);
`scripts/synthesize_desk_picks.py` (cap conviction on high disagreement);
tests alongside.

**Behavior:** NFL desk bundle gains `sim` block `{home_win_prob, margin, total,
disagreement}` (from `nfl_sim_current`, LEFT-joined; None when absent). The
synthesis methodology gains: when `disagreement` exceeds a threshold (default
0.15), **cap `conviction_tier` at "medium"** and note the model/sim split in the
rationale. Pure `apply_disagreement_cap(pick, disagreement, threshold)` is unit-
tested; bundle wiring is thin.

- [ ] **Step 1: failing test** — `apply_disagreement_cap` downgrades high→medium
  when disagreement>threshold, leaves low/medium unchanged, no-op when sim
  absent.
- [ ] Steps 2–5 (implement; commit `feat(sim/nfl): desk disagreement conviction cap`).

---

### Task 12: Workflow + docs

**Files:** `.github/workflows/generate-sim-nfl.yml` (schedule before desk-auto-nfl
so disagreement is fresh; DATABASE_URL secret); update
`docs/desk-auto-runbook.md` (sim step + disagreement); note the migration in the
spec's rollout.

- [ ] **Step 1:** add workflow (mirror `generate-nfl.yml`), `uv run python
  scripts/generate_sim_nfl.py`.
- [ ] **Step 2:** doc update.
- [ ] **Step 3:** commit `ci(sim/nfl): scheduled sim generation + docs`.

## Self-Review

- **Spec coverage:** correlated dists (T4/T5), player dists (T5), disagreement
  (T5/T9/T11), leakage-free inputs (T6/T7), validation gate (T10), storage
  (T8), slate runner (T9), integration (T11/T12) — all mapped. ✔
- **Type consistency:** `TeamRates`/`PlayerInput`/`NflGameSpec`/`NflGameSims`
  names used identically across T1–T9. Score helpers reused via duck-typing. ✔
- **No placeholders:** each task has concrete interfaces + test intent; modeling
  algorithms specified. Implementers fill exact numeric constants from data.
- **Ordering:** 1→2→3→4 (kernel) then 5 (agg), 6→7 (data), 8 (db), 9 (runner),
  10 (backtest), 11 (desk), 12 (ci). T10/T11 depend on 9; 9 on 4/5/7/8.
