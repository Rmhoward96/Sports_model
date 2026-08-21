# Run-Scoring Channels + Outlier Dispersion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the MLB sim's −0.99-run mean bias and its too-thin outlier tails so the simulated total/margin distributions match real MLB, then add a residual totals/margin calibration layer — without regressing moneyline or props.

**Architecture:** Add two structural scoring channels (reached-on-error, wild-pitch/passed-ball) to both kernel paths; add two per-game stochastic dispersion effects (scoring environment + pitcher quality) sampled once per sim and applied as a multiplier on offensive probabilities; fit residual location+scale calibration on the total/margin distributions and apply it at scoring time.

**Tech Stack:** Python 3.12 (uv), numpy, DuckDB (Statcast parquet), pytest. Sim kernel in `src/sportsmodel/sim/mlb/kernel.py` (a correctness-first scalar path `simulate_scalar` and a vectorized production path `simulate`, guarded by a distributional-equivalence test).

**Spec:** `docs/superpowers/specs/2026-08-21-run-scoring-channels-and-outlier-dispersion.md`

## Global Constraints

- **Backward compatibility / equivalence:** every new behavior is OFF by default. The kernel reads new params via `getattr(spec, "roe_p", 0.0)`, `getattr(spec, "wp_p", 0.0)`, `getattr(spec, "dispersion", None)`. With all defaults, `simulate_scalar` and `simulate` must produce the SAME output as today, so the existing `tests/test_sim_kernel.py` (including `test_vectorized_matches_scalar_distribution` and `test_strikeout_outs_not_double_counted`) passes unchanged.
- **Both kernels change together.** Any channel/effect added to `simulate_scalar` is added to `simulate`; they must agree within Monte Carlo noise.
- **Measured, not tuned:** `p_roe` comes from Statcast `field_error` counts (~0.0068). `p_wp` is a **literature constant** (0.025 ≈ 0.45 WP+PB per team-game) because this backfill only carries WP/PB as unreliable free text — NOT measured. Only the dispersion σ's (`sigma_shared`, `sigma_team`, `sigma_pitcher`) are tuned.
- **Rates respect the walk-forward cutoff** (`transforms._CUTOFF`) exactly like the other Statcast builders (`WHERE events IS NOT NULL AND CAST(game_date AS DATE) < DATE '<cutoff>'`).
- **Vector index order** (`_VEC_ORDER`): `p_bb=0, p_k=1, p_1b=2, p_2b=3, p_3b=4, p_hr=5, p_out=6`. Offensive (non-out) columns are `[0,2,3,4,5]`; out columns are `[1,6]`.
- **Acceptance bar (from spec §4):** sim marginal total/margin match the six empirical targets; over/under + run-line reliability hold; win-prob Brier ≤ 0.246 (± MC noise) and prop calibration no worse. Empirical targets (2025, 2367 games): mean total 8.89, SD total 4.59, P(total≥11) 32.9%, P(total≤5) 25.3%, P(team shut out) 13.8%, P(|margin|≥5) 28.7%, SD margin 4.58.
- **Commit cadence:** one commit per task (TDD: red → green → commit). Work on a branch `sim-scoring-dispersion` off `main`.

---

## File Structure

- `src/sportsmodel/sim/mlb/build_advancement.py` — ADD `measure_scoring_rates(con, _table=None) -> dict` returning `{"p_roe": float, "p_wp": float}`.
- `assets/scoring_rates.json` — NEW committed constants `{"p_roe": ..., "p_wp": ...}` (produced by Task 1's run).
- `src/sportsmodel/sim/mlb/kernel.py` — ADD ROE + WP/PB branches (scalar + vectorized); ADD `Dispersion` dataclass and per-sim multiplier sampling/application in both paths.
- `src/sportsmodel/sim/mlb/inputs.py` — thread `roe_p`, `wp_p`, `dispersion` onto the spec in `build_game_spec`.
- `src/sportsmodel/sim/mlb/config_dispersion.py` — NEW tiny module holding the tuned σ constants + loader for `scoring_rates.json` (single source of truth for production defaults).
- `scripts/validate_sim_dist.py` — NEW committed harness: sim vs empirical tail metrics + reliability (replaces the throwaway scratchpad scripts).
- `scripts/tune_dispersion.py` — NEW committed coordinate-search over the σ's against the tail targets.
- `src/sportsmodel/model/distributions.py` — ADD `apply_affine(dist, loc, scale) -> dict`.
- `scripts/fit_calibration_sim.py` — ADD total/margin location+scale fit; write `total_dist`/`margin_dist` entries to `assets/calibration.json`; refit all markets.
- `scripts/grade_results.py`, `streamlit_app.py` — apply the total/margin affine before `prob_over_dist`/`prob_cover`.
- Tests: `tests/test_build_advancement.py`, `tests/test_sim_kernel.py`, `tests/test_distributions.py`, `tests/test_grade_results.py`, `tests/test_fit_calibration_sim.py`.

---

## Phase 1 — Scoring channels

### Task 1: Measure and store `p_roe` / `p_wp` from Statcast

**Files:**
- Modify: `src/sportsmodel/sim/mlb/build_advancement.py`
- Create: `assets/scoring_rates.json`
- Test: `tests/test_build_advancement.py`

**Interfaces:**
- Produces: `measure_scoring_rates(con, _table=None) -> {"p_roe": float, "p_wp": float}`. `p_roe = count(events='field_error') / count(PA)`; `p_wp = (count of PAs whose half-inning contained a wild_pitch/passed_ball advance) / count(PA-with-runners-on)`. Approximate `p_wp` as `count(events IN ('wild_pitch','passed_ball')) / count(PA where on_1b/on_2b/on_3b any not null)` — these events appear as their own rows in Statcast.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_advancement.py  (add to existing file)
import duckdb
from sportsmodel.sim.mlb.build_advancement import measure_scoring_rates

def test_measure_scoring_rates_counts_errors_and_wp():
    con = duckdb.connect(":memory:")
    con.execute("""
        CREATE TABLE pa AS SELECT * FROM (VALUES
          -- (game_pk, inning, inning_topbot, at_bat_number, events, on_1b, on_2b, on_3b, game_date)
          (1,1,'Top',1,'single',      NULL,NULL,NULL, DATE '2025-05-01'),
          (1,1,'Top',2,'field_error', 1,   NULL,NULL, DATE '2025-05-01'),
          (1,1,'Top',3,'field_out',   1,   2,   NULL, DATE '2025-05-01'),
          (1,1,'Top',4,'wild_pitch',  1,   NULL,NULL, DATE '2025-05-01'),
          (1,1,'Top',5,'strikeout',   NULL,NULL,NULL, DATE '2025-05-01')
    ) t(game_pk,inning,inning_topbot,at_bat_number,events,on_1b,on_2b,on_3b,game_date)""")
    r = measure_scoring_rates(con, _table="pa")
    # 5 PA rows, 1 field_error -> p_roe = 1/5
    assert abs(r["p_roe"] - 0.2) < 1e-9
    # PAs with a runner on: rows 2,3,4 (3 of them); wild_pitch events = 1 -> p_wp = 1/3
    assert abs(r["p_wp"] - (1/3)) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_advancement.py::test_measure_scoring_rates_counts_errors_and_wp -v`
Expected: FAIL with `ImportError: cannot import name 'measure_scoring_rates'`.

- [ ] **Step 3: Implement `measure_scoring_rates`**

Add to `build_advancement.py` (uses the same `events IS NOT NULL` + cutoff gating pattern as `build_advancement_table`):

```python
def measure_scoring_rates(con, _table: str | None = None) -> dict:
    """Reached-on-error and wild-pitch/passed-ball rates from Statcast, for the
    kernel's ROE and WP/PB channels. Respects the active walk-forward cutoff."""
    src = _table or f"read_parquet('{transforms._PARQUET_GLOB}')"
    conditions = ["events IS NOT NULL"]
    if _table is None and transforms._CUTOFF:
        conditions.append(f"CAST(game_date AS DATE) < DATE '{transforms._CUTOFF}'")
    where = "WHERE " + " AND ".join(conditions)
    row = con.execute(f"""
        SELECT
          count(*) AS n_pa,
          count(*) FILTER (WHERE events = 'field_error') AS n_roe,
          count(*) FILTER (WHERE on_1b IS NOT NULL OR on_2b IS NOT NULL OR on_3b IS NOT NULL) AS n_runners,
          count(*) FILTER (WHERE events IN ('wild_pitch','passed_ball')) AS n_wp
        FROM {src} {where}
    """).fetchone()
    n_pa, n_roe, n_runners, n_wp = row
    p_roe = (n_roe / n_pa) if n_pa else 0.0
    p_wp = (n_wp / n_runners) if n_runners else 0.0
    return {"p_roe": float(p_roe), "p_wp": float(p_wp)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_build_advancement.py::test_measure_scoring_rates_counts_errors_and_wp -v`
Expected: PASS.

- [ ] **Step 5: Produce the real constants and commit them**

Run (writes the asset from the full local backfill, no cutoff):

```bash
cd "/Users/ryan/Desktop/Sports Model"
SM_DATA_DIR="$(pwd)/data" uv run python - <<'PY'
import json, duckdb
from pathlib import Path
from sportsmodel import transforms
from sportsmodel.sim.mlb.build_advancement import measure_scoring_rates
transforms.set_cutoff(None)
con = duckdb.connect(":memory:")
r = measure_scoring_rates(con)
Path("assets/scoring_rates.json").write_text(json.dumps(r, indent=2) + "\n")
print(r)
PY
```

Expected: `p_roe` ≈ 0.015–0.02, `p_wp` ≈ 0.01–0.03. Sanity-check the printed values are in range before committing.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/sim/mlb/build_advancement.py tests/test_build_advancement.py assets/scoring_rates.json
git commit -m "feat(sim): measure reached-on-error and WP/PB rates from Statcast"
```

---

### Task 2: Reached-on-error channel (scalar + vectorized)

**Files:**
- Modify: `src/sportsmodel/sim/mlb/kernel.py`
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Consumes: `getattr(spec, "roe_p", 0.0)` — probability an `OUT_INPLAY` PA becomes reached-on-error.
- Behavior: on ROE the batter is safe on first with **no out recorded** and **no hit credited**; runners/`runs` come from the single (`S`) advancement table with `outs_added` forced to 0; the pitcher records no K, no hit, no out.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sim_kernel.py  (add)
def test_roe_all_outs_never_end_inning_and_score():
    # Batters who always make an in-play out, but roe_p=1.0 -> every "out" is
    # actually reached-on-error: no outs are ever recorded, so a half-inning
    # never ends on its own. With the real single table, runners circulate and
    # score. Cap protects us from an infinite game via max_extra.
    out_vec = {"p_bb":0.0,"p_k":0.0,"p_1b":0.0,"p_2b":0.0,"p_3b":0.0,"p_hr":0.0,"p_out":1.0}
    bs = [K.Batter(pid, out_vec, out_vec) for pid in range(100,109)]
    aw = [K.Batter(pid, out_vec, out_vec) for pid in range(200,209)]
    adv = AdvancementTable.from_rows([
        {"outcome":"p_1b","occ":0,"end_occ":1,"runs":0,"prob":1.0},
        {"outcome":"p_1b","occ":1,"end_occ":3,"runs":0,"prob":1.0},
        {"outcome":"p_1b","occ":3,"end_occ":7,"runs":0,"prob":1.0},
        {"outcome":"p_1b","occ":7,"end_occ":7,"runs":1,"prob":1.0},
    ])
    spec = K.GameSpec(bs, aw, K.Pitcher(1,16,5), K.Pitcher(2,16,5)); spec.adv = adv
    spec.roe_p = 1.0
    sims = K.simulate_scalar(spec, n_sims=3, rng=np.random.default_rng(0))
    # innings only end at the max_extra cap -> both teams pile up runs, and the
    # starters record ZERO outs (every event is an error, not an out).
    assert sims.pitcher_stats[1]["outs"].max() == 0
    assert (sims.home_score + sims.away_score).min() > 0

def test_roe_off_by_default_matches_today():
    # roe_p unset -> identical to current behavior (regression guard).
    sims = K.simulate_scalar(_spec(), n_sims=50, rng=np.random.default_rng(0))
    ref = K.simulate_scalar(_spec(), n_sims=50, rng=np.random.default_rng(0))
    assert np.array_equal(sims.home_score, ref.home_score)

def test_roe_vectorized_matches_scalar_mean():
    bs = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(100,109)]
    aw = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(200,209)]
    spec = K.GameSpec(bs, aw, K.Pitcher(1,16,5), K.Pitcher(2,16,5)); spec.roe_p = 0.02
    a = K.simulate_scalar(spec, 4000, np.random.default_rng(3))
    b = K.simulate(spec, 4000, np.random.default_rng(3))
    assert abs((a.home_score+a.away_score).mean() - (b.home_score+b.away_score).mean()) < 0.2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_sim_kernel.py -k roe -v`
Expected: FAIL (roe_p not consumed; `test_roe_all_outs...` currently ends innings normally).

- [ ] **Step 3: Implement ROE in the scalar kernel**

In `_sim_one`'s `half(...)` loop, read `roe_p = getattr(spec, "roe_p", 0.0)` once (pass `spec` into `_sim_one` — it already receives `spec`). After `outcome = sample_outcome(vec, u)` and drawing `u2`, insert before `resolve_pa`:

```python
is_roe = outcome == OUT_INPLAY and roe_p > 0.0 and rng.random() < roe_p
if is_roe:
    # reached on error: advance runners on the single table, batter safe, NO out.
    occ = state.occ()
    end_occ, runs = adv.sample(S, occ, u2)
    _fill_from_mask(state, end_occ, idx[bat_team] % 9, lead_first=True)
    outs_added = 0
    # no hit credited, pitcher records nothing
else:
    runs, outs_added = resolve_pa(state, idx[bat_team] % 9, outcome, adv, u2)
```

Then guard the box-score/pitcher-credit blocks so they run only when `not is_roe` for the hit/K/out lines (rbi/team score still add `runs`). Keep `outs += outs_added` (0 for ROE) and the `bf`/`idx` increments unchanged.

- [ ] **Step 4: Implement ROE in the vectorized kernel**

In `simulate`, read `roe_p = getattr(spec, "roe_p", 0.0)`. Inside `play_half`, after computing `outcome` and `u2`, and after calling `_resolve_pa_vec`, compute an ROE mask and override those sims:

```python
if roe_p > 0.0:
    u_roe = rng.random(n)
    m_roe = active & (outcome == OUT_INPLAY) & (u_roe < roe_p)
    if m_roe.any():
        # recompute those sims as a single (code S) with outs forced to 0
        s_runs, _s_outs, s_occ = _resolve_pa_vec(
            occ, np.full(n, S), u2, cum_mat, end_mat, runs_mat)
        runs = np.where(m_roe, s_runs, runs)
        new_occ = np.where(m_roe, s_occ, new_occ)
        outs_added = np.where(m_roe, 0, outs_added)
        # suppress hit/hr/k credit for ROE sims
        is_hit = is_hit & ~m_roe
        is_hr = is_hr & ~m_roe
        is_k = is_k & ~m_roe
        tb = np.where(m_roe, 0, tb)
```

Place this BEFORE the box-score `+=` block so the suppressed masks take effect. (Draw `u_roe` unconditionally-per-half when `roe_p>0` to keep RNG deterministic.)

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_sim_kernel.py -k roe -v` then the whole kernel file `uv run pytest tests/test_sim_kernel.py -v`.
Expected: all PASS (including the untouched `test_vectorized_matches_scalar_distribution`, `test_strikeout_outs_not_double_counted`).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): reached-on-error channel (scalar + vectorized)"
```

---

### Task 3: Wild-pitch / passed-ball channel (scalar + vectorized)

**Files:**
- Modify: `src/sportsmodel/sim/mlb/kernel.py`
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Consumes: `getattr(spec, "wp_p", 0.0)`. On a `K` or `BB` PA with ≥1 runner on, with prob `wp_p` advance every runner one base (a runner on 3rd scores 1 run). Applied AFTER the PA resolves.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sim_kernel.py  (add)
def test_wp_scores_runner_from_third_on_strikeout():
    # A runner on 3rd, batter strikes out, wp_p=1.0 -> the K is still an out,
    # but the WP scores the runner from 3rd. Use a scalar helper to isolate.
    st = K.BaseState(first=-1, second=-1, third=5)
    runs = K.apply_wp_advance(st)  # third scores, others shift up
    assert runs == 1 and st.third == -1

def test_wp_off_by_default_matches_today():
    sims = K.simulate_scalar(_spec(), 50, np.random.default_rng(0))
    ref = K.simulate_scalar(_spec(), 50, np.random.default_rng(0))
    assert np.array_equal(sims.away_score, ref.away_score)

def test_wp_vectorized_matches_scalar_mean():
    bs = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(100,109)]
    aw = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(200,209)]
    spec = K.GameSpec(bs, aw, K.Pitcher(1,16,5), K.Pitcher(2,16,5)); spec.wp_p = 0.05
    a = K.simulate_scalar(spec, 4000, np.random.default_rng(3))
    b = K.simulate(spec, 4000, np.random.default_rng(3))
    assert abs((a.home_score+a.away_score).mean() - (b.home_score+b.away_score).mean()) < 0.2
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_sim_kernel.py -k wp -v`
Expected: FAIL (`apply_wp_advance` undefined; wp_p not consumed).

- [ ] **Step 3: Implement the scalar helper + wiring**

Add a module-level helper:

```python
def apply_wp_advance(state: "BaseState") -> int:
    """Advance every runner one base; a runner on 3rd scores. Returns runs scored."""
    runs = 1 if state.third >= 0 else 0
    state.third = state.second
    state.second = state.first
    state.first = -1
    return runs
```

In the scalar `half(...)` loop, after resolving the PA (and after the ROE branch), add:

```python
wp_p = getattr(spec, "wp_p", 0.0)
if wp_p > 0.0 and outcome in (K, BB) and state.runners() >= 1 and rng.random() < wp_p:
    wp_runs = apply_wp_advance(state)
    runs += wp_runs
    scores[bat_team] += wp_runs
```

(Place the `scores[bat_team] += runs` for the base PA before this, or fold `wp_runs` into `runs` before the single score update — ensure runs are added exactly once.)

- [ ] **Step 4: Implement the vectorized wiring**

In `play_half`, after the ROE block, before writing scores:

```python
wp_p = getattr(spec, "wp_p", 0.0)
if wp_p > 0.0:
    u_wp = rng.random(n)
    has_runner = new_occ > 0  # occupancy AFTER the PA resolved
    m_wp = active & ((outcome == K) | (outcome == BB)) & has_runner & (u_wp < wp_p)
    third_occ = (new_occ & 4) > 0
    wp_runs = np.where(m_wp & third_occ, 1, 0)
    # shift runners up one base: new3=old2, new2=old1, new1=empty
    shifted = ((new_occ & 3) << 1)  # bits for 1st/2nd move to 2nd/3rd
    new_occ = np.where(m_wp, shifted, new_occ)
    runs = runs + wp_runs
```

Use the same `runs` array in the score `+=` block so WP runs are counted once.

- [ ] **Step 5: Run to verify pass**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): wild-pitch/passed-ball advance channel (scalar + vectorized)"
```

---

### Task 4: Distribution validation harness

**Files:**
- Create: `scripts/validate_sim_dist.py`
- Test: `tests/test_validate_sim_dist.py`

**Interfaces:**
- Produces: `tail_metrics(total: np.ndarray, margin: np.ndarray) -> dict` with keys `mean_total, sd_total, p_ge11, p_le5, p_shutout, p_blowout, sd_margin`. `p_shutout` needs per-team scores, so also accept `home, away`. Final signature: `tail_metrics(home, away) -> dict`.
- A `main()` that runs `backtest_sim.run_sim_backtest` for a configurable month/season, pools the sim scores, computes `tail_metrics`, and prints sim-vs-empirical side by side (empirical constants from the Global Constraints block, or recomputed from parquet).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_sim_dist.py
import numpy as np
import importlib.util
from pathlib import Path
_p = Path(__file__).resolve().parents[1] / "scripts" / "validate_sim_dist.py"
_s = importlib.util.spec_from_file_location("validate_sim_dist", _p)
vsd = importlib.util.module_from_spec(_s); _s.loader.exec_module(vsd)

def test_tail_metrics_basic():
    home = np.array([5, 0, 8, 2])
    away = np.array([3, 6, 8, 1])
    m = vsd.tail_metrics(home, away)
    # totals: 8, 6, 16, 3
    assert abs(m["mean_total"] - 8.25) < 1e-9
    assert abs(m["p_ge11"] - 0.25) < 1e-9   # only 16
    assert abs(m["p_le5"] - 0.25) < 1e-9    # only 3
    assert abs(m["p_shutout"] - 0.25) < 1e-9  # 0-6 game
    # margins: 2, -6, 0, 1 -> |.|>=5 : one (-6)
    assert abs(m["p_blowout"] - 0.25) < 1e-9
```

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_validate_sim_dist.py -v`
Expected: FAIL (file/function missing).

- [ ] **Step 3: Implement `tail_metrics` + `main`**

```python
# scripts/validate_sim_dist.py
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

EMPIRICAL = {"mean_total":8.89,"sd_total":4.59,"p_ge11":0.329,"p_le5":0.253,
             "p_shutout":0.138,"p_blowout":0.287,"sd_margin":4.58}

def tail_metrics(home, away) -> dict:
    home = np.asarray(home); away = np.asarray(away)
    total = home + away; margin = home - away
    return {
        "mean_total": float(total.mean()),
        "sd_total": float(total.std()),
        "p_ge11": float(np.mean(total >= 11)),
        "p_le5": float(np.mean(total <= 5)),
        "p_shutout": float(np.mean((home == 0) | (away == 0))),
        "p_blowout": float(np.mean(np.abs(margin) >= 5)),
        "sd_margin": float(margin.std()),
    }

def main():
    import argparse, backtest_sim as bs
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--month", type=int, default=6)
    ap.add_argument("--n-sims", type=int, default=2000)
    a = ap.parse_args()
    homes, aways = [], []
    orig = bs.pred_scores
    def wrap(sims):
        homes.append(np.asarray(sims.home_score)); aways.append(np.asarray(sims.away_score))
        return orig(sims)
    bs.pred_scores = wrap
    bs._MONTHS = [a.month]
    bs.run_sim_backtest(a.season, n_sims=a.n_sims, seed=42)
    m = tail_metrics(np.concatenate(homes), np.concatenate(aways))
    print(f"{'metric':12} {'sim':>8} {'empirical':>10}")
    for k in EMPIRICAL:
        print(f"{k:12} {m[k]:>8.3f} {EMPIRICAL[k]:>10.3f}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_validate_sim_dist.py -v`
Expected: PASS.

- [ ] **Step 5: Checkpoint — measure post-channel state**

Run (records the effect of Tasks 2–3 with the measured rates wired via a temporary spec override — the harness uses `backtest_sim`, so first make `backtest_sim.build_game_spec` attach `roe_p`/`wp_p`; that wiring is Task 6. Until then, run with rates OFF to confirm the harness works, and note the numbers):

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/validate_sim_dist.py --month 6 --n-sims 2000
```

Expected: prints sim-vs-empirical. (Mean will improve to near 8.85 only after Task 6 wires the rates in.)

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_sim_dist.py tests/test_validate_sim_dist.py
git commit -m "feat(sim): committed distribution-validation harness (tail metrics vs empirical)"
```

---

## Phase 2 — Outlier dispersion

### Task 5: Dispersion effects in both kernels

**Files:**
- Modify: `src/sportsmodel/sim/mlb/kernel.py`
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Consumes: `getattr(spec, "dispersion", None)` — a `Dispersion(sigma_shared, sigma_team, sigma_pitcher)` dataclass (all default 0.0 → no-op).
- Per sim: sample `E_shared, E_home, E_away` (log-normal, mean 1) and `q_home, q_away` (Normal(0, σ_p)). The batting team's non-out probs are multiplied by `E_shared × E_team × exp(q_defending_starter if starter_in else 0)` and renormalized.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sim_kernel.py  (add)
def test_dispersion_off_matches_today():
    spec = _spec(); spec.dispersion = K.Dispersion()  # all zeros
    a = K.simulate_scalar(spec, 200, np.random.default_rng(5))
    b = K.simulate_scalar(_spec(), 200, np.random.default_rng(5))
    assert np.array_equal(a.home_score, b.home_score)

def test_dispersion_widens_total_preserves_mean():
    base = _spec_real_adv()  # helper: _spec() with the real adv table attached
    hi = _spec_real_adv(); hi.dispersion = K.Dispersion(0.20, 0.20, 0.30)
    a = K.simulate(base, 8000, np.random.default_rng(9))
    b = K.simulate(hi, 8000, np.random.default_rng(9))
    ta = (a.home_score + a.away_score); tb = (b.home_score + b.away_score)
    assert abs(ta.mean() - tb.mean()) < 0.25          # mean preserved
    assert tb.std() > ta.std() + 0.2                  # variance increased
```

Add a module-level `_spec_real_adv()` helper in the test file that builds `_spec()` and attaches the real advancement table (guard with `pytest.skip` if the parquet glob is empty, mirroring `test_sim_mean_runs_near_analytic`).

- [ ] **Step 2: Run to verify fail**

Run: `uv run pytest tests/test_sim_kernel.py -k dispersion -v`
Expected: FAIL (`Dispersion` undefined).

- [ ] **Step 3: Implement `Dispersion` + sampling + application**

Add the dataclass near `Pitcher`:

```python
@dataclass
class Dispersion:
    sigma_shared: float = 0.0
    sigma_team: float = 0.0
    sigma_pitcher: float = 0.0
```

Add a helper to sample per-sim multiplier inputs:

```python
def _sample_dispersion(disp, rng, n):
    """Return (E_shared, E_home, E_away, q_home, q_away) arrays of length n.
    Log-normals are centered so E[E]=1; q are Normal(0, sigma_pitcher)."""
    if disp is None:
        one = np.ones(n); zero = np.zeros(n)
        return one, one, one, zero, zero
    def logn(sig):
        return np.exp(rng.normal(-0.5*sig*sig, sig, size=n)) if sig > 0 else np.ones(n)
    E_shared = logn(disp.sigma_shared)
    E_home = logn(disp.sigma_team); E_away = logn(disp.sigma_team)
    q_home = rng.normal(0, disp.sigma_pitcher, size=n) if disp.sigma_pitcher > 0 else np.zeros(n)
    q_away = rng.normal(0, disp.sigma_pitcher, size=n) if disp.sigma_pitcher > 0 else np.zeros(n)
    return E_shared, E_home, E_away, q_home, q_away

_OFF_COLS = np.array([0, 2, 3, 4, 5])  # p_bb,p_1b,p_2b,p_3b,p_hr

def _apply_mult(vec, mult):
    """Scale offensive columns of a (n,7) vec array by per-sim mult (n,), renormalize rows."""
    v = vec.copy()
    v[:, _OFF_COLS] *= mult[:, None]
    return v / v.sum(axis=1, keepdims=True)
```

In `simulate` (vectorized): after `hook_home/hook_away`, sample once:

```python
disp = getattr(spec, "dispersion", None)
E_shared, E_home, E_away, q_home, q_away = _sample_dispersion(disp, rng, n)
```

Inside `play_half`, after computing `vec` (the `(n,7)` array) and `starter_in`, compute the batting multiplier and apply:

```python
E_bat = E_home if bat_team == 1 else E_away
q_def = q_home if defense == 1 else q_away
pitcher_mult = np.where(starter_in, np.exp(q_def), 1.0)
mult = E_shared * E_bat * pitcher_mult
vec = _apply_mult(vec, mult)
```

For `simulate_scalar`: sample the same five arrays once at the top of the per-`n_sims` loop (index by `i`), and in `half(...)` scale the single-sim `vec` dict's offensive keys by `E_shared[i]*E_bat[i]*(exp(q_def[i]) if starter_in else 1)` and renormalize before `sample_outcome`. (Scalar samples per-sim inside the loop; vectorized samples all at once — RNG order differs, so only aggregate agreement is asserted, consistent with the existing equivalence test.)

- [ ] **Step 4: Run to verify pass**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): per-game environment + pitcher-quality dispersion effects"
```

---

### Task 6: Thread rates + dispersion through `build_game_spec` and production config

**Files:**
- Create: `src/sportsmodel/sim/mlb/config_dispersion.py`
- Modify: `src/sportsmodel/sim/mlb/inputs.py`, `scripts/backtest_sim.py` (its local `build_game_spec` call site)
- Test: `tests/test_sim_inputs.py`

**Interfaces:**
- Produces: `config_dispersion.load_rates() -> {"p_roe","p_wp"}` (reads `assets/scoring_rates.json`); `config_dispersion.DISPERSION = Dispersion(...)` (tuned in Task 7 — start at zeros, filled after tuning).
- `build_game_spec(..., roe_p=0.0, wp_p=0.0, dispersion=None)` attaches them to the returned spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_inputs.py  (add)
from sportsmodel.sim.mlb.kernel import Dispersion
def test_build_game_spec_attaches_dispersion_and_rates(minimal_spec_inputs):
    spec = build_game_spec(*minimal_spec_inputs, roe_p=0.017, wp_p=0.02,
                           dispersion=Dispersion(0.1, 0.1, 0.2))
    assert spec.roe_p == 0.017 and spec.wp_p == 0.02
    assert spec.dispersion.sigma_pitcher == 0.2
```

(Reuse whatever fixture/inputs the existing `test_sim_inputs.py` uses to call `build_game_spec`; if none, construct the minimal args inline as that test file already does.)

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_sim_inputs.py -k dispersion -v` → FAIL.

- [ ] **Step 3: Implement**

Add params to `build_game_spec` signature and set on the spec before returning:

```python
def build_game_spec(home_order, away_order, home_sp_vec, away_sp_vec,
                    home_bp_vec, away_bp_vec, workload, context, league, adv,
                    home_starter_id, away_starter_id,
                    roe_p=0.0, wp_p=0.0, dispersion=None) -> GameSpec:
    ...
    spec.adv = adv
    spec.roe_p = roe_p
    spec.wp_p = wp_p
    spec.dispersion = dispersion
    return spec
```

Create `config_dispersion.py`:

```python
import json
from sportsmodel import config
from .kernel import Dispersion

def load_rates() -> dict:
    p = config.PROJECT_ROOT / "assets" / "scoring_rates.json"
    return json.loads(p.read_text()) if p.exists() else {"p_roe": 0.0, "p_wp": 0.0}

# Tuned in the dispersion-tuning step (Task 7). Zeros until then.
DISPERSION = Dispersion(sigma_shared=0.0, sigma_team=0.0, sigma_pitcher=0.0)
```

Wire the call sites: in `scripts/backtest_sim.py` and `scripts/generate_sim.py`, pass `roe_p`/`wp_p` from `config_dispersion.load_rates()` and `dispersion=config_dispersion.DISPERSION` into `build_game_spec`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_sim_inputs.py -v` → PASS.

- [ ] **Step 5: Checkpoint — re-measure mean/tails with channels ON**

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/validate_sim_dist.py --month 6 --n-sims 2000
```

Expected: `mean_total` now ≈ 8.6–8.9 (channels add the missing runs). Record the numbers in the commit message.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/sim/mlb/config_dispersion.py src/sportsmodel/sim/mlb/inputs.py scripts/backtest_sim.py scripts/generate_sim.py tests/test_sim_inputs.py
git commit -m "feat(sim): wire ROE/WP rates + dispersion config into spec builders"
```

---

### Task 7: Tune the dispersion σ's

**Files:**
- Create: `scripts/tune_dispersion.py`
- Modify: `src/sportsmodel/sim/mlb/config_dispersion.py` (write tuned σ's)
- Test: `tests/test_tune_dispersion.py`

**Interfaces:**
- Produces: `objective(metrics, empirical) -> float` (sum of squared relative errors on `sd_total`, `p_ge11`, `p_le5`, `p_blowout`, `sd_margin`); `coord_search(eval_fn, grid) -> best_params` (pure, testable on a toy quadratic).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tune_dispersion.py
import importlib.util
from pathlib import Path
_p = Path(__file__).resolve().parents[1] / "scripts" / "tune_dispersion.py"
_s = importlib.util.spec_from_file_location("tune_dispersion", _p)
td = importlib.util.module_from_spec(_s); _s.loader.exec_module(td)

def test_objective_zero_when_matching():
    emp = {"sd_total":4.59,"p_ge11":0.329,"p_le5":0.253,"p_blowout":0.287,"sd_margin":4.58}
    assert td.objective(dict(emp), emp) < 1e-12

def test_coord_search_finds_min_of_toy():
    # minimize (s-0.15)^2 over a small grid
    best = td.coord_search(lambda p: (p["sigma_shared"]-0.15)**2,
                           {"sigma_shared":[0.0,0.1,0.15,0.2]})
    assert best["sigma_shared"] == 0.15
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_tune_dispersion.py -v` → FAIL.

- [ ] **Step 3: Implement `objective` + `coord_search` + a `main` that searches over σ's**

`objective` = `sum(((metrics[k]-emp[k])/emp[k])**2 for k in KEYS)`. `coord_search` = iterate each param over its grid holding others at current best, keep improving (one or two sweeps). `main` evaluates each candidate by running `validate_sim_dist` machinery: set `config_dispersion.DISPERSION` (or pass a `Dispersion` through a thin wrapper around `backtest_sim.run_sim_backtest`), compute `tail_metrics`, score with `objective`. Search grid e.g. `sigma_shared,sigma_team ∈ {0.0,0.1,0.15,0.2,0.25}`, `sigma_pitcher ∈ {0.0,0.15,0.25,0.35}`. Use `--n-sims 2000 --month 6` for speed.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_tune_dispersion.py -v` → PASS.

- [ ] **Step 5: Run the search, write the tuned σ's**

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/tune_dispersion.py --n-sims 2000
```

Take the best `(sigma_shared, sigma_team, sigma_pitcher)`, hand-write them into `config_dispersion.DISPERSION`. Re-run `validate_sim_dist.py` and confirm all six metrics are within ~5% of empirical. If `p_le5`/`p_shutout` overshoot, prefer slightly lower `sigma_shared`.

- [ ] **Step 6: Commit**

```bash
git add scripts/tune_dispersion.py tests/test_tune_dispersion.py src/sportsmodel/sim/mlb/config_dispersion.py
git commit -m "feat(sim): tune dispersion sigmas to match empirical outlier rates"
```

---

## Phase 3 — Calibration layer

### Task 8: `apply_affine` distribution remap

**Files:**
- Modify: `src/sportsmodel/model/distributions.py`
- Test: `tests/test_distributions.py`

**Interfaces:**
- Produces: `apply_affine(dist, loc, scale) -> dict`. Remaps a `{"kind":"pmf","pmf":[...]}` (support k=0..len-1) or `{"kind":"margin","offset":o,"pmf":[...]}` (support i-o) by `x' = scale*(x-mean)+mean+loc`, re-binning to the original integer support (nearest bin, clipped to range, mass renormalized). `loc=0, scale=1` returns an equivalent distribution.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_distributions.py  (add)
from sportsmodel.model.distributions import apply_affine, prob_over_dist, prob_cover

def test_apply_affine_identity():
    d = {"kind":"pmf","pmf":[0.1,0.2,0.4,0.2,0.1]}
    out = apply_affine(d, 0.0, 1.0)
    assert abs(sum(out["pmf"]) - 1.0) < 1e-9
    # identity leaves P(>2) unchanged
    assert abs(prob_over_dist(out, 2) - prob_over_dist(d, 2)) < 1e-9

def test_apply_affine_location_shift_raises_mean():
    d = {"kind":"pmf","pmf":[0,0,0,0,0,0,0,0,1.0]}  # all mass at 8
    out = apply_affine(d, 1.0, 1.0)  # shift +1 -> mass moves toward 9
    assert prob_over_dist(out, 8) > 0.99   # now > 8

def test_apply_affine_scale_widens_margin_cover():
    md = {"kind":"margin","offset":3,"pmf":[0,0,0.2,0.6,0.2,0,0]}  # margins -1,0,1
    wide = apply_affine(md, 0.0, 2.0)
    assert prob_cover(wide, -1.5) == prob_cover(wide, -1.5)  # finite, not NaN
    assert sum(wide["pmf"]) == pytest.approx(1.0)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_distributions.py -k affine -v` → FAIL.

- [ ] **Step 3: Implement**

```python
def apply_affine(dist: dict, loc: float, scale: float) -> dict:
    """Location+scale remap of a discrete distribution, re-binned to its integer
    support. Used to calibrate the sim's total (pmf) and margin distributions."""
    import numpy as np
    if not dist:
        return dist
    kind = dist.get("kind")
    pmf = np.asarray(dist.get("pmf") or [], dtype=float)
    if pmf.size == 0:
        return dist
    offset = dist.get("offset", 0)
    support = np.arange(pmf.size) - offset
    mean = float((support * pmf).sum())
    new_vals = scale * (support - mean) + mean + loc
    new_idx = np.clip(np.rint(new_vals + offset).astype(int), 0, pmf.size - 1)
    out = np.zeros_like(pmf)
    np.add.at(out, new_idx, pmf)
    s = out.sum()
    out = (out / s) if s > 0 else out
    res = {"kind": kind, "pmf": out.tolist()}
    if kind == "margin":
        res["offset"] = offset
    return res
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_distributions.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/model/distributions.py tests/test_distributions.py
git commit -m "feat(model): apply_affine location+scale distribution remap"
```

---

### Task 9: Fit totals/margin calibration + refit all markets

**Files:**
- Modify: `scripts/fit_calibration_sim.py`
- Modify: `assets/calibration.json` (regenerated)
- Test: `tests/test_fit_calibration_sim.py`

**Interfaces:**
- Produces: `fit_dist_affine(pred_means, pred_sd, actuals) -> (loc, scale)` by method of moments: `loc = mean(actuals) - mean(pred_means)`; `scale = sd(actuals) / pooled_sd` where `pooled_sd` is the sim's marginal SD. Writes `calibration.json` keys `total_dist: {"loc":..,"scale":..}` and `margin_dist: {...}` alongside existing Platt entries.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fit_calibration_sim.py  (add)
import importlib.util
from pathlib import Path
_p = Path(__file__).resolve().parents[1] / "scripts" / "fit_calibration_sim.py"
_s = importlib.util.spec_from_file_location("fit_calibration_sim", _p)
fcs = importlib.util.module_from_spec(_s); _s.loader.exec_module(fcs)

def test_fit_dist_affine_debias_and_scale():
    import numpy as np
    rng = np.random.default_rng(0)
    actuals = rng.normal(8.85, 4.6, 5000)
    pred_means = np.full(5000, 7.9)   # sim mean is 0.95 low
    loc, scale = fcs.fit_dist_affine(pred_means, sim_marginal_sd=3.96, actuals=actuals)
    assert abs(loc - (actuals.mean() - 7.9)) < 0.05     # ~+0.95
    assert abs(scale - (actuals.std() / 3.96)) < 0.05   # ~1.16
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/test_fit_calibration_sim.py -k affine -v` → FAIL.

- [ ] **Step 3: Implement `fit_dist_affine` + wire into the fit script**

```python
def fit_dist_affine(pred_means, sim_marginal_sd, actuals):
    import numpy as np
    pred_means = np.asarray(pred_means, float); actuals = np.asarray(actuals, float)
    loc = float(actuals.mean() - pred_means.mean())
    scale = float(actuals.std() / sim_marginal_sd) if sim_marginal_sd > 0 else 1.0
    return loc, scale
```

In `fit_calibration_sim.py`'s main flow, after the existing per-market Platt fit, run the sim backtest to collect per-game `pred_total`/`pred_margin` means, the pooled sim marginal SDs, and the actual totals/margins, call `fit_dist_affine` for total and margin, and add `total_dist`/`margin_dist` entries to the written JSON. Keep all existing Platt entries (refit them in the same run).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_fit_calibration_sim.py -v` → PASS.

- [ ] **Step 5: Regenerate calibration.json**

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/fit_calibration_sim.py --season 2025
```

Confirm the printed before/after table and that `total_dist`/`margin_dist` loc/scale are near (loc≈0, scale≈1) — if the kernel work succeeded they should be small residuals. Large values mean the kernel didn't fully close the gap; note it but proceed (calibration absorbs it).

- [ ] **Step 6: Commit**

```bash
git add scripts/fit_calibration_sim.py tests/test_fit_calibration_sim.py assets/calibration.json
git commit -m "feat(model): fit totals/margin location+scale calibration; refit all markets"
```

---

### Task 10: Apply totals/margin calibration in the grader

**Files:**
- Modify: `scripts/grade_results.py`
- Test: `tests/test_grade_results.py`

**Interfaces:**
- Consumes: `calibration.load()` entries `total_dist`/`margin_dist`; `distributions.apply_affine`.
- Behavior: before `prob_over_dist(total_dist, line)` and `prob_cover(margin_dist, sl)`, remap the stored dist with the calibrated `(loc, scale)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grade_results.py  (add)
def test_total_calibration_shifts_over_prob(monkeypatch):
    # a +1.0 location calibration should raise the total's P(over) at a fixed line.
    md = grade_results
    monkeypatch.setattr(md, "_TOTAL_CAL", (1.0, 1.0), raising=False)
    d = {"kind":"pmf","pmf":[0,0,0,0,0,0,0,0,1.0]}  # mass at 8
    raw = md.prob_over_dist(d, 8)
    cal = md.prob_over_dist(md._calibrated_total(d), 8)
    assert cal > raw
```

(Adjust to whatever small helper you introduce — the test asserts calibration raises P(over) for a positive `loc`.)

- [ ] **Step 2: Run to verify fail** — FAIL (`_calibrated_total` undefined).

- [ ] **Step 3: Implement**

At import time in `grade_results.py`, load the calibration once:

```python
from sportsmodel.model import calibration
from sportsmodel.model.distributions import apply_affine
_cal = calibration.load()
_TOTAL_CAL = tuple(_cal["total_dist"].values()) if "total_dist" in _cal else (0.0, 1.0)
_MARGIN_CAL = tuple(_cal["margin_dist"].values()) if "margin_dist" in _cal else (0.0, 1.0)

def _calibrated_total(d):
    return apply_affine(d, _TOTAL_CAL[0], _TOTAL_CAL[1]) if d else d

def _calibrated_margin(d):
    return apply_affine(d, _MARGIN_CAL[0], _MARGIN_CAL[1]) if d else d
```

In `_grade_game`, replace `prob_over_dist(td, line)` with `prob_over_dist(_calibrated_total(td), line)` and `prob_cover(md, sl)` with `prob_cover(_calibrated_margin(md), sl)` (where `td`/`md` are the parsed total/margin dists). Ensure `total_dist.values()` order is `(loc, scale)` — write the JSON as an ordered `{"loc":..,"scale":..}` in Task 9 and read by key, not by `.values()`, to be safe: use `(_cal["total_dist"]["loc"], _cal["total_dist"]["scale"])`.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/test_grade_results.py -v` → PASS (all 19 + new).

- [ ] **Step 5: Commit**

```bash
git add scripts/grade_results.py tests/test_grade_results.py
git commit -m "feat(grade): apply totals/margin calibration before over/cover probs"
```

---

### Task 11: Apply totals/margin calibration on the board

**Files:**
- Modify: `streamlit_app.py`

**Interfaces:**
- Consumes: the same calibration entries; the board already parses `g.margin_dist` (Task from the run-line fix) and computes `prob_cover`. Totals board uses `prob_over_dist(total_dist, line)` — add the affine.

- [ ] **Step 1: Load calibration + add local helpers**

In `streamlit_app.py`, near the existing local `prob_over_dist`/`prob_cover`, load calibration (reuse the existing `assets/calibration.json` read the app already does for Platt) and add:

```python
_TOTAL_CAL = (_CAL.get("total_dist", {}).get("loc", 0.0), _CAL.get("total_dist", {}).get("scale", 1.0))
_MARGIN_CAL = (_CAL.get("margin_dist", {}).get("loc", 0.0), _CAL.get("margin_dist", {}).get("scale", 1.0))
```

Add a local `apply_affine` (copy of the model helper — the app deliberately keeps pure-math helpers local, matching the existing local `prob_over_dist`).

- [ ] **Step 2: Wire into `game_board`**

In the `total` branch, before computing the lean/EV from `total_dist`, remap: `td = apply_affine(json.loads(g.total_dist), *_TOTAL_CAL)` and use `prob_over_dist(td, line)`. In the `spread` branch, remap `g.margin_dist` with `_MARGIN_CAL` before `prob_cover`. (The board's totals branch currently only shows model-vs-market totals; if it does not yet compute P(over) from the dist, add it consistently with the grader so board and grader agree.)

- [ ] **Step 3: Syntax-check + manual verify**

Run: `python3 -c "import ast; ast.parse(open('streamlit_app.py').read()); print('OK')"`.
Then a scripted check that the calibrated P(over) differs from raw for a sample stored dist (pull one `game_predictions` row's `total_dist` via a local query if creds available, else construct one).

- [ ] **Step 4: Commit**

```bash
git add streamlit_app.py
git commit -m "feat(dashboard): apply totals/margin calibration on the board"
```

---

## Phase 4 — Validation & deploy

### Task 12: Run the acceptance bar

**Files:** none (measurement + report); Create: `docs/superpowers/reports/2026-08-21-sim-dispersion-validation.md`

- [ ] **Step 1: Distribution match**

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/validate_sim_dist.py --month 6 --n-sims 4000
```
Record all six metrics. PASS if each is within ~5% of empirical.

- [ ] **Step 2: No-regression on moneyline**

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/backtest_sim.py --season 2025 --n-sims 2000
```
PASS if win-prob Brier ≤ 0.246 + MC noise (compare to the pre-change number recorded in memory).

- [ ] **Step 3: No-regression on props**

```bash
SM_DATA_DIR="$(pwd)/data" uv run python scripts/backtest_sim_props.py --season 2025 --n-sims 1000
```
PASS if each market's calibration is no worse than the pre-change baseline.

- [ ] **Step 4: Write the report**

Capture the three results in `docs/superpowers/reports/2026-08-21-sim-dispersion-validation.md` with the before/after numbers and a PASS/FAIL per gate. If any gate FAILS, stop and reduce the σ's (or lean on calibration) — do not proceed to deploy.

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/reports/2026-08-21-sim-dispersion-validation.md
git commit -m "docs: sim dispersion validation report (acceptance bar)"
```

---

### Task 13: Merge & deploy

**Files:** none (git + ops)

- [ ] **Step 1: Full test suite**

```bash
uv run pytest tests/ -q
```
Expected: all green.

- [ ] **Step 2: Merge to main and push**

```bash
git checkout main && git merge --ff-only sim-scoring-dispersion && git push origin main
```

- [ ] **Step 3: Confirm production wiring**

`generate_sim.py` already builds specs via `build_game_spec` and now passes `roe_p`/`wp_p`/`dispersion` from `config_dispersion` (Task 6). No Supabase migration (schema unchanged). The next scheduled `daily-ingest`/`refresh-props` run picks up the new kernel + `calibration.json` automatically.

- [ ] **Step 4: Update memory**

Append the outcome (channels added, tuned σ's, calibration loc/scale, validation numbers) to `/Users/ryan/.claude/projects/-Users-ryan-Desktop-Sports-Model/memory/sports-model-project.md`.

---

## Self-Review

**Spec coverage:**
- Scoring channels (spec §1) → Tasks 1–3. ✓
- Outlier dispersion, both mechanisms (spec §2) → Tasks 5–7. ✓
- Calibration layer totals+margin (spec §3) → Tasks 8–11. ✓
- Validation/acceptance bar (spec §4) → Tasks 4, 12. ✓
- Build sequence (spec §5) → phases match. ✓
- Refit all markets (spec §3) → Task 9. ✓

**Type consistency:** `Dispersion(sigma_shared, sigma_team, sigma_pitcher)` used consistently (Tasks 5,6,7). `apply_affine(dist, loc, scale)` signature consistent (Tasks 8,10,11). `tail_metrics(home, away)` consistent (Tasks 4,7,12). Calibration JSON keys `total_dist`/`margin_dist` with `{"loc","scale"}` read by key (Tasks 9,10,11).

**Placeholder scan:** every code step has real code; measurement steps name exact commands and expected ranges. The two search-based steps (Task 7 tuning) provide the grid, objective, and acceptance check rather than a fixed answer, which is correct for a fit.
