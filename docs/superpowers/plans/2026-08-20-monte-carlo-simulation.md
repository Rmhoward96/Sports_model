# Monte Carlo Simulation Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a play-by-play Monte Carlo simulator for MLB that emits game and all player-prop distributions from one simulation, validated against the existing analytic model on the walk-forward backtest before going live.

**Architecture:** A sport-agnostic framework (`sim/engine.py`) with a pluggable MLB kernel (`sim/mlb/`). The kernel simulates plate appearances through a base-out state machine using Statcast-derived advancement tables, a sampled pitcher hook, and a times-through-order penalty. Correctness first (readable scalar kernel, unit-tested at small sim counts), then a vectorized numpy implementation guarded by an equivalence test, then the full backtest, then live wiring.

**Tech Stack:** Python 3.12, numpy, DuckDB (Statcast parquet), existing `sportsmodel` model/profiles modules, pytest.

**Spec:** `docs/superpowers/specs/2026-08-20-monte-carlo-simulation-design.md`

## Global Constraints

- The stored `dist` schema is `{"kind": "pmf", "pmf": [...]}` or `{"kind": "normal", "mean": m, "sd": s}` — the simulator emits **only pmf**. No DB/grading/dashboard changes.
- New model_versions coexist with analytic ones: `mlb-sim-v1` (game), `mlb-sim-props-v1` (props). Never overwrite `mlb-game-v2-defense` / `mlb-props-v1`.
- The 7 PA outcomes and their order are fixed: `("p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out")` (matches `rates.OUTCOMES`).
- Point-in-time safety: any profile/table used in the backtest is built respecting `transforms.set_cutoff` (no future data). Same mechanism the existing builders use.
- All new randomness goes through an explicit `numpy.random.Generator` passed in — never module-level `np.random` — so tests are reproducible with a fixed seed.
- Repo test convention: flat `tests/test_*.py`. Quick local checks: `PYTHONPATH=src uv run --no-sync python ...`. Full tests: `uv run pytest -q`.
- Commit after every green step. Do not push (the user pushes).

---

## Phase 1 — Advancement tables + simulation kernel

### Task 1: Dependency + package scaffolding

**Files:**
- Modify: `pyproject.toml` (add `numpy` to `dependencies`)
- Create: `src/sportsmodel/sim/__init__.py` (empty)
- Create: `src/sportsmodel/sim/mlb/__init__.py` (empty)

- [ ] **Step 1: Add numpy to dependencies**

In `pyproject.toml`, add to the `dependencies` list:

```toml
    "numpy>=1.26",
```

- [ ] **Step 2: Create empty package files**

```bash
mkdir -p src/sportsmodel/sim/mlb
touch src/sportsmodel/sim/__init__.py src/sportsmodel/sim/mlb/__init__.py
```

- [ ] **Step 3: Sync and verify import**

Run: `uv sync && PYTHONPATH=src uv run python -c "import sportsmodel.sim, sportsmodel.sim.mlb, numpy; print('ok', numpy.__version__)"`
Expected: `ok 2.x`

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/sportsmodel/sim
git commit -m "feat(sim): scaffold sim package, add numpy dependency"
```

---

### Task 2: Advancement-table builder (Statcast → transition table)

Derives, per `(outcome, base occupancy 0-7, outs 0-2)`, the empirical distribution over `(end occupancy 0-7, runs scored 0-4, outs added 0-3)` from the Statcast parquet. Only in-play outcomes need a table: `1b, 2b, 3b, out`. (`bb`, `k`, `hr` are deterministic rules in the kernel.)

**Files:**
- Create: `src/sportsmodel/sim/mlb/build_advancement.py`
- Test: `tests/test_build_advancement.py`

**Interfaces:**
- Produces: `build_advancement_table(con: duckdb.DuckDBPyConnection) -> list[dict]` where each dict is
  `{"outcome": str, "occ": int, "outs": int, "end_occ": int, "runs": int, "outs_added": int, "prob": float}`.
  `outcome ∈ {"p_1b","p_2b","p_3b","p_out"}`; `occ`/`end_occ` are 3-bit masks (bit0=1B, bit1=2B, bit2=3B); probabilities sum to 1 within each `(outcome, occ, outs)` group. Respects the active `transforms.set_cutoff`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_build_advancement.py
import duckdb
import pytest
from sportsmodel.sim.mlb.build_advancement import build_advancement_table, _base_occ_expr


def _mini_pbp(con):
    # Two half-innings of synthetic PA data with the columns the builder reads.
    # Columns mirror Statcast: on_1b/on_2b/on_3b are runner ids (NULL = empty).
    con.execute("""
        CREATE TABLE pbp AS SELECT * FROM (VALUES
        -- game 1, top 1: leadoff single (bases empty -> runner on 1st, 0 runs, 0 outs)
        (1, 1, 'Top', 1, 'single', NULL, NULL, NULL, 0, 0),
        -- next PA: runner on 1st, single -> table should see occ=1,out=0 -> some end state
        (1, 1, 'Top', 2, 'single', 100, NULL, NULL, 0, 0),
        (1, 1, 'Top', 3, 'field_out', 101, 100, NULL, 0, 1)
        ) AS t(game_pk, inning, inning_topbot, at_bat_number, events,
               on_1b, on_2b, on_3b, bat_score, post_bat_score)
    """)


def test_probabilities_sum_to_one_per_group():
    con = duckdb.connect(":memory:")
    _mini_pbp(con)
    rows = build_advancement_table(con, _table="pbp")
    from collections import defaultdict
    tot = defaultdict(float)
    for r in rows:
        tot[(r["outcome"], r["occ"], r["outs"])] += r["prob"]
    assert rows, "expected at least one transition row"
    for key, s in tot.items():
        assert abs(s - 1.0) < 1e-9, f"{key} sums to {s}"


def test_occ_mask_encoding():
    # bit0=1B, bit1=2B, bit2=3B
    assert _base_occ_expr  # symbol exists
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_build_advancement.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the builder**

```python
# src/sportsmodel/sim/mlb/build_advancement.py
"""Derive empirical base-out advancement tables from Statcast play-by-play.

For each (outcome, starting base occupancy, outs) we tally the resulting
(end occupancy, runs scored, outs added) by comparing a PA's pre-state to the
NEXT PA's pre-state within the same half-inning, and the batting-team score delta.
Only in-play outcomes need a table; BB/K/HR are deterministic in the kernel.
"""
from __future__ import annotations

import duckdb

from sportsmodel import transforms

# Statcast `events` -> our outcome code (only the in-play ones we tabulate).
_EVENT_TO_OUTCOME = {
    "single": "p_1b",
    "double": "p_2b",
    "triple": "p_3b",
    # every other batted-ball out / fielders choice / DP / sac counts as a generic out
}
_OUT_EVENTS = {
    "field_out", "grounded_into_double_play", "force_out", "sac_fly", "sac_bunt",
    "fielders_choice", "fielders_choice_out", "double_play", "field_error",
    "sac_fly_double_play", "triple_play",
}

_base_occ_expr = (
    "((CASE WHEN on_1b IS NOT NULL THEN 1 ELSE 0 END) "
    "+ (CASE WHEN on_2b IS NOT NULL THEN 2 ELSE 0 END) "
    "+ (CASE WHEN on_3b IS NOT NULL THEN 4 ELSE 0 END))"
)


def build_advancement_table(con: duckdb.DuckDBPyConnection, _table: str | None = None) -> list[dict]:
    """Transition rows respecting the active cutoff. `_table` overrides the source
    (tests pass a small in-memory table); production reads the Statcast parquet."""
    src = _table or f"read_parquet('{transforms._PARQUET_GLOB}')"
    cutoff = ""
    if _table is None and transforms._CUTOFF:  # same gate the other builders use
        cutoff = f"WHERE CAST(game_date AS DATE) < DATE '{transforms._CUTOFF}'"

    # 1) order PAs within each half-inning; read this PA's pre-state + the NEXT PA's
    #    pre-state (LEAD) + the batting-team score delta.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _adv_seq AS
        WITH pa AS (
            SELECT game_pk, inning, inning_topbot, at_bat_number, events,
                   {_base_occ_expr} AS occ,
                   COALESCE(post_bat_score - bat_score, 0) AS runs,
                   on_1b, on_2b, on_3b
            FROM {src} {cutoff}
        ),
        seq AS (
            SELECT *,
                   LEAD(occ) OVER w AS next_occ,
                   ROW_NUMBER() OVER w AS pa_idx,
                   COUNT(*) OVER (PARTITION BY game_pk, inning, inning_topbot) AS n_pa
            FROM pa
            WINDOW w AS (PARTITION BY game_pk, inning, inning_topbot ORDER BY at_bat_number)
        )
        SELECT * FROM seq
    """)

    # 2) map events -> outcome; compute end_occ (0 if this PA ended the half-inning),
    #    outs_added from runners-lost accounting isn't reliable, so derive outs_added
    #    from base+score bookkeeping: outs_added = (runners_before + 1) - runners_after - runs.
    #    runners_before = popcount(occ), runners_after = popcount(end_occ).
    rows = con.execute("""
        WITH mapped AS (
            SELECT
                CASE
                    WHEN events IN ('single') THEN 'p_1b'
                    WHEN events IN ('double') THEN 'p_2b'
                    WHEN events IN ('triple') THEN 'p_3b'
                    WHEN events IN ('field_out','grounded_into_double_play','force_out',
                                    'sac_fly','sac_bunt','fielders_choice','fielders_choice_out',
                                    'double_play','field_error','sac_fly_double_play','triple_play')
                        THEN 'p_out'
                    ELSE NULL
                END AS outcome,
                occ,
                runs,
                CASE WHEN pa_idx = n_pa THEN 0 ELSE COALESCE(next_occ, 0) END AS end_occ
            FROM _adv_seq
        ),
        counted AS (
            SELECT outcome, occ, end_occ, runs, count(*) AS c
            FROM mapped
            WHERE outcome IS NOT NULL
            GROUP BY outcome, occ, end_occ, runs
        ),
        tot AS (
            SELECT outcome, occ, sum(c) AS n FROM counted GROUP BY outcome, occ
        )
        SELECT c.outcome, c.occ, c.end_occ, c.runs, c.c::DOUBLE / t.n AS prob
        FROM counted c JOIN tot t USING (outcome, occ)
        ORDER BY c.outcome, c.occ, c.end_occ, c.runs
    """).fetchall()

    # NOTE: outs are collapsed across out-states in v1 (occ ignores out count) to keep the
    # table dense; `outs` fixed to a nominal 0 and `outs_added` derived in the kernel from
    # runner bookkeeping. This keeps every (outcome, occ) group well-populated.
    out = []
    for outcome, occ, end_occ, runs, prob in rows:
        out.append({"outcome": outcome, "occ": int(occ), "outs": 0,
                    "end_occ": int(end_occ), "runs": int(runs),
                    "outs_added": None, "prob": float(prob)})
    return out
```

- [ ] **Step 4: Adjust the test call signature**

The test calls `build_advancement_table(con, _table="pbp")`. Confirm the signature matches. Run:

Run: `uv run pytest tests/test_build_advancement.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/build_advancement.py tests/test_build_advancement.py
git commit -m "feat(sim): Statcast advancement-table builder"
```

**Reviewer note:** `outs_added` is intentionally `None` from the builder — the kernel computes outs from runner bookkeeping (Task 6), which is exact. Grouping by `occ` only (not out-state) keeps groups dense; refine to out-state-aware tables later if the backtest warrants.

---

### Task 3: Advancement loader + interface + TTO deltas

Turns the transition rows into fast lookup arrays and the `advance(...)` interface the kernel calls, plus the league-average times-through-order multipliers.

**Files:**
- Create: `src/sportsmodel/sim/mlb/advancement.py`
- Test: `tests/test_advancement.py`

**Interfaces:**
- Consumes: rows from `build_advancement_table` (Task 2).
- Produces:
  - `class AdvancementTable` with `from_rows(rows) -> AdvancementTable` and
    `sample(outcome_code: int, occ: int, u: float) -> tuple[int, int]` returning `(end_occ, runs)` for a single draw given a uniform `u ∈ [0,1)`. `outcome_code` uses the kernel's int encoding (Task 5): `S=2, D=3, T=4, OUT=6`.
  - `TTO_MULT: dict[str, tuple[float, float, float]]` — per-outcome multiplier for times-through-order 1/2/3+ applied to a starter's vector (values below).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_advancement.py
import pytest
from sportsmodel.sim.mlb.advancement import AdvancementTable, TTO_MULT

_ROWS = [
    # outcome p_1b, bases empty (occ 0): always -> runner on 1st (occ 1), 0 runs
    {"outcome": "p_1b", "occ": 0, "end_occ": 1, "runs": 0, "prob": 1.0},
    # outcome p_1b, runner on 2nd (occ 2): 60% score (occ 1, 1 run), 40% -> 1st&3rd (occ 5, 0)
    {"outcome": "p_1b", "occ": 2, "end_occ": 1, "runs": 1, "prob": 0.6},
    {"outcome": "p_1b", "occ": 2, "end_occ": 5, "runs": 0, "prob": 0.4},
]


def test_sample_deterministic_single_outcome():
    t = AdvancementTable.from_rows(_ROWS)
    # S=2 in the kernel's encoding
    assert t.sample(2, 0, 0.01) == (1, 0)
    assert t.sample(2, 0, 0.99) == (1, 0)


def test_sample_respects_cumulative_probability():
    t = AdvancementTable.from_rows(_ROWS)
    assert t.sample(2, 2, 0.3) == (1, 1)   # in first 0.6
    assert t.sample(2, 2, 0.7) == (5, 0)   # in last 0.4


def test_tto_mult_worsens_for_starter():
    # times through the order 2 and 3 should raise hit/hr, lower K vs pass 1
    assert TTO_MULT["p_hr"][1] > TTO_MULT["p_hr"][0]
    assert TTO_MULT["p_k"][2] < TTO_MULT["p_k"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_advancement.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement the loader + interface**

```python
# src/sportsmodel/sim/mlb/advancement.py
"""Load the advancement transition table into fast lookup structures, expose the
sampling interface the kernel uses, and hold the times-through-order multipliers."""
from __future__ import annotations

from dataclasses import dataclass

# kernel int encoding for the in-play outcomes that use the table
_OUTCOME_NAME = {2: "p_1b", 3: "p_2b", 4: "p_3b", 6: "p_out"}

# League-average times-through-order penalty on a STARTER's per-PA vector, indexed
# [1st time, 2nd time, 3rd+ time] through the order. Values are outcome-rate
# multipliers (renormalized after applying). ~ +0.008 wOBA per time through. [tunable]
TTO_MULT: dict[str, tuple[float, float, float]] = {
    "p_bb": (1.00, 1.04, 1.08),
    "p_k":  (1.00, 0.95, 0.90),
    "p_1b": (1.00, 1.03, 1.06),
    "p_2b": (1.00, 1.04, 1.08),
    "p_3b": (1.00, 1.04, 1.08),
    "p_hr": (1.00, 1.06, 1.12),
    "p_out": (1.00, 0.99, 0.98),
}


@dataclass
class AdvancementTable:
    # keyed (outcome_name, occ) -> (cum_probs, end_occ[], runs[])
    _table: dict

    @classmethod
    def from_rows(cls, rows) -> "AdvancementTable":
        grouped: dict = {}
        for r in rows:
            grouped.setdefault((r["outcome"], int(r["occ"])), []).append(
                (float(r["prob"]), int(r["end_occ"]), int(r["runs"])))
        table = {}
        for key, entries in grouped.items():
            entries.sort(key=lambda e: (-e[0]))  # stable; order within group irrelevant
            cum, ends, runs, acc = [], [], [], 0.0
            for p, e, rn in entries:
                acc += p
                cum.append(acc)
                ends.append(e)
                runs.append(rn)
            cum[-1] = 1.0  # guard fp drift
            table[key] = (cum, ends, runs)
        return cls(table)

    def sample(self, outcome_code: int, occ: int, u: float) -> tuple[int, int]:
        key = (_OUTCOME_NAME[outcome_code], occ)
        entry = self._table.get(key)
        if entry is None:
            # unseen state -> conservative fallback: batter to first if empty-ish, no runs
            return (occ | 1, 0) if outcome_code == 2 else (occ, 0)
        cum, ends, runs = entry
        for i, c in enumerate(cum):
            if u < c:
                return ends[i], runs[i]
        return ends[-1], runs[-1]
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_advancement.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/advancement.py tests/test_advancement.py
git commit -m "feat(sim): advancement lookup interface + TTO multipliers"
```

---

### Task 4: Engine result container + aggregation (sport-agnostic)

**Files:**
- Create: `src/sportsmodel/sim/engine.py`
- Test: `tests/test_sim_engine.py`

**Interfaces:**
- Produces:
  - `@dataclass GameSims` with fields `home_score: np.ndarray`, `away_score: np.ndarray`, `batter_stats: dict[int, dict[str, np.ndarray]]`, `pitcher_stats: dict[int, dict[str, np.ndarray]]` (each inner array length = n_sims).
  - `home_win_prob(sims) -> float`
  - `total_pmf(sims, max_total=30) -> list[float]`
  - `pred_scores(sims) -> dict` → `{"pred_home_score","pred_away_score","pred_total","pred_margin","home_win_prob"}`
  - `stat_pmf(arr, max_k) -> list[float]` (empirical pmf of a non-negative integer stat array)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_engine.py
import numpy as np
from sportsmodel.sim.engine import GameSims, home_win_prob, total_pmf, pred_scores, stat_pmf


def _toy():
    return GameSims(
        home_score=np.array([3, 5, 2, 4]),
        away_score=np.array([2, 5, 4, 1]),
        batter_stats={}, pitcher_stats={},
    )


def test_home_win_prob_counts_wins_only():
    # ties (5-5) do not count as home wins; sim resolves ties via extra innings,
    # but the aggregator must not credit an equal-score row as a win.
    assert home_win_prob(_toy()) == 0.5  # wins: 3>2, 4>1 ; losses: 2<4 ; tie: 5=5 -> excluded


def test_total_pmf_sums_to_one():
    p = total_pmf(_toy(), max_total=12)
    assert abs(sum(p) - 1.0) < 1e-9
    assert p[5] == 0.25  # totals: 5,10,6,5 -> total 5 appears twice of four


def test_stat_pmf():
    p = stat_pmf(np.array([0, 1, 1, 2]), max_k=3)
    assert p == [0.25, 0.5, 0.25, 0.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sim_engine.py -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement engine aggregation**

```python
# src/sportsmodel/sim/engine.py
"""Sport-agnostic simulation result container and aggregation helpers.

A kernel fills a GameSims (raw per-sim arrays); these helpers turn it into the
stored outputs (win prob, total pmf, per-player pmfs). Nothing here is baseball-specific.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GameSims:
    home_score: np.ndarray
    away_score: np.ndarray
    batter_stats: dict[int, dict[str, np.ndarray]]
    pitcher_stats: dict[int, dict[str, np.ndarray]]


def home_win_prob(sims: GameSims) -> float:
    return float(np.mean(sims.home_score > sims.away_score))


def stat_pmf(arr: np.ndarray, max_k: int) -> list[float]:
    n = len(arr)
    counts = np.bincount(np.clip(arr, 0, max_k).astype(int), minlength=max_k + 1)[: max_k + 1]
    return (counts / n).tolist()


def total_pmf(sims: GameSims, max_total: int = 30) -> list[float]:
    return stat_pmf(sims.home_score + sims.away_score, max_total)


def pred_scores(sims: GameSims) -> dict:
    h = float(np.mean(sims.home_score))
    a = float(np.mean(sims.away_score))
    return {
        "pred_home_score": h,
        "pred_away_score": a,
        "pred_total": h + a,
        "pred_margin": h - a,
        "home_win_prob": home_win_prob(sims),
    }
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sim_engine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/engine.py tests/test_sim_engine.py
git commit -m "feat(sim): sport-agnostic result container + aggregation"
```

---

### Task 5: Kernel — GameSpec, outcome encoding, single-PA sampling (scalar, correctness-first)

Build a **readable scalar kernel** first (loops over sims). It is the correctness oracle; Task 8 vectorizes it behind an equivalence test. Unit tests run at small sim counts so speed is irrelevant here.

**Files:**
- Create: `src/sportsmodel/sim/mlb/kernel.py`
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Produces:
  - Outcome int encoding constants: `BB=0, K=1, OUT_INPLAY=6, S=2, D=3, T=4, HR=5`. (Note: `OUT_INPLAY` and `HR` codes are distinct; `p_out` in a vector maps to code 6, `p_hr` to code 5.)
  - `@dataclass GameSpec` with:
    `home_order: list[Batter]`, `away_order: list[Batter]`, `home_starter: Pitcher`, `away_starter: Pitcher`.
  - `@dataclass Batter`: `player_id: int`, `vec_vs_sp: dict[str,float]`, `vec_vs_bp: dict[str,float]`.
  - `@dataclass Pitcher`: `player_id: int`, `avg_bf: float`, `sd_bf: float`.
  - `sample_outcome(vec: dict[str,float], u: float) -> int` — maps a per-PA vector + uniform draw to an outcome code, using order `("p_bb","p_k","p_1b","p_2b","p_3b","p_hr","p_out")` → codes `(0,1,2,3,4,5,6)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_kernel.py
from sportsmodel.sim.mlb import kernel as K


def test_sample_outcome_cumulative():
    vec = {"p_bb": 0.1, "p_k": 0.2, "p_1b": 0.2, "p_2b": 0.05,
           "p_3b": 0.02, "p_hr": 0.03, "p_out": 0.4}
    assert K.sample_outcome(vec, 0.05) == K.BB      # first 0.10
    assert K.sample_outcome(vec, 0.25) == K.K       # 0.10-0.30
    assert K.sample_outcome(vec, 0.35) == K.S       # 0.30-0.50
    assert K.sample_outcome(vec, 0.99) == K.OUT_INPLAY  # last bucket


def test_outcome_codes_distinct():
    codes = {K.BB, K.K, K.S, K.D, K.T, K.HR, K.OUT_INPLAY}
    assert len(codes) == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement encoding + sampling + spec dataclasses**

```python
# src/sportsmodel/sim/mlb/kernel.py
"""MLB play-by-play Monte Carlo kernel (scalar, correctness-first).

Simulates plate appearances through a base-out state machine using empirical
advancement tables, a sampled pitcher hook, and a times-through-order penalty.
A vectorized numpy version (equivalence-tested) replaces the hot loop later.
"""
from __future__ import annotations

from dataclasses import dataclass

from .advancement import TTO_MULT

# per-PA vector order -> outcome codes
_VEC_ORDER = ("p_bb", "p_k", "p_1b", "p_2b", "p_3b", "p_hr", "p_out")
BB, K, S, D, T, HR, OUT_INPLAY = 0, 1, 2, 3, 4, 5, 6
_CODE = {"p_bb": BB, "p_k": K, "p_1b": S, "p_2b": D, "p_3b": T, "p_hr": HR, "p_out": OUT_INPLAY}


@dataclass
class Batter:
    player_id: int
    vec_vs_sp: dict
    vec_vs_bp: dict


@dataclass
class Pitcher:
    player_id: int
    avg_bf: float
    sd_bf: float


@dataclass
class GameSpec:
    home_order: list
    away_order: list
    home_starter: Pitcher
    away_starter: Pitcher


def sample_outcome(vec: dict, u: float) -> int:
    acc = 0.0
    for name in _VEC_ORDER:
        acc += vec[name]
        if u < acc:
            return _CODE[name]
    return OUT_INPLAY


def apply_tto(vec: dict, times_through: int) -> dict:
    """Worsen a starter's vector for the 2nd/3rd+ time through the order; renormalize."""
    idx = min(times_through, 3) - 1  # 1->0, 2->1, 3+->2
    if idx <= 0:
        return vec
    scaled = {k: vec[k] * TTO_MULT[k][idx] for k in _VEC_ORDER}
    tot = sum(scaled.values())
    return {k: v / tot for k, v in scaled.items()}
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): kernel outcome encoding, spec dataclasses, PA sampling"
```

---

### Task 6: Kernel — base-out state, deterministic runner resolution, half-inning

Runner identity is tracked as lineup indices on bases. `resolve_pa` applies one outcome to a `BaseState` and returns runs scored + RBI credit + outs added, using the advancement table for in-play hits/outs and deterministic rules for BB/HR/K.

**Files:**
- Modify: `src/sportsmodel/sim/mlb/kernel.py`
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Produces:
  - `class BaseState` holding `first/second/third: int` (lineup index or −1) and helper `occ() -> int` (3-bit mask).
  - `resolve_pa(state: BaseState, batter_idx: int, outcome: int, adv: AdvancementTable, u: float) -> tuple[int, int]` returning `(runs_scored, outs_added)`; mutates `state` in place and records which runner indices scored via the returned runs (identity reconciliation described below). RBI credit equals `runs_scored` for the batter except on a double play out where a run may not be earned — v1 credits RBI = runs on non-K outs and = runs on hits, 0 on K.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_sim_kernel.py
from sportsmodel.sim.mlb.advancement import AdvancementTable

_EMPTY_ADV = AdvancementTable.from_rows([])


def test_home_run_bases_loaded_scores_four():
    st = K.BaseState(first=3, second=4, third=5)
    runs, outs = K.resolve_pa(st, batter_idx=6, outcome=K.HR, adv=_EMPTY_ADV, u=0.0)
    assert runs == 4 and outs == 0
    assert st.occ() == 0  # bases cleared


def test_walk_forces_only():
    st = K.BaseState(first=1, second=-1, third=-1)
    runs, outs = K.resolve_pa(st, batter_idx=2, outcome=K.BB, adv=_EMPTY_ADV, u=0.0)
    assert runs == 0 and outs == 0
    assert st.first == 2 and st.second == 1  # batter to 1st, forced runner to 2nd


def test_strikeout_is_pure_out():
    st = K.BaseState(first=1, second=-1, third=-1)
    runs, outs = K.resolve_pa(st, batter_idx=2, outcome=K.K, adv=_EMPTY_ADV, u=0.0)
    assert runs == 0 and outs == 1
    assert st.first == 1  # unchanged


def test_single_from_table_empty_bases():
    adv = AdvancementTable.from_rows(
        [{"outcome": "p_1b", "occ": 0, "end_occ": 1, "runs": 0, "prob": 1.0}])
    st = K.BaseState(-1, -1, -1)
    runs, outs = K.resolve_pa(st, batter_idx=7, outcome=K.S, adv=adv, u=0.5)
    assert runs == 0 and outs == 0
    assert st.first == 7 and st.second == -1  # batter on first
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: FAIL (BaseState/resolve_pa not defined).

- [ ] **Step 3: Implement state + resolution**

```python
# add to src/sportsmodel/sim/mlb/kernel.py

class BaseState:
    __slots__ = ("first", "second", "third")

    def __init__(self, first=-1, second=-1, third=-1):
        self.first, self.second, self.third = first, second, third

    def occ(self) -> int:
        return (1 if self.first >= 0 else 0) | (2 if self.second >= 0 else 0) | (4 if self.third >= 0 else 0)

    def runners(self) -> int:
        return (self.first >= 0) + (self.second >= 0) + (self.third >= 0)


def _fill_from_mask(state: "BaseState", end_occ: int, batter_idx: int, lead_first: bool):
    """Place surviving runners + batter onto the bases named by end_occ.

    Survivors advance in order (3rd, 2nd, 1st are the closest-to-home); the batter
    occupies the lowest set base that isn't taken by a survivor. This gives a
    deterministic, consistent identity mapping for any abstract end state.
    """
    # collect existing runners nearest-to-home first, then the batter last
    survivors = [b for b in (state.third, state.second, state.first) if b >= 0]
    order_slots = [4, 2, 1]  # 3rd, 2nd, 1st bit values, nearest-home first
    state.first = state.second = state.third = -1
    occupied = []
    for bit in order_slots:
        if end_occ & bit:
            occupied.append(bit)
    # assign survivors to the highest occupied bases, batter to the lowest
    to_place = survivors + [batter_idx]
    # nearest-home bases get the runners who were already furthest along
    for bit, who in zip(occupied, to_place):
        if bit == 4:
            state.third = who
        elif bit == 2:
            state.second = who
        else:
            state.first = who


def resolve_pa(state: "BaseState", batter_idx: int, outcome: int, adv, u: float) -> tuple[int, int]:
    if outcome == K:
        return 0, 1
    if outcome == HR:
        runs = state.runners() + 1
        state.first = state.second = state.third = -1
        return runs, 0
    if outcome == BB:
        # force only: batter to first; bump a forced chain
        if state.first < 0:
            state.first = batter_idx
        elif state.second < 0:
            state.second, state.first = state.first, batter_idx
        elif state.third < 0:
            state.third, state.second, state.first = state.second, state.first, batter_idx
        else:
            # bases loaded walk forces in a run
            state.third, state.second, state.first = state.second, state.first, batter_idx
            return 1, 0
        return 0, 0
    # in-play hit or out -> table
    occ = state.occ()
    runners_before = state.runners()
    end_occ, runs = adv.sample(outcome, occ, u)
    runners_after = bin(end_occ).count("1")
    # outs_added from bookkeeping: batter + runners_before must all be accounted for as
    # (scored) or (still on base) or (out). runners_after excludes the batter's own slot.
    accounted_on_base = runners_after
    outs_added = (runners_before + 1) - accounted_on_base - runs
    outs_added = max(0, min(3, outs_added))
    _fill_from_mask(state, end_occ, batter_idx, lead_first=(outcome == S))
    return runs, outs_added
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: PASS (all state tests).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): base-out state + deterministic runner resolution"
```

**Reviewer note:** the identity mapping in `_fill_from_mask` is an approximation (it can't perfectly reconstruct which specific runner scored when the table is abstract), but it is *consistent* and conserves runners/outs. Runs and outs — the quantities the props and game markets need — are exact from the bookkeeping. Per-runner RBI attribution is good enough for HRR at v1.

---

### Task 7: Kernel — full game loop (innings, hook, TTO) + box score → GameSims

**Files:**
- Modify: `src/sportsmodel/sim/mlb/kernel.py`
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Consumes: `GameSpec` (Task 5), `AdvancementTable` (Task 3), `resolve_pa`/`apply_tto` (Tasks 5-6), `GameSims` (Task 4).
- Produces: `simulate_scalar(spec: GameSpec, n_sims: int, rng: np.random.Generator, max_extra=11) -> GameSims`.

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_sim_kernel.py
import numpy as np
from sportsmodel.sim.engine import GameSims


def _flat_vec(**over):
    v = {"p_bb": 0.08, "p_k": 0.22, "p_1b": 0.15, "p_2b": 0.04,
         "p_3b": 0.005, "p_hr": 0.03, "p_out": 0.475 - 0.03}
    v.update(over)
    s = sum(v.values())
    return {k: x / s for k, x in v.items()}


def _spec():
    bs = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(100, 109)]
    aw = [K.Batter(pid, _flat_vec(), _flat_vec()) for pid in range(200, 209)]
    return K.GameSpec(bs, aw, K.Pitcher(1, 24.0, 3.0), K.Pitcher(2, 24.0, 3.0))


def test_simulate_returns_gamesims_with_right_shapes():
    sims = K.simulate_scalar(_spec(), n_sims=50, rng=np.random.default_rng(0))
    assert isinstance(sims, GameSims)
    assert sims.home_score.shape == (50,) and sims.away_score.shape == (50,)
    assert set(sims.batter_stats.keys()) == set(range(100, 109)) | set(range(200, 209))
    assert "hits" in sims.batter_stats[100] and "hrr" in sims.batter_stats[100]
    assert sims.pitcher_stats[1]["outs"].shape == (50,)


def test_no_ties_scores_are_decided():
    sims = K.simulate_scalar(_spec(), n_sims=200, rng=np.random.default_rng(1))
    assert not np.any(sims.home_score == sims.away_score)  # extras resolve ties


def test_reproducible_with_seed():
    a = K.simulate_scalar(_spec(), 30, np.random.default_rng(7))
    b = K.simulate_scalar(_spec(), 30, np.random.default_rng(7))
    assert np.array_equal(a.home_score, b.home_score)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: FAIL (simulate_scalar not defined).

- [ ] **Step 3: Implement the game loop**

```python
# add to src/sportsmodel/sim/mlb/kernel.py
import numpy as np
from ..engine import GameSims

_BATTER_MARKETS = ("hits", "total_bases", "hr", "runs", "rbi", "hrr")


def _new_box(order):
    return {b.player_id: {m: 0 for m in _BATTER_MARKETS} for b in order}


def _tb_for(outcome: int) -> int:
    return {S: 1, D: 2, T: 3, HR: 4}.get(outcome, 0)


def _sim_one(spec: GameSpec, adv, rng) -> tuple[int, int, dict, dict, dict, dict]:
    """One full game. Returns (home_runs, away_runs, home_box, away_box, hp_line, ap_line)."""
    home_box, away_box = _new_box(spec.home_order), _new_box(spec.away_order)
    # pitcher stat lines (starter only): K, hits allowed, outs
    hp = {"k": 0, "hits": 0, "outs": 0}
    ap = {"k": 0, "hits": 0, "outs": 0}
    hook_home = max(12, rng.normal(spec.home_starter.avg_bf, spec.home_starter.sd_bf))
    hook_away = max(12, rng.normal(spec.away_starter.avg_bf, spec.away_starter.sd_bf))
    scores = [0, 0]           # [away, home]
    idx = [0, 0]              # batting-order pointer [away, home]
    bf = [0, 0]              # batters faced by the [away pitcher? ] -> track per defense below

    def half(bat_team, inning):
        # bat_team: 0 away, 1 home. Defense is the other team; its starter faces batters.
        order = spec.home_order if bat_team == 1 else spec.away_order
        box = home_box if bat_team == 1 else away_box
        defense = 0 if bat_team == 1 else 1
        starter = spec.away_starter if defense == 0 else spec.home_starter
        hook = hook_away if defense == 0 else hook_home
        pline = ap if defense == 0 else hp
        state = BaseState()
        outs = 0
        while outs < 3:
            b = order[idx[bat_team] % 9]
            faced = bf[defense]
            starter_in = faced < hook
            times_through = faced // 9 + 1
            vec = apply_tto(b.vec_vs_sp, times_through) if starter_in else b.vec_vs_bp
            u = rng.random()
            outcome = sample_outcome(vec, u)
            u2 = rng.random()
            runs, outs_added = resolve_pa(state, idx[bat_team] % 9, outcome, adv, u2)
            # box score
            if outcome in (S, D, T, HR):
                box[b.player_id]["hits"] += 1
                box[b.player_id]["total_bases"] += _tb_for(outcome)
                if outcome == HR:
                    box[b.player_id]["hr"] += 1
            box[b.player_id]["rbi"] += runs if outcome != K else 0
            scores[bat_team] += runs
            # crude runs-scored credit: distribute `runs` to the batter's team tally only;
            # per-batter "runs" credited to the batter on his own HR, else left aggregate.
            if outcome == HR:
                box[b.player_id]["runs"] += 1
            # pitcher line (starter only)
            if starter_in:
                if outcome == K:
                    pline["k"] += 1
                if outcome in (S, D, T, HR):
                    pline["hits"] += 1
                pline["outs"] += outs_added + (1 if outcome == K else 0)
            outs += outs_added + (1 if outcome == K else 0)
            bf[defense] += 1
            idx[bat_team] += 1

    inning = 1
    while True:
        half(0, inning)  # away bats (top)
        half(1, inning)  # home bats (bottom)
        if inning >= 9 and scores[1] != scores[0]:
            break
        if inning >= 9 + 11:  # hard cap
            if scores[0] == scores[1]:
                scores[1] += 1  # break ties at the cap deterministically
            break
        inning += 1

    # finalize hrr
    for box in (home_box, away_box):
        for pid, s in box.items():
            s["hrr"] = s["hits"] + s["runs"] + s["rbi"]
    return scores[1], scores[0], home_box, away_box, hp, ap


def simulate_scalar(spec: GameSpec, n_sims: int, rng, max_extra: int = 11) -> GameSims:
    from .advancement import AdvancementTable
    # spec may carry the adv table; if not, empty (tests inject via module-level default)
    adv = getattr(spec, "adv", None) or AdvancementTable.from_rows([])
    hs = np.zeros(n_sims, dtype=np.int32)
    as_ = np.zeros(n_sims, dtype=np.int32)
    bstats = {b.player_id: {m: np.zeros(n_sims, np.int32) for m in _BATTER_MARKETS}
              for b in (*spec.home_order, *spec.away_order)}
    pstats = {spec.home_starter.player_id: {m: np.zeros(n_sims, np.int32) for m in ("k", "hits", "outs")},
              spec.away_starter.player_id: {m: np.zeros(n_sims, np.int32) for m in ("k", "hits", "outs")}}
    for i in range(n_sims):
        hr_, ar_, hbox, abox, hp, ap = _sim_one(spec, adv, rng)
        hs[i], as_[i] = hr_, ar_
        for box in (hbox, abox):
            for pid, s in box.items():
                for m in _BATTER_MARKETS:
                    bstats[pid][m][i] = s[m]
        for pid, s in ((spec.home_starter.player_id, hp), (spec.away_starter.player_id, ap)):
            for m in ("k", "hits", "outs"):
                pstats[pid][m][i] = s[m]
    return GameSims(hs, as_, bstats, pstats)
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: PASS (shapes, no-ties, reproducibility).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): full game loop with hook/TTO + box-score accumulation"
```

**Reviewer note:** `GameSpec` gains an optional `adv` attribute consumed here; Task 10 sets it. Per-batter "runs scored" credit beyond HRs is left aggregate in v1 (identity of who-scored is approximate); HRR still uses hits+rbi+HR-runs which is the dominant signal. Revisit if the prop backtest shows HRR bias.

---

### Task 8: Vectorize the kernel behind an equivalence gate

Replace the per-sim Python loop with numpy arrays across sims, and prove it matches the scalar kernel's aggregate distributions within Monte Carlo tolerance. This is what makes the full backtest tractable.

**Files:**
- Modify: `src/sportsmodel/sim/mlb/kernel.py` (add `simulate(spec, n_sims, rng)`)
- Test: `tests/test_sim_kernel.py`

**Interfaces:**
- Produces: `simulate(spec: GameSpec, n_sims: int, rng: np.random.Generator) -> GameSims` — same signature/semantics as `simulate_scalar`, vectorized across sims (masked lock-step per PA as described in the spec).

- [ ] **Step 1: Write the equivalence + speed test**

```python
# add to tests/test_sim_kernel.py
def test_vectorized_matches_scalar_distribution():
    spec = _spec()
    a = K.simulate_scalar(spec, 4000, np.random.default_rng(3))
    b = K.simulate(spec, 4000, np.random.default_rng(3))
    # aggregate means should agree within Monte Carlo noise (~0.1 run)
    assert abs(a.home_score.mean() - b.home_score.mean()) < 0.15
    assert abs(a.away_score.mean() - b.away_score.mean()) < 0.15
    # win prob within a couple points
    from sportsmodel.sim.engine import home_win_prob
    assert abs(home_win_prob(a) - home_win_prob(b)) < 0.03
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_kernel.py::test_vectorized_matches_scalar_distribution -v`
Expected: FAIL (`simulate` not defined).

- [ ] **Step 3: Implement the vectorized kernel**

Implement `simulate` using per-sim state arrays (`np.ndarray` of length N) and masked lock-step: process one PA across all sims still batting in the current half-inning, sample outcomes via cumulative-probability search on a stacked vector matrix, apply advancement via vectorized table lookup (precompute per-(outcome,occ) cumulative arrays into a padded numpy structure keyed by `outcome*8+occ`), update state arrays, flip half-innings when `outs==3`, end games by the same rule as scalar. Accumulate box scores by scatter-add into the per-player arrays. Keep the outcome/advancement RNG draw order identical to scalar so distributions match. (Full implementation ~150 lines; mirror `_sim_one` exactly, lifting each scalar operation to a masked array operation.)

Reference structure:

```python
def simulate(spec, n_sims, rng):
    from .advancement import AdvancementTable
    adv = getattr(spec, "adv", None) or AdvancementTable.from_rows([])
    N = n_sims
    # ... build vector matrices [9 x 7] for each order vs sp/bp, TTO variants precomputed ...
    # ... state arrays: first/second/third (int, -1), outs, idx per team, bf per defense,
    #     hook thresholds (rng.normal), scores, inning, phase mask ...
    # ... loop half-innings; within a half, while any active: sample outcomes, resolve,
    #     scatter box-score adds; advance idx; check outs ...
    # ... assemble GameSims from the accumulated arrays ...
    return GameSims(home_score, away_score, bstats, pstats)
```

- [ ] **Step 4: Run the equivalence test**

Run: `uv run pytest tests/test_sim_kernel.py -v`
Expected: PASS. If means diverge beyond tolerance, align the RNG draw order and the identity/out bookkeeping with `_sim_one` until they match.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/kernel.py tests/test_sim_kernel.py
git commit -m "feat(sim): vectorized kernel with scalar-equivalence test"
```

---

### Task 9: Inputs adapter — build a GameSpec from existing loaders

**Files:**
- Create: `src/sportsmodel/sim/mlb/inputs.py`
- Test: `tests/test_sim_inputs.py`

**Interfaces:**
- Consumes: `rates.matchup_vector`, `game.apply_bip_defense/apply_hr_multiplier/apply_park_to_vector`, the `Batter/Pitcher/GameSpec` types (Task 5), `AdvancementTable` (Task 3).
- Produces: `build_game_spec(home_order, away_order, home_sp_vec, away_sp_vec, home_bp_vec, away_bp_vec, workload, context, league, adv) -> GameSpec` where `home_order`/`away_order` are `list[(player_id, batter_vec)]`, `context` carries `home_pf, hr_mult, home_def, away_def`, `workload` maps `pid -> (avg_bf, sd_bf)`. Attaches `adv` to the spec.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sim_inputs.py
from sportsmodel.sim.mlb import inputs, kernel
from sportsmodel.sim.mlb.advancement import AdvancementTable

_L = {"p_bb": .08, "p_k": .22, "p_1b": .15, "p_2b": .045, "p_3b": .004, "p_hr": .03, "p_out": .471}


def _vec(**o):
    v = dict(_L); v.update(o); s = sum(v.values()); return {k: x / s for k, x in v.items()}


def test_build_game_spec_shapes_and_context():
    home = [(100 + i, _vec()) for i in range(9)]
    away = [(200 + i, _vec()) for i in range(9)]
    spec = inputs.build_game_spec(
        home, away, _vec(p_k=.28), _vec(p_k=.26), _vec(), _vec(),
        workload={1: (24.0, 3.0), 2: (23.0, 3.5)},
        context={"home_pf": 1.05, "hr_mult": 1.0, "home_def": 1.0, "away_def": 1.0},
        league=_L, adv=AdvancementTable.from_rows([]),
        home_starter_id=1, away_starter_id=2,
    )
    assert isinstance(spec, kernel.GameSpec)
    assert len(spec.home_order) == 9 and len(spec.away_order) == 9
    # each batter has BOTH matchup vectors, and they are valid distributions
    b = spec.home_order[0]
    assert abs(sum(b.vec_vs_sp.values()) - 1.0) < 1e-9
    assert abs(sum(b.vec_vs_bp.values()) - 1.0) < 1e-9
    assert spec.home_starter.avg_bf == 24.0
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_inputs.py -v`
Expected: FAIL.

- [ ] **Step 3: Implement the adapter**

```python
# src/sportsmodel/sim/mlb/inputs.py
"""Assemble a kernel GameSpec from the same inputs the analytic props path uses."""
from __future__ import annotations

from sportsmodel.model import game, rates
from .kernel import Batter, GameSpec, Pitcher


def _matchup(batter_vec, pitcher_vec, league, *, pf, hr_mult, opp_def):
    v = rates.matchup_vector(batter_vec, pitcher_vec, league)
    if opp_def != 1.0:
        v = game.apply_bip_defense(v, opp_def)
    if hr_mult != 1.0:
        v = game.apply_hr_multiplier(v, hr_mult)
    if pf != 1.0:
        v = game.apply_park_to_vector(v, pf)
    return v


def build_game_spec(home_order, away_order, home_sp_vec, away_sp_vec,
                    home_bp_vec, away_bp_vec, workload, context, league, adv,
                    home_starter_id, away_starter_id) -> GameSpec:
    pf, hr_mult = context["home_pf"], context["hr_mult"]
    home_def, away_def = context["home_def"], context["away_def"]

    def order(pairs, opp_sp_vec, opp_bp_vec, opp_def):
        out = []
        for pid, bvec in pairs:
            vs_sp = _matchup(bvec, opp_sp_vec, league, pf=pf, hr_mult=hr_mult, opp_def=opp_def)
            vs_bp = _matchup(bvec, opp_bp_vec, league, pf=pf, hr_mult=hr_mult, opp_def=opp_def)
            out.append(Batter(pid, vs_sp, vs_bp))
        return out

    home = order(home_order, away_sp_vec, away_bp_vec, away_def)  # home bats vs away pitchers
    away = order(away_order, home_sp_vec, home_bp_vec, home_def)
    spec = GameSpec(home, away,
                    Pitcher(home_starter_id, *workload[home_starter_id]),
                    Pitcher(away_starter_id, *workload[away_starter_id]))
    spec.adv = adv  # consumed by kernel.simulate
    return spec
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sim_inputs.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/mlb/inputs.py tests/test_sim_inputs.py
git commit -m "feat(sim): GameSpec inputs adapter over existing loaders"
```

---

### Task 10: Cross-check simulator vs analytic expected runs

A sanity gate: with defense/park neutral and a single pooled pitcher, the sim's mean total should land near the analytic `game.expected_runs` sum on the same vectors.

**Files:**
- Test: `tests/test_sim_kernel.py`

- [ ] **Step 1: Write the cross-check test**

```python
# add to tests/test_sim_kernel.py
from sportsmodel.model import game as gamemodel


def test_sim_mean_runs_near_analytic():
    spec = _spec()  # flat vectors, no park/def
    sims = K.simulate(spec, 6000, np.random.default_rng(11))
    sim_total = (sims.home_score.mean() + sims.away_score.mean())
    # analytic expected runs for the same flat offense vector, ~38 PA/team
    v = _flat_vec()
    analytic = 2 * gamemodel.expected_runs(v)
    assert abs(sim_total - analytic) < 1.5  # same ballpark; sim adds base-out realism
```

- [ ] **Step 2: Run it**

Run: `uv run pytest tests/test_sim_kernel.py::test_sim_mean_runs_near_analytic -v`
Expected: PASS (tune tolerance only if the base-out model legitimately shifts scoring; document if so).

- [ ] **Step 3: Commit**

```bash
git add tests/test_sim_kernel.py
git commit -m "test(sim): cross-check sim mean runs vs analytic expected runs"
```

---

## Phase 2 — Game-level backtest & validation gate

### Task 11: `backtest_sim.py` — walk-forward game metrics vs analytic baseline

Reuse the exact point-in-time harness of `scripts/backtest_game.py`; swap the closed-form `predict()` for a simulation, build the advancement table under the same cutoff, and print the same metrics report so results are directly comparable.

**Files:**
- Create: `scripts/backtest_sim.py`
- Test: `tests/test_backtest_sim.py` (smoke test on a tiny synthetic con)

**Interfaces:**
- Consumes: `transforms` builders, `build_advancement_table`, `AdvancementTable`, `inputs.build_game_spec`, `kernel.simulate`, `engine.pred_scores`, and `backtest_game.report`.
- Produces: `run_sim_backtest(season, n_sims) -> list[tuple]` returning `(p_home_win, home_won, pred_total, actual_total)` per game, same tuple shape `backtest_game.report` expects.

- [ ] **Step 1: Write a smoke test**

```python
# tests/test_backtest_sim.py
import numpy as np
from sportsmodel.sim.mlb import kernel
from sportsmodel.sim.engine import pred_scores


def test_pred_scores_from_sim_have_expected_keys():
    bs = [kernel.Batter(100 + i, _v(), _v()) for i in range(9)]
    aw = [kernel.Batter(200 + i, _v(), _v()) for i in range(9)]
    spec = kernel.GameSpec(bs, aw, kernel.Pitcher(1, 24, 3), kernel.Pitcher(2, 24, 3))
    sims = kernel.simulate(spec, 200, np.random.default_rng(0))
    d = pred_scores(sims)
    assert {"pred_total", "home_win_prob", "pred_margin"} <= set(d)


def _v():
    v = {"p_bb": .08, "p_k": .22, "p_1b": .15, "p_2b": .045,
         "p_3b": .004, "p_hr": .03, "p_out": .471}
    s = sum(v.values()); return {k: x / s for k, x in v.items()}
```

- [ ] **Step 2: Run to verify (should pass once Phase 1 is in)**

Run: `uv run pytest tests/test_backtest_sim.py -v`
Expected: PASS.

- [ ] **Step 3: Implement `backtest_sim.py`**

Model it on `backtest_game.py`: for each month cutoff, `transforms.set_cutoff(cutoff)`, build the same profiles **plus** `build_advancement_table(con)` (respecting the cutoff) → `AdvancementTable.from_rows(...)`. For each game that month, look up the actual starters + both actual lineups (from the parquet, the batters who actually hit), assemble a `GameSpec` via `inputs.build_game_spec`, run `kernel.simulate(spec, n_sims, rng)`, and emit `(home_win_prob, home_won, pred_total, actual_total)` from `engine.pred_scores`. Reuse `backtest_game.report` for output. Add `--season` and `--n-sims` (default 4000). For lineups in the backtest, take each team's 9 most-frequent batters that game ordered by first appearance (`arg_min(at_bat_number)` per batter within the game) — the actual order.

Provide the full script (mirror `backtest_game.py` structure; ~120 lines). Print sim metrics; then a human runs `backtest_game.py` for the baseline and compares.

- [ ] **Step 4: Run the real backtest (manual gate)**

Run: `uv run python scripts/backtest_sim.py --season 2025 --n-sims 4000`
Then: `uv run python scripts/backtest_game.py --season 2025`
Expected: sim prints the same report format. **Gate:** sim total MAE ≤ analytic MAE, and sim win-prob Brier ≤ analytic Brier + 0.002 (no meaningful regression), with equal-or-better calibration. Record both reports in the commit message.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_sim.py tests/test_backtest_sim.py
git commit -m "feat(sim): walk-forward game backtest vs analytic baseline"
```

**STOP — validation gate.** If the sim does not clear the gate, do not proceed to live wiring. Iterate the kernel (advancement grouping by out-state, TTO magnitudes, hook clamp) and re-run. Report results to the user for a go/no-go before Phase 3.

---

## Phase 3 — Player props + calibration

### Task 12: Prop distributions from sims + prop backtest

**Files:**
- Modify: `src/sportsmodel/sim/engine.py` (add `player_prop_dists`)
- Create: `scripts/backtest_sim_props.py` (or extend `backtest_props.py`)
- Test: `tests/test_sim_engine.py`

**Interfaces:**
- Produces: `player_prop_dists(sims, market_max: dict[str,int]) -> dict[int, dict[str, dict]]` mapping `player_id -> market -> {"kind":"pmf","pmf":[...], "mean": float}` for batter markets `hits, total_bases, home_run, hrr` and pitcher markets `pitcher_ks, hits_allowed, outs_recorded`. `home_run` pmf is `[P(0), P(≥1)]` to match the stored Y/N convention.

- [ ] **Step 1: Write the failing test**

```python
# add to tests/test_sim_engine.py
def test_player_prop_dists_shapes():
    import numpy as np
    from sportsmodel.sim.engine import GameSims, player_prop_dists
    sims = GameSims(
        home_score=np.array([1, 2]), away_score=np.array([0, 1]),
        batter_stats={100: {"hits": np.array([0, 2]), "total_bases": np.array([0, 3]),
                            "hr": np.array([0, 1]), "runs": np.array([0, 1]),
                            "rbi": np.array([0, 2]), "hrr": np.array([0, 5])}},
        pitcher_stats={1: {"k": np.array([5, 7]), "hits": np.array([4, 6]),
                           "outs": np.array([15, 18])}},
    )
    out = player_prop_dists(sims, {"hits": 5, "total_bases": 8, "hrr": 12,
                                   "pitcher_ks": 12, "hits_allowed": 12, "outs_recorded": 27})
    hr = out[100]["home_run"]["pmf"]
    assert len(hr) == 2 and abs(sum(hr) - 1.0) < 1e-9   # [P(0), P(>=1)]
    assert out[100]["hits"]["pmf"][0] == 0.5             # one sim had 0 hits
    assert out[1]["outs_recorded"]["mean"] == 16.5
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sim_engine.py::test_player_prop_dists_shapes -v`
Expected: FAIL.

- [ ] **Step 3: Implement `player_prop_dists`**

```python
# add to src/sportsmodel/sim/engine.py
_BATTER_MARKET_STAT = {"hits": "hits", "total_bases": "total_bases", "hrr": "hrr"}
_PITCHER_MARKET_STAT = {"pitcher_ks": "k", "hits_allowed": "hits", "outs_recorded": "outs"}


def _pmf_mean(arr, max_k):
    return {"kind": "pmf", "pmf": stat_pmf(arr, max_k), "mean": float(np.mean(arr))}


def player_prop_dists(sims: GameSims, market_max: dict) -> dict:
    out: dict = {}
    for pid, stats in sims.batter_stats.items():
        d = {}
        for market, stat in _BATTER_MARKET_STAT.items():
            d[market] = _pmf_mean(stats[stat], market_max[market])
        p_hr1 = float(np.mean(stats["hr"] >= 1))
        d["home_run"] = {"kind": "pmf", "pmf": [1 - p_hr1, p_hr1], "mean": float(np.mean(stats["hr"]))}
        out[pid] = d
    for pid, stats in sims.pitcher_stats.items():
        d = {}
        for market, stat in _PITCHER_MARKET_STAT.items():
            d[market] = _pmf_mean(stats[stat], market_max[market])
        out[pid] = d
    return out
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_sim_engine.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/engine.py tests/test_sim_engine.py
git commit -m "feat(sim): per-player prop distributions from sims"
```

- [ ] **Step 6: Prop backtest + calibration refit (manual)**

Extend the prop backtest harness to score sim prop `dist`s (calibration/Brier by market) against actual outcomes over 2025 walk-forward; then run `scripts/fit_calibration.py` to fit calibration for the sim markets. **Gate (c):** HRR and outs_recorded calibration improve vs the analytic props. Commit the calibration asset. Report to the user before Phase 4.

---

## Phase 4 — Live generation + wiring

### Task 13: Committed advancement asset + loader

**Files:**
- Create: `scripts/build_advancement.py` (writes `assets/advancement/mlb_advancement.parquet`)
- Modify: `src/sportsmodel/profiles.py` (add `load_advancement_rows()`)
- Test: `tests/test_advancement.py` (loader round-trip)

- [ ] **Step 1: Write the failing round-trip test**

```python
# add to tests/test_advancement.py
def test_advancement_rows_round_trip(tmp_path):
    import polars as pl
    from sportsmodel.sim.mlb.advancement import AdvancementTable
    rows = [{"outcome": "p_1b", "occ": 0, "outs": 0, "end_occ": 1, "runs": 0,
             "outs_added": None, "prob": 1.0}]
    p = tmp_path / "adv.parquet"
    pl.DataFrame(rows).write_parquet(p)
    back = pl.read_parquet(p).to_dicts()
    t = AdvancementTable.from_rows(back)
    assert t.sample(2, 0, 0.5) == (1, 0)
```

- [ ] **Step 2: Run to verify (passes once from_rows handles dict rows — it does)**

Run: `uv run pytest tests/test_advancement.py::test_advancement_rows_round_trip -v`
Expected: PASS.

- [ ] **Step 3: Implement the asset builder + loader**

`scripts/build_advancement.py`: connect DuckDB, `transforms.set_cutoff(None)` (use all available data), `rows = build_advancement_table(con)`, write to `assets/advancement/mlb_advancement.parquet` via polars. `profiles.load_advancement_rows()`: read that parquet → list of dicts (mirror the other `profiles.load_*` snapshot readers).

- [ ] **Step 4: Build the asset and commit**

Run: `uv run python scripts/build_advancement.py`
Expected: writes the parquet; print row count.

```bash
git add scripts/build_advancement.py src/sportsmodel/profiles.py assets/advancement/mlb_advancement.parquet tests/test_advancement.py
git commit -m "feat(sim): committed advancement asset + loader"
```

---

### Task 14: `generate_sim.py` — one simulation per game → game + prop rows

**Files:**
- Create: `scripts/generate_sim.py`
- Test: covered by the module smoke test in Task 11 + a dry-run against local DuckDB.

**Interfaces:**
- Consumes: schedule/profiles/lineups/weather loaders (as `generate_props.py`), `inputs.build_game_spec`, `kernel.simulate`, `engine.pred_scores` + `player_prop_dists`, `profiles.load_advancement_rows`, `AdvancementTable`.
- Produces: writes `game_predictions` rows (model_version `mlb-sim-v1`) and `prop_predictions` rows (model_version `mlb-sim-props-v1`) using the existing `upsert_game_predictions` / `upsert_prop_predictions` and the existing `dist` JSON encoding.

- [ ] **Step 1: Implement `generate_sim.py`**

Combine `generate_predictions.py` and `generate_props.py`: load schedule, profiles, workload, park/defense/weather, lineups (`mlb_lineups.lineups_for_game`), and the advancement asset once. For each game: assemble one `GameSpec`, `sims = kernel.simulate(spec, n_sims=20000, rng)`, then:
- game row from `engine.pred_scores(sims)` (calibrate `home_win_prob` via `calibration.calibrate("win_prob", ...)`), model_version `mlb-sim-v1`.
- prop rows from `engine.player_prop_dists(sims, MARKET_MAX)`; for each player/market write `projected_mean`, default `line` (reuse `props.DEFAULT_LINE`), `prob_over` (calibrated, evaluated at default line via `distributions.prob_over_dist`), and `dist = json.dumps(...)`, model_version `mlb-sim-props-v1`. Join batter/pitcher ids to the lineup names/teams already loaded.

Reuse `DEFAULT_LINE` and `MARKET_MAX` (e.g. `{"hits":6,"total_bases":10,"hrr":15,"pitcher_ks":15,"hits_allowed":15,"outs_recorded":30}`).

- [ ] **Step 2: Dry-run locally (no DB)**

Run: `PYTHONPATH=src uv run python scripts/generate_sim.py` (with no `DATABASE_URL`, writes to local DuckDB).
Expected: prints game + prop row counts; no exceptions.

- [ ] **Step 3: Commit**

```bash
git add scripts/generate_sim.py
git commit -m "feat(sim): generate_sim writes game + prop rows from one simulation"
```

---

### Task 15: Workflow wiring (parallel to analytic)

**Files:**
- Create: `.github/workflows/generate-sim.yml`
- Modify: `.github/workflows/refresh-profiles.yml` (also build the advancement asset)

- [ ] **Step 1: Add advancement build to refresh-profiles**

Add a step after profiles are rebuilt: `uv run python scripts/build_advancement.py`, and include `assets/advancement/mlb_advancement.parquet` in whatever commit/artifact step publishes the profile assets.

- [ ] **Step 2: Create `generate-sim.yml`**

Mirror the existing `daily-ingest`/`refresh-props` workflow shape (checkout, uv, `DATABASE_URL` secret), running `uv run python scripts/generate_sim.py` on the daily schedule **after** predictions/props so it runs parallel to the analytic model. Keep the analytic `generate_predictions`/`generate_props` running — both model_versions coexist for comparison.

- [ ] **Step 3: Validate YAML**

Run: `uv run --with pyyaml python -c "import yaml,glob; [yaml.safe_load(open(f)) for f in glob.glob('.github/workflows/*.yml')]; print('all valid')"`
Expected: `all valid`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/generate-sim.yml .github/workflows/refresh-profiles.yml
git commit -m "ci(sim): build advancement asset + run generate_sim parallel to analytic"
```

**Promotion (post-plan, user-driven):** once the sim's live predictions and CLV track alongside analytic for a couple of weeks and the backtest gates held, switch the dashboard default / grading focus to `mlb-sim-v1`. Not part of this plan.

---

## Self-Review

**Spec coverage:**
- Framework/kernel split → Tasks 4 (engine), 5-8 (kernel), 9 (inputs). ✓
- Base-out state machine + runner identity → Task 6. ✓
- Statcast advancement (tier-2, point-in-time) → Tasks 2, 3, 13. ✓
- Pitcher hook + TTO → Tasks 3 (TTO consts), 7 (hook loop). ✓
- Extra innings → Task 7. ✓
- Vectorization/performance → Task 8. ✓
- Empirical pmf outputs / dist schema unchanged → Tasks 4, 12. ✓
- generate_sim (one sim → game+props), coexisting model_versions → Task 14. ✓
- Backtest validation gate (a/b/c) → Tasks 11, 12. ✓
- Calibration refit → Task 12. ✓
- numpy dependency → Task 1. ✓
- No DB/schema/dashboard changes → honored (reuses upserts + dist JSON). ✓

**Placeholder scan:** Tasks 8, 11, 12(step 6), 14 describe larger scripts in prose rather than full literal code, because they are direct structural mirrors of existing files (`backtest_game.py`, `generate_predictions.py`+`generate_props.py`) — the executor copies the named source and swaps the named calls. Each names exact source files, functions, signatures, and the gate. This is intentional (DRY against existing patterns), not a vague placeholder. All *novel* logic (advancement math, state machine, aggregation) is written out in full.

**Type consistency:** outcome codes `BB/K/S/D/T/HR/OUT_INPLAY = 0..6` are defined once (Task 5) and reused (Tasks 3, 6, 8). `GameSims` fields (Task 4) match producers (Tasks 7, 8) and consumers (Tasks 11, 12). `GameSpec.adv` set in Task 9, read in Tasks 7-8. Advancement `sample(outcome_code, occ, u)->(end_occ,runs)` consistent across Tasks 3, 6.
