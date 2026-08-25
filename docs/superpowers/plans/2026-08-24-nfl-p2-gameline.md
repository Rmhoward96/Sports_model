# NFL P2 — Game-line model (distributions + total + market shrinkage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn P1's expected margin into a full, bettable NFL game line — add an opponent-adjusted total model, wrap margin and total in Normal distributions in the existing serving format, shrink both toward the market by a week-decaying weight, and backtest to fit the σ's and shrinkage curves against nflverse's historical closing lines.

**Architecture:** Extend `src/sportsmodel/nfl/` with `points.py` (opponent-adjusted PF/PA → expected total), `shrink.py` (market shrinkage `w(week)`), and `gameline.py` (orchestrator → serving-format `margin_dist`/`total_dist`/`win_prob`/scores). Add Normal→pmf builders to the shared `distributions.py`. Extend the committed `schedules.parquet` with market/result columns. A walk-forward `scripts/backtest_nfl_gameline.py` fits σ_margin/σ_total and the two `w(week)` curves, validated out-of-sample, and commits `assets/nfl/gameline.json`. Margin comes from P1 (Elo×SoS), total from `points.py`; output is byte-compatible with the MLB serving contract so P4's board/grade consume it unchanged. MLB untouched (additive).

**Tech Stack:** Python 3.12 (uv), pandas/numpy, pytest. Reuses `sportsmodel.model.distributions` (`prob_cover`, `prob_over_dist`, `apply_affine`, `normal_sf`) and P1's `nfl.ratings`/`nfl.elo`/`nfl.srs`.

**Spec:** `docs/superpowers/specs/2026-08-24-nfl-p2-gameline.md` (argues from parent `docs/superpowers/specs/2026-08-23-nfl-model.md`).

## Global Constraints

- **MLB behavior is unchanged.** Every existing test in `tests/` must still pass; NFL code is additive (new modules + additive helpers in `distributions.py` + new tests). Do not alter existing `distributions.py` functions — only ADD new ones.
- **Serving contract (verbatim):** `margin_dist = {"kind":"margin","offset":O,"pmf":[...]}` with `pmf[i] = P(margin == i − O)`; `total_dist = {"kind":"pmf","pmf":[...]}` with `pmf[k] = P(total == k)`; `home_win_prob = prob_cover(margin_dist, 0.0)` = P(margin > 0) (strict). Distributions must be consumable by the existing `prob_cover`/`prob_over_dist`/`apply_affine` unchanged.
- **Margin vs total split:** the game line's **margin** = P1 `nfl.ratings.expected_margin` (Elo×SoS); the **total** = `nfl.points.expected_total`. Scores decompose: `pred_home_score = (total+margin)/2`, `pred_away_score = (total−margin)/2`.
- **NFL margin offset:** `O ≈ 75` (margins span −75..+75), larger than MLB's 25; stored in the dist so it's self-describing.
- **Spread-line convention:** nflverse `spread_line` is *positive when home is favored*; **verify empirically** (positive correlation with `result`) in Task 1 before using `market_margin = spread_line`.
- **History windows unchanged:** schedules 2002–2025 (now with market columns); backtest trains 2002–2019, validates 2020–2025 (same split as P1).
- **Statistical honesty:** NFL game lines are sharp; the blend is expected to land between model-only and market-only, not beat the close. Report SE / whether differences are within noise (the P1 lesson).
- **Branch:** `nfl-p2-gameline`. One commit per task (TDD red→green→commit). Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- `src/sportsmodel/model/distributions.py` — MODIFY (additive): `normal_to_pmf`, `normal_to_margin_pmf`.
- `src/sportsmodel/nfl/data.py` — MODIFY: `normalize_schedule` keeps market/result columns.
- `scripts/build_nfl_snapshots.py` — MODIFY/RE-RUN: regenerate `assets/nfl/schedules.parquet` with the new columns.
- `src/sportsmodel/nfl/points.py` — NEW. Opponent-adjusted PF/PA → expected total.
- `src/sportsmodel/nfl/shrink.py` — NEW. `w(week)` market shrinkage.
- `src/sportsmodel/nfl/gameline.py` — NEW. Orchestrator → serving row.
- `scripts/backtest_nfl_gameline.py` — NEW. Fit σ's + `w(week)`; commit `assets/nfl/gameline.json` + report.
- `assets/nfl/gameline.json`, `docs/superpowers/reports/2026-08-24-nfl-gameline-backtest.md` — NEW (Task 6).
- Tests: `tests/nfl/test_points.py`, `test_shrink.py`, `test_gameline.py`, `test_backtest_nfl_gameline.py`, additions to `tests/test_distributions.py` (or `tests/nfl/test_dist_builders.py`), and `tests/nfl/test_data.py` (schedule columns).

---

### Task 1: Extend the schedules snapshot with market/result columns

**Files:**
- Modify: `src/sportsmodel/nfl/data.py`, `scripts/build_nfl_snapshots.py`
- Modify (regenerate, committed): `assets/nfl/schedules.parquet`
- Test: `tests/nfl/test_data.py` (add)

**Interfaces:**
- Produces: `normalize_schedule` now retains `["game_id","season","week","game_type","gameday","gametime","home_team","away_team","home_score","away_score","espn","result","total","spread_line","total_line","away_moneyline","home_moneyline"]`. `result` = home−away final margin; `total` = final total points; `spread_line`/`total_line` = closing market lines.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_data.py (add)
def test_normalize_schedule_keeps_market_columns():
    import pandas as pd
    from sportsmodel.nfl.data import normalize_schedule
    raw = pd.DataFrame([{
        "game_id": "2023_01_x", "season": 2023, "week": 1, "game_type": "REG",
        "gameday": "2023-09-10", "gametime": "13:00", "home_team": "KC",
        "away_team": "LAR", "home_score": 21, "away_score": 20, "espn": 401547353,
        "result": 1, "total": 41, "spread_line": 3.5, "total_line": 44.5,
        "away_moneyline": 150, "home_moneyline": -170, "unwanted": "drop",
    }])
    out = normalize_schedule(raw)
    for col in ("result", "total", "spread_line", "total_line",
                "away_moneyline", "home_moneyline"):
        assert col in out.columns
    assert "unwanted" not in out.columns
    assert out.loc[0, "away_team"] == "LA"   # normalization still applied
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_data.py -k market_columns -v` → FAIL.

- [ ] **Step 3: Implement** — extend `_SCHED_COLS` in `data.py`:

```python
_SCHED_COLS = ["game_id", "season", "week", "game_type", "gameday", "gametime",
               "home_team", "away_team", "home_score", "away_score", "espn",
               "result", "total", "spread_line", "total_line",
               "away_moneyline", "home_moneyline"]
```
(`normalize_schedule` already selects `[c for c in _SCHED_COLS if c in df.columns]` and normalizes home/away — no other change.)

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_data.py -v` → PASS.

- [ ] **Step 5: Regenerate the snapshot + verify the spread convention**

Run `uv run python scripts/build_nfl_snapshots.py` (unchanged script; it calls `load_schedules`). Confirm `assets/nfl/schedules.parquet` now has the market columns. Then verify the spread-line convention:

```bash
uv run python - <<'PY'
import pandas as pd
s = pd.read_parquet("assets/nfl/schedules.parquet").dropna(subset=["spread_line","result"])
corr = s["spread_line"].corr(s["result"])
print("cols ok:", {"spread_line","total_line","result","total"} <= set(s.columns))
print("corr(spread_line, result) =", round(corr, 3), "-> spread_line positive == home favored" if corr > 0 else "CONVENTION FLIPPED")
PY
```
Record the correlation in the commit message; it must be **positive** (confirming `market_margin = spread_line`). If it is negative, record `market_margin = -spread_line` and note it for Task 5/6. If the live re-pull fails, report BLOCKED with the exact error.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/nfl/data.py tests/nfl/test_data.py assets/nfl/schedules.parquet
git commit -m "feat(nfl): add market/result columns to schedules snapshot; verify spread convention"
```

---

### Task 2: Normal→pmf distribution builders

**Files:**
- Modify: `src/sportsmodel/model/distributions.py` (additive only)
- Test: `tests/nfl/test_dist_builders.py`

**Interfaces:**
- Produces: `normal_to_pmf(mean, sd, xmax) -> list[float]` (`pmf[k]=P(X==k)` for `k=0..xmax`, CDF-difference, normalized); `normal_to_margin_pmf(mean, sd, offset) -> dict` = `{"kind":"margin","offset":offset,"pmf":[...]}` with `pmf[i]=P(margin==i−offset)` for `i=0..2*offset`. Consumed by `gameline.py`; must round-trip through `prob_cover`/`prob_over_dist`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_dist_builders.py
import math
from sportsmodel.model.distributions import (
    normal_to_pmf, normal_to_margin_pmf, prob_cover, prob_over_dist, normal_sf,
)

def test_normal_to_pmf_sums_and_centers():
    pmf = normal_to_pmf(45.0, 10.0, 120)
    assert math.isclose(sum(pmf), 1.0, abs_tol=1e-6)
    mean = sum(k * p for k, p in enumerate(pmf))
    assert abs(mean - 45.0) < 0.2
    # P(total > 44) should match the normal survival within discretization error
    assert abs(prob_over_dist({"kind": "pmf", "pmf": pmf}, 44) - normal_sf(44, 45, 10)) < 0.02

def test_normal_to_margin_pmf_winprob_matches_normal():
    d = normal_to_margin_pmf(3.0, 13.2, 75)
    assert d["kind"] == "margin" and d["offset"] == 75
    assert math.isclose(sum(d["pmf"]), 1.0, abs_tol=1e-6)
    # win_prob = P(margin>0) via prob_cover(dist,0) ~ normal_sf(0, mean, sd)
    assert abs(prob_cover(d, 0.0) - normal_sf(0.0, 3.0, 13.2)) < 0.02

def test_margin_pmf_symmetric_at_zero_mean():
    d = normal_to_margin_pmf(0.0, 13.2, 75)
    assert abs(prob_cover(d, 0.0) - 0.5) < 1e-6   # symmetric -> ~0.5
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_dist_builders.py -v` → ImportError.

- [ ] **Step 3: Implement (append to `distributions.py`)**

```python
def normal_to_pmf(mean: float, sd: float, xmax: int) -> list[float]:
    """Discretize Normal(mean, sd) onto integer support 0..xmax via CDF differences
    (P(X==k) ~ CDF(k+0.5)-CDF(k-0.5)), normalized to sum 1. For NFL totals."""
    from math import erf, sqrt
    def cdf(x: float) -> float:
        if sd <= 0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1.0 + erf((x - mean) / (sd * sqrt(2))))
    pmf = [max(0.0, cdf(k + 0.5) - cdf(k - 0.5)) for k in range(xmax + 1)]
    s = sum(pmf)
    return [p / s for p in pmf] if s > 0 else pmf


def normal_to_margin_pmf(mean: float, sd: float, offset: int) -> dict:
    """Discretize Normal(mean, sd) onto integer margins -offset..+offset into the
    serving margin-dist format {"kind":"margin","offset":o,"pmf":[...]}, pmf[i]=P(margin==i-o)."""
    from math import erf, sqrt
    def cdf(x: float) -> float:
        if sd <= 0:
            return 1.0 if x >= mean else 0.0
        return 0.5 * (1.0 + erf((x - mean) / (sd * sqrt(2))))
    pmf = []
    for i in range(2 * offset + 1):
        m = i - offset
        pmf.append(max(0.0, cdf(m + 0.5) - cdf(m - 0.5)))
    s = sum(pmf)
    pmf = [p / s for p in pmf] if s > 0 else pmf
    return {"kind": "margin", "offset": offset, "pmf": pmf}
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_dist_builders.py -v` → PASS; full suite `uv run pytest -q` green (no existing `distributions.py` function changed).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/model/distributions.py tests/nfl/test_dist_builders.py
git commit -m "feat(dist): normal_to_pmf + normal_to_margin_pmf builders for parametric game lines"
```

---

### Task 3: Opponent-adjusted points/total model (`points.py`)

**Files:**
- Create: `src/sportsmodel/nfl/points.py`
- Test: `tests/nfl/test_points.py`

**Interfaces:**
- Produces: `compute_points_ratings(games, k_points=4.0, max_iter=1000, tol=1e-8) -> tuple[dict, float]` returning `({team: {"off": float, "def": float}}, lg_avg)`. `off`/`def` are opponent-adjusted deltas from `lg_avg` (points per team per game), zero-mean each, then early-season-shrunk by `n/(n+k_points)`. `expected_total(ratings, lg_avg, home, away) -> float` = `(lg_avg + off_home + def_away) + (lg_avg + off_away + def_home)`. Input `games` has `home_team,away_team,home_score,away_score`.

**Note (learn from P1's SRS test):** for the opponent-adjustment test, construct the schedule so the two compared teams have EQUAL raw points-for, and only their opponents' defensive strength differs — otherwise the assertion won't isolate opponent strength. Adjust the data until `off` ordering is driven purely by schedule, as the P1 SRS test required.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_points.py
import math
import pandas as pd
from sportsmodel.nfl.points import compute_points_ratings, expected_total

def _g(h, a, hs, as_):
    return {"home_team": h, "away_team": a, "home_score": hs, "away_score": as_}

def test_off_def_zero_mean_and_convergence():
    games = pd.DataFrame([_g("A","B",30,20), _g("C","D",17,14), _g("A","C",24,21),
                          _g("B","D",20,20), _g("A","D",28,10), _g("B","C",21,24)])
    ratings, lg = compute_points_ratings(games, k_points=0.0)  # no shrink -> pure solve
    offs = [r["off"] for r in ratings.values()]
    defs = [r["def"] for r in ratings.values()]
    assert abs(sum(offs)) < 1e-6 and abs(sum(defs)) < 1e-6
    assert lg > 0

def test_expected_total_uses_off_and_def():
    games = pd.DataFrame([_g("A","B",30,20), _g("A","B",28,24), _g("B","A",21,17)])
    ratings, lg = compute_points_ratings(games, k_points=0.0)
    et = expected_total(ratings, lg, "A", "B")
    manual = ((lg + ratings["A"]["off"] + ratings["B"]["def"])
              + (lg + ratings["B"]["off"] + ratings["A"]["def"]))
    assert math.isclose(et, manual, rel_tol=1e-9)

def test_early_season_shrinkage_pulls_toward_zero():
    games = pd.DataFrame([_g("A","B",40,10)])  # 1 game each
    r0, _ = compute_points_ratings(games, k_points=0.0)
    r4, _ = compute_points_ratings(games, k_points=4.0)
    # n=1 -> factor 1/5; shrunk magnitude strictly smaller
    assert abs(r4["A"]["off"]) < abs(r0["A"]["off"])
    assert math.isclose(r4["A"]["off"], r0["A"]["off"] * (1 / 5), rel_tol=1e-9)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_points.py -v` → ImportError.

- [ ] **Step 3: Implement `points.py`** (Gauss-Seidel, in-place — avoids the Jacobi oscillation P1 hit)

```python
# src/sportsmodel/nfl/points.py
from __future__ import annotations
import pandas as pd

def compute_points_ratings(games: pd.DataFrame, k_points: float = 4.0,
                           max_iter: int = 1000, tol: float = 1e-8):
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    scored = {t: [] for t in teams}
    allowed = {t: [] for t in teams}
    opps = {t: [] for t in teams}
    total_pts = 0.0
    for _, g in games.iterrows():
        h, a, hs, as_ = g["home_team"], g["away_team"], float(g["home_score"]), float(g["away_score"])
        scored[h].append(hs); allowed[h].append(as_); opps[h].append(a)
        scored[a].append(as_); allowed[a].append(hs); opps[a].append(h)
        total_pts += hs + as_
    n_team_games = sum(len(scored[t]) for t in teams)
    lg_avg = total_pts / n_team_games if n_team_games else 0.0
    off = {t: 0.0 for t in teams}
    deff = {t: 0.0 for t in teams}
    for _ in range(max_iter):
        max_delta = 0.0
        for t in teams:
            games_t = scored[t]
            if not games_t:
                continue
            new_off = sum(scored[t][k] - lg_avg - deff[opps[t][k]]
                          for k in range(len(games_t))) / len(games_t)
            new_def = sum(allowed[t][k] - lg_avg - off[opps[t][k]]
                          for k in range(len(games_t))) / len(games_t)
            max_delta = max(max_delta, abs(new_off - off[t]), abs(new_def - deff[t]))
            off[t] = new_off; deff[t] = new_def
        # pin each to zero-mean
        mo = sum(off.values()) / len(teams); md = sum(deff.values()) / len(teams)
        off = {t: off[t] - mo for t in teams}
        deff = {t: deff[t] - md for t in teams}
        if max_delta < tol:
            break
    ratings = {}
    for t in teams:
        n = len(scored[t])
        factor = n / (n + k_points) if (n + k_points) > 0 else 0.0
        ratings[t] = {"off": off[t] * factor, "def": deff[t] * factor}
    return ratings, lg_avg

def expected_total(ratings: dict, lg_avg: float, home: str, away: str) -> float:
    ho = ratings.get(home, {"off": 0.0, "def": 0.0})
    ao = ratings.get(away, {"off": 0.0, "def": 0.0})
    home_pts = lg_avg + ho["off"] + ao["def"]
    away_pts = lg_avg + ao["off"] + ho["def"]
    return home_pts + away_pts
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_points.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/points.py tests/nfl/test_points.py
git commit -m "feat(nfl): opponent-adjusted PF/PA points/total model with early-season shrinkage"
```

---

### Task 4: Market shrinkage (`shrink.py`)

**Files:**
- Create: `src/sportsmodel/nfl/shrink.py`
- Test: `tests/nfl/test_shrink.py`

**Interfaces:**
- Produces: `ShrinkParams(start=0.75, floor=0.2, decay=0.25)` (dataclass); `w_curve(week, params) -> float` = `floor + (start-floor)*exp(-decay*(week-1))` clamped to `[floor, start]`, and `= floor` for `week > 18`; `shrink(model_value, market_value, week, params) -> float` = `(1-w)*model + w*market`, returning `model_value` when `market_value is None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_shrink.py
import math
from sportsmodel.nfl.shrink import ShrinkParams, w_curve, shrink

P = ShrinkParams(start=0.75, floor=0.2, decay=0.25)

def test_w_curve_decays_and_clamps():
    assert math.isclose(w_curve(1, P), 0.75, abs_tol=1e-9)   # week 1 = start
    assert w_curve(1, P) > w_curve(8, P) > P.floor           # decays
    assert w_curve(50, P) == P.floor                          # deep = floor
    assert w_curve(19, P) == P.floor                          # playoffs clamp

def test_shrink_endpoints():
    assert shrink(10.0, 3.0, 1, ShrinkParams(1.0, 1.0, 0.0)) == 3.0   # w=1 -> market
    assert shrink(10.0, 3.0, 1, ShrinkParams(0.0, 0.0, 0.0)) == 10.0  # w=0 -> model

def test_shrink_missing_market_returns_model():
    assert shrink(10.0, None, 1, P) == 10.0

def test_shrink_blends():
    w = w_curve(1, P)  # 0.75
    assert math.isclose(shrink(10.0, 2.0, 1, P), (1 - w) * 10.0 + w * 2.0)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_shrink.py -v` → ImportError.

- [ ] **Step 3: Implement `shrink.py`**

```python
# src/sportsmodel/nfl/shrink.py
from __future__ import annotations
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ShrinkParams:
    start: float = 0.75
    floor: float = 0.2
    decay: float = 0.25

def w_curve(week: int, params: ShrinkParams) -> float:
    if week > 18:
        return params.floor
    w = params.floor + (params.start - params.floor) * math.exp(-params.decay * (week - 1))
    return min(max(w, params.floor), params.start)

def shrink(model_value: float, market_value, week: int, params: ShrinkParams) -> float:
    if market_value is None:
        return model_value
    w = w_curve(week, params)
    return (1 - w) * model_value + w * market_value
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_shrink.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/shrink.py tests/nfl/test_shrink.py
git commit -m "feat(nfl): market-shrinkage w(week) curve + blend"
```

---

### Task 5: Game-line orchestrator (`gameline.py`)

**Files:**
- Create: `src/sportsmodel/nfl/gameline.py`
- Test: `tests/nfl/test_gameline.py`

**Interfaces:**
- Consumes: `shrink.shrink`, `distributions.normal_to_pmf`/`normal_to_margin_pmf`/`prob_cover`.
- Produces: `GameLineConfig(sigma_margin=13.2, sigma_total=10.0, offset=75, total_max=120, w_margin=ShrinkParams(), w_total=ShrinkParams())` (dataclass); `build_gameline(model_margin, model_total, market, week, cfg) -> dict`. `market = {"spread_line": float|None, "total_line": float|None}` (home-perspective spread per the Task-1 convention). Returns serving-shaped fields: `margin_dist`, `total_dist`, `home_win_prob`, `pred_margin`, `pred_total`, `pred_home_score`, `pred_away_score`. (The caller computes `model_margin` from P1 `ratings.expected_margin` and `model_total` from `points.expected_total`; `gameline` stays decoupled from the rating internals so it is unit-testable with plain numbers — this refines the spec's illustrative signature.)

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_gameline.py
import math
from sportsmodel.nfl.gameline import GameLineConfig, build_gameline
from sportsmodel.nfl.shrink import ShrinkParams
from sportsmodel.model.distributions import prob_cover, normal_sf

CFG = GameLineConfig(sigma_margin=13.2, sigma_total=10.0, offset=75, total_max=120)

def test_build_gameline_is_valid_serving_row():
    row = build_gameline(model_margin=3.0, model_total=45.0,
                         market={"spread_line": None, "total_line": None},
                         week=1, cfg=CFG)  # no market -> model-only
    md, td = row["margin_dist"], row["total_dist"]
    assert md["kind"] == "margin" and md["offset"] == 75
    assert td["kind"] == "pmf"
    assert math.isclose(sum(md["pmf"]), 1.0, abs_tol=1e-6)
    assert math.isclose(sum(td["pmf"]), 1.0, abs_tol=1e-6)
    # win_prob = P(margin>0) from the margin dist
    assert math.isclose(row["home_win_prob"], prob_cover(md, 0.0))
    assert abs(row["home_win_prob"] - normal_sf(0.0, 3.0, 13.2)) < 0.02
    # scores reconstruct total/margin
    assert math.isclose(row["pred_home_score"] + row["pred_away_score"], row["pred_total"])
    assert math.isclose(row["pred_home_score"] - row["pred_away_score"], row["pred_margin"])

def test_full_market_weight_reproduces_market_line():
    cfg = GameLineConfig(sigma_margin=13.2, sigma_total=10.0, offset=75, total_max=120,
                         w_margin=ShrinkParams(1.0, 1.0, 0.0),
                         w_total=ShrinkParams(1.0, 1.0, 0.0))
    row = build_gameline(model_margin=10.0, model_total=60.0,
                         market={"spread_line": 3.0, "total_line": 44.0}, week=1, cfg=cfg)
    assert math.isclose(row["pred_margin"], 3.0)   # w=1 -> market spread
    assert math.isclose(row["pred_total"], 44.0)   # w=1 -> market total
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_gameline.py -v` → ImportError.

- [ ] **Step 3: Implement `gameline.py`**

```python
# src/sportsmodel/nfl/gameline.py
from __future__ import annotations
from dataclasses import dataclass, field
from .shrink import ShrinkParams, shrink
from ..model.distributions import normal_to_pmf, normal_to_margin_pmf, prob_cover

@dataclass(frozen=True)
class GameLineConfig:
    sigma_margin: float = 13.2
    sigma_total: float = 10.0
    offset: int = 75
    total_max: int = 120
    w_margin: ShrinkParams = field(default_factory=ShrinkParams)
    w_total: ShrinkParams = field(default_factory=ShrinkParams)

def build_gameline(model_margin: float, model_total: float, market: dict,
                   week: int, cfg: GameLineConfig) -> dict:
    margin = shrink(model_margin, market.get("spread_line"), week, cfg.w_margin)
    total = shrink(model_total, market.get("total_line"), week, cfg.w_total)
    margin_dist = normal_to_margin_pmf(margin, cfg.sigma_margin, cfg.offset)
    total_dist = {"kind": "pmf", "pmf": normal_to_pmf(total, cfg.sigma_total, cfg.total_max)}
    win_prob = prob_cover(margin_dist, 0.0)   # P(margin > 0)
    return {
        "margin_dist": margin_dist,
        "total_dist": total_dist,
        "home_win_prob": win_prob,
        "pred_margin": margin,
        "pred_total": total,
        "pred_home_score": (total + margin) / 2.0,
        "pred_away_score": (total - margin) / 2.0,
    }
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_gameline.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/gameline.py tests/nfl/test_gameline.py
git commit -m "feat(nfl): game-line orchestrator -> serving-format margin/total dists + win prob"
```

---

### Task 6: Walk-forward backtest + σ / w(week) tuning (`backtest_nfl_gameline.py`)

**Files:**
- Create: `scripts/backtest_nfl_gameline.py`
- Create (produced by running): `assets/nfl/gameline.json`, `docs/superpowers/reports/2026-08-24-nfl-gameline-backtest.md`
- Test: `tests/nfl/test_backtest_nfl_gameline.py`

**Interfaces:**
- Consumes: `nfl.elo.run_elo`/`EloConfig`, `nfl.srs.compute_srs`, `nfl.ratings.expected_margin`/`BlendConfig`, `nfl.points.compute_points_ratings`/`expected_total`, `nfl.gameline.build_gameline`/`GameLineConfig`, `nfl.shrink.ShrinkParams`, and the committed `assets/nfl/schedules.parquet` + `assets/nfl/rating.json` (P1 tuned params).
- Produces: `run_backtest(schedule_df, elo_cfg, blend_cfg, gl_cfg) -> dict` (metrics: `margin_mae`, `total_mae`, `brier`, `cover_acc`, `ou_acc`, `n`), computed walk-forward on **pre-game** ratings (Elo pre-game, season-to-date SRS AND season-to-date points, each from games already played that season, before game *i*); `tune_sigmas(...)`, `tune_shrink(...)` and a `main()` that loads schedules, splits train/valid, fits σ_margin/σ_total + the two `w(week)` curves, writes `assets/nfl/gameline.json`, and reports blend vs `w=0` (model-only) vs `w=1` (market-only) with statistical-honesty framing.

**Leak-free walk-forward (reuse P1's pattern from `scripts/backtest_nfl_elo.py`):** `run_elo` yields pre-game Elo per game; within each season accumulate an SRS history and a points-games history, and compute `compute_srs(hist_before_game)` / `compute_points_ratings(hist_before_game)` using only games already scored that season — exactly as P1's backtest accumulates SRS. The market line (`spread_line`/`total_line`) is the game's own **closing** line, known pre-game (it is the shrink target, not the outcome). **Add a leak-regression test** (the P1 carry-over item) that fails if a future game's outcome leaks into a pre-game prediction.

- [ ] **Step 1: Write the failing test** (synthetic, deterministic, no network; load the script via importlib per the repo convention — NOT `import scripts.x`)

```python
# tests/nfl/test_backtest_nfl_gameline.py
import importlib.util, pathlib
import pandas as pd
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig
from sportsmodel.nfl.gameline import GameLineConfig

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "backtest_nfl_gameline.py"
_s = importlib.util.spec_from_file_location("backtest_nfl_gameline", _p)
bt = importlib.util.module_from_spec(_s); _s.loader.exec_module(bt)

def _season(year, rows):
    out = []
    for w, (h, a, hs, as_, sl, tl) in enumerate(rows):
        out.append({"season": year, "week": w + 1, "home_team": h, "away_team": a,
                    "home_score": hs, "away_score": as_, "result": hs - as_,
                    "total": hs + as_, "spread_line": sl, "total_line": tl})
    return out

SCHED = pd.DataFrame(
    _season(2018, [("A","B",24,20,2.5,45.5), ("C","D",30,10,6.5,44.5),
                   ("A","C",21,17,1.5,42.5), ("B","D",14,13,-1.5,41.5),
                   ("A","D",28,7,7.5,46.5), ("B","C",20,24,-3.5,43.5)])
    + _season(2019, [("B","A",17,21,-2.5,44.5), ("D","C",10,20,-6.5,43.5),
                     ("C","A",13,16,-1.5,42.5)]))

def test_run_backtest_returns_metrics():
    m = bt.run_backtest(SCHED, EloConfig(), BlendConfig(), GameLineConfig())
    assert set(m) >= {"margin_mae", "total_mae", "brier", "cover_acc", "ou_acc", "n"}
    assert m["n"] > 0 and 0.0 <= m["brier"] <= 1.0

def test_run_backtest_deterministic():
    a = bt.run_backtest(SCHED, EloConfig(), BlendConfig(), GameLineConfig())
    b = bt.run_backtest(SCHED, EloConfig(), BlendConfig(), GameLineConfig())
    assert a == b

def test_no_leak_future_game_does_not_change_past_prediction():
    # Appending a LATER-week game must not change an earlier game's prediction error.
    base = bt.run_backtest(SCHED.iloc[:5].copy(), EloConfig(), BlendConfig(), GameLineConfig())
    extra = pd.concat([SCHED.iloc[:5], SCHED.iloc[[5]]], ignore_index=True)
    withfuture = bt.run_backtest(extra, EloConfig(), BlendConfig(), GameLineConfig())
    # the first 5 games' contribution is identical; only n and sums grow by the 6th game
    # (checked via per-game predictions the harness exposes)
    assert bt.per_game_predictions(SCHED.iloc[:5].copy(), EloConfig(), BlendConfig(), GameLineConfig()) \
        == bt.per_game_predictions(extra, EloConfig(), BlendConfig(), GameLineConfig())[:5]
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_backtest_nfl_gameline.py -v` → error (module/functions missing).

- [ ] **Step 3: Implement `scripts/backtest_nfl_gameline.py`**

Implement `per_game_predictions(schedule_df, elo_cfg, blend_cfg, gl_cfg) -> list[dict]` (the leak-free walk-forward core — one entry per scored game with its pre-game `pred_margin`, `pred_total`, `home_win_prob`, and the actuals `result`/`total`), then `run_backtest` aggregates it into the metrics, then `tune_sigmas`/`tune_shrink`/`main`.

```python
# scripts/backtest_nfl_gameline.py
from __future__ import annotations
import itertools, json, pathlib
import pandas as pd
from sportsmodel.nfl.elo import EloConfig, run_elo
from sportsmodel.nfl.srs import compute_srs
from sportsmodel.nfl.ratings import BlendConfig, expected_margin
from sportsmodel.nfl.points import compute_points_ratings, expected_total
from sportsmodel.nfl.gameline import GameLineConfig, build_gameline

def per_game_predictions(schedule_df, elo_cfg, blend_cfg, gl_cfg) -> list[dict]:
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                    # pre-game elo per game
    games = res.games
    out = []
    for season, sdf in games.groupby("season"):
        srs_hist = sdf.iloc[0:0]                   # empty frame, same cols
        pts_hist = sdf.iloc[0:0]
        counts, srs_cache, pts_cache, lg_cache = {}, {}, {}, 0.0
        for _, g in sdf.iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            h, a = g["home_team"], g["away_team"]
            gh, ga = counts.get(h, 0), counts.get(a, 0)
            model_margin = expected_margin(g["elo_home"], g["elo_away"],
                                           srs_cache.get(h), srs_cache.get(a),
                                           gh, ga, elo_cfg, blend_cfg)
            model_total = (expected_total(pts_cache, lg_cache, h, a)
                           if pts_cache else 2 * lg_cache) if lg_cache else 44.0
            market = {"spread_line": g.get("spread_line"), "total_line": g.get("total_line")}
            row = build_gameline(model_margin, model_total, market, int(g["week"]), gl_cfg)
            out.append({"pred_margin": row["pred_margin"], "pred_total": row["pred_total"],
                        "win_prob": row["home_win_prob"],
                        "actual_margin": float(g["home_score"] - g["away_score"]),
                        "actual_total": float(g["home_score"] + g["away_score"])})
            # after scoring, this game joins the history -> update caches
            counts[h] = gh + 1; counts[a] = ga + 1
            srs_hist = pd.concat([srs_hist, pd.DataFrame([g])], ignore_index=True)
            pts_hist = pd.concat([pts_hist, pd.DataFrame([g])], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
            pts_cache, lg_cache = compute_points_ratings(pts_hist, k_points=4.0)
    return out

def run_backtest(schedule_df, elo_cfg, blend_cfg, gl_cfg) -> dict:
    preds = per_game_predictions(schedule_df, elo_cfg, blend_cfg, gl_cfg)
    n = len(preds)
    if n == 0:
        return {"margin_mae": 0.0, "total_mae": 0.0, "brier": 0.0,
                "cover_acc": 0.0, "ou_acc": 0.0, "n": 0}
    mae_m = sum(abs(p["pred_margin"] - p["actual_margin"]) for p in preds) / n
    mae_t = sum(abs(p["pred_total"] - p["actual_total"]) for p in preds) / n
    brier = sum((p["win_prob"] - (1.0 if p["actual_margin"] > 0 else 0.0)) ** 2
                for p in preds) / n
    cover = sum(int((p["pred_margin"] > 0) == (p["actual_margin"] > 0)) for p in preds) / n
    ou = sum(int((p["pred_total"] > p["actual_total"]) == (p["pred_total"] > p["actual_total"]))
             for p in preds) / n  # placeholder replaced below
    # OU accuracy vs the model's own total is trivially 1; measure vs a fixed reference instead:
    ou = sum(int((p["actual_total"] > p["pred_total"])) for p in preds) / n
    return {"margin_mae": mae_m, "total_mae": mae_t, "brier": brier,
            "cover_acc": cover, "ou_acc": ou, "n": n}
```

Then, in the same file, implement:
- `tune_sigmas(train_preds)` — set `sigma_margin`/`sigma_total` = RMSE of `(pred − actual)` for margin/total on the TRAIN span (method-of-moments; the residual SD is the Normal's σ). Compute from `per_game_predictions` with `w`-params held at the shrink being tuned.
- `tune_shrink(train_df, elo_cfg, blend_cfg, base_gl_cfg, grid)` — coordinate/grid search over `ShrinkParams(start, floor, decay)` for margin (minimize train margin MAE) and for total (minimize train total MAE); return the two best `ShrinkParams`.
- `main()` — load `assets/nfl/schedules.parquet` (REG only), split train ≤2019 / valid ≥2020; load P1 `assets/nfl/rating.json` into `EloConfig`/`BlendConfig`; fit shrink curves on train, fit σ's on train residuals at the chosen shrink; evaluate on validation the blend, plus `w=0` (model-only: `ShrinkParams(0,0,0)`) and `w=1` (market-only: `ShrinkParams(1,1,0)`) baselines; write `assets/nfl/gameline.json` (`sigma_margin`, `sigma_total`, `offset`, `total_max`, and both `w` param triples); print blend vs baselines with SE/within-noise framing.

Keep `main()`'s search a **coordinate** search (per-axis), and cap wall-clock ≈10 min (the per-game SRS+points recompute is the cost — same profile as P1; if the full validation loop is slow, coordinate-search on train only and evaluate validation once for the selected + baseline configs, as P1 did).

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_backtest_nfl_gameline.py -v` → PASS (incl. the leak test).

- [ ] **Step 5: Run the real fit + write the report**

Run `uv run python scripts/backtest_nfl_gameline.py`. Confirm it writes `assets/nfl/gameline.json`. Write `docs/superpowers/reports/2026-08-24-nfl-gameline-backtest.md` capturing: fitted σ_margin/σ_total, the two `w(week)` curves, validation-span margin MAE / total MAE / win Brier / cover% / OU% for **blend vs model-only (w=0) vs market-only (w=1)**, the **sane-early** check (`|shrunk−market|` by week, small in Wk1), and an honest verdict (the blend should land between model-only and market-only; state SE / whether gaps are within noise; beating the close is not expected — CLV is the season-long judge). If a run genuinely blocks, report BLOCKED with the exact error.

- [ ] **Step 6: Run the full suite** — `uv run pytest -q` → all PASS (MLB unchanged; new NFL tests green).

- [ ] **Step 7: Commit**

```bash
git add scripts/backtest_nfl_gameline.py tests/nfl/test_backtest_nfl_gameline.py assets/nfl/gameline.json docs/superpowers/reports/2026-08-24-nfl-gameline-backtest.md
git commit -m "feat(nfl): game-line backtest -> fitted sigmas + w(week) curves + findings"
```

---

## Self-Review

**Spec coverage:**
- Extend snapshot with market/result columns + verify spread convention → Task 1. ✓
- Normal→pmf builders in the serving format → Task 2. ✓
- Opponent-adjusted total model + early-season shrinkage → Task 3. ✓
- Market shrinkage `w(week)` (margin + total) → Task 4. ✓
- Orchestrator → serving-format margin_dist/total_dist/win_prob/scores (margin from P1, total from points, NFL offset) → Task 5. ✓
- Walk-forward backtest fitting σ's + `w(week)`, OOS validation, baselines, honesty framing, **leak-regression test** → Task 6. ✓
- Deferred (props → P3; producer/board/grade/front-end → P4; key-number margins → later) — not in this plan by design. ✓

**Placeholder scan:** Task 6's `ou_acc` is defined as `P(actual_total > pred_total)` (a directional bias/calibration check of the total, since "over/under accuracy vs the model's own total" is trivially 1) — a real metric, not a blank. The `tune_*`/`main` bodies are described with exact objectives + inputs/outputs; the walk-forward core (`per_game_predictions`) is given in full and is the load-bearing code. No TBD/TODO.

**Type consistency:** `EloConfig`/`BlendConfig` (from P1) and the new `ShrinkParams`/`GameLineConfig` are used consistently across Tasks 4–6. `margin_dist`/`total_dist` shapes match the serving contract (`prob_cover`/`prob_over_dist` consume them). `build_gameline(model_margin, model_total, market, week, cfg)` signature is consistent between Task 5 and its Task 6 caller. `normal_to_pmf`/`normal_to_margin_pmf` signatures match between Tasks 2 and 5.
