# NFL Props ML — Props-1 (feature table + learned sim inputs + gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the leakage-safe player-week feature table, the learned volume/efficiency models that replace the sim's hand-set inputs (approach A), and the walk-forward ablation-ladder harness that delivers the first ship-gate verdict for A.

**Architecture:** Pure feature builders (`nfl/context.py`, `nfl/player_features.py`) produce one table used for both training and prediction. `sim/nfl/learned.py` trains scikit-learn HistGradientBoosting models and applies them to the sim's `TeamRates`/`PlayerInput`s through a new `spec_hook` in `backtest_sim_nfl.run_backtest`. `model/props_eval.py` holds the scoring rules (RPS, PIT, clustered bootstrap, rung decision); `scripts/train_props_ml.py` runs the ladder and writes the report. Nothing in serving changes in Props-1.

**Tech Stack:** Python 3.12 (uv), pandas, numpy, scikit-learn 1.9 (HistGradientBoosting), pytest. nflverse release parquets over HTTPS (never in tests).

**Spec:** `docs/superpowers/specs/2026-09-24-nfl-props-ml-design.md`

## Global Constraints

- Leakage: a feature for (season S, week w) uses only games strictly before (S, w). Pre-game information for week w (injury report, depth chart, schedule/roof/rest, pre-kickoff line) is allowed. Enforced by a perturbation test (Task 5).
- Walk-forward only — no random train/test splits anywhere.
- Features absent for an era stay NaN (HistGradientBoosting handles NaN); no imputation except the documented indoor-weather fill (indoor ⇒ temp 70°F, wind 0 mph).
- Tests never touch the network: monkeypatch `pd.read_parquet` / use synthetic frames.
- No DDL, no Supabase writes, no secrets printed. Nothing in serving (`generate_sim_nfl.py`, workflows, board) changes in Props-1.
- No new dependencies.
- TDD: failing test first, then implementation. Run `uv run pytest -q` (full suite green) before each commit.
- Commit messages end with a blank line then `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Team codes go through `sportsmodel.nfl.teams.normalize_team` (nflverse uses `LA`, `LAC`, `LV`, plus historical `OAK`/`SD`).
- nflverse `spread_line` is POSITIVE when the HOME team is favored.

## Rulings made while planning (recorded; the spec is the authority)

1. **Route participation dropped.** Only derivable from `pbp_participation`, which is not published for 2026 (404 on 2026-09-24) — training on a feature that can't be served live is a train/serve mismatch. `snap_pct` stays.
2. **Precipitation dropped.** Not in nflverse schedules historically; forecast-only would be train/serve mismatch. Temp + wind stay (schedule history; Open-Meteo forecast is Props-2 serving work).
3. **Rookie flag → `p_career_games`** (prior games since 2016, capped at 100) — rookie status isn't in the weekly feed; career games carries the same signal.
4. **Props-1 gates the four markets the backtest already pairs with actuals**: pass_yds, rush_yds, rec_yds, receptions. rush_att / pass_tds / anytime_td join in Props-2 with the B models.
5. **Market rung in Props-1 = market features in the volume models only** (spread, total, implied team total). The score-tilt use of the implied total is deferred to Props-2.
6. **2025+ depth charts are fixed first (Task 2).** nflverse switched to daily snapshots in 2025; the current backtest collapses them, so every 2025 week reused a stale chart. Each (season, week, team) now takes the latest snapshot at or before kickoff. The baseline must be honest before anything is compared to it.
7. **Pooled score = mean over markets of relative RPS improvement** `1 − RPS_cand/RPS_base` (scale-free; raw pooling would be dominated by yardage markets).
8. **PIT "does not worsen" tolerance = +0.005 decile ECE** (sampling noise on ~5k records per market).
9. **Refit blocks** before weeks 1, 5, 9, 13, 17 of each test season; inner tuning picks `season_decay ∈ {1.0, 0.8, 0.6}` × `max_iter ∈ {150, 300}` by Poisson deviance of the targets model on season S−1 (trained on seasons < S−1).
10. **Live `generate_sim_nfl.py` depth handling is untouched** in Props-1 (its live snapshot path already picks the latest chart); unification happens in Props-2.

## File Structure

| File | Responsibility |
|---|---|
| `src/sportsmodel/nfl/nflverse.py` (modify) | Add `depth`, `schedules`, `ngs_receiving/rushing/passing` release datasets + `EXPECTED_COLUMNS` validation |
| `src/sportsmodel/sim/nfl/usage.py` (modify) | `depth_charts_asof(raw, schedules)` — per (season, week, team) chart, both schemas |
| `scripts/backtest_sim_nfl.py` (modify) | Use `depth_charts_asof`; add `spec_hook` + `record` params |
| `assets/nfl/stadiums.json` (create) | stadium_id → lat, lon, tz, default roof |
| `src/sportsmodel/nfl/context.py` (create) | Per team-game context features (`cx_*`) + market features (`mk_*`) |
| `src/sportsmodel/nfl/player_features.py` (create) | Population, labels, team-game aggregates, rolling features (`p_*`, `ngs_*`, `tm_*`, `op_*`, `st_*`) |
| `scripts/build_player_features.py` (create) | IO: fetch sources → write `data/props_ml/player_week_features.parquet` + `team_week_features.parquet` |
| `src/sportsmodel/sim/nfl/learned.py` (create) | Feature sets per rung, training, inner tuning, prediction, apply to `TeamRates`/`PlayerInput` |
| `src/sportsmodel/model/props_eval.py` (create) | RPS, PIT, decile ECE, clustered bootstrap, rung decision |
| `scripts/train_props_ml.py` (create) | Walk-forward ablation ladder → `assets/nfl/props_ml/a_gate.json` + report markdown |
| `.gitignore` (modify) | ignore `data/props_ml/` (large parquet — Release asset in Props-2) |

Tests: `tests/nfl/test_nflverse.py` (extend), `tests/sim/nfl/test_usage_asof.py`, `tests/sim/nfl/test_backtest_sim_nfl.py` (extend), `tests/nfl/test_context.py`, `tests/nfl/test_player_features.py`, `tests/sim/nfl/test_learned.py`, `tests/model/test_props_eval.py`, `tests/scripts/test_train_props_ml.py`.

---

### Task 1: nflverse loaders with schema validation

**Files:**
- Modify: `src/sportsmodel/nfl/nflverse.py`
- Test: `tests/nfl/test_nflverse.py`

**Interfaces:**
- Produces: `load_release(dataset, seasons, *, required=True)` now also accepts `"depth"`, `"schedules"`, `"ngs_receiving"`, `"ngs_rushing"`, `"ngs_passing"`. Single-file datasets (schedules, ngs_*) are read once and filtered to `seasons`. `EXPECTED_COLUMNS: dict[str, frozenset[str]]`; `validate_columns(df, dataset) -> None` raises `ValueError` naming the missing columns.

- [ ] **Step 1: Write the failing tests** (append to `tests/nfl/test_nflverse.py`)

```python
import pandas as pd
import pytest

from sportsmodel.nfl import nflverse


def test_single_file_dataset_is_read_once_and_filtered(monkeypatch):
    calls = []
    frame = pd.DataFrame({"season": [2020, 2021, 2022], "week": [1, 1, 1],
                          "player_gsis_id": ["a", "b", "c"], "avg_separation": [3.0, 3.1, 3.2],
                          "avg_cushion": [6.0] * 3, "avg_intended_air_yards": [9.0] * 3,
                          "avg_yac_above_expectation": [0.1] * 3, "season_type": ["REG"] * 3})
    monkeypatch.setattr(nflverse.pd, "read_parquet", lambda url: calls.append(url) or frame)
    out = nflverse.load_release("ngs_receiving", [2021, 2022])
    assert len(calls) == 1 and calls[0].endswith("nextgen_stats/ngs_receiving.parquet")
    assert sorted(out["season"].tolist()) == [2021, 2022]


def test_validate_columns_names_missing(monkeypatch):
    with pytest.raises(ValueError, match="avg_separation"):
        nflverse.validate_columns(pd.DataFrame({"season": [2021]}), "ngs_receiving")


def test_depth_release_url_is_per_season(monkeypatch):
    urls = []
    old = pd.DataFrame({"season": [2021], "club_code": ["KC"], "week": [1], "depth_team": ["1"],
                        "gsis_id": ["x"], "position": ["QB"], "football_name": ["P"], "last_name": ["M"],
                        "first_name": ["P"]})
    monkeypatch.setattr(nflverse.pd, "read_parquet", lambda url: urls.append(url) or old)
    nflverse.load_release("depth", [2021])
    assert urls == ["https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_2021.parquet"]


def test_schedules_validated(monkeypatch):
    bad = pd.DataFrame({"season": [2021], "week": [1]})
    monkeypatch.setattr(nflverse.pd, "read_parquet", lambda url: bad)
    with pytest.raises(ValueError, match="home_rest"):
        nflverse.load_release("schedules", [2021])
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/nfl/test_nflverse.py -q`
Expected: FAIL (`KeyError: 'ngs_receiving'` / `AttributeError: validate_columns`).

- [ ] **Step 3: Implement** (in `src/sportsmodel/nfl/nflverse.py`; keep existing functions)

```python
_BASE = "https://github.com/nflverse/nflverse-data/releases/download"
_RELEASE_URL = {
    "pbp": f"{_BASE}/pbp/play_by_play_{{year}}.parquet",
    "weekly": f"{_BASE}/stats_player/stats_player_week_{{year}}.parquet",
    "snaps": f"{_BASE}/snap_counts/snap_counts_{{year}}.parquet",
    "depth": f"{_BASE}/depth_charts/depth_charts_{{year}}.parquet",
}
# One file covering every season; read once, filtered to the requested seasons.
_SINGLE_FILE_URL = {
    "schedules": f"{_BASE}/schedules/games.parquet",
    "ngs_receiving": f"{_BASE}/nextgen_stats/ngs_receiving.parquet",
    "ngs_rushing": f"{_BASE}/nextgen_stats/ngs_rushing.parquet",
    "ngs_passing": f"{_BASE}/nextgen_stats/ngs_passing.parquet",
}
# Columns the props-ML features read. A release that drops one fails loudly
# here instead of silently producing NaN features (lesson of the 2025
# depth-chart schema change).
EXPECTED_COLUMNS: dict[str, frozenset[str]] = {
    "schedules": frozenset({"game_id", "season", "week", "game_type", "gameday", "gametime",
                            "home_team", "away_team", "home_rest", "away_rest", "roof", "surface",
                            "temp", "wind", "div_game", "stadium_id", "spread_line", "total_line"}),
    "ngs_receiving": frozenset({"season", "week", "player_gsis_id", "avg_separation", "avg_cushion",
                                "avg_intended_air_yards", "avg_yac_above_expectation"}),
    "ngs_rushing": frozenset({"season", "week", "player_gsis_id", "efficiency",
                              "percent_attempts_gte_eight_defenders", "rush_yards_over_expected_per_att"}),
    "ngs_passing": frozenset({"season", "week", "player_gsis_id", "avg_time_to_throw",
                              "completion_percentage_above_expectation", "aggressiveness"}),
}


def validate_columns(df: pd.DataFrame, dataset: str) -> None:
    """Raise ValueError if `df` lacks any EXPECTED_COLUMNS[dataset] column."""
    missing = sorted(EXPECTED_COLUMNS.get(dataset, frozenset()) - set(df.columns))
    if missing:
        raise ValueError(f"nflverse {dataset}: missing expected columns {missing}")
```

Change `load_release` so single-file datasets short-circuit, and validate before returning:

```python
def load_release(dataset: str, seasons: list[int], *, required: bool = True) -> pd.DataFrame:
    """... (keep docstring; add:) ``schedules`` and ``ngs_*`` are single files
    read once and filtered to ``seasons``; every dataset in EXPECTED_COLUMNS is
    schema-validated."""
    if dataset in _SINGLE_FILE_URL:
        df = pd.read_parquet(_SINGLE_FILE_URL[dataset])
        validate_columns(df, dataset)
        df = df[df["season"].isin(seasons)].reset_index(drop=True)
        if df.empty and required:
            raise RuntimeError(f"nflverse {dataset}: no rows for seasons {seasons}")
        return df
    ... existing per-season loop unchanged ...
    out = pd.concat(frames, ignore_index=True)
    validate_columns(out, dataset)
    return out
```

- [ ] **Step 4: Run tests** — `uv run pytest tests/nfl/test_nflverse.py -q` → PASS; `uv run pytest -q` → all green.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/nflverse.py tests/nfl/test_nflverse.py
git commit -m "feat(nflverse): depth/schedules/NGS release loaders with schema validation"
```

---

### Task 2: Per-week depth charts for 2025+ snapshots (baseline fix)

**Files:**
- Modify: `src/sportsmodel/sim/nfl/usage.py`, `scripts/backtest_sim_nfl.py`
- Test: `tests/sim/nfl/test_usage_asof.py` (create), `tests/sim/nfl/test_backtest_sim_nfl.py` (extend)

**Interfaces:**
- Consumes: `load_release("depth", seasons)` (Task 1) — old schema for ≤2024 (`season, club_code, week, depth_team, gsis_id, position, football_name, first_name, last_name`), snapshot schema for 2025+ (`dt, team, player_name, gsis_id, pos_abb, pos_rank`).
- Produces: `depth_charts_asof(raw: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame` in the OLD columns `active_usage` reads (`season, week, club_code, depth_team, position, gsis_id, full_name, football_name`). `schedules` needs `season, week, game_type, gameday, gametime, home_team, away_team`. `run_backtest(..., spec_hook=None, record=None)`.

- [ ] **Step 1: Write the failing tests** (`tests/sim/nfl/test_usage_asof.py`)

```python
import pandas as pd

from sportsmodel.sim.nfl.usage import depth_charts_asof

SCHED = pd.DataFrame({
    "season": [2025, 2025], "week": [1, 2], "game_type": ["REG", "REG"],
    "gameday": ["2025-09-07", "2025-09-14"], "gametime": ["13:00", "13:00"],
    "home_team": ["KC", "LAC"], "away_team": ["LAC", "KC"],
})


def _snap(dt, team, gsis, pos, rank, name):
    return {"dt": dt, "team": team, "gsis_id": gsis, "pos_abb": pos, "pos_rank": rank, "player_name": name}


def test_snapshot_rows_take_latest_chart_at_or_before_kickoff():
    raw = pd.DataFrame([
        _snap("2025-09-01T10:00:00Z", "KC", "q1", "QB", 1, "Old Starter"),
        _snap("2025-09-06T10:00:00Z", "KC", "q2", "QB", 1, "Week1 Starter"),
        _snap("2025-09-12T10:00:00Z", "KC", "q3", "QB", 1, "Week2 Starter"),
        _snap("2025-09-20T10:00:00Z", "KC", "q4", "QB", 1, "Future"),
    ])
    out = depth_charts_asof(raw, SCHED)
    kc = out[out["club_code"] == "KC"].set_index("week")["gsis_id"]
    assert kc.loc[1] == "q2" and kc.loc[2] == "q3"      # never the future snapshot
    assert set(out.columns) >= {"season", "week", "club_code", "depth_team", "position",
                                "gsis_id", "full_name", "football_name"}


def test_old_schema_rows_pass_through_and_mix_with_snapshots():
    old = pd.DataFrame({"season": [2024], "week": [3], "club_code": ["KC"], "depth_team": ["1"],
                        "gsis_id": ["p"], "position": ["QB"], "football_name": ["Pat"],
                        "first_name": ["Pat"], "last_name": ["M"]})
    new = pd.DataFrame([_snap("2025-09-06T10:00:00Z", "KC", "q2", "QB", 1, "X")])
    out = depth_charts_asof(pd.concat([old, new], ignore_index=True), SCHED)
    assert ((out["season"] == 2024) & (out["week"] == 3) & (out["gsis_id"] == "p")).any()
    assert ((out["season"] == 2025) & (out["week"] == 1) & (out["gsis_id"] == "q2")).any()
    assert out.loc[out["gsis_id"] == "p", "full_name"].iloc[0] == "Pat M"


def test_team_with_no_snapshot_before_kickoff_gets_no_rows():
    raw = pd.DataFrame([_snap("2025-09-20T10:00:00Z", "LAC", "z", "QB", 1, "Late")])
    out = depth_charts_asof(raw, SCHED)
    assert out[(out["club_code"] == "LAC") & (out["week"] == 1)].empty
```

Extend `tests/sim/nfl/test_backtest_sim_nfl.py`:

```python
import inspect
import importlib.util, sys
from pathlib import Path

def _load_backtest():
    p = Path(__file__).resolve().parents[3] / "scripts" / "backtest_sim_nfl.py"
    spec = importlib.util.spec_from_file_location("backtest_sim_nfl_mod", p)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_run_backtest_accepts_spec_hook_and_record():
    params = inspect.signature(_load_backtest().run_backtest).parameters
    assert "spec_hook" in params and "record" in params
```

(If the test module already defines a loader for the script, reuse it instead of `_load_backtest`.)

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/sim/nfl/test_usage_asof.py tests/sim/nfl/test_backtest_sim_nfl.py -q` → FAIL (ImportError / missing params).

- [ ] **Step 3: Implement `depth_charts_asof`** (append to `src/sportsmodel/sim/nfl/usage.py`)

```python
_ET = "America/New_York"


def _kickoffs_utc(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team) with that team's kickoff in UTC.
    nflverse `gameday`/`gametime` are US-Eastern local."""
    s = schedules[schedules["game_type"] == "REG"]
    local = pd.to_datetime(s["gameday"].astype(str) + " " + s["gametime"].fillna("13:00").astype(str),
                           errors="coerce")
    ko = local.dt.tz_localize(_ET, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    rows = []
    for side in ("home_team", "away_team"):
        rows.append(pd.DataFrame({"season": s["season"].astype(int), "week": s["week"].astype(int),
                                  "team": s[side].astype(str), "kickoff": ko}))
    return pd.concat(rows, ignore_index=True).dropna(subset=["kickoff"])


def depth_charts_asof(raw: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Per-(season, week, team) depth charts in the OLD columns `active_usage`
    reads, from a frame mixing nflverse's two schemas. PURE.

    - Old weekly schema (has a non-null `club_code`, seasons <= 2024): passed
      through; `full_name` = "first last" (falls back to football_name).
    - Snapshot schema (2025+: `dt`, `team`, `pos_abb`, `pos_rank`): for each
      team's REG-season game, take that team's latest snapshot with
      `dt <= kickoff` (UTC) — never a later one — stamped with that game's
      (season, week). A team with no snapshot before kickoff gets no rows
      (active_usage's _latest_depth_week fallback then applies).
    """
    cols = ["season", "week", "club_code", "depth_team", "position", "gsis_id", "full_name", "football_name"]
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=cols)
    parts: list[pd.DataFrame] = []
    is_old = raw["club_code"].notna() if "club_code" in raw.columns else pd.Series(False, index=raw.index)
    old = raw[is_old]
    if len(old):
        first = old.get("first_name", pd.Series(pd.NA, index=old.index)).astype("string")
        last = old.get("last_name", pd.Series(pd.NA, index=old.index)).astype("string")
        full = (first.fillna("") + " " + last.fillna("")).str.strip()
        full = full.where(full != "", old["football_name"].astype("string"))
        parts.append(pd.DataFrame({
            "season": old["season"].astype(int), "week": old["week"].astype(int),
            "club_code": old["club_code"].astype("string"),
            "depth_team": pd.to_numeric(old["depth_team"], errors="coerce"),
            "position": old["position"].astype("string"), "gsis_id": old["gsis_id"].astype("string"),
            "full_name": full, "football_name": old["football_name"].astype("string"),
        }))
    new = raw[~is_old]
    if len(new) and {"dt", "team", "pos_abb", "gsis_id"}.issubset(new.columns):
        snaps = new.assign(_dt=pd.to_datetime(new["dt"], errors="coerce", utc=True)).dropna(subset=["_dt"])
        ko = _kickoffs_utc(schedules)
        for team, tsnaps in snaps.groupby("team"):
            times = tsnaps["_dt"].drop_duplicates().sort_values()
            for g in ko[ko["team"] == team].itertuples(index=False):
                eligible = times[times <= g.kickoff]
                if eligible.empty:
                    continue
                chart = tsnaps[tsnaps["_dt"] == eligible.iloc[-1]]
                name = chart["player_name"].astype("string")
                parts.append(pd.DataFrame({
                    "season": g.season, "week": g.week, "club_code": str(team),
                    "depth_team": pd.to_numeric(chart.get("pos_rank"), errors="coerce"),
                    "position": chart["pos_abb"].astype("string"), "gsis_id": chart["gsis_id"].astype("string"),
                    "full_name": name, "football_name": name,
                }))
    return pd.concat(parts, ignore_index=True)[cols] if parts else pd.DataFrame(columns=cols)
```

- [ ] **Step 4: Wire into `scripts/backtest_sim_nfl.py`**

In `run_backtest`, replace the `normalize_depth_charts(usage_src["depth"], ...)` line with:

```python
    from sportsmodel.nfl.nflverse import load_release
    # Per-week charts: old weekly schema <= 2024; 2025+ snapshots resolved to the
    # latest chart at/before each team's kickoff (the prior collapse applied one
    # stale chart to every 2025 week).
    depth_df = depth_charts_asof(load_release("depth", fetch_seasons), schedules)
```

Move `schedules = load_schedules(fetch_seasons)` above this line; import `depth_charts_asof` alongside the other `usage` imports; drop the now-unused `normalize_depth_charts` import (keep `fetch_usage_sources` for snaps/ids).

Add parameters and behavior:

```python
def run_backtest(
    seasons, n_sims, seed=SIM_SEED, on_game=None, season_decay=1.0, questionable_weight=1.0,
    home_field=0.0, use_defense=True, ratings_weight=0.0,
    spec_hook: Callable[[int, int, str, str, "NflGameSpec"], "NflGameSpec"] | None = None,
    record: list | None = None,
) -> dict:
    """... (append to docstring)
    spec_hook: optional `spec_hook(season, week, home, away, spec) -> spec`
        applied after `build_spec_from_usage` and before `simulate_game` -- the
        props-ML harness uses it to swap in learned team volume / player shares.
    record: optional list; when given, one dict per (player, market in
        PLAYER_MARKETS) with a paired actual is appended: season, week, home,
        player_id, market, mean, p50, p90, rps, pit, actual. Ungated (the
        caller applies the population gate). Small records, not pmfs, so five
        full walk-forwards fit in memory.
    """
```

Right after `spec = build_spec_from_usage(...)`:

```python
            if spec_hook is not None:
                spec = spec_hook(season, week, home, away, spec)
```

In the per-player loop, after `a = actual[market]`:

```python
                if record is not None:
                    record.append({
                        "season": season, "week": week, "home": home, "player_id": player_id,
                        "market": market, "mean": float(dist["mean"]),
                        "p50": quantile_from_pmf(dist, 0.50), "p90": quantile_from_pmf(dist, 0.90),
                        "rps": rps_pmf(dist["pmf"], a),
                        "pit": pit_pmf(dist["pmf"], a, pit_uniform(season, week, player_id, market)),
                        "actual": a,
                    })
```

with `from sportsmodel.model.props_eval import pit_pmf, pit_uniform, rps_pmf` — **these land in Task 8**. To keep this task green now, add them in this task as the first three functions of `src/sportsmodel/model/props_eval.py` (Task 8 adds the rest):

```python
"""Scoring rules for the props-ML ship gate (pure, no IO)."""
from __future__ import annotations

import zlib

import numpy as np


def rps_pmf(pmf, actual: float) -> float:
    """Ranked probability score (discrete CRPS) of a pmf over 0..K for an
    integer-valued actual; the actual is clipped into [0, K]. Lower is better."""
    p = np.asarray(pmf, dtype=float)
    k = len(p) - 1
    a = int(min(max(round(actual), 0), k))
    cdf = np.cumsum(p)
    step = (np.arange(k + 1) >= a).astype(float)
    return float(np.sum((cdf - step) ** 2))


def pit_uniform(season: int, week: int, player_id: str, market: str) -> float:
    """Deterministic U(0,1) per record (crc32 of the key), identical across
    candidates so randomized PITs are paired."""
    h = zlib.crc32(f"{season}|{week}|{player_id}|{market}".encode())
    return (h % 1_000_003) / 1_000_003


def pit_pmf(pmf, actual: float, u: float) -> float:
    """Randomized PIT for a discrete forecast: F(a-1) + u * P(a)."""
    p = np.asarray(pmf, dtype=float)
    k = len(p) - 1
    a = int(min(max(round(actual), 0), k))
    below = float(p[:a].sum())
    return below + u * float(p[a])
```

and tests in `tests/model/test_props_eval.py`:

```python
import numpy as np
from sportsmodel.model.props_eval import pit_pmf, pit_uniform, rps_pmf

def test_rps_zero_for_point_mass_on_actual():
    assert rps_pmf([0, 0, 1, 0], 2) == 0.0

def test_rps_penalizes_distance():
    near, far = [0, 1, 0, 0, 0], [0, 0, 0, 0, 1]
    assert rps_pmf(near, 2) < rps_pmf(far, 2)

def test_pit_bounds_and_determinism():
    u = pit_uniform(2024, 3, "p", "rec_yds")
    assert u == pit_uniform(2024, 3, "p", "rec_yds") and 0 <= u < 1
    assert pit_pmf([0.25, 0.25, 0.5], 1, 0.0) == 0.25
    assert pit_pmf([0.25, 0.25, 0.5], 1, 1.0) == 0.5
```

- [ ] **Step 5: Run tests** — targeted files PASS, then `uv run pytest -q` all green.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/sim/nfl/usage.py scripts/backtest_sim_nfl.py src/sportsmodel/model/props_eval.py tests/sim/nfl/test_usage_asof.py tests/sim/nfl/test_backtest_sim_nfl.py tests/model/test_props_eval.py
git commit -m "fix(sim): per-week 2025+ depth charts in the backtest; spec_hook + record for the props-ML harness"
```

---

### Task 3: Stadium asset + context and market features

**Files:**
- Create: `assets/nfl/stadiums.json`, `src/sportsmodel/nfl/context.py`
- Test: `tests/nfl/test_context.py`

**Interfaces:**
- Consumes: schedules frame (Task 1 `load_release("schedules", ...)`).
- Produces: `load_stadiums() -> dict[str, dict]`; `team_game_context(schedules, stadiums) -> pd.DataFrame` one row per (season, week, team) REG game with columns `season, week, team, opponent, is_home, stadium_id` and features `cx_rest, cx_rest_diff, cx_short_week, cx_off_bye, cx_travel_km, cx_tz_shift, cx_west_early, cx_indoor, cx_turf, cx_temp, cx_wind, cx_div`, plus market `mk_spread, mk_total, mk_implied`.

- [ ] **Step 1: Create `assets/nfl/stadiums.json`** — every stadium_id seen 2016–2026 (47). `roof`: `dome` | `retractable` | `outdoors`.

```json
{
  "ATL00": {"lat": 33.755, "lon": -84.401, "tz": "America/New_York", "roof": "dome"},
  "ATL97": {"lat": 33.755, "lon": -84.401, "tz": "America/New_York", "roof": "retractable"},
  "BAL00": {"lat": 39.278, "lon": -76.623, "tz": "America/New_York", "roof": "outdoors"},
  "BOS00": {"lat": 42.091, "lon": -71.264, "tz": "America/New_York", "roof": "outdoors"},
  "BUF00": {"lat": 42.774, "lon": -78.787, "tz": "America/New_York", "roof": "outdoors"},
  "CAR00": {"lat": 35.226, "lon": -80.853, "tz": "America/New_York", "roof": "outdoors"},
  "CHI98": {"lat": 41.862, "lon": -87.617, "tz": "America/Chicago", "roof": "outdoors"},
  "CIN00": {"lat": 39.095, "lon": -84.516, "tz": "America/New_York", "roof": "outdoors"},
  "CLE00": {"lat": 41.506, "lon": -81.700, "tz": "America/New_York", "roof": "outdoors"},
  "DAL00": {"lat": 32.748, "lon": -97.093, "tz": "America/Chicago", "roof": "retractable"},
  "DEN00": {"lat": 39.744, "lon": -105.020, "tz": "America/Denver", "roof": "outdoors"},
  "DET00": {"lat": 42.340, "lon": -83.046, "tz": "America/New_York", "roof": "dome"},
  "FRA00": {"lat": 50.069, "lon": 8.645, "tz": "Europe/Berlin", "roof": "outdoors"},
  "GER00": {"lat": 48.219, "lon": 11.625, "tz": "Europe/Berlin", "roof": "outdoors"},
  "GNB00": {"lat": 44.501, "lon": -88.062, "tz": "America/Chicago", "roof": "outdoors"},
  "HOU00": {"lat": 29.685, "lon": -95.411, "tz": "America/Chicago", "roof": "retractable"},
  "IND00": {"lat": 39.760, "lon": -86.164, "tz": "America/Indiana/Indianapolis", "roof": "retractable"},
  "JAX00": {"lat": 30.324, "lon": -81.637, "tz": "America/New_York", "roof": "outdoors"},
  "KAN00": {"lat": 39.049, "lon": -94.484, "tz": "America/Chicago", "roof": "outdoors"},
  "LAX01": {"lat": 33.953, "lon": -118.339, "tz": "America/Los_Angeles", "roof": "dome"},
  "LAX97": {"lat": 33.864, "lon": -118.261, "tz": "America/Los_Angeles", "roof": "outdoors"},
  "LAX99": {"lat": 34.014, "lon": -118.288, "tz": "America/Los_Angeles", "roof": "outdoors"},
  "LON00": {"lat": 51.556, "lon": -0.280, "tz": "Europe/London", "roof": "outdoors"},
  "LON01": {"lat": 51.456, "lon": -0.342, "tz": "Europe/London", "roof": "outdoors"},
  "LON02": {"lat": 51.604, "lon": -0.066, "tz": "Europe/London", "roof": "retractable"},
  "MAD01": {"lat": 40.453, "lon": -3.688, "tz": "Europe/Madrid", "roof": "retractable"},
  "MEL00": {"lat": -37.820, "lon": 144.983, "tz": "Australia/Melbourne", "roof": "outdoors"},
  "MEX00": {"lat": 19.303, "lon": -99.150, "tz": "America/Mexico_City", "roof": "outdoors"},
  "MIA00": {"lat": 25.958, "lon": -80.239, "tz": "America/New_York", "roof": "outdoors"},
  "MIN01": {"lat": 44.974, "lon": -93.258, "tz": "America/Chicago", "roof": "dome"},
  "MUN01": {"lat": 48.219, "lon": 11.625, "tz": "Europe/Berlin", "roof": "outdoors"},
  "NAS00": {"lat": 36.166, "lon": -86.771, "tz": "America/Chicago", "roof": "outdoors"},
  "NOR00": {"lat": 29.951, "lon": -90.081, "tz": "America/Chicago", "roof": "dome"},
  "NYC01": {"lat": 40.814, "lon": -74.074, "tz": "America/New_York", "roof": "outdoors"},
  "OAK00": {"lat": 37.752, "lon": -122.201, "tz": "America/Los_Angeles", "roof": "outdoors"},
  "PAR00": {"lat": 48.924, "lon": 2.360, "tz": "Europe/Paris", "roof": "outdoors"},
  "PHI00": {"lat": 39.901, "lon": -75.168, "tz": "America/New_York", "roof": "outdoors"},
  "PHO00": {"lat": 33.528, "lon": -112.263, "tz": "America/Phoenix", "roof": "retractable"},
  "PIT00": {"lat": 40.447, "lon": -80.016, "tz": "America/New_York", "roof": "outdoors"},
  "RIO00": {"lat": -22.912, "lon": -43.230, "tz": "America/Sao_Paulo", "roof": "outdoors"},
  "SAO00": {"lat": -23.545, "lon": -46.474, "tz": "America/Sao_Paulo", "roof": "outdoors"},
  "SDG00": {"lat": 32.783, "lon": -117.120, "tz": "America/Los_Angeles", "roof": "outdoors"},
  "SEA00": {"lat": 47.595, "lon": -122.332, "tz": "America/Los_Angeles", "roof": "outdoors"},
  "SFO01": {"lat": 37.403, "lon": -121.970, "tz": "America/Los_Angeles", "roof": "outdoors"},
  "TAM00": {"lat": 27.976, "lon": -82.503, "tz": "America/New_York", "roof": "outdoors"},
  "VEG00": {"lat": 36.091, "lon": -115.184, "tz": "America/Los_Angeles", "roof": "dome"},
  "WAS00": {"lat": 38.908, "lon": -76.864, "tz": "America/New_York", "roof": "outdoors"}
}
```

- [ ] **Step 2: Write the failing tests** (`tests/nfl/test_context.py`)

```python
import pandas as pd
import pytest

from sportsmodel.nfl.context import haversine_km, load_stadiums, team_game_context

STAD = {"KAN00": {"lat": 39.049, "lon": -94.484, "tz": "America/Chicago", "roof": "outdoors"},
        "LAX01": {"lat": 33.953, "lon": -118.339, "tz": "America/Los_Angeles", "roof": "dome"},
        "LON02": {"lat": 51.604, "lon": -0.066, "tz": "Europe/London", "roof": "retractable"}}


def _games():
    return pd.DataFrame({
        "game_id": ["g1", "g2", "g3"], "season": [2024] * 3, "week": [1, 2, 3], "game_type": ["REG"] * 3,
        "gameday": ["2024-09-08", "2024-09-15", "2024-09-22"], "gametime": ["16:25", "13:00", "09:30"],
        "home_team": ["KC", "LAC", "KC"], "away_team": ["LAC", "KC", "LAC"],
        "home_rest": [7, 7, 7], "away_rest": [7, 7, 14], "roof": ["outdoors", "dome", ""],
        "surface": ["grass", "matrixturf", "grass"], "temp": [80.0, None, None], "wind": [9.0, None, None],
        "div_game": [1, 1, 1], "stadium_id": ["KAN00", "LAX01", "LON02"],
        "spread_line": [3.0, -2.5, 6.0], "total_line": [48.0, 45.0, 47.5],
    })


def test_haversine_known_distance():
    assert haversine_km(39.049, -94.484, 33.953, -118.339) == pytest.approx(2200, rel=0.03)


def test_all_stadium_ids_have_coords():
    st = load_stadiums()
    assert len(st) == 47 and all({"lat", "lon", "tz", "roof"} <= set(v) for v in st.values())


def test_context_rows_and_features():
    cx = team_game_context(_games(), STAD).set_index(["week", "team"])
    assert len(cx) == 6
    # home game: no travel; LAC at KC travels ~2200 km, crosses 2 time zones
    assert cx.loc[(1, "KC"), "cx_travel_km"] == 0.0
    assert cx.loc[(1, "LAC"), "cx_travel_km"] == pytest.approx(2200, rel=0.03)
    assert cx.loc[(1, "LAC"), "cx_tz_shift"] == 2.0
    # dome: indoor fill (70F, 0 mph); retractable w/ blank roof: unknown -> NaN weather
    assert cx.loc[(2, "KC"), "cx_indoor"] == 1 and cx.loc[(2, "KC"), "cx_temp"] == 70.0 and cx.loc[(2, "KC"), "cx_wind"] == 0.0
    assert pd.isna(cx.loc[(3, "KC"), "cx_indoor"]) and pd.isna(cx.loc[(3, "KC"), "cx_wind"])
    # rest + bye
    assert cx.loc[(3, "LAC"), "cx_off_bye"] == 1 and cx.loc[(3, "LAC"), "cx_rest_diff"] == 7
    # market: home favored by 3, total 48 -> KC implied 25.5, LAC 22.5
    assert cx.loc[(1, "KC"), "mk_spread"] == 3.0 and cx.loc[(1, "LAC"), "mk_spread"] == -3.0
    assert cx.loc[(1, "KC"), "mk_implied"] == 25.5 and cx.loc[(1, "LAC"), "mk_implied"] == 22.5
    # turf flag
    assert cx.loc[(2, "LAC"), "cx_turf"] == 1 and cx.loc[(1, "KC"), "cx_turf"] == 0
```

- [ ] **Step 3: Run to verify failure** — `uv run pytest tests/nfl/test_context.py -q` → FAIL (module missing).

- [ ] **Step 4: Implement `src/sportsmodel/nfl/context.py`**

```python
"""Per team-game context features (rest, travel, time zones, roof/surface,
weather) and market features (spread, total, implied team total) for the
props-ML feature table. PURE except `load_stadiums` (reads a committed asset).

Leakage: every value here is known before kickoff (schedule, rest days, roof,
the line). Historical weather is the recorded game weather (live will use a
forecast -- a stated train/serve difference in the spec).
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

_STADIUMS = Path(__file__).resolve().parents[3] / "assets" / "nfl" / "stadiums.json"
_INDOOR_TEMP_F, _INDOOR_WIND_MPH = 70.0, 0.0
_PACIFIC = {"America/Los_Angeles"}


def load_stadiums() -> dict[str, dict]:
    return json.loads(_STADIUMS.read_text())


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _utc_offset_h(tz: str, day: str) -> float:
    return ZoneInfo(tz).utcoffset(datetime.fromisoformat(f"{day}T12:00:00")).total_seconds() / 3600


def _home_stadiums(games: pd.DataFrame) -> dict[tuple[int, str], str]:
    """(season, team) -> the stadium the team played most home games in that
    season -- data-driven, so relocations (OAK->LV, SD->LAC) and
    international 'home' games resolve correctly."""
    def _mode_first(s: pd.Series) -> str:
        counts = s.value_counts()
        top = counts.max()
        return next(v for v in s if counts[v] == top)   # tie -> earliest game (a London 'home' game can't win a tie)

    g = games.sort_values(["season", "week"])
    h = g.groupby(["season", "home_team"])["stadium_id"].agg(_mode_first)
    return {(int(k[0]), str(k[1])): v for k, v in h.items()}


def _indoor(roof: str, stadium_roof: str) -> float:
    r = (roof or "").strip().lower()
    if r in ("dome", "closed"):
        return 1.0
    if r in ("outdoors", "open"):
        return 0.0
    # blank roof (future games): known from the stadium unless retractable
    if stadium_roof == "dome":
        return 1.0
    if stadium_roof == "outdoors":
        return 0.0
    return float("nan")


def team_game_context(schedules: pd.DataFrame, stadiums: dict[str, dict]) -> pd.DataFrame:
    g = schedules[schedules["game_type"] == "REG"].copy()
    for c in ("home_team", "away_team"):
        g[c] = g[c].map(normalize_team)
    homes = _home_stadiums(g)
    rows = []
    for r in g.itertuples(index=False):
        st = stadiums.get(r.stadium_id, {})
        indoor = _indoor(r.roof, st.get("roof", ""))
        if indoor == 1.0:
            temp, wind = _INDOOR_TEMP_F, _INDOOR_WIND_MPH
        elif indoor == 0.0:
            temp = float(r.temp) if pd.notna(r.temp) else float("nan")
            wind = float(r.wind) if pd.notna(r.wind) else float("nan")
        else:
            temp = wind = float("nan")
        turf = 0.0 if str(r.surface or "").strip().lower() in ("grass", "") else 1.0
        spread = float(r.spread_line) if pd.notna(r.spread_line) else float("nan")
        total = float(r.total_line) if pd.notna(r.total_line) else float("nan")
        kick_h = int(str(r.gametime or "13:00").split(":")[0])
        for side, team, opp, rest, opp_rest in (
            ("home", r.home_team, r.away_team, r.home_rest, r.away_rest),
            ("away", r.away_team, r.home_team, r.away_rest, r.home_rest),
        ):
            base_id = homes.get((int(r.season), team))
            base = stadiums.get(base_id or "", {})
            if st and base:
                travel = haversine_km(base["lat"], base["lon"], st["lat"], st["lon"])
                tz_shift = abs(_utc_offset_h(st["tz"], str(r.gameday)) - _utc_offset_h(base["tz"], str(r.gameday)))
            else:
                travel = tz_shift = float("nan")
            sign = 1.0 if side == "home" else -1.0
            rows.append({
                "season": int(r.season), "week": int(r.week), "team": team, "opponent": opp,
                "is_home": 1 if side == "home" else 0, "stadium_id": r.stadium_id,
                "cx_rest": float(rest), "cx_rest_diff": float(rest) - float(opp_rest),
                "cx_short_week": float(rest <= 5), "cx_off_bye": float(rest >= 13),
                "cx_travel_km": round(travel, 1) if not math.isnan(travel) else travel,
                "cx_tz_shift": tz_shift,
                "cx_west_early": float(base.get("tz") in _PACIFIC and kick_h < 14 and st.get("tz") == "America/New_York"),
                "cx_indoor": indoor, "cx_turf": turf, "cx_temp": temp, "cx_wind": wind,
                "cx_div": float(r.div_game),
                "mk_spread": sign * spread, "mk_total": total,
                "mk_implied": (total + sign * spread) / 2 if not (math.isnan(total) or math.isnan(spread)) else float("nan"),
            })
    return pd.DataFrame(rows)
```

Note: in `_games()` the home team's own travel is 0 because `travel = haversine(base, st)` where base == st. Home games in London (week 3) produce real travel for both teams — correct.

- [ ] **Step 5: Run tests** — `uv run pytest tests/nfl/test_context.py -q` PASS; full suite green.

- [ ] **Step 6: Commit**

```bash
git add assets/nfl/stadiums.json src/sportsmodel/nfl/context.py tests/nfl/test_context.py
git commit -m "feat(nfl): team-game context + market features and stadium asset for props ML"
```

---

### Task 4: Population, labels and team-game aggregates

**Files:**
- Create: `src/sportsmodel/nfl/player_features.py`
- Test: `tests/nfl/test_player_features.py`

**Interfaces:**
- Consumes: nflverse `weekly` (`player_id, season, week, season_type, team|recent_team, opponent_team, position, targets, carries, attempts, completions, receptions, receiving_yards, rushing_yards, passing_yards, passing_tds, receiving_tds, rushing_tds, receiving_air_yards, receiving_yards_after_catch, target_share, air_yards_share`), `snaps` (`season, week, game_type, pfr_player_id, team, opponent, position, offense_snaps, offense_pct`), `pfr2gsis` (from `usage.build_pfr_to_gsis`), `pbp`.
- Produces: `SKILL = ("QB","RB","WR","TE")`; `player_games(weekly, snaps, pfr2gsis) -> DataFrame` (one row per played player-game, keys `player_id, season, week, team, opponent, position`, labels `y_targets, y_carries, y_pass_att, y_receptions, y_rec_yds, y_rush_yds, y_pass_yds, y_pass_tds, y_anytime_td`, raw per-game stats `snap_pct, target_share, air_yards_share, rec_air_yards, yac`); `team_games(pbp) -> DataFrame` keys `season, week, team, opponent`, cols `pass_att, rush_att, plays, dropbacks, pressures_allowed, neutral_pass_rate`; `player_redzone(pbp) -> DataFrame` keys `player_id, season, week`, cols `rz_targets, rz_carries, gl_carries`.

- [ ] **Step 1: Write the failing tests**

```python
import pandas as pd

from sportsmodel.nfl.player_features import player_games, player_redzone, team_games


def _weekly():
    return pd.DataFrame({
        "player_id": ["w1", "q1"], "season": [2024, 2024], "week": [1, 1], "season_type": ["REG", "REG"],
        "team": ["KC", "KC"], "opponent_team": ["BAL", "BAL"], "position": ["WR", "QB"],
        "targets": [8, 0], "carries": [1, 3], "attempts": [0, 30], "completions": [0, 20], "receptions": [6, 0],
        "receiving_yards": [90, 0], "rushing_yards": [4, 12], "passing_yards": [0, 250], "passing_tds": [0, 2],
        "receiving_tds": [1, 0], "rushing_tds": [0, 0], "receiving_air_yards": [70, 0],
        "receiving_yards_after_catch": [30, 0], "target_share": [0.27, 0.0], "air_yards_share": [0.3, 0.0],
    })


def _snaps():
    return pd.DataFrame({
        "season": [2024] * 4, "week": [1] * 4, "game_type": ["REG"] * 4,
        "pfr_player_id": ["W1", "Q1", "T1", "K1"], "team": ["KC"] * 4, "opponent": ["BAL"] * 4,
        "position": ["WR", "QB", "TE", "K"], "offense_snaps": [60, 65, 10, 0], "offense_pct": [0.92, 1.0, 0.15, 0.0],
    })


def test_population_is_skill_players_with_offensive_snaps_and_zero_filled_labels():
    pg = player_games(_weekly(), _snaps(), {"W1": "w1", "Q1": "q1", "T1": "t1", "K1": "k1"})
    assert sorted(pg["player_id"]) == ["q1", "t1", "w1"]          # kicker + 0-snap dropped
    t1 = pg.set_index("player_id").loc["t1"]
    assert t1["y_targets"] == 0 and t1["y_rec_yds"] == 0            # played, no stats -> 0 not NaN
    w1 = pg.set_index("player_id").loc["w1"]
    assert w1["y_anytime_td"] == 1 and w1["y_rec_yds"] == 90 and w1["snap_pct"] == 0.92


def _pbp():
    return pd.DataFrame({
        "season": [2024] * 6, "week": [1] * 6, "season_type": ["REG"] * 6, "play_id": range(6),
        "posteam": ["KC"] * 6, "defteam": ["BAL"] * 6,
        "play_type": ["pass", "pass", "run", "run", "pass", "no_play"],
        "sack": [0, 1, 0, 0, 0, 0], "qb_hit": [1, 0, 0, 0, 0, 0],
        "wp": [0.5, 0.5, 0.5, 0.9, 0.5, 0.5], "down": [1, 2, 1, 1, 3, 1], "qtr": [1, 1, 2, 4, 1, 1],
        "yardline_100": [15, 40, 4, 30, 60, 50],
        "receiver_player_id": ["w1", None, None, None, "w1", None],
        "rusher_player_id": [None, None, "r1", "r1", None, None],
    })


def test_team_games_counts_attempts_excluding_sacks():
    tg = team_games(_pbp()).iloc[0]
    assert tg["pass_att"] == 2 and tg["rush_att"] == 2 and tg["dropbacks"] == 3 and tg["plays"] == 5
    assert tg["pressures_allowed"] == 2                     # qb_hit + sack
    assert tg["neutral_pass_rate"] == 2 / 3                 # neutral: wp .2-.8, downs 1-2, qtr<=3


def test_player_redzone():
    rz = player_redzone(_pbp()).set_index("player_id")
    assert rz.loc["w1", "rz_targets"] == 1 and rz.loc["r1", "rz_carries"] == 1 and rz.loc["r1", "gl_carries"] == 1
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/nfl/test_player_features.py -q` → FAIL.

- [ ] **Step 3: Implement** (`src/sportsmodel/nfl/player_features.py`, first part)

```python
"""Player-week feature table for the props-ML models. PURE (DataFrames in,
DataFrames out). Leakage contract: every feature for (season S, week w) is
built from rows strictly before (S, w) -- enforced by shift(1) before any
rolling statistic and tested by perturbing week-w-and-later box scores.

Column prefixes select feature groups per ladder rung (sim.nfl.learned):
p_ usage/efficiency, ngs_ Next Gen Stats, tm_ own team, op_ opponent,
st_ status, cx_ context, mk_ market; y_ are labels (never features).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

SKILL = ("QB", "RB", "WR", "TE")
KEYS = ["player_id", "season", "week"]
_STAT_COLS = {
    "targets": "y_targets", "carries": "y_carries", "attempts": "y_pass_att", "receptions": "y_receptions",
    "receiving_yards": "y_rec_yds", "rushing_yards": "y_rush_yds", "passing_yards": "y_pass_yds",
    "passing_tds": "y_pass_tds", "receiving_air_yards": "rec_air_yards", "receiving_yards_after_catch": "yac",
    "target_share": "target_share", "air_yards_share": "air_yards_share",
    "receiving_tds": "_rec_tds", "rushing_tds": "_rush_tds",
}


def _norm(code) -> str | None:
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def player_games(weekly: pd.DataFrame, snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> pd.DataFrame:
    """One row per REG-season player-game for QB/RB/WR/TE with offense_snaps > 0.
    Snap counts define who played (weekly omits players with no stats); weekly
    stats are left-joined and zero-filled (played + no stat = 0, not NaN)."""
    s = snaps[(snaps["game_type"] == "REG") & (snaps["offense_snaps"] > 0)
              & snaps["position"].isin(SKILL)].copy()
    s["player_id"] = s["pfr_player_id"].map(pfr2gsis)
    s["team"], s["opponent"] = s["team"].map(_norm), s["opponent"].map(_norm)
    s = s.dropna(subset=["player_id", "team", "opponent"])
    s = s.rename(columns={"offense_pct": "snap_pct"})[
        ["player_id", "season", "week", "team", "opponent", "position", "snap_pct"]]
    w = weekly[weekly["season_type"] == "REG"] if "season_type" in weekly.columns else weekly
    stats = w[KEYS + list(_STAT_COLS)].rename(columns=_STAT_COLS)
    out = s.merge(stats, on=KEYS, how="left").drop_duplicates(KEYS)
    stat_cols = list(_STAT_COLS.values())
    out[stat_cols] = out[stat_cols].fillna(0.0)
    out["y_anytime_td"] = ((out["_rec_tds"] + out["_rush_tds"]) > 0).astype(float)
    return out.drop(columns=["_rec_tds", "_rush_tds"]).reset_index(drop=True)


def team_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, offense team) REG-season volume: pass attempts exclude
    sacks (matches sim rates B.3), neutral pass rate over wp 0.2-0.8, downs 1-2,
    quarters 1-3."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()].copy()
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    sack = p["sack"].fillna(0) == 1
    p["_pass"] = (p["play_type"] == "pass") & ~sack
    p["_rush"] = p["play_type"] == "run"
    p["_db"] = p["play_type"] == "pass"
    p["_press"] = p["_db"] & (sack | (p["qb_hit"].fillna(0) == 1))
    neutral = p["wp"].between(0.2, 0.8) & p["down"].isin([1, 2]) & (p["qtr"] <= 3)
    p["_n"], p["_npass"] = neutral, neutral & p["_db"]
    g = p.groupby(["season", "week", "posteam", "defteam"], as_index=False).agg(
        pass_att=("_pass", "sum"), rush_att=("_rush", "sum"), plays=("play_type", "size"),
        dropbacks=("_db", "sum"), pressures_allowed=("_press", "sum"), _n=("_n", "sum"), _npass=("_npass", "sum"))
    g["neutral_pass_rate"] = np.where(g["_n"] > 0, g["_npass"] / g["_n"].where(g["_n"] > 0, 1), np.nan)
    g["team"], g["opponent"] = g["posteam"].map(_norm), g["defteam"].map(_norm)
    return g.drop(columns=["posteam", "defteam", "_n", "_npass"]).dropna(subset=["team", "opponent"])


def player_redzone(pbp: pd.DataFrame) -> pd.DataFrame:
    """Per player-game red-zone targets/carries (yardline_100 <= 20) and
    goal-line carries (<= 5)."""
    p = pbp[pbp["yardline_100"].notna()]
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    rz = p[p["yardline_100"] <= 20]
    t = rz.dropna(subset=["receiver_player_id"]).groupby(["receiver_player_id", "season", "week"]).size().rename("rz_targets")
    c = rz.dropna(subset=["rusher_player_id"]).groupby(["rusher_player_id", "season", "week"]).size().rename("rz_carries")
    gl = p[p["yardline_100"] <= 5].dropna(subset=["rusher_player_id"]).groupby(
        ["rusher_player_id", "season", "week"]).size().rename("gl_carries")
    for s in (t, c, gl):
        s.index = s.index.set_names(KEYS)
    return pd.concat([t, c, gl], axis=1).fillna(0.0).reset_index()
```

- [ ] **Step 4: Run tests** — PASS; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/player_features.py tests/nfl/test_player_features.py
git commit -m "feat(nfl): props-ML population, labels, team-game and red-zone aggregates"
```

---

### Task 5: Leakage-safe rolling feature table

**Files:**
- Modify: `src/sportsmodel/nfl/player_features.py`
- Test: `tests/nfl/test_player_features.py` (extend)

**Interfaces:**
- Consumes: Task 4 outputs; Task 3 `team_game_context`; NGS frames (Task 1); injuries (`season, week, team, gsis_id, report_status`); `depth` as-of frame (Task 2); `efficiency.team_game_epa` / `adjusted_efficiency`.
- Produces: `build_feature_table(pg, tg, rz, ctx, ngs, injuries, depth, game_epa, stubs=None) -> pd.DataFrame` — one row per (player_id, season, week) for every played row plus every stub row (active-but-no-snap players, labels NaN); and `build_team_table(tg, ctx, game_epa) -> pd.DataFrame` — one row per (team, season, week) with labels `y_team_pass_att, y_team_rush_att` and `tm_*`, `op_*`, `cx_*`, `mk_*` features. Helper `roll_features(df, group, cols, prefix)`.

Feature definitions (all shifted by one game within the group before any statistic):
- `roll_features` makes, per col: `{prefix}{col}_r3`, `_r5`, `_r10` (rolling mean, min_periods=1), `_ewm` (halflife 4 games), `_std` (season-to-date mean), `_prev` (previous season's full mean), plus one `{prefix}n_season` (games so far this season). Groups: player (`player_id`) for `p_`/`ngs_`; team for `tm_`; opponent for `op_`.
- Player cols (`p_`): snap_pct, target_share, air_yards_share, y_targets, y_carries, y_pass_att, rz_targets, rz_carries, gl_carries, and efficiency ratios computed per game then rolled: `ypt = rec_yds/targets`, `catch_rate = receptions/targets`, `ypr = rec_yds/receptions`, `ypc = rush_yds/carries`, `yac_pr = yac/receptions` (NaN when the denominator is 0, so rolling means skip them). Also `p_carry_share` (carries / team rush_att) rolled, `p_career_games` (prior games since 2016, capped 100), `p_depth_rank` (as-of depth chart `depth_team` for (S,w) — pre-game), `p_pos` (category QB/RB/WR/TE).
- NGS cols (`ngs_`): receiving `avg_separation, avg_cushion, avg_intended_air_yards, avg_yac_above_expectation`; rushing `rush_yards_over_expected_per_att, efficiency, percent_attempts_gte_eight_defenders`; passing `avg_time_to_throw, completion_percentage_above_expectation, aggressiveness` — only `_ewm` and `_prev` (sparse), joined on `player_gsis_id == player_id`, REG weeks > 0.
- Team (`tm_`): pass_att, rush_att, plays, neutral_pass_rate rolled per team; `tm_off_adj`, `tm_def_adj` from `adjusted_efficiency(game_epa, S, w)` (same-season weeks < w; NaN week 1) and `tm_off_prev`, `tm_def_prev` = `adjusted_efficiency(game_epa, S-1, 99)`.
- Opponent (`op_`): opponent's rolled allowed volume (their `pass_att`/`rush_att` faced) and `op_press_rate` (pressures generated / dropbacks faced), opponent `op_def_adj` / `op_def_prev`, and position-specific `op_ypt_allowed_{pos}` = rolled (rec yards / targets) allowed to that position (shrunk: `(yds + 7.5*20) / (tgts + 20)` per game before rolling).
- Status (`st_`): `st_questionable` (player's report_status == Questionable that week, by gsis_id), `st_vacated_tgt` / `st_vacated_car` (sum of teammates' latest post-game `target_share`/`p_carry_share` EWM for teammates Out/Doubtful that week), `st_qb_changed` (as-of depth QB1 differs from the team's previous game's attempts leader).
- Context/market: merged from `ctx` on (season, week, team).

- [ ] **Step 1: Write the failing tests** (append)

```python
import numpy as np

from sportsmodel.nfl.player_features import roll_features


def _series_df():
    return pd.DataFrame({"player_id": ["a"] * 5, "season": [2023, 2023, 2024, 2024, 2024],
                         "week": [1, 2, 1, 2, 3], "x": [1.0, 3.0, 5.0, 7.0, 9.0]})


def test_roll_features_are_strictly_prior():
    out = roll_features(_series_df(), "player_id", ["x"], "p_").set_index(["season", "week"])
    assert np.isnan(out.loc[(2023, 1), "p_x_r3"])                 # no history
    assert out.loc[(2024, 3), "p_x_r3"] == np.mean([3.0, 5.0, 7.0])  # excludes own game (9)
    assert out.loc[(2024, 3), "p_x_std"] == 6.0                   # season-to-date: 5, 7
    assert out.loc[(2024, 1), "p_x_prev"] == 2.0                  # 2023 mean
    assert out.loc[(2024, 3), "p_n_season"] == 2


def test_perturbing_target_week_and_later_does_not_change_features():
    """Leakage guard: garbage box scores at/after (2024, 3) must leave the
    (2024, 3) feature row identical."""
    from tests.nfl.fixtures_props import feature_inputs     # synthetic 2-season league (see Step 3)
    base = feature_inputs()
    t0 = _build(base).set_index(["player_id", "season", "week"])
    bad = feature_inputs(perturb_from=(2024, 3))
    t1 = _build(bad).set_index(["player_id", "season", "week"])
    feats = [c for c in t0.columns if not c.startswith("y_")]
    key = [k for k in t0.index if k[1:] == (2024, 3)]
    pd.testing.assert_frame_equal(t0.loc[key, feats], t1.loc[key, feats])
    assert not t0.loc[key, "y_targets"].equals(t1.loc[key, "y_targets"])  # labels did change


def _build(inp):
    from sportsmodel.nfl.player_features import build_feature_table
    return build_feature_table(**inp)
```

Create the fixture module `tests/nfl/fixtures_props.py` with `feature_inputs(perturb_from=None) -> dict` returning synthetic `pg, tg, rz, ctx, ngs, injuries, depth, game_epa, stubs` for 4 teams × 2 seasons × 4 weeks (deterministic numbers from `np.random.default_rng(0)`); when `perturb_from=(S, w)`, multiply every box-score stat in `pg`, `tg`, `rz`, `ngs` and every `game_epa` entry at or after (S, w) by 10 (pre-game inputs — `ctx`, `injuries`, `depth` — untouched). Build `game_epa` by calling `efficiency.team_game_epa` on a tiny synthetic pbp so its dict shape is the real one.

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/nfl/test_player_features.py -q` → FAIL.

- [ ] **Step 3: Implement** (append to `player_features.py`)

```python
_ORDER = ["season", "week"]


def roll_features(df: pd.DataFrame, group: str, cols: list[str], prefix: str,
                  *, windows=(3, 5, 10), halflife: float = 4.0, sparse: bool = False) -> pd.DataFrame:
    """Add strictly-prior rolling features for `cols` within `group`.
    Every statistic is computed on the group's values shifted by one game, so
    a row never sees its own game or any later one."""
    out = df.sort_values([group, *_ORDER]).reset_index(drop=True)
    g = out.groupby(group, sort=False)
    out[f"{prefix}n_season"] = out.groupby([group, "season"]).cumcount()
    for c in cols:
        prev = g[c].shift(1)
        pg = prev.groupby(out[group])
        if not sparse:
            for n in windows:
                out[f"{prefix}{c}_r{n}"] = pg.transform(lambda x, n=n: x.rolling(n, min_periods=1).mean())
            std = out[c].groupby([out[group], out["season"]]).transform(lambda x: x.shift(1).expanding().mean())
            out[f"{prefix}{c}_std"] = std
        out[f"{prefix}{c}_ewm"] = pg.transform(lambda x: x.ewm(halflife=halflife, ignore_na=True).mean())
    # Previous season's full mean -- merged once AFTER the loop so the groupby
    # objects above stay aligned with `out`'s index.
    prev_season = out.groupby([group, "season"])[cols].mean().add_prefix(prefix).add_suffix("_prev").reset_index()
    prev_season["season"] += 1
    return out.merge(prev_season, on=[group, "season"], how="left")
```

Then `build_feature_table` and `build_team_table` implementing the definitions above. Structure (write each as a small private helper with a docstring):

```python
def _ratios(pg: pd.DataFrame) -> pd.DataFrame:
    """Per-game efficiency ratios; NaN when the denominator is 0 so rolling
    means ignore games with no opportunity."""
    d = pg.copy()
    safe = lambda n, den: np.where(d[den] > 0, d[n] / d[den].where(d[den] > 0, 1), np.nan)
    d["ypt"], d["catch_rate"] = safe("y_rec_yds", "y_targets"), safe("y_receptions", "y_targets")
    d["ypr"], d["ypc"], d["yac_pr"] = safe("y_rec_yds", "y_receptions"), safe("y_rush_yds", "y_carries"), safe("yac", "y_receptions")
    return d


_P_COLS = ["snap_pct", "target_share", "air_yards_share", "y_targets", "y_carries", "y_pass_att",
           "rz_targets", "rz_carries", "gl_carries", "p_carry_share_raw",
           "ypt", "catch_rate", "ypr", "ypc", "yac_pr"]
_NGS_COLS = {
    "rec": ["avg_separation", "avg_cushion", "avg_intended_air_yards", "avg_yac_above_expectation"],
    "rush": ["rush_yards_over_expected_per_att", "efficiency", "percent_attempts_gte_eight_defenders"],
    "pass": ["avg_time_to_throw", "completion_percentage_above_expectation", "aggressiveness"],
}


def build_feature_table(pg, tg, rz, ctx, ngs, injuries, depth, game_epa, stubs=None) -> pd.DataFrame:
    """One row per (player_id, season, week): every played row plus `stubs`
    (active-but-no-snap players the sim may still need; labels NaN). `ngs` is
    {"rec": df, "rush": df, "pass": df}. Returns KEYS, team, opponent,
    position, y_* labels and p_/ngs_/tm_/op_/st_/cx_/mk_ features."""
    ...
```

Implementation steps inside `build_feature_table` (each a helper, each unit-covered by the perturbation test):
1. `base = _ratios(pg).merge(rz, on=KEYS, how="left")` (rz NaN→0); `p_carry_share_raw = y_carries / team rush_att` via `tg`.
2. Append `stubs` (columns `player_id, season, week, team, opponent, position`, labels NaN) — stubs sort into their (season, week) slot so `shift(1)` gives them history-only features.
3. `roll_features(base, "player_id", _P_COLS, "p_")`; rename `p_n_season` kept; `p_career_games = groupby(player).cumcount().clip(upper=100)`; `p_pos` = `position` as `category`.
4. NGS: for each of rec/rush/pass, filter `week > 0` and `season_type == "REG"` if present, rename `player_gsis_id→player_id`, left-merge onto base, `roll_features(..., "ngs_", sparse=True)`.
5. Team + opponent: `team_tbl = build_team_table(tg, ctx, game_epa)`; merge `tm_*`, `op_*`, `cx_*`, `mk_*` on (season, week, team).
6. `op_ypt_allowed_{pos}`: per (season, week, opponent, position) sum rec yds/targets from `pg`, shrink `(yds + 150) / (tgts + 20)`, roll per (opponent, position) with `roll_features(..., "op_", windows=(5,), sparse=False)` keeping `_r5`/`_ewm`; pivot to columns by position; merge on (season, week, opponent).
7. Status: `st_questionable` from `injuries` (gsis_id, Questionable); vacated shares — compute each player's post-game EWM (no shift) of `target_share`/`p_carry_share_raw`, then for each team-week sum over players whose injury status is Out/Doubtful that (S,w), using their latest EWM from games strictly before (S,w) (`pd.merge_asof` on ordinal `season*100+week`, `allow_exact_matches=False`); `st_qb_changed` from as-of `depth` QB1 (min depth_team among QBs) vs previous game's `y_pass_att` leader per team.
8. `p_depth_rank`: `depth` (as-of, per week) `depth_team` joined on (gsis_id, season, week); NaN if absent.
9. Return sorted by KEYS.

`build_team_table(tg, ctx, game_epa)`: `roll_features(tg, "team", ["pass_att","rush_att","plays","neutral_pass_rate"], "tm_")`; opponent-faced volume/pressure via the same helper grouped by `opponent` on the defensive view (`tg` rows re-keyed: team := opponent), prefix `op_`, with `op_press_rate = pressures_allowed/dropbacks` per game before rolling; `tm_off_adj/tm_def_adj` from `adjusted_efficiency(game_epa, S, w)` per (S, w) (cache per (S,w)); `tm_*_prev` from `adjusted_efficiency(game_epa, S-1, 99)`; opponent's `op_def_adj/op_def_prev` the same lookup for the opponent; merge `ctx` for `cx_*`/`mk_*`; labels `y_team_pass_att = pass_att`, `y_team_rush_att = rush_att`.

- [ ] **Step 4: Run tests** — `uv run pytest tests/nfl/test_player_features.py -q` PASS; full suite green.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/player_features.py tests/nfl/test_player_features.py tests/nfl/fixtures_props.py
git commit -m "feat(nfl): leakage-safe rolling player/team feature tables for props ML"
```

---

### Task 6: Feature-table build script

**Files:**
- Create: `scripts/build_player_features.py`
- Modify: `.gitignore`
- Test: `tests/scripts/test_build_player_features.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: `data/props_ml/player_week_features.parquet`, `data/props_ml/team_week_features.parquet`; pure helper `active_stubs(depth, injuries, pg, seasons) -> DataFrame` (active depth-chart skill players minus Out/Doubtful for every REG team-week, excluding those already in `pg`), testable without IO.

- [ ] **Step 1: Failing test** for `active_stubs`: synthetic depth (2 QBs, 2 WRs for KC week 1), injuries (one WR Out), pg containing the QB1 → stubs = the other QB and the healthy WR only, with `team`, `opponent` (from a schedules frame arg), `position`.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement** `active_stubs(depth, injuries, pg, schedules)` and `main()`:

```python
SEASONS = list(range(2016, 2027))
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "props_ml"


def main() -> None:
    from sportsmodel.nfl.nflverse import load_release
    import nfl_data_py as nfl
    sched = load_release("schedules", SEASONS)
    pbp = load_release("pbp", SEASONS)
    weekly = load_release("weekly", SEASONS)
    snaps = load_release("snaps", SEASONS)
    depth = depth_charts_asof(load_release("depth", SEASONS), sched)
    injuries = nfl.import_injuries([s for s in SEASONS if s < 2026])   # 2026 via the same resilient path
    ngs = {k: load_release(f"ngs_{v}", SEASONS) for k, v in (("rec", "receiving"), ("rush", "rushing"), ("pass", "passing"))}
    pfr2gsis = build_pfr_to_gsis(nfl.import_ids())
    ...build pg, tg, rz, ctx, game_epa, stubs; write both parquets; print row counts,
       per-season rows, and NaN share per feature group
```

Use `import_by_season(nfl.import_injuries, SEASONS, "injuries", required=False)` (existing helper) so a not-yet-published season doesn't abort. Add `data/props_ml/` to `.gitignore`.

- [ ] **Step 4: Run tests** → PASS; full suite green. Then run the script once: `uv run python scripts/build_player_features.py` (network; ~5–10 min). Expected output: ~65k played rows 2016–2026 plus stubs; record the printed counts in the commit message body.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_player_features.py tests/scripts/test_build_player_features.py .gitignore
git commit -m "feat(props-ml): feature-table build script (2016-2026)"
```

---

### Task 7: Learned volume/efficiency models

**Files:**
- Create: `src/sportsmodel/sim/nfl/learned.py`
- Test: `tests/sim/nfl/test_learned.py`

**Interfaces:**
- Consumes: feature tables (Task 5/6), `PlayerInput`, `TeamRates`, `NflGameSpec` (`sim/nfl/spec.py`).
- Produces:
  - `RUNG_PREFIXES = {"volume": ("p_", "ngs_", "tm_", "op_", "st_"), "context": ("cx_",), "market": ("mk_",)}`
  - `feature_columns(df, toggles: frozenset[str]) -> list[str]` — `volume` is always on when any learned model is used; `efficiency` toggles the efficiency models, not columns.
  - `@dataclass LearnedModels(team_pass, team_rush, targets, carries, eff: dict[str, model] | None, player_cols, team_cols)`
  - `fit_models(player_df, team_df, toggles, *, upto: tuple[int,int], test_season: int, decay: float, max_iter: int) -> LearnedModels` — trains on rows strictly before `upto` with labels present; sample weight `decay ** (test_season - season)`.
  - `tune(player_df, test_season, toggles) -> tuple[float, int]` — inner search over `decay ∈ {1.0, 0.8, 0.6}` × `max_iter ∈ {150, 300}`: train on seasons < test_season−1, score Poisson deviance of the targets model on season test_season−1.
  - `apply_to_spec(spec, models, player_rows, team_rows, questionable: set[str], q_weight: float) -> NflGameSpec`.

Model definitions (all `HistGradientBoostingRegressor(learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50, l2_regularization=1.0, categorical_features="from_dtype", random_state=0, max_iter=max_iter)`):
- `team_pass`, `team_rush`: `loss="poisson"` on `y_team_pass_att` / `y_team_rush_att` with team-table columns.
- `targets`, `carries`: `loss="poisson"` on `y_targets` / `y_carries`; `monotonic_cst={"p_target_share_ewm": 1}` / `{"p_p_carry_share_raw_ewm": 1}`.
- efficiency (only when `"efficiency" in toggles`): `ypr` (label `y_rec_yds/y_receptions`, weight receptions, rows receptions>0), `catch_rate` (label receptions/targets, weight targets, rows targets>0), `ypc` (label rush_yds/carries, weight carries, rows carries>0); `loss="squared_error"`.

`apply_to_spec` behavior:
- Team rates: `dataclasses.replace(spec.home, pass_att_pg=pred_pass, rush_att_pg=pred_rush)` (same for away) when the team row exists; otherwise unchanged.
- Shares: for each side, predict targets/carries for every active player that has a feature row; multiply by `q_weight` for players whose gsis is in `questionable`; if **every** active player has a prediction, `target_share = pred_i / Σpred`, `carry_share` likewise; else leave the side's shares unchanged (rare — stubs cover the active set) and count it.
- Efficiency (when models present): `ypr`, `catch_rate` (clipped to [0.05, 1.0]), `ypc` replaced for players with rows; `ypt = catch_rate * ypr`. TD shares untouched.

- [ ] **Step 1: Failing tests** (`tests/sim/nfl/test_learned.py`), using a small synthetic player/team table (3 seasons, 2 teams, 4 players each, labels drawn from Poisson(λ) where λ rises with `p_target_share_ewm`):

```python
def test_feature_columns_by_toggle(tbl):
    cols = feature_columns(tbl, frozenset({"volume"}))
    assert all(c.startswith(("p_", "ngs_", "tm_", "op_", "st_")) for c in cols)
    assert not any(c.startswith(("cx_", "mk_", "y_")) for c in cols)
    assert any(c.startswith("mk_") for c in feature_columns(tbl, frozenset({"volume", "market"})))

def test_fit_uses_only_rows_before_upto(tbl, monkeypatch):
    seen = {}
    real_fit = HistGradientBoostingRegressor.fit
    def spy(self, X, y, sample_weight=None):
        seen.setdefault("n", len(X)); return real_fit(self, X, y, sample_weight=sample_weight)
    monkeypatch.setattr(HistGradientBoostingRegressor, "fit", spy)
    fit_models(tbl.player, tbl.team, frozenset({"volume"}), upto=(2024, 1), test_season=2024, decay=1.0, max_iter=20)
    assert seen["n"] == int(((tbl.player.season < 2024) & tbl.player.y_targets.notna()).sum())

def test_apply_renormalizes_shares_and_downweights_questionable(tbl, spec):
    m = fit_models(tbl.player, tbl.team, frozenset({"volume"}), upto=(2024, 1), test_season=2024, decay=1.0, max_iter=20)
    out = apply_to_spec(spec, m, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1), questionable={"h2"}, q_weight=0.5)
    shares = [p.target_share for p in out.home_players]
    assert abs(sum(shares) - 1.0) < 1e-9
    base = apply_to_spec(spec, m, tbl.rows_for(2024, 1), tbl.team_rows_for(2024, 1), questionable=set(), q_weight=0.5)
    h2 = lambda s: next(p.target_share for p in s.home_players if p.player_id == "h2")
    assert h2(out) < h2(base)
    assert out.home.pass_att_pg > 0 and out.home.pass_att_pg != spec.home.pass_att_pg

def test_monotone_in_target_share(tbl):
    m = fit_models(tbl.player, tbl.team, frozenset({"volume"}), upto=(2024, 1), test_season=2024, decay=1.0, max_iter=50)
    row = tbl.player[tbl.player.season == 2023].iloc[[0]].copy()
    lo, hi = row.copy(), row.copy()
    lo["p_target_share_ewm"], hi["p_target_share_ewm"] = 0.05, 0.30
    assert m.targets.predict(hi[m.player_cols])[0] >= m.targets.predict(lo[m.player_cols])[0]
```

Put the synthetic `tbl` and `spec` fixtures in `tests/sim/nfl/conftest.py` (append if it exists) — a `SimpleNamespace(player=..., team=..., rows_for=..., team_rows_for=...)` and an `NflGameSpec` with two 4-player sides whose ids match the table.

- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement `learned.py`** per the interfaces above (docstring states the leakage contract: `fit_models` filters `(season, week) < upto` itself, so callers cannot leak by passing a full table).
- [ ] **Step 4: Run tests** → PASS; full suite green.
- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/sim/nfl/learned.py tests/sim/nfl/test_learned.py tests/sim/nfl/conftest.py
git commit -m "feat(sim): learned team volume, player shares and efficiency models (props ML A)"
```

---

### Task 8: Gate metrics — ECE, clustered bootstrap, rung decision

**Files:**
- Modify: `src/sportsmodel/model/props_eval.py` (Task 2 created the file)
- Test: `tests/model/test_props_eval.py` (extend)

**Interfaces:**
- Produces:
  - `decile_ece(pits) -> float` — mean |share in each tenth − 0.1|.
  - `paired_frame(base: list[dict], cand: list[dict], population: set[tuple]) -> pd.DataFrame` — inner-join on `(season, week, player_id, market)` restricted to `population`; columns `rps_b, rps_c, pit_b, pit_c, cluster` (`cluster = f"{season}-{week}-{home}"`).
  - `relative_skill(df) -> float` — mean over markets of `1 − mean(rps_c)/mean(rps_b)`.
  - `cluster_bootstrap(df, stat, n_boot=1000, seed=0) -> tuple[float, float, float]` — (point, 2.5%, 97.5%) resampling clusters with replacement.
  - `rung_decision(df) -> dict` — `{"skill", "lo", "hi", "per_market": {m: {"n","rps_b","rps_c","ece_b","ece_c"}}, "pass": bool, "reasons": [..]}`; pass iff `lo > 0` and no market `rps_c > 1.01 * rps_b` and no market `ece_c > ece_b + 0.005`.
  - `population_from_baseline(base_records) -> set[tuple]` — keys whose baseline `mean` passes `serving.props_ev.is_propable_projected(market, mean)`.

- [ ] **Step 1: Failing tests**

```python
def test_decile_ece_uniform_is_small():
    rng = np.random.default_rng(0)
    assert decile_ece(rng.uniform(size=20000)) < 0.01
    assert decile_ece(np.full(1000, 0.05)) > 0.15

def _recs(rps, market="rec_yds", n=200, shift=0.0):
    return [{"season": 2024, "week": 1 + i % 17, "home": f"T{i % 8}", "player_id": f"p{i}", "market": market,
             "rps": rps + shift * (i % 3), "pit": (i % 10) / 10 + 0.05, "mean": 50.0} for i in range(n)]

def test_rung_passes_on_clear_improvement():
    base, cand = _recs(10.0), _recs(9.0)
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    d = rung_decision(paired_frame(base, cand, pop))
    assert d["pass"] and d["lo"] > 0

def test_rung_fails_when_one_market_worsens():
    base = _recs(10.0) + _recs(2.0, market="receptions")
    cand = _recs(8.0) + _recs(2.1, market="receptions")        # receptions 5% worse
    pop = {(r["season"], r["week"], r["player_id"], r["market"]) for r in base}
    d = rung_decision(paired_frame(base, cand, pop))
    assert not d["pass"] and any("receptions" in r for r in d["reasons"])

def test_population_uses_projected_gate():
    recs = [{"season": 2024, "week": 1, "player_id": "a", "market": "rec_yds", "mean": 60.0},
            {"season": 2024, "week": 1, "player_id": "b", "market": "rec_yds", "mean": 1.0}]
    assert population_from_baseline(recs) == {(2024, 1, "a", "rec_yds")}
```

(Check `PROJECTED_USAGE_GATE` in `serving/props_ev.py` for the exact rec_yds threshold; pick the two means on either side of it.)

- [ ] **Step 2: Run → FAIL.**  **Step 3: Implement.**  **Step 4: Run → PASS, full suite green.**
- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/model/props_eval.py tests/model/test_props_eval.py
git commit -m "feat(props-ml): gate metrics (ECE, clustered bootstrap, rung decision, population)"
```

---

### Task 9: Walk-forward ablation-ladder harness

**Files:**
- Create: `scripts/train_props_ml.py`
- Test: `tests/scripts/test_train_props_ml.py`

**Interfaces:**
- Consumes: Tasks 2, 5–8; `backtest_sim_nfl.run_backtest(..., spec_hook, record)` with the production config (`season_decay=0.4, questionable_weight=0.75, home_field=0.07, ratings_weight=0.5`).
- Produces: `assets/nfl/props_ml/a_gate.json` (committed; small) and `docs/superpowers/reports/<date>-props-ml-a-gate.md`.

Pure helpers (unit-tested):
- `TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]`, `REFIT_WEEKS = (1, 5, 9, 13, 17)`, `LADDER = ("volume", "efficiency", "context", "market")`.
- `refit_block(week) -> int` — the largest refit week ≤ `week`.
- `next_candidate(kept: frozenset[str], rung: str) -> frozenset[str]` — `kept | {rung}`; `efficiency`/`context`/`market` imply `volume` (if `volume` failed, later rungs are evaluated as `{"volume", rung}` anyway, and the report says so).
- `make_hook(models_by_block, player_tbl, team_tbl, injuries_q) -> Callable` — returns `spec_hook(season, week, home, away, spec)` that looks up `models_by_block[(season, refit_block(week))]`, selects the (season, week) rows for both teams, and calls `learned.apply_to_spec`.

`main()` flow:
1. Load `data/props_ml/*.parquet` (fail with a clear message pointing at `scripts/build_player_features.py` if absent).
2. `seasons = env PROPS_ML_SEASONS` (comma list) or `TEST_SEASONS`; `n = int(env PROPS_ML_N_SIMS or 1000)`.
3. Baseline: `run_backtest(seasons, n, record=base_recs, **PROD)`; `population = population_from_baseline(base_recs)`.
4. For each rung in `LADDER`: `toggles = next_candidate(kept, rung)`; for each test season: `(decay, it) = tune(player_tbl, S, toggles)`; for each refit week r: `models[(S, r)] = fit_models(..., upto=(S, r), test_season=S, decay=decay, max_iter=it)`; run `run_backtest(seasons, n, spec_hook=make_hook(...), record=cand_recs, **PROD)` (same seed → common random numbers); `d = rung_decision(paired_frame(base_recs_or_kept_recs, cand_recs, population))` where the comparison is against the **currently kept** configuration's records (baseline until a rung passes); if `d["pass"]`: `kept = toggles`, `kept_recs = cand_recs`.
5. Final gate: `rung_decision` of `kept_recs` vs `base_recs` on all seasons and again filtered to season 2025; `final_pass = both pass and kept != ∅`.
6. Write `a_gate.json` (`{"seasons", "n_sims", "ladder": [{rung, toggles, decision}], "kept", "final": {...}, "tuned": {season: [decay, max_iter]}}`) and the markdown report (tables per rung: skill with CI, per-market RPS base→cand and ECE base→cand, pass/fail reasons; final verdict in the first paragraph, stated plainly whether it passes or not).

- [ ] **Step 1: Failing tests** for `refit_block` (week 1→1, 4→1, 5→5, 18→17), `next_candidate`, and `make_hook` (with a stub `models_by_block` and `apply_to_spec` monkeypatched to record its arguments: asserts the right block key and that only rows for the given (season, week) and the two teams are passed).
- [ ] **Step 2: Run → FAIL.**  **Step 3: Implement.**  **Step 4: Run → PASS; full suite green.**
- [ ] **Step 5: Smoke run** (network; small): `PROPS_ML_SEASONS=2025 PROPS_ML_N_SIMS=200 uv run python scripts/train_props_ml.py` — must complete and write both outputs. Do not commit smoke outputs.
- [ ] **Step 6: Commit**

```bash
git add scripts/train_props_ml.py tests/scripts/test_train_props_ml.py
git commit -m "feat(props-ml): walk-forward ablation-ladder harness for the A gate"
```

---

### Task 10: Run the gate and report (controller task)

- [ ] **Step 1:** `uv run python scripts/build_player_features.py` (if Task 6's output is stale).
- [ ] **Step 2:** Full run in the background: `PROPS_ML_N_SIMS=1000 uv run python scripts/train_props_ml.py` (5 test seasons × up to 5 walk-forwards; expect hours — run with `run_in_background`).
- [ ] **Step 3:** Validate the every-4-weeks refit shortcut on one season: rerun the kept configuration for 2025 with `REFIT_WEEKS = tuple(range(1, 19))` via env `PROPS_ML_WEEKLY_REFIT=1` (add this switch in Task 9's `main`) and confirm the rung decision is unchanged; note the result in the report.
- [ ] **Step 4:** Commit `assets/nfl/props_ml/a_gate.json` and the report:

```bash
git add assets/nfl/props_ml/a_gate.json docs/superpowers/reports/*-props-ml-a-gate.md
git commit -m "docs(props-ml): A gate verdict (walk-forward 2021-2025)"
```

- [ ] **Step 5:** Tell the user the verdict plainly — which rungs passed, the pooled skill with CI, per-market changes, and what Props-2 inherits (kept toggles). A negative verdict is reported the same way.

---

## Self-review notes

- Spec coverage (Props-1 scope): feature table §2 → Tasks 3–6; A models §3 → Task 7; validation/gate §4 → Tasks 2, 8–10; loaders fail loudly §5 → Task 1. B models, blend, calibration, quantile mapping, Release artifacts, workflows, shadow serving → Props-2 plan. Game-context ridge → Phase-2 plan.
- Deviations from the spec are listed under "Rulings" (route participation, precipitation, rookie flag, four markets, market rung scope, depth-chart fix, pooled-score definition, ECE tolerance, refit/tuning grid, live depth path).
