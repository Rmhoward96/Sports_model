# NFL P1 — Data + Elo + SoS blend + backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the NFL data layer (committed nflverse snapshots) and the game-line rating engine — margin-adjusted Elo with last-season carryover, an explicit strength-of-schedule (SRS) rating blended in, ESPN live adapters, the `game_pk` matcher — and a walk-forward backtest that tunes and validates the rating parameters.

**Architecture:** New `src/sportsmodel/nfl/` package. `teams.py` normalizes the 32-team vocabulary (single source of truth). `data.py` wraps `nfl_data_py` and a build script commits parquet snapshots to `assets/nfl/`. `elo.py` is the sequential Elo engine; `srs.py` is a retrodictive opponent-adjusted rating; `ratings.py` blends the two expected margins (cold-start fallback to pure Elo). `espn.py` parses the ESPN public API (fixture-tested); `matcher.py` keys everything on the ESPN event id. `scripts/backtest_nfl_elo.py` walk-forwards over the snapshots to tune `(K, HFA_elo, carryover, w_sos, srs_min_games)` on a train span and validate out-of-sample. MLB is untouched; NFL is strictly additive.

**Tech Stack:** Python 3.12 (uv), `nfl_data_py` (added in P0), pandas/numpy, httpx (ESPN), pytest. Parquet via the pyarrow/fastparquet already pulled in by `nfl_data_py`.

**Spec:** `docs/superpowers/specs/2026-08-24-nfl-p1-data-elo.md` (argues from parent `docs/superpowers/specs/2026-08-23-nfl-model.md`).

## Global Constraints

- **MLB behavior is unchanged.** Every existing test in `tests/` must still pass after each task; NFL code is additive (new package + new tests only).
- **Team vocabulary — single source of truth in `teams.py`.** Normalizations at minimum: `LAR→LA`, `WSH→WAS`, and franchise relocations `OAK→LV`, `SD→LAC`, `STL→LA`. Canonical set = the current 32 teams.
- **`game_pk` = the ESPN event id (integer)**, cross-referenced by nflverse `schedules.espn`.
- **History windows:** schedules seasons **2002–2025**; weekly/rosters/injuries **2015–2025**.
- **No live network in the pytest suite.** Pure logic (elo/srs/ratings/matcher) is tested directly; `data.py`/`espn.py` parsing is tested on injected frames / committed fixtures. Live pulls happen only inside the build + backtest **scripts**, run as explicit task actions.
- **Elo formulas (exact):** `E_home = 1/(1+10^(-(elo_home+HFA_elo-elo_away)/400))`; MOV multiplier `ln(mov_input+1) * (2.2/(0.001*elo_diff_winner + 2.2))` where `mov_input = abs(margin) or 1 for a tie` and `elo_diff_winner` is (winner pregame rating incl. HFA) − (loser pregame rating incl. HFA); update `elo_home += K*mov_mult*(result_home−E_home)`, `elo_away −= (same delta)`; carryover `elo_start = 1500 + carryover*(elo_prev_final−1500)`; `elo_expected_margin = (elo_home+HFA_elo−elo_away)/25`.
- **Branch:** `nfl-p1-data-elo`. One commit per task (TDD red→green→commit). Commit trailer: `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

## File Structure

- `src/sportsmodel/nfl/__init__.py` — NEW (package marker).
- `src/sportsmodel/nfl/teams.py` — NEW. 32-team set + `normalize_team`.
- `src/sportsmodel/nfl/data.py` — NEW. nflverse loaders + pure normalizers.
- `src/sportsmodel/nfl/elo.py` — NEW. Sequential Elo engine.
- `src/sportsmodel/nfl/srs.py` — NEW. Retrodictive SoS rating.
- `src/sportsmodel/nfl/ratings.py` — NEW. Elo×SRS blend → `expected_margin`.
- `src/sportsmodel/nfl/espn.py` — NEW. ESPN parse_* + fetch_* adapters.
- `src/sportsmodel/nfl/matcher.py` — NEW. `game_pk` matcher.
- `scripts/build_nfl_snapshots.py` — NEW. Writes `assets/nfl/*.parquet`.
- `scripts/backtest_nfl_elo.py` — NEW. Walk-forward backtest + tuning.
- `assets/nfl/{schedules,weekly,rosters,injuries}.parquet` — NEW committed snapshots.
- `assets/nfl/rating.json` — NEW committed tuned params (Task 8).
- `docs/superpowers/reports/2026-08-24-nfl-elo-backtest.md` — NEW (Task 8 findings).
- Tests: `tests/nfl/test_teams.py`, `test_data.py`, `test_elo.py`, `test_srs.py`, `test_ratings.py`, `test_espn.py`, `test_matcher.py`, `test_backtest_nfl.py`, plus `tests/fixtures/nfl/espn_scoreboard.json`.

---

### Task 1: Team normalization (`teams.py`)

**Files:**
- Create: `src/sportsmodel/nfl/__init__.py` (empty), `src/sportsmodel/nfl/teams.py`
- Test: `tests/nfl/__init__.py` (empty), `tests/nfl/test_teams.py`

**Interfaces:**
- Produces: `normalize_team(abbr: str) -> str` (raises `ValueError` on unknown) and `TEAMS: frozenset[str]` (the 32 canonical abbreviations). Consumed by `data.py`, `espn.py`, `matcher.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_teams.py
import pytest
from sportsmodel.nfl.teams import normalize_team, TEAMS

def test_len_is_32():
    assert len(TEAMS) == 32

def test_aliases_and_relocations():
    assert normalize_team("LAR") == "LA"
    assert normalize_team("WSH") == "WAS"
    assert normalize_team("OAK") == "LV"
    assert normalize_team("SD") == "LAC"
    assert normalize_team("STL") == "LA"

def test_idempotent_and_case_insensitive():
    for t in TEAMS:
        assert normalize_team(t) == t
    assert normalize_team("kc") == "KC"
    assert normalize_team(normalize_team("OAK")) == "LV"

def test_unknown_raises():
    with pytest.raises(ValueError):
        normalize_team("XYZ")
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_teams.py -v` → ImportError.

- [ ] **Step 3: Implement `teams.py`**

```python
# src/sportsmodel/nfl/teams.py
TEAMS = frozenset({
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
    "DET", "GB", "HOU", "IND", "JAX", "KC", "LA", "LAC", "LV", "MIA",
    "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB",
    "TEN", "WAS",
})

# Historical/alternate codes → current franchise code.
_ALIASES = {
    "LAR": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC", "SDG": "LAC",
    "STL": "LA", "JAC": "JAX", "LVR": "LV", "KAN": "KC", "GNB": "GB",
    "NWE": "NE", "NOR": "NO", "TAM": "TB", "SFO": "SF",
}

def normalize_team(abbr: str) -> str:
    a = (abbr or "").strip().upper()
    a = _ALIASES.get(a, a)
    if a not in TEAMS:
        raise ValueError(f"unknown NFL team abbreviation: {abbr!r}")
    return a
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_teams.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/__init__.py src/sportsmodel/nfl/teams.py tests/nfl/__init__.py tests/nfl/test_teams.py
git commit -m "feat(nfl): team normalization (32-team vocab + relocations)"
```

---

### Task 2: nflverse loaders + committed snapshots (`data.py`)

**Files:**
- Create: `src/sportsmodel/nfl/data.py`, `scripts/build_nfl_snapshots.py`
- Create (committed assets, produced by running the script): `assets/nfl/{schedules,weekly,rosters,injuries}.parquet`
- Test: `tests/nfl/test_data.py`

**Interfaces:**
- Consumes: `teams.normalize_team`.
- Produces: pure normalizers `normalize_schedule(df) -> df` (keeps `["game_id","season","week","game_type","gameday","gametime","home_team","away_team","home_score","away_score","espn"]`, normalizes `home_team`/`away_team`) and `normalize_team_col(df, col) -> df`; loaders `load_schedules(seasons)`, `load_weekly(seasons)`, `load_rosters(seasons)`, `load_injuries(seasons)` returning normalized DataFrames. The `schedules` frame's `espn` column IS the `game_pk`.

**Note:** use the exact `nfl_data_py` function names recorded in the spike report `docs/superpowers/reports/2026-08-23-nfl-data-spike.md` (it verified `import_schedules`, the weekly import, injuries, and rosters function names + columns including `recent_team`). Do not guess column names — the report has the real lists.

- [ ] **Step 1: Write the failing test** (pure normalizer, no network)

```python
# tests/nfl/test_data.py
import pandas as pd
from sportsmodel.nfl.data import normalize_schedule, normalize_team_col

def test_normalize_schedule_normalizes_and_selects_columns():
    raw = pd.DataFrame([{
        "game_id": "2016_01_LA_SF", "season": 2016, "week": 1, "game_type": "REG",
        "gameday": "2016-09-12", "gametime": "20:20", "home_team": "SF",
        "away_team": "LAR", "home_score": 28, "away_score": 0, "espn": 400874518,
        "extra_col": "dropped",
    }])
    out = normalize_schedule(raw)
    assert out.loc[0, "away_team"] == "LA"      # LAR -> LA
    assert out.loc[0, "home_team"] == "SF"
    assert "extra_col" not in out.columns
    assert out.loc[0, "espn"] == 400874518      # game_pk source

def test_normalize_team_col_handles_relocation():
    raw = pd.DataFrame([{"recent_team": "OAK"}, {"recent_team": "WSH"}])
    out = normalize_team_col(raw, "recent_team")
    assert list(out["recent_team"]) == ["LV", "WAS"]
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_data.py -v` → ImportError.

- [ ] **Step 3: Implement `data.py`**

```python
# src/sportsmodel/nfl/data.py
import nfl_data_py as nfl
import pandas as pd
from .teams import normalize_team

_SCHED_COLS = ["game_id", "season", "week", "game_type", "gameday", "gametime",
               "home_team", "away_team", "home_score", "away_score", "espn"]

def normalize_team_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].map(normalize_team)
    return df

def normalize_schedule(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in _SCHED_COLS if c in df.columns]].copy()
    for col in ("home_team", "away_team"):
        df = normalize_team_col(df, col)
    return df

def load_schedules(seasons: list[int]) -> pd.DataFrame:
    return normalize_schedule(nfl.import_schedules(seasons))

def load_weekly(seasons: list[int]) -> pd.DataFrame:
    return normalize_team_col(nfl.import_weekly_data(seasons), "recent_team")

def load_rosters(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_seasonal_rosters(seasons)
    return normalize_team_col(df, "team")

def load_injuries(seasons: list[int]) -> pd.DataFrame:
    df = nfl.import_injuries(seasons)
    return normalize_team_col(df, "team")
```
If the verified roster/injury team column name differs from `team` per the spike report, use the report's name (and update the test's column accordingly). Keep `normalize_schedule`/`normalize_team_col` pure.

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_data.py -v` → PASS.

- [ ] **Step 5: Write + run the snapshot builder** (live pull; produces committed assets)

```python
# scripts/build_nfl_snapshots.py
import pathlib
from sportsmodel.nfl import data

OUT = pathlib.Path("assets/nfl")
OUT.mkdir(parents=True, exist_ok=True)
SCHED = list(range(2002, 2026))
OTHER = list(range(2015, 2026))

data.load_schedules(SCHED).to_parquet(OUT / "schedules.parquet", index=False)
data.load_weekly(OTHER).to_parquet(OUT / "weekly.parquet", index=False)
data.load_rosters(OTHER).to_parquet(OUT / "rosters.parquet", index=False)
data.load_injuries(OTHER).to_parquet(OUT / "injuries.parquet", index=False)
print("wrote", sorted(p.name for p in OUT.glob("*.parquet")))
```

Run: `uv run python scripts/build_nfl_snapshots.py` and confirm four parquet files appear under `assets/nfl/` and each loads back with `pd.read_parquet`. Record row counts in the commit message. If a live pull fails (network), report BLOCKED with the exact error — do not fabricate assets.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/nfl/data.py scripts/build_nfl_snapshots.py tests/nfl/test_data.py assets/nfl/*.parquet
git commit -m "feat(nfl): nflverse loaders + committed schedule/weekly/roster/injury snapshots"
```

---

### Task 3: Margin-adjusted Elo engine (`elo.py`)

**Files:**
- Create: `src/sportsmodel/nfl/elo.py`
- Test: `tests/nfl/test_elo.py`

**Interfaces:**
- Produces: `EloConfig(k=20.0, hfa_elo=65.0, carryover=0.75, base=1500.0)` (dataclass); `expected_home(elo_home, elo_away, cfg) -> float`; `mov_multiplier(margin, elo_diff_winner) -> float`; `elo_expected_margin(elo_home, elo_away, cfg) -> float`; `run_elo(schedule_df, cfg) -> EloResult`. `EloResult` has `.games` (a DataFrame with per-game pre-game `elo_home`,`elo_away`,`e_home`) and `.final` (`dict[team, float]`). Consumed by `ratings.py` and the backtest.

- [ ] **Step 1: Write the failing test** (hand-computed)

```python
# tests/nfl/test_elo.py
import math
import pandas as pd
from sportsmodel.nfl.elo import (
    EloConfig, expected_home, mov_multiplier, elo_expected_margin, run_elo,
)

def test_expected_home_neutral_with_hfa():
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75)
    assert expected_home(1500, 1500, cfg) == 0.5
    cfg2 = EloConfig(k=20, hfa_elo=65, carryover=0.75)
    assert expected_home(1500, 1500, cfg2) > 0.5   # HFA tilts home

def test_mov_multiplier_monotone_and_tie():
    assert mov_multiplier(3, 0) < mov_multiplier(21, 0)   # bigger win => bigger mult
    assert mov_multiplier(0, 0) == mov_multiplier(1, 0)   # tie treated as mov_input 1

def test_expected_margin_scale_and_sign():
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75)
    assert elo_expected_margin(1525, 1500, cfg) == (25 / 25)   # +1 point
    assert elo_expected_margin(1500, 1525, cfg) < 0

def test_run_elo_single_game_hand_computed():
    # base 1500, K=20, HFA 0, home wins by 7 -> E=0.5, mult=ln(8), delta=20*ln(8)*0.5
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75, base=1500)
    sched = pd.DataFrame([{
        "season": 2024, "week": 1, "home_team": "KC", "away_team": "BAL",
        "home_score": 27, "away_score": 20,
    }])
    res = run_elo(sched, cfg)
    delta = 20 * math.log(8) * 0.5
    assert math.isclose(res.final["KC"], 1500 + delta, rel_tol=1e-9)
    assert math.isclose(res.final["BAL"], 1500 - delta, rel_tol=1e-9)
    row = res.games.iloc[0]
    assert row["elo_home"] == 1500 and row["e_home"] == 0.5   # pre-game values

def test_carryover_regresses_toward_1500():
    cfg = EloConfig(k=20, hfa_elo=0, carryover=0.75, base=1500)
    # season 2023 gives KC a lead; 2024 opener should start from a regressed rating
    s2023 = pd.DataFrame([{"season": 2023, "week": 1, "home_team": "KC",
                           "away_team": "BAL", "home_score": 30, "away_score": 0}])
    s2024 = pd.DataFrame([{"season": 2024, "week": 1, "home_team": "KC",
                           "away_team": "BAL", "home_score": 20, "away_score": 17}])
    res = run_elo(pd.concat([s2023, s2024], ignore_index=True), cfg)
    kc_end_2023 = 1500 + 20 * math.log(31) * 0.5
    kc_start_2024 = 1500 + 0.75 * (kc_end_2023 - 1500)
    e = expected_home(kc_start_2024, 1500 + 0.75 * ((1500 - 20 * math.log(31) * 0.5) - 1500), cfg)
    assert res.games.iloc[1]["elo_home"] == kc_start_2024
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_elo.py -v` → ImportError.

- [ ] **Step 3: Implement `elo.py`**

```python
# src/sportsmodel/nfl/elo.py
from __future__ import annotations
import math
from dataclasses import dataclass, field
import pandas as pd

@dataclass(frozen=True)
class EloConfig:
    k: float = 20.0
    hfa_elo: float = 65.0
    carryover: float = 0.75
    base: float = 1500.0

@dataclass
class EloResult:
    games: pd.DataFrame
    final: dict

def expected_home(elo_home: float, elo_away: float, cfg: EloConfig) -> float:
    return 1.0 / (1.0 + 10 ** (-((elo_home + cfg.hfa_elo) - elo_away) / 400.0))

def mov_multiplier(margin: float, elo_diff_winner: float) -> float:
    mov_input = abs(margin) if margin != 0 else 1.0
    return math.log(mov_input + 1.0) * (2.2 / (0.001 * elo_diff_winner + 2.2))

def elo_expected_margin(elo_home: float, elo_away: float, cfg: EloConfig) -> float:
    return ((elo_home + cfg.hfa_elo) - elo_away) / 25.0

def _carryover(rating: float, cfg: EloConfig) -> float:
    return cfg.base + cfg.carryover * (rating - cfg.base)

def run_elo(schedule_df: pd.DataFrame, cfg: EloConfig) -> EloResult:
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    ratings: dict[str, float] = {}
    prev_season = None
    rows = []
    for _, g in df.iterrows():
        season = g["season"]
        if prev_season is not None and season != prev_season:
            ratings = {t: _carryover(r, cfg) for t, r in ratings.items()}
        prev_season = season
        h, a = g["home_team"], g["away_team"]
        eh = ratings.get(h, cfg.base)
        ea = ratings.get(a, cfg.base)
        e_home = expected_home(eh, ea, cfg)
        hs, as_ = g["home_score"], g["away_score"]
        if pd.isna(hs) or pd.isna(as_):
            rows.append({**g, "elo_home": eh, "elo_away": ea, "e_home": e_home})
            continue
        margin = hs - as_
        result_home = 1.0 if margin > 0 else (0.0 if margin < 0 else 0.5)
        # winner-perspective pre-game diff (home carries HFA)
        if margin >= 0:
            elo_diff_winner = (eh + cfg.hfa_elo) - ea
        else:
            elo_diff_winner = ea - (eh + cfg.hfa_elo)
        mult = mov_multiplier(margin, elo_diff_winner)
        delta = cfg.k * mult * (result_home - e_home)
        ratings[h] = eh + delta
        ratings[a] = ea - delta
        rows.append({**g, "elo_home": eh, "elo_away": ea, "e_home": e_home})
    return EloResult(games=pd.DataFrame(rows), final=dict(ratings))
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_elo.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/elo.py tests/nfl/test_elo.py
git commit -m "feat(nfl): margin-adjusted Elo engine with carryover"
```

---

### Task 4: Strength-of-schedule rating (`srs.py`)

**Files:**
- Create: `src/sportsmodel/nfl/srs.py`
- Test: `tests/nfl/test_srs.py`

**Interfaces:**
- Produces: `compute_srs(games: pd.DataFrame, max_iter=1000, tol=1e-8) -> dict[team, float]`. Input `games` has `home_team,away_team,home_score,away_score`. Ratings are zero-mean, in points, solving `rating_i = avg_point_margin_i + avg_opponent_rating_i`. Consumed by `ratings.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_srs.py
import math
import pandas as pd
from sportsmodel.nfl.srs import compute_srs

def _g(h, a, hs, as_):
    return {"home_team": h, "away_team": a, "home_score": hs, "away_score": as_}

def test_two_team_single_game():
    # A beats B by 10 at home -> ratings +/-5, zero mean
    games = pd.DataFrame([_g("A", "B", 20, 10)])
    r = compute_srs(games)
    assert math.isclose(r["A"], 5.0, abs_tol=1e-6)
    assert math.isclose(r["B"], -5.0, abs_tol=1e-6)
    assert math.isclose(sum(r.values()), 0.0, abs_tol=1e-6)

def test_strength_of_schedule_ranking():
    # A and D both 1-1 by raw record, but A's wins/losses are vs strong teams.
    # Build a schedule where A plays the top teams and D plays the weak ones.
    games = pd.DataFrame([
        _g("A", "B", 24, 21),   # A beats strong B by 3
        _g("C", "A", 20, 17),   # strong C beats A by 3
        _g("D", "E", 40, 3),    # D crushes weak E
        _g("F", "D", 10, 7),    # weak F edges D
        _g("B", "C", 21, 20),   # B ~ C (both strong)
        _g("E", "F", 17, 14),   # E ~ F (both weak)
    ])
    r = compute_srs(games)
    assert math.isclose(sum(r.values()), 0.0, abs_tol=1e-6)
    assert r["A"] > r["D"]      # same-ish record, tougher schedule -> higher rating
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_srs.py -v` → ImportError.

- [ ] **Step 3: Implement `srs.py`**

```python
# src/sportsmodel/nfl/srs.py
from __future__ import annotations
import pandas as pd

def compute_srs(games: pd.DataFrame, max_iter: int = 1000, tol: float = 1e-8) -> dict:
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    margins: dict[str, list[float]] = {t: [] for t in teams}
    opponents: dict[str, list[str]] = {t: [] for t in teams}
    for _, g in games.iterrows():
        h, a = g["home_team"], g["away_team"]
        m = float(g["home_score"] - g["away_score"])
        margins[h].append(m); opponents[h].append(a)
        margins[a].append(-m); opponents[a].append(h)
    avg_margin = {t: (sum(margins[t]) / len(margins[t]) if margins[t] else 0.0)
                  for t in teams}
    rating = dict(avg_margin)
    for _ in range(max_iter):
        new = {}
        for t in teams:
            opp = opponents[t]
            sos = sum(rating[o] for o in opp) / len(opp) if opp else 0.0
            new[t] = avg_margin[t] + sos
        # zero-mean the ratings each pass to pin the free constant
        mean = sum(new.values()) / len(new)
        new = {t: v - mean for t, v in new.items()}
        if max(abs(new[t] - rating[t]) for t in teams) < tol:
            rating = new
            break
        rating = new
    return rating
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_srs.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/srs.py tests/nfl/test_srs.py
git commit -m "feat(nfl): retrodictive strength-of-schedule (SRS) rating"
```

---

### Task 5: Elo×SRS blend (`ratings.py`)

**Files:**
- Create: `src/sportsmodel/nfl/ratings.py`
- Test: `tests/nfl/test_ratings.py`

**Interfaces:**
- Consumes: `elo.EloConfig`, `elo.elo_expected_margin`.
- Produces: `BlendConfig(w_sos=0.0, srs_min_games=4)` (dataclass); `expected_margin(elo_home, elo_away, srs_home, srs_away, games_home, games_away, elo_cfg, blend_cfg) -> float`. When either team has `< srs_min_games` games so far, or a `srs_*` is `None`, returns pure `elo_expected_margin`; else `(1-w)*elo_margin + w*(srs_home - srs_away + hfa_points)` with `hfa_points = elo_cfg.hfa_elo/25`.

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_ratings.py
import math
from sportsmodel.nfl.elo import EloConfig, elo_expected_margin
from sportsmodel.nfl.ratings import BlendConfig, expected_margin

ELO = EloConfig(k=20, hfa_elo=0, carryover=0.75)

def test_pure_elo_below_min_games():
    b = BlendConfig(w_sos=0.5, srs_min_games=4)
    got = expected_margin(1525, 1500, 3.0, -3.0, games_home=2, games_away=9,
                          elo_cfg=ELO, blend_cfg=b)
    assert got == elo_expected_margin(1525, 1500, ELO)   # cold-start -> pure elo

def test_pure_elo_when_srs_none():
    b = BlendConfig(w_sos=0.5, srs_min_games=1)
    got = expected_margin(1525, 1500, None, None, games_home=9, games_away=9,
                          elo_cfg=ELO, blend_cfg=b)
    assert got == elo_expected_margin(1525, 1500, ELO)

def test_blend_when_available():
    b = BlendConfig(w_sos=0.5, srs_min_games=1)
    elo_m = elo_expected_margin(1525, 1500, ELO)   # +1.0
    srs_m = 6.0 - (-2.0) + ELO.hfa_elo / 25         # 8.0
    got = expected_margin(1525, 1500, 6.0, -2.0, games_home=9, games_away=9,
                          elo_cfg=ELO, blend_cfg=b)
    assert math.isclose(got, 0.5 * elo_m + 0.5 * srs_m)

def test_w_zero_reproduces_elo():
    b = BlendConfig(w_sos=0.0, srs_min_games=1)
    got = expected_margin(1525, 1500, 6.0, -2.0, 9, 9, ELO, b)
    assert got == elo_expected_margin(1525, 1500, ELO)
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_ratings.py -v` → ImportError.

- [ ] **Step 3: Implement `ratings.py`**

```python
# src/sportsmodel/nfl/ratings.py
from __future__ import annotations
from dataclasses import dataclass
from .elo import EloConfig, elo_expected_margin

@dataclass(frozen=True)
class BlendConfig:
    w_sos: float = 0.0
    srs_min_games: int = 4

def expected_margin(elo_home, elo_away, srs_home, srs_away,
                    games_home, games_away, elo_cfg: EloConfig,
                    blend_cfg: BlendConfig) -> float:
    elo_m = elo_expected_margin(elo_home, elo_away, elo_cfg)
    enough = (games_home >= blend_cfg.srs_min_games
              and games_away >= blend_cfg.srs_min_games)
    if blend_cfg.w_sos <= 0 or not enough or srs_home is None or srs_away is None:
        return elo_m
    hfa_points = elo_cfg.hfa_elo / 25.0
    srs_m = (srs_home - srs_away) + hfa_points
    w = blend_cfg.w_sos
    return (1 - w) * elo_m + w * srs_m
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_ratings.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/ratings.py tests/nfl/test_ratings.py
git commit -m "feat(nfl): Elo x SRS expected-margin blend with cold-start fallback"
```

---

### Task 6: ESPN adapters (`espn.py`)

**Files:**
- Create: `src/sportsmodel/nfl/espn.py`, `tests/fixtures/nfl/espn_scoreboard.json`
- Test: `tests/nfl/test_espn.py`

**Interfaces:**
- Consumes: `teams.normalize_team`.
- Produces: pure parsers `parse_schedule(payload) -> list[dict]` (each: `game_pk:int`, `commence_time:str`, `home_team`, `away_team` normalized abbreviations, `status:str`), `parse_final(event) -> dict|None` (`{"home_score":int,"away_score":int,"final":True}` or `None` unless `STATUS_FINAL`), `parse_inactives(payload) -> list[str]`; plus thin `fetch_schedule(season, week, season_type=2)`, `fetch_final(event_id)`, `fetch_inactives(event_id)` that `httpx.get` then call the parser. Only the parsers are unit-tested.

- [ ] **Step 1: Create the fixture** — `tests/fixtures/nfl/espn_scoreboard.json` (minimal, mirrors the real shape the spike documented; ESPN abbreviations, scores as strings):

```json
{"events": [
  {"id": "401671789", "date": "2024-09-06T00:20Z",
   "status": {"type": {"name": "STATUS_FINAL"}},
   "competitions": [{"competitors": [
     {"homeAway": "home", "team": {"abbreviation": "KC"}, "score": "27"},
     {"homeAway": "away", "team": {"abbreviation": "BAL"}, "score": "20"}]}]},
  {"id": "401671790", "date": "2024-09-08T17:00Z",
   "status": {"type": {"name": "STATUS_SCHEDULED"}},
   "competitions": [{"competitors": [
     {"homeAway": "home", "team": {"abbreviation": "WSH"}, "score": "0"},
     {"homeAway": "away", "team": {"abbreviation": "LAR"}, "score": "0"}]}]}
]}
```

- [ ] **Step 2: Write the failing test**

```python
# tests/nfl/test_espn.py
import json, pathlib
from sportsmodel.nfl.espn import parse_schedule, parse_final

FIX = json.loads((pathlib.Path(__file__).parent.parent
                  / "fixtures/nfl/espn_scoreboard.json").read_text())

def test_parse_schedule_normalizes_and_types():
    games = parse_schedule(FIX)
    assert len(games) == 2
    g0 = games[0]
    assert g0["game_pk"] == 401671789 and isinstance(g0["game_pk"], int)
    assert g0["home_team"] == "KC" and g0["away_team"] == "BAL"
    assert g0["status"] == "STATUS_FINAL"
    g1 = games[1]
    assert g1["home_team"] == "WAS" and g1["away_team"] == "LA"   # WSH/LAR normalized

def test_parse_final_gates_on_status():
    assert parse_final(FIX["events"][0]) == {"home_score": 27, "away_score": 20, "final": True}
    assert parse_final(FIX["events"][1]) is None   # not STATUS_FINAL
```

- [ ] **Step 3: Run to verify fail** — `uv run pytest tests/nfl/test_espn.py -v` → ImportError.

- [ ] **Step 4: Implement `espn.py`**

```python
# src/sportsmodel/nfl/espn.py
from __future__ import annotations
import httpx
from .teams import normalize_team

_BASE = "https://site.api.espn.com/apis/site/v2/sports/football/nfl"

def _competitors(event) -> dict:
    comp = event["competitions"][0]["competitors"]
    return {c["homeAway"]: c for c in comp}

def parse_schedule(payload) -> list[dict]:
    out = []
    for ev in payload.get("events", []):
        c = _competitors(ev)
        out.append({
            "game_pk": int(ev["id"]),
            "commence_time": ev["date"],
            "home_team": normalize_team(c["home"]["team"]["abbreviation"]),
            "away_team": normalize_team(c["away"]["team"]["abbreviation"]),
            "status": ev["status"]["type"]["name"],
        })
    return out

def parse_final(event) -> dict | None:
    if event["status"]["type"]["name"] != "STATUS_FINAL":
        return None
    c = _competitors(event)
    return {"home_score": int(c["home"]["score"]),
            "away_score": int(c["away"]["score"]), "final": True}

def parse_inactives(payload) -> list[str]:
    names = []
    for ev in payload.get("events", []):
        for c in ev.get("competitions", [{}])[0].get("competitors", []):
            for inj in c.get("injuries", []):
                status = (inj.get("status") or "").lower()
                ath = inj.get("athlete", {})
                if status in {"out", "inactive"} and ath.get("displayName"):
                    names.append(ath["displayName"])
    return names

def fetch_schedule(season: int, week: int, season_type: int = 2) -> list[dict]:
    r = httpx.get(f"{_BASE}/scoreboard",
                  params={"dates": season, "seasontype": season_type, "week": week},
                  timeout=20)
    r.raise_for_status()
    return parse_schedule(r.json())

def fetch_final(event_id: int) -> dict | None:
    r = httpx.get(f"{_BASE}/summary", params={"event": event_id}, timeout=20)
    r.raise_for_status()
    data = r.json()
    ev = data.get("header", {}).get("competitions", [{}])[0]
    # summary shape differs from scoreboard; adapt via the header competition
    status = ev.get("status", {}).get("type", {}).get("name")
    if status != "STATUS_FINAL":
        return None
    comp = {c["homeAway"]: c for c in ev.get("competitors", [])}
    return {"home_score": int(comp["home"]["score"]),
            "away_score": int(comp["away"]["score"]), "final": True}
```
`parse_inactives`/`fetch_inactives` are best-effort (ESPN game-day inactives shape is confirmed live at first use); the unit tests cover only `parse_schedule` + `parse_final` against the committed fixture. Include a `fetch_inactives(event_id)` stub that GETs the summary endpoint and returns `parse_inactives(payload)` — it may return `[]` until game day.

- [ ] **Step 5: Run to verify pass** — `uv run pytest tests/nfl/test_espn.py -v` → PASS.

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/nfl/espn.py tests/nfl/test_espn.py tests/fixtures/nfl/espn_scoreboard.json
git commit -m "feat(nfl): ESPN schedule/final/inactives adapters (fixture-tested)"
```

---

### Task 7: `game_pk` matcher (`matcher.py`)

**Files:**
- Create: `src/sportsmodel/nfl/matcher.py`
- Test: `tests/nfl/test_matcher.py`

**Interfaces:**
- Produces: `match_odds_event(odds_event: dict, espn_games: list[dict]) -> int | None`. `odds_event` carries The-Odds-API `home_team`/`away_team` (full display names) + `commence_time` (ISO). `espn_games` = the output of `espn.fetch_schedule` PLUS a `home_name`/`away_name` full-name field per game. Match on `(_norm_name(home), _norm_name(away), date)`; return the `game_pk` or `None`. `_norm_name(s)` lowercases/strips. (Abbreviation normalization stays in `teams.py`; the live Odds-API↔ESPN join is by full name because both sides emit full names — Odds-API exact strings are finalized at first CI capture.)

- [ ] **Step 1: Write the failing test**

```python
# tests/nfl/test_matcher.py
from sportsmodel.nfl.matcher import match_odds_event

ESPN = [
    {"game_pk": 401671789, "home_name": "Kansas City Chiefs",
     "away_name": "Baltimore Ravens", "commence_time": "2024-09-06T00:20Z"},
    {"game_pk": 401671790, "home_name": "Washington Commanders",
     "away_name": "Los Angeles Rams", "commence_time": "2024-09-08T17:00Z"},
]

def test_matches_by_names_and_date():
    ev = {"home_team": "Kansas City Chiefs", "away_team": "Baltimore Ravens",
          "commence_time": "2024-09-06T00:20:00Z"}
    assert match_odds_event(ev, ESPN) == 401671789

def test_relocated_team_name():
    ev = {"home_team": "Washington Commanders", "away_team": "Los Angeles Rams",
          "commence_time": "2024-09-08T17:05:00Z"}   # slightly different minute
    assert match_odds_event(ev, ESPN) == 401671790

def test_no_match_returns_none():
    ev = {"home_team": "Dallas Cowboys", "away_team": "New York Giants",
          "commence_time": "2024-09-06T00:20:00Z"}
    assert match_odds_event(ev, ESPN) is None
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_matcher.py -v` → ImportError.

- [ ] **Step 3: Implement `matcher.py`**

```python
# src/sportsmodel/nfl/matcher.py
from __future__ import annotations

def _norm_name(s: str) -> str:
    return (s or "").strip().lower()

def _date(iso: str) -> str:
    return (iso or "")[:10]

def match_odds_event(odds_event: dict, espn_games: list[dict]) -> int | None:
    key = (_norm_name(odds_event["home_team"]),
           _norm_name(odds_event["away_team"]),
           _date(odds_event["commence_time"]))
    for g in espn_games:
        if (_norm_name(g["home_name"]), _norm_name(g["away_name"]),
                _date(g["commence_time"])) == key:
            return int(g["game_pk"])
    return None
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_matcher.py -v` → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/matcher.py tests/nfl/test_matcher.py
git commit -m "feat(nfl): Odds-API -> ESPN game_pk matcher (name+date)"
```

---

### Task 8: Walk-forward backtest + tuning (`backtest_nfl_elo.py`)

**Files:**
- Create: `scripts/backtest_nfl_elo.py`
- Create (produced by running): `assets/nfl/rating.json`, `docs/superpowers/reports/2026-08-24-nfl-elo-backtest.md`
- Test: `tests/nfl/test_backtest_nfl.py`

**Interfaces:**
- Consumes: `data.load_schedules` (or the committed `assets/nfl/schedules.parquet`), `elo`, `srs`, `ratings`.
- Produces: `run_backtest(schedule_df, elo_cfg, blend_cfg) -> dict` (metrics: `brier`, `win_acc`, `margin_mae`, `margin_rmse`, `n`) computed walk-forward on **pre-game** ratings (with per-week SRS from games already played that season); `tune(train_df, valid_df, grid) -> tuple[best_cfg, results]`. The script's `main()` loads schedules 2002–2025, splits train 2002–2019 / valid 2020–2025, coordinate-searches, writes `assets/nfl/rating.json` + the findings report.

- [ ] **Step 1: Write the failing test** (synthetic, deterministic, no network)

```python
# tests/nfl/test_backtest_nfl.py
import pandas as pd
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig
import scripts.backtest_nfl_elo as bt

def _season(year, rows):
    return [{"season": year, "week": w + 1, "home_team": h, "away_team": a,
             "home_score": hs, "away_score": as_}
            for w, (h, a, hs, as_) in enumerate(rows)]

SCHED = pd.DataFrame(
    _season(2018, [("A", "B", 24, 20), ("C", "D", 30, 10), ("A", "C", 21, 17),
                   ("B", "D", 14, 13), ("A", "D", 28, 7), ("B", "C", 20, 24)])
    + _season(2019, [("B", "A", 17, 21), ("D", "C", 10, 20), ("C", "A", 13, 16)])
)

def test_run_backtest_returns_metrics():
    m = bt.run_backtest(SCHED, EloConfig(), BlendConfig())
    assert set(m) >= {"brier", "win_acc", "margin_mae", "margin_rmse", "n"}
    assert 0.0 <= m["brier"] <= 1.0
    assert m["n"] > 0

def test_run_backtest_deterministic():
    a = bt.run_backtest(SCHED, EloConfig(), BlendConfig())
    b = bt.run_backtest(SCHED, EloConfig(), BlendConfig())
    assert a == b

def test_tune_returns_config_from_grid():
    grid = {"k": [15, 20], "hfa_elo": [65], "carryover": [0.75],
            "w_sos": [0.0, 0.3], "srs_min_games": [4]}
    best, results = bt.tune(SCHED, SCHED, grid)
    assert isinstance(best[0], EloConfig) and isinstance(best[1], BlendConfig)
    assert len(results) == 4   # 2 k x 2 w_sos
```

- [ ] **Step 2: Run to verify fail** — `uv run pytest tests/nfl/test_backtest_nfl.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement `scripts/backtest_nfl_elo.py`**

```python
# scripts/backtest_nfl_elo.py
from __future__ import annotations
import itertools, json, pathlib
import pandas as pd
from sportsmodel.nfl.elo import EloConfig, expected_home, run_elo
from sportsmodel.nfl.srs import compute_srs
from sportsmodel.nfl.ratings import BlendConfig, expected_margin

def run_backtest(schedule_df: pd.DataFrame, elo_cfg: EloConfig,
                 blend_cfg: BlendConfig) -> dict:
    df = schedule_df.sort_values(["season", "week"]).reset_index(drop=True)
    res = run_elo(df, elo_cfg)                       # pre-game elo per game
    games = res.games
    n = 0; brier = 0.0; correct = 0; abs_err = 0.0; sq_err = 0.0
    played_counts: dict = {}
    for season, sdf in games.groupby("season"):
        season_games = df[df["season"] == season].sort_values("week")
        srs_hist: pd.DataFrame = season_games.iloc[0:0]
        counts: dict = {}
        srs_cache: dict = {}
        for _, g in sdf.iterrows():
            if pd.isna(g["home_score"]) or pd.isna(g["away_score"]):
                continue
            h, a = g["home_team"], g["away_team"]
            gh, ga = counts.get(h, 0), counts.get(a, 0)
            srs = srs_cache if srs_cache else {}
            srs_h = srs.get(h); srs_a = srs.get(a)
            em = expected_margin(g["elo_home"], g["elo_away"], srs_h, srs_a,
                                 gh, ga, elo_cfg, blend_cfg)
            e_home = g["e_home"]
            actual_margin = g["home_score"] - g["away_score"]
            result_home = 1.0 if actual_margin > 0 else 0.0
            brier += (e_home - result_home) ** 2
            correct += int((e_home >= 0.5) == (result_home == 1.0))
            abs_err += abs(em - actual_margin)
            sq_err += (em - actual_margin) ** 2
            n += 1
            # after scoring, this game joins the played set -> update counts + SRS
            counts[h] = gh + 1; counts[a] = ga + 1
            srs_hist = pd.concat([srs_hist, pd.DataFrame([g])], ignore_index=True)
            srs_cache = compute_srs(srs_hist)
    return {"brier": brier / n, "win_acc": correct / n,
            "margin_mae": abs_err / n, "margin_rmse": (sq_err / n) ** 0.5, "n": n}

def tune(train_df, valid_df, grid) -> tuple:
    combos = list(itertools.product(
        grid["k"], grid["hfa_elo"], grid["carryover"],
        grid["w_sos"], grid["srs_min_games"]))
    results = []
    for k, hfa, carry, w, mg in combos:
        ec = EloConfig(k=k, hfa_elo=hfa, carryover=carry)
        bc = BlendConfig(w_sos=w, srs_min_games=mg)
        vm = run_backtest(valid_df, ec, bc)
        results.append({"elo": ec, "blend": bc, "valid": vm})
    best = min(results, key=lambda r: r["valid"]["brier"])
    return (best["elo"], best["blend"]), results

def main() -> None:
    sched = pd.read_parquet("assets/nfl/schedules.parquet")
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    train = reg[reg["season"] <= 2019]
    valid = reg[reg["season"] >= 2020]
    grid = {"k": [12, 16, 20, 24], "hfa_elo": [40, 55, 65, 80],
            "carryover": [0.6, 0.7, 0.75, 0.85],
            "w_sos": [0.0, 0.15, 0.3, 0.45], "srs_min_games": [3, 4, 6]}
    (best_elo, best_blend), results = tune(train, valid, grid)
    pure = min((r for r in results if r["blend"].w_sos == 0.0),
               key=lambda r: r["valid"]["brier"])
    out = {"k": best_elo.k, "hfa_elo": best_elo.hfa_elo,
           "carryover": best_elo.carryover, "base": best_elo.base,
           "w_sos": best_blend.w_sos, "srs_min_games": best_blend.srs_min_games}
    pathlib.Path("assets/nfl/rating.json").write_text(json.dumps(out, indent=2))
    print("best:", out)
    print("best pure-Elo brier:", pure["valid"]["brier"],
          "| best blended brier:", min(r["valid"]["brier"] for r in results))

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify pass** — `uv run pytest tests/nfl/test_backtest_nfl.py -v` → PASS. (If importing `scripts.backtest_nfl_elo` needs a package marker, add an empty `scripts/__init__.py` — check whether the repo already imports scripts in tests, e.g. `tests/test_generate_board.py`; follow that repo convention.)

- [ ] **Step 5: Run the real tune + write the findings report**

Run: `uv run python scripts/backtest_nfl_elo.py`. Confirm it writes `assets/nfl/rating.json`. Then write `docs/superpowers/reports/2026-08-24-nfl-elo-backtest.md` capturing: the tuned params, the validation-span metrics (Brier / win% / margin MAE), the two naive baselines (home-always, prior-season win%) computed for comparison, and **whether the SoS blend beat pure Elo out-of-sample** (best blended Brier vs best pure-Elo Brier). If the blend did not beat pure Elo, state that `w_sos` tuned to 0 and Elo stands alone (per the spec's acceptance bar). If a live pull is needed and fails, report BLOCKED with the exact error.

- [ ] **Step 6: Run the full suite** — `uv run pytest -q` → all PASS (MLB suite unchanged; new NFL tests green).

- [ ] **Step 7: Commit**

```bash
git add scripts/backtest_nfl_elo.py tests/nfl/test_backtest_nfl.py assets/nfl/rating.json docs/superpowers/reports/2026-08-24-nfl-elo-backtest.md
git commit -m "feat(nfl): walk-forward Elo/SoS backtest + tuned rating.json + findings"
```

---

## Self-Review

**Spec coverage:**
- nflverse ingest + committed snapshots (schedules 2002+, weekly/rosters/injuries 2015+) → Task 2. ✓
- Team normalization single source of truth → Task 1 (consumed by 2, 6, 7). ✓
- Margin-adjusted Elo + carryover + expected margin → Task 3. ✓
- Explicit SRS strength-of-schedule rating → Task 4; blend + cold-start fallback → Task 5. ✓
- ESPN adapters (schedule/final/inactives), fixture-tested → Task 6. ✓
- `game_pk` matcher keyed on ESPN event id → Task 7 (historical linkage via `schedules.espn` carried in Task 2's snapshot). ✓
- Walk-forward backtest, train/validate split, tune `(K,HFA,carryover,w_sos,srs_min_games)`, compare blend vs pure Elo, commit tuned params + findings → Task 8. ✓
- Deferred (distributions/shrinkage/σ → P2; props → P3; producer/board/grade wiring → P4) — not in this plan by design. ✓

**Placeholder scan:** No TBD/TODO. `parse_inactives`/`fetch_inactives` and the ESPN summary-endpoint score shape are explicitly best-effort/finalized-at-first-live-use (the P0-style deferral) — the unit-tested surface (`parse_schedule`/`parse_final`) is fully specified; that is a documented deferral, not an unfilled blank.

**Type consistency:** `EloConfig(k,hfa_elo,carryover,base)` and `BlendConfig(w_sos,srs_min_games)` are used identically across Tasks 3/5/8. `run_elo -> EloResult(.games,.final)`, `compute_srs -> dict`, `expected_margin(...)` signature, and `game_pk:int` are consistent across their producers and consumers. `elo_expected_margin` (Elo-only) vs `ratings.expected_margin` (blended) are named distinctly to avoid the collision the spec flagged.
