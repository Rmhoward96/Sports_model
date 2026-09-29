# Matchups, QB Profiles and Defensive Injuries (props-ML v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opponent-adjusted pass/run unit matchups, per-unit defensive injuries and QB career profiles as inputs to the props-ML models, refit and re-gate them, and serve the result as `nfl-sim-ml-v2`.

**Architecture:** Three new PURE feature modules (`nfl/unit_efficiency.py`, `nfl/def_injuries.py`, `nfl/qb_profile.py`) feed the single feature builder (`nfl/player_features.py` via `scripts/build_player_features.py`) under new column prefixes `mx_`, `di_`, `qb_`. `sim/nfl/learned.py` gains three ladder rungs selecting those prefixes plus a QB-driven drive-TD shift; B's QB role requires recent passes. The existing A ladder (`train_props_ml.py`) decides rungs on 2021–23; the B ladder (`train_props_ml_b.py`) runs for v1 and v2 and writes served composites; a new `gate_props_ml_v2.py` compares v2 vs v1 on 2024–25 with QB-change and two-way matchup sub-checks. Serving picks artifacts by the served version.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn HGB, nflverse parquet releases, Supabase Postgres, GitHub Actions; site = CappingAlpha static JS (external, user deploys).

**Spec:** `docs/superpowers/specs/2026-09-28-matchup-qb-design.md`

## Global Constraints

- Every feature for (season S, week w) uses only games strictly before (S, w); the pre-game injury report and as-of depth chart are allowed (same rule as today's `st_` features).
- The builder (`build_player_features.build_tables`) stays the ONE builder for training, backtest and live serving.
- New columns use ONLY the new prefixes `mx_`, `di_`, `qb_` (plus labels `y_*`). No new `tm_`/`op_`/`p_`/`st_` columns — v1's rungs must see exactly v1's columns.
- Strong run D / weak pass D ⇒ RB rushing down, QB/WR/TE passing up; strong pass D / weak run D ⇒ the reverse (gate sub-check, both directions).
- QB profiles use every QB game from 1999 through the last completed week; weighting settings (H, k) are chosen on 2021–2023 for the gate and on 2021→latest for serving.
- Gate verdict seasons: 2024 and 2025 only; rung decisions and tuning: 2021–2023.
- v1 (`nfl-sim-ml-v1`) artifacts, rows and release stay intact; rollback = `UPDATE nfl_sim_serving SET model_version='nfl-sim-ml-v1'`.
- The user runs all DB writes/migrations; commit/push/merge/publish only when the user asks.
- End every commit message with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Tests: `.venv/bin/python -m pytest -q` must be green after every task.

## Rulings made while planning (spec deviations, recorded)

- R1 `mx_pass_minus_rush` is DEFENSE-only: `mx_op_pass_epa_allowed_adj − mx_op_rush_epa_allowed_adj` (+ ⇒ pass D weak relative to run D). The spec's formula mixed offense in and had the sign backwards for the sub-check.
- R2 No explicit early-season shrink: both the as-of current-season adjusted value (`_adj`) and last season's full value (`_prev`) are features; the HGB learns the blend (same pattern as `tm_off_adj`/`tm_off_prev`).
- R3 H, k are tuned by direct next-game prediction of opponent-adjusted YPA (cheap), not by ladder reruns; stored in `assets/nfl/props_ml/qb_profile_params.json` (`gate`: fit on 2021–23; `serving`: fit on 2021→latest).
- R4 The drive-TD elasticity `e` is fit inside `learned.fit_models` on the training window (OLS, no tuning loop).
- R5 Position-specific opponent-adjusted ypt allowed is dropped (existing raw `op_ypt_allowed_*` stay); unit-level pass metrics carry the adjustment.
- R6 PFR's `LB` (includes edge OLBs) counts as run defense; `DE`/`OLB`/`EDGE` as pass rush; `CB`/`S`/`FS`/`SS`/`DB` as coverage; `DT`/`NT`/`DL`/`ILB`/`MLB`/`LB` as run defense.

---

### Task 1: Pass/run unit efficiency, opponent-adjusted (`mx_` features)

**Files:**
- Create: `src/sportsmodel/nfl/unit_efficiency.py`
- Test: `tests/nfl/test_unit_efficiency.py`

**Interfaces:**
- Consumes: `sportsmodel.nfl.efficiency.adjusted_efficiency(game_epa, season, upto_week)` (existing; generic over any per-game `{"off","def","opp"}` dict).
- Produces:
  - `UNIT_METRICS: tuple[str, ...] = ("pass_epa", "nypd", "sack_rate", "pass_success", "rush_epa", "ypc", "rush_success", "stuff_rate")`
  - `unit_games(pbp: pd.DataFrame) -> pd.DataFrame` — columns `season, week, team, opponent, *UNIT_METRICS`
  - `unit_features(unit_games_df: pd.DataFrame, team_weeks: pd.DataFrame) -> pd.DataFrame` — `team_weeks` has `season, week, team, opponent`; returns those keys plus `mx_tm_<m>_adj`, `mx_tm_<m>_prev`, `mx_op_<m>_allowed_adj`, `mx_op_<m>_allowed_prev` for every metric, and `mx_pass_edge`, `mx_rush_edge`, `mx_pass_minus_rush`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/nfl/test_unit_efficiency.py
import numpy as np
import pandas as pd

from sportsmodel.nfl.unit_efficiency import UNIT_METRICS, unit_features, unit_games


def _play(season, week, off, de, kind, yds, epa, success, sack=0):
    return {"season": season, "week": week, "season_type": "REG", "posteam": off, "defteam": de,
            "play_type": kind, "yards_gained": yds, "epa": epa, "success": success, "sack": sack}


def _pbp():
    rows = []
    # week 1: KC offense vs BUF D, BAL offense vs PIT D (and reverse)
    for wk, pairs in ((1, [("KC", "BUF"), ("BUF", "KC"), ("BAL", "PIT"), ("PIT", "BAL")]),
                      (2, [("KC", "BAL"), ("BAL", "KC"), ("BUF", "PIT"), ("PIT", "BUF")])):
        for off, de in pairs:
            # BUF's defense: stout vs the run (-1 yd carries), leaky vs the pass (+12 yd dropbacks)
            rush_yds = -1 if de == "BUF" else 5
            pass_yds = 12 if de == "BUF" else 6
            rows += [_play(2024, wk, off, de, "run", rush_yds, 0.1 if rush_yds > 0 else -0.5, int(rush_yds > 3))
                     for _ in range(10)]
            rows += [_play(2024, wk, off, de, "pass", pass_yds, 0.4 if pass_yds > 8 else 0.0, 1)
                     for _ in range(10)]
            rows.append(_play(2024, wk, off, de, "pass", -7, -1.5, 0, sack=1))
    return pd.DataFrame(rows)


def test_unit_games_rates():
    ug = unit_games(_pbp())
    kc1 = ug[(ug.week == 1) & (ug.team == "KC")].iloc[0]
    assert kc1.opponent == "BUF"
    assert kc1.ypc == -1.0 and kc1.stuff_rate == 1.0
    assert np.isclose(kc1.nypd, (12 * 10 - 7) / 11) and np.isclose(kc1.sack_rate, 1 / 11)
    assert set(UNIT_METRICS) <= set(ug.columns)


def test_features_are_strictly_prior_and_relative_to_league():
    ug = unit_games(_pbp())
    tw = pd.DataFrame({"season": [2024, 2024, 2024], "week": [1, 3, 3],
                       "team": ["KC", "KC", "PIT"], "opponent": ["BUF", "BUF", "BUF"]})
    f = unit_features(ug, tw)
    w1 = f[f.week == 1].iloc[0]
    assert np.isnan(w1.mx_op_ypc_allowed_adj)          # nothing before week 1 of 2024
    w3 = f[(f.week == 3) & (f.team == "KC")].iloc[0]
    # BUF's run D is the league's best (negative allowed), its pass D the worst (positive allowed)
    assert w3.mx_op_ypc_allowed_adj < 0 < w3.mx_op_nypd_allowed_adj
    assert w3.mx_pass_minus_rush > 0                    # R1: + means pass D weak vs run D
    assert w3.mx_pass_edge == w3.mx_tm_pass_epa_adj + w3.mx_op_pass_epa_allowed_adj


def test_changing_week_3_plays_does_not_move_week_3_features():
    pbp = _pbp()
    tw = pd.DataFrame({"season": [2024], "week": [3], "team": ["KC"], "opponent": ["BUF"]})
    base = unit_features(unit_games(pbp), tw)
    extra = pd.DataFrame([_play(2024, 3, "KC", "BUF", "run", 80, 5.0, 1)] * 50)
    moved = unit_features(unit_games(pd.concat([pbp, extra], ignore_index=True)), tw)
    cols = [c for c in base.columns if c.startswith("mx_")]
    pd.testing.assert_frame_equal(base[cols], moved[cols])


def test_prev_uses_last_season_full():
    pbp = _pbp()
    tw = pd.DataFrame({"season": [2025], "week": [1], "team": ["KC"], "opponent": ["BUF"]})
    f = unit_features(unit_games(pbp), tw).iloc[0]
    assert f.mx_op_ypc_allowed_prev < 0 and np.isnan(f.mx_op_ypc_allowed_adj)
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest -q tests/nfl/test_unit_efficiency.py`
Expected: FAIL — `ModuleNotFoundError: sportsmodel.nfl.unit_efficiency`

- [ ] **Step 3: Implement**

```python
# src/sportsmodel/nfl/unit_efficiency.py
"""Pass / run unit efficiency per team-game, opponent-adjusted and relative to
league (the `mx_` matchup features). PURE.

Offense metric per team-game from REG play-by-play (pass = dropbacks incl.
sacks, run = designed runs): pass_epa, nypd (net yards / dropback), sack_rate,
pass_success, rush_epa, ypc, rush_success, stuff_rate (carries <= 0 yds).
A defense's "allowed" value for a game is its opponent offense's value.

Adjustment reuses `efficiency.adjusted_efficiency` per metric (single pass,
same-season weeks < w only): `_adj` = as-of this season, `_prev` = last season
in full (week 99). Both are made relative to league (minus the league mean of
the same window), so + always means "above average" (more EPA / yards / sacks
/ success / stuffs). For defenses + means ALLOWS more than average (weaker),
except `sack_rate` / `stuff_rate` allowed, where + means the offense suffers
more (the defense is stronger at that). Plan rulings R1/R2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.efficiency import adjusted_efficiency
from sportsmodel.nfl.teams import normalize_team

UNIT_METRICS: tuple[str, ...] = ("pass_epa", "nypd", "sack_rate", "pass_success",
                                 "rush_epa", "ypc", "rush_success", "stuff_rate")
_KEYS = ["season", "week", "team"]


def _norm(code):
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def unit_games(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, offense team) with the UNIT_METRICS."""
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna() & pbp["defteam"].notna()]
    if "season_type" in p.columns:
        p = p[p["season_type"] == "REG"]
    p = p.assign(team=p["posteam"].map(_norm), opponent=p["defteam"].map(_norm),
                 _db=p["play_type"] == "pass", _run=p["play_type"] == "run",
                 _sack=p["sack"].fillna(0) == 1, _yds=p["yards_gained"].fillna(0.0),
                 _succ=p["success"].fillna(0.0), _epa=p["epa"])
    p = p.dropna(subset=["team", "opponent"])

    def agg(g: pd.DataFrame) -> pd.Series:
        db, run = g[g["_db"]], g[g["_run"]]
        nd, nr = len(db), len(run)
        return pd.Series({
            "opponent": g["opponent"].iloc[0],
            "pass_epa": db["_epa"].mean() if nd else np.nan,
            "nypd": db["_yds"].sum() / nd if nd else np.nan,
            "sack_rate": db["_sack"].mean() if nd else np.nan,
            "pass_success": db["_succ"].mean() if nd else np.nan,
            "rush_epa": run["_epa"].mean() if nr else np.nan,
            "ypc": run["_yds"].sum() / nr if nr else np.nan,
            "rush_success": run["_succ"].mean() if nr else np.nan,
            "stuff_rate": (run["_yds"] <= 0).mean() if nr else np.nan,
        })

    out = p.groupby(_KEYS).apply(agg, include_groups=False).reset_index()
    out[["season", "week"]] = out[["season", "week"]].astype("int64")
    return out


def _game_dict(ug: pd.DataFrame, metric: str) -> dict:
    """{(s, w, team): {"off": team's value, "def": value its opponent posted vs it, "opp"}}."""
    off = {(int(s), int(w), t): (v, o) for s, w, t, o, v in
           ug[["season", "week", "team", "opponent", metric]].itertuples(index=False)}
    out: dict = {}
    for (s, w, t), (v, o) in off.items():
        allowed = off.get((s, w, o), (None, None))[0]
        out[(s, w, t)] = {"off": None if pd.isna(v) else float(v),
                          "def": None if allowed is None or pd.isna(allowed) else float(allowed),
                          "opp": o}
    return out


def _relative(adj: dict, key: str) -> dict[str, float]:
    if not adj:
        return {}
    league = float(np.mean([v[key] for v in adj.values()]))
    return {t: v[key] - league for t, v in adj.items()}


def unit_features(unit_games_df: pd.DataFrame, team_weeks: pd.DataFrame) -> pd.DataFrame:
    """`team_weeks` (season, week, team, opponent) + mx_ columns (see module doc)."""
    games = {m: _game_dict(unit_games_df, m) for m in UNIT_METRICS}
    cache: dict = {}

    def rel(metric: str, s: int, w: int) -> tuple[dict, dict]:
        k = (metric, s, w)
        if k not in cache:
            adj = adjusted_efficiency(games[metric], s, w)
            cache[k] = (_relative(adj, "off_adj"), _relative(adj, "def_adj"))
        return cache[k]

    rows = []
    for s, w, team, opp in team_weeks[["season", "week", "team", "opponent"]].itertuples(index=False):
        s, w = int(s), int(w)
        r: dict = {}
        for m in UNIT_METRICS:
            off_now, def_now = rel(m, s, w)
            off_prev, def_prev = rel(m, s - 1, 99)
            r[f"mx_tm_{m}_adj"] = off_now.get(team, np.nan)
            r[f"mx_tm_{m}_prev"] = off_prev.get(team, np.nan)
            r[f"mx_op_{m}_allowed_adj"] = def_now.get(opp, np.nan)
            r[f"mx_op_{m}_allowed_prev"] = def_prev.get(opp, np.nan)
        r["mx_pass_edge"] = r["mx_tm_pass_epa_adj"] + r["mx_op_pass_epa_allowed_adj"]
        r["mx_rush_edge"] = r["mx_tm_rush_epa_adj"] + r["mx_op_rush_epa_allowed_adj"]
        r["mx_pass_minus_rush"] = r["mx_op_pass_epa_allowed_adj"] - r["mx_op_rush_epa_allowed_adj"]
        rows.append(r)
    feats = pd.DataFrame(rows, index=team_weeks.index)
    return pd.concat([team_weeks[["season", "week", "team", "opponent"]], feats], axis=1)
```

- [ ] **Step 4: Run tests** — `.venv/bin/python -m pytest -q tests/nfl/test_unit_efficiency.py` → PASS. Note: `adjusted_efficiency`'s window for (s, 99) includes the whole season s; for week 1 of season s the current window is empty → `{}` → NaN (as asserted).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/nfl/unit_efficiency.py tests/nfl/test_unit_efficiency.py
git commit -m "feat(features): opponent-adjusted pass/run unit efficiency (mx_)"
```

---

### Task 2: Defensive injuries by unit (`di_` features)

**Files:**
- Create: `src/sportsmodel/nfl/def_injuries.py`
- Test: `tests/nfl/test_def_injuries.py`

**Interfaces:**
- Produces:
  - `DEF_GROUPS: dict[str, str]` — position → `"cov" | "rush" | "run"` (R6)
  - `defender_snaps(snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> pd.DataFrame` — REG rows with `defense_snaps > 0`: `player_id, season, week, team, group, share` (`share` = `defense_pct` as a 0–1 fraction; if the source is 0–100, divide by 100)
  - `vacated_by_defense(dsnaps: pd.DataFrame, injuries: pd.DataFrame | None, team_weeks: pd.DataFrame) -> pd.DataFrame` — keys `season, week, team` (the DEFENSE) + `di_vacated_cov`, `di_vacated_rush`, `di_vacated_run`: summed as-of snap-share EWM (halflife 4 games, the player's games strictly before (S, w) for that team) of that team's Out/Doubtful defenders at (S, w); a (S, w) with no injury rows at all → NaN; with rows but none of this team's defenders → 0.0.

- [ ] **Step 1: Write the failing tests**

```python
# tests/nfl/test_def_injuries.py
import numpy as np
import pandas as pd

from sportsmodel.nfl.def_injuries import DEF_GROUPS, defender_snaps, vacated_by_defense


def _snaps():
    rows = []
    for wk in (1, 2, 3):
        rows += [
            {"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "cb1", "position": "CB",
             "team": "BUF", "opponent": "KC", "defense_snaps": 60, "defense_pct": 1.0},
            {"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "de1", "position": "DE",
             "team": "BUF", "opponent": "KC", "defense_snaps": 30, "defense_pct": 0.5},
            {"season": 2024, "week": wk, "game_type": "REG", "pfr_player_id": "lb1", "position": "LB",
             "team": "BUF", "opponent": "KC", "defense_snaps": 45, "defense_pct": 0.75},
        ]
    return pd.DataFrame(rows)


IDS = {"cb1": "00-CB", "de1": "00-DE", "lb1": "00-LB"}


def test_groups():
    assert DEF_GROUPS["CB"] == "cov" and DEF_GROUPS["DE"] == "rush" and DEF_GROUPS["LB"] == "run"


def test_vacated_sums_out_defenders_prior_share_by_group():
    ds = defender_snaps(_snaps(), IDS)
    inj = pd.DataFrame({"season": [2024, 2024], "week": [4, 4], "team": ["BUF", "BUF"],
                        "gsis_id": ["00-CB", "00-DE"], "report_status": ["Out", "Questionable"]})
    tw = pd.DataFrame({"season": [2024], "week": [4], "team": ["BUF"]})
    v = vacated_by_defense(ds, inj, tw).iloc[0]
    assert np.isclose(v.di_vacated_cov, 1.0) and v.di_vacated_rush == 0.0 and v.di_vacated_run == 0.0


def test_no_report_week_is_nan_and_own_week_snaps_never_count():
    ds = defender_snaps(_snaps(), IDS)
    tw = pd.DataFrame({"season": [2024, 2024], "week": [2, 5], "team": ["BUF", "BUF"]})
    inj = pd.DataFrame({"season": [2024], "week": [2], "team": ["BUF"],
                        "gsis_id": ["00-LB"], "report_status": ["Doubtful"]})
    v = vacated_by_defense(ds, inj, tw)
    w2 = v[v.week == 2].iloc[0]
    assert np.isclose(w2.di_vacated_run, 0.75)      # only week-1 snaps are before week 2
    assert np.isnan(v[v.week == 5].iloc[0].di_vacated_cov)
```

- [ ] **Step 2: Run to verify failure** — `.venv/bin/python -m pytest -q tests/nfl/test_def_injuries.py` → FAIL (module missing).

- [ ] **Step 3: Implement**

```python
# src/sportsmodel/nfl/def_injuries.py
"""Opponent defensive injuries by unit (the `di_` features). PURE.

A defender's weight is his recent share of his team's defensive snaps (EWM,
halflife 4 of his games strictly before the target week, for that team). A
defense's `di_vacated_<group>` at (S, w) sums the weights of its Out/Doubtful
defenders in that group on the (S, w) report. Groups (plan ruling R6):
coverage, pass rush, run defense; each defender counts in his primary
(most frequent) snap position's group.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from sportsmodel.nfl.teams import normalize_team

DEF_GROUPS: dict[str, str] = {
    "CB": "cov", "S": "cov", "FS": "cov", "SS": "cov", "DB": "cov",
    "DE": "rush", "OLB": "rush", "EDGE": "rush",
    "DT": "run", "NT": "run", "DL": "run", "LB": "run", "ILB": "run", "MLB": "run",
}
GROUPS = ("cov", "rush", "run")
_HALFLIFE = 4.0
_KEYS = ["season", "week", "team"]


def _norm(code):
    try:
        return normalize_team(str(code))
    except ValueError:
        return None


def _ord(df: pd.DataFrame) -> pd.Series:
    return df["season"].astype("int64") * 100 + df["week"].astype("int64")


def defender_snaps(snaps: pd.DataFrame, pfr2gsis: dict[str, str]) -> pd.DataFrame:
    s = snaps[(snaps["game_type"] == "REG") & (snaps["defense_snaps"] > 0)
              & snaps["position"].isin(list(DEF_GROUPS))].copy()
    s["player_id"] = s["pfr_player_id"].map(pfr2gsis)
    s["team"] = s["team"].map(_norm)
    s = s.dropna(subset=["player_id", "team"])
    share = s["defense_pct"].astype(float)
    s["share"] = np.where(share.max() > 1.0, share / 100.0, share)
    primary = s.groupby("player_id")["position"].agg(lambda x: x.value_counts().index[0])
    s["group"] = s["player_id"].map(primary).map(DEF_GROUPS)
    return s[["player_id", "season", "week", "team", "group", "share"]].astype(
        {"season": "int64", "week": "int64", "player_id": object})


def vacated_by_defense(dsnaps: pd.DataFrame, injuries: pd.DataFrame | None,
                       team_weeks: pd.DataFrame) -> pd.DataFrame:
    cols = [f"di_vacated_{g}" for g in GROUPS]
    out = team_weeks[_KEYS].copy()
    if injuries is None or not len(injuries):
        return out.assign(**{c: np.nan for c in cols})
    h = dsnaps.sort_values(["player_id", "season", "week"]).copy()
    h["_ewm"] = h.groupby("player_id")["share"].transform(
        lambda x: x.ewm(halflife=_HALFLIFE, ignore_na=True).mean())
    h = h.assign(_ord=_ord(h)).rename(columns={"team": "_hteam"})
    inj = injuries.dropna(subset=["gsis_id"]).assign(team=injuries["team"].map(_norm))
    o = inj[inj["report_status"].isin(["Out", "Doubtful"])].dropna(subset=["team"])
    o = o.rename(columns={"gsis_id": "player_id"})[["player_id", "season", "week", "team"]]
    o = o.astype({"season": "int64", "week": "int64", "player_id": object}).drop_duplicates()
    o = o.assign(_ord=_ord(o)).sort_values("_ord")
    m = pd.merge_asof(o, h[["player_id", "_ord", "_hteam", "group", "_ewm"]].sort_values("_ord"),
                      on="_ord", by="player_id", allow_exact_matches=False)
    m = m[(m["_hteam"] == m["team"]) & m["group"].notna()]
    tot = m.pivot_table(index=_KEYS, columns="group", values="_ewm", aggfunc="sum")
    tot = tot.reindex(columns=list(GROUPS)).add_prefix("di_vacated_").reset_index()
    out = out.merge(tot, on=_KEYS, how="left")
    reported = pd.MultiIndex.from_frame(out[["season", "week"]].astype("int64")).isin(
        pd.MultiIndex.from_frame(injuries[["season", "week"]].dropna().astype("int64").drop_duplicates()))
    out[cols] = out[cols].fillna(0.0)
    out.loc[~reported, cols] = np.nan
    return out
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Commit** — `git add src/sportsmodel/nfl/def_injuries.py tests/nfl/test_def_injuries.py && git commit -m "feat(features): opponent defensive injuries by unit (di_)"`

---

### Task 3: QB career profiles (`qb_` features)

**Files:**
- Create: `src/sportsmodel/nfl/qb_profile.py`
- Create: `assets/nfl/props_ml/qb_profile_params.json` (written by Step 6's tuning command)
- Create: `scripts/tune_qb_profile.py`
- Test: `tests/nfl/test_qb_profile.py`

**Interfaces:**
- Produces:
  - `RATES = ("ypa", "td_rate", "int_rate", "sack_rate")`
  - `qb_games(weekly: pd.DataFrame) -> pd.DataFrame` — REG rows with `attempts + sacks_suffered > 0`: `player_id, season, week, team, opponent, att, yds, tds, ints, sacks` + raw rates (`ypa = yds/att`, `td_rate = tds/att`, `int_rate = ints/att`, `sack_rate = sacks/(att+sacks)`; NaN where the denominator is 0).
  - `opponent_adjust(qg: pd.DataFrame) -> pd.DataFrame` — adds `<rate>_adj` = rate − (opponent's allowed rate − league rate), both attempt-weighted over that season's games strictly before the game's week; fewer than 100 attempts faced by the opponent in the window → use the opponent's previous-season allowed rate; nothing known → no adjustment.
  - `replacement(qga: pd.DataFrame, before_season: int) -> dict[str, float]` — attempt-weighted `<rate>_adj` of NON-regular QBs (not their team's season attempts leader) over seasons `< before_season`.
  - `profile_asof(qga, keys: pd.DataFrame, H: float, k: float, repl: dict) -> pd.DataFrame` — `keys` has `player_id, season, week`; returns keys + `qb_ypa, qb_td_rate, qb_int_rate, qb_sack_rate, qb_n_eff` from games strictly before (season, week): weight = att × 0.5^(age/H), age = (season + (week−1)/18) of the key minus that of the game; `shrunk = (n·raw + k·repl)/(n + k)`; a QB with no games → `repl`, `qb_n_eff = 0`.
  - `qb1_by_team_week(depth: pd.DataFrame, injuries: pd.DataFrame | None, team_weeks: pd.DataFrame, override: dict | None = None) -> pd.DataFrame` — `season, week, team, qb1_id`: lowest `depth_team` QB on the as-of chart (exact week else latest earlier, `usage.chart_weeks_asof`) not Out/Doubtful that week; `override[(season, week, team)] = gsis` wins.
  - `team_qb_features(team_weeks, qb1, qga, H, k, repl) -> pd.DataFrame` — `season, week, team` + `qb_ypa, qb_td_rate, qb_int_rate, qb_sack_rate, qb_n_eff` (QB1's profile), `qb_ratio_ypa, qb_ratio_td, qb_ratio_sack` (QB1 ÷ the attempt×0.5^(games_ago/4)-weighted mix of profiles, as of (S, w), of the QBs who threw the team's passes in its previous 10 games; clamped [0.6, 1.3]; NaN when the team has no prior games), `qb_changed` (1.0 if QB1 ≠ the team's previous game's attempts leader, 0.0 if equal, NaN if unknown).
  - `tune(qga, seasons: list[int], grid_h=(1, 2, 3, 4), grid_k=(100, 200, 400, 800)) -> dict` — for each (H, k): predict `ypa_adj` of every QB-game with att ≥ 10 in `seasons` from `profile_asof` at that game (repl from seasons before the game's season); score attempt-weighted MSE; return `{"H": .., "k": .., "mse": .., "grid": [...]}` (ties → first grid point).

- [ ] **Step 1: Write the failing tests**

```python
# tests/nfl/test_qb_profile.py
import numpy as np
import pandas as pd

from sportsmodel.nfl import qb_profile as qp


def _w(pid, season, week, team, opp, att, yds, tds=1, ints=0, sacks=2):
    return {"player_id": pid, "season": season, "week": week, "season_type": "REG",
            "recent_team": team, "opponent_team": opp, "position": "QB", "attempts": att,
            "passing_yards": yds, "passing_tds": tds, "passing_interceptions": ints,
            "sacks_suffered": sacks}


def _weekly():
    rows = []
    for yr in (2016, 2017, 2018):                       # veteran starter, then benched
        for wk in range(1, 17):
            rows.append(_w("VET", yr, wk, "MIN", "GB", 30, 210))
    for wk in range(1, 17):                              # 2019-2024: other starters, league avg 7.0
        for yr in range(2019, 2025):
            rows.append(_w(f"S{yr}", yr, wk, "MIN", "GB", 30, 210))
    rows.append(_w("ROOK", 2024, 16, "MIN", "GB", 5, 20, tds=0))   # thin backup
    return pd.DataFrame(rows)


def test_qb_games_rates():
    g = qp.qb_games(_weekly())
    r = g[g.player_id == "VET"].iloc[0]
    assert r.ypa == 7.0 and np.isclose(r.sack_rate, 2 / 32)


def test_profile_weights_long_career_and_shrinks_thin_one():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    keys = pd.DataFrame({"player_id": ["VET", "ROOK", "NEW"], "season": [2025] * 3, "week": [4] * 3})
    p = qp.profile_asof(qga, keys, H=2.0, k=200.0, repl=repl).set_index("player_id")
    # VET: 1,440 attempts but 7-9 seasons old -> n_eff ~ 90 at H=2 -> (90*7 + 200*5)/290 ~ 5.6
    assert 5.4 < p.loc["VET", "qb_ypa"] < 7.0
    # ROOK: 5 attempts at 4.0 ypa -> (5*4 + 200*5)/205 ~ 4.98: essentially replacement
    assert 4.9 < p.loc["ROOK", "qb_ypa"] < 5.0
    # a longer half-life keeps more of the veteran's own record
    p4 = qp.profile_asof(qga, keys, H=4.0, k=200.0, repl=repl).set_index("player_id")
    assert p4.loc["VET", "qb_ypa"] > p.loc["VET", "qb_ypa"]
    assert p.loc["NEW", "qb_ypa"] == 5.0 and p.loc["NEW", "qb_n_eff"] == 0.0


def test_profile_is_strictly_prior():
    w = _weekly()
    qga = qp.opponent_adjust(qp.qb_games(w))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    keys = pd.DataFrame({"player_id": ["VET"], "season": [2017], "week": [5]})
    a = qp.profile_asof(qga, keys, 2.0, 200.0, repl)
    w2 = pd.concat([w, pd.DataFrame([_w("VET", 2017, 5, "MIN", "GB", 40, 900)])], ignore_index=True)
    b = qp.profile_asof(qp.opponent_adjust(qp.qb_games(w2)), keys, 2.0, 200.0, repl)
    pd.testing.assert_frame_equal(a, b)


def test_qb1_skips_out_and_honors_override():
    depth = pd.DataFrame({"season": [2024] * 3, "week": [5] * 3, "club_code": ["CHI"] * 3,
                          "position": ["QB"] * 3, "gsis_id": ["CW", "TB", "CK"], "depth_team": [1, 2, 3]})
    inj = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"], "gsis_id": ["CW"],
                        "report_status": ["Out"]})
    tw = pd.DataFrame({"season": [2024], "week": [5], "team": ["CHI"]})
    assert qp.qb1_by_team_week(depth, inj, tw).iloc[0].qb1_id == "TB"
    assert qp.qb1_by_team_week(depth, inj, tw, override={(2024, 5, "CHI"): "CK"}).iloc[0].qb1_id == "CK"


def test_ratio_is_one_for_the_usual_starter_and_below_one_for_a_weaker_backup():
    qga = qp.opponent_adjust(qp.qb_games(_weekly()))
    repl = {"ypa_adj": 5.0, "td_rate_adj": 0.02, "int_rate_adj": 0.03, "sack_rate_adj": 0.08}
    tw = pd.DataFrame({"season": [2024, 2024], "week": [10, 10], "team": ["MIN", "MIN"],
                       "opponent": ["GB", "GB"]})
    qb1 = pd.DataFrame({"season": [2024, 2024], "week": [10, 10], "team": ["MIN", "MIN"],
                        "qb1_id": ["S2024", "ROOK"]})
    same = qp.team_qb_features(tw.iloc[:1], qb1.iloc[:1], qga, 2.0, 200.0, repl).iloc[0]
    back = qp.team_qb_features(tw.iloc[1:], qb1.iloc[1:], qga, 2.0, 200.0, repl).iloc[0]
    assert np.isclose(same.qb_ratio_ypa, 1.0, atol=0.02) and same.qb_changed == 0.0
    assert back.qb_ratio_ypa < 0.9 and back.qb_changed == 1.0
```

- [ ] **Step 2: Run to verify failure** → FAIL (module missing).

- [ ] **Step 3: Implement `src/sportsmodel/nfl/qb_profile.py`** exactly to the Interfaces above. Implementation notes (binding):
  - `qb_games`: filter `season_type == "REG"` and `position == "QB"`; `team = normalize_team(recent_team)`, `opponent = normalize_team(opponent_team)`; drop unmappable teams.
  - `opponent_adjust`: for each (season, week) compute, per defense, attempt-weighted allowed sums from rows with `week < w` of that season (cumulative sums by defense ordered by week, shifted so the game's own week is excluded); league = all defenses' pooled window; previous season's full allowed per defense for the < 100-attempt case. Vectorize by season (≤ 18 weeks × 32 defenses) — no per-row Python loops over all rows.
  - `profile_asof`: group `qga` by `player_id` once; for each key row use only that player's games with `_ord < key_ord` (`_ord = season*100 + week`); `n_eff = Σ w` where `w = att × 0.5**(age/H)`; each `qb_<rate>` = `(Σ w·rate_adj + k·repl[rate_adj]) / (n_eff + k)`, rates with NaN `rate_adj` skipped in both sums for that rate.
  - `team_qb_features`: previous-10-game passer mix from `qga` rows of `team` strictly before (S, w); `games_ago` = 1 for the latest game; profiles for the mix QBs via `profile_asof` at (S, w).
  - `replacement`: regular = the QB with the most attempts for (season, team).

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Tuning script** — create `scripts/tune_qb_profile.py`:

```python
"""Choose the QB-profile weighting (H, k) by next-game prediction (plan R3).

  uv run python scripts/tune_qb_profile.py --mode gate      # fit seasons 2021-2023
  uv run python scripts/tune_qb_profile.py --mode serving   # fit 2021 -> last completed week

Loads nflverse weekly stats 1999..current (load_release("weekly", ...)), runs
qb_profile.tune, and writes assets/nfl/props_ml/qb_profile_params.json
{"gate": {...}, "serving": {...}} (the other mode's block is kept).
"""
```
  `main()` parses `--mode`, loads weekly 1999..`nfl_season(now)`, calls `qb_profile.tune(opponent_adjust(qb_games(w)), seasons)` with `seasons = [2021, 2022, 2023]` (gate) or `list(range(2021, current + 1))` (serving; only labelled weeks are scored), merges into the json with `fit_seasons`, `created_at`, `git`. Test in `tests/nfl/test_qb_profile.py`: `tune` on the fixture returns a grid point and a finite mse, and ties keep the first grid point (pass a one-point grid twice).

- [ ] **Step 6: Run the gate tuning** — `.venv/bin/python scripts/tune_qb_profile.py --mode gate`; confirm the json has a `gate` block.

- [ ] **Step 7: Commit** — `git add src/sportsmodel/nfl/qb_profile.py scripts/tune_qb_profile.py tests/nfl/test_qb_profile.py assets/nfl/props_ml/qb_profile_params.json && git commit -m "feat(features): QB career profiles, opponent-adjusted (qb_)"`

---

### Task 4: Wire the new features into the one builder

**Files:**
- Modify: `src/sportsmodel/nfl/player_features.py` (`team_games`, `build_team_table`, `build_feature_table`, `_FEATURE_PREFIXES`)
- Modify: `scripts/build_player_features.py` (`_PBP_COLS`, `fetch_sources`, `build_tables`, `FEATURE_GROUPS`)
- Test: `tests/nfl/test_player_features.py`, `tests/scripts/test_build_player_features.py`

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces:
  - `build_team_table(tg, ctx, game_epa, extra: pd.DataFrame | None = None)` — `extra` (keys `season, week, team`) holds `mx_`/`di_`/`qb_` columns merged into the team table.
  - `build_feature_table(..., stubs=None, extra=None)` — same `extra` merged onto every player row by `(season, week, team)`.
  - `team_games` adds label column `off_tds` (sum of `pass_touchdown + rush_touchdown` for `posteam`); `build_team_table` exposes it as `y_team_off_tds`.
  - `build_tables(src, ctx_fill=None, qb1_override: dict | None = None)` builds `extra` = unit features (opponent side from `unit_features`) + `di_op_vacated_<g>` (the OPPONENT's `di_vacated_<g>`, i.e. `vacated_by_defense` re-keyed from defense to offense via the team-week's opponent) + `team_qb_features` with the H, k from `qb_profile_params.json` (`serving` block when `src["qb_params_mode"] == "serving"`, else `gate`).
  - `fetch_sources` adds `"weekly_qb"`: `load_release("weekly", list(range(1999, max(seasons) + 1)))` filtered to QB rows, and `"qb_params_mode"` (default `"gate"`).

- [ ] **Step 1: Failing tests** — add to `tests/nfl/test_player_features.py`:

```python
def test_extra_columns_join_every_player_row_and_only_new_prefixes():
    inp = _inputs()                     # the existing scripted fixture used by _build
    tw = inp["tg"][["season", "week", "team"]].drop_duplicates()
    extra = tw.assign(mx_pass_edge=0.1, di_op_vacated_cov=0.0, qb_ypa=7.0)
    out = _build({**inp, "extra": extra})
    assert {"mx_pass_edge", "di_op_vacated_cov", "qb_ypa"} <= set(out.columns)
    assert out["mx_pass_edge"].notna().all()
    before = set(_build(inp).columns)
    assert set(out.columns) - before == {"mx_pass_edge", "di_op_vacated_cov", "qb_ypa"}
```
  (Adapt `_build`/fixture names to the file's existing helpers — `_build(inp)` at line 113; extend it to pass `extra=inp.get("extra")`.) Add a `team_games` test that `off_tds` counts pass + rush TDs per offense. In `tests/scripts/test_build_player_features.py` add a `build_tables` test with a stubbed `src` (tiny pbp incl. `yards_gained, success, pass_touchdown, rush_touchdown`, snaps with defenders, weekly_qb) asserting `mx_`, `di_op_vacated_cov`, `qb_ypa` columns exist and that `qb1_override` changes `qb_ypa` for that team-week only.

- [ ] **Step 2: Run** → FAIL.
- [ ] **Step 3: Implement** — `_PBP_COLS` += `"yards_gained", "success", "pass_touchdown", "rush_touchdown"`; `_FEATURE_PREFIXES` and `FEATURE_GROUPS` += `"mx_", "di_", "qb_"`; merges as specified; `y_team_off_tds` added to the team table's label columns (NOT to `_TM_COLS` — no new `tm_` features, Global Constraint). The existing leakage perturbation test (`test_perturbing_target_week_and_later_does_not_change_features`) must cover the new columns: extend its fixture with the new pbp columns and snaps/weekly_qb.
- [ ] **Step 4: Run full suite** → green.
- [ ] **Step 5: Build the tables** — `.venv/bin/python scripts/build_player_features.py`; paste the NaN share line for `mx_`, `di_`, `qb_` into the report file.
- [ ] **Step 6: Commit** — `git commit -m "feat(features): wire mx_/di_/qb_ into the feature builder"`

---

### Task 5: Learned rungs, drive-TD shift and the stale-QB role

**Files:**
- Modify: `src/sportsmodel/sim/nfl/learned.py`
- Modify: `src/sportsmodel/model/props_ml/dist_models.py`
- Test: `tests/sim/nfl/test_learned.py` (or the existing learned test file — `grep -l fit_models tests -r`), `tests/model/props_ml/test_dist_models.py` (same rule)

**Interfaces:**
- Produces:
  - `RUNG_PREFIXES` += `"matchup": ("mx_",), "def_injuries": ("di_",), "qb_profile": ("qb_",)`; `feature_columns` includes a rung's prefixes only when the rung is toggled (the three new rungs, like `context`/`market`).
  - `LearnedModels.td_elasticity: float | None = None` — set by `fit_models` when `"qb_profile" in toggles`: no-intercept OLS slope of `y = y_team_off_tds / trailing − 1` on `x = qb_ratio_ypa − 1` over team rows before `upto` with finite x, y and `trailing > 0`, where `trailing` = the team's EWM (halflife 4) of `y_team_off_tds` over its games strictly before; None when < 50 such rows.
  - `apply_to_spec` (unchanged signature): when `models.td_elasticity` is not None and the team row has finite `qb_ratio_ypa`, the team's `drive_outcomes["td"]` is multiplied by `m = clip(1 + e·(r − 1), 0.7, 1.3)` and the TD mass change is taken from / given to `fg` and `punt` in proportion to their current mass (outcomes still sum to 1).
  - `dist_models._QB_ROLE` requires `p_y_pass_att_r10 > 0` in addition to `p_y_pass_att_ewm >= 10` (`in_role` supports a list of `(col, min, strict)` conditions; keep the existing single-condition roles working).

- [ ] **Step 1: Failing tests**

```python
def test_new_rungs_select_only_their_prefix():
    df = pd.DataFrame(columns=["p_a", "tm_b", "mx_c", "di_d", "qb_e", "cx_f", "y_x"])
    assert learned.feature_columns(df, frozenset({"volume"})) == ["p_a", "tm_b"]
    assert learned.feature_columns(df, frozenset({"volume", "matchup"})) == ["p_a", "tm_b", "mx_c"]
    assert "qb_e" in learned.feature_columns(df, frozenset({"volume", "qb_profile"}))


def test_drive_td_shift_keeps_outcomes_summing_to_one():
    rates = TeamRates(drive_outcomes={"td": 0.25, "fg": 0.15, "punt": 0.4, "turnover": 0.12,
                                      "downs": 0.05, "end": 0.03}, pass_rate=0.6,
                      drives_per_game=11, rz_td_rate=0.55)
    out = learned._shift_td(rates, e=1.0, ratio=0.8)
    assert np.isclose(sum(out.drive_outcomes.values()), 1.0)
    assert np.isclose(out.drive_outcomes["td"], 0.25 * 0.8)
    assert out.drive_outcomes["fg"] > 0.15 and out.drive_outcomes["punt"] > 0.4
    assert out.drive_outcomes["turnover"] == 0.12


def test_stale_qb_is_out_of_role():
    df = pd.DataFrame({"position": ["QB", "QB"], "p_y_pass_att_ewm": [15.7, 30.0],
                       "p_y_pass_att_r10": [np.nan, 28.0]})
    assert dist_models.in_role(df, "pass_yds").tolist() == [False, True]
```
  Plus a `fit_models` test that `td_elasticity` recovers a planted slope (synthetic team rows where `y_team_off_tds = trailing × (1 + 0.8·(r − 1))`) within 0.05, and is None without `qb_profile`.

- [ ] **Step 2: Run** → FAIL. **Step 3: Implement** (`_shift_td(rates, e, ratio) -> TeamRates` is the pure helper used by `_team_rates`). **Step 4: Full suite** → green (existing v1 tests must still pass: v1 toggles never include the new rungs). **Step 5: Commit** `feat(learned): matchup/def_injuries/qb_profile rungs, QB drive-TD shift, stale-QB role`.

---

### Task 6: A ladder — new rungs, decisions on 2021–23

**Files:**
- Modify: `scripts/train_props_ml.py`
- Test: `tests/scripts/test_train_props_ml.py`

**Interfaces:**
- Produces:
  - `LADDER = ("volume", "efficiency", "context", "market", "matchup", "def_injuries", "qb_profile")`.
  - Env `PROPS_ML_DECIDE_SEASONS` (comma list; default = all run seasons): `rung_decision` / `baseline_ece_check` in the ladder use only records of these seasons; the final section reports kept-vs-baseline on all seasons, on the decide seasons, and on each other season separately.
  - Env `PROPS_ML_GATE_NAME` (default `""`): suffix for outputs — `a_gate{name}.json`, report `...-props-ml-a-gate{name}.md`, checkpoints `records_<run>__<tag>{name}.parquet` — so the v2 run never overwrites v1 files.
  - Gate json gains `"decide_seasons"`.

- [ ] **Step 1: Failing tests** in `tests/scripts/test_train_props_ml.py` using the existing stubbed-backtest `run_harness` pattern: (a) with `PROPS_ML_DECIDE_SEASONS=2021` and records where a rung helps in 2021 but hurts in 2022, the rung is kept; (b) `PROPS_ML_GATE_NAME=_v2` writes `a_gate_v2.json` and leaves `a_gate.json` untouched; (c) `toggles_from_env` accepts `qb_profile`.
- [ ] **Step 2–4:** implement, run, full suite green.
- [ ] **Step 5: Commit** `feat(props-ml): ladder rungs matchup/def_injuries/qb_profile; decide seasons; gate name`.

---

### Task 7: B ladder per version, served composites, and the v2 gate

**Files:**
- Modify: `scripts/train_props_ml_b.py`
- Create: `src/sportsmodel/model/props_ml/v2_checks.py`
- Create: `scripts/gate_props_ml_v2.py`
- Test: `tests/scripts/test_train_props_ml_b.py`, `tests/model/props_ml/test_v2_checks.py`, `tests/scripts/test_gate_props_ml_v2.py`

**Interfaces:**
- `train_props_ml_b.run_b_ladder` honors `PROPS_ML_GATE_NAME` (reads `a_gate{name}.json`; writes `b_gate{name}.json`, `pipeline{name}.json`, `calibration{name}.json`, checkpoints/OOF with the suffix) and ALSO writes the final served composite's post-calibration records to `data/props_ml/served__<tag>{name}.parquet` (columns `season, week, home, player_id, market, mean, rps, pit, actual`, one row per population key).
- `v2_checks.py` (PURE):
  - `qb_change_mask(records: pd.DataFrame, feats: pd.DataFrame) -> pd.Series` — True where the record's team-week has `qb_changed == 1` or `qb_ratio_ypa` outside [0.95, 1.05] (join on `season, week, player_id` → `team`, then team-week).
  - `player_baseline(records, feats, n=8) -> pd.Series` — the player's mean actual of that market over his previous `n` played games (labels `y_pass_yds`, `y_rush_yds`, `y_rec_yds`, `y_receptions` …; market→label map), NaN with < 3 games.
  - `matchup_response(v1: pd.DataFrame, v2: pd.DataFrame, feats: pd.DataFrame) -> dict` — quintiles of the OPPONENT defense's `mx_pass_minus_rush` (as of the record's week); for groups RB×rush_yds, QB×pass_yds, WR×rec_yds, TE×rec_yds: per quintile `pred_dev = Σmean/Σbaseline − 1` (v1 and v2) and `act_dev = Σactual/Σbaseline − 1`; returns the table plus `pass_top` (top quintile: v2 RB dev < 0 and v2 QB/WR/TE dev > 0, and each sign equals act_dev's sign), `pass_bottom` (the reverse), `corr_v1`, `corr_v2` (Pearson over the 20 group×quintile cells of pred_dev vs act_dev) and `pass = pass_top and pass_bottom and corr_v2 > corr_v1`.
  - `verdict(paired_rps_df, ece_v1, ece_v2, game_diffs, qb_change_ci, matchup) -> dict` — the spec §6 ship rule.
- `scripts/gate_props_ml_v2.py`: loads `served__<tag>.parquet` (v1) and `served__<tag>_v2.parquet`, the player table, and game-level records from two `backtest_sim_nfl.run_backtest` runs with the kept A of `a_gate.json` vs `a_gate_v2.json` (reuse `scripts/gate_ml_game_lines.py`'s ML-side runner + `game_gate.game_metrics` / `paired_diffs` / `gate_decision`); verdict seasons 2024–2025 (and 2025 alone); writes `assets/nfl/props_ml/v2_gate.json` and `docs/superpowers/reports/<date>-matchup-qb-gate.md` including the matchup table and a Keenum 2026-09-28 spot-check row when graded.

- [ ] **Step 1: Failing tests** — `test_v2_checks.py`: synthetic records where the top quintile's actual RB rush is 20 % under baseline and v2 predicts −15 % while v1 predicts 0 % → `pass_top` True, `corr_v2 > corr_v1`; a mirrored bottom quintile → `pass_bottom` True; flip v2's RB sign → `pass` False. `qb_change_mask` picks exactly the planted team-weeks. `test_train_props_ml_b.py`: gate name routes every output path; served file written with the listed columns. `test_gate_props_ml_v2.py`: verdict fails when 2025 alone is worse even if pooled is better.
- [ ] **Step 2–4:** implement, run, full suite green.
- [ ] **Step 5: Commit** `feat(props-ml): per-version B ladder, served composites, v2 gate with QB-change and two-way matchup checks`.

---

### Task 8: Serve by version (v1 ⇄ v2), live QB override, site label

**Files:**
- Modify: `scripts/generate_sim_nfl.py`, `scripts/fit_props_ml_final.py`, `src/sportsmodel/model/props_ml/artifacts.py`
- Modify: `.github/workflows/generate-sim-nfl.yml`, `.github/workflows/injury-watch.yml` (download steps), `.github/workflows/train-props-ml.yml` (publish tag per version)
- Modify (external, not git): `/Users/ryan/Desktop/CappingAlpha/app.js` (label), all `*.html` cache key → `20260929a`
- Test: `tests/sim/nfl/test_generate_sim_nfl.py`, `tests/scripts/test_fit_props_ml_final.py`, `tests/test_workflows_props_ml.py`

**Interfaces:**
- `ML_VERSIONS = ("nfl-sim-ml-v1", "nfl-sim-ml-v2")`; `ml_model_dir(version) -> Path` = `data/props_ml/models/<version>/`. The served version (from `nfl_sim_serving`) selects the artifacts dir; every place that compared `served == ML_MODEL_VERSION` now checks `served in ML_VERSIONS` and writes rows / game_predictions under the SERVED version string. v1 artifacts stay usable (their config has no `model_version` → treated as v1).
- `fit_props_ml_final.py --gate-name _v2` reads `pipeline_v2.json`/`calibration_v2.json`/`a_gate_v2.json`, re-tunes nothing, writes `props_ml_config.json` with `"model_version": "nfl-sim-ml-v2"`, `"qb_profile_params": <serving block>` into `data/props_ml/models/nfl-sim-ml-v2/`.
- Workflows download release `props-ml-latest` → `models/nfl-sim-ml-v1/` (unchanged release) and `props-ml-v2` → `models/nfl-sim-ml-v2/` (missing release → warning, v2 disabled), each manifest-verified as today.
- Live: `generate_sim_nfl` passes `qb1_override = {(season, week, team): promoted gsis}` from the books QB check into `_build_ml_tables` → `build_tables`, and uses the `serving` QB params; logs one `matchup:` line per team (QB1, `qb_ypa`, `qb_ratio_ypa`, `mx_pass_minus_rush`, `di_op_vacated_*`).
- Site: `mlModelTag(v)` → `v.startsWith("nfl-sim-ml-")` → `Model: ML v<N>` from the suffix.
- Live defensive injuries: `_inject_live_injuries` → `injury_report.live_injury_rows` maps live names to gsis through the target week's depth chart. Verify defenders map (the as-of chart must keep Defense-formation rows; add a test with a live CB on the report). If defenders are filtered out anywhere on that path, extend the mapping to all positions so `di_op_vacated_*` is not 0 for the live week by construction.

- [ ] **Step 1: Failing tests** — served v2 with only v1 artifacts present → ML skipped with a `::warning::` and the v1 copy rule unchanged; served v2 with v2 artifacts → rows written under `nfl-sim-ml-v2` and game_predictions `model_version == "nfl-sim-ml-v2"`; the books override reaches `build_tables` (spy); workflow test asserts both download targets and manifest checks.
- [ ] **Step 2–4:** implement, run, full suite green; `node --check` the site app.js.
- [ ] **Step 5: Commit** `feat(serving): props-ML by served version (v1/v2), live QB override, matchup log`.

---

### Task 9: Run the gate (controller, long compute) and report

- [ ] **Step 1:** `.venv/bin/python scripts/build_player_features.py` (fresh tables).
- [ ] **Step 2:** A ladder v2: `PROPS_ML_GATE_NAME=_v2 PROPS_ML_DECIDE_SEASONS=2021,2022,2023 .venv/bin/python scripts/train_props_ml.py` (background; hours; `PROPS_ML_RESUME=1` on restart).
- [ ] **Step 3:** B ladder v1 (served composite for the comparison) and v2: `.venv/bin/python scripts/train_props_ml_b.py` then `PROPS_ML_GATE_NAME=_v2 .venv/bin/python scripts/train_props_ml_b.py`.
- [ ] **Step 4:** `.venv/bin/python scripts/gate_props_ml_v2.py`; commit `v2_gate.json` + report.
- [ ] **Step 5:** Report the verdict to the user with the numbers (pass or fail). ONLY on pass and the user's go-ahead: `scripts/tune_qb_profile.py --mode serving`, `fit_props_ml_final.py --gate-name _v2`, quick gate, publish release `props-ml-v2`, user redeploys the site, user runs `UPDATE nfl_sim_serving SET model_version='nfl-sim-ml-v2'`, controller dispatches generate-sim-nfl and verifies rows/labels.
