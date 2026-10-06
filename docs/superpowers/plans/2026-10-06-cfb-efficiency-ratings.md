# CFB Efficiency Ratings (cfb-ratings-v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add opponent-adjusted per-play efficiency ratings and game-day context from CFBD to the live margin/Elo + priors CFB model, and ship it as `cfb-ratings-v3` (behind the `CFB_MODEL_VERSION` repo variable, default `v2`) only if it beats v2 on held-out 2023-25.

**Architecture:** New committed CFBD parquet assets (per-game havoc, drives, weather, talent, venues, game meta, postseason advanced stats) feed one leak-free walk-forward feature table (`v3_table.build_table`, built on the existing `walkforward.walk`) that carries v2's own margin/total, a decaying prior margin, efficiency "side features" (weekly weighted ridge ratings with a previous-season prior) and context features. Weights are fitted on 2016-22 (`v3_fit`), judged on 2023-25 by a pure gate (`v3_gate`) that first reproduces v2's published numbers, and served by `generate_cfb.py` only when `CFB_MODEL_VERSION=v3`.

**Tech Stack:** Python 3.11+ (uv), pandas, numpy, scikit-learn (Lasso / Ridge / HistGradientBoosting), httpx + tenacity, pytest, GitHub Actions. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-06-cfb-efficiency-ratings-design.md` (the authority; executors read it alongside this plan).

## Global Constraints

Every task's requirements implicitly include this section. Values are copied from the spec.

- **Key:** the CFBD key (paid tier, Tier 3 or higher) stays in repo secret `CFBD_API_KEY`; code reads it from the environment only, never logs it, never puts it in a URL or a file. 75,000 calls/month budget; full backfill is a few hundred calls, weekly refresh ~20-40 calls/week + game-day weather (about 1 call per run).
- **Data pulls:** every pull caches to committed parquet under `assets/cfb/`, maps CFBD team names to ESPN ids with the existing `cfbd_to_espn` (rows whose team does not map to an FBS id are dropped and counted in `df.attrs["dropped"]`), and stamps every row with `season`, `week`, `season_type`, `game_id`. Missing / null numbers are NaN, never 0.
- **Leak rule:** a game's features for week W use only games completed before week W (the invariant of `walkforward.py`: appending a later game never changes an earlier game's features). Season-level CFBD aggregates computed at season end (e.g. `/ratings/sp`, `/ratings/fpi`, season advanced stats) are NEVER used for in-season games of that season.
- **Model-only:** market data (lines, spreads, CFBD pregame win probability) is never an input to v3's predictions (as v2). Closing lines are used only to grade.
- **Efficiency ratings:** metrics PPA (overall, rush, pass), success rate, explosiveness, havoc rate, points per scoring opportunity; one ridge regression per metric per week over all season-to-date FBS-vs-FBS team-games (FCS games excluded), `metric(off team, def team) = league_mean + off_adj[team] - def_adj[opp] + hfa*home`, weighted by plays, ridge penalty fitted in the walk-forward; early season = previous-season final adjusted rating shrunk toward the league mean by returning production and talent, moving to the season-to-date estimate as games accumulate.
- **Blend (v3):** `margin_v3 = a1*margin_v2_ratings + a2*margin_eff + a3*prior_margin(decaying) + hfa + sum(context_m)`; `total_v3 = b1*total_eff + b2*total_v2 + sum(context_t)`. Context (additive, each fitted, each can be fitted to 0): weather on totals (wind >= ~15 mph, temperature < ~40 F, precipitation; dome = 0), travel (distance bucket, >= 2 time zones east/west), rest (short week, bye week before the game), talent gap (early season only, decays with games played). Weights and context sizes fitted on 2016-22 in the walk-forward. Moneyline: margin -> win probability through the existing calibrated mapping, refit for v3 (`gameline_v3.json`).
- **Gate (verdict on 2023-25 combined):** margin MAE < v2; total MAE < v2; ATS vs closing spread >= v2; O/U vs closing total >= v2; ML log-loss <= v2 and ECE <= v2 + 0.005. Per season (2023, 2024, 2025) reported and any season where v3 is worse is flagged; the verdict uses the combined numbers. **v2 baseline reproduced FIRST: margin MAE 12.61, total MAE 13.10, ATS 49.0% on held-out 2023-25 within rounding; if not, STOP and reconcile before comparing.**
- **Residual check (report only):** ridge and a shallow gradient-boosted model on v3's 2016-22 residuals (margin and total) using features v3 does not use; report out-of-sample R^2, MAE change on 2023-25 and top features; a feature that reduces held-out MAE in all three seasons is a v4 candidate. Nothing from it ships.
- **Serving:** `cfb-ratings-v3` is selected by repo variable `CFB_MODEL_VERSION` (default `v2`; switching back = one variable change); committed weights in `assets/cfb/v3_weights.json`; gate output `assets/cfb/v3_gate.json`. Weekly Monday job (`build-cfb-advanced.yml`, 12:00 UTC) also refreshes postseason advanced stats, havoc and drives and recomputes efficiency ratings; a game-day weather pull runs before the daily CFB generate; weather missing for a game -> that game's weather adjustment = 0 (never fabricated). Track record continues across the version change (no restart). Site unchanged.
- **The live model stays v2.** No task in this plan sets or changes the `CFB_MODEL_VERSION` repo variable or its default. Flipping to v3 is the user's decision after reading the gate report.
- **Splits:** fit/tune on 2016-2022 (2015 only warms Elo / has no previous-season prior); held-out 2023-2025 is only ever reported / gated, nothing is selected on it.
- **Testing:** unit tests are network-free (inline fixtures); existing CFB tests stay green; every task ends with `uv run pytest -q <touched tests>` green.
- **Out of scope:** site panels (project 2), live in-game model / GraphQL (project 3), CFB player props, changing the +EV board logic, restarting the CFB track record, market-informed predictions.
- **Conventions:** run everything from the repo root (`/Users/ryan/Desktop/Sports Model` - the path contains a space, quote it); `uv run pytest`, `uv run python scripts/...`; commit messages `feat(cfb): ...` / `data(cfb): ...` / `test(cfb): ...`, ending with the Co-Authored-By trailer the executing session is configured with; subagents and coding run on Sonnet (pass `model: "sonnet"`), per the user's model-routing rule.
- **Network / data steps** are marked **[NETWORK]** or **[LOCAL RUN]** below: they are run by the executing agent itself (not unit tests), need `CFBD_API_KEY` (network ones) and end by committing the produced assets. Load the key without printing it: `set -a; source .env; set +a` (or export it from the repo secret store); never `echo` it.

---

## File Structure

New code (all under `src/sportsmodel/cfb/` unless noted), one responsibility each:

| File | Responsibility |
|---|---|
| `cfbd.py` (modify) | add `CfbdClient`: Bearer auth, retry/backoff, per-run call counter |
| `cfbd_games.py` (new) | pure parsers: games meta, havoc, drives, weather, talent, venues, prior ratings |
| `data_checks.py` (new) | pure sanity checks of backfilled assets (units, havoc sides, coverage) |
| `efficiency.py` (new) | weighted ridge adjustment, prior, blend, `state_before`, matchup `side_features`, `build_eff_games` |
| `eff_fit.py` (new) | walk-forward fit of the efficiency hyperparameters |
| `context.py` (new) | weather / travel / time-zone / rest / talent-gap features |
| `v3.py` (new) | `LinearBlend`, `V3Weights`, `predict_v3`, gameline config loader |
| `v3_table.py` (new) | the leak-free feature table + live frame (shared by fit, gate, serving) |
| `v3_fit.py` (new) | points map, margin/total blends (lasso context), sigmas |
| `v3_gate.py` (new) | pure gate metrics, baseline check, verdict |
| `residual.py` (new) | report-only residual check |
| `v3_data.py` (new) | committed-asset loaders (`load_eff_games`, `load_v3_inputs`, ...) |
| `live_weather.py` (new) | game-day weather pull (never fatal) + merge |

Scripts: `scripts/build_cfb_advanced.py` (modify: postseason + splits), `scripts/build_team_context.py` (modify: regular-season guard), `scripts/build_cfb_game_data.py`, `scripts/check_cfb_game_data.py`, `scripts/fit_cfb_eff.py`, `scripts/fit_cfb_v3.py`, `scripts/gate_cfb_v3.py`, `scripts/build_cfb_efficiency.py` (all new), `scripts/generate_cfb.py` (modify: `CFB_MODEL_VERSION`). Workflows: `build-cfb-advanced.yml`, `generate-cfb.yml`, `injury-watch.yml` (cfb job - it also runs `generate_cfb.py`).

New committed assets under `assets/cfb/`: `cfbd_games`, `havoc_games`, `drive_games`, `weather_games`, `talent`, `prior_ratings`, `venues`, `efficiency_ratings` (parquet); `eff_config.json`, `v3_weights.json`, `gameline_v3.json`, `v3_gate.json`. `advanced_games.parquet` gains `season_type`, postseason rows and down/distance columns.

Task order follows the spec's dependency chain: **1-4 data pulls/assets**, **5-6 efficiency adjustment and priors**, **7-9 context, points mapping, blend fitted in the walk-forward**, **10 gate (+ v2 baseline reproduction + residual check -> `v3_gate.json`)**, **11-12 serving behind `CFB_MODEL_VERSION` + workflow refreshes + game-day weather**.

---

### Task 1: CFBD client and per-game parsers

**Files:**
- Modify: `src/sportsmodel/cfb/cfbd.py`
- Create: `src/sportsmodel/cfb/cfbd_games.py`
- Test: `tests/cfb/test_cfbd_games.py`

**Interfaces:**
- Consumes: `cfb.teams.cfbd_to_espn(name: str) -> str | None`.
- Produces:
  - `cfbd.CfbdClient(api_key: str, *, http: httpx.Client | None = None, attempts: int = 4, wait=None, timeout: float = 60.0)`; `CfbdClient.from_env(**kw)` (SystemExit if `CFBD_API_KEY` unset); `.get(path: str, params: dict | None = None) -> Any`; `.calls: int`; `.summary() -> str`.
  - `cfbd_games.num(x) -> float`, `season_type(x) -> str`.
  - `parse_games_meta(payload) -> DataFrame[GAMES_COLUMNS]` (game_id, season, week, season_type, start_date, neutral_site, venue_id, home_team, away_team, home_points, away_points, home_pregame_elo, away_pregame_elo).
  - `parse_havoc_games(payload) -> DataFrame[HAVOC_COLUMNS]` (season, week, season_type, game_id, team, opponent, `off_*`/`def_*` of havoc, front7_havoc, db_havoc, havoc_events, plays).
  - `parse_drive_games(payload, games_meta: DataFrame) -> DataFrame[DRIVE_COLUMNS]` (… `off_drives, off_points, off_ppd, off_opps, off_points_per_opp, off_start_yd` and `def_*`).
  - `parse_weather_games(payload) -> DataFrame[WEATHER_COLUMNS]`, `parse_talent(payload) -> [season, team, talent]`, `parse_venues(payload) -> DataFrame[VENUE_COLUMNS]`, `parse_prior_ratings(fpi_payload, srs_payload, season) -> [season, team, fpi, srs]`.
  - Every game-level parser sets `df.attrs["dropped"]`; response shapes follow the CFBD OpenAPI document and are **not** live-verified (Task 4 verifies them).

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_cfbd_games.py`:

```python
"""Pure-parser tests for cfbd_games + the CfbdClient. No network: payloads are
inline dicts shaped like the CFBD v2 OpenAPI schema."""
import math

import httpx
import pandas as pd
import pytest
from tenacity import wait_none

from sportsmodel.cfb import cfbd, cfbd_games as cg
from sportsmodel.cfb.teams import cfbd_to_espn

BAMA, UGA = cfbd_to_espn("Alabama"), cfbd_to_espn("Georgia")


# ---------------------------------------------------------------- the client --

def _client(handler, **kw):
    http = httpx.Client(base_url="https://x.test", transport=httpx.MockTransport(handler))
    return cfbd.CfbdClient("sekret", http=http, wait=wait_none(), **kw)


def test_client_sends_bearer_counts_calls_and_hides_key():
    seen = {}

    def handler(req):
        seen["auth"] = req.headers["Authorization"]
        seen["url"] = str(req.url)
        return httpx.Response(200, json=[{"a": 1}])

    c = _client(handler)
    assert c.get("/talent", {"year": 2024}) == [{"a": 1}]
    assert seen["auth"] == "Bearer sekret" and "sekret" not in seen["url"]
    assert c.calls == 1 and "1" in c.summary()
    assert "sekret" not in repr(c)


def test_client_retries_5xx_then_succeeds_and_counts_attempts():
    n = {"i": 0}

    def handler(req):
        n["i"] += 1
        return httpx.Response(503) if n["i"] < 3 else httpx.Response(200, json={"ok": True})

    c = _client(handler)
    assert c.get("/x") == {"ok": True}
    assert c.calls == 3


def test_client_does_not_retry_4xx():
    c = _client(lambda req: httpx.Response(401))
    with pytest.raises(httpx.HTTPStatusError):
        c.get("/x")
    assert c.calls == 1


def test_client_gives_up_after_attempts():
    c = _client(lambda req: httpx.Response(500), attempts=2)
    with pytest.raises(httpx.HTTPStatusError):
        c.get("/x")
    assert c.calls == 2


def test_from_env_requires_key(monkeypatch):
    monkeypatch.delenv("CFBD_API_KEY", raising=False)
    with pytest.raises(SystemExit):
        cfbd.CfbdClient.from_env()


# ---------------------------------------------------------------- games meta --

GAMES = [
    {"id": 401, "season": 2023, "week": 3, "seasonType": "regular", "startDate": "2023-09-16T16:00:00.000Z",
     "neutralSite": False, "venueId": 3657, "homeTeam": "Alabama", "awayTeam": "Georgia",
     "homePoints": 27, "awayPoints": 24, "homePregameElo": 2100, "awayPregameElo": None},
    {"id": 402, "season": 2023, "week": 3, "seasonType": "regular", "homeTeam": "Alabama",
     "awayTeam": "Nowhere State Fighting Pickles"},
    {"id": 403, "season": 2023, "week": 1, "seasonType": "postseason", "neutralSite": True,
     "homeTeam": "Georgia", "awayTeam": "Alabama", "homePoints": None, "awayPoints": None},
]


def test_games_meta_maps_teams_drops_fcs_and_nans():
    df = cg.parse_games_meta(GAMES)
    assert len(df) == 2 and df.attrs["dropped"] == 1
    r = df.iloc[0]
    assert (r["game_id"], r["home_team"], r["away_team"]) == (401, BAMA, UGA)
    assert r["season_type"] == "regular" and r["venue_id"] == 3657 and not r["neutral_site"]
    assert r["home_pregame_elo"] == 2100 and math.isnan(r["away_pregame_elo"])
    p = df.iloc[1]
    assert p["season_type"] == "postseason" and p["neutral_site"] and math.isnan(p["home_points"])
    assert math.isnan(p["venue_id"])


def test_games_meta_empty_keeps_schema():
    df = cg.parse_games_meta([])
    assert list(df.columns) == cg.GAMES_COLUMNS and len(df) == 0


# -------------------------------------------------------------------- havoc --

def _hv(rate, ev=10, plays=60):
    return {"havocRate": rate, "frontSevenHavocRate": rate / 2, "dbHavocRate": rate / 2,
            "totalHavocEvents": ev, "totalPlays": plays, "frontSevenHavocEvents": 4, "dbHavocEvents": 6}


HAVOC = [
    {"gameId": 401, "season": 2023, "seasonType": "regular", "week": 3, "team": "Alabama",
     "opponent": "Georgia", "offense": _hv(0.12), "defense": _hv(0.20)},
    {"gameId": 402, "season": 2023, "seasonType": "regular", "week": 3, "team": "Alabama",
     "opponent": "Nowhere State Fighting Pickles", "offense": _hv(0.1), "defense": _hv(0.1)},
    {"gameId": 404, "season": 2023, "seasonType": "regular", "week": 4, "team": "Georgia",
     "opponent": "Alabama", "offense": None, "defense": {"havocRate": None}},
]


def test_havoc_maps_units_and_drops():
    df = cg.parse_havoc_games(HAVOC)
    assert len(df) == 2 and df.attrs["dropped"] == 1
    r = df.iloc[0]
    assert (r["team"], r["opponent"], r["season_type"]) == (BAMA, UGA, "regular")
    assert r["off_havoc"] == 0.12 and r["def_havoc"] == 0.20
    assert r["off_front7_havoc"] == 0.06 and r["off_havoc_events"] == 10 and r["off_plays"] == 60


def test_havoc_missing_units_are_nan_not_zero():
    r = cg.parse_havoc_games(HAVOC).iloc[1]
    for c in ("off_havoc", "off_plays", "def_havoc", "def_front7_havoc"):
        assert math.isnan(r[c]), c


# ------------------------------------------------------------------- drives --

META = cg.parse_games_meta(GAMES)


def _drive(gid, off, dfn, s_ytg, e_ytg, s_pts, e_pts, result="PUNT", opp_score=0):
    return {"gameId": gid, "offense": off, "defense": dfn, "startYardsToGoal": s_ytg,
            "endYardsToGoal": e_ytg, "startOffenseScore": s_pts, "endOffenseScore": e_pts,
            "driveResult": result, "plays": 6, "yards": 40}


DRIVES = [
    _drive(401, "Alabama", "Georgia", 75, 0, 0, 7, "TD"),          # scores from own 25: opp, 7 pts
    _drive(401, "Alabama", "Georgia", 70, 45, 7, 7, "PUNT"),       # stalled at the 45: no opp
    _drive(401, "Alabama", "Georgia", 60, 22, 7, 10, "FG"),        # FG: opp, 3 pts
    _drive(401, "Georgia", "Alabama", 80, 35, 0, 0, "FUMBLE"),     # reached the 35: opp, 0 pts
    _drive(401, "Alabama", "Georgia", 90, 90, 10, 10, "END OF HALF"),  # excluded
    _drive(401, "Alabama", "Nowhere State Fighting Pickles", 70, 0, 0, 7, "TD"),  # unmapped -> dropped
    _drive(999, "Alabama", "Georgia", 70, 0, 0, 7, "TD"),          # game not in meta -> dropped
    {"gameId": 401, "offense": "Georgia", "defense": "Alabama", "startYardsToGoal": 70,
     "endYardsToGoal": 60, "startOffenseScore": None, "endOffenseScore": None, "driveResult": "PUNT"},
]


def test_drive_games_aggregate_points_and_opps():
    df = cg.parse_drive_games(DRIVES, META)
    assert df.attrs["dropped"] == 2
    a = df[df["team"] == BAMA].iloc[0]
    assert (a["season"], a["week"], a["season_type"], a["game_id"]) == (2023, 3, "regular", 401)
    assert a["off_drives"] == 3 and a["off_points"] == 10
    assert a["off_opps"] == 2 and a["off_points_per_opp"] == 5.0
    assert a["off_ppd"] == pytest.approx(10 / 3)
    assert a["off_start_yd"] == pytest.approx(((100 - 75) + (100 - 70) + (100 - 60)) / 3)


def test_drive_games_defense_is_opponents_offense_and_unreadable_scores_skipped():
    df = cg.parse_drive_games(DRIVES, META)
    a, g = df[df["team"] == BAMA].iloc[0], df[df["team"] == UGA].iloc[0]
    assert a["def_drives"] == g["off_drives"] == 1      # the None-score Georgia drive was skipped
    assert g["off_opps"] == 1 and g["off_points_per_opp"] == 0.0
    assert g["def_points"] == a["off_points"]


def test_drive_games_no_opps_gives_nan_not_zero():
    d = [_drive(401, "Alabama", "Georgia", 80, 70, 0, 0, "PUNT")]
    r = cg.parse_drive_games(d, META).iloc[0]
    assert r["off_opps"] == 0 and math.isnan(r["off_points_per_opp"])


# ------------------------------------------------------------------ weather --

def _w(gid, home, away, **kw):
    base = {"id": gid, "season": 2023, "week": 3, "seasonType": "regular",
            "startTime": "2023-09-16T16:00:00.000Z", "gameIndoors": False, "homeTeam": home,
            "awayTeam": away, "venueId": 3657, "venue": "Bryant-Denny", "temperature": 71.0,
            "windSpeed": 9.5, "precipitation": 0.0, "snowfall": None, "humidity": 55.0}
    base.update(kw)
    return base


def test_weather_parse_and_nan_for_missing():
    df = cg.parse_weather_games([_w(401, "Alabama", "Georgia"),
                                 _w(402, "Alabama", "Georgia", temperature=None, windSpeed=None, gameIndoors=True),
                                 _w(403, "Alabama", "Nowhere State Fighting Pickles")])
    assert len(df) == 2 and df.attrs["dropped"] == 1
    r0, r1 = df.iloc[0], df.iloc[1]
    assert (r0["home_team"], r0["away_team"], r0["venue_id"]) == (BAMA, UGA, 3657)
    assert r0["temperature"] == 71.0 and r0["wind_speed"] == 9.5 and not r0["game_indoors"]
    assert math.isnan(r0["snowfall"])
    assert r1["game_indoors"] and math.isnan(r1["temperature"]) and math.isnan(r1["wind_speed"])


# ------------------------------------------------- talent / venues / ratings --

def test_talent_maps_and_drops_unmapped_or_null():
    df = cg.parse_talent([{"year": 2024, "team": "Georgia", "talent": 988.2},
                          {"year": 2024, "team": "Nowhere State Fighting Pickles", "talent": 200.0},
                          {"year": 2024, "team": "Alabama", "talent": None}])
    assert len(df) == 1 and df.attrs["dropped"] == 2
    assert df.iloc[0]["team"] == UGA and df.iloc[0]["talent"] == 988.2 and df.iloc[0]["season"] == 2024


def test_venues_parse_coords_elevation_string_and_dome():
    df = cg.parse_venues([
        {"id": 1, "name": "Dome", "timezone": "America/Chicago", "latitude": 29.7, "longitude": -95.4,
         "elevation": "12.3", "dome": True},
        {"id": 2, "name": "Open", "timezone": None, "latitude": None, "longitude": None,
         "elevation": None, "dome": None},
        {"id": None, "name": "no id"}])
    assert len(df) == 2
    d, o = df.iloc[0], df.iloc[1]
    assert d["timezone"] == "America/Chicago" and d["elevation"] == 12.3 and d["dome"] == 1.0
    assert math.isnan(o["latitude"]) and math.isnan(o["elevation"]) and math.isnan(o["dome"])


def test_prior_ratings_join_fpi_and_srs_by_team():
    df = cg.parse_prior_ratings([{"year": 2023, "team": "Georgia", "fpi": 28.1},
                                 {"year": 2023, "team": "Alabama", "fpi": None}],
                                [{"year": 2023, "team": "Georgia", "rating": 20.5},
                                 {"year": 2023, "team": "Nowhere State Fighting Pickles", "rating": 1.0}], 2023)
    u = df[df["team"] == UGA].iloc[0]
    assert (u["season"], u["fpi"], u["srs"]) == (2023, 28.1, 20.5)
    b = df[df["team"] == BAMA].iloc[0]
    assert math.isnan(b["fpi"]) and math.isnan(b["srs"])
    assert df.attrs["dropped"] == 1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_cfbd_games.py -q`
Expected: FAIL at collection with `ImportError: cannot import name 'cfbd_games' from 'sportsmodel.cfb'` (and, once that exists, `AttributeError: module 'sportsmodel.cfb.cfbd' has no attribute 'CfbdClient'`).

- [ ] **Step 3: Implement the client**

Apply this diff to `src/sportsmodel/cfb/cfbd.py` (`git apply` it, or make the same edits by hand):

```diff
--- a/src/sportsmodel/cfb/cfbd.py
+++ b/src/sportsmodel/cfb/cfbd.py
@@ -20,11 +20,16 @@
 """
 from __future__ import annotations
 
+import logging
+import os
 from typing import Any
 
 import httpx
-from tenacity import retry, stop_after_attempt, wait_exponential
+from tenacity import (Retrying, retry, retry_if_exception, stop_after_attempt,
+                      wait_exponential)
 
+log = logging.getLogger("cfbd")
+
 _BASE = "https://api.collegefootballdata.com"
 
 # v1 proxy threshold for "QB production is returning": fraction of a team's
@@ -52,6 +57,60 @@
     return r.json()
 
 
+def _is_transient(exc: BaseException) -> bool:
+    """Retry network errors, 5xx and 429; fail fast on other 4xx (bad params/key)."""
+    if isinstance(exc, httpx.HTTPStatusError):
+        code = exc.response.status_code
+        return code >= 500 or code == 429
+    return isinstance(exc, httpx.TransportError)
+
+
+class CfbdClient:
+    """Shared CFBD client for the paid-tier pulls: Bearer auth, retry with
+    exponential backoff on transient failures, and a per-run call counter.
+
+    The key is held privately and sent only in the Authorization header: it is
+    never logged, never put in a URL, and never appears in `repr`. `calls`
+    counts every HTTP attempt (retries included) so the budget line printed at
+    the end of a run is honest. `http`/`wait` are injectable for tests.
+    """
+
+    def __init__(self, api_key: str, *, http: httpx.Client | None = None,
+                 attempts: int = 4, wait=None, timeout: float = 60.0):
+        if not api_key:
+            raise ValueError("CFBD api_key is empty")
+        self._key = api_key
+        self._http = http or httpx.Client(base_url=_BASE, timeout=timeout)
+        self._attempts = attempts
+        self._wait = wait or wait_exponential(multiplier=1, max=20)
+        self.calls = 0
+
+    @classmethod
+    def from_env(cls, **kw) -> "CfbdClient":
+        key = os.environ.get("CFBD_API_KEY")
+        if not key:
+            raise SystemExit("CFBD_API_KEY not set in environment (export it / repo secret).")
+        return cls(key, **kw)
+
+    def __repr__(self) -> str:
+        return f"CfbdClient(calls={self.calls})"
+
+    def get(self, path: str, params: dict | None = None) -> Any:
+        for attempt in Retrying(stop=stop_after_attempt(self._attempts), wait=self._wait,
+                                retry=retry_if_exception(_is_transient), reraise=True):
+            with attempt:
+                self.calls += 1
+                r = self._http.get(path, params=params,
+                                   headers={"Authorization": f"Bearer {self._key}"})
+                r.raise_for_status()
+                out = r.json()
+        log.info("CFBD GET %s (%d calls this run)", path, self.calls)
+        return out
+
+    def summary(self) -> str:
+        return f"CFBD calls this run: {self.calls}"
+
+
 def parse_sp(payload) -> dict[str, float]:
     """SP+ overall rating by team, from CFBD `/ratings/sp` (list of
     {"team": ..., "rating": ..., ...}).
```

- [ ] **Step 4: Implement the parsers**

Create `src/sportsmodel/cfb/cfbd_games.py`:

```python
"""Pure parsers for the per-game CFBD endpoints behind cfb-ratings-v3.

Every parser takes an already-decoded JSON payload (no network, no file reads)
and returns a pandas DataFrame with one row per team-game (or game / team /
venue). CFBD school names are mapped to ESPN ids with `cfbd_to_espn`; a row
whose team (or opponent) does not map to an FBS id is dropped and counted in
`df.attrs["dropped"]`. Missing / null numbers are NaN, never 0. Every
game-level frame carries season, week, season_type and game_id.

Response shapes follow the CFBD v2 OpenAPI document (api-docs.json). They have
NOT been verified against live responses with the project key; the parsers read
every field defensively (`.get`, NaN on anything non-numeric) and
`scripts/build_cfb_game_data.py --probe` prints the real key structure.
"""
from __future__ import annotations

import math

import pandas as pd

from sportsmodel.cfb.teams import cfbd_to_espn

NAN = float("nan")


def num(x) -> float:
    """float(x) for real numbers; NaN for None / bool / strings / anything else."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return NAN
    return float(x)


def season_type(x) -> str:
    """CFBD seasonType -> 'regular' | 'postseason' (lower-cased pass-through otherwise)."""
    return str(x or "regular").lower()


def _frame(rows: list[dict], columns: list[str], ints=(), strs=(), bools=(), dropped: int = 0):
    df = pd.DataFrame(rows, columns=columns)
    for c in columns:
        if c in ints:
            df[c] = df[c].astype("int64")
        elif c in strs:
            df[c] = df[c].astype(str)
        elif c in bools:
            df[c] = df[c].astype(bool)
        else:
            df[c] = df[c].astype("float64")
    df.attrs["dropped"] = dropped
    return df


# --------------------------------------------------------------- games meta --

GAMES_COLUMNS = ["game_id", "season", "week", "season_type", "start_date", "neutral_site",
                 "venue_id", "home_team", "away_team", "home_points", "away_points",
                 "home_pregame_elo", "away_pregame_elo"]


def parse_games_meta(payload) -> pd.DataFrame:
    """CFBD `/games` -> one row per FBS-vs-FBS game: ids, week, season_type, venue,
    neutral flag, final points (NaN until played) and CFBD's PRE-game Elo (leak-free;
    the post-game fields are deliberately not read)."""
    rows, dropped = [], 0
    for g in payload:
        h, a = cfbd_to_espn(g.get("homeTeam") or ""), cfbd_to_espn(g.get("awayTeam") or "")
        if not h or not a:
            dropped += 1
            continue
        rows.append({"game_id": int(g["id"]), "season": int(g["season"]), "week": int(g["week"]),
                     "season_type": season_type(g.get("seasonType")),
                     "start_date": str(g.get("startDate") or ""),
                     "neutral_site": bool(g.get("neutralSite")),
                     "venue_id": num(g.get("venueId")), "home_team": h, "away_team": a,
                     "home_points": num(g.get("homePoints")), "away_points": num(g.get("awayPoints")),
                     "home_pregame_elo": num(g.get("homePregameElo")),
                     "away_pregame_elo": num(g.get("awayPregameElo"))})
    return _frame(rows, GAMES_COLUMNS, ints=("game_id", "season", "week"),
                  strs=("season_type", "start_date", "home_team", "away_team"),
                  bools=("neutral_site",), dropped=dropped)


# -------------------------------------------------------------------- havoc --

_HAVOC_FIELDS = (("havoc", "havocRate"), ("front7_havoc", "frontSevenHavocRate"),
                 ("db_havoc", "dbHavocRate"), ("havoc_events", "totalHavocEvents"),
                 ("plays", "totalPlays"))
HAVOC_COLUMNS = (["season", "week", "season_type", "game_id", "team", "opponent"]
                 + [f"{u}_{k}" for u in ("off", "def") for k, _ in _HAVOC_FIELDS])


def parse_havoc_games(payload) -> pd.DataFrame:
    """CFBD `/stats/game/havoc` -> one row per team-game. `off_*` is the havoc the
    team's OFFENSE suffered, `def_*` the havoc its DEFENSE created, with rates per
    play (the unit objects also carry event counts and total plays)."""
    rows, dropped = [], 0
    for g in payload:
        team, opp = cfbd_to_espn(g.get("team") or ""), cfbd_to_espn(g.get("opponent") or "")
        if not team or not opp:
            dropped += 1
            continue
        row = {"season": int(g["season"]), "week": int(g["week"]),
               "season_type": season_type(g.get("seasonType")),
               "game_id": int(g["gameId"]), "team": team, "opponent": opp}
        for unit, key in (("off", "offense"), ("def", "defense")):
            obj = g.get(key) if isinstance(g.get(key), dict) else {}
            for k, src in _HAVOC_FIELDS:
                row[f"{unit}_{k}"] = num(obj.get(src))
        rows.append(row)
    return _frame(rows, HAVOC_COLUMNS, ints=("season", "week", "game_id"),
                  strs=("season_type", "team", "opponent"), dropped=dropped)


# ------------------------------------------------------------------- drives --

# CFBD driveResult values that are not real possessions (matched upper-case).
EXCLUDED_DRIVE_RESULTS = frozenset({"END OF HALF", "END OF GAME", "END OF 4TH QUARTER"})
SCORING_OPP_YARDS = 40          # a "scoring opportunity" = the drive reached the opp's 40
DRIVE_COLUMNS = (["season", "week", "season_type", "game_id", "team", "opponent"]
                 + [f"{u}_{k}" for u in ("off", "def")
                    for k in ("drives", "points", "ppd", "opps", "points_per_opp", "start_yd")])


def _drive_points(d: dict) -> float:
    s, e = num(d.get("startOffenseScore")), num(d.get("endOffenseScore"))
    return NAN if math.isnan(s) or math.isnan(e) else max(0.0, e - s)


def parse_drive_games(payload, games_meta: pd.DataFrame) -> pd.DataFrame:
    """CFBD `/drives` -> one row per team-game of drive efficiency.

    points = offense score at drive end minus at drive start; a scoring
    opportunity is a drive whose closest point to the opponent's goal line
    (min of start/end yards-to-goal) was inside the 40 (CFBD has no per-play
    drive progress, so this is an approximation); start_yd is the average own-
    yard-line start (100 - startYardsToGoal). Non-possession results and drives
    with unreadable scores are skipped. `games_meta` (parse_games_meta output)
    supplies season / week / season_type; drives of unknown games are dropped
    and counted. `def_*` is the opponent offense's value in the same game.
    """
    meta = games_meta.set_index("game_id")[["season", "week", "season_type"]].to_dict("index")
    agg: dict[tuple[int, str], dict] = {}
    opp_of: dict[tuple[int, str], str] = {}
    dropped = 0
    for d in payload:
        gid = int(d["gameId"])
        off, dfn = cfbd_to_espn(d.get("offense") or ""), cfbd_to_espn(d.get("defense") or "")
        if not off or not dfn or gid not in meta:
            dropped += 1
            continue
        if str(d.get("driveResult") or "").upper() in EXCLUDED_DRIVE_RESULTS:
            continue
        pts, ytg_s, ytg_e = _drive_points(d), num(d.get("startYardsToGoal")), num(d.get("endYardsToGoal"))
        if math.isnan(pts) or math.isnan(ytg_s) or math.isnan(ytg_e):
            continue
        a = agg.setdefault((gid, off), {"drives": 0, "points": 0.0, "opps": 0, "start": 0.0})
        a["drives"] += 1
        a["points"] += pts
        a["opps"] += int(min(ytg_s, ytg_e) <= SCORING_OPP_YARDS)
        a["start"] += 100.0 - ytg_s
        opp_of[(gid, off)] = dfn

    def unit(a: dict | None) -> dict:
        if not a:
            return {k: NAN for k in ("drives", "points", "ppd", "opps", "points_per_opp", "start_yd")}
        return {"drives": float(a["drives"]), "points": a["points"], "ppd": a["points"] / a["drives"],
                "opps": float(a["opps"]),
                "points_per_opp": a["points"] / a["opps"] if a["opps"] else NAN,
                "start_yd": a["start"] / a["drives"]}

    rows = []
    for (gid, team), a in sorted(agg.items()):
        opp = opp_of[(gid, team)]
        m = meta[gid]
        row = {"season": int(m["season"]), "week": int(m["week"]), "season_type": m["season_type"],
               "game_id": gid, "team": team, "opponent": opp}
        row.update({f"off_{k}": v for k, v in unit(a).items()})
        row.update({f"def_{k}": v for k, v in unit(agg.get((gid, opp))).items()})
        rows.append(row)
    return _frame(rows, DRIVE_COLUMNS, ints=("season", "week", "game_id"),
                  strs=("season_type", "team", "opponent"), dropped=dropped)


# ------------------------------------------------------------------ weather --

WEATHER_COLUMNS = ["game_id", "season", "week", "season_type", "start_time", "home_team",
                   "away_team", "venue_id", "game_indoors", "temperature", "wind_speed",
                   "precipitation", "snowfall", "humidity"]


def parse_weather_games(payload) -> pd.DataFrame:
    """CFBD `/games/weather` -> one row per FBS-vs-FBS game (historical observation
    or forecast). `game_indoors` is CFBD's dome flag; every numeric field is NaN when
    CFBD has no reading. Units are CFBD's own (expected: temperature F, wind mph,
    precipitation inches; checked by the backfill sanity step, not assumed here)."""
    rows, dropped = [], 0
    for w in payload:
        h, a = cfbd_to_espn(w.get("homeTeam") or ""), cfbd_to_espn(w.get("awayTeam") or "")
        if not h or not a:
            dropped += 1
            continue
        rows.append({"game_id": int(w["id"]), "season": int(w["season"]), "week": int(w["week"]),
                     "season_type": season_type(w.get("seasonType")),
                     "start_time": str(w.get("startTime") or ""), "home_team": h, "away_team": a,
                     "venue_id": num(w.get("venueId")), "game_indoors": bool(w.get("gameIndoors")),
                     "temperature": num(w.get("temperature")), "wind_speed": num(w.get("windSpeed")),
                     "precipitation": num(w.get("precipitation")), "snowfall": num(w.get("snowfall")),
                     "humidity": num(w.get("humidity"))})
    return _frame(rows, WEATHER_COLUMNS, ints=("game_id", "season", "week"),
                  strs=("season_type", "start_time", "home_team", "away_team"),
                  bools=("game_indoors",), dropped=dropped)


# ------------------------------------------------------------------- talent --

def parse_talent(payload) -> pd.DataFrame:
    """CFBD `/talent?year=Y` (247 team talent composite) -> season, team, talent."""
    rows, dropped = [], 0
    for t in payload:
        team = cfbd_to_espn(t.get("team") or "")
        if not team or math.isnan(num(t.get("talent"))):
            dropped += 1
            continue
        rows.append({"season": int(t["year"]), "team": team, "talent": num(t["talent"])})
    return _frame(rows, ["season", "team", "talent"], ints=("season",), strs=("team",), dropped=dropped)


# ------------------------------------------------------------------- venues --

VENUE_COLUMNS = ["venue_id", "name", "timezone", "latitude", "longitude", "elevation", "dome"]


def parse_venues(payload) -> pd.DataFrame:
    """CFBD `/venues` -> venue_id, name, IANA timezone, lat/lon, elevation (CFBD sends it
    as a string), dome (NaN when unknown). Venues without an id are skipped."""
    rows = []
    for v in payload:
        if v.get("id") is None:
            continue
        try:
            elev = float(v.get("elevation"))
        except (TypeError, ValueError):
            elev = NAN
        dome = v.get("dome")
        rows.append({"venue_id": int(v["id"]), "name": str(v.get("name") or ""),
                     "timezone": str(v.get("timezone") or ""), "latitude": num(v.get("latitude")),
                     "longitude": num(v.get("longitude")), "elevation": elev,
                     "dome": float(dome) if isinstance(dome, bool) else NAN})
    return _frame(rows, VENUE_COLUMNS, ints=("venue_id",), strs=("name", "timezone"))


# ------------------------------------------------------------ prior ratings --

def parse_prior_ratings(fpi_payload, srs_payload, season: int) -> pd.DataFrame:
    """CFBD `/ratings/fpi` + `/ratings/srs` for `season` -> season, team, fpi, srs.
    These are END-of-season ratings: the residual check only ever joins them to the
    NEXT season's games (season + 1), never to in-season games."""
    fpi = {}
    for r in fpi_payload:
        team = cfbd_to_espn(r.get("team") or "")
        if team:
            fpi[team] = num(r.get("fpi"))
    srs, dropped = {}, 0
    for r in srs_payload:
        team = cfbd_to_espn(r.get("team") or "")
        if team:
            srs[team] = num(r.get("rating"))
        else:
            dropped += 1
    rows = [{"season": int(season), "team": t, "fpi": fpi.get(t, NAN), "srs": srs.get(t, NAN)}
            for t in sorted(set(fpi) | set(srs))]
    return _frame(rows, ["season", "team", "fpi", "srs"], ints=("season",), strs=("team",),
                  dropped=dropped)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_cfbd_games.py tests/cfb/test_cfbd_parsers.py tests/cfb/test_cfbd_ppa.py -q`
Expected: PASS (16 new tests plus the existing parser tests).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/cfb/cfbd.py src/sportsmodel/cfb/cfbd_games.py tests/cfb/test_cfbd_games.py
git commit -m "feat(cfb): CfbdClient (retry, call counter) + per-game CFBD parsers for v3"
```

---

### Task 2: Advanced stats - postseason, season_type, down/distance splits

**Files:**
- Modify: `scripts/build_cfb_advanced.py`
- Modify: `scripts/build_team_context.py` (the matchup grades must never see postseason rows: CFBD restarts week numbering for bowls)
- Test: `tests/scripts/test_build_cfb_advanced.py`, `tests/scripts/test_build_team_context.py`

**Interfaces:**
- Consumes: existing `parse_advanced`, `merge_frames`, `fetch_year(year, key)` in `build_cfb_advanced.py`.
- Produces:
  - `parse_advanced(payload)` rows now carry `season_type` (`"regular"` default) and the extra unit columns `std_down_success, pass_down_success, std_down_ppa, pass_down_ppa, line_yards, stuff_rate, power_success` (each as `off_*` and `def_*`).
  - `fetch_year(year: int, key: str, season_type: str = "regular") -> list`; CLI `--season-types regular postseason` (default: both).
  - `keep_postseason(existing: DataFrame | None, new: DataFrame, year: int) -> DataFrame`.
  - `build_team_context.regular_season_only(advanced: DataFrame | None) -> DataFrame | None`.

- [ ] **Step 1: Write the failing tests**

Apply this diff to the two existing test files (it also updates `_run_main`'s `fetch_year` stub to the new 3-argument signature):

```diff
--- a/tests/scripts/test_build_cfb_advanced.py
+++ b/tests/scripts/test_build_cfb_advanced.py
@@ -117,7 +117,8 @@
     monkeypatch.setattr(bca, "_OUT", out)
     monkeypatch.setattr(bca, "_SCHEDULES", tmp_path / "missing.parquet")
     monkeypatch.setattr(bca, "ROOT", tmp_path)
-    monkeypatch.setattr(bca, "fetch_year", lambda y, key: payloads[y])
+    monkeypatch.setattr(bca, "fetch_year", lambda y, key, st="regular":
+                        payloads[y] if st == "regular" else payloads.get(("post", y), []))
     monkeypatch.setenv("CFBD_API_KEY", "test-key")
     monkeypatch.setattr("sys.argv", ["build_cfb_advanced.py", *argv])
     bca.main()
@@ -172,3 +173,46 @@
         r = bca.parse_advanced([g]).iloc[0]
         assert math.isnan(r["off_pass_plays"]), (ppa, total)
         assert r["off_rush_plays"] == 30
+
+
+def _post(payload_row, game_id):
+    return dict(payload_row, gameId=game_id, seasonType="postseason", week=1)
+
+
+def test_parse_stamps_season_type_and_reads_down_distance_splits():
+    g = dict(PAYLOAD[0], seasonType="postseason")
+    g["offense"] = dict(g["offense"], standardDowns={"ppa": 0.2, "successRate": 0.55},
+                        passingDowns={"ppa": -0.1, "successRate": 0.31},
+                        lineYards=3.1, stuffRate=0.17, powerSuccess=0.7)
+    df = bca.parse_advanced([g, PAYLOAD[0]])
+    assert list(df["season_type"]) == ["postseason", "regular"]      # missing seasonType = regular
+    r = df.iloc[0]
+    assert r["off_std_down_success"] == 0.55 and r["off_pass_down_ppa"] == -0.1
+    assert r["off_line_yards"] == 3.1 and r["off_stuff_rate"] == 0.17 and r["off_power_success"] == 0.7
+    assert math.isnan(df.iloc[1]["off_std_down_success"])            # absent split -> NaN, not 0
+
+
+def test_keep_postseason_preserves_committed_bowls_on_empty_post_pull():
+    reg = bca.parse_advanced([PAYLOAD[0]])
+    old_post = bca.parse_advanced([_post(PAYLOAD[0], 777)])
+    existing = pd.concat([reg, old_post], ignore_index=True)
+    kept = bca.keep_postseason(existing, reg, 2023)
+    assert set(kept["game_id"]) == {401, 777}
+    fresh_post = bca.parse_advanced([_post(PAYLOAD[0], 888)])
+    new = pd.concat([reg, fresh_post], ignore_index=True)
+    assert set(bca.keep_postseason(existing, new, 2023)["game_id"]) == {401, 888}   # real pull wins
+    assert bca.keep_postseason(None, reg, 2023) is reg
+
+
+def test_main_pulls_both_season_types_by_default(monkeypatch, tmp_path):
+    payloads = {2023: [PAYLOAD[0]], ("post", 2023): [_post(PAYLOAD[0], 555)]}
+    got = _run_main(monkeypatch, tmp_path, payloads, None, ["--seasons", "2023"])
+    assert dict(zip(got["game_id"], got["season_type"])) == {401: "regular", 555: "postseason"}
+
+
+def test_main_empty_postseason_keeps_existing_bowls(monkeypatch, tmp_path):
+    old = pd.concat([bca.parse_advanced([PAYLOAD[0]]),
+                     bca.parse_advanced([_post(PAYLOAD[0], 777)])], ignore_index=True)
+    got = _run_main(monkeypatch, tmp_path, {2023: [dict(PAYLOAD[0], gameId=402)]}, old,
+                    ["--merge", "--seasons", "2023"])
+    assert set(got["game_id"]) == {402, 777}
```

```diff
--- a/tests/scripts/test_build_team_context.py
+++ b/tests/scripts/test_build_team_context.py
@@ -514,3 +514,13 @@
     home_w = rk["home_record"].str.split("-").str[0].astype(int)
     road_w = rk["road_record"].str.split("-").str[0].astype(int)
     assert (home_w + road_w == wins).all()      # fixtures have no neutral games
+
+
+def test_regular_season_only_drops_postseason_rows_and_tolerates_old_parquets():
+    adv = pd.DataFrame({"season": [2024, 2024, 2024], "week": [1, 1, 2], "game_id": [1, 2, 3],
+                        "season_type": ["regular", "postseason", None]})
+    got = btc.regular_season_only(adv)
+    assert list(got["game_id"]) == [1, 3]                       # postseason out, null = regular
+    old = adv.drop(columns="season_type")
+    assert btc.regular_season_only(old) is old                  # pre-v3 parquet untouched
+    assert btc.regular_season_only(None) is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_build_cfb_advanced.py tests/scripts/test_build_team_context.py -q`
Expected: FAIL (`KeyError: 'season_type'`, `AttributeError: ... 'keep_postseason'`, `AttributeError: ... 'regular_season_only'`).

- [ ] **Step 3: Implement**

```diff
--- a/scripts/build_cfb_advanced.py
+++ b/scripts/build_cfb_advanced.py
@@ -1,7 +1,9 @@
 """Pull CFBD advanced per-game team stats -> assets/cfb/advanced_games.parquet.
 
 Reads CFBD_API_KEY from the environment (never hardcoded, never logged). Source is
-`/stats/game/advanced?year=Y&seasonType=regular`, which returns one object per
+`/stats/game/advanced?year=Y&seasonType=regular|postseason` (both pulled by default; each
+row is stamped with `season_type`, so consumers that want the regular season only filter on
+it), which returns one object per
 team-game: {gameId, season, week, team, opponent, offense{...}, defense{...}} where each
 unit carries plays / ppa / successRate / explosiveness plus `passingPlays` and
 `rushingPlays` sub-objects ({ppa, totalPPA, successRate, explosiveness}). Missing
@@ -50,11 +52,19 @@
     ("rush_ppa", ("rushingPlays", "ppa")),
     ("rush_success", ("rushingPlays", "successRate")),
     ("rush_explosiveness", ("rushingPlays", "explosiveness")),
+    # down/distance splits + run-game shape (residual-check features for cfb-ratings-v3)
+    ("std_down_success", ("standardDowns", "successRate")),
+    ("pass_down_success", ("passingDowns", "successRate")),
+    ("std_down_ppa", ("standardDowns", "ppa")),
+    ("pass_down_ppa", ("passingDowns", "ppa")),
+    ("line_yards", ("lineYards",)),
+    ("stuff_rate", ("stuffRate",)),
+    ("power_success", ("powerSuccess",)),
 )
 _SPLIT_PLAYS = (("pass_plays", ("passingPlays", "plays")),
                 ("rush_plays", ("rushingPlays", "plays")))
 _UNIT_FIELDS = tuple(s for s, _ in _METRICS)
-_ID_COLS = ("season", "week", "game_id", "team", "opponent")
+_ID_COLS = ("season", "week", "season_type", "game_id", "team", "opponent")
 
 
 def _unit_cols(prefix: str) -> list[str]:
@@ -111,13 +121,14 @@
             dropped += 1
             continue
         row = {"season": int(g["season"]), "week": int(g["week"]),
+               "season_type": str(g.get("seasonType") or "regular").lower(),
                "game_id": int(g["gameId"]), "team": team, "opponent": opp}
         row.update(_unit_row("off", g.get("offense")))
         row.update(_unit_row("def", g.get("defense")))
         rows.append(row)
     df = pd.DataFrame(rows, columns=COLUMNS)
     for c in COLUMNS:
-        if c in ("team", "opponent"):
+        if c in ("team", "opponent", "season_type"):
             df[c] = df[c].astype(str)
         elif c in ("season", "week", "game_id"):
             df[c] = df[c].astype("int64")
@@ -134,10 +145,23 @@
     else:
         kept = existing[~existing["season"].isin(list(seasons))]
         out = pd.concat([kept, new], ignore_index=True) if len(kept) else new.copy()
+    if "season_type" in out.columns:           # pre-v3 parquets carry no season_type: all regular
+        out["season_type"] = out["season_type"].fillna("regular")
     out = out.drop_duplicates(subset=["season", "game_id", "team"], keep="last")
     return out.sort_values(["season", "week", "game_id", "team"]).reset_index(drop=True)
 
 
+def keep_postseason(existing: pd.DataFrame | None, new: pd.DataFrame, year: int) -> pd.DataFrame:
+    """A season refresh whose postseason pull came back empty (mid-season, or an API
+    blip after bowls) must not drop the postseason rows already committed."""
+    if existing is None or "season_type" not in existing.columns:
+        return new
+    if (new["season_type"] == "postseason").any():
+        return new
+    old = existing[(existing["season"] == year) & (existing["season_type"] == "postseason")]
+    return pd.concat([new, old], ignore_index=True) if len(old) else new
+
+
 def coverage(adv: pd.DataFrame, sched: pd.DataFrame, fbs: set[str] | None = None) -> pd.DataFrame:
     """Per season: FBS-vs-FBS schedule games and the share with both teams' rows present."""
     fbs = load_fbs_ids() if fbs is None else fbs
@@ -149,12 +173,12 @@
     return g[g.index.isin(adv["season"].unique())]
 
 
-def fetch_year(year: int, key: str) -> list:
-    """CFBD advanced game stats for a season, retrying transient 5xx / network errors."""
+def fetch_year(year: int, key: str, season_type: str = "regular") -> list:
+    """CFBD advanced game stats for a season + season type, retrying transient 5xx / network errors."""
     last: Exception | None = None
     for attempt in range(4):
         try:
-            r = httpx.get(_API, params={"year": year, "seasonType": "regular"},
+            r = httpx.get(_API, params={"year": year, "seasonType": season_type},
                           headers={"Authorization": f"Bearer {key}"}, timeout=60)
             r.raise_for_status()
             return r.json()
@@ -174,6 +198,8 @@
     ap = argparse.ArgumentParser()
     ap.add_argument("--seasons", type=int, nargs="+",
                     default=list(range(2015, dt.date.today().year + 1)))
+    ap.add_argument("--season-types", nargs="+", default=["regular", "postseason"],
+                    choices=["regular", "postseason"], help="CFBD seasonType values to pull")
     ap.add_argument("--merge", action="store_true",
                     help="refresh only the given seasons, keeping the rest of the parquet")
     args = ap.parse_args()
@@ -185,10 +211,12 @@
     existing = pd.read_parquet(_OUT) if args.merge and _OUT.exists() else None
     frames, dropped, replace = [], 0, []
     for y in args.seasons:
-        payload = fetch_year(y, key)
-        df = parse_advanced(payload)
-        dropped += df.attrs["dropped"]
-        print(f"{y}: {len(payload)} team-games, {len(df)} kept, "
+        parts = [parse_advanced(fetch_year(y, key, st)) for st in args.season_types]
+        dropped += sum(p.attrs["dropped"] for p in parts)
+        df = pd.concat(parts, ignore_index=True)
+        if len(df) and "postseason" in args.season_types:
+            df = keep_postseason(existing, df, y)
+        print(f"{y}: {len(df)} team-games kept ({', '.join(args.season_types)}), "
               f"{df['off_pass_plays'].notna().mean() if len(df) else 0:.0%} with pass/rush counts",
               flush=True)
         if df.empty and existing is not None and (existing["season"] == y).any():
```

```diff
--- a/scripts/build_team_context.py
+++ b/scripts/build_team_context.py
@@ -382,6 +382,15 @@
     if "game_type" in s.columns:
         s = s[s["game_type"] == "REG"]
     return played_games(s.assign(neutral=s["neutral_site"].fillna(False).astype(bool)))
+
+
+def regular_season_only(advanced: pd.DataFrame | None) -> pd.DataFrame | None:
+    """advanced_games.parquet also carries postseason rows (cfb-ratings-v3); CFBD numbers bowl
+    weeks from 1 again, so the matchup grades (which key on season + week) must never see them.
+    A parquet without a season_type column (pre-v3) is all regular season."""
+    if advanced is None or "season_type" not in advanced.columns:
+        return advanced
+    return advanced[advanced["season_type"].fillna("regular") == "regular"].reset_index(drop=True)
 
 
 def load_cfb_sources(now: pd.Timestamp) -> dict:
@@ -391,7 +400,8 @@
     print(f"cfb: schedule {len(asset)} asset rows + {len(sched) - len(asset)} from ESPN")
     season = int(sched["season"].max())
     pks = sched.loc[sched["season"] == season, "game_pk"].astype("int64").tolist()
-    advanced = pd.read_parquet(ADVANCED_PATH) if ADVANCED_PATH.exists() else None
+    advanced = regular_season_only(
+        pd.read_parquet(ADVANCED_PATH) if ADVANCED_PATH.exists() else None)
     return {"schedules": sched, "lines": pd.read_parquet(CFB_ASSETS / "lines.parquet"),
             "live_close": load_cfb_odds(pks), "advanced": advanced}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_build_cfb_advanced.py tests/scripts/test_build_team_context.py tests/test_workflows_cfb_advanced.py -q`
Expected: PASS (14 + 17 + 2 existing workflow tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_cfb_advanced.py scripts/build_team_context.py tests/scripts/test_build_cfb_advanced.py tests/scripts/test_build_team_context.py
git commit -m "feat(cfb): advanced stats pull postseason + season_type + down/distance splits; matchup grades stay regular-season"
```

---

### Task 3: Game-data build script and asset sanity checks

**Files:**
- Create: `scripts/build_cfb_game_data.py`, `src/sportsmodel/cfb/data_checks.py`, `scripts/check_cfb_game_data.py`
- Test: `tests/scripts/test_build_cfb_game_data.py`, `tests/cfb/test_data_checks.py`

**Interfaces:**
- Consumes: `CfbdClient.get`, the `cfbd_games` parsers (Task 1).
- Produces:
  - `build_cfb_game_data.run(client, datasets: list[str], seasons: list[int], out_dir: Path) -> dict[str, int]` (merge-by-season into `cfbd_games.parquet`, `havoc_games.parquet`, `drive_games.parquet`, `weather_games.parquet`, `talent.parquet`, `prior_ratings.parquet`, `venues.parquet`; `games` always runs before `drives`); `merge_asset(existing, new, keys)`, `describe_shape(obj, depth=3)`; CLI `--seasons`, `--datasets`, `--probe`.
  - `data_checks.check_all(assets: dict, sched: DataFrame) -> list[str]` (empty list = OK) and `check_weather / check_havoc / check_drives / check_coverage`; CLI `scripts/check_cfb_game_data.py` (exit 1 on any problem).

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_cfb_game_data.py`:

```python
"""build_cfb_game_data: pure merge + describe helpers and a stubbed-client run (no network)."""
import importlib.util
import pathlib

import pandas as pd
import pytest

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_game_data.py"
_s = importlib.util.spec_from_file_location("build_cfb_game_data", _p)
bgd = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bgd)


def test_merge_asset_replaces_pulled_seasons_and_keeps_empty_ones():
    existing = pd.DataFrame({"season": [2022, 2023], "team": ["a", "a"], "talent": [1.0, 2.0]})
    new = pd.DataFrame({"season": [2023, 2023], "team": ["a", "b"], "talent": [9.0, 8.0]})
    out = bgd.merge_asset(existing, new, ["season", "team"])
    assert list(out["talent"]) == [1.0, 9.0, 8.0]                   # 2022 kept, 2023 replaced
    empty = new.iloc[0:0]
    assert bgd.merge_asset(existing, empty, ["season", "team"]).equals(existing)   # empty pull wipes nothing
    assert len(bgd.merge_asset(None, new, ["season", "team"])) == 2


def test_merge_asset_venues_replaced_whole():
    old = pd.DataFrame({"venue_id": [1, 2], "name": ["a", "b"]})
    new = pd.DataFrame({"venue_id": [2, 3], "name": ["B", "C"]})
    assert list(bgd.merge_asset(old, new, ["venue_id"])["name"]) == ["B", "C"]


def test_describe_shape_hides_values_and_follows_first_record():
    shape = bgd.describe_shape([{"id": 7, "team": "Alabama", "offense": {"ppa": 0.3, "tags": [1, 2]}}, {"x": 1}])
    assert shape == [{"id": "int", "team": "str", "offense": {"ppa": "float", "tags": ["int"]}}]
    assert bgd.describe_shape([]) == []


class StubClient:
    def __init__(self, routes):
        self.routes, self.calls, self.seen = routes, 0, []

    def get(self, path, params=None):
        self.calls += 1
        self.seen.append((path, dict(params or {})))
        return self.routes[path]

    def summary(self):
        return f"CFBD calls this run: {self.calls}"


GAMES = [{"id": 401, "season": 2023, "week": 3, "seasonType": "regular", "homeTeam": "Alabama",
          "awayTeam": "Georgia", "venueId": 7, "homePoints": 20, "awayPoints": 10}]
DRIVES = [{"gameId": 401, "offense": "Alabama", "defense": "Georgia", "startYardsToGoal": 70,
           "endYardsToGoal": 0, "startOffenseScore": 0, "endOffenseScore": 7, "driveResult": "TD"}]


def test_run_pulls_games_then_drives_with_meta_and_merges(tmp_path):
    client = StubClient({"/games": GAMES, "/drives": DRIVES})
    # same payload for both season types: the (season, game_id, team) merge de-duplicates it
    wrote = bgd.run(client, ["drives", "games"], [2023], tmp_path)
    assert wrote["games"] == 1 and wrote["drives"] >= 1
    drives = pd.read_parquet(tmp_path / "drive_games.parquet")
    assert set(drives["week"]) == {3} and drives["off_points"].iloc[0] == 7.0
    assert [p for p, _ in client.seen[:2]] == ["/games", "/games"]         # games ran first (meta for drives)
    assert {q["seasonType"] for p, q in client.seen if p == "/drives"} == {"regular", "postseason"}


def test_run_drives_without_games_asset_stops(tmp_path):
    with pytest.raises(SystemExit):
        bgd.run(StubClient({"/drives": DRIVES}), ["drives"], [2023], tmp_path)


def test_run_empty_pull_keeps_committed_season_and_warns(tmp_path, capsys):
    pd.DataFrame({"season": [2023], "team": ["333"], "talent": [900.0]}).to_parquet(tmp_path / "talent.parquet")
    bgd.run(StubClient({"/talent": []}), ["talent"], [2023], tmp_path)
    got = pd.read_parquet(tmp_path / "talent.parquet")
    assert list(got["talent"]) == [900.0]
    assert "::warning::" in capsys.readouterr().out
```

Create `tests/cfb/test_data_checks.py`:

```python
import numpy as np
import pandas as pd

from sportsmodel.cfb import data_checks as dc


def weather(temp=62.0, wind=7.0, precip=0.0, indoors=False, season=2023):
    return pd.DataFrame({"season": [season] * 4, "game_id": range(4), "game_indoors": [indoors] * 4,
                         "temperature": [temp] * 4, "wind_speed": [wind] * 4, "precipitation": [precip] * 4})


def test_weather_units_flagged_when_not_fahrenheit_mph_inches():
    assert dc.check_weather(weather()) == []
    assert any("degrees F" in p for p in dc.check_weather(weather(temp=17.0)))          # Celsius
    assert any("mph" in p for p in dc.check_weather(weather(wind=2.5 * 0.44)))          # m/s-ish
    assert any("precipitation" in p for p in dc.check_weather(weather(precip=80.0)))
    assert dc.check_weather(weather(temp=17.0, indoors=True)) == []                    # domes are ignored
    assert dc.check_weather(weather().iloc[0:0]) == ["weather: no rows"]


def havoc(pair_gap=0.0):
    rows = []
    for gid in (1, 2, 3):
        rows.append({"season": 2023, "game_id": gid, "team": "a", "opponent": "b", "off_havoc": 0.12,
                     "def_havoc": 0.14})
        rows.append({"season": 2023, "game_id": gid, "team": "b", "opponent": "a", "off_havoc": 0.14 + pair_gap,
                     "def_havoc": 0.12})
    return pd.DataFrame(rows)


def test_havoc_sides_must_pair_up_across_the_two_rows_of_a_game():
    assert dc.check_havoc(havoc()) == []
    bad = dc.check_havoc(havoc(pair_gap=0.08))
    assert any("other way round" in p for p in bad)
    out_of_range = havoc().assign(off_havoc=0.9, def_havoc=0.9)
    assert any("outside" in p for p in dc.check_havoc(out_of_range))


def drives(points=(10.0, 7.0)):
    base = {"season": 2023, "game_id": 1, "off_start_yd": 30.0}
    a = {**base, "team": "a", "opponent": "b", "off_ppd": 2.0, "off_points": points[0], "def_points": points[1]}
    b = {**base, "team": "b", "opponent": "a", "off_ppd": 1.5, "off_points": points[1], "def_points": points[0]}
    return pd.DataFrame([a, b])


def test_drive_checks_ranges_and_defense_equals_opponent_offense():
    assert dc.check_drives(drives()) == []
    assert any("def_points" in p for p in dc.check_drives(drives().assign(def_points=[1.0, 1.0])))
    assert any("points/drive" in p for p in dc.check_drives(drives().assign(off_ppd=9.0)))


def test_coverage_flags_thin_seasons_but_not_pre_2018():
    sched = pd.DataFrame({"season": [2016] * 2 + [2023] * 2, "game_pk": [1, 2, 3, 4], "game_type": ["REG"] * 4,
                          "home_team": ["a"] * 4, "away_team": ["b"] * 4})
    thin = pd.DataFrame({"season": [2016, 2023], "game_id": [1, 3], "team": ["a", "a"], "season_type": ["regular"] * 2})
    full = pd.DataFrame({"season": [2023] * 4, "game_id": [3, 3, 4, 4], "team": ["a", "b"] * 2,
                         "season_type": ["regular"] * 4})
    wx = pd.DataFrame({"season": [2016, 2016, 2023, 2023], "game_id": [1, 2, 3, 4], "season_type": ["regular"] * 4})
    p = dc.check_coverage({"advanced": full, "havoc": full, "drives": full, "weather": wx}, sched)
    assert p == []
    p = dc.check_coverage({"advanced": full, "havoc": thin, "drives": full, "weather": wx}, sched)
    assert len(p) == 1 and p[0].startswith("havoc") and "2023" in p[0]
    assert dc.check_coverage({"advanced": full}, sched)[0].startswith("havoc: asset missing")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_build_cfb_game_data.py tests/cfb/test_data_checks.py -q`
Expected: FAIL (`FileNotFoundError` for `scripts/build_cfb_game_data.py`; `ImportError: cannot import name 'data_checks'`).

- [ ] **Step 3: Implement the build script**

Create `scripts/build_cfb_game_data.py`:

```python
"""Pull the per-game CFBD datasets behind cfb-ratings-v3 -> committed parquet under assets/cfb/.

Datasets (CFBD endpoint -> asset):
  games    /games              -> cfbd_games.parquet     game meta: week, season_type, venue, neutral, pre-game Elo
  havoc    /stats/game/havoc   -> havoc_games.parquet    per team-game havoc rates
  drives   /drives             -> drive_games.parquet    per team-game drive efficiency (needs `games`)
  weather  /games/weather      -> weather_games.parquet  per game weather (historical + forecast)
  talent   /talent             -> talent.parquet         yearly 247 talent composite
  ratings  /ratings/fpi + srs  -> prior_ratings.parquet  yearly FPI/SRS (residual check; next-season use only)
  venues   /venues             -> venues.parquet         static lat/lon/timezone/dome

Reads CFBD_API_KEY from the environment through CfbdClient (never logged). Every dataset is
always merged season-by-season: a season whose pull returns no rows keeps its committed rows. The
call counter is printed at the end.

Usage:
  CFBD_API_KEY=... uv run python scripts/build_cfb_game_data.py --probe          # print real response shapes
  CFBD_API_KEY=... uv run python scripts/build_cfb_game_data.py --seasons 2015 2016 ... 2026
  CFBD_API_KEY=... uv run python scripts/build_cfb_game_data.py --seasons 2026   # weekly refresh (other seasons kept)
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sportsmodel.cfb import cfbd_games as cg  # noqa: E402
from sportsmodel.cfb.cfbd import CfbdClient  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
FILES = {"games": "cfbd_games.parquet", "havoc": "havoc_games.parquet", "drives": "drive_games.parquet",
         "weather": "weather_games.parquet", "talent": "talent.parquet",
         "ratings": "prior_ratings.parquet", "venues": "venues.parquet"}
KEYS = {"games": ["season", "game_id"], "havoc": ["season", "game_id", "team"],
        "drives": ["season", "game_id", "team"], "weather": ["season", "game_id"],
        "talent": ["season", "team"], "ratings": ["season", "team"], "venues": ["venue_id"]}
ORDER = ["games", "havoc", "drives", "weather", "talent", "ratings", "venues"]
SEASON_TYPES = ("regular", "postseason")


def merge_asset(existing: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Replace the seasons present in `new`, keep every other committed season (a season whose
    pull came back empty is therefore kept). Season-less assets (venues) are replaced whole
    when `new` has rows."""
    if new.empty:
        return existing.copy() if existing is not None else new.copy()
    if existing is None or existing.empty:
        out = new.copy()
    elif "season" in new.columns:
        out = pd.concat([existing[~existing["season"].isin(new["season"].unique())], new],
                        ignore_index=True)
    else:
        out = new.copy()
    return out.drop_duplicates(subset=keys, keep="last").sort_values(keys).reset_index(drop=True)


def describe_shape(obj, depth: int = 3):
    """Key/type skeleton of a decoded JSON value (values are never shown) -- the --probe output."""
    if isinstance(obj, dict) and depth > 0:
        return {k: describe_shape(v, depth - 1) for k, v in obj.items()}
    if isinstance(obj, list):
        return [describe_shape(obj[0], depth)] if obj else []
    return type(obj).__name__


def _pull(client, dataset: str, year: int, meta: pd.DataFrame | None) -> pd.DataFrame:
    parts = []
    if dataset == "games":
        parts = [cg.parse_games_meta(client.get("/games", {"year": year, "seasonType": st}))
                 for st in SEASON_TYPES]
    elif dataset == "havoc":
        parts = [cg.parse_havoc_games(client.get("/stats/game/havoc", {"year": year, "seasonType": st}))
                 for st in SEASON_TYPES]
    elif dataset == "drives":
        if meta is None or meta.empty:
            raise SystemExit("drives needs cfbd_games.parquet (run the `games` dataset first)")
        parts = [cg.parse_drive_games(client.get("/drives", {"year": year, "seasonType": st}), meta)
                 for st in SEASON_TYPES]
    elif dataset == "weather":
        parts = [cg.parse_weather_games(client.get("/games/weather", {"year": year, "seasonType": st}))
                 for st in SEASON_TYPES]
    elif dataset == "talent":
        parts = [cg.parse_talent(client.get("/talent", {"year": year}))]
    elif dataset == "ratings":
        parts = [cg.parse_prior_ratings(client.get("/ratings/fpi", {"year": year}),
                                        client.get("/ratings/srs", {"year": year}), year)]
    df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    df.attrs["dropped"] = sum(p.attrs.get("dropped", 0) for p in parts)
    return df


def run(client, datasets: list[str], seasons: list[int], out_dir: Path) -> dict:
    """Pull `datasets` for `seasons`, merge into the committed parquet under out_dir
    (seasons not pulled are kept), and return {dataset: rows in the written file}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wrote = {}
    games_path = out_dir / FILES["games"]
    meta = pd.read_parquet(games_path) if games_path.exists() else None
    for ds in [d for d in ORDER if d in datasets]:
        path = out_dir / FILES[ds]
        existing = pd.read_parquet(path) if path.exists() else None
        if ds == "venues":
            new = cg.parse_venues(client.get("/venues"))
        else:
            frames = [_pull(client, ds, y, meta) for y in seasons]
            new = pd.concat(frames, ignore_index=True)
            print(f"{ds}: {len(new)} rows pulled for {seasons[0]}-{seasons[-1]} "
                  f"({sum(f.attrs['dropped'] for f in frames)} dropped: unmapped/FCS/unknown game)",
                  flush=True)
            for y in seasons:
                if not (new["season"] == y).any():
                    kept = existing is not None and (existing["season"] == y).any()
                    print(f"::warning::build-cfb-game-data: {ds} {y} returned no rows"
                          + ("; KEEPING the committed rows" if kept else ""), flush=True)
        out = merge_asset(existing, new, KEYS[ds])
        out.to_parquet(path)
        wrote[ds] = len(out)
        if ds == "games":
            meta = out
        print(f"wrote {len(out)} rows -> {path.name}", flush=True)
    print(client.summary(), flush=True)
    return wrote


def probe(client, year: int = 2023) -> None:
    """One real call per endpoint: print the key/type skeleton of the first record."""
    import json
    calls = {"/games": {"year": year, "seasonType": "regular", "week": 3},
             "/stats/game/havoc": {"year": year, "week": 3},
             "/drives": {"year": year, "week": 3},
             "/games/weather": {"year": year, "week": 3},
             "/talent": {"year": year}, "/venues": None,
             "/ratings/fpi": {"year": year}, "/ratings/srs": {"year": year},
             "/stats/game/advanced": {"year": year, "week": 3}}
    for path, params in calls.items():
        data = client.get(path, params)
        print(f"== {path} {params or ''}: {len(data)} records")
        print(json.dumps(describe_shape(data[:1]), indent=1)[:2500])
    print(client.summary())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[dt.date.today().year])
    ap.add_argument("--datasets", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--probe", action="store_true", help="print one record's key/type skeleton per endpoint")
    args = ap.parse_args(argv)
    client = CfbdClient.from_env()
    if args.probe:
        probe(client)
        return
    run(client, args.datasets, sorted(args.seasons), ASSETS)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement the checks**

Create `src/sportsmodel/cfb/data_checks.py`:

```python
"""Sanity checks of the freshly backfilled CFBD game-data assets (PURE; returns problem strings).

The parsers were written from the CFBD OpenAPI document, not from live responses, so the backfill
step runs these checks to catch what a schema cannot say: units (temperature / wind / precipitation),
which side of the havoc object is "suffered" vs "created", drive points, and coverage. An empty list
means the assets look right; any problem string must be resolved (fix the parser / mapping and
re-pull) before fitting anything on them.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TEMP_F = (40.0, 80.0)            # season mean of outdoor game temperatures, degrees F
WIND_MPH = (2.0, 16.0)           # season mean wind
PRECIP_MAX = 30.0                # a single reading above this is not inches (or mm of a real game)
HAVOC_RATE = (0.04, 0.30)
HAVOC_PAIR_MAX_GAP = 0.02        # mean |def_havoc[A] - off_havoc[B]|: the two sides of one game agree
POINTS_PER_DRIVE = (1.0, 3.5)
START_YD = (20.0, 40.0)
MIN_COVERAGE = 0.80              # share of a season's FBS-vs-FBS games a per-game asset must cover
COVERAGE_FROM_SEASON = 2018      # CFBD weather / havoc history is patchier before this (spec risk)


def _seasons(df):
    return sorted(df["season"].unique())


def check_weather(w: pd.DataFrame) -> list[str]:
    out = []
    if w.empty:
        return ["weather: no rows"]
    for s in _seasons(w):
        d = w[(w["season"] == s) & ~w["game_indoors"]]
        temp, wind, pr = d["temperature"].dropna(), d["wind_speed"].dropna(), d["precipitation"].dropna()
        if len(temp) and not TEMP_F[0] <= temp.mean() <= TEMP_F[1]:
            out.append(f"weather {s}: mean outdoor temperature {temp.mean():.1f} is not plausible degrees F "
                       f"({TEMP_F}); Celsius? convert in parse_weather_games")
        if len(wind) and not WIND_MPH[0] <= wind.mean() <= WIND_MPH[1]:
            out.append(f"weather {s}: mean wind {wind.mean():.1f} is not plausible mph ({WIND_MPH}); "
                       "m/s or km/h? convert in parse_weather_games")
        if len(pr) and (pr.max() > PRECIP_MAX or pr.min() < 0):
            out.append(f"weather {s}: precipitation range {pr.min():.2f}..{pr.max():.2f} looks wrong")
    return out


def check_havoc(h: pd.DataFrame) -> list[str]:
    out = []
    if h.empty:
        return ["havoc: no rows"]
    for s in _seasons(h):
        d = h[h["season"] == s]
        for col in ("off_havoc", "def_havoc"):
            m = d[col].dropna().mean()
            if not HAVOC_RATE[0] <= m <= HAVOC_RATE[1]:
                out.append(f"havoc {s}: mean {col} {m:.3f} outside {HAVOC_RATE}")
    j = h.merge(h, left_on=["game_id", "opponent"], right_on=["game_id", "team"], suffixes=("", "_opp"))
    gap = (j["def_havoc"] - j["off_havoc_opp"]).abs().dropna()
    if len(gap) and gap.mean() > HAVOC_PAIR_MAX_GAP:
        out.append(f"havoc: team's DEFENSE havoc differs from its opponent's OFFENSE havoc by {gap.mean():.3f} "
                   "on average -- the offense/defense sides may be the other way round")
    return out


def check_drives(d: pd.DataFrame) -> list[str]:
    out = []
    if d.empty:
        return ["drives: no rows"]
    for s in _seasons(d):
        x = d[d["season"] == s]
        ppd, sy = x["off_ppd"].dropna().mean(), x["off_start_yd"].dropna().mean()
        if not POINTS_PER_DRIVE[0] <= ppd <= POINTS_PER_DRIVE[1]:
            out.append(f"drives {s}: mean points/drive {ppd:.2f} outside {POINTS_PER_DRIVE}")
        if not START_YD[0] <= sy <= START_YD[1]:
            out.append(f"drives {s}: mean start yard line {sy:.1f} outside {START_YD}")
    j = d.merge(d, left_on=["game_id", "opponent"], right_on=["game_id", "team"], suffixes=("", "_opp"))
    gap = (j["def_points"] - j["off_points_opp"]).abs().dropna()
    if len(gap) and gap.max() > 0:
        out.append("drives: a team's def_points differs from its opponent's off_points")
    return out


def coverage(asset: pd.DataFrame, sched: pd.DataFrame, per: str = "team") -> pd.DataFrame:
    """Per season: share of REG FBS-vs-FBS schedule games `asset` covers (per='team': both team rows;
    per='game': one row per game)."""
    s = sched[(sched["game_type"] == "REG") & (sched["home_team"] != "FCS") & (sched["away_team"] != "FCS")]
    n = asset.groupby(["season", "game_id"]).size()
    need = 2 if per == "team" else 1
    ok = {k for k, v in n.items() if v >= need}
    s = s.assign(covered=[(int(a), int(b)) in ok for a, b in zip(s["season"], s["game_pk"])])
    g = s.groupby("season")["covered"].mean().rename("share")
    return g[g.index.isin(asset["season"].unique())].to_frame()


def check_coverage(assets: dict, sched: pd.DataFrame) -> list[str]:
    out = []
    for name, df, per in (("advanced", assets.get("advanced"), "team"), ("havoc", assets.get("havoc"), "team"),
                          ("drives", assets.get("drives"), "team"), ("weather", assets.get("weather"), "game")):
        if df is None or df.empty:
            out.append(f"{name}: asset missing or empty")
            continue
        reg = df[df["season_type"] == "regular"] if "season_type" in df else df
        cov = coverage(reg, sched, per)
        low = [int(s) for s, v in cov["share"].items() if s >= COVERAGE_FROM_SEASON and v < MIN_COVERAGE]
        if low:
            out.append(f"{name}: coverage below {MIN_COVERAGE:.0%} of FBS-vs-FBS games in seasons {low}")
    return out


def check_all(assets: dict, sched: pd.DataFrame) -> list[str]:
    """assets: {advanced, havoc, drives, weather, ...} frames (None = missing)."""
    out = check_coverage(assets, sched)
    for key, fn in (("weather", check_weather), ("havoc", check_havoc), ("drives", check_drives)):
        if assets.get(key) is not None:
            out += fn(assets[key])
    return out
```

Create `scripts/check_cfb_game_data.py`:

```python
"""Sanity-check the committed CFBD game-data assets (units, havoc sides, drive points, coverage).

Exit 0 and print "OK" when everything looks right; otherwise print each problem and exit 1.
Run it after every backfill (plan Task 4) and before fitting. Local only (no network, no key).

Usage:
    uv run python scripts/check_cfb_game_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import data_checks, v3_data  # noqa: E402


def main() -> int:
    names = {"advanced": "advanced_games.parquet", "havoc": "havoc_games.parquet",
             "drives": "drive_games.parquet", "weather": "weather_games.parquet"}
    assets = {k: v3_data.read_asset(f) for k, f in names.items()}
    problems = data_checks.check_all(assets, v3_data.require_asset("schedules.parquet"))
    for k, df in assets.items():
        print(f"{k}: " + ("missing" if df is None else f"{len(df)} rows, seasons {df.season.min()}-{df.season.max()}"))
    if problems:
        print("\nPROBLEMS:\n- " + "\n- ".join(problems))
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_build_cfb_game_data.py tests/cfb/test_data_checks.py -q`
Expected: PASS (6 + 4).

- [ ] **Step 6: Commit**

```bash
git add scripts/build_cfb_game_data.py scripts/check_cfb_game_data.py src/sportsmodel/cfb/data_checks.py tests/scripts/test_build_cfb_game_data.py tests/cfb/test_data_checks.py
git commit -m "feat(cfb): build_cfb_game_data (games/havoc/drives/weather/talent/ratings/venues) + asset sanity checks"
```

---

### Task 4: [NETWORK] Probe the real CFBD shapes, backfill, verify, commit the assets

No new code unless the probe or the checks show a shape / unit that differs from what the parsers assume. The executing agent runs this task itself (needs `CFBD_API_KEY`; about 170 CFBD calls in total, far below the 75,000/month budget). The response shapes the parsers rely on were read from the public CFBD OpenAPI document and were **not** verified live when this plan was written.

**Files:**
- Create (data, committed): `assets/cfb/cfbd_games.parquet`, `havoc_games.parquet`, `drive_games.parquet`, `weather_games.parquet`, `talent.parquet`, `prior_ratings.parquet`, `venues.parquet`
- Modify (data, committed): `assets/cfb/advanced_games.parquet` (regular + postseason, new columns)
- Modify only if the probe/checks demand it: `src/sportsmodel/cfb/cfbd_games.py`, `tests/cfb/test_cfbd_games.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Produces: the committed assets above, with `season_type` / `week` / `game_id` on every game-level row; `check_cfb_game_data.py` printing `OK`.

- [ ] **Step 1: [NETWORK] Probe the live response shapes (9 calls)**

```bash
set -a; source .env; set +a      # loads CFBD_API_KEY without printing it
uv run python scripts/build_cfb_game_data.py --probe
```
Expected: for each of `/games`, `/stats/game/havoc`, `/drives`, `/games/weather`, `/talent`, `/venues`, `/ratings/fpi`, `/ratings/srs`, `/stats/game/advanced` a key/type skeleton of the first record. Compare against the parsers: `havoc` has `offense`/`defense` objects with `havocRate, frontSevenHavocRate, dbHavocRate, totalHavocEvents, totalPlays`; `drives` has `offense, defense, gameId, startYardsToGoal, endYardsToGoal, startOffenseScore, endOffenseScore, driveResult`; `weather` has `id, season, week, seasonType, gameIndoors, homeTeam, awayTeam, venueId, temperature, windSpeed, precipitation`; `games` has `id, season, week, seasonType, neutralSite, venueId, homeTeam, awayTeam, homePoints, awayPoints, homePregameElo, awayPregameElo`; `venues` has `id, timezone, latitude, longitude, elevation, dome`; `talent` has `year, team, talent`. If a field name or nesting differs: change the inline fixture in `tests/cfb/test_cfbd_games.py` to the real shape first, run it to see it fail, then fix the parser (TDD), and note the difference in the commit message.

- [ ] **Step 2: [NETWORK] Backfill advanced stats (regular + postseason), 2015-2026**

```bash
set -a; source .env; set +a
uv run python scripts/build_cfb_advanced.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```
Expected: one line per season (`<year>: N team-games kept (regular, postseason), ...`), a final `wrote ... team-game rows`, a coverage table. 2026 has regular-season rows only until bowls.

- [ ] **Step 3: [NETWORK] Backfill the per-game datasets, 2015-2026**

```bash
set -a; source .env; set +a
uv run python scripts/build_cfb_game_data.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```
Expected: per dataset `rows pulled ... (D dropped: unmapped/FCS/unknown game)`, `wrote N rows -> <asset>`, then `CFBD calls this run: ~160`. A `::warning::` for a season with no rows (e.g. havoc / weather before ~2018, or postseason 2026) is expected - those features stay NaN for that season and are handled by the missing-indicator / neutral logic.

- [ ] **Step 4: Verify units, havoc sides, drive points and coverage**

Run: `uv run python scripts/check_cfb_game_data.py`
Expected: `OK` (exit 0). If it prints PROBLEMS, resolve each before continuing - do not fit on unverified units:
- `degrees F` / `mph` problem -> convert in `parse_weather_games` (add a test with the real units first), re-pull weather.
- `other way round` (havoc sides) -> swap `off_*` / `def_*` in `parse_havoc_games` (test first), re-pull havoc.
- coverage below 80% for a season >= 2018 -> check `dropped` counts and `cfbd_to_espn` coverage for that season's teams (add aliases to `teams._CFBD_ALIASES` with a test in `tests/cfb/test_teams.py`), re-pull.

- [ ] **Step 5: Run the touched tests**

Run: `uv run pytest tests/cfb tests/scripts/test_build_cfb_advanced.py tests/scripts/test_build_cfb_game_data.py tests/scripts/test_build_team_context.py -q`
Expected: PASS.

- [ ] **Step 6: Commit the data**

```bash
git add assets/cfb/advanced_games.parquet assets/cfb/cfbd_games.parquet assets/cfb/havoc_games.parquet assets/cfb/drive_games.parquet assets/cfb/weather_games.parquet assets/cfb/talent.parquet assets/cfb/prior_ratings.parquet assets/cfb/venues.parquet
git commit -m "data(cfb): backfill CFBD advanced (+postseason), games meta, havoc, drives, weather, talent, ratings, venues (2015-2026)"
```
(Add `src/` / `tests/` files too if Step 1 or 4 required a parser change.)

---

### Task 5: Efficiency ratings - ridge adjustment, prior, blend, matchup features

**Files:**
- Create: `src/sportsmodel/cfb/efficiency.py`
- Test: `tests/cfb/test_efficiency.py`

**Interfaces:**
- Consumes: `priors.DecayConfig`, `priors.prior_weight(games_played, cfg)`, `priors.season_features_z(rows)`, `priors.zscore(dict)`.
- Produces:
  - `METRICS = ("ppa", "pass_ppa", "rush_ppa", "success", "explosiveness", "havoc", "ppo")`, `POINT_FEATURES = ("ppa_plays", "success", "explosiveness", "ppo", "havoc", "plays_dev")`.
  - `EffConfig(ridge=4.0, half_life_games=3.0, prior_floor=0.0, k0=0.6, k_ret=0.0, k_tal=0.0)` (+ `.decay -> DecayConfig`), `load_eff_config(path) -> EffConfig` (missing file -> defaults).
  - `MetricRatings(off: dict, deff: dict, mu: float, hfa: float)`, `EffState(ratings, games, pace, rush_share, lg_plays, lg_rush_share)`.
  - `ridge_adjust(obs: DataFrame[team, opponent, home, y, w], ridge: float) -> MetricRatings` (NaN y/w rows dropped; none left -> `EMPTY`).
  - `raw_state(games: DataFrame, ridge) -> EffState`, `retention_for(teams, z_ret, z_tal, cfg) -> dict`, `build_prior(final, retention) -> EffState`, `season_prior(eff_games, season, priors_rows, talent, cfg, _cache=None) -> EffState | None`, `blend_state(cur, prior, cfg) -> EffState`, `state_before(eff_games, season, week, prior, cfg, _cache=None) -> EffState`.
  - `side_features(state, x: str, y: str, hs: float) -> dict[POINT_FEATURES -> float]` (offense x vs defense y; hs = +1 home / -1 away / 0 neutral).
  - `build_eff_games(adv, havoc, drives, meta, sched) -> DataFrame` with columns `season, game_id, team, opponent, season_type, week (NaN for postseason), home, plays, rush_plays, pass_plays, y_<metric>, w_<metric>`.

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_efficiency.py`:

```python
"""Efficiency ratings: ridge recovery on a synthetic league, NaN handling, prior blend, leak invariant."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import efficiency as eff
from sportsmodel.cfb.priors import DecayConfig, prior_weight


def synthetic_obs(n_teams=16, n_games=3000, noise=0.02, seed=0, hfa=0.05, mu=0.1):
    rng = np.random.default_rng(seed)
    off, dfn = rng.normal(0, 0.2, n_teams), rng.normal(0, 0.2, n_teams)
    rows = []
    for _ in range(n_games):
        a, b = rng.choice(n_teams, 2, replace=False)
        h = int(rng.choice([-1, 0, 1]))
        rows.append((str(a), str(b), h, mu + off[a] - dfn[b] + hfa * h + rng.normal(0, noise), 60.0))
    return pd.DataFrame(rows, columns=["team", "opponent", "home", "y", "w"]), off, dfn


def test_ridge_recovers_known_offsets_and_home_edge():
    obs, off, dfn = synthetic_obs()
    r = eff.ridge_adjust(obs, ridge=0.5)
    t = len(off)
    assert np.corrcoef([r.off[str(i)] for i in range(t)], off)[0, 1] > 0.99
    assert np.corrcoef([r.deff[str(i)] for i in range(t)], dfn)[0, 1] > 0.99
    assert r.hfa == pytest.approx(0.05, abs=0.01)
    assert r.mu == pytest.approx(obs["y"].mean(), abs=1e-9)         # equal weights -> plain mean


def test_bigger_ridge_shrinks_ratings_toward_zero():
    obs, *_ = synthetic_obs(n_games=200)
    lo, hi = eff.ridge_adjust(obs, 0.5), eff.ridge_adjust(obs, 50.0)
    assert np.mean(np.abs(list(hi.off.values()))) < np.mean(np.abs(list(lo.off.values())))


def test_nan_rows_are_dropped_not_zeroed_and_empty_is_nan_mu():
    obs, *_ = synthetic_obs(n_games=300)
    base = eff.ridge_adjust(obs, 2.0)
    junk = pd.DataFrame([("0", "1", 1, np.nan, 60.0), ("0", "1", 1, 0.5, np.nan), ("0", "1", 1, 0.5, 0.0)],
                        columns=obs.columns)
    got = eff.ridge_adjust(pd.concat([obs, junk], ignore_index=True), 2.0)
    assert got.off == base.off and got.mu == base.mu
    empty = eff.ridge_adjust(obs.iloc[0:0], 2.0)
    assert math.isnan(empty.mu) and empty.off == {}


def test_weights_pull_the_fit_toward_heavy_observations():
    obs = pd.DataFrame([("a", "b", 0, 1.0, 100.0), ("a", "b", 0, 0.0, 1.0)],
                       columns=["team", "opponent", "home", "y", "w"])
    r = eff.ridge_adjust(obs, 0.01)
    assert r.mu > 0.95                                                # heavy row dominates the league mean


# ----------------------------------------------------------- eff_games frames --

def league_games(seasons=(2022, 2023), weeks=4, n_teams=8, seed=3):
    """team-game rows (one per team per game) on a tiny league with real column names."""
    rng = np.random.default_rng(seed)
    off = {s: rng.normal(0, 0.2, n_teams) for s in seasons}
    rows, gid = [], 1000
    for s in seasons:
        for w in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for i in range(0, n_teams, 2):
                h, a = int(order[i]), int(order[i + 1])
                gid += 1
                for team, opp, hs in ((h, a, 1), (a, h, -1)):
                    y = 0.1 + off[s][team] - 0.5 * off[s][opp] + 0.03 * hs
                    rows.append({"season": s, "game_id": gid, "team": str(team), "opponent": str(opp),
                                 "season_type": "regular", "week": float(w), "home": float(hs),
                                 "plays": 70.0, "rush_plays": 35.0, "pass_plays": 35.0,
                                 **{f"y_{m}": y for m in eff.METRICS}, **{f"w_{m}": 60.0 for m in eff.METRICS}})
    return pd.DataFrame(rows)


def test_state_before_uses_only_earlier_weeks_and_ignores_postseason():
    g = league_games()
    cfg = eff.EffConfig()
    base = eff.state_before(g, 2023, 3, None, cfg)
    assert set(base.games.values()) == {2}                            # weeks 1-2 only
    later = g[(g.season == 2023) & (g.week == 3)].copy()
    later["y_ppa"] += 5.0                                             # week 3 data must not matter
    post = g[(g.season == 2023) & (g.week == 1)].copy()
    post["season_type"], post["week"], post["y_ppa"] = "postseason", np.nan, 9.0
    for extra in (later, post):
        got = eff.state_before(pd.concat([g, extra], ignore_index=True), 2023, 3, None, cfg)
        assert got.ratings["ppa"].off == base.ratings["ppa"].off
        assert got.games == base.games


def test_state_before_empty_week_one_is_prior_only():
    g = league_games()
    cfg = eff.EffConfig(k0=0.5)
    final = eff.raw_state(g[g.season == 2022], cfg.ridge)
    prior = eff.build_prior(final, {t: 0.5 for t in final.games})
    st = eff.state_before(g, 2023, 1, prior, cfg)
    t = "0"
    assert st.ratings["ppa"].off[t] == pytest.approx(0.5 * final.ratings["ppa"].off[t])
    assert st.games == {}


def test_blend_weight_follows_prior_weight_curve():
    cfg = eff.EffConfig(half_life_games=2.0, prior_floor=0.1)
    prior = eff.EffState({m: eff.MetricRatings({"a": 1.0}, {"a": 0.0}, 0.0, 0.0) for m in eff.METRICS},
                         {}, {"a": 80.0}, {"a": 0.4}, 70.0, 0.5)
    cur = eff.EffState({m: eff.MetricRatings({"a": 0.0}, {"a": 0.0}, 0.0, 0.0) for m in eff.METRICS},
                       {"a": 4}, {"a": 60.0}, {"a": 0.6}, 70.0, 0.5)
    w = prior_weight(4, DecayConfig(2.0, 0.1))
    out = eff.blend_state(cur, prior, cfg)
    assert out.ratings["ppa"].off["a"] == pytest.approx(w * 1.0)
    assert out.pace["a"] == pytest.approx(w * 80.0 + (1 - w) * 60.0)
    assert out.rush_share["a"] == pytest.approx(w * 0.4 + (1 - w) * 0.6)
    assert eff.blend_state(cur, None, cfg) is cur


def test_retention_is_clipped_and_scaled_by_returning_and_talent():
    cfg = eff.EffConfig(k0=0.6, k_ret=0.2, k_tal=0.1)
    k = eff.retention_for(["a", "b", "c"], {"a": 3.0, "b": -5.0}, {"a": 1.0}, cfg)
    assert k["a"] == 1.0 and k["b"] == 0.0 and k["c"] == pytest.approx(0.6)


def test_season_prior_uses_previous_season_final_and_z_scored_inputs():
    g = league_games()
    cfg = eff.EffConfig(k0=0.5, k_ret=0.1, k_tal=0.0)
    rows = [{"team_espn_id": str(t), "returning_pct": 0.2 + 0.1 * t, "recruiting_points": None,
             "portal_net": None, "prior_sos": None} for t in range(8)]
    prior = eff.season_prior(g, 2023, rows, None, cfg)
    final = eff.raw_state(g[g.season == 2022], cfg.ridge)
    vals = np.array([0.2 + 0.1 * t for t in range(8)])
    z7 = (vals[7] - vals.mean()) / vals.std()
    assert prior.ratings["ppa"].off["7"] == pytest.approx(
        min(1.0, max(0.0, 0.5 + 0.1 * z7)) * final.ratings["ppa"].off["7"])
    assert eff.season_prior(g, 2022, rows, None, cfg) is None          # no 2021 data -> no prior


# -------------------------------------------------------- matchup features --

def test_side_features_are_deviations_and_missing_metrics_are_neutral():
    r = eff.MetricRatings({"x": 0.3}, {"y": 0.1}, 0.1, 0.02)
    st = eff.EffState({"rush_ppa": r, "pass_ppa": r, "success": r}, {"x": 5, "y": 5},
                      {"x": 80.0, "y": 60.0}, {"x": 0.25}, 70.0, 0.5)
    f = eff.side_features(st, "x", "y", 1.0)
    assert f["success"] == pytest.approx(0.3 - 0.1 + 0.02)
    plays = 70.0 + 0.5 * ((80 - 70) + (60 - 70))
    assert f["plays_dev"] == pytest.approx(plays - 70.0)
    assert f["ppa_plays"] == pytest.approx((0.3 - 0.1 + 0.02) * plays)
    assert f["havoc"] == 0.0 and f["ppo"] == 0.0 and f["explosiveness"] == 0.0   # absent metrics: 0, not NaN
    assert eff.side_features(st, "x", "y", -1.0)["success"] == pytest.approx(0.3 - 0.1 - 0.02)


# ----------------------------------------------------------------- the frame --

def test_build_eff_games_joins_home_week_and_nan_for_missing_metrics():
    adv = pd.DataFrame({
        "season": [2023] * 4, "week": [1, 1, 1, 1], "season_type": ["regular", "regular", "postseason", "regular"],
        "game_id": [1, 1, 2, 3], "team": ["a", "b", "a", "a"], "opponent": ["b", "a", "b", "b"],
        "off_plays": [70.0, 60.0, 65.0, 50.0], "off_ppa": [0.2, 0.1, 0.3, 0.0],
        "off_success": [0.5, 0.4, 0.5, 0.4], "off_explosiveness": [1.2, 1.1, 1.3, 1.0],
        "off_pass_ppa": [0.3, 0.2, 0.3, 0.0], "off_rush_ppa": [0.1, 0.0, 0.2, 0.0],
        "off_pass_plays": [40.0, np.nan, 30.0, 20.0], "off_rush_plays": [30.0, 30.0, 35.0, 30.0]})
    havoc = pd.DataFrame({"season": [2023], "game_id": [1], "team": ["a"], "off_havoc": [0.11], "off_plays": [72.0]})
    meta = pd.DataFrame({"game_id": [1, 2], "home_team": ["a", "b"], "neutral_site": [False, True]})
    sched = pd.DataFrame({"game_pk": [1, 3], "week": [2, 5], "game_type": ["REG", "REG"]})
    g = eff.build_eff_games(adv, havoc, None, meta, sched)
    assert len(g) == 3                                               # game 3 unknown to meta -> dropped
    r_a, r_b, r_post = g.iloc[0], g.iloc[1], g.iloc[2]
    assert (r_a["home"], r_b["home"], r_post["home"]) == (1.0, -1.0, 0.0)   # neutral bowl -> 0
    assert r_a["week"] == 2.0 and math.isnan(r_post["week"])         # schedule week; postseason NaN
    assert r_a["y_havoc"] == 0.11 and r_a["w_havoc"] == 72.0
    assert math.isnan(r_b["y_havoc"]) and r_b["w_havoc"] == 60.0     # no havoc row: NaN y, plays as weight
    assert math.isnan(r_a["y_ppo"])                                   # no drives asset: NaN
    assert r_b["w_pass_ppa"] == 30.0                                  # missing pass count -> half of plays
    assert r_a["w_pass_ppa"] == 40.0


def test_load_eff_config_roundtrip_and_default(tmp_path):
    assert eff.load_eff_config(tmp_path / "missing.json") == eff.EffConfig()
    p = tmp_path / "eff_config.json"
    p.write_text(json.dumps({"ridge": 8.0, "half_life_games": 2.0, "prior_floor": 0.1,
                             "k0": 0.7, "k_ret": 0.05, "k_tal": 0.0}))
    assert eff.load_eff_config(p) == eff.EffConfig(8.0, 2.0, 0.1, 0.7, 0.05, 0.0)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_efficiency.py -q`
Expected: FAIL at collection: `ImportError: cannot import name 'efficiency' from 'sportsmodel.cfb'`.

- [ ] **Step 3: Implement**

Create `src/sportsmodel/cfb/efficiency.py`:

```python
"""Opponent-adjusted per-play efficiency ratings for CFB (cfb-ratings-v3). PURE.

One weighted ridge regression per metric per week over the season's completed
team-games:

    y(team vs opp) = league_mean + off[team] - def[opp] + hfa * home

(`home` is +1 for the home team, -1 for the away team, 0 at a neutral site; `off` > 0 is a
better offense, `def` > 0 a better defense.) Observations are weighted by plays (scoring
opportunities for `ppo`), normalised to mean 1, and ridge-shrunk toward the league mean with
`ridge` pseudo-games of strength.

Early season: the season-to-date ratings are blended with a prior = the team's previous-
season FINAL ratings (whole season, bowls included) shrunk by a per-team retention
k = clip(k0 + k_ret*z(returning production) + k_tal*z(talent), 0, 1). The blend weight on the
prior is `priors.prior_weight(games, decay)` (exponential half-life in games + floor, the
same family as priors_decay.json).

LEAK RULE: the state entering week W of season S is built from regular-season team-games
with schedule week < W of season S plus the previous season's final ratings; nothing else.
Appending a later game never changes an earlier state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportsmodel.cfb.priors import DecayConfig, prior_weight, season_features_z, zscore

METRICS = ("ppa", "pass_ppa", "rush_ppa", "success", "explosiveness", "havoc", "ppo")
LEAGUE_PLAYS_SEED = 70.0
LEAGUE_RUSH_SHARE_SEED = 0.5
RETENTION_BOUNDS = (0.0, 1.0)


@dataclass(frozen=True)
class EffConfig:
    ridge: float = 4.0            # pseudo-games of shrinkage toward the league mean
    half_life_games: float = 3.0  # prior weight halves after this many games
    prior_floor: float = 0.0      # prior weight never drops below this
    k0: float = 0.6               # baseline retention of last season's rating
    k_ret: float = 0.0            # + per z of returning production
    k_tal: float = 0.0            # + per z of talent composite

    @property
    def decay(self) -> DecayConfig:
        return DecayConfig(half_life_games=self.half_life_games, prior_floor=self.prior_floor)


@dataclass(frozen=True)
class MetricRatings:
    off: dict
    deff: dict
    mu: float
    hfa: float


EMPTY = MetricRatings({}, {}, float("nan"), 0.0)


@dataclass(frozen=True)
class EffState:
    ratings: dict        # metric -> MetricRatings
    games: dict          # team -> team-games used this season
    pace: dict           # team -> offensive plays per game
    rush_share: dict     # team -> rushes / (rushes + passes)
    lg_plays: float
    lg_rush_share: float


# ------------------------------------------------------------------ the ridge --

def ridge_adjust(obs: pd.DataFrame, ridge: float) -> MetricRatings:
    """obs columns: team, opponent, home, y, w. Rows with a NaN y or w (or w <= 0) are dropped,
    never treated as 0. No usable rows -> EMPTY."""
    obs = obs.dropna(subset=["y", "w"])
    obs = obs[obs["w"] > 0]
    if obs.empty:
        return EMPTY
    teams = sorted(set(obs["team"]) | set(obs["opponent"]))
    ix = {t: i for i, t in enumerate(teams)}
    t_n, n = len(teams), len(obs)
    w = obs["w"].to_numpy(float)
    w = w / w.mean()
    y = obs["y"].to_numpy(float)
    mu = float((w * y).sum() / w.sum())
    x = np.zeros((n, 2 * t_n + 1))
    r = np.arange(n)
    x[r, obs["team"].map(ix).to_numpy()] = 1.0
    x[r, t_n + obs["opponent"].map(ix).to_numpy()] = -1.0
    x[:, 2 * t_n] = obs["home"].to_numpy(float)
    xw = x * w[:, None]
    a = x.T @ xw
    diag = np.arange(2 * t_n)
    a[diag, diag] += ridge
    a[2 * t_n, 2 * t_n] += 1e-6            # hfa: unpenalised but kept invertible
    theta = np.linalg.solve(a, xw.T @ (y - mu))
    return MetricRatings({t: float(theta[ix[t]]) for t in teams},
                         {t: float(theta[t_n + ix[t]]) for t in teams}, mu, float(theta[2 * t_n]))


def metric_obs(games: pd.DataFrame, metric: str) -> pd.DataFrame:
    """The (team, opponent, home, y, w) observations of one metric from an eff_games frame."""
    nan = pd.Series(np.nan, index=games.index)
    return pd.DataFrame({"team": games["team"], "opponent": games["opponent"], "home": games["home"],
                         "y": games.get(f"y_{metric}", nan), "w": games.get(f"w_{metric}", nan)})


def raw_state(games: pd.DataFrame, ridge: float) -> EffState:
    """Season-only (no prior) state from team-games."""
    if games.empty:
        return EffState({m: EMPTY for m in METRICS}, {}, {}, {}, LEAGUE_PLAYS_SEED, LEAGUE_RUSH_SHARE_SEED)
    ratings = {m: ridge_adjust(metric_obs(games, m), ridge) for m in METRICS}
    rush = games.groupby("team")["rush_plays"].sum()
    pas = games.groupby("team")["pass_plays"].sum()
    share = (rush / (rush + pas)).dropna()
    tot = float(games["rush_plays"].sum() + games["pass_plays"].sum())
    lg_rs = float(games["rush_plays"].sum() / tot) if tot > 0 else LEAGUE_RUSH_SHARE_SEED
    plays = games["plays"].dropna()
    return EffState(ratings, games.groupby("team").size().to_dict(),
                    games.groupby("team")["plays"].mean().dropna().to_dict(), share.to_dict(),
                    float(plays.mean()) if len(plays) else LEAGUE_PLAYS_SEED, lg_rs)


# ------------------------------------------------------------- prior + blend --

def retention_for(teams, z_ret: dict, z_tal: dict, cfg: EffConfig) -> dict:
    lo, hi = RETENTION_BOUNDS
    return {t: float(np.clip(cfg.k0 + cfg.k_ret * z_ret.get(t, 0.0) + cfg.k_tal * z_tal.get(t, 0.0), lo, hi))
            for t in teams}


def build_prior(final: EffState, retention: dict) -> EffState:
    """Previous-season final state with every team's ratings shrunk by its retention."""
    ratings = {}
    for m, r in final.ratings.items():
        ratings[m] = MetricRatings({t: retention.get(t, 0.0) * v for t, v in r.off.items()},
                                   {t: retention.get(t, 0.0) * v for t, v in r.deff.items()},
                                   r.mu, r.hfa)
    return EffState(ratings, {}, dict(final.pace), dict(final.rush_share), final.lg_plays, final.lg_rush_share)


def season_prior(eff_games: pd.DataFrame, season: int, priors_rows: list[dict],
                 talent: pd.DataFrame | None, cfg: EffConfig,
                 _cache: dict | None = None) -> EffState | None:
    """The prior entering `season`: last season's final ratings (all its games, bowls included)
    shrunk by returning production (priors.parquet `returning_pct`, season-S preseason value) and
    the talent composite. None when there is no previous-season data (no prior). `_cache`
    (optional) memoises the previous season's ridge by (ridge, season - 1)."""
    prev = eff_games[eff_games["season"] == season - 1]
    if prev.empty:
        return None
    fkey = ("final", cfg.ridge, season - 1)
    if _cache is not None and fkey in _cache:
        final = _cache[fkey]
    else:
        final = raw_state(prev, cfg.ridge)
        if _cache is not None:
            _cache[fkey] = final
    z_ret = {t: z["returning_pct"] for t, z in season_features_z(priors_rows).items()} if priors_rows else {}
    tal = {}
    if talent is not None and len(talent):
        cur = talent[talent["season"] == season].dropna(subset=["talent"])
        tal = zscore({str(t): float(v) for t, v in zip(cur["team"], cur["talent"])}) if len(cur) else {}
    return build_prior(final, retention_for(final.games.keys(), z_ret, tal, cfg))


def blend_state(cur: EffState, prior: EffState | None, cfg: EffConfig) -> EffState:
    """Per-team decay blend: w * prior + (1 - w) * season-to-date, w = prior_weight(games)."""
    if prior is None:
        return cur
    ratings = {}
    for m in METRICS:
        c, p = cur.ratings.get(m, EMPTY), prior.ratings.get(m, EMPTY)
        off, deff = {}, {}
        for t in set(c.off) | set(p.off):
            w = prior_weight(cur.games.get(t, 0), cfg.decay)
            off[t] = w * p.off.get(t, 0.0) + (1 - w) * c.off.get(t, 0.0)
            deff[t] = w * p.deff.get(t, 0.0) + (1 - w) * c.deff.get(t, 0.0)
        ratings[m] = MetricRatings(off, deff, c.mu if c.mu == c.mu else p.mu, c.hfa if c.off else p.hfa)
    lg_plays = cur.lg_plays if cur.games else prior.lg_plays
    lg_rs = cur.lg_rush_share if cur.games else prior.lg_rush_share

    def mix(c: dict, p: dict, lg: float) -> dict:
        return {t: prior_weight(cur.games.get(t, 0), cfg.decay) * p.get(t, lg)
                + (1 - prior_weight(cur.games.get(t, 0), cfg.decay)) * c.get(t, lg)
                for t in set(c) | set(p)}

    return EffState(ratings, cur.games, mix(cur.pace, prior.pace, lg_plays),
                    mix(cur.rush_share, prior.rush_share, lg_rs), lg_plays, lg_rs)


def state_before(eff_games: pd.DataFrame, season: int, week: int,
                 prior: EffState | None, cfg: EffConfig, _cache: dict | None = None) -> EffState:
    """The efficiency state ENTERING schedule week `week` of `season`: regular-season team-games
    with week < `week` (NaN-week postseason rows never qualify) blended with `prior`.
    `_cache` (optional) memoises the season-only ridge by (ridge, season, week)."""
    key = (cfg.ridge, season, week)
    if _cache is not None and key in _cache:
        cur = _cache[key]
    else:
        rows = eff_games[(eff_games["season"] == season) & (eff_games["week"] < week)]
        cur = raw_state(rows, cfg.ridge)
        if _cache is not None:
            _cache[key] = cur
    return blend_state(cur, prior, cfg)


# --------------------------------------------------------- matchup features --

def _dev(state: EffState, metric: str, x: str, y: str, hs: float) -> float:
    r = state.ratings.get(metric, EMPTY)
    return r.off.get(x, 0.0) - r.deff.get(y, 0.0) + r.hfa * hs


def side_features(state: EffState, x: str, y: str, hs: float) -> dict:
    """Deviation-from-league features of offense `x` against defense `y` (hs: +1 home offense,
    -1 away offense, 0 neutral). Missing metrics / unrated teams contribute a 0.0 deviation, so a
    metric absent for a whole season is neutral rather than NaN. PPA per play mixes the rush and
    pass ratings by x's season-to-date rush share; plays is the pace both teams imply."""
    rs = state.rush_share.get(x, state.lg_rush_share)
    ppa_mix = rs * _dev(state, "rush_ppa", x, y, hs) + (1 - rs) * _dev(state, "pass_ppa", x, y, hs)
    plays = state.lg_plays + 0.5 * ((state.pace.get(x, state.lg_plays) - state.lg_plays)
                                    + (state.pace.get(y, state.lg_plays) - state.lg_plays))
    return {"ppa_plays": ppa_mix * plays, "success": _dev(state, "success", x, y, hs),
            "explosiveness": _dev(state, "explosiveness", x, y, hs), "ppo": _dev(state, "ppo", x, y, hs),
            "havoc": _dev(state, "havoc", x, y, hs), "plays_dev": plays - state.lg_plays}


POINT_FEATURES = ("ppa_plays", "success", "explosiveness", "ppo", "havoc", "plays_dev")


# ------------------------------------------------------------ the eff frame --

def build_eff_games(adv: pd.DataFrame, havoc: pd.DataFrame | None, drives: pd.DataFrame | None,
                    meta: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """One row per FBS team-game (regular season AND postseason) with the observations the ridge
    needs: y_<metric> / w_<metric>, plays, rush_plays, pass_plays, `home` (+1/-1/0) and `week`
    (the ESPN schedule week, NaN for postseason or games absent from the REG schedule -- those rows
    feed only the previous-season final ratings, never an in-season state)."""
    key = ["season", "game_id", "team"]
    a = adv.copy()
    a["season_type"] = a["season_type"].fillna("regular") if "season_type" in a else "regular"
    if havoc is not None and len(havoc):
        a = a.merge(havoc[key + ["off_havoc", "off_plays"]].rename(
            columns={"off_plays": "havoc_plays"}), on=key, how="left")
    else:
        a["off_havoc"], a["havoc_plays"] = np.nan, np.nan
    if drives is not None and len(drives):
        a = a.merge(drives[key + ["off_points_per_opp", "off_opps"]], on=key, how="left")
    else:
        a["off_points_per_opp"], a["off_opps"] = np.nan, np.nan
    a = a.merge(meta[["game_id", "home_team", "neutral_site"]], on="game_id", how="inner")
    a["home"] = np.where(a["neutral_site"], 0.0, np.where(a["team"] == a["home_team"], 1.0, -1.0))
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    wk = reg[["game_pk", "week"]].rename(columns={"game_pk": "game_id", "week": "sched_week"})
    a = a.drop(columns=["week"]).merge(wk, on="game_id", how="left")
    a["week"] = a["sched_week"].where(a["season_type"] == "regular")
    split_w = lambda col: a[col].fillna(0.5 * a["off_plays"])      # noqa: E731
    out = pd.DataFrame({
        "season": a["season"].astype("int64"), "game_id": a["game_id"].astype("int64"),
        "team": a["team"].astype(str), "opponent": a["opponent"].astype(str),
        "season_type": a["season_type"], "week": a["week"].astype("float64"), "home": a["home"],
        "plays": a["off_plays"], "rush_plays": a["off_rush_plays"], "pass_plays": a["off_pass_plays"],
        "y_ppa": a["off_ppa"], "w_ppa": a["off_plays"],
        "y_pass_ppa": a["off_pass_ppa"], "w_pass_ppa": split_w("off_pass_plays"),
        "y_rush_ppa": a["off_rush_ppa"], "w_rush_ppa": split_w("off_rush_plays"),
        "y_success": a["off_success"], "w_success": a["off_plays"],
        "y_explosiveness": a["off_explosiveness"], "w_explosiveness": a["off_plays"],
        "y_havoc": a["off_havoc"], "w_havoc": a["havoc_plays"].fillna(a["off_plays"]),
        "y_ppo": a["off_points_per_opp"], "w_ppo": a["off_opps"],
    })
    return out.reset_index(drop=True)


def load_eff_config(path) -> EffConfig:
    """EffConfig from assets/cfb/eff_config.json; a missing file -> the defaults."""
    import json
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        return EffConfig()
    return EffConfig(**{k: float(v) for k, v in json.loads(p.read_text()).items()})
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_efficiency.py -q`
Expected: PASS (12 tests: recovery, ridge shrinkage, NaN handling, weights, leak invariant (later weeks and postseason rows), prior-only week 1, blend curve, retention, season prior, side features, `build_eff_games`, `load_eff_config`).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/cfb/efficiency.py tests/cfb/test_efficiency.py
git commit -m "feat(cfb): opponent-adjusted efficiency ratings (weekly weighted ridge + previous-season prior)"
```

---

### Task 6: Asset loaders and the efficiency-hyperparameter fit

**Files:**
- Create: `src/sportsmodel/cfb/v3_data.py` (first part), `src/sportsmodel/cfb/eff_fit.py`, `scripts/fit_cfb_eff.py`
- Test: `tests/cfb/test_v3_data.py` (first three tests), `tests/cfb/test_eff_fit.py`

**Interfaces:**
- Consumes: `efficiency.build_eff_games`, `EffConfig`, `season_prior`, `state_before` (Task 5).
- Produces:
  - `v3_data.ASSETS`, `read_asset(name, assets=ASSETS) -> DataFrame | None`, `require_asset(name, assets=ASSETS) -> DataFrame` (FileNotFoundError naming the backfill), `load_merged_schedule(assets=ASSETS) -> DataFrame` (REG schedules x lines, self-matches dropped), `load_priors_rows(assets=ASSETS) -> dict[int, list[dict]]`, `load_eff_games(assets=ASSETS) -> DataFrame`.
  - `eff_fit.TRAIN_SEASONS`, `HOLDOUT_SEASONS`, `GRID`, `ORDER`; `ppa_holdout_loss(eff_games, seasons, cfg, priors_rows_by_season, talent, cache=None) -> float`; `fit_eff_config(eff_games, priors_rows_by_season, talent, train_seasons=TRAIN_SEASONS, grid=GRID, start=None, n_passes=4) -> (EffConfig, float)`.
  - `assets/cfb/eff_config.json` (`ridge, half_life_games, prior_floor, k0, k_ret, k_tal`).

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_v3_data.py` (more tests are appended to this file in Task 9):

```python
import pandas as pd
import pytest

from sportsmodel.cfb import v3_data


def write(tmp, name, df):
    df.to_parquet(tmp / name)


def test_merged_schedule_drops_self_matches_and_joins_lines(tmp_path):
    write(tmp_path, "schedules.parquet", pd.DataFrame({
        "season": [2023] * 3, "week": [1] * 3, "home_team": ["a", "b", "c"], "away_team": ["b", "a", "c"],
        "home_score": [1, 2, 3], "away_score": [0, 1, 2], "game_type": ["REG", "REG", "REG"],
        "game_pk": [1, 2, 3]}))
    write(tmp_path, "lines.parquet", pd.DataFrame({
        "season": [2023], "week": [1], "home_team": ["a"], "away_team": ["b"], "market_spread": [-3.0],
        "market_total": [50.0]}))
    m = v3_data.load_merged_schedule(tmp_path)
    assert list(m["game_pk"]) == [1, 2]                              # c-vs-c self match dropped
    assert m.loc[m.game_pk == 1, "market_spread"].iloc[0] == -3.0
    assert pd.isna(m.loc[m.game_pk == 2, "market_spread"].iloc[0])


def test_missing_required_asset_names_the_backfill(tmp_path):
    with pytest.raises(FileNotFoundError, match="CFBD_API_KEY"):
        v3_data.load_eff_games(tmp_path)
    assert v3_data.read_asset("talent.parquet", tmp_path) is None
    assert v3_data.load_priors_rows(tmp_path) == {}


def test_load_eff_games_works_without_optional_assets(tmp_path):
    write(tmp_path, "advanced_games.parquet", pd.DataFrame({
        "season": [2023, 2023], "week": [1, 1], "season_type": ["regular"] * 2, "game_id": [1, 1],
        "team": ["a", "b"], "opponent": ["b", "a"], "off_plays": [70.0, 60.0], "off_ppa": [0.2, 0.1],
        "off_success": [0.5, 0.4], "off_explosiveness": [1.2, 1.1], "off_pass_ppa": [0.2, 0.1],
        "off_rush_ppa": [0.1, 0.0], "off_pass_plays": [40.0, 30.0], "off_rush_plays": [30.0, 30.0]}))
    write(tmp_path, "cfbd_games.parquet", pd.DataFrame({"game_id": [1], "home_team": ["a"],
                                                         "neutral_site": [False]}))
    write(tmp_path, "schedules.parquet", pd.DataFrame({"game_pk": [1], "week": [1], "game_type": ["REG"]}))
    g = v3_data.load_eff_games(tmp_path)
    assert len(g) == 2 and g["y_havoc"].isna().all() and g["y_ppo"].isna().all()
    assert list(g["home"]) == [1.0, -1.0]
```

Create `tests/cfb/test_eff_fit.py`:

```python
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import eff_fit, efficiency as eff


def persistent_league(seasons=(2020, 2021, 2022, 2023), weeks=6, n_teams=12, seed=11):
    """Team strength persists across seasons (AR(1)), so a previous-season prior is informative."""
    rng = np.random.default_rng(seed)
    off = rng.normal(0, 0.25, n_teams)
    dfn = rng.normal(0, 0.25, n_teams)
    rows, gid = [], 0
    for s in seasons:
        off, dfn = 0.8 * off + rng.normal(0, 0.08, n_teams), 0.8 * dfn + rng.normal(0, 0.08, n_teams)
        for w in range(1, weeks + 1):
            order = rng.permutation(n_teams)
            for i in range(0, n_teams, 2):
                h, a = int(order[i]), int(order[i + 1])
                gid += 1
                for team, opp, hs in ((h, a, 1), (a, h, -1)):
                    y = 0.1 + off[team] - dfn[opp] + 0.03 * hs + rng.normal(0, 0.15)
                    rows.append({"season": s, "game_id": gid, "team": str(team), "opponent": str(opp),
                                 "season_type": "regular", "week": float(w), "home": float(hs),
                                 "plays": 70.0, "rush_plays": 35.0, "pass_plays": 35.0,
                                 **{f"y_{m}": y for m in eff.METRICS}, **{f"w_{m}": 60.0 for m in eff.METRICS}})
    return pd.DataFrame(rows)


def test_loss_is_finite_and_a_prior_helps_early_weeks():
    g = persistent_league()
    no_prior = eff_fit.ppa_holdout_loss(g, (2022, 2023), eff.EffConfig(k0=0.0), {}, None)
    with_prior = eff_fit.ppa_holdout_loss(g, (2022, 2023), eff.EffConfig(k0=0.8), {}, None)
    assert np.isfinite(no_prior) and np.isfinite(with_prior)
    assert with_prior < no_prior


def test_loss_only_uses_games_before_each_week():
    g = persistent_league()
    base = eff_fit.ppa_holdout_loss(g, (2022,), eff.EffConfig(), {}, None)
    # corrupting the LAST week's observations changes that week's targets but never an earlier
    # week's prediction: losses differ, yet the prediction path up to week 5 is identical
    last = g[(g.season == 2022) & (g.week == 6)].index
    g2 = g.copy()
    g2.loc[last, "y_ppa"] += 1.0
    assert eff_fit.ppa_holdout_loss(g2, (2022,), eff.EffConfig(), {}, None) != base
    g3 = g[~((g.season == 2022) & (g.week == 6))]
    g4 = g3.copy()
    g4_future = g[(g.season == 2023)].copy()
    g4_future["y_ppa"] += 3.0                       # a later season must not change 2022's loss
    assert eff_fit.ppa_holdout_loss(pd.concat([g3, g4_future]), (2022,), eff.EffConfig(), {}, None) == \
        eff_fit.ppa_holdout_loss(g3, (2022,), eff.EffConfig(), {}, None)


def test_fit_never_worse_than_start_and_stays_on_grid():
    g = persistent_league()
    grid = {"ridge": [1.0, 4.0, 16.0], "half_life_games": [1.0, 3.0], "prior_floor": [0.0, 0.3],
            "k0": [0.2, 0.8], "k_ret": [0.0], "k_tal": [0.0]}
    start = eff.EffConfig(k0=0.2)
    cfg, loss = eff_fit.fit_eff_config(g, {}, None, train_seasons=(2021, 2022), grid=grid, start=start)
    assert loss <= eff_fit.ppa_holdout_loss(g, (2021, 2022), start, {}, None) + 1e-12
    assert cfg.ridge in grid["ridge"] and cfg.k0 in (0.2, 0.8)
    assert cfg.k0 == 0.8                                   # persistent strengths -> keep most of last year
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_v3_data.py tests/cfb/test_eff_fit.py -q`
Expected: FAIL (`ImportError: cannot import name 'v3_data'` / `'eff_fit'`).

- [ ] **Step 3: Implement the loaders**

Create `src/sportsmodel/cfb/v3_data.py` (Task 9 appends `load_rating` / `load_v3_inputs` to it):

```python
"""Committed-asset loaders shared by the cfb-ratings-v3 fit, gate and serving code.

All assets live under assets/cfb/. The three the v3 walk cannot run without are
schedules.parquet, advanced_games.parquet and cfbd_games.parquet; everything else
(havoc, drives, weather, talent, venues, priors, prior ratings) is optional and a
missing file simply means that feature is NaN / neutral.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .. import config
from .efficiency import build_eff_games

ASSETS = config.PROJECT_ROOT / "assets" / "cfb"
BACKFILL_HINT = ("run scripts/build_cfb_advanced.py and scripts/build_cfb_game_data.py with "
                 "CFBD_API_KEY (plan Task 3) to create it")


def read_asset(name: str, assets: Path = ASSETS) -> pd.DataFrame | None:
    """assets/cfb/<name> as a DataFrame, or None when the file is absent."""
    p = Path(assets) / name
    return pd.read_parquet(p) if p.exists() else None


def require_asset(name: str, assets: Path = ASSETS) -> pd.DataFrame:
    df = read_asset(name, assets)
    if df is None:
        raise FileNotFoundError(f"{Path(assets) / name} is missing: {BACKFILL_HINT}")
    return df


def load_merged_schedule(assets: Path = ASSETS) -> pd.DataFrame:
    """REG schedules left-joined to closing/opening lines on (season, week, home, away), with the
    CFBD self-match rows (home == away) dropped from both sides first -- identical to
    backtest_cfb_gameline.load_merged_schedule, so the v2 walk-forward numbers are unchanged."""
    sched = require_asset("schedules.parquet", assets)
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    reg = reg[reg["home_team"] != reg["away_team"]].copy()
    lines = require_asset("lines.parquet", assets)
    lines = lines[lines["home_team"] != lines["away_team"]].drop_duplicates(
        subset=["season", "week", "home_team", "away_team"], keep="first")
    return reg.merge(lines, on=["season", "week", "home_team", "away_team"], how="left",
                     validate="one_to_one")


def load_priors_rows(assets: Path = ASSETS) -> dict[int, list[dict]]:
    """priors.parquet -> {season: [row dicts]} (empty dict when the asset is absent)."""
    df = read_asset("priors.parquet", assets)
    if df is None:
        return {}
    return {int(s): sdf.to_dict("records") for s, sdf in df.groupby("season")}


def load_eff_games(assets: Path = ASSETS) -> pd.DataFrame:
    """The efficiency frame (efficiency.build_eff_games) from the committed assets."""
    return build_eff_games(require_asset("advanced_games.parquet", assets),
                           read_asset("havoc_games.parquet", assets),
                           read_asset("drive_games.parquet", assets),
                           require_asset("cfbd_games.parquet", assets),
                           require_asset("schedules.parquet", assets))
```

- [ ] **Step 4: Implement the fit**

Create `src/sportsmodel/cfb/eff_fit.py`:

```python
"""Walk-forward fit of the efficiency-rating hyperparameters (cfb-ratings-v3). PURE.

Objective: one-week-ahead, play-weighted squared error of PPA per play. For every week W of
every training season the state entering W (season-to-date ridge + previous-season prior)
predicts each team-game of week W:  mu + off[team] - def[opp] + hfa * home.  PPA alone drives
the fit; the chosen (ridge, half-life, floor, k0, k_ret, k_tal) are shared by all metrics.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sportsmodel.cfb.efficiency import EffConfig, season_prior, state_before

TRAIN_SEASONS = tuple(range(2016, 2023))      # 2016-2022 (2015 has no previous-season prior)
HOLDOUT_SEASONS = (2023, 2024, 2025)

GRID = {
    "ridge": [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
    "half_life_games": [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
    "prior_floor": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    "k0": [round(0.3 + 0.1 * i, 2) for i in range(8)],            # 0.3 .. 1.0
    "k_ret": [round(-0.2 + 0.05 * i, 2) for i in range(9)],       # -0.2 .. 0.2
    "k_tal": [round(-0.2 + 0.05 * i, 2) for i in range(9)],
}
ORDER = ("ridge", "half_life_games", "prior_floor", "k0", "k_ret", "k_tal")


def ppa_holdout_loss(eff_games: pd.DataFrame, seasons, cfg: EffConfig, priors_rows_by_season: dict,
                     talent: pd.DataFrame | None, cache: dict | None = None) -> float:
    """Play-weighted one-week-ahead MSE of PPA per play over `seasons` (NaN if nothing scored)."""
    cache = {} if cache is None else cache
    sse = wsum = 0.0
    for season in seasons:
        prior = season_prior(eff_games, season, priors_rows_by_season.get(season, []), talent, cfg, cache)
        rows = eff_games[(eff_games["season"] == season) & eff_games["week"].notna()
                         & eff_games["y_ppa"].notna() & (eff_games["w_ppa"] > 0)]
        for week, wk in rows.groupby("week"):
            r = state_before(eff_games, season, week, prior, cfg, cache).ratings["ppa"]
            if r.mu != r.mu:                                  # no data and no prior yet
                continue
            pred = (r.mu + wk["team"].map(r.off).fillna(0.0).to_numpy()
                    - wk["opponent"].map(r.deff).fillna(0.0).to_numpy() + r.hfa * wk["home"].to_numpy())
            err = pred - wk["y_ppa"].to_numpy()
            w = wk["w_ppa"].to_numpy()
            sse += float((w * err ** 2).sum())
            wsum += float(w.sum())
    return sse / wsum if wsum else float("nan")


def fit_eff_config(eff_games: pd.DataFrame, priors_rows_by_season: dict, talent: pd.DataFrame | None,
                   train_seasons=TRAIN_SEASONS, grid: dict = GRID, start: EffConfig | None = None,
                   n_passes: int = 4) -> tuple[EffConfig, float]:
    """Coordinate search (one parameter at a time over `grid`, up to n_passes passes) on the
    training seasons only. Returns (config, train loss)."""
    cache: dict = {}
    seen: dict = {}

    def loss(cfg: EffConfig) -> float:
        key = tuple(getattr(cfg, p) for p in ORDER)
        if key not in seen:
            seen[key] = ppa_holdout_loss(eff_games, train_seasons, cfg, priors_rows_by_season, talent, cache)
        return seen[key]

    cur = start or EffConfig()
    for _ in range(n_passes):
        improved = False
        for p in ORDER:
            best, best_loss = cur, loss(cur)
            for v in grid[p]:
                cand = replace(cur, **{p: v})
                l = loss(cand)
                if l < best_loss - 1e-12:
                    best, best_loss = cand, l
            if best is not cur:
                improved, cur = True, best
        if not improved:
            break
    return cur, loss(cur)
```

Create `scripts/fit_cfb_eff.py`:

```python
"""Fit the efficiency-rating hyperparameters (ridge, prior half-life/floor, retention) on
2016-2022 and write assets/cfb/eff_config.json. Report-only on 2023-2025 (nothing is selected
there). Pure local compute from the committed assets -- no network, no key.

Usage:
    uv run python scripts/fit_cfb_eff.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import eff_fit, v3_data  # noqa: E402
from sportsmodel.cfb.efficiency import EffConfig  # noqa: E402

OUT = ROOT / "assets" / "cfb" / "eff_config.json"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="fit and print; do not write eff_config.json")
    args = ap.parse_args(argv)
    t0 = time.time()
    eff_games = v3_data.load_eff_games()
    talent = v3_data.read_asset("talent.parquet")
    priors = v3_data.load_priors_rows()
    default = EffConfig()
    cfg, train_loss = eff_fit.fit_eff_config(eff_games, priors, talent)
    print(f"fitted {cfg}\n  train {list(eff_fit.TRAIN_SEASONS)[0]}-{list(eff_fit.TRAIN_SEASONS)[-1]} "
          f"one-week-ahead PPA MSE: default={eff_fit.ppa_holdout_loss(eff_games, eff_fit.TRAIN_SEASONS, default, priors, talent):.5f}"
          f" fitted={train_loss:.5f}")
    print(f"  holdout {eff_fit.HOLDOUT_SEASONS} (report only): "
          f"default={eff_fit.ppa_holdout_loss(eff_games, eff_fit.HOLDOUT_SEASONS, default, priors, talent):.5f} "
          f"fitted={eff_fit.ppa_holdout_loss(eff_games, eff_fit.HOLDOUT_SEASONS, cfg, priors, talent):.5f}")
    edge = [p for p in eff_fit.ORDER if getattr(cfg, p) in (eff_fit.GRID[p][0], eff_fit.GRID[p][-1])]
    if edge:
        print(f"  NOTE: fitted value sits on the grid edge for {edge}")
    if args.dry_run:
        print("--dry-run: nothing written")
    else:
        OUT.write_text(json.dumps({k: float(v) for k, v in asdict(cfg).items()}, indent=2) + "\n")
        print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"total {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_v3_data.py tests/cfb/test_eff_fit.py tests/cfb/test_efficiency.py -q`
Expected: PASS (3 + 3 + 12).

- [ ] **Step 6: [LOCAL RUN] Fit on 2016-22 and commit `eff_config.json`**

Needs the Task 4 assets; no network. About 30 seconds.

```bash
uv run python scripts/fit_cfb_eff.py
```
Expected: `fitted EffConfig(...)`, the one-week-ahead PPA MSE for default vs fitted on train (fitted <= default) and on 2023-25 (report only), `wrote assets/cfb/eff_config.json`. If it prints `NOTE: fitted value sits on the grid edge for [...]`, widen that parameter's grid in `eff_fit.GRID` (keep the test passing) and re-run.

```bash
git add src/sportsmodel/cfb/v3_data.py src/sportsmodel/cfb/eff_fit.py scripts/fit_cfb_eff.py tests/cfb/test_v3_data.py tests/cfb/test_eff_fit.py assets/cfb/eff_config.json
git commit -m "feat(cfb): asset loaders + walk-forward fit of efficiency hyperparameters (eff_config.json)"
```

---

### Task 7: Context features - weather, travel, time zones, rest, talent gap

**Files:**
- Create: `src/sportsmodel/cfb/context.py`
- Test: `tests/cfb/test_context.py`

**Interfaces:**
- Consumes: asset frames `venues.parquet`, `cfbd_games.parquet`, `weather_games.parquet`, `talent.parquet` (any may be `None`).
- Produces:
  - Column groups `MARGIN_CTX = ("travel_mid_diff", "travel_far_diff", "tz_east_diff", "tz_west_diff", "short_week_diff", "bye_diff", "talent_gap")`, `TOTAL_CTX = ("wind_excess", "cold_excess", "precip")`, `NUISANCE = ("weather_missing",)`, `CONTEXT_COLS = MARGIN_CTX + TOTAL_CTX + NUISANCE`.
  - `ContextAssets(venues, game_venue, home_venue, weather, talent_z)`; `build_context_assets(venues, meta, weather, talent) -> ContextAssets`.
  - `haversine_miles(lat1, lon1, lat2, lon2)`, `utc_offset_hours(tz, when_iso)`, `weather_features(row, venue) -> dict`, `travel_features(assets, season, home, away, game_vid, when_iso) -> dict`, `rest_features(rest_home, rest_away) -> dict`, `talent_gap(assets, season, home, away, games_home, games_away) -> float`, `rest_table(frame) -> {(game_pk, team): days}`, `context_features(game: dict, assets, rests, games_home, games_away) -> dict[CONTEXT_COLS -> float]` (never NaN).

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_context.py`:

```python
import math

import pandas as pd
import pytest

from sportsmodel.cfb import context as cx

NYC, LA = (40.7128, -74.0060), (34.0522, -118.2437)


def test_haversine_known_distance():
    assert cx.haversine_miles(*NYC, *LA) == pytest.approx(2445, abs=10)
    assert cx.haversine_miles(*NYC, *NYC) == 0.0


def test_utc_offset_is_dst_aware_and_unknown_is_nan():
    sept, jan = "2024-09-14T16:00Z", "2024-01-08T16:00Z"
    assert cx.utc_offset_hours("America/New_York", sept) == -4.0
    assert cx.utc_offset_hours("America/New_York", jan) == -5.0
    assert cx.utc_offset_hours("America/Los_Angeles", sept) == -7.0
    assert math.isnan(cx.utc_offset_hours("Not/AZone", sept)) and math.isnan(cx.utc_offset_hours("", sept))


# ----------------------------------------------------------------- weather --

def test_weather_thresholds_and_zero_fill():
    f = cx.weather_features({"wind_speed": 22.0, "temperature": 31.0, "precipitation": 0.2,
                             "game_indoors": False}, None)
    assert f == {"wind_excess": 7.0, "cold_excess": 9.0, "precip": 0.2, "weather_missing": 0.0}
    calm = cx.weather_features({"wind_speed": 5.0, "temperature": 75.0, "precipitation": 0.0}, None)
    assert calm["wind_excess"] == 0.0 and calm["cold_excess"] == 0.0 and calm["weather_missing"] == 0.0


def test_dome_means_no_weather_effect_even_with_bad_readings():
    row = {"wind_speed": 30.0, "temperature": 10.0, "precipitation": 1.0, "game_indoors": True}
    assert cx.weather_features(row, None) == {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0,
                                              "weather_missing": 0.0}
    assert cx.weather_features({"wind_speed": 30.0}, {"dome": True})["wind_excess"] == 0.0


def test_missing_weather_is_flagged_not_fabricated():
    for row in (None, {}, {"wind_speed": float("nan"), "temperature": None, "precipitation": float("nan")}):
        f = cx.weather_features(row, None)
        assert f == {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0, "weather_missing": 1.0}
    part = cx.weather_features({"wind_speed": 20.0, "temperature": float("nan")}, None)
    assert part["wind_excess"] == 5.0 and part["cold_excess"] == 0.0 and part["weather_missing"] == 0.0


# ------------------------------------------------------------ travel / tz --

def assets_for_travel():
    venues = pd.DataFrame({"venue_id": [1, 2, 3], "name": ["ny", "la", "ny2"],
                           "timezone": ["America/New_York", "America/Los_Angeles", "America/New_York"],
                           "latitude": [NYC[0], LA[0], NYC[0] + 0.1], "longitude": [NYC[1], LA[1], NYC[1]],
                           "elevation": [0.0] * 3, "dome": [0.0, 0.0, float("nan")]})
    meta = pd.DataFrame({"game_id": [10, 11], "season": [2023, 2023], "venue_id": [1, 2],
                         "home_team": ["EAST", "WEST"], "neutral_site": [False, False]})
    return cx.build_context_assets(venues, meta, None, None)


def test_home_venue_comes_from_non_neutral_home_games():
    a = assets_for_travel()
    assert cx.home_venue_of(a, "EAST", 2023) == 1 and cx.home_venue_of(a, "EAST", 2025) == 1
    assert cx.home_venue_of(a, "EAST", 2022) is None and cx.home_venue_of(a, "NOBODY", 2023) is None


def test_travel_diff_sign_away_team_burden_is_positive_for_home():
    a = assets_for_travel()
    f = cx.travel_features(a, 2023, "EAST", "WEST", 1, "2023-09-16T16:00Z")   # west team flies to the east
    assert f["travel_far_diff"] == 1.0 and f["travel_mid_diff"] == 0.0
    assert f["tz_east_diff"] == 1.0 and f["tz_west_diff"] == 0.0              # traveled 3h east
    g = cx.travel_features(a, 2023, "WEST", "EAST", 2, "2023-09-16T16:00Z")   # east team flies west
    assert g["tz_west_diff"] == 1.0 and g["tz_east_diff"] == 0.0


def test_neutral_site_burdens_both_teams_and_unknowns_are_zero():
    a = assets_for_travel()
    f = cx.travel_features(a, 2023, "EAST", "WEST", 3, "2023-09-16T16:00Z")   # neutral venue 3 (near NYC)
    assert f["travel_far_diff"] == 1.0                                        # WEST far, EAST ~7 miles
    assert f["travel_mid_diff"] == 0.0
    assert cx.travel_features(a, 2023, "EAST", "WEST", None, "")["travel_far_diff"] == 0.0
    assert cx.travel_features(a, 2023, "EAST", "GHOST", 1, "2023-09-16T16:00Z")["travel_far_diff"] == 0.0


# ------------------------------------------------------------ rest / talent --

def test_rest_features_signs():
    assert cx.rest_features(10, 5) == {"short_week_diff": 1.0, "bye_diff": 0.0}    # away on a short week helps home
    assert cx.rest_features(5, 14) == {"short_week_diff": -1.0, "bye_diff": 1.0}
    assert cx.rest_features(float("nan"), 7) == {"short_week_diff": 0.0, "bye_diff": 0.0}


def test_rest_table_days_since_previous_game_and_opener_nan():
    frame = pd.DataFrame({"season": [2023] * 3, "game_pk": [1, 2, 3],
                          "start_date": ["2023-09-02T17:00Z", "2023-09-09T17:00Z", "2023-09-21T23:30Z"],
                          "home_team": ["a", "b", "a"], "away_team": ["c", "a", "b"]})
    r = cx.rest_table(frame)
    assert math.isnan(r[(1, "a")]) and r[(2, "a")] == 7.0 and r[(3, "a")] == 12.0
    assert math.isnan(r[(2, "b")]) and r[(3, "b")] == 12.0


def test_talent_gap_fades_with_games_and_z_scores_per_season():
    talent = pd.DataFrame({"season": [2023, 2023, 2023], "team": ["a", "b", "c"], "talent": [900.0, 700.0, 500.0]})
    a = cx.build_context_assets(None, None, None, talent)
    z = a.talent_z
    assert z[(2023, "a")] == pytest.approx(-z[(2023, "c")]) and z[(2023, "b")] == pytest.approx(0.0)
    g0 = cx.talent_gap(a, 2023, "a", "c", 0, 0)
    assert g0 == pytest.approx(z[(2023, "a")] - z[(2023, "c")])
    assert cx.talent_gap(a, 2023, "a", "c", 3, 3) == pytest.approx(0.5 * g0)
    assert cx.talent_gap(a, 2023, "a", "ghost", 0, 0) == 0.0


# --------------------------------------------------------------- assembled --

def test_context_features_assembles_every_column_and_never_nan():
    a = assets_for_travel()
    game = {"season": 2023, "game_pk": 77, "home_team": "EAST", "away_team": "WEST",
            "start_date": "2023-09-16T16:00Z", "neutral_site": False}
    f = cx.context_features(game, a, {(77, "EAST"): 6.0, (77, "WEST"): 14.0}, 2, 2)
    assert set(f) == set(cx.CONTEXT_COLS)
    assert f["travel_far_diff"] == 1.0 and f["short_week_diff"] == -1.0 and f["bye_diff"] == 1.0
    assert f["weather_missing"] == 1.0 and not any(math.isnan(v) for v in f.values())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_context.py -q`
Expected: FAIL at collection: `ImportError: cannot import name 'context' from 'sportsmodel.cfb'`.

- [ ] **Step 3: Implement**

Create `src/sportsmodel/cfb/context.py`:

```python
"""Game-day context features for cfb-ratings-v3 (weather, travel, time zones, rest, talent gap). PURE.

Sign convention: margin = home - away, so every *_diff feature is (away team's burden) minus
(home team's burden): a positive value should push the margin UP. Totals features are
non-negative magnitudes. Everything unknown is 0.0 (never fabricated); `weather_missing` flags a
game with no usable weather reading so the fit can absorb that group's mean shift (it is a
nuisance column: the fitted model drops its coefficient at prediction time).

Leak rule: nothing here reads a result. Weather/venue/schedule facts are known (or forecast)
before kickoff; rest uses only earlier start dates; talent is the season's preseason composite.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

WIND_MPH = 15.0               # wind above this slows the passing game / scoring
COLD_F = 40.0                 # temperatures below this suppress scoring
TRAVEL_MID_MI, TRAVEL_FAR_MI = 500.0, 1500.0
TZ_SHIFT_HOURS = 2.0
SHORT_WEEK_DAYS, BYE_DAYS = 6, 13
TALENT_HALF_LIFE_GAMES = 3.0

MARGIN_CTX = ("travel_mid_diff", "travel_far_diff", "tz_east_diff", "tz_west_diff",
              "short_week_diff", "bye_diff", "talent_gap")
TOTAL_CTX = ("wind_excess", "cold_excess", "precip")
NUISANCE = ("weather_missing",)
CONTEXT_COLS = MARGIN_CTX + TOTAL_CTX + NUISANCE


@dataclass(frozen=True)
class ContextAssets:
    venues: dict = field(default_factory=dict)       # venue_id -> {"lat", "lon", "tz", "dome"}
    game_venue: dict = field(default_factory=dict)   # game_id -> venue_id
    home_venue: dict = field(default_factory=dict)   # team -> {season: venue_id} (non-neutral home games)
    weather: dict = field(default_factory=dict)      # game_id -> weather row dict
    talent_z: dict = field(default_factory=dict)     # (season, team) -> z-score of the talent composite


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 3958.8 * 2 * math.asin(math.sqrt(a))


def utc_offset_hours(tz: str, when_iso: str) -> float:
    """UTC offset (hours, DST-aware) of IANA zone `tz` at ISO instant `when_iso`; NaN if unknown."""
    try:
        when = datetime.fromisoformat(when_iso.replace("Z", "+00:00"))
        return when.astimezone(ZoneInfo(tz)).utcoffset().total_seconds() / 3600.0
    except Exception:  # noqa: BLE001 -- unknown zone / unparseable date -> unknown, never guessed
        return float("nan")


def build_context_assets(venues: pd.DataFrame | None, meta: pd.DataFrame | None,
                         weather: pd.DataFrame | None, talent: pd.DataFrame | None) -> ContextAssets:
    """Index the committed frames (any may be None -> that feature family is neutral)."""
    v = {}
    if venues is not None:
        for r in venues.itertuples():
            v[int(r.venue_id)] = {"lat": r.latitude, "lon": r.longitude, "tz": r.timezone or "",
                                  "dome": r.dome == 1.0}
    gv, hv = {}, {}
    if meta is not None:
        m = meta.dropna(subset=["venue_id"])
        gv = {int(g): int(x) for g, x in zip(m["game_id"], m["venue_id"])}
        home = m[~m["neutral_site"].astype(bool)]
        mode = home.groupby(["home_team", "season"])["venue_id"].agg(lambda s: int(s.mode().iloc[0]))
        for (team, season), vid in mode.items():
            hv.setdefault(team, {})[int(season)] = int(vid)
    wx = {}
    if weather is not None:
        wx = {int(r["game_id"]): r for r in weather.to_dict("records")}
        for gid, r in wx.items():                  # weather rows know their own venue (incl. forecasts)
            if _ok(r.get("venue_id")):
                gv.setdefault(gid, int(r["venue_id"]))
    tz = {}
    if talent is not None:
        for season, sdf in talent.dropna(subset=["talent"]).groupby("season"):
            vals = sdf["talent"].astype(float)
            sd = vals.std(ddof=0)
            for team, x in zip(sdf["team"], vals):
                tz[(int(season), str(team))] = float((x - vals.mean()) / sd) if sd > 0 else 0.0
    return ContextAssets(v, gv, hv, wx, tz)


def home_venue_of(assets: ContextAssets, team: str, season: int):
    """The team's home venue id for `season` (latest earlier season if that one is unknown)."""
    seasons = assets.home_venue.get(team, {})
    for s in sorted((s for s in seasons if s <= season), reverse=True):
        return seasons[s]
    return None


def weather_features(row: dict | None, venue: dict | None) -> dict:
    """wind_excess (mph over 15), cold_excess (F under 40), precip (CFBD units, >= 0).
    A dome / indoor game is a known zero. No usable reading -> zeros + weather_missing=1."""
    if (row is not None and bool(row.get("game_indoors"))) or (venue is not None and venue.get("dome")):
        return {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0, "weather_missing": 0.0}
    wind, temp, prec = (row.get(k) if row else None for k in ("wind_speed", "temperature", "precipitation"))
    if not any(_ok(x) for x in (wind, temp, prec)):
        return {"wind_excess": 0.0, "cold_excess": 0.0, "precip": 0.0, "weather_missing": 1.0}
    return {"wind_excess": max(0.0, wind - WIND_MPH) if _ok(wind) else 0.0,
            "cold_excess": max(0.0, COLD_F - temp) if _ok(temp) else 0.0,
            "precip": max(0.0, prec) if _ok(prec) else 0.0, "weather_missing": 0.0}


def _burden(assets: ContextAssets, team: str, season: int, game_vid, when_iso: str) -> dict:
    """One team's travel burden: distance bucket flags and time-zone shift flags (0 = unknown/none)."""
    zero = {"mid": 0.0, "far": 0.0, "east": 0.0, "west": 0.0}
    hv = home_venue_of(assets, team, season)
    if game_vid is None or hv is None or hv == game_vid:
        return zero
    a, b = assets.venues.get(hv), assets.venues.get(game_vid)
    if not a or not b or not (_ok(a["lat"]) and _ok(a["lon"]) and _ok(b["lat"]) and _ok(b["lon"])):
        return zero
    d = haversine_miles(a["lat"], a["lon"], b["lat"], b["lon"])
    shift = utc_offset_hours(b["tz"], when_iso) - utc_offset_hours(a["tz"], when_iso)   # + = traveled east
    shift = 0.0 if math.isnan(shift) else shift
    return {"mid": float(TRAVEL_MID_MI <= d < TRAVEL_FAR_MI), "far": float(d >= TRAVEL_FAR_MI),
            "east": float(shift >= TZ_SHIFT_HOURS), "west": float(shift <= -TZ_SHIFT_HOURS)}


def travel_features(assets: ContextAssets, season: int, home: str, away: str, game_vid, when_iso: str) -> dict:
    h, a = (_burden(assets, t, season, game_vid, when_iso) for t in (home, away))
    return {"travel_mid_diff": a["mid"] - h["mid"], "travel_far_diff": a["far"] - h["far"],
            "tz_east_diff": a["east"] - h["east"], "tz_west_diff": a["west"] - h["west"]}


def rest_features(rest_home, rest_away) -> dict:
    """short_week_diff / bye_diff = away flag - home flag. Unknown rest (season opener) = no flag."""
    def flags(r):
        return (float(_ok(r) and r <= SHORT_WEEK_DAYS), float(_ok(r) and r >= BYE_DAYS))
    (sh, bh), (sa, ba) = flags(rest_home), flags(rest_away)
    return {"short_week_diff": sa - sh, "bye_diff": ba - bh}


def talent_gap(assets: ContextAssets, season: int, home: str, away: str, games_home: int, games_away: int) -> float:
    """(talent z home - talent z away), fading with games played: 0.5 ** (mean games / 3)."""
    zh, za = assets.talent_z.get((season, home)), assets.talent_z.get((season, away))
    if zh is None or za is None:
        return 0.0
    return (zh - za) * 0.5 ** (((games_home + games_away) / 2.0) / TALENT_HALF_LIFE_GAMES)


def rest_table(frame: pd.DataFrame) -> dict:
    """{(game_pk, team): days since that team's previous game of the season} (NaN for an opener).
    Reads start dates only, so unscored (upcoming) games are fine."""
    d = frame[["season", "game_pk", "start_date", "home_team", "away_team"]].copy()
    d["t"] = pd.to_datetime(d["start_date"], utc=True, errors="coerce")
    long = pd.concat([d.rename(columns={"home_team": "team"})[["season", "game_pk", "team", "t"]],
                      d.rename(columns={"away_team": "team"})[["season", "game_pk", "team", "t"]]])
    long = long.sort_values(["season", "team", "t"])
    prev = long.groupby(["season", "team"])["t"].shift(1)
    days = (long["t"].dt.normalize() - prev.dt.normalize()).dt.days
    return {(int(g), str(t)): (float(x) if pd.notna(x) else float("nan"))
            for g, t, x in zip(long["game_pk"], long["team"], days)}


def context_features(game: dict, assets: ContextAssets, rests: dict, games_home: int, games_away: int) -> dict:
    """All CONTEXT_COLS for one game dict (season, game_pk, home_team, away_team, start_date)."""
    pk, season = int(game["game_pk"]), int(game["season"])
    h, a = game["home_team"], game["away_team"]
    vid = assets.game_venue.get(pk)
    if vid is None and not game.get("neutral_site"):
        vid = home_venue_of(assets, h, season)
    when = str(game.get("start_date") or "")
    out = {**travel_features(assets, season, h, a, vid, when),
           **rest_features(rests.get((pk, h), float("nan")), rests.get((pk, a), float("nan"))),
           "talent_gap": talent_gap(assets, season, h, a, games_home, games_away),
           **weather_features(assets.weather.get(pk), assets.venues.get(vid) if vid is not None else None)}
    return {c: out[c] for c in CONTEXT_COLS}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_context.py -q`
Expected: PASS (12 tests: haversine, DST-aware offsets, thresholds, dome = no weather effect, missing flagged not fabricated, home-venue mode, travel/time-zone sign, neutral sites, rest, talent fade, assembled columns).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/cfb/context.py tests/cfb/test_context.py
git commit -m "feat(cfb): game-day context features (weather, travel, time zones, rest, talent gap)"
```

---

### Task 8: v3 model core and the leak-free feature table

**Files:**
- Create: `src/sportsmodel/cfb/v3.py`, `src/sportsmodel/cfb/v3_table.py`
- Test: `tests/cfb/test_v3.py`, `tests/cfb/test_v3_table.py`

**Interfaces:**
- Consumes: `walkforward.walk`, `walkforward.model_margin_total`, `priors.{season_priors, blend_rating, prior_weight}`, Task 5 `efficiency.*`, Task 7 `context.*`, `nfl.gameline.GameLineConfig`.
- Produces:
  - `v3.LinearBlend(intercept: float, coefs: dict)` with `.predict(feats) -> float` (NaN / missing = 0), `.to_dict()`, `LinearBlend.from_dict`.
  - `v3.V3Weights(points_map, margin, total, meta)` with `.to_json()`, `V3Weights.from_json(text)`; `load_v3_weights(path)` (strict: FileNotFoundError if missing); `eff_margin_total(pm, row) -> (margin_eff, total_eff)`; `predict_v3(row, w) -> (margin, total)`; `gameline_v3_dict(sigma_margin, sigma_total) -> dict`; `load_gameline_config(path) -> GameLineConfig`; constants `MARGIN_CORE = ("margin_v2", "margin_eff", "prior_margin", "non_neutral")`, `TOTAL_CORE = ("total_eff", "total_v2")`, `VERSION = "cfb-ratings-v3"`.
  - `v3_table.V3Inputs(elo_cfg, blend_cfg, prior_weights, decay, eff_cfg, eff_games, priors_rows, talent, ctx)`; `TABLE_COLUMNS` (season, week, game_pk, home_team, away_team, neutral, start_date, actual_margin, actual_total, market_spread, market_total, `margin_v2`, `total_v2` (raw, pre-bias), `prior_margin`, `non_neutral`, games_home, games_away, `h_*`/`a_*` for each `POINT_FEATURES`, every `CONTEXT_COLS`); `build_table(frame, inp, *, seasons=None, include_unscored=False) -> DataFrame`; `live_frame(sched, season, week, games) -> DataFrame`.

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_v3.py`:

```python
"""v3 blend math, JSON round trip, gameline config."""
import json

import pytest

from sportsmodel.cfb import v3


def test_linear_blend_treats_nan_and_missing_as_zero():
    b = v3.LinearBlend(1.0, {"a": 2.0, "b": 3.0, "c": 4.0})
    assert b.predict({"a": 1.0, "b": float("nan"), "c": None}) == 3.0
    assert b.predict({}) == 1.0


def test_predict_v3_combines_points_map_and_blend():
    pm = v3.LinearBlend(20.0, {"success": 10.0})
    w = v3.V3Weights(pm, v3.LinearBlend(0.5, {"margin_v2": 0.8, "margin_eff": 0.2, "wind_excess": 9.9}),
                     v3.LinearBlend(2.0, {"total_eff": 0.5, "total_v2": 0.4, "wind_excess": -0.3}))
    row = {"margin_v2": 6.0, "total_v2": 50.0, "h_success": 0.1, "a_success": -0.1, "wind_excess": 10.0}
    m, t = v3.predict_v3(row, w)
    eff_m, eff_t = (21.0 - 19.0), (21.0 + 19.0)
    assert m == pytest.approx(0.5 + 0.8 * 6.0 + 0.2 * eff_m + 9.9 * 10.0)   # margin blend reads the margin ctx it has
    assert t == pytest.approx(2.0 + 0.5 * eff_t + 0.4 * 50.0 - 0.3 * 10.0)


def test_weights_json_roundtrip_and_strict_load(tmp_path):
    w = v3.V3Weights(v3.LinearBlend(1.0, {"ppa_plays": 2.0}), v3.LinearBlend(0.1, {"margin_v2": 1.0}),
                     v3.LinearBlend(3.0, {"total_v2": 0.9}), {"n_train": 5})
    p = tmp_path / "v3_weights.json"
    p.write_text(w.to_json())
    assert v3.load_v3_weights(p) == w
    assert json.loads(p.read_text())["version"] == "cfb-ratings-v3"
    with pytest.raises(FileNotFoundError):
        v3.load_v3_weights(tmp_path / "nope.json")


def test_gameline_v3_dict_has_every_key_generate_cfb_reads():
    d = v3.gameline_v3_dict(15.5, 16.0)
    assert set(d) >= {"sigma_margin", "sigma_total", "offset", "total_max", "w_margin", "w_total",
                      "bias_margin", "bias_total"}
    assert d["w_margin"] == {"start": 0.0, "floor": 0.0, "decay": 0.0} and d["bias_margin"] == 0.0


def test_load_gameline_config_reads_gameline_shaped_json_for_v2_and_v3(tmp_path):
    p = tmp_path / "gameline_v3.json"
    p.write_text(json.dumps(v3.gameline_v3_dict(15.5, 16.0)))
    cfg = v3.load_gameline_config(p)
    assert (cfg.sigma_margin, cfg.sigma_total, cfg.offset, cfg.total_max) == (15.5, 16.0, 110, 150)
    assert cfg.bias_margin == 0.0 and cfg.w_margin.start == 0.0
    q = tmp_path / "old.json"                                    # a pre-bias gameline.json still loads
    d = v3.gameline_v3_dict(1.0, 2.0)
    del d["bias_margin"], d["bias_total"]
    q.write_text(json.dumps(d))
    assert v3.load_gameline_config(q).bias_total == 0.0
```

Create `tests/cfb/test_v3_table.py`:

```python
"""v3 feature table: v2 parity, leak invariant, FCS skip, live frame. Tiny synthetic league."""
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import context as cx, efficiency as eff, v3_table as vt
from sportsmodel.cfb.priors import DecayConfig, PriorWeights
from sportsmodel.cfb.walkforward import raw_model_predictions
from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig

ELO = EloConfig(k=40, hfa_elo=70, carryover=0.9, base=1500.0)
BLEND = BlendConfig(w_sos=0.45, srs_min_games=3)
TEAMS = [str(i) for i in range(1, 9)]


def world(seasons=(2021, 2022, 2023), weeks=5, seed=5, with_fcs=True):
    """(schedule frame, eff_games) over a tiny league; game_pk == eff game_id."""
    rng = np.random.default_rng(seed)
    strength = rng.normal(0, 8, len(TEAMS))
    sched, effrows, pk = [], [], 1000
    for s in seasons:
        for w in range(1, weeks + 1):
            order = list(rng.permutation(len(TEAMS)))
            for i in range(0, len(order), 2):
                hi, ai = order[i], order[i + 1]
                h, a = TEAMS[hi], TEAMS[ai]
                margin = strength[hi] - strength[ai] + 2.5 + rng.normal(0, 10)
                hs = int(max(0, round(28 + margin / 2 + rng.normal(0, 4))))
                as_ = int(max(0, round(28 - margin / 2 + rng.normal(0, 4))))
                pk += 1
                sched.append({"season": s, "week": w, "home_team": h, "away_team": a, "home_score": hs,
                              "away_score": as_, "game_type": "REG", "game_pk": pk,
                              "start_date": f"{s}-09-{w * 7 - 3:02d}T17:00Z", "neutral_site": bool(i == 2),
                              "market_spread": float(round(margin)) + 0.5, "market_total": 55.5})
                for team, opp, sgn in ((hi, ai, 1), (ai, hi, -1)):
                    y = 0.1 + 0.01 * (strength[team] - strength[opp]) + rng.normal(0, 0.03)
                    effrows.append({"season": s, "game_id": pk, "team": TEAMS[team], "opponent": TEAMS[opp],
                                    "season_type": "regular", "week": float(w), "home": float(sgn),
                                    "plays": 70.0, "rush_plays": 35.0, "pass_plays": 35.0,
                                    **{f"y_{m}": y for m in eff.METRICS}, **{f"w_{m}": 60.0 for m in eff.METRICS}})
        if with_fcs:                                           # an FCS game every season, week 1
            pk += 1
            sched.append({"season": s, "week": 1, "home_team": "1", "away_team": "FCS", "home_score": 45,
                          "away_score": 7, "game_type": "REG", "game_pk": pk, "start_date": f"{s}-09-01T17:00Z",
                          "neutral_site": False, "market_spread": np.nan, "market_total": np.nan})
    return pd.DataFrame(sched), pd.DataFrame(effrows)


def inputs(eff_games, priors_rows=None):
    return vt.V3Inputs(ELO, BLEND, PriorWeights(sp_scale=17.5, sp_offset=1500.0), DecayConfig(3.0, 0.35),
                       eff.EffConfig(), eff_games, priors_rows or {}, None, cx.ContextAssets())


def test_table_margin_and_total_match_the_v2_walk_when_no_prior():
    sched, eg = world()
    table = vt.build_table(sched, inputs(eg))
    raw = pd.DataFrame(raw_model_predictions(sched, ELO, BLEND))
    raw = raw[(raw.home_team != "FCS") & (raw.away_team != "FCS")].set_index(["season", "week", "home_team"])
    got = table.set_index(["season", "week", "home_team"])
    assert len(got) == len(raw)
    assert np.allclose(got["margin_v2"], raw.loc[got.index, "model_margin"])
    assert np.allclose(got["total_v2"], raw.loc[got.index, "model_total"])
    assert not table[["home_team", "away_team"]].isin(["FCS"]).any().any()


def test_prior_margin_zero_without_priors_and_blends_into_v2_with_them():
    sched, eg = world()
    rows = {s: [{"team_espn_id": t, "sp_rating": float(i), "returning_pct": 0.5, "recruiting_points": 100.0,
                 "portal_net": 0.0, "prior_sos": 0.0, "qb_returning": True} for i, t in enumerate(TEAMS)]
            for s in (2020, 2021, 2022, 2023)}
    assert (vt.build_table(sched, inputs(eg)).prior_margin == 0).all()
    with_p = vt.build_table(sched, inputs(eg, rows))
    wk1 = with_p[(with_p.season == 2022) & (with_p.week == 1)]
    assert (wk1.prior_margin != 0).any()                     # week 1: prior carries full weight
    late = with_p[(with_p.season == 2022) & (with_p.week == 5)]
    assert late.prior_margin.abs().mean() < wk1.prior_margin.abs().mean()   # decays toward the floor


def test_appending_a_later_game_never_changes_earlier_rows():
    sched, eg = world()
    base = vt.build_table(sched, inputs(eg))
    extra = sched[(sched.season == 2023) & (sched.week == 5)].copy()
    extra["week"], extra["game_pk"] = 6, extra["game_pk"] + 5000
    extra["home_score"], extra["away_score"] = 70, 0
    eg_extra = eg[(eg.season == 2023) & (eg.week == 5)].copy()
    eg_extra["week"], eg_extra["game_id"], eg_extra["y_ppa"] = 6.0, eg_extra["game_id"] + 5000, 9.0
    later = vt.build_table(pd.concat([sched, extra], ignore_index=True),
                           inputs(pd.concat([eg, eg_extra], ignore_index=True)))
    keep = later[later.game_pk.isin(base.game_pk)].sort_values("game_pk").reset_index(drop=True)
    pd.testing.assert_frame_equal(base.sort_values("game_pk").reset_index(drop=True), keep[base.columns],
                                  check_dtype=False)


def test_features_are_finite_and_columns_complete():
    sched, eg = world()
    t = vt.build_table(sched, inputs(eg))
    assert list(t.columns) == vt.TABLE_COLUMNS
    feat = [c for c in vt.TABLE_COLUMNS if c not in ("market_spread", "market_total", "start_date",
                                                      "home_team", "away_team", "neutral")]
    assert np.isfinite(t[feat].to_numpy(float)).all()
    assert set(t["non_neutral"]) <= {0.0, 1.0} and (t["neutral"] == (t["non_neutral"] == 0.0)).all()


def test_unscored_games_emit_only_when_asked_and_have_nan_actuals():
    sched, eg = world()
    sched.loc[sched.index[-3:], ["home_score", "away_score"]] = np.nan
    assert vt.build_table(sched, inputs(eg)).actual_margin.notna().all()
    live = vt.build_table(sched, inputs(eg), include_unscored=True)
    assert live.actual_margin.isna().sum() >= 2 and live.actual_total.isna().sum() >= 2


def test_live_frame_has_only_prior_games_plus_unscored_slate():
    sched, _ = world()
    slate = [{"game_pk": 9001, "home_team": "1", "away_team": "2", "neutral_site": False,
              "start_date": "2023-10-07T17:00Z"}]
    f = vt.live_frame(sched, 2023, 4, slate)
    assert (f[f.game_pk != 9001][["season", "week"]].apply(tuple, axis=1)
            .map(lambda sw: sw < (2023, 4))).all()
    up = f[f.game_pk == 9001].iloc[0]
    assert up.week == 4 and pd.isna(up.home_score) and up.start_date == "2023-10-07T17:00Z"
    assert (f.game_pk != 9001).sum() == ((sched.season < 2023) | ((sched.season == 2023) & (sched.week < 4))).sum()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_v3.py tests/cfb/test_v3_table.py -q`
Expected: FAIL at collection: `ImportError: cannot import name 'v3' from 'sportsmodel.cfb'`.

- [ ] **Step 3: Implement the model core**

Create `src/sportsmodel/cfb/v3.py`:

```python
"""cfb-ratings-v3: the points map, the blend and the serialisable weights. PURE.

    side points   = pm.intercept + sum_f pm.coefs[f] * side_feature[f]     (efficiency -> points, per offense)
    margin_eff    = points(home offense) - points(away offense)
    total_eff     = points(home offense) + points(away offense)
    margin_v3     = c + a1*margin_v2 + a2*margin_eff + a3*prior_margin + h*non_neutral + sum ctx_m * x
    total_v3      = c + b1*total_eff + b2*total_v2 + sum ctx_t * x

`margin_v2` / `total_v2` are the live v2 model's own pre-bias margin / total (Elo + SRS, prior-
blended; walkforward.model_margin_total), `prior_margin` the decaying preseason-prior margin
(Elo scale, /25). Any NaN feature counts as 0.0 so a missing input never poisons a prediction.
Weights live in assets/cfb/v3_weights.json; the market is never an input.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from sportsmodel.nfl.gameline import GameLineConfig
from sportsmodel.nfl.shrink import ShrinkParams

from .context import MARGIN_CTX, TOTAL_CTX
from .efficiency import POINT_FEATURES

MARGIN_CORE = ("margin_v2", "margin_eff", "prior_margin", "non_neutral")
TOTAL_CORE = ("total_eff", "total_v2")
GAMELINE_OFFSET, GAMELINE_TOTAL_MAX = 110, 150      # same CFB constants as gameline.json
VERSION = "cfb-ratings-v3"


def _f(x) -> float:
    return 0.0 if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


@dataclass(frozen=True)
class LinearBlend:
    intercept: float
    coefs: dict = field(default_factory=dict)

    def predict(self, feats) -> float:
        return self.intercept + sum(c * _f(feats.get(k)) for k, c in self.coefs.items())

    def to_dict(self) -> dict:
        return {"intercept": float(self.intercept), "coefs": {k: float(v) for k, v in self.coefs.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearBlend":
        return cls(float(d["intercept"]), {k: float(v) for k, v in d["coefs"].items()})


@dataclass(frozen=True)
class V3Weights:
    points_map: LinearBlend
    margin: LinearBlend
    total: LinearBlend
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({"version": VERSION, "points_map": self.points_map.to_dict(),
                           "margin": self.margin.to_dict(), "total": self.total.to_dict(),
                           "meta": self.meta}, indent=2, allow_nan=False) + "\n"

    @classmethod
    def from_json(cls, text: str) -> "V3Weights":
        d = json.loads(text)
        return cls(LinearBlend.from_dict(d["points_map"]), LinearBlend.from_dict(d["margin"]),
                   LinearBlend.from_dict(d["total"]), d.get("meta", {}))


def load_v3_weights(path) -> V3Weights:
    """V3Weights from assets/cfb/v3_weights.json. STRICT: a missing file raises (v3 must never be
    served on guessed weights); the v2 path does not call this."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing; run scripts/fit_cfb_v3.py (plan Task 8) before serving v3")
    return V3Weights.from_json(p.read_text())


def eff_margin_total(pm: LinearBlend, row) -> tuple[float, float]:
    """(margin_eff, total_eff) from a table row's h_<feature> / a_<feature> side features."""
    home = pm.predict({f: row.get(f"h_{f}") for f in POINT_FEATURES})
    away = pm.predict({f: row.get(f"a_{f}") for f in POINT_FEATURES})
    return home - away, home + away


def predict_v3(row, w: V3Weights) -> tuple[float, float]:
    """The v3 (margin, total) for one feature-table row (a dict / Series). No bias is applied:
    the fitted intercepts already centre the residuals."""
    margin_eff, total_eff = eff_margin_total(w.points_map, row)
    feats = {**row, "margin_eff": margin_eff, "total_eff": total_eff}
    return w.margin.predict(feats), w.total.predict(feats)


def gameline_v3_dict(sigma_margin: float, sigma_total: float) -> dict:
    """assets/cfb/gameline_v3.json: same keys as gameline.json (so generate_cfb.load_gameline reads
    it); sigmas are the v3 train RMSEs, the market-shrink curves are zero (model-only) and the
    bias terms are 0 (the blend intercepts already centre v3)."""
    zero = {"start": 0.0, "floor": 0.0, "decay": 0.0}
    return {"sigma_margin": float(sigma_margin), "sigma_total": float(sigma_total),
            "offset": GAMELINE_OFFSET, "total_max": GAMELINE_TOTAL_MAX,
            "w_margin": dict(zero), "w_total": dict(zero), "bias_margin": 0.0, "bias_total": 0.0}


ALL_MARGIN_FEATURES = MARGIN_CORE + MARGIN_CTX
ALL_TOTAL_FEATURES = TOTAL_CORE + TOTAL_CTX


def load_gameline_config(path) -> GameLineConfig:
    """GameLineConfig from a gameline.json-shaped file (gameline.json for v2, gameline_v3.json for v3)."""
    j = json.loads(Path(path).read_text())
    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"], offset=j["offset"],
                          total_max=j["total_max"], w_margin=ShrinkParams(**j["w_margin"]),
                          w_total=ShrinkParams(**j["w_total"]), bias_margin=j.get("bias_margin", 0.0),
                          bias_total=j.get("bias_total", 0.0))
```

- [ ] **Step 4: Implement the feature table**

Create `src/sportsmodel/cfb/v3_table.py`:

```python
"""The v3 feature table: one row per FBS-vs-FBS game with the v2 model's own margin/total, the
decaying prior margin, the efficiency side features and the context features -- all built in ONE
leak-free walk-forward (walkforward.walk), so the fit, the gate and the live producer share a
single implementation.

Row for a game of week W uses only: games before W of that season (Elo is continuous across
seasons), the previous season's final efficiency ratings, schedule/venue/weather facts known
before kickoff. Appending a later game never changes an earlier row.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from sportsmodel.nfl.elo import EloConfig
from sportsmodel.nfl.ratings import BlendConfig

from .context import CONTEXT_COLS, ContextAssets, context_features, rest_table
from .efficiency import (POINT_FEATURES, EffConfig, season_prior, side_features, state_before)
from .priors import DecayConfig, PriorWeights, blend_rating, prior_weight, season_priors
from .teams import FCS
from .walkforward import _clean_market, model_margin_total, walk


@dataclass(frozen=True)
class V3Inputs:
    elo_cfg: EloConfig
    blend_cfg: BlendConfig
    prior_weights: PriorWeights
    decay: DecayConfig
    eff_cfg: EffConfig
    eff_games: pd.DataFrame
    priors_rows: dict            # {season: [priors.parquet row dicts]}
    talent: pd.DataFrame | None
    ctx: ContextAssets


SIDE_COLS = [f"{s}_{f}" for s in ("h", "a") for f in POINT_FEATURES]
TABLE_COLUMNS = (["season", "week", "game_pk", "home_team", "away_team", "neutral", "start_date",
                  "actual_margin", "actual_total", "market_spread", "market_total",
                  "margin_v2", "total_v2", "prior_margin", "non_neutral", "games_home", "games_away"]
                 + SIDE_COLS + list(CONTEXT_COLS))


def _r_pre(inp: V3Inputs, season: int) -> dict:
    """{team: R_pre} for `season` via the leak-free priors.season_priors (empty if no rows)."""
    if season not in inp.priors_rows:
        return {}
    rows = {s: r for s, r in inp.priors_rows.items() if s in (season - 1, season)}
    return season_priors(rows, season, inp.prior_weights)


def build_table(frame: pd.DataFrame, inp: V3Inputs, *, seasons=None,
                include_unscored: bool = False) -> pd.DataFrame:
    """Feature table for the games of `frame` (needs season, week, home_team, away_team,
    home_score, away_score, game_pk, neutral_site, start_date; market_spread / market_total
    optional). `seasons` restricts the emitted seasons (Elo still runs over the whole frame);
    include_unscored also emits games with no result yet (the live slate)."""
    rests = rest_table(frame)
    r_pre_cache, prior_cache, state_cache, eff_cache = {}, {}, {}, {}
    rows = []
    for season, week, g, st in walk(frame, inp.elo_cfg, include_unscored=include_unscored,
                                    seasons=None if seasons is None else set(seasons)):
        h, a = g["home_team"], g["away_team"]
        if FCS in (h, a):
            continue
        season, week = int(season), int(week)
        if season not in r_pre_cache:
            r_pre_cache[season] = _r_pre(inp, season)
            prior_cache[season] = season_prior(inp.eff_games, season, inp.priors_rows.get(season, []),
                                               inp.talent, inp.eff_cfg, eff_cache)
        rp = r_pre_cache[season]
        gh, ga = st.counts.get(h, 0), st.counts.get(a, 0)
        eh, ea = g["elo_home"], g["elo_away"]
        if h in rp:
            eh = blend_rating(rp[h], eh, gh, inp.decay)
        if a in rp:
            ea = blend_rating(rp[a], ea, ga, inp.decay)
        margin_v2, total_v2 = model_margin_total(h, a, eh, ea, st, inp.elo_cfg, inp.blend_cfg)
        if h in rp and a in rp:
            w_pair = 0.5 * (prior_weight(gh, inp.decay) + prior_weight(ga, inp.decay))
            prior_margin = w_pair * ((rp[h] + inp.elo_cfg.hfa_elo) - rp[a]) / 25.0
        else:
            prior_margin = 0.0
        if (season, week) not in state_cache:
            state_cache[(season, week)] = state_before(inp.eff_games, season, week, prior_cache[season],
                                                       inp.eff_cfg, eff_cache)
        neutral = bool(g.get("neutral_site"))
        hs = 0.0 if neutral else 1.0
        fh = side_features(state_cache[(season, week)], h, a, hs)
        fa = side_features(state_cache[(season, week)], a, h, -hs)
        ctx = context_features({"season": season, "game_pk": g["game_pk"], "home_team": h, "away_team": a,
                                "start_date": g.get("start_date"), "neutral_site": neutral},
                               inp.ctx, rests, gh, ga)
        scored = not (pd.isna(g["home_score"]) or pd.isna(g["away_score"]))
        rows.append({"season": season, "week": week, "game_pk": int(g["game_pk"]), "home_team": h,
                     "away_team": a, "neutral": neutral, "start_date": str(g.get("start_date") or ""),
                     "actual_margin": float(g["home_score"] - g["away_score"]) if scored else np.nan,
                     "actual_total": float(g["home_score"] + g["away_score"]) if scored else np.nan,
                     "market_spread": _clean_market(g.get("market_spread")),
                     "market_total": _clean_market(g.get("market_total")),
                     "margin_v2": margin_v2, "total_v2": total_v2, "prior_margin": prior_margin,
                     "non_neutral": 0.0 if neutral else 1.0, "games_home": gh, "games_away": ga,
                     **{f"h_{k}": v for k, v in fh.items()}, **{f"a_{k}": v for k, v in fa.items()}, **ctx})
    out = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    for c in ("market_spread", "market_total"):
        out[c] = out[c].astype("float64")
    return out


def live_frame(sched: pd.DataFrame, season: int, week: int, games: list[dict]) -> pd.DataFrame:
    """The frame the walk sees for the live week: every scored REG game strictly before (season,
    week) plus the slate's `games` appended UNSCORED (dicts carry game_pk, home_team, away_team,
    neutral_site, start_date). Mirrors walkforward.live_week_state, so a live row equals the
    backtest row the same game would have had."""
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched.columns else sched
    reg = reg[reg["home_team"] != reg["away_team"]]
    before = reg[(reg["season"] < season) | ((reg["season"] == season) & (reg["week"] < week))]
    before = before.dropna(subset=["home_score", "away_score"])
    cols = ["season", "week", "home_team", "away_team", "home_score", "away_score", "game_pk",
            "neutral_site", "start_date"]
    upcoming = pd.DataFrame([{"season": season, "week": week, "home_team": g["home_team"],
                              "away_team": g["away_team"], "home_score": np.nan, "away_score": np.nan,
                              "game_pk": int(g["game_pk"]), "neutral_site": bool(g.get("neutral_site")),
                              "start_date": g.get("start_date") or g.get("commence_time") or ""}
                             for g in games], columns=cols)
    base = before[cols].astype({"home_score": "float64", "away_score": "float64"})
    return pd.concat([base, upcoming], ignore_index=True)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_v3.py tests/cfb/test_v3_table.py tests/cfb/test_walkforward.py -q`
Expected: PASS (5 + 6 new, plus the existing walk-forward tests: the table's v2 columns equal `raw_model_predictions` when no prior is configured, and appending a later game leaves every earlier row unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/cfb/v3.py src/sportsmodel/cfb/v3_table.py tests/cfb/test_v3.py tests/cfb/test_v3_table.py
git commit -m "feat(cfb): v3 model core (points map, blend, weights json) + leak-free feature table"
```

---

### Task 9: Fit the v3 weights (points map, blends, context by lasso, sigmas)

**Files:**
- Create: `src/sportsmodel/cfb/v3_fit.py`, `scripts/fit_cfb_v3.py`
- Modify: `src/sportsmodel/cfb/v3_data.py` (append `load_rating`, `load_v3_inputs`)
- Test: `tests/cfb/test_v3_fit.py`, `tests/scripts/test_fit_cfb_v3.py`, `tests/cfb/test_v3_data.py` (append)

**Interfaces:**
- Consumes: `v3.{LinearBlend, V3Weights, MARGIN_CORE, TOTAL_CORE, predict_v3, eff_margin_total, gameline_v3_dict}`, `context.{MARGIN_CTX, TOTAL_CTX, NUISANCE}`, `v3_table.{build_table, V3Inputs}`, `v3_data.*` (Task 6).
- Produces:
  - `v3_fit.fit_points_map(table, train_seasons=TRAIN_SEASONS, ridge=1.0) -> LinearBlend`; `add_eff(table, pm) -> DataFrame` (adds `margin_eff`, `total_eff`); `fit_blend(df, y, core, ctx, nuisance=(), alphas=LASSO_ALPHAS) -> (LinearBlend, info)` (lasso with the one-standard-error rule, kept terms refit unpenalised, unselected context terms exactly 0.0, nuisance dropped); `fit_v3(table, train_seasons=TRAIN_SEASONS) -> V3Weights` (`meta` holds `sigma_margin`, `sigma_total`, `n_train`, `train_mae`).
  - `v3_data.load_rating(assets=ASSETS) -> (EloConfig, BlendConfig)`; `v3_data.load_v3_inputs(assets=ASSETS, *, weather=None) -> V3Inputs` (strict about `priors_weights.json` when `priors.parquet` exists; `weather` overrides `weather_games.parquet`).
  - `scripts/fit_cfb_v3.write_outputs(w, assets_dir)`; assets `assets/cfb/v3_weights.json`, `assets/cfb/gameline_v3.json`.

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_v3_fit.py`:

```python
"""The v3 fit on a synthetic table with known structure."""
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3_fit
from sportsmodel.cfb.context import CONTEXT_COLS
from sportsmodel.cfb.efficiency import POINT_FEATURES


def synthetic_table(n=2400, seed=2):
    """Known truth: margin = 0.7*v2 + 0.4*eff - 1 + 1.5*non_neutral + 2*short_week_diff + noise;
    total = 0.5*eff_total + 0.5*v2 - 0.4*wind_excess + noise. Everything else has zero effect."""
    rng = np.random.default_rng(seed)
    seasons = rng.choice(np.arange(2016, 2023), n)
    t = pd.DataFrame({"season": seasons, "week": rng.integers(1, 14, n), "game_pk": np.arange(n)})
    for f in POINT_FEATURES:
        t[f"h_{f}"], t[f"a_{f}"] = rng.normal(0, 1, n), rng.normal(0, 1, n)
    pm_true = {"intercept": 28.0, "ppa_plays": 0.8, "success": 1.5}
    hp = pm_true["intercept"] + 0.8 * t["h_ppa_plays"] + 1.5 * t["h_success"]
    ap = pm_true["intercept"] + 0.8 * t["a_ppa_plays"] + 1.5 * t["a_success"]
    t["margin_v2"], t["total_v2"] = rng.normal(3, 12, n), rng.normal(55, 8, n)
    t["prior_margin"] = rng.normal(0, 3, n)
    t["non_neutral"] = (rng.random(n) > 0.1).astype(float)
    for c in CONTEXT_COLS:
        t[c] = 0.0
    t["short_week_diff"] = rng.choice([-1.0, 0.0, 1.0], n, p=[0.15, 0.7, 0.15])
    t["wind_excess"] = np.where(rng.random(n) < 0.3, rng.uniform(0, 15, n), 0.0)
    t["weather_missing"] = (rng.random(n) < 0.25).astype(float)
    t["travel_far_diff"] = rng.choice([-1.0, 0.0, 1.0], n)               # pure noise columns
    t["talent_gap"] = rng.normal(0, 1, n)
    eff_m, eff_t = hp - ap, hp + ap
    t["actual_margin"] = (0.7 * t["margin_v2"] + 0.4 * eff_m - 1.0 + 1.5 * t["non_neutral"]
                          + 2.0 * t["short_week_diff"] + rng.normal(0, 3, n))
    t["actual_total"] = (0.5 * eff_t + 0.5 * t["total_v2"] - 0.4 * t["wind_excess"]
                         + 5 * t["weather_missing"] + rng.normal(0, 3, n))
    return t


def test_points_map_recovers_known_coefficients():
    t = synthetic_table()
    # make actual points consistent with the true points map for this check
    hp = 28.0 + 0.8 * t["h_ppa_plays"] + 1.5 * t["h_success"]
    ap = 28.0 + 0.8 * t["a_ppa_plays"] + 1.5 * t["a_success"]
    t["actual_margin"], t["actual_total"] = hp - ap, hp + ap
    pm = v3_fit.fit_points_map(t)
    assert pm.intercept == pytest.approx(28.0, abs=0.05)
    assert pm.coefs["ppa_plays"] == pytest.approx(0.8, abs=0.02) and pm.coefs["success"] == pytest.approx(1.5, abs=0.02)
    assert abs(pm.coefs["havoc"]) < 0.02


def test_fit_v3_recovers_blend_zeroes_irrelevant_context_and_drops_nuisance():
    t = synthetic_table()
    w = v3_fit.fit_v3(t)
    m, tot = w.margin.coefs, w.total.coefs
    assert m["margin_v2"] == pytest.approx(0.7, abs=0.05) and m["non_neutral"] == pytest.approx(1.5, abs=0.3)
    assert m["short_week_diff"] == pytest.approx(2.0, abs=0.3)
    assert m["travel_far_diff"] == 0.0 and m["talent_gap"] == 0.0 and m["bye_diff"] == 0.0   # can be fitted to 0
    assert tot["wind_excess"] == pytest.approx(-0.4, abs=0.05)
    assert tot["cold_excess"] == 0.0 and tot["precip"] == 0.0
    assert "weather_missing" not in tot                       # nuisance dropped at serve time
    assert w.meta["sigma_margin"] == pytest.approx(3.0, abs=0.4) and w.meta["n_train"] == len(t)
    assert w.meta["train_mae"]["margin"] < 3.0


def test_fit_v3_only_reads_training_seasons():
    t = synthetic_table()
    base = v3_fit.fit_v3(t)
    poisoned = t.copy()
    mask = poisoned["season"] == 2022
    poisoned.loc[mask, "actual_margin"] = 999.0
    held = v3_fit.fit_v3(poisoned, train_seasons=tuple(range(2016, 2022)))
    ref = v3_fit.fit_v3(t[t["season"] != 2022], train_seasons=tuple(range(2016, 2022)))
    assert held.margin == ref.margin and held.total == ref.total
    assert base.margin != held.margin
```

Create `tests/scripts/test_fit_cfb_v3.py`:

```python
import importlib.util
import json
import pathlib

from sportsmodel.cfb import v3

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "fit_cfb_v3.py"
_s = importlib.util.spec_from_file_location("fit_cfb_v3", _p)
fit_script = importlib.util.module_from_spec(_s)
_s.loader.exec_module(fit_script)


def test_write_outputs_roundtrips_weights_and_emits_a_loadable_gameline(tmp_path):
    w = v3.V3Weights(v3.LinearBlend(20.0, {"success": 3.0}), v3.LinearBlend(0.0, {"margin_v2": 1.0}),
                     v3.LinearBlend(1.0, {"total_v2": 1.0}), {"sigma_margin": 15.5, "sigma_total": 16.5})
    fit_script.write_outputs(w, tmp_path)
    assert v3.load_v3_weights(tmp_path / "v3_weights.json") == w
    gl = json.loads((tmp_path / "gameline_v3.json").read_text())
    assert gl["sigma_margin"] == 15.5 and gl["sigma_total"] == 16.5 and gl["bias_margin"] == 0.0
```

Append to `tests/cfb/test_v3_data.py`:

```python
def test_load_v3_inputs_reads_configs_and_tolerates_missing_optional_assets(tmp_path):
    import json
    write(tmp_path, "advanced_games.parquet", pd.DataFrame({
        "season": [2023, 2023], "week": [1, 1], "season_type": ["regular"] * 2, "game_id": [1, 1],
        "team": ["a", "b"], "opponent": ["b", "a"], "off_plays": [70.0, 60.0], "off_ppa": [0.2, 0.1],
        "off_success": [0.5, 0.4], "off_explosiveness": [1.2, 1.1], "off_pass_ppa": [0.2, 0.1],
        "off_rush_ppa": [0.1, 0.0], "off_pass_plays": [40.0, 30.0], "off_rush_plays": [30.0, 30.0]}))
    write(tmp_path, "cfbd_games.parquet", pd.DataFrame({
        "game_id": [1], "season": [2023], "venue_id": [7.0], "home_team": ["a"], "neutral_site": [False]}))
    write(tmp_path, "schedules.parquet", pd.DataFrame({"game_pk": [1], "week": [1], "game_type": ["REG"]}))
    (tmp_path / "rating.json").write_text(json.dumps({"k": 40, "hfa_elo": 70, "carryover": 0.9,
                                                      "w_sos": 0.45, "srs_min_games": 3}))
    inp = v3_data.load_v3_inputs(tmp_path)                    # no weights/decay/eff json: documented defaults
    assert inp.elo_cfg.k == 40 and inp.blend_cfg.srs_min_games == 3
    assert inp.eff_cfg.ridge == 4.0 and inp.decay.half_life_games > 0
    assert inp.talent is None and inp.priors_rows == {} and len(inp.eff_games) == 2
    assert inp.ctx.game_venue == {1: 7}


def test_load_v3_inputs_is_strict_about_prior_weights_when_priors_exist(tmp_path):
    import json
    write(tmp_path, "priors.parquet", pd.DataFrame({"season": [2023], "team_espn_id": ["a"]}))
    (tmp_path / "rating.json").write_text(json.dumps({"k": 40, "hfa_elo": 70, "carryover": 0.9,
                                                      "w_sos": 0.45, "srs_min_games": 3}))
    with pytest.raises(FileNotFoundError, match="priors_weights.json"):
        v3_data.load_v3_inputs(tmp_path)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_v3_fit.py tests/scripts/test_fit_cfb_v3.py tests/cfb/test_v3_data.py -q`
Expected: FAIL (`ImportError: cannot import name 'v3_fit'`; `AttributeError: module ... has no attribute 'load_v3_inputs'`).

- [ ] **Step 3: Implement the fit**

Create `src/sportsmodel/cfb/v3_fit.py`:

```python
"""Fit the v3 weights on the training seasons of the v3 feature table (2016-2022). PURE.

1. points map: ridge-regularised least squares of each offense's actual points on its efficiency
   side features (both sides of every game stacked).
2. blends: stage 1 = least squares on the core terms (v2 / efficiency / prior / home field);
   stage 2 = lasso on the stage-1 residual for the context terms, the penalty chosen by
   leave-one-season-out MAE with the one-standard-error rule, so a context term that does not
   help is fitted to EXACTLY 0 (the kept terms are refit unpenalised, "relaxed lasso").
   `weather_missing` is a nuisance column: fitted (so the other weather terms are estimated from
   games that have weather) but dropped from the served model.
3. sigmas: RMSE of the fitted v3 margin / total on the training seasons (moneyline mapping).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso

from .context import MARGIN_CTX, NUISANCE, TOTAL_CTX
from .efficiency import POINT_FEATURES
from .v3 import (MARGIN_CORE, TOTAL_CORE, LinearBlend, V3Weights, eff_margin_total, predict_v3)

TRAIN_SEASONS = tuple(range(2016, 2023))
LASSO_ALPHAS = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)
POINTS_RIDGE = 1.0


def _design(df: pd.DataFrame, cols) -> np.ndarray:
    return np.nan_to_num(df[list(cols)].to_numpy(float), nan=0.0)


def fit_points_map(table: pd.DataFrame, train_seasons=TRAIN_SEASONS, ridge: float = POINTS_RIDGE) -> LinearBlend:
    """Ridge LS of a side's actual points on POINT_FEATURES (intercept unpenalised)."""
    t = table[table["season"].isin(train_seasons) & table["actual_margin"].notna()]
    home_pts, away_pts = (t["actual_total"] + t["actual_margin"]) / 2, (t["actual_total"] - t["actual_margin"]) / 2
    x = np.vstack([_design(t.rename(columns={f"h_{f}": f for f in POINT_FEATURES}), POINT_FEATURES),
                   _design(t.rename(columns={f"a_{f}": f for f in POINT_FEATURES}), POINT_FEATURES)])
    y = np.concatenate([home_pts.to_numpy(float), away_pts.to_numpy(float)])
    x1 = np.c_[np.ones(len(x)), x]
    pen = np.eye(x1.shape[1]) * ridge
    pen[0, 0] = 0.0
    beta = np.linalg.solve(x1.T @ x1 + pen, x1.T @ y)
    return LinearBlend(float(beta[0]), {f: float(b) for f, b in zip(POINT_FEATURES, beta[1:])})


def add_eff(table: pd.DataFrame, pm: LinearBlend) -> pd.DataFrame:
    """table + margin_eff / total_eff columns from the points map."""
    out = table.copy()
    mt = [eff_margin_total(pm, r) for r in table.to_dict("records")]
    out["margin_eff"] = [m for m, _ in mt]
    out["total_eff"] = [t for _, t in mt]
    return out


def fit_blend(df: pd.DataFrame, y: np.ndarray, core, ctx, nuisance=(), alphas=LASSO_ALPHAS) -> tuple[LinearBlend, dict]:
    """Two-stage fit (see module docstring). Returns (blend, info)."""
    x1 = np.c_[np.ones(len(df)), _design(df, core)]
    beta = np.linalg.lstsq(x1, y, rcond=None)[0]
    resid = y - x1 @ beta
    zc = list(ctx) + list(nuisance)
    z = _design(df, zc)
    sd = z.std(axis=0)
    sd[sd == 0] = 1.0
    zs = z / sd
    seasons = df["season"].to_numpy()

    def lasso(a, tr):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            return Lasso(alpha=a, max_iter=20000).fit(zs[tr], resid[tr])

    abs_err = {None: np.abs(resid)}
    for a in alphas:
        errs = []
        for s in np.unique(seasons):
            te, tr = seasons == s, seasons != s
            errs.append(np.abs(resid[te] - lasso(a, tr).predict(zs[te])))
        abs_err[a] = np.concatenate(errs)
    cv = {a: float(e.mean()) for a, e in abs_err.items()}
    best_a = min(cv, key=cv.get)
    se = float(abs_err[best_a].std(ddof=1) / np.sqrt(len(abs_err[best_a])))
    # one-standard-error rule: the most penalised model within 1 SE of the best CV MAE, so a
    # context term has to earn its place (None = no context terms at all = the largest penalty)
    ok = [a for a in cv if cv[a] <= cv[best_a] + se]
    best = None if None in ok else max(ok)
    coefs = {c: 0.0 for c in ctx}
    intercept = float(beta[0])
    if best is not None:
        m = lasso(best, np.ones(len(df), bool))
        # relaxed lasso: the penalty only SELECTS the context terms; the kept ones (plus the
        # nuisance columns) are refit by plain least squares so their sizes are not shrunk
        keep = [i for i, c in enumerate(zc) if m.coef_[i] != 0.0 or c in nuisance]
        if keep:
            xs = np.c_[np.ones(len(df)), zs[:, keep]]
            b = np.linalg.lstsq(xs, resid, rcond=None)[0]
            coefs.update({zc[i]: float(b[1 + j] / sd[i]) for j, i in enumerate(keep) if zc[i] not in nuisance})
            intercept += float(b[0])
    core_coefs = {c: float(b) for c, b in zip(core, beta[1:])}
    return (LinearBlend(intercept, {**core_coefs, **coefs}),
            {"lasso_alpha": best, "cv_mae": {str(k): v for k, v in cv.items()}})


def fit_v3(table: pd.DataFrame, train_seasons=TRAIN_SEASONS) -> V3Weights:
    """Fit the whole of v3 on `train_seasons` of `table` (build_table output)."""
    pm = fit_points_map(table, train_seasons)
    tr = add_eff(table[table["season"].isin(train_seasons) & table["actual_margin"].notna()], pm)
    margin, mi = fit_blend(tr, tr["actual_margin"].to_numpy(float), MARGIN_CORE, MARGIN_CTX)
    total, ti = fit_blend(tr, tr["actual_total"].to_numpy(float), TOTAL_CORE, TOTAL_CTX, NUISANCE)
    w = V3Weights(pm, margin, total, {})
    preds = np.array([predict_v3(r, w) for r in tr.to_dict("records")])
    sig_m = float(np.sqrt(np.mean((preds[:, 0] - tr["actual_margin"].to_numpy()) ** 2)))
    sig_t = float(np.sqrt(np.mean((preds[:, 1] - tr["actual_total"].to_numpy()) ** 2)))
    meta = {"train_seasons": [int(s) for s in train_seasons], "n_train": int(len(tr)),
            "sigma_margin": sig_m, "sigma_total": sig_t, "margin": mi, "total": ti,
            "train_mae": {"margin": float(np.abs(preds[:, 0] - tr["actual_margin"]).mean()),
                          "total": float(np.abs(preds[:, 1] - tr["actual_total"]).mean())}}
    return V3Weights(pm, margin, total, meta)
```

- [ ] **Step 4: Append the v3 input loaders**

Append to `src/sportsmodel/cfb/v3_data.py`:

```python
def load_rating(assets: Path = ASSETS):
    """(EloConfig, BlendConfig) from assets/cfb/rating.json -- same fields generate_cfb.load_rating reads."""
    import json

    from sportsmodel.nfl.elo import EloConfig
    from sportsmodel.nfl.ratings import BlendConfig
    j = json.loads((Path(assets) / "rating.json").read_text())
    return (EloConfig(k=j["k"], hfa_elo=j["hfa_elo"], carryover=j["carryover"], base=j.get("base", 1500.0)),
            BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"]))


def load_v3_inputs(assets: Path = ASSETS, *, weather: pd.DataFrame | None = None):
    """Everything build_table needs, from the committed assets. `weather` (optional) overrides
    weather_games.parquet -- the live producer passes the day's forecast rows merged over it."""
    from .context import build_context_assets
    from .efficiency import load_eff_config
    from .priors import load_decay_config, load_weights
    from .v3_table import V3Inputs
    assets = Path(assets)
    if (assets / "priors.parquet").exists() and not (assets / "priors_weights.json").exists():
        # same strictness as generate_cfb.load_live_prior_weights: PriorWeights() defaults are the
        # wrong unit (one SP+ point = one Elo point) and must never be a silent fallback
        raise FileNotFoundError(f"{assets / 'priors_weights.json'} is missing but priors.parquet exists; "
                                "run scripts/backtest_cfb_priors.py")
    elo_cfg, blend_cfg = load_rating(assets)
    talent = read_asset("talent.parquet", assets)
    wx = weather if weather is not None else read_asset("weather_games.parquet", assets)
    ctx = build_context_assets(read_asset("venues.parquet", assets), require_asset("cfbd_games.parquet", assets),
                               wx, talent)
    return V3Inputs(elo_cfg, blend_cfg, load_weights(assets / "priors_weights.json"),
                    load_decay_config(assets / "priors_decay.json"), load_eff_config(assets / "eff_config.json"),
                    load_eff_games(assets), load_priors_rows(assets), talent, ctx)
```

- [ ] **Step 5: Implement the fit script**

Create `scripts/fit_cfb_v3.py`:

```python
"""Fit cfb-ratings-v3 on 2016-2022 and write assets/cfb/v3_weights.json + gameline_v3.json.

Builds the leak-free v3 feature table over 2015-2025 (one walk-forward), fits the efficiency
points map, the margin/total blends (context terms by lasso, any of them can land on exactly 0)
and the moneyline sigmas on the TRAINING seasons only, and prints the fit. Nothing is judged
here -- the held-out comparison against v2 is scripts/gate_cfb_v3.py. Local compute only.

Usage:
    uv run python scripts/fit_cfb_v3.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import v3, v3_data, v3_fit, v3_table  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
WALK_SEASONS = tuple(range(2015, 2026))


def write_outputs(w: v3.V3Weights, assets_dir: Path) -> None:
    (assets_dir / "v3_weights.json").write_text(w.to_json())
    gl = v3.gameline_v3_dict(w.meta["sigma_margin"], w.meta["sigma_total"])
    (assets_dir / "gameline_v3.json").write_text(json.dumps(gl, indent=2) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="fit and print; write nothing")
    args = ap.parse_args(argv)
    t0 = time.time()
    inp = v3_data.load_v3_inputs(ASSETS)
    table = v3_table.build_table(v3_data.load_merged_schedule(ASSETS), inp, seasons=WALK_SEASONS)
    print(f"feature table: {len(table)} FBS-vs-FBS games, {time.time() - t0:.0f}s", flush=True)
    w = v3_fit.fit_v3(table)
    print(f"points map: {json.dumps(w.points_map.to_dict())}")
    print(f"margin blend: {json.dumps(w.margin.to_dict())}")
    print(f"total blend:  {json.dumps(w.total.to_dict())}")
    zeros = sorted(k for b in (w.margin, w.total) for k, c in b.coefs.items() if c == 0.0)
    print(f"terms fitted to exactly 0: {zeros}")
    print(f"train 2016-2022: n={w.meta['n_train']} margin MAE={w.meta['train_mae']['margin']:.3f} "
          f"total MAE={w.meta['train_mae']['total']:.3f} sigma_margin={w.meta['sigma_margin']:.3f} "
          f"sigma_total={w.meta['sigma_total']:.3f}")
    if args.dry_run:
        print("--dry-run: nothing written")
        return
    write_outputs(w, ASSETS)
    print(f"wrote assets/cfb/v3_weights.json and gameline_v3.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_v3_fit.py tests/scripts/test_fit_cfb_v3.py tests/cfb/test_v3_data.py tests/cfb/test_v3.py tests/cfb/test_v3_table.py -q`
Expected: PASS (3 + 1 + 5 + 5 + 6): the points map recovers known coefficients, the blend recovers known weights, irrelevant context terms are exactly 0.0, `weather_missing` is dropped, and the fit never reads a season outside `train_seasons`.

- [ ] **Step 7: [LOCAL RUN] Fit on 2016-22 and commit the weights**

Needs the Task 4 assets and Task 6's `eff_config.json`; no network (about 30 seconds).

```bash
uv run python scripts/fit_cfb_v3.py
```
Expected: `feature table: ~7,900 FBS-vs-FBS games`, the printed points map (positive `ppa_plays` / `success`), margin and total blends, `terms fitted to exactly 0: [...]`, train MAE / sigma, `wrote assets/cfb/v3_weights.json and gameline_v3.json`. Sanity: `margin_v2` coefficient between 0.3 and 1.0 and no coefficient is NaN; nothing here is a pass/fail - the verdict is Task 10.

```bash
git add src/sportsmodel/cfb/v3_fit.py src/sportsmodel/cfb/v3_data.py scripts/fit_cfb_v3.py tests/cfb/test_v3_fit.py tests/cfb/test_v3_data.py tests/scripts/test_fit_cfb_v3.py assets/cfb/v3_weights.json assets/cfb/gameline_v3.json
git commit -m "feat(cfb): fit cfb-ratings-v3 on 2016-22 (points map, blends, context lasso) -> v3_weights.json, gameline_v3.json"
```

---

### Task 10: The gate - v2 baseline reproduction, v3 vs v2, residual check -> `v3_gate.json`

**Files:**
- Create: `src/sportsmodel/cfb/v3_gate.py`, `src/sportsmodel/cfb/residual.py`, `scripts/gate_cfb_v3.py`
- Test: `tests/cfb/test_v3_gate.py`, `tests/cfb/test_residual.py`, `tests/scripts/test_gate_cfb_v3.py`

**Interfaces:**
- Consumes: `v3_table.build_table`, `v3.{load_v3_weights, load_gameline_config, predict_v3, V3Weights}`, `v3_data.*`, `nfl.gameline.build_gameline`, Task 2's `advanced_games.parquet` extra columns, Task 4's `cfbd_games` / `prior_ratings`.
- Produces:
  - `v3_gate.HOLDOUT_SEASONS = (2023, 2024, 2025)`, `V2_EXPECTED = {"margin_mae": 12.61, "total_mae": 13.10, "ats": 0.490}`, `BASELINE_TOL`, `BaselineMismatch`; `eval_set(table, seasons=HOLDOUT_SEASONS)`, `mae`, `side_accuracy(pred, line, actual) -> (acc, n)`, `log_loss`, `ece(p, y, bins=10)`, `metrics(df, margin_col, total_col, prob_col) -> dict` (`n, margin_mae, total_mae, ats, n_ats, ou, n_ou, ml_logloss, ml_ece`), `by_season(...)`, `check_baseline(measured, expected=V2_EXPECTED, tol=BASELINE_TOL)` (raises `BaselineMismatch`), `verdict(v2, v3, v2_seasons=None, v3_seasons=None) -> {"criteria", "ship", "season_flags"}`, `clean(x)` (JSON-safe).
  - `residual.prior_game_means(adv, sched)`, `residual.residual_features(table, means, meta, priors_rows, prior_ratings, margin_coefs, total_coefs, points_coefs) -> DataFrame`, `residual.run_residual_check(feat, resid_margin, resid_total, seasons, train_seasons, test_seasons) -> dict` (ridge + gradient boosting: `r2_oos`, `mae_change`, per-season changes, top features, `v4_candidates`).
  - `gate_cfb_v3.add_predictions(table, w, gl_v2, gl_v3)` (adds `v2_/v3_` `margin, total, wp`), `evaluate_gate(scored, expected=..., tol=...) -> dict`, `render_report(res) -> str`, `main() -> int` (exit 2 and nothing written if the v2 baseline is not reproduced); outputs `assets/cfb/v3_gate.json` and `docs/superpowers/reports/<date>-cfb-v3-gate.md`.

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_v3_gate.py`:

```python
import json
import math

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3_gate as g


def frame(**cols):
    base = {"season": [2023, 2023, 2024, 2025], "actual_margin": [7.0, -3.0, 10.0, -14.0],
            "actual_total": [50.0, 41.0, 63.0, 38.0], "market_spread": [3.5, -6.5, 14.0, -3.0],
            "market_total": [47.5, 44.5, np.nan, 38.0]}
    base.update(cols)
    return pd.DataFrame(base)


def test_mae_and_side_accuracy_skip_pushes_and_no_pick_games():
    assert g.mae([1, 5], [3, 1]) == 3.0
    # lines 3.5 / -6.5 / 14 / -3 ; model margins 6 (home) / -10 (away) / 14 (no pick) / 0 (home)
    acc, n = g.side_accuracy([6.0, -10.0, 14.0, 0.0], [3.5, -6.5, 14.0, -3.0], [7.0, -3.0, 10.0, -14.0])
    # g1: model home, actual 7>3.5 home -> hit ; g2: model away(-10<-6.5), actual -3>-6.5 home -> miss
    # g3: pred == line -> skipped ; g4: model home (0>-3), actual -14<-3 away -> miss
    assert n == 3 and acc == pytest.approx(1 / 3)
    acc2, n2 = g.side_accuracy([1.0], [1.0], [1.0])
    assert math.isnan(acc2) and n2 == 0
    _, n3 = g.side_accuracy([5.0], [4.0], [4.0])                 # actual == line: a push
    assert n3 == 0


def test_log_loss_and_ece():
    assert g.log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))
    assert g.log_loss([1.0], [0.0]) > 10                          # clipped, finite
    perfect = g.ece([0.1] * 10 + [0.9] * 10, [0] * 9 + [1] + [1] * 9 + [0])
    assert perfect == pytest.approx(0.0)
    assert g.ece([0.9] * 10, [0] * 10) == pytest.approx(0.9)


def test_metrics_on_the_eval_frame():
    df = frame(pm=[6.0, -10.0, 14.0, 0.0], pt=[49.0, 40.0, 60.0, 40.0], pw=[0.8, 0.2, 0.9, 0.4])
    m = g.metrics(df, "pm", "pt", "pw")
    assert m["n"] == 4 and m["margin_mae"] == pytest.approx((1 + 7 + 4 + 14) / 4)
    assert m["total_mae"] == pytest.approx((1 + 1 + 3 + 2) / 4)
    assert m["n_ou"] == 2 and m["ou"] == 1.0   # game 3 has no closing total, game 4 pushes (38 vs 38)
    assert 0 < m["ml_logloss"] < 2 and 0 <= m["ml_ece"] <= 1


def test_eval_set_filters_season_scored_and_priced():
    t = frame()
    t.loc[1, "market_spread"] = np.nan
    t.loc[2, "actual_margin"] = np.nan
    assert list(g.eval_set(t)["season"]) == [2023, 2025]
    assert len(g.eval_set(t, seasons=(2024,))) == 0


def test_check_baseline_passes_within_rounding_and_stops_otherwise():
    ok = {"margin_mae": 12.6069, "total_mae": 13.0980, "ats": 0.48964}
    assert g.check_baseline(ok)["margin_mae"] == (12.61, 12.6069)
    for bad in ({**ok, "margin_mae": 12.70}, {**ok, "total_mae": 13.2}, {**ok, "ats": 0.50}):
        with pytest.raises(g.BaselineMismatch, match="NOT reproduced"):
            g.check_baseline(bad)


V2 = {"margin_mae": 12.61, "total_mae": 13.10, "ats": 0.49, "ou": 0.50, "ml_logloss": 0.60, "ml_ece": 0.020}


def test_verdict_every_criterion_decides_ship():
    better = {"margin_mae": 12.5, "total_mae": 13.0, "ats": 0.49, "ou": 0.50, "ml_logloss": 0.60, "ml_ece": 0.024}
    v = g.verdict(V2, better)
    assert v["ship"] and all(c["pass"] for c in v["criteria"].values())     # ties pass ats/ou/logloss; ece within slack
    for key, bad in (("margin_mae", 12.61), ("total_mae", 13.10), ("ats", 0.4899), ("ou", 0.4999),
                     ("ml_logloss", 0.6001), ("ml_ece", 0.0251)):
        worse = {**better, key: bad}
        out = g.verdict(V2, worse)
        assert not out["ship"] and not out["criteria"][key]["pass"], key


def test_verdict_flags_a_worse_season_without_changing_the_combined_verdict():
    better = {"margin_mae": 12.5, "total_mae": 13.0, "ats": 0.5, "ou": 0.5, "ml_logloss": 0.59, "ml_ece": 0.02}
    s2 = {2023: dict(V2), 2024: dict(V2)}
    s3 = {2023: {**better}, 2024: {**better, "margin_mae": 12.9}}
    v = g.verdict(V2, better, s2, s3)
    assert v["ship"] is True
    assert v["season_flags"] == [{"season": 2024, "criterion": "margin_mae", "v2": 12.61, "v3": 12.9}]


def test_clean_makes_json_safe():
    out = g.clean({2023: {"a": np.float64("nan"), "b": np.int64(3), "c": [np.float64(1.5), np.bool_(True)]}})
    assert out == {"2023": {"a": None, "b": 3, "c": [1.5, True]}}
    json.dumps(out, allow_nan=False)
```

Create `tests/cfb/test_residual.py`:

```python
import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import residual as rs
from sportsmodel.cfb.efficiency import POINT_FEATURES


def test_prior_game_means_are_expanding_means_of_earlier_games_only():
    adv = pd.DataFrame({"season": [2023] * 4, "season_type": ["regular"] * 3 + ["postseason"],
                        "week": [1, 2, 3, 1], "game_id": [1, 2, 3, 9], "team": ["a"] * 4,
                        "off_std_down_success": [0.4, 0.6, 0.8, 0.1]})
    sched = pd.DataFrame({"game_pk": [1, 2, 3], "week": [1, 2, 3], "game_type": ["REG"] * 3})
    m = rs.prior_game_means(adv, sched).set_index("game_id")
    assert np.isnan(m.loc[1, "std_down_success"])                 # opener: nothing earlier
    assert m.loc[2, "std_down_success"] == pytest.approx(0.4)
    assert m.loc[3, "std_down_success"] == pytest.approx(0.5)
    assert 9 not in m.index                                       # postseason never feeds in-season means
    assert np.isnan(m.loc[2, "line_yards"])                       # column absent from the pull -> NaN


def test_residual_features_are_home_minus_away_zero_filled_and_leak_scoped():
    table = pd.DataFrame({"season": [2023, 2023], "game_pk": [1, 2], "home_team": ["a", "a"],
                          "away_team": ["b", "c"], "week": [2, 3], "margin_v2": [0.0, 0.0],
                          "talent_gap": [0.3, 0.1], "travel_far_diff": [1.0, 0.0]})
    for f in POINT_FEATURES:
        table[f"h_{f}"], table[f"a_{f}"] = 0.0, 0.0
    table["h_havoc"], table["a_havoc"] = [0.2, 0.1], [0.05, 0.0]
    table["h_ppa_plays"] = [1.0, 1.0]
    means = pd.DataFrame({"game_id": [1, 1, 2, 2], "team": ["a", "b", "a", "c"],
                          "std_down_success": [0.6, 0.5, 0.7, np.nan]})
    for c in rs.DOWN_COLS:
        if c not in means:
            means[c] = np.nan
    meta = pd.DataFrame({"game_id": [1, 2], "home_pregame_elo": [1700.0, 1650.0],
                         "away_pregame_elo": [1600.0, np.nan]})
    ratings = pd.DataFrame({"season": [2022, 2022, 2023], "team": ["a", "b", "a"],
                            "fpi": [10.0, 4.0, 99.0], "srs": [8.0, 3.0, 99.0]})
    priors = {2023: [{"team_espn_id": "a", "returning_starters": 0.7},
                     {"team_espn_id": "b", "returning_starters": 0.5}]}
    pts = {f: 1.0 for f in POINT_FEATURES}
    pts["havoc"] = 0.0
    f = rs.residual_features(table, means, meta, priors, ratings,
                             {"talent_gap": 0.4, "travel_far_diff": 0.0}, {}, pts)
    assert f.loc[0, "cfbd_elo_diff"] == 100.0 and f.loc[1, "cfbd_elo_diff"] == 0.0     # NaN -> 0
    assert f.loc[0, "fpi_diff_prev"] == 6.0                       # 2022 ratings for 2023 games, not the 2023 row
    assert f.loc[1, "fpi_diff_prev"] == 0.0                       # team c unrated
    assert f.loc[0, "returning_usage_diff"] == pytest.approx(0.2)
    assert f.loc[0, "std_down_success_diff"] == pytest.approx(0.1)
    assert "unused_travel_far_diff" in f and "unused_talent_gap" not in f   # only 0-weight terms
    assert f.loc[0, "unused_pt_havoc"] == pytest.approx(0.15) and "unused_pt_ppa_plays" not in f
    assert not f.isna().any().any()


def synth(n=3000, seed=4, signal=True):
    rng = np.random.default_rng(seed)
    feat = pd.DataFrame({"signal": rng.normal(size=n), "noise1": rng.normal(size=n), "noise2": rng.normal(size=n)})
    seasons = rng.choice(np.arange(2016, 2026), n)
    base = rng.normal(0, 3, n)
    resid_m = base + (0.8 * feat["signal"].to_numpy() if signal else 0.0)
    return feat, resid_m, base.copy(), seasons


def test_check_finds_a_real_signal_in_all_held_out_seasons():
    feat, rm, rt, seasons = synth()
    out = rs.run_residual_check(feat, rm, rt, seasons, range(2016, 2023), (2023, 2024, 2025))
    m = out["margin"]
    assert m["ridge"]["r2_oos"] > 0.03 and m["ridge"]["mae_change"] < 0
    assert m["ridge"]["top_features"][0]["feature"] == "signal"
    assert m["hgb"]["top_features"][0]["feature"] == "signal"
    assert {c["feature"] for c in m["v4_candidates"]} == {"signal"}
    assert set(m["ridge"]["mae_change_by_season"]) == {2023, 2024, 2025}
    assert out["total"]["v4_candidates"] == [] and out["total"]["ridge"]["r2_oos"] < 0.01
    assert "report only" in out["note"]


def test_pure_noise_gives_no_edge():
    feat, rm, rt, seasons = synth(signal=False)
    out = rs.run_residual_check(feat, rm, rt, seasons, range(2016, 2023), (2023, 2024, 2025))
    assert out["margin"]["ridge"]["r2_oos"] < 0.01
    assert out["margin"]["hgb"]["r2_oos"] < 0.01
```

Create `tests/scripts/test_gate_cfb_v3.py`:

```python
import importlib.util
import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import v3, v3_gate
from sportsmodel.cfb.context import CONTEXT_COLS
from sportsmodel.cfb.efficiency import POINT_FEATURES
from sportsmodel.nfl.gameline import GameLineConfig

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "gate_cfb_v3.py"
_s = importlib.util.spec_from_file_location("gate_cfb_v3", _p)
gate = importlib.util.module_from_spec(_s)
_s.loader.exec_module(gate)

GL = GameLineConfig(sigma_margin=15.0, sigma_total=15.0, offset=110, total_max=150)


def raw_table(n=60, seed=0):
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({"season": np.repeat([2023, 2024, 2025], n // 3), "week": 5, "game_pk": np.arange(n)})
    t["margin_v2"], t["total_v2"] = rng.normal(3, 10, n), rng.normal(55, 6, n)
    t["actual_margin"] = t["margin_v2"] + rng.normal(0, 12, n)
    t["actual_total"] = t["total_v2"] + rng.normal(0, 12, n)
    t["market_spread"] = t["margin_v2"].round() + 0.5
    t["market_total"] = t["total_v2"].round() + 0.5
    t["prior_margin"], t["non_neutral"] = 0.0, 1.0
    for f in POINT_FEATURES:
        t[f"h_{f}"], t[f"a_{f}"] = 0.0, 0.0
    for c in CONTEXT_COLS:
        t[c] = 0.0
    return t


W = v3.V3Weights(v3.LinearBlend(28.0, {}), v3.LinearBlend(0.0, {"margin_v2": 1.0}),
                 v3.LinearBlend(0.0, {"total_v2": 1.0}))


def test_add_predictions_serves_v2_through_its_gameline_and_v3_through_the_blend():
    t = raw_table()
    gl2 = GameLineConfig(sigma_margin=15.0, sigma_total=15.0, offset=110, total_max=150, bias_margin=-0.5)
    out = gate.add_predictions(t, W, gl2, GL)
    assert np.allclose(out["v2_margin"], t["margin_v2"] + 0.5)             # served = model - bias
    assert np.allclose(out["v3_margin"], t["margin_v2"])                   # identity blend, no bias
    assert ((out["v3_wp"] > 0) & (out["v3_wp"] < 1)).all()
    assert (out["v3_wp"] > 0.5).eq(out["v3_margin"] > 0).all()


def test_evaluate_gate_reproduces_baseline_then_compares_v3():
    out = gate.add_predictions(raw_table(), W, GL, GL)       # v3 == v2 here: ties
    ev = v3_gate.eval_set(out)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    expected = {k: round(v2c[k], 4) for k in ("margin_mae", "total_mae", "ats")}
    res = gate.evaluate_gate(out, expected=expected)
    assert res["baseline"]["reproduced"] and res["n_games"] == len(ev)
    assert res["ship"] is False                              # strict < on the MAEs: a tie does not ship
    assert res["criteria"]["margin_mae"]["pass"] is False and res["criteria"]["ats"]["pass"] is True
    assert set(res["v3"]["by_season"]) == {2023, 2024, 2025}
    json.dumps(v3_gate.clean(res), allow_nan=False)


def test_evaluate_gate_stops_on_baseline_mismatch_before_scoring_v3():
    out = gate.add_predictions(raw_table(), W, GL, GL)
    with pytest.raises(v3_gate.BaselineMismatch):
        gate.evaluate_gate(out)                              # the real published numbers do not match noise data


def test_better_v3_ships_and_report_renders():
    t = raw_table()
    t2 = t.copy()
    w_better = v3.V3Weights(v3.LinearBlend(28.0, {}), v3.LinearBlend(0.0, {"margin_v2": 0.5}),
                            v3.LinearBlend(0.0, {"total_v2": 1.0}))
    out = gate.add_predictions(t2, w_better, GL, GL)
    # make v3's margin strictly better on the eval set by moving it toward the actual margin
    out["v3_margin"] = 0.5 * out["v2_margin"] + 0.5 * out["actual_margin"]
    out["v3_total"] = 0.5 * out["v2_total"] + 0.5 * out["actual_total"]
    ev = v3_gate.eval_set(out)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    res = gate.evaluate_gate(out, expected={k: round(v2c[k], 4) for k in ("margin_mae", "total_mae", "ats")})
    res.update({"generated": "2026-10-07", "residual_check": None})
    assert res["criteria"]["margin_mae"]["pass"] and res["criteria"]["total_mae"]["pass"]
    md = gate.render_report(res)
    assert "Live model: v2 (unchanged)" in md and "Held-out [2023, 2024, 2025]" in md
    assert "| 2024 | v3 |" in md
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_v3_gate.py tests/cfb/test_residual.py tests/scripts/test_gate_cfb_v3.py -q`
Expected: FAIL (`ImportError: cannot import name 'v3_gate'` / `'residual'`; `FileNotFoundError` for `scripts/gate_cfb_v3.py`).

- [ ] **Step 3: Implement the pure gate**

Create `src/sportsmodel/cfb/v3_gate.py`:

```python
"""The v3 ship gate (spec section 4): pure metrics, the v2-baseline reproduction check and the
pass/fail verdict. No IO.

Eval set (identical to the one v2's 12.61 / 13.10 / 49.0 % were measured on): held-out 2023-2025
REG FBS-vs-FBS games that have a closing spread. O/U is scored on the subset with a closing total.
`market_spread` is in HOME-MARGIN convention (positive = home favored): the model picks home when
its margin is above the number. Pushes and no-pick games are excluded from ATS / O/U.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

HOLDOUT_SEASONS = (2023, 2024, 2025)
# v2's held-out numbers (compare_cfb_live_fix.py "fixed", 2026-09-30) -- reproduced FIRST
V2_EXPECTED = {"margin_mae": 12.61, "total_mae": 13.10, "ats": 0.490}
BASELINE_TOL = {"margin_mae": 0.006, "total_mae": 0.006, "ats": 0.0006}   # rounding of the published figures
ECE_SLACK = 0.005
ECE_BINS = 10


class BaselineMismatch(RuntimeError):
    """v2 run through the gate harness did not reproduce its published held-out numbers."""


def eval_set(table: pd.DataFrame, seasons=HOLDOUT_SEASONS) -> pd.DataFrame:
    return table[table["season"].isin(seasons) & table["actual_margin"].notna()
                 & table["market_spread"].notna()].reset_index(drop=True)


def mae(pred, actual) -> float:
    return float(np.mean(np.abs(np.asarray(pred, float) - np.asarray(actual, float))))


def side_accuracy(pred, line, actual) -> tuple[float, int]:
    """Share of decided games where the model's side of `line` matches the actual side
    (pushes and games with pred == line are skipped). Returns (accuracy, n decided)."""
    pred, line, actual = (np.asarray(x, float) for x in (pred, line, actual))
    ok = ~np.isnan(line) & (actual != line) & (pred != line)
    if not ok.any():
        return float("nan"), 0
    return float(np.mean((pred[ok] > line[ok]) == (actual[ok] > line[ok]))), int(ok.sum())


def log_loss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def ece(p, y, bins: int = ECE_BINS) -> float:
    """Expected calibration error: bin-size-weighted mean |mean(p) - mean(y)| over equal-width bins."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum()
                     for b in range(bins) if (idx == b).any()) / len(p))


def metrics(df: pd.DataFrame, margin_col: str, total_col: str, prob_col: str) -> dict:
    """Gate metrics of one model on an eval-set frame (needs actual_*, market_*, the three columns)."""
    ats, n_ats = side_accuracy(df[margin_col], df["market_spread"], df["actual_margin"])
    has_t = df["market_total"].notna()
    ou, n_ou = side_accuracy(df.loc[has_t, total_col], df.loc[has_t, "market_total"], df.loc[has_t, "actual_total"])
    dec = df["actual_margin"] != 0
    y = (df.loc[dec, "actual_margin"] > 0).astype(float)
    return {"n": int(len(df)), "margin_mae": mae(df[margin_col], df["actual_margin"]),
            "total_mae": mae(df[total_col], df["actual_total"]),
            "ats": ats, "n_ats": n_ats, "ou": ou, "n_ou": n_ou,
            "ml_logloss": log_loss(df.loc[dec, prob_col], y), "ml_ece": ece(df.loc[dec, prob_col], y)}


def by_season(df: pd.DataFrame, margin_col: str, total_col: str, prob_col: str) -> dict:
    return {int(s): metrics(g, margin_col, total_col, prob_col) for s, g in df.groupby("season")}


def check_baseline(measured: dict, expected: dict = V2_EXPECTED, tol: dict = BASELINE_TOL) -> dict:
    """Raise BaselineMismatch unless v2's measured combined numbers equal the published ones within
    rounding; returns {metric: (expected, measured)} on success."""
    diffs = {k: (expected[k], measured[k]) for k in expected}
    bad = {k: v for k, v in diffs.items() if abs(v[0] - v[1]) > tol[k]}
    if bad:
        raise BaselineMismatch("v2 baseline NOT reproduced (expected vs measured): "
                               + ", ".join(f"{k} {e} vs {m:.4f}" for k, (e, m) in bad.items()))
    return diffs


CRITERIA = (
    ("margin_mae", "v3 margin MAE < v2", lambda v2, v3: v3 < v2),
    ("total_mae", "v3 total MAE < v2", lambda v2, v3: v3 < v2),
    ("ats", "ATS vs closing spread >= v2", lambda v2, v3: v3 >= v2),
    ("ou", "O/U vs closing total >= v2", lambda v2, v3: v3 >= v2),
    ("ml_logloss", "ML log-loss <= v2", lambda v2, v3: v3 <= v2),
    ("ml_ece", f"ML ECE <= v2 + {ECE_SLACK}", lambda v2, v3: v3 <= v2 + ECE_SLACK),
)


def verdict(v2: dict, v3: dict, v2_seasons: dict | None = None, v3_seasons: dict | None = None) -> dict:
    """Pass/fail per criterion on the COMBINED numbers (the verdict) plus a list of (season,
    criterion) pairs where v3 is worse than v2 (flagged, never part of the verdict)."""
    crit = {k: {"label": label, "v2": v2[k], "v3": v3[k], "pass": bool(rule(v2[k], v3[k]))}
            for k, label, rule in CRITERIA}
    flags = []
    for s in sorted(v3_seasons or {}):
        for k, label, rule in CRITERIA:
            if not rule(v2_seasons[s][k], v3_seasons[s][k]):
                flags.append({"season": int(s), "criterion": k, "v2": v2_seasons[s][k], "v3": v3_seasons[s][k]})
    return {"criteria": crit, "ship": all(c["pass"] for c in crit.values()), "season_flags": flags}


def clean(x):
    """JSON-safe copy: NaN/inf -> None, numpy scalars -> python, int keys -> str."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x
```

- [ ] **Step 4: Implement the residual check**

Create `src/sportsmodel/cfb/residual.py`:

```python
"""Residual check (REPORT ONLY -- nothing here ships). Regularised models on v3's own training
residuals, using features v3 does not use, scored out of sample on the held-out seasons.

Features (all leak-free; each is home minus away):
  cfbd_elo_diff          CFBD's own PRE-game Elo (games.homePregameElo / awayPregameElo)
  fpi_diff_prev, srs_diff_prev       CFBD FPI / SRS of the PREVIOUS season (end-of-season ratings are
                         only ever joined to the next season)
  returning_usage_diff   priors.parquet returning_starters (CFBD returning-usage share)
  std_down_*, pass_down_*, line_yards, stuff_rate, power_success   season-to-date raw means of
                         earlier games (down/distance splits and run-game shape)
  unused_<ctx>, unused_pt_<f>   v3 context terms / efficiency points-map features fitted to weight 0
No market-derived field (CFBD pregame win probability, spreads, lines) is ever used.

A feature is a v4 candidate when, fitted alone on the training residuals, it cuts held-out MAE by
at least MIN_GAIN points in EVERY held-out season (the spec's "all three seasons" rule, with a
materiality floor so a 0.0001 coincidence does not count).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge, RidgeCV

from .efficiency import POINT_FEATURES

DOWN_COLS = {"std_down_success": "off_std_down_success", "pass_down_success": "off_pass_down_success",
             "std_down_ppa": "off_std_down_ppa", "pass_down_ppa": "off_pass_down_ppa",
             "line_yards": "off_line_yards", "stuff_rate": "off_stuff_rate",
             "power_success": "off_power_success"}
RIDGE_ALPHAS = (1.0, 10.0, 100.0, 1000.0, 10000.0)
MIN_GAIN = 0.005      # points of held-out MAE a feature must save in EVERY held-out season to be a v4 candidate


def prior_game_means(adv: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    """For every regular-season team-game: the team's mean of each DOWN_COLS metric over its EARLIER
    games of the season (NaN for the opener). Rows: game_id, team, <name>."""
    reg = sched[sched["game_type"] == "REG"] if "game_type" in sched else sched
    a = adv[adv["season_type"].fillna("regular") == "regular"] if "season_type" in adv else adv
    a = a.drop(columns=["week"]).merge(reg[["game_pk", "week"]].rename(columns={"game_pk": "game_id"}),
                                       on="game_id", how="inner")
    a = a.sort_values(["season", "team", "week"]).reset_index(drop=True)
    out = a[["game_id", "team"]].copy()
    grp = a.groupby(["season", "team"])
    for name, col in DOWN_COLS.items():
        out[name] = grp[col].transform(lambda s: s.expanding().mean().shift(1)) if col in a else np.nan
    return out


def residual_features(table: pd.DataFrame, means: pd.DataFrame, meta: pd.DataFrame | None,
                      priors_rows: dict, prior_ratings: pd.DataFrame | None,
                      margin_coefs: dict, total_coefs: dict, points_coefs: dict) -> pd.DataFrame:
    """Residual-check feature frame aligned to `table` rows (home - away; unknown -> 0.0)."""
    out = pd.DataFrame(index=table.index)
    pk, home, away, season = table["game_pk"], table["home_team"], table["away_team"], table["season"]
    if meta is not None:
        m = meta.drop_duplicates("game_id").set_index("game_id")
        out["cfbd_elo_diff"] = pk.map(m["home_pregame_elo"]) - pk.map(m["away_pregame_elo"])
    if prior_ratings is not None:
        pr = prior_ratings.set_index(["season", "team"])
        for col, name in (("fpi", "fpi_diff_prev"), ("srs", "srs_diff_prev")):
            lut = pr[col].to_dict()
            out[name] = [lut.get((s - 1, h), np.nan) - lut.get((s - 1, a), np.nan)
                         for s, h, a in zip(season, home, away)]
    usage = {(s, r["team_espn_id"]): r.get("returning_starters")
             for s, rows in priors_rows.items() for r in rows}
    out["returning_usage_diff"] = [_f(usage.get((s, h))) - _f(usage.get((s, a)))
                                   for s, h, a in zip(season, home, away)]
    idx = means.set_index(["game_id", "team"])
    for name in DOWN_COLS:
        lut = idx[name].to_dict()
        out[f"{name}_diff"] = [lut.get((g, h), np.nan) - lut.get((g, a), np.nan)
                               for g, h, a in zip(pk, home, away)]
    for c, v in {**margin_coefs, **total_coefs}.items():
        if v == 0.0 and c in table:
            out[f"unused_{c}"] = table[c]
    for f in POINT_FEATURES:
        if points_coefs.get(f, 0.0) == 0.0:
            out[f"unused_pt_{f}"] = table[f"h_{f}"] - table[f"a_{f}"]
    return out.astype(float).fillna(0.0)


def _f(x) -> float:
    return float("nan") if x is None else float(x)


def _mae_change(resid: np.ndarray, pred: np.ndarray, base: float) -> float:
    """MAE(resid - pred) minus MAE of the constant train-mean prediction (negative = the features
    help). The constant baseline keeps a pure intercept shift from counting as a feature win."""
    return float(np.mean(np.abs(resid - pred)) - np.mean(np.abs(resid - base)))


def check_target(feat: pd.DataFrame, resid: np.ndarray, seasons: np.ndarray,
                 train_seasons, test_seasons) -> dict:
    """Ridge + shallow gradient boosting fit on the train residuals; held-out R^2, MAE change
    vs the constant train-mean residual (negative = better) combined and per season, top features and v4 candidates."""
    tr, te = np.isin(seasons, list(train_seasons)), np.isin(seasons, list(test_seasons))
    feat = feat.loc[:, feat[tr].std(ddof=0) > 0]            # constant (all-neutral) columns carry nothing
    cols = list(feat.columns)
    mu, sd = feat[tr].mean(), feat[tr].std(ddof=0).replace(0, 1.0)
    z = ((feat - mu) / sd).to_numpy(float)
    base = float(resid[tr].mean())

    def summarize(pred_te: np.ndarray) -> dict:
        r = resid[te]
        sst = float(np.sum((r - base) ** 2))
        out = {"r2_oos": float(1 - np.sum((r - pred_te) ** 2) / sst) if sst > 0 else float("nan"),
               "mae_change": _mae_change(r, pred_te, base),
               "mae_change_by_season": {int(s): _mae_change(r[seasons[te] == s], pred_te[seasons[te] == s], base)
                                        for s in sorted(set(seasons[te]))}}
        return out

    ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(z[tr], resid[tr])
    res = {"ridge": {**summarize(ridge.predict(z[te])), "alpha": float(ridge.alpha_),
                     "top_features": [{"feature": cols[i], "std_coef": float(ridge.coef_[i])}
                                      for i in np.argsort(-np.abs(ridge.coef_))[:5]]}}
    hgb = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150, min_samples_leaf=100,
                                        l2_regularization=1.0, random_state=0).fit(z[tr], resid[tr])
    imp = permutation_importance(hgb, z[te], resid[te], n_repeats=3, random_state=0,
                                 scoring="neg_mean_absolute_error").importances_mean
    res["hgb"] = {**summarize(hgb.predict(z[te])),
                  "top_features": [{"feature": cols[i], "perm_importance": float(imp[i])}
                                   for i in np.argsort(-imp)[:5]]}
    cands = []
    for i, c in enumerate(cols):
        m = Ridge(alpha=float(ridge.alpha_)).fit(z[tr][:, [i]], resid[tr])
        per = summarize(m.predict(z[te][:, [i]]))["mae_change_by_season"]
        if per and all(v < -MIN_GAIN for v in per.values()):
            cands.append({"feature": c, "mae_change_by_season": per})
    res["v4_candidates"] = cands
    return res


def run_residual_check(feat: pd.DataFrame, resid_margin, resid_total, seasons,
                       train_seasons, test_seasons) -> dict:
    seasons = np.asarray(seasons)
    return {"features": list(feat.columns),
            "margin": check_target(feat, np.asarray(resid_margin, float), seasons, train_seasons, test_seasons),
            "total": check_target(feat, np.asarray(resid_total, float), seasons, train_seasons, test_seasons),
            "note": "report only: nothing from the residual models ships in cfb-ratings-v3"}
```

- [ ] **Step 5: Implement the gate script**

Create `scripts/gate_cfb_v3.py`:

```python
"""cfb-ratings-v3 ship gate: v2 baseline first, then v3 vs v2 on held-out 2023-2025, then the
report-only residual check. Writes assets/cfb/v3_gate.json and a markdown summary.

Order (spec section 4):
  1. v2 is run through THIS harness and must reproduce its published held-out numbers (margin
     MAE 12.61, total MAE 13.10, ATS 49.0 % on 2023-2025 FBS-vs-FBS games with a closing spread)
     within rounding. If it does not, the run STOPS (exit 2) before v3 is scored and nothing is
     written -- reconcile the harness first (compare against scripts/compare_cfb_live_fix.py).
  2. v3 ships only if, on the combined 2023-2025 numbers: margin MAE < v2, total MAE < v2,
     ATS >= v2, O/U >= v2, ML log-loss <= v2 and ECE <= v2 + 0.005. Per-season numbers are
     reported and any season where v3 is worse is flagged; the verdict uses the combined ones.
  3. Residual check (ridge + shallow gradient boosting on v3's 2016-2022 residuals, features v3
     does not use): reported only, nothing ships.
This script never changes which model is live: the repo variable CFB_MODEL_VERSION stays on v2
until the user decides.

Usage:
    uv run python scripts/gate_cfb_v3.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from sportsmodel.cfb import residual, v3, v3_data, v3_gate, v3_table  # noqa: E402
from sportsmodel.nfl.gameline import build_gameline  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"
WALK_SEASONS = tuple(range(2015, 2026))
TRAIN_SEASONS = tuple(range(2016, 2023))
EMPTY_MARKET = {"spread_line": None, "total_line": None}


def add_predictions(table: pd.DataFrame, w: v3.V3Weights, gl_v2, gl_v3) -> pd.DataFrame:
    """table + v2_/v3_ margin, total and home-win-prob columns. Both go through build_gameline with
    an empty market (model-only), so the served pred = model - bias and the win prob uses each
    version's own sigma -- exactly what generate_cfb serves."""
    out = table.copy()
    cols = {k: [] for k in ("v2_margin", "v2_total", "v2_wp", "v3_margin", "v3_total", "v3_wp")}
    for r in table.to_dict("records"):
        a = build_gameline(r["margin_v2"], r["total_v2"], EMPTY_MARKET, r["week"], gl_v2)
        m3, t3 = v3.predict_v3(r, w)
        b = build_gameline(m3, t3, EMPTY_MARKET, r["week"], gl_v3)
        for pre, row in (("v2", a), ("v3", b)):
            cols[f"{pre}_margin"].append(row["pred_margin"])
            cols[f"{pre}_total"].append(row["pred_total"])
            cols[f"{pre}_wp"].append(row["home_win_prob"])
    for k, v in cols.items():
        out[k] = v
    return out


def evaluate_gate(scored: pd.DataFrame, expected=v3_gate.V2_EXPECTED, tol=v3_gate.BASELINE_TOL) -> dict:
    """Baseline reproduction (raises BaselineMismatch) then the v3-vs-v2 comparison."""
    ev = v3_gate.eval_set(scored)
    v2c = v3_gate.metrics(ev, "v2_margin", "v2_total", "v2_wp")
    baseline = v3_gate.check_baseline(v2c, expected, tol)
    v3c = v3_gate.metrics(ev, "v3_margin", "v3_total", "v3_wp")
    v2s = v3_gate.by_season(ev, "v2_margin", "v2_total", "v2_wp")
    v3s = v3_gate.by_season(ev, "v3_margin", "v3_total", "v3_wp")
    return {"holdout_seasons": list(v3_gate.HOLDOUT_SEASONS), "n_games": int(len(ev)),
            "baseline": {"expected": expected, "measured": {k: v2c[k] for k in expected},
                         "reproduced": True},
            "v2": {"combined": v2c, "by_season": v2s}, "v3": {"combined": v3c, "by_season": v3s},
            **v3_gate.verdict(v2c, v3c, v2s, v3s)}


def residual_summary(scored: pd.DataFrame, w: v3.V3Weights, assets: Path) -> dict:
    """Report-only residual check on v3's residuals (train 2016-2022, test = the gate eval set)."""
    sched = v3_data.require_asset("schedules.parquet", assets)
    adv = v3_data.require_asset("advanced_games.parquet", assets)
    means = residual.prior_game_means(adv, sched)
    feats = residual.residual_features(
        scored, means, v3_data.read_asset("cfbd_games.parquet", assets), v3_data.load_priors_rows(assets),
        v3_data.read_asset("prior_ratings.parquet", assets), w.margin.coefs, w.total.coefs, w.points_map.coefs)
    ev = v3_gate.eval_set(scored)
    is_eval = scored["game_pk"].isin(ev["game_pk"]) & scored["season"].isin(v3_gate.HOLDOUT_SEASONS)
    is_train = scored["season"].isin(TRAIN_SEASONS) & scored["actual_margin"].notna()
    keep = is_eval | is_train
    sub = scored[keep]
    return residual.run_residual_check(
        feats[keep], (sub["actual_margin"] - sub["v3_margin"]).to_numpy(),
        (sub["actual_total"] - sub["v3_total"]).to_numpy(), sub["season"].to_numpy(),
        TRAIN_SEASONS, v3_gate.HOLDOUT_SEASONS)


def _fmt(x, nd=3, pct=False) -> str:
    return "n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else (
        f"{100 * x:.1f}%" if pct else f"{x:.{nd}f}")


def render_report(res: dict) -> str:
    L = [f"# cfb-ratings-v3 gate ({res['generated']})", "",
         f"Held-out {res['holdout_seasons']}, {res['n_games']} FBS-vs-FBS games with a closing spread.", "",
         "**Live model: v2 (unchanged). Switching to v3 is the user's decision "
         "(repo variable CFB_MODEL_VERSION).**", "",
         f"**Verdict: {'PASS - v3 meets every criterion' if res['ship'] else 'FAIL - v3 does not ship'}**", "",
         "## v2 baseline reproduction", "",
         "| metric | published | measured |", "|---|---|---|"]
    for k, e in res["baseline"]["expected"].items():
        L.append(f"| {k} | {e} | {_fmt(res['baseline']['measured'][k], 4)} |")
    L += ["", "## Criteria (combined)", "", "| criterion | v2 | v3 | pass |", "|---|---|---|---|"]
    for k, c in res["criteria"].items():
        pct = k in ("ats", "ou")
        L.append(f"| {c['label']} | {_fmt(c['v2'], 4, pct)} | {_fmt(c['v3'], 4, pct)} | {'yes' if c['pass'] else 'NO'} |")
    L += ["", "## Per season", "", "| season | model | n | margin MAE | total MAE | ATS | O/U | ML log-loss | ECE |",
          "|---|---|---|---|---|---|---|---|---|"]
    for s in sorted(res["v2"]["by_season"]):
        for name in ("v2", "v3"):
            m = res[name]["by_season"][s]
            L.append(f"| {s} | {name} | {m['n']} | {_fmt(m['margin_mae'])} | {_fmt(m['total_mae'])} | "
                     f"{_fmt(m['ats'], pct=True)} | {_fmt(m['ou'], pct=True)} | {_fmt(m['ml_logloss'], 4)} | "
                     f"{_fmt(m['ml_ece'], 4)} |")
    L += ["", "Seasons where v3 is worse than v2: "
          + (", ".join(f"{f['season']} {f['criterion']}" for f in res["season_flags"]) or "none"), ""]
    rc = res.get("residual_check")
    if rc:
        L += ["## Residual check (report only)", ""]
        for tgt in ("margin", "total"):
            r = rc[tgt]
            L.append(f"- {tgt}: ridge R2_oos {_fmt(r['ridge']['r2_oos'], 4)}, MAE change "
                     f"{_fmt(r['ridge']['mae_change'], 4)}; boosting R2_oos {_fmt(r['hgb']['r2_oos'], 4)}, MAE change "
                     f"{_fmt(r['hgb']['mae_change'], 4)}; top ridge features "
                     f"{[f['feature'] for f in r['ridge']['top_features'][:3]]}; v4 candidates "
                     f"{[c['feature'] for c in r['v4_candidates']] or 'none'}")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-out", default=str(ASSETS / "v3_gate.json"))
    ap.add_argument("--report-out", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    inp = v3_data.load_v3_inputs(ASSETS)
    table = v3_table.build_table(v3_data.load_merged_schedule(ASSETS), inp, seasons=WALK_SEASONS)
    w = v3.load_v3_weights(ASSETS / "v3_weights.json")
    scored = add_predictions(table, w, v3.load_gameline_config(ASSETS / "gameline.json"),
                             v3.load_gameline_config(ASSETS / "gameline_v3.json"))
    try:
        res = evaluate_gate(scored)
    except v3_gate.BaselineMismatch as e:
        print(f"STOP: {e}\nNothing written. Reconcile the harness with scripts/compare_cfb_live_fix.py "
              "before comparing v3.", file=sys.stderr)
        return 2
    res["generated"] = date.today().isoformat()
    res["residual_check"] = residual_summary(scored, w, ASSETS)
    res["fit"] = w.meta
    res["live_model"] = "v2 (unchanged); CFB_MODEL_VERSION stays v2 until the user approves this gate"
    res["runtime_s"] = time.time() - t0
    out = Path(args.json_out)
    rep = Path(args.report_out) if args.report_out else (
        ROOT / "docs" / "superpowers" / "reports" / f"{res['generated']}-cfb-v3-gate.md")
    out.write_text(json.dumps(v3_gate.clean(res), indent=2, allow_nan=False) + "\n")
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(render_report(res))
    print(render_report(res))
    print(f"wrote {out} and {rep} ({res['runtime_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_v3_gate.py tests/cfb/test_residual.py tests/scripts/test_gate_cfb_v3.py -q`
Expected: PASS (8 + 4 + 4).

- [ ] **Step 7: [LOCAL RUN] Run the gate. v2's baseline is reproduced FIRST; if it is not, the run stops.**

Needs Task 4 assets + Task 9 weights; no network (about 20 seconds).

```bash
uv run python scripts/gate_cfb_v3.py
```
- **Baseline first.** The script scores v2 through this harness on held-out 2023-25 (2,169 FBS-vs-FBS games with a closing spread) and requires margin MAE 12.61, total MAE 13.10, ATS 49.0% within rounding (tolerances 0.006 / 0.006 / 0.0006). On a mismatch it prints `STOP: v2 baseline NOT reproduced ...` to stderr, writes **nothing** and exits 2. Do not compare v3 then: reconcile first (run `uv run python scripts/compare_cfb_live_fix.py --seasons 2023 2024 2025`, whose "fixed" row is where the published numbers come from, and diff its per-game rows against the table; check that the asset parquets were not re-pulled in a way that changed `schedules.parquet` / `lines.parquet` / `priors.parquet`), fix the harness, re-run.
- On success it prints the report (baseline table, the six criteria with pass/fail, per-season table, flagged seasons, residual-check summary) and writes `assets/cfb/v3_gate.json` and `docs/superpowers/reports/<today>-cfb-v3-gate.md`. A FAIL verdict is a valid, committable outcome: it means v3 does not ship. **Do not change `CFB_MODEL_VERSION` either way.**

```bash
git add src/sportsmodel/cfb/v3_gate.py src/sportsmodel/cfb/residual.py scripts/gate_cfb_v3.py tests/cfb/test_v3_gate.py tests/cfb/test_residual.py tests/scripts/test_gate_cfb_v3.py assets/cfb/v3_gate.json docs/superpowers/reports/*-cfb-v3-gate.md
git commit -m "feat(cfb): v3 ship gate (v2 baseline reproduction, criteria, residual check) + v3_gate.json"
```

---

### Task 11: Serve v3 behind `CFB_MODEL_VERSION` (default v2) with game-day weather

**Files:**
- Create: `src/sportsmodel/cfb/live_weather.py`
- Modify: `scripts/generate_cfb.py`
- Test: `tests/cfb/test_live_weather.py`, `tests/cfb/test_generate_cfb.py`

**Interfaces:**
- Consumes: `v3.{load_v3_weights, predict_v3, load_gameline_config, V3Weights, VERSION}`, `v3_table.{live_frame, build_table, V3Inputs}`, `v3_data.{load_v3_inputs, read_asset}`, `cfbd.CfbdClient`, `cfbd_games.parse_weather_games`.
- Produces:
  - `live_weather.fetch_live_weather(client, season: int, espn_season_type: int) -> DataFrame` (one `/games/weather?year&seasonType` call; **never raises**: no client / error / empty -> empty frame + `::warning::`), `merge_weather(history, live) -> DataFrame | None` (live rows replace the same game_id).
  - `generate_cfb.selected_model(env=None) -> "v2" | "v3"` (unset/blank -> `"v2"`; anything else raises `ValueError`); `GAME_MODEL_VERSION = "cfb-ratings-v2"`, `GAME_MODEL_VERSION_V3 = "cfb-ratings-v3"`; `build_game_row(game, ctx, gl_cfg, model_version=GAME_MODEL_VERSION)`; `build_game_rows_v3(games, feature_rows, weights, gl_cfg, week) -> list[dict]`; `v3_feature_rows(sched, season, week, games, inp) -> {game_pk: row}`; `load_gameline(path=None)`.
  - Default behaviour (variable unset) is byte-identical to today's v2.

- [ ] **Step 1: Write the failing tests**

Create `tests/cfb/test_live_weather.py`:

```python
import pandas as pd

from sportsmodel.cfb import live_weather as lw
from sportsmodel.cfb.teams import cfbd_to_espn


def _row(gid, temp, wind=8.0):
    return {"id": gid, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-10T16:00:00.000Z",
            "gameIndoors": False, "homeTeam": "Alabama", "awayTeam": "Georgia", "venueId": 3657,
            "temperature": temp, "windSpeed": wind, "precipitation": 0.0}


class Stub:
    def __init__(self, payload=None, boom=False):
        self.payload, self.boom, self.seen = payload, boom, []

    def get(self, path, params=None):
        self.seen.append((path, params))
        if self.boom:
            raise RuntimeError("503")
        return self.payload


def test_fetch_makes_one_call_for_the_season_and_maps_espn_season_type():
    c = Stub([_row(1, 55.0)])
    df = lw.fetch_live_weather(c, 2026, 2)
    assert c.seen == [("/games/weather", {"year": 2026, "seasonType": "regular"})]
    assert len(df) == 1 and df.iloc[0]["home_team"] == cfbd_to_espn("Alabama")
    lw.fetch_live_weather(c, 2026, 3)
    assert c.seen[-1][1]["seasonType"] == "postseason"


def test_fetch_never_raises_and_warns(capsys):
    for client in (Stub(boom=True), None, Stub([])):
        df = lw.fetch_live_weather(client, 2026, 2)
        assert df.empty
    assert capsys.readouterr().out.count("::warning::cfb-v3") == 3


def test_live_rows_replace_history_rows_of_the_same_game():
    hist = lw.parse_weather_games([_row(1, 40.0), _row(2, 70.0)])
    live = lw.parse_weather_games([_row(2, 33.0)])
    out = lw.merge_weather(hist, live).set_index("game_id")
    assert out.loc[1, "temperature"] == 40.0 and out.loc[2, "temperature"] == 33.0 and len(out) == 2
    assert lw.merge_weather(hist, lw.parse_weather_games([])) is hist
    assert lw.merge_weather(None, live) is live
    assert lw.merge_weather(None, None) is None
```

Append the v3 serving tests to `tests/cfb/test_generate_cfb.py` (existing tests stay untouched):

```diff
--- a/tests/cfb/test_generate_cfb.py
+++ b/tests/cfb/test_generate_cfb.py
@@ -399,3 +399,117 @@
     # no priors asset -> no prior is served, so no weights are required
     assert gc.load_live_prior_weights(tmp_path / "missing.json",
                                       tmp_path / "no_priors.parquet") == PriorWeights()
+
+
+# ---------------------------------------------------------------------------
+# cfb-ratings-v3 serving behind CFB_MODEL_VERSION
+# ---------------------------------------------------------------------------
+import numpy as np
+import pandas as pd
+import pytest
+
+from sportsmodel.cfb import v3, v3_table
+from tests.cfb.test_v3_table import world, inputs as v3_inputs
+
+
+def test_selected_model_defaults_to_v2_and_rejects_typos():
+    assert gc.selected_model({}) == "v2"
+    assert gc.selected_model({"CFB_MODEL_VERSION": ""}) == "v2"
+    assert gc.selected_model({"CFB_MODEL_VERSION": "  V3 "}) == "v3"
+    assert gc.selected_model({"CFB_MODEL_VERSION": "v2"}) == "v2"
+    with pytest.raises(ValueError, match="CFB_MODEL_VERSION"):
+        gc.selected_model({"CFB_MODEL_VERSION": "v4"})
+
+
+def test_build_game_row_model_version_defaults_to_v2_and_can_be_v3():
+    game = {"game_pk": 1, "game_date": "2026-10-10", "home_name": "H", "away_name": "A"}
+    ctx = {"model_margin": 3.0, "model_total": 55.0, "week": 6}
+    assert gc.build_game_row(game, ctx, GL_CFG)["model_version"] == "cfb-ratings-v2"
+    assert gc.build_game_row(game, ctx, GL_CFG, gc.GAME_MODEL_VERSION_V3)["model_version"] == "cfb-ratings-v3"
+    assert gc.GAME_MODEL_VERSION_V3 == "cfb-ratings-v3"
+
+
+def _weights():
+    return v3.V3Weights(v3.LinearBlend(20.0, {"success": 10.0}),
+                        v3.LinearBlend(1.0, {"margin_v2": 0.5, "margin_eff": 0.5}),
+                        v3.LinearBlend(2.0, {"total_v2": 1.0}))
+
+
+def test_build_game_rows_v3_uses_predict_v3_skips_fcs_and_unfeatured_games():
+    games = [{"game_pk": 1, "home_team": "96", "away_team": "61", "home_name": "K", "away_name": "G",
+              "game_date": "2026-10-10"},
+             {"game_pk": 2, "home_team": "158", "away_team": "FCS", "home_name": "N", "away_name": "U",
+              "game_date": "2026-10-10"},
+             {"game_pk": 3, "home_team": "12", "away_team": "2", "home_name": "A", "away_name": "B",
+              "game_date": "2026-10-10"}]
+    feats = {1: {"margin_v2": 6.0, "total_v2": 50.0, "h_success": 0.1, "a_success": -0.1},
+             2: {"margin_v2": 1.0, "total_v2": 50.0}}                      # game 3 has no feature row
+    rows = gc.build_game_rows_v3(games, feats, _weights(), GL_CFG, week=6)
+    assert [r["game_pk"] for r in rows] == [1]
+    r = rows[0]
+    assert r["model_version"] == "cfb-ratings-v3" and r["sport"] == "cfb"
+    assert r["pred_margin"] == pytest.approx(1.0 + 0.5 * 6.0 + 0.5 * 2.0)  # eff margin = 21 - 19 = 2
+    assert r["pred_total"] == pytest.approx(2.0 + 50.0)
+    assert r["total_dist"]["kind"] == "pmf" and r["margin_dist"]["kind"] == "margin"
+
+
+def test_live_v3_feature_row_equals_the_backtest_row_for_the_same_game():
+    sched, eg = world()
+    inp = v3_inputs(eg)
+    hist = v3_table.build_table(sched, inp)
+    wk = sched[(sched.season == 2023) & (sched.week == 4)]
+    slate = [{"game_pk": int(r.game_pk), "home_team": r.home_team, "away_team": r.away_team,
+              "neutral_site": bool(r.neutral_site), "start_date": r.start_date}
+             for r in wk.itertuples() if "FCS" not in (r.home_team, r.away_team)]
+    live = gc.v3_feature_rows(sched, 2023, 4, slate, inp)
+    assert set(live) == {g["game_pk"] for g in slate}
+    for pk, row in live.items():
+        ref = hist[hist.game_pk == pk].iloc[0]
+        for c in v3_table.TABLE_COLUMNS:
+            if c in ("actual_margin", "actual_total", "market_spread", "market_total"):
+                continue                                                  # unscored live game: no result / line
+            assert row[c] == pytest.approx(ref[c]) if isinstance(ref[c], float) else row[c] == ref[c], c
+        assert np.isnan(row["actual_margin"])
+
+
+def _patch_main(monkeypatch, tmp_path, version, calls):
+    sched, eg = world()
+    games = [{"game_pk": 9001, "home_team": "1", "away_team": "2", "home_name": "One", "away_name": "Two",
+              "commence_time": "2023-10-07T17:00Z", "start_date": "2023-10-07T17:00Z", "neutral_site": False,
+              "week": 4, "season": 2023}]
+    monkeypatch.setattr(gc.espn, "fetch_current_week", lambda: {"season": 2023, "week": 4, "season_type": 2})
+    monkeypatch.setattr(gc.espn, "fetch_schedule", lambda *a, **k: games)
+    monkeypatch.setattr(gc, "_load_committed", lambda name: sched)
+    monkeypatch.setattr(gc.config, "DATABASE_URL", "postgres://x")
+    monkeypatch.setattr(gc, "upsert_game_predictions", lambda rows: calls.setdefault("rows", rows))
+    monkeypatch.setattr(gc.live_weather, "fetch_live_weather",
+                        lambda client, season, st: calls.setdefault("weather", (season, st)) and pd.DataFrame())
+    monkeypatch.setattr(gc.v3_data, "load_v3_inputs", lambda assets, weather=None: v3_inputs(eg))
+    monkeypatch.setattr(gc.v3, "load_v3_weights", lambda path: _weights())
+    monkeypatch.setattr(gc, "load_gameline", lambda path=None: GL_CFG)
+    monkeypatch.setattr(gc, "_cfbd_client_or_none", lambda: None)
+    if version:
+        monkeypatch.setenv("CFB_MODEL_VERSION", version)
+    else:
+        monkeypatch.delenv("CFB_MODEL_VERSION", raising=False)
+
+
+def test_main_v3_pulls_weather_and_upserts_v3_rows(monkeypatch, tmp_path, capsys):
+    calls = {}
+    _patch_main(monkeypatch, tmp_path, "v3", calls)
+    gc.main()
+    assert calls["weather"] == (2023, 2)
+    assert [r["model_version"] for r in calls["rows"]] == ["cfb-ratings-v3"]
+    assert "(v3)" in capsys.readouterr().out
+
+
+def test_main_default_is_v2_and_never_touches_v3(monkeypatch, tmp_path):
+    calls = {}
+    _patch_main(monkeypatch, tmp_path, None, calls)
+    monkeypatch.setattr(gc, "load_rating", lambda: (ELO_CFG, BLEND_CFG))
+    monkeypatch.setattr(gc, "load_live_prior_weights", lambda: PriorWeights())
+    monkeypatch.setattr(gc, "load_priors_for_season", lambda season, w: {})
+    monkeypatch.setattr(gc.v3_data, "load_v3_inputs", lambda *a, **k: pytest.fail("v2 must not build v3 inputs"))
+    gc.main()
+    assert "weather" not in calls
+    assert [r["model_version"] for r in calls["rows"]] == ["cfb-ratings-v2"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/cfb/test_live_weather.py tests/cfb/test_generate_cfb.py -q`
Expected: FAIL (`ImportError: cannot import name 'live_weather'`; the appended tests fail with `AttributeError: module 'generate_cfb' has no attribute 'selected_model'`).

- [ ] **Step 3: Implement the live weather pull**

Create `src/sportsmodel/cfb/live_weather.py`:

```python
"""Game-day weather for the live v3 slate (one CFBD call per run).

`fetch_live_weather` pulls `/games/weather?year=Y&seasonType=T` (historical + forecast rows for the
whole season in ONE call -- no week alignment to get wrong) through the shared CfbdClient and
returns the parsed frame. It NEVER raises: no key, a CFBD error or an empty answer returns an empty
frame and prints a ::warning::, and the v3 model then applies no weather adjustment to those games
(0, never fabricated).
"""
from __future__ import annotations

import pandas as pd

from .cfbd_games import WEATHER_COLUMNS, parse_weather_games

ESPN_SEASON_TYPE = {2: "regular", 3: "postseason"}


def _empty() -> pd.DataFrame:
    return parse_weather_games([])


def fetch_live_weather(client, season: int, espn_season_type: int) -> pd.DataFrame:
    """Weather rows for the season of the live slate; empty (never an exception) on any failure."""
    try:
        if client is None:
            raise RuntimeError("no CFBD client (CFBD_API_KEY not set)")
        st = ESPN_SEASON_TYPE.get(int(espn_season_type), "regular")
        df = parse_weather_games(client.get("/games/weather", {"year": int(season), "seasonType": st}))
        if df.empty:
            print(f"::warning::cfb-v3: CFBD returned no weather rows for {season} {st}; "
                  "games get no weather adjustment", flush=True)
        return df
    except Exception as e:  # noqa: BLE001 -- weather must never take the prediction job down
        print(f"::warning::cfb-v3: weather pull failed ({type(e).__name__}); "
              "games get no weather adjustment", flush=True)
        return _empty()


def merge_weather(history: pd.DataFrame | None, live: pd.DataFrame | None) -> pd.DataFrame | None:
    """Committed history + the day's live rows; a live row replaces the same game_id's history row."""
    if live is None or live.empty:
        return history
    if history is None or history.empty:
        return live
    keep = history[~history["game_id"].isin(live["game_id"])]
    return pd.concat([keep[WEATHER_COLUMNS], live[WEATHER_COLUMNS]], ignore_index=True)
```

- [ ] **Step 4: Implement the serving switch**

Apply this diff to `scripts/generate_cfb.py`:

```diff
--- a/scripts/generate_cfb.py
+++ b/scripts/generate_cfb.py
@@ -30,12 +30,22 @@
 against -- only the CFB-specific team-id normalization (cfb.teams, via
 cfb.espn) and the FCS-pseudo-team skip below are CFB-specific.
 
+Model selection: the repo variable CFB_MODEL_VERSION (workflows pass `vars.CFB_MODEL_VERSION ||
+'v2'`) picks the served model -- `v2` (default, the ratings model above) or `v3`
+(cfb-ratings-v3: v2 + opponent-adjusted efficiency ratings + game-day context, weights in
+assets/cfb/v3_weights.json, gameline_v3.json). v3 first pulls the day's weather (one CFBD call;
+never fatal -- missing weather = no weather adjustment). Switching back is one variable change;
+rows upsert on (game_pk, model_version), so both versions' rows can coexist and the serving views
+show the latest-generated one.
+
 Usage:
-    uv run python scripts/generate_cfb.py
+    uv run python scripts/generate_cfb.py                       # v2
+    CFB_MODEL_VERSION=v3 uv run python scripts/generate_cfb.py  # v3
 """
 from __future__ import annotations
 
 import json
+import os
 import sys
 from datetime import datetime, timedelta
 from pathlib import Path
@@ -45,7 +55,8 @@
 import pandas as pd
 
 from sportsmodel import config
-from sportsmodel.cfb import espn
+from sportsmodel.cfb import espn, live_weather, v3, v3_data, v3_table
+from sportsmodel.cfb.cfbd import CfbdClient
 from sportsmodel.cfb.priors import (
     DEFAULT_HALF_LIFE_GAMES,
     DecayConfig,
@@ -66,6 +77,8 @@
 # v2 (fix/cfb-live-ratings): walk-forward season-to-date state + leak-free prior.
 # The upsert key is (game_pk, model_version), so v1 rows stay alongside.
 GAME_MODEL_VERSION = "cfb-ratings-v2"
+GAME_MODEL_VERSION_V3 = v3.VERSION            # "cfb-ratings-v3"
+MODEL_CHOICES = ("v2", "v3")
 
 _ASSETS = Path(__file__).resolve().parents[1] / "assets" / "cfb"
 
@@ -81,15 +94,22 @@
             BlendConfig(w_sos=j["w_sos"], srs_min_games=j["srs_min_games"]))
 
 
-def load_gameline() -> GameLineConfig:
-    j = json.loads((_ASSETS / "gameline.json").read_text())
-    return GameLineConfig(sigma_margin=j["sigma_margin"], sigma_total=j["sigma_total"],
-                          offset=j["offset"], total_max=j["total_max"],
-                          w_margin=ShrinkParams(**j["w_margin"]),
-                          w_total=ShrinkParams(**j["w_total"]),
-                          bias_margin=j.get("bias_margin", 0.0), bias_total=j.get("bias_total", 0.0))
+def load_gameline(path: Path | None = None) -> GameLineConfig:
+    """GameLineConfig from gameline.json (v2, default) or another gameline-shaped file."""
+    return v3.load_gameline_config(path or (_ASSETS / "gameline.json"))
 
 
+def selected_model(env=None) -> str:
+    """The served model from CFB_MODEL_VERSION: 'v2' (default when unset/blank) or 'v3'.
+    Anything else raises -- a typo in the repo variable must fail the job loudly, not silently
+    serve a model nobody chose."""
+    raw = (os.environ if env is None else env).get("CFB_MODEL_VERSION")
+    version = (raw or "v2").strip().lower() or "v2"
+    if version not in MODEL_CHOICES:
+        raise ValueError(f"CFB_MODEL_VERSION must be one of {MODEL_CHOICES}, got {raw!r}")
+    return version
+
+
 def load_priors_for_season(season: int, weights: PriorWeights,
                            path: Path | None = None) -> dict[str, float]:
     """{team_espn_id: R_pre} for `season` from `assets/cfb/priors.parquet`,
@@ -170,7 +190,8 @@
     return (dt - timedelta(hours=8)).date().isoformat()
 
 
-def build_game_row(game: dict, ctx: dict, gl_cfg: GameLineConfig) -> dict:
+def build_game_row(game: dict, ctx: dict, gl_cfg: GameLineConfig,
+                   model_version: str = GAME_MODEL_VERSION) -> dict:
     """Pure: model margin/total -> a `game_predictions`-shaped row.
 
     `ctx` = {"model_margin", "model_total", "week"}. No market line is ever
@@ -187,7 +208,7 @@
     return {
         **row,
         "sport": "cfb",
-        "model_version": GAME_MODEL_VERSION,
+        "model_version": model_version,
         "game_pk": game["game_pk"],
         "game_date": game["game_date"],
         "commence_time": game.get("commence_time"),
@@ -263,9 +284,42 @@
                                                        elo_cfg, blend_cfg)
         ctx = {"model_margin": model_margin, "model_total": model_total, "week": week}
         rows.append(build_game_row(g, ctx, gl_cfg))
+    return rows
+
+
+def build_game_rows_v3(games: list[dict], feature_rows: dict, weights: v3.V3Weights,
+                       gl_cfg: GameLineConfig, week: int) -> list[dict]:
+    """Pure: the slate's `games` + each game's v3 feature row (`v3_table` row dict keyed by game_pk)
+    -> `game_predictions`-shaped rows tagged cfb-ratings-v3. FCS games are skipped exactly as in v2;
+    a game with no feature row is skipped (never guessed)."""
+    rows = []
+    for g in games:
+        if g["home_team"] == FCS or g["away_team"] == FCS:
+            continue
+        feats = feature_rows.get(int(g["game_pk"]))
+        if feats is None:
+            continue
+        margin, total = v3.predict_v3(feats, weights)
+        ctx = {"model_margin": margin, "model_total": total, "week": week}
+        rows.append(build_game_row(g, ctx, gl_cfg, GAME_MODEL_VERSION_V3))
     return rows
 
 
+def v3_feature_rows(sched: pd.DataFrame, season: int, week: int, games: list[dict],
+                    inp: v3_table.V3Inputs) -> dict:
+    """{game_pk: v3 feature row} for the live slate, from the SAME walk-forward the fit and the gate
+    use (v3_table.build_table on the live frame: scored REG games before (season, week) + the slate
+    appended unscored), so a served row equals the backtest row the game would have had."""
+    frame = v3_table.live_frame(sched, season, week, games)
+    table = v3_table.build_table(frame, inp, seasons={season}, include_unscored=True)
+    return {int(r["game_pk"]): r for r in table[table["week"] == week].to_dict("records")}
+
+
+def _cfbd_client_or_none() -> CfbdClient | None:
+    key = os.environ.get("CFBD_API_KEY")
+    return CfbdClient(key) if key else None
+
+
 def _season_to_date_ratings(sched: pd.DataFrame, season: int, week: int,
                             elo_cfg: EloConfig, blend_cfg: BlendConfig,
                             games: list[dict]) -> dict:
@@ -301,29 +355,39 @@
     cur = espn.fetch_current_week()
     season, week, season_type = int(cur["season"]), int(cur["week"]), int(cur["season_type"])
 
-    elo_cfg, blend_cfg = load_rating()
-    gl_cfg = load_gameline()
-    prior_weights = load_live_prior_weights()
-    decay_cfg = load_decay_config()
-
+    version = selected_model()
     sched = _load_committed("schedules.parquet")
     espn_games = espn.fetch_schedule(season, week, season_type=season_type)
     games_for_rows = [{**g, "game_date": _game_date_from_commence(g["commence_time"])}
                       for g in espn_games]
-
     s_week = state_week(sched, season, week, season_type)
-    ratings = _season_to_date_ratings(sched, season, s_week, elo_cfg, blend_cfg, games_for_rows)
-    ratings["r_pre"] = load_priors_for_season(season, prior_weights)
-    ratings["decay_cfg"] = decay_cfg
 
-    game_rows_raw = build_game_rows(games_for_rows, ratings, s_week, gl_cfg)
+    if version == "v3":
+        # game-day weather first (one call, never fatal), then the v3 walk-forward feature rows
+        live_wx = live_weather.fetch_live_weather(_cfbd_client_or_none(), season, season_type)
+        inp = v3_data.load_v3_inputs(
+            _ASSETS, weather=live_weather.merge_weather(v3_data.read_asset("weather_games.parquet", _ASSETS),
+                                                         live_wx))
+        feats = v3_feature_rows(sched, season, s_week, games_for_rows, inp)
+        game_rows_raw = build_game_rows_v3(games_for_rows, feats,
+                                           v3.load_v3_weights(_ASSETS / "v3_weights.json"),
+                                           load_gameline(_ASSETS / "gameline_v3.json"), s_week)
+    else:
+        elo_cfg, blend_cfg = load_rating()
+        gl_cfg = load_gameline()
+        prior_weights = load_live_prior_weights()
+        decay_cfg = load_decay_config()
+        ratings = _season_to_date_ratings(sched, season, s_week, elo_cfg, blend_cfg, games_for_rows)
+        ratings["r_pre"] = load_priors_for_season(season, prior_weights)
+        ratings["decay_cfg"] = decay_cfg
+        game_rows_raw = build_game_rows(games_for_rows, ratings, s_week, gl_cfg)
     game_rows = [{**row, "margin_dist": json.dumps(row["margin_dist"]),
                  "total_dist": json.dumps(row["total_dist"])} for row in game_rows_raw]
 
     if config.DATABASE_URL:
         upsert_game_predictions(game_rows)
 
-    print(f"predicted {len(game_rows)} games")
+    print(f"predicted {len(game_rows)} games ({version})")
 
 
 if __name__ == "__main__":
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/cfb/test_live_weather.py tests/cfb/test_generate_cfb.py tests/cfb/test_backtest_cfb_priors.py tests/scripts/test_compare_cfb_live_fix.py -q`
Expected: PASS (3 new live-weather tests + 25 in `test_generate_cfb.py`, of which 6 are new: the unset-variable `main()` test proves the v2 path never builds v3 inputs and still upserts `cfb-ratings-v2` rows; the live v3 feature row equals the backtest row for the same game).

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/cfb/live_weather.py scripts/generate_cfb.py tests/cfb/test_live_weather.py tests/cfb/test_generate_cfb.py
git commit -m "feat(cfb): serve cfb-ratings-v3 behind CFB_MODEL_VERSION (default v2) + game-day weather pull"
```

---

### Task 12: Workflows (weekly refresh, model switch), efficiency snapshot, final verification

**Files:**
- Create: `scripts/build_cfb_efficiency.py`
- Modify: `.github/workflows/build-cfb-advanced.yml`, `.github/workflows/generate-cfb.yml`, `.github/workflows/injury-watch.yml` (the `cfb` job also runs `generate_cfb.py`: a regenerate under the wrong model would overwrite the served rows)
- Test: `tests/scripts/test_build_cfb_efficiency.py`, `tests/test_workflows_cfb_advanced.py`

**Interfaces:**
- Consumes: `efficiency.{season_prior, state_before, load_eff_config, METRICS}`, `v3_data.*`, the scripts of Tasks 2-3.
- Produces:
  - `build_cfb_efficiency.ratings_snapshot(eff_games, season, priors_rows, talent, cfg) -> DataFrame[COLUMNS]` (season, week, team, games, `<metric>_off`, `<metric>_def`; one block per week entering + the live week) and `assets/cfb/efficiency_ratings.parquet`.
  - `build-cfb-advanced.yml` weekly job: advanced pull -> game-data pull -> efficiency recompute -> one commit of all assets; cron/blank dispatch pulls the current AND previous season (January bowls).
  - `generate-cfb.yml` and `injury-watch.yml` (cfb) pass `CFB_MODEL_VERSION: ${{ vars.CFB_MODEL_VERSION || 'v2' }}` and `CFBD_API_KEY`.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_cfb_efficiency.py`:

```python
import importlib.util
import pathlib

import numpy as np
import pandas as pd
import pytest

from sportsmodel.cfb import efficiency as eff
from tests.cfb.test_eff_fit import persistent_league

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_efficiency.py"
_s = importlib.util.spec_from_file_location("build_cfb_efficiency", _p)
bce = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bce)


def test_snapshot_has_one_row_per_team_week_with_entering_state_and_a_live_week():
    g = persistent_league()
    snap = bce.ratings_snapshot(g, 2023, [], None, eff.EffConfig())
    assert list(snap.columns) == bce.COLUMNS
    weeks = sorted(snap["week"].unique())
    assert weeks == list(range(1, 8))                          # weeks 1-6 played + the live week 7
    wk3 = snap[snap.week == 3].set_index("team")
    assert (wk3["games"] == 2).all()                           # entering week 3 = two games played
    wk7 = snap[snap.week == 7].set_index("team")
    assert (wk7["games"] == 6).all()
    assert not snap[["ppa_off", "ppa_def"]].isna().any().any()
    # entering-week state equals the model's state_before for the same week
    st = eff.state_before(g, 2023, 3, eff.season_prior(g, 2023, [], None, eff.EffConfig()), eff.EffConfig())
    assert wk3.loc["4", "ppa_off"] == pytest.approx(st.ratings["ppa"].off["4"])
    assert wk3.loc["4", "ppa_def"] == pytest.approx(st.ratings["ppa"].deff["4"])


def test_snapshot_of_a_season_without_games_is_empty():
    g = persistent_league(seasons=(2022,))
    assert bce.ratings_snapshot(g, 2030, [], None, eff.EffConfig()).empty
```

Extend the workflow structure tests:

```diff
--- a/tests/test_workflows_cfb_advanced.py
+++ b/tests/test_workflows_cfb_advanced.py
@@ -3,8 +3,10 @@
 concurrency group serializes runs without cancelling an in-flight pull."""
 from __future__ import annotations
 
-from tests.test_workflows_props_ml import WF, load
+import pytest
 
+from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of
+
 NAME = "build-cfb-advanced.yml"
 
 
@@ -19,3 +21,45 @@
     wf = load(NAME)
     assert wf["concurrency"]["group"] == "build-cfb-advanced"
     assert str(wf["concurrency"]["cancel-in-progress"]).lower() == "false"
+
+
+# ---- cfb-ratings-v3 wiring -----------------------------------------------------------------
+
+GAME_DATA_FILES = ("advanced_games", "cfbd_games", "havoc_games", "drive_games", "weather_games",
+                   "talent", "prior_ratings", "venues", "efficiency_ratings")
+
+
+def test_weekly_job_pulls_game_data_and_recomputes_ratings_before_commit():
+    steps = steps_of(load(NAME)["jobs"]["build"])
+    adv = step_index(steps, runs("build_cfb_advanced.py"))
+    data = step_index(steps, runs("build_cfb_game_data.py"))
+    eff = step_index(steps, runs("build_cfb_efficiency.py"))
+    commit = step_index(steps, runs("git commit"))
+    assert adv < data < eff < commit
+    assert steps[data]["env"]["CFBD_API_KEY"] == "${{ secrets.CFBD_API_KEY }}"
+    assert "CFBD_API_KEY" not in (steps[eff].get("env") or {})          # local compute: no key needed
+    add = steps[commit]["run"]
+    for f in GAME_DATA_FILES:
+        assert f"assets/cfb/{f}.parquet" in add, f
+
+
+def test_cron_refreshes_current_and_previous_season_for_january_bowls():
+    text = (WF / NAME).read_text()
+    assert text.count("$(date +%Y) $(( $(date +%Y) - 1 ))") == 3       # advanced, game data, efficiency
+    desc = load(NAME)["on"]["workflow_dispatch"]["inputs"]["seasons"]["description"]
+    assert "current season" in desc and "previous" in desc
+
+
+@pytest.mark.parametrize("name,job", [("generate-cfb.yml", "generate"), ("injury-watch.yml", "cfb")])
+def test_cfb_generate_steps_carry_the_model_switch_and_the_key(name, job):
+    steps = steps_of(load(name)["jobs"][job])
+    gen = steps[step_index(steps, runs("generate_cfb.py"))]
+    assert gen["env"]["CFB_MODEL_VERSION"] == "${{ vars.CFB_MODEL_VERSION || 'v2' }}"   # default stays v2
+    assert gen["env"]["CFBD_API_KEY"] == "${{ secrets.CFBD_API_KEY }}"
+
+
+def test_no_other_workflow_runs_generate_cfb_without_the_switch():
+    for path in sorted(WF.glob("*.yml")):
+        text = path.read_text()
+        if "generate_cfb.py" in text:
+            assert "CFB_MODEL_VERSION: ${{ vars.CFB_MODEL_VERSION || 'v2' }}" in text, path.name
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/scripts/test_build_cfb_efficiency.py tests/test_workflows_cfb_advanced.py -q`
Expected: FAIL (`FileNotFoundError` for `scripts/build_cfb_efficiency.py`; workflow assertions such as `expected exactly one matching step, got []`).

- [ ] **Step 3: Implement the snapshot script**

Create `scripts/build_cfb_efficiency.py`:

```python
"""Recompute the opponent-adjusted efficiency ratings -> assets/cfb/efficiency_ratings.parquet.

One row per (season, week, team): `week` is the schedule week the ratings ENTER (built only from
earlier weeks of that season + the previous-season prior), `games` the team-games used, and
`<metric>_off` / `<metric>_def` for every efficiency metric (off > 0 better offense, def > 0 better
defense, vs league average). This is the audit / site-panel snapshot of exactly the state the v3
model walks through; v3 itself recomputes it from the committed parquets at predict time, so the
two cannot drift. Local compute only (no network, no key).

Usage:
    uv run python scripts/build_cfb_efficiency.py --seasons 2026 [2025 ...]
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sportsmodel.cfb import v3_data  # noqa: E402
from sportsmodel.cfb.efficiency import (METRICS, EffConfig, load_eff_config, season_prior,  # noqa: E402
                                        state_before)

ASSETS = ROOT / "assets" / "cfb"
OUT = ASSETS / "efficiency_ratings.parquet"
COLUMNS = ["season", "week", "team", "games"] + [f"{m}_{s}" for m in METRICS for s in ("off", "def")]


def ratings_snapshot(eff_games: pd.DataFrame, season: int, priors_rows: list[dict],
                     talent: pd.DataFrame | None, cfg: EffConfig) -> pd.DataFrame:
    """Entering-week ratings for every regular-season week of `season` plus the week after the last
    played one (the live week)."""
    wk = eff_games[(eff_games["season"] == season) & eff_games["week"].notna()]["week"]
    if wk.empty:
        return pd.DataFrame(columns=COLUMNS)
    cache: dict = {}
    prior = season_prior(eff_games, season, priors_rows, talent, cfg, cache)
    rows = []
    for week in [int(w) for w in sorted(wk.unique())] + [int(wk.max()) + 1]:
        st = state_before(eff_games, season, week, prior, cfg, cache)
        teams = set(st.games)
        for r in st.ratings.values():
            teams |= set(r.off)
        for team in sorted(teams):
            row = {"season": season, "week": week, "team": team, "games": int(st.games.get(team, 0))}
            for m in METRICS:
                r = st.ratings[m]
                row[f"{m}_off"] = r.off.get(team, float("nan"))
                row[f"{m}_def"] = r.deff.get(team, float("nan"))
            rows.append(row)
    return pd.DataFrame(rows, columns=COLUMNS)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seasons", type=int, nargs="+", default=[dt.date.today().year])
    args = ap.parse_args(argv)
    eff_games = v3_data.load_eff_games(ASSETS)
    talent = v3_data.read_asset("talent.parquet", ASSETS)
    priors = v3_data.load_priors_rows(ASSETS)
    cfg = load_eff_config(ASSETS / "eff_config.json")
    frames = [ratings_snapshot(eff_games, s, priors.get(s, []), talent, cfg) for s in args.seasons]
    new = pd.concat(frames, ignore_index=True)
    got = sorted(new["season"].unique()) if len(new) else []
    existing = pd.read_parquet(OUT) if OUT.exists() else None
    if existing is not None:
        # a season with no rows yet (e.g. the new year before week 1) keeps its committed rows
        new = pd.concat([existing[~existing["season"].isin(got)], new], ignore_index=True)
    new = new.sort_values(["season", "week", "team"]).reset_index(drop=True)
    new.to_parquet(OUT)
    print(f"wrote {len(new)} rows ({len(got)} season(s) recomputed: {got}) -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Update the workflows**

```diff
--- a/.github/workflows/build-cfb-advanced.yml
+++ b/.github/workflows/build-cfb-advanced.yml
@@ -1,15 +1,18 @@
 name: build-cfb-advanced
 
-# CFBD advanced per-game team stats (PPA, success rate, explosiveness, pass/rush splits),
-# committed as assets/cfb/advanced_games.parquet to the dispatched ref. Manual dispatch pulls
-# the given seasons; the weekly cron (Mon 12:00 UTC) refreshes the current season. Always
-# --merge: only the pulled seasons are replaced, the rest of the parquet is kept.
+# CFBD per-game data behind cfb-ratings-v3, committed to the dispatched ref:
+#   advanced_games.parquet (regular + postseason PPA / success / explosiveness / splits),
+#   cfbd_games / havoc_games / drive_games / weather_games / talent / prior_ratings / venues
+#   (scripts/build_cfb_game_data.py), then efficiency_ratings.parquet (recomputed ratings snapshot).
+# Manual dispatch pulls the given seasons; the weekly cron (Mon 12:00 UTC) refreshes the current
+# season AND the previous one (January bowls belong to last season's postseason). Always merged:
+# only the pulled seasons are replaced, the rest of each parquet is kept.
 # CFBD_API_KEY is a repo secret; the key is never printed.
 on:
   workflow_dispatch:
     inputs:
       seasons:
-        description: "Space-separated seasons to pull (blank = the current season only)"
+        description: "Space-separated seasons to pull (blank = the current season, plus the previous one for January bowls)"
         required: false
         default: "2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025"
   schedule:
@@ -26,7 +29,7 @@
 jobs:
   build:
     runs-on: ubuntu-latest
-    timeout-minutes: 20
+    timeout-minutes: 30
     steps:
       - uses: actions/checkout@v5
       - name: Install uv
@@ -38,18 +41,33 @@
           CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
           SEASONS: ${{ inputs.seasons }}
         run: |
-          # cron runs have no inputs -> current season only
-          uv run python scripts/build_cfb_advanced.py --merge --seasons ${SEASONS:-$(date +%Y)}
-      - name: Commit advanced asset
+          # cron runs have no inputs -> current season + previous season
+          uv run python scripts/build_cfb_advanced.py --merge --seasons ${SEASONS:-$(date +%Y) $(( $(date +%Y) - 1 ))}
+      - name: Pull CFB game data (games meta, havoc, drives, weather, talent, ratings, venues)
+        env:
+          CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
+          SEASONS: ${{ inputs.seasons }}
         run: |
+          uv run python scripts/build_cfb_game_data.py --seasons ${SEASONS:-$(date +%Y) $(( $(date +%Y) - 1 ))}
+      - name: Recompute efficiency ratings
+        env:
+          SEASONS: ${{ inputs.seasons }}
+        run: |
+          uv run python scripts/build_cfb_efficiency.py --seasons ${SEASONS:-$(date +%Y) $(( $(date +%Y) - 1 ))}
+      - name: Commit CFB data assets
+        run: |
           git config user.name "github-actions[bot]"
           git config user.email "github-actions[bot]@users.noreply.github.com"
-          git add assets/cfb/advanced_games.parquet
+          git add assets/cfb/advanced_games.parquet assets/cfb/cfbd_games.parquet \
+            assets/cfb/havoc_games.parquet assets/cfb/drive_games.parquet \
+            assets/cfb/weather_games.parquet assets/cfb/talent.parquet \
+            assets/cfb/prior_ratings.parquet assets/cfb/venues.parquet \
+            assets/cfb/efficiency_ratings.parquet
           if git diff --staged --quiet; then
             echo "no change"
             exit 0
           fi
-          git commit -m "data(cfb): CFBD advanced game stats (build-cfb-advanced) [skip ci]"
+          git commit -m "data(cfb): CFBD advanced + game data + efficiency ratings (build-cfb-advanced) [skip ci]"
           # other data workflows push to the same branch: rebase onto theirs and retry
           for attempt in 1 2 3; do
             if git push; then
```

```diff
--- a/.github/workflows/generate-cfb.yml
+++ b/.github/workflows/generate-cfb.yml
@@ -32,4 +32,8 @@
         env:
           DATABASE_URL: ${{ secrets.DATABASE_URL }}
           SM_DATA_DIR: ${{ github.workspace }}/data
+          # repo variable: v2 (default) or v3 (cfb-ratings-v3). Switching back = one variable change.
+          CFB_MODEL_VERSION: ${{ vars.CFB_MODEL_VERSION || 'v2' }}
+          # v3 pulls the day's weather (one call, never fatal) before predicting
+          CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
         run: uv run python scripts/generate_cfb.py
```

```diff
--- a/.github/workflows/injury-watch.yml
+++ b/.github/workflows/injury-watch.yml
@@ -175,6 +175,9 @@
         env:
           DATABASE_URL: ${{ secrets.DATABASE_URL }}
           SM_DATA_DIR: ${{ github.workspace }}/data
+          # must match generate-cfb.yml: a regenerate under the wrong model would overwrite the served rows
+          CFB_MODEL_VERSION: ${{ vars.CFB_MODEL_VERSION || 'v2' }}
+          CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
         run: uv run python scripts/generate_cfb.py
 
       - name: Rebuild +EV board
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/scripts/test_build_cfb_efficiency.py tests/test_workflows_cfb_advanced.py tests/test_workflows_props_ml.py tests/test_workflows_ml_only.py -q`
Expected: PASS (2 + 7 new/extended workflow tests; every workflow still parses; injury-watch's cfb re-run chain order is unchanged).

- [ ] **Step 6: [LOCAL RUN] Build the first efficiency snapshot and commit it**

```bash
uv run python scripts/build_cfb_efficiency.py --seasons 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026
```
Expected: `wrote N rows (12 season(s) recomputed: [...]) -> assets/cfb/efficiency_ratings.parquet`.

- [ ] **Step 7: Final verification - full suite, and the live model is still v2**

Run: `uv run pytest -q`
Expected: the whole suite passes (the existing CFB tests included). Then confirm nothing flipped the live model:

```bash
grep -n "CFB_MODEL_VERSION" .github/workflows/*.yml
```
Expected: exactly the `${{ vars.CFB_MODEL_VERSION || 'v2' }}` lines in `generate-cfb.yml` and `injury-watch.yml` (default `'v2'`); no workflow sets the variable.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_cfb_efficiency.py tests/scripts/test_build_cfb_efficiency.py tests/test_workflows_cfb_advanced.py .github/workflows/build-cfb-advanced.yml .github/workflows/generate-cfb.yml .github/workflows/injury-watch.yml assets/cfb/efficiency_ratings.parquet
git commit -m "feat(cfb): weekly CFBD game-data + efficiency refresh, CFB_MODEL_VERSION wired into the cfb generate jobs"
```

**After this plan (not a task - the user's decision):** read `docs/superpowers/reports/<date>-cfb-v3-gate.md` (also the PR summary). Only if the user approves the gate: `gh variable set CFB_MODEL_VERSION --body v3`; switching back is `gh variable set CFB_MODEL_VERSION --body v2`. The serving views take the latest-generated row per game, and the upsert key is `(game_pk, model_version)`, so both versions' rows coexist and the track record continues.

---

## Self-Review

**1. Spec coverage** (spec section -> task):

| Spec requirement | Task |
|---|---|
| Sec. 1 data: advanced incl. postseason + `season_type` | 2 (+4 backfill) |
| Sec. 1 data: havoc, drives (PPD, trips inside the 40, start position), weather (+ forecasts), talent, venues | 1 (parsers), 3 (script), 4 (backfill) |
| Sec. 1 mapping via `cfbd_to_espn`, drop + count unmapped/FCS, stamp season/week/season_type/game_id | 1, 2, 3 (`/games` meta supplies week/season_type to `/drives`) |
| Sec. 1 rest days from schedules; travel + time-zone change from venue coordinates (neutral sites included) | 7 |
| Sec. 1 shared client with retry/backoff + logged call counter | 1 (`CfbdClient`), 3 (`summary()` printed) |
| Leak rule | 5 (`state_before`), 7, 8 (leak tests), 10 (residual features use prior-season FPI/SRS only) |
| Sec. 2 weekly ridge per metric (play-weighted, ridge fitted, FBS-only), early-season prior from returning production + talent, decay curve | 5, 6 |
| Sec. 2 to points: rush/pass-mixed PPA x expected plays, success, explosiveness, finishing; efficiency margin/total | 5 (`side_features`), 9 (`fit_points_map`), 8 (`eff_margin_total`) |
| Sec. 3 blend + context (weather, travel, rest, talent), fitted on 2016-22, each can be 0; moneyline refit (`gameline_v3.json`) | 7, 8, 9 |
| Sec. 3 model-only | 8 (no market column enters `predict_v3`) |
| Sec. 4 v2 baseline reproduced first, stop on mismatch | 10 (`check_baseline`, exit 2) |
| Sec. 4 six criteria, per-season report + flags, combined verdict | 10 |
| Sec. 4 residual check (ridge + boosting, 2016-22 train, 2023-25 out-of-sample R^2 / MAE change / top features / v4 candidates), report only | 10 |
| Sec. 4 output `assets/cfb/v3_gate.json` + markdown summary | 10 |
| Sec. 5 `generate_cfb.py` `cfb-ratings-v3` via `CFB_MODEL_VERSION` default v2, weights in `v3_weights.json` | 9 (weights), 11 |
| Sec. 5 weekly Monday job refreshes postseason advanced, havoc, drives, recomputes ratings; game-day weather before daily generate; missing weather = 0 | 12 (workflows, snapshot), 11 (weather) |
| Sec. 5 track record continues, site unchanged | 11 (upsert key unchanged), no site task |
| Testing list (parsers incl. unmapped/FCS drops + NaN, ridge recovers offsets, leak invariant, context construction incl. dome / travel / time zone, gate evaluator, `CFB_MODEL_VERSION` switching, existing CFB tests green) | 1, 5, 8, 7, 10, 11, 12 (full suite) |
| Out of scope | respected (no site, GraphQL, props, board, track-record changes) |
| "Live model stays on v2" | Global Constraints; Task 12 Step 7 greps for it |

**2. Placeholder scan:** every step carries concrete code, diffs or exact commands; the only runs that cannot be unit-tested (Tasks 4, 6, 9, 10, 12 `[NETWORK]` / `[LOCAL RUN]`) give exact commands, expected output and the action for each deviation. No placeholder wording anywhere (no deferred details, no "add appropriate handling", no "similar to Task N").

**3. Type consistency:** names were checked against each task's Interfaces block - `EffConfig` fields (`ridge, half_life_games, prior_floor, k0, k_ret, k_tal`) are the same in `efficiency.py`, `eff_fit.GRID`, `eff_config.json` and `load_eff_config`; `POINT_FEATURES` order is shared by `side_features`, `v3.eff_margin_total`, `v3_table.SIDE_COLS` and `v3_fit.fit_points_map`; `CONTEXT_COLS` (`MARGIN_CTX + TOTAL_CTX + NUISANCE`) is shared by `context.context_features`, `v3_table.TABLE_COLUMNS` and the fit; `V3Inputs` is built only by `v3_data.load_v3_inputs` (and the tests); table columns `margin_v2` / `total_v2` are raw (pre-bias) everywhere, and the gate applies v2's bias through `build_gameline` exactly as `generate_cfb` does; `v3.VERSION` equals `generate_cfb.GAME_MODEL_VERSION_V3`.

**Spec ambiguities resolved in this plan:**
- *Postseason week numbering collides with regular weeks* -> `season_type` column; the efficiency walk and `build_team_context` use regular rows only, bowls feed only the previous-season final ratings.
- *Spec table lists no `/games` pull, but `/drives` has no week / season_type and the walk needs venue, neutral flag and home team* -> a `cfbd_games.parquet` meta pull (also supplies CFBD's pre-game Elo for the residual check).
- *"margin_v2_ratings" vs "prior_margin(decaying)"* -> `margin_v2` = v2's own prior-blended margin (so a1 = 1, others 0 reproduces v2) and `prior_margin` = an extra decaying-prior term the fit may re-weight.
- *"weight by plays; decay curve of the same shape family as priors_decay.json"* -> the ridge is play-weighted; the prior-to-season blend uses the exponential half-life + floor curve (`priors.prior_weight`) on games played.
- *Fitting the rating hyperparameters* -> one shared set (ridge, half-life, floor, retention), chosen on one-week-ahead PPA error over 2016-22.
- *Havoc* is rated per spec but absent from the spec's points-map list -> included as a points-map feature (the fit can zero it).
- *Missing-indicator handling vs "missing weather = 0 adjustment"* -> `weather_missing` is a nuisance column during the fit and dropped at prediction time.
- *"Each context term can be fitted to 0"* -> lasso with the one-standard-error rule, kept terms refit unpenalised.
- *"ML calibration"* -> the existing Normal-to-pmf mapping with sigma refit on 2016-22 residuals; ECE = 10 equal-width bins.
- *Residual-check inputs* -> FPI / SRS only from the previous season; CFBD pregame win probability excluded (market-derived); "player usage" = team returning-usage share already in `priors.parquet`; a v4 candidate must save at least 0.005 MAE points in every held-out season.
- *"A game-day weather pull runs before the daily CFB generate"* -> done inside the v3 path of `generate_cfb.py` (one call, never fatal) because `injury-watch.yml` also regenerates CFB; both workflows carry the model switch and the key.
- *Weekly refresh in January* -> the cron / blank dispatch pulls the previous season too, so bowls are captured.
