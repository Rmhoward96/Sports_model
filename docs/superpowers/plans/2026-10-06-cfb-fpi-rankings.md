# CFB FPI-Blend Power Rankings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** CFB power rankings rank teams by 60% ESPN FPI + 40% our results-based model rating. FPI and the model rating are stored and shown alongside the blend.

**Architecture:** `cfb/espn.py` gains an FPI fetch/parse for ESPN's public core API. `context/power.py` gains two pure helpers: `blend_fpi` (adds `fpi`/`model_rating`, replaces `rating` with the blend) and `prev_ranks_from_published` (last published blended week -> {team: rank}). `scripts/build_team_context.py` wires them into `build_cfb`, fed by two new loaders (FPI, published rankings). `power_rankings` gains two columns (migration). `site/app.js` shows FPI/MODEL columns and the blend wording for CFB. NFL is untouched.

**Tech Stack:** Python 3.12, pandas, httpx + tenacity, psycopg (via `sportsmodel.db.get_postgres`), pytest (`uv run pytest`), vanilla JS site with `node --test` (`cd site && npm test`).

**Spec:** `docs/superpowers/specs/2026-10-06-cfb-fpi-rankings-design.md`

## Global Constraints

- Blend: `rating = 0.6 * fpi + 0.4 * model_rating` (raw points, no z-scoring). Constant `W_FPI = 0.6` in `sportsmodel/context/power.py`.
- CFB only. NFL rankings, ratings and `prev_rank` logic are unchanged. NFL rows write NULL `fpi`/`model_rating`.
- FPI source: `https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{season}/powerindex` with `limit=200&page=N`. No key. Team id = the digits after `/teams/` in `item["team"]["$ref"]`. Value = the `predictives` entry with `name == "fpi"`.
- Missing FPI (fetch failure, empty, or a team absent): warn, and the team's rating = model rating with `fpi` NULL. The job never fails because of FPI.
- `prev_rank`/`move` (CFB) come only from the previous published CFB week of the same season, and only if that week has at least one non-null `fpi`. Otherwise they are NULL. That covers the rollout week, a week with an FPI outage, and running without DATABASE_URL.
- SOS/SOV use the blended rating (they are computed from `rating` inside `rankings()`).
- Site wording constant: `60% ESPN FPI + 40% CappingAlpha model`. New site cache key: `20261007c` (replaces `20261007b`).
- Migration `db/migration_power_fpi.sql` is run by the user in Supabase BEFORE merge. Use one `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` statement per column (the migration test parses that form).
- Run Python tests with `uv run --no-sync pytest -q <path>`. Run site tests with `cd site && npm test`.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. No pushes or merges.

---

### Task 1: ESPN FPI client

**Files:**
- Modify: `src/sportsmodel/cfb/espn.py` (after `_get`, near line 30)
- Create: `tests/fixtures/cfb/espn_fpi.json`
- Test: `tests/cfb/test_espn.py` (append)

**Interfaces:**
- Produces: `sportsmodel.cfb.espn.parse_fpi(payload: dict) -> dict[str, float]` and `sportsmodel.cfb.espn.fetch_fpi(season: int) -> dict[str, float]` (team id str -> FPI points). `fetch_fpi` calls the module-level `_get_core(path, params)`. Tests monkeypatch `_get_core`.

- [ ] **Step 1: Create the fixture** `tests/fixtures/cfb/espn_fpi.json`. It is a trimmed real payload (2026-10-06), plus one item without an fpi entry and one with a non-team ref:

```json
{
  "count": 4, "pageIndex": 1, "pageSize": 200, "pageCount": 1,
  "items": [
    {"team": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/2026/teams/194?lang=en&region=us"},
     "season": 2026, "lastUpdated": "2026-10-06T08:00Z",
     "predictives": [{"name": "fpi", "value": 28.837, "displayValue": "28.8"},
                     {"name": "fpirank", "value": 1.0, "displayValue": "1st"}]},
    {"team": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/2026/teams/61?lang=en&region=us"},
     "season": 2026, "lastUpdated": "2026-10-06T08:00Z",
     "predictives": [{"name": "fpirank", "value": 2.0}, {"name": "fpi", "value": 28.2, "displayValue": "28.2"}]},
    {"team": {"$ref": "http://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/2026/teams/2032?lang=en&region=us"},
     "season": 2026, "lastUpdated": "2026-10-06T08:00Z",
     "predictives": [{"name": "fpirank", "value": 90.0}]},
    {"team": {"$ref": "http://example.invalid/not-a-team"},
     "season": 2026, "predictives": [{"name": "fpi", "value": 1.0}]}
  ]
}
```

- [ ] **Step 2: Write the failing tests** (append to `tests/cfb/test_espn.py`; add `import json, pathlib` at the top if absent, and `from sportsmodel.cfb import espn` if the file doesn't already import it under that name):

```python
_FPI_FIXTURE = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "cfb" / "espn_fpi.json"


def test_parse_fpi_maps_team_id_to_fpi_value_and_skips_items_without_one():
    payload = json.loads(_FPI_FIXTURE.read_text())
    assert espn.parse_fpi(payload) == {"194": 28.837, "61": 28.2}
    assert espn.parse_fpi({}) == {} and espn.parse_fpi(None) == {}


def test_parse_fpi_ignores_non_numeric_values():
    item = {"team": {"$ref": ".../teams/5?lang=en"},
            "predictives": [{"name": "fpi", "value": None}]}
    bad = {"team": {"$ref": ".../teams/6?lang=en"},
           "predictives": [{"name": "fpi", "value": "n/a"}]}
    assert espn.parse_fpi({"items": [item, bad]}) == {}


def test_fetch_fpi_follows_every_page(monkeypatch):
    pages = {
        1: {"pageIndex": 1, "pageCount": 2, "items": [
            {"team": {"$ref": ".../teams/194?x"}, "predictives": [{"name": "fpi", "value": 28.8}]}]},
        2: {"pageIndex": 2, "pageCount": 2, "items": [
            {"team": {"$ref": ".../teams/61?x"}, "predictives": [{"name": "fpi", "value": 28.2}]}]},
    }
    calls = []

    def fake(path, params=None):
        calls.append((path, dict(params)))
        return pages[params["page"]]
    monkeypatch.setattr(espn, "_get_core", fake)
    assert espn.fetch_fpi(2026) == {"194": 28.8, "61": 28.2}
    assert calls == [("/seasons/2026/powerindex", {"limit": 200, "page": 1}),
                     ("/seasons/2026/powerindex", {"limit": 200, "page": 2})]
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/cfb/test_espn.py -k fpi`
Expected: FAIL with `AttributeError: module 'sportsmodel.cfb.espn' has no attribute 'parse_fpi'`

- [ ] **Step 4: Implement.** In `src/sportsmodel/cfb/espn.py`, add `import math` and `import re` to the imports. Then insert after `_get`:

```python
# ESPN's core API (FPI lives here, not on the site API under _BASE).
_CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
_TEAM_REF = re.compile(r"/teams/(\d+)")


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get_core(path: str, params: dict | None = None) -> Any:
    """GET {_CORE}{path} -> parsed JSON, same retry policy as `_get`."""
    r = httpx.get(f"{_CORE}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_fpi(payload: dict | None) -> dict[str, float]:
    """ESPN /powerindex page -> {ESPN team id: FPI} (points vs an average team on a neutral
    field). Items whose team ref has no numeric id or whose `fpi` predictive is missing or
    non-numeric are skipped. Pure."""
    out: dict[str, float] = {}
    for item in (payload or {}).get("items") or []:
        m = _TEAM_REF.search(((item.get("team") or {}).get("$ref")) or "")
        if not m:
            continue
        val = next((p.get("value") for p in item.get("predictives") or []
                    if p.get("name") == "fpi"), None)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(val):
            out[m.group(1)] = float(val)
    return out


def fetch_fpi(season: int) -> dict[str, float]:
    """Current ESPN FPI for every team ESPN rates in `season` (all pages). ESPN serves only the
    latest FPI run; there is no weekly history."""
    out: dict[str, float] = {}
    page = 1
    while True:
        payload = _get_core(f"/seasons/{season}/powerindex", {"limit": 200, "page": page})
        out.update(parse_fpi(payload))
        if page >= int(payload.get("pageCount") or 1):
            return out
        page += 1
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run --no-sync pytest -q tests/cfb/test_espn.py`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/sportsmodel/cfb/espn.py tests/cfb/test_espn.py tests/fixtures/cfb/espn_fpi.json
git commit -m "feat(cfb): ESPN FPI fetch/parse (core API powerindex)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Blend and previous-rank helpers in `context/power.py`

**Files:**
- Modify: `src/sportsmodel/context/power.py` (append after `rankings`, end of file)
- Test: `tests/context/test_power.py` (append; reuses the file's `_power`, `_log_row`, `_log` helpers)

**Interfaces:**
- Produces:
  - `W_FPI: float = 0.6`
  - `blend_fpi(power_df: pd.DataFrame, fpi: dict[str, float] | None, w_fpi: float = W_FPI) -> pd.DataFrame`. Returns a copy with `model_rating` (the input `rating`), `fpi` (float, NaN when missing) and the blended `rating`.
  - `prev_ranks_from_published(published: pd.DataFrame | None, season: int, week: int) -> dict[str, int]`. `published` has columns `season, week, team, rank, fpi`.

- [ ] **Step 1: Write the failing tests** (append to `tests/context/test_power.py`; extend the import line to `from sportsmodel.context.power import W_FPI, blend_fpi, cfb_power, nfl_market_scale, nfl_power, prev_ranks_from_published, rankings`):

```python
def test_blend_fpi_is_sixty_forty_and_keeps_both_parts():
    assert W_FPI == 0.6
    b = blend_fpi(_power({"A": 10.0, "B": 0.0}), {"A": 20.0, "B": -5.0}).set_index("team")
    assert b.loc["A", "rating"] == pytest.approx(0.6 * 20.0 + 0.4 * 10.0)   # 16.0
    assert b.loc["B", "rating"] == pytest.approx(0.6 * -5.0 + 0.4 * 0.0)    # -3.0
    assert b.loc["A", "model_rating"] == 10.0 and b.loc["A", "fpi"] == 20.0


def test_blend_fpi_team_without_fpi_keeps_its_model_rating():
    b = blend_fpi(_power({"A": 10.0, "FCS": -30.0}), {"A": 20.0}).set_index("team")
    assert b.loc["FCS", "rating"] == -30.0 and pd.isna(b.loc["FCS", "fpi"])
    for none in (None, {}):
        nb = blend_fpi(_power({"A": 10.0}), none)
        assert nb["rating"].tolist() == [10.0] and nb["fpi"].isna().all()
        assert nb["model_rating"].tolist() == [10.0]


def test_blend_fpi_does_not_mutate_its_input():
    p = _power({"A": 10.0})
    blend_fpi(p, {"A": 20.0})
    assert p["rating"].tolist() == [10.0] and "fpi" not in p.columns


def test_rankings_on_blended_frame_rank_and_sos_use_the_blend():
    # model order A > B > C > D; FPI flips B above A
    cur = blend_fpi(_power({"A": 6.0, "B": 2.0, "C": -1.0, "D": -7.0}),
                    {"A": 0.0, "B": 10.0, "C": -1.0, "D": -7.0})
    r = rankings(cur, None, None, _log()).set_index("team")
    assert r["rank"].to_dict() == {"B": 1, "A": 2, "C": 3, "D": 4}
    # B played A once before week 5: SOS = A's blended rating 0.6*0 + 0.4*6 = 2.4
    assert r.loc["B", "sos"] == pytest.approx(2.4)
    assert r.loc["A", "model_rating"] == 6.0 and r.loc["A", "fpi"] == 0.0


def _published(rows):
    return pd.DataFrame(rows, columns=["season", "week", "team", "rank", "fpi"])


def test_prev_ranks_from_published_uses_the_last_blended_week_of_the_season():
    pub = _published([
        (2026, 4, "A", 2, 10.0), (2026, 4, "B", 1, 12.0),
        (2026, 5, "A", 1, 11.0), (2026, 5, "B", 2, 9.0),
        (2026, 6, "A", 2, 11.0),                     # the current week itself: ignored
        (2025, 15, "A", 9, 1.0),                     # last season: ignored
    ])
    assert prev_ranks_from_published(pub, 2026, 6) == {"A": 1, "B": 2}
    assert prev_ranks_from_published(pub, 2026, 5) == {"A": 2, "B": 1}


def test_prev_ranks_from_published_is_empty_when_last_week_was_not_blended_or_absent():
    model_only = _published([(2026, 5, "A", 1, None), (2026, 5, "B", 2, None)])
    assert prev_ranks_from_published(model_only, 2026, 6) == {}
    assert prev_ranks_from_published(model_only, 2026, 5) == {}     # nothing earlier
    assert prev_ranks_from_published(None, 2026, 6) == {}
    assert prev_ranks_from_published(_published([]), 2026, 6) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/context/test_power.py -k "blend or published"`
Expected: FAIL with `ImportError: cannot import name 'W_FPI'`

- [ ] **Step 3: Implement** (append to `src/sportsmodel/context/power.py`):

```python
# CFB power rankings: 60% ESPN FPI + 40% our results-based rating (both points vs an average
# team on a neutral field; spec 2026-10-06-cfb-fpi-rankings-design).
W_FPI = 0.6


def blend_fpi(power_df: pd.DataFrame, fpi: dict[str, float] | None,
              w_fpi: float = W_FPI) -> pd.DataFrame:
    """Copy of ``power_df`` with ``model_rating`` (its ``rating``), ``fpi`` (ESPN FPI by team,
    NaN when missing) and ``rating = w_fpi * fpi + (1 - w_fpi) * model_rating``. A team without
    an FPI value (incl. the "FCS" pseudo-team, or every team when ``fpi`` is None/empty) keeps
    its model rating. Ranking, SOS and SOV in ``rankings`` then all use the blend."""
    out = power_df.copy()
    out["model_rating"] = out["rating"].astype(float)
    out["fpi"] = out["team"].astype(str).map(fpi or {}).astype(float)
    has = out["fpi"].notna()
    out.loc[has, "rating"] = (w_fpi * out.loc[has, "fpi"]
                              + (1 - w_fpi) * out.loc[has, "model_rating"])
    return out


def prev_ranks_from_published(published: pd.DataFrame | None, season: int,
                              week: int) -> dict[str, int]:
    """{team: rank} of the last published week before (``season``, ``week``), same season only.
    Returns {} when there is none, or when that week was not blended (no non-null ``fpi``: the
    model-only rankings from before the FPI blend, or a week whose FPI fetch failed). That way
    ``move`` never compares a blended rank with a model-only one."""
    if published is None or published.empty:
        return {}
    p = published[(published["season"] == season) & (published["week"] < week)]
    if p.empty:
        return {}
    last = p[p["week"] == p["week"].max()]
    if last["fpi"].isna().all():
        return {}
    return {str(t): int(r) for t, r in zip(last["team"], last["rank"])}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run --no-sync pytest -q tests/context/test_power.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/context/power.py tests/context/test_power.py
git commit -m "feat(context): 60/40 FPI blend + previous published rank helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `power_rankings` columns (db + migration)

**Files:**
- Create: `db/migration_power_fpi.sql`
- Modify: `src/sportsmodel/db.py:1015-1017` (`TEAM_CONTEXT_COLUMNS["power_rankings"]`)
- Modify: `scripts/build_team_context.py:163-171` (`_rank_frame`)
- Test: `tests/scripts/test_build_team_context.py` (`test_migration_matches_db_columns_and_keys` ~line 392; NFL rankings test ~line 502)

**Interfaces:**
- Produces: `db.TEAM_CONTEXT_COLUMNS["power_rankings"]` ends with `..., "sov", "home_record", "road_record", "fpi", "model_rating"`. `_rank_frame` fills `fpi`/`model_rating` with NaN when the frame lacks them (NFL).

- [ ] **Step 1: Update the tests.**
  - In `test_migration_matches_db_columns_and_keys`, replace the `later = ...` line with:

```python
    later = "\n".join((dbdir / f).read_text() for f in
                      ("migration_power_results.sql", "migration_power_fpi.sql"))   # ALTER ... ADD COLUMN
```

  - Append to the end of that test:

```python
    fpi = (dbdir / "migration_power_fpi.sql").read_text()
    assert "CREATE OR REPLACE VIEW power_rankings_current" in fpi
    assert "GRANT SELECT ON power_rankings_current TO anon, authenticated" in fpi
```

  - In `test_nfl_rankings_use_the_results_rating`, append:

```python
    assert rk["fpi"].isna().all() and rk["model_rating"].isna().all()   # CFB-only blend
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/scripts/test_build_team_context.py -k "migration or nfl_rankings"`
Expected: FAIL (`migration_power_fpi.sql` not found; `KeyError: 'fpi'`)

- [ ] **Step 3: Create `db/migration_power_fpi.sql`:**

```sql
-- CFB power rankings = 60% ESPN FPI + 40% our results-based rating (2026-10-06).
-- Run BEFORE merging feat/cfb-fpi-rankings (the daily build-team-context job writes and
-- reads these columns).
--   fpi           ESPN Football Power Index used in the blend (CFB; NULL: NFL, or no FPI)
--   model_rating  our results-based rating before the blend (CFB; NULL: NFL)
--   rating        CFB: 0.6 * fpi + 0.4 * model_rating (model_rating alone when fpi is NULL)

ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS fpi double precision;
ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS model_rating double precision;

-- p.* is expanded when a view is created: recreate so the view carries the new
-- columns (they are appended at the end, which CREATE OR REPLACE allows).
CREATE OR REPLACE VIEW power_rankings_current WITH (security_invoker = true) AS
  SELECT p.*
  FROM power_rankings p
  JOIN (SELECT DISTINCT ON (sport) sport, season, week
        FROM power_rankings
        ORDER BY sport, season DESC, week DESC) l
    ON l.sport = p.sport AND l.season = p.season AND l.week = p.week;

GRANT SELECT ON power_rankings_current TO anon, authenticated;
```

- [ ] **Step 4: Update `src/sportsmodel/db.py`.** `TEAM_CONTEXT_COLUMNS["power_rankings"]` becomes:

```python
    "power_rankings": [
        "sport", "season", "week", "team", "rank", "rating", "prev_rank", "move", "units",
        "sos", "su", "ats", "games", "conf", "sov", "home_record", "road_record",
        "fpi", "model_rating"],
```

- [ ] **Step 5: Update `_rank_frame` in `scripts/build_team_context.py`.** Add after the `games` fill:

```python
    for c in ("fpi", "model_rating"):     # CFB-only blend parts; NFL rows store NULL
        if c not in rk.columns:
            rk[c] = np.nan
```

- [ ] **Step 6: Run the file's tests**

Run: `uv run --no-sync pytest -q tests/scripts/test_build_team_context.py`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add db/migration_power_fpi.sql src/sportsmodel/db.py scripts/build_team_context.py tests/scripts/test_build_team_context.py
git commit -m "feat(db): power_rankings fpi + model_rating columns (migration_power_fpi.sql)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Wire the blend into the CFB team-context job

**Files:**
- Modify: `scripts/build_team_context.py`
  - the imports near line 52 (`from sportsmodel.context.power import rankings`)
  - `load_cfb_sources` (~line 395)
  - `build_cfb` (~line 427: signature, and the rankings block ~lines 457-467)
  - `print_sample` (~line 531)
- Test: `tests/scripts/test_build_team_context.py` (append)

**Interfaces:**
- Consumes: `espn.fetch_fpi(season)` (Task 1); `blend_fpi`, `prev_ranks_from_published` (Task 2); the `fpi`/`model_rating` columns (Task 3).
- Produces:
  - `build_cfb(..., fpi: dict[str, float] | None = None, published: pd.DataFrame | None = None)`
  - `load_cfb_fpi(season: int) -> dict[str, float] | None`
  - `load_cfb_published(season: int) -> pd.DataFrame | None` (columns `season, week, team, rank, fpi`)
  - `load_cfb_sources(now)` returns the extra keys `fpi` and `published`.

- [ ] **Step 1: Write the failing tests** (append to `tests/scripts/test_build_team_context.py`):

```python
def test_cfb_rankings_blend_fpi_sixty_forty():
    src = cfb_sources()
    teams = _cfb_teams()
    priors = {t: 1500.0 + 40.0 * i for i, t in enumerate(teams)}
    base = btc.build_cfb(**src, now=NOW, priors=priors, rp=_rp(), fbs=set(teams))["power_rankings"]
    fpi = {t: float(-3 * i) for i, t in enumerate(teams)}         # reverses the prior order
    fpi.pop(teams[-1])                                            # one team without FPI
    rk = btc.build_cfb(**src, now=NOW, priors=priors, rp=_rp(), fbs=set(teams),
                       fpi=fpi)["power_rankings"].set_index("team")
    model = base.set_index("team")["rating"]
    for t in teams[:-1]:
        assert rk.loc[t, "model_rating"] == pytest.approx(model[t])
        assert rk.loc[t, "fpi"] == fpi[t]
        assert rk.loc[t, "rating"] == pytest.approx(0.6 * fpi[t] + 0.4 * model[t])
    last = teams[-1]
    assert pd.isna(rk.loc[last, "fpi"]) and rk.loc[last, "rating"] == pytest.approx(model[last])
    assert list(rk.sort_values("rank").index) == list(rk["rating"].sort_values(ascending=False).index)


def test_cfb_move_comes_from_the_last_published_blended_week():
    src = cfb_sources()
    teams = _cfb_teams()
    fpi = {t: float(i) for i, t in enumerate(teams)}
    rk0 = btc.build_cfb(**src, now=NOW, fpi=fpi)["power_rankings"]
    s, w = int(rk0["season"].iloc[0]), int(rk0["week"].iloc[0])
    assert rk0["prev_rank"].isna().all() and rk0["move"].isna().all()   # nothing published
    prev = {t: i + 1 for i, t in enumerate(reversed(teams))}
    pub = pd.DataFrame([(s, w - 1, t, r, 1.0) for t, r in prev.items()],
                       columns=["season", "week", "team", "rank", "fpi"])
    rk = btc.build_cfb(**src, now=NOW, fpi=fpi, published=pub)["power_rankings"].set_index("team")
    for t in teams:
        assert rk.loc[t, "prev_rank"] == prev[t]
        assert rk.loc[t, "move"] == prev[t] - rk.loc[t, "rank"]
    model_only = pub.assign(fpi=np.nan)                                 # rollout week
    rk2 = btc.build_cfb(**src, now=NOW, fpi=fpi, published=model_only)["power_rankings"]
    assert rk2["prev_rank"].isna().all() and rk2["move"].isna().all()


def test_load_cfb_fpi_warns_and_returns_none_on_failure_or_empty(monkeypatch, capsys):
    from sportsmodel.cfb import espn

    def boom(season):
        raise RuntimeError("503")
    monkeypatch.setattr(espn, "fetch_fpi", boom)
    assert btc.load_cfb_fpi(2026) is None
    monkeypatch.setattr(espn, "fetch_fpi", lambda season: {})
    assert btc.load_cfb_fpi(2026) is None
    monkeypatch.setattr(espn, "fetch_fpi", lambda season: {"194": 28.8})
    assert btc.load_cfb_fpi(2026) == {"194": 28.8}
    out = capsys.readouterr().out
    assert out.count("::warning::team-context: cfb: ESPN FPI") == 2


def test_load_cfb_published_reads_this_seasons_cfb_rows(monkeypatch, capsys):
    monkeypatch.setattr(db.config, "DATABASE_URL", None)
    assert btc.load_cfb_published(2026) is None
    assert "no published rankings" in capsys.readouterr().out
    seen = {}

    class Cur:
        def execute(self, sql, params):
            seen["sql"], seen["params"] = sql, params

        def fetchall(self):
            return [(2026, 5, "194", 1, 28.8)]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class Conn:
        def cursor(self):
            return Cur()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    monkeypatch.setattr(db.config, "DATABASE_URL", "postgres://fake")
    monkeypatch.setattr(db, "get_postgres", lambda: Conn())
    df = btc.load_cfb_published(2026)
    assert list(df.columns) == ["season", "week", "team", "rank", "fpi"]
    assert df.iloc[0].tolist() == [2026, 5, "194", 1, 28.8]
    assert "sport = 'cfb'" in seen["sql"] and seen["params"] == (2026,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run --no-sync pytest -q tests/scripts/test_build_team_context.py -k "fpi or published"`
Expected: FAIL (`TypeError: build_cfb() got an unexpected keyword argument 'fpi'`; `AttributeError: ... 'load_cfb_fpi'`)

- [ ] **Step 3: Implement.**
  - Change the import line `from sportsmodel.context.power import rankings  # noqa: E402` to:

```python
from sportsmodel.context.power import (  # noqa: E402
    blend_fpi, prev_ranks_from_published, rankings)
```

  - Add the two loaders above `load_cfb_sources`:

```python
def load_cfb_fpi(season: int) -> dict[str, float] | None:
    """ESPN FPI {team id: points} for the ranking season, or None when the fetch fails or comes
    back empty. CFB is then ranked on the model rating alone for this run; the job never fails
    over FPI."""
    from sportsmodel.cfb import espn
    try:
        fpi = espn.fetch_fpi(season)
    except Exception as exc:  # noqa: BLE001
        warn(f"cfb: ESPN FPI fetch failed ({type(exc).__name__}: {exc}) -- "
             "ranking on the model rating only")
        return None
    if not fpi:
        warn("cfb: ESPN FPI came back empty -- ranking on the model rating only")
        return None
    return fpi


def load_cfb_published(season: int) -> pd.DataFrame | None:
    """This season's published CFB power_rankings rows (season, week, team, rank, fpi): the
    previous blended week sets prev_rank / move. None (both left blank) without DATABASE_URL."""
    if not config.DATABASE_URL:
        warn("cfb: DATABASE_URL unset -- no published rankings, prev_rank / move left blank")
        return None
    with db.get_postgres() as pg, pg.cursor() as cur:
        cur.execute("SELECT season, week, team, rank, fpi FROM power_rankings "
                    "WHERE sport = 'cfb' AND season = %s", (season,))
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=["season", "week", "team", "rank", "fpi"])
```

  - In `load_cfb_sources`, before the `return`, add `rs = _cfb_season(sched, now)`. Then extend the returned dict with:

```python
            "fpi": load_cfb_fpi(rs), "published": load_cfb_published(rs)}
```

  (keep the existing keys; the dict now ends `..., "advanced": advanced, "fpi": ..., "published": ...}`).
  - In the `build_cfb` signature, after `rp: ResultsParams | None = None`, add:

```python
              fpi: dict[str, float] | None = None,
              published: pd.DataFrame | None = None
```

  - In `build_cfb`, replace

```python
        power, prev = power_asof(cfb_results_games(schedules), rp or load_params("cfb"), s, w,
                                 preseason=pre, members=members)
        ur = unit_ratings_asof(ug, s, w, blend_k=POWER_BLEND_K) if ug is not None else None
        rk = _rank_frame(rankings(power, prev, ur, log), "cfb", _cfb_conf(sched))
```

  with

```python
        power, _ = power_asof(cfb_results_games(schedules), rp or load_params("cfb"), s, w,
                              preseason=pre, members=members)
        ur = unit_ratings_asof(ug, s, w, blend_k=POWER_BLEND_K) if ug is not None else None
        # 60% ESPN FPI + 40% the results rating; move vs the last PUBLISHED blended week
        ranked = rankings(blend_fpi(power, fpi), None, ur, log)
        prev = prev_ranks_from_published(published, s, w)
        ranked["prev_rank"] = ranked["team"].astype(str).map(prev).astype("Int64")
        ranked["move"] = (ranked["prev_rank"] - ranked["rank"]).astype("Int64")
        rk = _rank_frame(ranked, "cfb", _cfb_conf(sched))
```

  - In `print_sample`, replace the `cols = [...]` line with:

```python
        cols = ["rank", "team", "rating", "prev_rank", "move", "sos", "su", "ats"]
        if sport == "cfb":
            cols[3:3] = ["fpi", "model_rating"]
```

- [ ] **Step 4: Run the file's tests and the context suite**

Run: `uv run --no-sync pytest -q tests/scripts/test_build_team_context.py tests/context tests/test_workflows_team_context.py`
Expected: all PASS. The existing `test_cfb_rankings_use_the_results_rating_with_sov_and_venue_records` passes unchanged, because no `fpi` means rating = model rating.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_team_context.py tests/scripts/test_build_team_context.py
git commit -m "feat(team-context): CFB power rankings = 60% ESPN FPI + 40% model; move vs last published blended week

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Site: FPI/MODEL columns, blend wording, cache key

**Files:**
- Modify: `site/app.js`
  - `rkPowerLine` ~line 1434
  - `rkColumns` ~line 1619
  - `buildRankings` ~line 1697
- Modify: `site/tests/board.test.mjs` (append a test)
- Modify: every `site/*.html` (cache key `20261007b` -> `20261007c`) and `site/tests/smoke.test.mjs`

**Interfaces:**
- Consumes: `power_rankings_current` rows now carry `fpi` and `model_rating` (Task 3).
- Produces: `RK_CFB_BLEND = "60% ESPN FPI + 40% CappingAlpha model"` (top-level const in app.js).

- [ ] **Step 1: Write the failing test** (append to `site/tests/board.test.mjs`):

```js
test("CFB rankings: Rating is the 60/40 FPI blend with FPI and MODEL columns right after it; NFL keeps its table and intro", async () => {
  const cfbRows = [{ sport: "cfb", rank: 1, team: "194", rating: 25.14, fpi: 28.8, model_rating: 19.64, conf: 5, season: 2026, week: 6, units: {} },
                   { sport: "cfb", rank: 2, team: "61", rating: 20, fpi: null, model_rating: 20, conf: 8, season: 2026, week: 6, units: {} }];
  const A = loadScripts(FILES, { page: "rankings", globals: { location: { search: "?sport=cfb", href: "http://localhost/rankings.html?sport=cfb" },
    fetch: async () => ({ ok: true, json: async () => cfbRows }) } });
  const html = await A.buildRankings();
  const heads = [...html.matchAll(/<th data-sort="(\w+)"/g)].map((m) => m[1]);
  assert.deepEqual(heads.slice(0, 6), ["rank", "team", "conf", "rating", "fpi", "model_rating"]);
  assert.ok(html.includes(">FPI") && html.includes(">MODEL"), "column headers");
  assert.ok(html.includes("+28.8") && html.includes("+19.6"), "signed FPI and model values");
  assert.ok(html.includes("60% ESPN FPI + 40% CappingAlpha model"), "the blend is explained");
  assert.ok(!/NaN|undefined/.test(html), "a missing FPI renders as a dash");
  const N = loadScripts(FILES, { page: "rankings", globals: { location: { search: "?sport=nfl", href: "http://localhost/rankings.html?sport=nfl" },
    fetch: async () => ({ ok: true, json: async () => [{ sport: "nfl", rank: 1, team: "Detroit Lions", rating: 5, season: 2026, week: 5, units: {} }] }) } });
  const nfl = await N.buildRankings();
  assert.ok(!nfl.includes('data-sort="fpi"') && !nfl.includes('data-sort="model_rating"') && !nfl.includes("ESPN FPI"));
  assert.ok(nfl.includes("Rating = points better than an average team on a neutral field, earned from this season's results"));
  assert.ok(A.rkPowerLine({ sport: "cfb", rank: 1, rating: 25 }).includes('title="Power rating: 60% ESPN FPI + 40% CappingAlpha model"'));
  assert.ok(A.rkPowerLine({ sport: "nfl", rank: 1, rating: 5 }).includes('title="Power rating weighted to this season"'));
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd site && node --test tests/board.test.mjs`
Expected: FAIL on the `heads.slice(0, 6)` assertion

- [ ] **Step 3: Implement in `site/app.js`.**
  - Above `rkPowerLine` (after the `rkRating` line), add:

```js
// CFB power rating = 60% ESPN FPI + 40% the CappingAlpha results rating (power_rankings.fpi / model_rating).
const RK_CFB_BLEND = "60% ESPN FPI + 40% CappingAlpha model";
```

  - Replace `rkPowerLine` with:

```js
const rkPowerLine = (p, hasPrev = true) => p
  ? `<span class="ctx-power" title="${p.sport === "cfb" ? `Power rating: ${RK_CFB_BLEND}` : "Power rating weighted to this season"}">#${ctxEsc(p.rank)} power · ${rkRating(p.rating)} ${rkMove(p, hasPrev)}</span>` : "";
```

  - In `rkColumns`, after the existing `if (sport === "cfb") cols.splice(2, 0, [...conf...]);` line and before `return cols;`, add:

```js
  if (sport === "cfb") {   // the two halves of the blended Rating, right after it
    const at = cols.findIndex((c) => c[0] === "rating") + 1;
    cols.splice(at, 0,
      ["fpi", "FPI", "desc", (x) => ctxNum(x.fpi), (x) => ctxSigned(x.fpi)],
      ["model_rating", "MODEL", "desc", (x) => ctxNum(x.model_rating), (x) => ctxSigned(x.model_rating)]);
  }
```

  - In `buildRankings`, replace the single `const intro = ...;` line with:

```js
  const results = "earned from this season's results: point margin (blowouts capped, adjusted for home vs road), wins weighed against how likely they were (road upsets earn more, bad home losses cost more) and strength of schedule. This season counts games ÷ (games + 1) — 75% after 3 games — the rest is the preseason rating.";
  const head = sport === "cfb"
    ? `Rating = ${RK_CFB_BLEND} rating, in points better than an average team on a neutral field. FPI = ESPN's Football Power Index. Model = the CappingAlpha results rating, ${results}`
    : `Rating = points better than an average team on a neutral field, ${results}`;
  const intro = `${head} SOS = average rating of opponents played; SOV = average rating of teams beaten; unit ranks by opponent-adjusted EPA/play (shown, not rated); records are regular season · ${CTX_NOT_A_PICK}`;
```

- [ ] **Step 4: Bump the cache key.** In every `site/*.html`, replace `20261007b` with `20261007c`:

```bash
cd site && for f in *.html; do sed -i '' 's/20261007b/20261007c/g' "$f"; done && cd ..
```

  In `site/tests/smoke.test.mjs`:
  - Change every `20261007b` to `20261007c`.
  - Change the old-key line `const old = ["2026", "1007", "a"].join("")` to `const old = ["2026", "1007", "b"].join("")`.

- [ ] **Step 5: Run the site suite**

Run: `cd site && npm test`
Expected: all PASS (incl. smoke cache-key tests)

- [ ] **Step 6: Commit**

```bash
git add site
git commit -m "feat(site): CFB rankings show the 60/40 FPI blend (FPI + MODEL columns); cache 20261007c

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Verification against live data

**Files:** none (verification only; no commit unless a fix is needed)

- [ ] **Step 1: Full Python suite.** Run: `uv run --no-sync pytest -q`. Expected: all pass (baseline 2500 passed, 2 skipped, plus the new tests).
- [ ] **Step 2: Site suite.** Run: `cd site && npm test`. Expected: all pass.
- [ ] **Step 3: Live dry run (no DB writes; DATABASE_URL unset locally).** Run: `uv run --no-sync python scripts/build_team_context.py --sport cfb --dry-run`.
  Expected:
  - one warning that there are no published rankings (`prev_rank / move left blank`);
  - no FPI warning;
  - the top-5 sample shows `fpi` and `model_rating` columns, with `rating ≈ 0.6*fpi + 0.4*model_rating`.
  On 2026-10-06 the top 5 was Alabama (333), Notre Dame (87), Ohio State (194), Texas (251), Georgia (61).
- [ ] **Step 4: Spot-check the blend arithmetic** on the dry-run frame:

```bash
uv run --no-sync python - <<'EOF'
import sys; sys.path[:0] = ["src", "scripts"]
import pandas as pd, build_team_context as b
now = pd.Timestamp.now(tz="UTC")
rk = b.build_cfb(**b.load_cfb_sources(now), now=now)["power_rankings"]
assert rk["fpi"].notna().sum() >= 130, rk["fpi"].notna().sum()
d = (rk["rating"] - (0.6 * rk["fpi"] + 0.4 * rk["model_rating"])).abs().max()
assert d < 1e-9, d
print(rk.head(15)[["rank", "team", "rating", "fpi", "model_rating"]].round(2).to_string(index=False))
EOF
```

  Expected: no assertion error; the top 15 matches the design check (Oregon about 10th, Ohio State 3rd).
