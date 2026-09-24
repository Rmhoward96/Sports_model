# Injury Freshness Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One ESPN-verified NFL injury source for sim + desk that never trusts a stale weekly report, a desk-visible freshness block, and a twice-daily injury watch that re-runs model → sim → desk → boards when an Out/Doubtful status changes for a game within ~30h.

**Architecture:** Pure merge in `sportsmodel/nfl/injury_report.py`; pure fingerprinting in `sportsmodel/serving/injury_watch.py`; thin script `scripts/injury_watch.py`; new workflow `injury-watch.yml`; `desk-auto-*` record a baseline.

**Tech Stack:** Python / uv / pytest, Supabase Postgres, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-24-injury-freshness-design.md`

## Global Constraints
- The user runs every migration; never apply DDL. Never print secrets.
- Branch `feat/injury-freshness` (from main); merge/push only when the user asks.
- Stale = nflverse report_week is None or < target_week. With ESPN available, stale nflverse rows are never used; without ESPN, they are used and flagged.
- Re-run trigger: any change into/out of Out/Doubtful (IR/PUP/suspension/NFI count as Out); Questionable changes never trigger. Missing stored snapshot = changed.
- Sim and desk behavior otherwise unchanged (same weighting of Out/Doubtful/Questionable).
- `uv run pytest`.

---

### Task 1: Shared NFL injury report (pure merge + IO)

**Files:** Create `src/sportsmodel/nfl/injury_report.py`; Test `tests/nfl/test_injury_report.py`.

**Produces:** `normalize_status(s) -> "Out"|"Doubtful"|"Questionable"|None`, `merge_report(nflverse_by_abbr, report_week, target_week, espn_rows, name_to_abbr, espn_available=True) -> dict`, `latest_report_week(df) -> int|None`, `current_report(now, target_week, name_to_abbr) -> dict`.

- [ ] **Step 1: Failing tests**

```python
from sportsmodel.nfl.injury_report import merge_report, normalize_status

N2A = {"Atlanta Falcons": "ATL", "Green Bay Packers": "GB"}
NFLV = {"ATL": [{"player": "Michael Penix Jr.", "position": "QB", "status": "Out", "note": None},
                {"player": "Samson Ebukam", "position": "DE", "status": "Out", "note": None}]}

def test_normalize_status():
    assert normalize_status("Injured Reserve") == "Out" and normalize_status("Suspension") == "Out"
    assert normalize_status("doubtful") == "Doubtful" and normalize_status("Questionable") == "Questionable"
    assert normalize_status("Active") is None and normalize_status("Day-To-Day") is None

def test_stale_report_dropped_when_espn_available():
    espn = [{"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Active"}]
    r = merge_report(NFLV, report_week=2, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    assert r["stale"] is True
    assert r["by_team"].get("ATL", []) == []            # Ebukam (stale-only) dropped, Penix cleared
    assert {"team": "ATL", "player": "Michael Penix Jr.", "nflverse": "Out", "espn": None} in r["conflicts"]

def test_current_report_kept_and_espn_overrides_per_player():
    espn = [{"team": "Atlanta Falcons", "player": "Michael Penix Jr.", "status": "Questionable"},
            {"team": "Green Bay Packers", "player": "Jordan Love", "status": "Out"}]
    r = merge_report(NFLV, report_week=3, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    atl = {x["player"]: (x["status"], x["source"]) for x in r["by_team"]["ATL"]}
    assert atl == {"Michael Penix Jr.": ("Questionable", "espn"), "Samson Ebukam": ("Out", "nflverse")}
    assert [(x["player"], x["status"]) for x in r["by_team"]["GB"]] == [("Jordan Love", "Out")]
    assert r["stale"] is False

def test_stale_without_espn_uses_nflverse_flagged():
    r = merge_report(NFLV, report_week=2, target_week=3, espn_rows=[], name_to_abbr=N2A, espn_available=False)
    assert r["stale"] is True and r["espn_available"] is False
    assert len(r["by_team"]["ATL"]) == 2

def test_name_matching_normalized_and_unknown_team_skipped():
    espn = [{"team": "Atlanta Falcons", "player": "michael penix jr", "status": "Active"},
            {"team": "Nowhere", "player": "X", "status": "Out"}]
    r = merge_report(NFLV, report_week=3, target_week=3, espn_rows=espn, name_to_abbr=N2A)
    assert [x["player"] for x in r["by_team"]["ATL"]] == ["Samson Ebukam"]
```

- [ ] **Step 2:** run → FAIL. **Step 3: Implement**

```python
"""One NFL injury report for the sim AND the desk: nflverse's official weekly
report, verified per player against ESPN's live injury list, and never trusted
when stale.

nflverse's newest report is LAST week's until the Wed-Fri report posts. A player
Out last week (Penix, Week 2) would otherwise stay Out in this week's sim/desk.
So: if the report's week is older than the target week and ESPN is reachable,
the report is dropped entirely and ESPN's current designations are used; ESPN
also overrides the current report per player. PURE merge + thin IO wrapper.
"""
from __future__ import annotations

import re
from datetime import datetime

_OUT_WORDS = ("out", "injured reserve", "reserve", "suspension", "physically unable", "non football")


def normalize_status(s) -> str | None:
    st = str(s or "").strip().lower()
    if st == "questionable":
        return "Questionable"
    if st == "doubtful":
        return "Doubtful"
    if any(w in st for w in _OUT_WORDS):
        return "Out"
    return None


def _key(name) -> str:
    s = re.sub(r"[.'’]", "", str(name or "").strip().lower())
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", s)


def merge_report(nflverse_by_abbr: dict, report_week, target_week, espn_rows: list[dict],
                 name_to_abbr: dict[str, str], espn_available: bool = True) -> dict:
    stale = report_week is None or (target_week is not None and int(report_week) < int(target_week))
    use_nflverse = not (stale and espn_available)
    by_team: dict[str, dict[str, dict]] = {}
    nfl_status: dict[tuple, str] = {}
    for abbr, rows in (nflverse_by_abbr or {}).items():
        for r in rows or []:
            st = normalize_status(r.get("status"))
            nfl_status[(abbr, _key(r.get("player")))] = st
            if use_nflverse and st:
                by_team.setdefault(abbr, {})[_key(r.get("player"))] = {
                    "player": r.get("player"), "position": r.get("position"), "status": st,
                    "note": r.get("note"), "source": "nflverse"}
    conflicts = []
    for e in espn_rows or []:
        abbr = name_to_abbr.get(e.get("team"))
        k = _key(e.get("player"))
        if not abbr or not k:
            continue
        st = normalize_status(e.get("status"))
        before = nfl_status.get((abbr, k))
        prior_row = by_team.get(abbr, {}).pop(k, None)
        name = prior_row["player"] if prior_row else e.get("player")
        if st:
            by_team.setdefault(abbr, {})[k] = {"player": name, "position": prior_row.get("position") if prior_row else None,
                                               "status": st, "note": None, "source": "espn"}
        if (before in ("Out", "Doubtful")) != (st in ("Out", "Doubtful")) and (before or st):
            conflicts.append({"team": abbr, "player": name, "nflverse": before, "espn": st})
    return {"by_team": {a: list(v.values()) for a, v in by_team.items()}, "stale": bool(stale),
            "report_week": report_week, "target_week": target_week,
            "espn_available": espn_available, "conflicts": conflicts}


def latest_report_week(df) -> int | None:
    """Latest week with a real designation in the nflverse injuries frame."""
    if df is None or len(df) == 0 or "report_status" not in df.columns:
        return None
    rep = df[df["report_status"].isin(["Out", "Doubtful", "Questionable"])]
    return int(rep["week"].max()) if len(rep) else None


def current_report(now: datetime, target_week, name_to_abbr: dict[str, str]) -> dict:
    """IO: nflverse season report + ESPN live injuries, merged."""
    import nfl_data_py as nfl

    from sportsmodel.nfl import espn
    from sportsmodel.nfl.injuries_nflverse import nfl_season, parse_injuries
    from sportsmodel.nfl.nflverse import import_by_season

    df = import_by_season(nfl.import_injuries, [nfl_season(now)], "injuries", required=False)
    week = latest_report_week(df)
    by_abbr = parse_injuries(df, week=week) if week is not None else {}
    try:
        espn_rows, ok = espn.fetch_injuries(), True
    except Exception as exc:  # noqa: BLE001 -- ESPN down: fall back to nflverse, flagged
        print(f"WARN espn injuries unavailable ({exc!r})")
        espn_rows, ok = [], False
    return merge_report(by_abbr, week, target_week, espn_rows, name_to_abbr, espn_available=ok)
```

(The conflict rule records a player once when the Out/Doubtful-ness differs between nflverse and ESPN, including a stale nflverse Out cleared by ESPN; a player neither source designates is not a conflict. Match the tests exactly.)

- [ ] **Step 4:** tests → PASS; full suite. **Step 5: Commit** `feat(injuries): shared NFL injury report, ESPN-verified and stale-safe`.

### Task 2: Use it in the sim and the desk; desk freshness block + prompt

**Files:** Modify `scripts/generate_sim_nfl.py`, `scripts/desk_inputs.py`, `scripts/synthesize_desk_picks.py`; Tests `tests/sim/nfl/test_generate_sim_nfl.py`, `tests/cfb/test_desk_inputs.py`, prompt test file.

- [ ] **Sim:** replace the `current_injuries` + `_merge_espn_injuries` block in `main()` with `report = current_report(now, target_week, name_to_abbr)` where `target_week` = the week being simmed (use `sportsmodel.nfl.espn.resolve_target_week()["week"]`, falling back to None on error → treated as not stale) and `name_to_abbr` = inverse of the crosswalk (display name → abbr). Build `out_names_by_team` / `q_names_by_team` from `report["by_team"]` with the existing `_out_names_by_team` / `_questionable_names_by_team` (statuses are now normalized Out/Doubtful/Questionable). Print `injuries: report_week=W target_week=T stale=S espn=OK conflicts=N`. Delete `_espn_injury_names` and `_merge_espn_injuries` and move their test intent into Task 1's module tests (delete their tests here; keep `_out_names_by_team`/`_questionable_names_by_team` tests).
- [ ] **Desk:** `_nfl_injuries_by_name` returns `(by_name, meta)` using `current_report` (target week as above; name_to_abbr from the crosswalk); `_cfb_injuries_by_name` returns `(by_name, {"source": "sportsdata", "stale": False, "conflicts": []})`; `main()` unpacks both. `build_bundle(..., injury_meta: dict | None = None)` adds per game `news.injury_report = {"source": meta.source ("nflverse+espn" for NFL), "stale": meta.stale, "report_week", "target_week", "conflicts": [c for c in meta.conflicts if c.team (mapped to display name) is this game's home or away team]}` (None when meta is None). Test: conflicts filtered to the game's teams; default None.
- [ ] **Prompt** (`SYSTEM_PROMPT`, keep all other text verbatim): add after the injuries-related sport note a bullet:

```
- INJURY FRESHNESS (`news.injury_report`): injury statuses are the CURRENT
  report, verified against ESPN's live list. Never state a player is out,
  doubtful or questionable unless he appears in `news.injuries` with that
  status. If `stale` is true, the official weekly report for this week hasn't
  posted yet and statuses come from ESPN only -- say so if an injury drives
  your call. If `conflicts` lists a player, the sources disagreed; the
  bundle's status is the resolved one -- name the conflict if you rely on it.
```

Extend the prompt test to assert the phrase `INJURY FRESHNESS` and `Never state a player is out`.
- [ ] Full suite green. **Commit** `fix(desk,sim): one ESPN-verified injury report; desk sees freshness + conflicts`.

### Task 3: Injury snapshots + watch script

**Files:** Create `db/migration_injury_snapshots.sql`, `src/sportsmodel/serving/injury_watch.py`, `scripts/injury_watch.py`; Modify `src/sportsmodel/db.py`; Tests `tests/serving/test_injury_watch.py`, `tests/test_db_injury_snapshots.py`, `tests/scripts/test_injury_watch_script.py`.

- [ ] Migration:

```sql
-- Injury snapshots for the day-before injury watch. Internal (no anon access).
CREATE TABLE IF NOT EXISTS injury_snapshots (
    sport        TEXT NOT NULL,
    game_pk      BIGINT NOT NULL,
    fingerprint  TEXT NOT NULL,
    statuses     JSONB NOT NULL,       -- [{team, player, status}] Out/Doubtful only
    captured_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, game_pk)
);
ALTER TABLE injury_snapshots ENABLE ROW LEVEL SECURITY;
```

- [ ] Pure `serving/injury_watch.py`: `game_statuses(home, away, injuries_by_team) -> list[{team, player, status}]` (Out/Doubtful only via `normalize_status`, sorted by team, player); `fingerprint(statuses) -> str` (sha1 of `team|player|status` lines); `diff_statuses(old, new) -> list[str]` human lines like `"Atlanta Falcons: Michael Penix Jr. Out -> (none)"`; `changed_games(current: dict[int, str], stored: dict[int, str]) -> list[int]` (missing stored = changed). Tests: Questionable-only change → same fingerprint; Out→Active → different; order-insensitive; missing stored → changed.
- [ ] `db.py`: `load_injury_snapshots(sport, game_pks) -> dict[int, dict]` and `upsert_injury_snapshots(rows) -> int` (ON CONFLICT (sport, game_pk) DO UPDATE, captured_at = now(), statuses JSON-encoded). FakeConn tests.
- [ ] Script `scripts/injury_watch.py --sport {nfl,cfb} (--check | --record)`:
  - Games: `predictions_current` for the sport with `commence_time > now()` and, for `--check`, `commence_time <= now() + 30h`; for `--record`, `<= now() + 7 days`. Team names = home_team_name/away_team_name.
  - Injuries by team display name: NFL → `injury_report.current_report` mapped abbr→name via the crosswalk (target week via `espn.resolve_target_week`); CFB → load `scripts/desk_inputs.py` via importlib and call its `_cfb_injuries_by_name(adapter, api_key, crosswalk, now)` + `_rekey_by_espn_name(..., espn_names)` exactly as desk_inputs.main does (needs SPORTSDATA_API_KEY; missing key → print warning and exit 0 with changed=false).
  - `--check`: compute fingerprints, load stored, `changed_games`; print each changed game's matchup + `diff_statuses`; append `changed=true|false` to `$GITHUB_OUTPUT` when set. Exit 0.
  - `--record`: upsert fingerprints/statuses for the games.
  - Keep IO thin; unit-test a pure `plan_check(games, injuries_by_team, stored) -> (changed_pks, lines)` helper.
- [ ] Full suite green. **Commit** `feat(injuries): injury snapshots + watch script`.

### Task 4: Workflows

**Files:** Create `.github/workflows/injury-watch.yml`; Modify `.github/workflows/desk-auto-nfl.yml`, `.github/workflows/desk-auto-cfb.yml`.

- [ ] `injury-watch.yml`: `on: schedule: - cron: "0 15,22 * * *"` + `workflow_dispatch`; `permissions: contents: read`; two jobs `nfl` and `cfb`, each with `concurrency: {group: injury-pipeline-<sport>, cancel-in-progress: false}`, `timeout-minutes: 40`, steps: checkout, setup-uv (`astral-sh/setup-uv@v10.0.1`), `uv sync`, then:
  1. `id: check` — `uv run python scripts/injury_watch.py --sport <s> --check` (env DATABASE_URL, SPORTSDATA_API_KEY).
  2–N (each `if: steps.check.outputs.changed == 'true'`): model `scripts/generate_<s>.py` (env DATABASE_URL, `SM_DATA_DIR: ${{ github.workspace }}/data`); NFL only: `scripts/generate_sim_nfl.py` (same env); `desk_inputs.py --sport <s> --out desk_bundle.json --days-ahead 7` (DATABASE_URL, SPORTSDATA_API_KEY); `synthesize_desk_picks.py --sport <s> --bundle desk_bundle.json --out picks.json` (ANTHROPIC_API_KEY, `DESK_SYNTH_MODEL: ${{ vars.DESK_SYNTH_MODEL }}`); `write_desk_picks.py --in picks.json`; `build_ev_board.py --sport <s>`; `build_best_parlays.py`; `injury_watch.py --sport <s> --record`.
- [ ] `desk-auto-nfl.yml` / `desk-auto-cfb.yml`: append a final step `Record injury snapshot` → `uv run python scripts/injury_watch.py --sport "$SPORT" --record` (DATABASE_URL, SPORTSDATA_API_KEY).
- [ ] Validate all three YAML files parse (`uv run --with pyyaml python -c ...`). Commit `feat(injuries): twice-daily injury watch re-runs model/sim/desk on Out/Doubtful changes`.
