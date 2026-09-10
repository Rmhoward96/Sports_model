# NFL Decision Desk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the CFB decision desk to NFL — an NFL SportsDataIO injuries adapter, an NFL team crosswalk asset, a `--sport`-parameterized `desk_inputs.py`, the workflow input, and the runbook — reusing the existing (already sport-generic) schema, writer, grader, and front-end component.

**Architecture:** Add `src/sportsmodel/nfl/sportsdata.py` (mirror of the CFB adapter, NFL base + endpoint paths) and `assets/nfl/nfl_teams.json` (abbreviation → ESPN displayName crosswalk, the NFL analog of `fbs_teams.json`). Parameterize `scripts/desk_inputs.py` by `--sport {cfb,nfl}` via a small `_sport_config` block; everything else (`build_bundle`, `compute_recent_form`, `_rekey_by_espn_name`, the writer, the grader) is reused unchanged.

**Tech Stack:** Python 3.12, httpx, tenacity, pandas, pytest; uv for env/test.

**Spec:** `docs/superpowers/specs/2026-09-07-nfl-decision-desk-design.md`

## Global Constraints

- **No schema/migration change.** `desk_picks`/`desk_pick_results`/`desk_current`/`desk_record` are already `(sport, …)`-keyed. Do not touch `db/migration_decision_desk.sql`.
- **No change to the writer or grader.** `scripts/write_desk_picks.py` is sport-agnostic; `scripts/grade_desk_picks.py` already loops `{"nfl": nfl_espn, "cfb": cfb_espn}`. Leave both alone.
- **Back-compat:** `scripts/desk_inputs.py` must still work with no `--sport` arg (default `cfb`) — the existing workflow and runbook invocations must not break.
- **Line/identifier conventions (unchanged from CFB):** market lines are ESPN pickcenter (sportsbook convention, home favored → negative). `predictions_current.home_team_name`/`away_team_name` are ESPN displayNames for both sports. NFL `schedules.parquet` keys teams by **abbreviation**; `game_type == "REG"` for regular season.
- **Secrets never enter code or the session:** `SPORTSDATA_API_KEY`/`DATABASE_URL` are GitHub Actions secrets, read only in `main()` from the environment.
- Run tests with `PYTHONPATH=src uv run pytest`.
- The NFL injuries endpoint is under `/projections/json/`, NOT `/scores/json/` like CFB. Do not guess paths — use exactly those in this plan (verified against the NFL OpenAPI swagger).

---

### Task 1: NFL SportsDataIO adapter

**Files:**
- Create: `src/sportsmodel/nfl/sportsdata.py`
- Modify: `src/sportsmodel/cfb/sportsdata.py` (add two path constants only)
- Create: `tests/nfl/test_sportsdata_parsers.py`
- Create: `tests/fixtures/nfl/sportsdata_injuries.json`
- Create: `tests/fixtures/nfl/sportsdata_teams.json`

**Interfaces:**
- Produces: `sportsmodel.nfl.sportsdata` with `_BASE`, `INJURED_PLAYERS_PATH = "/projections/json/InjuredPlayers"`, `TEAMS_PATH = "/scores/json/Teams"`, `_get(path, api_key, params=None)`, `parse_injuries(payload) -> dict[str, list[dict]]` (keyed by team abbreviation; each item `{player, position, status, note}`), `parse_teams(payload) -> dict[str, str]` (Key → FullName).
- Also adds `INJURED_PLAYERS_PATH = "/scores/json/InjuredPlayers"` and `TEAMS_PATH = "/scores/json/Teams"` to `sportsmodel.cfb.sportsdata` (consumed by Task 3 so `desk_inputs.py` is adapter-generic).

- [ ] **Step 1: Write the adapter**

Create `src/sportsmodel/nfl/sportsdata.py` with exactly this content:

```python
"""SportsDataIO adapter: live NFL injuries (+ team crosswalk) for the news
agent. NFL sibling of cfb/sportsdata.py.

Field names mirror SportsDataIO's REAL NFL schemas, verified against their
published OpenAPI swagger:

  - Injuries: `GET /v3/nfl/projections/json/InjuredPlayers` (note the
    `/projections/` base -- the NFL injuries feed is NOT under `/scores/`
    like CFB). Returns a `Player[]` array with the SAME relevant fields the
    CFB `Player` carries: `FirstName`, `LastName` (no single "Name" field),
    `Team` (the team ABBREVIATION/Key, e.g. "PHI"), `TeamID`, `Position`,
    `InjuryStatus` (Probable/Questionable/Doubtful/Out), `InjuryBodyPart`,
    `InjuryNotes` -- all nullable in practice.

  - Teams (abbreviation -> full name crosswalk): `GET
    /v3/nfl/scores/json/Teams`. Returns a `Team[]` array: `Key`
    (abbreviation, e.g. "PHI"), `FullName` (e.g. "Philadelphia Eagles"),
    `City`, `Name` (mascot). NOTE: unlike the CFB `Team`, there is NO
    `School` field -- use `FullName`, which equals ESPN's NFL displayName,
    so desk_inputs.py's `_rekey_by_espn_name` prefix match joins injuries
    onto games with no new logic.

parse_injuries is deliberately a near-duplicate of the CFB parser rather
than a shared helper: the repo keeps a full per-sport module (cfb/espn.py
and nfl/espn.py already duplicate substantially), and isolating the sports
is preferred here over a shared abstraction.

`_get` takes the API key as a parameter (main()'s job to read
SPORTSDATA_API_KEY and fail fast) and uses SportsDataIO's subscription-key
header auth, mirroring cfb.sportsdata._get exactly.
"""
from __future__ import annotations

from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

_BASE = "https://api.sportsdata.io/v3/nfl"

# Endpoint paths (see module docstring). Exposed as constants so
# scripts/desk_inputs.py can stay adapter-generic across sports. NOTE the
# NFL injuries path is under /projections/, unlike CFB's /scores/.
INJURED_PLAYERS_PATH = "/projections/json/InjuredPlayers"
TEAMS_PATH = "/scores/json/Teams"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8))
def _get(path: str, api_key: str, params: dict | None = None) -> Any:
    """GET {_BASE}{path} with SportsDataIO subscription-key auth and return
    parsed JSON, retrying transient failures (tenacity: 3 attempts,
    exponential backoff -- same policy as cfb.sportsdata._get)."""
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    r = httpx.get(f"{_BASE}{path}", params=params, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()


def parse_injuries(payload) -> dict[str, list[dict]]:
    """Injury report grouped by team ABBREVIATION, from SportsDataIO's NFL
    `InjuredPlayers` endpoint (list of `Player` objects). Same shape and
    field names as the CFB parser.

    Rows with a null `Team` are skipped. Per player:
      - "player": "{FirstName} {LastName}", stripped (either half may be null).
      - "position": Position
      - "status": InjuryStatus
      - "note": InjuryBodyPart and InjuryNotes joined with " - ", skipping
        whichever is null, or None when both are null.

    Keyed by SportsDataIO's team ABBREVIATION (e.g. "PHI") -- callers that
    need ESPN display names first rekey through `parse_teams`'s
    abbreviation -> FullName map (see scripts/desk_inputs.py)."""
    out: dict[str, list[dict]] = {}
    for row in payload:
        team = row.get("Team")
        if team is None:
            continue
        first = row.get("FirstName") or ""
        last = row.get("LastName") or ""
        player = f"{first} {last}".strip()
        body_part, notes = row.get("InjuryBodyPart"), row.get("InjuryNotes")
        parts = [p for p in (body_part, notes) if p is not None]
        note = " - ".join(parts) if parts else None
        out.setdefault(team, []).append({
            "player": player,
            "position": row.get("Position"),
            "status": row.get("InjuryStatus"),
            "note": note,
        })
    return out


def parse_teams(payload) -> dict[str, str]:
    """Abbreviation ({Key}) -> full name ({FullName}) crosswalk, from
    SportsDataIO's NFL `Teams` endpoint (list of `Team` objects). NFL's
    `Team` has NO `School` field (unlike CFB) -- `FullName` (e.g.
    "Philadelphia Eagles") is the ESPN-displayName-equivalent used to rekey
    abbreviation-keyed injuries onto ESPN display names (see
    scripts/desk_inputs.py's `_rekey_by_espn_name`).

    Rows with a null `Key` or `FullName` are skipped."""
    out: dict[str, str] = {}
    for row in payload:
        key, full = row.get("Key"), row.get("FullName")
        if key is None or full is None:
            continue
        out[key] = full
    return out
```

- [ ] **Step 2: Add matching path constants to the CFB adapter**

In `src/sportsmodel/cfb/sportsdata.py`, immediately after the `_BASE = "https://api.sportsdata.io/v3/cfb"` line, add:

```python

# Endpoint paths (see module docstring). Exposed as constants so
# scripts/desk_inputs.py can stay adapter-generic across sports.
INJURED_PLAYERS_PATH = "/scores/json/InjuredPlayers"
TEAMS_PATH = "/scores/json/Teams"
```

Do not change anything else in the CFB adapter.

- [ ] **Step 3: Write the test fixtures**

Create `tests/fixtures/nfl/sportsdata_injuries.json`:

```json
[
  {
    "Team": "PHI",
    "TeamID": 22,
    "FirstName": "John",
    "LastName": "Smith",
    "Position": "QB",
    "InjuryStatus": "Questionable",
    "InjuryBodyPart": "Shoulder",
    "InjuryNotes": "Limited in practice",
    "InjuryStartDate": "2026-09-03T00:00:00"
  },
  {
    "Team": "PHI",
    "TeamID": 22,
    "FirstName": "Mike",
    "LastName": "Jones",
    "Position": "RB",
    "InjuryStatus": "Out",
    "InjuryBodyPart": null,
    "InjuryNotes": null,
    "InjuryStartDate": null
  },
  {
    "Team": "DAL",
    "TeamID": 9,
    "FirstName": "Dan",
    "LastName": "Lee",
    "Position": "WR",
    "InjuryStatus": "Probable",
    "InjuryBodyPart": "Ankle",
    "InjuryNotes": null,
    "InjuryStartDate": "2026-09-04T00:00:00"
  },
  {
    "Team": null,
    "TeamID": null,
    "FirstName": "Unknown",
    "LastName": "Player",
    "Position": "OL",
    "InjuryStatus": "Out",
    "InjuryBodyPart": "Knee",
    "InjuryNotes": "Season-ending",
    "InjuryStartDate": null
  }
]
```

Create `tests/fixtures/nfl/sportsdata_teams.json`:

```json
[
  {
    "Key": "PHI",
    "TeamID": 22,
    "City": "Philadelphia",
    "Name": "Eagles",
    "FullName": "Philadelphia Eagles"
  },
  {
    "Key": "DAL",
    "TeamID": 9,
    "City": "Dallas",
    "Name": "Cowboys",
    "FullName": "Dallas Cowboys"
  },
  {
    "Key": null,
    "TeamID": 999,
    "City": "Unknown",
    "Name": "Unknown",
    "FullName": "Unknown"
  },
  {
    "Key": "XXX",
    "TeamID": 998,
    "City": null,
    "Name": null,
    "FullName": null
  }
]
```

- [ ] **Step 4: Write the parser tests**

Create `tests/nfl/test_sportsdata_parsers.py`:

```python
"""Pure-parser tests for the NFL SportsDataIO adapter (injuries + team
crosswalk). No network; fixtures under tests/fixtures/nfl/ match
SportsDataIO's REAL NFL schema field names (Player[]: Team/FirstName/
LastName/Position/InjuryStatus/InjuryBodyPart/InjuryNotes; Team[]: Key/
FullName/City/Name -- no School field, unlike CFB)."""
import json
import pathlib

from sportsmodel.nfl import sportsdata

FIX = pathlib.Path(__file__).parent.parent / "fixtures" / "nfl"


def _load(name: str):
    return json.loads((FIX / name).read_text())


def test_endpoint_path_constants():
    # NFL injuries live under /projections/, not /scores/ like CFB.
    assert sportsdata.INJURED_PLAYERS_PATH == "/projections/json/InjuredPlayers"
    assert sportsdata.TEAMS_PATH == "/scores/json/Teams"


def test_parse_injuries_groups_by_team_abbreviation_with_status():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    assert "PHI" in out
    qb = next(p for p in out["PHI"] if p["position"] == "QB")
    assert qb["player"] == "John Smith"
    assert qb["status"] == "Questionable"
    assert qb["note"] == "Shoulder - Limited in practice"


def test_parse_injuries_note_none_when_bodypart_and_notes_null():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    rb = next(p for p in out["PHI"] if p["position"] == "RB")
    assert rb["note"] is None
    assert rb["status"] == "Out"


def test_parse_injuries_skips_rows_with_null_team():
    out = sportsdata.parse_injuries(_load("sportsdata_injuries.json"))
    assert None not in out
    assert set(out) == {"PHI", "DAL"}


def test_parse_teams_maps_key_to_fullname_skipping_nulls():
    out = sportsdata.parse_teams(_load("sportsdata_teams.json"))
    assert out == {"PHI": "Philadelphia Eagles", "DAL": "Dallas Cowboys"}
```

- [ ] **Step 5: Ensure the test package dir exists**

If `tests/nfl/__init__.py` does not exist and `tests/cfb/` has one, create an empty `tests/nfl/__init__.py` to match the layout. (Check `tests/cfb/`; mirror whatever convention it uses.)

- [ ] **Step 6: Run the tests**

Run: `PYTHONPATH=src uv run pytest tests/nfl/test_sportsdata_parsers.py -v`
Expected: PASS (5 tests).

- [ ] **Step 7: Commit**

```bash
git add src/sportsmodel/nfl/sportsdata.py src/sportsmodel/cfb/sportsdata.py tests/nfl/ tests/fixtures/nfl/
git commit -m "feat(nfl-desk): NFL SportsDataIO injuries adapter + path constants"
```

---

### Task 2: NFL team crosswalk asset + builder

**Files:**
- Create: `assets/nfl/nfl_teams.json`
- Create: `scripts/build_nfl_teams.py`
- Create: `tests/nfl/test_nfl_teams_asset.py`

**Interfaces:**
- Produces: `assets/nfl/nfl_teams.json` — a JSON object `{abbreviation -> ESPN displayName}` for all 32 teams, consumed by Task 3's NFL recent-form join (the NFL analog of `assets/cfb/fbs_teams.json`).

- [ ] **Step 1: Write the crosswalk asset**

Create `assets/nfl/nfl_teams.json` with exactly this content (authoritative ESPN NFL displayNames, keyed by the normalized franchise abbreviation used in `assets/nfl/schedules.parquet`):

```json
{
  "ARI": "Arizona Cardinals",
  "ATL": "Atlanta Falcons",
  "BAL": "Baltimore Ravens",
  "BUF": "Buffalo Bills",
  "CAR": "Carolina Panthers",
  "CHI": "Chicago Bears",
  "CIN": "Cincinnati Bengals",
  "CLE": "Cleveland Browns",
  "DAL": "Dallas Cowboys",
  "DEN": "Denver Broncos",
  "DET": "Detroit Lions",
  "GB": "Green Bay Packers",
  "HOU": "Houston Texans",
  "IND": "Indianapolis Colts",
  "JAX": "Jacksonville Jaguars",
  "KC": "Kansas City Chiefs",
  "LA": "Los Angeles Rams",
  "LAC": "Los Angeles Chargers",
  "LV": "Las Vegas Raiders",
  "MIA": "Miami Dolphins",
  "MIN": "Minnesota Vikings",
  "NE": "New England Patriots",
  "NO": "New Orleans Saints",
  "NYG": "New York Giants",
  "NYJ": "New York Jets",
  "PHI": "Philadelphia Eagles",
  "PIT": "Pittsburgh Steelers",
  "SEA": "Seattle Seahawks",
  "SF": "San Francisco 49ers",
  "TB": "Tampa Bay Buccaneers",
  "TEN": "Tennessee Titans",
  "WAS": "Washington Commanders"
}
```

- [ ] **Step 2: Write the one-shot builder**

Create `scripts/build_nfl_teams.py` (regenerates the asset above from ESPN, so it can be refreshed if ESPN ever renames a team):

```python
"""Regenerate assets/nfl/nfl_teams.json: {abbreviation -> ESPN displayName}.

The NFL analog of assets/cfb/fbs_teams.json. desk_inputs.py uses it to
translate assets/nfl/schedules.parquet's abbreviation-keyed teams onto the
ESPN displayNames that predictions_current (and thus the desk bundle) use,
so recent-form joins line up. Keys are normalized via nfl.teams.normalize_team
so historical/alternate codes collapse onto the current franchise code.

Run on demand (not in CI):
    PYTHONPATH=src uv run python scripts/build_nfl_teams.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import httpx

from sportsmodel.nfl.teams import normalize_team

URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
OUT = Path(__file__).resolve().parents[1] / "assets" / "nfl" / "nfl_teams.json"


def main() -> None:
    data = httpx.get(URL, timeout=20).json()
    teams = data["sports"][0]["leagues"][0]["teams"]
    out = {}
    for t in teams:
        tm = t["team"]
        out[normalize_team(tm["abbreviation"])] = tm["displayName"]
    OUT.write_text(json.dumps(dict(sorted(out.items())), indent=2) + "\n")
    print(f"Wrote {OUT} ({len(out)} teams)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Write the asset test**

Create `tests/nfl/test_nfl_teams_asset.py`:

```python
"""The committed NFL team crosswalk asset is well-formed and complete."""
import json
import pathlib

from sportsmodel.nfl.teams import TEAMS

ASSET = pathlib.Path(__file__).parents[2] / "assets" / "nfl" / "nfl_teams.json"


def test_asset_covers_all_32_current_franchises():
    crosswalk = json.loads(ASSET.read_text())
    assert set(crosswalk) == set(TEAMS)  # every current franchise, no extras
    assert len(crosswalk) == 32


def test_known_mappings():
    crosswalk = json.loads(ASSET.read_text())
    assert crosswalk["PHI"] == "Philadelphia Eagles"
    assert crosswalk["WAS"] == "Washington Commanders"
    assert crosswalk["LA"] == "Los Angeles Rams"
    assert all(isinstance(v, str) and v for v in crosswalk.values())
```

- [ ] **Step 4: Run the test**

Run: `PYTHONPATH=src uv run pytest tests/nfl/test_nfl_teams_asset.py -v`
Expected: PASS (2 tests). (If it fails on the 32-set check, the asset in Step 1 was mistyped — fix the asset, not the test.)

- [ ] **Step 5: Commit**

```bash
git add assets/nfl/nfl_teams.json scripts/build_nfl_teams.py tests/nfl/test_nfl_teams_asset.py
git commit -m "data(nfl-desk): NFL abbreviation->displayName crosswalk + builder"
```

---

### Task 3: Parameterize `desk_inputs.py` by `--sport`

**Files:**
- Modify: `scripts/desk_inputs.py`
- Create: `tests/nfl/test_desk_inputs_sport_config.py`

**Interfaces:**
- Consumes: `nfl.sportsdata` / `cfb.sportsdata` (`INJURED_PLAYERS_PATH`, `TEAMS_PATH`, `_get`, `parse_injuries`, `parse_teams`) from Task 1; `assets/nfl/nfl_teams.json` from Task 2.
- Produces: `desk_inputs._sport_config(sport) -> dict` with keys `adapter` (module), `schedules_path` (Path), `crosswalk_path` (Path), `out_default` (Path); a `--sport {cfb,nfl}` CLI arg (default `cfb`).

**Background for the implementer:** `build_bundle`, `compute_recent_form`, `_parse_iso`, `_norm_name`, and `_rekey_by_espn_name` are already pure and sport-generic — **do not change them**. The injuries rekey chain (abbrev → SDIO name → ESPN displayName via `_rekey_by_espn_name`) is identical for both sports once `parse_teams` returns `{abbrev -> name}` (CFB: School; NFL: FullName). The only sport-specific inputs are: which adapter, the two endpoint paths (now constants on the adapter), the DB `sport` filter, the schedules parquet path, the schedule-team→displayName crosswalk file, and the default output path.

The current CFB name-mapping in `main()` reads `assets/cfb/fbs_teams.json` (an `{id -> name}` map) and applies it to `schedule["home_team"]/["away_team"]` (which are string ids). For NFL the crosswalk is `assets/nfl/nfl_teams.json` (`{abbrev -> name}`) applied to the schedule's abbreviation columns — the **same operation**, different file. So the crosswalk load + `.map(...)` stays as-is; only the file path becomes sport-dependent. A team key absent from the crosswalk falls through unchanged (`crosswalk.get(t, t)`), so a stale/renamed franchise degrades to no-form rather than crashing.

- [ ] **Step 1: Write the failing test**

Create `tests/nfl/test_desk_inputs_sport_config.py`:

```python
"""_sport_config selects the right adapter, paths, and crosswalk per sport."""
import importlib.util
import pathlib

_p = pathlib.Path(__file__).parents[2] / "scripts" / "desk_inputs.py"
_spec = importlib.util.spec_from_file_location("desk_inputs", _p)
desk_inputs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(desk_inputs)


def test_cfb_config():
    cfg = desk_inputs._sport_config("cfb")
    assert cfg["adapter"].__name__ == "sportsmodel.cfb.sportsdata"
    assert cfg["adapter"].INJURED_PLAYERS_PATH == "/scores/json/InjuredPlayers"
    assert cfg["schedules_path"].as_posix().endswith("assets/cfb/schedules.parquet")
    assert cfg["crosswalk_path"].as_posix().endswith("assets/cfb/fbs_teams.json")
    assert cfg["out_default"].as_posix().endswith("cfb/desk_bundle.json")


def test_nfl_config():
    cfg = desk_inputs._sport_config("nfl")
    assert cfg["adapter"].__name__ == "sportsmodel.nfl.sportsdata"
    assert cfg["adapter"].INJURED_PLAYERS_PATH == "/projections/json/InjuredPlayers"
    assert cfg["schedules_path"].as_posix().endswith("assets/nfl/schedules.parquet")
    assert cfg["crosswalk_path"].as_posix().endswith("assets/nfl/nfl_teams.json")
    assert cfg["out_default"].as_posix().endswith("nfl/desk_bundle.json")


def test_unknown_sport_raises():
    import pytest
    with pytest.raises((KeyError, ValueError, SystemExit)):
        desk_inputs._sport_config("mlb")
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `PYTHONPATH=src uv run pytest tests/nfl/test_desk_inputs_sport_config.py -v`
Expected: FAIL (`AttributeError: module 'desk_inputs' has no attribute '_sport_config'`).

- [ ] **Step 3: Add `_sport_config` and the adapter import**

In `scripts/desk_inputs.py`, change the CFB adapter import to import both adapters:

```python
from sportsmodel.cfb import sportsdata as cfb_sportsdata
from sportsmodel.nfl import sportsdata as nfl_sportsdata
```

(Replace the existing `from sportsmodel.cfb import sportsdata` line. Update the two `sportsdata._get(...)` / `sportsdata.parse_*` call sites in `main()` in Step 5 to use the selected adapter.)

Add this function above `main()` (after `_rekey_by_espn_name`):

```python
def _sport_config(sport: str) -> dict:
    """Per-sport paths/adapter for main(). Everything else in this module is
    sport-generic. Raises SystemExit on an unsupported sport."""
    root = config.PROJECT_ROOT
    configs = {
        "cfb": {
            "adapter": cfb_sportsdata,
            "schedules_path": root / "assets" / "cfb" / "schedules.parquet",
            "crosswalk_path": root / "assets" / "cfb" / "fbs_teams.json",
            "out_default": config.DATA_DIR / "cfb" / "desk_bundle.json",
        },
        "nfl": {
            "adapter": nfl_sportsdata,
            "schedules_path": root / "assets" / "nfl" / "schedules.parquet",
            "crosswalk_path": root / "assets" / "nfl" / "nfl_teams.json",
            "out_default": config.DATA_DIR / "nfl" / "desk_bundle.json",
        },
    }
    if sport not in configs:
        raise SystemExit(f"unsupported --sport {sport!r} (expected one of {sorted(configs)})")
    return configs[sport]
```

- [ ] **Step 4: Run the test to confirm it passes**

Run: `PYTHONPATH=src uv run pytest tests/nfl/test_desk_inputs_sport_config.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Thread `--sport` through `main()`**

Edit `main()` so it uses the sport config. Specifically:

1. Add the CLI arg (keep `--out` defaulting to the *sport's* default, so resolve sport first):

```python
    ap = argparse.ArgumentParser(description="Build the decision-desk input bundle.")
    ap.add_argument("--sport", choices=["cfb", "nfl"], default="cfb")
    ap.add_argument("--out", type=Path, default=None,
                     help="Output path (defaults to data/<sport>/desk_bundle.json).")
    ap.add_argument("--days-ahead", type=int, default=7,
                     help="Only include games within this many days of now (default 7).")
    args = ap.parse_args()

    cfg = _sport_config(args.sport)
    out_path = args.out if args.out is not None else cfg["out_default"]
    adapter = cfg["adapter"]
```

   (Remove the module-level `OUT_PATH` constant, or keep it only as the CFB default — but the `--out` default must now come from `cfg`. Do not leave a stale `default=OUT_PATH`.)

2. In the predictions query, replace the hard-coded `WHERE sport = 'cfb'` with a parameterized filter:

```python
        cur.execute("""
            SELECT game_pk, game_date, home_team_name, away_team_name,
                   home_win_prob, pred_home_score, pred_away_score,
                   commence_time, market_spread, market_total
            FROM predictions_current
            WHERE sport = %(sport)s
        """, {"sport": args.sport})
```

   Update the "No upcoming games" print to say `{args.sport}` instead of the literal "CFB".

3. Replace the schedule + crosswalk load to use `cfg`:

```python
    schedule = pd.read_parquet(cfg["schedules_path"])
    crosswalk_path = cfg["crosswalk_path"]
    crosswalk = json.loads(crosswalk_path.read_text()) if crosswalk_path.exists() else {}
    schedule_named = schedule.assign(
        home_team=schedule["home_team"].astype(str).map(lambda t: crosswalk.get(t, t)),
        away_team=schedule["away_team"].astype(str).map(lambda t: crosswalk.get(t, t)),
    )
    form_rows = compute_recent_form(schedule_named, set(espn_names))
```

   (The CFB `fbs_teams.json` is `{id -> name}` and NFL `nfl_teams.json` is `{abbrev -> name}`; both applied to the schedule's `home_team`/`away_team` string keys — same code, sport-specific file. Keep the explanatory comment updated to say the crosswalk maps the schedule's team key to the ESPN displayName for whichever sport.)

4. Replace the two SportsDataIO calls to use `adapter` and its path constants:

```python
        injuries_by_abbrev = adapter.parse_injuries(
            adapter._get(adapter.INJURED_PLAYERS_PATH, api_key)
        )
        teams_by_abbrev = adapter.parse_teams(
            adapter._get(adapter.TEAMS_PATH, api_key)
        )
```

   (The `injuries_by_school` / `_rekey_by_espn_name` lines below are unchanged — for NFL the "school" name is the FullName; the variable name can stay or be generalized to `injuries_by_name`, implementer's choice, but do not change the logic.)

5. Replace `args.out` with `out_path` in the final write block and the print.

- [ ] **Step 6: Run the full desk-inputs + adapter test suite**

Run: `PYTHONPATH=src uv run pytest tests/cfb/test_desk_inputs.py tests/nfl/ -v`
Expected: PASS (existing CFB build_bundle tests still green — proving back-compat — plus the new NFL tests).

- [ ] **Step 7: Byte-compile check (no live network/DB needed)**

Run: `PYTHONPATH=src uv run python -c "import importlib.util,pathlib; p=pathlib.Path('scripts/desk_inputs.py'); s=importlib.util.spec_from_file_location('di',p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('cfb',m._sport_config('cfb')['adapter'].__name__); print('nfl',m._sport_config('nfl')['adapter'].__name__)"`
Expected: prints `cfb sportsmodel.cfb.sportsdata` and `nfl sportsmodel.nfl.sportsdata`.

- [ ] **Step 8: Commit**

```bash
git add scripts/desk_inputs.py tests/nfl/test_desk_inputs_sport_config.py
git commit -m "feat(nfl-desk): parameterize desk_inputs by --sport (cfb default)"
```

---

### Task 4: Workflow — `sport` input on `desk-inputs.yml`

**Files:**
- Modify: `.github/workflows/desk-inputs.yml`

**Interfaces:**
- Consumes: `scripts/desk_inputs.py`'s `--sport` from Task 3.

- [ ] **Step 1: Add the `sport` input and pass `--sport`**

In `.github/workflows/desk-inputs.yml`, add a `sport` input under `workflow_dispatch.inputs` (place it before `days_ahead`):

```yaml
      sport:
        description: "Sport to build the bundle for"
        type: choice
        options: [cfb, nfl]
        default: cfb
```

And change the run step to pass it:

```yaml
        run: uv run python scripts/desk_inputs.py --sport "${{ github.event.inputs.sport || 'cfb' }}" --out desk_bundle.json --days-ahead "${{ github.event.inputs.days_ahead || '7' }}"
```

Update the workflow's top comment to say "CFB or NFL decision-desk input bundle" rather than "CFB".

- [ ] **Step 2: Validate the YAML parses**

Run: `uv run python -c "import yaml; yaml.safe_load(open('.github/workflows/desk-inputs.yml'))" ` (if PyYAML isn't a dep, use `python -c "import json,subprocess"` alternative: `uv run python -c "import pathlib; pathlib.Path('.github/workflows/desk-inputs.yml').read_text()"` and eyeball). Prefer the yaml parse if available.
Expected: no error.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/desk-inputs.yml
git commit -m "ci(nfl-desk): add sport input to desk-inputs workflow"
```

---

### Task 5: Runbook — generalize to both sports

**Files:**
- Modify: `docs/decision-desk-runbook.md`

- [ ] **Step 1: Generalize the runbook**

Update `docs/decision-desk-runbook.md` so it covers NFL as well as CFB. Concretely:

- Retitle the doc "Decision-desk runbook (CFB + NFL)" (or similar) and reword the intro from "CFB decision desk" to "the CFB or NFL decision desk".
- Prerequisites: note that `desk_inputs.py` takes `--sport {cfb,nfl}`; that the schedules + crosswalk assets are `assets/<sport>/schedules.parquet` and `assets/<sport>/{fbs_teams.json|nfl_teams.json}`; and that `predictions_current` must be populated for the chosen `sport` (`generate_cfb.py` / `generate_nfl.py`).
- Step 1 (bundle): show both invocations, e.g.
  `... scripts/desk_inputs.py --sport cfb --out data/cfb/desk_bundle.json`
  and `... scripts/desk_inputs.py --sport nfl --out data/nfl/desk_bundle.json`.
- Add a short note that the NFL injuries endpoint is `/projections/json/InjuredPlayers` (vs CFB's `/scores/json/InjuredPlayers`), injuries are keyed by team abbreviation and crosswalked to ESPN display names via SportsDataIO `FullName`, and that the data-quality caveat (free/trial SportsDataIO tiers return scrambled values — spot-check one known injury per slate) applies to both sports.
- Leave the picks-JSON contract section unchanged except to note it is sport-agnostic (`sport` field is `"cfb"` or `"nfl"`); `write_desk_picks.py` remains the authority.

Keep the edits tight — this is a procedure doc, not new code.

- [ ] **Step 2: Commit**

```bash
git add docs/decision-desk-runbook.md
git commit -m "docs(nfl-desk): generalize decision-desk runbook to CFB + NFL"
```

---

## Post-plan (controller-applied, outside the worktree)

These are **not** worktree subagent tasks — the controller does them after the branch is reviewed/merged:

1. **Front-end (`/Users/ryan/Desktop/CappingAlpha/app.js`):** in `buildLeague`, change `if (sport === "cfb")` to `if (sport === "cfb" || sport === "nfl")` so the desk section renders on the NFL page. Everything below it is already sport-generic (`deskCurrent`, `deskRecord`, `deskSection`, `logoImg`, `NFL_TEAM_ABBR`). User redeploys.
2. **No Supabase change** — schema is already sport-tagged; nothing for the user to run.
3. First NFL desk run is on-demand per the runbook (`--sport nfl`), producing NFL `desk_picks` graded forward by the already-both-sports grader.
