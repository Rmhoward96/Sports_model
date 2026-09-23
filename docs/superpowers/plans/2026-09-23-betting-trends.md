# Betting Trends Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture Action Network season betting records (NFL + CFB) and compute NFL situational ATS/O-U trends; show both on game pages and feed them to the Decision Desk's sports-analyst.

**Architecture:** Pure parsers/calculators (`action_network.parse_team_records`, `nfl/trends.py`) + thin daily scripts (`capture_team_records.py`, `build_nfl_trends.py`) in a new `build-trends.yml` workflow (before the desk). Two tables. `desk_inputs` adds a `trends` block; the desk prompt makes the analyst cite them as supporting evidence. CappingAlpha adds a "Trends" section.

**Tech Stack:** Python 3 / uv / pytest, Supabase Postgres, GitHub Actions, Apify actor `zen-studio/action-network-odds`, vanilla JS (`/Users/ryan/Desktop/CappingAlpha/app.js`, not a git repo).

**Spec:** `docs/superpowers/specs/2026-09-23-betting-trends-design.md`

## Global Constraints
- The user runs every Supabase migration; never apply DDL. Secrets live in GitHub Actions; never print them.
- Branch `feat/betting-trends`; merge/push only when the user asks.
- Situational trend window = current season + last 3; keep only n ≥ 5 games.
- Desk weighting: trends are SUPPORTING evidence only — never the sole basis for a spread lean; analyst note must cite every applicable trend for both teams.
- Game-page highlight rule: ≥70% one way with ≥8 decided games.
- Run tests with `uv run pytest`.

---

### Task 1: Action Network season records — parser, name matching, DB, capture script

**Files:**
- Modify: `src/sportsmodel/nfl/action_network.py` (add `parse_team_records`, extend `fetch_splits` usage only via `extra_input` — do not change its defaults)
- Create: `src/sportsmodel/serving/team_names.py` (`norm_team`, `match_team_names`)
- Modify: `src/sportsmodel/db.py` (`upsert_team_betting_records`)
- Create: `scripts/capture_team_records.py`
- Create: `db/migration_trends.sql` (BOTH tables for this plan — Task 3 relies on `nfl_game_trends` existing in it)
- Tests: `tests/nfl/test_action_network.py` (append), `tests/serving/test_team_names.py`, `tests/test_db_trends.py`

**Produces:** `parse_team_records(items) -> list[dict]` rows `{sport ("nfl"|"cfb"), season, an_team_name, abbr, records}`; `match_team_names(an_names: list[str], ours: list[str]) -> dict[str, str]` (AN name → our name, only matches); `upsert_team_betting_records(rows) -> int`.

- [ ] **Step 1: Migration** `db/migration_trends.sql`:

```sql
-- Betting trends: Action Network season betting records (NFL + CFB) and our
-- computed NFL situational trends. Idempotent. Run in the Supabase SQL Editor.
CREATE TABLE IF NOT EXISTS team_betting_records (
    sport          TEXT NOT NULL,          -- nfl | cfb
    season         INTEGER NOT NULL,
    team_name      TEXT NOT NULL,          -- our display name (AN name if unmatched)
    an_team_name   TEXT,
    abbr           TEXT,
    records        JSONB NOT NULL,         -- {category: {w,l,p,o,u}}
    captured_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (sport, season, team_name)
);

CREATE TABLE IF NOT EXISTS nfl_game_trends (
    game_pk        BIGINT NOT NULL,
    team_name      TEXT NOT NULL,
    situation      TEXT NOT NULL,          -- home|road|favorite|underdog|off_road|off_home|off_win|off_loss|off_bye|division|primetime
    label          TEXT,                   -- "off a road game"
    ats_w INTEGER, ats_l INTEGER, ats_p INTEGER,
    ou_o  INTEGER, ou_u  INTEGER, ou_p  INTEGER,
    n              INTEGER,
    since_season   INTEGER,
    computed_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, team_name, situation)
);

ALTER TABLE team_betting_records ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read team_betting_records" ON team_betting_records;
CREATE POLICY "public read team_betting_records" ON team_betting_records FOR SELECT USING (true);
ALTER TABLE nfl_game_trends ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_game_trends" ON nfl_game_trends;
CREATE POLICY "public read nfl_game_trends" ON nfl_game_trends FOR SELECT USING (true);
GRANT SELECT ON team_betting_records, nfl_game_trends TO anon, authenticated;
```

- [ ] **Step 2: Failing tests.** `tests/nfl/test_action_network.py` (append):

```python
from sportsmodel.nfl.action_network import parse_team_records

_STANDINGS = {"recordType": "standings", "league": "nfl", "season": 2026, "standings": [
    {"team": {"name": "Buffalo Bills", "abbreviation": "BUF"}, "records": [
        {"category": "ats", "wins": 2, "losses": 1, "ties": None, "draws": 1, "overs": None, "unders": None},
        {"category": "over_under_road", "wins": None, "losses": None, "overs": 3, "unders": 1, "draws": 0},
        {"category": "units", "wins": 1.2, "losses": 0, "draws": 0},
        {"category": "home", "wins": 1, "losses": 0, "ties": 0, "draws": None}]},
    {"team": {"name": None}, "records": []}]}


def test_parse_team_records_flattens_categories():
    rows = parse_team_records([{"eventType": "game"}, _STANDINGS])
    assert len(rows) == 1
    r = rows[0]
    assert (r["sport"], r["season"], r["an_team_name"], r["abbr"]) == ("nfl", 2026, "Buffalo Bills", "BUF")
    assert r["records"]["ats"] == {"w": 2, "l": 1, "p": 1, "o": None, "u": None}
    assert r["records"]["over_under_road"] == {"w": None, "l": None, "p": 0, "o": 3, "u": 1}
    assert r["records"]["units"]["w"] == 1.2 and r["records"]["home"]["p"] == 0


def test_parse_team_records_maps_ncaaf_to_cfb_and_ignores_games():
    item = {**_STANDINGS, "league": "ncaaf"}
    assert parse_team_records([item])[0]["sport"] == "cfb"
    assert parse_team_records([{"eventType": "game", "gameId": 1}]) == []
```

`tests/serving/test_team_names.py`:

```python
from sportsmodel.serving.team_names import match_team_names, norm_team


def test_norm_team_strips_accents_punct_case():
    assert norm_team("San José State Spartans") == "san jose state spartans"
    assert norm_team("Texas A&M Aggies") == "texas am aggies"


def test_match_team_names_only_exact_normalized_matches():
    m = match_team_names(["Buffalo Bills", "Hawai'i Rainbow Warriors", "Nowhere U"],
                         ["Buffalo Bills", "Hawaii Rainbow Warriors"])
    assert m == {"Buffalo Bills": "Buffalo Bills", "Hawai'i Rainbow Warriors": "Hawaii Rainbow Warriors"}
```

`tests/test_db_trends.py`: FakeConn/FakeCursor pattern from `tests/test_db_nfl_player_actuals.py`; assert `upsert_team_betting_records([row])` SQL contains `ON CONFLICT (sport, season, team_name) DO UPDATE` and `captured_at = now()`, `records` is JSON-encoded, returns 1; empty list returns 0 without touching the DB.

- [ ] **Step 3:** run → FAIL.

- [ ] **Step 4: Implement.** In `action_network.py`:

```python
_LEAGUE_TO_SPORT = {"nfl": "nfl", "ncaaf": "cfb"}


def parse_team_records(items) -> list[dict]:
    """PURE. The actor's standings item(s) (includeStandings: true) -> one row
    per team: {sport, season, an_team_name, abbr, records}, where records maps
    each category (ats, ats_road, over_under_home, units_fav, last_5, ...) to
    {w, l, p, o, u}: wins/losses, pushes (draws, else ties), overs/unders
    (over_under_* only; units_* carry the unit value in w). Game items and
    rows without a team name are skipped."""
    out: list[dict] = []
    for item in items or []:
        if item.get("recordType") != "standings":
            continue
        sport = _LEAGUE_TO_SPORT.get(str(item.get("league") or "").lower())
        if sport is None:
            continue
        for row in item.get("standings") or []:
            team = row.get("team") or {}
            name = team.get("name")
            if not name:
                continue
            records = {}
            for rec in row.get("records") or []:
                cat = rec.get("category")
                if not cat:
                    continue
                push = rec.get("draws") if rec.get("draws") is not None else rec.get("ties")
                records[cat] = {"w": rec.get("wins"), "l": rec.get("losses"), "p": push,
                                "o": rec.get("overs"), "u": rec.get("unders")}
            out.append({"sport": sport, "season": item.get("season"), "an_team_name": name,
                        "abbr": team.get("abbreviation"), "records": records})
    return out
```

(Duplicate categories — AN lists `home`/`road` twice for SU — keep the LAST occurrence; that is what the loop does.)

`src/sportsmodel/serving/team_names.py`:

```python
"""Team-name normalization shared by the trend captures (Action Network names ->
our ESPN display names). Same normalization as scripts/desk_inputs._norm_name."""
from __future__ import annotations

import unicodedata


def norm_team(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    return " ".join("".join(c for c in s if c.isalnum() or c.isspace()).split())


def match_team_names(an_names: list[str], ours: list[str]) -> dict[str, str]:
    """{AN name -> our name} for exact normalized-name matches only."""
    by_norm = {norm_team(n): n for n in ours}
    return {a: by_norm[norm_team(a)] for a in an_names if norm_team(a) in by_norm}
```

`db.py` (next to the other upserts, module-level `json`):

```python
_TEAM_RECORD_COLS = ["sport", "season", "team_name", "an_team_name", "abbr", "records"]


def upsert_team_betting_records(rows: list[dict]) -> int:
    """Upsert Action Network season betting records (db/migration_trends.sql).
    Idempotent on (sport, season, team_name); captured_at refreshed."""
    if not rows:
        return 0
    key = ("sport", "season", "team_name")
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in _TEAM_RECORD_COLS if c not in key)
    sql = (f"INSERT INTO team_betting_records ({', '.join(_TEAM_RECORD_COLS)}) "
           f"VALUES ({', '.join(['%s'] * len(_TEAM_RECORD_COLS))}) "
           f"ON CONFLICT (sport, season, team_name) DO UPDATE SET {updates}, captured_at = now()")
    vals = [tuple(json.dumps(r[c]) if c == "records" else r.get(c) for c in _TEAM_RECORD_COLS) for r in rows]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(sql, vals)
        conn.commit()
    return len(vals)
```

`scripts/capture_team_records.py`:

```python
"""Capture Action Network season betting records (ATS / O-U / units by split)
for NFL + CFB into team_betting_records. One actor call per league with
includeStandings (maxGames 1 keeps it cheap). Daily, before the desk.

Usage: APIFY_TOKEN=... DATABASE_URL=... uv run python scripts/capture_team_records.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel import config
from sportsmodel.db import upsert_team_betting_records
from sportsmodel.nfl.action_network import fetch_splits, parse_team_records
from sportsmodel.serving.team_names import match_team_names

ROOT = Path(__file__).resolve().parents[1]
LEAGUES = {"nfl": "nfl", "cfb": "ncaaf"}


def our_team_names(sport: str) -> list[str]:
    path = ROOT / "assets" / ("nfl/nfl_teams.json" if sport == "nfl" else "cfb/fbs_teams.json")
    return list(json.loads(path.read_text()).values())


def current_season(sport: str) -> int:
    if sport == "nfl":
        from sportsmodel.nfl import espn
    else:
        from sportsmodel.cfb import espn
    return int(espn.fetch_current_week()["season"])


def resolve_names(rows: list[dict], ours: list[str]) -> tuple[list[dict], list[str]]:
    """Attach team_name (our name when matched, else the AN name); return the
    unmatched AN names for logging. PURE."""
    m = match_team_names([r["an_team_name"] for r in rows], ours)
    unmatched = [r["an_team_name"] for r in rows if r["an_team_name"] not in m]
    return [{**r, "team_name": m.get(r["an_team_name"], r["an_team_name"])} for r in rows], unmatched


def main() -> None:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        raise SystemExit("APIFY_TOKEN required")
    for sport, league in LEAGUES.items():
        try:
            season = current_season(sport)
            items = fetch_splits(token, leagues=(league,), game_status=("scheduled",),
                                 extra_input={"season": season, "maxGames": 1, "includeStandings": True})
            rows, unmatched = resolve_names(parse_team_records(items), our_team_names(sport))
            n = upsert_team_betting_records(rows)
            print(f"[{sport}] season={season} teams={n} unmatched={len(unmatched)} {unmatched[:10]}")
        except Exception as exc:  # noqa: BLE001 -- one league failing must not abort the other
            print(f"[{sport}] team records capture failed: {exc}")


if __name__ == "__main__":
    main()
```

Add a test (importlib-load the script) for the pure `resolve_names`: matched name replaced, unmatched kept + reported.

- [ ] **Step 5:** tests → PASS; full suite. **Step 6: Commit** `feat(trends): capture Action Network season betting records`.

### Task 2: NFL situational trends (pure)

**Files:** Create `src/sportsmodel/nfl/trends.py`; Test `tests/nfl/test_trends.py`.

**Produces:** `NFL_DIVISIONS: dict[str, str]`, `SITUATION_LABELS: dict[str, str]`, `team_game_log(sched) -> pandas.DataFrame`, `situations_for(team, is_home, team_line, kickoff_et, log_before) -> list[str]`, `compute_game_trends(sched, upcoming, current_season, min_n=5) -> list[dict]` rows `{game_pk, team_name?..}` — see below.

- [ ] **Step 1: Failing tests** `tests/nfl/test_trends.py` — build a tiny synthetic schedule DataFrame (columns: season, week, game_type "REG", gameday "YYYY-MM-DD", gametime "HH:MM", home_team, away_team, home_score, away_score, result (=home−away), total, spread_line (home favored positive), total_line, espn). Cover:

```python
import pandas as pd
from sportsmodel.nfl.trends import (NFL_DIVISIONS, compute_game_trends, situations_for, team_game_log)

def _g(season, day, home, away, hs, as_, spread, tot, time="13:00", espn=None):
    return {"season": season, "week": 1, "game_type": "REG", "gameday": day, "gametime": time,
            "home_team": home, "away_team": away, "home_score": hs, "away_score": as_,
            "result": None if hs is None else hs - as_, "total": None if hs is None else hs + as_,
            "spread_line": spread, "total_line": tot, "espn": espn}

def test_team_game_log_cover_and_ou_from_each_side():
    sched = pd.DataFrame([_g(2025, "2025-09-07", "BUF", "NYJ", 27, 20, 6.5, 44.5)])
    log = team_game_log(sched)
    buf = log[log.team == "BUF"].iloc[0]; nyj = log[log.team == "NYJ"].iloc[0]
    assert buf.is_home and buf.team_line == -6.5 and buf.ats == "W"      # won by 7, laid 6.5
    assert (not nyj.is_home) and nyj.team_line == 6.5 and nyj.ats == "L"
    assert buf.ou == "O" and nyj.ou == "O"                                # 47 > 44.5

def test_push_and_missing_line_rows():
    sched = pd.DataFrame([_g(2025, "2025-09-07", "BUF", "NYJ", 27, 20, 7.0, 47.0),
                          _g(2025, "2025-09-14", "BUF", "MIA", 20, 17, None, None)])
    log = team_game_log(sched)
    assert log[(log.team == "BUF") & (log.gameday == "2025-09-07")].iloc[0].ats == "P"
    assert log[(log.team == "BUF") & (log.gameday == "2025-09-14")].iloc[0].ats is None

def test_situations_for():
    prev = pd.DataFrame([{"gameday": "2026-09-13", "is_home": False, "su": "W", "season": 2026}])
    s = situations_for("BUF", True, -3.0, "2026-09-27 20:20", prev, opp="MIA", season=2026, gameday="2026-09-27")
    assert set(s) >= {"home", "favorite", "off_road", "off_win", "off_bye", "division", "primetime"}
    assert "underdog" not in s and "road" not in s

def test_compute_game_trends_counts_matching_past_games_and_min_n():
    rows = []
    # BUF: 6 past road games each followed by a game (to build "off a road game" history)
    days = pd.date_range("2025-09-07", periods=12, freq="7D").strftime("%Y-%m-%d")
    for i, d in enumerate(days):
        if i % 2 == 0:
            rows.append(_g(2025, d, "NYJ", "BUF", 10, 20, 3.0, 40.0))   # BUF road win, covers
        else:
            rows.append(_g(2025, d, "BUF", "NE", 24, 21, 1.0, 40.0))    # home after road game; covers
    rows.append(_g(2026, "2026-09-27", "BUF", "MIA", None, None, 2.5, 45.0, espn="401"))
    sched = pd.DataFrame(rows)
    upcoming = [{"game_pk": 401, "home_team": "BUF", "away_team": "MIA", "home_line": -2.5,
                 "gameday": "2026-09-27", "gametime": "13:00"}]
    out = compute_game_trends(sched, upcoming, current_season=2026, min_n=5)
    off_road = [r for r in out if r["team"] == "BUF" and r["situation"] == "off_road"]
    # previous BUF game (2025-12-07? last of 12) decides off_road for the upcoming game; if it applies,
    # history must count only past games that were themselves off a road game
    for r in off_road:
        assert r["n"] >= 5 and r["ats_w"] + r["ats_l"] + r["ats_p"] == r["n"]
    assert all(r["n"] >= 5 for r in out)
    assert all(r["since_season"] == 2023 for r in out)
```

(The implementer may reshape this last fixture so that `off_road` definitely applies to the upcoming game — e.g. make BUF's final 2025 game a road game — and must then assert its exact counts. Keep the other assertions.)

- [ ] **Step 2:** run → FAIL. **Step 3: Implement** `src/sportsmodel/nfl/trends.py`:

```python
"""NFL situational betting trends, computed from nflverse schedules.

For each upcoming game and each team: which situations apply THIS week (home/
road, favorite/underdog, off a road/home game, off a win/loss, off a bye,
division game, primetime), and the team's ATS (W-L-P) and O/U (O-U-P) record in
past games that were in the same situation, over the current season + last 3.
Descriptive only -- the model does not use these. PURE (DataFrames in, dicts out).

nflverse conventions: spread_line > 0 means the HOME team was favored by that
many; result = home_score - away_score; gametime is Eastern "HH:MM".
"""
from __future__ import annotations

from datetime import date

import pandas as pd

NFL_DIVISIONS = {
    **dict.fromkeys(["BUF", "MIA", "NE", "NYJ"], "AFC East"),
    **dict.fromkeys(["BAL", "CIN", "CLE", "PIT"], "AFC North"),
    **dict.fromkeys(["HOU", "IND", "JAX", "TEN"], "AFC South"),
    **dict.fromkeys(["DEN", "KC", "LV", "LAC"], "AFC West"),
    **dict.fromkeys(["DAL", "NYG", "PHI", "WAS"], "NFC East"),
    **dict.fromkeys(["CHI", "DET", "GB", "MIN"], "NFC North"),
    **dict.fromkeys(["ATL", "CAR", "NO", "TB"], "NFC South"),
    **dict.fromkeys(["ARI", "LA", "SF", "SEA"], "NFC West"),
}
SITUATION_LABELS = {
    "home": "at home", "road": "on the road", "favorite": "as a favorite", "underdog": "as an underdog",
    "off_road": "off a road game", "off_home": "off a home game", "off_win": "off a win",
    "off_loss": "off a loss", "off_bye": "off a bye", "division": "in division games",
    "primetime": "in primetime",
}
BYE_DAYS = 13
PRIMETIME_ET = "19:00"


def _vs(x: float) -> str:
    return "W" if x > 0 else "L" if x < 0 else "P"


def team_game_log(sched: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) for completed games: team, opp, season, gameday,
    gametime, is_home, team_line (the team's own spread: home -spread_line),
    su (W/L/P), ats (W/L/P or None without a line), ou (O/U/P or None), sorted
    by team then date."""
    done = sched[sched["home_score"].notna() & sched["away_score"].notna()]
    rows = []
    for g in done.itertuples(index=False):
        margin_home = float(g.home_score) - float(g.away_score)
        has_line = pd.notna(g.spread_line)
        has_total = pd.notna(g.total_line)
        total = float(g.home_score) + float(g.away_score)
        ou = None if not has_total else ("O" if total > g.total_line else "U" if total < g.total_line else "P")
        for team, opp, is_home, margin in ((g.home_team, g.away_team, True, margin_home),
                                           (g.away_team, g.home_team, False, -margin_home)):
            line = None if not has_line else (-float(g.spread_line) if is_home else float(g.spread_line))
            rows.append({"team": team, "opp": opp, "season": int(g.season), "gameday": str(g.gameday),
                         "gametime": str(g.gametime or ""), "is_home": is_home, "team_line": line,
                         "su": _vs(margin), "ats": None if line is None else _vs(margin + line), "ou": ou})
    log = pd.DataFrame(rows)
    return log.sort_values(["team", "gameday"]).reset_index(drop=True) if len(log) else log


def situations_for(team, is_home, team_line, kickoff_et, log_before, *, opp, season, gameday) -> list[str]:
    """Situations that apply to `team` for one game. `log_before` = the team's
    completed games strictly before this one (team_game_log rows, sorted).
    team_line: the team's own spread for this game (negative = favored; None or
    0 -> neither favorite nor underdog). kickoff_et: "YYYY-MM-DD HH:MM" Eastern."""
    s = ["home" if is_home else "road"]
    if team_line is not None and team_line != 0:
        s.append("favorite" if team_line < 0 else "underdog")
    if len(log_before):
        prev = log_before.iloc[-1]
        s.append("off_home" if prev["is_home"] else "off_road")
        if prev["su"] in ("W", "L"):
            s.append("off_win" if prev["su"] == "W" else "off_loss")
        if int(prev["season"]) == int(season):
            gap = (date.fromisoformat(gameday) - date.fromisoformat(str(prev["gameday"])[:10])).days
            if gap >= BYE_DAYS:
                s.append("off_bye")
    if NFL_DIVISIONS.get(team) and NFL_DIVISIONS.get(team) == NFL_DIVISIONS.get(opp):
        s.append("division")
    if kickoff_et and kickoff_et[-5:] >= PRIMETIME_ET:
        s.append("primetime")
    return s


def compute_game_trends(sched: pd.DataFrame, upcoming: list[dict], current_season: int,
                        min_n: int = 5) -> list[dict]:
    """Rows {game_pk, team, situation, label, ats_w, ats_l, ats_p, ou_o, ou_u,
    ou_p, n, since_season} for each upcoming game x team x applicable situation
    with >= min_n past games in that situation (seasons current-3..current).
    `upcoming`: {game_pk, home_team, away_team (nflverse abbrs), home_line (home
    spread, negative = home favored; None allowed), gameday "YYYY-MM-DD",
    gametime "HH:MM" ET}."""
    since = current_season - 3
    log = team_game_log(sched[sched["season"] >= since - 1])   # one extra season so the first game has a "previous"
    out: list[dict] = []
    for g in upcoming:
        for team, opp, is_home in ((g["home_team"], g["away_team"], True), (g["away_team"], g["home_team"], False)):
            tlog = log[log["team"] == team] if len(log) else log
            before = tlog[tlog["gameday"] < g["gameday"]] if len(tlog) else tlog
            line = g.get("home_line")
            team_line = None if line is None else (line if is_home else -line)
            now_sits = situations_for(team, is_home, team_line, f"{g['gameday']} {g.get('gametime') or ''}".strip(),
                                      before, opp=opp, season=current_season, gameday=g["gameday"])
            # Situations of each past game (in-window), evaluated against ITS own previous game.
            hist = {k: [] for k in now_sits}
            games = before.reset_index(drop=True)
            for i in range(len(games)):
                row = games.iloc[i]
                if int(row["season"]) < since or row["ats"] is None:
                    continue
                past_sits = situations_for(team, bool(row["is_home"]), row["team_line"],
                                           f"{row['gameday']} {row['gametime']}", games.iloc[:i],
                                           opp=row["opp"], season=int(row["season"]), gameday=str(row["gameday"])[:10])
                for k in now_sits:
                    if k in past_sits:
                        hist[k].append(row)
            for k, rows in hist.items():
                if len(rows) < min_n:
                    continue
                ats = [r["ats"] for r in rows]
                ou = [r["ou"] for r in rows if r["ou"] is not None]
                out.append({"game_pk": g["game_pk"], "team": team, "situation": k,
                            "label": SITUATION_LABELS[k],
                            "ats_w": ats.count("W"), "ats_l": ats.count("L"), "ats_p": ats.count("P"),
                            "ou_o": ou.count("O"), "ou_u": ou.count("U"), "ou_p": ou.count("P"),
                            "n": len(rows), "since_season": since})
    return out
```

- [ ] **Step 4:** tests → PASS; full suite. **Step 5: Commit** `feat(trends): NFL situational ATS/O-U trends (pure)`.

### Task 3: Trend builder script + daily workflow

**Files:** Create `scripts/build_nfl_trends.py`, `.github/workflows/build-trends.yml`; Modify `src/sportsmodel/db.py` (`replace_nfl_game_trends`); Tests `tests/test_db_trends.py` (append), `tests/test_build_nfl_trends.py`.

- [ ] **Step 1:** `db.replace_nfl_game_trends(game_pks: list[int], rows: list[dict]) -> int` — one transaction: `DELETE FROM nfl_game_trends WHERE game_pk = ANY(%s)` then insert rows (cols: game_pk, team_name, situation, label, ats_w, ats_l, ats_p, ou_o, ou_u, ou_p, n, since_season); returns len(rows); empty game_pks → 0 without touching the DB. Test with FakeConn (delete before insert, commit).
- [ ] **Step 2:** Script:

```python
"""Compute NFL situational trends for the upcoming slate -> nfl_game_trends.
Daily (build-trends.yml), before the desk. See sportsmodel/nfl/trends.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sportsmodel.db import get_postgres, replace_nfl_game_trends
from sportsmodel.nfl.data import load_schedules
from sportsmodel.nfl.injuries_nflverse import nfl_season
from sportsmodel.nfl.trends import compute_game_trends

ROOT = Path(__file__).resolve().parents[1]


def upcoming_games(sched) -> list[dict]:
    """Upcoming NFL games from predictions_current, mapped to nflverse abbrs +
    schedule date/time via the schedule's `espn` id. PURE-ish helper kept small."""
    with get_postgres() as pg, pg.cursor() as cur:
        cur.execute("SELECT game_pk, market_spread FROM predictions_current WHERE sport = 'nfl' AND commence_time > now()")
        preds = {int(pk): spread for pk, spread in cur.fetchall()}
    return build_upcoming(sched, preds)


def build_upcoming(sched, preds: dict[int, float | None]) -> list[dict]:
    """PURE: schedule rows whose `espn` id is an upcoming predicted game_pk."""
    out = []
    for g in sched.itertuples(index=False):
        try:
            pk = int(g.espn)
        except (TypeError, ValueError):
            continue
        if pk not in preds:
            continue
        line = preds[pk]
        if line is None and g.spread_line == g.spread_line:          # not NaN
            line = -float(g.spread_line)                              # nflverse: + = home favored
        out.append({"game_pk": pk, "home_team": g.home_team, "away_team": g.away_team,
                    "home_line": None if line is None else float(line),
                    "gameday": str(g.gameday)[:10], "gametime": str(g.gametime or "")})
    return out


def main() -> None:
    from datetime import datetime, timezone
    season = nfl_season(datetime.now(timezone.utc))
    sched = load_schedules(list(range(season - 4, season + 1)))
    crosswalk = json.loads((ROOT / "assets/nfl/nfl_teams.json").read_text())
    games = upcoming_games(sched)
    rows = compute_game_trends(sched, games, current_season=season)
    for r in rows:
        abbr = r.pop("team")
        r["team_name"] = crosswalk.get(abbr, abbr)
    n = replace_nfl_game_trends([g["game_pk"] for g in games], rows)
    print(f"[build_nfl_trends] season={season} games={len(games)} trend_rows={n}")


if __name__ == "__main__":
    main()
```

Test the pure `build_upcoming` (predicted pk kept, market_spread used, NaN spread fallback flips sign, non-predicted/NaN espn dropped).

- [ ] **Step 3:** Workflow `.github/workflows/build-trends.yml`:

```yaml
name: build-trends
# Daily betting trends, ahead of the desk (desk-auto-* 22:15 UTC): Action
# Network season betting records (NFL + CFB) and NFL situational trends.
on:
  schedule:
    - cron: "30 20 * * *"
  workflow_dispatch:
permissions:
  contents: read
jobs:
  trends:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v5
      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1
      - name: Sync deps
        run: uv sync
      - name: Capture Action Network team betting records
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          APIFY_TOKEN: ${{ secrets.APIFY_TOKEN }}
        run: uv run python scripts/capture_team_records.py
      - name: Build NFL situational trends
        if: ${{ always() }}
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/build_nfl_trends.py
```

- [ ] **Step 4:** tests; YAML parses; full suite. **Step 5: Commit** `feat(trends): daily trend build (records + NFL situational)`.

### Task 4: Decision Desk integration

**Files:** Modify `scripts/desk_inputs.py` (pure `trend_block`, `build_bundle` gains optional `trends`, `main` loads them), `scripts/synthesize_desk_picks.py` (SYSTEM_PROMPT); Tests `tests/cfb/test_desk_inputs.py` (append) and a prompt test.

- [ ] **Step 1: Failing tests** (append to `tests/cfb/test_desk_inputs.py`, importing `trend_block` from the same module the file already imports `build_bundle` from):

```python
def test_trend_block_formats_records_and_situational():
    records = {"ats": {"w": 2, "l": 1, "p": 0}, "ats_road": {"w": 1, "l": 0, "p": 0},
               "ats_dog": {"w": 0, "l": 0, "p": 0}, "ats_last_5": {"w": 2, "l": 1, "p": 0},
               "over_under": {"o": 2, "u": 1, "p": 0}, "over_under_road": {"o": 1, "u": 0, "p": 0},
               "units": {"w": 1.2, "l": 0}}
    sit = [{"label": "off a road game", "ats_w": 7, "ats_l": 2, "ats_p": 0, "ou_o": 4, "ou_u": 5, "ou_p": 0, "since_season": 2023}]
    b = trend_block(records, sit, is_home=False, is_fav=False)
    assert b["records"]["ATS"] == "2-1-0" and b["records"]["ATS on the road"] == "1-0-0"
    assert b["records"]["ATS as underdog"] == "0-0-0" and b["records"]["ATS last 5"] == "2-1-0"
    assert b["records"]["O/U"] == "2-1-0" and b["records"]["O/U on the road"] == "1-0-0"
    assert b["records"]["Units"] == "+1.2"
    assert b["situational"] == ["7-2-0 ATS, O/U 4-5-0 off a road game since 2023"]


def test_trend_block_none_when_no_data():
    assert trend_block(None, [], is_home=True, is_fav=True) is None


def test_build_bundle_attaches_trends_when_given():
    trends = {GAMES[0]["game_pk"]: {"home": {"records": {"ATS": "1-0-0"}, "situational": []}, "away": None}}
    bundle = build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW, trends=trends)
    assert bundle[0]["trends"]["home"]["records"]["ATS"] == "1-0-0"
    # default: no trends arg -> key present with both sides None
    assert build_bundle(GAMES, MODEL_ROWS, FORM_ROWS, INJURIES, WEATHER, NOW)[0]["trends"] == {"home": None, "away": None}
```

And a prompt test (new `tests/test_synthesize_desk_prompt.py`, importlib or package import as other tests do): `SYSTEM_PROMPT` contains `"trends"`, contains `"agent_notes.analyst"` or `"analyst note"`, and contains the phrase `"never the sole"`.

- [ ] **Step 2: Implement** in `desk_inputs.py`:

```python
def _rec(r: dict | None, a: str = "w", b: str = "l") -> str | None:
    if not r or (r.get(a) is None and r.get(b) is None):
        return None
    return f"{r.get(a) or 0}-{r.get(b) or 0}-{r.get('p') or 0}"


def trend_block(records: dict | None, situational: list[dict], *, is_home: bool, is_fav: bool | None) -> dict | None:
    """PURE. One team's trends for the desk: Action Network season records
    relevant to this game (ATS overall / this venue / this role / last 5, O/U
    overall / this venue, units) + NFL situational trend lines. None when empty."""
    out: dict = {}
    if records:
        venue = "home" if is_home else "road"
        venue_lbl = "at home" if is_home else "on the road"
        picks = [("ATS", _rec(records.get("ats"))), (f"ATS {venue_lbl}", _rec(records.get(f"ats_{venue}"))),
                 ("ATS last 5", _rec(records.get("ats_last_5"))),
                 ("O/U", _rec(records.get("over_under"), "o", "u")),
                 (f"O/U {venue_lbl}", _rec(records.get(f"over_under_{venue}"), "o", "u"))]
        if is_fav is not None:
            role, role_lbl = ("fav", "as favorite") if is_fav else ("dog", "as underdog")
            picks.insert(2, (f"ATS {role_lbl}", _rec(records.get(f"ats_{role}"))))
        units = records.get("units")
        if units and units.get("w") is not None:
            net = float(units.get("w") or 0) - float(units.get("l") or 0)
            picks.append(("Units", f"{net:+.1f}"))
        out["records"] = {k: v for k, v in picks if v is not None}
    sit = [f"{t['ats_w']}-{t['ats_l']}-{t['ats_p']} ATS, O/U {t['ou_o']}-{t['ou_u']}-{t['ou_p']} "
           f"{t['label']} since {t['since_season']}" for t in (situational or [])]
    if sit:
        out["situational"] = sit
    if not out:
        return None
    out.setdefault("records", {})
    out.setdefault("situational", [])
    return out
```

`build_bundle(..., now, trends: dict[int, dict] | None = None)`: each entry gets `"trends": (trends or {}).get(game_pk) or {"home": None, "away": None}`; update the docstring's return shape. In `main()`, after the games/model rows load: read `team_betting_records` for the sport + current season (`SELECT team_name, records FROM team_betting_records WHERE sport = %s AND season = %s`) and, for NFL, `nfl_game_trends` for the slate's game_pks; build `trends[game_pk] = {"home": trend_block(recs.get(home), sit[(pk, home)], is_home=True, is_fav=...), "away": ...}` where is_fav comes from the game's market_spread (home line; negative = home favored; None -> None). Wrap the DB reads in try/except so a missing table logs a warning and yields no trends (desk must still run). Pass `trends=trends` to `build_bundle`.

In `synthesize_desk_picks.py` SYSTEM_PROMPT, add after the METHODOLOGY bullets (keep everything else verbatim):

```
- TRENDS (`trends.home` / `trends.away`): season betting records (ATS, O/U,
  units -- overall, by venue, by favorite/underdog role, last 5) and, for NFL,
  situational trends ("7-2-0 ATS off a road game since 2023"). The SPORTS-ANALYST
  note (`agent_notes.analyst`) MUST cite every trend given for BOTH teams and say
  which way each points. Trends are SUPPORTING EVIDENCE: they may raise or lower
  conviction or tip a close call, but they are never the sole basis for a spread
  lean -- a lean still needs a concrete edge (injury, form, model-vs-line). Most
  ATS trends are small-sample noise; weigh lopsided, larger-sample ones more. If
  a trend influenced the call, say so in the rationale. If `trends` is null for a
  team, say trends were unavailable.
```

- [ ] **Step 3:** tests → PASS; full suite. **Step 4: Commit** `feat(desk): feed betting trends to the analyst agent`.

### Task 5: Game page "Trends" section (CappingAlpha)

**Files:** `/Users/ryan/Desktop/CappingAlpha/app.js`, all `*.html` → `app.js?v=20260923h`.

- [ ] In the game page build (where splits are fetched for nfl/cfb), also fetch `team_betting_records?sport=eq.${sport}&team_name=in.(${enc(away)},${enc(home)})&order=season.desc` (URL-encode names; wrap each in double quotes inside `in.()` as PostgREST requires: `in.("Buffalo Bills","Miami Dolphins")`) and, for NFL, `nfl_game_trends?game_pk=eq.${game}`; both `.catch(() => [])`. Use the latest season row per team.
- [ ] New `trendsSection(awayName, homeName, recs, sits, awayCol, homeCol, homeLine)` rendered directly after the Public Betting section (`${predictionsSec}${splitsSec}${trendsSec}${simSec}…`), titled `Trends`, two columns (away | home, team-colored headers). Each column lists (label · record · %): ATS, ATS at this venue, ATS in this role (fav/dog from `r.market_spread` home line: negative = home favored; neither on pick'em/None), ATS last 5, O/U, O/U at this venue, Units (net = w − l, signed); then (NFL) situational lines `"{W-L-P} ATS · O/U {O-U-P} {label} since {since_season}"`. Highlight (class `trend-hot`) any ATS or O/U record where one side ≥ 70% of decided (W+L or O+U) and decided ≥ 8. Footnote: `Trends are descriptive, not predictive — the model doesn't use them. Records via Action Network; situational trends computed from nflverse (current season + last 3, ≥5 games).` Empty state when neither team has data: `No betting trends for this matchup yet.`
- [ ] CSS for the two-column layout (stacks on ≤620px), `node --check`, bump cache version, browser check with `sb` stubbed for the two endpoints (tables won't exist until the migration).
