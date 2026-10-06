# CFB Site Panels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Light up the game-page panels the Phase A redesign left hidden: a venue + weather hero line (CFB + NFL), new Key Insights rows (CFB havoc and turnovers; CFB + NFL weather) and a CFB Projected Game Flow chart, all from real data and hidden when the data is missing.

**Architecture:** A daily job (`scripts/build_game_info.py`, workflow `build-game-info.yml`) and the Monday CFB job (`build-cfb-advanced.yml`, extended) publish three small Supabase tables (`game_info`, `cfb_team_insights`, `cfb_quarter_shares`) by idempotent upserts; the site reads them with the anon key through the existing `sb()` / `safeCard` patterns and each panel hides when its table is missing or empty. Pure builders (`cfb/panels.py`, `game_info.py`) hold all the logic and take already-decoded data, so every Python test is network-free; new CFBD data (venue city/state, line scores, turnovers) is parsed by extending `cfb/cfbd_games.py` and committed as parquet under `assets/cfb/`. The migration is a file the USER runs; nothing in this plan writes to Supabase except the scheduled jobs after merge.

**Tech Stack:** Python 3 (`uv`, pandas, httpx, tenacity, psycopg, pytest), GitHub Actions, Supabase Postgres + PostgREST (RLS, anon read), plain-JS site (`site/`, vm-based `node --test` suite, inline SVG chart helpers in `site/js/ui.js`).

**Spec:** `docs/superpowers/specs/2026-10-06-cfb-site-panels-design.md` (the authority; this plan argues from it, executors read both).

## Global Constraints

Copied from the spec (every task's requirements include these):

- Real data only; anything missing is hidden (a missing table, a missing row, a missing field).
- Venue/weather for CFB **and** NFL; havoc/turnover insights and game flow CFB-only.
- `game_info` (CFB + NFL): one row per (sport, game_pk), upcoming window (today − 2 days … +10 days). Columns: `sport`, `game_pk`, `venue_name`, `city`, `state`, `indoor` (bool), `temp_f`, `wind_mph`, `precip_chance` (0–100, nullable), `precip_in` (nullable), `conditions` (short text, nullable), `weather_kind` ('forecast' | 'observed'), `source` ('cfbd' | 'espn'), `captured_at`, `updated_at`. PK (sport, game_pk). Indoor → weather columns NULL and the site shows "Indoors". Any missing field stays NULL (never guessed).
- `cfb_team_insights` (current season): one row per (season, team). Season-to-date, regular season, FBS games, through the last completed week: `games`, `def_havoc_rate` + national rank (higher = better), `off_havoc_allowed_rate` + rank (lower allowed = better), `turnover_margin` (total and per game) + rank, `off_explosiveness` + rank, `def_explosiveness_allowed` + rank, `through_week`, `updated_at`. Ranks among FBS teams with ≥ 3 games.
- `cfb_quarter_shares`: one row per team. For the current season's FBS teams: share of points **scored** and **allowed** in Q1–Q4 (OT excluded; shares sum to 1) over the last two completed seasons plus the current season to date, shrunk toward the league average with weight n/(n+k), k = 12 games, plus `games_used`, `updated_at`.
- Jobs: `build-game-info.yml` daily at 14:00 UTC and on football days (Thu/Fri/Sat/Sun/Mon) at 16:00 and 22:00 UTC; the Monday CFB job (`build-cfb-advanced.yml`) additionally runs the turnover pull + line-score refresh and rebuilds `cfb_team_insights` and `cfb_quarter_shares`. CFBD budget ≈ 60–100 calls/week; ESPN summary calls ≈ NFL games in window per run (free). Jobs write only these three tables (upsert; never delete rows outside the window they rebuilt).
- Hero line (game page, CFB + NFL), under the kickoff line: `Venue · City, ST · 41°F, wind 14 mph, 20% rain` (pieces joined with " · "; any missing piece omitted; indoor → `Venue · City, ST · Indoors`; finished games label observed weather). Hidden entirely when `game_info` has no row.
- Key Insights (`gmKeyInsights`): candidate rows each with a strength score, existing four-row cap kept by strength. CFB *Havoc mismatch*: one defense's havoc rank vs the other offense's havoc-allowed rank, shown when the rank gap ≥ 30 or one side is in the top/bottom 15% nationally. CFB *Turnover edge*: turnover-margin rank gap ≥ 30 or a top/bottom-15% team. CFB + NFL *Weather*: wind ≥ 15 mph, temp < 40°F, or precip chance ≥ 50% / measurable precip, outdoor games only. Existing rows keep their logic; strength for existing rows is derived from their current thresholds so the ordering is deterministic. New icons follow `GM_INSIGHT_ICON`.
- Projected Game Flow (CFB, new card after Key Insights): per team, projected points Q1–Q4 = projected team score × blended share, where blended share = average of the team's scored share and the opponent's allowed share (renormalised to 1). Small grouped bar or line chart via existing `ui.js` chart helpers, labelled "Projected pace", with an ⓘ tooltip explaining the method. Finished games: actual quarter scores beside the projection. Hidden when there is no projection or no share row for either team.
- Fetches use the existing `sb()` / `safeCard` patterns; escaping via `ctxEsc`; cache key bumped once at the end (`?v=20261007a` → `?v=20261007b`); panels hide cleanly if the tables don't exist yet (the site may deploy before the migration).
- `db/migration_site_panels.sql`: the three tables, PKs, anon read policies (SELECT) consistent with existing serving tables. **The user runs it.**
- Tests are network-free. Visual check: NFL and CFB game pages at 1280 / 1440 / 1920 (hard reload), with and without data.
- Out of scope: TV network, stadium photos, NFL havoc/turnover insights, NFL game flow, live in-game data (project 3), any model change.

Project rules for the executor:

- Every task ends green: `uv run pytest -q <touched files>` and/or `cd site && npm test` (baseline before this plan: site 307 tests pass).
- Steps marked **[NETWORK]** are run by the executing agent itself (never delegated to a reviewer subagent). CFBD: load the key with `set -a; source .env; set +a` and never print it; the whole plan uses well under 100 CFBD calls. ESPN needs no key.
- No task writes to Supabase. No task runs the migration. The first writes happen only when the scheduled / dispatched jobs run after the user has run the migration and merged.
- `site/tests/board-left.test.mjs` crashed once at file level (`test failed`, no assertion) during one parallel `npm test` run while this plan was being validated and passed on every re-run, including the same tree; if it fails alone with no assertion message, re-run once before investigating.
- Browser verification copies `site/` to `/tmp` and serves it there (macOS blocks serving from `~/Desktop`).
- Commits end with the trailer `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Any subagent used for coding or review runs on Sonnet (pass `model: "sonnet"` explicitly).

---

## Live API shapes (checked 2026-10-06, read-only)

CFBD calls made while planning: `/venues`, `/games` (2025 wk 6 and 2026 wk 5), `/games/teams` (2025 wk 6), `/games/weather` (2026 wk 6) = 5 calls. ESPN: one NFL `/summary` (event 401872971, a finished game). Shapes the plan is built on:

| Shape | Status | What was seen / what is assumed |
|---|---|---|
| CFBD `/venues` | **verified** | 852 venues; `city` always present, `state` null for 6 (non-US); also `dome` bool, `timezone` (null for some), `latitude`, `longitude`, `elevation` (string), `countryCode`. |
| CFBD `/games` line scores | **verified** | `homeLineScores` / `awayLineScores` are lists of 4 (regulation) or 5+ (overtime periods appended, e.g. `[7, 7, 7, 3, 7]`); `null` for a few completed games and for unplayed ones; the sum equals `homePoints` except 1 game of 301 in week 5 (so quarter sums are not forced to equal the final). `classification` fields exist (`fbs`, `fcs`, `ii`, `iii`). |
| CFBD `/games/teams` | **verified** | `[{id, teams: [{teamId, team, conference, homeAway, points, stats: [{category, stat(str)}]}, x2]}]` with no season / week / seasonType in the payload (the caller stamps them). `turnovers` is always present; `turnovers == fumblesLost + interceptions` for all 102 team-games; `interceptions` = interceptions THROWN by that team (equals the opponent's `passesIntercepted`, which is absent when 0). |
| CFBD `/games/weather` | **verified** | Same shape for forecast (before kickoff) and observed: `id, season, week, seasonType, startTime, gameIndoors, homeTeam, awayTeam, venueId, venue (name), temperature, windSpeed, precipitation (inches, e.g. 0.035), humidity, ... weatherConditionCode, weatherCondition`. `weatherCondition` is null for ~70% of rows. **No precipitation probability.** Returns every division (D-III included) so the FBS filter is needed. Indoor games still carry readings (the job nulls them). |
| ESPN NFL `/summary` venue | **verified** | `gameInfo.venue = {id, fullName, address: {city, state, zipCode, country}, grass, images}`; `header.competitions[0]` has no venue. |
| ESPN NFL `/summary` weather | **NOT verified** | The finished game had no weather block at all (neither `gameInfo.weather` nor top-level `weather`). Key names (`temperature`, `displayValue`, `windSpeed`, `precipitation` as a % chance) are assumed; the Task 2 [NETWORK] probe checks them against the current slate. |
| ESPN NFL `venue.indoor` | **NOT verified** | Absent on an outdoor venue; assumed to be `true` for domes. Probed in Task 2. |
| ESPN CFB scoreboard `competitors[].linescores[].value` | **NOT verified** | Assumed ESPN's usual shape; probed in Task 2. If absent, `line_score` stays NULL and the Game Flow card shows the projection only. |
| CFBD forecast depth (10 days ahead), postseason `/games/teams`, multi-OT (6+ periods) | **NOT verified** | Parsers are defensive; missing -> NULL -> panel hides. |

## Spec ambiguities resolved in this plan

1. **Venue <-> game link.** The spec says `/games`; `/games/weather` already carries `venueId` and the venue name (verified), so the daily job needs no `/games` call. FBS-vs-FCS games are kept (the site has pages for them); FCS-vs-FCS games are dropped (`has_fbs`).
2. **`precip_chance`.** CFBD sends only `precipitation` in inches, so CFB rows fill `precip_in` and leave `precip_chance` NULL; NFL rows fill `precip_chance` (when ESPN gives a 0-100 value) and leave `precip_in` NULL. The hero shows a chance when there is one, else measurable rain (>= 0.01 in).
3. **Actual quarter scores for finished games.** The three tables had no source for them, so `game_info` gains a nullable `line_score jsonb` (`{"home": [q1..q4], "away": [q1..q4]}`, CFB, OT excluded) filled from ESPN's free scoreboard for games inside the −2-day window. Older finished games show the projection only.
4. **Havoc is not refreshed weekly today.** The spec says `havoc_games` is "already pulled weekly after merge"; no workflow runs `build_cfb_game_data.py`. The Monday job now pulls the current season's `games` (line scores), `havoc` and `team_stats` (turnovers) too.
5. **`cfb_team_insights.n_ranked`** (size of the ranked field) is added so the site can apply the "top/bottom 15% nationally" rule without hard-coding 134 teams.
6. **Insight ordering.** The four rows are chosen and displayed strongest first. Existing rows: grade `round(30 + 0.4 x percentile)`, explosive `45`, power `30 + min(40, rank gap)`, streak `min(70, 30 + 8 x (n − 2))`; new rows start at 50 (`round(50 + min(45, gap / 3))` for havoc / turnover; weather `55 +` severity). One existing test's order assertions change accordingly.
7. **Havoc mismatch direction.** Only a defense edge is a "mismatch" (gap = offense havoc-allowed rank − defense havoc rank > 0); the 15% extremity rule applies only when the gap is positive; the better of the two matchups (home D vs away O, away D vs home O) is shown.
8. **Turnover edge** ranks by turnover margin per game; equal ranks show nothing.
9. **`weather_kind`.** CFB: `observed` once the game has kicked off, else `forecast`. NFL: always `forecast` (ESPN's block is not known to be an observation).
10. **Upserts never erase knowledge.** `game_info` upserts keep an earlier reading when a later run has none (`COALESCE(new, old)`; ESPN drops its weather after a game) and NULL the weather when the new row says indoor.
11. **Index.** The `(sport, game_pk)` primary key is the lookup index, so no second index is created.
12. **Hero text.** Conditions (when present) lead the weather piece (`Cloudy, 41°F, wind 14 mph`); `0%` rain is omitted; observed readings read `Observed 41°F, ...`.
13. **Rank field and window.** Ranks need >= 3 games; `through_week` = the last week with havoc data, and every source is clipped to it; the window is UTC (`now − 2 days … now + 10 days`).
14. **Quarter shares** use regular + postseason games of the three seasons; a team with no usable game is absent (so its game page hides the card).
15. **Layout.** The Game Flow card sits directly under Key Insights in the Overview's left column (the Cover Probability card stays in the right column).

## File Structure

New (one responsibility each):

- `src/sportsmodel/cfb/panels.py`: pure CFB panel tables: `quarter_shares`, `team_insights`, `to_rows`.
- `src/sportsmodel/game_info.py`: pure `game_info` row builders for CFB (CFBD weather + venues + ESPN line scores) and NFL (ESPN summaries), the window rule.
- `scripts/build_game_info.py`: daily job (ESPN week discovery, CFBD weather calls, upsert).
- `scripts/build_cfb_panels.py`: weekly job (assets in, two upserts out; no network).
- `db/migration_site_panels.sql`: the three tables, PKs, RLS + anon read, grants (run by the user).
- `.github/workflows/build-game-info.yml`: schedule + dispatch for the daily job.
- `site/tests/game_panels.test.mjs`: tests for everything new on the game page (hero line, insights, game flow).
- Tests: `tests/cfb/test_panels.py`, `tests/test_game_info.py`, `tests/test_db_site_panels.py`, `tests/scripts/test_build_game_info.py`, `tests/scripts/test_build_cfb_panels.py`, `tests/test_workflows_game_info.py`.
- Data: `assets/cfb/team_game_stats.parquet`.

Modified:

- `src/sportsmodel/cfb/cfbd_games.py` (venue city/state, line scores, team stats + weather-window parsers); `src/sportsmodel/nfl/espn.py` and `src/sportsmodel/cfb/espn.py` (game info, scoreboard line scores); `src/sportsmodel/db.py` (three upserts).
- `scripts/build_cfb_game_data.py` (`team_stats` dataset); `.github/workflows/build-cfb-advanced.yml` (Monday pulls + table rebuild).
- `assets/cfb/venues.parquet`, `cfbd_games.parquet`, `havoc_games.parquet` (re-pulled).
- `site/js/pages/game.js` (hero line, strength-ranked insights, game flow; kept in one file with its `gm*` siblings: the three features share `D`, `gmTeamCode` and the card chrome, and the file stays well under 900 lines), `site/css/theme.css`, `site/tests/game.test.mjs` (one ordering test), `site/tests/smoke.test.mjs`, `site/*.html` (cache key).

Task order: data parsers + backfill (1-3) -> table builders + migration (4-7) -> daily job, workflow and Monday additions (8-9) -> site hero line (10), insights (11), game flow (12) -> cache bump + final verification (13).

---

## Tasks

### Task 1: CFBD parsers: venue city/state, line scores, team turnovers, weather window

**Files:**
- Modify: `src/sportsmodel/cfb/cfbd_games.py` (`parse_games_meta`, `parse_venues`, two new parsers)
- Test: `tests/cfb/test_cfbd_games.py`

**Interfaces:**
- Consumes: existing `cfbd_games` helpers `cfbd_to_espn(name) -> str | None`, `_ints(rec, *keys) -> list[int] | None`, `num(x) -> float`, `_frame(rows, columns, ints=, strs=, bools=, dropped=)`, `season_type(x) -> str`, `NAN`.
- Produces: `cfbd_games.LINE_SCORE_COLUMNS` (`home_q1..home_q4, home_ot, away_q1..away_q4, away_ot`, appended to `GAMES_COLUMNS`); `line_scores(v) -> list[float]` = `[q1, q2, q3, q4, ot]` (all NaN unless the CFBD list has >= 4 numeric entries); `parse_venues` now also returns `city`, `state` (`""` when CFBD has none; `VENUE_COLUMNS` gains both); `TEAM_STAT_COLUMNS` and `parse_team_game_stats(payload, season: int, week: int, stype: str = "regular") -> DataFrame` (`season, week, season_type, game_id, team, opponent, fumbles_lost, interceptions_thrown, giveaways, takeaways`; `df.attrs["dropped"]`); `WEATHER_WINDOW_COLUMNS` and `parse_weather_window(payload) -> DataFrame` (`game_id, season, week, season_type, start_time, venue_id, venue, game_indoors, temperature, wind_speed, precipitation, condition, has_fbs`).

- [ ] **Step 1: Write the failing tests**

The `/games` rows below mirror the live shapes verified on 2026-10-06 (`homeLineScores: [0, 10, 3, 24]`, a 5-element overtime list, `null` for a completed game CFBD has no line score for); the `/games/teams` rows use the real `[{category, stat}]` string-valued stats list; the `/games/weather` rows are trimmed live rows.

Append to the end of `tests/cfb/test_cfbd_games.py` — the file already imports `math`, `pd`, `cg` and `cfbd_to_espn`:

```python
AZ, OKST = cfbd_to_espn("Arizona"), cfbd_to_espn("Oklahoma State")

def test_line_scores():
    assert cg.line_scores([0, 10, 3, 24]) == [0.0, 10.0, 3.0, 24.0, 0.0]
    assert cg.line_scores([7, 7, 7, 3, 7, 6]) == [7.0, 7.0, 7.0, 3.0, 13.0]
    for bad in (None, [], [1, 2, 3], [1, 2, None, 4], "x"):
        assert all(math.isnan(x) for x in cg.line_scores(bad))

def test_games_line_scores():
    df = cg.parse_games_meta([
        {"id": 1, "season": 2025, "week": 6, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
         "homePoints": 37, "awayPoints": 10, "homeLineScores": [0, 10, 3, 24], "awayLineScores": [0, 3, 7, 0]},
        {"id": 2, "season": 2025, "week": 6, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
         "homePoints": 31, "awayPoints": 27, "homeLineScores": [7, 7, 7, 3, 7], "awayLineScores": [7, 7, 7, 3, 3]},
        {"id": 3, "season": 2025, "week": 7, "seasonType": "regular", "homeTeam": "Alabama", "awayTeam": "Georgia",
         "homeLineScores": None, "awayLineScores": None}])
    a, b, c = df.iloc[0], df.iloc[1], df.iloc[2]
    assert [a["home_q1"], a["home_q2"], a["home_q3"], a["home_q4"], a["home_ot"]] == [0, 10, 3, 24, 0]
    assert (b["home_ot"], b["away_ot"]) == (7.0, 3.0)
    assert math.isnan(c["home_q1"]) and math.isnan(c["away_ot"])
    assert list(df.columns) == cg.GAMES_COLUMNS

def test_venue_city_state():
    df = cg.parse_venues([{"id": 3657, "name": "Bryant-Denny Stadium", "city": "Tuscaloosa", "state": "AL"},
                          {"id": 9, "name": "Somewhere", "city": "Dublin", "state": None}])
    assert list(df["city"]) == ["Tuscaloosa", "Dublin"] and list(df["state"]) == ["AL", ""]

def _t(name, ha, stats): return {"teamId": 1, "team": name, "homeAway": ha, "points": 1, "stats": [{"category": k, "stat": str(v)} for k, v in stats.items()]}

def test_team_stats():
    pay = [{"id": 401756910, "teams": [_t("Arizona", "home", {"turnovers": 3, "fumblesLost": 1, "interceptions": 2}),
                                       _t("Oklahoma State", "away", {"turnovers": 1, "fumblesLost": 1})]},
           {"id": 5, "teams": [_t("Arizona", "home", {}), _t("Nowhere State Fighting Pickles", "away", {"turnovers": 2})]},
           {"id": 6, "teams": [_t("Arizona", "home", {"fumblesLost": 1, "interceptions": 0}), _t("Oklahoma State", "away", {})]}]
    df = cg.parse_team_game_stats(pay, 2025, 6)
    assert df.attrs["dropped"] == 1 and len(df) == 4
    az = df[(df.game_id == 401756910) & (df.team == AZ)].iloc[0]
    ok = df[(df.game_id == 401756910) & (df.team == OKST)].iloc[0]
    assert (az["giveaways"], az["takeaways"], az["fumbles_lost"], az["interceptions_thrown"]) == (3, 1, 1, 2)
    assert (ok["giveaways"], ok["takeaways"]) == (1, 3) and math.isnan(ok["interceptions_thrown"])
    g6 = df[(df.game_id == 6) & (df.team == AZ)].iloc[0]
    assert g6["giveaways"] == 1 and math.isnan(g6["takeaways"])
    assert set(df["season"]) == {2025} and set(df["week"]) == {6} and set(df["season_type"]) == {"regular"}
    assert list(cg.parse_team_game_stats([], 2025, 1).columns) == cg.TEAM_STAT_COLUMNS

def test_weather_window():
    pay = [{"id": 401862794, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-08T23:30:00.000Z",
            "gameIndoors": True, "homeTeam": "UTSA", "awayTeam": "South Florida", "venueId": 3604, "venue": "Alamodome",
            "temperature": 82.8, "windSpeed": 9.2, "precipitation": 0, "weatherCondition": "Fair"},
           {"id": 401908581, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z",
            "gameIndoors": False, "homeTeam": "Bridgewater State", "awayTeam": "Framingham State", "venueId": 5808,
            "venue": "Swenson", "temperature": 66, "windSpeed": 10.1, "precipitation": 0.035, "weatherCondition": None},
           {"id": 7, "season": 2026, "week": 6, "homeTeam": "Alabama", "awayTeam": "Mercer", "windSpeed": None},
           {"season": 2026, "week": 6}]
    df = cg.parse_weather_window(pay)
    assert len(df) == 3 and df.attrs["dropped"] == 1
    assert list(df["has_fbs"]) == [True, False, True]
    r = df.iloc[0]; assert (r["venue_id"], r["venue"], r["game_indoors"], r["condition"]) == (3604, "Alamodome", True, "Fair")
    assert df.iloc[1]["condition"] == "" and df.iloc[1]["precipitation"] == 0.035
    assert math.isnan(df.iloc[2]["temperature"]) and math.isnan(df.iloc[2]["venue_id"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/cfb/test_cfbd_games.py`
Expected: FAIL — `FAILED tests/cfb/test_cfbd_games.py::test_line_scores - AttributeError: modul...`.

- [ ] **Step 3: Implement the parsers**

In `src/sportsmodel/cfb/cfbd_games.py`, replace this text (it occurs exactly once):

```python
GAMES_COLUMNS = ["game_id", "season", "week", "season_type", "start_date", "neutral_site",
                 "venue_id", "home_team", "away_team", "home_points", "away_points",
                 "home_pregame_elo", "away_pregame_elo"]
```

with:

```python
LINE_SCORE_PERIODS = ("q1", "q2", "q3", "q4", "ot")
LINE_SCORE_COLUMNS = [f"{side}_{p}" for side in ("home", "away") for p in LINE_SCORE_PERIODS]
GAMES_COLUMNS = ["game_id", "season", "week", "season_type", "start_date", "neutral_site",
                 "venue_id", "home_team", "away_team", "home_points", "away_points",
                 "home_pregame_elo", "away_pregame_elo"] + LINE_SCORE_COLUMNS


def line_scores(v) -> list[float]:
    """CFBD `homeLineScores` / `awayLineScores` -> [q1, q2, q3, q4, ot]. `ot` is the sum of every period
    after the fourth (0.0 when the game ended in regulation). All five are NaN unless the list holds at
    least four numeric entries (an unplayed game, or a completed one CFBD has no line score for)."""
    nan5 = [NAN] * 5
    if not isinstance(v, list) or len(v) < 4:
        return nan5
    vals = [num(x) for x in v]
    if any(math.isnan(x) for x in vals):
        return nan5
    return vals[:4] + [float(sum(vals[4:]))]
```

In `src/sportsmodel/cfb/cfbd_games.py`, replace this text (it occurs exactly once):

```python
                     "home_pregame_elo": num(g.get("homePregameElo")),
                     "away_pregame_elo": num(g.get("awayPregameElo"))})
```

with:

```python
                     "home_pregame_elo": num(g.get("homePregameElo")),
                     "away_pregame_elo": num(g.get("awayPregameElo")),
                     **dict(zip(LINE_SCORE_COLUMNS[:5], line_scores(g.get("homeLineScores")))),
                     **dict(zip(LINE_SCORE_COLUMNS[5:], line_scores(g.get("awayLineScores"))))})
```

In `src/sportsmodel/cfb/cfbd_games.py`, replace this text (it occurs exactly once):

```python
VENUE_COLUMNS = ["venue_id", "name", "timezone", "latitude", "longitude", "elevation", "dome"]
```

with:

```python
VENUE_COLUMNS = ["venue_id", "name", "timezone", "latitude", "longitude", "elevation", "dome", "city", "state"]
```

In `src/sportsmodel/cfb/cfbd_games.py`, replace this text (it occurs exactly once):

```python
    as a string), dome (NaN when unknown). Venues without an id are skipped."""
```

with:

```python
    as a string), dome (NaN when unknown), city and state ("" when CFBD has none; a few venues have no
    state). Venues without an id are skipped."""
```

In `src/sportsmodel/cfb/cfbd_games.py`, replace this text (it occurs exactly once):

```python
                     "dome": float(dome) if isinstance(dome, bool) else NAN})
    return _frame(rows, VENUE_COLUMNS, ints=("venue_id",), strs=("name", "timezone"))
```

with:

```python
                     "dome": float(dome) if isinstance(dome, bool) else NAN,
                     "city": str(v.get("city") or ""), "state": str(v.get("state") or "")})
    return _frame(rows, VENUE_COLUMNS, ints=("venue_id",), strs=("name", "timezone", "city", "state"))
```

Append to the end of `src/sportsmodel/cfb/cfbd_games.py` — the two new parsers, after `parse_prior_ratings`:

```python
# -------------------------------------------------------------- team stats --

TEAM_STAT_COLUMNS = ["season", "week", "season_type", "game_id", "team", "opponent",
                     "fumbles_lost", "interceptions_thrown", "giveaways", "takeaways"]


def _stat(stats, category: str) -> float:
    """One entry of a CFBD `/games/teams` stats list ([{"category": "turnovers", "stat": "3"}, ...]) as a
    float; NaN when the category is absent or its value is not a number (the values arrive as strings)."""
    for s in stats if isinstance(stats, list) else []:
        if isinstance(s, dict) and s.get("category") == category:
            try:
                return float(s.get("stat"))
            except (TypeError, ValueError):
                return NAN
    return NAN


def parse_team_game_stats(payload, season: int, week: int, stype: str = "regular") -> pd.DataFrame:
    """CFBD `/games/teams?year=&week=&seasonType=` -> one row per team-game of ball security.

    The payload is [{"id": gameId, "teams": [{"team", "homeAway", "stats": [{"category", "stat"}]}, x2]}] and
    carries no season / week, so the caller stamps `season`, `week` and `stype` (the request's own params).
    `giveaways` = the team's `turnovers` stat (fumblesLost + interceptions THROWN: verified live that CFBD's
    `interceptions` is the offense's, `passesIntercepted` the defense's); `takeaways` = the opponent's
    giveaways in the same game. A missing stat is NaN, never 0. A game whose two sides do not both map to
    FBS ids (an FCS opponent) is dropped and counted in `df.attrs["dropped"]`, matching havoc / advanced."""
    rows, dropped = [], 0
    for g in payload:
        gid = _ints(g, "id")
        teams = g.get("teams") if isinstance(g.get("teams"), list) else []
        ids = [cfbd_to_espn((t or {}).get("team") or "") for t in teams]
        if gid is None or len(teams) != 2 or not all(ids):
            dropped += 1
            continue
        per = []
        for t in teams:
            st = t.get("stats")
            fl, ints, tot = _stat(st, "fumblesLost"), _stat(st, "interceptions"), _stat(st, "turnovers")
            if math.isnan(tot) and not (math.isnan(fl) or math.isnan(ints)):
                tot = fl + ints
            per.append((fl, ints, tot))
        for i in (0, 1):
            rows.append({"season": int(season), "week": int(week), "season_type": season_type(stype),
                         "game_id": gid[0], "team": ids[i], "opponent": ids[1 - i],
                         "fumbles_lost": per[i][0], "interceptions_thrown": per[i][1],
                         "giveaways": per[i][2], "takeaways": per[1 - i][2]})
    return _frame(rows, TEAM_STAT_COLUMNS, ints=("season", "week", "game_id"),
                  strs=("season_type", "team", "opponent"), dropped=dropped)


# ---------------------------------------------------------- weather window --

WEATHER_WINDOW_COLUMNS = ["game_id", "season", "week", "season_type", "start_time", "venue_id", "venue",
                          "game_indoors", "temperature", "wind_speed", "precipitation", "condition", "has_fbs"]


def parse_weather_window(payload) -> pd.DataFrame:
    """CFBD `/games/weather` (forecast before kickoff, observation after) -> one row per game for the
    site's game_info table. Unlike `parse_weather_games` this keeps FBS-vs-FCS games (the site has pages
    for them): `has_fbs` says at least one side maps to an FBS id, and the caller drops the rest (CFBD returns
    every division). `venue_id` / `venue` come straight from the weather row (verified live: it carries both),
    `condition` is CFBD's weatherCondition text ("" when absent, which is most rows). CFBD sends no
    precipitation probability, only `precipitation` (inches), so no chance is derived here."""
    rows, dropped = [], 0
    for w in payload:
        ids = _ints(w, "id", "season", "week")
        if ids is None:
            dropped += 1
            continue
        h, a = cfbd_to_espn(w.get("homeTeam") or ""), cfbd_to_espn(w.get("awayTeam") or "")
        rows.append({"game_id": ids[0], "season": ids[1], "week": ids[2],
                     "season_type": season_type(w.get("seasonType")),
                     "start_time": str(w.get("startTime") or ""), "venue_id": num(w.get("venueId")),
                     "venue": str(w.get("venue") or ""), "game_indoors": bool(w.get("gameIndoors")),
                     "temperature": num(w.get("temperature")), "wind_speed": num(w.get("windSpeed")),
                     "precipitation": num(w.get("precipitation")),
                     "condition": str(w.get("weatherCondition") or ""), "has_fbs": bool(h or a)})
    return _frame(rows, WEATHER_WINDOW_COLUMNS, ints=("game_id", "season", "week"),
                  strs=("season_type", "start_time", "venue", "condition"),
                  bools=("game_indoors", "has_fbs"), dropped=dropped)
```

- [ ] **Step 4: Run the tests to verify they pass, plus the consumers of the games / venues frames**

Run: `uv run pytest -q tests/cfb/test_cfbd_games.py`
Expected: PASS (`39 passed` in this plan's dry run).

Run: `uv run pytest -q tests/cfb/test_cfbd_games.py tests/cfb/test_v3_data.py tests/cfb/test_context.py tests/cfb/test_data_checks.py tests/scripts/test_build_cfb_game_data.py`
Expected: PASS (`91 passed` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/cfb/cfbd_games.py tests/cfb/test_cfbd_games.py
git commit -m "feat(cfb): CFBD parsers for venue city/state, line scores, team turnovers, weather window

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 2: ESPN parsers: NFL venue/weather, CFB final line scores

**Files:**
- Modify: `src/sportsmodel/nfl/espn.py` (append `parse_game_info`, `fetch_game_info`)
- Modify: `src/sportsmodel/cfb/espn.py` (append `fetch_scoreboard`, `parse_line_scores`)
- Test: `tests/nfl/test_espn.py`, `tests/cfb/test_espn.py`

**Interfaces:**
- Consumes: existing `_get(path, params=None)` in both ESPN modules.
- Produces: `nfl.espn.parse_game_info(summary: dict) -> dict` with keys `venue_name, city, state, indoor (bool), temp_f, wind_mph, precip_chance, conditions` (every value `None` when ESPN has none; `indoor` True only when `venue.indoor is True`, which also nulls the weather keys); `nfl.espn.fetch_game_info(event_id: int) -> dict`; `cfb.espn.fetch_scoreboard(season: int, week: int, season_type: int = 2) -> dict` (raw payload); `cfb.espn.parse_line_scores(payload: dict) -> dict[int, dict]` = `{game_pk: {"home": [q1..q4], "away": [q1..q4]}}` for STATUS_FINAL events only.

- [ ] **Step 1: Write the failing tests**

The NFL venue fixture (`LIVE_VENUE`) is copied from the live summary of event 401872971 (2026-10-06). The weather-block keys in the other NFL tests are **assumed** (not verified live: the live finished game had no weather block) and the CFB `linescores` shape is assumed; Step 6 probes both against live ESPN and this task's fixtures are the only place to change if a key differs.

Append to the end of `tests/nfl/test_espn.py`:

```python
# ---------------------------------------------------------------- game info --

# Venue block copied from a LIVE finished-game summary (2026-10-06, event 401872971); it has no weather block.
LIVE_VENUE = {"id": "11938", "fullName": "Highmark Stadium",
              "address": {"city": "Orchard Park", "state": "NY", "zipCode": "14127", "country": "USA"},
              "grass": True, "images": []}


def test_parse_game_info_live_venue_shape_without_weather():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "attendance": 60606}})
    assert got == {"venue_name": "Highmark Stadium", "city": "Orchard Park", "state": "NY", "indoor": False,
                   "temp_f": None, "wind_mph": None, "precip_chance": None, "conditions": None}


def test_parse_game_info_weather_block_assumed_keys():
    from sportsmodel.nfl.espn import parse_game_info
    wx = {"temperature": 41, "displayValue": "Mostly Cloudy", "windSpeed": "14", "precipitation": 20, "gust": 30}
    got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "weather": wx}})
    assert (got["temp_f"], got["wind_mph"], got["precip_chance"], got["conditions"]) == (41.0, 14.0, 20.0, "Mostly Cloudy")


def test_parse_game_info_gust_is_not_wind_and_bad_values_are_none():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": LIVE_VENUE, "weather": {"temperature": "n/a", "gust": 30,
                                                                            "precipitation": 250, "displayValue": " "}}})
    assert (got["temp_f"], got["wind_mph"], got["precip_chance"], got["conditions"]) == (None, None, None, None)


def test_parse_game_info_indoor_venue_drops_weather():
    from sportsmodel.nfl.espn import parse_game_info
    got = parse_game_info({"gameInfo": {"venue": {**LIVE_VENUE, "indoor": True},
                                        "weather": {"temperature": 72, "displayValue": "Clear"}}})
    assert got["indoor"] is True and got["temp_f"] is None and got["conditions"] is None
    assert got["venue_name"] == "Highmark Stadium"


def test_parse_game_info_missing_blocks_never_raise():
    from sportsmodel.nfl.espn import parse_game_info
    for payload in ({}, {"gameInfo": None}, {"gameInfo": {"venue": None, "weather": []}}):
        got = parse_game_info(payload)
        assert got["venue_name"] is None and got["indoor"] is False and got["temp_f"] is None


def test_fetch_game_info_reads_the_summary_of_one_event(monkeypatch):
    from sportsmodel.nfl import espn
    seen = {}
    monkeypatch.setattr(espn, "_get", lambda path, params=None: seen.update(path=path, params=params) or {"gameInfo": {"venue": LIVE_VENUE}})
    assert espn.fetch_game_info(401872971)["venue_name"] == "Highmark Stadium"
    assert seen == {"path": "/summary", "params": {"event": 401872971}}
```

Append to the end of `tests/cfb/test_espn.py`:

```python
# ------------------------------------------------------------- line scores --

def _final_event(pk, home, away, status="STATUS_FINAL"):
    ls = lambda xs: [{"value": float(x), "displayValue": str(x)} for x in xs]  # noqa: E731
    return {"id": str(pk), "status": {"type": {"name": status}},
            "competitions": [{"competitors": [{"homeAway": "home", "linescores": ls(home)},
                                              {"homeAway": "away", "linescores": ls(away)}]}]}


def test_parse_line_scores_final_games_only_overtime_dropped():
    from sportsmodel.cfb.espn import parse_line_scores
    payload = {"events": [_final_event(1, [0, 10, 3, 24], [0, 3, 7, 0]),
                          _final_event(2, [7, 7, 7, 3, 7], [7, 7, 7, 3, 3]),
                          _final_event(3, [7, 7, 7, 3], [7, 7, 7, 3], status="STATUS_IN_PROGRESS"),
                          _final_event(4, [7, 7], [3, 3]),
                          {"id": "5"}]}
    assert parse_line_scores(payload) == {1: {"home": [0, 10, 3, 24], "away": [0, 3, 7, 0]},
                                          2: {"home": [7, 7, 7, 3], "away": [7, 7, 7, 3]}}
    assert parse_line_scores({}) == {}


def test_fetch_scoreboard_asks_for_the_fbs_group_and_returns_the_raw_payload(monkeypatch):
    from sportsmodel.cfb import espn
    seen = {}
    monkeypatch.setattr(espn, "_get", lambda path, params=None: seen.update(path=path, params=params) or {"events": []})
    assert espn.fetch_scoreboard(2026, 6, 2) == {"events": []}
    assert seen == {"path": "/scoreboard", "params": {"dates": 2026, "seasontype": 2, "week": 6, "groups": 80}}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/nfl/test_espn.py tests/cfb/test_espn.py`
Expected: FAIL — `FAILED tests/nfl/test_espn.py::test_parse_game_info_live_venue_shape_without_weather`.

- [ ] **Step 3: Implement the NFL game-info parser**

Append to the end of `src/sportsmodel/nfl/espn.py`:

```python
# ------------------------------------------------------------------ game info
# Weather keys read from `gameInfo.weather` (or a top-level `weather`). Verified live 2026-10-06 on a FINISHED
# game (event 401872971): `gameInfo.venue` = {id, fullName, address: {city, state, zipCode, country}, grass,
# images} -- no `indoor` key on an outdoor venue -- and NO weather block at all. NOT verified: the key names
# below (ESPN's usual upcoming-game block: temperature, displayValue, windSpeed?, precipitation = chance %)
# and whether a domed venue carries `indoor: true`. The Task 2 [NETWORK] probe prints an upcoming game's real
# block; if a key differs, change ONLY this mapping and the matching fixture in tests/nfl/test_espn.py.
_WX_TEMP, _WX_COND = ("temperature",), ("displayValue", "conditions")
_WX_WIND, _WX_PRECIP = ("windSpeed", "wind"), ("precipitation",)


def _first_num(block: dict, keys: tuple[str, ...]) -> float | None:
    for k in keys:
        v = block.get(k)
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v.strip().rstrip("%"))
            except ValueError:
                continue
    return None


def parse_game_info(summary) -> dict:
    """Venue + weather of one game from a /summary payload's `gameInfo` block (PURE).

    Returns {venue_name, city, state, indoor, temp_f, wind_mph, precip_chance, conditions}; every field is
    None when ESPN does not provide it (never guessed). `indoor` is True only when the venue says so
    (`venue.indoor is True`); then every weather field is None (the site shows "Indoors"). `wind_mph` reads
    a sustained wind key only -- `gust` is deliberately NOT used. `precip_chance` is ESPN's precipitation
    value when it lies in 0..100. ESPN gives no rainfall amount, so there is no precip_in here."""
    info = summary.get("gameInfo") if isinstance(summary.get("gameInfo"), dict) else {}
    venue = info.get("venue") if isinstance(info.get("venue"), dict) else {}
    addr = venue.get("address") if isinstance(venue.get("address"), dict) else {}
    wx = info.get("weather") if isinstance(info.get("weather"), dict) else (
        summary.get("weather") if isinstance(summary.get("weather"), dict) else {})
    text = lambda v: (str(v).strip() or None) if isinstance(v, str) else None  # noqa: E731
    indoor = venue.get("indoor") is True
    out = {"venue_name": text(venue.get("fullName")), "city": text(addr.get("city")),
           "state": text(addr.get("state")), "indoor": indoor,
           "temp_f": None, "wind_mph": None, "precip_chance": None, "conditions": None}
    if indoor:
        return out
    out["temp_f"] = _first_num(wx, _WX_TEMP)
    out["wind_mph"] = _first_num(wx, _WX_WIND)
    chance = _first_num(wx, _WX_PRECIP)
    out["precip_chance"] = chance if chance is not None and 0 <= chance <= 100 else None
    out["conditions"] = next((t[:40] for t in (text(wx.get(k)) for k in _WX_COND) if t), None)
    return out


def fetch_game_info(event_id: int) -> dict:
    return parse_game_info(_get("/summary", {"event": event_id}))
```

- [ ] **Step 4: Implement the CFB scoreboard + line-score parser**

Append to the end of `src/sportsmodel/cfb/espn.py`:

```python
def fetch_scoreboard(season: int, week: int, season_type: int = 2) -> dict:
    """The raw FBS scoreboard payload for one week (`fetch_schedule` parses the same call)."""
    return _get("/scoreboard", {"dates": season, "seasontype": season_type, "week": week, "groups": 80})


def parse_line_scores(payload) -> dict[int, dict]:
    """{game_pk: {"home": [q1, q2, q3, q4], "away": [...]}} for every STATUS_FINAL event of a scoreboard
    payload whose two competitors both carry at least four `linescores` periods (overtime is dropped).
    Anything else is omitted. NOT verified against a live payload (the shape read is ESPN's usual
    competitors[].linescores[].value); the Task 2 [NETWORK] probe confirms it."""
    out: dict[int, dict] = {}
    for ev in payload.get("events", []):
        try:
            if ev["status"]["type"]["name"] != "STATUS_FINAL":
                continue
            sides = {}
            for c in ev["competitions"][0]["competitors"]:
                vals = [p.get("value") for p in (c.get("linescores") or [])]
                if len(vals) < 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in vals[:4]):
                    raise ValueError("no usable line score")
                sides[c["homeAway"]] = [int(v) for v in vals[:4]]
            if set(sides) == {"home", "away"}:
                out[int(ev["id"])] = sides
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return out
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/nfl/test_espn.py tests/cfb/test_espn.py`
Expected: PASS (`28 passed` in this plan's dry run).

- [ ] **Step 6: [NETWORK] Probe the two unverified ESPN shapes and reconcile**

Both probes are free ESPN reads. Run them from the repo root. The NFL probe prints the venue / weather blocks of every game in the target week; the CFB probe prints one finished game's competitor `linescores`.

**[NETWORK] — run by the executing agent, not a subagent reviewer.**

```bash
uv run python - <<'PY'
import json
from sportsmodel.nfl import espn

cur = espn.target_week(espn.fetch_current_week())
games = espn.fetch_schedule(cur["season"], cur["week"], season_type=cur["season_type"])
print("week", cur, "games", len(games))
for g in games:                                   # one free /summary per game: the venue + weather blocks of the whole slate
    s = espn._get("/summary", {"event": g["game_pk"]})
    gi = s.get("gameInfo") or {}
    v = gi.get("venue") or {}
    print(g["game_pk"], g["status"], "|", v.get("fullName"), "| indoor key:", v.get("indoor"),
          "| gameInfo.weather:", json.dumps(gi.get("weather")), "| top-level weather:", json.dumps(s.get("weather")))
PY
```

**[NETWORK] — run by the executing agent, not a subagent reviewer.**

```bash
uv run python - <<'PY'
import json
from sportsmodel.cfb import espn

cur = espn.fetch_current_week()
payload = espn.fetch_scoreboard(cur["season"], max(1, cur["week"] - 1), cur["season_type"])
final = [e for e in payload["events"] if e["status"]["type"]["name"] == "STATUS_FINAL"]
print("events", len(payload["events"]), "final", len(final))
c = final[0]["competitions"][0]["competitors"][0]
print("competitor keys:", sorted(c))
print("linescores:", json.dumps(c.get("linescores")))
print("parse_line_scores found:", len(espn.parse_line_scores(payload)), "of", len(final))
PY
```

Reconcile, then re-run Step 5:

- NFL `gameInfo.weather` keys: if the printed block uses different key names than `temperature` / `displayValue` / `windSpeed` / `precipitation`, edit **only** the `_WX_TEMP`, `_WX_COND`, `_WX_WIND`, `_WX_PRECIP` tuples in `src/sportsmodel/nfl/espn.py` and the weather fixture in `test_parse_game_info_weather_block_assumed_keys` (rename that test to drop `_assumed_keys`). Do not map `gust` to wind. If `precipitation` is not a 0-100 percentage (e.g. inches), remove it from `_WX_PRECIP` (leave `()`), because the site would mislabel it.
- If some game prints `indoor key: True`, the dome flag works as assumed. If a domed stadium prints `None`, add its ESPN `fullName` to a module-level `_INDOOR_VENUES` frozenset in `nfl/espn.py` and treat `venue_name in _INDOOR_VENUES` as indoor in `parse_game_info`, with a test.
- If no game has a weather block yet (ESPN posts it close to kickoff), record `weather block still unverified` in the commit message: the parser degrades to `None` and the site hides the piece, so shipping is safe.
- CFB `linescores`: if the printed list is not `[{"value": n, ...}, ...]` (for example bare numbers), change the comprehension in `cfb.espn.parse_line_scores` and the `_final_event` helper in `tests/cfb/test_espn.py` together.

- [ ] **Step 7: Commit**

```bash
git add src/sportsmodel/nfl/espn.py src/sportsmodel/cfb/espn.py tests/nfl/test_espn.py tests/cfb/test_espn.py
git commit -m "feat(espn): NFL venue/weather parser and CFB scoreboard line scores

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 3: Backfill script (`team_stats` dataset) and the [NETWORK] data backfill

**Files:**
- Modify: `scripts/build_cfb_game_data.py` (new `team_stats` dataset, `played_weeks`)
- Test: `tests/scripts/test_build_cfb_game_data.py`
- Modify (data, committed): `assets/cfb/venues.parquet`, `assets/cfb/cfbd_games.parquet`, `assets/cfb/havoc_games.parquet`; Create (data): `assets/cfb/team_game_stats.parquet`

**Interfaces:**
- Consumes: T1 `cg.parse_team_game_stats`, `cg.parse_games_meta` (line scores), `cg.parse_venues` (city/state); existing `run(client, datasets, seasons, out_dir)`, `merge_asset`, `ORDER`.
- Produces: dataset name `team_stats` (`--datasets team_stats`) writing `assets/cfb/team_game_stats.parquet` keyed `[season, game_id, team]`; `played_weeks(meta: DataFrame | None, year: int) -> list[int]`; committed assets: `venues.parquet` with `city`/`state`, `cfbd_games.parquet` with `home_q1..away_ot` for 2024-2026, `team_game_stats.parquet` (2026 regular season), `havoc_games.parquet` through the latest week.

- [ ] **Step 1: Write the failing tests**

Append to the end of `tests/scripts/test_build_cfb_game_data.py`:

```python
def test_played_weeks_counts_only_finished_regular_weeks():
    meta = pd.DataFrame({"season": [2026] * 5 + [2025], "season_type": ["regular"] * 4 + ["postseason", "regular"],
                         "week": [1, 1, 2, 3, 1, 9], "home_points": [3.0, 7.0, 10.0, float("nan"), 14.0, 21.0]})
    assert bgd.played_weeks(meta, 2026) == [1, 2]
    assert bgd.played_weeks(meta, 2024) == [] and bgd.played_weeks(None, 2026) == []


def test_run_team_stats_pulls_each_played_week_and_writes_the_asset(tmp_path):
    games = [{"id": 400 + w, "season": 2026, "week": w, "seasonType": "regular", "homeTeam": "Alabama",
              "awayTeam": "Georgia", "homePoints": 20, "awayPoints": 10} for w in (1, 2)]
    teams = [{"id": 401, "teams": [{"team": "Alabama", "stats": [{"category": "turnovers", "stat": "1"}]},
                                   {"team": "Georgia", "stats": [{"category": "turnovers", "stat": "2"}]}]}]
    class WeekStub(StubClient):             # one distinct game id per requested week
        def get(self, path, params=None):
            out = super().get(path, params)
            return [{**g, "id": 400 + params["week"]} for g in out] if path == "/games/teams" else out

    client = WeekStub({"/games": games, "/games/teams": teams})
    bgd.run(client, ["games"], [2026], tmp_path)
    wrote = bgd.run(client, ["team_stats"], [2026], tmp_path)
    ts = pd.read_parquet(tmp_path / "team_game_stats.parquet")
    assert wrote["team_stats"] == len(ts) and set(ts["week"]) == {1, 2}
    assert [q["week"] for p, q in client.seen if p == "/games/teams"] == [1, 2]
    assert {q["seasonType"] for p, q in client.seen if p == "/games/teams"} == {"regular"}
    assert set(ts["giveaways"]) == {1.0, 2.0}


def test_run_team_stats_without_games_asset_stops(tmp_path):
    with pytest.raises(SystemExit):
        bgd.run(StubClient({}), ["team_stats"], [2026], tmp_path)


def test_run_team_stats_preseason_writes_an_empty_typed_asset(tmp_path):
    client = StubClient({"/games": [{"id": 1, "season": 2026, "week": 1, "seasonType": "regular",
                                     "homeTeam": "Alabama", "awayTeam": "Georgia"}]})
    bgd.run(client, ["games"], [2026], tmp_path)
    assert bgd.run(client, ["team_stats"], [2026], tmp_path)["team_stats"] == 0
    assert not [p for p, _ in client.seen if p == "/games/teams"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/scripts/test_build_cfb_game_data.py`
Expected: FAIL — `FAILED tests/scripts/test_build_cfb_game_data.py::test_played_weeks_counts_only_finished_regular_weeks`.

- [ ] **Step 3: Implement the dataset**

In `scripts/build_cfb_game_data.py`, replace this text (it occurs exactly once):

```python
  venues   /venues             -> venues.parquet         static lat/lon/timezone/dome
```

with:

```python
  venues   /venues             -> venues.parquet         static lat/lon/timezone/dome + city/state
  team_stats /games/teams      -> team_game_stats.parquet per team-game giveaways / takeaways (regular season; one
                                                           call per PLAYED week, so pass a single current season)
```

In `scripts/build_cfb_game_data.py`, replace this text (it occurs exactly once):

```python
         "ratings": "prior_ratings.parquet", "venues": "venues.parquet"}
```

with:

```python
         "ratings": "prior_ratings.parquet", "venues": "venues.parquet",
         "team_stats": "team_game_stats.parquet"}
```

In `scripts/build_cfb_game_data.py`, replace this text (it occurs exactly once):

```python
        "talent": ["season", "team"], "ratings": ["season", "team"], "venues": ["venue_id"]}
ORDER = ["games", "havoc", "drives", "weather", "talent", "ratings", "venues"]
```

with:

```python
        "talent": ["season", "team"], "ratings": ["season", "team"], "venues": ["venue_id"],
        "team_stats": ["season", "game_id", "team"]}
ORDER = ["games", "havoc", "drives", "weather", "talent", "ratings", "venues", "team_stats"]
```

In `scripts/build_cfb_game_data.py`, replace this text (it occurs exactly once):

```python
def _pull(client, dataset: str, year: int, meta: pd.DataFrame | None) -> pd.DataFrame:
```

with:

```python
def played_weeks(meta: pd.DataFrame | None, year: int) -> list[int]:
    """Regular-season weeks of `year` with at least one finished game in the committed `games` asset (its
    final points are not NaN) -- the weeks worth a /games/teams call."""
    if meta is None or meta.empty:
        return []
    d = meta[(meta["season"] == year) & (meta["season_type"] == "regular") & meta["home_points"].notna()]
    return sorted(int(w) for w in d["week"].unique())


def _pull(client, dataset: str, year: int, meta: pd.DataFrame | None) -> pd.DataFrame:
```

In `scripts/build_cfb_game_data.py`, replace this text (it occurs exactly once):

```python
    elif dataset == "ratings":
        parts = [cg.parse_prior_ratings(
```

with:

```python
    elif dataset == "team_stats":
        if meta is None or meta.empty:
            raise SystemExit("team_stats needs cfbd_games.parquet (run the `games` dataset first)")
        parts = [cg.parse_team_game_stats(client.get("/games/teams", {"year": year, "week": w, "seasonType": "regular"}),
                                          year, w, "regular") for w in played_weeks(meta, year)]
        parts = parts or [cg.parse_team_game_stats([], year, 0, "regular")]    # preseason: an empty, typed frame
    elif dataset == "ratings":
        parts = [cg.parse_prior_ratings(
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/scripts/test_build_cfb_game_data.py tests/cfb/test_cfbd_games.py`
Expected: PASS (`58 passed` in this plan's dry run).

- [ ] **Step 5: [NETWORK] Backfill the assets (about 15-20 CFBD calls)**

The executing agent runs this (it needs `CFBD_API_KEY` from `.env`). Each command prints `CFBD calls this run: N`; the four together must stay at or under about 20 calls. `team_stats` must be run on its own with `--seasons 2026`: it makes one call per played week of every season passed.

**[NETWORK] — run by the executing agent, not a subagent reviewer.**

```bash
set -a; source .env; set +a            # loads CFBD_API_KEY into this shell; never print or echo it
uv run python scripts/build_cfb_game_data.py --datasets venues                          # 1 call: adds city / state
uv run python scripts/build_cfb_game_data.py --datasets games --seasons 2024 2025 2026   # 6 calls: line scores (regular + postseason per season)
uv run python scripts/build_cfb_game_data.py --datasets team_stats --seasons 2026        # one call per PLAYED 2026 regular-season week (about 5-6)
uv run python scripts/build_cfb_game_data.py --datasets havoc --seasons 2026             # 2 calls: bring 2026 havoc up to the latest week
```

- [ ] **Step 6: Sanity-check the written assets (local, no network)**

Every assertion must print without raising. If one fails, stop: fix the parser (not the assertion), re-pull and do not commit the parquet.

```bash
uv run python - <<'PY'
import pandas as pd

g = pd.read_parquet("assets/cfb/cfbd_games.parquet")
r = g[g.season.isin([2024, 2025, 2026]) & g.home_points.notna() & g.away_points.notna()]
has = r.home_q1.notna() & r.away_q1.notna()
print("finished games", len(r), "with line scores", round(has.mean(), 3))
assert has.mean() >= 0.97, "line scores missing for too many finished games"
for side in ("home", "away"):
    tot = sum(r[f"{side}_q{q}"] for q in (1, 2, 3, 4)) + r[f"{side}_ot"]
    ok = ((tot - r[f"{side}_points"]).abs() < 0.5)[has].mean()
    print(side, "quarters + OT == final points:", round(ok, 3))
    assert ok >= 0.98, "line scores do not add up to the final score"
assert g[g.season < 2024].home_q1.isna().all(), "older seasons were not re-pulled and must stay NaN"

v = pd.read_parquet("assets/cfb/venues.parquet")
print("venues", len(v), "without a city:", int((v.city == "").sum()), "without a state:", int((v.state == "").sum()))
assert (v.city != "").mean() > 0.98 and v.loc[v.name == "Bryant-Denny Stadium", "city"].iloc[0] == "Tuscaloosa"

t = pd.read_parquet("assets/cfb/team_game_stats.parquet")
print("team_stats rows", len(t), "weeks", sorted(t.week.unique()), "giveaways mean", round(t.giveaways.mean(), 2))
assert set(t.season) == {2026} and set(t.season_type) == {"regular"}
assert 0.8 <= t.giveaways.mean() <= 2.0, "giveaways per team-game outside the plausible range"
assert t.giveaways.isna().mean() < 0.02
pair = t.merge(t, left_on=["game_id", "opponent"], right_on=["game_id", "team"], suffixes=("", "_o"))
assert (pair.takeaways == pair.giveaways_o).all(), "takeaways must equal the opponent's giveaways"

h = pd.read_parquet("assets/cfb/havoc_games.parquet")
a = pd.read_parquet("assets/cfb/advanced_games.parquet")
print("2026 max week: havoc", h[h.season == 2026].week.max(), "advanced", a[a.season == 2026].week.max())
print("OK")
PY
```

Run: `uv run python scripts/check_cfb_game_data.py`
Expected: prints `OK` (the existing weather / havoc / drives checks still hold with the refreshed rows).

Run: `uv run pytest -q tests/cfb tests/scripts/test_build_cfb_game_data.py`
Expected: PASS (`355 passed` in this plan's dry run). (about 1-2 minutes; the new columns must not break any consumer of the games / venues frames)

- [ ] **Step 7: Commit the code and the parquet**

```bash
git add scripts/build_cfb_game_data.py tests/scripts/test_build_cfb_game_data.py assets/cfb/venues.parquet assets/cfb/cfbd_games.parquet assets/cfb/team_game_stats.parquet assets/cfb/havoc_games.parquet
git commit -m "feat(cfb): team_stats dataset + backfill (venue city/state, line scores 2024-26, 2026 turnovers, havoc refresh)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 4: Quarter shares builder (`cfb_quarter_shares`)

**Files:**
- Create: `src/sportsmodel/cfb/panels.py` (`quarter_shares`, `to_rows`)
- Test: `tests/cfb/test_panels.py` (create)

**Interfaces:**
- Consumes: `assets/cfb/cfbd_games.parquet` frame from T1 / T3: columns `season, season_type, home_team, away_team, home_q1..home_q4, away_q1..away_q4` (ESPN team ids as strings).
- Produces: `panels.SHARE_K = 12.0`, `panels.SHARE_SEASONS_BACK = 2`, `panels.QUARTERS`, `panels.SHARE_COLUMNS` (`team, scored_q1..scored_q4, allowed_q1..allowed_q4, games_used`); `panels.quarter_shares(games: DataFrame, season: int, *, k: float = SHARE_K, fbs: set[str] | None = None) -> DataFrame` (one row per team, shares of each side sum to 1, OT excluded, seasons `season-2..season`, shrunk to the league with `w = n / (n + k)`); `panels.to_rows(df: DataFrame) -> list[dict]` (numpy -> python, NaN/NA -> `None`, whole-number rank/count columns -> `int`).

- [ ] **Step 1: Write the failing tests**

The expected numbers are worked by hand in the tests: a team's raw share is its points in a quarter over its points in all four; the shrunk share is `w * raw + (1 - w) * league` with `w = n / (n + 12)` and the league share is the all-team-games share of each quarter.

Create `tests/cfb/test_panels.py`:

```python
import math

import pandas as pd
import pytest

from sportsmodel.cfb import panels


def _game(season, home, away, hq, aq):
    return {"season": season, "season_type": "regular", "home_team": home, "away_team": away,
            **{f"home_q{i + 1}": float(v) for i, v in enumerate(hq)},
            **{f"away_q{i + 1}": float(v) for i, v in enumerate(aq)}}


def test_quarter_shares_sum_to_one_and_follow_the_pattern():
    games = pd.DataFrame([_game(2025, "A", "B", [14, 0, 0, 0], [0, 7, 0, 0]),
                          _game(2025, "A", "C", [7, 0, 0, 0], [0, 0, 7, 0]),
                          _game(2026, "B", "C", [0, 0, 0, 7], [0, 0, 0, 7])])
    sh = panels.quarter_shares(games, 2026).set_index("team")
    assert list(sh.index) == ["A", "B", "C"] and list(sh["games_used"]) == [2, 2, 2]
    for t in sh.index:
        assert sum(sh.loc[t, f"scored_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
        assert sum(sh.loc[t, f"allowed_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
    lq = [21 / 49, 7 / 49, 7 / 49, 14 / 49]          # league scored points per quarter over all six team-games: 21 / 7 / 7 / 14 of 49
    w = 2 / 14                                                           # n / (n + 12)
    assert sh.loc["A", "scored_q1"] == pytest.approx(w * 1.0 + (1 - w) * lq[0])
    assert sh.loc["A", "scored_q2"] == pytest.approx((1 - w) * lq[1])
    assert sh.loc["A", "allowed_q2"] == pytest.approx(w * 0.5 + (1 - w) * lq[1])


def test_quarter_shares_small_n_is_close_to_the_league_and_large_n_to_the_team():
    big = [_game(2026, "A", "B", [10, 0, 0, 0], [0, 0, 0, 10]) for _ in range(120)]
    one = [_game(2026, "C", "D", [0, 10, 0, 0], [0, 0, 10, 0])]
    sh = panels.quarter_shares(pd.DataFrame(big + one), 2026).set_index("team")
    lg = 10 / 2420                                   # league share of q2: only C's single game scored there (10 of 2,420 points)
    assert sh.loc["A", "scored_q1"] == pytest.approx(120 / 132 * 1.0 + 12 / 132 * (1200 / 2420))      # 120 games: w = 120 / 132
    assert sh.loc["C", "scored_q2"] == pytest.approx(1 / 13 + 12 / 13 * lg)                          # one game: w = 1 / 13, so mostly league
    assert sh.loc["C", "games_used"] == 1


def test_quarter_shares_window_missing_quarters_fbs_filter_and_empty():
    games = pd.DataFrame([_game(2023, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]),          # too old (2026 - 2 = 2024)
                          _game(2025, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]),
                          {**_game(2025, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7]), "home_q3": float("nan")}])
    sh = panels.quarter_shares(games, 2026)
    assert list(sh["games_used"]) == [1, 1]
    assert list(panels.quarter_shares(games, 2026, fbs={"A"})["team"]) == ["A"]
    assert panels.quarter_shares(games.iloc[0:1], 2026).empty
    assert list(panels.quarter_shares(games.iloc[0:0], 2026).columns) == panels.SHARE_COLUMNS


def test_quarter_shares_a_team_that_never_scored_falls_back_to_the_league():
    games = pd.DataFrame([_game(2026, "A", "B", [0, 0, 0, 0], [7, 7, 7, 7]), _game(2026, "A", "C", [0, 0, 0, 0], [3, 3, 3, 3])])
    sh = panels.quarter_shares(games, 2026).set_index("team")
    assert sum(sh.loc["A", f"scored_q{q}"] for q in (1, 2, 3, 4)) == pytest.approx(1.0)
    assert sh.loc["A", "scored_q1"] == pytest.approx(0.25)


def test_to_rows_python_scalars_and_none():
    rows = panels.to_rows(panels.quarter_shares(pd.DataFrame([_game(2026, "A", "B", [7, 7, 7, 7], [7, 7, 7, 7])]), 2026))
    assert type(rows[0]["games_used"]) is int and type(rows[0]["scored_q1"]) is float and rows[0]["team"] == "A"
    assert panels.to_rows(pd.DataFrame({"x": [float("nan"), 1.5], "n_ranked": [3.0, None]})) == [{"x": None, "n_ranked": 3}, {"x": 1.5, "n_ranked": None}]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/cfb/test_panels.py`
Expected: FAIL — `ERROR tests/cfb/test_panels.py`.

- [ ] **Step 3: Implement `quarter_shares` and `to_rows`**

Create `src/sportsmodel/cfb/panels.py`:

```python
"""CFB site-panel tables (pure): per-team quarter scoring shares and season-to-date insight ranks.

* `quarter_shares` -> `cfb_quarter_shares` (db/migration_site_panels.sql): the share of a team's points
  SCORED and ALLOWED in Q1-Q4 over the last two completed seasons plus the current one to date, shrunk toward
  the league average with weight n / (n + k) (n = games used, k = SHARE_K). Overtime is excluded; each
  vector of four shares sums to 1. The site's Projected Game Flow multiplies a team's projected score by
  the average of its own scored shares and its opponent's allowed shares.
* `team_insights` -> `cfb_team_insights`: current-season, regular-season, FBS-vs-FBS havoc / turnover /
  explosiveness rates with national ranks (1 = best) among teams with at least MIN_GAMES games (added with it).

Inputs are the committed assets under assets/cfb/ (team ids are ESPN ids as strings).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

SHARE_K = 12.0                 # games of shrinkage: a team with n games keeps n / (n + 12) of its own pattern
SHARE_SEASONS_BACK = 2         # completed seasons before the current one that feed the shares
QUARTERS = (1, 2, 3, 4)
SHARE_COLUMNS = (["team"] + [f"scored_q{q}" for q in QUARTERS] + [f"allowed_q{q}" for q in QUARTERS]
                 + ["games_used"])


def quarter_shares(games: pd.DataFrame, season: int, *, k: float = SHARE_K,
                   fbs: set[str] | None = None) -> pd.DataFrame:
    """One row per team (columns SHARE_COLUMNS) from the cfbd_games frame.

    Games of seasons season-2..season (regular + postseason) with all of home/away q1..q4 present are used;
    the rest are skipped. Raw share = a team's points in a quarter / its points in all four quarters
    (scored) and the same for the points its opponents scored on it (allowed); the shrunk share is
    w * raw + (1 - w) * league with w = n / (n + k). The league share is the all-team-games share, the same
    for scored and allowed. A team with no usable game is absent; with `fbs`, only those teams are kept."""
    cols = [f"{s}_q{q}" for s in ("home", "away") for q in QUARTERS]
    g = games[games["season"].between(season - SHARE_SEASONS_BACK, season)]
    g = g.dropna(subset=cols)
    if g.empty:
        return pd.DataFrame(columns=SHARE_COLUMNS)
    home = pd.DataFrame({"team": g["home_team"].astype(str)})
    away = pd.DataFrame({"team": g["away_team"].astype(str)})
    for q in QUARTERS:
        home[f"scored_q{q}"], home[f"allowed_q{q}"] = g[f"home_q{q}"].to_numpy(), g[f"away_q{q}"].to_numpy()
        away[f"scored_q{q}"], away[f"allowed_q{q}"] = g[f"away_q{q}"].to_numpy(), g[f"home_q{q}"].to_numpy()
    tg = pd.concat([home, away], ignore_index=True)
    sc, al = [f"scored_q{q}" for q in QUARTERS], [f"allowed_q{q}" for q in QUARTERS]
    league = tg[sc].sum().to_numpy(dtype=float)
    league = league / league.sum()                       # all-team-games share of each quarter
    agg = tg.groupby("team")[sc + al].sum()
    agg["games_used"] = tg.groupby("team").size()
    if fbs is not None:
        agg = agg[agg.index.isin(fbs)]
    rows = []
    for team, r in agg.iterrows():
        n = float(r["games_used"])
        w = n / (n + k)
        out = {"team": str(team), "games_used": int(n)}
        for names in (sc, al):
            tot = float(r[names].sum())
            raw = r[names].to_numpy(dtype=float) / tot if tot > 0 else league
            shrunk = w * raw + (1 - w) * league
            out.update(dict(zip(names, (float(x) for x in shrunk / shrunk.sum()))))
        rows.append(out)
    return pd.DataFrame(rows, columns=SHARE_COLUMNS).sort_values("team").reset_index(drop=True)


def to_rows(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> DB-ready python rows: numpy scalars to int / float / str, NaN / NA to None. Rank and
    count columns (whole numbers) come back as int."""
    out = []
    for rec in df.to_dict("records"):
        row = {}
        for c, v in rec.items():
            if v is None or v is pd.NA or (isinstance(v, float) and math.isnan(v)):
                row[c] = None
            elif isinstance(v, (np.integer, int)) and not isinstance(v, bool):
                row[c] = int(v)
            elif isinstance(v, (np.floating, float)):
                row[c] = int(v) if (c.endswith("_rank") or c in ("games", "games_used", "n_ranked", "through_week", "season")) else float(v)
            else:
                row[c] = v
        out.append(row)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/cfb/test_panels.py`
Expected: PASS (`5 passed` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/cfb/panels.py tests/cfb/test_panels.py
git commit -m "feat(cfb): quarter scoring shares builder

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 5: Team insights builder (`cfb_team_insights`)

**Files:**
- Modify: `src/sportsmodel/cfb/panels.py` (append `team_insights`)
- Test: `tests/cfb/test_panels.py` (append)

**Interfaces:**
- Consumes: `havoc_games.parquet` (`season, week, season_type, team, def_havoc_events, def_plays, off_havoc_events, off_plays`: `off_*` is the havoc the offense suffered, `def_*` the havoc the defense created), `advanced_games.parquet` (`off_explosiveness, off_plays, def_explosiveness, def_plays`: the defense value is explosiveness ALLOWED), `team_game_stats.parquet` from T3 (`giveaways, takeaways`) or `None`; T4 `panels.to_rows`.
- Produces: `panels.MIN_GAMES = 3`, `panels.INSIGHT_COLUMNS` (`season, team, games, def_havoc_rate, def_havoc_rank, off_havoc_allowed_rate, off_havoc_allowed_rank, turnover_margin, turnover_margin_per_game, turnover_margin_rank, off_explosiveness, off_explosiveness_rank, def_explosiveness_allowed, def_explosiveness_allowed_rank, n_ranked, through_week`); `panels.team_insights(havoc: DataFrame, advanced: DataFrame, team_stats: DataFrame | None, season: int, *, min_games: int = MIN_GAMES, fbs: set[str] | None = None) -> DataFrame` (regular season, weeks <= the last havoc week; ranks 1 = best among teams with >= `min_games` games, ranks NaN otherwise; `n_ranked` = size of the ranked field).

- [ ] **Step 1: Write the failing tests**

The fixture is a four-team round robin over three weeks (teams 1-4, each with a known per-100-play havoc rate, turnover and explosiveness ordering) plus team 5 / 6 with two games (so they carry values but no rank).

Append to the end of `tests/cfb/test_panels.py`:

```python
def _h(team, opp, week, gid, def_ev, def_pl, off_ev, off_pl):
    return {"season": 2026, "week": week, "season_type": "regular", "game_id": gid, "team": team, "opponent": opp,
            "def_havoc_events": def_ev, "def_plays": def_pl, "off_havoc_events": off_ev, "off_plays": off_pl}


def _adv(team, week, gid, off_x, off_pl, def_x, def_pl):
    return {"season": 2026, "week": week, "season_type": "regular", "game_id": gid, "team": team,
            "off_explosiveness": off_x, "off_plays": off_pl, "def_explosiveness": def_x, "def_plays": def_pl}


def _fixture():
    """Teams 1..4, three weeks each (weeks 1-3), plus team 5 with two games. Havoc created: 1 > 2 > 3 > 4."""
    pairs = {1: [(1, 2), (3, 4)], 2: [(1, 3), (2, 4)], 3: [(1, 4), (2, 3)]}
    created = {"1": 20, "2": 15, "3": 10, "4": 5}          # per 100 plays
    allowed = {"1": 5, "2": 10, "3": 15, "4": 20}
    hv, ad, ts, gid = [], [], [], 0
    for w, ps in pairs.items():
        for a, b in ps:
            gid += 1
            for t, o in ((str(a), str(b)), (str(b), str(a))):
                hv.append(_h(t, o, w, gid, created[t], 100, allowed[t], 100))
                ad.append(_adv(t, w, gid, 0.1 * int(t), 50 + int(t), 0.5 - 0.1 * int(t), 60))
                ts.append({"season": 2026, "week": w, "season_type": "regular", "game_id": gid, "team": t,
                           "opponent": o, "giveaways": float(int(t)), "takeaways": float(5 - int(t))})
    for w in (1, 2):                                          # team 5 plays itself... a second opponent pool
        gid += 1
        hv.append(_h("5", "6", w, gid, 99, 100, 1, 100)); hv.append(_h("6", "5", w, gid, 1, 100, 99, 100))
    return pd.DataFrame(hv), pd.DataFrame(ad), pd.DataFrame(ts)


def test_team_insights_rates_ranks_and_n_ranked():
    hv, ad, ts = _fixture()
    ins = panels.team_insights(hv, ad, ts, 2026).set_index("team")
    assert ins.loc["1", "games"] == 3 and ins.loc["1", "through_week"] == 3
    assert ins.loc["1", "def_havoc_rate"] == pytest.approx(0.20)
    assert ins.loc["1", "off_havoc_allowed_rate"] == pytest.approx(0.05)
    assert [int(ins.loc[t, "def_havoc_rank"]) for t in "1234"] == [1, 2, 3, 4]             # higher created = better
    assert [int(ins.loc[t, "off_havoc_allowed_rank"]) for t in "1234"] == [1, 2, 3, 4]     # lower allowed = better
    assert ins.loc["1", "turnover_margin"] == pytest.approx(3 * (4 - 1))                  # takeaways 4, giveaways 1 per game
    assert ins.loc["1", "turnover_margin_per_game"] == pytest.approx(3.0)
    assert [int(ins.loc[t, "turnover_margin_rank"]) for t in "1234"] == [1, 2, 3, 4]
    assert [int(ins.loc[t, "off_explosiveness_rank"]) for t in "1234"] == [4, 3, 2, 1]     # 0.1 * team id
    assert [int(ins.loc[t, "def_explosiveness_allowed_rank"]) for t in "1234"] == [4, 3, 2, 1]   # 0.5 - 0.1 * id: lower allowed = better
    assert set(ins["n_ranked"]) == {4}
    for t in ("5", "6"):                                                                    # two games: values yes, ranks no
        assert ins.loc[t, "games"] == 2 and pd.isna(ins.loc[t, "def_havoc_rank"]) and pd.isna(ins.loc[t, "turnover_margin_rank"])
    assert ins.loc["5", "def_havoc_rate"] == pytest.approx(0.99)
    assert math.isnan(ins.loc["5", "turnover_margin"])                                      # no team_stats rows -> NaN, never 0
    assert list(panels.team_insights(hv, ad, None, 2026).loc[:, "turnover_margin"].isna()) == [True] * 6


def test_team_insights_filters_season_postseason_and_weeks_beyond_havoc():
    hv, ad, ts = _fixture()
    post = hv.iloc[:2].assign(season_type="postseason", week=9)
    old = hv.iloc[:2].assign(season=2025)
    ad2 = pd.concat([ad, ad.iloc[:1].assign(week=8)], ignore_index=True)        # advanced is ahead of havoc: ignored
    got = panels.team_insights(pd.concat([hv, post, old]), ad2, ts, 2026)
    assert got["through_week"].eq(3).all() and got.set_index("team").loc["1", "games"] == 3
    assert panels.team_insights(hv, ad, ts, 2030).empty
    assert list(panels.team_insights(hv.iloc[0:0], ad, ts, 2026).columns) == panels.INSIGHT_COLUMNS
    assert set(panels.team_insights(hv, ad, ts, 2026, fbs={"1", "2"})["team"]) == {"1", "2"}


def test_to_rows_insight_ranks_are_ints_and_missing_values_are_none():
    hv, ad, ts = _fixture()
    rows = panels.to_rows(panels.team_insights(hv, ad, ts, 2026))
    r5 = next(r for r in rows if r["team"] == "5")
    assert r5["def_havoc_rank"] is None and r5["turnover_margin"] is None and r5["games"] == 2
    r1 = next(r for r in rows if r["team"] == "1")
    assert type(r1["def_havoc_rank"]) is int and r1["def_havoc_rank"] == 1 and type(r1["def_havoc_rate"]) is float
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/cfb/test_panels.py`
Expected: FAIL — `FAILED tests/cfb/test_panels.py::test_team_insights_rates_ranks_and_n_ranked`.

- [ ] **Step 3: Implement `team_insights`**

Append to the end of `src/sportsmodel/cfb/panels.py`:

```python
MIN_GAMES = 3                  # a team needs this many games to be ranked
INSIGHT_COLUMNS = ["season", "team", "games", "def_havoc_rate", "def_havoc_rank", "off_havoc_allowed_rate",
                   "off_havoc_allowed_rank", "turnover_margin", "turnover_margin_per_game",
                   "turnover_margin_rank", "off_explosiveness", "off_explosiveness_rank",
                   "def_explosiveness_allowed", "def_explosiveness_allowed_rank", "n_ranked", "through_week"]


def _wmean(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean ignoring NaN values; plain mean when the weights are unusable; NaN when no value."""
    v = values.astype(float)
    ok = v.notna()
    if not ok.any():
        return float("nan")
    w = weights.astype(float).where(ok)
    if w.notna().any() and w[ok].sum() > 0 and not w[ok].isna().any():
        return float((v[ok] * w[ok]).sum() / w[ok].sum())
    return float(v[ok].mean())


def _rank(s: pd.Series, eligible: pd.Series, *, best_high: bool) -> pd.Series:
    """1 = best among eligible teams (ties share the lowest rank); NaN for the rest."""
    r = s.where(eligible).rank(method="min", ascending=not best_high)
    return r.astype("Float64").where(r.notna(), pd.NA)


def team_insights(havoc: pd.DataFrame, advanced: pd.DataFrame, team_stats: pd.DataFrame | None,
                  season: int, *, min_games: int = MIN_GAMES, fbs: set[str] | None = None) -> pd.DataFrame:
    """One row per team (columns INSIGHT_COLUMNS) for `season`, regular season, through the last week that
    has havoc data. `havoc` = havoc_games (off_* = havoc the offense suffered, def_* = havoc the defense
    created), `advanced` = advanced_games (def_explosiveness = explosiveness ALLOWED), `team_stats` =
    team_game_stats (giveaways / takeaways; may be None -> turnover columns NULL). Rates are play-weighted
    season totals. Ranks (1 = best): defensive havoc created high, havoc allowed low, turnover margin per
    game high, explosiveness high (offense) / low (allowed). Teams under `min_games` games keep their
    values but get no rank; `n_ranked` is the size of the ranked field."""
    def reg(df):
        d = df[(df["season"] == season) & (df["season_type"].fillna("regular") == "regular")]
        return d[d["team"].astype(str).isin(fbs)] if fbs is not None else d
    h = reg(havoc)
    if h.empty:
        return pd.DataFrame(columns=INSIGHT_COLUMNS)
    through = int(h["week"].max())
    h = h[h["week"] <= through]
    a = reg(advanced)
    a = a[a["week"] <= through]
    ts = reg(team_stats) if team_stats is not None and len(team_stats) else None
    if ts is not None:
        ts = ts[ts["week"] <= through].dropna(subset=["giveaways", "takeaways"])

    rows = []
    for team, d in h.groupby(h["team"].astype(str)):
        def rate(ev, pl):
            plays = float(d[pl].sum())
            return float(d[ev].sum()) / plays if plays > 0 else float("nan")
        da = a[a["team"].astype(str) == team]
        row = {"season": int(season), "team": team, "games": int(len(d)),
               "def_havoc_rate": rate("def_havoc_events", "def_plays"),
               "off_havoc_allowed_rate": rate("off_havoc_events", "off_plays"),
               "off_explosiveness": _wmean(da["off_explosiveness"], da["off_plays"]) if len(da) else float("nan"),
               "def_explosiveness_allowed": _wmean(da["def_explosiveness"], da["def_plays"]) if len(da) else float("nan"),
               "turnover_margin": float("nan"), "turnover_margin_per_game": float("nan"),
               "through_week": through}
        if ts is not None:
            dt = ts[ts["team"].astype(str) == team]
            if len(dt):
                row["turnover_margin"] = float((dt["takeaways"] - dt["giveaways"]).sum())
                row["turnover_margin_per_game"] = row["turnover_margin"] / len(dt)
        rows.append(row)
    df = pd.DataFrame(rows)
    ok = df["games"] >= min_games
    for col, src, high in (("def_havoc_rank", "def_havoc_rate", True),
                           ("off_havoc_allowed_rank", "off_havoc_allowed_rate", False),
                           ("turnover_margin_rank", "turnover_margin_per_game", True),
                           ("off_explosiveness_rank", "off_explosiveness", True),
                           ("def_explosiveness_allowed_rank", "def_explosiveness_allowed", False)):
        df[col] = _rank(df[src], ok, best_high=high)
    df["n_ranked"] = int(ok.sum())
    return df[INSIGHT_COLUMNS].sort_values("team").reset_index(drop=True)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/cfb/test_panels.py`
Expected: PASS (`8 passed` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/cfb/panels.py tests/cfb/test_panels.py
git commit -m "feat(cfb): team insights (havoc, turnover margin, explosiveness) with national ranks

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 6: `game_info` row builders (CFB + NFL)

**Files:**
- Create: `src/sportsmodel/game_info.py`
- Test: `tests/test_game_info.py` (create)

**Interfaces:**
- Consumes: T1 `parse_weather_window` frame; T2 `cfb.espn.parse_line_scores` dict and `nfl.espn.parse_game_info` dict; ESPN schedule dicts from `cfb.espn.parse_schedule` / `nfl.espn.parse_schedule` (`game_pk`, `commence_time`); `assets/cfb/venues.parquet` (`venue_id, name, city, state, dome`) from T3.
- Produces: `game_info.WINDOW_BEFORE` (2 days), `WINDOW_AFTER` (10 days), `WEATHER_FIELDS`; `utc(ts) -> pd.Timestamp`; `in_window(kickoff, now) -> bool` (inclusive, False when unreadable); `cfb_game_info(games: list[dict], line_scores: dict[int, dict], weather: DataFrame, venues: DataFrame | None, now) -> list[dict]`; `nfl_game_info(games: list[dict], fetch_info: Callable[[int], dict], now, warn=lambda m: None) -> list[dict]`. Row keys (exactly `db.SITE_PANEL_COLUMNS["game_info"]`): `sport, game_pk, venue_name, city, state, indoor, temp_f, wind_mph, precip_chance, precip_in, conditions, weather_kind, source, captured_at, line_score`.

- [ ] **Step 1: Write the failing tests**

Cases: window bounds; CFB forecast / indoor (weather NULL) / observed (kicked off) / venue missing from `venues.parquet` / FBS-less game skipped / game with no inputs skipped / line-score-only row; NFL window, duplicate game, weather present vs absent, indoor, a failing summary call.

Create `tests/test_game_info.py`:

```python
import pandas as pd
import pytest

from sportsmodel import game_info as gi
from sportsmodel.cfb.cfbd_games import parse_weather_window

NOW = pd.Timestamp("2026-10-07T14:00:00Z")


def _g(pk, kick):
    return {"game_pk": pk, "commence_time": kick}


def test_in_window_bounds_and_junk():
    assert gi.in_window("2026-10-05T14:00:00Z", NOW) and gi.in_window("2026-10-17T14:00:00Z", NOW)
    assert not gi.in_window("2026-10-05T13:59:00Z", NOW) and not gi.in_window("2026-10-17T14:01:00Z", NOW)
    assert gi.in_window("2026-10-10T00:00", NOW)                     # naive -> UTC
    assert not gi.in_window("garbage", NOW) and not gi.in_window(None, NOW)


WEATHER = parse_weather_window([
    {"id": 1, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z", "gameIndoors": False,
     "homeTeam": "Alabama", "awayTeam": "Georgia", "venueId": 3657, "venue": "Bryant-Denny Stadium",
     "temperature": 41.2, "windSpeed": 14.0, "precipitation": 0.04, "weatherCondition": "Cloudy"},
    {"id": 2, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-08T23:30:00.000Z", "gameIndoors": True,
     "homeTeam": "UTSA", "awayTeam": "South Florida", "venueId": 3604, "venue": "Alamodome",
     "temperature": 82.8, "windSpeed": 9.2, "precipitation": 0, "weatherCondition": "Fair"},
    {"id": 3, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-06T00:00:00.000Z", "gameIndoors": False,
     "homeTeam": "Alabama", "awayTeam": "Mercer", "venueId": 9999, "venue": "Unlisted Field",
     "temperature": 70, "windSpeed": 5, "precipitation": 0},
    {"id": 4, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z", "gameIndoors": False,
     "homeTeam": "Bridgewater State", "awayTeam": "Framingham State", "venueId": 5808, "venue": "Swenson",
     "temperature": 66, "windSpeed": 10.1, "precipitation": 0}])
VENUES = pd.DataFrame({"venue_id": [3657, 3604], "name": ["Bryant-Denny Stadium", "Alamodome"],
                       "city": ["Tuscaloosa", "San Antonio"], "state": ["AL", "TX"], "dome": [0.0, 1.0]})
GAMES = [_g(1, "2026-10-09T22:00:00Z"), _g(2, "2026-10-08T23:30:00Z"), _g(3, "2026-10-06T00:00:00Z"),
         _g(4, "2026-10-09T22:00:00Z"), _g(5, "2026-10-09T22:00:00Z"), _g(6, "2026-12-01T22:00:00Z")]


def test_cfb_rows_forecast_indoor_observed_window_and_skips():
    rows = {r["game_pk"]: r for r in gi.cfb_game_info(GAMES, {3: {"home": [7, 7, 7, 7], "away": [0, 0, 0, 3]}}, WEATHER, VENUES, NOW)}
    assert set(rows) == {1, 2, 3}                       # 4: no FBS side, 5: no weather and no line score, 6: outside the window
    a = rows[1]
    assert (a["sport"], a["source"], a["venue_name"], a["city"], a["state"]) == ("cfb", "cfbd", "Bryant-Denny Stadium", "Tuscaloosa", "AL")
    assert (a["indoor"], a["temp_f"], a["wind_mph"], a["precip_in"], a["conditions"], a["weather_kind"]) == (False, 41.2, 14.0, 0.04, "Cloudy", "forecast")
    assert a["precip_chance"] is None and a["line_score"] is None
    b = rows[2]                                          # indoors: weather columns stay NULL
    assert b["indoor"] is True and b["temp_f"] is None and b["wind_mph"] is None and b["conditions"] is None and b["weather_kind"] is None
    assert b["city"] == "San Antonio"
    c = rows[3]                                          # kicked off already; venue not in venues.parquet -> CFBD's own name, no city
    assert c["weather_kind"] == "observed" and c["venue_name"] == "Unlisted Field" and c["city"] is None and c["state"] is None
    assert c["line_score"] == {"home": [7, 7, 7, 7], "away": [0, 0, 0, 3]} and c["indoor"] is False
    assert all(r["captured_at"].isoformat().startswith("2026-10-07T14:00:00") for r in rows.values())


def test_cfb_line_score_only_row_and_no_inputs():
    only = gi.cfb_game_info([_g(9, "2026-10-06T00:00:00Z")], {9: {"home": [1, 2, 3, 4], "away": [4, 3, 2, 1]}},
                            WEATHER.iloc[0:0], None, NOW)
    assert len(only) == 1 and only[0]["line_score"]["home"] == [1, 2, 3, 4] and only[0]["venue_name"] is None and only[0]["indoor"] is None
    assert gi.cfb_game_info([], {}, WEATHER, VENUES, NOW) == []


def _info(**kw):
    base = {"venue_name": "Highmark Stadium", "city": "Orchard Park", "state": "NY", "indoor": False,
            "temp_f": None, "wind_mph": None, "precip_chance": None, "conditions": None}
    return {**base, **kw}


def test_nfl_rows_window_dedupe_weather_kind_and_errors():
    calls, warns = [], []

    def fetch(pk):
        calls.append(pk)
        if pk == 13:
            raise RuntimeError("boom")
        return {10: _info(temp_f=41, wind_mph=14, precip_chance=20, conditions="Cloudy"), 11: _info(), 12: _info(indoor=True, venue_name="Dome")}[pk]

    games = [_g(10, "2026-10-11T17:00:00Z"), _g(10, "2026-10-11T17:00:00Z"), _g(11, "2026-10-12T00:20:00Z"),
             _g(12, "2026-10-11T20:25:00Z"), _g(13, "2026-10-11T20:25:00Z"), _g(14, "2027-01-01T00:00:00Z")]
    rows = {r["game_pk"]: r for r in gi.nfl_game_info(games, fetch, NOW, warns.append)}
    assert sorted(calls) == [10, 11, 12, 13] and set(rows) == {10, 11, 12}      # duplicate and out-of-window not fetched
    assert len(warns) == 1 and "13" in warns[0]
    a = rows[10]
    assert (a["sport"], a["source"], a["indoor"], a["temp_f"], a["wind_mph"], a["precip_chance"], a["weather_kind"]) == ("nfl", "espn", False, 41.0, 14.0, 20.0, "forecast")
    assert rows[11]["weather_kind"] is None and rows[11]["temp_f"] is None and rows[11]["city"] == "Orchard Park"
    assert rows[12]["indoor"] is True and rows[12]["venue_name"] == "Dome"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_game_info.py`
Expected: FAIL — `ERROR tests/test_game_info.py`.

- [ ] **Step 3: Implement the builders**

Create `src/sportsmodel/game_info.py`:

```python
"""game_info rows (pure): venue + weather (+ CFB line score) for the games of the upcoming window.

One row per (sport, game_pk) for games that kick off between now - WINDOW_BEFORE and now + WINDOW_AFTER,
written by scripts/build_game_info.py into Supabase `game_info` (db/migration_site_panels.sql); the game
page's hero line, weather insight and Projected Game Flow actuals read it.

* CFB: weather + venue link from CFBD `/games/weather` (forecast before kickoff, observation after),
  city / state / dome from assets/cfb/venues.parquet, the final quarter scores from ESPN's scoreboard.
* NFL: ESPN `/summary` gameInfo (`nfl.espn.parse_game_info`); weather_kind is always 'forecast' (ESPN's
  block is not known to be an observation).
Indoor games carry no weather (the site says "Indoors"). A missing field stays None, never guessed.
"""
from __future__ import annotations

import math
from typing import Callable

import pandas as pd

WINDOW_BEFORE = pd.Timedelta(days=2)
WINDOW_AFTER = pd.Timedelta(days=10)
WEATHER_FIELDS = ("temp_f", "wind_mph", "precip_chance", "precip_in", "conditions")


def utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def in_window(kickoff, now) -> bool:
    """now - 2 days <= kickoff <= now + 10 days (both inclusive); False for an unreadable kickoff."""
    try:
        k, n = utc(kickoff), utc(now)
    except (ValueError, TypeError):
        return False
    return n - WINDOW_BEFORE <= k <= n + WINDOW_AFTER


def _num(v):
    """float, or None for None / NaN / non-numbers."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _text(v):
    s = "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v).strip()
    return s or None


def _blank(sport: str, pk: int, source: str, now) -> dict:
    return {"sport": sport, "game_pk": int(pk), "venue_name": None, "city": None, "state": None,
            "indoor": None, "temp_f": None, "wind_mph": None, "precip_chance": None, "precip_in": None,
            "conditions": None, "weather_kind": None, "source": source,
            "captured_at": utc(now).to_pydatetime(), "line_score": None}


def cfb_game_info(games: list[dict], line_scores: dict[int, dict], weather: pd.DataFrame,
                  venues: pd.DataFrame | None, now) -> list[dict]:
    """CFB rows. `games` = ESPN `cfb.espn.parse_schedule` dicts (game_pk, commence_time); `line_scores` =
    `cfb.espn.parse_line_scores`; `weather` = `cfbd_games.parse_weather_window` (only has_fbs rows are
    used); `venues` = venues.parquet (venue_id, name, city, state, dome) or None. A game with neither a
    weather row nor a line score is skipped. weather_kind is 'observed' once the game has kicked off."""
    now = utc(now)
    wx = {}
    if weather is not None and len(weather):
        for r in weather[weather["has_fbs"]].drop_duplicates("game_id", keep="last").to_dict("records"):
            wx[int(r["game_id"])] = r
    vmap = {} if venues is None or venues.empty else venues.drop_duplicates("venue_id").set_index("venue_id").to_dict("index")
    out = []
    for g in games:
        pk, kick = int(g["game_pk"]), g.get("commence_time") or g.get("start_date")
        if not in_window(kick, now):
            continue
        w, ls = wx.get(pk), line_scores.get(pk)
        if w is None and ls is None:
            continue
        row = _blank("cfb", pk, "cfbd", now)
        row["line_score"] = ls
        if w is not None:
            vid = _num(w.get("venue_id"))
            v = vmap.get(int(vid)) if vid is not None else None
            row["venue_name"] = _text(v["name"]) if v else _text(w.get("venue"))
            if v:
                row["city"], row["state"] = _text(v.get("city")), _text(v.get("state"))
            dome = _num(v.get("dome")) if v else None
            row["indoor"] = bool(w.get("game_indoors")) or dome == 1.0
            if not row["indoor"]:
                row.update(temp_f=_num(w.get("temperature")), wind_mph=_num(w.get("wind_speed")),
                           precip_in=_num(w.get("precipitation")), conditions=_text(w.get("condition")))
                if any(row[f] is not None for f in WEATHER_FIELDS):
                    row["weather_kind"] = "observed" if utc(kick) <= now else "forecast"
        out.append(row)
    return out


def nfl_game_info(games: list[dict], fetch_info: Callable[[int], dict], now,
                  warn: Callable[[str], None] = lambda m: None) -> list[dict]:
    """NFL rows. `games` = `nfl.espn.parse_schedule` dicts; `fetch_info(game_pk)` = `nfl.espn.fetch_game_info`.
    A game whose summary call raises is skipped with a warning (one bad call never loses the slate)."""
    now = utc(now)
    out, seen = [], set()
    for g in games:
        pk = int(g["game_pk"])
        if pk in seen or not in_window(g.get("commence_time"), now):
            continue
        seen.add(pk)
        try:
            info = fetch_info(pk)
        except Exception as exc:  # noqa: BLE001
            warn(f"nfl: summary for {pk} unavailable ({type(exc).__name__})")
            continue
        row = _blank("nfl", pk, "espn", now)
        row.update(venue_name=info.get("venue_name"), city=info.get("city"), state=info.get("state"),
                   indoor=bool(info.get("indoor")), temp_f=_num(info.get("temp_f")),
                   wind_mph=_num(info.get("wind_mph")), precip_chance=_num(info.get("precip_chance")),
                   conditions=_text(info.get("conditions")))
        if any(row[f] is not None for f in WEATHER_FIELDS):
            row["weather_kind"] = "forecast"
        out.append(row)
    return out
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_game_info.py`
Expected: PASS (`4 passed` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add src/sportsmodel/game_info.py tests/test_game_info.py
git commit -m "feat: game_info row builders for CFB (CFBD) and NFL (ESPN)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 7: Migration file and DB upserts

**Files:**
- Create: `db/migration_site_panels.sql` (a file the USER runs; nothing in this plan executes it)
- Modify: `src/sportsmodel/db.py` (append site-panel upserts)
- Test: `tests/test_db_site_panels.py` (create)

**Interfaces:**
- Consumes: row dicts from T4 (`panels.to_rows(quarter_shares(...))`), T5 (`panels.to_rows(team_insights(...))`) and T6 (`game_info` rows); existing `db.get_postgres()`.
- Produces: `db.SITE_PANEL_COLUMNS: dict[str, list[str]]` and `db.SITE_PANEL_KEYS: dict[str, tuple[str, ...]]` for `game_info` / `cfb_team_insights` / `cfb_quarter_shares`; `db.upsert_game_info(rows) -> int`, `db.upsert_cfb_team_insights(rows) -> int`, `db.upsert_cfb_quarter_shares(rows) -> int` (0 and no connection for an empty list; upserts only, no deletes; `game_info` keeps earlier readings via `COALESCE(new, old)` and NULLs weather for indoor rows). The three tables with PKs `(sport, game_pk)`, `(season, team)`, `(team)`, RLS on, anon `SELECT` policies and grants.

- [ ] **Step 1: Write the failing tests**

The tests check the SQL text and the tuples handed to the cursor (the repo's `FakeConn` pattern, no live DB), and that every column the code writes exists in the migration's `CREATE TABLE` (so the two cannot drift).

Create `tests/test_db_site_panels.py`:

```python
"""db.upsert_game_info / upsert_cfb_team_insights / upsert_cfb_quarter_shares and
db/migration_site_panels.sql: SQL shape, tuple building and column agreement. No live DB (FakeConn)."""
import json
import re
from pathlib import Path

import pytest

from sportsmodel import db

MIGRATION = (Path(__file__).parent.parent / "db" / "migration_site_panels.sql").read_text()


class FakeCursor:
    def __init__(self, sink):
        self.sink = sink

    def executemany(self, sql, rows):
        self.sink["sql"], self.sink["rows"] = sql, list(rows)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, sink):
        self.sink = sink

    def cursor(self):
        return FakeCursor(self.sink)

    def commit(self):
        self.sink["committed"] = True

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _boom():
    raise AssertionError("get_postgres should not be called")


@pytest.mark.parametrize("fn", ["upsert_game_info", "upsert_cfb_team_insights", "upsert_cfb_quarter_shares"])
def test_empty_rows_never_connect(monkeypatch, fn):
    monkeypatch.setattr(db, "get_postgres", _boom)
    assert getattr(db, fn)([]) == 0


def test_game_info_sql_keeps_earlier_readings_and_nulls_indoor_weather(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    row = {"sport": "cfb", "game_pk": 7, "venue_name": "V", "city": "C", "state": "AL", "indoor": False,
           "temp_f": 41.0, "wind_mph": 14.0, "precip_chance": None, "precip_in": 0.04, "conditions": None,
           "weather_kind": "forecast", "source": "cfbd", "captured_at": "2026-10-07T14:00:00Z",
           "line_score": {"home": [1, 2, 3, 4], "away": [4, 3, 2, 1]}}
    assert db.upsert_game_info([row]) == 1
    (tup,) = sink["rows"]
    cols = db.SITE_PANEL_COLUMNS["game_info"]
    assert len(tup) == len(cols) and tup[cols.index("sport")] == "cfb" and tup[cols.index("game_pk")] == 7
    assert json.loads(tup[cols.index("line_score")]) == row["line_score"]
    sql = sink["sql"]
    assert "INSERT INTO game_info" in sql and "ON CONFLICT (sport, game_pk) DO UPDATE" in sql
    assert "temp_f = CASE WHEN EXCLUDED.indoor IS TRUE THEN NULL ELSE COALESCE(EXCLUDED.temp_f, game_info.temp_f) END" in sql
    assert "weather_kind = CASE WHEN EXCLUDED.indoor IS TRUE" in sql
    assert "venue_name = COALESCE(EXCLUDED.venue_name, game_info.venue_name)" in sql
    assert "line_score = COALESCE(EXCLUDED.line_score, game_info.line_score)" in sql
    assert "indoor = COALESCE(EXCLUDED.indoor, game_info.indoor)" in sql
    assert "source = EXCLUDED.source" in sql and "captured_at = EXCLUDED.captured_at" in sql
    assert sql.rstrip().endswith("updated_at = now()") and sink["committed"] is True
    row2 = {**row, "line_score": None}
    db.upsert_game_info([row2])
    assert sink["rows"][0][cols.index("line_score")] is None          # NULL stays SQL NULL, not the string "null"


def test_cfb_tables_overwrite_every_column(monkeypatch):
    sink = {}
    monkeypatch.setattr(db, "get_postgres", lambda: FakeConn(sink))
    ins = {c: 1 for c in db.SITE_PANEL_COLUMNS["cfb_team_insights"]} | {"team": "333"}
    assert db.upsert_cfb_team_insights([ins]) == 1
    assert "ON CONFLICT (season, team) DO UPDATE" in sink["sql"] and "COALESCE" not in sink["sql"]
    assert "def_havoc_rank = EXCLUDED.def_havoc_rank" in sink["sql"] and "season = EXCLUDED.season" not in sink["sql"]
    sh = {c: 0.25 for c in db.SITE_PANEL_COLUMNS["cfb_quarter_shares"]} | {"team": "333", "games_used": 30}
    assert db.upsert_cfb_quarter_shares([sh]) == 1
    assert "ON CONFLICT (team) DO UPDATE" in sink["sql"] and sink["rows"][0][0] == "333" and sink["rows"][0][-1] == 30


def _create_table(name):
    m = re.search(rf"CREATE TABLE IF NOT EXISTS {name} \((.+?)\n\);", MIGRATION, re.DOTALL)
    assert m, f"no CREATE TABLE for {name}"
    return m.group(1)


@pytest.mark.parametrize("table", ["game_info", "cfb_team_insights", "cfb_quarter_shares"])
def test_migration_table_has_every_written_column_pk_rls_policy_and_grant(table):
    body = _create_table(table)
    cols = re.findall(r"^\s{2}(\w+)\s", body, re.MULTILINE)
    assert set(db.SITE_PANEL_COLUMNS[table]) <= set(cols), set(db.SITE_PANEL_COLUMNS[table]) - set(cols)
    assert "updated_at" in cols
    key = ", ".join(db.SITE_PANEL_KEYS[table])
    assert f"PRIMARY KEY ({key})" in body or (len(db.SITE_PANEL_KEYS[table]) == 1 and "PRIMARY KEY" in body)
    assert f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY;" in MIGRATION
    assert f'CREATE POLICY "public read {table}" ON {table} FOR SELECT USING (true);' in MIGRATION
    assert re.search(rf"GRANT SELECT ON [^;]*\b{table}\b[^;]* TO anon, authenticated;", MIGRATION)


def test_migration_is_idempotent_and_writes_nothing():
    assert MIGRATION.count("CREATE TABLE IF NOT EXISTS") == 3
    assert MIGRATION.count("DROP POLICY IF EXISTS") == 3
    assert not re.search(r"^\s*(INSERT|UPDATE|DELETE|DROP TABLE|TRUNCATE)\b", MIGRATION, re.MULTILINE | re.IGNORECASE)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/test_db_site_panels.py`
Expected: FAIL — `ERROR tests/test_db_site_panels.py - FileNotFoundError: [Errno 2] No such fil...`.

- [ ] **Step 3: Write the migration**

Three tables, primary keys, RLS and anon read policies in the style of `db/migration_team_context.sql`. The `(sport, game_pk)` primary key is the lookup index the site uses (`game_info?sport=eq.cfb&game_pk=eq.<pk>`), so no separate index is created (a second one would only duplicate it).

Create `db/migration_site_panels.sql`:

```sql
-- =============================================================================
-- CFB site panels: game hero venue/weather line + weather insight, CFB havoc / turnover insights,
-- and the CFB Projected Game Flow. Descriptive data only -- not picks.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor BEFORE the jobs first run (the site
-- hides each panel while its table is missing or empty, so it can deploy before or after this).
--
--   game_info           one row per (sport, game_pk) for games from 2 days ago to 10 days ahead (CFB + NFL).
--                       Written by scripts/build_game_info.py (.github/workflows/build-game-info.yml).
--                       Weather columns are NULL for indoor games and whenever the source has no reading;
--                       weather_kind: forecast | observed. line_score (CFB, finished games only):
--                       {"home": [q1..q4], "away": [q1..q4]}, overtime excluded. The primary key is the
--                       (sport, game_pk) lookup index the site uses, so no second index is created.
--   cfb_team_insights   one row per (season, team): season-to-date regular-season FBS-vs-FBS havoc,
--                       turnover margin and explosiveness with national ranks (1 = best) among teams with
--                       >= 3 games (n_ranked = size of that field). Written weekly by
--                       scripts/build_cfb_panels.py (build-cfb-advanced.yml, Monday).
--   cfb_quarter_shares  one row per team: share of points scored / allowed in Q1-Q4 over the last two
--                       completed seasons plus this one, shrunk toward the league average; each set of
--                       four sums to 1. Written weekly by the same job.
-- Team ids are ESPN team ids as text (the same ids as team_history.team).

CREATE TABLE IF NOT EXISTS game_info (
  sport          text NOT NULL,              -- nfl | cfb
  game_pk        bigint NOT NULL,            -- ESPN event id (= CFBD game id)
  venue_name     text,
  city           text,
  state          text,
  indoor         boolean,
  temp_f         double precision,
  wind_mph       double precision,
  precip_chance  double precision,           -- 0-100; NULL for CFB (CFBD sends no probability)
  precip_in      double precision,           -- inches; NULL for NFL (ESPN sends none)
  conditions     text,
  weather_kind   text CHECK (weather_kind IN ('forecast', 'observed')),
  source         text NOT NULL CHECK (source IN ('cfbd', 'espn')),
  line_score     jsonb,
  captured_at    timestamptz NOT NULL DEFAULT now(),
  updated_at     timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (sport, game_pk)
);

CREATE TABLE IF NOT EXISTS cfb_team_insights (
  season                          integer NOT NULL,
  team                            text NOT NULL,
  games                           integer NOT NULL,
  def_havoc_rate                  double precision,
  def_havoc_rank                  integer,
  off_havoc_allowed_rate          double precision,
  off_havoc_allowed_rank          integer,
  turnover_margin                 double precision,
  turnover_margin_per_game        double precision,
  turnover_margin_rank            integer,
  off_explosiveness               double precision,
  off_explosiveness_rank          integer,
  def_explosiveness_allowed       double precision,
  def_explosiveness_allowed_rank  integer,
  n_ranked                        integer,
  through_week                    integer,
  updated_at                      timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (season, team)
);

CREATE TABLE IF NOT EXISTS cfb_quarter_shares (
  team         text NOT NULL PRIMARY KEY,
  scored_q1    double precision NOT NULL,
  scored_q2    double precision NOT NULL,
  scored_q3    double precision NOT NULL,
  scored_q4    double precision NOT NULL,
  allowed_q1   double precision NOT NULL,
  allowed_q2   double precision NOT NULL,
  allowed_q3   double precision NOT NULL,
  allowed_q4   double precision NOT NULL,
  games_used   integer NOT NULL,
  updated_at   timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE game_info ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read game_info" ON game_info;
CREATE POLICY "public read game_info" ON game_info FOR SELECT USING (true);
ALTER TABLE cfb_team_insights ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read cfb_team_insights" ON cfb_team_insights;
CREATE POLICY "public read cfb_team_insights" ON cfb_team_insights FOR SELECT USING (true);
ALTER TABLE cfb_quarter_shares ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read cfb_quarter_shares" ON cfb_quarter_shares;
CREATE POLICY "public read cfb_quarter_shares" ON cfb_quarter_shares FOR SELECT USING (true);

GRANT SELECT ON game_info, cfb_team_insights, cfb_quarter_shares TO anon, authenticated;
```

- [ ] **Step 4: Implement the upserts**

Append to the end of `src/sportsmodel/db.py`:

```python
# ------------------------------------------------------------------ site panels
# db/migration_site_panels.sql -- game_info is written by scripts/build_game_info.py, the two cfb_* tables by
# scripts/build_cfb_panels.py. Upserts only: nothing is ever deleted (rows outside a run's window survive).
SITE_PANEL_COLUMNS: dict[str, list[str]] = {
    "game_info": ["sport", "game_pk", "venue_name", "city", "state", "indoor", "temp_f", "wind_mph",
                  "precip_chance", "precip_in", "conditions", "weather_kind", "source", "captured_at",
                  "line_score"],
    "cfb_team_insights": ["season", "team", "games", "def_havoc_rate", "def_havoc_rank",
                          "off_havoc_allowed_rate", "off_havoc_allowed_rank", "turnover_margin",
                          "turnover_margin_per_game", "turnover_margin_rank", "off_explosiveness",
                          "off_explosiveness_rank", "def_explosiveness_allowed",
                          "def_explosiveness_allowed_rank", "n_ranked", "through_week"],
    "cfb_quarter_shares": ["team", "scored_q1", "scored_q2", "scored_q3", "scored_q4", "allowed_q1",
                           "allowed_q2", "allowed_q3", "allowed_q4", "games_used"],
}
SITE_PANEL_KEYS: dict[str, tuple[str, ...]] = {
    "game_info": ("sport", "game_pk"), "cfb_team_insights": ("season", "team"),
    "cfb_quarter_shares": ("team",),
}
_SITE_PANEL_JSON = frozenset({"line_score"})
_GAME_INFO_WEATHER = frozenset({"temp_f", "wind_mph", "precip_chance", "precip_in", "conditions",
                                "weather_kind"})


def _site_panel_sql(table: str) -> str:
    """INSERT ... ON CONFLICT DO UPDATE for one site-panel table. cfb_* tables overwrite every column.
    game_info keeps what an earlier run learned: a later run that has no reading (ESPN drops its weather
    block after a game, CFBD has no forecast yet) must not erase it, so every non-key column except
    `source` / `captured_at` is COALESCE(new, old); the weather columns are additionally NULLed when the
    new row says the venue is indoors."""
    cols, key = SITE_PANEL_COLUMNS[table], SITE_PANEL_KEYS[table]
    sets = []
    for c in cols:
        if c in key:
            continue
        if table != "game_info" or c in ("source", "captured_at"):
            sets.append(f"{c} = EXCLUDED.{c}")
        elif c in _GAME_INFO_WEATHER:
            sets.append(f"{c} = CASE WHEN EXCLUDED.indoor IS TRUE THEN NULL "
                        f"ELSE COALESCE(EXCLUDED.{c}, game_info.{c}) END")
        else:
            sets.append(f"{c} = COALESCE(EXCLUDED.{c}, game_info.{c})")
    return (f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) "
            f"ON CONFLICT ({', '.join(key)}) DO UPDATE SET {', '.join(sets)}, updated_at = now()")


def _upsert_site_panel(table: str, rows: list[dict]) -> int:
    if not rows:
        return 0
    cols = SITE_PANEL_COLUMNS[table]
    vals = [tuple((None if r.get(c) is None else json.dumps(r[c])) if c in _SITE_PANEL_JSON else r.get(c)
                  for c in cols) for r in rows]
    with get_postgres() as conn, conn.cursor() as cur:
        cur.executemany(_site_panel_sql(table), vals)
        conn.commit()
    return len(vals)


def upsert_game_info(rows: list[dict]) -> int:
    """Upsert `game_info` rows (SITE_PANEL_COLUMNS['game_info']; line_score a dict, JSON-encoded here)."""
    return _upsert_site_panel("game_info", rows)


def upsert_cfb_team_insights(rows: list[dict]) -> int:
    return _upsert_site_panel("cfb_team_insights", rows)


def upsert_cfb_quarter_shares(rows: list[dict]) -> int:
    return _upsert_site_panel("cfb_quarter_shares", rows)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/test_db_site_panels.py tests/test_db_injury_snapshots.py tests/test_migration_site_redesign_a.py`
Expected: PASS (`16 passed` in this plan's dry run).

- [ ] **Step 6: Commit**

```bash
git add db/migration_site_panels.sql src/sportsmodel/db.py tests/test_db_site_panels.py
git commit -m "feat(db): site panel tables migration and upserts (game_info, cfb_team_insights, cfb_quarter_shares)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 8: Daily game-info job and workflow

**Files:**
- Create: `scripts/build_game_info.py`
- Create: `.github/workflows/build-game-info.yml`
- Test: `tests/scripts/test_build_game_info.py` (create), `tests/test_workflows_game_info.py` (create)

**Interfaces:**
- Consumes: T2 `cfb.espn.fetch_current_week/fetch_scoreboard/parse_schedule/parse_line_scores`, `nfl.espn.fetch_current_week/target_week/fetch_schedule/fetch_game_info`; T1 `parse_weather_window`; T6 `game_info.cfb_game_info/nfl_game_info/in_window/utc`; T7 `db.upsert_game_info`; existing `CfbdClient.from_env()`, `config.DATABASE_URL`.
- Produces: `build_game_info.window_weeks(games: list[dict], now) -> list[tuple[int, int]]`; `build_cfb(now, client, espn=cfb_espn, venues_path=VENUES_PATH) -> list[dict]`; `build_nfl(now, espn=nfl_espn) -> list[dict]`; `run_sport(sport: str, now, dry_run: bool) -> int`; `main(argv=None)` with `--sport {nfl,cfb,all}`, `--dry-run`, `--now ISO`. Workflow `build-game-info` (crons `0 14 * * *`, `0 16 * * 0,1,4,5,6`, `0 22 * * 0,1,4,5,6`; secrets `DATABASE_URL`, `CFBD_API_KEY`; never commits).

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_game_info.py`:

```python
"""build_game_info.py: the daily game-info job on stubbed ESPN / CFBD clients. No network, no DB."""
from __future__ import annotations

import importlib.util
import pathlib
from types import SimpleNamespace

import pandas as pd
import pytest

from sportsmodel import db
from sportsmodel.cfb import espn as real_cfb_espn

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_game_info.py"
_s = importlib.util.spec_from_file_location("build_game_info", _p)
bgi = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bgi)

NOW = pd.Timestamp("2026-10-07T14:00:00Z")


def _ev(pk, home_id, away_id, kick, status="STATUS_SCHEDULED", ls=None):
    comp = [{"homeAway": "home", "team": {"id": home_id, "displayName": "H"}, "score": "0"},
            {"homeAway": "away", "team": {"id": away_id, "displayName": "A"}, "score": "0"}]
    if ls:
        comp[0]["linescores"] = [{"value": v} for v in ls[0]]
        comp[1]["linescores"] = [{"value": v} for v in ls[1]]
    return {"id": str(pk), "date": kick, "status": {"type": {"name": status}}, "week": {"number": 6},
            "season": {"year": 2026}, "competitions": [{"competitors": comp}]}


class FakeCfbEspn:
    """The real parsers over canned scoreboards; only the two network calls are faked."""
    parse_schedule, parse_line_scores = staticmethod(real_cfb_espn.parse_schedule), staticmethod(real_cfb_espn.parse_line_scores)

    def __init__(self, cur, fail_weeks=()):
        self.cur, self.fail, self.asked = cur, set(fail_weeks), []

    def fetch_current_week(self):
        return self.cur

    def fetch_scoreboard(self, season, week, st):
        self.asked.append((season, week, st))
        if week in self.fail:
            raise RuntimeError("espn down")
        if week != 6:
            return {"events": []}
        return {"events": [_ev(1, "333", "61", "2026-10-09T22:00Z"),
                           _ev(2, "333", "61", "2026-10-06T00:00Z", "STATUS_FINAL", ([7, 7, 7, 7], [0, 3, 3, 0]))]}


class FakeClient:
    def __init__(self):
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, dict(params)))
        return [{"id": 1, "season": 2026, "week": 6, "seasonType": "regular", "startTime": "2026-10-09T22:00:00.000Z",
                 "gameIndoors": False, "homeTeam": "Alabama", "awayTeam": "Georgia", "venueId": 3657, "venue": "Bryant-Denny Stadium",
                 "temperature": 41, "windSpeed": 14, "precipitation": 0}]


VENUES = pd.DataFrame({"venue_id": [3657], "name": ["Bryant-Denny Stadium"], "city": ["Tuscaloosa"], "state": ["AL"], "dome": [0.0]})


def test_window_weeks_only_counts_games_inside_the_window():
    games = [{"season": 2026, "week": 6, "commence_time": "2026-10-09T22:00Z"},
             {"season": 2026, "week": 7, "commence_time": "2026-10-16T22:00Z"},
             {"season": 2026, "week": 12, "commence_time": "2026-11-20T22:00Z"},
             {"season": None, "week": 6, "commence_time": "2026-10-09T22:00Z"}]
    assert bgi.window_weeks(games, NOW) == [(2026, 6), (2026, 7)]


def test_build_cfb_one_weather_call_per_window_week_and_rows(tmp_path):
    vp = tmp_path / "venues.parquet"
    VENUES.to_parquet(vp)
    espn, client = FakeCfbEspn({"season": 2026, "week": 6, "season_type": 2}), FakeClient()
    rows = bgi.build_cfb(NOW, client, espn, vp)
    assert espn.asked == [(2026, 5, 2), (2026, 6, 2), (2026, 7, 2)]
    assert client.calls == [("/games/weather", {"year": 2026, "week": 6, "seasonType": "regular"})]
    by = {r["game_pk"]: r for r in rows}
    assert by[1]["venue_name"] == "Bryant-Denny Stadium" and by[1]["city"] == "Tuscaloosa" and by[1]["weather_kind"] == "forecast"
    assert by[2]["line_score"] == {"home": [7, 7, 7, 7], "away": [0, 3, 3, 0]} and by[2]["venue_name"] is None


def test_build_cfb_offseason_and_a_down_espn_week(tmp_path):
    assert bgi.build_cfb(NOW, FakeClient(), FakeCfbEspn({"season": 2026, "week": 1, "season_type": 4}), tmp_path / "x") == []
    espn = FakeCfbEspn({"season": 2026, "week": 6, "season_type": 3}, fail_weeks={5})
    client = FakeClient()
    rows = bgi.build_cfb(NOW, client, espn, tmp_path / "missing.parquet")           # no venues asset: names only, still runs
    assert client.calls[0][1]["seasonType"] == "postseason" and len(rows) == 2


def test_build_nfl_walks_three_weeks_and_fetches_each_game_once():
    asked, fetched = [], []

    def fetch_schedule(season, week, season_type=2):
        asked.append((season, week, season_type))
        return [{"game_pk": week * 10, "commence_time": "2026-10-11T17:00:00Z"}] if week == 5 else []

    espn = SimpleNamespace(fetch_current_week=lambda: {"season": 2026, "week": 5, "season_type": 2},
                           target_week=lambda cur: cur, fetch_schedule=fetch_schedule,
                           fetch_game_info=lambda pk: fetched.append(pk) or {"venue_name": "V", "city": "C", "state": "S", "indoor": False,
                                                                           "temp_f": 50, "wind_mph": None, "precip_chance": None, "conditions": None})
    rows = bgi.build_nfl(NOW, espn)
    assert asked == [(2026, 4, 2), (2026, 5, 2), (2026, 6, 2)] and fetched == [50]
    assert rows[0]["temp_f"] == 50.0 and rows[0]["weather_kind"] == "forecast"


def test_main_dry_run_writes_nothing_and_a_failed_sport_exits_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(db, "get_postgres", lambda: (_ for _ in ()).throw(AssertionError("no DB in a dry run")))
    monkeypatch.setattr(bgi, "build_nfl", lambda now, espn=None: [{"sport": "nfl", "game_pk": 1, "weather_kind": "forecast", "indoor": False, "line_score": None, "temp_f": 40.0}])
    bgi.main(["--sport", "nfl", "--dry-run", "--now", "2026-10-07T14:00:00Z"])
    assert "rows=1" in capsys.readouterr().out

    def boom(now, espn=None):
        raise RuntimeError("espn down")

    monkeypatch.setattr(bgi, "build_nfl", boom)
    with pytest.raises(SystemExit) as e:
        bgi.main(["--sport", "nfl", "--dry-run"])
    assert "nfl" in str(e.value)


def test_main_upserts_rows_and_requires_database_url(monkeypatch):
    sent = []
    monkeypatch.setattr(bgi.config, "DATABASE_URL", "postgres://x")
    monkeypatch.setattr(bgi.db, "upsert_game_info", lambda rows: sent.append(rows) or len(rows))
    monkeypatch.setattr(bgi, "build_nfl", lambda now, espn=None: [{"sport": "nfl", "game_pk": 1, "weather_kind": None, "indoor": None, "line_score": None}])
    bgi.main(["--sport", "nfl"])
    assert sent == [[{"sport": "nfl", "game_pk": 1, "weather_kind": None, "indoor": None, "line_score": None}]]
    monkeypatch.setattr(bgi.config, "DATABASE_URL", None)
    with pytest.raises(SystemExit):
        bgi.main(["--sport", "nfl"])
```

Create `tests/test_workflows_game_info.py`:

```python
"""Structural checks of .github/workflows/build-game-info.yml: daily 14:00 UTC plus 16:00 / 22:00 UTC on
football days, runs the game-info job with DATABASE_URL + CFBD_API_KEY, never commits."""
from __future__ import annotations

from tests.test_workflows_props_ml import WF, load, runs, step_index, steps_of

NAME = "build-game-info.yml"


def test_schedule_daily_plus_football_days_and_dispatch():
    wf = load(NAME)
    assert [c["cron"] for c in wf["on"]["schedule"]] == ["0 14 * * *", "0 16 * * 0,1,4,5,6", "0 22 * * 0,1,4,5,6"]
    inp = wf["on"]["workflow_dispatch"]["inputs"]["sport"]
    assert inp["default"] == "all" and inp["options"] == ["all", "nfl", "cfb"]
    days = wf["on"]["schedule"][1]["cron"].split()[4].split(",")
    assert sorted(days) == ["0", "1", "4", "5", "6"]          # Sun Mon Thu Fri Sat


def test_job_runs_the_script_with_both_secrets_and_serializes_runs():
    wf = load(NAME)
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    steps = steps_of(job)
    step = steps[step_index(steps, runs("scripts/build_game_info.py"))]
    assert step["run"] == 'uv run python scripts/build_game_info.py --sport "$SPORT"'
    assert step["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert step["env"]["CFBD_API_KEY"] == "${{ secrets.CFBD_API_KEY }}"
    assert step["env"]["SPORT"] == "${{ inputs.sport || 'all' }}"
    assert job["concurrency"]["group"] == "build-game-info"
    assert str(job["concurrency"]["cancel-in-progress"]).lower() == "false"
    assert step_index(steps, runs("uv sync")) < step_index(steps, runs("build_game_info.py"))


def test_workflow_never_commits():
    text = (WF / NAME).read_text()
    assert "git push" not in text and "git commit" not in text
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/scripts/test_build_game_info.py tests/test_workflows_game_info.py`
Expected: FAIL — `ERROR tests/scripts/test_build_game_info.py - FileNotFoundError: [Errno 2] No...`.

- [ ] **Step 3: Implement the script**

Create `scripts/build_game_info.py`:

```python
"""Daily game-info job (NFL + CFB): venue + weather (and the CFB final line score) for every game from
2 days ago to 10 days ahead -> Supabase `game_info` (db/migration_site_panels.sql). Descriptive only.

* CFB: ESPN's FBS scoreboard for the current week +- 1 gives the games, kickoffs and final quarter scores;
  CFBD `/games/weather` (one call per distinct week inside the window) gives venue id, indoor flag and the
  forecast / observed weather; assets/cfb/venues.parquet gives city / state. Needs CFBD_API_KEY.
* NFL: ESPN scoreboard for the current week +- 1, then one free `/summary` per game in the window.
* `game_info` upserts keep an earlier reading when a later run has none (see db._site_panel_sql).
* `--dry-run` prints counts and a sample and writes nothing (CFB still calls CFBD, read-only).

Usage:
    uv run python scripts/build_game_info.py --sport all [--dry-run] [--now ISO]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sportsmodel import config, db, game_info  # noqa: E402
from sportsmodel.cfb import cfbd_games, espn as cfb_espn  # noqa: E402
from sportsmodel.cfb.cfbd import CfbdClient  # noqa: E402
from sportsmodel.nfl import espn as nfl_espn  # noqa: E402

VENUES_PATH = ROOT / "assets" / "cfb" / "venues.parquet"


def warn(msg: str) -> None:
    print(f"::warning::game-info: {msg}", flush=True)


def window_weeks(games: list[dict], now) -> list[tuple[int, int]]:
    """Sorted distinct (season, week) of the games whose kickoff is inside the window."""
    return sorted({(int(g["season"]), int(g["week"])) for g in games
                   if g.get("season") is not None and g.get("week") is not None
                   and game_info.in_window(g.get("commence_time"), now)})


def build_cfb(now, client, espn=cfb_espn, venues_path: Path = VENUES_PATH) -> list[dict]:
    cur = espn.fetch_current_week()
    season, week, st = int(cur["season"]), int(cur["week"]), int(cur["season_type"])
    if st not in (2, 3):
        print("cfb: ESPN is between seasons; no rows", flush=True)
        return []
    games, line_scores = [], {}
    for w in (week - 1, week, week + 1):
        if w < 1:
            continue
        try:
            payload = espn.fetch_scoreboard(season, w, st)
        except Exception as exc:  # noqa: BLE001 -- one week down must not lose the rest
            warn(f"cfb: ESPN week {w} unavailable ({type(exc).__name__})")
            continue
        games += espn.parse_schedule(payload)
        line_scores.update(espn.parse_line_scores(payload))
    stype = "postseason" if st == 3 else "regular"
    frames = []
    for s, w in window_weeks(games, now):
        try:
            frames.append(cfbd_games.parse_weather_window(
                client.get("/games/weather", {"year": s, "week": w, "seasonType": stype})))
        except Exception as exc:  # noqa: BLE001
            warn(f"cfb: CFBD weather week {w} unavailable ({type(exc).__name__})")
    weather = pd.concat(frames, ignore_index=True) if frames else cfbd_games.parse_weather_window([])
    venues = pd.read_parquet(venues_path) if venues_path.exists() else None
    if venues is None or "city" not in venues.columns:
        warn(f"cfb: {venues_path.name} has no city / state (run the venues backfill); venue names only")
    return game_info.cfb_game_info(games, line_scores, weather, venues, now)


def build_nfl(now, espn=nfl_espn) -> list[dict]:
    tw = espn.target_week(espn.fetch_current_week())
    season, week, st = int(tw["season"]), int(tw["week"]), int(tw["season_type"])
    games = []
    for w in (week - 1, week, week + 1):
        if w < 1 or (st == 2 and w > 18):
            continue
        try:
            games += espn.fetch_schedule(season, w, season_type=st)
        except Exception as exc:  # noqa: BLE001
            warn(f"nfl: ESPN week {w} unavailable ({type(exc).__name__})")
    return game_info.nfl_game_info(games, espn.fetch_game_info, now, warn)


def run_sport(sport: str, now, dry_run: bool) -> int:
    rows = build_cfb(now, CfbdClient.from_env()) if sport == "cfb" else build_nfl(now)
    withwx = sum(1 for r in rows if r["weather_kind"])
    print(f"[{sport}] game_info rows={len(rows)} (with weather {withwx}, indoor "
          f"{sum(1 for r in rows if r['indoor'])}, with line score {sum(1 for r in rows if r['line_score'])})",
          flush=True)
    if dry_run:
        for r in rows[:3]:
            print("  ", {k: v for k, v in r.items() if v is not None})
        return len(rows)
    print(f"[{sport}] upserted {db.upsert_game_info(rows)} rows", flush=True)
    return len(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sport", choices=["nfl", "cfb", "all"], default="all")
    ap.add_argument("--dry-run", action="store_true", help="compute and print; no DB writes")
    ap.add_argument("--now", default=None, help="as-of time (ISO, default: now UTC)")
    args = ap.parse_args(argv)
    if not args.dry_run and not config.DATABASE_URL:
        sys.exit("DATABASE_URL is not set (use --dry-run to compute without writing)")
    now = game_info.utc(args.now) if args.now else pd.Timestamp.now(tz="UTC")
    failed = []
    for sport in (["nfl", "cfb"] if args.sport == "all" else [args.sport]):
        try:   # one sport failing must not block the other
            run_sport(sport, now, args.dry_run)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 -- SystemExit: CfbdClient.from_env without a key
            print(f"::error::game-info: {sport} failed: {type(exc).__name__}: {exc}", flush=True)
            failed.append(sport)
    if failed:
        sys.exit(f"game-info failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Add the workflow**

Cron `0 16 * * 0,1,4,5,6` and `0 22 * * 0,1,4,5,6` run on Sun, Mon, Thu, Fri, Sat (the football days) about two hours before the typical first kickoffs; `0 14 * * *` is the daily run. CFB costs one `/games/weather` call per distinct week in the window (about 2) per run, so the job stays at roughly 17 runs x 2 = 35 CFBD calls a week, inside the spec's 60-100 calls/week together with the Monday pulls.

Create `.github/workflows/build-game-info.yml`:

```yaml
name: build-game-info

# Venue + weather (and the CFB final line score) for the game page, NFL + CFB: every game from 2 days ago to
# 10 days ahead -> Supabase game_info (scripts/build_game_info.py; db/migration_site_panels.sql). Descriptive,
# not picks. Daily 14:00 UTC, plus 16:00 and 22:00 UTC on football days (Thu / Fri / Sat / Sun / Mon: cron
# days 4 5 6 0 1), about 2 hours before the typical first kickoffs, so a forecast that firmed up is captured.
# CFB reads CFBD (CFBD_API_KEY, one /games/weather call per week in the window); NFL reads ESPN (free).
# Idempotent upserts that keep an earlier reading when a later run has none; never commits to the repo.
on:
  schedule:
    - cron: "0 14 * * *"
    - cron: "0 16 * * 0,1,4,5,6"
    - cron: "0 22 * * 0,1,4,5,6"
  workflow_dispatch:
    inputs:
      sport:
        description: "Sport to build"
        type: choice
        options:
          - all
          - nfl
          - cfb
        default: all

permissions:
  contents: read

jobs:
  build:
    runs-on: ubuntu-latest
    timeout-minutes: 15
    concurrency:
      group: build-game-info
      cancel-in-progress: false
    steps:
      - uses: actions/checkout@v5

      - name: Install uv
        uses: astral-sh/setup-uv@v10.0.1

      - name: Sync deps
        run: uv sync

      - name: Build game info (venue, weather, CFB line scores)
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
          CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
          SPORT: ${{ inputs.sport || 'all' }}
        run: uv run python scripts/build_game_info.py --sport "$SPORT"
```

- [ ] **Step 5: Run the tests to verify they pass (including every workflow still parsing)**

Run: `uv run pytest -q tests/scripts/test_build_game_info.py tests/test_workflows_game_info.py tests/test_workflows_props_ml.py`
Expected: PASS (`114 passed` in this plan's dry run).

- [ ] **Step 6: [NETWORK] Dry-run against the live APIs (reads only; writes nothing)**

The executing agent runs this once. Expected: a `[nfl] game_info rows=N ...` and a `[cfb] game_info rows=M ...` line, `M` > 0 in season, rows printing venue / city / state, a CFB game with `temp_f` / `wind_mph`, and no `::error::`. CFB makes about 2 CFBD calls. A `::warning::` that a week or the venues asset is unavailable is acceptable only if it names a real outage; investigate anything else before committing.

**[NETWORK] — run by the executing agent, not a subagent reviewer.**

```bash
set -a; source .env; set +a                  # CFBD_API_KEY for the CFB half; never print it
uv run python scripts/build_game_info.py --sport all --dry-run
```

- [ ] **Step 7: Commit**

```bash
git add scripts/build_game_info.py .github/workflows/build-game-info.yml tests/scripts/test_build_game_info.py tests/test_workflows_game_info.py
git commit -m "feat: daily game_info job and workflow (venue, weather, CFB line scores)

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 9: Weekly CFB panels job and the Monday workflow additions

**Files:**
- Create: `scripts/build_cfb_panels.py`
- Modify: `.github/workflows/build-cfb-advanced.yml`
- Test: `tests/scripts/test_build_cfb_panels.py` (create), `tests/test_workflows_cfb_advanced.py` (append)

**Interfaces:**
- Consumes: T4 `panels.quarter_shares`, T5 `panels.team_insights`, `panels.to_rows`; T7 `db.upsert_cfb_team_insights`, `db.upsert_cfb_quarter_shares`; the T3 assets; existing `cfb.teams.load_fbs_ids()`; the T3 `build_cfb_game_data.py --datasets games havoc team_stats` interface.
- Produces: `build_cfb_panels.build(assets: Path | None = None, season: int | None = None, fbs: set[str] | None = None) -> tuple[int, DataFrame, DataFrame]` (season, insights, shares; default season = latest regular-season havoc season; exits with a message when `havoc_games`, `advanced_games` or `cfbd_games` is missing, warns when `team_game_stats` is); `main(argv=None)` with `--season`, `--dry-run`. Workflow `build-cfb-advanced` gains, in order: the current-season pull (`--datasets games havoc team_stats`), a commit step that stages all four parquets, then the `DATABASE_URL` panel rebuild step.

- [ ] **Step 1: Write the failing tests**

Create `tests/scripts/test_build_cfb_panels.py`:

```python
"""build_cfb_panels.py: assets in, two table row lists out (no network, no DB)."""
from __future__ import annotations

import importlib.util
import pathlib

import pandas as pd
import pytest

from tests.cfb.test_panels import _fixture, _game

_p = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "build_cfb_panels.py"
_s = importlib.util.spec_from_file_location("build_cfb_panels", _p)
bcp = importlib.util.module_from_spec(_s)
_s.loader.exec_module(bcp)


def _write(tmp_path, team_stats=True):
    hv, ad, ts = _fixture()
    hv.to_parquet(tmp_path / "havoc_games.parquet")
    ad.to_parquet(tmp_path / "advanced_games.parquet")
    pd.DataFrame([_game(2026, "1", "2", [7, 7, 7, 7], [3, 3, 3, 3]), _game(2025, "3", "4", [0, 7, 0, 7], [7, 0, 7, 0])]).to_parquet(
        tmp_path / "cfbd_games.parquet")
    if team_stats:
        ts.to_parquet(tmp_path / "team_game_stats.parquet")


def test_build_defaults_to_the_latest_havoc_season_and_filters_to_fbs(tmp_path):
    _write(tmp_path)
    season, ins, shares = bcp.build(tmp_path)
    assert season == 2026 and set(ins["team"]) == {"1", "2", "3", "4", "5", "6"} and set(shares["team"]) == {"1", "2", "3", "4"}
    _, ins2, shares2 = bcp.build(tmp_path, fbs={"1", "2"})
    assert set(ins2["team"]) == {"1", "2"} and set(shares2["team"]) == {"1", "2"}


def test_build_without_team_stats_warns_and_nulls_turnovers(tmp_path, capsys):
    _write(tmp_path, team_stats=False)
    _, ins, _ = bcp.build(tmp_path)
    assert ins["turnover_margin"].isna().all() and "team_game_stats.parquet missing" in capsys.readouterr().out


def test_build_missing_required_asset_exits(tmp_path):
    with pytest.raises(SystemExit):
        bcp.build(tmp_path)


def test_main_dry_run_and_upsert(tmp_path, monkeypatch, capsys):
    _write(tmp_path)
    monkeypatch.setattr(bcp, "ASSETS", tmp_path)
    monkeypatch.setattr(bcp, "load_fbs_ids", lambda: {"1", "2", "3", "4"})
    sent = {}
    monkeypatch.setattr(bcp.db, "upsert_cfb_team_insights", lambda rows: sent.setdefault("ins", rows) and len(rows))
    monkeypatch.setattr(bcp.db, "upsert_cfb_quarter_shares", lambda rows: sent.setdefault("sh", rows) and len(rows))
    bcp.main(["--dry-run"])
    assert not sent and "team_insights=4" in capsys.readouterr().out
    monkeypatch.setattr(bcp.config, "DATABASE_URL", "postgres://x")
    bcp.main(["--season", "2026"])
    assert {r["team"] for r in sent["ins"]} == {"1", "2", "3", "4"} and sent["sh"][0]["games_used"] >= 1
    monkeypatch.setattr(bcp.config, "DATABASE_URL", None)
    with pytest.raises(SystemExit):
        bcp.main([])
```

Append to the end of `tests/test_workflows_cfb_advanced.py`:

```python
def test_monday_job_refreshes_game_data_commits_four_parquets_then_rebuilds_the_panel_tables():
    from tests.test_workflows_props_ml import runs, step_index, steps_of
    (job,) = load(NAME)["jobs"].values()
    steps = steps_of(job)
    adv = step_index(steps, runs("scripts/build_cfb_advanced.py"))
    pull = step_index(steps, runs("scripts/build_cfb_game_data.py"))
    commit = step_index(steps, runs("git commit"))
    panels = step_index(steps, runs("scripts/build_cfb_panels.py"))
    assert adv < pull < commit < panels                       # the assets are committed before the DB rebuild
    assert steps[pull]["env"]["CFBD_API_KEY"] == "${{ secrets.CFBD_API_KEY }}"
    assert 'scripts/build_cfb_game_data.py --datasets games havoc team_stats --seasons "$(date +%Y)"' in steps[pull]["run"]
    for f in ("advanced_games", "cfbd_games", "havoc_games", "team_game_stats"):
        assert f"assets/cfb/{f}.parquet" in steps[commit]["run"]
    assert steps[panels]["env"]["DATABASE_URL"] == "${{ secrets.DATABASE_URL }}"
    assert "CFBD_API_KEY" not in steps[panels].get("env", {})
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest -q tests/scripts/test_build_cfb_panels.py tests/test_workflows_cfb_advanced.py`
Expected: FAIL — `ERROR tests/scripts/test_build_cfb_panels.py - FileNotFoundError: [Errno 2] N...`.

- [ ] **Step 3: Implement the script**

Create `scripts/build_cfb_panels.py`:

```python
"""Weekly CFB panel tables from the committed CFBD assets -> Supabase (db/migration_site_panels.sql):

* `cfb_team_insights`  -- season-to-date havoc / turnover margin / explosiveness with national ranks
  (cfb.panels.team_insights) from havoc_games, advanced_games and team_game_stats.
* `cfb_quarter_shares` -- per-team Q1-Q4 scoring shares, last two completed seasons + this one, shrunk to the
  league (cfb.panels.quarter_shares) from cfbd_games' line scores.

Runs after build-cfb-advanced.yml has refreshed those assets (Mondays). Pure assets in, upserts out: no CFBD
call here. `--season` defaults to the latest season with regular-season havoc rows. `--dry-run` prints counts
and a sample and writes nothing.

Usage:
    uv run python scripts/build_cfb_panels.py [--season 2026] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from sportsmodel import config, db  # noqa: E402
from sportsmodel.cfb import panels  # noqa: E402
from sportsmodel.cfb.teams import load_fbs_ids  # noqa: E402

ASSETS = ROOT / "assets" / "cfb"


def warn(msg: str) -> None:
    print(f"::warning::cfb-panels: {msg}", flush=True)


def _read(assets: Path, name: str, required: bool) -> pd.DataFrame | None:
    p = assets / name
    if p.exists():
        return pd.read_parquet(p)
    if required:
        sys.exit(f"{p} is missing (run scripts/build_cfb_game_data.py / build_cfb_advanced.py first)")
    warn(f"{name} missing; its columns stay NULL")
    return None


def build(assets: Path | None = None, season: int | None = None, fbs: set[str] | None = None
          ) -> tuple[int, pd.DataFrame, pd.DataFrame]:
    """(season, insights frame, quarter-share frame) from the assets under `assets` (default: ASSETS)."""
    assets = assets or ASSETS
    havoc = _read(assets, "havoc_games.parquet", True)
    advanced = _read(assets, "advanced_games.parquet", True)
    games = _read(assets, "cfbd_games.parquet", True)
    team_stats = _read(assets, "team_game_stats.parquet", False)
    if season is None:
        reg = havoc[havoc["season_type"].fillna("regular") == "regular"]
        if reg.empty:
            sys.exit("havoc_games.parquet has no regular-season rows")
        season = int(reg["season"].max())
    ins = panels.team_insights(havoc, advanced, team_stats, season, fbs=fbs)
    shares = panels.quarter_shares(games, season, fbs=fbs)
    if "home_q1" not in games.columns or games["home_q1"].notna().sum() == 0:
        warn("cfbd_games.parquet has no line scores (run the games backfill); no quarter shares")
    return season, ins, shares


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--season", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true", help="compute and print; no DB writes")
    args = ap.parse_args(argv)
    if not args.dry_run and not config.DATABASE_URL:
        sys.exit("DATABASE_URL is not set (use --dry-run to compute without writing)")
    season, ins, shares = build(season=args.season, fbs=set(load_fbs_ids()))
    print(f"cfb panels season {season}: team_insights={len(ins)} (ranked {int(ins['n_ranked'].max()) if len(ins) else 0}, "
          f"through week {int(ins['through_week'].max()) if len(ins) else '-'}), quarter_shares={len(shares)}", flush=True)
    if args.dry_run:
        print(ins.head(3).to_string(index=False))
        print(shares.head(3).to_string(index=False))
        return
    print(f"upserted: cfb_team_insights={db.upsert_cfb_team_insights(panels.to_rows(ins))} "
          f"cfb_quarter_shares={db.upsert_cfb_quarter_shares(panels.to_rows(shares))}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Extend the Monday workflow**

`build-cfb-advanced.yml` already refreshes `advanced_games.parquet` every Monday 12:00 UTC. No workflow refreshes `havoc_games.parquet` today (the spec assumed one does), so this run now also pulls the current season's `games` (line scores), `havoc` and `team_stats` (turnovers), commits all four parquets, and only then rebuilds the two Supabase tables, so a failed table rebuild never blocks the asset commit. The Monday CFBD cost is about 2 (advanced) + 2 (games) + 2 (havoc) + one per played week (turnovers, at most about 15).

In `.github/workflows/build-cfb-advanced.yml`, replace this text (it occurs exactly once):

```yaml
# --merge: only the pulled seasons are replaced, the rest of the parquet is kept.
# CFBD_API_KEY is a repo secret; the key is never printed.
```

with:

```yaml
# --merge: only the pulled seasons are replaced, the rest of the parquet is kept.
# The same run also refreshes the CURRENT season's CFBD havoc, game results with line scores and
# turnover stats (scripts/build_cfb_game_data.py --datasets games havoc team_stats; the turnover pull
# is one call per played week, hence current season only), commits all four parquets, and then rebuilds
# the site's cfb_team_insights + cfb_quarter_shares tables in Supabase (scripts/build_cfb_panels.py,
# DATABASE_URL; db/migration_site_panels.sql). A failed table rebuild never blocks the asset commit.
# CFBD_API_KEY is a repo secret; the key is never printed.
```

In `.github/workflows/build-cfb-advanced.yml`, replace this text (it occurs exactly once):

```yaml
      - name: Commit advanced asset
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add assets/cfb/advanced_games.parquet
```

with:

```yaml
      - name: Pull CFBD havoc, line scores and turnovers (current season)
        env:
          CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
        run: |
          uv run python scripts/build_cfb_game_data.py --datasets games havoc team_stats --seasons "$(date +%Y)"
      - name: Commit advanced asset
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add assets/cfb/advanced_games.parquet assets/cfb/cfbd_games.parquet assets/cfb/havoc_games.parquet assets/cfb/team_game_stats.parquet
```

In `.github/workflows/build-cfb-advanced.yml`, replace this text (it occurs exactly once):

```yaml
          git commit -m "data(cfb): CFBD advanced game stats (build-cfb-advanced) [skip ci]"
```

with:

```yaml
          git commit -m "data(cfb): CFBD advanced game stats, havoc, line scores, turnovers (build-cfb-advanced) [skip ci]"
```

In `.github/workflows/build-cfb-advanced.yml`, replace this text (it occurs exactly once):

```yaml
          echo "push failed after 3 attempts" >&2
          exit 1
```

with:

```yaml
          echo "push failed after 3 attempts" >&2
          exit 1
      - name: Rebuild site panel tables (team insights, quarter shares)
        env:
          DATABASE_URL: ${{ secrets.DATABASE_URL }}
        run: uv run python scripts/build_cfb_panels.py
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest -q tests/scripts/test_build_cfb_panels.py tests/test_workflows_cfb_advanced.py tests/test_workflows_props_ml.py tests/test_workflows_game_info.py`
Expected: PASS (`115 passed` in this plan's dry run).

- [ ] **Step 6: Dry-run on the real committed assets (local, no network, no DB)**

Expected: `cfb panels season 2026: team_insights=~135 (ranked ~130, through week N), quarter_shares=~130` and three sample rows of each; `through week` equals the latest 2026 havoc week printed by T3's check. If `quarter_shares` is 0 the games backfill did not land: re-check T3.

```bash
uv run python scripts/build_cfb_panels.py --dry-run
```

- [ ] **Step 7: Commit**

```bash
git add scripts/build_cfb_panels.py .github/workflows/build-cfb-advanced.yml tests/scripts/test_build_cfb_panels.py tests/test_workflows_cfb_advanced.py
git commit -m "feat: weekly CFB panel tables job (team insights, quarter shares) and Monday asset refresh

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 10: Site: venue / weather hero line

**Files:**
- Modify: `site/js/pages/game.js` (header comment, `gmWeatherPieces`, `gmVenueLine`, `gmLoad`, `gmHero`)
- Modify: `site/css/theme.css` (`.ca-gm-venue`)
- Test: `site/tests/game_panels.test.mjs` (create)

**Interfaces:**
- Consumes: Supabase table `game_info` (T7; columns `sport, game_pk, venue_name, city, state, indoor, temp_f, wind_mph, precip_chance, precip_in, conditions, weather_kind`) read with `sb()`; existing `numOrNull`, `ctxEsc` (`gmEsc`), `sb`.
- Produces: `GM_MEASURABLE_IN = 0.01` (inches); `gmWeatherPieces(info) -> string[]` (conditions, `41°F`, `wind 14 mph`, `20% rain` or `0.04 in rain`; missing pieces omitted, `0%` not shown, rain under 0.01 in not shown); `gmVenueLine(info) -> string` (`Venue · City, ST · weather`, `Indoors` when `info.indoor === true`, weather prefixed `Observed ` when `weather_kind === "observed"`, `""` when nothing to show); `D.gameInfo` (the row or `null`) on the object `gmLoad` returns; `<p class="ca-gm-venue">` inside the hero, between the kickoff row and the teams row.

- [ ] **Step 1: Write the failing tests**

The new test file loads the same scripts as `game.test.mjs` through the vm loader and drives `buildGamePage()` with a stubbed `fetch`; a table that does not exist yet is modelled as a 404 (the real `sb()` throws, `gmLoad` catches it and the hero line is simply absent).

Create `site/tests/game_panels.test.mjs`:

```javascript
import test from "node:test";
import assert from "node:assert/strict";
import { loadScripts } from "./load.mjs";
const FILES = ["app.js", "js/metrics.js", "js/ui.js", "js/shell.js", "js/data.js", "js/pages/game.js", "js/boot.js"];
const g = loadScripts(FILES);

/* ── hero venue / weather line ──────────────────────────────────────────── */
const INFO = { venue_name: "Bryant-Denny Stadium", city: "Tuscaloosa", state: "AL", indoor: false, temp_f: 41.2, wind_mph: 14.4, precip_chance: 20, precip_in: null, conditions: null, weather_kind: "forecast" };

test("gmVenueLine: every piece, in order, joined with ' · '", () => {
  assert.equal(g.gmVenueLine(INFO), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph, 20% rain");
});

test("gmVenueLine: a missing piece is omitted, never guessed", () => {
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: null }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph");
  assert.equal(g.gmVenueLine({ ...INFO, state: null }), "Bryant-Denny Stadium · Tuscaloosa · 41°F, wind 14 mph, 20% rain");
  assert.equal(g.gmVenueLine({ ...INFO, city: "", state: "" }), "Bryant-Denny Stadium · 41°F, wind 14 mph, 20% rain");
  assert.equal(g.gmVenueLine({ ...INFO, venue_name: null, temp_f: null, wind_mph: null, precip_chance: null }), "Tuscaloosa, AL");
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: 0 }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph", "0% is not shown");
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: null, precip_in: 0.04 }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph, 0.04 in rain");
  assert.equal(g.gmVenueLine({ ...INFO, precip_chance: null, precip_in: 0.004 }), "Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 14 mph", "below 0.01 in is not measurable");
  assert.equal(g.gmVenueLine({ ...INFO, conditions: "Cloudy", temp_f: "38" }), "Bryant-Denny Stadium · Tuscaloosa, AL · Cloudy, 38°F, wind 14 mph, 20% rain");
  assert.equal(g.gmVenueLine({ venue_name: "V", temp_f: null, wind_mph: "n/a" }), "V");
});

test("gmVenueLine: indoors replaces the weather; observed readings are labelled; nothing to show -> empty", () => {
  assert.equal(g.gmVenueLine({ ...INFO, indoor: true }), "Bryant-Denny Stadium · Tuscaloosa, AL · Indoors");
  assert.equal(g.gmVenueLine({ ...INFO, weather_kind: "observed", precip_chance: null }), "Bryant-Denny Stadium · Tuscaloosa, AL · Observed 41°F, wind 14 mph");
  for (const none of [null, undefined, {}, "x", { venue_name: "", city: " ", state: null, indoor: null }]) assert.equal(g.gmVenueLine(none), "");
});

/* ── page fixtures ──────────────────────────────────────────────────────── */
const urlPath = (url) => new URL(url).pathname.split("/").pop();
const PRED = (sport, over = {}) => ({ game_pk: 9, sport, home_team_name: sport === "cfb" ? "Alabama Crimson Tide" : "Kansas City Chiefs", away_team_name: sport === "cfb" ? "Georgia Bulldogs" : "Buffalo Bills",
  commence_time: "2026-10-12T23:00:00Z", home_win_prob: 0.7, pred_home_score: 30, pred_away_score: 20, market_spread: -7.5, market_total: 50.5, ...over });
const HIST = [{ side: "away", team: "61", windows: { season: { n: 5, su: "4-1" } }, streaks: {} }, { side: "home", team: "333", windows: { season: { n: 5, su: "5-0" } }, streaks: {} }];
function page({ sport = "cfb", rows = {}, missing = [], pred = {} } = {}) {
  const search = `?sport=${sport}&game=9`;
  const requested = [];
  const fetch = async (url) => {
    const path = urlPath(url); requested.push(decodeURIComponent(path + new URL(url).search));
    if (missing.includes(path)) return { ok: false, status: 404, text: async () => "relation does not exist", json: async () => ({}) };
    const base = { predictions_any: [PRED(sport, pred)], team_history: sport === "cfb" ? HIST : [] };
    return { ok: true, json: async () => rows[path] ?? base[path] ?? [] };
  };
  const D = loadScripts(FILES, { page: "game", globals: { fetch, location: { search, href: `http://localhost/game.html${search}` } } });
  return { D, requested };
}
const heroOf = (html) => html.slice(html.indexOf("ca-gm-hero"), html.indexOf("ca-gm-read"));

test("hero: the venue line sits under the kickoff line, escaped, for CFB and NFL; absent without a game_info row or table", async () => {
  const evil = { ...INFO, venue_name: 'Bryant <b>"Denny"</b>' };
  for (const sport of ["cfb", "nfl"]) {
    const html = await page({ sport, rows: { game_info: [evil] } }).D.buildGamePage();
    const hero = heroOf(html);
    assert.ok(hero.includes('<p class="ca-gm-venue">Bryant &lt;b&gt;&quot;Denny&quot;&lt;/b&gt; · Tuscaloosa, AL · 41°F, wind 14 mph, 20% rain</p>'), sport);
    assert.ok(hero.indexOf("ca-gm-when") < hero.indexOf("ca-gm-venue") && hero.indexOf("ca-gm-venue") < hero.indexOf("ca-gm-teams"));
    assert.ok(!html.includes("<b>\"Denny\"</b>"));
  }
  const none = await page({ rows: { game_info: [] } }).D.buildGamePage();
  assert.ok(!none.includes("ca-gm-venue"), "no row: no line");
  const noTable = await page({ missing: ["game_info"] }).D.buildGamePage();
  assert.ok(!noTable.includes("ca-gm-venue") && noTable.includes("ca-gm-hero"), "missing table: the page still renders");
  const empty = await page({ rows: { game_info: [{ sport: "cfb", game_pk: 9 }] } }).D.buildGamePage();
  assert.ok(!empty.includes("ca-gm-venue"), "a row with nothing in it: no empty line");
});

test("game_info is read by sport and game_pk for NFL and CFB only", async () => {
  const { D, requested } = page();
  await D.buildGamePage();
  assert.ok(requested.includes("game_info?sport=eq.cfb&game_pk=eq.9"), requested.join("\n"));
  const mlb = page({ sport: "mlb" });
  await mlb.D.buildGamePage();
  assert.ok(!mlb.requested.some((r) => r.startsWith("game_info")), "MLB has no game_info rows");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd site && npm test`
Expected: FAIL — `✖ gmVenueLine: every piece, in order, joined with ' · ' (0.434042ms)`.

- [ ] **Step 3: Implement the line, the read and the hero**

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
(window.__caGameData) without a refetch. Phase B items (venue / weather line, Projected Game Flow, TV network) are omitted.
```

with:

```javascript
(window.__caGameData) without a refetch. The venue / weather hero line (game_info, NFL + CFB), the havoc / turnover / weather Key Insights
   (cfb_team_insights, game_info) and the CFB Projected Game Flow (cfb_quarter_shares, game_info.line_score) render only from rows that
   exist: a missing table or row hides its panel. The TV network is still omitted.
```

In `site/js/pages/game.js` — insert the two helpers above `consensusAmerican`'s comment, replace this text (it occurs exactly once):

```javascript
// Market consensus: the median implied probability
```

with:

```javascript
// Rain below this many inches is not "measurable" (CFBD precipitation is inches).
const GM_MEASURABLE_IN = 0.01;
// Weather pieces in reading order: conditions, "41°F", "wind 14 mph", "20% rain" (or "0.04 in rain"). A missing reading is omitted.
function gmWeatherPieces(i) {
  const t = numOrNull(i.temp_f), w = numOrNull(i.wind_mph), pc = numOrNull(i.precip_chance), pi = numOrNull(i.precip_in);
  const out = [];
  if (i.conditions && String(i.conditions).trim()) out.push(String(i.conditions).trim());
  if (t != null) out.push(`${Math.round(t)}°F`);
  if (w != null) out.push(`wind ${Math.round(w)} mph`);
  if (pc != null && pc > 0) out.push(`${Math.round(pc)}% rain`);
  else if (pi != null && pi >= GM_MEASURABLE_IN) out.push(`${pi.toFixed(2)} in rain`);
  return out;
}
// The hero's venue line from a game_info row: "Venue · City, ST · 41°F, wind 14 mph, 20% rain". Any missing piece is omitted; an indoor
// venue says "Indoors" in place of the weather; a CFBD reading taken after kickoff is prefixed "Observed". Plain text ("" = hide the line).
function gmVenueLine(info) {
  if (!info || typeof info !== "object") return "";
  const place = [info.city, info.state].filter((x) => x && String(x).trim()).join(", ");
  let wx = "";
  if (info.indoor === true) wx = "Indoors";
  else { const p = gmWeatherPieces(info); if (p.length) wx = (info.weather_kind === "observed" ? "Observed " : "") + p.join(", "); }
  return [info.venue_name, place, wx].filter((x) => x && String(x).trim()).join(" · ");
}

// Market consensus: the median implied probability
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
  const [predsAny, predsCur, evRows, accRows, servedVersion, mls, opps, moves, lineBy] = await Promise.all([
```

with:

```javascript
  const [predsAny, predsCur, evRows, accRows, servedVersion, mls, opps, moves, lineBy, gameInfoRows] = await Promise.all([
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
    live ? evBestLines().catch(() => new Map()) : none(new Map()),
  ]);
```

with:

```javascript
    live ? evBestLines().catch(() => new Map()) : none(new Map()),
    // venue / weather (+ CFB line score); the table may not exist yet -> [] and the hero line / weather row / actual quarters hide
    isNfl || sport === "cfb" ? sb(`game_info?sport=eq.${sport}&game_pk=eq.${game}`).catch(() => []) : none([]),
  ]);
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
trendRecs, trendSits, ctxHist, ctxGrades, ctxPower, awayCol, homeCol,
    isNfl, simTag:
```

with:

```javascript
trendRecs, trendSits, ctxHist, ctxGrades, ctxPower, awayCol, homeCol,
    gameInfo: (gameInfoRows || [])[0] || null,
    isNfl, simTag:
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
  return `<section class="ca-gm-hero${boxes ? " has-odds" : ""}"><div class="ca-gm-top"><a class="ca-gm-back" href="${sport}.html">‹ ${SPORT_NAME[sport] || sport.toUpperCase()} Board</a><span class="ca-gm-when">${gmEsc(when)}</span><span></span></div>
    <div class="ca-gm-teams">
```

with:

```javascript
  const venue = gmVenueLine(D.gameInfo);
  return `<section class="ca-gm-hero${boxes ? " has-odds" : ""}"><div class="ca-gm-top"><a class="ca-gm-back" href="${sport}.html">‹ ${SPORT_NAME[sport] || sport.toUpperCase()} Board</a><span class="ca-gm-when">${gmEsc(when)}</span><span></span></div>
    ${venue ? `<p class="ca-gm-venue">${gmEsc(venue)}</p>` : ""}<div class="ca-gm-teams">
```

In `site/css/theme.css`, replace this text (it occurs exactly once):

```css
.ca-gm-when{color:#DCE6F2;font-weight:500;text-align:center}
```

with:

```css
.ca-gm-when{color:#DCE6F2;font-weight:500;text-align:center}
.ca-gm-venue{margin:8px 0 0;text-align:center;color:#C9D6E6;font-size:14px;line-height:1.35}
```

- [ ] **Step 4: Run the tests to verify they pass (every existing game-page test included)**

Run: `cd site && npm test`
Expected: PASS (`ℹ pass 312, ℹ fail 0` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add site/js/pages/game.js site/css/theme.css site/tests/game_panels.test.mjs
git commit -m "feat(site): venue and weather line in the game hero

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 11: Site: Key Insights strength model + havoc, turnover and weather rows

**Files:**
- Modify: `site/js/pages/game.js` (icons, `gmKeyInsights` and its helpers, `GM_INSIGHT_ICON`, `gmLoad`)
- Modify: `site/tests/game.test.mjs` (one existing test: ordering is now by strength)
- Test: `site/tests/game_panels.test.mjs` (append)

**Interfaces:**
- Consumes: T10 `D.gameInfo`, `GM_MEASURABLE_IN`; Supabase `cfb_team_insights` (T7: `season, team, games, n_ranked, def_havoc_rate/rank, off_havoc_allowed_rate/rank, turnover_margin_per_game/rank, ...`) and `cfb_quarter_shares` (T7) read with `sb()`; existing `gmTeamCode(hist, grades, side)`, `gmPoss`, `shortTeam`, `signedStr`, `finite`, `numOrNull`.
- Produces: `gmKeyInsights(D) -> {kind, strength, title, body}[]` (at most 4, strongest first, ties in generation order); `GM_STRENGTH` (`grade: pct -> round(30 + 0.4 * pct)`, `explosive: () -> 45`, `power: gap -> 30 + min(40, gap)`, `streak: n -> min(70, 30 + 8 * (n - 2))`); `GM_NOTABLE_GAP = 30`, `GM_NOTABLE_PCT = 0.15`, `gmCut(n) = ceil(0.15 * n)`; `gmInsRow(D, side) -> row | null` (newest-season `cfb_team_insights` row of the side's team code); `gmHavocInsight(D, name)`, `gmTurnoverInsight(D, name)` (strength `round(50 + min(45, gap / 3))`), `gmWeatherInsight(D)` (strength 55 + severity, max over wind / cold / rain factors); `GM_INSIGHT_ICON` gains `havoc`, `turnover`, `weather`; `D.cfbIns`, `D.cfbShares` (arrays, `[]` when the table is missing or empty).

- [ ] **Step 1: Write the failing tests**

Thresholds under test: havoc = rank gap (offense havoc-allowed rank minus defense havoc rank) >= 30, or an edge (gap > 0) where the defense is in the top 15% or the offense in the bottom 15% of the ranked field (`ceil(0.15 * n_ranked)`); turnover = margin-rank gap >= 30 or either team in the top / bottom 15%, equal ranks excluded; weather = wind >= 15 mph, temp < 40°F, precip chance >= 50% or >= 0.01 in, outdoor only. The existing four-row test changes only in its ordering assertions (now by strength), updated in Step 3.

Append to the end of `site/tests/game_panels.test.mjs`:

```javascript
/* ── Key Insights: havoc, turnover, weather candidates ──────────────────── */
const R = (sport = "cfb") => PRED(sport);
const insRow = (team, o = {}) => ({ season: 2026, team, games: 5, n_ranked: 134, def_havoc_rate: 0.2132, def_havoc_rank: 60, off_havoc_allowed_rate: 0.1812, off_havoc_allowed_rank: 60,
  turnover_margin: 0, turnover_margin_per_game: 0, turnover_margin_rank: 67, ...o });
const base = (o = {}) => ({ sport: "cfb", r: R(), ctxGrades: [], ctxHist: HIST, ctxPower: [], cfbIns: [], gameInfo: null, ...o });
const kinds = (D) => g.gmKeyInsights(D).map((i) => i.kind);

test("havoc mismatch: a 30+ rank gap (defense edge), or an edge involving a top / bottom 15% unit; strength 50 + gap / 3 (max 95)", () => {
  const D = base({ cfbIns: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118, off_havoc_allowed_rate: 0.2491 })] });
  const [h] = g.gmKeyInsights(D);
  assert.equal(h.kind, "havoc"); assert.equal(h.strength, 87);      // 50 + min(45, 110 / 3)
  assert.equal(h.title, "Alabama defense creates havoc against Georgia");
  assert.equal(h.body, "Alabama ranks #8 in havoc created (21.3% of plays); Georgia's offense ranks #118 in havoc allowed (24.9%), where #1 allows the least.");
  // gap under 30 but the defense is top 15% (<= ceil(.15 * 134) = 21) and has the edge
  const ext = g.gmKeyInsights(base({ cfbIns: [insRow("333", { def_havoc_rank: 5 }), insRow("61", { off_havoc_allowed_rank: 15 })] }));
  assert.deepEqual([ext[0].kind, ext[0].strength], ["havoc", 53]);
  // the offense is bottom 15% (> 134 - 21 = 113) with a small edge
  assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: 110 }), insRow("61", { off_havoc_allowed_rank: 114 })] })).includes("havoc"), true);
  // not notable: gap 29 in the middle of the pack; a negative gap (the offense protects the ball better); extremes without the edge
  for (const [d, o] of [[40, 69], [90, 30], [10, 5], [60, 60]])
    assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: d }), insRow("61", { off_havoc_allowed_rank: o })] })).includes("havoc"), false, `${d}/${o}`);
});

test("havoc mismatch: the stronger direction wins; unranked / missing rows and non-CFB games show nothing", () => {
  const D = base({ cfbIns: [insRow("333", { def_havoc_rank: 8, off_havoc_allowed_rank: 125 }), insRow("61", { def_havoc_rank: 40, off_havoc_allowed_rank: 100 })] });
  const [h] = g.gmKeyInsights(D);                                    // home D 8 vs away O 100 = 92; away D 40 vs home O 125 = 85
  assert.match(h.title, /^Alabama defense creates havoc against Georgia$/); assert.equal(h.strength, 81);
  const unranked = base({ cfbIns: [insRow("333", { def_havoc_rank: null }), insRow("61", { off_havoc_allowed_rank: 118 })] });
  assert.equal(kinds(unranked).includes("havoc"), false);
  assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: 8, n_ranked: null }), insRow("61", { off_havoc_allowed_rank: 118, n_ranked: undefined })] })).includes("havoc"), false, "no field size, no 15% rule");
  assert.equal(kinds(base({ cfbIns: [insRow("333", { def_havoc_rank: 8 })] })).includes("havoc"), false, "one team only");
  assert.equal(kinds(base({ sport: "nfl", cfbIns: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })] })).includes("havoc"), false, "CFB only");
  const old = [insRow("333", { season: 2025, def_havoc_rank: 90 }), insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })];   // newest season first in practice
  assert.equal(g.gmKeyInsights(base({ cfbIns: [old[1], old[0], old[2]] }))[0].kind, "havoc", "the first (newest) row per team is used");
});

test("turnover edge: rank gap 30+ or a top / bottom 15% team; names the better team; equal ranks show nothing", () => {
  const D = base({ cfbIns: [insRow("333", { turnover_margin_rank: 6, turnover_margin_per_game: 1.8 }), insRow("61", { turnover_margin_rank: 112, turnover_margin_per_game: -0.9 })] });
  const [t] = g.gmKeyInsights(D);
  assert.equal(t.kind, "turnover"); assert.equal(t.strength, 85);   // 50 + min(45, 106 / 3)
  assert.equal(t.title, "Alabama owns the turnover edge");
  assert.equal(t.body, "Alabama is +1.8 per game (#6 nationally); Georgia is −0.9 per game (#112).");
  const away = g.gmKeyInsights(base({ cfbIns: [insRow("333", { turnover_margin_rank: 90 }), insRow("61", { turnover_margin_rank: 20, turnover_margin_per_game: 1.1 })] }));
  assert.equal(away[0].title, "Georgia owns the turnover edge", "gap 70, rank 20 is inside the top 21");
  assert.equal(kinds(base({ cfbIns: [insRow("333", { turnover_margin_rank: 10 }), insRow("61", { turnover_margin_rank: 25 })] })).includes("turnover"), true, "gap 15 but a top-15% team");
  for (const [a, b] of [[50, 60], [67, 67], [30, 59]]) assert.equal(kinds(base({ cfbIns: [insRow("333", { turnover_margin_rank: a }), insRow("61", { turnover_margin_rank: b })] })).includes("turnover"), false, `${a}/${b}`);
  assert.equal(kinds(base({ cfbIns: [insRow("333", { turnover_margin_rank: null }), insRow("61", { turnover_margin_rank: 112 })] })).includes("turnover"), false);
  const bare = g.gmKeyInsights(base({ cfbIns: [insRow("333", { turnover_margin_rank: 6, turnover_margin_per_game: null }), insRow("61", { turnover_margin_rank: 112, turnover_margin_per_game: null })] }));
  assert.equal(bare[0].body, "Alabama ranks (#6 nationally); Georgia ranks (#112).", "no per-game number, no invented one");
});

test("weather insight: wind 15+, under 40°F, 50%+ chance or measurable rain; outdoor only; CFB and NFL; strength from the worst factor", () => {
  const wx = (o) => g.gmKeyInsights(base({ gameInfo: { ...INFO, wind_mph: 5, temp_f: 60, precip_chance: null, precip_in: null, ...o } })).find((i) => i.kind === "weather");
  const a = wx({ wind_mph: 18 });
  assert.deepEqual([a.strength, a.title], [64, "Weather: wind 18 mph"]);        // 55 + (18 - 15) * 3
  assert.equal(a.body, "Forecast conditions at Bryant-Denny Stadium. Descriptive only, not a pick.");
  const b = wx({ wind_mph: 20, temp_f: 35, precip_chance: 60 });
  assert.deepEqual([b.strength, b.title], [70, "Weather: wind 20 mph, 35°F, 60% chance of rain"]);   // max(70, 65, 61)
  assert.equal(wx({ temp_f: 20 }).strength, 85);                                 // 55 + min(30, (40 - 20) * 2)
  assert.equal(wx({ temp_f: 39.4 }).title, "Weather: 39°F");
  assert.equal(wx({ precip_in: 0.05 }).title, "Weather: 0.05 in of rain");
  assert.equal(wx({ precip_in: 0.05 }).strength, 60);
  assert.match(wx({ wind_mph: 18, weather_kind: "observed" }).body, /^Observed conditions at/);
  for (const calm of [{}, { wind_mph: 14.9 }, { temp_f: 40 }, { precip_chance: 49 }, { precip_in: 0.004 }, { indoor: true, wind_mph: 30 }, { wind_mph: null, temp_f: null }]) assert.equal(wx(calm), undefined, JSON.stringify(calm));
  assert.equal(g.gmKeyInsights(base({ gameInfo: null })).find((i) => i.kind === "weather"), undefined);
  const nfl = g.gmKeyInsights({ sport: "nfl", r: R("nfl"), ctxGrades: [], ctxHist: [], ctxPower: [], gameInfo: { ...INFO, wind_mph: 22, venue_name: null } });
  assert.equal(nfl[0].kind, "weather"); assert.equal(nfl[0].body, "Forecast conditions. Descriptive only, not a pick.");
});

test("Key Insights keeps the strongest four, strongest first, ties in generation order", () => {
  const grades = [{ side: "home", team: "333", overall: "A", pass: "A", run: "B", overall_pct: 90, units: null }];
  const power = [{ team: "333", rank: 4, rating: 20 }, { team: "61", rank: 40, rating: 2 }];
  const D = base({ ctxGrades: grades, ctxPower: power, ctxHist: [{ side: "home", team: "333", streaks: { su: "W5" } }, { side: "away", team: "61", streaks: {} }],
    cfbIns: [insRow("333", { def_havoc_rank: 8, turnover_margin_rank: 6, turnover_margin_per_game: 1.8 }), insRow("61", { off_havoc_allowed_rank: 118, turnover_margin_rank: 112, turnover_margin_per_game: -0.9 })],
    gameInfo: { ...INFO, wind_mph: 18 } });
  const out = g.gmKeyInsights(D);
  // grade 30 + 0.4 * 90 = 66 and power 30 + min(40, 36) = 66 tie: generation order (grade first). weather 64 and streak W5 54 are cut.
  assert.deepEqual(out.map((i) => [i.kind, i.strength]), [["havoc", 87], ["turnover", 85], ["grade", 66], ["power", 66]]);
  const again = g.gmKeyInsights(D);
  assert.deepEqual(again, out, "deterministic");
});

test("Key Insights card: the new rows render with their icons and escaped text; missing tables leave the old rows alone", async () => {
  const rows = { cfb_team_insights: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })],
    game_info: [{ ...INFO, venue_name: "<i>Field</i>", wind_mph: 18 }], team_history: HIST };
  const html = await page({ rows }).D.buildGamePage();
  const ins = html.slice(html.indexOf('id="gm-insights"'), html.indexOf('id="gm-cover"'));
  assert.ok(ins.includes("Alabama defense creates havoc against Georgia") && ins.includes("Weather: wind 18 mph"));
  assert.ok(ins.includes("&lt;i&gt;Field&lt;/i&gt;") && !ins.includes("<i>Field</i>"));
  assert.equal((ins.match(/class="ca-gm-ins"/g) || []).length, 2);
  assert.ok(ins.includes("ca-gm-ic-red") && ins.includes("ca-gm-ic-blue"));
  const none = await page({ missing: ["cfb_team_insights", "cfb_quarter_shares", "game_info"], rows: { team_history: HIST } }).D.buildGamePage();
  assert.ok(none.includes("ca-gm-hero") && !none.includes("creates havoc") && !none.includes("Weather:"), "tables missing: the page renders without the new rows");
});

test("cfb_team_insights / cfb_quarter_shares are read for both team codes (CFB only)", async () => {
  const cfb = page();
  await cfb.D.buildGamePage();
  assert.ok(cfb.requested.includes('cfb_team_insights?team=in.("61","333")&order=season.desc'), cfb.requested.join("\n"));
  assert.ok(cfb.requested.includes('cfb_quarter_shares?team=in.("61","333")'));
  const nfl = page({ sport: "nfl" });
  await nfl.D.buildGamePage();
  assert.ok(!nfl.requested.some((r) => r.startsWith("cfb_")), "NFL never reads the CFB tables");
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd site && npm test`
Expected: FAIL — `✖ havoc mismatch: a 30+ rank gap (defense edge), or an edge involving a top / bottom 15% unit; strength 50 + gap / 3 (max 95) (0.301583ms)`.

- [ ] **Step 3: Implement the strength model, the three new rows and the loads**

In `site/js/pages/game.js` — three new 22px line icons, replace this text (it occurs exactly once):

```javascript
const ICON_RANK = `
```

with:

```javascript
const ICON_HAVOC = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2L4 14h7l-1 8 9-12h-7z"/></svg>`;
const ICON_SWAP = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 7h11l-3-3M17 17H6l3 3"/></svg>`;
const ICON_CLOUD = `<svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M7 18a4 4 0 0 1-.5-7.97A5.5 5.5 0 0 1 17 8.5 4.5 4.5 0 0 1 17 18z"/></svg>`;
const ICON_RANK = `
```

In `site/js/pages/game.js` — replace the whole `gmKeyInsights` block, from its comment line through `return out.slice(0, 4);` and its closing brace, replace this text (it occurs exactly once):

```javascript
// Up to four generated sentences from data that exists: offense-vs-defense unit grade, explosive-play edge, power-rank
// gap, current streak. Each is {kind, title, body} (plain text; escaped when rendered).
const gmPoss = (n) => (/s$/i.test(n) ? `${n}'` : `${n}'s`);
// team_history / matchup_grades / power_rankings identify a side by team code.
const gmTeamCode = (hist, grades, side) => { const x = (hist || []).find((y) => y.side === side) || (grades || []).find((y) => y.side === side); return x ? String(x.team) : null; };
const GM_STREAK = { su: { W: ["won", "won"], L: ["lost", "lost"] }, ats: { W: ["covered", "covered"], L: ["failed to cover", "failed to cover"] },
  ou: { O: ["has gone over in", "have gone over in"], U: ["has gone under in", "have gone under in"] } };
function gmKeyInsights(D) {
  const r = (D && D.r) || {}, grades = (D && D.ctxGrades) || [], hist = (D && D.ctxHist) || [], power = (D && D.ctxPower) || [];
  const name = (side) => shortTeam(side === "home" ? r.home_team_name : r.away_team_name, D && D.sport);
  const other = (s) => (s === "home" ? "away" : "home");
  const pl = (D && D.sport) === "nfl" ? 1 : 0;   // NFL short names are plural nicknames ("Saints rank"), CFB schools singular ("Alabama ranks")
  const out = [];
  const graded = grades.filter((g) => g && g.overall && finite(g.overall_pct)).sort((x, y) => y.overall_pct - x.overall_pct)[0];
  if (graded) out.push({ kind: "grade", title: `${name(graded.side)} offense grades ${graded.overall} vs ${name(other(graded.side))}`,
    body: `Pass ${graded.pass || "–"}, run ${graded.run || "–"} against ${gmPoss(name(other(graded.side)))} defense (${ctxOrd(graded.overall_pct)} percentile overall).` });
  const xpl = (g) => {
    const u = g && ctxJson(g.units), o = u && u.off, d = u && u.def;
    if (!o || !d) return null;
    const v = [o.pass_explosive, d.pass_explosive, o.run_explosive, d.run_explosive].map(numOrNull);
    if (v.some((x) => x == null)) return null;
    const pr = numOrNull(u.pass_rate) ?? 0.5;
    return pr * (v[0] + v[1]) + (1 - pr) * (v[2] + v[3]);
  };
  const xh = xpl(grades.find((g) => g.side === "home")), xa = xpl(grades.find((g) => g.side === "away"));
  if (xh != null && xa != null && Math.abs(xh - xa) > 1e-9) {
    const w = xh > xa ? "home" : "away";
    out.push({ kind: "explosive", title: `Big-play grades favor ${name(w)}`,
      body: `${gmPoss(name(w))} offense vs ${gmPoss(name(other(w)))} defense is ahead of the reverse matchup on pass-rate-weighted explosive-play ratings.` });
  }
  const pr = (side) => power.find((x) => String(x.team) === gmTeamCode(hist, grades, side));
  const ph = pr("home"), pa = pr("away");
  if (ph && pa && finite(ph.rank) && finite(pa.rank) && ph.rank !== pa.rank) {
    const [bs, b, w] = ph.rank < pa.rank ? ["home", ph, pa] : ["away", pa, ph];
    const gap = finite(b.rating) && finite(w.rating) ? ` Rating gap: ${(b.rating - w.rating).toFixed(1)} pts.` : "";
    out.push({ kind: "power", title: `${name(bs)} ${pl ? "rank" : "ranks"} #${b.rank} in the power rankings`, body: `${name(other(bs))} ${pl ? "rank" : "ranks"} #${w.rank}.${gap}` });
  }
  let st = null;
  for (const h of hist) {
    const S = ctxJson(h.streaks) || {};
    for (const k of ["su", "ats", "ou"]) {
      const m = /^([A-Z])(\d+)$/.exec(String(S[k] || ""));
      if (m && GM_STREAK[k][m[1]] && +m[2] >= 3 && (!st || +m[2] > st.n)) st = { side: h.side, k, kind: m[1], n: +m[2] };
    }
  }
  if (st) out.push({ kind: "streak", title: `${name(st.side)} ${GM_STREAK[st.k][st.kind][pl]} ${st.n} straight`,
    body: `Current ${{ su: "straight-up", ats: "against-the-spread", ou: "over/under" }[st.k]} streak entering this game.` });
  return out.slice(0, 4);
}
```

with:

```javascript
// Up to four generated sentences from data that exists, strongest first. Candidates: offense-vs-defense unit grade, explosive-play
// edge, power-rank gap, current streak (all sports) and, from game_info / cfb_team_insights, a CFB havoc mismatch, a CFB turnover edge
// and a weather row (CFB + NFL). Each is {kind, strength, title, body} (plain text; escaped when rendered); strength is 0-100 and
// derived below, ties keep the order the candidates are generated in, so the pick of four and its order are deterministic.
const gmPoss = (n) => (/s$/i.test(n) ? `${n}'` : `${n}'s`);
// team_history / matchup_grades / power_rankings identify a side by team code.
const gmTeamCode = (hist, grades, side) => { const x = (hist || []).find((y) => y.side === side) || (grades || []).find((y) => y.side === side); return x ? String(x.team) : null; };
const GM_STREAK = { su: { W: ["won", "won"], L: ["lost", "lost"] }, ats: { W: ["covered", "covered"], L: ["failed to cover", "failed to cover"] },
  ou: { O: ["has gone over in", "have gone over in"], U: ["has gone under in", "have gone under in"] } };
// Strengths. Existing rows sit in 30-70 from their own thresholds (grade: the offense's percentile; explosive: shown whenever both
// sides are computable; power: the rank gap; streak: its length, 3 or more); the new rows start at 50 and are only generated past
// their notable thresholds, so a real mismatch / weather outranks a routine row but a routine row still fills an empty slot.
const GM_STRENGTH = { grade: (pct) => Math.round(30 + 0.4 * pct), explosive: () => 45, power: (gap) => 30 + Math.min(40, gap), streak: (n) => Math.min(70, 30 + 8 * (n - 2)) };
const GM_NOTABLE_GAP = 30, GM_NOTABLE_PCT = 0.15;
const gmCut = (n) => Math.ceil(GM_NOTABLE_PCT * n);   // how many teams are the top (or bottom) 15% of an n-team field
// The newest-season cfb_team_insights row of a side's team, or null (rows arrive newest season first).
const gmInsRow = (D, side) => { const c = gmTeamCode(D.ctxHist, D.ctxGrades, side); return c == null ? null : (D.cfbIns || []).find((x) => String(x.team) === c) || null; };
const gmPct1 = (x) => (finite(x) ? `${(+x * 100).toFixed(1)}%` : "");
// CFB: one defense's havoc-created rank against the other offense's havoc-allowed rank (1 = best on both). Notable when the gap
// (offense rank minus defense rank, positive = defense edge) is 30+, or the defense has the edge and is top 15% / the offense bottom 15%.
function gmHavocInsight(D, name) {
  let best = null;
  for (const dSide of ["home", "away"]) {
    const d = gmInsRow(D, dSide), o = gmInsRow(D, dSide === "home" ? "away" : "home");
    if (!d || !o || !finite(d.def_havoc_rank) || !finite(o.off_havoc_allowed_rank) || !finite(d.n_ranked ?? o.n_ranked)) continue;
    const n = +(d.n_ranked ?? o.n_ranked), gap = o.off_havoc_allowed_rank - d.def_havoc_rank, cut = gmCut(n);
    const extreme = gap > 0 && (d.def_havoc_rank <= cut || o.off_havoc_allowed_rank > n - cut);
    if (!(gap >= GM_NOTABLE_GAP || extreme)) continue;
    const strength = Math.round(50 + Math.min(45, gap / 3));
    if (best && best.strength >= strength) continue;
    const dn = name(dSide), on = name(dSide === "home" ? "away" : "home"), dr = gmPct1(d.def_havoc_rate), or = gmPct1(o.off_havoc_allowed_rate);
    best = { kind: "havoc", strength, title: `${dn} defense creates havoc against ${on}`,
      body: `${dn} ranks #${d.def_havoc_rank} in havoc created${dr ? ` (${dr} of plays)` : ""}; ${gmPoss(on)} offense ranks #${o.off_havoc_allowed_rank} in havoc allowed${or ? ` (${or})` : ""}, where #1 allows the least.` };
  }
  return best;
}
// CFB: turnover-margin rank gap of 30+, or either team in the top / bottom 15% (equal ranks: no edge).
function gmTurnoverInsight(D, name) {
  const h = gmInsRow(D, "home"), a = gmInsRow(D, "away");
  if (!h || !a || !finite(h.turnover_margin_rank) || !finite(a.turnover_margin_rank) || !finite(h.n_ranked ?? a.n_ranked)) return null;
  const n = +(h.n_ranked ?? a.n_ranked), hr = +h.turnover_margin_rank, ar = +a.turnover_margin_rank, gap = Math.abs(hr - ar), cut = gmCut(n);
  const ext = (r) => r <= cut || r > n - cut;
  if (hr === ar || !(gap >= GM_NOTABLE_GAP || ext(hr) || ext(ar))) return null;
  const [bs, b, w] = hr < ar ? ["home", h, a] : ["away", a, h], per = (x) => (finite(x.turnover_margin_per_game) ? ` is ${signedStr(+x.turnover_margin_per_game, 1)} per game` : "");
  return { kind: "turnover", strength: Math.round(50 + Math.min(45, gap / 3)), title: `${name(bs)} owns the turnover edge`,
    body: `${name(bs)}${per(b) || " ranks"} (#${b.turnover_margin_rank} nationally); ${name(bs === "home" ? "away" : "home")}${per(w) || " ranks"} (#${w.turnover_margin_rank}).` };
}
// CFB + NFL: outdoor games only. Wind 15+ mph, under 40°F, 50%+ chance of rain or measurable rain; strength from the worst factor.
function gmWeatherInsight(D) {
  const i = D && D.gameInfo;
  if (!i || i.indoor === true) return null;
  const t = numOrNull(i.temp_f), w = numOrNull(i.wind_mph), pc = numOrNull(i.precip_chance), pi = numOrNull(i.precip_in), f = [], sc = [];
  if (w != null && w >= 15) { f.push(`wind ${Math.round(w)} mph`); sc.push(55 + Math.min(35, (w - 15) * 3)); }
  if (t != null && t < 40) { f.push(`${Math.round(t)}°F`); sc.push(55 + Math.min(30, (40 - t) * 2)); }
  if (pc != null && pc >= 50) { f.push(`${Math.round(pc)}% chance of rain`); sc.push(55 + Math.min(30, (pc - 50) * 0.6)); }
  else if (pi != null && pi >= GM_MEASURABLE_IN) { f.push(`${pi.toFixed(2)} in of rain`); sc.push(60); }
  if (!f.length) return null;
  return { kind: "weather", strength: Math.round(Math.max(...sc)), title: `Weather: ${f.join(", ")}`,
    body: `${i.weather_kind === "observed" ? "Observed" : "Forecast"} conditions${i.venue_name ? ` at ${i.venue_name}` : ""}. Descriptive only, not a pick.` };
}
function gmKeyInsights(D) {
  const r = (D && D.r) || {}, grades = (D && D.ctxGrades) || [], hist = (D && D.ctxHist) || [], power = (D && D.ctxPower) || [];
  const name = (side) => shortTeam(side === "home" ? r.home_team_name : r.away_team_name, D && D.sport);
  const other = (s) => (s === "home" ? "away" : "home");
  const pl = (D && D.sport) === "nfl" ? 1 : 0;   // NFL short names are plural nicknames ("Saints rank"), CFB schools singular ("Alabama ranks")
  const out = [];
  const graded = grades.filter((g) => g && g.overall && finite(g.overall_pct)).sort((x, y) => y.overall_pct - x.overall_pct)[0];
  if (graded) out.push({ kind: "grade", strength: GM_STRENGTH.grade(+graded.overall_pct), title: `${name(graded.side)} offense grades ${graded.overall} vs ${name(other(graded.side))}`,
    body: `Pass ${graded.pass || "–"}, run ${graded.run || "–"} against ${gmPoss(name(other(graded.side)))} defense (${ctxOrd(graded.overall_pct)} percentile overall).` });
  const xpl = (g) => {
    const u = g && ctxJson(g.units), o = u && u.off, d = u && u.def;
    if (!o || !d) return null;
    const v = [o.pass_explosive, d.pass_explosive, o.run_explosive, d.run_explosive].map(numOrNull);
    if (v.some((x) => x == null)) return null;
    const pr = numOrNull(u.pass_rate) ?? 0.5;
    return pr * (v[0] + v[1]) + (1 - pr) * (v[2] + v[3]);
  };
  const xh = xpl(grades.find((g) => g.side === "home")), xa = xpl(grades.find((g) => g.side === "away"));
  if (xh != null && xa != null && Math.abs(xh - xa) > 1e-9) {
    const w = xh > xa ? "home" : "away";
    out.push({ kind: "explosive", strength: GM_STRENGTH.explosive(), title: `Big-play grades favor ${name(w)}`,
      body: `${gmPoss(name(w))} offense vs ${gmPoss(name(other(w)))} defense is ahead of the reverse matchup on pass-rate-weighted explosive-play ratings.` });
  }
  const pr = (side) => power.find((x) => String(x.team) === gmTeamCode(hist, grades, side));
  const ph = pr("home"), pa = pr("away");
  if (ph && pa && finite(ph.rank) && finite(pa.rank) && ph.rank !== pa.rank) {
    const [bs, b, w] = ph.rank < pa.rank ? ["home", ph, pa] : ["away", pa, ph];
    const gap = finite(b.rating) && finite(w.rating) ? ` Rating gap: ${(b.rating - w.rating).toFixed(1)} pts.` : "";
    out.push({ kind: "power", strength: GM_STRENGTH.power(Math.abs(ph.rank - pa.rank)), title: `${name(bs)} ${pl ? "rank" : "ranks"} #${b.rank} in the power rankings`, body: `${name(other(bs))} ${pl ? "rank" : "ranks"} #${w.rank}.${gap}` });
  }
  let st = null;
  for (const h of hist) {
    const S = ctxJson(h.streaks) || {};
    for (const k of ["su", "ats", "ou"]) {
      const m = /^([A-Z])(\d+)$/.exec(String(S[k] || ""));
      if (m && GM_STREAK[k][m[1]] && +m[2] >= 3 && (!st || +m[2] > st.n)) st = { side: h.side, k, kind: m[1], n: +m[2] };
    }
  }
  if (st) out.push({ kind: "streak", strength: GM_STRENGTH.streak(st.n), title: `${name(st.side)} ${GM_STREAK[st.k][st.kind][pl]} ${st.n} straight`,
    body: `Current ${{ su: "straight-up", ats: "against-the-spread", ou: "over/under" }[st.k]} streak entering this game.` });
  if (D && D.sport === "cfb") for (const x of [gmHavocInsight(D, name), gmTurnoverInsight(D, name)]) if (x) out.push(x);
  const wx = gmWeatherInsight(D);
  if (wx) out.push(wx);
  return out.map((x, i) => ({ x, i })).sort((p, q) => q.x.strength - p.x.strength || p.i - q.i).slice(0, 4).map((p) => p.x);
}
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
const GM_INSIGHT_ICON = { grade: ["ca-gm-ic-red", ICON_GRADE], explosive: ["ca-gm-ic-amber", ICON_TREND], power: ["ca-gm-ic-blue", ICON_RANK], streak: ["ca-gm-ic-green", ICON_WAVE] };
```

with:

```javascript
const GM_INSIGHT_ICON = { grade: ["ca-gm-ic-red", ICON_GRADE], explosive: ["ca-gm-ic-amber", ICON_TREND], power: ["ca-gm-ic-blue", ICON_RANK], streak: ["ca-gm-ic-green", ICON_WAVE],
  havoc: ["ca-gm-ic-red", ICON_HAVOC], turnover: ["ca-gm-ic-amber", ICON_SWAP], weather: ["ca-gm-ic-blue", ICON_CLOUD] };
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
ctxHist = [], ctxGrades = [], ctxPower = [];
  if (isNfl) {
```

with:

```javascript
ctxHist = [], ctxGrades = [], ctxPower = [], cfbIns = [], cfbShares = [];
  if (isNfl) {
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
      ctxPower = await sb(`power_rankings_current?sport=eq.${sport}&team=in.(${codes.map((c) => q(String(c))).join(",")})`).catch(() => []);
  }
```

with:

```javascript
      ctxPower = await sb(`power_rankings_current?sport=eq.${sport}&team=in.(${codes.map((c) => q(String(c))).join(",")})`).catch(() => []);
    // CFB only: both teams' havoc / turnover ranks (every stored season, newest first) and quarter-scoring shares; a missing table -> []
    if (sport === "cfb" && codes.length) {
      const list = codes.map((c) => q(String(c))).join(",");
      [cfbIns, cfbShares] = await Promise.all([
        sb(`cfb_team_insights?team=in.(${list})&order=season.desc`).catch(() => []),
        sb(`cfb_quarter_shares?team=in.(${list})`).catch(() => []),
      ]);
    }
  }
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
    gameInfo: (gameInfoRows || [])[0] || null,
```

with:

```javascript
    gameInfo: (gameInfoRows || [])[0] || null, cfbIns: cfbIns || [], cfbShares: cfbShares || [],
```

In `site/tests/game.test.mjs` — the existing test `gmKeyInsights: up to four sentences ...`; same data, now asserted by strength, replace this text (it occurs exactly once):

```javascript
  const ins = g.gmKeyInsights(D);
  assert.equal(ins.length, 4);
  assert.deepEqual(ins.map((i) => i.kind), ["grade", "explosive", "power", "streak"]);
  assert.match(ins[0].title, /Alabama offense grades A vs South Carolina/); assert.match(ins[0].body, /Pass A, run C/);
  assert.match(ins[1].title, /Big-play grades favor Alabama/); assert.ok(!/pass and run/.test(ins[1].body), "only what is computed");
  assert.match(ins[2].title, /Alabama ranks #2/); assert.match(ins[2].body, /South Carolina ranks #31/); assert.match(ins[2].body, /20\.5/);
  assert.match(ins[3].title, /Alabama won 4 straight/);
```

with:

```javascript
  const ins = g.gmKeyInsights(D);
  assert.equal(ins.length, 4);
  // strongest first: grade 30 + 0.4 * 92 = 67, power 30 + gap 29 = 59, streak W4 = 46, explosive = 45
  assert.deepEqual(ins.map((i) => [i.kind, i.strength]), [["grade", 67], ["power", 59], ["streak", 46], ["explosive", 45]]);
  const by = (k) => ins.find((i) => i.kind === k);
  assert.match(by("grade").title, /Alabama offense grades A vs South Carolina/); assert.match(by("grade").body, /Pass A, run C/);
  assert.match(by("explosive").title, /Big-play grades favor Alabama/); assert.ok(!/pass and run/.test(by("explosive").body), "only what is computed");
  assert.match(by("power").title, /Alabama ranks #2/); assert.match(by("power").body, /South Carolina ranks #31/); assert.match(by("power").body, /20\.5/);
  assert.match(by("streak").title, /Alabama won 4 straight/);
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd site && npm test`
Expected: PASS (`ℹ pass 319, ℹ fail 0` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add site/js/pages/game.js site/tests/game.test.mjs site/tests/game_panels.test.mjs
git commit -m "feat(site): Key Insights by strength; CFB havoc and turnover rows; weather row

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 12: Site: CFB Projected Game Flow card

**Files:**
- Modify: `site/js/pages/game.js` (`gmShareRow`, `gmLineScore`, `gmGameFlow`, `gmFlowCard`, `gmOverview`)
- Modify: `site/css/theme.css` (`.ca-gm-col`, `.ca-gm-flow*`)
- Modify: `site/tests/game.test.mjs` (one assertion message)
- Test: `site/tests/game_panels.test.mjs` (append)

**Interfaces:**
- Consumes: T10 `D.gameInfo` (with `line_score`), T11 `D.cfbShares`, `gmTeamCode`; existing `groupedBars(groups, {h, bw})` and `infoTip(text)` (`ui.js`), `shortTeam`, `D.awayCol` / `D.homeCol`, `numOrNull`, `finite`, `ctxJson`, `gmEsc`, `safeCard`.
- Produces: `gmShareRow(D, side) -> row | null`; `gmLineScore(info) -> {home: number[4], away: number[4]} | null`; `gmGameFlow(D) -> {home: number[4], away: number[4], actual: {home, away} | null, games: {home, away}} | null` (null unless CFB with a projected score and a share row for both teams; `blend = mean(own scored, opponent allowed)` renormalised to sum 1; quarters = projected score x blend); `GM_FLOW_TIP`; `gmFlowCard(D) -> string` (`<section id="gm-flow">`, `""` when hidden); the Overview's left column stacks Key Insights then the flow card.

- [ ] **Step 1: Write the failing tests**

Worked example: home 30 pts, own scored shares `[.30 .20 .20 .30]`, opponent allowed `[.28 .22 .22 .28]` -> blend `[.29 .21 .21 .29]` -> `[8.7 6.3 6.3 8.7]`; away 20 pts, scored `[.20 .30 .30 .20]` against allowed `[.25 .25 .25 .25]` -> `[4.5 5.5 5.5 4.5]`. Hide conditions: no projection, either share row missing or unusable, NFL, missing table.

Append to the end of `site/tests/game_panels.test.mjs`:

```javascript
/* ── Projected Game Flow ────────────────────────────────────────────────── */
const share = (team, scored, allowed, games = 31) => ({ team, scored_q1: scored[0], scored_q2: scored[1], scored_q3: scored[2], scored_q4: scored[3],
  allowed_q1: allowed[0], allowed_q2: allowed[1], allowed_q3: allowed[2], allowed_q4: allowed[3], games_used: games });
const SHARES = [share("333", [0.30, 0.20, 0.20, 0.30], [0.25, 0.25, 0.25, 0.25], 31), share("61", [0.20, 0.30, 0.30, 0.20], [0.28, 0.22, 0.22, 0.28], 29)];
const flowD = (o = {}) => ({ sport: "cfb", r: PRED("cfb"), ctxGrades: [], ctxHist: HIST, cfbShares: SHARES, gameInfo: null, ...o });
const close = (a, b) => assert.ok(a.length === b.length && a.every((x, i) => Math.abs(x - b[i]) < 1e-9), `${a} vs ${b}`);

test("gmGameFlow: projected score x the average of the team's scored shares and the opponent's allowed shares", () => {
  const f = g.gmGameFlow(flowD());
  close(f.home, [8.7, 6.3, 6.3, 8.7]);       // home 30 pts: (.30 + .28) / 2, (.20 + .22) / 2, ...
  close(f.away, [4.5, 5.5, 5.5, 4.5]);       // away 20 pts: (.20 + .25) / 2, (.30 + .25) / 2, ...
  assert.equal(f.actual, null); assert.deepEqual({ ...f.games }, { home: 31, away: 29 });
  assert.ok(Math.abs(f.home.reduce((a, b) => a + b) - 30) < 1e-9 && Math.abs(f.away.reduce((a, b) => a + b) - 20) < 1e-9, "quarters add up to the projected score");
});

test("gmGameFlow: share vectors that do not sum to 1 are renormalised, so the quarters still add up to the projected score", () => {
  const f = g.gmGameFlow(flowD({ cfbShares: [share("333", [0.6, 0.4, 0.4, 0.4], [0.25, 0.25, 0.25, 0.25]), share("61", [0.2, 0.3, 0.3, 0.2], [0.25, 0.25, 0.25, 0.25])] }));
  close(f.home, [30 * 0.425 / 1.4, 30 * 0.325 / 1.4, 30 * 0.325 / 1.4, 30 * 0.325 / 1.4]);
  assert.ok(Math.abs(f.home.reduce((a, b) => a + b) - 30) < 1e-9);
});

test("gmGameFlow: the actual line score rides along only when it is four non-negative numbers per side", () => {
  const ok = g.gmGameFlow(flowD({ gameInfo: { line_score: { home: [7, 3, 10, 7], away: ["0", 7, 6, 7] } } }));
  assert.deepEqual({ ...ok.actual }, { home: [7, 3, 10, 7], away: [0, 7, 6, 7] });
  assert.deepEqual({ ...g.gmGameFlow(flowD({ gameInfo: { line_score: JSON.stringify({ home: [1, 2, 3, 4], away: [4, 3, 2, 1] }) } })).actual }, { home: [1, 2, 3, 4], away: [4, 3, 2, 1] }, "a JSON string works too");
  for (const bad of [{ home: [7, 3, 10], away: [0, 7, 6, 7] }, { home: [7, 3, 10, -1], away: [0, 7, 6, 7] }, { home: [7, 3, 10, null], away: [0, 7, 6, 7] }, null, "junk", {}])
    assert.equal(g.gmGameFlow(flowD({ gameInfo: { line_score: bad } })).actual, null, JSON.stringify(bad));
});

test("gmGameFlow hides (null) without a projection, a share row for either team, a usable share vector, or for non-CFB", () => {
  assert.equal(g.gmGameFlow(flowD({ r: PRED("cfb", { pred_home_score: null }) })), null);
  assert.equal(g.gmGameFlow(flowD({ r: PRED("cfb", { pred_away_score: "x" }) })), null);
  assert.equal(g.gmGameFlow(flowD({ r: PRED("cfb", { pred_home_score: 0, pred_away_score: 0 }) })), null);
  assert.equal(g.gmGameFlow(flowD({ cfbShares: [SHARES[0]] })), null, "one team has no row");
  assert.equal(g.gmGameFlow(flowD({ cfbShares: [] })), null);
  assert.equal(g.gmGameFlow(flowD({ cfbShares: undefined })), null);
  assert.equal(g.gmGameFlow(flowD({ ctxHist: [] })), null, "no team codes, no rows to match");
  assert.equal(g.gmGameFlow(flowD({ cfbShares: [share("333", [0.3, null, 0.2, 0.3], [0.25, 0.25, 0.25, 0.25]), SHARES[1]] })), null);
  assert.equal(g.gmGameFlow(flowD({ cfbShares: [share("333", [0, 0, 0, 0], [0, 0, 0, 0]), share("61", [0, 0, 0, 0], [0, 0, 0, 0])] })), null, "all-zero shares cannot be normalised");
  assert.equal(g.gmGameFlow(flowD({ sport: "nfl" })), null);
  assert.equal(g.gmGameFlow(null), null);
});

test("Overview: the Projected Game Flow card sits under Key Insights with its tooltip, legend, projected chart and (final games) the actual chart", async () => {
  const rows = { cfb_quarter_shares: SHARES, game_info: [{ ...INFO, line_score: { home: [7, 3, 10, 7], away: [0, 7, 6, 7] } }],
    cfb_team_insights: [insRow("333", { def_havoc_rank: 8 }), insRow("61", { off_havoc_allowed_rank: 118 })] };
  const html = await page({ rows }).D.buildGamePage();
  const i = html.indexOf('id="gm-insights"'), f = html.indexOf('id="gm-flow"'), c = html.indexOf('id="gm-cover"');
  assert.ok(i > 0 && i < f && f < c, "insights, then game flow, then cover");
  const card = html.slice(f, c);
  assert.ok(card.includes("<h2>Projected Game Flow</h2>") && card.includes("Projected pace (points by quarter)") && card.includes("Actual by quarter (excludes overtime)"));
  assert.ok(card.includes('class="ca-info"') && card.includes("A pace estimate, not a prediction of the score by quarter"));
  assert.ok(card.includes(">Georgia<") && card.includes(">Alabama<"), "legend names both teams");
  for (const t of [">Q1<", ">Q4<", ">8.7<", ">4.5<", ">6.3<", ">10<"]) assert.ok(card.includes(t), t);
  assert.ok(card.includes("Shares come from Georgia's 29 and Alabama's 31 games"));
  assert.equal((card.match(/<svg class="ca-bars"/g) || []).length, 2);
  const upcoming = await page({ rows: { cfb_quarter_shares: SHARES } }).D.buildGamePage();
  const up = upcoming.slice(upcoming.indexOf('id="gm-flow"'), upcoming.indexOf('id="gm-cover"'));
  assert.ok(up.includes("Projected pace") && !up.includes("Actual by quarter"), "no line score yet: projection only");
});

test("Overview: no shares, a missing table, no projection or an NFL game -> no Projected Game Flow card, page intact", async () => {
  for (const o of [{ rows: { cfb_quarter_shares: [] } }, { missing: ["cfb_quarter_shares"] }, { rows: { cfb_quarter_shares: SHARES }, pred: { pred_home_score: null } },
                   { sport: "nfl", rows: { cfb_quarter_shares: SHARES } }]) {
    const html = await page(o).D.buildGamePage();
    assert.ok(!html.includes("gm-flow") && !html.includes("Projected Game Flow") && html.includes('id="gm-cover"'), JSON.stringify(Object.keys(o)));
    assert.ok(!/NaN|undefined/.test(html.replace(/\bnull\b(?=[^<]*>)/g, "")));
  }
});
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd site && npm test`
Expected: FAIL — `✖ gmGameFlow: projected score x the average of the team's scored shares and the opponent's allowed shares (0.230584ms)`.

- [ ] **Step 3: Implement the math, the card and the layout**

In `site/js/pages/game.js` — insert the flow helpers just above `gmKeyInsights`, replace this text (it occurs exactly once):

```javascript
function gmKeyInsights(D) {
```

with:

```javascript
// CFB Projected Game Flow. Each team's projected points by quarter = its projected score x the average of its own scored shares and its
// opponent's allowed shares (cfb_quarter_shares: Q1-Q4 shares of points, shrunk to the league, each vector sums to 1), renormalised to
// 1. Returns {home: [q1..q4], away: [q1..q4], actual: {home, away} | null, games: {home, away}} or null when it is not CFB, has no
// projected score, or either team lacks a share row (the card hides). `actual` = game_info.line_score for a finished game.
const gmShareRow = (D, side) => { const c = gmTeamCode(D.ctxHist, D.ctxGrades, side); return c == null ? null : (D.cfbShares || []).find((x) => String(x.team) === c) || null; };
function gmLineScore(info) {
  const ls = info && ctxJson(info.line_score);
  const ok = (a) => Array.isArray(a) && a.length === 4 && a.every((v) => finite(v) && +v >= 0);
  return ls && ok(ls.home) && ok(ls.away) ? { home: ls.home.map(Number), away: ls.away.map(Number) } : null;
}
function gmGameFlow(D) {
  if (!D || D.sport !== "cfb") return null;
  const r = D.r || {}, h = numOrNull(r.pred_home_score), a = numOrNull(r.pred_away_score);
  if (h == null || a == null || h < 0 || a < 0 || h + a <= 0) return null;
  const sh = gmShareRow(D, "home"), sa = gmShareRow(D, "away");
  if (!sh || !sa) return null;
  const vec = (row, kind) => [1, 2, 3, 4].map((q) => numOrNull(row[`${kind}_q${q}`]));
  const blend = (own, opp) => {
    const sc = vec(own, "scored"), al = vec(opp, "allowed");
    if ([...sc, ...al].some((x) => x == null || x < 0)) return null;
    const m = sc.map((x, i) => (x + al[i]) / 2), tot = m.reduce((x, y) => x + y, 0);
    return tot > 0 ? m.map((x) => x / tot) : null;
  };
  const bh = blend(sh, sa), ba = blend(sa, sh);
  if (!bh || !ba) return null;
  return { home: bh.map((x) => x * h), away: ba.map((x) => x * a), actual: gmLineScore(D.gameInfo), games: { home: numOrNull(sh.games_used), away: numOrNull(sa.games_used) } };
}
function gmKeyInsights(D) {
```

In `site/js/pages/game.js` — insert the card above `gmCoverCard`, replace this text (it occurs exactly once):

```javascript
function gmCoverCard(D) {
```

with:

```javascript
const GM_FLOW_TIP = "Each team's own quarter-by-quarter scoring pattern over the last two seasons and this one (shrunk toward the league average), averaged with what its opponent allows by quarter, applied to the projected score. A pace estimate, not a prediction of the score by quarter. Overtime is excluded.";
function gmFlowCard(D) {
  const f = gmGameFlow(D);
  if (!f) return "";
  const away = shortTeam(D.r.away_team_name, "cfb"), home = shortTeam(D.r.home_team_name, "cfb");
  const groups = (src, fmt) => ["Q1", "Q2", "Q3", "Q4"].map((q, i) => ({ label: q, bars: [{ value: src.away[i], color: D.awayCol, label: fmt(src.away[i]) }, { value: src.home[i], color: D.homeCol, label: fmt(src.home[i]) }] }));
  const chart = (cap, src, fmt) => `<figure><figcaption>${cap}</figcaption>${groupedBars(groups(src, fmt), { h: 200, bw: 28 })}</figure>`;
  const legend = `<div class="ca-legend"><span class="ca-legend-item"><i style="background:${gmEsc(D.awayCol)}"></i>${gmEsc(away)}</span><span class="ca-legend-item"><i style="background:${gmEsc(D.homeCol)}"></i>${gmEsc(home)}</span></div>`;
  const n = f.games.away != null && f.games.home != null ? `<p class="ca-gm-cap">Shares come from ${gmEsc(away)}'s ${f.games.away} and ${gmEsc(home)}'s ${f.games.home} games over the last two seasons and this one.</p>` : "";
  return `<section class="ca-card ca-gm-flow" id="gm-flow"><div class="ca-card-head"><h2>Projected Game Flow</h2>${infoTip(GM_FLOW_TIP)}</div>${legend}
    <div class="ca-gm-flow-charts">${chart("Projected pace (points by quarter)", f, (v) => v.toFixed(1))}${f.actual ? chart("Actual by quarter (excludes overtime)", f.actual, (v) => String(Math.round(v))) : ""}</div>${n}</section>`;
}
function gmCoverCard(D) {
```

In `site/js/pages/game.js`, replace this text (it occurs exactly once):

```javascript
  const ins = safeCard("Key Insights", gmInsightsCard, D, "ca-card ca-gm-ins-card", "gm-insights");
  return `<div class="ca-gm-cards" id="gm-cards">${cards}</div>${safeCard("Model Projection", gmProjectionCard, D, "ca-card ca-gm-proj", "gm-projection")}
    <div class="ca-gm-lower${ins ? "" : " solo"}">${ins}${safeCard("Cover Probability", gmCoverCard, D, "ca-card ca-gm-cover", "gm-cover")}</div>`;
```

with:

```javascript
  const ins = safeCard("Key Insights", gmInsightsCard, D, "ca-card ca-gm-ins-card", "gm-insights");
  const flow = safeCard("Projected Game Flow", gmFlowCard, D, "ca-card ca-gm-flow", "gm-flow");
  const left = ins + flow;       // CFB: the Projected Game Flow card sits under Key Insights
  return `<div class="ca-gm-cards" id="gm-cards">${cards}</div>${safeCard("Model Projection", gmProjectionCard, D, "ca-card ca-gm-proj", "gm-projection")}
    <div class="ca-gm-lower${left ? "" : " solo"}">${flow ? `<div class="ca-gm-col">${left}</div>` : left}${safeCard("Cover Probability", gmCoverCard, D, "ca-card ca-gm-cover", "gm-cover")}</div>`;
```

In `site/css/theme.css`, replace this text (it occurs exactly once):

```css
.ca-gm-books h2,.ca-gm-power h2{font:700 24px var(--sans)}
```

with:

```css
.ca-gm-books h2,.ca-gm-power h2,.ca-gm-flow h2{font:700 24px var(--sans)}
```

In `site/css/theme.css`, replace this text (it occurs exactly once):

```css
.ca-gm-lower.solo{grid-template-columns:minmax(0,1fr)}
```

with:

```css
.ca-gm-lower.solo{grid-template-columns:minmax(0,1fr)}
.ca-gm-col{display:grid;gap:16px;min-width:0}
.ca-gm-flow .ca-legend{margin:0 0 6px}
.ca-gm-flow-charts{display:flex;flex-wrap:wrap;gap:12px 28px}
.ca-gm-flow figure{margin:0;min-width:0;max-width:100%}
.ca-gm-flow figcaption{font:600 13px var(--sans);color:var(--muted);margin-bottom:2px}
```

In `site/tests/game.test.mjs`, replace this text (it occurs exactly once):

```javascript
"Phase B items and mockup placeholders are never rendered"
```

with:

```javascript
"the CFB-only Projected Game Flow and the mockup placeholders are never rendered for an NFL game without game_info"
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd site && npm test`
Expected: PASS (`ℹ pass 325, ℹ fail 0` in this plan's dry run).

- [ ] **Step 5: Commit**

```bash
git add site/js/pages/game.js site/css/theme.css site/tests/game.test.mjs site/tests/game_panels.test.mjs
git commit -m "feat(site): CFB Projected Game Flow card

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

### Task 13: Cache-key bump and final verification (tests + browser)

**Files:**
- Modify: `site/*.html` (ten pages: cache key `?v=20261007a` -> `?v=20261007b`)
- Modify: `site/tests/smoke.test.mjs` (the three places that name the key)

**Interfaces:**
- Consumes: everything above; the cache key convention `?v=YYYYMMDD<letter>` on every `<script>` and on `theme.css` in all ten pages.
- Produces: cache key `20261007b` on every page (one bump for the whole feature); a verified feature branch ready for the user to run `db/migration_site_panels.sql`, review and merge. Nothing in this task writes to Supabase.

- [ ] **Step 1: Update the smoke test to the new key (failing first)**

In `site/tests/smoke.test.mjs`, replace this text (it occurs exactly once):

```javascript
assert.equal(srcs[0][2], "20261007a", `${f}: the current cache key`);
```

with:

```javascript
assert.equal(srcs[0][2], "20261007b", `${f}: the current cache key`);
```

In `site/tests/smoke.test.mjs`, replace this text (it occurs exactly once):

```javascript
assert.match(html, /css\/theme\.css\?v=20261007a"/, `${f}: theme.css carries the same key`);
```

with:

```javascript
assert.match(html, /css\/theme\.css\?v=20261007b"/, `${f}: theme.css carries the same key`);
```

In `site/tests/smoke.test.mjs`, replace this text (it occurs exactly once):

```javascript
test("the cache key was bumped to 20261007a everywhere: no file in site/ still carries the previous key", () => {
  const old = ["2026", "1006", "a"].join("")
```

with:

```javascript
test("the cache key was bumped to 20261007b everywhere: no file in site/ still carries the previous key", () => {
  const old = ["2026", "1007", "a"].join("")
```

In `site/tests/smoke.test.mjs`, replace this text (it occurs exactly once):

```javascript
.filter((m) => m[1] !== "20261007a").length, 0);
```

with:

```javascript
.filter((m) => m[1] !== "20261007b").length, 0);
```

Run: `cd site && npm test`
Expected: FAIL — `✖ every page loads the scripts in order with the current cache key (1.273584ms)`.

- [ ] **Step 2: Bump the key in every page**

Replace every `20261007a` with `20261007b` in `site/*.html`:

```bash
perl -pi -e 's/20261007a/20261007b/g' site/*.html
```

Run: `cd site && npm test`
Expected: PASS (`ℹ pass 325, ℹ fail 0` in this plan's dry run).

- [ ] **Step 3: Run the touched Python suites and the whole site suite**

Run: `uv run pytest -q tests/cfb tests/nfl/test_espn.py tests/scripts tests/test_game_info.py tests/test_db_site_panels.py tests/test_db_injury_snapshots.py tests/test_migration_site_redesign_a.py tests/test_workflows_game_info.py tests/test_workflows_cfb_advanced.py tests/test_workflows_team_context.py tests/test_workflows_props_ml.py`
Expected: PASS (`956 passed` in this plan's dry run). (about 2-4 minutes)

Run: `cd site && npm test`
Expected: PASS (`ℹ pass 325, ℹ fail 0` in this plan's dry run).

Then the whole Python suite once, to catch anything outside the touched files (about 3-6 minutes):

Run: `uv run pytest -q`
Expected: PASS (no failures; a test that fails here and also fails on `main` is pre-existing: confirm with `git stash` / `git checkout main` before blaming this branch).

- [ ] **Step 4: Serve a /tmp copy with verification fixtures**

The three new tables do not exist in Supabase until the user runs the migration, so the real site shows the **without data** state by itself (every new panel hidden). The **with data** state needs fixtures: this step copies `site/` to `/tmp` and injects a tiny `fetch` wrapper (only in the copy) that answers `game_info`, `cfb_team_insights` and `cfb_quarter_shares` and passes every other request to the real Supabase. Add `&mock=data` (upcoming game), `&mock=final` (adds an observed reading and a final line score) or `&mock=empty` (the tables exist but have no rows) to the game URL.

```bash
rm -rf /tmp/site-panels && cp -R site /tmp/site-panels          # macOS blocks serving from ~/Desktop, so serve a copy from /tmp
cat > /tmp/site-panels/panels-mock.js <<'JS'
// Verification-only fixtures for game_info / cfb_team_insights / cfb_quarter_shares. Lives in the /tmp copy, never committed.
(() => {
  const real = window.fetch.bind(window);
  const mode = new URLSearchParams(location.search).get("mock") || "data";    // data | final | empty
  const codes = (u) => [...decodeURIComponent(u).matchAll(/"([^"]+)"/g)].map((m) => m[1]);
  const info = (sport) => (sport === "nfl"
    ? { sport, game_pk: 0, venue_name: "Highmark Stadium", city: "Orchard Park", state: "NY", indoor: false, temp_f: 36, wind_mph: 18, precip_chance: 55, precip_in: null, conditions: "Cloudy", weather_kind: "forecast", source: "espn", line_score: null }
    : { sport, game_pk: 0, venue_name: "Bryant-Denny Stadium", city: "Tuscaloosa", state: "AL", indoor: false, temp_f: 41, wind_mph: 16, precip_chance: null, precip_in: 0.04, conditions: null,
        weather_kind: mode === "final" ? "observed" : "forecast", source: "cfbd", line_score: mode === "final" ? { home: [7, 3, 10, 7], away: [0, 7, 6, 7] } : null });
  const shares = (team, i) => ({ team, scored_q1: i ? 0.20 : 0.30, scored_q2: i ? 0.30 : 0.20, scored_q3: i ? 0.30 : 0.20, scored_q4: i ? 0.20 : 0.30,
    allowed_q1: 0.25, allowed_q2: 0.25, allowed_q3: 0.25, allowed_q4: 0.25, games_used: 30 });
  const ins = (team, i) => ({ season: 2026, team, games: 5, n_ranked: 134, through_week: 5, def_havoc_rate: i ? 0.17 : 0.23, def_havoc_rank: i ? 70 : 8,
    off_havoc_allowed_rate: i ? 0.24 : 0.15, off_havoc_allowed_rank: i ? 118 : 12, turnover_margin: i ? -4 : 9, turnover_margin_per_game: i ? -0.8 : 1.8, turnover_margin_rank: i ? 112 : 6 });
  window.fetch = async (url, opts) => {
    const u = String(url), m = /\/rest\/v1\/(game_info|cfb_team_insights|cfb_quarter_shares)\?/.exec(u);
    if (!m) return real(url, opts);
    let rows = [];
    if (mode !== "empty") rows = m[1] === "game_info" ? [info(/sport=eq\.nfl/.test(u) ? "nfl" : "cfb")]
      : codes(u).map((c, i) => (m[1] === "cfb_team_insights" ? ins(c, i % 2) : shares(c, i % 2)));
    return { ok: true, status: 200, json: async () => rows, text: async () => JSON.stringify(rows) };
  };
})();
JS
# load the mock before app.js on the game page of the COPY only
perl -pi -e 's#<script src="app\.js#<script src="panels-mock.js"></script><script src="app.js#' /tmp/site-panels/game.html
node --check /tmp/site-panels/panels-mock.js && grep -c panels-mock /tmp/site-panels/game.html     # expect: 1
cd /tmp/site-panels && python3 -m http.server 4173 >/tmp/site-panels.log 2>&1 &                      # run in the background; stop it at the end
```

- [ ] **Step 5: Browser check: built-in browser, 1280 / 1440 / 1920, hard reload**

Use the built-in browser (`mcp__Claude_Browser__*`). First get a live CFB and a live NFL game id: `mcp__Claude_Browser__preview_start` with `url: "http://localhost:4173/cfb.html"`, then `mcp__Claude_Browser__javascript_tool` with `[...document.querySelectorAll('a[href*="game.html"]')].slice(0, 3).map((a) => a.getAttribute('href'))`; do the same on `nfl.html`. Call the results `<CFB>` (`game.html?sport=cfb&game=<pk>`) and `<NFL>`.

For each viewport in **1280x800, 1440x900, 1920x1080** (`mcp__Claude_Browser__resize_window` with `width` / `height`) and each URL below: `mcp__Claude_Browser__navigate` to it, then a **hard reload** (`mcp__Claude_Browser__computer` `key` `cmd+shift+r`, so the new `?v=20261007b` scripts and CSS are fetched fresh), `mcp__Claude_Browser__computer` `screenshot`, `mcp__Claude_Browser__read_console_messages` with `onlyErrors: true`, and `mcp__Claude_Browser__javascript_tool` with `document.documentElement.scrollWidth <= window.innerWidth` (must be `true`: no horizontal scroll).

| URL (append to `http://localhost:4173/`) | Must show | Must not show |
|---|---|---|
| `<CFB>` (no mock; live Supabase, tables not migrated) | the page exactly as before: hero, odds boxes, Read card, Key Insights (existing rows), Cover Probability | any venue line, havoc / turnover / weather row, Projected Game Flow card, empty card shells, console errors |
| `<CFB>&mock=empty` | same as the row above | the same |
| `<CFB>&mock=data` | one centered line under the kickoff line: `Bryant-Denny Stadium · Tuscaloosa, AL · 41°F, wind 16 mph, 0.04 in rain`; Key Insights with a havoc row, a turnover row and a weather row (at most 4 rows, strongest first, new icons); a Projected Game Flow card directly under Key Insights with the legend, the `Projected pace (points by quarter)` chart (8 bars labelled to one decimal) and the ⓘ tooltip (hover it) | the `Actual by quarter` chart; any `NaN` / `undefined` text |
| `<CFB>&mock=final` | the venue line with `Observed 41°F, wind 16 mph, 0.04 in rain`; the Game Flow card with both the projected and the `Actual by quarter (excludes overtime)` charts side by side (they wrap under each other when the column is narrow) | overlap between charts and the Cover Probability card |
| `<NFL>&mock=data` | the venue line `Highmark Stadium · Orchard Park, NY · Cloudy, 36°F, wind 18 mph, 55% rain`; a `Weather: wind 18 mph, 36°F, 55% chance of rain` Key Insights row | any Projected Game Flow card, havoc or turnover row |
| `<NFL>` (no mock) | the page exactly as before | any venue line or weather row |

At 1280 the hero line may wrap to two lines but must stay centered inside the hero; the Key Insights / Game Flow left column and the Cover Probability column must not overflow or overlap. Fix any defect in `site/css/theme.css` or `game.js` with a test where the defect is logic, re-run `cd site && npm test`, re-copy `site/` to `/tmp/site-panels` (keep `panels-mock.js` and the `game.html` edit: re-run the first two lines of the previous step) and repeat the failing viewport.

- [ ] **Step 6: Clean up**

```bash
pkill -f "http.server 4173"; rm -rf /tmp/site-panels /tmp/site-panels.log
git status --short            # only the intended files; no /tmp copy, no .env
```

Reset the browser viewport with `mcp__Claude_Browser__resize_window` `preset: "desktop"`.

- [ ] **Step 7: Commit**

```bash
git add site/*.html site/tests/smoke.test.mjs
git commit -m "chore(site): bump the cache key to 20261007b for the game-page panels

Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 8: Hand off to the user (nothing here is automated)**

Report to the user, in this order: (1) run `db/migration_site_panels.sql` in the Supabase SQL Editor (it is idempotent); (2) merge the branch; (3) dispatch `build-cfb-advanced` once (blank seasons = current season) so `cfb_team_insights` / `cfb_quarter_shares` fill, and `build-game-info` once so `game_info` fills, instead of waiting for the schedules (Monday 12:00 UTC and the next 14:00 UTC run). Until then the site keeps hiding every new panel. The only writes to Supabase are those scheduled or dispatched jobs.

---

## Self-Review

**1. Spec coverage** (spec section -> task):

| Spec requirement | Task |
|---|---|
| §1 `game_info` table + CFB venue/weather from CFBD (`/venues` city/state extended, weather forecast/observed) | 1 (parsers), 3 (venues backfill), 6 (rows), 7 (table), 8 (job) |
| §1 `game_info` NFL from ESPN summary (venue, weather when present, indoor) | 2, 6, 8 |
| §1 indoor -> weather NULL; missing stays NULL | 2 (`indoor` drops weather), 6, 7 (`CASE ... indoor`) |
| §1 `cfb_team_insights` (havoc, turnovers, explosiveness, ranks >= 3 games, `through_week`) | 1 (`parse_team_game_stats`), 3 (turnover pull), 5, 7, 9 |
| §1 `cfb_quarter_shares` (scored/allowed shares, k = 12, last two seasons + current, OT excluded) and the games-parser extension + 2024-2026 backfill | 1, 3, 4, 7, 9 |
| §1 Job: `build_game_info.py` + `build-game-info.yml` (daily 14:00, football days 16:00 / 22:00); Monday job additions; budget; upserts only | 8, 9 |
| §2 Hero line (format, omitted pieces, indoor, observed, hidden without a row) | 10 |
| §2 Key Insights candidates, thresholds, strength, four-row cap, icons | 11 |
| §2 Projected Game Flow (blend math, chart via `ui.js` helpers, "Projected pace", ⓘ tooltip, finished-game actuals, hide conditions) | 12 (+3 spec clarification 3 for the actuals' source) |
| §2 `sb()` / `safeCard`, `ctxEsc`, one cache bump, panels hide when tables are missing | 10-12 (tests with 404 stubs), 13 |
| §3 Migration (3 tables, PKs, anon read policies; user runs it) | 7 (file), 13 (hand-off) |
| §3 Network-free tests listed in the spec (parsers, ESPN parser indoor/missing, shrinkage, insights, builder window/indoor/upsert, site hero/insights/flow, workflow schedule) | 1, 2, 4, 5, 6, 7, 8, 9, 10-12 |
| §3 Visual check at 1280 / 1440 / 1920 with and without data | 13 |
| Out of scope respected (no TV network, NFL havoc, live game, model change) | all |

No gaps. Spec items the plan changed or extended are listed under "Spec ambiguities resolved".

**2. Placeholder scan:** no "TBD", "TODO", "implement later", "fill in", "appropriate error handling" or "similar to Task N" anywhere; every code step carries the full code; every `Replace` states the exact text it replaces (and its uniqueness was checked), and every test file / appended chunk was applied to a clean checkout of `HEAD` and run. The only deliberately open items are the two [NETWORK] reconcile steps (Task 2 probes, Task 3 backfill checks), which say exactly what to change for each possible outcome.

**3. Type consistency:** names are used identically across tasks.

- Row keys of `game_info` rows (T6) = `db.SITE_PANEL_COLUMNS["game_info"]` (T7) = the migration's columns (T7 test) = what the site reads (`venue_name, city, state, indoor, temp_f, wind_mph, precip_chance, precip_in, conditions, weather_kind, line_score` in T10-T12).
- `INSIGHT_COLUMNS` / `SHARE_COLUMNS` (T4-T5) = `db.SITE_PANEL_COLUMNS["cfb_team_insights" / "cfb_quarter_shares"]` (T7: `to_rows` output is what `upsert_*` writes) = the columns `gmHavocInsight` / `gmTurnoverInsight` / `gmGameFlow` read (`def_havoc_rank`, `off_havoc_allowed_rank`, `turnover_margin_rank`, `turnover_margin_per_game`, `n_ranked`, `scored_q1..4`, `allowed_q1..4`, `games_used`).
- `parse_weather_window` columns (T1) are exactly what `cfb_game_info` reads (`game_id, venue_id, venue, game_indoors, temperature, wind_speed, precipitation, condition, has_fbs`) (T6).
- `parse_line_scores` -> `{pk: {"home": [4], "away": [4]}}` (T2) = `game_info.line_score` (T6/T7) = `gmLineScore` (T12).
- Team ids are ESPN ids as strings everywhere (`cfbd_to_espn`, `team_history.team`, `gmTeamCode`).
- `build_game_info.build_cfb(now, client, espn, venues_path)` / `build_nfl(now, espn)` (T8) call only functions T2 / T6 define; `build_cfb_panels.build` (T9) calls only T4 / T5.
- JS: `D.gameInfo` (T10) -> `gmWeatherInsight` (T11) and `gmGameFlow` (T12); `D.cfbIns` / `D.cfbShares` (T11) -> `gmInsRow` (T11) and `gmShareRow` (T12); `GM_MEASURABLE_IN` (T10) used in T11.
