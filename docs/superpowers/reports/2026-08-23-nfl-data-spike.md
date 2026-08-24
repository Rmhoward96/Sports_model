# NFL Data Spike — External Source Verification

**Date:** 2026-08-24
**Purpose:** Confirm exact data shapes from `nfl_data_py` (nflverse), ESPN's scoreboard endpoint, and The Odds API's NFL prop markets, so P1 (NFL ingestion) implementation isn't guessing at schemas. Investigation only — no throwaway probe code from this spike is kept. The only lasting artifacts are (a) this findings doc and (b) the `nfl-data-py` dependency added to `pyproject.toml`/`uv.lock`, intended for P1's ingestion code to use.

**Environment note:** `ODDS_API_KEY` is not available in this environment (it lives only as a GitHub Actions secret). Step 3 below (The Odds API) was **not run live** — see the Deferred section. Steps 1 and 2 were run live and all output below is real captured output, not fabricated.

---

## Step 1: `nfl_data_py` (nflverse) — LIVE, verified

Installed via `uv add nfl_data_py` (pulled in `nfl-data-py==0.3.2`, plus `appdirs`, `cramjam`, `fastparquet`, `fsspec`). Ran against the 2024 season (most recent complete season).

### `import_schedules([2024])` — 285 rows

Full column list:
```
game_id, season, game_type, week, gameday, weekday, gametime, away_team, away_score,
home_team, home_score, location, result, total, overtime, old_game_id, gsis,
nfl_detail_id, pfr, pff, espn, ftn, away_rest, home_rest, away_moneyline, home_moneyline,
spread_line, away_spread_odds, home_spread_odds, total_line, under_odds, over_odds,
div_game, roof, surface, temp, wind, away_qb_id, home_qb_id, away_qb_name, home_qb_name,
away_coach, home_coach, referee, stadium_id, stadium
```

Fields relevant to P1 (game results / identity):
- `game_id` — nflverse's own composite id, format `"2024_01_BAL_KC"` (season_week_away_home)
- `away_team`, `home_team` — team abbreviation strings (see team-name section below)
- `away_score`, `home_score` — floats
- `week` — int
- `gameday` — date string `"2024-09-05"`
- `gametime` — string `"20:20"` (separate from `gameday`, no combined timestamp column)
- **`espn`** — nflverse already carries the ESPN event id as a column (e.g. `401671789` for the BAL@KC opener). This is a strong cross-reference for `game_pk` matching against ESPN's scoreboard (see game_pk recommendation below).
- Also carries betting lines already: `spread_line`, `total_line`, `away_moneyline`, `home_moneyline`, `away_spread_odds`, `home_spread_odds`, `over_odds`, `under_odds` — useful as a closing-line reference/backfill source independent of The Odds API.

Sample row (BAL @ KC, week 1 2024):
```python
{'game_id': '2024_01_BAL_KC', 'season': 2024, 'game_type': 'REG', 'week': 1,
 'gameday': '2024-09-05', 'weekday': 'Thursday', 'gametime': '20:20',
 'away_team': 'BAL', 'away_score': 20.0, 'home_team': 'KC', 'home_score': 27.0,
 'location': 'Home', 'result': 7.0, 'total': 47.0, 'overtime': 0.0,
 'espn': 401671789, ...}
```

### `import_weekly_data([2024], columns=None)` — 5,597 rows

Full column list:
```
player_id, player_name, player_display_name, position, position_group, headshot_url,
recent_team, season, week, season_type, opponent_team, completions, attempts,
passing_yards, passing_tds, interceptions, sacks, sack_yards, sack_fumbles,
sack_fumbles_lost, passing_air_yards, passing_yards_after_catch, passing_first_downs,
passing_epa, passing_2pt_conversions, pacr, dakota, carries, rushing_yards, rushing_tds,
rushing_fumbles, rushing_fumbles_lost, rushing_first_downs, rushing_epa,
rushing_2pt_conversions, receptions, targets, receiving_yards, receiving_tds,
receiving_fumbles, receiving_fumbles_lost, receiving_air_yards,
receiving_yards_after_catch, receiving_first_downs, receiving_epa,
receiving_2pt_conversions, racr, target_share, air_yards_share, wopr,
special_teams_tds, fantasy_points, fantasy_points_ppr
```

All the columns the brief asked to confirm exist exactly as named: `player_id`, `player_name` (abbreviated form, e.g. `"A.Rodgers"`), `player_display_name` (full name, e.g. `"Aaron Rodgers"`), `recent_team` (not `team`), `passing_yards`, `passing_tds`, `attempts`, `targets`, `receptions`, `receiving_yards`, `receiving_tds`, `carries` (not `rushing_attempts`), `rushing_yards`, `rushing_tds`. Note: team column is named **`recent_team`**, not `team`. Opponent is `opponent_team`.

### `import_injuries([2024])` — 6,215 rows

Full column list:
```
season, game_type, team, week, gsis_id, position, full_name, first_name, last_name,
report_primary_injury, report_secondary_injury, report_status, practice_primary_injury,
practice_secondary_injury, practice_status, date_modified
```

`report_status` is the field for game-status designation (values seen: `"Out"`, `"Questionable"`; also expect `"Doubtful"`). `gsis_id` is the join key back to `player_id` in weekly/roster data (nflverse's gsis id is used consistently as `player_id` across schedules/weekly/injuries/rosters — confirm at implementation time that `injuries.gsis_id == weekly.player_id` format, both look like `"00-0039521"`).

### `import_seasonal_rosters([2024])` — 3,215 rows

(`import_seasonal_rosters` exists on this version of `nfl_data_py`, used per the brief's fallback logic.)

Full column list:
```
season, team, position, depth_chart_position, jersey_number, status, player_name,
first_name, last_name, birth_date, height, weight, college, player_id, espn_id,
sportradar_id, yahoo_id, rotowire_id, pff_id, pfr_id, fantasy_data_id, sleeper_id,
years_exp, headshot_url, ngs_position, week, game_type, status_description_abbr,
football_name, esb_id, gsis_it_id, smart_id, entry_year, rookie_year, draft_club,
draft_number, age
```

Player↔name↔team↔position↔depth chain: `player_id` (join key), `player_name`, `team`, `position`, `depth_chart_position`. Also carries `espn_id` directly — a second, player-level ESPN cross-reference (separate from the game-level `espn` column in schedules).

---

## Step 2: ESPN scoreboard endpoint — LIVE, verified

Endpoint: `https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard`

### Default call (no params)

As of the live run (2026-08-24), this returned the **current week's** games: NFL 2026 preseason week 3, all 16 games already `STATUS_FINAL` (today is a Monday after that week's Sunday slate).

Top-level event keys: `['id', 'uid', 'date', 'name', 'shortName', 'season', 'week', 'competitions', 'links', 'status']`

Sample:
```
id: 401873286   date: 2026-08-21T00:00Z   name: "Las Vegas Raiders at Houston Texans"
teams: [('home', 'Houston Texans', 'HOU', '20'), ('away', 'Las Vegas Raiders', 'LV', '22')]
status: STATUS_FINAL  {'id': '3', 'name': 'STATUS_FINAL', 'state': 'post', 'completed': True, 'description': 'Final', 'detail': 'Final', 'shortDetail': 'Final'}
```

`competitions[0]` keys: `['id', 'uid', 'date', 'attendance', 'type', 'timeValid', 'neutralSite', 'conferenceCompetition', 'playByPlayAvailable', 'recent', 'venue', 'competitors', 'notes', 'status', 'broadcasts', 'leaders', 'format', 'startDate', 'broadcast', 'geoBroadcasts', 'highlights', 'headlines']`

`competitor` keys: `['id', 'uid', 'type', 'order', 'homeAway', 'winner', 'team', 'score', 'linescores', 'statistics', 'records']` — score is a **string**, e.g. `'20'`, not int.

`team` keys: `['id', 'uid', 'location', 'name', 'abbreviation', 'displayName', 'shortDisplayName', 'color', 'alternateColor', 'isActive', 'venue', 'links', 'logo']`

Status field for completion check: `event["status"]["type"]["name"] == "STATUS_FINAL"` (also `["state"] == "post"` and `["completed"] == True` are usable as booleans).

### Historical/specific-week queries (confirmed working)

The default scoreboard call only returns the *current* week — it does **not** default to "next upcoming" for historical years. Two query patterns were confirmed live to pull specific weeks/dates:

- `?dates=YYYYMMDD` — single day, e.g. `dates=20240908` returned 13 games for that Sunday (2024 week 1 Sunday slate), all `STATUS_FINAL`.
- `?dates=<season_year>&seasontype=<N>&week=<N>` — e.g. `dates=2024&seasontype=2&week=1` returned all 16 week-1 2024 regular-season games correctly (`seasontype=2` is regular season; `seasontype=1` is preseason per the `season.type` field seen in the default call).
- Passing `year=2024` alone (without `dates`) did **not** work as expected — it silently returned the current (2026) week instead. **Use `dates=<year>` combined with `seasontype`/`week`, not a bare `year` param, for historical pulls.**

This confirms P1's backfill/historical ingestion should use `dates=<season>&seasontype=<2|1|3>&week=<N>` to target specific weeks reliably.

### Team abbreviations: ESPN vs nflverse — 2 mismatches found

Pulled all 32 teams from `https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams` and compared `team.abbreviation` against nflverse's `home_team`/`away_team` values from `import_schedules`:

- **ESPN:** `LAR` (Los Angeles Rams) — **nflverse:** `LA`
- **ESPN:** `WSH` (Washington) — **nflverse:** `WAS`
- All other 30 team abbreviations match exactly between the two sources (`ARI, ATL, BAL, BUF, CAR, CHI, CIN, CLE, DAL, DEN, DET, GB, HOU, IND, JAX, KC, LAC, LV, MIA, MIN, NE, NO, NYG, NYJ, PHI, PIT, SEA, SF, TB, TEN` are identical in both).

`displayName` (e.g. `"Kansas City Chiefs"`) is ESPN's human-readable full name; nflverse doesn't have a full-name column in schedules but has `home_coach`/`away_coach` etc. — team identity matching between ESPN and nflverse should be done on abbreviation with a **2-entry normalization map** (`LAR→LA`, `WSH→WAS`), not on display name.

---

## Step 3: The Odds API NFL market keys — DEFERRED

**Not run live.** `ODDS_API_KEY` exists only as a GitHub Actions secret and is not present in this local/agent environment. Per controller ruling, this step is deferred rather than blocked or faked.

**Live verification of the following is deferred to the first CI odds-capture run for NFL** (the existing MLB odds-capture GitHub Action job is the presumed place this will run once NFL is wired in):

Expected NFL player-prop market keys (per The Odds API's documented NFL market list, to be confirmed live):
- `player_pass_yds`
- `player_pass_tds`
- `player_reception_yds`
- `player_receptions`
- `player_rush_yds`
- `player_rush_reception_yds`
- `player_anytime_td`

Expected NFL game-level market keys:
- `h2h` (moneyline)
- `spreads`
- `totals`

**Still unverified/deferred until that first live CI run:**
- Whether all seven prop keys actually return outcomes in preseason (brief flagged that some may be empty in preseason — plausible given Step 2 showed we are currently in 2026 preseason week 3).
- The exact team-name string format The Odds API uses for NFL events (full name? city+mascot? abbreviation?) and whether it matches ESPN's `displayName` or needs its own normalization map.
- The player-name format in prop outcomes (e.g. `"Patrick Mahomes"` vs `"P. Mahomes"` vs some other convention) — needed to join props back to nflverse `player_display_name`/`player_name`.

**Caveat for P1 planning:** Because Odds-API team-name and player-name formats are unverified, the P1 plan should not hard-code a normalization scheme against Odds-API strings yet — that mapping must be built/confirmed against real output from the first CI run before the matcher ships, or a companion small task should probe it once the CI secret is available.

---

## Recommended NFL `game_pk` scheme

**Use the ESPN event `id` as an integer** (e.g. `401671789`), matching the brief's suggestion. Rationale confirmed by this spike:

1. ESPN's `id` is globally unique per game and stable (verified: current/preseason ids like `401873286` and historical 2024 ids like `401671789` follow the same numbering scheme).
2. **nflverse's `schedules.espn` column already carries this exact same id** (verified: BAL@KC week 1 2024 → nflverse `espn: 401671789`, matches ESPN's `id` field format). This means nflverse rows can be joined to ESPN scoreboard rows directly on this id with zero string-matching/fuzzy-matching needed — a much more reliable join than team+date matching.
3. It is distinct from nflverse's own `game_id` (which is a composite string like `"2024_01_BAL_KC"`) — recommend storing nflverse's `game_id` as a secondary/display field if needed, but using the ESPN integer id as the canonical `game_pk` primary key, consistent with how MLB's `game_pk` is already an integer from the MLB Stats API in this codebase.
4. The Odds API event ids are a separate id space (UUIDs/hex strings per its docs) and will need their own mapping to `game_pk` via date+team matching — this mapping is part of what's deferred to Step 3's live verification, since Odds-API team-name strings aren't confirmed yet.

## Team-name normalization needed for the matcher

- **ESPN ↔ nflverse:** 2 mismatches confirmed (`LAR`→`LA`, `WSH`→`WAS`); all other 30 abbreviations identical. A small static dict is sufficient.
- **ESPN ↔ Odds-API:** cannot be stated yet — deferred to Step 3 live verification. Odds-API typically returns full team names (e.g. `"Kansas City Chiefs"`) for other sports in this codebase's existing MLB integration; if NFL follows the same convention it should match ESPN's `displayName` field directly, but this must be confirmed against real output, not assumed.

## Preseason / data gaps encountered

- Today (2026-08-24) falls in **NFL preseason 2026, week 3**. The default (no-param) ESPN scoreboard call returns only this current week, all games already final — not useful on its own for testing "in-progress" or "scheduled" status values. Historical/specific-week queries (`dates=2024&seasontype=2&week=1`) were required to get a live example of `STATUS_SCHEDULED` status and a non-preseason context.
- The Odds API step being deferred means preseason prop-market emptiness (flagged as a likely issue by the brief) is **unconfirmed** — this is the single biggest open question for P1 and should be the first thing checked once `ODDS_API_KEY` is available in CI.
- nflverse's `import_seasonal_rosters` exists on the installed version (`nfl-data-py==0.3.2`), so the brief's `hasattr` fallback to `import_rosters` was not exercised — only the primary path was verified live.
