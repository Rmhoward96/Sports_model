# CFB site panels (venue/weather, CFB insights, projected game flow) — design spec

Date: 2026-10-06 · Status: design approved in chat; awaiting user review of this file
Project 2 of 3 using the user's CFBD Tier 3 key (project 1 = CFB v3 efficiency model, merged as data + records, failed the gate, v2 live; project 3 = live in-game model).

## Goal

Light up the game-page panels that the Phase A redesign left hidden (site redesign spec §4.2/§4.3/§4.5), using CFBD data for CFB and the free ESPN summary for NFL: a venue + weather line in the hero (CFB + NFL), new CFB Key Insights rows (havoc, turnovers) plus a weather row (CFB + NFL), and a CFB Projected Game Flow chart. Real data only; anything missing is hidden.

## User decisions (2026-10-06)

- Venue/weather for CFB **and** NFL; havoc/turnover insights and game flow CFB-only.
- Game flow = each team's own quarter scoring pattern (last two seasons, shrunk to league), applied to its projected score; finished games also show the actual line score.
- Architecture A: a daily job publishes three small Supabase tables; the site reads them (the site reads Supabase, not repo files). The user runs the migration.

## Current state

- `site/js/pages/game.js` header comment: Phase B items (venue/weather line, Projected Game Flow, TV network) omitted. Key Insights exists (`gmKeyInsights`, up to four rows: grade, explosive, power, streak; icons in `GM_INSIGHT_ICON`).
- CFBD assets in `assets/cfb/` (from project 1): `cfbd_games` (no line scores), `weather_games` (temp, wind, precip, snowfall, humidity, game_indoors — historical/actual), `venues` (name, timezone, lat/lon, elevation, dome — **no city/state yet**), `havoc_games`, `advanced_games` (incl. explosiveness), `drive_games`. CFBD game ids equal ESPN event ids (= our `game_pk`). Client: `src/sportsmodel/cfb/cfbd.py` (`CfbdClient`), parsers `cfb/cfbd_games.py`.
- ESPN helpers exist: `src/sportsmodel/nfl/espn.py`, `src/sportsmodel/cfb/espn.py`.
- Supabase: DB writes via `DATABASE_URL` in Actions; site reads with the anon key; tables need a read policy for anon (RLS). Migrations live in `db/migration_*.sql` and are run by the user.

## 1. Data (three new Supabase tables)

### `game_info` (CFB + NFL) — one row per (sport, game_pk), upcoming window (today − 2 days … +10 days)
Columns: `sport`, `game_pk`, `venue_name`, `city`, `state`, `indoor` (bool), `temp_f`, `wind_mph`, `precip_chance` (0–100, nullable), `precip_in` (nullable), `conditions` (short text, nullable), `weather_kind` ('forecast' | 'observed'), `source` ('cfbd' | 'espn'), `captured_at`, `updated_at`. PK (sport, game_pk).
- CFB: venue from CFBD `/venues` (extend the venue parser to keep `city`, `state`); game ↔ venue from CFBD `/games`; weather from CFBD `/games/weather` for the window (forecast before kickoff, observed after).
- NFL: ESPN game summary `gameInfo.venue` (fullName, address.city/state, indoor) and `gameInfo.weather` / `weather` block when present (temperature, conditions; wind/precip only if ESPN provides them).
- Indoor → weather columns NULL and the site shows "Indoors". Any missing field stays NULL (never guessed).

### `cfb_team_insights` (current season) — one row per (season, team)
Season-to-date, regular season, FBS games, through the last completed week: `games`, `def_havoc_rate` + national rank (higher = better), `off_havoc_allowed_rate` + rank (lower allowed = better), `turnover_margin` (total and per game) + rank, `off_explosiveness` + rank, `def_explosiveness_allowed` + rank, `through_week`, `updated_at`. Ranks among FBS teams with ≥ 3 games. Sources: `havoc_games`, `advanced_games` (already pulled weekly after merge), plus a new small CFBD pull for turnovers (`/games/teams` per week: fumbles lost, interceptions thrown, and the opponent's) stored as `assets/cfb/team_game_stats.parquet`.

### `cfb_quarter_shares` — one row per team
For the current season's FBS teams: share of points **scored** and **allowed** in Q1–Q4 (OT excluded; shares sum to 1) over the last two completed seasons plus the current season to date, shrunk toward the league average with weight n/(n+k), k = 12 games, plus `games_used`, `updated_at`. Needs CFBD line scores: extend the games parser to keep `homeLineScores` / `awayLineScores` (new columns `home_q1..home_q4`, `away_q1..away_q4`, `home_ot`, `away_ot`; NaN when absent) and backfill 2024–2026 (≈ 3–6 calls).

### Job
- New `scripts/build_game_info.py` + workflow `build-game-info.yml`: daily at 14:00 UTC and on football days (Thu/Fri/Sat/Sun/Mon) at 16:00 and 22:00 UTC (≈ 2 h before typical first kickoffs); upserts `game_info` for both sports.
- The Monday CFB job (`build-cfb-advanced.yml`) additionally runs the turnover pull + line-score refresh and rebuilds `cfb_team_insights` and `cfb_quarter_shares`.
- CFBD budget: ≈ 60–100 calls/week; ESPN summary calls ≈ NFL games in window per run (free).
- Jobs write only these three tables (upsert; never delete rows outside the window they rebuilt).

## 2. Site

- **Hero line** (game page, CFB + NFL), under the kickoff line: `Venue · City, ST · 41°F, wind 14 mph, 20% rain` (pieces joined with " · "; any missing piece omitted; indoor → `Venue · City, ST · Indoors`; finished games label observed weather). Hidden entirely when `game_info` has no row.
- **Key Insights** (`gmKeyInsights`): add candidate rows, each with a strength score, and keep the existing four-row cap by strength:
  - CFB *Havoc mismatch*: one defense's havoc rank vs the other offense's havoc-allowed rank — shown when the rank gap ≥ 30 or one side is in the top/bottom 15% nationally.
  - CFB *Turnover edge*: turnover-margin rank gap ≥ 30 or a top/bottom-15% team.
  - CFB + NFL *Weather*: wind ≥ 15 mph, temp < 40°F, or precip chance ≥ 50% / measurable precip — outdoor games only.
  - Existing rows keep their logic; strength for existing rows is derived from their current thresholds so the ordering is deterministic. New icons follow `GM_INSIGHT_ICON`.
- **Projected Game Flow** (CFB, new card after Key Insights): per team, projected points Q1–Q4 = projected team score × blended share, where blended share = average of the team's scored share and the opponent's allowed share (renormalised to 1). Small grouped bar or line chart via existing `ui.js` chart helpers, labelled "Projected pace", with an ⓘ tooltip explaining the method. Finished games: actual quarter scores beside the projection. Hidden when there is no projection or no share row for either team.
- Fetches use the existing `sb()` / `safeCard` patterns; escaping via `ctxEsc`; cache key bumped once at the end; panels hide cleanly if the tables don't exist yet (site may deploy before the migration).

## 3. Operations and testing

- `db/migration_site_panels.sql`: the three tables, PKs, anon read policies (SELECT) consistent with existing serving tables, index on `game_info (sport, game_pk)`. User runs it.
- Tests (network-free): CFBD venue/line-score/team-stats parser extensions; ESPN summary venue/weather parser (indoor, missing weather); quarter-share shrinkage (sums to 1, small-n → league); insights stats and ranks; game_info builder (window, indoor nulls, upsert rows); site: hero-line formatting (all pieces, missing pieces, indoor, observed), insight candidates + thresholds + top-four ordering, game-flow math and hide conditions; workflow schedule test.
- Visual check: NFL and CFB game pages at 1280 / 1440 / 1920 (hard reload), with and without data.

## Out of scope

TV network, stadium photos, NFL havoc/turnover insights, NFL game flow, live in-game data (project 3), any model change.

## Risks

- ESPN NFL weather is often missing until close to kickoff → hidden until present.
- CFBD forecast availability varies by game → hidden when absent.
- Quarter shares are noisy even after shrinkage → labelled "projected pace", not a prediction of the score by quarter.
