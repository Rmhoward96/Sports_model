# Betting trends on game pages + in the Decision Desk — design

Date: 2026-09-23 · Status: approved in chat ("Both"; "Yes, that looks right";
desk weight = "Supporting evidence")

## Goal
Show each game's betting trends on its game page, and feed them to the Decision
Desk so the sports-analyst note cites every applicable trend when deciding.

## Findings (spike, 2026-09-23)
The Apify actor `zen-studio/action-network-odds` with `includeStandings: true`
returns, besides game items, ONE extra item per league:
`{"recordType": "standings", "league", "season", "entryCount", "standings": [...]}`
(NFL 32 teams, NCAAF 138). Each standings row has `team` ({name, abbreviation,
conference, division, ...}) and `records`: a list of
`{category, wins, losses, ties, draws, overs, unders, ...}`. Betting categories:
`ats, ats_last_5, ats_last_10, ats_home, ats_road, ats_fav, ats_dog, ats_even,
ats_division, ats_conference`; the same splits for `over_under_*` (counts in
`overs`/`unders`) and `units_*` (value in `wins`); SU `last_5, last_10, home,
road, fav, dog`. Pushes are in `draws` (or `ties`). Situational PRO trends
("off a road game") are NOT returned.

## 1. Season betting records (NFL + CFB, Action Network)
- New daily capture (`scripts/capture_team_records.py`, workflow
  `build-trends.yml`, 20:30 UTC daily — before the desk at 22:15): one actor call
  per league with `maxGames: 1, includeStandings: true`, current season.
- Pure parser → one row per team: `{sport, season, team_name, an_team_name,
  abbr, records}` where `records = {category: {"w","l","p","o","u"}}`
  (p = draws or ties or 0; o/u only for over_under_*).
- `team_name` = our canonical display name, matched by normalized name against
  the sport's known names (NFL crosswalk values; CFB fbs_teams.json values);
  unmatched teams keep the AN name and are logged.
- Table `team_betting_records` PK (sport, season, team_name).

## 2. NFL situational trends (computed)
- From nflverse schedules (`load_schedules`, seasons current−3..current;
  REG+POST, completed games with a spread_line): per-team game log with
  team_margin, team line (home: −spread_line, away: +spread_line), cover
  (margin + line: >0 W, <0 L, 0 P), total vs total_line (O/U/P), home/road,
  date, kickoff time, opponent.
- For each upcoming game (predictions_current, NFL) and each team, the
  situations that apply THIS week: `home`/`road`; `favorite`/`underdog` (current
  line from predictions_current.market_spread, fallback schedule spread_line;
  pick'em → neither); `off_road`/`off_home` (previous game); `off_win`/`off_loss`
  (previous SU); `off_bye` (≥13 days since previous game, same season);
  `division` (fixed 2002+ division map); `primetime` (kickoff ≥ 19:00 ET).
- For each applicable situation: the team's ATS W-L-P and O/U O-U-P over its
  past games in that same situation (situation evaluated per historical game),
  window = current season + last 3. Keep only n ≥ 5.
- Table `nfl_game_trends` (game_pk, team_name, situation, label, ats_w, ats_l,
  ats_p, ou_o, ou_u, ou_p, n, since_season); rows for a game are replaced each run.
  Builder `scripts/build_nfl_trends.py` in the same daily workflow.

## 3. Decision Desk
- `desk_inputs` adds per game `trends: {"home": {...}, "away": {...}}`, each
  `{"records": {<compact labeled strings>}, "situational": [<labels>]}` e.g.
  `"ATS 2-0-0"`, `"ATS road 1-0-0"`, `"O/U 2-0-0 (overs-unders-pushes)"`,
  `"Units +1.2"`, `"7-2-0 ATS off a road game since 2023"`.
- SYSTEM_PROMPT: the sports-analyst note (`agent_notes.analyst`) MUST cite every
  applicable trend for both teams. Trends are SUPPORTING EVIDENCE: they may raise
  or lower conviction or tip a close call, but a spread lean can never rest on
  trends alone — it still needs a concrete edge (injury, form, model-vs-line).
  Rationale says when a trend influenced the call.

## 4. Game page (CappingAlpha)
- New "Trends" section directly below Public Betting (NFL + CFB): away | home
  columns; season records relevant to this game (ATS overall, ATS at this
  venue, ATS in this role fav/dog, ATS last 5, O/U overall, O/U at this venue,
  units), then (NFL) situational trends. A record is highlighted when ≥70% one
  way with ≥8 decided games. Footnote: trends are descriptive, not predictive;
  the model doesn't use them.

## Out of scope
CFB situational trends (no historical CFB lines), PRO systems, trend-based
model features.

## User actions
Run `db/migration_trends.sql`; redeploy CappingAlpha.
