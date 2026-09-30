# Matchup grades, team history and power rankings — NFL + CFB (Phase 1 design spec)

Date: 2026-09-29 · Status: approved in chat, pending user review of this file

## Goal

Give bettors three context tools on every NFL and CFB game page, plus a Power
Rankings page per sport, all built from one leak-free weekly team data layer:

1. **Matchup grade** (A–F per side): offense unit vs the opponent's defense unit.
2. **History**: SU / ATS / O-U records over L5, L10, L20 and the season, current
   streaks, home/away and favorite/underdog splits.
3. **Power rankings**: our model's rating (projected margin vs an average team
   on a neutral field), rank, weekly move, unit ranks, SOS.

## Decisions (user, 2026-09-29)

- Both sports. Purpose: show on the site now AND test as model inputs later.
- Matchup grade = unit vs unit (not model-vs-market edge, not a composite).
- Power rankings = our model rating.
- Phasing: **Phase 1 (this spec)** = data layer + History + Power rankings +
  Matchup grades on the site. **Phase 2** = published projection scorecard
  (5+ held-out seasons). **Phase 3** = test the new data as model inputs (CFB
  profit gate re-run; NFL v2 ladder rung). Phases 2–3 get their own specs.

## Design

### 1. Team game log (one row per team per game)

Columns: sport, season, week, game_pk / game_id, date (ET), team, opponent,
home/away/neutral, points for/against, margin, closing spread from the team's
perspective (team line: + = favored by that many), closing total, favorite /
underdog / pick, result SU (W/L/T), ATS (W/L/P; team covers iff margin > team
line), O/U (O/U/P).

- NFL: nflverse schedules (`spread_line` / `total_line` = closing; 1999→now).
- CFB: `assets/cfb/schedules.parquet` + `assets/cfb/lines.parquet` (2015→now)
  plus the current season's live results and closing lines from
  `odds_snapshot` (the last pre-kickoff consensus) where CFBD hasn't posted.
- A game without a closing line has SU but no ATS/O-U.

### 2. History (per team, as of each game — strictly prior games)

- Windows: L5, L10, L20 (across seasons), season-to-date.
- Per window: SU W-L(-T), ATS W-L-P, O/U O-U-P, average margin, average cover
  margin (margin − team line), average total vs the line.
- Streaks: current SU, ATS and O/U streaks (e.g. "ATS W4", "Under 6 of 7" as
  "U6 of last 7" when ≥ 6 of 7).
- Season splits: home / away, as favorite / as underdog (SU and ATS).

### 3. Unit ratings

Opponent-adjusted, relative to league average, as of the target week
(same-season weeks before it; early season blends toward the previous season's
final value by games played: weight on current = games / (games + 3)):
- pass offense / pass defense, run offense / run defense — EPA per play and
  success rate; plus explosiveness (share of plays ≥ 20 yds pass / ≥ 10 yds
  run) per unit.
- NFL: `sportsmodel.nfl.unit_efficiency` (built for v2; nflverse pbp).
- CFB: CFBD per-game advanced stats (`/stats/game/advanced`: offense and
  defense PPA, success rate, explosiveness, passing/rushing splits) → the same
  opponent adjustment. Pulled through a workflow (API key secret), 2015→now,
  committed as `assets/cfb/advanced_games.parquet`; coverage reported per
  season.

### 4. Matchup grade (per side per game)

- Pass matchup = (offense pass EPA adj) + (opponent pass-defense EPA allowed
  adj); run matchup likewise; each is also computed for success rate and
  explosiveness and combined (weights 0.6 EPA, 0.25 success, 0.15 explosive).
- Overall = pass and run matchups weighted by the offense's pass rate (season
  to date, blended early like §3).
- Grades are percentiles against the distribution of all team-games from the
  previous three complete seasons (frozen at season start per sport): A ≥ 90th,
  B 70–90, C 30–70, D 10–30, F < 10. Overall, pass and run each get a grade.
- Fewer than 3 games played this season → shown with an "early" tag.

### 5. Power rating and rankings

- CFB rating = the ratings model's expected margin vs an average FBS team on a
  neutral field (Elo + SRS blend via `ratings.expected_margin` with the average
  team's ratings; points model for offense/defense scoring).
- NFL rating = points per game vs an average team, from the opponent-adjusted
  unit ratings: Σ units (EPA/play adj × the team's plays/game in that unit)
  for offense minus the same allowed for defense, plus 0 home field (neutral).
- Per team per week: rating, rank, previous week's rank and move, offense and
  defense unit ranks (pass off, run off, pass def, run def), SOS (average
  opponent rating so far), SU and ATS record.
- Recomputed daily; "week" = the upcoming slate's week.

### 6. Storage and job

- Tables (migration, user runs): `team_game_log`, `team_history` (latest per
  team, and per upcoming game for both sides), `matchup_grades` (per upcoming
  game and side), `power_rankings` (sport, season, week, team, …); anon read.
- Job `build-team-context` (daily, after grading/odds; also dispatchable) for
  NFL and CFB; idempotent upserts; history/rankings for upcoming games only
  plus the current rankings table (past weeks kept for the weekly move).
- Leakage: every value for a game uses only games before its kickoff.

### 7. Site (CappingAlpha, user redeploys)

- Game page (NFL + CFB): **Matchup** section (each side: overall grade + pass
  and run grades with unit bars) and **History** section (L5/L10/L20/season
  table, streak chips, splits, power rank + rating). Labelled "descriptive —
  not a pick".
- New **Power Rankings** page per sport (nav link): rank, team, rating, ▲/▼
  move, unit ranks, SOS, record, ATS; sortable; conference filter for CFB.
- Hidden gracefully when tables are missing/empty.

## Out of scope (Phase 1)

- Projection scorecard (Phase 2) and model-input tests (Phase 3).
- Situational trends beyond the splits above (NFL situational trends already
  exist and stay).

## Risks

- CFBD advanced-stats coverage before ~2016 and for FCS opponents may be thin
  → unit ratings for FCS games are skipped (grades shown only FBS vs FBS).
- The NFL power rating is a unit-efficiency points estimate, not the v2 sim
  itself (v2 has no single team number); it can disagree with a v2 game
  projection — the page shows both clearly labelled.
- Closing lines for the current CFB season come from our own captures, not
  CFBD — a game we didn't capture has no ATS/O-U.
