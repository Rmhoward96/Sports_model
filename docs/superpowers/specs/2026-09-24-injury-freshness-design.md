# Injury freshness gate + day-before re-run — design

Date: 2026-09-24 · Status: approved in chat (trigger = any change into/out of Out/Doubtful).

## Problem (verified 2026-09-24)
The desk said Michael Penix Jr. (ATL) is Out; he is Active. The NFL desk reads
nflverse's weekly injury report directly (`desk_inputs._nfl_injuries_by_name` →
`injuries_nflverse.current_injuries`), which uses the NEWEST report — last week's
until the Wed–Fri report posts. Week 2 listed Penix Out; the desk ran Wednesday
(before the Week 3 report) and won't re-run until Saturday. The sim already
overlays ESPN's current statuses (`generate_sim_nfl._merge_espn_injuries`); the
desk never got that fix.

## 1. One NFL injury source (`src/sportsmodel/nfl/injury_report.py`)
- PURE `merge_report(nflverse_by_abbr, report_week, target_week, espn_rows,
  name_to_abbr)` → `{"by_team": {abbr: [{player, position, status, note,
  source}]}, "stale": bool, "report_week", "target_week", "espn_available": bool,
  "conflicts": [{team, player, nflverse, espn}]}`.
  - `stale` = report_week is None or report_week < target_week.
  - If stale AND ESPN is available: nflverse rows are DROPPED (never used).
    If ESPN is unavailable: nflverse rows are used even if stale (flagged).
  - ESPN overlay per player (normalized name): ESPN's designation replaces any
    nflverse row; ESPN "Active"/other clears it; IR/PUP/suspension/NFI → "Out".
  - `conflicts`: players where the nflverse status (current or stale) and ESPN
    disagree on Out/Doubtful vs not.
- IO `current_report(now, target_week, name_to_abbr)`: nflverse season report
  (report_week = latest week with a designation) + `espn.fetch_injuries()`
  (non-fatal).
- Used by BOTH `generate_sim_nfl` (replacing its local ESPN merge) and
  `desk_inputs` (NFL).

## 2. Gate in the desk
- Each bundle game gets `news.injury_report = {source, stale, as_of,
  conflicts (this game's teams only)}`.
- Prompt: statuses are the CURRENT report (ESPN-verified); never state a player
  is out unless he's listed; if `stale`, say the official report wasn't posted
  and statuses are ESPN's; mention any conflict you relied on.

## 3. Day-before re-check + re-run (`injury-watch`)
- Table `injury_snapshots(sport, game_pk, fingerprint, statuses jsonb,
  captured_at)`, PK (sport, game_pk). Internal (RLS on, no anon policy).
- Fingerprint per game = sorted `team|player|status` over both teams' Out/
  Doubtful players (Questionable ignored — per the approved trigger).
- `scripts/injury_watch.py --sport {nfl,cfb} --check|--record`:
  - `--check`: games kicking off within the next 30h; compare current
    fingerprints to stored; missing stored snapshot = changed; print the player
    diffs; write `changed=true|false` to `$GITHUB_OUTPUT`.
  - `--record`: store fingerprints for all upcoming games (7 days).
  - Same injury sources as the desk (NFL: `injury_report`; CFB: SportsDataIO
    via `desk_inputs`).
- Workflow `injury-watch.yml` (15:00 + 22:00 UTC daily), per sport: check → if
  changed: model (`generate_{nfl,cfb}.py`) → sim (NFL) → desk (inputs →
  synthesize → write) → `build_ev_board.py` → `build_best_parlays.py` →
  `--record`. Concurrency group per sport.
- `desk-auto-{nfl,cfb}` get a final `--record` step (baseline = what the desk saw).

## Out of scope
CFB stale-report detection (SportsDataIO is a live feed), changing sim/desk
injury weighting.
