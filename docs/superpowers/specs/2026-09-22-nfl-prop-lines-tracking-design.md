# NFL prop lines on every projection + sim prop accuracy tracking — design

Date: 2026-09-22 · Status: approved in chat ("That works")

## Goal

1. Show the book line and over/under prices next to **every** sim player-prop
   projection on the game page (not only +EV picks).
2. Track how the sim does on prop totals — passing / rushing / receiving yards,
   receptions, and rush attempts — graded against the book line and the actual.

## Current state (verified 2026-09-22)

- `odds_snapshot` holds NFL prop lines (pass_yds, pass_tds, reception_yds,
  receptions, rush_yds, rush_reception_yds, anytime_td), but only for events
  inside the pre-kick `PROP_WINDOW_MIN` (150 min) window on game-day crons —
  7 games last week. Props are fetched with `regions=us` (no Pinnacle).
- `build_ev_props_board.py` joins sim → odds but emits rows only for
  projected-usage-gated players, excludes pass_yds, and the site shows only
  `is_pick` rows.
- The sim allocates carries per player (`kernel.py` box stats) but does not
  emit them as a market.
- `nfl_player_actuals` is **empty**: `capture_nfl_player_actuals.py` (and
  `grade_ev_props.py`) read weekly stats through `nfl_data_py.import_weekly_data`,
  whose URL nflverse retired; the failure is swallowed (`required=False`), so
  the job is green and writes nothing. The canonical
  `stats_player/stats_player_week_2026.parquet` exists (last-modified
  2026-09-22) and `nflverse.load_release("weekly", ...)` already reads it.

## Design

### 1. Actuals fix (prerequisite)
- `capture_nfl_player_actuals.py` and `grade_ev_props.py` load weekly stats via
  `sportsmodel.nfl.nflverse.load_release("weekly", [season], required=False)`.
- `weekly_actuals.WEEKLY_STAT_COLS` gains `"rush_att": ("carries",)`.

### 2. Odds capture
- `SportConfig["nfl"].prop_market_map` gains `"rush_att": "player_rush_attempts"`.
- New daily full-slate prop snapshot: `ingest_odds.py` accepts
  `PROP_SCOPE=slate` (env), which captures props for every matched upcoming
  event of the target week instead of only the in-window ones. A new daily
  cron in `capture-odds.yml` (15:00 UTC) runs with `PROP_SCOPE=slate`;
  existing crons keep the window behavior. Budget: ~8 markets × 1 region ×
  ~16 events ≈ 130 credits/day, approved.

### 3. Sim
- Kernel box stats track carries per player; `rush_att` added to
  `_PLAYER_STAT_NAMES` and to the aggregate pmf markets; `MARKET_MAX["rush_att"] = 40`.
  Lands in `nfl_player_sim` with no schema change.

### 4. `nfl_prop_lines` table (new)
One row per `(game_pk, player_id, market)` for every sim projection with a
book line — **no usage gate, no +EV filter, pass_yds included**. Markets:
pass_yds, rush_yds, rec_yds, receptions, rush_att.

Columns: `game_pk, player_id, player_name, team, market, line, over_price,
over_book, under_price, under_book, projection, p_over, lean ('over'|'under'),
n_books, commence_time, updated_at`.

- Main line = the line posted by the most books (ties → lowest), same rule as
  the +EV board. Best price per side across all books.
- `lean` = over if `p_over > 0.5` else under (from the sim distribution at the
  line).
- Written by `build_ev_props_board.py` on every run (upsert). Rows are only
  built for upcoming games (the sim view filters `commence_time > now()`), so
  the last write before kickoff freezes the graded line.
- The pure builder lives in `sportsmodel.serving.props_ev` next to
  `assemble_prop_rows`, reusing its market mapping and name normalization
  (`SIM_TO_ODDS_MARKET` gains pass_yds and rush_att; the +EV path keeps
  excluding pass_yds/rush_att via its own gate so its behavior is unchanged).

### 5. Grading view `nfl_prop_grades`
`nfl_prop_lines` ⋈ `nfl_player_actuals` on (game_pk, player_id, market):
`actual`, `result` (`hit`/`miss`/`push` — actual vs line on the sim's lean),
`sim_err = |projection − actual|`, `line_err = |line − actual|`,
`sim_bias = projection − actual`. Anon SELECT grant. No new job.

### 6. Site (CappingAlpha `app.js`)
- Game page projection table: new RUSH ATT column. Each cell shows the
  projection, then (when a line exists) `o/u {line} · {over}/{under}`; after
  the game, the actual with ✓/✗ **vs the line on the sim's lean**
  (replacing the current "met/beat projection" check where a line exists;
  unchanged fallback where none does). +EV flag kept. Reads
  `nfl_prop_lines?game_pk=eq.X`.
- Track Record (+EV view): new "Sim prop accuracy" section from
  `nfl_prop_grades`: per market N, lean hit % (vs 52.4% breakeven), sim MAE vs
  line MAE, mean bias; plus an all-markets row.

## Out of scope
anytime_td / pass_tds tracking, CFB props, any change to +EV pick selection.

## User actions
Run `db/migration_nfl_prop_lines.sql`; redeploy CappingAlpha.

## Testing
Pure-function unit tests (prop-line builder, actuals `rush_att`, kernel carries,
ingest slate scope); local end-to-end run of build_ev_props_board against the
DB after the migration; browser check of the game page + Track Record.
