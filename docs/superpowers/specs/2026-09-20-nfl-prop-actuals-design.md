# NFL Prop Actuals Pipeline — Design

**Date:** 2026-09-20
**Status:** Draft for review
**Author:** Ryan + Claude

## Goal

Capture each NFL game's **actual** player box-score stats after it finishes and
surface them next to the sim's **projected** player props on the game page, so
you can see how the projection did — actual value beside the projected mean,
with a hit/miss marker. This mirrors the game-level "Model vs Simulation vs
Actual" comparison already shipped (which reads `prediction_accuracy`); this
adds the equivalent at the player-prop level.

Today `ev_prop_results` records CLV only for the handful of **+EV picks** the
board took. There is no store of realized stats for the full projected slate,
and nothing renders actuals on the projected-props table. This pipeline fills
that gap.

## Scope

- **In:** the six markets the sim already projects and the props table already
  shows — `pass_yds`, `pass_tds`, `rush_yds`, `rec_yds`, `receptions`,
  `anytime_td` — for every player the sim projected in a finished game.
- **Out:** no new markets, no book-line grading (that's `ev_prop_results`), no
  change to how projections are produced, no other sport (NFL only, matching
  the sim). CFB has no player sim, so no actuals there.

## Data source & resolution (reuse, don't reinvent)

`scripts/grade_ev_props.py` already resolves a player's realized weekly stat
and is the proven recipe. It:
1. derives the season from `commence_time` via
   `sportsmodel.nfl.injuries_nflverse.nfl_season`;
2. resolves the game's NFL **week** by matching `game_pk` (an ESPN event id)
   against nflverse `import_schedules`' own `espn` column
   (`_week_for_game`);
3. fetches that season's `import_weekly_data` (via the resilient
   `import_by_season`, `required=False`), joins by `player_id == gsis_id` and
   `week`, and reads the market's weekly column
   (`props_ev.SIM_MARKET_TO_WEEKLY`).

**Refactor (light):** extract the pure pieces into a new module
`src/sportsmodel/nfl/weekly_actuals.py`:

- `WEEKLY_STAT_COLS: dict[str, str | tuple]` — the market → nflverse weekly
  column map for **all six** projected markets (superset of
  `SIM_MARKET_TO_WEEKLY`, which stays as-is for the grader). Values:
  - `pass_yds → "passing_yards"`
  - `pass_tds → "passing_tds"`
  - `rush_yds → "rushing_yards"`
  - `rec_yds → "receiving_yards"`
  - `receptions → "receptions"`
  - `anytime_td → ("rushing_tds", "receiving_tds")` (summed; the realized
    "did they score" quantity)
- `week_for_game(sched_df, game_pk) -> int | None` — the exact `espn`-column
  join, pure (DataFrame in).
- `player_market_actual(weekly_row, market) -> float | None` — reads (and for
  `anytime_td`, sums) the column(s), NaN/missing → None.

`grade_ev_props.py` keeps working unchanged; optionally (nice-to-have, not
required) it can later import `week_for_game` from this module to drop its
private copy. The new capture script is the only required consumer.

## Storage

New table `nfl_player_actuals` (migration `db/migration_nfl_player_actuals.sql`,
**you run it**), anon-readable like the other serving tables:

```sql
CREATE TABLE IF NOT EXISTS nfl_player_actuals (
    game_pk      BIGINT NOT NULL,
    player_id    TEXT   NOT NULL,   -- gsis_id, same namespace as nfl_player_sim
    player_name  TEXT,
    market       TEXT   NOT NULL,   -- pass_yds|pass_tds|rush_yds|rec_yds|receptions|anytime_td
    actual       DOUBLE PRECISION,  -- realized stat; for anytime_td = total TDs (>=1 == scored)
    season       INT,
    week         INT,
    captured_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market)
);
CREATE INDEX IF NOT EXISTS idx_nfl_player_actuals_game ON nfl_player_actuals (game_pk);
ALTER TABLE nfl_player_actuals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_player_actuals" ON nfl_player_actuals;
CREATE POLICY "public read nfl_player_actuals" ON nfl_player_actuals FOR SELECT USING (true);
GRANT SELECT ON nfl_player_actuals TO anon, authenticated;
```

Actuals are **not** model-versioned (a realized stat is a fact, not a model
output). One row per (game_pk, player_id, market). Kept in its own table rather
than as a column on `nfl_player_sim` so the replace-per-game sim write path
(which deletes/re-inserts projection rows) never touches captured facts, and so
a started game's actuals persist independently of its frozen projection.

`db.upsert_nfl_player_actuals(records)` — idempotent on the PK, plain upsert
(`ON CONFLICT (game_pk, player_id, market) DO UPDATE`), mirroring the existing
`upsert_*` helpers.

## Capture job

`scripts/capture_nfl_player_actuals.py` (NFL only; reads Supabase + nflverse,
**no** Odds API):

1. Select finished games in a rolling window: distinct `game_pk` +
   `commence_time` from `nfl_player_sim` where `commence_time <= now()` and
   `commence_time >= now() - INTERVAL '<days> days'` (default 8), and that don't yet have
   a full set of `nfl_player_actuals` rows (a cheap `NOT EXISTS` / left-join
   count, matching the grader's idempotent-rerun posture).
2. Per game: derive season, resolve week via schedules (skip the game if the
   week/schedule can't be resolved — weekly data not published yet — and try
   again next run, never crash the batch).
3. Fetch that season's weekly frame once (per-run season cache, like the
   grader's `_weekly_cache`). For **each player_id the sim projected for that
   game** (join to the game's `nfl_player_sim` player_ids), read all six
   markets' actuals via `weekly_actuals`, and upsert the resolved ones.
4. Print a one-line summary (`games=N players=M actuals=K skipped=S`), matching
   the other producers.

Only players the sim projected are captured — the table aligns 1:1 with the
projected-props table the UI renders, and we don't store box lines for players
we never showed.

**Pure core / thin IO:** `assemble_actual_rows(projected_player_ids,
weekly_frame, game_pk, season, week)` is pure (frames + ids in, records out)
and unit-tested; `main()` does the DB/nflverse IO and isn't unit-tested (same
split as `generate_sim_nfl.py`).

## Workflow

`.github/workflows/capture-player-actuals.yml` — scheduled a few hours after
the game windows and again the next morning (NFL games finalize late; weekly
data lands after), e.g. `cron: "0 4,15 * * 1,2,5,6,0"` (Tue/Wed/Sat/Sun/Mon
mornings + afternoons UTC), plus `workflow_dispatch`. Reuses `DATABASE_URL`.
Rolling window (default `--days 8`) so a missed run self-heals. Free to run
often (no odds credits).

## Front-end (external CappingAlpha `app.js`, you redeploy)

`buildGame()` additionally fetches
`nfl_player_actuals?game_pk=eq.${game}` (NFL, `.catch(() => [])`) and passes it
to `propsProjectionSection(sims, props, actuals)`.

In the projected-props table, when actuals exist for the game, each stat cell
shows the actual beneath the projection with a hit/miss marker:

- Yardage / count markets (`pass_yds`, `pass_tds`, `rush_yds`, `rec_yds`,
  `receptions`): render `proj → actual` with a ✓ when the player **met or beat**
  the projected mean, ✗ when under (a projection-vs-result read, since these
  rows have no single book line; the +EV flag already shows the book line where
  one was offered). Example cell: `268 → 250 ✗`.
- `anytime_td` (ANY TD column): show the projected `%` with the realized result
  — `TD ✓` if actual TDs ≥ 1, else `— ✗` — beneath it.

A `hasActuals` flag toggles the extra sub-row/line and the section note
("→ actual, ✓/✗ vs the projected number"). When no actuals exist yet (upcoming
or not-yet-captured game) the table renders exactly as today. Add the small CSS
(`.prop-actual`, `.hit`, `.miss`) to `injectStylesOnce()`, and bump the
`app.js?v=` cache-buster.

## Testing

- `weekly_actuals`: `week_for_game` (match / no-match / missing `espn` col);
  `player_market_actual` per market incl. `anytime_td` summing rush+rec TDs and
  NaN → None.
- `assemble_actual_rows`: builds one row per (player, market) for projected
  players present in the weekly frame; skips players/markets absent; correct
  `anytime_td` sum; deterministic column order.
- `db.upsert_nfl_player_actuals`: FakeCursor test (column-order tuple, ON
  CONFLICT clause, empty-list no-op, commit called) — mirrors
  `tests/test_db_nfl_sim.py`.
- Full suite stays green.

## Deployment (your actions, in order)

1. Run `db/migration_nfl_player_actuals.sql` in Supabase.
2. Merge to `main`; enable `capture-player-actuals.yml` (it errors on a missing
   table if run before step 1, so migration first).
3. Dispatch the workflow once for a slate that has finished games to backfill,
   and confirm rows land.
4. Redeploy `app.js` (+ bumped cache-buster) to Cloudflare.

## Non-goals / honest caveats

- This shows **projection accuracy**, not betting results — hit/miss is vs our
  own number, not a closing line. CLV on actual +EV picks stays in
  `ev_prop_results`.
- Actuals depend on nflverse weekly publishing (can lag a day+); the rolling
  window + idempotent re-runs handle the delay. A game whose week can't be
  resolved is simply retried next run.
- Only players the sim projected are captured, so a surprise contributor the
  sim didn't list won't appear (acceptable — the table is the projection's
  report card).
