# CFB power rankings: 60% ESPN FPI + 40% CappingAlpha model

Date: 2026-10-06. Status: design approved in chat; awaiting spec review.

## Goal

CFB power rankings rank teams by a blended rating: 60% ESPN Football Power Index (FPI) plus 40% our
current results-based model rating (the CappingAlpha rating, `context/power.py` via
`power_asof`). NFL rankings are unchanged.

## Decisions (user, 2026-10-06)

- ESPN source = FPI (not the AP/Coaches polls): it rates every FBS team.
- Blend on ratings, not ranks: `rating = 0.6 * fpi + 0.4 * model_rating`, then rank descending.
  Both are points vs an average team on a neutral field. On 2026-10-06 they correlate 0.97 (sd 12.7
  vs 11.8), and a z-scored blend never differs from the raw blend by more than 1 rank spot, so the
  raw average is used.
- `move` in the first (rollout) week is blank rather than compared with the old model-only rank.
- The site shows the blended Rating plus FPI and Model columns.

## Data source

`https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/seasons/{season}/powerindex?limit=200`
(public, no key; the same numbers as espn.com/college-football/fpi). Each item has `team.$ref`
(ESPN team id in the URL path, the id our CFB tables use) and `predictives[]` with `name == "fpi"`.
The feed returns about 138 items (FBS plus transitional teams); only FBS ids
(`cfb.teams.load_fbs_ids`) are used. Only the current FPI is available, with no weekly history.
`lastUpdated` is the freshness stamp. (`runDateTimeKey` can lag and is not used.)

## Components

1. `sportsmodel/cfb/espn.py`
   - `parse_fpi(payload) -> dict[str, float]`: pure. Returns team id -> FPI value. Items without a
     numeric fpi are skipped.
   - `fetch_fpi(season) -> dict[str, float]`: GET of the core-API URL above (not the module's
     site-API `_BASE`), with the same tenacity retry (3 tries, exponential backoff) and 20s timeout
     as `_get`. Follows `pageCount` if more than one page comes back.
2. `sportsmodel/context/power.py`
   - `blend_fpi(power_df, fpi, w_fpi=0.6) -> DataFrame`: pure.
     - Adds `model_rating` (the original `rating`) and `fpi` (NaN when missing).
     - Sets `rating = w_fpi * fpi + (1 - w_fpi) * model_rating`. A team without FPI keeps
       `rating = model_rating`.
     - `W_FPI = 0.6` is a module constant.
   - `rankings()` is unchanged. It is called with the blended frame, so rank, SOS and SOV all use
     the blended rating. `prev_power_df` is passed as None for CFB; `prev_rank` is set afterwards
     (item 3).
   - `prev_ranks_from_published(rows) -> dict[str, int]`: pure. Takes the stored rows of the
     previous published CFB week and returns {} unless that week was blended (any non-null `fpi`).
3. `scripts/build_team_context.py` (`build_cfb`)
   - New keyword arguments `fpi: dict | None` and `prev_published: list[dict] | None` (injected in
     tests, the same pattern as `priors`/`rp`).
   - `load_cfb_sources` fetches FPI for the ranking season. If the fetch fails or returns nothing,
     it warns (`::warning::`) and passes `fpi=None`, which ranks on the model rating alone for that
     run.
   - It also reads the previous published week from `power_rankings`: sport = 'cfb', same season,
     the max week below the current ranking week. This requires DATABASE_URL; without it the job
     warns and leaves `move` blank.
   - After `rankings()`, set `prev_rank` from `prev_ranks_from_published` and recompute `move`.
     Teams absent last week stay NULL.
4. `db.py`: `TEAM_CONTEXT_COLUMNS["power_rankings"]` gains `fpi` and `model_rating`. NFL rows write
   NULL for both.
5. `db/migration_power_fpi.sql` (user runs it in Supabase BEFORE the merge, like
   `migration_power_results.sql`):
   - `ALTER TABLE power_rankings ADD COLUMN IF NOT EXISTS fpi double precision, ADD COLUMN IF NOT
     EXISTS model_rating double precision;`
   - Recreate `power_rankings_current` so the view carries the new columns, and keep its grants.
6. Site (`site/app.js`)
   - CFB only: add `FPI` and `MODEL` columns after `RATING`, signed points, sortable, with "—" when
     null.
   - Add a CFB-specific intro: "Rating = 60% ESPN FPI + 40% CappingAlpha model rating (points
     better than an average team on a neutral field) ...", keeping the existing model explanation
     for the Model column.
   - The game-page power chip tooltip reads "60% ESPN FPI + 40% CappingAlpha model" for CFB.
   - Bump the cache key on every HTML page.

## Error handling

| Failure | Behavior |
|---|---|
| FPI fetch fails / empty | warn; CFB ranked on the model rating; rows have `fpi` NULL; job succeeds |
| Team missing from FPI | that team's rating = model rating; `fpi` NULL |
| No DATABASE_URL / no previous published week / previous week not blended | `prev_rank`/`move` NULL |
| Migration not run | the upsert fails loudly (the go-live order prevents this) |

## Testing

- `parse_fpi` against a trimmed real payload fixture (`tests/fixtures/cfb/espn_fpi.json`), including
  an item with no fpi value.
- `blend_fpi`:
  - Hand-checked 60/40 values.
  - A missing-FPI team falls back to its model rating.
  - Rank order follows the blend, and SOS uses blended opponent ratings.
- `prev_ranks_from_published`:
  - A blended previous week maps ranks.
  - An unblended (all-NULL fpi) week returns {}.
- `build_cfb` with injected fpi/prev:
  - The `fpi`/`model_rating` columns are present.
  - `move` = prev − rank.
  - `fpi=None` gives model-only ranking, with output identical to today's except `prev_rank`/`move`.
- NFL `build_nfl` output unchanged (existing tests).
- Site tests (`site/tests`): the CFB table has FPI/MODEL columns, NFL does not, and both sort and
  render null as "—".

## Out of scope

- NFL FPI blend. The weights stay fixed (no fitting).
- Storing FPI history beyond the weekly `power_rankings` rows.
- No backtest is possible (no FPI history). The rankings stay descriptive and are not read by the
  betting models.

## Go-live

1. The user runs `db/migration_power_fpi.sql`.
2. Merge.
3. Dispatch `build-team-context` once.
4. The user syncs and redeploys the site with the new cache key.
