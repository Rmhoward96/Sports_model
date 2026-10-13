# Pick ledger: go-live checklist (2026-10-13)

Branch `feat/pick-ledger`. Everything below is for you to run; nothing here has been executed. No migration was run, nothing was pushed, no workflow was dispatched, and no code ever connected to the database during the build.

Until step 1 is done every ledger job prints a `::warning::` and exits 0, and the pick writers behave exactly as before. Step 1 is safe to do before or after the merge.

## 1. Run the migration

Paste `db/migration_pick_ledger.sql` into the Supabase SQL editor and run it once (idempotent, safe to re-run; you may already have run it). It creates `pick_ledger` (append-only: trigger plus revoked privileges), `close_capture` and `pick_ledger_results`.

Optional append-only check (rolls back; run it yourself in the SQL editor):

```sql
BEGIN;
INSERT INTO pick_ledger (pick_key, source, sport, game_pk, market, side, observed_at, commence_time,
                         minutes_to_kick, writer, writer_run)
VALUES ('t', 'ev_game', 'nfl', 1, 'spread', 'home', now(), now() + interval '1 day', 1440, 'manual', 'local');
UPDATE pick_ledger SET best_price = 100;   -- must fail: pick_ledger is append-only: UPDATE is not allowed
ROLLBACK;
```

## 2. Merge and push

Push `feat/pick-ledger`, open the PR, merge to `main`. Merge before Thursday Oct 15 if you want Thursday-night football captured. Schedules begin on merge:

| Workflow | Schedule (UTC) |
|---|---|
| `capture-close` | every 10 min in game-day hours (:03 football, :08 baseball) |
| `grade-ledger` | daily 14:10 |
| `report-ledger-gate` | Tuesday 15:47 |

Existing workflows are unchanged. The PR description should carry the probe verdict (step 3), the credit table (about 360 credits a week added, covering football and the MLB postseason only; a full MLB regular season, Mar-Sep, adds roughly 500-800 more a week, and the 30,000 a month budget still holds; props stop below 600 credits, everything below 200; the regular odds ingest stops props below 2,500 and game lines below 800, so it runs out first) and the list of resolved ambiguities from the plan's decisions table.

Note: the "close" is captured about 15-25 minutes before kickoff, not at kickoff.

## 3. Dispatch the Pinnacle props probe

Costs about 8 Odds API credits, never touches the database. It must run from `main`, so only after the merge:

```
gh workflow run probe-pinnacle-props.yml --ref main
```

It needs an upcoming NFL event in the Odds API (there is one every week in season).

## 4. Read the verdict

Open the run log or job summary and find `PROBE_VERDICT=`:

- `pinnacle_has_props`: Pinnacle publishes NFL props. Leave `CLOSE_PROP_REGIONS` unset (default `us,eu`).
- `pinnacle_absent`: go to step 5.
- `inconclusive`: usually no upcoming event, or a transient error. Re-dispatch during a game week. Until you have a verdict, the default `us,eu` is safe (it only costs extra credits).

## 5. Set CLOSE_PROP_REGIONS if Pinnacle is absent

GitHub repo, Settings, Secrets and variables, Actions, Variables tab: create variable `CLOSE_PROP_REGIONS` = `us`. This drops the `eu` region from prop captures (roughly halves props credits). No redeploy needed; the next `capture-close` run reads it. Props coverage then relies on the gate's other route (3 or more books at the pick's exact line).

## 6. First football weekend (Thu Oct 15 to Mon Oct 19)

Expect, per NFL/CFB game:

- one `close_capture` row, `kind = lines`, `status = ok`, `minutes_to_kick` between 5 and 25;
- props capture rows for games that hold prop picks;
- `pick_ledger` rows appearing from the next board and generator builds (a row is added on first sight of a pick, then only when price, line or model/ref probability moves by the append rule);
- `pick_ledger_results` filling in the day after games finish (`grade-ledger` 14:10 UTC), with `clv_prob`, `clv_pts`, `beat_close`.

Where to look: `[capture-close] {...}` summary lines in each run log carry credits used and `x-requests-remaining`. Idle runs (no kickoff in range) make zero API calls. Failed captures are retried by the next run; a missed game shows up as a gap in the gate report, not as a failed job.

Not covered (accepted): postponed or cancelled games never grade; a game flexed after the generators ran can miss the prefilter until the next generator run; MLB regular-season props are not budgeted here.

## 7. First Tuesday gate report (Oct 20, 15:47 UTC)

`report-ledger-gate` commits `docs/superpowers/reports/2026-10-20-ledger-gate.md` (pull to see it). The first run only establishes the audit baseline. Per weekend it checks:

- at least 95% of graded games have a game-lines close within 30 minutes of kickoff;
- at least 90% of graded prop picks have a Pinnacle or 3-book close at their exact line;
- zero ledger rows updated or deleted.

Step 1 passes after two consecutive weekends meeting every check, so the earliest pass is the Oct 27 report if both Oct 17-19 and Oct 24-26 pass. A miss blocks steps 2 and 3 of the profit roadmap.

## Rollback

Disable the four workflows in the Actions tab (or revert the merge). The writers' ledger hooks are best-effort and never fail a producer. The tables can stay; they are append-only and unread by anything else.
