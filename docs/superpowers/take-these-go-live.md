# Take these: go-live checklist

Spec and plan: `docs/superpowers/specs/` and `docs/superpowers/plans/` (2026-10-17 take-these). Do these in order.

1. **Apply the migration.** Paste `db/migration_take_these.sql` into the Supabase SQL editor. It has never been applied.
   Caution: if an earlier version of this file was ever pasted (before the `feat/take-these` branch), `take_verdict.mean_clv`
   and `take_verdict.lower` are still NOT NULL there. Run
   `ALTER TABLE take_verdict ALTER COLUMN mean_clv DROP NOT NULL, ALTER COLUMN lower DROP NOT NULL;` (the CREATE TABLE
   here has them nullable, and `IF NOT EXISTS` will not change an existing table).

2. **Dry-run the pull.** Actions, "take-these-pull", Run workflow, `dry_run` = true. (A manual run only: the pull and the
   close no longer have their own schedules. See the table in step 4.) It makes the paid odds calls (about 3
   credits per in-season league, at most 24) and writes nothing. Check the "books seen / missing" lines. The list runs on 7
   books (DraftKings, FanDuel, BetMGM, Caesars, Fanatics, Hard Rock, Bally Bet) plus Pinnacle as the reference. Bet365 is
   not offered by the Odds API for these leagues (checked 2026-10-17: only `bet365_au`, so there is no US Bet365).
   The close, scores and watch workflows also take a `dry_run` input (paid calls, no writes) if you want to check them too.
   A watch dry run stays alive for 340 minutes: cancel it in the Actions tab once you have seen a tick or two.

3. **Check the sport keys.** The dry run calls the free `/sports` list and prints any of the 8 sport keys (nfl, cfb, mlb,
   nhl, nba, wnba, epl, mls) it does not list. Expect none for the leagues in season; an out-of-season league may be absent.

4. **Turn it on.** Set repo variable `TAKE_THESE` = `on`. Optionally set `TAKE_PULL_EVERY_HOURS` (default 2; the pull
   script only runs on hours divisible by it). Credits are short in October: leave `TAKE_THESE` off and run dispatches
   by hand instead if the balance is tight. The scheduled jobs are gated on `vars.TAKE_THESE == 'on'`; a manual
   `workflow_dispatch` always runs.

   GitHub cron is best-effort (the pull ran 6 times, 15-55 minutes late, and the close never landed inside a game's
   pre-start window on 2026-10-18), so ONE long-running job, `take-these-watch`, does the pull and the close. It starts
   every 2 hours, stays alive 340 minutes, and every 5 minutes runs the close and then the pull when due (pull hours
   12-23 and 0-4 UTC, at least `TAKE_PULL_EVERY_HOURS` since the last logged pull). The pull and close workflows keep only
   `workflow_dispatch` for manual runs. No workflow starts or restarts another; the concurrency group runs one watcher at a
   time and queues at most one more, which starts the moment the running one ends. Closes target picks starting in
   5-12 minutes.

   | Workflow | Schedule (UTC) | Credit floor | Gate |
   |---|---|---|---|
   | `take-these-watch.yml` | `7 */2 * * *`, runs 340 min each; ticks every 5 min (close every tick, pull every 2 hours in pull hours) | pull skips below 1,500 remaining, close paid calls below 300 | `TAKE_THESE` |
   | `take-these-pull.yml` | none (manual `workflow_dispatch` only) | skips below 1,500 remaining | none (manual) |
   | `take-these-close.yml` | none (manual `workflow_dispatch` only) | skips paid calls below 300 remaining | none (manual) |
   | `take-these-scores.yml` | `30 9 * * *` (daily) | skips below 300 remaining | `TAKE_THESE` |
   | `report-take-these.yml` | `0 13 * * 1` (Mondays) | none (database only) | `TAKE_THESE` |

   The weekly report writes `docs/superpowers/reports/YYYY-MM-DD-take-these.md` and commits it. With no picks yet it writes
   nothing.

5. **Deploy the site.** Cache key is `20261017a`. `take.html` shows "Couldn't load data" until the migration has run.
   `scorecard.html` (browser-only Pikkit CSV scorecard) works without the migration; its side-by-side comparison shows
   "list data unavailable" until the list has data.

6. **Test timing.** October is a shakedown and is not judged. The CLV test counts picks from 2026-11-01 and is judged once,
   at 300 graded picks. Pass criteria: beat-close rate >= 55%, mean CLV > 0, and a cluster-bootstrap lower bound > 0. The
   verdict row is append-only.

7. **Rollback.** Set `TAKE_THESE` to `off` (or delete it), then CANCEL the running `take-these-watch` run (and any pending
   one) in the Actions tab: setting the variable only stops new scheduled starts, the run in progress keeps going until its
   340-minute deadline. Then disable all five workflows: `take-these-watch`, `take-these-pull`, `take-these-close`,
   `take-these-scores` and `report-take-these`. The watch, scores and report schedules are gated on `TAKE_THESE`, so
   setting it to `off` already stops new scheduled runs; disabling them also stops manual runs. To go back to the old cron
   timing instead (not recommended: it drops and delays runs), restore the `schedule:` blocks of `take-these-pull.yml`
   (`0 12-23,0-4 * * *`, with `--scheduled`) and `take-these-close.yml` (`3-59/10 * * * *`) from git history and disable the
   watcher. To drop the data,
   use the commented rollback block at the top of `db/migration_take_these.sql`.
