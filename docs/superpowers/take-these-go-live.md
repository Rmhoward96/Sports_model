# Take these: go-live checklist

Spec and plan: `docs/superpowers/specs/` and `docs/superpowers/plans/` (2026-10-17 take-these). Do these in order.

1. **Apply the migration.** Paste `db/migration_take_these.sql` into the Supabase SQL editor. It has never been applied.
   Caution: if an earlier version of this file was ever pasted (before the `feat/take-these` branch), `take_verdict.mean_clv`
   and `take_verdict.lower` are still NOT NULL there. Run
   `ALTER TABLE take_verdict ALTER COLUMN mean_clv DROP NOT NULL, ALTER COLUMN lower DROP NOT NULL;` (the CREATE TABLE
   here has them nullable, and `IF NOT EXISTS` will not change an existing table).

2. **Dry-run the pull.** Actions, "take-these-pull", Run workflow, `dry_run` = true. It makes the paid odds calls (about 3
   credits per in-season league, at most 24) and writes nothing. Check the "books seen / missing" lines. The list runs on 7
   books (DraftKings, FanDuel, BetMGM, Caesars, Fanatics, Hard Rock, Bally Bet) plus Pinnacle as the reference. Bet365 is
   not offered by the Odds API for these leagues (checked 2026-10-17: only `bet365_au`, so there is no US Bet365).
   The close and scores workflows also take a `dry_run` input (paid calls, no writes) if you want to check them too.

3. **Check the sport keys.** The dry run calls the free `/sports` list and prints any of the 8 sport keys (nfl, cfb, mlb,
   nhl, nba, wnba, epl, mls) it does not list. Expect none for the leagues in season; an out-of-season league may be absent.

4. **Turn it on.** Set repo variable `TAKE_THESE` = `on`. Optionally set `TAKE_PULL_EVERY_HOURS` (default 2; the pull
   script only runs on hours divisible by it). Credits are short in October: leave `TAKE_THESE` off and run dispatches
   by hand instead if the balance is tight. The scheduled jobs are gated on `vars.TAKE_THESE == 'on'`; a manual
   `workflow_dispatch` always runs.

   | Workflow | Schedule (UTC) | Credit floor | Gate |
   |---|---|---|---|
   | `take-these-pull.yml` | `0 12-23,0-4 * * *` (every 2 hours by default) | skips below 1,500 remaining | `TAKE_THESE` |
   | `take-these-close.yml` | `3-59/10 * * * *` (every 10 min) | skips paid calls below 300 remaining | `TAKE_THESE` |
   | `take-these-scores.yml` | `30 9 * * *` (daily) | skips below 300 remaining | `TAKE_THESE` |
   | `report-take-these.yml` | `0 13 * * 1` (Mondays) | none (database only) | none; always runs on schedule |

   The weekly report writes `docs/superpowers/reports/YYYY-MM-DD-take-these.md` and commits it. With no picks yet it writes
   nothing.

5. **Deploy the site.** Cache key is `20261017a`. `take.html` shows "Couldn't load data" until the migration has run.
   `scorecard.html` (browser-only Pikkit CSV scorecard) works without the migration; its side-by-side comparison shows
   "list data unavailable" until the list has data.

6. **Test timing.** October is a shakedown and is not judged. The CLV test counts picks from 2026-11-01 and is judged once,
   at 300 graded picks. Pass criteria: beat-close rate >= 55%, mean CLV > 0, and a cluster-bootstrap lower bound > 0. The
   verdict row is append-only.

7. **Rollback.** Set `TAKE_THESE` to `off` (or delete it), then disable all four workflows in the Actions tab:
   `take-these-pull`, `take-these-close`, `take-these-scores` and `report-take-these`. The report has no `TAKE_THESE` gate,
   so disabling it is the only way to stop it. To drop the data, use the commented rollback block at the top of
   `db/migration_take_these.sql`.
