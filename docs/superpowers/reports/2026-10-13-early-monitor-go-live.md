# Early-week monitor: go-live checklist (2026-10-13)

Branch `feat/early-monitor`. Everything below is for you to run; nothing here has been executed. No migration was run, nothing was pushed, no workflow was dispatched, and no code ever connected to the database or the Odds API during the build.

Until step 1 is done every early-monitor job prints a `::warning::` and exits 0, and nothing existing changes. `capture-close` ignores `early_tickets` until the table exists. Step 1 is safe to do before or after the merge.

## 1. Run the migration

Paste `db/migration_early_tickets.sql` into the Supabase SQL editor and run it once (idempotent, safe to re-run). It creates `early_tickets` (append-only: trigger plus revoked privileges) and `early_ticket_results`.

Optional append-only check (rolls back; run it yourself in the SQL editor):

```sql
BEGIN;
INSERT INTO early_tickets (sport, game_pk, market, side, book, book_line, price, pinnacle_line, pinnacle_fair,
                           p_win, p_push, p_loss, ev, observed_at, commence_time, hours_to_kick, poll_run_id)
VALUES ('cfb', 1, 'total', 'over', 'draftkings', 54.5, -110, 55.5, 0.5, 0.53, 0, 0.47, 0.016,
        now(), now() + interval '3 days', 72, 'manual');
UPDATE early_tickets SET price = 100;   -- must fail: early_tickets is append-only: UPDATE is not allowed
ROLLBACK;
```

## 2. Check the credit balance before merging

To see your balance: open The Odds API account dashboard, or look for the `credits remaining` line in the log of any capture-odds run (GitHub, Actions). The plan has 30,000 credits a month, reset on the 1st. On 2026-10-13 about 5,700 were left for October. The monitor costs about 3 credits a poll, about 180 a week, and stops itself below 1,500 remaining (the regular prop pulls stop at 2,500, regular game lines at 800, close-capture props at 600, all close capture at 200). With the regular ingest and the close capture also spending, expect the monitor to be paused by its floor, with `::warning::early-monitor: ... skipping its remaining calls`, in the last days of October. That is the floors working as designed, not a fault; polls resume after the reset on the 1st. If you would rather start from a full balance, merge on or after Nov 1.

## 3. Merge and push

Push `feat/early-monitor`, open the PR, merge to `main`. Schedules begin on merge:

| Workflow | Schedule (UTC) |
|---|---|
| `early-monitor` | hourly at minute 0, Sunday 16:00 to Wednesday 03:00, months Aug to Feb |
| `grade-early-tickets` | daily 14:20 |
| `report-early-tickets` | Tuesday 15:52 |

The first scheduled poll is the first Sunday at 16:00 UTC after the merge (that is noon ET until Nov 1, and 11am ET after).

Existing workflows are unchanged except that `capture-close` now also treats games that hold an early-week ticket as candidates, so every ticketed game gets a pre-kick line capture (it reads `early_tickets` only once the migration has run).

## 4. Dry run (REQUIRED before the first Sunday)

Do step 1 (the migration) first: the dry run stops with a "table missing" warning until it has been run.

The dry run costs about 3 Odds API credits (one CFB call of 2, one NFL call of 1), writes nothing, and prints the paper tickets it would have logged. It must run from `main`, so only after the merge:

```
gh workflow run early-monitor.yml --ref main -f dry_run=true
```

This is a required check, not an option. In the run log find the two lines that look like `[early-monitor] cfb books seen: [...]; missing: [...]` (one for CFB, one for NFL). Confirm that:

- `books seen` lists all 10 keys (draftkings, fanduel, betmgm, williamhill_us, espnbet, fanatics, hardrockbet, ballybet, betrivers, pinnacle), and
- `pinnacle` is among them. Without Pinnacle there is nothing to compare against and no ticket can ever be logged.

If any key is missing, for example `espnbet` (ESPN BET may have rebranded) or `williamhill_us`, tell Claude before Sunday so the book list can be fixed. A book that is missing is simply never ticketed, so the test would quietly run on fewer books.

Also look for the `[early-monitor] DRY RUN would log: ...` lines and the summary line (`events`, `matched`, `rows`, `tickets` per sport). A `matched` count far below `events` means the ESPN matcher is missing games; those games produce no tickets (conservative).

## 5. Storage

Each poll stores every price it saw in `odds_snapshot`, in the same row shape the regular ingest writes: roughly 3,000 to 4,000 rows per poll (a CFB week of 60 to 100 events times spreads and totals times up to 10 books, plus NFL), so about 200,000 rows a week while the season runs. Check your Supabase database size after the first week; if it is heading toward your plan's limit, decide whether to prune old `odds_snapshot` rows or to limit what the monitor stores (a small change to `scripts/early_monitor.py`).

## 6. First week (Sunday after the merge)

- After each poll: `early_tickets` gains zero or a few rows. Look at them:

```sql
SELECT sport, market, side, book, book_line, price, round(ev::numeric, 4) AS ev, hours_to_kick, observed_at
FROM early_tickets ORDER BY id DESC LIMIT 20;
```

- Games with a ticket get a `close_capture` lines row 5 to 25 minutes before kickoff (`SELECT * FROM close_capture WHERE game_pk IN (SELECT game_pk FROM early_tickets)`).
- The day after the games, `grade-early-tickets` writes `early_ticket_results`:

```sql
SELECT close_kind, count(*) FROM early_ticket_results GROUP BY 1;
```

  A large `missing` share means Pinnacle often has no two-way quote in the close capture; the weekly report states the share. A `missing` close can also mean the close capture skipped that game altogether (credits were low, or its 5 to 25 minute window was missed), not only that Pinnacle had no quote.
- The weekly report is committed to the repo, and appears on GitHub under `docs/superpowers/reports/` (the first one the Tuesday after, named `YYYY-MM-DD-early-monitor.md`). Pull `main` to see it on your computer. It says COLLECTING until there are 150 graded tickets with a Pinnacle close, 60 of them CFB totals. Nothing can pass or fail before that.

## 7. What the result means

The test is judged once, at the first report with at least 150 graded tickets (Pinnacle close) of which at least 60 are CFB totals. CFB spreads and NFL spreads each need at least 40 graded tickets at the moment of judgement, otherwise that market is "insufficient". Because the test is judged once, a market that is insufficient at that moment is not re-judged later. Each market is judged on its own: mean close EV at least +1.5%, game-clustered 95% lower bound above 0, beat-close rate at least 55%, and a positive mean close EV where the number was still available a poll later. A market that passes goes to a separate short design for live alerts at small stakes (a later profit check, at least 300 tickets with the ROI 95% lower bound above -3%, decides any stake increase). A market that fails is dropped. If the minimum is not reached by the end of the CFB bowl season, collection continues next season. Realized ROI is reported, never gated.

Tickets on games with no model final in `prediction_accuracy` (for example an FBS team against an FCS team, which the CFB model skips) never get a final score from the grader. The weekly report counts them as stale-ungraded tickets (still ungraded 48 hours after kickoff); they do not count toward the test.

## 8. Turning it off (rollback)

Disable the `early-monitor` workflow in GitHub (Actions, early-monitor, the three-dot menu, Disable workflow), or `gh workflow disable early-monitor.yml`. That stops new tickets and stops all credit spend. `grade-early-tickets` and `report-early-tickets` keep running harmlessly (they only read stored rows and spend no credits); disable them the same way to stop everything. The data stays: the tables are additive and harmless to leave, and `early_tickets` is append-only by design, so only the table owner can drop it.

## 9. Good to know

- The monitor's paid odds calls are single-attempt: there are no retries, because a retry after a billed timeout would bill again. A failed poll is simply skipped and the next hourly run goes on normally.
- A red `early-monitor` run means every poll in that run failed (both sports). Check the `ODDS_API_KEY` secret and the `books seen` line in the log. If only one sport fails, or the credit floor skips sports, the run stays green.
