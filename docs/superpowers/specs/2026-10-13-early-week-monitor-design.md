# Early-week stale-number monitor (profit project, step 3) — design spec

Date: 2026-10-13 · Status: design approved in chat (three sections); awaiting user review of this file
Evidence: `docs/superpowers/reports/2026-10-13-edge-audit.md` §5 and §6 item 3 (the only positive lead: early CFB totals at +2.9% EV vs Pinnacle valid close, 95% CI [+1.0%, +4.9%], n = 57, unproven). Depends on step 1 (`docs/superpowers/specs/2026-10-13-pick-ledger-close-capture-design.md`, merged 2d16461) for the pre-kick close capture.

## Goal

Find out, with no money at risk, whether early-week numbers at the user's sportsbooks can be bet at prices that beat Pinnacle's pre-kick close. The system logs paper tickets, never edits them, grades them against the close, and judges them once against a pass test that was fixed before any results came in.

## User decisions (2026-10-13)

- Books: all the user's funded US books, which are DraftKings, FanDuel, BetMGM, Caesars, ESPN BET, Fanatics, Hard Rock, Bally and BetRivers.
- Reaction: if this ever goes live, the user would act on fast alerts within about 30 minutes. There are no alerts during the paper test. A ticket counts at the price first seen, and whether the number was still available an hour later is recorded and gated (see §4).
- Approach 1: a dedicated hourly early-week poll with its own tickets table. This rejected two alternatives: reusing the existing sparse pulls, and putting tickets into `pick_ledger`.
- Flag threshold: **+1.5% EV** (raised by the user from the audit's 1.0%).
- Odds API: 30,000 credits/month, reset on the 1st. On 2026-10-13 about 5,700 were left for October.

## 1. Flag rule and ticket valuation

**Markets in the test:**
- CFB totals, CFB spreads and NFL spreads (the audit's pre-registered set).
- NFL totals and all moneylines are excluded.

**Reference price:**
- Pinnacle's two-way no-vig fair probability at Pinnacle's current line, from the **same poll**.
- No Pinnacle two-way quote for a game and market in that poll means no tickets for that game and market from that poll.
- Pinnacle fair outside [0.15, 0.85] means no ticket.

**Landing tables (line adjustment).** These are committed JSON assets under `assets/early/`, built once by `scripts/build_landing_tables.py` from history:
- **CFB:** `assets/cfb/lines.parquet` closing `market_spread` / `market_total`, 2015–2025, joined to final scores from `assets/cfb/schedules.parquet`.
- **NFL:** `assets/nfl/schedules.parquet` `spread_line` / `total_line` and scores, 2010–2025.
- **Result variable:**
  - spreads use the home margin M;
  - totals use the total points T.
- **Table contents:** for each sport and market, and for each line value L on a half-point grid over the observed range, the empirical probability of each integer result value.
  - The sample is games whose closing line is within ±1.0 point of L.
  - It widens to ±2.0 if that gives fewer than 300 games.
- **Effect:** the tables keep key-number mass (NFL margins 3, 7, 10, 14; common CFB totals) and the push mass on whole-number lines.

**Line adjustment from Pinnacle's line Lp to the book's line Lb, for one side:**
1. Take the landing table at Lp.
2. Pinnacle's fair is conditional on no push: P_win(Lp) = fair × (1 − P_push(Lp)), with P_push from the table (0 on half-point lines).
3. Move to Lb by adding or removing the table mass of every result value between the two lines:
   - mass at Lp's push value becomes win or loss according to the direction of the move;
   - mass at Lb's push value becomes P_push(Lb).
4. That gives P_win(Lb), P_push(Lb) and P_loss(Lb).
5. If |Lb − Lp| > 3.0 points, there is no ticket (likely a data error, or too much reliance on the table).

**EV at the book's price:**
- Formula: EV = P_win × (decimal − 1) − P_loss. A push returns the stake and contributes 0.
- **Flag:** EV ≥ **+0.015**, and kickoff ≥ 24 h after the poll.
- If several of the user's books qualify for the same side, the ticket goes to the highest-EV book.
- Ties are broken by book name, so results are deterministic.

**Ticket rules:**
- **One ticket per (sport, game_pk, market, side)**, logged the first time it is flagged.
- Later sightings never create a ticket. The weekly report may compute "best EV seen" from snapshots, but it is not stored on the ticket.
- Each ticket stores:
  - sport, game_pk, market, side;
  - book, book line, American price;
  - Pinnacle line, Pinnacle fair, P_win, P_push, P_loss and EV;
  - observed_at, commence_time, hours_to_kick;
  - poll run id.
- Game matching uses the existing odds-event to game_pk matchers that `scripts/ingest_odds.py` uses.
- `early_tickets` is append-only, enforced by a DB trigger and revokes, the same pattern as `pick_ledger`.

## 2. Polling and credits

**Schedule.** New `scripts/early_monitor.py` and workflow `early-monitor.yml`:
- Hourly at minute 0, from Sunday 16:00 UTC to Wednesday 04:00 UTC. That is 60 polls a week, every week of the football season (Aug–Jan).
- Crons: `0 16-23 * 1,2,8-12 0`, `0 * * 1,2,8-12 1,2` and `0 0-3 * 1,2,8-12 3`.
- The workflow also has a `workflow_dispatch` with a `dry_run` input, which is passed through `env`.

**Calls per poll.** One Odds API odds call per sport, using the `bookmakers` parameter rather than `regions`. The 9 US books plus `pinnacle` make 10 books, which bills as one region.
- CFB: `spreads,totals`, 2 credits.
- NFL: `spreads`, 1 credit.
- **About 3 credits per poll, about 180 per week.**

**Credit floor.**
- The monitor skips its remaining calls when `x-requests-remaining` < **1,500**, and warns. The value is unknown before the first response, so the first call always runs.
- The full ordering as credits fall:

  | Remaining | Stops |
  |---|---|
  | < 2,500 | regular prop pulls |
  | < 1,500 | this monitor |
  | < 800 | regular game-line pulls |
  | < 600 | close-capture props |
  | < 200 | all close capture |

**Storage.**
- Each poll's prices are written to the existing `odds_snapshot` table, in the same row shape `ingest_odds` writes.
- The monitor then computes flags and appends new tickets.
- Tickets are written incrementally (after each sport), so a failure part-way through keeps what was already found.

**Failure handling.**
- A failed poll is not retried. The next hourly run proceeds normally, with no catch-up burst.
- Missing tables: warn and exit 0, using the same missing-table guard as the ledger jobs.
- Under the test guard `SM_LEDGER_OFF`, the job warns and exits 0, dry-run included.
- It writes only `odds_snapshot`, `early_tickets` and `early_ticket_results`. It never touches boards, picks, models or the site.

**Close coverage.** The close-capture candidate query (`scripts/capture_close.py`) adds `early_tickets` to its sources, so every ticketed game gets a pre-kick line capture.

## 3. Grading

New `scripts/grade_early_tickets.py`, run by `grade-early-tickets.yml` daily at 14:20 UTC. It upserts `early_ticket_results`, one row per ticket. This table is written by the grader, not append-only, matching `pick_ledger_results`.

**Close.** Pinnacle's two-way no-vig fair from the `close_capture` lines snapshot for the game (5–25 min pre-kick).
- If Pinnacle is absent at the close, `close_kind = 'missing'`. The ticket is excluded from the test, and the share of such tickets is reported.

**Metrics:**
- **`close_ev`** (primary): the ticket's book line and price, valued at Pinnacle's closing fair through the same landing-table adjustment and EV formula as §1.
- `beat_close`: `close_ev` > 0.
- `clv_pts`: line points gained versus Pinnacle's closing line, signed in the ticket side's favour.
- `result` (W/L/P) from finals keyed by (sport, game_pk), using the same final providers as `grade_ledger`.
- `units` at the ticket's price.
- **`available_next_poll`:** whether the same book offered the same line at the same or a better price in the next monitor poll. That poll must be 30 min to 3 h after the ticket; otherwise the value is NULL. It is computed from `odds_snapshot`.

Postponed or uncaptured games stay ungraded and are reported.

## 4. Pass test (pre-registered; frozen as of this spec)

`scripts/report_early_tickets.py`, run by `report-early-tickets.yml` on Tue 15:52 UTC, commits a markdown report to `docs/superpowers/reports/`.
- **Weekly content:**
  - tickets by market;
  - close coverage;
  - running mean `close_ev`, beat-close rate and `available_next_poll` share;
  - W/L/P and units;
  - the decision status.

**Judgement point.** The test is judged **once**, at the first weekly report where both of these hold:
- there are ≥ 150 graded tickets with a Pinnacle close across the three markets;
- ≥ 60 of them are CFB totals.

Before that, the status is "collecting", and running numbers cannot trigger a pass or a fail. If the minimum is not reached by the end of the CFB bowl season, collection continues next season.

**Per-market pass.** Each market is judged separately at the judgement point. CFB spreads and NFL spreads need ≥ 40 graded tickets; otherwise they are marked "insufficient". A market passes only if **all** of these hold:
1. mean `close_ev` ≥ **+0.015**;
2. game-clustered 95% lower bound of mean `close_ev` > **0**. This uses a cluster bootstrap over games: 10,000 resamples, fixed seed 20261013, 2.5th percentile;
3. `beat_close` rate ≥ **55%**;
4. mean `close_ev` over tickets with `available_next_poll = true` > **0**.

**What the result means:**
- **Realized ROI is reported, never gated**, at this stage.
- **A market that passes** goes to a separate short design for live alerts at small stakes. A later profit check (≥ 300 tickets, ROI 95% lower bound > −3%) decides any stake increase.
- **A market that fails** is dropped.

## Migration (user runs)

`db/migration_early_tickets.sql` is idempotent and additive only:
- **`early_tickets`:**
  - bigserial id;
  - unique (sport, game_pk, market, side);
  - append-only trigger on UPDATE, DELETE and TRUNCATE, with `SET search_path = ''`.
- **`early_ticket_results`:** PK ticket_id.
- **Access:** RLS with anon SELECT only. Writes are revoked from anon, authenticated, PUBLIC and service_role on `early_tickets`, as in `db/migration_pick_ledger.sql`.

## Testing

Network-free and DB-free unit tests, plus the existing suite staying green:
- **Landing tables:** pmf sums to 1, key numbers kept, ±1.0 to ±2.0 widening, push mass on whole lines.
- **Line adjustment:**
  - same line gives an identity;
  - a half-point across a key number moves win probability by the table mass;
  - push conversion;
  - the 3-point cap.
- **EV and flags:**
  - EV formula, with a push contributing 0;
  - the 1.5% threshold boundary;
  - the 24 h rule;
  - the Pinnacle-missing skip;
  - the fair-range filter;
  - best-book choice and deterministic ties;
  - one ticket per side across polls.
- **Polling:** the `bookmakers` parameter list (10 books) and cost per poll.
- **Credit floor and guards:**
  - 1,499 / 1,500 boundary and unknown-first-call behaviour;
  - the floor-ordering assertion against the ingest and close floors;
  - `SM_LEDGER_OFF` and missing-table exits.
- **Writes:** incremental ticket writes.
- **Grading:**
  - close EV;
  - missing close;
  - `available_next_poll` (same or better price, 30 min–3 h window, NULL outside it);
  - `clv_pts` sign;
  - pushes.
- **Pass logic:**
  - collecting versus judged;
  - per-market minimums;
  - each of the four criteria;
  - the cluster bootstrap with a fixed seed.
- **Schedules:** workflow crons and the env-passed dry run.

## Out of scope

- Alerts and notifications.
- Any change to models, boards, picks or the site.
- NFL totals, moneylines and props.
- Steps 2 (probability shrink) and any bet sizing.

## Risks

- **Execution realism.** Paper prices may not be bettable: limits, fast moves, geolocation. The `available_next_poll` criterion and the later live-alert design address this.
- **Accuracy of the landing tables.** History closes come from one consensus source (CFB) or nflverse (NFL). The ±1.0/±2.0 windows and the 3-point cap limit extrapolation.
- **Sample time.** At 1.5%, CFB totals may take until bowl season or next season to reach 60. The test waits rather than judging early.
- **Pinnacle coverage.** Small CFB games may lack early Pinnacle lines, so those games produce no tickets, which is conservative.
- **Account limits.** A passing market can still be limited by books. That is outside what this test can measure.
