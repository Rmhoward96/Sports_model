# "Take these" list, Pikkit scorecard and CLV test — design (2026-10-17)

## Why

The lean backtest (`docs/superpowers/reports/2026-10-16-lean-backtest.md`) found that the models add no information beyond the closing line in any sport. The one edge that is still realistic for a single bettor with 9 US books is **price**: a soft book posting a number better than the sharp market's fair price. Pinnacle is the sharp reference.

The user bets a couple of hundred straights and parlays a week (0.25u–4u), tracked in Pikkit. They shop lines, try to bet at least a day before the game, and place quickly once they decide. Picking is intuition, a little research, and sometimes tailing pros. Their goal is consistent, profitable weeks, about +25u a month. They are on a 5-day losing streak and want to know whether it is variance or something deeper. In the Pikkit export (382 bets, Sep 1 – Oct 16), their beat-the-close rate fell from about 44% to 28% after Oct 5.

What is missing is **which bets to take**. This project gives them:

1. a list of bets where a book beats Pinnacle's fair price right now;
2. a scorecard that tells them, from their own Pikkit export, whether their prices are beating the close and where they are slipping;
3. a pre-registered test that says whether the list's picks actually beat the close.

**Success** is that the list's picks beat Pinnacle's close often enough to pass the test (section 3), and that the user can see on the scorecard when their own beat-close rate drops, and in which slice.

## Goals

- **G1 — "Take these" page:** a new page on the existing site (CappingAlpha, `site/`).
  - Leagues: NFL, CFB, MLB, NHL, NBA, WNBA, EPL, MLS.
  - Markets: moneyline, spread and total only.
- **G2 — Scorecard page:** the user drops a Pikkit CSV export and sees:
  - beat-close rate and average price vs. the close, weekly and rolling;
  - breakdowns by league, book, market, bet type, parlay legs, stake size and Pikkit tag;
  - an alert when the rolling beat-close rate falls below 40%, naming the slices that fell.
- **G3 — Test:** every list pick is graded against Pinnacle's close. The list is judged once, at 300 graded picks, by the rule in section 3. The user's own bets are shown side by side over the same period.

## Non-goals

- **No player props in this project.** The user said: "Don't worry about props then. Lets test with ML, Spread, Total first." College props are never in scope; the user cannot bet them.
- **Model leans are never picks.** For NFL, CFB and MLB, the model's lean may appear on a pick as context ("model agrees" / "model disagrees"). It never creates, removes or ranks a pick.
- **Parlays are never recommended.** They are tracked on the scorecard only.
- **No stake sizing.** Stakes stay the user's decision. The list shows edge, not units.
- No change to the existing +EV board, early-week monitor, ledger, shrink or Track Record. This is a separate path that may reuse their code.
- No relabelling of existing pages or features. The user said: "don't relabel anything. It's just me using this tool."

## Anti-goals (a failure even if it works)

- **Fake edges from stale prices.** An edge built on a Pinnacle price that has not updated, or on a soft price from an earlier pull, is noise. Freshness rules are mandatory (section 1).
- **Moving the goalposts.** The test's numbers are fixed now and judged once.
- **Running the Odds API dry.** The list must not starve the existing close capture and ingest jobs (section 4).
- **Personal betting data leaving the browser.** The Pikkit CSV is parsed in the browser and is not uploaded anywhere (section 2).

---

## 1. The "take these" list

### What a pick is

A **pick** is a (game, market, side) where one of the user's 9 books offers a price whose expected value against Pinnacle's no-vig fair probability, **taken from the same pull**, is at least the threshold:

- Edge (EV) = `fair_prob × decimal(book_price) − 1`.
- The edge must be **≥ 1.5%**. *My call:* the user raised the early-week monitor's flag from 1.0% to 1.5%, which is the clearest signal of their preference.

**The user's 9 books:** DraftKings, FanDuel, BetMGM, Caesars, ESPN BET, Fanatics, Hard Rock, Bally and BetRivers. Their Odds API keys are in `board.USER_BOOKS`: draftkings, fanduel, betmgm, williamhill_us, espnbet, fanatics, hardrockbet, ballybet, betrivers.
- The builder must confirm that each key returns data for each league.
- ESPN BET may have been rebranded in the Odds API feed; flag any key that never appears.
- A book that is missing simply contributes no prices. It is not an error.

**Fair price:**
- Pinnacle's no-vig probability, using the **power** de-vig (`board.novig_power`).
- **Soccer moneylines are 3-way** (home / draw / away). They need an n-way power de-vig: find k such that Σ pᵢᵏ = 1. `novig_power` is 2-way only, so add an n-way function beside it. Do not change the existing function's behaviour.
- Draw is a pickable side.

**Lines:**
- **Moneyline:** always comparable.
- **Spread and total, same line:** the book's line equals Pinnacle's line, and the price is compared directly.
- **Spread and total, better line:** *my call.* The book's line is better for the bettor than Pinnacle's (e.g. +3.5 vs +3, or under 45.5 vs under 45), and the book's price still gives ≥ 1.5% EV against Pinnacle's fair probability at Pinnacle's own line. This edge is a **lower bound**, and the pick is marked "better line".
- **Any other line difference:** no pick. There are no line-conversion tables for most of these leagues.

**Spread signs:** a minus spread is the favourite and a plus spread is the underdog, as the user stated. The Odds API's per-team `point` already follows this.

**Price range:** *my call.* Only prices from −250 to +500 are eligible. −250 matches ev_pilot's `MAX_STRAIGHT_JUICE`. The +500 cap is there because de-vig error on longshots manufactures edges.

### Freshness rules (mandatory)

- **Same pull:** Pinnacle and the book must come from the same API response. One bulk call per league returns both.
- **Pinnacle staleness:** *my call.* Pinnacle's `last_update` for that market must be within 30 minutes of the pull time. Otherwise the market makes no picks in that pull.
- **Game started:** no picks once the game has started.

### What each pick shows

| Field | Meaning |
|---|---|
| League, matchup, start time | Start time is shown as both a countdown and a local time |
| Bet | e.g. "Chiefs −3.5", "Over 8.5", "Arsenal ML", "Draw" |
| Book and price | The best eligible book. Other books that also clear 1.5% are listed smaller |
| Fair price | Pinnacle's no-vig price, in American odds |
| Edge % | As defined above, marked "≥" for better-line picks |
| Good until | The worst price at this line that still clears 1.5% (`decimal = 1.015 / fair_prob`, rounded toward the safe side) |
| Found / checked | When it was first seen and when it was last confirmed |
| Model (NFL, CFB, MLB) | Context only: the model's lean on this market, if any |

**Pick lifecycle:**
- A pick stays on the list while each pull still confirms it, with its price and edge updated.
- It drops off when a pull no longer finds it, or when the game starts.
- The page also shows the last 24 hours of dropped picks, greyed out, so the user can see what moved.

**Test identity:** *my call.* For the test, a pick is counted once per (event, market, side). It is recorded at its **first** appearance: that book, that price and that line. Later re-flags of the same side do not add picks.

### Pulls

- **One bulk odds call per league per pull:** `markets=h2h,spreads,totals`, `bookmakers=` the 9 books plus pinnacle. Ten bookmakers are billed as one region, so a pull costs **3 credits**. This is the same billing trick as `early/polling.py`; reuse its helpers.
- **Odds API sport keys:** americanfootball_nfl, americanfootball_ncaaf, baseball_mlb, icehockey_nhl, basketball_nba, basketball_wnba, soccer_epl, soccer_usa_mls. The builder verifies them against the free `/sports` endpoint.
- **Skip leagues with no games:** before buying odds, check the free `/events` endpoint and skip any league with no events in the next 7 days. Out-of-season leagues cost nothing.
- **Cadence:** *my call.* Every 2 hours from 12:00 to 04:00 UTC, which is 9 pulls a day. It runs in a new GitHub Actions workflow, and the cadence is a repo variable.
- **Game identity:** the Odds API event id. Leagues other than NFL, CFB and MLB have no `game_pk`. For NFL, CFB and MLB, the event is matched to `game_pk` with the existing team-name matcher only for the model-context field. A failed match just means no model context.
- **Storage:** new tables keyed by event id. Do not overload `odds_snapshot`, which is keyed by `game_pk` and has no sport column. Schema is left to the builder. It needs events (league, teams, start time), every pulled price with its pull time, picks with first-seen and last-seen, Pinnacle closes, and graded results.

### Closing line and grading (for the test)

**Close capture:**
- A close job runs frequently (*my call:* every 10 minutes).
- For every league with an ungraded pick whose game starts within the next 5–20 minutes, it makes one bulk call with `bookmakers=pinnacle` and only the markets that hold such picks. That costs 1 credit per market.
- The close is Pinnacle's last price before the start.
- Model this on `scripts/capture_close.py` and reuse its window logic.
- GitHub cron can run late, so record the minutes-to-start of each close. A close taken after the start is not a close; that pick stays ungraded and is reported as "no close".

**Grading each pick:**
- **Close fair:** Pinnacle's no-vig closing probability for the pick's side, using the n-way de-vig for soccer moneylines.
- **Edge vs. close (CLV)** = `close_fair × decimal(pick_price) − 1`, at the pick's line.
- **Beat the close** = CLV > 0.
- **If Pinnacle's closing line differs from the pick's line** (spreads and totals): *my call.*
  - Beat-close is decided by direction. A move of ≥ 0.5 toward the pick (the pick got the better number) is a beat; a move away is a miss.
  - CLV is left blank, because there is no line-conversion table.
  - The report shows how many picks were graded this way.
- **Results:** won / lost / push and units at the pick price. *My call:* finals come from the Odds API `/scores` endpoint, once a day per league that has picks to settle, at about 2 credits a call. Results are shown but are **not** part of the verdict.

---

## 2. The scorecard (Pikkit CSV)

**Input:** the user's Pikkit export, the same format as `/Users/ryan/Downloads/transactions.csv`. Columns:
- bet_id, sportsbook, type (straight / parlay), status (SETTLED_WIN / SETTLED_LOSS / CASH_OUT / PUSH / VOID / PLACED);
- odds and closing_line, both in decimal; closing_line is sometimes blank;
- ev, amount, profit;
- time_placed_iso, time_settled_iso;
- bet_info (legs separated by " | ");
- tags, sports, leagues.

**Where it runs:** *my call.* On a new site page, entirely in the browser.
- The user drags the CSV in. It is parsed client-side and **never uploaded**. That keeps personal betting data off the server and avoids a public write policy on Supabase.
- The last import may be kept in `localStorage` as a convenience.
- Each new export contains the full history, so re-dropping the file rebuilds everything.

**Close source:** *my call.* Pikkit's own `closing_line` column.
- In the sample it is present for about 85% of straights (MLB 49/59, CFB 48/54, NFL 45/72, NHL 24/26) and 59 of 156 parlays.
- Matching Pikkit's free-text `bet_info` to our Pinnacle closes is brittle and is left out.
- Pikkit's close is not necessarily Pinnacle's, so the side-by-side with the list (section 3) carries that caveat on screen.

**Definitions:**
- **Beat the close:** `odds > closing_line`.
- **Price vs. close:** `odds / closing_line − 1`, which matches Pikkit's `ev` column.
- **What is excluded:** bets with no closing_line are left out of these two numbers but counted in volume and profit; VOID bets are left out entirely; PLACED (unsettled) bets count once they have a close.

**Views:**
- **Weekly:** Monday to Sunday, US Eastern. For each week: bets, bets with a close, beat-close rate, average price vs. close, units and profit.
- **Rolling:** *my call.* The last 100 bets that have a close, recomputed bet by bet, and shown as a line over time.

**Breakdowns** (the same metrics for each):
- League.
- Book, normalised. For example "Draftkings Sportsbook" becomes DraftKings.
- Market: moneyline, spread, total, prop or other, parsed from bet_info.
- Bet type: straight or parlay.
- Parlay legs: 1, 2, 3, 4 and 5+.
- Stake size, in units. *My call:* a unit-size input on the page, saved locally. Buckets are ≤0.5u, ≤1u, ≤2u and >2u.
- Pikkit tag. For example "tail-ProName" or "own". Untagged bets are shown as "untagged".

**Alert:** when the rolling-100 beat-close rate is below 40%, show a banner at the top of the page.
- It lists every slice with at least 15 bets in the rolling window whose own rate is below 40%, worst first.
- It shows each slice's rate in the prior 100 for comparison.
- *My call:* the alert appears when the user drops a file. There are no push notifications, because the data only arrives when they drop it.

---

## 3. The test (pre-registered, judged once)

- **Sample:** list picks as defined in section 1 (first appearance per event, market and side), graded against a valid Pinnacle close. Picks with no close are reported but not counted.
- **When:** the verdict is computed once, the first time **300 graded picks** exist across all leagues. The weekly report shows progress until then.
- **Pass** requires all three of the following. *My call:* these mirror the early-week monitor's frozen test.
  1. **Beat-close rate ≥ 55%.**
  2. **Mean CLV > 0**, over picks that have a CLV.
  3. **Cluster-bootstrap 95% lower bound of mean CLV > 0.** Clusters are events, with 10,000 resamples and seed 20261017.
- **Reported, not judged:** per-league and per-market breakdowns, results and units, the number of picks graded by line direction, and the user's own Pikkit beat-close rate over the same dates.
- **Where it shows:** the weekly report runs as a workflow and writes a markdown report, like `report-ledger-gate`. A small panel on the "take these" page shows progress and, once judged, the verdict.
- **After the verdict:** the list keeps running either way. What to do next is the user's call. The verdict is never recomputed with different numbers.

---

## 4. Credits

The user's Odds API plan is 30,000 credits a month, reset on the 1st.

**Estimate for this project:**
- **Pulls:** 3 credits × 9 pulls × in-season leagues. In November, with NFL, CFB, NHL, NBA, EPL and MLS in season, that is about 160 a day, or roughly 4,900 a month.
- **Closes:** about 1–3 credits per start window that holds picks, roughly 20 windows a day, which comes to about 1,000–1,800 a month.
- **Scores:** about 2 credits per league per day, roughly 400 a month.
- **Total:** about 6,000–7,000 credits a month.

**Guards:** *my call.*
- Pulls skip when fewer than 1,500 credits remain. Close calls skip below 300. Scores skip below 300.
- Every call logs the credits it used and the remaining balance.
- Existing floors are unchanged: ingest skips props below 2,500 and lines below 800; close capture skips props below 600 and everything below 200.
- October is tight, since only about 5,700 credits remained on Oct 13. The system must degrade by skipping, not by failing.
- Full operation is realistic from Nov 1.
- If the budget is short, the first lever is the existing daily prop slate pull, about 140 credits a day. That is the user's decision, not this project's.

---

## Constraints

- **Secrets:**
  - Never print, commit or source `.env`.
  - `ODDS_API_KEY` lives only in GitHub secrets.
  - Workflows use the existing `DATABASE_URL` secret.
- **Migrations:** the user runs them in the Supabase SQL editor. Ship an idempotent `db/migration_take_these.sql`.
- **Site:**
  - Follow the existing pages' patterns: `site/*.html`, `site/js/pages/*.js` and the `shell.js` nav.
  - Bump the cache version on deploy.
  - Desktop-first, but the page must work at phone width.
  - The user deploys the site themselves.
- **Tests:** Python (pytest) and site tests (`site/tests`), following the existing structure.
- **Pure engine, IO at the edges:** pick-finding, de-vig, good-until and grading are pure functions with unit tests. This is the ev_pilot / early-monitor pattern.

## Left to the builder

- Table schemas and view names.
- Exact module layout. A new package, e.g. `src/sportsmodel/take/`, is suggested.
- How the page polls Supabase. Reuse `data.js` patterns.
- The visual design of both pages, following the current site theme.
- Book-name normalisation for Pikkit, and parsing the market type from bet_info.
