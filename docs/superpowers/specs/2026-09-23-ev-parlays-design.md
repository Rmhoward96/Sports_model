# +EV Parlays — design

Date: 2026-09-23 · Status: approved in chat ("Yes, but just call the section Parlays")

## Goal
Build up to 3 three-leg parlays per slate from the highest-confidence, highest-EV
current +EV picks, each priced at ONE book (the best single-book price for all
three legs), show them on the +EV page under **"Parlays"**, and track them
($10/ticket P&L) on Track Record.

## Decisions (user-approved)
- Leg pool: every current +EV pick — NFL/CFB moneyline/spread/total
  (`ev_current`, `is_pick`) and NFL player props (`ev_prop_picks_current`,
  `is_pick`).
- Confidence: leg win probability ≥ **0.55** (`true_prob` for game lines,
  `model_prob` for props); legs ranked by EV.
- **One leg per game** within a ticket (independent legs; true prob = product).
- Up to **3 tickets per build, no shared legs** across tickets.
- Section title on the +EV page: **"Parlays"**.

## Pricing
- A leg's per-book prices come from `odds_snapshot`, each book's LATEST capture
  before kickoff (same "latest capture per book" rule as
  `build_ev_props_board.latest_capture_only`), restricted to US books in
  `serving.board.MAJOR_BOOKS` (no Pinnacle).
- A book counts for a leg only at the SAME line the pick was made at (spread/
  total: the best book's line from `ev_best_lines` logic; props: the pick's
  `line`). Moneyline has no line.
- At a book, a leg is eligible only if it is still +EV there
  (`prob × decimal(price) − 1 > 0`).

## Building (pure)
For each book: eligible legs, one per game (keep the highest EV-at-book), top 3
by EV-at-book; parlay decimal = product of the three book prices; true prob =
product of leg probs; EV = true_prob × decimal − 1. Keep the book with the best
EV. Require exactly 3 legs and `0.01 < EV ≤ 1.0`. That is ticket #1; remove
its legs from the pool and repeat, up to 3 tickets.

`parlay_id` = `book|` + sorted leg keys (`g:{game_pk}:{market}:{side}`,
`p:{game_pk}:{player_id}:{market}:{side}`). Ticket `sport` = the legs' common
sport, else `"mixed"`.

## Lifecycle / tracking
- Each build: DELETE unlocked tickets (first leg not yet started), then insert
  the new set. A ticket whose first leg has started is **locked** — never
  deleted, never rebuilt; its legs are excluded from new builds until graded.
- Grading (locked tickets, not yet graded):
  - Game legs from `prediction_accuracy` (actual_margin = home − away,
    actual_total): ML win iff the side won (tie → push); spread cover =
    side-margin + line (>0 win, <0 loss, 0 push); total vs line.
  - Prop legs from `nfl_player_actuals` (player_id, market) vs line; a player
    with no actual once that game's actuals exist = **void (push)**.
  - Any leg lost → ticket loss (−$10) as soon as known. Otherwise when all legs
    resolve: push legs drop out; win pays `10 × (Π decimal of winning legs − 1)`;
    all legs push → push ($0).
- Results in `ev_best_parlay_results` with per-leg results.

## Surfaces
- +EV page: "Parlays" section — one card per ticket: book, combined American
  price, ticket EV and win prob, each leg (label, price at that book, prob).
- Track Record → +EV tab: "+EV PARLAYS" card in the +EV profit tracker
  (`ev_pnl` gains parlay rows, market `parlay`), plus a graded-parlays list with
  per-leg ✓/✗.

## Plumbing
- Migration: `ev_best_parlays`, `ev_best_parlay_results`,
  `ev_best_parlays_current` view, `ev_pnl` updated; anon read.
- Builder `scripts/build_best_parlays.py` runs after the full (non-lines-only)
  `build-ev-props` runs. Grader `scripts/grade_best_parlays.py` runs in
  `grade-ev-props` (daily) and every 6h with grade-predictions.
- No new Odds API calls.

## Out of scope
Same-game parlays, parlays > 3 legs, changing +EV pick selection.
