# Best-book serving layer + CLV (Blue Edge front-end)

**Date:** 2026-08-21
**Status:** Approved design (pre-plan)
**Replaces:** the Streamlit dashboard (`streamlit_app.py`) and the `prediction_results`
grading path, as the serving layer for the new Lovable front-end ("Blue Edge",
https://capalpha.lovable.app).

## Problem

The new front-end is a polished multi-sport tracker with a **board** (per game:
moneyline/spread/total + player props, each showing best-book odds, model %, no-vig
implied %, EV %, and the sportsbook offering the price) and a **track record** (W-L-P,
ROI, net units, avg EV, a cumulative-units chart, and a by-segment table with a **CLV**
column). It currently runs on hardcoded mock data — no backend.

The Streamlit app computes picks at read time (EV, calibration, main-line consensus,
pass gating) from raw predictions + odds + `calibration.json`. The Lovable app can only
read tables/views. So the pick math must move into precomputed Supabase tables, and the
serving/grading layer must be redesigned to support **best-book pricing** and **CLV from
day one**.

## Decisions (settled)

- **Precompute to Supabase tables**; the front-end is pure read-only.
- **Public read-only** access (RLS `SELECT USING (true)`); no auth.
- **Best-book pricing** for both the board and the graded bet; the **book name is stored
  and displayed**.
- **CLV live from day one**: lock a pick's price the **first time it turns +EV**; CLV is
  measured against the **consensus no-vig closing** price.
- **MLB only** — NFL/NBA pages stay empty until those models exist.

## Goals

- Two Supabase tables + two views that fully drive the board and track record, refreshed
  by the Python pipeline (GitHub Actions), with best-book prices and working CLV.
- One implementation of the pick math (shared module), reused by the board producer and
  the grader, so board and track record never drift.
- A documented data contract (schema + example queries) the user wires into Lovable.

## Non-goals

- No NFL/NBA models. No Supabase Auth. No changes inside the Lovable repo (user wires it
  via Lovable's Supabase integration). No live per-request computation (all precomputed).

## Data model

### `board_picks` — live board (display)
One row per game×market and per player×prop for **today's** slate, fully refreshed on each
run (upsert; stale rows for past dates may be pruned). Shows the model's better side with
the **current best-book** price.

```
sport TEXT, game_pk BIGINT, game_date DATE, commence_time TIMESTAMPTZ, matchup TEXT,
market TEXT, market_label TEXT, player_id BIGINT, player_name TEXT, team TEXT,
pick_label TEXT, side TEXT, line REAL,
odds INT, book TEXT, model_prob REAL, implied_prob REAL, ev REAL, is_pick BOOLEAN,
generated_at TIMESTAMPTZ DEFAULT now(),
PRIMARY KEY (game_pk, market, player_id)
```
- `market` ∈ moneyline|spread|total|hits|total_bases|home_run|hrr|pitcher_ks|hits_allowed|outs_recorded.
- `is_pick = ev > 0` for every market. The UI shows all moneyline rows regardless of
  `is_pick`, and filters other markets to `is_pick = true` (the front-end's choice).
- `player_id = 0` for game lines.

### `picks` — bet log / track record
Insert-once when a pick **first turns +EV**, locking the bet price; grader fills the
outcome. Feeds the track record.

```
game_pk BIGINT, market TEXT, player_id BIGINT,
sport TEXT, game_date DATE, commence_time TIMESTAMPTZ, matchup TEXT,
market_label TEXT, player_name TEXT, team TEXT, pick_label TEXT, side TEXT, line REAL,
bet_odds INT, bet_book TEXT, model_prob REAL, novig_bet REAL, ev_bet REAL,
bet_at TIMESTAMPTZ,
status TEXT DEFAULT 'pending',      -- pending | graded
actual REAL, result TEXT, profit REAL,   -- graded fields
novig_close REAL, clv REAL, graded_at TIMESTAMPTZ,
PRIMARY KEY (game_pk, market, player_id)
```
- Locked once via `ON CONFLICT (game_pk, market, player_id) DO NOTHING` — the first +EV
  price/side/book is the tracked bet and is never overwritten.
- Only +EV picks are ever inserted (moneyline included when +EV). Non-+EV = no row.

### Deprecated
`prediction_results` and the old `grade_results` per-game-lean grading are retired with
Streamlit. Keep the table until the new track record is validated, then drop.

## Best-book pricing

`odds_snapshot` stores per-book prices over time. For a given (game, market, side, line,
player) at a capture, the **best book** is the one with the highest decimal odds (most
favorable to the bettor → highest EV). Store that book's `price` and name. No-vig
**consensus** implied prob (used for `implied_prob` and CLV close) is computed from the
**median** price per side across books, then de-vigged with both sides:
`novig = io / (io + iu)`.

## CLV

- `novig_bet` = consensus no-vig implied prob of the picked side at **bet time** (the
  capture where it first turned +EV).
- `novig_close` = consensus no-vig implied prob of the picked side at **close** (last
  capture before `commence_time`).
- `clv = novig_close − novig_bet` (percentage points; store as a fraction, display as %).
  Positive = the market moved toward the picked side after the bet was locked.

## Grading (refactor of `grade_results`)

For each `picks` row with `status='pending'` whose game is final:
1. Fetch the actual score/stat via `mlb_results` (existing).
2. Decide win/loss/push at the pick's `side`/`line`; `profit = decimal(bet_odds) − 1` on a
   win, `−1` on a loss, `0` on a push (flat 1u at the **bet** price).
3. Compute `novig_close` (consensus no-vig at close) and `clv`.
4. Update the row: `status='graded'`, `actual`, `result`, `profit`, `novig_close`, `clv`,
   `graded_at`.

The existing StatsAPI results fetch, closing-line window guard, and the fresh-start floor
(`FRESH_START`) are reused. The multi-version dedup already in place is moot here (picks
are logged from the latest board, one per game/market/player).

## Track-record views (public-read)

- `track_record_segments`: from graded `picks`, `GROUP BY sport, market` →
  `wins, losses, pushes, win_pct, units (sum profit), roi (units/count), avg_ev, avg_clv`.
- `cumulative_units_weekly`: from graded `picks`, `date_trunc('week', game_date)` →
  weekly units and a running cumulative total.

## Shared pick math

Refactor the board/EV/calibration/main-line logic out of `streamlit_app.py` into
`src/sportsmodel/serving/board.py` (pure functions: best-book selection, model prob via
calibrated dist, EV, pick label, is_pick). `generate_board.py` and the grader import it, so
there is a single source of truth.

## Producers & wiring

- `scripts/generate_board.py`: read latest-version `game_predictions`/`prop_predictions`
  (reuse the dedup helpers) + latest `odds_snapshot` per (game, market, side, line, book) +
  `calibration.json`; compute the best-book board; **upsert `board_picks`** (full refresh)
  and **insert `picks`** for any pick that is +EV and not already logged. Runs after each
  `capture-odds` run (a new step in that workflow, or a matching schedule).
- `grade_results.py`: refactored per above; unchanged cadence (every 3h).
- RLS: `ENABLE ROW LEVEL SECURITY` + `SELECT USING (true)` on `board_picks`, `picks`, and
  the two views. Writers connect as the table owner via the Session pooler (bypasses RLS);
  verify a write still succeeds after enabling RLS.

## Front-end contract (handed to the user for Lovable)

- **Board (`/mlb`):** `board_picks` where `sport='mlb' AND game_date = <today>`; ML rows
  always shown, others filtered `is_pick`. Columns map 1:1 to matchup/market/pick/odds/
  model/implied/ev/book.
- **Dashboard top edges:** `board_picks` order by `ev desc` limit N.
- **Track record:** `track_record_segments` (the by-segment table + headline aggregate) and
  `cumulative_units_weekly` (the chart).
- Access via the Supabase **anon** key + auto REST API (`supabase.from(...).select()`).

## Scope & retirement

MLB only. After parity is confirmed for a day, retire the Streamlit app; keep
`streamlit_app.py` in the repo briefly as reference. Drop `prediction_results` once the new
track record is trusted.

## Acceptance

- `board_picks` populated for today's MLB slate with best-book prices + book names; the
  board's model %, implied %, EV match the shared module (parity with the calibrated model).
- `picks` locks the first-+EV price once; the grader fills result/profit/CLV correctly
  (unit-tested: win/loss/push profit at bet odds; `clv = novig_close − novig_bet`).
- Track-record views return correct W-L-P / win% / units / ROI / avg CLV and weekly
  cumulative units.
- Anon `SELECT` works on all four objects; the Actions writers still write after RLS.
- The Lovable app renders board + track record from these objects (user wires; parity
  vs Streamlit for one day before retiring it).

## Open gaps / follow-ups

- NFL/NBA models (front-end already has the tabs).
- Pruning old `board_picks`/`odds_snapshot` rows (housekeeping).
- Optional: show both the locked bet price and the current best price on the board (the
  board currently shows the current best-book price; the locked price lives in `picks`).
