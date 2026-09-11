-- +EV pilot: freeze the pick-time (opening) Pinnacle price so CLV is real.
--
-- upsert_ev_picks overwrites pinnacle_price on every board rebuild, and the
-- board rebuilds constantly (scheduled + every desk-auto run), so by kickoff
-- the stored price has drifted to the close and clv = implied(close) -
-- implied(pick) collapsed to ~0 for every pick. open_pinnacle_price is set
-- once on first INSERT and never updated (see db.py _EV_PICKS_IMMUTABLE), so
-- grade_ev.py can measure CLV against the price the pick was first surfaced at.
--
-- Idempotent; safe to re-run. Existing rows keep open_pinnacle_price = NULL and
-- grade_ev falls back to pinnacle_price for them (their true pick-time price is
-- already lost); every row written after this migration freezes correctly.

ALTER TABLE ev_picks ADD COLUMN IF NOT EXISTS open_pinnacle_price INTEGER;
