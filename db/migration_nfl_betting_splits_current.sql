-- =============================================================================
-- Expose CURRENT NFL betting splits to the browser (anon key).
-- =============================================================================
-- nfl_betting_splits (see migration_nfl_betting_splits.sql) is captured near
-- close as one row per (game_pk, market, side, captured_at) -- a time series.
-- The game page only needs the LATEST snapshot per (game_pk, market, side), so
-- this adds a `nfl_betting_splits_current` view (DISTINCT ON, newest capture)
-- and grants anon read on the base table + view, mirroring migration_nfl_sim's
-- public-read pattern. Idempotent -- safe to re-run in the Supabase SQL Editor.
--
-- Run this in the SERVING project (the one the GitHub Actions write to / the
-- front-end's SUPABASE_URL points at), NOT a local/other project.
-- =============================================================================

-- Latest captured splits per (game_pk, market, side). No commence_time filter:
-- splits are only captured for near-kickoff games anyway, and the game page
-- reads by game_pk, so keep the last snapshot visible even at/after kickoff.
CREATE OR REPLACE VIEW nfl_betting_splits_current AS
  SELECT DISTINCT ON (game_pk, market, side)
    game_pk,
    market,          -- moneyline | spread | total
    side,            -- home | away | over | under
    cash_pct,        -- % of money on this side (0..100)
    ticket_pct,      -- % of tickets/bets on this side (0..100)
    commence_time,
    captured_at
  FROM nfl_betting_splits
  ORDER BY game_pk, market, side, captured_at DESC;

-- Public read-only access for the browser (anon key).
ALTER TABLE nfl_betting_splits ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_betting_splits" ON nfl_betting_splits;
CREATE POLICY "public read nfl_betting_splits" ON nfl_betting_splits FOR SELECT USING (true);

GRANT SELECT ON nfl_betting_splits, nfl_betting_splits_current TO anon, authenticated;
