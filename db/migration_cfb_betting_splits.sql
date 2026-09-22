-- =============================================================================
-- cfb_betting_splits: public betting splits (cash% vs ticket%) per CFB game/
-- market, captured near close from Action Network (Apify actor, league=ncaaf).
-- Mirrors nfl_betting_splits, but browser-readable from the start (the game page
-- reads cfb_betting_splits_current).
-- =============================================================================
-- Idempotent -- safe to re-run in the Supabase SQL Editor (SERVING project).
-- One row per (game_pk, market, side, captured_at). game_pk is the ESPN event
-- id, resolved from the actor's team names against the ESPN CFB slate.
CREATE TABLE IF NOT EXISTS cfb_betting_splits (
    game_pk       BIGINT NOT NULL,
    market        TEXT   NOT NULL,   -- moneyline | spread | total
    side          TEXT   NOT NULL,   -- home | away | over | under
    cash_pct      DOUBLE PRECISION,  -- % of money on this side
    ticket_pct    DOUBLE PRECISION,  -- % of tickets on this side
    commence_time TIMESTAMPTZ,
    captured_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (game_pk, market, side, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_cfb_betting_splits_game ON cfb_betting_splits (game_pk, market);

-- Latest captured splits per (game_pk, market, side) for the browser.
CREATE OR REPLACE VIEW cfb_betting_splits_current AS
  SELECT DISTINCT ON (game_pk, market, side)
    game_pk, market, side, cash_pct, ticket_pct, commence_time, captured_at
  FROM cfb_betting_splits
  ORDER BY game_pk, market, side, captured_at DESC;

-- Public read-only access for the browser (anon key).
ALTER TABLE cfb_betting_splits ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read cfb_betting_splits" ON cfb_betting_splits;
CREATE POLICY "public read cfb_betting_splits" ON cfb_betting_splits FOR SELECT USING (true);

GRANT SELECT ON cfb_betting_splits, cfb_betting_splits_current TO anon, authenticated;
