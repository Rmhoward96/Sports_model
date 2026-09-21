-- =============================================================================
-- nfl_betting_splits: public betting splits (cash% vs ticket%) per game/market,
-- captured near close from SportsDataIO's Betting tier. Training-side input for
-- the cover/total ensemble's market-microstructure features (Phase 2).
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- One row per (game_pk, market, side, captured_at). `cash_pct`/`ticket_pct` are
-- 0..100 consensus percentages for that side. No anon read policy: this feeds
-- model training, not the browser.
CREATE TABLE IF NOT EXISTS nfl_betting_splits (
    game_pk       BIGINT NOT NULL,
    market        TEXT   NOT NULL,   -- spread | total | moneyline
    side          TEXT   NOT NULL,   -- home | away | over | under
    cash_pct      DOUBLE PRECISION,  -- % of money on this side
    ticket_pct    DOUBLE PRECISION,  -- % of tickets on this side
    commence_time TIMESTAMPTZ,
    captured_at   TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (game_pk, market, side, captured_at)
);

CREATE INDEX IF NOT EXISTS idx_nfl_betting_splits_game ON nfl_betting_splits (game_pk, market);
