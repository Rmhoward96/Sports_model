-- =============================================================================
-- odds_snapshot (market, captured_at) index: lets the 48h price windows
-- (game_moneylines_current, the +EV board loaders) read only recent captures.
-- =============================================================================
-- Idempotent -- safe to re-run. Run in the Supabase SQL Editor.
--
-- Without it, game_moneylines_current walks every moneyline capture ever
-- stored (186k rows on 2026-09-24, growing ~13k/day) to keep the ~6.6k from
-- the last 48h. Warm that is ~200 ms, but on a cold cache it passes the anon
-- role's 3s statement_timeout, the request 500s, and the site's game tiles
-- fall back to the model's fair line until a refresh (cache now warm).
CREATE INDEX IF NOT EXISTS idx_odds_snapshot_market_captured
  ON odds_snapshot (market, captured_at);

ANALYZE odds_snapshot;
