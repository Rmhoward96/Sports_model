-- =============================================================================
-- game_moneylines_current: best available moneyline per side for every
-- upcoming game, for the site's game tiles.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- Each US book's latest pre-kickoff moneyline capture (within 48h, so a book
-- that stopped posting doesn't linger), then the best price per side -- which
-- can be two different books. US books only: the same set as the +EV board
-- (sportsmodel.serving.board.MAJOR_BOOKS minus Pinnacle); the odds capture also
-- pulls EU/offshore books that aren't bettable in the US. Requires
-- american_to_decimal() from migration_pnl_views.sql. odds_snapshot itself is
-- RLS-locked from the browser; this view exposes only these prices.
CREATE OR REPLACE VIEW game_moneylines_current AS
  WITH last AS (
    SELECT DISTINCT ON (game_pk, side, book) game_pk, side, book, price
    FROM odds_snapshot
    WHERE market = 'moneyline' AND COALESCE(player_name, '') = ''
      AND commence_time > now() AND captured_at <= commence_time
      AND captured_at > now() - interval '48 hours'
      AND price IS NOT NULL AND price <> 0
      AND book IN ('draftkings', 'fanduel', 'fanatics', 'hardrockbet', 'thescore', 'espnbet',
                   'williamhill_us', 'caesars', 'bet365', 'betmgm', 'ballybet')
    ORDER BY game_pk, side, book, captured_at DESC
  ), best AS (
    SELECT DISTINCT ON (game_pk, side) game_pk, side, book, price
    FROM last
    ORDER BY game_pk, side, american_to_decimal(price) DESC, book
  )
  SELECT game_pk,
         max(price) FILTER (WHERE side = 'home') AS home_price,
         max(book)  FILTER (WHERE side = 'home') AS home_book,
         max(price) FILTER (WHERE side = 'away') AS away_price,
         max(book)  FILTER (WHERE side = 'away') AS away_book
  FROM best
  GROUP BY game_pk;

GRANT SELECT ON game_moneylines_current TO anon, authenticated;
