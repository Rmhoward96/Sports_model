-- Site redesign Phase A: opening -> current Pinnacle line per upcoming game market/side,
-- for the dashboard's Market Movers and the +EV Market Pulse. Run before deploying the site.
CREATE INDEX IF NOT EXISTS idx_odds_snapshot_book_commence ON odds_snapshot (book, commence_time);

-- Owner-rights view (no security_invoker): odds_snapshot has RLS with no read policy, so an
-- invoker view would return 0 rows to the site's anon key; matches the existing ev_current views.
CREATE OR REPLACE VIEW line_moves_current AS
WITH s AS (
  SELECT game_pk, market, side, line, price, captured_at, commence_time
  FROM odds_snapshot
  WHERE book = 'pinnacle' AND coalesce(player_name, '') = ''
    AND market IN ('moneyline', 'spread', 'total') AND commence_time > now()
), o AS (
  SELECT DISTINCT ON (game_pk, market, side) game_pk, market, side, line AS open_line,
         price AS open_price, captured_at AS open_at
  FROM s ORDER BY game_pk, market, side, captured_at ASC
), c AS (
  SELECT DISTINCT ON (game_pk, market, side) game_pk, market, side, line AS cur_line,
         price AS cur_price, captured_at AS cur_at, commence_time
  FROM s ORDER BY game_pk, market, side, captured_at DESC
)
SELECT c.game_pk, c.market, c.side, o.open_line, o.open_price, o.open_at,
       c.cur_line, c.cur_price, c.cur_at, c.commence_time
FROM c JOIN o USING (game_pk, market, side)
WHERE o.open_at < c.cur_at;

GRANT SELECT ON line_moves_current TO anon, authenticated;
