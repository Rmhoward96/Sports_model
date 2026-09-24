-- =============================================================================
-- Settings page data: per-book prices for current +EV picks, and per-book
-- moneylines for the game tiles, so the site can re-price at the user's books.
-- =============================================================================
-- Idempotent -- safe to re-run. Run in the Supabase SQL Editor. Requires
-- ev_best_lines (migration_ev_best_lines.sql) and american_to_decimal
-- (migration_pnl_views.sql). US books = serving.board.MAJOR_BOOKS minus Pinnacle.

-- One row per current +EV pick: each US book's latest pre-kickoff capture
-- (within 48h) at the pick's own line, as {book: american_price}.
CREATE OR REPLACE VIEW ev_pick_prices_current AS
  WITH gpicks AS (
    SELECT e.sport, e.game_pk, e.market, e.side, b.line
    FROM ev_current e LEFT JOIN ev_best_lines b USING (sport, game_pk, market, side)
    WHERE e.is_pick
  ), glast AS (
    SELECT DISTINCT ON (o.game_pk, o.market, o.side, o.book) o.game_pk, o.market, o.side, o.book, o.line, o.price
    FROM odds_snapshot o
    WHERE o.game_pk IN (SELECT game_pk FROM gpicks) AND o.market IN ('moneyline', 'spread', 'total')
      AND COALESCE(o.player_name, '') = '' AND o.captured_at <= o.commence_time
      AND o.captured_at > now() - interval '48 hours' AND o.price IS NOT NULL AND o.price <> 0
      AND o.book = ANY (ARRAY['draftkings','fanduel','fanatics','hardrockbet','thescore','espnbet',
                              'williamhill_us','caesars','bet365','betmgm','ballybet'])
    ORDER BY o.game_pk, o.market, o.side, o.book, o.captured_at DESC
  ), g AS (
    SELECT 'game'::text AS kind, p.sport, p.game_pk, p.market, p.side, NULL::text AS player_id, p.line,
           jsonb_object_agg(l.book, l.price) AS prices
    FROM gpicks p JOIN glast l ON l.game_pk = p.game_pk AND l.market = p.market AND l.side = p.side
     AND (p.market = 'moneyline' OR l.line = p.line)
    GROUP BY 1, 2, 3, 4, 5, 6, 7
  ), ppicks AS (
    SELECT game_pk, player_id, market, side, line,
           CASE market WHEN 'rec_yds' THEN 'reception_yds' ELSE market END AS omarket,
           lower(regexp_replace(regexp_replace(regexp_replace(trim(player_name), '[.''’]', '', 'g'),
                                             '\s+', ' ', 'g'),
                              '\s+(jr|sr|ii|iii|iv|v)$', '', 'i')) AS nkey
    FROM ev_prop_picks_current WHERE is_pick
  ), pcap AS (
    SELECT o.game_pk, o.market, o.player_name, o.book, max(o.captured_at) AS cap
    FROM odds_snapshot o
    WHERE o.game_pk IN (SELECT game_pk FROM ppicks) AND o.market IN (SELECT omarket FROM ppicks)
      AND o.captured_at <= o.commence_time AND o.captured_at > now() - interval '48 hours'
      AND o.book = ANY (ARRAY['draftkings','fanduel','fanatics','hardrockbet','thescore','espnbet',
                              'williamhill_us','caesars','bet365','betmgm','ballybet'])
    GROUP BY 1, 2, 3, 4
  ), prows AS (
    SELECT o.game_pk, o.market, o.side, o.book, o.line, o.price,
           lower(regexp_replace(regexp_replace(regexp_replace(trim(o.player_name), '[.''’]', '', 'g'),
                                             '\s+', ' ', 'g'),
                              '\s+(jr|sr|ii|iii|iv|v)$', '', 'i')) AS nkey
    FROM odds_snapshot o
    JOIN pcap c ON c.game_pk = o.game_pk AND c.market = o.market AND c.player_name = o.player_name
               AND c.book = o.book AND c.cap = o.captured_at
    WHERE o.price IS NOT NULL AND o.price <> 0
  ), p AS (
    SELECT 'prop'::text AS kind, 'nfl'::text AS sport, pp.game_pk, pp.market, pp.side, pp.player_id, pp.line,
           jsonb_object_agg(r.book, r.price) AS prices
    FROM ppicks pp JOIN prows r ON r.game_pk = pp.game_pk AND r.market = pp.omarket AND r.side = pp.side
                               AND r.line = pp.line AND r.nkey = pp.nkey
    GROUP BY 1, 2, 3, 4, 5, 6, 7
  )
  SELECT * FROM g UNION ALL SELECT * FROM p;

-- game_moneylines_current (migration_game_moneylines.sql) + per-book maps.
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
  ), maps AS (
    SELECT game_pk,
           jsonb_object_agg(book, price) FILTER (WHERE side = 'home') AS home_prices,
           jsonb_object_agg(book, price) FILTER (WHERE side = 'away') AS away_prices
    FROM last GROUP BY game_pk
  )
  SELECT b.game_pk,
         max(b.price) FILTER (WHERE b.side = 'home') AS home_price,
         max(b.book)  FILTER (WHERE b.side = 'home') AS home_book,
         max(b.price) FILTER (WHERE b.side = 'away') AS away_price,
         max(b.book)  FILTER (WHERE b.side = 'away') AS away_book,
         (array_agg(m.home_prices))[1] AS home_prices,
         (array_agg(m.away_prices))[1] AS away_prices
  FROM best b JOIN maps m USING (game_pk)
  GROUP BY b.game_pk;

GRANT SELECT ON ev_pick_prices_current, game_moneylines_current TO anon, authenticated;
