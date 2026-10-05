-- Site redesign Phase A: opening -> current Pinnacle line per upcoming game market/side,
-- for the dashboard's Market Movers and the +EV Market Pulse; predictions_any gains the
-- model's margin / total distributions (the CFB game page). Run before deploying the site.
-- Run at a quiet time: CREATE INDEX (not CONCURRENTLY, so it can run in one transaction in the
-- SQL Editor) briefly blocks writes to odds_snapshot while it builds -- avoid odds-capture runs.
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

-- predictions_any: append the model's margin_dist / total_dist (home - away margin pmf, total pmf;
-- game_predictions columns from migration_game_dist.sql) so the game page reads the CFB model's own
-- cover / over probabilities at the market line (NFL already falls back to nfl_sim's distributions).
-- CREATE OR REPLACE VIEW must keep the live column list EXACTLY, in order, and may only add new
-- columns at the END; otherwise it fails with 42P16 (cannot drop / rename / reorder view columns).
-- The list below is the live pg_get_viewdef definition plus the two new columns.
CREATE OR REPLACE VIEW predictions_any AS
SELECT DISTINCT ON (sport, game_pk)
  sport, game_pk, game_date, home_team_name, away_team_name, home_win_prob,
  pred_home_score, pred_away_score, commence_time, market_spread, market_total, model_version,
  margin_dist, total_dist
FROM game_predictions
ORDER BY sport, game_pk, generated_at DESC;

GRANT SELECT ON predictions_any TO anon, authenticated;
