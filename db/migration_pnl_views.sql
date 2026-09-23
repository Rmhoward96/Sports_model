-- =============================================================================
-- Profit trackers: what a flat $10 bet on every pick would have returned.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
--   prediction_pnl / _daily : every model prediction (ML winner, spread pick,
--                             total pick) bet at the CLOSING price.
--   ev_pnl / _daily         : every graded +EV pick (game lines at the pick's
--                             best price; +EV props from ev_prop_results).
--   nfl_prop_pnl            : every graded sim prop lean (nfl_prop_grades) bet
--                             at the leaned side's best price.
--   nfl_prop_pnl_by_game    : the prop P&L rolled up per game.
--
-- Stake is $10. Win pays stake * (decimal - 1); loss -10; push 0.

-- American -> decimal odds (NULL/0 -> NULL).
CREATE OR REPLACE FUNCTION american_to_decimal(price DOUBLE PRECISION)
RETURNS DOUBLE PRECISION LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN price IS NULL OR price = 0 THEN NULL
              WHEN price > 0 THEN 1 + price / 100.0
              ELSE 1 + 100.0 / (-price) END
$$;

-- Consensus closing price per (game, market, side): each book's LAST capture
-- before kickoff, median across books. Medianed in DECIMAL odds so a mix of
-- -102/+102 can't average to a nonsense American number. Restricted to graded
-- games so the scan stays small.
CREATE OR REPLACE VIEW game_closing_prices AS
  WITH last AS (
    SELECT DISTINCT ON (o.game_pk, o.market, o.side, o.book)
           o.game_pk, o.market, o.side, o.book, o.price
    FROM odds_snapshot o
    WHERE COALESCE(o.player_name, '') = ''   -- game lines store '' (not NULL)
      AND o.market IN ('moneyline', 'spread', 'total')
      AND o.captured_at <= o.commence_time
      AND o.price IS NOT NULL AND o.price <> 0
      AND o.game_pk IN (SELECT game_pk FROM prediction_accuracy WHERE actual_winner IS NOT NULL)
    ORDER BY o.game_pk, o.market, o.side, o.book, o.captured_at DESC
  )
  SELECT game_pk, market, side,
         percentile_cont(0.5) WITHIN GROUP (ORDER BY american_to_decimal(price)) AS close_dec,
         count(*) AS n_books
  FROM last
  GROUP BY game_pk, market, side;

-- One row per model bet. Pick sides mirror scripts/grade_predictions.py:
-- ML = predicted winner; spread = side the projected margin covers vs the
-- closing home line (no bet when it lands exactly on the line); total = over
-- iff projected total > closing total (no bet when equal). Spread/total with
-- no captured price assume -110; ML with no price is excluded (can't guess).
CREATE OR REPLACE VIEW prediction_pnl AS
  WITH p AS (
    SELECT * FROM prediction_accuracy WHERE actual_winner IS NOT NULL
  ), bets AS (
    SELECT sport, game_pk, game_date, 'moneyline'::text AS market,
           CASE WHEN predicted_winner = home_team_name THEN 'home' ELSE 'away' END AS side,
           CASE WHEN actual_winner = predicted_winner THEN 'win' ELSE 'loss' END AS result
    FROM p WHERE predicted_winner IS NOT NULL
    UNION ALL
    SELECT sport, game_pk, game_date, 'spread',
           CASE WHEN pred_margin + market_spread > 0 THEN 'home' ELSE 'away' END,
           CASE WHEN actual_margin + market_spread = 0 THEN 'push'
                WHEN (pred_margin + market_spread > 0) = (actual_margin + market_spread > 0) THEN 'win'
                ELSE 'loss' END
    FROM p WHERE pred_margin IS NOT NULL AND market_spread IS NOT NULL
             AND pred_margin + market_spread <> 0
    UNION ALL
    SELECT sport, game_pk, game_date, 'total',
           CASE WHEN pred_total > market_total THEN 'over' ELSE 'under' END,
           CASE WHEN actual_total = market_total THEN 'push'
                WHEN (pred_total > market_total) = (actual_total > market_total) THEN 'win'
                ELSE 'loss' END
    FROM p WHERE pred_total IS NOT NULL AND market_total IS NOT NULL
             AND pred_total <> market_total
  ), priced AS (
    SELECT b.*, c.n_books,
           COALESCE(c.close_dec,
                    CASE WHEN b.market <> 'moneyline' THEN american_to_decimal(-110) END) AS bet_dec
    FROM bets b
    LEFT JOIN game_closing_prices c
      ON c.game_pk = b.game_pk AND c.market = b.market AND c.side = b.side
  )
  SELECT *,
         CASE result WHEN 'win' THEN 10 * (bet_dec - 1) WHEN 'loss' THEN -10.0 ELSE 0.0 END AS pnl
  FROM priced
  WHERE bet_dec IS NOT NULL;

-- prediction_pnl_daily is a MATERIALIZED view -- see db/migration_pnl_materialize.sql
-- (a plain view here timed out under the anon role's 3s statement limit).

-- Every graded +EV pick. Game lines: ev_results joined to the pick row for its
-- best (pick-time) price; won NULL = push. Props: ev_prop_results already
-- carries profit in units at the pick price -> x10.
-- ev_pnl is redefined with a parlay branch in migration_ev_best_parlays.sql; run that after this file.
CREATE OR REPLACE VIEW ev_pnl AS
  SELECT r.sport, r.game_pk, r.market, r.side, p.commence_time,
         CASE WHEN r.won IS NULL THEN 'push' WHEN r.won THEN 'win' ELSE 'loss' END AS result,
         CASE WHEN r.won IS NULL THEN 0.0
              WHEN r.won THEN 10 * (american_to_decimal(p.best_price) - 1)
              ELSE -10.0 END AS pnl
  FROM ev_results r
  JOIN LATERAL (
    SELECT commence_time, best_price FROM ev_picks e
    WHERE e.game_pk = r.game_pk AND e.market = r.market AND e.side = r.side AND e.is_pick
    ORDER BY e.created_at DESC LIMIT 1
  ) p ON true
  WHERE r.won IS NOT NULL OR p.best_price IS NOT NULL
  UNION ALL
  SELECT sport, game_pk, 'prop', side, commence_time,
         CASE result WHEN 'win' THEN 'win' WHEN 'loss' THEN 'loss' ELSE 'push' END,
         10 * profit
  FROM ev_prop_results
  WHERE result IS NOT NULL AND profit IS NOT NULL;

CREATE OR REPLACE VIEW ev_pnl_daily AS
  SELECT (commence_time AT TIME ZONE 'America/New_York')::date AS game_date, sport, market,
         count(*)                                   AS n,
         count(*) FILTER (WHERE result = 'win')     AS wins,
         count(*) FILTER (WHERE result = 'loss')    AS losses,
         count(*) FILTER (WHERE result = 'push')    AS pushes,
         sum(pnl)                                   AS pnl
  FROM ev_pnl
  GROUP BY 1, sport, market;

-- Every graded sim prop lean, bet at the leaned side's best price. Rows whose
-- leaned side had no posted price get pnl NULL (not counted). ev_side tags the
-- props that were also +EV picks (the +EV side can differ from the lean).
CREATE OR REPLACE VIEW nfl_prop_pnl AS
  SELECT g.game_pk, g.player_id, g.player_name, g.team, g.market, g.line,
         g.lean, g.projection, g.actual, g.result, g.season, g.week, g.commence_time,
         CASE g.lean WHEN 'over' THEN g.over_price ELSE g.under_price END AS bet_price,
         CASE WHEN (CASE g.lean WHEN 'over' THEN g.over_price ELSE g.under_price END) IS NULL THEN NULL
              WHEN g.result = 'hit'  THEN 10 * (american_to_decimal(
                   CASE g.lean WHEN 'over' THEN g.over_price ELSE g.under_price END) - 1)
              WHEN g.result = 'miss' THEN -10.0
              ELSE 0.0 END AS pnl,
         (SELECT e.side FROM ev_prop_picks e
          WHERE e.game_pk = g.game_pk AND e.player_id = g.player_id
            AND e.market = g.market AND e.is_pick
          ORDER BY e.created_at DESC LIMIT 1) AS ev_side
  FROM nfl_prop_grades g;

CREATE OR REPLACE VIEW nfl_prop_pnl_by_game AS
  WITH lean AS (
    SELECT game_pk, max(commence_time) AS commence_time, max(season) AS season, max(week) AS week,
           count(*)                                  AS n,
           count(*) FILTER (WHERE result = 'hit')    AS hits,
           count(*) FILTER (WHERE result = 'miss')   AS misses,
           count(*) FILTER (WHERE result = 'push')   AS pushes,
           count(pnl)                                AS n_priced,
           sum(pnl)                                  AS pnl
    FROM nfl_prop_pnl GROUP BY game_pk
  ), ev AS (
    SELECT game_pk, count(*) AS ev_n,
           count(*) FILTER (WHERE result = 'win') AS ev_wins,
           count(*) FILTER (WHERE result = 'loss') AS ev_losses,
           sum(10 * profit) AS ev_pnl
    FROM ev_prop_results WHERE result IS NOT NULL AND profit IS NOT NULL
    GROUP BY game_pk
  )
  SELECT l.*, (SELECT s.matchup FROM nfl_sim s WHERE s.game_pk = l.game_pk
               ORDER BY s.created_at DESC LIMIT 1) AS matchup,
         COALESCE(ev.ev_n, 0) AS ev_n, COALESCE(ev.ev_wins, 0) AS ev_wins,
         COALESCE(ev.ev_losses, 0) AS ev_losses, ev.ev_pnl
  FROM lean l LEFT JOIN ev ON ev.game_pk = l.game_pk;

GRANT EXECUTE ON FUNCTION american_to_decimal(DOUBLE PRECISION) TO anon, authenticated;
GRANT SELECT ON game_closing_prices, prediction_pnl,
                ev_pnl, ev_pnl_daily, nfl_prop_pnl, nfl_prop_pnl_by_game
  TO anon, authenticated;
