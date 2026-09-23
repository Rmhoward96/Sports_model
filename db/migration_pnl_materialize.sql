-- =============================================================================
-- Precompute prediction_pnl_daily (was a plain view).
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- Pricing every model pick at the close scans every captured odds row (~2.6s
-- and growing), past the anon role's 3s statement timeout, so the browser's
-- read of the profit tracker timed out and showed nothing. A materialized view
-- computes it once; scripts/grade_predictions.py refreshes it after each run
-- (db.refresh_prediction_pnl). The underlying prediction_pnl view is unchanged.
DO $$
BEGIN
  IF EXISTS (SELECT 1 FROM pg_views WHERE schemaname = 'public' AND viewname = 'prediction_pnl_daily') THEN
    DROP VIEW prediction_pnl_daily;
  END IF;
END $$;

CREATE MATERIALIZED VIEW IF NOT EXISTS prediction_pnl_daily AS
  SELECT game_date, sport, market,
         count(*)                                   AS n,
         count(*) FILTER (WHERE result = 'win')     AS wins,
         count(*) FILTER (WHERE result = 'loss')    AS losses,
         count(*) FILTER (WHERE result = 'push')    AS pushes,
         sum(pnl)                                   AS pnl
  FROM prediction_pnl
  GROUP BY game_date, sport, market;

REFRESH MATERIALIZED VIEW prediction_pnl_daily;

GRANT SELECT ON prediction_pnl_daily TO anon, authenticated;
