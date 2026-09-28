-- =============================================================================
-- predictions_any / predictions_current: expose model_version.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- The site labels the NFL prediction block "Model: ML v1" when the served
-- prediction came from the ML sim (model_version = 'nfl-sim-ml-v1'). Anon can't
-- read game_predictions rows directly (RLS on, no policies), so both views
-- append model_version as their LAST column (CREATE OR REPLACE VIEW may only
-- add columns at the end). Everything else is unchanged from
-- migration_predictions_any.sql and the live predictions_current definition.
CREATE OR REPLACE VIEW predictions_any AS
  SELECT DISTINCT ON (sport, game_pk)
    sport,
    game_pk,
    game_date,
    home_team_name,
    away_team_name,
    home_win_prob,
    pred_home_score,
    pred_away_score,
    commence_time,
    market_spread,
    market_total,
    model_version
  FROM game_predictions
  ORDER BY sport, game_pk, generated_at DESC;

CREATE OR REPLACE VIEW predictions_current AS
  SELECT DISTINCT ON (sport, game_pk)
    sport,
    game_pk,
    game_date,
    home_team_name,
    away_team_name,
    home_win_prob,
    pred_home_score,
    pred_away_score,
    commence_time,
    market_spread,
    market_total,
    model_version
  FROM game_predictions
  WHERE game_date >= (now() - interval '8 hours')::date
    AND commence_time IS NOT NULL
  ORDER BY sport, game_pk, generated_at DESC;

GRANT SELECT ON predictions_any TO anon, authenticated;
GRANT SELECT ON predictions_current TO anon, authenticated;
