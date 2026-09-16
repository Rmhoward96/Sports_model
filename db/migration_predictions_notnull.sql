-- =============================================================================
-- predictions_current: hide future rows with no kickoff time.
--
-- game_predictions can hold stale future-week placeholder rows (generated early,
-- before ESPN set kickoff times) with commence_time = NULL. The view filtered
-- only on game_date, so those leaked onto the predictions page with null times
-- and sorted oddly. The desk / +EV board already ignore them (they filter
-- commence_time > now()); this makes the front-end view consistent by requiring
-- a real kickoff time. Idempotent -- safe to re-run in the Supabase SQL Editor.
-- =============================================================================

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
    market_total
  FROM game_predictions
  WHERE game_date >= (now() - interval '8 hours')::date
    AND commence_time IS NOT NULL
  ORDER BY sport, game_pk, generated_at DESC;

GRANT SELECT ON predictions_current TO anon, authenticated;
