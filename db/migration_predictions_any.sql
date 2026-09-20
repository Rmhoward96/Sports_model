-- =============================================================================
-- predictions_any: the latest model prediction per game with NO date floor.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- `predictions_current` filters to the upcoming/current slate (game_date >=
-- now()-8h) so the home/league lists only show games you can still bet. But a
-- game DETAIL page should stay reachable after kickoff and after the game is
-- over -- the simulation and projections should never disappear, so you can
-- come back and see how the sim did against the actual result. This view is
-- the same DISTINCT-ON-latest projection per game as predictions_current, just
-- without the date floor, so the front-end can load any game by game_pk
-- regardless of when it was played.
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
    market_total
  FROM game_predictions
  ORDER BY sport, game_pk, generated_at DESC;

GRANT SELECT ON predictions_any TO anon, authenticated;
