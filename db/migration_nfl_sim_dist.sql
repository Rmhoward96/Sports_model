-- =============================================================================
-- nfl_sim game-level distributions: margin / total / each team's score pmf, so
-- the game page can draw the simulated Spread/Total/team-total histograms
-- (Apple/MVPEAV style) with an interactive line -> Under/At/Over.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- Formats (consumed by the front-end):
--   margin_dist  = {"kind":"margin","offset":O,"pmf":[...]}  value = i - O
--   *_dist (pmf) = {"kind":"pmf","pmf":[...]}                value = i (points)
ALTER TABLE nfl_sim ADD COLUMN IF NOT EXISTS margin_dist      JSONB;
ALTER TABLE nfl_sim ADD COLUMN IF NOT EXISTS total_dist       JSONB;
ALTER TABLE nfl_sim ADD COLUMN IF NOT EXISTS away_score_dist  JSONB;
ALTER TABLE nfl_sim ADD COLUMN IF NOT EXISTS home_score_dist  JSONB;

-- Re-expose the current-slate view with the new distribution columns.
CREATE OR REPLACE VIEW nfl_sim_current AS
  SELECT DISTINCT ON (game_pk)
    game_pk,
    model_version,
    matchup,
    commence_time,
    sim_home_win_prob,
    sim_margin,
    sim_total,
    disagreement,
    margin_dist,
    total_dist,
    away_score_dist,
    home_score_dist
  FROM nfl_sim
  WHERE commence_time > now()
  ORDER BY game_pk, created_at DESC;

GRANT SELECT ON nfl_sim_current TO anon, authenticated;
