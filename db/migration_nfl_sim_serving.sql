-- =============================================================================
-- nfl_sim_serving: which NFL sim model_version the site/board read.
-- =============================================================================
-- Idempotent -- safe to re-run. Run in the Supabase SQL Editor.
-- Shadow mode writes a second model_version (nfl-sim-ml-v1) into nfl_sim /
-- nfl_player_sim; the *_current views serve ONLY the version named here.
-- Go live:   UPDATE nfl_sim_serving SET model_version = 'nfl-sim-ml-v1';
-- Roll back: UPDATE nfl_sim_serving SET model_version = 'sim-nfl-v1';
CREATE TABLE IF NOT EXISTS nfl_sim_serving (
    id            INT PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    model_version TEXT NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO nfl_sim_serving (id, model_version) VALUES (1, 'sim-nfl-v1')
ON CONFLICT (id) DO NOTHING;
ALTER TABLE nfl_sim_serving ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS nfl_sim_serving_read ON nfl_sim_serving;
CREATE POLICY nfl_sim_serving_read ON nfl_sim_serving FOR SELECT USING (true);
GRANT SELECT ON nfl_sim_serving TO anon, authenticated;

CREATE OR REPLACE VIEW nfl_sim_current AS
  SELECT DISTINCT ON (game_pk)
    game_pk,
    model_version,
    matchup,
    commence_time,
    sim_home_win_prob,
    sim_margin,
    sim_total,
    disagreement
  FROM nfl_sim
  WHERE commence_time > now()
    AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)
  ORDER BY game_pk, created_at DESC;

CREATE OR REPLACE VIEW nfl_player_sim_current AS
  SELECT DISTINCT ON (game_pk, player_id, market)
    game_pk,
    player_id,
    model_version,
    name,
    pos,
    team,
    market,
    mean,
    dist,
    commence_time
  FROM nfl_player_sim
  WHERE commence_time > now()
    AND model_version = (SELECT model_version FROM nfl_sim_serving WHERE id = 1)
  ORDER BY game_pk, player_id, market, created_at DESC;

GRANT SELECT ON nfl_sim_current, nfl_player_sim_current TO anon, authenticated;
