-- =============================================================================
-- nfl_player_actuals: realized player box stats for finished NFL games, shown
-- next to the sim's projected props on the game page.
-- =============================================================================
-- Idempotent -- safe to re-run. Run this in the Supabase SQL Editor.
--
-- One row per (game_pk, player_id, market). `actual` is the realized stat for
-- that market; for anytime_td it is the player's total TDs (>= 1 means scored).
-- Written by scripts/capture_nfl_player_actuals.py from nflverse weekly data.
-- Kept in its own table (not a column on nfl_player_sim) so the sim's
-- replace-per-game projection write never touches captured facts.
CREATE TABLE IF NOT EXISTS nfl_player_actuals (
    game_pk      BIGINT NOT NULL,
    player_id    TEXT   NOT NULL,   -- gsis_id, same namespace as nfl_player_sim
    player_name  TEXT,
    market       TEXT   NOT NULL,   -- pass_yds|pass_tds|rush_yds|rec_yds|receptions|anytime_td
    actual       DOUBLE PRECISION,  -- realized stat; anytime_td = total TDs
    season       INT,
    week         INT,
    captured_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (game_pk, player_id, market)
);

CREATE INDEX IF NOT EXISTS idx_nfl_player_actuals_game ON nfl_player_actuals (game_pk);

-- Public read-only access for the browser (anon key), mirroring nfl_player_sim.
ALTER TABLE nfl_player_actuals ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public read nfl_player_actuals" ON nfl_player_actuals;
CREATE POLICY "public read nfl_player_actuals" ON nfl_player_actuals FOR SELECT USING (true);
GRANT SELECT ON nfl_player_actuals TO anon, authenticated;
